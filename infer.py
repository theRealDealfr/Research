import os
import argparse
import torch
import soundfile as sf
import numpy as np
import librosa
from torch import nn
from transformers import AutoFeatureExtractor, AutoModel
import peft
from peft import LoraConfig, get_peft_model

class AAFFClassifier(nn.Module):
    """
    Stage 1 Age-Aware Feature Fusion (AAFF) Binary Architecture:
    1024-d speech embedding + 1-d normalized age (1025-d)
    -> Linear(1025, proj_dim) -> ReLU() -> Dropout(0.2) -> Linear(proj_dim, num_classes)
    """
    def __init__(self, in_features=1025, proj_dim=256, dropout=0.2, num_classes=2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, emb, age):
        fused = torch.cat([emb, age], dim=1)
        projected = self.projection(fused)
        return self.classifier(projected)

class AAFFDisorderClassifier(nn.Module):
    """
    Stage 2 Age-Aware Feature Fusion (AAFF) Multiclass Disorder Architecture:
    1024-d speech embedding + 1-d normalized age (1025-d)
    -> Linear(1025, proj_dim) -> ReLU() -> Dropout(0.2) -> Linear(proj_dim, num_classes)
    """
    def __init__(self, in_features=1025, proj_dim=256, dropout=0.2, num_classes=4):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, emb, age):
        fused = torch.cat([emb, age], dim=1)
        projected = self.projection(fused)
        return self.classifier(projected)

DISORDER_CLASSES = ["substitution", "stuttering", "omission", "addition"]
_MODEL_CACHE = {}

def inspect_checkpoint(checkpoint_path):
    """
    Inspects a checkpoint and returns architecture metadata, detecting whether it is:
      1. Legacy Frozen Baseline (flat state_dict)
      2. LoRA Fine-Tuned (consolidated dictionary with lora_state_dict)
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {"exists": False, "architecture_type": "Unknown", "model_type": "none"}
    
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"exists": True, "error": str(e), "architecture_type": "Unknown"}

    if isinstance(ckpt, dict) and ("lora_state_dict" in ckpt or "model_type" in ckpt):
        unfrozen = ckpt.get("unfreeze_top_layers", 0)
        arch_desc = f"LoRA Fine-Tuned (WavLM-Large + LoRA r={ckpt.get('lora_r', 8)}"
        if unfrozen > 0:
            arch_desc += f" + Top {unfrozen} Layers"
        arch_desc += " + AAFF)"

        return {
            "exists": True,
            "architecture_type": "LoRA Fine-Tuned",
            "architecture_description": arch_desc,
            "model_type": ckpt.get("model_type", "wavlm_lora_aaff"),
            "model_checkpoint": ckpt.get("model_checkpoint", "microsoft/wavlm-large"),
            "unfreeze_top_layers": unfrozen,
            "unfrozen_layers": ckpt.get("unfrozen_layers", []),
            "lora_r": ckpt.get("lora_r", 8),
            "lora_alpha": ckpt.get("lora_alpha", 16),
            "target_modules": ckpt.get("target_modules", "q_proj,v_proj"),
            "num_classes": ckpt.get("num_classes", 2),
            "unique_labels": ckpt.get("unique_labels", ["speech_disorder", "typically_developing"]),
            "proj_dim": ckpt.get("proj_dim", 256),
            "age_min": float(ckpt.get("age_min", 5.67)),
            "age_max": float(ckpt.get("age_max", 12.83)),
            "metrics": ckpt.get("metrics", {})
        }
    else:
        # Legacy flat state_dict
        proj_weight = ckpt.get("projection.0.weight", None) if isinstance(ckpt, dict) else None
        clf_weight = ckpt.get("classifier.weight", None) if isinstance(ckpt, dict) else None
        proj_dim = proj_weight.shape[0] if proj_weight is not None else 256
        num_classes = clf_weight.shape[0] if clf_weight is not None else 2
        return {
            "exists": True,
            "architecture_type": "Frozen Baseline",
            "architecture_description": "Frozen Baseline (WavLM-Large + AAFF)",
            "model_type": "frozen_wavlm_aaff",
            "model_checkpoint": "microsoft/wavlm-large",
            "unfreeze_top_layers": 0,
            "lora_r": None,
            "num_classes": num_classes,
            "unique_labels": ["speech_disorder", "typically_developing"] if num_classes == 2 else DISORDER_CLASSES,
            "proj_dim": proj_dim,
            "age_min": 5.67,
            "age_max": 12.83,
            "metrics": {}
        }

def get_loaded_hierarchical_models(
    stage1_checkpoint="./results_controlled_baselines/model_c_aaff_best.pt",
    stage2_checkpoint="./results_controlled_baselines/model_stage2_disorder_best.pt",
    model_checkpoint="microsoft/wavlm-large",
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_key = (
        os.path.abspath(stage1_checkpoint) if stage1_checkpoint else None,
        os.path.abspath(stage2_checkpoint) if stage2_checkpoint and os.path.exists(stage2_checkpoint) else None,
        model_checkpoint,
        str(device)
    )

    if cache_key not in _MODEL_CACHE:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
        stage1_meta = inspect_checkpoint(stage1_checkpoint)

        # 1. Determine if Stage 1 uses LoRA backbone or Frozen backbone
        if stage1_meta.get("architecture_type") == "LoRA Fine-Tuned":
            base_wavlm = AutoModel.from_pretrained(model_checkpoint, use_safetensors=True)
            for param in base_wavlm.parameters():
                param.requires_grad = False

            ckpt = torch.load(stage1_checkpoint, map_location=device, weights_only=False)
            target_mods = [m.strip() for m in ckpt.get("target_modules", "q_proj,v_proj").split(",") if m.strip()]
            lora_config = LoraConfig(
                r=ckpt.get("lora_r", 8),
                lora_alpha=ckpt.get("lora_alpha", 16),
                target_modules=target_mods,
                lora_dropout=0.0,
                bias="none"
            )
            lora_wavlm = get_peft_model(base_wavlm, lora_config)

            # Load LoRA adapter weights
            if "lora_state_dict" in ckpt and hasattr(lora_wavlm, "load_state_dict"):
                peft.set_peft_model_state_dict(lora_wavlm, ckpt["lora_state_dict"])

            # Load unfrozen backbone layers if present
            if "backbone_trainable_state" in ckpt:
                for name, param in lora_wavlm.named_parameters():
                    if name in ckpt["backbone_trainable_state"]:
                        param.data.copy_(ckpt["backbone_trainable_state"][name].to(device))

            lora_wavlm.eval()
            lora_wavlm.to(device)
            wavlm = lora_wavlm

            # Load AAFF Head
            s1_proj_dim = ckpt.get("proj_dim", 256)
            s1_num_classes = ckpt.get("num_classes", 2)
            stage1_model = AAFFClassifier(in_features=1025, proj_dim=s1_proj_dim, dropout=0.0, num_classes=s1_num_classes).to(device)
            if "head_state_dict" in ckpt:
                stage1_model.projection.load_state_dict(ckpt["head_state_dict"]["projection"])
                stage1_model.classifier.load_state_dict(ckpt["head_state_dict"]["classifier"])
            stage1_model.eval()

        else:
            # Legacy Frozen Backbone
            wavlm = AutoModel.from_pretrained(model_checkpoint, use_safetensors=True)
            wavlm.eval()
            wavlm.to(device)
            for param in wavlm.parameters():
                param.requires_grad = False

            stage1_model = None
            if stage1_checkpoint and os.path.exists(stage1_checkpoint):
                state_dict = torch.load(stage1_checkpoint, map_location=device, weights_only=False)
                if isinstance(state_dict, dict) and "projection.0.weight" in state_dict:
                    s1_proj_dim = state_dict["projection.0.weight"].shape[0]
                    s1_num_classes = state_dict["classifier.weight"].shape[0]
                    stage1_model = AAFFClassifier(in_features=1025, proj_dim=s1_proj_dim, dropout=0.0, num_classes=s1_num_classes).to(device)
                    stage1_model.load_state_dict(state_dict)
                    stage1_model.eval()

        # 2. Load Stage 2 Multiclass Model (if exists)
        stage2_model = None
        if stage2_checkpoint and os.path.exists(stage2_checkpoint):
            state_dict = torch.load(stage2_checkpoint, map_location=device, weights_only=False)
            if isinstance(state_dict, dict) and "projection.0.weight" in state_dict:
                s2_proj_dim = state_dict["projection.0.weight"].shape[0]
                s2_num_classes = state_dict["classifier.weight"].shape[0]
                stage2_model = AAFFDisorderClassifier(in_features=1025, proj_dim=s2_proj_dim, dropout=0.0, num_classes=s2_num_classes).to(device)
                stage2_model.load_state_dict(state_dict)
                stage2_model.eval()

        _MODEL_CACHE[cache_key] = (feature_extractor, wavlm, stage1_model, stage2_model, stage1_meta)

    return _MODEL_CACHE[cache_key]

def extract_audio_features(audio_path, feature_extractor, wavlm, device, target_sr=16000, max_duration=12.0):
    audio_array, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    duration = len(audio_array) / float(target_sr)
    max_length = int(target_sr * max_duration)

    inputs = feature_extractor(
        audio_array,
        sampling_rate=target_sr,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    input_values = inputs["input_values"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = wavlm(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        raw_backbone = wavlm.get_base_model() if hasattr(wavlm, "get_base_model") else wavlm
        if hasattr(raw_backbone, "model"):
            raw_backbone = raw_backbone.model

        feature_mask = raw_backbone._get_feature_vector_attention_mask(
            hidden_states.shape[1],
            attention_mask
        )
        mask = feature_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    return pooled, duration

def predict_hierarchical_speech_disorder(
    audio_path,
    age,
    stage1_checkpoint="./results_controlled_baselines/model_c_aaff_best.pt",
    stage2_checkpoint="./results_controlled_baselines/model_stage2_disorder_best.pt",
    model_checkpoint="microsoft/wavlm-large"
):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.exists(stage1_checkpoint):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor, wavlm, stage1_model, stage2_model, stage1_meta = get_loaded_hierarchical_models(
        stage1_checkpoint=stage1_checkpoint,
        stage2_checkpoint=stage2_checkpoint,
        model_checkpoint=model_checkpoint,
        device=device
    )

    # 1. Feature Extraction
    pooled_emb, duration = extract_audio_features(audio_path, feature_extractor, wavlm, device)

    # 2. Age Normalization (dynamically from stage1_meta or default training bounds)
    age_min = float(stage1_meta.get("age_min", 5.67))
    age_max = float(stage1_meta.get("age_max", 12.83))
    norm_age = (float(age) - age_min) / (age_max - age_min)
    norm_age_clamped = max(0.0, min(1.0, norm_age))
    age_tensor = torch.tensor([[norm_age_clamped]], dtype=torch.float32).to(device)

    # 3. STAGE 1: Binary Classification (Speech Disorder vs Typically Developing)
    with torch.no_grad():
        s1_logits = stage1_model(pooled_emb, age_tensor)
        s1_probs = torch.softmax(s1_logits, dim=-1)[0].cpu().numpy()

    # Index 0: speech_disorder (SSD), Index 1: typically_developing (TD)
    prob_ssd = float(s1_probs[0])
    prob_td  = float(s1_probs[1])

    is_disordered = prob_ssd >= prob_td
    stage1_prediction = "Speech Sound Disorder (SSD)" if is_disordered else "Typical Developing (TD)"
    stage1_confidence = prob_ssd if is_disordered else prob_td

    stage2_prediction = None
    stage2_confidence = None
    stage2_probs = {}
    final_prediction = None
    final_confidence = None

    # 4. HIERARCHICAL ROUTING
    if not is_disordered:
        # TD Path
        final_prediction = "Typical Developing"
        final_confidence = prob_td
        stage2_prediction = "N/A (Screened Typical)"
        stage2_confidence = 0.0
    else:
        # SSD Path: Run Stage 2 Disorder Classifier
        if stage2_model is None:
            # Fallback if Stage 2 weights not yet trained
            final_prediction = "Speech Sound Disorder (Unspecified)"
            final_confidence = prob_ssd
            stage2_prediction = "Model weights pending training"
            stage2_confidence = prob_ssd
        else:
            with torch.no_grad():
                s2_logits = stage2_model(pooled_emb, age_tensor)
                s2_probs = torch.softmax(s2_logits, dim=-1)[0].cpu().numpy()

            stage2_probs = {
                cls_name: float(s2_probs[i]) * 100.0
                for i, cls_name in enumerate(DISORDER_CLASSES)
            }
            best_disorder_idx = int(np.argmax(s2_probs))
            predicted_disorder_raw = DISORDER_CLASSES[best_disorder_idx]
            disorder_cond_prob = float(s2_probs[best_disorder_idx])

            stage2_prediction = predicted_disorder_raw.capitalize()
            stage2_confidence = disorder_cond_prob

            # Combined Hierarchical Confidence: P(SSD) * P(Disorder | SSD)
            combined_hierarchical_conf = prob_ssd * disorder_cond_prob
            final_prediction = stage2_prediction
            final_confidence = combined_hierarchical_conf

    # 5. Display Clinical Output
    print("\n" + "="*60)
    print("      PEDIATRIC SPEECH ASSESSMENT")
    print("="*60)
    print(f"Age: {float(age):.2f} years (Normalized: {norm_age_clamped:.4f})")
    print(f"Audio File: {os.path.basename(audio_path)} ({duration:.2f}s, 16,000 Hz Mono)")
    print("-"*60)
    print("Stage 1:")
    print("TD vs SSD Screening")
    print(f"Prediction: {'SSD' if is_disordered else 'TD'}")
    print(f"Confidence: {stage1_confidence * 100.0:.1f}%")
    print("-"*60)
    print("Stage 2:")
    print("Disorder Classification")
    print(f"Prediction: {stage2_prediction}")
    print(f"Confidence: {stage2_confidence * 100.0:.1f}%" if is_disordered else "Confidence: N/A (Typical)")
    print("-"*60)
    print("Final Prediction:")
    print(f"{final_prediction}")
    print("\nFinal Confidence:")
    print(f"{final_confidence * 100.0:.1f}%")
    if is_disordered and stage2_probs:
        print("-"*60)
        print("Stage 2 Symptom Breakdown:")
        for c, p in stage2_probs.items():
            print(f"  • {c.capitalize():14s}: {p:.2f}%")
    print("="*60 + "\n")

    return {
        "audio_file": os.path.basename(audio_path),
        "duration": float(duration),
        "age": float(age),
        "normalized_age": float(norm_age_clamped),
        "stage1": {
            "prediction": stage1_prediction,
            "is_disordered": bool(is_disordered),
            "confidence": float(stage1_confidence * 100.0),
            "prob_ssd": float(prob_ssd * 100.0),
            "prob_td": float(prob_td * 100.0)
        },
        "stage2": {
            "prediction": stage2_prediction,
            "confidence": float(stage2_confidence * 100.0) if stage2_confidence is not None else 0.0,
            "probabilities": stage2_probs
        },
        "final_prediction": final_prediction,
        "final_confidence": float(final_confidence * 100.0),
        "model_architecture": stage1_meta.get("architecture_type", "Frozen Baseline"),
        "model_description": stage1_meta.get("architecture_description", "Frozen Baseline (WavLM-Large + AAFF)"),
        "stage1_meta": stage1_meta,
        "hierarchical_explanation": (
            f"Stage 1 detected {stage1_prediction} with {stage1_confidence*100.0:.1f}% certainty. "
            + (f"Stage 2 subtyped the symptom as {stage2_prediction} ({stage2_confidence*100.0:.1f}% conditional). Combined hierarchical confidence: {final_confidence*100.0:.1f}%."
               if is_disordered else "Screened as Typically Developing; Stage 2 disorder classifier bypassed.")
        )
    }

# Backward compatibility alias
def predict_speech_disorder(audio_path, age, checkpoint_path=None, model_checkpoint="microsoft/wavlm-large"):
    res = predict_hierarchical_speech_disorder(
        audio_path=audio_path,
        age=age,
        stage1_checkpoint=checkpoint_path or "./results_controlled_baselines/model_c_aaff_best.pt",
        stage2_checkpoint="./results_controlled_baselines/model_stage2_disorder_best.pt",
        model_checkpoint=model_checkpoint
    )
    return {
        "prediction": res["final_prediction"],
        "confidence": res["final_confidence"],
        "prob_disorder": res["stage1"]["prob_ssd"],
        "prob_typical": res["stage1"]["prob_td"],
        "duration": res["duration"],
        "normalized_age": res["normalized_age"],
        "model_architecture": res.get("model_architecture", "Frozen Baseline"),
        "model_description": res.get("model_description", "Frozen Baseline (WavLM-Large + AAFF)"),
        "hierarchical_details": res
    }

def main():
    parser = argparse.ArgumentParser(description="Two-Stage Hierarchical Pediatric Speech Assessment")
    parser.add_argument("--audio", type=str, required=True, help="Path to .wav audio file")
    parser.add_argument("--age", type=float, required=True, help="Child's chronological age (in decimal years)")
    parser.add_argument("--stage1_checkpoint", type=str, default="./results_controlled_baselines/model_c_aaff_best.pt", help="Stage 1 Binary Model weights")
    parser.add_argument("--stage2_checkpoint", type=str, default="./results_controlled_baselines/model_stage2_disorder_best.pt", help="Stage 2 Multiclass Disorder weights")
    parser.add_argument("--model_checkpoint", type=str, default="microsoft/wavlm-large", help="WavLM HuggingFace checkpoint")
    args = parser.parse_args()

    predict_hierarchical_speech_disorder(
        audio_path=args.audio,
        age=args.age,
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        model_checkpoint=args.model_checkpoint
    )

if __name__ == "__main__":
    main()
