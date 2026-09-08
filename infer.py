import os
import argparse
import torch
import soundfile as sf
import numpy as np
import librosa
from torch import nn
from transformers import AutoFeatureExtractor, AutoModel

class AAFFClassifier(nn.Module):
    """
    Trained Age-Aware Feature Fusion (AAFF) Architecture:
    1024-d speech embedding + 1-d normalized age (1025-d)
    -> Linear(1025, 256) -> ReLU() -> Dropout(0.2) -> Linear(256, 2)
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

def predict_speech_disorder(audio_path, age, checkpoint_path=None, model_checkpoint="microsoft/wavlm-large"):
    if checkpoint_path is None:
        checkpoint_path = "./results_controlled_baselines/model_c_aaff_best.pt"

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Trained model checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Audio and Resample to 16,000 Hz Mono
    audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio_array) / 16000.0

    # 2. Audio Preprocessing (identical to training)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    target_sr = 16000
    max_duration = 12.0
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

    # 3. WavLM-Large Feature Extraction (Frozen)
    wavlm = AutoModel.from_pretrained(model_checkpoint, use_safetensors=True)
    wavlm.eval()
    wavlm.to(device)
    for param in wavlm.parameters():
        param.requires_grad = False

    with torch.no_grad():
        outputs = wavlm(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Temporal downsampling feature vector mask
        feature_mask = wavlm._get_feature_vector_attention_mask(
            hidden_states.shape[1],
            attention_mask
        )
        mask = feature_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    # 4. Age Normalization (using Training Set bounds [5.67, 12.83])
    age_min = 5.67
    age_max = 12.83
    norm_age = (float(age) - age_min) / (age_max - age_min)
    norm_age_clamped = max(0.0, min(1.0, norm_age))
    age_tensor = torch.tensor([[norm_age_clamped]], dtype=torch.float32).to(device)

    # 5. Load Trained AAFF Model and Predict
    model = AAFFClassifier(in_features=1025, proj_dim=256, dropout=0.2, num_classes=2).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(pooled, age_tensor)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    p_disorder = probs[0] * 100.0
    p_typical = probs[1] * 100.0

    prediction = "Speech Disorder" if probs[0] >= probs[1] else "Typically Developing"
    confidence = max(p_disorder, p_typical)

    # 6. Display Clinical Output
    print("\n" + "="*60)
    print("      AAFF CLINICAL DECISION SUPPORT - INFERENCE")
    print("="*60)
    print(f"Input Audio:        {os.path.basename(audio_path)}")
    print(f"File Path:          {audio_path}")
    print(f"Audio Duration:     {duration:.2f} seconds (16,000 Hz, Mono)")
    print(f"Child Age:          {float(age):.2f} years (Normalized: {norm_age_clamped:.4f})")
    print(f"Trained Checkpoint: {checkpoint_path}")
    print(f"WavLM Backbone:     {model_checkpoint} (Frozen 1024-d)")
    print("-"*60)
    print(f"FINAL PREDICTION:   {prediction.upper()}")
    print(f"CONFIDENCE:         {confidence:.2f}%")
    print("-"*60)
    print("Class Probabilities:")
    print(f"  • Typically Developing: {p_typical:.2f}%")
    print(f"  • Speech Disorder:       {p_disorder:.2f}%")
    print("="*60 + "\n")

    return {
        "prediction": prediction,
        "confidence": confidence,
        "prob_disorder": p_disorder,
        "prob_typical": p_typical,
        "duration": duration,
        "normalized_age": norm_age_clamped
    }

def main():
    parser = argparse.ArgumentParser(description="Test trained AAFF model with a single .wav file")
    parser.add_argument("--audio", type=str, required=True, help="Path to .wav audio file")
    parser.add_argument("--age", type=float, required=True, help="Child's chronological age (in decimal years)")
    parser.add_argument("--checkpoint", type=str, default="./results_controlled_baselines/model_c_aaff_best.pt", help="Path to trained AAFF weights")
    parser.add_argument("--model_checkpoint", type=str, default="microsoft/wavlm-large", help="WavLM HuggingFace model checkpoint")
    args = parser.parse_args()

    predict_speech_disorder(
        audio_path=args.audio,
        age=args.age,
        checkpoint_path=args.checkpoint,
        model_checkpoint=args.model_checkpoint
    )

if __name__ == "__main__":
    main()
