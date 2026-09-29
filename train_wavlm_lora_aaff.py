"""
train_wavlm_lora_aaff.py
========================
End-to-End Fine-Tuning of WavLM-Large with Age-Aware Feature Fusion (AAFF)
for Pediatric Speech Assessment.

Strategy:
  - Primary: Unfreeze top N transformer encoder layers for direct gradient flow
  - Secondary (optional): LoRA on feed-forward layers (intermediate_dense, output_dense)
    NOTE: WavLM's attention (q_proj/v_proj) uses F.multi_head_attention_forward which
    bypasses LoRA-wrapped layers, so we target feed-forward layers instead.
  - AAFF head: 1024-d WavLM pooled embedding + 1-d normalized age -> 256 -> N classes
  - Class-balanced weighted Cross-Entropy loss
  - Speaker-disjoint train/val/test splits (zero speaker leakage)
  - Gender-stratified Pitch Augmentation
  - Differential learning rates (backbone vs. fresh AAFF head)
  - Gradient and sanity check mode (--check_only)
"""

import os
import re
import sys
import json
import random
import argparse
import datetime
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from datasets import load_dataset, Audio, ClassLabel, concatenate_datasets
import peft
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    TrainingArguments,
    Trainer,
    set_seed,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split as sklearn_split
from audiomentations import PitchShift, AddGaussianNoise
import wandb

warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")
warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask")

TEST_SPLIT_SEED = 42

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune WavLM with LoRA and AAFF Head")
    
    # Task & Model Setup
    parser.add_argument("--target_column", type=str, default="disorder_class",
                        choices=["disorder_class", "disorder_type", "disorder_symptom"],
                        help="Target classification column (disorder_class for T1 screening)")
    parser.add_argument("--model_checkpoint", type=str, default="microsoft/wavlm-large",
                        help="HuggingFace model identifier")
    parser.add_argument("--exclude_healthy", type=str2bool, default=False,
                        help="Exclude typically_developing (auto-set for T2/T3)")

    # LoRA Configuration
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj",
                        help="Comma-separated LoRA target module names (default: q_proj,v_proj; also supports k_proj,out_proj)")
    parser.add_argument("--unfreeze_top_layers", type=int, default=0,
                        help="Number of top transformer encoder blocks to unfreeze (0 = LoRA-only backbone)")

    # AAFF Head Configuration
    parser.add_argument("--proj_dim", type=int, default=256, help="Intermediate projection dimension")
    parser.add_argument("--head_dropout", type=float, default=0.2, help="Dropout in AAFF projection head")

    # Optimization & Differential Learning Rates
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Per-device eval batch size")
    parser.add_argument("--lora_lr", "--learning_rate", dest="lora_lr", type=float, default=1e-4, help="Learning rate for LoRA adapters")
    parser.add_argument("--head_lr", "--head_learning_rate", dest="head_lr", type=float, default=5e-4, help="Learning rate for fresh AAFF head")
    parser.add_argument("--unfrozen_lr", type=float, default=1e-5, help="Learning rate for newly unfrozen WavLM top layers")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_accumulation", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--oversample_mult", type=int, default=5, help="Capped oversampling factor")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience")

    # Audio Augmentation
    parser.add_argument("--pitch_prob", type=float, default=0.5, help="Pitch shift probability")
    parser.add_argument("--aug_male", type=str2bool, default=True, help="Augment male audio")
    parser.add_argument("--aug_female", type=str2bool, default=True, help="Augment female audio")
    parser.add_argument("--noise_prob", type=float, default=0.0, help="Noise augmentation probability")
    parser.add_argument("--pitch_shift_min", type=int, default=6)
    parser.add_argument("--pitch_shift_max", type=int, default=8)

    # Output & Tracking
    parser.add_argument("--output_dir", type=str, default="./results_wavlm_lora_aaff", help="Output directory")
    parser.add_argument("--wandb_project", type=str, default="SLP-Gender-CDA", help="W&B project")
    parser.add_argument("--wandb_mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--check_only", type=str2bool, default=False, help="Run gradient & architecture checks only")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Debug limit for training samples")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Debug limit for eval samples")
    parser.add_argument("--test_reload", type=str2bool, default=True, help="Verify checkpoint reload after saving")

    return parser.parse_args()


# ==============================================================================
# Model Architecture: WavLM + LoRA + Age-Aware Feature Fusion (AAFF)
# ==============================================================================

class WavLMLoRA_AAFFModel(nn.Module):
    """
    End-to-End Model combining:
      1. WavLM-Large backbone with LoRA parameter-efficient adapters
      2. Temporal attention-pooling across non-padded speech frames
      3. Fusion with normalized chronological age (1024-d speech + 1-d age = 1025-d)
      4. Nonlinear projection MLP (1025 -> proj_dim -> ReLU -> Dropout)
      5. Classification head (proj_dim -> num_classes)
    """
    def __init__(self, backbone, num_labels, proj_dim=256, dropout=0.2):
        super().__init__()
        self.wavlm = backbone  # PeftModel wrapping WavLMModel
        self.config = backbone.config
        self.config.num_labels = num_labels
        self.proj_dim = proj_dim

        # Age-Aware Feature Fusion (AAFF) Head
        self.projection = nn.Sequential(
            nn.Linear(1025, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(proj_dim, num_labels)

    def forward(
        self,
        input_values=None,
        attention_mask=None,
        age=None,
        labels=None,
        **kwargs
    ):
        # Forward pass through LoRA-adapted WavLM
        outputs = self.wavlm(
            input_values=input_values,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state  # [batch, time, 1024]

        # Receptive-field aligned attention pooling
        if attention_mask is not None:
            raw_backbone = self.wavlm.get_base_model() if hasattr(self.wavlm, "get_base_model") else self.wavlm
            if hasattr(raw_backbone, "model"):
                raw_backbone = raw_backbone.model

            feature_mask = raw_backbone._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask
            )
            mask = feature_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)

        # Ensure age tensor is [batch, 1]
        if age is not None:
            age = age.float().view(-1, 1)
        else:
            age = torch.zeros((pooled.size(0), 1), device=pooled.device, dtype=torch.float32)

        # Concatenate 1024-d acoustic features with 1-d chronological age -> 1025-d
        fused = torch.cat([pooled, age], dim=1)

        # Project and classify
        projected = self.projection(fused)
        logits = self.classifier(projected)

        return {"logits": logits, "pooled": pooled}


# ==============================================================================
# Custom Weighted Trainer with Differential Learning Rates
# ==============================================================================

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, head_lr=5e-4, lora_lr=1e-4, unfrozen_lr=1e-5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.head_lr = head_lr
        self.lora_lr = lora_lr
        self.unfrozen_lr = unfrozen_lr

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            weights = self.class_weights.to(labels.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """
        Differential learning rate optimizer with 3 distinct parameter groups:
          1. LoRA parameters (lora_lr, e.g. 1e-4)
          2. Newly unfrozen WavLM top layers if any (unfrozen_lr, e.g. 1e-5)
          3. Fresh AAFF head (head_lr, e.g. 5e-4)
        Frozen parameters are strictly excluded.
        """
        if self.optimizer is None:
            head_params = list(self.model.projection.parameters()) + list(self.model.classifier.parameters())
            lora_params = [p for n, p in self.model.wavlm.named_parameters() if p.requires_grad and "lora_" in n]
            unfrozen_params = [p for n, p in self.model.wavlm.named_parameters() if p.requires_grad and "lora_" not in n]

            optimizer_grouped_parameters = [
                {
                    "params": lora_params,
                    "lr": self.lora_lr,
                    "weight_decay": self.args.weight_decay
                },
                {
                    "params": head_params,
                    "lr": self.head_lr,
                    "weight_decay": self.args.weight_decay
                }
            ]
            if len(unfrozen_params) > 0:
                optimizer_grouped_parameters.append({
                    "params": unfrozen_params,
                    "lr": self.unfrozen_lr,
                    "weight_decay": self.args.weight_decay
                })
            
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        return self.optimizer


def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_micro": acc,
        "precision_macro": p_mac,
        "recall_macro": r_mac
    }


def balance_dataset(ds, target_col, max_multiplier):
    labels = ds[target_col]
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1

    if not counts:
        return ds

    majority_count = max(counts.values())
    dfs = [ds]

    print(f"\n[Balancing] Initial Distribution: {counts}")
    print(f"[Balancing] Capped Multiplier: {max_multiplier}x")

    for label, count in counts.items():
        if count == majority_count:
            continue
        ratio = majority_count / count
        multiplier = min(int(round(ratio)), max_multiplier)
        if multiplier > 1:
            print(f"   -> Oversampling '{label}': {count} -> {multiplier}x (~{count*multiplier})")
            minority_ds = ds.filter(lambda x: x[target_col] == label)
            for _ in range(multiplier - 1):
                dfs.append(minority_ds)

    if len(dfs) > 1:
        balanced = concatenate_datasets(dfs)
        print(f"[Balancing] Final Dataset Size: {len(balanced)}")
        return balanced
    return ds


# ==============================================================================
# Helper Functions: Architecture Construction, Parameter Audit & Verification
# ==============================================================================

def build_wavlm_lora_aaff_model(
    model_checkpoint="microsoft/wavlm-large",
    num_labels=2,
    proj_dim=256,
    head_dropout=0.2,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules="q_proj,v_proj",
    unfreeze_top_layers=0
):
    """
    Constructs WavLMLoRA_AAFFModel with deterministic layer unfreezing:
    1. Loads base WavLM model and freezes all base parameters.
    2. Injects PEFT LoRA adapters into target_modules (default: q_proj,v_proj).
    3. If unfreeze_top_layers > 0, unfreezes the last N transformer encoder blocks.
    4. Attaches the AAFF head (1025 -> proj_dim -> num_labels).
    """
    base_wavlm = AutoModel.from_pretrained(model_checkpoint, use_safetensors=True)
    for param in base_wavlm.parameters():
        param.requires_grad = False

    target_mods = [m.strip() for m in target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_mods,
        lora_dropout=lora_dropout,
        bias="none"
    )
    lora_wavlm = get_peft_model(base_wavlm, lora_config)

    # Deterministic top-layer unfreezing
    raw = lora_wavlm.get_base_model()
    if hasattr(raw, "model"):
        raw = raw.model

    unfrozen_layers = []
    if unfreeze_top_layers > 0:
        if hasattr(raw, "encoder") and hasattr(raw.encoder, "layers"):
            total_layers = len(raw.encoder.layers)
            start_idx = max(0, total_layers - unfreeze_top_layers)
            for idx in range(start_idx, total_layers):
                for p in raw.encoder.layers[idx].parameters():
                    p.requires_grad = True
                unfrozen_layers.append(f"encoder.layers[{idx}]")

    model = WavLMLoRA_AAFFModel(
        backbone=lora_wavlm,
        num_labels=num_labels,
        proj_dim=proj_dim,
        dropout=head_dropout
    )
    return model, unfrozen_layers


def get_parameter_breakdown(model):
    """
    Calculates exact parameter counts across:
      - Frozen WavLM parameters
      - Unfrozen top WavLM parameters
      - LoRA adapter parameters
      - AAFF head parameters
      - Total parameters & percentage trainable
    """
    wavlm_total = sum(p.numel() for p in model.wavlm.parameters())
    wavlm_lora = sum(p.numel() for n, p in model.wavlm.named_parameters() if p.requires_grad and "lora_" in n)
    wavlm_unfrozen = sum(p.numel() for n, p in model.wavlm.named_parameters() if p.requires_grad and "lora_" not in n)
    wavlm_frozen = sum(p.numel() for n, p in model.wavlm.named_parameters() if not p.requires_grad)
    head_trainable = sum(p.numel() for p in model.projection.parameters()) + sum(p.numel() for p in model.classifier.parameters())
    total_trainable = wavlm_lora + wavlm_unfrozen + head_trainable
    total_params = wavlm_total + head_trainable

    return {
        "wavlm_total": wavlm_total,
        "wavlm_frozen": wavlm_frozen,
        "wavlm_unfrozen": wavlm_unfrozen,
        "wavlm_lora": wavlm_lora,
        "head_trainable": head_trainable,
        "total_trainable": total_trainable,
        "total_params": total_params,
        "trainable_pct": 100.0 * total_trainable / max(1, total_params)
    }


def verify_gradients_for_mode(model, sample_batch, dev, mode_name, expected_unfrozen_layers=0):
    """
    Performs forward + backward on a single sample and verifies:
      - Lower / frozen original WavLM parameters have ZERO gradients
      - If expected_unfrozen_layers > 0, top encoder layers have NON-ZERO gradients
      - LoRA adapter parameters have NON-ZERO gradients
      - AAFF head has NON-ZERO gradients
    """
    model.to(dev)
    model.train()
    model.zero_grad()

    test_input = torch.tensor(sample_batch["input_values"], dtype=torch.float32).unsqueeze(0).to(dev)
    test_attn = torch.tensor(sample_batch["attention_mask"], dtype=torch.long).unsqueeze(0).to(dev)
    test_age = torch.tensor([sample_batch["age"]], dtype=torch.float32).to(dev)
    test_lbl = torch.tensor([sample_batch["labels"]], dtype=torch.long).to(dev)

    test_out = model(input_values=test_input, attention_mask=test_attn, age=test_age)
    test_logits = test_out["logits"]
    loss_val = nn.CrossEntropyLoss()(test_logits, test_lbl)
    loss_val.backward()

    # LoRA gradients
    lora_grads = [p.grad.abs().sum().item() for n, p in model.wavlm.named_parameters() if "lora_" in n and p.grad is not None]
    lora_has_grad = len(lora_grads) > 0 and any(g > 0 for g in lora_grads)

    # Base frozen gradients
    base_frozen_grads = [p.grad.abs().sum().item() for n, p in model.wavlm.named_parameters() if "lora_" not in n and not p.requires_grad and p.grad is not None]
    base_frozen_clean = len(base_frozen_grads) == 0 or all(g == 0 for g in base_frozen_grads)

    # Unfrozen top-layer gradients
    unfrozen_params = [(n, p) for n, p in model.wavlm.named_parameters() if "lora_" not in n and p.requires_grad]
    if expected_unfrozen_layers > 0:
        unfrozen_grads = [p.grad.abs().sum().item() for n, p in unfrozen_params if p.grad is not None]
        unfrozen_has_grad = len(unfrozen_grads) > 0 and any(g > 0 for g in unfrozen_grads)
    else:
        unfrozen_has_grad = (len(unfrozen_params) == 0)

    # Head gradients
    head_has_grad = (
        model.projection[0].weight.grad is not None and model.projection[0].weight.grad.abs().sum().item() > 0
        and model.classifier.weight.grad is not None and model.classifier.weight.grad.abs().sum().item() > 0
    )

    model.zero_grad()
    return {
        "lora_has_grad": lora_has_grad,
        "base_frozen_clean": base_frozen_clean,
        "unfrozen_has_grad": unfrozen_has_grad,
        "head_has_grad": head_has_grad,
        "passed": lora_has_grad and base_frozen_clean and unfrozen_has_grad and head_has_grad
    }


def verify_checkpoint_reload(checkpoint_path, original_model, sample_batch, dev):
    """
    Loads the saved checkpoint into a fresh model structure and validates that
    the reloaded model produces identical logits to original_model on a sample.
    """
    print("\n[Checkpoint Verification] Reloading checkpoint to verify serialization integrity...")
    data = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    
    reloaded_model, _ = build_wavlm_lora_aaff_model(
        model_checkpoint=data["model_checkpoint"],
        num_labels=data["num_classes"],
        proj_dim=data["proj_dim"],
        head_dropout=data.get("head_dropout", 0.2),
        lora_r=data.get("lora_r", 8),
        lora_alpha=data.get("lora_alpha", 16),
        target_modules=data.get("target_modules", "q_proj,v_proj"),
        unfreeze_top_layers=data.get("unfreeze_top_layers", 0)
    )
    reloaded_model.to(dev)

    # Load LoRA weights
    if "lora_state_dict" in data and hasattr(reloaded_model.wavlm, "load_state_dict"):
        peft.set_peft_model_state_dict(reloaded_model.wavlm, data["lora_state_dict"])

    # Load unfrozen backbone weights if any
    if "backbone_trainable_state" in data:
        for name, param in reloaded_model.wavlm.named_parameters():
            if name in data["backbone_trainable_state"]:
                param.data.copy_(data["backbone_trainable_state"][name])

    # Load head weights
    reloaded_model.projection.load_state_dict(data["head_state_dict"]["projection"])
    reloaded_model.classifier.load_state_dict(data["head_state_dict"]["classifier"])

    original_model.eval()
    reloaded_model.eval()

    test_input = torch.tensor(sample_batch["input_values"], dtype=torch.float32).unsqueeze(0).to(dev)
    test_attn = torch.tensor(sample_batch["attention_mask"], dtype=torch.long).unsqueeze(0).to(dev)
    test_age = torch.tensor([sample_batch["age"]], dtype=torch.float32).to(dev)

    with torch.no_grad():
        out_orig = original_model(input_values=test_input, attention_mask=test_attn, age=test_age)["logits"]
        out_reload = reloaded_model(input_values=test_input, attention_mask=test_attn, age=test_age)["logits"]

    is_close = torch.allclose(out_orig, out_reload, atol=1e-4)
    print(f"   Original Logits : {out_orig.cpu().numpy().tolist()}")
    print(f"   Reloaded Logits : {out_reload.cpu().numpy().tolist()}")
    print(f"   Match status    : {'PERFECT MATCH (Passed)' if is_close else 'MISMATCH (Failed)'}")
    if not is_close:
        raise ValueError("Reloaded checkpoint logits do not match original model logits!")


# ==============================================================================
# Main Execution Pipeline
# ==============================================================================

def main():
    args = parse_arguments()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup Seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    set_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    print("=" * 70)
    print("      END-TO-END WAVLM LoRA + AAFF FINE-TUNING")
    print("=" * 70)
    print(f"Target Column        : {args.target_column}")
    print(f"Base Backbone        : {args.model_checkpoint}")
    print(f"LoRA Rank (r)        : {args.lora_r}")
    print(f"LoRA Alpha           : {args.lora_alpha}")
    print(f"Target Modules       : {args.target_modules}")
    print(f"Unfreeze Top Layers  : {args.unfreeze_top_layers}")
    print(f"LoRA Learning Rate   : {args.lora_lr}")
    print(f"AAFF Head LR         : {args.head_lr}")
    print(f"Unfrozen WavLM LR    : {args.unfrozen_lr}")
    print(f"Device               : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Random Seed          : {args.random_seed}")
    print("=" * 70)

    # 1. Load Dataset
    print("\n[Step 1/6] Loading Dataset SAA-Lab/SLPHelmUltraSuitePlus...")
    dataset = load_dataset("SAA-Lab/SLPHelmUltraSuitePlus", split="train")

    # Extract Speaker IDs from paths
    ds_meta = dataset.cast_column("audio", Audio(decode=False))
    paths = [item["audio"]["path"] for item in ds_meta]
    pattern = re.compile(r"^(?:test_|train_|val_|audio_)?([a-zA-Z0-9]+)[_\-\.]", re.IGNORECASE)
    speaker_ids = [pattern.search(os.path.basename(p)).group(1) if pattern.search(os.path.basename(p)) else os.path.basename(p) for p in paths]
    dataset = dataset.add_column("speaker_id", speaker_ids)

    # Clean & standardize labels
    def clean_labels(batch):
        val = batch[args.target_column]
        if val is None:
            return batch
        val = str(val).replace("['", "").replace("']", "").replace("[", "").replace("]", "").strip()
        if args.target_column == "disorder_type":
            if val in ["addition", "omission"]: val = "articulation"
            elif val == "stuttering": val = "phonological"
        elif args.target_column == "disorder_symptom":
            if val in ["substitution_modified", "phonological"]: val = "stuttering"
        return {args.target_column: val}

    dataset = dataset.map(clean_labels)
    dataset = dataset.filter(lambda x: x[args.target_column] is not None)

    # Exclude healthy for T2 / T3 tasks
    if args.target_column in ["disorder_symptom", "disorder_type"]:
        args.exclude_healthy = True
    if args.exclude_healthy:
        print("[Filter] Excluding 'typically_developing' samples...")
        dataset = dataset.filter(lambda x: str(x[args.target_column]).strip().lower() != "typically_developing")

    # 2. Speaker-Disjoint Splitting (Strict Leakage Prevention)
    df = dataset.to_pandas()
    unique_speakers = sorted(df["speaker_id"].unique())
    print(f"Unique Speaker IDs identified: {len(unique_speakers)}")

    train_val_spk, test_spk = sklearn_split(unique_speakers, test_size=0.20, random_state=TEST_SPLIT_SEED)
    train_spk, val_spk = sklearn_split(train_val_spk, test_size=0.20, random_state=TEST_SPLIT_SEED)

    assert set(train_spk).isdisjoint(set(val_spk)), "Leakage between Train and Val"
    assert set(train_spk).isdisjoint(set(test_spk)), "Leakage between Train and Test"
    assert set(val_spk).isdisjoint(set(test_spk)), "Leakage between Val and Test"

    ds_train = dataset.filter(lambda x: x["speaker_id"] in train_spk)
    ds_val   = dataset.filter(lambda x: x["speaker_id"] in val_spk)
    ds_test  = dataset.filter(lambda x: x["speaker_id"] in test_spk)

    # Encode Labels
    all_labels = set(ds_train[args.target_column]) | set(ds_val[args.target_column]) | set(ds_test[args.target_column])
    unique_labels = sorted(list(all_labels))
    label_feature = ClassLabel(names=unique_labels)
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for i, l in enumerate(unique_labels)}

    ds_train = ds_train.map(lambda x: {"labels": label_feature.str2int(x[args.target_column])})
    ds_val   = ds_val.map(lambda x: {"labels": label_feature.str2int(x[args.target_column])})
    ds_test  = ds_test.map(lambda x: {"labels": label_feature.str2int(x[args.target_column])})

    print(f"\n[Dataset Splits] (Speaker Disjoint)")
    print(f"   Train : {len(ds_train)} samples")
    print(f"   Val   : {len(ds_val)} samples")
    print(f"   Test  : {len(ds_test)} samples")
    print(f"   Labels: {unique_labels} ({label2id})")

    # Optional fast debug subset
    if args.max_train_samples:
        ds_train = ds_train.select(range(min(args.max_train_samples, len(ds_train))))
    if args.max_eval_samples:
        ds_val = ds_val.select(range(min(args.max_eval_samples, len(ds_val))))
        ds_test = ds_test.select(range(min(args.max_eval_samples, len(ds_test))))

    # Age Normalization (Derived purely from training split)
    train_ages = np.array([float(x) for x in ds_train["age"]])
    age_min = float(train_ages.min())
    age_max = float(train_ages.max())
    print(f"\n[Age Normalization] Training bounds: [{age_min:.2f}, {age_max:.2f}] years")

    def normalize_age(a):
        if age_max == age_min:  # degenerate case (e.g. all debug samples same age)
            return 0.5
        val = (float(a) - age_min) / (age_max - age_min)
        return max(0.0, min(1.0, val))

    # Apply Train Oversampling / Balancing
    ds_train = balance_dataset(ds_train, args.target_column, args.oversample_mult)

    # Calculate Balanced Class Weights
    train_labels = ds_train["labels"]
    class_counts = np.bincount(train_labels, minlength=len(unique_labels))
    class_counts = np.maximum(class_counts, 1)
    computed_weights = len(train_labels) / (len(unique_labels) * class_counts)
    class_weights_tensor = torch.tensor(computed_weights, dtype=torch.float32)

    print("\n[Class Loss Weights]:")
    for idx, name in id2label.items():
        print(f"   - {name:20s}: {class_weights_tensor[idx]:.2f}x (Count: {class_counts[idx]})")

    # 3. Audio Preprocessor & Feature Extractor
    print(f"\n[Step 2/6] Loading Feature Extractor: {args.model_checkpoint}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_checkpoint)
    target_sr = feature_extractor.sampling_rate
    max_duration = 12.0

    ds_train = ds_train.cast_column("audio", Audio(sampling_rate=target_sr))
    ds_val   = ds_val.cast_column("audio", Audio(sampling_rate=target_sr))
    ds_test  = ds_test.cast_column("audio", Audio(sampling_rate=target_sr))

    p_min = min(args.pitch_shift_min, args.pitch_shift_max)
    p_max = max(args.pitch_shift_min, args.pitch_shift_max)
    augmenter_pitch_up = PitchShift(min_semitones=p_min, max_semitones=p_max, p=1.0)
    augmenter_pitch_down = PitchShift(min_semitones=-p_max, max_semitones=-p_min, p=1.0)
    augmenter_noise = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)

    def preprocess_train(batch):
        audio_arrays = [x["array"] for x in batch["audio"]]
        genders = batch["gender"]
        processed = []
        for x, g in zip(audio_arrays, genders):
            if np.random.rand() < args.pitch_prob:
                if g == 'M' and args.aug_male:
                    x = augmenter_pitch_up(samples=x, sample_rate=target_sr)
                elif g == 'F' and args.aug_female:
                    x = augmenter_pitch_down(samples=x, sample_rate=target_sr)
            if args.noise_prob > 0 and np.random.rand() < args.noise_prob:
                x = augmenter_noise(samples=x, sample_rate=target_sr)
            processed.append(x)

        inputs = feature_extractor(
            processed,
            sampling_rate=target_sr,
            max_length=int(target_sr * max_duration),
            truncation=True,
            padding=True
        )
        inputs["labels"] = batch["labels"]
        inputs["age"] = [normalize_age(a) for a in batch["age"]]
        return inputs

    def preprocess_eval(batch):
        audio_arrays = [x["array"] for x in batch["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=target_sr,
            max_length=int(target_sr * max_duration),
            truncation=True,
            padding=True
        )
        inputs["labels"] = batch["labels"]
        inputs["age"] = [normalize_age(a) for a in batch["age"]]
        return inputs

    ds_train.set_transform(preprocess_train)
    ds_val.set_transform(preprocess_eval)
    ds_test.set_transform(preprocess_eval)

    # 4. Construct LoRA WavLM Model
    print(f"\n[Step 3/6] Initializing WavLM Backbone & LoRA Adapters (unfreeze_top_layers={args.unfreeze_top_layers})...")
    model, unfrozen_layers = build_wavlm_lora_aaff_model(
        model_checkpoint=args.model_checkpoint,
        num_labels=len(unique_labels),
        proj_dim=args.proj_dim,
        head_dropout=args.head_dropout,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        unfreeze_top_layers=args.unfreeze_top_layers
    )

    counts = get_parameter_breakdown(model)
    print(f"\n[Parameter Audit - Active Configuration]")
    print(f"   Total Parameters                 : {counts['total_params']:,}")
    print(f"   Original WavLM Frozen Parameters : {counts['wavlm_frozen']:,}")
    print(f"   Top Unfrozen WavLM Parameters    : {counts['wavlm_unfrozen']:,}")
    print(f"   LoRA Trainable Parameters        : {counts['wavlm_lora']:,}")
    print(f"   AAFF Head Trainable Parameters   : {counts['head_trainable']:,}")
    print(f"   Total Trainable Parameters       : {counts['total_trainable']:,} ({counts['trainable_pct']:.2f}%)")
    print(f"   Unfrozen WavLM Layers            : {unfrozen_layers if unfrozen_layers else 'None (Fully Frozen Backbone)'}")

    # 5. Gradient & Architecture Sanity Check
    print("\n[Step 4/6] Executing Architecture & Gradient Verification...")
    sample = ds_train[0]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.check_only:
        print("\n" + "=" * 70)
        print("   DUAL-MODE GRADIENT & ARCHITECTURE VERIFICATION (--check_only True)")
        print("=" * 70)

        # MODE A: unfreeze_top_layers = 0 (LoRA only)
        print("\n[VERIFYING MODE A: LoRA Only (--unfreeze_top_layers 0)]")
        model_a, unfrozen_a = build_wavlm_lora_aaff_model(
            model_checkpoint=args.model_checkpoint,
            num_labels=len(unique_labels),
            proj_dim=args.proj_dim,
            head_dropout=args.head_dropout,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            unfreeze_top_layers=0
        )
        counts_a = get_parameter_breakdown(model_a)
        print(f"   Original WavLM frozen parameters : {counts_a['wavlm_frozen']:,}")
        print(f"   Top unfrozen WavLM parameters    : {counts_a['wavlm_unfrozen']:,}")
        print(f"   LoRA trainable parameters        : {counts_a['wavlm_lora']:,}")
        print(f"   AAFF trainable parameters        : {counts_a['head_trainable']:,}")
        print(f"   Total trainable parameters       : {counts_a['total_trainable']:,} ({counts_a['trainable_pct']:.2f}%)")
        print(f"   Unfrozen WavLM layers            : {unfrozen_a if unfrozen_a else 'None (Fully Frozen Backbone)'}")

        res_a = verify_gradients_for_mode(model_a, sample, dev, "Mode A (LoRA Only)", expected_unfrozen_layers=0)
        print(f"   - Original WavLM parameters frozen clean (no grad) : {res_a['base_frozen_clean']}")
        print(f"   - LoRA parameters gradient flow                    : {res_a['lora_has_grad']}")
        print(f"   - AAFF head gradient flow                          : {res_a['head_has_grad']}")
        print(f"   -> MODE A Status: {'PASSED' if res_a['passed'] else 'FAILED'}")

        # MODE B: unfreeze_top_layers = 2 (LoRA + Top 2 Layers)
        print("\n[VERIFYING MODE B: LoRA + Top 2 Layers (--unfreeze_top_layers 2)]")
        model_b, unfrozen_b = build_wavlm_lora_aaff_model(
            model_checkpoint=args.model_checkpoint,
            num_labels=len(unique_labels),
            proj_dim=args.proj_dim,
            head_dropout=args.head_dropout,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            unfreeze_top_layers=2
        )
        counts_b = get_parameter_breakdown(model_b)
        print(f"   Original WavLM frozen parameters : {counts_b['wavlm_frozen']:,}")
        print(f"   Top unfrozen WavLM parameters    : {counts_b['wavlm_unfrozen']:,}")
        print(f"   LoRA trainable parameters        : {counts_b['wavlm_lora']:,}")
        print(f"   AAFF trainable parameters        : {counts_b['head_trainable']:,}")
        print(f"   Total trainable parameters       : {counts_b['total_trainable']:,} ({counts_b['trainable_pct']:.2f}%)")
        print(f"   Unfrozen WavLM layers            : {unfrozen_b}")

        res_b = verify_gradients_for_mode(model_b, sample, dev, "Mode B (LoRA + Top 2)", expected_unfrozen_layers=2)
        print(f"   - Lower WavLM layers frozen clean (no grad)        : {res_b['base_frozen_clean']}")
        print(f"   - Top 2 WavLM layers gradient flow                 : {res_b['unfrozen_has_grad']}")
        print(f"   - LoRA parameters gradient flow                    : {res_b['lora_has_grad']}")
        print(f"   - AAFF head gradient flow                          : {res_b['head_has_grad']}")
        print(f"   -> MODE B Status: {'PASSED' if res_b['passed'] else 'FAILED'}")

        if not res_a['passed'] or not res_b['passed']:
            raise RuntimeError("Gradient verification failed in one or more modes!")

        print("\n" + "=" * 70)
        print("   ALL GRADIENT & ARCHITECTURE CHECKS PASSED SUCCESSFULLY!")
        print("=" * 70)
        return

    # Non-check_only: verify single active model before launching trainer
    active_check = verify_gradients_for_mode(model, sample, dev, "Active Model", expected_unfrozen_layers=args.unfreeze_top_layers)
    if not active_check["passed"]:
        raise RuntimeError(f"Sanity check failed for active configuration: {active_check}")
    print("   Active architecture sanity check: PASSED successfully.")

    # 6. Training Pipeline
    print(f"\n[Step 5/6] Launching Training ({args.epochs} epochs)...")
    if args.wandb_mode != "disabled":
        wandb.init(project=args.wandb_project, mode=args.wandb_mode, config=vars(args))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lora_lr,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=(torch.cuda.is_available()),
        remove_unused_columns=False,
        report_to="wandb" if args.wandb_mode != "disabled" else "none",
        run_name=f"wavlm_lora_aaff_{args.target_column}_{timestamp}",
        logging_steps=10
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=feature_extractor,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
        head_lr=args.head_lr,
        lora_lr=args.lora_lr,
        unfrozen_lr=args.unfrozen_lr,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )

    train_result = trainer.train()
    print("\nTraining complete!")

    # 7. Test Evaluation & Serialization
    print("\n[Step 6/6] Final Evaluation & Checkpoint Serialization...")
    test_results = trainer.predict(ds_test)
    y_true = test_results.label_ids
    y_pred = np.argmax(test_results.predictions, axis=1)

    ds_test.reset_format()
    df_eval = pd.DataFrame({"true": y_true, "pred": y_pred, "gender": ds_test["gender"]})

    print("\n" + "=" * 65)
    print(f"GLOBAL TEST EVALUATION (WavLM LoRA + AAFF, Target: {args.target_column})")
    print("=" * 65)
    report_text = classification_report(y_true, y_pred, labels=list(range(len(unique_labels))), target_names=unique_labels, zero_division=0)
    print(report_text)

    # Save text report
    report_path = os.path.join(args.output_dir, "test_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"=== WAVLM LORA + AAFF EVALUATION REPORT ===\n")
        f.write(f"Backbone: {args.model_checkpoint} | LoRA r={args.lora_r}, alpha={args.lora_alpha}\n")
        f.write(f"Target: {args.target_column} | Unfreeze Top Layers: {args.unfreeze_top_layers}\n\n")
        f.write(report_text)

    # Gender Disparity Breakdown
    print("\n--- Gender Breakdown ---")
    for gender in ["M", "F"]:
        gdf = df_eval[df_eval["gender"] == gender]
        if len(gdf) > 0:
            print(f"\nGender {gender} (N={len(gdf)}):")
            g_rep = classification_report(list(gdf["true"]), list(gdf["pred"]), labels=list(range(len(unique_labels))), target_names=unique_labels, zero_division=0)
            print(g_rep)

    # Save Consolidated Checkpoint
    consolidated_path = os.path.join(args.output_dir, "model_c_wavlm_lora_aaff_best.pt")
    lora_state = peft.get_peft_model_state_dict(model.wavlm) if hasattr(model.wavlm, "peft_type") else {}
    checkpoint_data = {
        "model_type": "wavlm_lora_aaff",
        "model_checkpoint": args.model_checkpoint,
        "target_column": args.target_column,
        "unique_labels": unique_labels,
        "num_classes": len(unique_labels),
        "proj_dim": args.proj_dim,
        "head_dropout": args.head_dropout,
        "unfreeze_top_layers": args.unfreeze_top_layers,
        "unfrozen_layers": unfrozen_layers,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
        "lora_state_dict": lora_state,
        "backbone_trainable_state": {n: p.data.clone() for n, p in model.wavlm.named_parameters() if p.requires_grad and "lora_" not in n},
        "head_state_dict": {
            "projection": model.projection.state_dict(),
            "classifier": model.classifier.state_dict(),
        },
        "age_min": age_min,
        "age_max": age_max,
        "metrics": {
            "test_accuracy": float(accuracy_score(y_true, y_pred)),
            "test_macro_f1": float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2])
        }
    }
    torch.save(checkpoint_data, consolidated_path)
    print(f"\nConsolidated checkpoint saved to: {consolidated_path}")

    # Also save PEFT adapter directory if LoRA was used
    if lora_state:
        adapter_dir = os.path.join(args.output_dir, "wavlm_lora_adapter")
        model.wavlm.save_pretrained(adapter_dir)
        print(f"PEFT adapter directory saved to: {adapter_dir}")

    # Verify reload if requested
    if args.test_reload:
        verify_checkpoint_reload(consolidated_path, model, sample, dev)

    if args.wandb_mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()

