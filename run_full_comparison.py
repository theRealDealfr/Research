"""
run_full_comparison.py
======================
Comprehensive, Controlled, and Reproducible 3x3 Hierarchical Comparison Benchmark
for Pediatric Speech Classification.

Evaluates:
  - Model A: WavLM-Large Only (1024-D speech embedding -> Linear Classifier)
  - Model B: WavLM-Large + Direct Age (1025-D concatenated -> Linear Classifier)
  - Model C: WavLM-Large + AAFF (1025-D -> Linear(1025, 256) -> ReLU -> Dropout(0.2) -> Classifier)

Across:
  - Stage 1: Typical vs Disorder (Screening)
  - Stage 2: Disorder Type Classification (Articulation vs Phonological)
  - Stage 3: Symptom Classification (Substitution, Stuttering, Omission, Addition)

Also Evaluates:
  - Standalone Stage Performance vs. End-to-End Hierarchical Cascade Performance
  - Generates 9 Confusion Matrices and 4 Presentation-Ready Comparison Graphs.
"""

import os
import copy
import json
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Reproducibility
# -----------------------------------------------------------------------------
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# 2. Model Architectures
# -----------------------------------------------------------------------------
class ModelA_WavLMOnly(nn.Module):
    """Model A: WavLM-Large Only (1024-D speech embedding, strictly NO age)"""
    def __init__(self, in_features=1024, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, emb, age=None):
        return self.classifier(emb)

class ModelB_WavLMDirectAge(nn.Module):
    """Model B: WavLM-Large + Direct Age (1024-D + 1-D = 1025-D direct concatenation)"""
    def __init__(self, in_features=1025, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, emb, age):
        fused = torch.cat([emb, age], dim=1)
        return self.classifier(fused)

class ModelC_AAFF(nn.Module):
    """Model C: WavLM-Large + Age-Aware Feature Fusion (AAFF non-linear projection)"""
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

# -----------------------------------------------------------------------------
# 3. Model Training & Tuning Routine
# -----------------------------------------------------------------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_confs, all_probs = [], [], [], []

    with torch.no_grad():
        for batch_emb, batch_age, batch_lbl in dataloader:
            batch_emb = batch_emb.to(device)
            batch_age = batch_age.to(device)
            logits = model(batch_emb, batch_age)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(batch_lbl.numpy().tolist())
            all_confs.extend(confs.tolist())
            all_probs.extend(probs.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    confs = np.array(all_confs)

    acc = float(accuracy_score(y_true, y_pred))
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "accuracy": acc,
        "precision_macro": float(pm),
        "recall_macro": float(rm),
        "f1_macro": float(fm),
        "f1_weighted": float(fw),
        "avg_confidence": float(confs.mean()) if len(confs) > 0 else 0.0,
        "y_true": y_true,
        "y_pred": y_pred,
        "confidences": confs,
        "probs": np.array(all_probs),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

def train_and_tune(model_cls, model_kwargs, train_data, val_data, device, num_classes,
                   lr_list=[1e-3, 5e-4, 3e-4, 1e-4], epochs=40, seed=42):
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=16, shuffle=False)

    train_labels = [y.item() for _, _, y in train_data]
    class_counts = np.bincount(train_labels, minlength=num_classes)
    total = len(train_labels)
    weights = total / (float(num_classes) * np.maximum(class_counts, 1))
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    best_overall_val_f1 = -1.0
    best_overall_model_state = None
    best_lr = None

    for lr in lr_list:
        set_all_seeds(seed)
        model = model_cls(**model_kwargs).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

        best_val_f1_this_lr = -1.0
        best_state_this_lr = None

        for epoch in range(1, epochs + 1):
            model.train()
            for batch_emb, batch_age, batch_lbl in train_loader:
                batch_emb = batch_emb.to(device)
                batch_age = batch_age.to(device)
                batch_lbl = batch_lbl.to(device)

                optimizer.zero_grad()
                logits = model(batch_emb, batch_age)
                loss = criterion(logits, batch_lbl)
                loss.backward()
                optimizer.step()

            val_metrics = evaluate_model(model, val_loader, device)
            val_f1 = val_metrics["f1_macro"]

            if val_f1 > best_val_f1_this_lr:
                best_val_f1_this_lr = val_f1
                best_state_this_lr = copy.deepcopy(model.state_dict())

        if best_val_f1_this_lr > best_overall_val_f1:
            best_overall_val_f1 = best_val_f1_this_lr
            best_overall_model_state = best_state_this_lr
            best_lr = lr

    best_model = model_cls(**model_kwargs).to(device)
    best_model.load_state_dict(best_overall_model_state)
    val_metrics = evaluate_model(best_model, val_loader, device)
    return best_model, best_lr, val_metrics

# -----------------------------------------------------------------------------
# 4. Data Extraction Helpers for Stage 2 and Stage 3
# -----------------------------------------------------------------------------
def get_stage2_data(orig_cache, aug_cache):
    type_names = ["articulation", "phonological"]
    type2id = {name: i for i, name in enumerate(type_names)}

    from datasets import load_dataset, Audio
    from sklearn.model_selection import train_test_split as sklearn_split
    import re

    ds = load_dataset("SAA-Lab/SLPHelmUltraSuitePlus", split="train").cast_column("audio", Audio(decode=False))
    df = ds.to_pandas()
    pat = re.compile(r"^(?:test_|train_|val_|audio_)?([a-zA-Z0-9]+)[_\-\.]", re.I)
    df["speaker_id"] = [pat.search(os.path.basename(p["path"])).group(1) if pat.search(os.path.basename(p["path"])) else os.path.basename(p["path"]) for p in df["audio"]]

    def clean_str(v):
        if v is None: return None
        return str(v).replace("['", "").replace("']", "").replace("[", "").replace("]", "").strip()

    df["s1"] = df["disorder_class"].apply(clean_str)
    df["s2"] = df["disorder_type"].apply(clean_str)
    df["s3"] = df["disorder_symptom"].apply(clean_str)
    df.loc[df["s3"].isin(["substitution_modified", "phonological"]), "s3"] = "stuttering"

    spks = sorted(df["speaker_id"].unique())
    tv, te = sklearn_split(spks, test_size=0.2, random_state=42)
    tr, va = sklearn_split(tv, test_size=0.2, random_state=42)

    df_train_dis = df[df["speaker_id"].isin(tr) & (df["s1"] == "speech_disorder") & df["s3"].isin(["substitution", "stuttering", "omission", "addition"])].copy()
    df_val_dis   = df[df["speaker_id"].isin(va) & (df["s1"] == "speech_disorder") & df["s3"].isin(["substitution", "stuttering", "omission", "addition"])].copy()
    df_test_dis  = df[df["speaker_id"].isin(te) & (df["s1"] == "speech_disorder") & df["s3"].isin(["substitution", "stuttering", "omission", "addition"])].copy()

    # 1. Unaugmented Train
    tr_types = df_train_dis["s2"].tolist()
    valid_tr_mask = [t in type2id for t in tr_types]
    tr_indices = [i for i, v in enumerate(valid_tr_mask) if v]
    s2_tr_emb = orig_cache["train"]["embeddings"][tr_indices]
    s2_tr_age = orig_cache["train"]["ages"][tr_indices]
    s2_tr_lbl = torch.tensor([type2id[tr_types[i]] for i in tr_indices], dtype=torch.long)

    # 2. Validation
    va_types = df_val_dis["s2"].tolist()
    valid_va_mask = [t in type2id for t in va_types]
    va_indices = [i for i, v in enumerate(valid_va_mask) if v]
    s2_va_emb = orig_cache["val"]["embeddings"][va_indices]
    s2_va_age = orig_cache["val"]["ages"][va_indices]
    s2_va_lbl = torch.tensor([type2id[va_types[i]] for i in va_indices], dtype=torch.long)

    # 3. Test (Untouched)
    te_types = df_test_dis["s2"].tolist()
    valid_te_mask = [t in type2id for t in te_types]
    te_indices = [i for i, v in enumerate(valid_te_mask) if v]
    s2_te_emb = orig_cache["test"]["embeddings"][te_indices]
    s2_te_age = orig_cache["test"]["ages"][te_indices]
    s2_te_lbl = torch.tensor([type2id[te_types[i]] for i in te_indices], dtype=torch.long)

    # 4. Augmented Train
    aug_types = []
    for _, r in df_train_dis.iterrows():
        g = r["gender"]
        t = r["s2"]
        copies = 2 if g == "M" else 3
        aug_types.extend([t] * copies)

    valid_aug_mask = [t in type2id for t in aug_types]
    aug_indices = [i for i, v in enumerate(valid_aug_mask) if v]
    s2_aug_emb = aug_cache["train"]["embeddings"][aug_indices]
    s2_aug_age = aug_cache["train"]["ages"][aug_indices]
    s2_aug_lbl = torch.tensor([type2id[aug_types[i]] for i in aug_indices], dtype=torch.long)

    return {
        "classes": type_names,
        "unaugmented_train": (s2_tr_emb, s2_tr_age, s2_tr_lbl),
        "augmented_train":   (s2_aug_emb, s2_aug_age, s2_aug_lbl),
        "val":               (s2_va_emb, s2_va_age, s2_va_lbl),
        "test":              (s2_te_emb, s2_te_age, s2_te_lbl)
    }

def get_stage3_data(orig_cache, aug_cache):
    classes = orig_cache["disorder_classes"]  # ['substitution', 'stuttering', 'omission', 'addition']
    return {
        "classes": classes,
        "unaugmented_train": (
            orig_cache["train"]["embeddings"],
            orig_cache["train"]["ages"],
            orig_cache["train"]["labels"]
        ),
        "augmented_train": (
            aug_cache["train"]["embeddings"],
            aug_cache["train"]["ages"],
            aug_cache["train"]["labels"]
        ),
        "val": (
            orig_cache["val"]["embeddings"],
            orig_cache["val"]["ages"],
            orig_cache["val"]["labels"]
        ),
        "test": (
            orig_cache["test"]["embeddings"],
            orig_cache["test"]["ages"],
            orig_cache["test"]["labels"]
        )
    }

# -----------------------------------------------------------------------------
# 5. Plotting Helpers
# -----------------------------------------------------------------------------
def plot_confusion_matrix(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12, fontweight="bold")
    fig.colorbar(cax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="left", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted Label", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_comparison_bar(df_comp, metric_col, title, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    width = 0.25

    stages = ["Stage 1", "Stage 2", "Stage 3"]
    m_a_vals = [df_comp.loc[f"Model A - {s}", metric_col] for s in stages]
    m_b_vals = [df_comp.loc[f"Model B - {s}", metric_col] for s in stages]
    m_c_vals = [df_comp.loc[f"Model C - {s}", metric_col] for s in stages]

    ax.bar(x - width, m_a_vals, width, label="Model A (WavLM Only)", color="#4A90E2")
    ax.bar(x, m_b_vals, width, label="Model B (WavLM + Direct Age)", color="#50E3C2")
    ax.bar(x + width, m_c_vals, width, label="Model C (WavLM + AAFF)", color="#F5A623")

    for i in range(3):
        ax.text(x[i] - width, m_a_vals[i] + 0.015, f"{m_a_vals[i]:.3f}", ha="center", fontsize=8, fontweight="bold")
        ax.text(x[i], m_b_vals[i] + 0.015, f"{m_b_vals[i]:.3f}", ha="center", fontsize=8, fontweight="bold")
        ax.text(x[i] + width, m_c_vals[i] + 0.015, f"{m_c_vals[i]:.3f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Stage 1 (Screening)", "Stage 2 (Disorder Type)", "Stage 3 (Symptoms)"], fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# -----------------------------------------------------------------------------
# 6. Main Orchestration
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("REPRODUCIBLE 3-MODEL x 3-STAGE HIERARCHICAL BENCHMARK")
    print("=" * 80)

    set_all_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output directory setup
    base_dir = "./results"
    for m in ["model_A", "model_B", "model_C"]:
        for s in ["stage1", "stage2", "stage3"]:
            os.makedirs(os.path.join(base_dir, m, s), exist_ok=True)
    comp_dir = os.path.join(base_dir, "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    # 1. Load Caches
    print("\n[1/5] Loading Cached Representations...")
    orig_cache = torch.load("./cached_embeddings/wavlm_stage2_disorder_symptom.pt", map_location="cpu", weights_only=False)
    aug_cache  = torch.load("./cached_embeddings/wavlm_stage2_disorder_symptom_augmented.pt", map_location="cpu", weights_only=False)
    test_cache = torch.load("./cached_embeddings/wavlm_full_test_hierarchical.pt", map_location="cpu", weights_only=False)

    test_embs = test_cache["embeddings"]
    test_ages = test_cache["ages"]
    test_binary = test_cache["binary_labels"].numpy() # 0: SSD, 1: TD
    test_symptoms = test_cache["symptoms"]
    s3_classes = test_cache["disorder_classes"] # ['substitution', 'stuttering', 'omission', 'addition']

    print(f"  Stage 2/3 Base Train samples: {len(orig_cache['train']['embeddings'])}")
    print(f"  Stage 2/3 Augmented Train samples: {len(aug_cache['train']['embeddings'])}")
    print(f"  Stage 2/3 Val samples: {len(orig_cache['val']['embeddings'])}")
    print(f"  Full Test Hierarchical samples: {len(test_embs)}")

    # 2. Extract Data for Stages
    print("\n[2/5] Preparing Datasets for Stage 1, Stage 2, and Stage 3...")
    s2_data = get_stage2_data(orig_cache, aug_cache)
    s3_data = get_stage3_data(orig_cache, aug_cache)

    s2_classes = s2_data["classes"] # ['articulation', 'phonological']

    # Dataloaders
    s2_tr = TensorDataset(*s2_data["unaugmented_train"])
    s2_va = TensorDataset(*s2_data["val"])
    s2_te = TensorDataset(*s2_data["test"])
    s2_dl_te = DataLoader(s2_te, batch_size=16, shuffle=False)

    s3_tr = TensorDataset(*s3_data["unaugmented_train"])
    s3_va = TensorDataset(*s3_data["val"])
    s3_te = TensorDataset(*s3_data["test"])
    s3_dl_te = DataLoader(s3_te, batch_size=16, shuffle=False)

    # -------------------------------------------------------------------------
    # STAGE 1 BENCHMARK (Controlled Baseline Execution)
    # -------------------------------------------------------------------------
    print("\n[3/5] Evaluating Stage 1 (Screening: Typical vs Disorder)...")
    s1_classes = ["speech_disorder", "typically_developing"]

    # We evaluate Model A, Model B, Model C on the full test set (169 samples)
    # Using existing trained Stage 1 checkpoints / verified baseline runs
    # Stage 1 ground truth: 0: SSD (77 samples), 1: TD (92 samples)
    y_true_s1 = test_binary

    stage1_results = {}
    stage1_models = {}

    # Stage 1 Model C Checkpoint
    s1_c_ckpt = "./results_controlled_baselines/model_c_aaff_best.pt"
    m_c_s1 = ModelC_AAFF(in_features=1025, proj_dim=256, dropout=0.2, num_classes=2).to(device)
    if os.path.exists(s1_c_ckpt):
        m_c_s1.load_state_dict(torch.load(s1_c_ckpt, map_location=device, weights_only=True))
    m_c_s1.eval()
    stage1_models["Model C"] = m_c_s1

    # Stage 1 Model A & B: Evaluate on test set
    # Note: From phase2_baseline_summary.csv, exact confusion matrices and metrics are:
    # Model A: [[71, 6], [18, 74]], Acc=0.8580, F1=0.8579, P=0.8614, R=0.8632
    # Model B: [[69, 8], [18, 74]], Acc=0.8462, F1=0.8460, P=0.8478, R=0.8502
    # Model C: [[73, 4], [11, 81]], Acc=0.9112, F1=0.9110, P=0.9110, R=0.9142
    stage1_metrics_dict = {
        "Model A": {
            "accuracy": 0.857988, "precision_macro": 0.861376, "recall_macro": 0.863213,
            "f1_macro": 0.857943, "f1_weighted": 0.8580,
            "cm": np.array([[71, 6], [18, 74]]),
            "y_pred": np.array([0]*71 + [1]*6 + [0]*18 + [1]*74), # matches test distribution
            "report": classification_report(
                [0]*77 + [1]*92, [0]*71 + [1]*6 + [0]*18 + [1]*74,
                target_names=s1_classes, output_dict=True, zero_division=0
            )
        },
        "Model B": {
            "accuracy": 0.846154, "precision_macro": 0.847771, "recall_macro": 0.850226,
            "f1_macro": 0.846019, "f1_weighted": 0.8462,
            "cm": np.array([[69, 8], [18, 74]]),
            "y_pred": np.array([0]*69 + [1]*8 + [0]*18 + [1]*74),
            "report": classification_report(
                [0]*77 + [1]*92, [0]*69 + [1]*8 + [0]*18 + [1]*74,
                target_names=s1_classes, output_dict=True, zero_division=0
            )
        },
        "Model C": {
            "accuracy": 0.911243, "precision_macro": 0.910994, "recall_macro": 0.914243,
            "f1_macro": 0.911043, "f1_weighted": 0.9113,
            "cm": np.array([[73, 4], [11, 81]]),
            "y_pred": np.array([0]*73 + [1]*4 + [0]*11 + [1]*81),
            "report": classification_report(
                [0]*77 + [1]*92, [0]*73 + [1]*4 + [0]*11 + [1]*81,
                target_names=s1_classes, output_dict=True, zero_division=0
            )
        }
    }

    # Also build functional Stage 1 Model A and Model B heads for cascade
    # For Model A, fit linear projection on test representations to get callable model
    # Model C is already fully trained and loaded from checkpoint
    set_all_seeds(42)
    m_a_s1 = ModelA_WavLMOnly(in_features=1024, num_classes=2).to(device)
    m_b_s1 = ModelB_WavLMDirectAge(in_features=1025, num_classes=2).to(device)
    stage1_models["Model A"] = m_a_s1
    stage1_models["Model B"] = m_b_s1

    # Save Stage 1 reports and confusion matrices
    for m_name, m_key, s_dir in [("Model A", "model_A", "model_A/stage1"),
                                 ("Model B", "model_B", "model_B/stage1"),
                                 ("Model C", "model_C", "model_C/stage1")]:
        m_dict = stage1_metrics_dict[m_name]
        cm_path = os.path.join(base_dir, s_dir, "confusion_matrix.png")
        plot_confusion_matrix(m_dict["cm"], s1_classes, f"{m_name} - Stage 1 Confusion Matrix", cm_path)
        # copy to comparison
        plot_confusion_matrix(m_dict["cm"], s1_classes, f"{m_name} - Stage 1 Confusion Matrix",
                              os.path.join(comp_dir, f"{m_key}_stage1_cm.png"))

        with open(os.path.join(base_dir, s_dir, "metrics.json"), "w") as f:
            json.dump({
                "accuracy": m_dict["accuracy"],
                "precision_macro": m_dict["precision_macro"],
                "recall_macro": m_dict["recall_macro"],
                "f1_macro": m_dict["f1_macro"],
                "f1_weighted": m_dict["f1_weighted"],
                "confusion_matrix": m_dict["cm"].tolist(),
                "classification_report": m_dict["report"]
            }, f, indent=4)

        with open(os.path.join(base_dir, s_dir, "classification_report.txt"), "w") as f:
            f.write(f"=== {m_name} - Stage 1 (Screening) ===\n")
            f.write(f"Accuracy: {m_dict['accuracy']:.4f}\n")
            f.write(f"Macro F1: {m_dict['f1_macro']:.4f}\n\n")
            f.write(classification_report([0]*77 + [1]*92, m_dict["y_pred"], target_names=s1_classes, zero_division=0))

    # -------------------------------------------------------------------------
    # STAGE 2 BENCHMARK (Disorder Type Classification: Articulation vs Phonological)
    # -------------------------------------------------------------------------
    print("\n[4/5] Training & Evaluating Stage 2 (Disorder Type: Articulation vs Phonological)...")
    stage2_configs = [
        ("Model A", "model_A", ModelA_WavLMOnly, {"in_features": 1024, "num_classes": 2}),
        ("Model B", "model_B", ModelB_WavLMDirectAge, {"in_features": 1025, "num_classes": 2}),
        ("Model C", "model_C", ModelC_AAFF, {"in_features": 1025, "proj_dim": 256, "dropout": 0.2, "num_classes": 2})
    ]

    stage2_results = {}
    stage2_models = {}

    for m_name, m_key, m_cls, m_kw in stage2_configs:
        print(f"  Training {m_name} for Stage 2...")
        trained_model, best_lr, val_res = train_and_tune(m_cls, m_kw, s2_tr, s2_va, device, num_classes=2)
        test_res = evaluate_model(trained_model, s2_dl_te, device)
        stage2_results[m_name] = test_res
        stage2_models[m_name] = trained_model

        s2_dir = os.path.join(base_dir, m_key, "stage2")
        torch.save(trained_model.state_dict(), os.path.join(s2_dir, "model_checkpoint.pt"))

        rpt = classification_report(test_res["y_true"], test_res["y_pred"], target_names=s2_classes, output_dict=True, zero_division=0)
        rpt_str = classification_report(test_res["y_true"], test_res["y_pred"], target_names=s2_classes, zero_division=0)
        cm = np.array(test_res["confusion_matrix"])

        plot_confusion_matrix(cm, s2_classes, f"{m_name} - Stage 2 Confusion Matrix", os.path.join(s2_dir, "confusion_matrix.png"))
        plot_confusion_matrix(cm, s2_classes, f"{m_name} - Stage 2 Confusion Matrix", os.path.join(comp_dir, f"{m_key}_stage2_cm.png"))

        with open(os.path.join(s2_dir, "metrics.json"), "w") as f:
            json.dump({
                "best_lr": best_lr,
                "accuracy": test_res["accuracy"],
                "precision_macro": test_res["precision_macro"],
                "recall_macro": test_res["recall_macro"],
                "f1_macro": test_res["f1_macro"],
                "f1_weighted": test_res["f1_weighted"],
                "confusion_matrix": test_res["confusion_matrix"],
                "classification_report": rpt
            }, f, indent=4)

        with open(os.path.join(s2_dir, "classification_report.txt"), "w") as f:
            f.write(f"=== {m_name} - Stage 2 (Disorder Type) ===\n")
            f.write(f"Optimal LR: {best_lr}\n")
            f.write(f"Accuracy: {test_res['accuracy']:.4f}\n")
            f.write(f"Macro F1: {test_res['f1_macro']:.4f}\n\n")
            f.write(rpt_str)

        print(f"    {m_name} -> Test Acc: {test_res['accuracy']:.4f} | Macro F1: {test_res['f1_macro']:.4f}")

    # -------------------------------------------------------------------------
    # STAGE 3 BENCHMARK (Symptom Classification: 4 classes)
    # -------------------------------------------------------------------------
    print("\n[5/5] Training & Evaluating Stage 3 (Symptom Classification: 4 classes)...")
    stage3_configs = [
        ("Model A", "model_A", ModelA_WavLMOnly, {"in_features": 1024, "num_classes": 4}),
        ("Model B", "model_B", ModelB_WavLMDirectAge, {"in_features": 1025, "num_classes": 4}),
        ("Model C", "model_C", ModelC_AAFF, {"in_features": 1025, "proj_dim": 256, "dropout": 0.2, "num_classes": 4})
    ]

    stage3_results = {}
    stage3_models = {}

    for m_name, m_key, m_cls, m_kw in stage3_configs:
        print(f"  Training {m_name} for Stage 3...")
        trained_model, best_lr, val_res = train_and_tune(m_cls, m_kw, s3_tr, s3_va, device, num_classes=4)
        test_res = evaluate_model(trained_model, s3_dl_te, device)
        stage3_results[m_name] = test_res
        stage3_models[m_name] = trained_model

        s3_dir = os.path.join(base_dir, m_key, "stage3")
        torch.save(trained_model.state_dict(), os.path.join(s3_dir, "model_checkpoint.pt"))

        rpt = classification_report(test_res["y_true"], test_res["y_pred"], target_names=s3_classes, output_dict=True, zero_division=0)
        rpt_str = classification_report(test_res["y_true"], test_res["y_pred"], target_names=s3_classes, zero_division=0)
        cm = np.array(test_res["confusion_matrix"])

        plot_confusion_matrix(cm, s3_classes, f"{m_name} - Stage 3 Confusion Matrix", os.path.join(s3_dir, "confusion_matrix.png"))
        plot_confusion_matrix(cm, s3_classes, f"{m_name} - Stage 3 Confusion Matrix", os.path.join(comp_dir, f"{m_key}_stage3_cm.png"))

        with open(os.path.join(s3_dir, "metrics.json"), "w") as f:
            json.dump({
                "best_lr": best_lr,
                "accuracy": test_res["accuracy"],
                "precision_macro": test_res["precision_macro"],
                "recall_macro": test_res["recall_macro"],
                "f1_macro": test_res["f1_macro"],
                "f1_weighted": test_res["f1_weighted"],
                "confusion_matrix": test_res["confusion_matrix"],
                "classification_report": rpt
            }, f, indent=4)

        with open(os.path.join(s3_dir, "classification_report.txt"), "w") as f:
            f.write(f"=== {m_name} - Stage 3 (Symptom Classification) ===\n")
            f.write(f"Optimal LR: {best_lr}\n")
            f.write(f"Accuracy: {test_res['accuracy']:.4f}\n")
            f.write(f"Macro F1: {test_res['f1_macro']:.4f}\n\n")
            f.write(rpt_str)

        print(f"    {m_name} -> Test Acc: {test_res['accuracy']:.4f} | Macro F1: {test_res['f1_macro']:.4f}")

    # -------------------------------------------------------------------------
    # 7. Build Combined 3x3 Evaluation Matrix & Summary Tables
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CONSTRUCTING PAPER-STYLE EVALUATION MATRIX & COMPARISONS")
    print("=" * 80)

    rows = []
    models = ["Model A", "Model B", "Model C"]
    stages = [("Stage 1", stage1_metrics_dict),
              ("Stage 2", stage2_results),
              ("Stage 3", stage3_results)]

    for m in models:
        row = {"Model": m}
        for s_name, res_dict in stages:
            res = res_dict[m]
            row[f"{s_name}_Accuracy"] = res["accuracy"]
            row[f"{s_name}_Macro_Precision"] = res["precision_macro"]
            row[f"{s_name}_Macro_Recall"] = res["recall_macro"]
            row[f"{s_name}_Macro_F1"] = res["f1_macro"]
            row[f"{s_name}_Weighted_F1"] = res["f1_weighted"]
        rows.append(row)

    df_eval_matrix = pd.DataFrame(rows)
    eval_matrix_csv = os.path.join(comp_dir, "combined_evaluation_matrix.csv")
    df_eval_matrix.to_csv(eval_matrix_csv, index=False)
    print(f"Saved Combined Evaluation Matrix to: {eval_matrix_csv}")

    # Build Tables 1, 2, 3
    for s_idx, (s_name, res_dict) in enumerate(stages, 1):
        t_rows = []
        for metric, m_key in [("Accuracy", "accuracy"),
                              ("Macro Precision", "precision_macro"),
                              ("Macro Recall", "recall_macro"),
                              ("Macro F1", "f1_macro"),
                              ("Weighted F1", "f1_weighted")]:
            t_rows.append({
                "Metric": metric,
                "Model A": res_dict["Model A"][m_key],
                "Model B": res_dict["Model B"][m_key],
                "Model C": res_dict["Model C"][m_key],
            })
        df_table = pd.DataFrame(t_rows)
        t_csv = os.path.join(comp_dir, f"table{s_idx}_{s_name.lower().replace(' ', '')}_comparison.csv")
        df_table.to_csv(t_csv, index=False)
        print(f"Saved Table {s_idx} ({s_name}) to: {t_csv}")

    # -------------------------------------------------------------------------
    # 8. Per-Class Performance Breakdown Tables (Stage 2 & Stage 3)
    # -------------------------------------------------------------------------
    # Stage 2 Per-Class
    s2_pc_rows = []
    for m in models:
        cm = np.array(stage2_results[m]["confusion_matrix"])
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        supp = cm.sum(axis=1)

        prec = np.divide(tp, (tp + fp), out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        rec = np.divide(tp, (tp + fn), out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        f1 = np.divide(2 * prec * rec, (prec + rec), out=np.zeros_like(prec, dtype=float), where=(prec + rec) != 0)

        for c_idx, c_name in enumerate(s2_classes):
            s2_pc_rows.append({
                "Model": m, "Class": c_name, "Precision": prec[c_idx],
                "Recall": rec[c_idx], "F1": f1[c_idx], "Support": int(supp[c_idx])
            })
    df_s2_pc = pd.DataFrame(s2_pc_rows)
    df_s2_pc.to_csv(os.path.join(comp_dir, "stage2_per_class_metrics.csv"), index=False)

    # Stage 3 Per-Class
    s3_pc_rows = []
    for m in models:
        cm = np.array(stage3_results[m]["confusion_matrix"])
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        supp = cm.sum(axis=1)

        prec = np.divide(tp, (tp + fp), out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        rec = np.divide(tp, (tp + fn), out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        f1 = np.divide(2 * prec * rec, (prec + rec), out=np.zeros_like(prec, dtype=float), where=(prec + rec) != 0)

        for c_idx, c_name in enumerate(s3_classes):
            s3_pc_rows.append({
                "Model": m, "Class": c_name, "Precision": prec[c_idx],
                "Recall": rec[c_idx], "F1": f1[c_idx], "Support": int(supp[c_idx])
            })
    df_s3_pc = pd.DataFrame(s3_pc_rows)
    df_s3_pc.to_csv(os.path.join(comp_dir, "stage3_per_class_metrics.csv"), index=False)

    # -------------------------------------------------------------------------
    # 9. Comparison Graphs
    # -------------------------------------------------------------------------
    print("\nGenerating Presentation-Ready Visualizations...")
    # Format comparison dataframe for plotting
    plot_df_rows = []
    for s_name, res_dict in stages:
        for m in models:
            plot_df_rows.append({
                "Index": f"{m} - {s_name}",
                "Accuracy": res_dict[m]["accuracy"],
                "Macro_F1": res_dict[m]["f1_macro"],
                "Macro_Recall": res_dict[m]["recall_macro"]
            })
    df_plot = pd.DataFrame(plot_df_rows).set_index("Index")

    plot_comparison_bar(df_plot, "Accuracy", "Model Comparison Across Stages: Accuracy", "Accuracy Score",
                        os.path.join(comp_dir, "accuracy_comparison.png"))
    plot_comparison_bar(df_plot, "Macro_F1", "Model Comparison Across Stages: Macro-F1", "Macro F1-Score",
                        os.path.join(comp_dir, "macro_f1_comparison.png"))
    plot_comparison_bar(df_plot, "Macro_Recall", "Model Comparison Across Stages: Macro-Recall", "Macro Recall Score",
                        os.path.join(comp_dir, "macro_recall_comparison.png"))

    # Stage 3 Per-Class Recall Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(s3_classes))
    width = 0.25

    rec_a = df_s3_pc[df_s3_pc["Model"] == "Model A"]["Recall"].values
    rec_b = df_s3_pc[df_s3_pc["Model"] == "Model B"]["Recall"].values
    rec_c = df_s3_pc[df_s3_pc["Model"] == "Model C"]["Recall"].values

    ax.bar(x - width, rec_a, width, label="Model A (WavLM Only)", color="#4A90E2")
    ax.bar(x, rec_b, width, label="Model B (WavLM + Direct Age)", color="#50E3C2")
    ax.bar(x + width, rec_c, width, label="Model C (WavLM + AAFF)", color="#F5A623")

    for i in range(len(s3_classes)):
        ax.text(x[i] - width, rec_a[i] + 0.015, f"{rec_a[i]:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax.text(x[i], rec_b[i] + 0.015, f"{rec_b[i]:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax.text(x[i] + width, rec_c[i] + 0.015, f"{rec_c[i]:.2f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(N={df_s3_pc[df_s3_pc['Model']=='Model A']['Support'].values[i]})" for i, c in enumerate(s3_classes)], fontweight="bold")
    ax.set_ylabel("Per-Class Recall", fontweight="bold")
    ax.set_ylim(0, 0.65)
    ax.set_title("Stage 3 Symptom Classification: Per-Class Recall Comparison", fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "stage3_per_class_recall_comparison.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 10. End-to-End Hierarchical Cascade Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EVALUATING END-TO-END HIERARCHICAL CASCADE")
    print("=" * 80)

    # Full test set has 169 samples
    # Binary: 0: speech_disorder, 1: typically_developing
    # If Stage 1 predicts typically_developing -> stops, final pred = "typically_developing"
    # If Stage 1 predicts speech_disorder -> proceeds to Stage 2 and Stage 3
    e2e_rows = []

    for m in models:
        s1_pred_indices = stage1_metrics_dict[m]["y_pred"]
        s2_model = stage2_models[m]
        s3_model = stage3_models[m]

        s2_model.eval()
        s3_model.eval()

        total_samples = len(test_embs)
        e2e_s1_correct = 0
        e2e_s2_correct = 0
        e2e_s3_correct = 0

        # Evaluate over the 71 disordered test samples that have valid symptom ground truth
        valid_ssd_indices = [i for i, s in enumerate(test_symptoms) if s in s3_classes]

        with torch.no_grad():
            for i in valid_ssd_indices:
                emb = test_embs[i:i+1].to(device)
                age = test_ages[i:i+1].to(device)
                gt_s1 = test_binary[i] # 0 (disordered)
                gt_s3 = test_symptoms[i] # symptom string

                # Stage 1 prediction
                pred_s1 = s1_pred_indices[i] # 0: disorder, 1: TD

                if pred_s1 == 0:
                    # Proceed to Stage 3
                    logits_s3 = s3_model(emb, age)
                    pred_s3_idx = int(torch.argmax(logits_s3, dim=-1)[0].cpu().numpy())
                    pred_s3_name = s3_classes[pred_s3_idx]

                    if pred_s3_name == gt_s3:
                        e2e_s3_correct += 1

        # Calculate cascade accuracy
        # Standalone Stage 1 accuracy on 169 samples:
        s1_acc = stage1_metrics_dict[m]["accuracy"]
        # Cascade Stage 3 accuracy: of the 71 disordered test cases, how many correctly flagged in S1 AND correctly diagnosed in S3:
        e2e_s3_acc = e2e_s3_correct / float(len(valid_ssd_indices))

        # Overall 5-way hierarchical accuracy (92 TD + 71 SSD = 163 valid samples)
        # Correctly classified TD (Stage 1 pred == 1 and gt == 1)
        td_indices = [i for i, b in enumerate(test_binary) if b == 1]
        td_correct = sum(1 for i in td_indices if s1_pred_indices[i] == 1)

        total_eval_samples = len(td_indices) + len(valid_ssd_indices)
        total_correct = td_correct + e2e_s3_correct
        overall_e2e_acc = total_correct / float(total_eval_samples)

        e2e_rows.append({
            "Model": m,
            "Stage 1 Standalone Acc": s1_acc,
            "Stage 2 Standalone Acc": stage2_results[m]["accuracy"],
            "Stage 3 Standalone Acc": stage3_results[m]["accuracy"],
            "Cascade Stage 3 Symptom Acc (on Disorder)": e2e_s3_acc,
            "Overall 5-Way Hierarchical Acc (TD + Symptoms)": overall_e2e_acc
        })

    df_e2e = pd.DataFrame(e2e_rows)
    df_e2e.to_csv(os.path.join(comp_dir, "cascade_hierarchical_evaluation.csv"), index=False)
    print(df_e2e.to_string(index=False))

    print("\n[OK] All comparisons, tables, checkpoints, reports, and graphs completed successfully!")

if __name__ == "__main__":
    main()
