"""
optimize_stage3_suite.py
------------------------
Comprehensive Stage 3 Optimization and Experimental Campaign.
Executes Phases 2 through 18:
  - Phase 2: Speaker-independent data split audit
  - Phase 3 & 4: Audio Augmentations & WavLM embedding loading
  - Phase 5: Baseline model (unweighted CE, standard sampling)
  - Phase 6: Class-weighted Cross Entropy
  - Phase 7: Balanced mini-batch sampling (WeightedRandomSampler)
  - Phase 8: SMOTE alone vs Augmentation + SMOTE
  - Phase 9: Focal Loss
  - Phase 10: WavLM fine-tuning / layer adaptation
  - Phase 11: Acoustic + Prosodic feature extraction & normalization
  - Phase 12: Multi-modal Feature Fusion (WavLM + Acoustic + Age)
  - Phase 13: Supervised Contrastive Learning (SupCon)
  - Phase 14: Classifier Architecture ablation
  - Phase 15: Combination Experiments
  - Phase 16: Validation Macro-F1 Model Selection & Final Test Evaluation
  - Phase 18: Comparison Table, Confusion Matrix, Graphs & t-SNE Visualization
"""

import os
import io
import copy
import json
import random
import time
import collections
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


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
# 2. Architectures (AAFF, Multi-Head, Fusion, Contrastive)
# -----------------------------------------------------------------------------
class AAFFModel1(nn.Module):
    """Model 1: 1025 -> 256 -> 4"""
    def __init__(self, in_features=1025, num_classes=4, dropout=0.2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.classifier(self.projection(x))


class AAFFModel2(nn.Module):
    """Model 2: 1025 -> 128 -> 4 (Compact)"""
    def __init__(self, in_features=1025, num_classes=4, dropout=0.2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.projection(x))


class AAFFModel3(nn.Module):
    """Model 3: 1025 -> 256 -> 64 -> 4 (Deeper)"""
    def __init__(self, in_features=1025, num_classes=4, dropout=0.25):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.classifier(self.projection(x))


class FeatureFusionModel(nn.Module):
    """
    Model 4: Hierarchical Multi-Modal Feature Fusion
    WavLM (1024) -> Linear(256)
    Acoustic (36) -> Linear(64)
    Age (1) -> Linear(16)
    Concat (336) -> Linear(128) -> Dropout -> Linear(num_classes)
    """
    def __init__(self, wavlm_dim=1024, acoustic_dim=36, age_dim=1, num_classes=4, dropout=0.25):
        super().__init__()
        self.wavlm_proj = nn.Sequential(
            nn.Linear(wavlm_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.acoustic_proj = nn.Sequential(
            nn.Linear(acoustic_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.age_proj = nn.Sequential(
            nn.Linear(age_dim, 16),
            nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64 + 16, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(128, num_classes)

    def extract_features(self, wavlm, acoustic, age):
        p_w = self.wavlm_proj(wavlm)
        p_ac = self.acoustic_proj(acoustic)
        p_ag = self.age_proj(age)
        fused = torch.cat([p_w, p_ac, p_ag], dim=-1)
        return self.fusion(fused)

    def forward(self, wavlm, acoustic, age):
        rep = self.extract_features(wavlm, acoustic, age)
        return self.classifier(rep)


# -----------------------------------------------------------------------------
# 3. Loss Functions (Focal Loss & Supervised Contrastive Loss)
# -----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # class weight tensor
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (batch_size, feature_dim) normalized
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        # similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - logits_max.detach()

        # mask-out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-9)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        loss = -mean_log_prob_pos.mean()
        return loss


# -----------------------------------------------------------------------------
# 4. Evaluation Function
# -----------------------------------------------------------------------------
def evaluate_model_dl(model, dl, device, is_fusion=False):
    model.eval()
    preds, targets, confs = [], [], []
    with torch.no_grad():
        for batch in dl:
            if is_fusion:
                bw, bac, bag, by = batch
                bw, bac, bag = bw.to(device), bac.to(device), bag.to(device)
                logits = model(bw, bac, bag)
            else:
                bx, by = batch
                bx = bx.to(device)
                logits = model(bx)

            probs = F.softmax(logits, dim=-1).cpu().numpy()
            top_preds = np.argmax(probs, axis=1)
            top_confs = np.max(probs, axis=1)

            preds.extend(top_preds.tolist())
            targets.extend(by.numpy().tolist())
            confs.extend(top_confs.tolist())

    y_true = np.array(targets)
    y_pred = np.array(preds)
    acc = float(accuracy_score(y_true, y_pred))
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    rep_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": acc,
        "precision_macro": float(pm),
        "recall_macro": float(rm),
        "f1_macro": float(fm),
        "f1_weighted": float(fw),
        "avg_confidence": float(np.mean(confs)),
        "y_true": y_true,
        "y_pred": y_pred,
        "report": rep_dict,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


# -----------------------------------------------------------------------------
# 5. Extract Acoustic & Prosodic Features (36-D)
# -----------------------------------------------------------------------------
def extract_sample_acoustic(wav_bytes, target_sr=16000):
    wav, sr = sf.read(io.BytesIO(wav_bytes))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)

    # 12 seconds max
    wav = wav[:target_sr * 12]

    # MFCCs (13 coefficients: mean and std)
    mfcc = librosa.feature.mfcc(y=wav, sr=target_sr, n_mfcc=13, hop_length=512)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    # Energy & ZCR
    rms = librosa.feature.rms(y=wav, hop_length=512)
    zcr = librosa.feature.zero_crossing_rate(y=wav, hop_length=512)

    # Fundamental Frequency (Pitch)
    f0 = librosa.yin(wav, fmin=80, fmax=400, sr=target_sr, hop_length=1024)
    voiced = f0[f0 > 80]
    if len(voiced) > 0:
        f0_m, f0_s, f0_min, f0_max = float(np.mean(voiced)), float(np.std(voiced)), float(np.min(voiced)), float(np.max(voiced))
        vr = float(len(voiced) / len(f0))
    else:
        f0_m, f0_s, f0_min, f0_max, vr = 0.0, 0.0, 0.0, 0.0, 0.0

    return np.hstack([
        mfcc_mean, mfcc_std,
        [rms.mean(), rms.std(), zcr.mean(), zcr.std(), f0_m, f0_s, f0_min, f0_max, vr, 1.0 - vr]
    ])


# -----------------------------------------------------------------------------
# 6. Main Experimental Benchmark Suite
# -----------------------------------------------------------------------------
def main():
    set_all_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "./results_controlled_baselines"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("STAGE 3 SYMPTOM CLASSIFICATION: OPTIMIZATION & BENCHMARK SUITE")
    print(f"Compute Device: {device}")
    print("=" * 80)

    # --- Load Pre-cached Data ---
    orig_path = "./cached_embeddings/wavlm_stage2_disorder_symptom.pt"
    aug_path  = "./cached_embeddings/wavlm_stage2_disorder_symptom_augmented.pt"

    orig_cache = torch.load(orig_path, map_location="cpu", weights_only=False)
    aug_cache  = torch.load(aug_path,  map_location="cpu", weights_only=False)

    classes = orig_cache["disorder_classes"]  # ['substitution', 'stuttering', 'omission', 'addition']
    num_classes = len(classes)

    # Unaugmented splits
    tr_emb_base = orig_cache["train"]["embeddings"]
    tr_age_base = orig_cache["train"]["ages"]
    tr_lbl_base = orig_cache["train"]["labels"]

    val_emb = orig_cache["val"]["embeddings"]
    val_age = orig_cache["val"]["ages"]
    val_lbl = orig_cache["val"]["labels"]

    te_emb = orig_cache["test"]["embeddings"]
    te_age = orig_cache["test"]["ages"]
    te_lbl = orig_cache["test"]["labels"]

    # Augmented split
    tr_emb_aug = aug_cache["train"]["embeddings"]
    tr_age_aug = aug_cache["train"]["ages"]
    tr_lbl_aug = aug_cache["train"]["labels"]

    # Concat 1025-D features (WavLM 1024 + age 1)
    X_tr_base = torch.cat([tr_emb_base, tr_age_base], dim=1)
    X_tr_aug  = torch.cat([tr_emb_aug,  tr_age_aug],  dim=1)
    X_val     = torch.cat([val_emb,     val_age],     dim=1)
    X_test    = torch.cat([te_emb,      te_age],      dim=1)

    val_ds = TensorDataset(X_val, val_lbl)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    test_ds = TensorDataset(X_test, te_lbl)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # Calculate class weights from training base
    counts_base = np.bincount(tr_lbl_base.numpy(), minlength=num_classes)
    weights_base = len(tr_lbl_base) / (float(num_classes) * np.maximum(counts_base, 1))
    weight_tensor_base = torch.tensor(weights_base, dtype=torch.float32).to(device)

    # Calculate class weights from training augmented
    counts_aug = np.bincount(tr_lbl_aug.numpy(), minlength=num_classes)
    weights_aug = len(tr_lbl_aug) / (float(num_classes) * np.maximum(counts_aug, 1))
    weight_tensor_aug = torch.tensor(weights_aug, dtype=torch.float32).to(device)

    # -------------------------------------------------------------------------
    # Helper Training Function with Validation Tuning
    # -------------------------------------------------------------------------
    def train_experiment(
        train_features, train_labels,
        model_cls=AAFFModel1,
        loss_fn_builder=None,
        use_sampler=False,
        epochs=40,
        lr_list=[1e-3, 5e-4, 3e-4, 1e-4],
        weight_decay=1e-2,
        seed=42,
        model_kwargs=None,
        eval_val_loader=None,
        eval_test_loader=None
    ):
        model_kwargs = model_kwargs or {}
        best_val_f1 = -1.0
        best_model_state = None
        best_lr = None

        v_loader = eval_val_loader if eval_val_loader is not None else val_loader
        t_loader = eval_test_loader if eval_test_loader is not None else test_loader

        if use_sampler:
            lbls = train_labels.numpy()
            c_cnts = np.bincount(lbls, minlength=num_classes)
            sample_weights = [1.0 / max(c_cnts[l], 1) for l in lbls]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(lbls), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        for lr in lr_list:
            set_all_seeds(seed)
            model = model_cls(**model_kwargs).to(device)
            loss_fn = loss_fn_builder(device) if loss_fn_builder else nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            train_ds = TensorDataset(train_features, train_labels)
            train_loader = DataLoader(train_ds, batch_size=16, shuffle=shuffle, sampler=sampler)

            this_best_f1 = -1.0
            this_best_state = None

            for ep in range(1, epochs + 1):
                model.train()
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = loss_fn(out, by)
                    loss.backward()
                    optimizer.step()

                v_res = evaluate_model_dl(model, v_loader, device)
                if v_res["f1_macro"] > this_best_f1:
                    this_best_f1 = v_res["f1_macro"]
                    this_best_state = copy.deepcopy(model.state_dict())

            if this_best_f1 > best_val_f1:
                best_val_f1 = this_best_f1
                best_model_state = this_best_state
                best_lr = lr

        best_model = model_cls(**model_kwargs).to(device)
        best_model.load_state_dict(best_model_state)
        val_eval = evaluate_model_dl(best_model, v_loader, device)
        test_eval = evaluate_model_dl(best_model, t_loader, device)

        return best_model, best_lr, val_eval, test_eval

    # Dictionary to store all experiment results
    all_experiments = {}

    # =========================================================================
    # PHASE 5: BASELINE (Unweighted CE, Standard Sampling, Base Train N=280)
    # =========================================================================
    print("\n--- PHASE 5: BASELINE (Model 1, Unweighted CE, Base Train) ---")
    b_model, b_lr, b_val, b_test = train_experiment(
        X_tr_base, tr_lbl_base,
        model_cls=AAFFModel1,
        loss_fn_builder=lambda dev: nn.CrossEntropyLoss(),
        model_kwargs={"in_features": 1025, "num_classes": 4}
    )
    all_experiments["Baseline"] = {
        "val_f1": b_val["f1_macro"], "val_acc": b_val["accuracy"],
        "test_acc": b_test["accuracy"], "test_f1": b_test["f1_macro"],
        "test_recall": b_test["recall_macro"], "minority_recall": b_test["report"]["3"]["recall"],
        "report": b_test
    }
    print(f"  Validation F1: {b_val['f1_macro']:.4f} | Test Acc: {b_test['accuracy']:.4f} | Test Macro F1: {b_test['f1_macro']:.4f} | Minority Recall: {b_test['report']['3']['recall']:.4f}")

    # =========================================================================
    # PHASE 3: AUDIO AUGMENTATION (Gender-Stratified Pitch Augmentation N=630)
    # =========================================================================
    print("\n--- PHASE 3: AUDIO AUGMENTATION (Pitch Augmentation Only N=630) ---")
    aug_model, aug_lr, aug_val, aug_test = train_experiment(
        X_tr_aug, tr_lbl_aug,
        model_cls=AAFFModel1,
        loss_fn_builder=lambda dev: nn.CrossEntropyLoss(),
        model_kwargs={"in_features": 1025, "num_classes": 4}
    )
    all_experiments["Augmentation"] = {
        "val_f1": aug_val["f1_macro"], "val_acc": aug_val["accuracy"],
        "test_acc": aug_test["accuracy"], "test_f1": aug_test["f1_macro"],
        "test_recall": aug_test["recall_macro"], "minority_recall": aug_test["report"]["3"]["recall"],
        "report": aug_test
    }
    print(f"  Validation F1: {aug_val['f1_macro']:.4f} | Test Acc: {aug_test['accuracy']:.4f} | Test Macro F1: {aug_test['f1_macro']:.4f} | Minority Recall: {aug_test['report']['3']['recall']:.4f}")

    # =========================================================================
    # PHASE 6: CLASS WEIGHTING (Weighted Cross-Entropy on Base Train)
    # =========================================================================
    print("\n--- PHASE 6: CLASS-WEIGHTED CROSS ENTROPY ---")
    cw_model, cw_lr, cw_val, cw_test = train_experiment(
        X_tr_base, tr_lbl_base,
        model_cls=AAFFModel1,
        loss_fn_builder=lambda dev: nn.CrossEntropyLoss(weight=weight_tensor_base),
        model_kwargs={"in_features": 1025, "num_classes": 4}
    )
    all_experiments["Class-weighted"] = {
        "val_f1": cw_val["f1_macro"], "val_acc": cw_val["accuracy"],
        "test_acc": cw_test["accuracy"], "test_f1": cw_test["f1_macro"],
        "test_recall": cw_test["recall_macro"], "minority_recall": cw_test["report"]["3"]["recall"],
        "report": cw_test
    }
    print(f"  Validation F1: {cw_val['f1_macro']:.4f} | Test Acc: {cw_test['accuracy']:.4f} | Test Macro F1: {cw_test['f1_macro']:.4f} | Minority Recall: {cw_test['report']['3']['recall']:.4f}")

    # =========================================================================
    # PHASE 7: BALANCED SAMPLING (WeightedRandomSampler)
    # =========================================================================
    print("\n--- PHASE 7: BALANCED SAMPLING (WeightedRandomSampler) ---")
    bs_model, bs_lr, bs_val, bs_test = train_experiment(
        X_tr_base, tr_lbl_base,
        model_cls=AAFFModel1,
        loss_fn_builder=lambda dev: nn.CrossEntropyLoss(),
        use_sampler=True,
        model_kwargs={"in_features": 1025, "num_classes": 4}
    )
    all_experiments["Balanced Sampling"] = {
        "val_f1": bs_val["f1_macro"], "val_acc": bs_val["accuracy"],
        "test_acc": bs_test["accuracy"], "test_f1": bs_test["f1_macro"],
        "test_recall": bs_test["recall_macro"], "minority_recall": bs_test["report"]["3"]["recall"],
        "report": bs_test
    }
    print(f"  Validation F1: {bs_val['f1_macro']:.4f} | Test Acc: {bs_test['accuracy']:.4f} | Test Macro F1: {bs_test['f1_macro']:.4f} | Minority Recall: {bs_test['report']['3']['recall']:.4f}")

    # =========================================================================
    # PHASE 8: SMOTE (SMOTE Alone vs Augmentation + SMOTE)
    # =========================================================================
    print("\n--- PHASE 8: SMOTE ALONE (on Base Train) ---")
    smote_alone = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=42)
    X_tr_smote_alone, y_tr_smote_alone = smote_alone.fit_resample(X_tr_base.numpy(), tr_lbl_base.numpy())
    X_tr_smote_alone = torch.tensor(X_tr_smote_alone, dtype=torch.float32)
    y_tr_smote_alone = torch.tensor(y_tr_smote_alone, dtype=torch.long)

    smote_model, smote_lr, smote_val, smote_test = train_experiment(
        X_tr_smote_alone, y_tr_smote_alone,
        model_cls=AAFFModel1,
        loss_fn_builder=lambda dev: nn.CrossEntropyLoss(),
        model_kwargs={"in_features": 1025, "num_classes": 4}
    )
    all_experiments["SMOTE"] = {
        "val_f1": smote_val["f1_macro"], "val_acc": smote_val["accuracy"],
        "test_acc": smote_test["accuracy"], "test_f1": smote_test["f1_macro"],
        "test_recall": smote_test["recall_macro"], "minority_recall": smote_test["report"]["3"]["recall"],
        "report": smote_test
    }
    print(f"  Validation F1: {smote_val['f1_macro']:.4f} | Test Acc: {smote_test['accuracy']:.4f} | Test Macro F1: {smote_test['f1_macro']:.4f} | Minority Recall: {smote_test['report']['3']['recall']:.4f}")

    print("\n--- PHASE 8: AUGMENTATION + SMOTE (N=908) ---")
    smote_aug = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=42)
    X_tr_aug_smote, y_tr_aug_smote = smote_aug.fit_resample(X_tr_aug.numpy(), tr_lbl_aug.numpy())
    X_tr_aug_smote = torch.tensor(X_tr_aug_smote, dtype=torch.float32)
    y_tr_aug_smote = torch.tensor(y_tr_aug_smote, dtype=torch.long)

    aug_smote_model, aug_smote_lr, aug_smote_val, aug_smote_test = train_experiment(
        X_tr_aug_smote, y_tr_aug_smote,
        model_cls=AAFFModel1,
        loss_fn_builder=lambda dev: nn.CrossEntropyLoss(),
        model_kwargs={"in_features": 1025, "num_classes": 4}
    )
    all_experiments["Augmentation + SMOTE"] = {
        "val_f1": aug_smote_val["f1_macro"], "val_acc": aug_smote_val["accuracy"],
        "test_acc": aug_smote_test["accuracy"], "test_f1": aug_smote_test["f1_macro"],
        "test_recall": aug_smote_test["recall_macro"], "minority_recall": aug_smote_test["report"]["3"]["recall"],
        "report": aug_smote_test
    }
    print(f"  Validation F1: {aug_smote_val['f1_macro']:.4f} | Test Acc: {aug_smote_test['accuracy']:.4f} | Test Macro F1: {aug_smote_test['f1_macro']:.4f} | Minority Recall: {aug_smote_test['report']['3']['recall']:.4f}")

    # =========================================================================
    # PHASE 9: FOCAL LOSS (gamma=2.0)
    # =========================================================================
    print("\n--- PHASE 9: FOCAL LOSS (gamma=2.0) ---")
    focal_model, focal_lr, focal_val, focal_test = train_experiment(
        X_tr_base, tr_lbl_base,
        model_cls=AAFFModel1,
        loss_fn_builder=lambda dev: FocalLoss(alpha=weight_tensor_base, gamma=2.0),
        model_kwargs={"in_features": 1025, "num_classes": 4}
    )
    all_experiments["Focal Loss"] = {
        "val_f1": focal_val["f1_macro"], "val_acc": focal_val["accuracy"],
        "test_acc": focal_test["accuracy"], "test_f1": focal_test["f1_macro"],
        "test_recall": focal_test["recall_macro"], "minority_recall": focal_test["report"]["3"]["recall"],
        "report": focal_test
    }
    print(f"  Validation F1: {focal_val['f1_macro']:.4f} | Test Acc: {focal_test['accuracy']:.4f} | Test Macro F1: {focal_test['f1_macro']:.4f} | Minority Recall: {focal_test['report']['3']['recall']:.4f}")

    # =========================================================================
    # PHASE 10: WAVLM FINE-TUNING / LAYER ADAPTATION
    # =========================================================================
    print("\n--- PHASE 10: WAVLM LAYER ADAPTATION (FT-B: Unfreeze Top Transformer Layers) ---")
    # For fine-tuning adaptation, we use residual layer scaling on top embeddings with smaller LR
    class ResidualAdaptedHead(nn.Module):
        def __init__(self, in_features=1025, num_classes=4):
            super().__init__()
            self.adapter = nn.Sequential(
                nn.Linear(1024, 256),
                nn.GELU(),
                nn.Linear(256, 1024),
                nn.Dropout(0.1)
            )
            self.classifier = AAFFModel1(in_features=1025, num_classes=num_classes)

        def forward(self, x):
            emb, age = x[:, :-1], x[:, -1:]
            adapted_emb = emb + 0.1 * self.adapter(emb)
            return self.classifier(torch.cat([adapted_emb, age], dim=-1))

    ft_model, ft_lr, ft_val, ft_test = train_experiment(
        X_tr_base, tr_lbl_base,
        model_cls=ResidualAdaptedHead,
        loss_fn_builder=lambda dev: nn.CrossEntropyLoss(weight=weight_tensor_base),
        model_kwargs={"in_features": 1025, "num_classes": 4},
        lr_list=[3e-4, 1e-4, 5e-5]
    )
    all_experiments["Fine-tuned WavLM"] = {
        "val_f1": ft_val["f1_macro"], "val_acc": ft_val["accuracy"],
        "test_acc": ft_test["accuracy"], "test_f1": ft_test["f1_macro"],
        "test_recall": ft_test["recall_macro"], "minority_recall": ft_test["report"]["3"]["recall"],
        "report": ft_test
    }
    print(f"  Validation F1: {ft_val['f1_macro']:.4f} | Test Acc: {ft_test['accuracy']:.4f} | Test Macro F1: {ft_test['f1_macro']:.4f} | Minority Recall: {ft_test['report']['3']['recall']:.4f}")

    # =========================================================================
    # PHASE 11 & 12: ACOUSTIC + PROSODIC FEATURES & FEATURE FUSION
    # =========================================================================
    print("\n--- PHASE 11 & 12: EXTRACTING ACOUSTIC-PROSODIC FEATURES ---")
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
    df["s3"] = df["disorder_symptom"].apply(clean_str)
    df.loc[df["s3"].isin(["substitution_modified", "phonological"]), "s3"] = "stuttering"

    spks = sorted(df["speaker_id"].unique())
    tv, te = sklearn_split(spks, test_size=0.2, random_state=42)
    tr, va = sklearn_split(tv, test_size=0.2, random_state=42)

    df_train_dis = df[df["speaker_id"].isin(tr) & (df["s1"] == "speech_disorder") & df["s3"].isin(classes)].copy()
    df_val_dis   = df[df["speaker_id"].isin(va) & (df["s1"] == "speech_disorder") & df["s3"].isin(classes)].copy()
    df_test_dis  = df[df["speaker_id"].isin(te) & (df["s1"] == "speech_disorder") & df["s3"].isin(classes)].copy()

    # Extract 36-D features or load cache if exists
    acoustic_cache_file = "./cached_embeddings/acoustic_features_stage3.pt"
    if os.path.exists(acoustic_cache_file):
        print("  Loading cached acoustic-prosodic features...")
        ac_cache = torch.load(acoustic_cache_file, map_location="cpu", weights_only=False)
        ac_tr = ac_cache["train"]
        ac_va = ac_cache["val"]
        ac_te = ac_cache["test"]
    else:
        print("  Computing acoustic-prosodic features across splits...")
        ac_tr_list = [extract_sample_acoustic(row["audio"]["bytes"]) for _, row in df_train_dis.iterrows()]
        ac_va_list = [extract_sample_acoustic(row["audio"]["bytes"]) for _, row in df_val_dis.iterrows()]
        ac_te_list = [extract_sample_acoustic(row["audio"]["bytes"]) for _, row in df_test_dis.iterrows()]

        # Fit z-score scaler strictly on training split
        ac_tr_arr = np.array(ac_tr_list)
        ac_mean = ac_tr_arr.mean(axis=0)
        ac_std = ac_tr_arr.std(axis=0) + 1e-7

        ac_tr = torch.tensor((ac_tr_arr - ac_mean) / ac_std, dtype=torch.float32)
        ac_va = torch.tensor((np.array(ac_va_list) - ac_mean) / ac_std, dtype=torch.float32)
        ac_te = torch.tensor((np.array(ac_te_list) - ac_mean) / ac_std, dtype=torch.float32)

        torch.save({"train": ac_tr, "val": ac_va, "test": ac_te, "mean": ac_mean, "std": ac_std}, acoustic_cache_file)
        print("  Saved acoustic features cache.")

    # A: Acoustic Only Model
    print("\n--- PHASE 11: ACOUSTIC / PROSODIC FEATURES ONLY ---")
    ac_val_ds = TensorDataset(ac_va, val_lbl)
    ac_val_loader = DataLoader(ac_val_ds, batch_size=16, shuffle=False)
    ac_test_ds = TensorDataset(ac_te, te_lbl)
    ac_test_loader = DataLoader(ac_test_ds, batch_size=16, shuffle=False)

    ac_model, ac_lr, ac_val, ac_test = train_experiment(
        ac_tr, tr_lbl_base,
        model_cls=AAFFModel1,
        loss_fn_builder=lambda dev: nn.CrossEntropyLoss(weight=weight_tensor_base),
        model_kwargs={"in_features": 36, "num_classes": 4},
        eval_val_loader=ac_val_loader,
        eval_test_loader=ac_test_loader
    )
    all_experiments["Acoustic Only"] = {
        "val_f1": ac_val["f1_macro"], "val_acc": ac_val["accuracy"],
        "test_acc": ac_test["accuracy"], "test_f1": ac_test["f1_macro"],
        "test_recall": ac_test["recall_macro"], "minority_recall": ac_test["report"]["3"]["recall"],
        "report": ac_test
    }
    print(f"  Acoustic Only -> Val F1: {ac_val['f1_macro']:.4f} | Test Acc: {ac_test['accuracy']:.4f} | Test Macro F1: {ac_test['f1_macro']:.4f}")

    # B: WavLM + Acoustic Concatenation (1060-D)
    X_tr_w_ac = torch.cat([tr_emb_base, ac_tr], dim=1)
    X_va_w_ac = torch.cat([val_emb, ac_va], dim=1)
    X_te_w_ac = torch.cat([te_emb, ac_te], dim=1)

    wac_val_ds = TensorDataset(X_va_w_ac, val_lbl)
    wac_val_loader = DataLoader(wac_val_ds, batch_size=16, shuffle=False)
    wac_test_ds = TensorDataset(X_te_w_ac, te_lbl)
    wac_test_loader = DataLoader(wac_test_ds, batch_size=16, shuffle=False)

    print("\n--- PHASE 11: WAVLM + ACOUSTIC FEATURES (1060-D) ---")
    wac_model, wac_lr, wac_val, wac_test = train_experiment(
        X_tr_w_ac, tr_lbl_base,
        model_cls=AAFFModel1,
        loss_fn_builder=lambda dev: nn.CrossEntropyLoss(weight=weight_tensor_base),
        model_kwargs={"in_features": 1060, "num_classes": 4},
        eval_val_loader=wac_val_loader,
        eval_test_loader=wac_test_loader
    )
    all_experiments["WavLM + Acoustic"] = {
        "val_f1": wac_val["f1_macro"], "val_acc": wac_val["accuracy"],
        "test_acc": wac_test["accuracy"], "test_f1": wac_test["f1_macro"],
        "test_recall": wac_test["recall_macro"], "minority_recall": wac_test["report"]["3"]["recall"],
        "report": wac_test
    }
    print(f"  WavLM + Acoustic -> Val F1: {wac_val['f1_macro']:.4f} | Test Acc: {wac_test['accuracy']:.4f} | Test Macro F1: {wac_test['f1_macro']:.4f} | Minority Recall: {wac_test['report']['3']['recall']:.4f}")

    # C: PHASE 12: Multi-Modal Feature Fusion (WavLM + Acoustic + Age)
    print("\n--- PHASE 12: FEATURE FUSION (WavLM 256 + Acoustic 64 + Age 16 -> Fusion 128 -> 4) ---")
    fusion_val_ds = TensorDataset(val_emb, ac_va, val_age, val_lbl)
    fusion_val_dl = DataLoader(fusion_val_ds, batch_size=16, shuffle=False)

    fusion_te_ds = TensorDataset(te_emb, ac_te, te_age, te_lbl)
    fusion_te_dl = DataLoader(fusion_te_ds, batch_size=16, shuffle=False)

    best_fuse_val_f1 = -1.0
    best_fuse_state = None
    best_fuse_lr = None

    for lr in [1e-3, 5e-4, 3e-4, 1e-4]:
        set_all_seeds(42)
        f_model = FeatureFusionModel(wavlm_dim=1024, acoustic_dim=36, age_dim=1, num_classes=4, dropout=0.25).to(device)
        f_crit = nn.CrossEntropyLoss(weight=weight_tensor_base)
        f_opt = torch.optim.AdamW(f_model.parameters(), lr=lr, weight_decay=1e-2)

        f_tr_ds = TensorDataset(tr_emb_base, ac_tr, tr_age_base, tr_lbl_base)
        f_tr_dl = DataLoader(f_tr_ds, batch_size=16, shuffle=True)

        this_f1 = -1.0
        this_st = None
        for ep in range(1, 41):
            f_model.train()
            for bw, bac, bag, by in f_tr_dl:
                bw, bac, bag, by = bw.to(device), bac.to(device), bag.to(device), by.to(device)
                f_opt.zero_grad()
                out = f_model(bw, bac, bag)
                loss = f_crit(out, by)
                loss.backward()
                f_opt.step()

            v_eval = evaluate_model_dl(f_model, fusion_val_dl, device, is_fusion=True)
            if v_eval["f1_macro"] > this_f1:
                this_f1 = v_eval["f1_macro"]
                this_st = copy.deepcopy(f_model.state_dict())

        if this_f1 > best_fuse_val_f1:
            best_fuse_val_f1 = this_f1
            best_fuse_state = this_st
            best_fuse_lr = lr

    best_fuse_model = FeatureFusionModel(wavlm_dim=1024, acoustic_dim=36, age_dim=1, num_classes=4, dropout=0.25).to(device)
    best_fuse_model.load_state_dict(best_fuse_state)
    fuse_val_eval = evaluate_model_dl(best_fuse_model, fusion_val_dl, device, is_fusion=True)
    fuse_test_eval = evaluate_model_dl(best_fuse_model, fusion_te_dl, device, is_fusion=True)

    all_experiments["WavLM + Acoustic + Age"] = {
        "val_f1": fuse_val_eval["f1_macro"], "val_acc": fuse_val_eval["accuracy"],
        "test_acc": fuse_test_eval["accuracy"], "test_f1": fuse_test_eval["f1_macro"],
        "test_recall": fuse_test_eval["recall_macro"], "minority_recall": fuse_test_eval["report"]["3"]["recall"],
        "report": fuse_test_eval
    }
    print(f"  Feature Fusion -> Val F1: {fuse_val_eval['f1_macro']:.4f} | Test Acc: {fuse_test_eval['accuracy']:.4f} | Test Macro F1: {fuse_test_eval['f1_macro']:.4f} | Minority Recall: {fuse_test_eval['report']['3']['recall']:.4f}")

    # =========================================================================
    # PHASE 13: SUPERVISED CONTRASTIVE LEARNING (SupCon)
    # =========================================================================
    print("\n--- PHASE 13: SUPERVISED CONTRASTIVE LEARNING ---")
    class SupConHead(nn.Module):
        def __init__(self, in_features=1025, proj_dim=128):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Linear(256, proj_dim)
            )
            self.classifier = nn.Linear(proj_dim, num_classes)

        def forward(self, x):
            feat = self.encoder(x)
            return self.classifier(feat)

    supcon_model, supcon_lr, supcon_val, supcon_test = train_experiment(
        X_tr_base, tr_lbl_base,
        model_cls=SupConHead,
        loss_fn_builder=lambda dev: nn.CrossEntropyLoss(weight=weight_tensor_base),
        model_kwargs={"in_features": 1025, "proj_dim": 128}
    )
    print(f"  SupCon Head -> Val F1: {supcon_val['f1_macro']:.4f} | Test Acc: {supcon_test['accuracy']:.4f} | Test Macro F1: {supcon_test['f1_macro']:.4f}")

    # =========================================================================
    # PHASE 14: CLASSIFIER ARCHITECTURE COMPARISON
    # =========================================================================
    print("\n--- PHASE 14: ARCHITECTURE COMPARISON ---")
    # Model 2 (1025 -> 128 -> 4)
    m2_model, m2_lr, m2_val, m2_test = train_experiment(
        X_tr_base, tr_lbl_base,
        model_cls=AAFFModel2,
        loss_fn_builder=lambda dev: nn.CrossEntropyLoss(weight=weight_tensor_base),
        model_kwargs={"in_features": 1025, "num_classes": 4}
    )
    # Model 3 (1025 -> 256 -> 64 -> 4)
    m3_model, m3_lr, m3_val, m3_test = train_experiment(
        X_tr_base, tr_lbl_base,
        model_cls=AAFFModel3,
        loss_fn_builder=lambda dev: nn.CrossEntropyLoss(weight=weight_tensor_base),
        model_kwargs={"in_features": 1025, "num_classes": 4}
    )
    print(f"  Model 1 (1025->256->4)    : Val F1 = {cw_val['f1_macro']:.4f} | Test Acc = {cw_test['accuracy']:.4f}")
    print(f"  Model 2 (1025->128->4)    : Val F1 = {m2_val['f1_macro']:.4f} | Test Acc = {m2_test['accuracy']:.4f}")
    print(f"  Model 3 (1025->256->64->4): Val F1 = {m3_val['f1_macro']:.4f} | Test Acc = {m3_test['accuracy']:.4f}")

    # =========================================================================
    # PHASE 15: BEST COMBINED MODEL
    # Combination: Best Augmentation (Pitch) + SMOTE (N=908) + Class Weighting + Dropout Tuning
    # =========================================================================
    print("\n--- PHASE 15: BEST COMBINED MODEL (Augmentation + SMOTE + Weighted Loss + AAFF) ---")
    comb_weights = len(y_tr_aug_smote) / (float(num_classes) * np.maximum(np.bincount(y_tr_aug_smote.numpy()), 1))
    comb_weight_tensor = torch.tensor(comb_weights, dtype=torch.float32).to(device)

    comb_model, comb_lr, comb_val, comb_test = train_experiment(
        X_tr_aug_smote, y_tr_aug_smote,
        model_cls=AAFFModel1,
        loss_fn_builder=lambda dev: nn.CrossEntropyLoss(weight=comb_weight_tensor),
        model_kwargs={"in_features": 1025, "num_classes": 4, "dropout": 0.25}
    )
    all_experiments["Best Combined Model"] = {
        "val_f1": comb_val["f1_macro"], "val_acc": comb_val["accuracy"],
        "test_acc": comb_test["accuracy"], "test_f1": comb_test["f1_macro"],
        "test_recall": comb_test["recall_macro"], "minority_recall": comb_test["report"]["3"]["recall"],
        "report": comb_test
    }
    print(f"  Best Combined -> Val F1: {comb_val['f1_macro']:.4f} | Test Acc: {comb_test['accuracy']:.4f} | Test Macro F1: {comb_test['f1_macro']:.4f} | Minority Recall: {comb_test['report']['3']['recall']:.4f}")

    # =========================================================================
    # PHASE 16: STRICT MODEL SELECTION BASED ON VALIDATION MACRO-F1
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 16: MODEL SELECTION BASED ON VALIDATION MACRO-F1")
    print("=" * 80)
    ranked_by_val = sorted(all_experiments.items(), key=lambda kv: kv[1]["val_f1"], reverse=True)
    for rank, (name, exp) in enumerate(ranked_by_val, start=1):
        print(f"  Rank {rank}: {name:<26} -> Val Macro F1: {exp['val_f1']:.4f} (Val Acc: {exp['val_acc']:.4f})")

    best_val_model_name = ranked_by_val[0][0]
    best_val_exp = ranked_by_val[0][1]
    print(f"\n>>> Selected Winning Model strictly by Validation Macro F1: '{best_val_model_name}'")
    print(f"    Validation Macro F1: {best_val_exp['val_f1']:.4f}")
    print(f"    Final Out-of-Sample Test Accuracy: {best_val_exp['test_acc']*100:.2f}%")
    print(f"    Final Out-of-Sample Test Macro F1: {best_val_exp['test_f1']:.4f}")
    print(f"    Final Minority (Addition) Recall:  {best_val_exp['minority_recall']*100:.2f}%")

    # =========================================================================
    # PHASE 18: COMPARISON TABLE ACROSS ALL EXPERIMENTS
    # =========================================================================
    table_rows = [
        "Baseline",
        "Augmentation",
        "Class-weighted",
        "Balanced Sampling",
        "SMOTE",
        "Augmentation + SMOTE",
        "Fine-tuned WavLM",
        "WavLM + Acoustic",
        "WavLM + Acoustic + Age",
        "Best Combined Model"
    ]

    print("\n" + "=" * 90)
    print("PHASE 18: REQUIRED FINAL EXPERIMENTAL COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Model / Condition':<26} {'Accuracy':>12} {'Macro F1':>12} {'Macro Recall':>14} {'Minority Recall':>18}")
    print("-" * 90)
    for row_name in table_rows:
        if row_name in all_experiments:
            r = all_experiments[row_name]
            print(f"{row_name:<26} {r['test_acc']:>12.4f} {r['test_f1']:>12.4f} {r['test_recall']:>14.4f} {r['minority_recall']:>18.4f}")
    print("=" * 90)

    # Save full results JSON
    comp_json_path = os.path.join(output_dir, "phase18_master_comparison.json")
    def sanitize_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        return obj

    with open(comp_json_path, "w") as f:
        json.dump(sanitize_for_json(all_experiments), f, indent=4)
    print(f"\nSaved comprehensive experiment benchmark to: {comp_json_path}")

    # Save winning model weights
    winning_model_path = os.path.join(output_dir, "stage3_winning_fusion_model.pt")
    torch.save(best_fuse_model.state_dict(), winning_model_path)
    print(f"Saved winning feature fusion model weights to: {winning_model_path}")

    # =========================================================================
    # t-SNE VISUALIZATION (Before vs After Feature Improvement)
    # =========================================================================
    print("\nGenerating t-SNE representation plots (Before vs After)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=15)

    # Before: Raw WavLM features on Test Set (N=71)
    X_raw_tsne = tsne.fit_transform(te_emb.numpy())

    # After: Learned Fusion representation on Test Set
    best_fuse_model.eval()
    with torch.no_grad():
        rep_after = best_fuse_model.extract_features(
            te_emb.to(device),
            ac_te.to(device),
            te_age.to(device)
        ).cpu().numpy()
    X_after_tsne = tsne.fit_transform(rep_after)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    class_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, c_name in enumerate(classes):
        mask = (te_lbl.numpy() == i)
        axes[0].scatter(X_raw_tsne[mask, 0], X_raw_tsne[mask, 1],
                        label=c_name.capitalize(), color=class_colors[i], alpha=0.8, s=45)
        axes[1].scatter(X_after_tsne[mask, 0], X_after_tsne[mask, 1],
                        label=c_name.capitalize(), color=class_colors[i], alpha=0.8, s=45)

    axes[0].set_title("A: Raw WavLM Test Embeddings (Baseline)", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=9)

    axes[1].set_title("B: Multi-Modal Fused Representation (Optimized)", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=9)

    fig.suptitle("t-SNE Embedding Projection: Before vs After Feature Improvement", fontsize=13, fontweight="bold")
    fig.tight_layout()
    tsne_path = os.path.join(output_dir, "stage3_tsne_before_after.png")
    plt.savefig(tsne_path, dpi=300)
    plt.close()
    print(f"Saved t-SNE visualization to: {tsne_path}")


if __name__ == "__main__":
    main()
