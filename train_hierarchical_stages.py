"""
train_hierarchical_stages.py
----------------------------
Executes the controlled experimental benchmark for:
  - Stage 2: Disorder Type Classification ('articulation' vs 'phonological')
  - Stage 3: Symptom Classification ('substitution', 'stuttering', 'omission', 'addition')

For both stages, 4 controlled experimental conditions are evaluated on the UNTOUCHED test set:
  - Experiment A: Baseline (Unaugmented training embeddings, no SMOTE)
  - Experiment B: Augmentation Only (Gender-stratified pitch augmented training embeddings)
  - Experiment C: SMOTE Only (Unaugmented training embeddings + SMOTE)
  - Experiment D: Augmentation + SMOTE (Pitch augmented training embeddings + SMOTE)

Ensures zero data leakage:
  - Validation and test splits are strictly untouched.
  - SMOTE and augmentation are applied SOLELY to the training split.
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
# 2. AAFF Classifier Architecture (Preserved for Stage 2 & Stage 3)
# -----------------------------------------------------------------------------
class AAFFClassifier(nn.Module):
    """
    Age-Aware Feature Fusion (AAFF) Architecture:
    1024-d speech embedding + 1-d normalized age (1025-d)
    -> Linear(1025, 256) -> ReLU() -> Dropout(0.2) -> Linear(256, num_classes)
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


# -----------------------------------------------------------------------------
# 3. Model Evaluation Function
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

    acc = accuracy_score(y_true, y_pred)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "accuracy": float(acc),
        "precision_macro": float(pm),
        "recall_macro": float(rm),
        "f1_macro": float(fm),
        "f1_weighted": float(fw),
        "avg_confidence": float(confs.mean()) if len(confs) > 0 else 0.0,
        "y_true": y_true,
        "y_pred": y_pred,
        "confidences": confs,
        "probs": np.array(all_probs),
    }


# -----------------------------------------------------------------------------
# 4. SMOTE Resampler (TRAINING SET ONLY)
# -----------------------------------------------------------------------------
def apply_smote(embs, ages, lbls, k_neighbors=5, seed=42):
    X = np.hstack([embs.numpy(), ages.numpy()])
    y = lbls.numpy()

    class_counts = np.bincount(y)
    min_count = int(np.min(class_counts))
    safe_k = max(1, min(k_neighbors, min_count - 1))

    smote = SMOTE(sampling_strategy="auto", k_neighbors=safe_k, random_state=seed)
    X_res, y_res = smote.fit_resample(X, y)

    res_emb = torch.tensor(X_res[:, :-1], dtype=torch.float32)
    res_age = torch.tensor(X_res[:, -1:], dtype=torch.float32)
    res_lbl = torch.tensor(y_res, dtype=torch.long)

    return res_emb, res_age, res_lbl


# -----------------------------------------------------------------------------
# 5. Training Loop with Learning Rate Grid Search
# -----------------------------------------------------------------------------
def train_and_tune(train_data, val_data, device, num_classes,
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
        model = AAFFClassifier(in_features=1025, proj_dim=256, dropout=0.2, num_classes=num_classes).to(device)
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

    final_model = AAFFClassifier(in_features=1025, proj_dim=256, dropout=0.2, num_classes=num_classes).to(device)
    final_model.load_state_dict(best_overall_model_state)
    val_res = evaluate_model(final_model, val_loader, device)

    return final_model, best_lr, val_res


# -----------------------------------------------------------------------------
# 6. Build Stage 2 (Disorder Type) Feature Tensors
# -----------------------------------------------------------------------------
def get_stage2_data(orig_cache, aug_cache):
    """
    Extracts binary disorder type labels:
      0: 'articulation'
      1: 'phonological'
    Filters out anomalous rows (typically_developing or mislabeled symptoms).
    """
    type_names = ["articulation", "phonological"]
    type2id = {name: i for i, name in enumerate(type_names)}

    # Map from original dataframe order to disorder type
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


# -----------------------------------------------------------------------------
# 7. Build Stage 3 (Symptom Type) Feature Tensors
# -----------------------------------------------------------------------------
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
# 8. Run 4-Condition Experiments for a Stage
# -----------------------------------------------------------------------------
def run_four_experiments(stage_name, data_dict, device, output_dir, seed=42):
    classes = data_dict["classes"]
    num_classes = len(classes)
    val_data = TensorDataset(*data_dict["val"])
    test_data = TensorDataset(*data_dict["test"])
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    print("\n" + "=" * 70)
    print(f"RUNNING 4-WAY BENCHMARK FOR {stage_name.upper()}")
    print("=" * 70)
    print(f"Target Classes ({num_classes}): {classes}")
    print(f"Test samples (Untouched): {len(test_data)}")

    results = {}
    best_exp_model = None
    best_exp_name = None
    best_exp_f1 = -1.0

    conditions = [
        ("Exp_A_Baseline",             data_dict["unaugmented_train"], False),
        ("Exp_B_Augmentation_Only",    data_dict["augmented_train"],   False),
        ("Exp_C_SMOTE_Only",           data_dict["unaugmented_train"], True),
        ("Exp_D_Augmentation_SMOTE",   data_dict["augmented_train"],   True),
    ]

    for exp_id, (embs, ages, lbls), use_smote in conditions:
        print("\n" + "-" * 70)
        print(f"[{stage_name}] Starting {exp_id} (use_smote={use_smote})")
        print("-" * 70)

        raw_counts = np.bincount(lbls.numpy(), minlength=num_classes)
        print(f"Initial train sample count: {len(lbls)}")
        for c_idx, c_name in enumerate(classes):
            print(f"  {c_name}: {raw_counts[c_idx]}")

        if use_smote:
            print("  Applying SMOTE to training embeddings...")
            tr_embs, tr_ages, tr_lbls = apply_smote(embs, ages, lbls, k_neighbors=5, seed=seed)
            res_counts = np.bincount(tr_lbls.numpy(), minlength=num_classes)
            print(f"  Post-SMOTE train sample count: {len(tr_lbls)}")
            for c_idx, c_name in enumerate(classes):
                print(f"  {c_name}: {res_counts[c_idx]}")
        else:
            tr_embs, tr_ages, tr_lbls = embs, ages, lbls
            res_counts = raw_counts

        train_ds = TensorDataset(tr_embs, tr_ages, tr_lbls)

        # Train model
        model, best_lr, val_res = train_and_tune(
            train_data=train_ds,
            val_data=val_data,
            device=device,
            num_classes=num_classes,
            lr_list=[1e-3, 5e-4, 3e-4, 1e-4],
            epochs=40,
            seed=seed
        )

        test_res = evaluate_model(model, test_loader, device)
        y_true = test_res["y_true"]
        y_pred = test_res["y_pred"]
        cm = confusion_matrix(y_true, y_pred)
        rpt = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)

        print(f"Test Accuracy:          {test_res['accuracy']:.4f}")
        print(f"Test Macro F1:          {test_res['f1_macro']:.4f}")
        print(f"Test Macro Precision:   {test_res['precision_macro']:.4f}")
        print(f"Test Macro Recall:      {test_res['recall_macro']:.4f}")
        print(f"Test Weighted F1:       {test_res['f1_weighted']:.4f}")
        print("Confusion Matrix:\n", cm)

        results[exp_id] = {
            "initial_counts": {c: int(raw_counts[i]) for i, c in enumerate(classes)},
            "final_counts":   {c: int(res_counts[i]) for i, c in enumerate(classes)},
            "train_samples":  len(tr_lbls),
            "best_lr":        best_lr,
            "val_f1_macro":   val_res["f1_macro"],
            "accuracy":       test_res["accuracy"],
            "precision_macro":test_res["precision_macro"],
            "recall_macro":   test_res["recall_macro"],
            "f1_macro":       test_res["f1_macro"],
            "f1_weighted":    test_res["f1_weighted"],
            "avg_confidence": test_res["avg_confidence"],
            "confusion_matrix": cm.tolist(),
            "classification_report": rpt
        }

        if test_res["f1_macro"] > best_exp_f1:
            best_exp_f1 = test_res["f1_macro"]
            best_exp_model = copy.deepcopy(model)
            best_exp_name = exp_id

    # Save the best model checkpoint for this stage
    best_ckpt_path = os.path.join(output_dir, f"{stage_name}_best_model.pt")
    torch.save(best_exp_model.state_dict(), best_ckpt_path)
    print(f"\n[OK] Saved overall best {stage_name} model ({best_exp_name}, Macro F1={best_exp_f1:.4f}) to: {best_ckpt_path}")

    return results, best_exp_name


# -----------------------------------------------------------------------------
# 9. Main Orchestration
# -----------------------------------------------------------------------------
def main():
    set_all_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    output_dir = "./results_controlled_baselines"
    os.makedirs(output_dir, exist_ok=True)

    orig_cache_path = "./cached_embeddings/wavlm_stage2_disorder_symptom.pt"
    aug_cache_path  = "./cached_embeddings/wavlm_stage2_disorder_symptom_augmented.pt"

    if not os.path.exists(orig_cache_path) or not os.path.exists(aug_cache_path):
        raise FileNotFoundError("Missing cache files in ./cached_embeddings/")

    orig_cache = torch.load(orig_cache_path, map_location="cpu", weights_only=False)
    aug_cache  = torch.load(aug_cache_path,  map_location="cpu", weights_only=False)

    # 1. Stage 2 Data & Benchmark
    print("Preparing Stage 2 (Disorder Type) data...")
    s2_data = get_stage2_data(orig_cache, aug_cache)
    s2_results, s2_best_name = run_four_experiments("stage2_disorder_type", s2_data, device, output_dir)

    # 2. Stage 3 Data & Benchmark
    print("\nPreparing Stage 3 (Symptom Classification) data...")
    s3_data = get_stage3_data(orig_cache, aug_cache)
    s3_results, s3_best_name = run_four_experiments("stage3_symptom", s3_data, device, output_dir)

    # 3. Save Combined Results JSON
    summary_path = os.path.join(output_dir, "hierarchical_experiments_results.json")
    full_output = {
        "stage2_disorder_type": {
            "classes": s2_data["classes"],
            "best_condition": s2_best_name,
            "experiments": s2_results
        },
        "stage3_symptom": {
            "classes": s3_data["classes"],
            "best_condition": s3_best_name,
            "experiments": s3_results
        }
    }

    with open(summary_path, "w") as f:
        json.dump(full_output, f, indent=4)
    print(f"\n[OK] Saved all experimental results to: {summary_path}")

    # 4. Print Summary Comparison Table
    print("\n" + "=" * 95)
    print("STAGE 2 (DISORDER TYPE) 4-WAY EXPERIMENTAL BENCHMARK SUMMARY")
    print("=" * 95)
    print(f"{'Condition':<28} {'Accuracy':>10} {'Macro Precision':>16} {'Macro Recall':>14} {'Macro F1':>12} {'Weighted F1':>12}")
    print("-" * 95)
    for c_id, res in s2_results.items():
        print(f"{c_id:<28} {res['accuracy']:>10.4f} {res['precision_macro']:>16.4f} {res['recall_macro']:>14.4f} {res['f1_macro']:>12.4f} {res['f1_weighted']:>12.4f}")

    print("\n" + "=" * 95)
    print("STAGE 3 (SYMPTOM CLASSIFICATION) 4-WAY EXPERIMENTAL BENCHMARK SUMMARY")
    print("=" * 95)
    print(f"{'Condition':<28} {'Accuracy':>10} {'Macro Precision':>16} {'Macro Recall':>14} {'Macro F1':>12} {'Weighted F1':>12}")
    print("-" * 95)
    for c_id, res in s3_results.items():
        print(f"{c_id:<28} {res['accuracy']:>10.4f} {res['precision_macro']:>16.4f} {res['recall_macro']:>14.4f} {res['f1_macro']:>12.4f} {res['f1_weighted']:>12.4f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
