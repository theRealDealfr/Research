"""
tune_bayesian_t2.py
-------------------
Bayesian Hyperparameter Tuning & Evaluation for Task 2 (disorder_type: Articulation vs Phonological)
Comparing:
  1. Unaugmented Baseline (Default Hyperparameters)
  2. Unaugmented + Bayesian Tuning
  3. Augmented (Gender-stratified Pitch Augmentation, Default Hyperparameters)
  4. Augmented + Bayesian Tuning (Combined)

Evaluates:
  - Global Accuracy, Macro F1, Macro Precision, Macro Recall
  - Per-class metrics: Articulation vs Phonological
  - Fairness & Gender Disparity: Male F1 vs Female F1 (|M - F|)
  - Confusion Matrices
"""

import os
import copy
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DynamicAAFFClassifier(nn.Module):
    """
    Model C: Age-Aware Feature Fusion (AAFF) Classifier
    Acoustic embedding (1024-d) + Age (1-d) -> Linear(1025, proj_dim) -> ReLU -> Dropout -> Linear(proj_dim, num_classes)
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


def evaluate(model, dl, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for be, ba, bl in dl:
            be, ba = be.to(device), ba.to(device)
            logits = model(be, ba)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(bl.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "accuracy": acc, "f1_macro": fm, "precision_macro": pm, "recall_macro": rm,
        "f1_weighted": fw, "y_true": y_true, "y_pred": y_pred
    }


def make_class_weight_tensor(labels_list, num_classes, device):
    counts = np.bincount(labels_list, minlength=num_classes)
    total = len(labels_list)
    weights = total / (float(num_classes) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32).to(device)


def oversample_minority(dataset, num_classes, multiplier=3):
    labels = np.array([y.item() for _, _, y in dataset])
    counts = np.bincount(labels, minlength=num_classes)
    max_count = counts.max()
    indices = list(range(len(dataset)))
    for cls in range(num_classes):
        cls_indices = np.where(labels == cls)[0].tolist()
        target = min(max_count, counts[cls] * multiplier)
        extra_needed = max(0, target - counts[cls])
        if extra_needed > 0 and len(cls_indices) > 0:
            indices += (cls_indices * (extra_needed // len(cls_indices) + 1))[:extra_needed]
    random.shuffle(indices)
    emb = torch.stack([dataset[i][0] for i in indices])
    age = torch.stack([dataset[i][1] for i in indices])
    lbl = torch.stack([dataset[i][2] for i in indices])
    return TensorDataset(emb, age, lbl)


def train_model(train_data, val_data, device, num_classes, lr, weight_decay,
                dropout, proj_dim, batch_size, epochs=40, seed=42,
                use_weights=True, oversample=True):
    set_all_seeds(seed)
    if oversample:
        train_data = oversample_minority(train_data, num_classes, multiplier=3)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    if use_weights:
        tr_labels = [y.item() for _, _, y in train_data]
        weight_tensor = make_class_weight_tensor(tr_labels, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    model = DynamicAAFFClassifier(in_features=1025, proj_dim=proj_dim, dropout=dropout, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for be, ba, bl in train_loader:
            be, ba, bl = be.to(device), ba.to(device), bl.to(device)
            optimizer.zero_grad()
            logits = model(be, ba)
            loss = criterion(logits, bl)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_res = evaluate(model, val_loader, device)
        if val_res["f1_macro"] > best_val_f1:
            best_val_f1 = val_res["f1_macro"]
            best_state = copy.deepcopy(model.state_dict())

    best_model = DynamicAAFFClassifier(in_features=1025, proj_dim=proj_dim, dropout=dropout, num_classes=num_classes).to(device)
    best_model.load_state_dict(best_state if best_state is not None else model.state_dict())
    val_res = evaluate(best_model, val_loader, device)
    return best_model, val_res


def cv_objective(train_dataset, device, num_classes, lr, weight_decay, dropout,
                 proj_dim, batch_size, epochs, seed, n_folds=3):
    emb = train_dataset.tensors[0].numpy()
    age = train_dataset.tensors[1].numpy()
    lbl = train_dataset.tensors[2].numpy()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_f1s = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(emb, lbl)):
        t_emb = torch.tensor(emb[tr_idx], dtype=torch.float32)
        t_age = torch.tensor(age[tr_idx], dtype=torch.float32)
        t_lbl = torch.tensor(lbl[tr_idx], dtype=torch.long)
        v_emb = torch.tensor(emb[va_idx], dtype=torch.float32)
        v_age = torch.tensor(age[va_idx], dtype=torch.float32)
        v_lbl = torch.tensor(lbl[va_idx], dtype=torch.long)

        tr_ds = TensorDataset(t_emb, t_age, t_lbl)
        va_ds = TensorDataset(v_emb, v_age, v_lbl)

        _, val_res = train_model(
            tr_ds, va_ds, device, num_classes,
            lr=lr, weight_decay=weight_decay, dropout=dropout,
            proj_dim=proj_dim, batch_size=batch_size,
            epochs=epochs, seed=seed + fold_idx,
            use_weights=True, oversample=True
        )
        fold_f1s.append(val_res["f1_macro"])

    return float(np.mean(fold_f1s))


def evaluate_with_fairness(model, test_dataset, test_genders, classes, device):
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    res = evaluate(model, test_loader, device)
    y_true = res["y_true"]
    y_pred = res["y_pred"]

    rep_dict = classification_report(y_true, y_pred, target_names=classes, zero_division=0, output_dict=True)
    rep_text = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    df_test = pd.DataFrame({"true": y_true, "pred": y_pred, "gender": test_genders})
    gender_metrics = {}
    for g in ["M", "F"]:
        gdf = df_test[df_test["gender"] == g]
        if len(gdf) > 0:
            gt = gdf["true"].values
            gp = gdf["pred"].values
            g_acc = accuracy_score(gt, gp)
            gp_m, gr_m, gf_m, _ = precision_recall_fscore_support(gt, gp, average="macro", zero_division=0)
            gender_metrics[g] = {
                "accuracy": g_acc, "f1_macro": gf_m, "precision_macro": gp_m, "recall_macro": gr_m,
                "support": len(gdf)
            }
        else:
            gender_metrics[g] = {"accuracy": 0.0, "f1_macro": 0.0, "support": 0}

    m_f1 = gender_metrics.get("M", {}).get("f1_macro", 0.0)
    f_f1 = gender_metrics.get("F", {}).get("f1_macro", 0.0)
    disparity = abs(m_f1 - f_f1)

    return {
        "accuracy": res["accuracy"],
        "f1_macro": res["f1_macro"],
        "precision_macro": res["precision_macro"],
        "recall_macro": res["recall_macro"],
        "per_class": {
            classes[0]: rep_dict[classes[0]]["f1-score"],
            classes[1]: rep_dict[classes[1]]["f1-score"]
        },
        "gender_breakdown": {
            "male_f1": m_f1,
            "female_f1": f_f1,
            "disparity": disparity
        },
        "confusion_matrix": cm.tolist(),
        "report_text": rep_text
    }


def run_experiment():
    parser = argparse.ArgumentParser(description="T2 Bayesian Tuning and Augmentation Experiment")
    parser.add_argument("--n_trials", type=int, default=30, help="Optuna Bayesian trials")
    parser.add_argument("--trial_epochs", type=int, default=25, help="Epochs per trial")
    parser.add_argument("--final_epochs", type=int, default=60, help="Epochs for final model training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(" TASK 2: DISORDER_TYPE CLASSIFICATION (ARTICULATION vs PHONOLOGICAL)")
    print(" COMPARING BASELINE vs AUGMENTATION vs BAYESIAN TUNING vs COMBINED")
    print(f" Device: {device} | Random Seed: {args.seed} | Trials: {args.n_trials}")
    print("=" * 80)

    unaug_cache_path = "cached_embeddings/wavlm_t2_disorder_type_unaugmented.pt"
    aug_cache_path   = "cached_embeddings/wavlm_t2_disorder_type_augmented.pt"

    unaug_cache = torch.load(unaug_cache_path, map_location="cpu", weights_only=False)
    aug_cache   = torch.load(aug_cache_path, map_location="cpu", weights_only=False)

    classes = unaug_cache["classes"]
    num_classes = len(classes)

    # 1. Unaugmented Datasets
    unaug_train_ds = TensorDataset(unaug_cache["train"]["embeddings"], unaug_cache["train"]["ages"], unaug_cache["train"]["labels"])
    val_ds         = TensorDataset(unaug_cache["val"]["embeddings"],   unaug_cache["val"]["ages"],   unaug_cache["val"]["labels"])
    test_ds        = TensorDataset(unaug_cache["test"]["embeddings"],  unaug_cache["test"]["ages"],  unaug_cache["test"]["labels"])
    test_genders   = unaug_cache["test"]["genders"]

    # 2. Augmented Datasets
    aug_train_ds   = TensorDataset(aug_cache["train"]["embeddings"],   aug_cache["train"]["ages"],   aug_cache["train"]["labels"])

    print(f"\nDataset Overview:")
    print(f"  Unaugmented Train : {len(unaug_train_ds)} samples")
    print(f"  Augmented Train   : {len(aug_train_ds)} samples (Gender-stratified pitch shifted)")
    print(f"  Validation        : {len(val_ds)} samples (Untouched)")
    print(f"  Test              : {len(test_ds)} samples (Untouched, held-out: 59 Articulation, 12 Phonological)")

    results_dir = "./results_controlled_baselines"
    os.makedirs(results_dir, exist_ok=True)

    summary_records = {}

    # -------------------------------------------------------------------------
    # EXPERIMENT 1: Unaugmented Baseline (Default Hyperparameters)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" [1/4] Running: UNAUGMENTED BASELINE (Default Hyperparams)")
    print("=" * 70)
    default_params = {
        "learning_rate": 1e-3, "weight_decay": 1e-2, "dropout": 0.2, "proj_dim": 256, "batch_size": 16
    }
    m1, _ = train_model(
        unaug_train_ds, val_ds, device, num_classes,
        lr=default_params["learning_rate"], weight_decay=default_params["weight_decay"],
        dropout=default_params["dropout"], proj_dim=default_params["proj_dim"],
        batch_size=default_params["batch_size"], epochs=args.final_epochs, seed=args.seed
    )
    res_m1 = evaluate_with_fairness(m1, test_ds, test_genders, classes, device)
    summary_records["1_Unaugmented_Baseline"] = {"params": default_params, "metrics": res_m1}
    print(f"Result -> Test Acc: {res_m1['accuracy']:.4f} | Test Macro F1: {res_m1['f1_macro']:.4f} | Disparity: {res_m1['gender_breakdown']['disparity']:.4f}")

    # -------------------------------------------------------------------------
    # EXPERIMENT 2: Augmented Only (Default Hyperparameters)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" [2/4] Running: AUGMENTATION ONLY (Default Hyperparams)")
    print("=" * 70)
    m2, _ = train_model(
        aug_train_ds, val_ds, device, num_classes,
        lr=default_params["learning_rate"], weight_decay=default_params["weight_decay"],
        dropout=default_params["dropout"], proj_dim=default_params["proj_dim"],
        batch_size=default_params["batch_size"], epochs=args.final_epochs, seed=args.seed
    )
    res_m2 = evaluate_with_fairness(m2, test_ds, test_genders, classes, device)
    summary_records["2_Augmented_Default"] = {"params": default_params, "metrics": res_m2}
    print(f"Result -> Test Acc: {res_m2['accuracy']:.4f} | Test Macro F1: {res_m2['f1_macro']:.4f} | Disparity: {res_m2['gender_breakdown']['disparity']:.4f}")

    # -------------------------------------------------------------------------
    # EXPERIMENT 3: Unaugmented + Bayesian Tuning
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" [3/4] Running: UNAUGMENTED + BAYESIAN TUNING")
    print("=" * 70)
    def objective_unaug(trial):
        lr       = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        wd       = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        dropout  = trial.suggest_float("dropout", 0.1, 0.5, step=0.05)
        proj_dim = trial.suggest_categorical("proj_dim", [128, 256, 512])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

        return cv_objective(
            unaug_train_ds, device, num_classes,
            lr=lr, weight_decay=wd, dropout=dropout,
            proj_dim=proj_dim, batch_size=batch_size,
            epochs=args.trial_epochs, seed=args.seed + trial.number
        )

    sampler = TPESampler(seed=args.seed)
    study_unaug = optuna.create_study(direction="maximize", sampler=sampler)
    study_unaug.enqueue_trial(default_params)
    study_unaug.optimize(objective_unaug, n_trials=args.n_trials)
    best_unaug_params = study_unaug.best_trial.params

    print(f"Optimal Unaugmented Params: {best_unaug_params}")
    m3, _ = train_model(
        unaug_train_ds, val_ds, device, num_classes,
        lr=best_unaug_params["learning_rate"], weight_decay=best_unaug_params["weight_decay"],
        dropout=best_unaug_params["dropout"], proj_dim=best_unaug_params["proj_dim"],
        batch_size=best_unaug_params["batch_size"], epochs=args.final_epochs, seed=args.seed
    )
    res_m3 = evaluate_with_fairness(m3, test_ds, test_genders, classes, device)
    summary_records["3_Unaugmented_Bayesian"] = {"params": best_unaug_params, "metrics": res_m3}
    print(f"Result -> Test Acc: {res_m3['accuracy']:.4f} | Test Macro F1: {res_m3['f1_macro']:.4f} | Disparity: {res_m3['gender_breakdown']['disparity']:.4f}")

    # -------------------------------------------------------------------------
    # EXPERIMENT 4: COMBINED AUGMENTATION + BAYESIAN TUNING
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" [4/4] Running: COMBINED AUGMENTATION + BAYESIAN TUNING")
    print("=" * 70)
    def objective_aug(trial):
        lr       = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        wd       = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        dropout  = trial.suggest_float("dropout", 0.1, 0.5, step=0.05)
        proj_dim = trial.suggest_categorical("proj_dim", [128, 256, 512])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

        return cv_objective(
            aug_train_ds, device, num_classes,
            lr=lr, weight_decay=wd, dropout=dropout,
            proj_dim=proj_dim, batch_size=batch_size,
            epochs=args.trial_epochs, seed=args.seed + trial.number
        )

    study_aug = optuna.create_study(direction="maximize", sampler=sampler)
    study_aug.enqueue_trial(default_params)
    study_aug.enqueue_trial(best_unaug_params)
    study_aug.optimize(objective_aug, n_trials=args.n_trials)
    best_aug_params = study_aug.best_trial.params

    print(f"Optimal Combined (Aug + Bayesian) Params: {best_aug_params}")
    m4, val_m4 = train_model(
        aug_train_ds, val_ds, device, num_classes,
        lr=best_aug_params["learning_rate"], weight_decay=best_aug_params["weight_decay"],
        dropout=best_aug_params["dropout"], proj_dim=best_aug_params["proj_dim"],
        batch_size=best_aug_params["batch_size"], epochs=args.final_epochs, seed=args.seed
    )
    res_m4 = evaluate_with_fairness(m4, test_ds, test_genders, classes, device)
    summary_records["4_Augmented_Bayesian_Combined"] = {"params": best_aug_params, "metrics": res_m4}
    print(f"Result -> Test Acc: {res_m4['accuracy']:.4f} | Test Macro F1: {res_m4['f1_macro']:.4f} | Disparity: {res_m4['gender_breakdown']['disparity']:.4f}")

    # Save Best Model Checkpoint
    best_model_path = os.path.join(results_dir, "model_t2_disorder_type_bayesian_augmented_best.pt")
    torch.save(m4.state_dict(), best_model_path)
    print(f"\nSaved Best Combined T2 Model to: {best_model_path}")

    # -------------------------------------------------------------------------
    # PRINT COMPREHENSIVE COMPARISON TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" TASK 2 PERFORMANCE COMPARISON: BASELINE vs AUGMENTATION vs BAYESIAN vs COMBINED")
    print("=" * 90)
    print(f"{'Configuration':<30} {'Accuracy':>10} {'Macro F1':>10} {'Artic F1':>10} {'Phon F1':>10} {'Male F1':>10} {'Fem F1':>10} {'Disparity':>10}")
    print("-" * 90)
    for name, data in summary_records.items():
        m = data["metrics"]
        print(f"{name:<30} {m['accuracy']*100:>9.2f}% {m['f1_macro']*100:>9.2f}% {m['per_class']['articulation']*100:>9.2f}% {m['per_class']['phonological']*100:>9.2f}% {m['gender_breakdown']['male_f1']*100:>9.2f}% {m['gender_breakdown']['female_f1']*100:>9.2f}% {m['gender_breakdown']['disparity']*100:>9.2f}%")
    print("=" * 90)

    # Save JSON summary
    summary_json_path = os.path.join(results_dir, "t2_bayesian_augmented_comparison.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_records, f, indent=4)
    print(f"Saved JSON comparison metrics to: {summary_json_path}")

    # Save Full Text Report
    report_path = os.path.join(results_dir, "t2_bayesian_augmented_report.txt")
    with open(report_path, "w") as f:
        f.write("=== TASK 2: DISORDER_TYPE CLASSIFICATION PERFORMANCE REPORT ===\n")
        f.write("Combining Bayesian Hyperparameter Tuning and Gender-Stratified Pitch Augmentation\n\n")
        for name, data in summary_records.items():
            m = data["metrics"]
            f.write(f"--- Configuration: {name} ---\n")
            f.write(f"Hyperparameters: {data['params']}\n")
            f.write(f"Test Accuracy:  {m['accuracy']:.4f}\n")
            f.write(f"Test Macro F1:  {m['f1_macro']:.4f}\n")
            f.write(f"Test Precision: {m['precision_macro']:.4f}\n")
            f.write(f"Test Recall:    {m['recall_macro']:.4f}\n")
            f.write(f"Male Macro F1:  {m['gender_breakdown']['male_f1']:.4f} | Female Macro F1: {m['gender_breakdown']['female_f1']:.4f} | Disparity: {m['gender_breakdown']['disparity']:.4f}\n")
            f.write(f"Confusion Matrix [articulation, phonological]: {m['confusion_matrix']}\n")
            f.write("\nClassification Report:\n")
            f.write(m["report_text"])
            f.write("\n" + "="*60 + "\n\n")
    print(f"Saved full text report to: {report_path}")


if __name__ == "__main__":
    run_experiment()
