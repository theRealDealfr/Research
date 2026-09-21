"""
tune_bayesian_stage2.py
Fast Bayesian Hyperparameter Tuning for Stage 2 Multi-Class Disorder Classifier (T3).
Runs on pre-cached WavLM representations (runs 30-50 trials in ~1 minute on CPU).

Optimizes:
  - learning_rate (log-uniform 1e-4 to 1e-2)
  - weight_decay (log-uniform 1e-4 to 1e-1)
  - dropout (0.1 to 0.5)
  - proj_dim (128, 256, 512)
  - batch_size (8, 16, 32)

Maximizes 3-Fold Stratified CV Macro F1 (so minority classes always appear in every
validation fold, closing the val→test gap). Final fine-tuning always uses forced
class-weighted loss + oversampling for the minority class.
"""

import os
import copy
import json
import random
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

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


def oversample_minority(dataset, num_classes, multiplier=3):
    """Repeat minority-class samples so the rarest class has at least
    `multiplier` times more samples in the returned dataset."""
    labels = np.array([y.item() for _, _, y in dataset])
    counts = np.bincount(labels, minlength=num_classes)
    max_count = counts.max()
    indices = list(range(len(dataset)))
    for cls in range(num_classes):
        cls_indices = np.where(labels == cls)[0].tolist()
        # how many extra copies of this class do we need?
        target = min(max_count, counts[cls] * multiplier)
        extra_needed = max(0, target - counts[cls])
        if extra_needed > 0:
            indices += (cls_indices * (extra_needed // len(cls_indices) + 1))[:extra_needed]
    random.shuffle(indices)
    emb = torch.stack([dataset[i][0] for i in indices])
    age = torch.stack([dataset[i][1] for i in indices])
    lbl = torch.stack([dataset[i][2] for i in indices])
    return TensorDataset(emb, age, lbl)


def make_class_weight_tensor(dataset, num_classes, device):
    labels = [y.item() for _, _, y in dataset]
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (float(num_classes) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_single_config(train_data, val_data, device, num_classes, lr, weight_decay,
                        dropout, proj_dim, batch_size, use_weights, epochs=30, seed=42,
                        oversample=False):
    set_all_seeds(seed)
    if oversample:
        train_data = oversample_minority(train_data, num_classes, multiplier=3)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    if use_weights:
        weight_tensor = make_class_weight_tensor(train_data, num_classes, device)
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
    best_model.load_state_dict(best_state)
    val_res = evaluate(best_model, val_loader, device)
    return best_model, val_res


def cv_objective(all_train_data, device, num_classes, lr, weight_decay, dropout,
                 proj_dim, batch_size, epochs, seed, n_folds=3):
    """3-fold stratified CV so every minority class is always in each val fold."""
    emb = all_train_data.tensors[0].numpy()
    age = all_train_data.tensors[1].numpy()
    lbl = all_train_data.tensors[2].numpy()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_f1s = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(emb, lbl)):
        t_emb = torch.tensor(emb[train_idx], dtype=torch.float32)
        t_age = torch.tensor(age[train_idx], dtype=torch.float32)
        t_lbl = torch.tensor(lbl[train_idx], dtype=torch.long)
        v_emb = torch.tensor(emb[val_idx], dtype=torch.float32)
        v_age = torch.tensor(age[val_idx], dtype=torch.float32)
        v_lbl = torch.tensor(lbl[val_idx], dtype=torch.long)

        tr_ds = TensorDataset(t_emb, t_age, t_lbl)
        va_ds = TensorDataset(v_emb, v_age, v_lbl)
        # Always use class weights inside CV — ensures minority class signal
        _, val_res = train_single_config(
            tr_ds, va_ds, device, num_classes,
            lr=lr, weight_decay=weight_decay, dropout=dropout,
            proj_dim=proj_dim, batch_size=batch_size,
            use_weights=True, oversample=True,
            epochs=epochs, seed=seed + fold_idx
        )
        fold_f1s.append(val_res["f1_macro"])

    return float(np.mean(fold_f1s))


def main():
    parser = argparse.ArgumentParser(description="Bayesian Optimization for Stage 2 Classifier")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Bayesian optimization trials")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per trial (per CV fold)")
    parser.add_argument("--final_epochs", type=int, default=80, help="Epochs for final fine-tuning with best params")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=3, help="Number of stratified CV folds for objective")
    args = parser.parse_args()

    data_path = "./cached_embeddings/wavlm_stage2_disorder_symptom.pt"
    if not os.path.exists(data_path):
        print(f"[Error] {data_path} not found.")
        return

    cache = torch.load(data_path, map_location="cpu", weights_only=False)
    disorder_classes = cache["disorder_classes"]
    num_classes = len(disorder_classes)

    train_data = TensorDataset(cache["train"]["embeddings"], cache["train"]["ages"], cache["train"]["labels"])
    val_data   = TensorDataset(cache["val"]["embeddings"],   cache["val"]["ages"],   cache["val"]["labels"])
    test_data  = TensorDataset(cache["test"]["embeddings"],  cache["test"]["ages"],  cache["test"]["labels"])

    # Merge train+val for CV objective so all samples are used in rotation
    all_train_emb = torch.cat([cache["train"]["embeddings"], cache["val"]["embeddings"]], dim=0)
    all_train_age = torch.cat([cache["train"]["ages"],       cache["val"]["ages"]],       dim=0)
    all_train_lbl = torch.cat([cache["train"]["labels"],     cache["val"]["labels"]],     dim=0)
    all_train_data = TensorDataset(all_train_emb, all_train_age, all_train_lbl)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("STAGE 2 BAYESIAN HYPERPARAMETER OPTIMIZATION (3-Fold Stratified CV)")
    print("=" * 70)
    print(f"Device: {device} | Disorder Classes: {disorder_classes}")
    print(f"Samples -> Train+Val (CV pool): {len(all_train_data)}, Test (held-out): {len(test_data)}")
    print(f"Running {args.n_trials} Bayesian Optimization trials x {args.n_folds} CV folds...")
    print("=" * 70)

    trial_history = []

    def objective(trial):
        lr       = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        wd       = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        dropout  = trial.suggest_float("dropout", 0.1, 0.5, step=0.05)
        proj_dim = trial.suggest_categorical("proj_dim", [128, 256, 512])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        # Note: class_weights + oversampling are now always forced;
        # we only tune architecture and optimiser.

        cv_f1 = cv_objective(
            all_train_data=all_train_data, device=device, num_classes=num_classes,
            lr=lr, weight_decay=wd, dropout=dropout, proj_dim=proj_dim,
            batch_size=batch_size, epochs=args.epochs,
            seed=args.seed + trial.number, n_folds=args.n_folds
        )

        trial_history.append({
            "trial": trial.number + 1, "cv_f1": cv_f1,
            "lr": lr, "wd": wd, "dropout": dropout,
            "proj_dim": proj_dim, "batch_size": batch_size
        })

        return cv_f1

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Warm-start: queue the previous best config as trial 0
    study.enqueue_trial({
        "learning_rate": 2.69e-3,
        "weight_decay":  0.0147,
        "dropout":       0.2,
        "proj_dim":      512,
        "batch_size":    8,
    })
    # Also try baseline defaults
    study.enqueue_trial({
        "learning_rate": 1e-3,
        "weight_decay":  1e-2,
        "dropout":       0.2,
        "proj_dim":      256,
        "batch_size":    16,
    })

    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    print("\n" + "=" * 70)
    print("STAGE 2 BAYESIAN OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Best Trial #{best.number + 1}:")
    print(f"  Best Val Macro F1:  {best.value:.4f}")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.2e}" if v < 0.01 else f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")
    print("=" * 70)

    # Save best parameters
    output_dir = "./results_controlled_baselines"
    os.makedirs(output_dir, exist_ok=True)
    best_json_path = os.path.join(output_dir, "best_stage2_bayesian_hyperparams.json")
    with open(best_json_path, "w") as f:
        json.dump(best.params, f, indent=4)
    print(f"\nSaved best hyperparameters to: {best_json_path}")

    # Final training: use ALL train+val data, forced class weights + oversampling
    print("\n" + "=" * 70)
    print("RUNNING FINAL FINE-TUNING (train+val pool, class weights, oversampling)")
    print("=" * 70)
    # A small held-out val slice for best-epoch selection during final tuning
    n_total = len(all_train_data)
    n_val_final = max(int(n_total * 0.15), num_classes * 2)
    lbl_np = all_train_data.tensors[2].numpy()
    # stratified split for final val selection
    from sklearn.model_selection import train_test_split
    tr_idx, va_idx = train_test_split(
        np.arange(n_total), test_size=n_val_final, stratify=lbl_np, random_state=args.seed
    )
    final_train_ds = TensorDataset(
        all_train_data.tensors[0][tr_idx],
        all_train_data.tensors[1][tr_idx],
        all_train_data.tensors[2][tr_idx],
    )
    final_val_ds = TensorDataset(
        all_train_data.tensors[0][va_idx],
        all_train_data.tensors[1][va_idx],
        all_train_data.tensors[2][va_idx],
    )
    final_model, final_val = train_single_config(
        train_data=final_train_ds, val_data=final_val_ds, device=device, num_classes=num_classes,
        lr=best.params["learning_rate"],
        weight_decay=best.params["weight_decay"],
        dropout=best.params["dropout"],
        proj_dim=best.params["proj_dim"],
        batch_size=best.params["batch_size"],
        use_weights=True,       # always force class weights in final model
        oversample=True,        # always oversample minority in final model
        epochs=args.final_epochs,
        seed=args.seed
    )

    test_loader = DataLoader(test_data, batch_size=best.params["batch_size"], shuffle=False)
    test_res = evaluate(final_model, test_loader, device)

    print("\n--- FINAL TEST EVALUATION (STAGE 2 BEST BAYESIAN MODEL) ---")
    print(f"Test Accuracy:  {test_res['accuracy']:.4f}")
    print(f"Test Macro F1:  {test_res['f1_macro']:.4f}")
    print(f"Test Macro Prec:{test_res['precision_macro']:.4f}")
    print(f"Test Macro Rec: {test_res['recall_macro']:.4f}")
    print("\nClassification Report:")
    report_text = classification_report(test_res["y_true"], test_res["y_pred"], target_names=disorder_classes, zero_division=0)
    print(report_text)

    # Save best model checkpoint
    model_save_path = os.path.join(output_dir, "model_stage2_disorder_bayesian_best.pt")
    torch.save(final_model.state_dict(), model_save_path)
    print(f"\nModel saved to: {model_save_path}")

    report_save_path = os.path.join(output_dir, "stage2_bayesian_classification_report.txt")
    with open(report_save_path, "w") as f:
        f.write("=== STAGE 2 BAYESIAN OPTIMIZED CLASSIFICATION REPORT ===\n")
        f.write(f"Best Hyperparameters: {best.params}\n")
        f.write(f"Val Macro F1: {final_val['f1_macro']:.4f} | Test Macro F1: {test_res['f1_macro']:.4f}\n\n")
        f.write(report_text)
    print(f"Report saved to: {report_save_path}")


if __name__ == "__main__":
    main()
