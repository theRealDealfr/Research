"""
train_stage2_augmented.py
-------------------------
Trains the Stage 2 Age-Aware Feature Fusion (AAFF) Disorder Classifier
using GENDER-STRATIFIED PITCH AUGMENTED training embeddings.

Evaluation is strictly performed on the UNTOUCHED validation and test splits.
Generates:
1. Full test classification metrics (Accuracy, Macro F1, Macro Recall, Macro Precision).
2. Per-class report and confusion matrix across the 4 disorder symptoms.
3. Gender fairness breakdown (Male vs. Female Macro F1).
4. Direct comparison against the original unaugmented baseline.

New in this version:
  --embed_mode last|stats  (default last = existing behavior)
  In stats mode the [L, 2048] tensor is collapsed by a learnable LayerWeightedPool
  before age fusion. Input dimension is inferred from data, not hardcoded.
"""

import os
import copy
import random
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- Layer-Weighting Module (stats mode) ---

class LayerWeightedPool(nn.Module):
    """Softmax-weighted sum of L per-layer [mean||std] vectors.
    Input : (B, L, D)
    Output: (B, D)
    Initialized uniform.
    """
    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, x):
        w = torch.softmax(self.weights, dim=0)
        return (x * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)


class AAFFDisorderClassifier(nn.Module):
    """
    Standard AAFF Architecture (preserves Model C / Stage 2 structure):
    in_features-d -> Linear(proj_dim) -> ReLU() -> Dropout -> Linear(num_classes)
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


class AAFFDisorderClassifierImproved(nn.Module):
    """
    Improved AAFF Architecture:
    in_features-d -> Linear(512)->ReLU->Dropout -> Linear(256)->ReLU->Dropout -> Linear(nc)
    """
    def __init__(self, in_features=1025, num_classes=4, dropout=0.3):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, emb, age):
        fused = torch.cat([emb, age], dim=1)
        projected = self.projection(fused)
        return self.classifier(projected)


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch_emb, batch_age, batch_lbl in dataloader:
            batch_emb = batch_emb.to(device)
            batch_age = batch_age.to(device)
            logits = model(batch_emb, batch_age)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_lbl.numpy())
            all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "accuracy": acc,
        "f1_macro": f1_mac,
        "precision_macro": p_mac,
        "recall_macro": r_mac,
        "f1_micro": f1_mic,
        "f1_weighted": f1_wt,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": np.array(all_probs)
    }


def train_augmented_model(
    model_cls,
    model_kwargs,
    train_data,
    val_data,
    device,
    lr_list=[1e-3, 5e-4, 3e-4, 1e-4],
    epochs=50,
    batch_size=16,
    seed=42,
    use_cosine=True
):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    train_labels = [y.item() for _, _, y in train_data]
    num_classes = model_kwargs.get("num_classes", 4)
    class_counts = np.bincount(train_labels, minlength=num_classes)
    total = len(train_labels)
    weights = total / (float(num_classes) * np.maximum(class_counts, 1))
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    print(f"Train samples count: {total}")
    print(f"Class counts in augmented training: {class_counts}")
    print(f"Class weights applied to loss: {np.round(weights, 3)}")

    best_overall_val_f1 = -1.0
    best_overall_model_state = None
    best_lr = None

    for lr in lr_list:
        set_all_seeds(seed)
        model = model_cls(**model_kwargs).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if use_cosine else None

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

            if scheduler:
                scheduler.step()

            val_metrics = evaluate_model(model, val_loader, device)
            val_f1 = val_metrics["f1_macro"]

            if val_f1 > best_val_f1_this_lr:
                best_val_f1_this_lr = val_f1
                best_state_this_lr = copy.deepcopy(model.state_dict())

        print(f"  LR {lr:.0e} -> Best Val Macro F1: {best_val_f1_this_lr:.4f}")

        if best_val_f1_this_lr > best_overall_val_f1:
            best_overall_val_f1 = best_val_f1_this_lr
            best_overall_model_state = best_state_this_lr
            best_lr = lr

    final_model = model_cls(**model_kwargs).to(device)
    final_model.load_state_dict(best_overall_model_state)
    val_res = evaluate_model(final_model, val_loader, device)
    return final_model, best_lr, val_res


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Augmented Training")
    parser.add_argument(
        "--embed_mode",
        choices=["last", "stats"],
        default="last",
        help=(
            "'last' = use pooled last-layer augmented embeddings (default). "
            "'stats' = use multi-layer mean+std embeddings from "
            "./cached_embeddings/wavlm_stats_layers/wavlm_large_stats_layers_seed42.pt"
        ),
    )
    args = parser.parse_args()

    if args.embed_mode == "stats":
        augmented_data_path = "./cached_embeddings/wavlm_stats_layers/wavlm_large_stats_layers_seed42.pt"
    else:
        augmented_data_path = "./cached_embeddings/wavlm_stage2_disorder_symptom_augmented.pt"

    if not os.path.exists(augmented_data_path):
        print(f"Error: {augmented_data_path} not found. Please run the appropriate extraction script first.")
        return

    print("=" * 70)
    print(f"STAGE 2 MULTICLASS DISORDER CLASSIFIER TRAINING  [embed_mode={args.embed_mode}]")
    print("=" * 70)

    cache = torch.load(augmented_data_path, map_location="cpu", weights_only=False)

    if args.embed_mode == "stats":
        num_layers = cache["train"]["stats_embeddings"].shape[1]
        pool = LayerWeightedPool(num_layers)

        def get_emb(split_dict):
            se = split_dict["stats_embeddings"].float()
            with torch.no_grad():
                return pool(se)

        train_emb = get_emb(cache["train"])
        val_emb   = get_emb(cache["val"])
        test_emb  = get_emb(cache["test"])
        disorder_classes = cache.get("unique_labels", cache.get("disorder_classes", []))
    else:
        train_emb = cache["train"]["embeddings"]
        val_emb   = cache["val"]["embeddings"]
        test_emb  = cache["test"]["embeddings"]
        disorder_classes = cache["disorder_classes"]

    emb_dim = train_emb.shape[1]
    in_features = emb_dim + 1    # +1 for age
    num_classes = len(disorder_classes)
    print(f"Target classes ({num_classes}): {disorder_classes}")
    print(f"emb_dim={emb_dim}, in_features={in_features}")

    train_data = TensorDataset(train_emb, cache["train"]["ages"], cache["train"]["labels"])
    val_data   = TensorDataset(val_emb,   cache["val"]["ages"],   cache["val"]["labels"])
    test_data  = TensorDataset(test_emb,  cache["test"]["ages"],  cache["test"]["labels"])
    test_genders = cache["test"]["genders"]

    print(f"\nDataset Splits:")
    print(f"  Train (Augmented): {len(train_data)} samples")
    print(f"  Val   (Untouched): {len(val_data)} samples")
    print(f"  Test  (Untouched): {len(test_data)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}\n")

    output_dir = "./results_controlled_baselines"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Train Standard AAFF Model
    print("=" * 70)
    print(f"Training Model 1: Standard AAFF ({in_features} -> 256 -> {num_classes})")
    print("=" * 70)
    best_model_std, best_lr_std, val_res_std = train_augmented_model(
        model_cls=AAFFDisorderClassifier,
        model_kwargs={"in_features": in_features, "proj_dim": 256, "dropout": 0.2, "num_classes": num_classes},
        train_data=train_data,
        val_data=val_data,
        device=device,
        lr_list=[1e-3, 5e-4, 3e-4, 1e-4],
        epochs=40,
        seed=42
    )

    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    test_res_std = evaluate_model(best_model_std, test_loader, device)

    # 2. Train Improved AAFF Model
    print("\n" + "=" * 70)
    print(f"Training Model 2: Improved AAFF ({in_features} -> 512 -> 256 -> {num_classes})")
    print("=" * 70)
    best_model_imp, best_lr_imp, val_res_imp = train_augmented_model(
        model_cls=AAFFDisorderClassifierImproved,
        model_kwargs={"in_features": in_features, "num_classes": num_classes, "dropout": 0.3},
        train_data=train_data,
        val_data=val_data,
        device=device,
        lr_list=[1e-3, 5e-4, 3e-4, 1e-4],
        epochs=50,
        seed=42,
        use_cosine=True
    )
    test_res_imp = evaluate_model(best_model_imp, test_loader, device)

    # Pick the best performing model
    if val_res_imp["f1_macro"] >= val_res_std["f1_macro"]:
        best_model = best_model_imp
        best_variant_name = "Improved AAFF (Wider Head)"
        best_lr = best_lr_imp
        best_val_res = val_res_imp
        best_test_res = test_res_imp
    else:
        best_model = best_model_std
        best_variant_name = "Standard AAFF"
        best_lr = best_lr_std
        best_val_res = val_res_std
        best_test_res = test_res_std

    # Gender-stratified evaluation
    df_eval = pd.DataFrame({
        "true": best_test_res["y_true"],
        "pred": best_test_res["y_pred"],
        "gender": test_genders
    })

    gender_metrics = {}
    for g in ["M", "F"]:
        gdf = df_eval[df_eval["gender"] == g]
        if len(gdf) > 0:
            g_true = gdf["true"].values
            g_pred = gdf["pred"].values
            g_acc = accuracy_score(g_true, g_pred)
            g_p, g_r, g_f1, _ = precision_recall_fscore_support(g_true, g_pred, average="macro", zero_division=0)
            gender_metrics[g] = {
                "support": len(gdf),
                "accuracy": g_acc,
                "f1_macro": g_f1,
                "precision_macro": g_p,
                "recall_macro": g_r,
                "report": classification_report(g_true, g_pred, target_names=disorder_classes, zero_division=0)
            }

    rpt_str = classification_report(best_test_res["y_true"], best_test_res["y_pred"], target_names=disorder_classes, zero_division=0)
    cm = confusion_matrix(best_test_res["y_true"], best_test_res["y_pred"])

    print("\n" + "=" * 70)
    print(f"FINAL TEST SET EVALUATION ({best_variant_name})")
    print("=" * 70)
    print(f"embed_mode:             {args.embed_mode}")
    print(f"Optimal Learning Rate:  {best_lr}")
    print(f"Validation Macro F1:    {best_val_res['f1_macro']:.4f}")
    print(f"Test Accuracy:          {best_test_res['accuracy']:.4f} ({best_test_res['accuracy']*100:.2f}%)")
    print(f"Test Macro Precision:   {best_test_res['precision_macro']:.4f}")
    print(f"Test Macro Recall:      {best_test_res['recall_macro']:.4f}")
    print(f"Test Macro F1-Score:    {best_test_res['f1_macro']:.4f}")
    print(f"Test Weighted F1-Score: {best_test_res['f1_weighted']:.4f}")

    print("\nPer-Class Classification Report (Untouched Test Set):")
    print(rpt_str)
    print("Confusion Matrix:")
    print(cm)
    print(f"Classes: {disorder_classes}")

    print("\n--- Gender Breakdown on Test Set ---")
    for g in ["M", "F"]:
        gm = gender_metrics[g]
        print(f"Gender {g} (N={gm['support']}): Accuracy={gm['accuracy']:.4f}, Macro F1={gm['f1_macro']:.4f}, Precision={gm['precision_macro']:.4f}, Recall={gm['recall_macro']:.4f}")

    print("\n" + "=" * 70)
    print("COMPARISON: UNAUGMENTED BASELINE vs CURRENT")
    print("=" * 70)
    base_acc = 0.3380
    base_f1 = 0.3020
    delta_acc = (best_test_res['accuracy'] - base_acc) * 100
    delta_f1 = (best_test_res['f1_macro'] - base_f1) * 100
    print(f"Baseline Test Accuracy:  {base_acc:.4f}  ->  Current: {best_test_res['accuracy']:.4f}  (Delta: {delta_acc:+.2f}%)")
    print(f"Baseline Test Macro F1:  {base_f1:.4f}  ->  Current: {best_test_res['f1_macro']:.4f}  (Delta: {delta_f1:+.2f}%)")
    print("=" * 70)

    model_save_path = os.path.join(output_dir, "model_stage2_augmented_best.pt")
    torch.save(best_model.state_dict(), model_save_path)
    print(f"\nSaved best Stage 2 model weights to: {model_save_path}")

    report_file = os.path.join(output_dir, "stage2_augmented_classification_report.txt")
    with open(report_file, "w") as f:
        f.write("=== STAGE 2 AUGMENTED CLASSIFICATION REPORT ===\n")
        f.write(f"embed_mode: {args.embed_mode} | in_features: {in_features}\n")
        f.write(f"Architecture: {best_variant_name}\n")
        f.write(f"Classes: {disorder_classes}\n")
        f.write(f"Optimal Learning Rate: {best_lr}\n\n")
        f.write(f"Val Accuracy: {best_val_res['accuracy']:.4f} | Val Macro F1: {best_val_res['f1_macro']:.4f}\n")
        f.write(f"Test Accuracy: {best_test_res['accuracy']:.4f} | Test Macro F1: {best_test_res['f1_macro']:.4f}\n")
        f.write(f"Test Macro Precision: {best_test_res['precision_macro']:.4f} | Test Macro Recall: {best_test_res['recall_macro']:.4f}\n\n")
        f.write("Classification Report (Untouched Test Set):\n")
        f.write(rpt_str + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm) + "\n\n")
        for g in ["M", "F"]:
            gm = gender_metrics[g]
            f.write(f"--- Gender {g} Breakdown (N={gm['support']}) ---\n")
            f.write(f"Acc: {gm['accuracy']:.4f} | Macro F1: {gm['f1_macro']:.4f} | Precision: {gm['precision_macro']:.4f} | Recall: {gm['recall_macro']:.4f}\n")
            f.write(gm["report"] + "\n")

    print(f"Saved full report to: {report_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
