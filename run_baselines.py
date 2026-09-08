import os
import copy
import random
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

# --- Model Definitions ---

class ModelA_WavLMOnly(nn.Module):
    """Baseline A: Frozen WavLM 1024-d -> Linear Classifier"""
    def __init__(self, in_features=1024, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, emb, age=None):
        return self.classifier(emb)

class ModelB_WavLMDirectAge(nn.Module):
    """Baseline B: Frozen WavLM 1024-d + Age 1-d (1025-d) -> Linear Classifier"""
    def __init__(self, in_features=1025, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, emb, age):
        fused = torch.cat([emb, age], dim=1)
        return self.classifier(fused)

class ModelC_AAFF(nn.Module):
    """Architecture C: Frozen WavLM 1024-d + Age 1-d -> Projection (256) -> Classifier"""
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

# --- Training and Evaluation Functions ---

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_emb, batch_age, batch_lbl in dataloader:
            batch_emb = batch_emb.to(device)
            batch_age = batch_age.to(device)
            logits = model(batch_emb, batch_age)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_lbl.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "f1_macro": f1_mac,
        "precision_macro": p_mac,
        "recall_macro": r_mac,
        "y_true": y_true,
        "y_pred": y_pred
    }

def train_and_tune(model_cls, model_kwargs, train_data, val_data, device, lr_list=[1e-3, 5e-4, 3e-4, 1e-4], use_class_weights=False, epochs=40, seed=42):
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    train_labels = [y.item() for _, _, y in train_data]
    class_counts = np.bincount(train_labels, minlength=2)
    total = len(train_labels)
    weights = total / (2.0 * class_counts)
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device) if use_class_weights else None

    best_overall_val_f1 = -1.0
    best_overall_model_state = None
    best_lr = None

    for lr in lr_list:
        set_all_seeds(seed)
        model = model_cls(**model_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        best_val_f1_this_lr = -1.0
        best_state_this_lr = None

        for epoch in range(epochs):
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

            scheduler.step()
            val_metrics = evaluate_model(model, val_loader, device)

            if val_metrics["f1_macro"] > best_val_f1_this_lr:
                best_val_f1_this_lr = val_metrics["f1_macro"]
                best_state_this_lr = copy.deepcopy(model.state_dict())

        if best_val_f1_this_lr > best_overall_val_f1:
            best_overall_val_f1 = best_val_f1_this_lr
            best_overall_model_state = best_state_this_lr
            best_lr = lr

    # Load the best overall model
    best_model = model_cls(**model_kwargs).to(device)
    best_model.load_state_dict(best_overall_model_state)
    val_metrics = evaluate_model(best_model, val_loader, device)
    return best_model, best_lr, val_metrics

def evaluate_test_with_fairness(model, test_data, test_genders, class_names, device):
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    results = evaluate_model(model, test_loader, device)
    y_true = results["y_true"]
    y_pred = results["y_pred"]

    global_report_dict = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    global_report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Stratified Gender Analysis
    df_eval = pd.DataFrame({"true": y_true, "pred": y_pred, "gender": test_genders})
    gender_reports = {}

    for g in ["M", "F"]:
        gdf = df_eval[df_eval["gender"] == g]
        if len(gdf) > 0:
            g_true = gdf["true"].values
            g_pred = gdf["pred"].values
            g_acc = accuracy_score(g_true, g_pred)
            g_p, g_r, g_f1, _ = precision_recall_fscore_support(g_true, g_pred, average="macro", zero_division=0)
            gender_reports[g] = {
                "accuracy": g_acc,
                "f1_macro": g_f1,
                "precision_macro": g_p,
                "recall_macro": g_r,
                "support": len(gdf),
                "report_str": classification_report(g_true, g_pred, target_names=class_names, zero_division=0)
            }

    return {
        "global_metrics": results,
        "global_report_dict": global_report_dict,
        "global_report_str": global_report_str,
        "confusion_matrix": cm,
        "gender_reports": gender_reports
    }

def main():
    data_path = "./cached_embeddings/wavlm_large_split_seed42.pt"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run extract_wavlm_embeddings.py first.")
        return

    print("=== Loading Cached WavLM Representations ===")
    cache = torch.load(data_path, map_location="cpu", weights_only=False)
    class_names = cache["unique_labels"]
    print(f"Loaded cache. Classes: {class_names}")

    train_data = TensorDataset(cache["train"]["embeddings"], cache["train"]["ages"], cache["train"]["labels"])
    val_data   = TensorDataset(cache["val"]["embeddings"], cache["val"]["ages"], cache["val"]["labels"])
    test_data  = TensorDataset(cache["test"]["embeddings"], cache["test"]["ages"], cache["test"]["labels"])
    test_genders = cache["test"]["genders"]

    print(f"Dataset Split Sizes -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    output_dir = "./results_controlled_baselines"
    os.makedirs(output_dir, exist_ok=True)

    experiments = [
        ("Model A (Frozen WavLM Only)", ModelA_WavLMOnly, {"in_features": 1024, "num_classes": 2}),
        ("Model B (Frozen WavLM + Direct Age)", ModelB_WavLMDirectAge, {"in_features": 1025, "num_classes": 2}),
        ("Model C (Frozen WavLM + AAFF)", ModelC_AAFF, {"in_features": 1025, "proj_dim": 256, "dropout": 0.2, "num_classes": 2})
    ]

    summary_rows = []

    print("\n" + "="*70)
    print("STARTING PHASE 2 CONTROLLED EXPERIMENTS")
    print("="*70)

    for name, model_cls, model_kwargs in experiments:
        print(f"\n--- Training & Tuning {name} ---")
        best_model, best_lr, val_metrics = train_and_tune(
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            train_data=train_data,
            val_data=val_data,
            device=device,
            lr_list=[1e-3, 5e-4, 3e-4, 1e-4],
            use_class_weights=False,
            epochs=40,
            seed=42
        )
        print(f"Optimal LR: {best_lr} | Val Acc: {val_metrics['accuracy']:.4f} | Val Macro F1: {val_metrics['f1_macro']:.4f}")

        # Final Test Evaluation
        test_results = evaluate_test_with_fairness(best_model, test_data, test_genders, class_names, device)
        gm = test_results["global_metrics"]
        gr = test_results["gender_reports"]

        print(f"Test Acc: {gm['accuracy']:.4f} | Test Macro F1: {gm['f1_macro']:.4f} | Precision: {gm['precision_macro']:.4f} | Recall: {gm['recall_macro']:.4f}")
        print(f"Male Test Macro F1: {gr['M']['f1_macro']:.4f} | Female Test Macro F1: {gr['F']['f1_macro']:.4f}")

        # Save individual experiment log
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        report_file = os.path.join(output_dir, f"{safe_name}_report.txt")
        with open(report_file, "w") as f:
            f.write(f"=== {name} ===\n")
            f.write(f"Optimal Learning Rate: {best_lr}\n")
            f.write(f"Val Accuracy: {val_metrics['accuracy']:.4f} | Val Macro F1: {val_metrics['f1_macro']:.4f}\n\n")
            f.write("Global Test Classification Report:\n")
            f.write(test_results["global_report_str"] + "\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(test_results["confusion_matrix"]) + "\n\n")
            for g in ["M", "F"]:
                f.write(f"--- Gender {g} Breakdown (N={gr[g]['support']}) ---\n")
                f.write(f"Acc: {gr[g]['accuracy']:.4f} | Macro F1: {gr[g]['f1_macro']:.4f}\n")
                f.write(gr[g]["report_str"] + "\n")

        summary_rows.append({
            "Model": name,
            "Best LR": best_lr,
            "Val Acc": val_metrics["accuracy"],
            "Val Macro F1": val_metrics["f1_macro"],
            "Test Acc": gm["accuracy"],
            "Test Macro F1": gm["f1_macro"],
            "Test Macro Precision": gm["precision_macro"],
            "Test Macro Recall": gm["recall_macro"],
            "Male Macro F1": gr["M"]["f1_macro"],
            "Female Macro F1": gr["F"]["f1_macro"],
            "Gender F1 Disparity (|M-F|)": abs(gr["M"]["f1_macro"] - gr["F"]["f1_macro"])
        })

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "="*80)
    print("PHASE 2 CONTROLLED BASELINE SUMMARY TABLE")
    print("="*80)
    print(df_summary.to_string(index=False))

    summary_csv = os.path.join(output_dir, "phase2_baseline_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"\nSaved summary to: {summary_csv}")

if __name__ == "__main__":
    main()
