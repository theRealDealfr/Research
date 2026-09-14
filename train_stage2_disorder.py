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

class AAFFDisorderClassifier(nn.Module):
    """
    Stage 2 Age-Aware Feature Fusion (AAFF) Architecture for Multiclass Disorder Classification:
    1024-d speech embedding + 1-d normalized age (1025-d)
    -> Linear(1025, 256) -> ReLU() -> Dropout(0.2) -> Linear(256, num_classes)
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

def evaluate_disorder_model(model, dataloader, device):
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
        "y_pred": y_pred
    }

def train_and_tune_disorder_model(train_data, val_data, device, num_classes=4, lr_list=[1e-3, 5e-4, 3e-4, 1e-4], epochs=40, seed=42):
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    train_labels = [y.item() for _, _, y in train_data]
    class_counts = np.bincount(train_labels, minlength=num_classes)
    total = len(train_labels)
    weights = total / (float(num_classes) * np.maximum(class_counts, 1))
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Class counts in training: {class_counts}")
    print(f"Calculated class weights: {np.round(weights, 3)}")

    best_overall_val_f1 = -1.0
    best_overall_model_state = None
    best_lr = None

    for lr in lr_list:
        set_all_seeds(seed)
        model = AAFFDisorderClassifier(in_features=1025, proj_dim=256, dropout=0.2, num_classes=num_classes).to(device)
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

            val_metrics = evaluate_disorder_model(model, val_loader, device)
            val_f1 = val_metrics["f1_macro"]

            if val_f1 > best_val_f1_this_lr:
                best_val_f1_this_lr = val_f1
                best_state_this_lr = copy.deepcopy(model.state_dict())

        print(f"  LR {lr:.0e} -> Best Val Macro F1: {best_val_f1_this_lr:.4f}")

        if best_val_f1_this_lr > best_overall_val_f1:
            best_overall_val_f1 = best_val_f1_this_lr
            best_overall_model_state = best_state_this_lr
            best_lr = lr

    final_model = AAFFDisorderClassifier(in_features=1025, proj_dim=256, dropout=0.2, num_classes=num_classes).to(device)
    final_model.load_state_dict(best_overall_model_state)

    val_res = evaluate_disorder_model(final_model, val_loader, device)
    return final_model, best_lr, val_res

def main():
    data_path = "./cached_embeddings/wavlm_stage2_disorder_symptom.pt"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run extract_stage2_embeddings.py first.")
        return

    print("="*70)
    print("STAGE 2 DISORDER CLASSIFICATION TRAINING (AAFF MULTICLASS)")
    print("="*70)

    cache = torch.load(data_path, map_location="cpu", weights_only=False)
    disorder_classes = cache["disorder_classes"]
    num_classes = len(disorder_classes)
    print(f"Loaded Stage 2 cache. Disorder classes: {disorder_classes}")

    train_data = TensorDataset(cache["train"]["embeddings"], cache["train"]["ages"], cache["train"]["labels"])
    val_data   = TensorDataset(cache["val"]["embeddings"], cache["val"]["ages"], cache["val"]["labels"])
    test_data  = TensorDataset(cache["test"]["embeddings"], cache["test"]["ages"], cache["test"]["labels"])

    print(f"Disorder Dataset Splits -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    output_dir = "./results_controlled_baselines"
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "model_stage2_disorder_best.pt")

    best_model, best_lr, val_metrics = train_and_tune_disorder_model(
        train_data=train_data,
        val_data=val_data,
        device=device,
        num_classes=num_classes,
        lr_list=[1e-3, 5e-4, 3e-4, 1e-4],
        epochs=40,
        seed=42
    )

    print(f"\n--- Optimal Model Parameters ---")
    print(f"Best Learning Rate: {best_lr}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f} | Validation Macro F1: {val_metrics['f1_macro']:.4f}")

    # Evaluate on Test Set
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    test_metrics = evaluate_disorder_model(best_model, test_loader, device)

    y_true = test_metrics["y_true"]
    y_pred = test_metrics["y_pred"]
    report_dict = classification_report(y_true, y_pred, target_names=disorder_classes, output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, target_names=disorder_classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "="*70)
    print("STAGE 2 TEST SET EVALUATION REPORT")
    print("="*70)
    print(f"Test Accuracy:          {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"Test Macro Precision:   {test_metrics['precision_macro']:.4f}")
    print(f"Test Macro Recall:      {test_metrics['recall_macro']:.4f}")
    print(f"Test Macro F1-Score:    {test_metrics['f1_macro']:.4f}")
    print(f"Test Weighted F1-Score: {test_metrics['f1_weighted']:.4f}")
    print("\nPer-Class Detailed Report:")
    print(report_str)
    print("Confusion Matrix:")
    print(cm)
    print(f"\nClasses in Matrix: {disorder_classes}")

    # Save trained checkpoint
    torch.save(best_model.state_dict(), best_model_path)
    print(f"\nSaved best Stage 2 model weights to: {best_model_path}")

    # Save detailed evaluation report
    report_file = os.path.join(output_dir, "stage2_disorder_classification_report.txt")
    with open(report_file, "w") as f:
        f.write("=== STAGE 2 MULTICLASS DISORDER CLASSIFICATION REPORT ===\n")
        f.write(f"Target Column: disorder_symptom (Disorder Subtypes)\n")
        f.write(f"Classes: {disorder_classes}\n")
        f.write(f"Optimal Learning Rate: {best_lr}\n\n")
        f.write(f"Val Accuracy: {val_metrics['accuracy']:.4f} | Val Macro F1: {val_metrics['f1_macro']:.4f}\n")
        f.write(f"Test Accuracy: {test_metrics['accuracy']:.4f} | Test Macro F1: {test_metrics['f1_macro']:.4f}\n")
        f.write(f"Test Macro Precision: {test_metrics['precision_macro']:.4f} | Test Macro Recall: {test_metrics['recall_macro']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_str + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm) + "\n")

    print(f"Saved report to: {report_file}")
    print("="*70)

if __name__ == "__main__":
    main()
