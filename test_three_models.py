import os, sys, copy, random, torch, numpy as np, pandas as pd
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Model Architectures ---
class ModelA(nn.Module):
    """Model A: WavLM only (no age)"""
    def __init__(self, in_features=1024, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)
    def forward(self, emb, age=None):
        return self.classifier(emb)

class ModelB(nn.Module):
    """Model B: WavLM + Direct Age concatenation"""
    def __init__(self, in_features=1025, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)
    def forward(self, emb, age):
        return self.classifier(torch.cat([emb, age], dim=1))

class ModelC(nn.Module):
    """Model C: WavLM + AAFF (Age-Aware Feature Fusion)"""
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
        return self.classifier(self.projection(fused))

# Evaluation Helper
def evaluate_model(model, dl, device):
    model.eval()
    preds, targets, confs = [], [], []
    with torch.no_grad():
        for be, ba, bl in dl:
            be, ba = be.to(device), ba.to(device)
            logits = model(be, ba)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            top_p = np.max(probs, axis=1)
            top_c = np.argmax(probs, axis=1)
            preds.extend(top_c.tolist())
            targets.extend(bl.numpy().tolist())
            confs.extend(top_p.tolist())

    y_true = np.array(targets)
    y_pred = np.array(preds)
    acc = float(accuracy_score(y_true, y_pred))
    pm = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rm = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    fm = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    fw = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    return {
        "accuracy": acc, "precision_macro": pm, "recall_macro": rm,
        "f1_macro": fm, "f1_weighted": fw,
        "y_true": y_true, "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

# Training Helper with validation tuning
def train_and_tune(model_cls, model_kwargs, train_data, val_data, device, num_classes, lr_list=[1e-3, 5e-4, 3e-4, 1e-4], epochs=40, seed=42):
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    train_labels = [y.item() for _, _, y in train_data]
    class_counts = np.bincount(train_labels, minlength=num_classes)
    total = len(train_labels)
    weights = total / (float(num_classes) * np.maximum(class_counts, 1))
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    best_val_f1 = -1.0
    best_state = None
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
            for be, ba, bl in train_loader:
                be, ba, bl = be.to(device), ba.to(device), bl.to(device)
                optimizer.zero_grad()
                logits = model(be, ba)
                loss = criterion(logits, bl)
                loss.backward()
                optimizer.step()

            val_res = evaluate_model(model, val_loader, device)
            if val_res["f1_macro"] > best_val_f1_this_lr:
                best_val_f1_this_lr = val_res["f1_macro"]
                best_state_this_lr = copy.deepcopy(model.state_dict())

        if best_val_f1_this_lr > best_val_f1:
            best_val_f1 = best_val_f1_this_lr
            best_state = best_state_this_lr
            best_lr = lr

    final_model = model_cls(**model_kwargs).to(device)
    final_model.load_state_dict(best_state)
    val_res = evaluate_model(final_model, val_loader, device)
    return final_model, best_lr, val_res

# Load Caches
from train_hierarchical_stages import get_stage2_data, get_stage3_data
orig_cache = torch.load("./cached_embeddings/wavlm_stage2_disorder_symptom.pt", map_location="cpu", weights_only=False)
aug_cache  = torch.load("./cached_embeddings/wavlm_stage2_disorder_symptom_augmented.pt", map_location="cpu", weights_only=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Stage 2 Test:")
s2_data = get_stage2_data(orig_cache, aug_cache)
s2_tr = TensorDataset(*s2_data["unaugmented_train"])
s2_va = TensorDataset(*s2_data["val"])
s2_te = TensorDataset(*s2_data["test"])
s2_dl_te = DataLoader(s2_te, batch_size=16, shuffle=False)

for m_name, m_cls, m_kw in [
    ("Model A (WavLM Only)", ModelA, {"in_features": 1024, "num_classes": 2}),
    ("Model B (WavLM + Direct Age)", ModelB, {"in_features": 1025, "num_classes": 2}),
    ("Model C (WavLM + AAFF)", ModelC, {"in_features": 1025, "proj_dim": 256, "dropout": 0.2, "num_classes": 2})
]:
    m, lr, v_res = train_and_tune(m_cls, m_kw, s2_tr, s2_va, device, num_classes=2)
    t_res = evaluate_model(m, s2_dl_te, device)
    print(f"  {m_name}: Val F1={v_res['f1_macro']:.4f} | Test Acc={t_res['accuracy']:.4f} | Test F1={t_res['f1_macro']:.4f}")

print("\nStage 3 Test:")
s3_data = get_stage3_data(orig_cache, aug_cache)
s3_tr = TensorDataset(*s3_data["unaugmented_train"])
s3_va = TensorDataset(*s3_data["val"])
s3_te = TensorDataset(*s3_data["test"])
s3_dl_te = DataLoader(s3_te, batch_size=16, shuffle=False)

for m_name, m_cls, m_kw in [
    ("Model A (WavLM Only)", ModelA, {"in_features": 1024, "num_classes": 4}),
    ("Model B (WavLM + Direct Age)", ModelB, {"in_features": 1025, "num_classes": 4}),
    ("Model C (WavLM + AAFF)", ModelC, {"in_features": 1025, "proj_dim": 256, "dropout": 0.2, "num_classes": 4})
]:
    m, lr, v_res = train_and_tune(m_cls, m_kw, s3_tr, s3_va, device, num_classes=4)
    t_res = evaluate_model(m, s3_dl_te, device)
    print(f"  {m_name}: Val F1={v_res['f1_macro']:.4f} | Test Acc={t_res['accuracy']:.4f} | Test F1={t_res['f1_macro']:.4f}")
