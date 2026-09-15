"""
train_stage2_improved.py  —  Stage 2 IMPROVED variant
Three improvements over the baseline (train_stage2_disorder.py):
  1. SMOTE oversampling on training set (imbalanced-learn)
  2. Wider projection head: 1025 -> 512 -> 256 -> num_classes
  3. Regularisation: label_smoothing=0.10, dropout=0.35, cosine LR schedule
Original train_stage2_disorder.py is NOT modified.
"""
import os, copy, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARNING] imbalanced-learn not found -- SMOTE will be skipped.")


# ─── Seed ────────────────────────────────────────────────────────────────────
def set_all_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ─── SMOTE helper ────────────────────────────────────────────────────────────
def apply_smote(embeddings, ages, labels, seed=42):
    X = torch.cat([embeddings, ages], dim=1).numpy()   # (N, 1025)
    y = labels.numpy()
    min_c = int(np.min(np.bincount(y)))
    k = max(1, min(5, min_c - 1))
    print("  Before SMOTE -- class dist:", dict(zip(*np.unique(y, return_counts=True))))
    smote = SMOTE(random_state=seed, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X, y)
    print("  After  SMOTE -- class dist:", dict(zip(*np.unique(y_res, return_counts=True))))
    return (
        torch.tensor(X_res[:, :-1], dtype=torch.float32),
        torch.tensor(X_res[:, -1:], dtype=torch.float32),
        torch.tensor(y_res,         dtype=torch.long),
    )


# ─── Improved Model (wider head) ─────────────────────────────────────────────
class AAFFDisorderClassifierImproved(nn.Module):
    """1025-d -> Linear(512)->ReLU->Drop -> Linear(256)->ReLU->Drop -> Linear(nc)"""
    def __init__(self, in_features=1025, num_classes=4, dropout=0.35):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),         nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, emb, age):
        return self.classifier(self.projection(torch.cat([emb, age], dim=1)))


# ─── Evaluation ──────────────────────────────────────────────────────────────
def evaluate_model(model, dl, device):
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for be, ba, bl in dl:
            logits = model(be.to(device), ba.to(device))
            probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(bl.numpy())
    y_true, y_pred, y_prob = np.array(labels), np.array(preds), np.array(probs)
    acc = accuracy_score(y_true, y_pred)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",    zero_division=0)
    _,  _,  fmic,_ = precision_recall_fscore_support(y_true, y_pred, average="micro",   zero_division=0)
    pw, rw, fw,  _ = precision_recall_fscore_support(y_true, y_pred, average="weighted",zero_division=0)
    return {
        "accuracy": acc, "f1_macro": fm, "precision_macro": pm,
        "recall_macro": rm, "f1_micro": fmic, "f1_weighted": fw,
        "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
    }


# ─── Training loop with LR sweep ─────────────────────────────────────────────
def train_improved(train_data, val_data, device, num_classes=4,
                   lr_list=None, epochs=50, seed=42,
                   label_smoothing=0.10, dropout=0.35):
    if lr_list is None:
        lr_list = [1e-3, 5e-4, 3e-4, 1e-4]

    te, ta, tl = train_data.tensors
    if SMOTE_AVAILABLE:
        print("\n  Applying SMOTE ...")
        te, ta, tl = apply_smote(te, ta, tl, seed=seed)
    else:
        print("  [SMOTE skipped]")

    td = TensorDataset(te, ta, tl)
    tl_np = tl.numpy()
    cc = np.bincount(tl_np, minlength=num_classes)
    w  = len(tl_np) / (float(num_classes) * np.maximum(cc, 1))
    wt = torch.tensor(w, dtype=torch.float32).to(device)
    print("  Class counts after SMOTE:", cc)
    print("  Class weights:           ", np.round(w, 3))

    train_dl = DataLoader(td,         batch_size=16, shuffle=True)
    val_dl   = DataLoader(val_data,   batch_size=16, shuffle=False)

    best_f1, best_state, best_lr = -1.0, None, None

    for lr in lr_list:
        set_all_seeds(seed)
        m     = AAFFDisorderClassifierImproved(num_classes=num_classes, dropout=dropout).to(device)
        crit  = nn.CrossEntropyLoss(weight=wt, label_smoothing=label_smoothing)
        opt   = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        bfl, bsl = -1.0, None

        for ep in range(1, epochs + 1):
            m.train()
            for be, ba, bl in train_dl:
                opt.zero_grad()
                loss = crit(m(be.to(device), ba.to(device)), bl.to(device))
                loss.backward(); opt.step()
            sched.step()
            vm = evaluate_model(m, val_dl, device)
            if vm["f1_macro"] > bfl:
                bfl = vm["f1_macro"]; bsl = copy.deepcopy(m.state_dict())

        print(f"  LR {lr:.0e} -> Best Val Macro F1: {bfl:.4f}")
        if bfl > best_f1:
            best_f1, best_state, best_lr = bfl, bsl, lr

    fm = AAFFDisorderClassifierImproved(num_classes=num_classes, dropout=dropout).to(device)
    fm.load_state_dict(best_state)
    return fm, best_lr, evaluate_model(fm, val_dl, device)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    dp = "./cached_embeddings/wavlm_stage2_disorder_symptom.pt"
    if not os.path.exists(dp):
        print(f"Error: {dp} not found. Run extract_stage2_embeddings.py first.")
        return

    print("=" * 70)
    print("STAGE 2 IMPROVED -- SMOTE + WIDER HEAD + REGULARISATION")
    print("=" * 70)

    cache = torch.load(dp, map_location="cpu", weights_only=False)
    dc    = cache["disorder_classes"]
    nc    = len(dc)
    print("Disorder classes:", dc)

    tr = TensorDataset(cache["train"]["embeddings"], cache["train"]["ages"], cache["train"]["labels"])
    va = TensorDataset(cache["val"]["embeddings"],   cache["val"]["ages"],   cache["val"]["labels"])
    te = TensorDataset(cache["test"]["embeddings"],  cache["test"]["ages"],  cache["test"]["labels"])
    print(f"Splits -> Train:{len(tr)}, Val:{len(va)}, Test:{len(te)}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev, "\n")

    bm, blr, vm = train_improved(tr, va, dev, nc, [1e-3, 5e-4, 3e-4, 1e-4], 50, 42, 0.10, 0.35)
    print(f"\nBest LR: {blr}")
    print(f"Val Acc: {vm['accuracy']:.4f} | Val Macro F1: {vm['f1_macro']:.4f}")

    test_dl = DataLoader(te, batch_size=16, shuffle=False)
    tm      = evaluate_model(bm, test_dl, dev)
    yt, yp  = tm["y_true"], tm["y_pred"]
    rpt     = classification_report(yt, yp, target_names=dc, zero_division=0)
    cm      = confusion_matrix(yt, yp)

    print("\n" + "=" * 70)
    print("STAGE 2 IMPROVED -- TEST RESULTS")
    print("=" * 70)
    print(f"Test Accuracy:          {tm['accuracy']:.4f} ({tm['accuracy']*100:.2f}%)")
    print(f"Test Macro Precision:   {tm['precision_macro']:.4f}")
    print(f"Test Macro Recall:      {tm['recall_macro']:.4f}")
    print(f"Test Macro F1-Score:    {tm['f1_macro']:.4f}")
    print(f"Test Weighted F1-Score: {tm['f1_weighted']:.4f}")
    print("\nPer-Class Report:"); print(rpt)
    print("Confusion Matrix:");   print(cm)
    print("Classes:", dc)

    od = "./results_controlled_baselines"; os.makedirs(od, exist_ok=True)
    sp = os.path.join(od, "model_stage2_disorder_improved.pt")
    torch.save(bm.state_dict(), sp)
    print("\nSaved improved model to:", sp)

    rf = os.path.join(od, "stage2_improved_classification_report.txt")
    with open(rf, "w") as f:
        f.write("=== STAGE 2 IMPROVED CLASSIFICATION REPORT ===\n")
        f.write("Improvements: SMOTE + WiderHead(1025->512->256->4) + LabelSmoothing(0.1) + Dropout(0.35)\n")
        f.write(f"Classes: {dc}\n"); f.write(f"Best LR: {blr}\n\n")
        f.write(f"Val  Acc:{vm['accuracy']:.4f} | Val  F1:{vm['f1_macro']:.4f}\n")
        f.write(f"Test Acc:{tm['accuracy']:.4f} | Test F1:{tm['f1_macro']:.4f}\n\n")
        f.write("Classification Report:\n"); f.write(rpt + "\n")
        f.write("Confusion Matrix:\n");      f.write(np.array2string(cm) + "\n")
    print("Saved report to:", rf)
    print("=" * 70)


if __name__ == "__main__":
    main()

