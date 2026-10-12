"""
compare_stage2_models.py
Load and compare the BASELINE vs IMPROVED Stage 2 disorder classifier
on the same test set.  Prints:
  - Side-by-side summary metrics
  - Per-class report for each model
  - Per-sample confidence scores (top prediction + probability)
"""
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)


# ─── Baseline architecture (must match train_stage2_disorder.py) ──────────────
class AAFFDisorderClassifier(nn.Module):
    def __init__(self, in_features=1025, proj_dim=256, dropout=0.2, num_classes=4):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, emb, age):
        return self.classifier(self.projection(torch.cat([emb, age], dim=1)))


# ─── Improved architecture (must match train_stage2_improved.py) ──────────────
class AAFFDisorderClassifierImproved(nn.Module):
    def __init__(self, in_features=1025, num_classes=4, dropout=0.35):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),         nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, emb, age):
        return self.classifier(self.projection(torch.cat([emb, age], dim=1)))


# ─── Evaluation with confidence ───────────────────────────────────────────────
def evaluate_with_confidence(model, dl, device):
    model.eval()
    preds, labels, confidences = [], [], []
    with torch.no_grad():
        for be, ba, bl in dl:
            logits = model(be.to(device), ba.to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            top_p  = probs.max(axis=1)
            top_c  = probs.argmax(axis=1)
            preds.extend(top_c.tolist())
            labels.extend(bl.numpy().tolist())
            confidences.extend(top_p.tolist())
    y_true = np.array(labels)
    y_pred = np.array(preds)
    confs  = np.array(confidences)

    acc = accuracy_score(y_true, y_pred)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",    zero_division=0)
    _,  _,  fmic,_ = precision_recall_fscore_support(y_true, y_pred, average="micro",   zero_division=0)
    pw, rw, fw,  _ = precision_recall_fscore_support(y_true, y_pred, average="weighted",zero_division=0)
    return {
        "accuracy": acc, "f1_macro": fm, "precision_macro": pm,
        "recall_macro": rm, "f1_micro": fmic, "f1_weighted": fw,
        "y_true": y_true, "y_pred": y_pred, "confidences": confs,
    }


def main():
    data_path     = "./cached_embeddings/wavlm_stage2_disorder_symptom.pt"
    baseline_ckpt = "./results_controlled_baselines/model_stage2_disorder_best.pt"
    improved_ckpt = "./results_controlled_baselines/model_stage2_disorder_improved.pt"

    # ── Check files ──
    for p in [data_path, baseline_ckpt, improved_ckpt]:
        if not os.path.exists(p):
            print(f"[ERROR] Missing: {p}")
            return

    cache = torch.load(data_path, map_location="cpu", weights_only=False)
    dc    = cache["disorder_classes"]
    nc    = len(dc)
    te    = TensorDataset(cache["test"]["embeddings"], cache["test"]["ages"], cache["test"]["labels"])
    dl    = DataLoader(te, batch_size=16, shuffle=False)
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("STAGE 2 MODEL COMPARISON: BASELINE vs IMPROVED")
    print("=" * 70)
    print(f"Disorder classes : {dc}")
    print(f"Test samples     : {len(te)}")
    print(f"Device           : {dev}\n")

    # ── Load models ──
    base_state = torch.load(baseline_ckpt, map_location=dev, weights_only=True)
    base_proj_dim = base_state["projection.0.weight"].shape[0] if "projection.0.weight" in base_state else 256
    baseline = AAFFDisorderClassifier(num_classes=nc, proj_dim=base_proj_dim).to(dev)
    baseline.load_state_dict(base_state)

    improved = AAFFDisorderClassifierImproved(num_classes=nc).to(dev)
    improved.load_state_dict(torch.load(improved_ckpt, map_location=dev, weights_only=True))

    # ── Evaluate ──
    bm = evaluate_with_confidence(baseline, dl, dev)
    im = evaluate_with_confidence(improved, dl, dev)

    # ─────────────────────────────────────────────────────────────
    # SIDE-BY-SIDE SUMMARY
    # ─────────────────────────────────────────────────────────────
    def delta(new, old):
        d = new - old
        sign = "+" if d >= 0 else ""
        return f"{sign}{d*100:.2f}%"

    print("=" * 70)
    print(f"{'METRIC':<28} {'BASELINE':>12} {'IMPROVED':>12} {'DELTA':>10}")
    print("-" * 70)
    rows = [
        ("Accuracy",          bm["accuracy"],         im["accuracy"]),
        ("Macro Precision",   bm["precision_macro"],  im["precision_macro"]),
        ("Macro Recall",      bm["recall_macro"],     im["recall_macro"]),
        ("Macro F1",          bm["f1_macro"],         im["f1_macro"]),
        ("Weighted F1",       bm["f1_weighted"],      im["f1_weighted"]),
        ("Avg Confidence",    bm["confidences"].mean(),im["confidences"].mean()),
    ]
    for name, bv, iv in rows:
        print(f"  {name:<26} {bv:>11.4f} {iv:>11.4f} {delta(iv,bv):>10}")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────
    # PER-CLASS REPORTS
    # ─────────────────────────────────────────────────────────────
    print("\n-- BASELINE Per-Class Report --")
    print(classification_report(bm["y_true"], bm["y_pred"], target_names=dc, zero_division=0))
    print("Baseline Confusion Matrix:")
    print(confusion_matrix(bm["y_true"], bm["y_pred"]))

    print("\n-- IMPROVED Per-Class Report --")
    print(classification_report(im["y_true"], im["y_pred"], target_names=dc, zero_division=0))
    print("Improved Confusion Matrix:")
    print(confusion_matrix(im["y_true"], im["y_pred"]))

    # ─────────────────────────────────────────────────────────────
    # PER-SAMPLE CONFIDENCE SCORES (first 30 shown)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-SAMPLE CONFIDENCE (first 30 test samples)")
    print("=" * 70)
    print(f"{'#':>4} {'TrueLabel':<14} {'Base-Pred':<14} {'Base-Conf':>10} {'Imp-Pred':<14} {'Imp-Conf':>9}")
    print("-" * 70)
    n_show = min(30, len(bm["y_true"]))
    for i in range(n_show):
        tl = dc[bm["y_true"][i]]
        bp = dc[bm["y_pred"][i]]
        ip = dc[im["y_pred"][i]]
        bc = bm["confidences"][i]
        ic = im["confidences"][i]
        marker = " <--" if bm["y_pred"][i] != im["y_pred"][i] else ""
        print(f"  {i+1:>2} {tl:<14} {bp:<14} {bc:>9.3f}    {ip:<14} {ic:>9.3f}{marker}")

    print("\n  (rows marked <-- are cases where the two models disagree)")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────
    # VERDICT
    # ─────────────────────────────────────────────────────────────
    print("\nVERDICT:")
    if im["f1_macro"] > bm["f1_macro"]:
        gain = (im["f1_macro"] - bm["f1_macro"]) * 100
        print(f"  [IMPROVED] Macro F1 increased by {gain:.2f}% points.")
    elif im["f1_macro"] == bm["f1_macro"]:
        print("  [TIED] Both models achieve the same Macro F1.")
    else:
        loss = (bm["f1_macro"] - im["f1_macro"]) * 100
        print(f"  [REGRESSED] Macro F1 dropped by {loss:.2f}% points. Consider more data / tuning.")
    print("=" * 70)


if __name__ == "__main__":
    main()

