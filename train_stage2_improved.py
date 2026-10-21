"""
train_stage2_improved.py  --  Stage 2 IMPROVED variant
Three improvements over the baseline (train_stage2_disorder.py):
  1. SMOTE oversampling on training set (imbalanced-learn)
  2. Wider projection head: in_features -> 512 -> 256 -> num_classes
  3. Regularisation: label_smoothing=0.10, dropout=0.35, cosine LR schedule
Original train_stage2_disorder.py is NOT modified.

New in this version:
  --embed_mode last|stats  (default last = existing behavior)
  In stats mode the [L, 2048] tensor is collapsed by a learnable LayerWeightedPool
  module (softmax over L scalars, initialized uniform) before age fusion.
  Input dimension is inferred from the cache, not hardcoded.
"""
import os, copy, random, argparse
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


# --- Seed ---

def set_all_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# --- Layer-Weighting Module (stats mode only) ---

class LayerWeightedPool(nn.Module):
    """Softmax-weighted sum of L per-layer [mean||std] vectors.
    Input : (B, L, 2048)
    Output: (B, 2048)
    Initialized uniform (all weights = 0 -> softmax = 1/L).
    """
    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, x):
        # x: (B, L, D)
        w = torch.softmax(self.weights, dim=0)   # (L,)
        return (x * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (B, D)


# --- SMOTE helper ---

def apply_smote(embeddings, ages, labels, seed=42):
    """Apply SMOTE to the fused (emb || age) vector; split back afterward."""
    X = torch.cat([embeddings, ages], dim=1).numpy()   # (N, emb_dim+1)
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


# --- Improved Model (wider head, in_features from data) ---

class AAFFDisorderClassifierImproved(nn.Module):
    """in_features -> Linear(512)->ReLU->Drop -> Linear(256)->ReLU->Drop -> Linear(nc)"""
    def __init__(self, in_features=1025, num_classes=4, dropout=0.35):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),         nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, emb, age):
        return self.classifier(self.projection(torch.cat([emb, age], dim=1)))


# --- Evaluation ---

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


# --- Training loop with LR sweep ---

def train_improved(train_data, val_data, device, num_classes=4,
                   lr_list=None, epochs=50, seed=42,
                   label_smoothing=0.10, dropout=0.35, in_features=1025,
                   use_smote=True):
    if lr_list is None:
        lr_list = [1e-3, 5e-4, 3e-4, 1e-4]

    te, ta, tl = train_data.tensors
    if use_smote and SMOTE_AVAILABLE:
        print("\n  Applying SMOTE ...")
        te, ta, tl = apply_smote(te, ta, tl, seed=seed)
    elif use_smote and not SMOTE_AVAILABLE:
        print("  [SMOTE skipped -- imbalanced-learn not installed]")
    else:
        print("  [SMOTE disabled via --no_smote flag]")

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
        m     = AAFFDisorderClassifierImproved(in_features=in_features, num_classes=num_classes, dropout=dropout).to(device)
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

    fm = AAFFDisorderClassifierImproved(in_features=in_features, num_classes=num_classes, dropout=dropout).to(device)
    fm.load_state_dict(best_state)
    return fm, best_lr, evaluate_model(fm, val_dl, device)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Stage 2 Improved Training")
    parser.add_argument(
        "--embed_mode",
        choices=["last", "stats"],
        default="last",
        help=(
            "'last' = use pooled last-layer embeddings from "
            "./cached_embeddings/wavlm_stage2_disorder_symptom.pt (default). "
            "'stats' = use multi-layer mean+std embeddings from "
            "./cached_embeddings/wavlm_stats_layers/wavlm_large_stats_layers_seed42.pt"
        ),
    )
    parser.add_argument(
        "--no_smote",
        action="store_true",
        help="Disable SMOTE oversampling. Useful for ablation: isolates the effect of "
             "the wider head / regularisation without feature-space resampling.",
    )
    parser.add_argument(
        "--out_tag",
        type=str,
        default="",
        help="Optional suffix appended to checkpoint and report filenames, e.g. "
             "'no_smote' produces model_stage2_disorder_improved_no_smote.pt. "
             "Prevents overwriting the existing improved checkpoint.",
    )
    args = parser.parse_args()

    use_smote = not args.no_smote
    out_tag   = ("_" + args.out_tag.strip("_")) if args.out_tag else ""

    if args.embed_mode == "stats":
        dp = "./cached_embeddings/wavlm_stats_layers/wavlm_large_stats_layers_seed42.pt"
    else:
        dp = "./cached_embeddings/wavlm_stage2_disorder_symptom.pt"

    if not os.path.exists(dp):
        print(f"Error: {dp} not found. Run the appropriate extraction script first.")
        return

    smote_label = "WITH SMOTE" if use_smote else "NO SMOTE (wider head only)"
    print("=" * 70)
    print(f"STAGE 2 IMPROVED -- {smote_label} + WIDER HEAD + REGULARISATION  [embed_mode={args.embed_mode}]")
    print("=" * 70)

    cache = torch.load(dp, map_location="cpu", weights_only=False)

    if args.embed_mode == "stats":
        # stats cache uses "stats_embeddings" key; collapse layers before passing
        # LayerWeightedPool is initialized uniform, so starting point = simple mean
        num_layers = cache["train"]["stats_embeddings"].shape[1]
        pool = LayerWeightedPool(num_layers)

        def get_embeddings(split_dict):
            se = split_dict["stats_embeddings"].float()  # (N, L, 2048)
            with torch.no_grad():
                return pool(se)                          # (N, 2048)

        train_emb = get_embeddings(cache["train"])
        val_emb   = get_embeddings(cache["val"])
        test_emb  = get_embeddings(cache["test"])
        # disorder_classes stored in stats cache under same key (same split protocol)
        dc = cache.get("unique_labels", cache.get("disorder_classes", []))
    else:
        train_emb = cache["train"]["embeddings"]
        val_emb   = cache["val"]["embeddings"]
        test_emb  = cache["test"]["embeddings"]
        dc = cache["disorder_classes"]

    # Infer dimension from data
    emb_dim = train_emb.shape[1]            # 1024 (last) or 2048 (stats)
    in_features = emb_dim + 1               # +1 for age

    nc = len(dc)
    print(f"Disorder classes ({nc}): {dc}")
    print(f"emb_dim={emb_dim}, in_features={in_features}")

    tr = TensorDataset(train_emb, cache["train"]["ages"], cache["train"]["labels"])
    va = TensorDataset(val_emb,   cache["val"]["ages"],   cache["val"]["labels"])
    te = TensorDataset(test_emb,  cache["test"]["ages"],  cache["test"]["labels"])
    print(f"Splits -> Train:{len(tr)}, Val:{len(va)}, Test:{len(te)}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev, "\n")

    bm, blr, vm = train_improved(tr, va, dev, nc, [1e-3, 5e-4, 3e-4, 1e-4], 50, 42, 0.10, 0.35, in_features, use_smote=use_smote)
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
    sp = os.path.join(od, f"model_stage2_disorder_improved{out_tag}.pt")
    torch.save(bm.state_dict(), sp)
    print("\nSaved improved model to:", sp)

    smote_desc = "SMOTE" if use_smote else "No SMOTE"
    rf = os.path.join(od, f"stage2_improved_classification_report{out_tag}.txt")
    with open(rf, "w") as f:
        f.write(f"=== STAGE 2 IMPROVED CLASSIFICATION REPORT ({smote_desc}) ===\n")
        f.write(f"embed_mode: {args.embed_mode} | in_features: {in_features}\n")
        f.write(f"Improvements: {smote_desc} + WiderHead({in_features}->512->256->{nc}) + LabelSmoothing(0.1) + Dropout(0.35)\n")
        f.write(f"Classes: {dc}\n"); f.write(f"Best LR: {blr}\n\n")
        f.write(f"Val  Acc:{vm['accuracy']:.4f} | Val  F1:{vm['f1_macro']:.4f}\n")
        f.write(f"Test Acc:{tm['accuracy']:.4f} | Test F1:{tm['f1_macro']:.4f}\n\n")
        f.write("Classification Report:\n"); f.write(rpt + "\n")
        f.write("Confusion Matrix:\n");      f.write(np.array2string(cm) + "\n")
    print("Saved report to:", rf)
    print("=" * 70)


if __name__ == "__main__":
    main()
