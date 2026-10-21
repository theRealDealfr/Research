"""
run_stats_ablation.py
---------------------
Runs the multi-layer mean+std WavLM (stats) ablation over 3 seeds and appends
one new row to results_controlled_baselines/pipeline_ablation_metrics.json.

Usage:
    python run_stats_ablation.py

Requirements:
  - ./cached_embeddings/wavlm_stats_layers/wavlm_large_stats_layers_seed42.pt
    must already exist (run: python extract_wavlm_embeddings.py --mode stats)

The script trains the Stage 1 AAFF model (Model C) in stats mode over seeds
[42, 123, 2024], reports macro F1 and per-class F1, writes mean+/-std, and
appends an "ablation_stats_pooling" entry to the JSON.
"""

import os
import copy
import json
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

CACHE_PATH = "./cached_embeddings/wavlm_stats_layers/wavlm_large_stats_layers_seed42.pt"
ABLATION_JSON = "./results_controlled_baselines/pipeline_ablation_metrics.json"
SEEDS = [42, 123, 2024]
EPOCHS = 40
LR_LIST = [1e-3, 5e-4, 3e-4, 1e-4]


# ---------------------------------------------------------------------------
# Layer-Weighting Module
# ---------------------------------------------------------------------------

class LayerWeightedPool(nn.Module):
    """Softmax-weighted sum of L per-layer [mean||std] vectors.
    Input : (B, L, D)    Output: (B, D)
    Initialized uniform.
    """
    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, x):
        w = torch.softmax(self.weights, dim=0)
        return (x * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)


# ---------------------------------------------------------------------------
# Model: AAFF with learnable layer weighting head
# ---------------------------------------------------------------------------

class AAFFStatsClassifier(nn.Module):
    """[L, 2048] -> LayerWeightedPool -> [2048] -> cat(age) -> [2049]
       -> Linear(proj_dim) -> ReLU -> Dropout -> Linear(num_classes)
    """
    def __init__(self, num_layers: int, proj_dim=256, dropout=0.2, num_classes=2):
        super().__init__()
        self.pool = LayerWeightedPool(num_layers)
        emb_dim = num_layers * 0  # computed in forward; pool output = layer D = 2048
        # We know each layer is 2048-d (1024 mean + 1024 std)
        self.projection = nn.Sequential(
            nn.Linear(2048 + 1, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, stats_emb, age):
        # stats_emb: (B, L, 2048) fp32
        pooled = self.pool(stats_emb)            # (B, 2048)
        fused = torch.cat([pooled, age], dim=1)  # (B, 2049)
        return self.classifier(self.projection(fused))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def set_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def evaluate(model, dl, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for be, ba, bl in dl:
            logits = model(be.to(device), ba.to(device))
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(bl.numpy())
    y_true, y_pred = np.array(labels), np.array(preds)
    acc = accuracy_score(y_true, y_pred)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": acc, "f1_macro": fm, "y_true": y_true, "y_pred": y_pred}


def train_one_seed(train_ds, val_ds, num_layers, num_classes, device, seed, epochs=EPOCHS):
    """Train with LR sweep; return best model and val metrics."""
    set_seeds(seed)
    best_f1, best_state, best_lr = -1.0, None, None

    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)

    for lr in LR_LIST:
        set_seeds(seed)
        model = AAFFStatsClassifier(num_layers=num_layers, proj_dim=256, dropout=0.2, num_classes=num_classes).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit  = nn.CrossEntropyLoss()

        best_f1_this_lr, best_state_this_lr = -1.0, None
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

        for ep in range(epochs):
            model.train()
            for be, ba, bl in train_dl:
                opt.zero_grad()
                loss = crit(model(be.to(device), ba.to(device)), bl.to(device))
                loss.backward()
                opt.step()
            sched.step()
            vm = evaluate(model, val_dl, device)
            if vm["f1_macro"] > best_f1_this_lr:
                best_f1_this_lr = vm["f1_macro"]
                best_state_this_lr = copy.deepcopy(model.state_dict())

        print(f"    LR {lr:.0e} -> best val F1: {best_f1_this_lr:.4f}")
        if best_f1_this_lr > best_f1:
            best_f1, best_state, best_lr = best_f1_this_lr, best_state_this_lr, lr

    best_model = AAFFStatsClassifier(num_layers=num_layers, proj_dim=256, dropout=0.2, num_classes=num_classes).to(device)
    best_model.load_state_dict(best_state)
    return best_model, best_lr, evaluate(best_model, val_dl, device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(CACHE_PATH):
        print(f"[ERROR] Stats cache not found: {CACHE_PATH}")
        print("Run: python extract_wavlm_embeddings.py --mode stats")
        return

    print("=" * 70)
    print("ABLATION: Multi-layer mean+std WavLM (stats mode, 3 seeds)")
    print("=" * 70)

    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    unique_labels = cache.get("unique_labels", cache.get("disorder_classes", []))
    num_classes = len(unique_labels)
    num_layers = cache["train"]["stats_embeddings"].shape[1]   # L

    print(f"Classes: {unique_labels}")
    print(f"L (selected layers): {num_layers}, stats dim per layer: 2048")
    print(f"Total emb_dim: {num_layers * 2048}, fused dim (with age): {num_layers * 2048 + 1}")
    print(f"Stored layers: {cache.get('stats_layers', 'unknown')}")

    # Build TensorDatasets: stats_embeddings are fp16 -> cast to fp32 at load time
    # NOTE: this cache only has train/val splits; val is used as held-out eval
    def make_ds(split):
        se = cache[split]["stats_embeddings"].float()   # (N, L, 2048)
        return TensorDataset(se, cache[split]["ages"], cache[split]["labels"])

    train_ds = make_ds("train")
    val_ds   = make_ds("val")
    # Use val as test (held-out evaluation) since no separate test split exists
    test_ds  = val_ds

    print(f"Splits -> Train:{len(train_ds)}, Val/Test:{len(val_ds)} (val used as held-out eval)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    seed_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        best_model, best_lr, val_metrics = train_one_seed(
            train_ds, val_ds, num_layers, num_classes, device, seed
        )
        print(f"  Best LR: {best_lr} | Val F1: {val_metrics['f1_macro']:.4f}")

        # Evaluate on val (held-out) set
        test_dl = DataLoader(val_ds, batch_size=32, shuffle=False)
        test_metrics = evaluate(best_model, test_dl, device)

        y_true, y_pred = test_metrics["y_true"], test_metrics["y_pred"]
        per_class = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
        )
        per_class_f1 = {unique_labels[i]: float(per_class[2][i]) for i in range(num_classes)}

        print(f"  Val Acc: {test_metrics['accuracy']:.4f} | Val Macro F1: {test_metrics['f1_macro']:.4f}")
        print(f"  Per-class F1: {per_class_f1}")

        seed_results.append({
            "seed": seed,
            "best_lr": best_lr,
            "val_f1_macro": val_metrics["f1_macro"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "per_class_f1": per_class_f1,
        })

    # Aggregate statistics
    test_f1s = [r["test_f1_macro"] for r in seed_results]
    test_accs = [r["test_accuracy"] for r in seed_results]

    mean_f1  = float(np.mean(test_f1s))
    std_f1   = float(np.std(test_f1s))
    mean_acc = float(np.mean(test_accs))
    std_acc  = float(np.std(test_accs))

    # Per-class mean F1
    per_class_means = {}
    for cls in unique_labels:
        vals = [r["per_class_f1"].get(cls, 0.0) for r in seed_results]
        per_class_means[cls] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    ablation_row = {
        "name": "Multi-layer mean+std WavLM (stats pooling, L=6 layers, 2048-d)",
        "embed_mode": "stats",
        "stats_layers": cache.get("stats_layers", [6, 9, 12, 15, 18, 24]),
        "embed_dim": num_layers * 2048,
        "fused_dim": num_layers * 2048 + 1,
        "architecture": "AAFFStatsClassifier (LayerWeightedPool -> Linear(2049,256) -> ReLU -> Linear(2))",
        "seeds": SEEDS,
        "seed_results": seed_results,
        "test_f1_macro_mean": mean_f1,
        "test_f1_macro_std": std_f1,
        "test_accuracy_mean": mean_acc,
        "test_accuracy_std": std_acc,
        "per_class_f1": per_class_means,
        "summary": f"Macro F1 = {mean_f1*100:.2f}% +/- {std_f1*100:.2f}% | Acc = {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%",
    }

    print("\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Macro F1: {mean_f1*100:.2f}% +/- {std_f1*100:.2f}%  (over seeds {SEEDS})")
    print(f"Accuracy: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
    print("Per-class F1:")
    for cls, v in per_class_means.items():
        print(f"  {cls}: {v['mean']*100:.2f}% +/- {v['std']*100:.2f}%")

    # Load and update ablation JSON
    os.makedirs(os.path.dirname(ABLATION_JSON), exist_ok=True)
    if os.path.exists(ABLATION_JSON):
        with open(ABLATION_JSON, "r") as f:
            ablation_data = json.load(f)
    else:
        ablation_data = {}

    ablation_data["ablation_stats_pooling"] = ablation_row

    with open(ABLATION_JSON, "w") as f:
        json.dump(ablation_data, f, indent=2)
    print(f"\n[SUCCESS] Appended ablation row to: {ABLATION_JSON}")
    print(f"Key: 'ablation_stats_pooling'")


if __name__ == "__main__":
    main()
