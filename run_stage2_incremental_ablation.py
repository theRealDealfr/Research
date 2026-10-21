"""
run_stage2_incremental_ablation.py
===================================
Runs Stage 2 disorder-subtype classification across 5 incremental stages,
each building on Model C (AAFF) as the base:

  Stage 0 : Model C baseline            (in->256->4,  unaugmented,  no Bayesian)
  Stage 1 : + Wider Head                (in->512->256->4, unaugmented, no Bayes)
  Stage 2 : + CDA Pitch Augmentation   (in->512->256->4, augmented,  no Bayes)
  Stage 3 : + Bayesian Search           (in->proj_dim->4 Bayes-tuned, unaugmented)
  Stage 4 : + Wider Head + CDA + Bayes (in->512->256->4, augmented,  Bayes-tuned)

Outputs:
  results_controlled_baselines/stage2_incremental_ablation.json
  stage2_incremental_ablation_accuracy.png
  stage2_incremental_ablation_f1.png
"""
import os, copy, random, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import StratifiedKFold
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Seeds --------------------------------------------------------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# --- Model Architectures ------------------------------------------------------
class AAFFBaseline(nn.Module):
    """Model C baseline: in -> proj_dim -> num_classes  (1 hidden layer)"""
    def __init__(self, in_features=1025, proj_dim=256, dropout=0.2, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, proj_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.head = nn.Linear(proj_dim, num_classes)
    def forward(self, emb, age):
        x = torch.cat([emb, age], dim=1)
        return self.head(self.net(x))

class AAFFWider(nn.Module):
    """Wider head: in -> 512 -> 256 -> num_classes  (2 hidden layers)"""
    def __init__(self, in_features=1025, dropout=0.35, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),         nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(256, num_classes)
    def forward(self, emb, age):
        x = torch.cat([emb, age], dim=1)
        return self.head(self.net(x))

class AAFFBayesian(nn.Module):
    """Bayesian-tunable: in -> proj_dim -> num_classes  (1 hidden layer, dim flexible)"""
    def __init__(self, in_features=1025, proj_dim=256, dropout=0.2, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, proj_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.head = nn.Linear(proj_dim, num_classes)
    def forward(self, emb, age):
        x = torch.cat([emb, age], dim=1)
        return self.head(self.net(x))

class AAFFWiderBayesian(nn.Module):
    """Wider head with Bayesian-tuned dropout: in -> 512 -> 256 -> num_classes"""
    def __init__(self, in_features=1025, dropout=0.35, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),         nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(256, num_classes)
    def forward(self, emb, age):
        x = torch.cat([emb, age], dim=1)
        return self.head(self.net(x))

# --- Utilities ----------------------------------------------------------------
def class_weight_tensor(dataset, nc, device):
    lbls = [y.item() for _, _, y in dataset]
    cnt  = np.bincount(lbls, minlength=nc)
    w    = len(lbls) / (float(nc) * np.maximum(cnt, 1))
    return torch.tensor(w, dtype=torch.float32).to(device)

def evaluate(model, dl, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for be, ba, bl in dl:
            logits = model(be.to(device), ba.to(device))
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(bl.numpy())
    y_t, y_p = np.array(labels), np.array(preds)
    acc = accuracy_score(y_t, y_p)
    _, _, fm, _ = precision_recall_fscore_support(y_t, y_p, average="macro",    zero_division=0)
    _, _, fw, _ = precision_recall_fscore_support(y_t, y_p, average="weighted", zero_division=0)
    return {"accuracy": acc, "f1_macro": fm, "f1_weighted": fw, "y_true": y_t, "y_pred": y_p}

def train_loop(model_fn, model_kw, train_ds, val_ds, test_ds, device, nc,
               lr_list, epochs, seed, label_smoothing=0.0, use_cosine=True):
    """Train over lr_list, return best model metrics (val-selected)."""
    wt     = class_weight_tensor(train_ds, nc, device)
    crit   = nn.CrossEntropyLoss(weight=wt, label_smoothing=label_smoothing)
    tr_dl  = DataLoader(train_ds,  batch_size=16, shuffle=True)
    va_dl  = DataLoader(val_ds,    batch_size=16, shuffle=False)
    te_dl  = DataLoader(test_ds,   batch_size=16, shuffle=False)

    best_f1, best_state, best_lr = -1.0, None, None
    for lr in lr_list:
        set_seed(seed)
        m   = model_fn(**model_kw).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if use_cosine else None
        bfl, bsl = -1.0, None
        for _ in range(epochs):
            m.train()
            for be, ba, bl in tr_dl:
                opt.zero_grad()
                nn.CrossEntropyLoss(weight=wt, label_smoothing=label_smoothing)(
                    m(be.to(device), ba.to(device)), bl.to(device)
                ).backward(); opt.step()
            if sch: sch.step()
            vf = evaluate(m, va_dl, device)["f1_macro"]
            if vf > bfl: bfl, bsl = vf, copy.deepcopy(m.state_dict())
        print(f"    lr={lr:.0e}  val_f1={bfl:.4f}")
        if bfl > best_f1: best_f1, best_state, best_lr = bfl, bsl, lr

    fm = model_fn(**model_kw).to(device)
    fm.load_state_dict(best_state)
    val_res  = evaluate(fm, va_dl, device)
    test_res = evaluate(fm, te_dl, device)
    return fm, best_lr, val_res, test_res

def bayesian_train(train_ds_all, val_ds, test_ds, device, nc, in_features,
                   model_cls, fixed_arch_kw,
                   n_trials=25, trial_epochs=25, final_epochs=60, seed=42):
    """
    Optuna TPE search over lr, weight_decay, dropout (and optionally proj_dim).
    fixed_arch_kw: dict of model kwargs that are NOT searched (e.g. {'in_features':..., 'num_classes':...})
    model_cls must accept dropout kwarg at minimum.
    """
    emb = train_ds_all.tensors[0].numpy()
    age = train_ds_all.tensors[1].numpy()
    lbl = train_ds_all.tensors[2].numpy()
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    def objective(trial):
        lr  = trial.suggest_float("lr",  1e-4, 1e-2, log=True)
        wd  = trial.suggest_float("wd",  1e-4, 1e-1, log=True)
        do  = trial.suggest_float("do",  0.10, 0.50, step=0.05)
        bs  = trial.suggest_categorical("bs", [8, 16, 32])
        # proj_dim only if model supports it (stage 3 single-layer Bayesian)
        if "proj_dim" in model_cls.__init__.__code__.co_varnames:
            pd = trial.suggest_categorical("proj_dim", [128, 256, 512])
            kw = {**fixed_arch_kw, "dropout": do, "proj_dim": pd}
        else:
            kw = {**fixed_arch_kw, "dropout": do}

        fold_f1s = []
        for fi, (ti, vi) in enumerate(skf.split(emb, lbl)):
            t_ds = TensorDataset(
                torch.tensor(emb[ti], dtype=torch.float32),
                torch.tensor(age[ti], dtype=torch.float32),
                torch.tensor(lbl[ti], dtype=torch.long)
            )
            v_ds = TensorDataset(
                torch.tensor(emb[vi], dtype=torch.float32),
                torch.tensor(age[vi], dtype=torch.float32),
                torch.tensor(lbl[vi], dtype=torch.long)
            )
            wt   = class_weight_tensor(t_ds, nc, device)
            crit = nn.CrossEntropyLoss(weight=wt)
            m    = model_cls(**kw).to(device)
            opt  = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
            sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=trial_epochs)
            tr_dl = DataLoader(t_ds, batch_size=bs, shuffle=True)
            va_dl = DataLoader(v_ds, batch_size=bs, shuffle=False)
            best_vf = -1.0
            for _ in range(trial_epochs):
                m.train()
                for be, ba, bl in tr_dl:
                    opt.zero_grad()
                    crit(m(be.to(device), ba.to(device)), bl.to(device)).backward(); opt.step()
                sch.step()
                vf = evaluate(m, va_dl, device)["f1_macro"]
                if vf > best_vf: best_vf = vf
            fold_f1s.append(best_vf)
        return float(np.mean(fold_f1s))

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    print(f"    Best params: {bp}  cv_f1={study.best_value:.4f}")

    # Determine final kw
    if "proj_dim" in model_cls.__init__.__code__.co_varnames:
        final_kw = {**fixed_arch_kw, "dropout": bp["do"],
                    "proj_dim": bp.get("proj_dim", fixed_arch_kw.get("proj_dim", 256))}
    else:
        final_kw = {**fixed_arch_kw, "dropout": bp["do"]}

    _, _, val_res, test_res = train_loop(
        model_cls, final_kw, train_ds_all, val_ds, test_ds, device, nc,
        lr_list=[bp["lr"]], epochs=final_epochs, seed=seed, label_smoothing=0.10
    )
    return val_res, test_res, bp

# --- Main ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials",      type=int, default=25)
    parser.add_argument("--trial_epochs",  type=int, default=25)
    parser.add_argument("--final_epochs",  type=int, default=60)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    UNAUG = "./cached_embeddings/wavlm_stage2_disorder_symptom.pt"
    AUG   = "./cached_embeddings/wavlm_stage2_disorder_symptom_augmented.pt"
    OUT   = "./results_controlled_baselines"
    os.makedirs(OUT, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load caches
    cu = torch.load(UNAUG, map_location="cpu", weights_only=False)
    ca = torch.load(AUG,   map_location="cpu", weights_only=False)
    dc = cu["disorder_classes"]
    nc = len(dc)

    def ds(cache, split): return TensorDataset(
        cache[split]["embeddings"], cache[split]["ages"], cache[split]["labels"]
    )
    tr_u = ds(cu, "train"); va = ds(cu, "val"); te = ds(cu, "test")
    tr_a = ds(ca, "train")  # augmented train (630 samples)
    in_f = 1025             # 1024-d emb + 1-d age

    results = []

    # -- Stage 0: Baseline AAFF (Model C head, unaugmented) ------------------
    print("=" * 60)
    print("Stage 0: Baseline AAFF  (in->256->4, unaugmented)")
    print("=" * 60)
    _, lr0, vr0, tr0 = train_loop(
        AAFFBaseline, {"in_features": in_f, "proj_dim": 256, "dropout": 0.2, "num_classes": nc},
        tr_u, va, te, device, nc, [1e-3, 5e-4, 3e-4, 1e-4], 50, args.seed, 0.0
    )
    results.append({
        "stage": 0, "label": "Baseline AAFF\n(in?256?4)",
        "test_acc": tr0["accuracy"]*100, "test_f1": tr0["f1_macro"]*100,
        "val_acc":  vr0["accuracy"]*100, "val_f1":  vr0["f1_macro"]*100,
    })
    print(f"  Test Acc={tr0['accuracy']*100:.2f}%  F1={tr0['f1_macro']*100:.2f}%\n")

    # -- Stage 1: Wider Head, unaugmented, no Bayes --------------------------
    print("=" * 60)
    print("Stage 1: Wider Head  (in->512->256->4, unaugmented)")
    print("=" * 60)
    _, lr1, vr1, tr1 = train_loop(
        AAFFWider, {"in_features": in_f, "dropout": 0.35, "num_classes": nc},
        tr_u, va, te, device, nc, [1e-3, 5e-4, 3e-4, 1e-4], 50, args.seed, 0.10
    )
    results.append({
        "stage": 1, "label": "+ Wider Head\n(in?512?256?4)",
        "test_acc": tr1["accuracy"]*100, "test_f1": tr1["f1_macro"]*100,
        "val_acc":  vr1["accuracy"]*100, "val_f1":  vr1["f1_macro"]*100,
    })
    print(f"  Test Acc={tr1['accuracy']*100:.2f}%  F1={tr1['f1_macro']*100:.2f}%\n")

    # -- Stage 2: Wider Head + CDA Augmentation ------------------------------
    print("=" * 60)
    print("Stage 2: Wider Head + CDA  (in->512->256->4, augmented)")
    print("=" * 60)
    _, lr2, vr2, tr2 = train_loop(
        AAFFWider, {"in_features": in_f, "dropout": 0.35, "num_classes": nc},
        tr_a, va, te, device, nc, [1e-3, 5e-4, 3e-4, 1e-4], 50, args.seed, 0.10
    )
    results.append({
        "stage": 2, "label": "+ CDA Augmentation\n(Pitch Shift)",
        "test_acc": tr2["accuracy"]*100, "test_f1": tr2["f1_macro"]*100,
        "val_acc":  vr2["accuracy"]*100, "val_f1":  vr2["f1_macro"]*100,
    })
    print(f"  Test Acc={tr2['accuracy']*100:.2f}%  F1={tr2['f1_macro']*100:.2f}%\n")

    # -- Stage 3: Bayesian on baseline head, unaugmented ---------------------
    print("=" * 60)
    print(f"Stage 3: Bayesian Tuning on Baseline Head  ({args.n_trials} trials)")
    print("=" * 60)
    vr3, tr3, bp3 = bayesian_train(
        tr_u, va, te, device, nc, in_f,
        AAFFBayesian, {"in_features": in_f, "num_classes": nc},
        n_trials=args.n_trials, trial_epochs=args.trial_epochs,
        final_epochs=args.final_epochs, seed=args.seed
    )
    results.append({
        "stage": 3, "label": "+ Bayesian Search\n(Baseline Head)",
        "test_acc": tr3["accuracy"]*100, "test_f1": tr3["f1_macro"]*100,
        "val_acc":  vr3["accuracy"]*100, "val_f1":  vr3["f1_macro"]*100,
        "best_params": bp3,
    })
    print(f"  Test Acc={tr3['accuracy']*100:.2f}%  F1={tr3['f1_macro']*100:.2f}%\n")

    # -- Stage 4: Wider Head + CDA + Bayesian --------------------------------
    print("=" * 60)
    print(f"Stage 4: Wider Head + CDA + Bayesian  ({args.n_trials} trials)")
    print("=" * 60)
    vr4, tr4, bp4 = bayesian_train(
        tr_a, va, te, device, nc, in_f,
        AAFFWiderBayesian, {"in_features": in_f, "num_classes": nc},
        n_trials=args.n_trials, trial_epochs=args.trial_epochs,
        final_epochs=args.final_epochs, seed=args.seed
    )
    results.append({
        "stage": 4, "label": "+ Bayesian on\nWider Head + CDA",
        "test_acc": tr4["accuracy"]*100, "test_f1": tr4["f1_macro"]*100,
        "val_acc":  vr4["accuracy"]*100, "val_f1":  vr4["f1_macro"]*100,
        "best_params": bp4,
    })
    print(f"  Test Acc={tr4['accuracy']*100:.2f}%  F1={tr4['f1_macro']*100:.2f}%\n")

    # -- Save JSON ------------------------------------------------------------
    json_path = os.path.join(OUT, "stage2_incremental_ablation.json")
    json_results = []
    for r in results:
        jr = {k: (v if not isinstance(v, np.floating) else float(v)) for k, v in r.items()}
        json_results.append(jr)
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved results -> {json_path}")

    # -- Print summary table --------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 2 INCREMENTAL ABLATION — SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Stage':<6} {'Configuration':<35} {'Test Acc':>9} {'Test F1':>8} {'Delta Acc':>10} {'Delta F1':>9}")
    print("-" * 70)
    base_acc = results[0]["test_acc"]
    base_f1  = results[0]["test_f1"]
    for r in results:
        da = r["test_acc"] - base_acc
        df = r["test_f1"]  - base_f1
        lbl = r["label"].replace("\n", " ")
        print(f"  {r['stage']}    {lbl:<35} {r['test_acc']:>8.2f}% {r['test_f1']:>7.2f}%  {da:>+9.2f}%  {df:>+8.2f}%")
    print("=" * 70)

    # -- Generate plots -------------------------------------------------------
    _plot(results)


def _plot(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[warn] matplotlib unavailable — skipping plots.")
        return

    BG      = "#0b0f19"
    CARD    = "#111827"
    GRID    = "#1f2937"
    TEXT    = "#e5e7eb"
    MUTED   = "#6b7280"
    CYAN    = "#06b6d4"
    INDIGO  = "#818cf8"
    EMERALD = "#10b981"
    AMBER   = "#f59e0b"
    RED     = "#f87171"

    stage_colors = [MUTED, CYAN, INDIGO, AMBER, EMERALD]
    labels = [r["label"] for r in results]
    accs   = [r["test_acc"] for r in results]
    f1s    = [r["test_f1"]  for r in results]
    xs     = list(range(len(results)))

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35)

    def ax_style(ax, title):
        ax.set_facecolor(CARD)
        for sp in ax.spines.values(): sp.set_color(GRID)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.set_title(title, color=CYAN, fontsize=10, fontweight="bold", pad=10)
        ax.yaxis.label.set_color(TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.6)

    # -- Panel 1 (top full): Line chart — Acc + F1 across stages -------------
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(xs, accs, marker="o", markersize=8, linewidth=2.0,
             color=CYAN, label="Test Accuracy (%)", zorder=3)
    ax1.plot(xs, f1s,  marker="s", markersize=8, linewidth=2.0,
             color=INDIGO, linestyle="--", label="Test Macro F1 (%)", zorder=3)
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax1.annotate(f"{a:.1f}%", (i, a), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8, color=CYAN, fontweight="bold")
        ax1.annotate(f"{f:.1f}%", (i, f), textcoords="offset points",
                     xytext=(0, -16), ha="center", fontsize=8, color=INDIGO)
    # Shade delta from baseline
    ax1.fill_between(xs, accs[0], accs, alpha=0.12, color=CYAN)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([r["label"] for r in results], fontsize=8, color=TEXT)
    ax1.set_ylabel("Score (%)", fontsize=9)
    ax1.set_ylim(max(0, min(accs+f1s) - 5), min(100, max(accs+f1s) + 8))
    ax1.legend(fontsize=9, facecolor=GRID, labelcolor=TEXT)
    ax_style(ax1, "Stage 2 Disorder Subtype — Incremental Ablation (Test Performance)")

    # -- Panel 2 (bottom-left): Bar chart — Accuracy --------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    bars = ax2.bar(xs, accs, color=stage_colors, width=0.55, zorder=3)
    ax2.axhline(accs[0], color=MUTED, linewidth=1.0, linestyle=":", label=f"Baseline {accs[0]:.1f}%")
    for bar, val in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                 f"{val:.2f}%", ha="center", va="bottom", fontsize=7.5,
                 color=TEXT, fontweight="bold")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([r["label"] for r in results], fontsize=7)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=9)
    ax2.set_ylim(0, min(100, max(accs) + 8))
    ax2.legend(fontsize=8, facecolor=GRID, labelcolor=TEXT)
    ax_style(ax2, "Test Accuracy by Stage")

    # -- Panel 3 (bottom-right): Bar chart — Macro F1 -------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    bars3 = ax3.bar(xs, f1s, color=stage_colors, width=0.55, zorder=3)
    ax3.axhline(f1s[0], color=MUTED, linewidth=1.0, linestyle=":", label=f"Baseline {f1s[0]:.1f}%")
    for bar, val in zip(bars3, f1s):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                 f"{val:.2f}%", ha="center", va="bottom", fontsize=7.5,
                 color=TEXT, fontweight="bold")
    ax3.set_xticks(xs)
    ax3.set_xticklabels([r["label"] for r in results], fontsize=7)
    ax3.set_ylabel("Test Macro F1 (%)", fontsize=9)
    ax3.set_ylim(0, min(100, max(f1s) + 8))
    ax3.legend(fontsize=8, facecolor=GRID, labelcolor=TEXT)
    ax_style(ax3, "Test Macro F1 by Stage")

    fig.suptitle("Stage 2 Incremental Ablation — Model C ? Wider Head ? CDA ? Bayesian",
                 color=TEXT, fontsize=12, fontweight="bold", y=0.99)

    for fname in ["stage2_incremental_ablation_accuracy.png",
                  "stage2_incremental_ablation_f1.png"]:
        plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("Plots saved -> stage2_incremental_ablation_accuracy.png / _f1.png")


if __name__ == "__main__":
    main()

