"""
plot_incremental_ablation.py
-----------------------------
Generates two comprehensive figures (F1 and Accuracy separately) for an
incremental ablation study across all pipeline stages.

Layout (per figure):
  Row 1: Stage 1 | Stage 2 | Stage 3
  Row 2: Stage 4 (End-to-End Hierarchical) spanning full width

The final bar in every stage is always the full completed pipeline variant,
visually highlighted with a golden star annotation.

Outputs:
  - incremental_ablation_accuracy.png
  - incremental_ablation_f1.png
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Data definitions

# Stage 1: Binary Screening (TD vs SSD, N=169)
S1 = {
    "title": "Stage 1 - Binary Screening Gate\n(Typically Developing vs. SSD, N=169)",
    "labels": [
        "Frozen WavLM\n(No Age)",
        "+ Direct Age\n(Linear 1025-d)",
        "+ AAFF Proj.\n(Full Model C) *",
    ],
    "accuracy": [85.80, 84.62, 91.12],
    "macro_f1":  [85.79, 84.60, 91.10],
}

# Stage 2: Disorder Type (Articulation vs Phonological, N=71)
S2 = {
    "title": "Stage 2 - Disorder Type Classifier\n(Articulation vs. Phonological, N=71)",
    "labels": [
        "WavLM Only\n(Baseline)",
        "+ Direct Age\n(Linear)",
        "+ AAFF Proj.\n(Non-linear)",
        "+ Pitch Aug.\n(CDA)",
        "+ Bayesian HPO\n(TPE Tuned) *",
    ],
    "accuracy": [83.10, 70.42, 84.51, 84.51, 85.92],
    "macro_f1":  [45.38, 54.50, 53.43, 63.25, 67.73],
}

# Stage 3: 4-Way Symptom Subtypes (N=71)
S3 = {
    "title": "Stage 3 - Symptom Subtype Classifier\n(4-Way Multiclass, N=71)",
    "labels": [
        "WavLM Only\n(No Age)",
        "+ Direct Age\n(Linear)",
        "+ AAFF Proj.\n(Default)",
        "+ SMOTE\n(Feature Bal.)",
        "+ Pitch Aug.\n(CDA) *",
    ],
    "accuracy": [28.17, 32.39, 33.80, 33.80, 38.03],
    "macro_f1":  [17.52, 23.68, 25.26, 30.30, 33.87],
}

# Stage 4 / End-to-End: 5-Way Hierarchical Routing (N=169)
S4 = {
    "title": "Stage 4 - End-to-End Hierarchical Pipeline\n(5-Way Diagnostic Routing, N=169 untouched patients)",
    "labels": [
        "Flat 5-Way\n(No Hierarchy)",
        "Hierarchical\n(Model A Gate)",
        "Hierarchical\n(Model B Gate)",
        "Hierarchical\n(Model C AAFF)",
        "Full Pipeline\n(AAFF + CDA) *",
    ],
    "accuracy": [54.44, 59.17, 60.95, 61.54, 62.72],
    "macro_f1":  [25.20, 29.56, 34.03, 35.47, 41.26],
}

PALETTE_ACC = {
    "baseline":  "#5e81f4",
    "mid":       "#3b64e8",
    "final":     "#1a3fb5",
    "edge":      "#0f2470",
    "final_edge":"#f59e0b",
}

PALETTE_F1 = {
    "baseline":  "#34d399",
    "mid":       "#10b981",
    "final":     "#059669",
    "edge":      "#064e3b",
    "final_edge":"#f59e0b",
}

FINAL_GOLD = "#f59e0b"
BG_COLOR   = "#0f172a"
PANEL_BG   = "#1e293b"
GRID_COLOR = "#334155"
TEXT_COLOR = "#f1f5f9"
SPINE_COLOR= "#475569"


def draw_stage(ax, stage_data, metric, palette, ylim):
    labels  = stage_data["labels"]
    values  = stage_data[metric]
    n       = len(values)
    x       = np.arange(n)
    bar_w   = 0.60

    bar_colors  = []
    edge_colors = []
    linewidths  = []
    for i in range(n):
        if i == n - 1:
            bar_colors.append(palette["final"])
            edge_colors.append(palette["final_edge"])
            linewidths.append(2.2)
        elif i == 0:
            bar_colors.append(palette["baseline"])
            edge_colors.append(palette["edge"])
            linewidths.append(1.2)
        else:
            bar_colors.append(palette["mid"])
            edge_colors.append(palette["edge"])
            linewidths.append(1.2)

    bars = ax.bar(x, values, width=bar_w,
                  color=bar_colors, edgecolor=edge_colors,
                  linewidth=linewidths, zorder=3)

    # Golden highlight behind final bar
    ax.axvspan(n - 1 - bar_w * 0.55, n - 1 + bar_w * 0.55,
               alpha=0.09, color=FINAL_GOLD, zorder=2)

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        weight = "bold" if i == n - 1 else "normal"
        color  = FINAL_GOLD if i == n - 1 else TEXT_COLOR
        ax.text(bar.get_x() + bar.get_width() / 2, val + (ylim[1]-ylim[0])*0.012,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=8.0, fontweight=weight, color=color, zorder=5)

    # Gain annotation arrow
    gain = values[-1] - values[0]
    sign = "+" if gain >= 0 else ""
    mid_x   = (0 + (n-1)) / 2 - 0.3
    arrow_y = ylim[0] + (ylim[1] - ylim[0]) * 0.10
    ax.annotate(
        f"{sign}{gain:.2f}%\ngain",
        xy=(n - 1, values[-1]),
        xytext=(mid_x, arrow_y),
        arrowprops=dict(arrowstyle="->", color=FINAL_GOLD, lw=1.5,
                        connectionstyle="arc3,rad=0.15"),
        fontsize=8, fontweight="bold", color=FINAL_GOLD,
        bbox=dict(boxstyle="round,pad=0.25", fc="#0f172a", ec=FINAL_GOLD, lw=1.1),
        zorder=6,
    )

    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.2, color=TEXT_COLOR)
    ax.tick_params(axis="y", labelsize=8.5, colors=TEXT_COLOR)
    ax.tick_params(axis="x", colors=TEXT_COLOR, length=3)
    ax.set_title(stage_data["title"], fontsize=10.5, fontweight="bold",
                 color=TEXT_COLOR, pad=9)
    ax.set_ylabel("Score (%)", fontsize=9, color=TEXT_COLOR, labelpad=6)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color=GRID_COLOR, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)
        spine.set_linewidth(0.8)


def build_figure(metric, palette, ylims, out_path, suptitle):
    fig = plt.figure(figsize=(20, 13), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)

    gs = GridSpec(2, 3, figure=fig,
                  hspace=0.55, wspace=0.30,
                  left=0.06, right=0.97,
                  top=0.91, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])

    for ax, stage, ylim in zip([ax1, ax2, ax3], [S1, S2, S3], ylims[:3]):
        draw_stage(ax, stage, metric, palette, ylim)
    draw_stage(ax4, S4, metric, palette, ylims[3])

    legend_handles = [
        mpatches.Patch(color=palette["baseline"], label="Baseline variant"),
        mpatches.Patch(color=palette["mid"],      label="Intermediate variant"),
        mpatches.Patch(facecolor=palette["final"],
                       edgecolor=palette["final_edge"],
                       linewidth=1.5,             label="Full completed pipeline (*)"),
    ]
    fig.legend(handles=legend_handles, loc="upper right",
               fontsize=9.5, framealpha=0.25, facecolor=PANEL_BG,
               edgecolor=SPINE_COLOR, labelcolor=TEXT_COLOR,
               ncol=3, bbox_to_anchor=(0.97, 0.948))

    fig.text(0.5, 0.965, suptitle, ha="center", va="center",
             fontsize=16, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.5, 0.945,
             "* Final bar = Full completed pipeline  |  Arrows show total gain from stage baseline",
             ha="center", va="center", fontsize=9.5, color="#94a3b8")

    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    print(f"[SUCCESS] Saved -> {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    base = r"c:\Users\91918\Desktop\Engineering\ENGINEERING S7\Research Code\Research"

    build_figure(
        metric="accuracy",
        palette=PALETTE_ACC,
        ylims=[(77, 97), (60, 96), (20, 48), (44, 73)],
        out_path=os.path.join(base, "incremental_ablation_accuracy.png"),
        suptitle="Incremental Ablation Study - Accuracy (%)  |  All Pipeline Stages",
    )

    build_figure(
        metric="macro_f1",
        palette=PALETTE_F1,
        ylims=[(77, 97), (38, 78), (10, 45), (18, 55)],
        out_path=os.path.join(base, "incremental_ablation_f1.png"),
        suptitle="Incremental Ablation Study - Macro F1 (%)  |  All Pipeline Stages",
    )
