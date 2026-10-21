"""
plot_model_c_tables.py
-----------------------
Recreates the model_c_accuracy_table and model_c_f1_table style figures,
with an additional row for the Full Improved Pipeline (AAFF + CDA + Bayesian) and
a final row for the new stats implementation (LayerWeightedPool over 6 WavLM layers).

Stats ablation results (3-seed mean, val split as held-out eval):
  Stage 1 Accuracy: 96.35% +/- 0.37%
  Stage 1 Macro F1: 96.29% +/- 0.37%

Outputs:
  - model_c_accuracy_table_v2.png
  - model_c_f1_table_v2.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────

COLS = [
    "Stage 1: Screening\n(TD vs SSD)",
    "Task 2: Type\n(Artic vs Phon)",
    "Stage 3: Symptoms\n(4-Way Subtypes)",
    "End-to-End\nHierarchical (5-Way)",
]

# ── Accuracy ──────────────────────────────────────────────────
# Tuple: (config_label, [S1, T2, S3, E2E], mean_acc, delta_from_row1)
# Row 6 Stage 1 = 96.35% measured (stats ablation, 3-seed mean)
# Row 6 T2/S3   = row 4 values (classifiers unchanged)
# Row 6 E2E     = conservative estimate: row4 E2E + proportional gate lift
#                 gate improved +4.04pp on S1 -> ~+1.17pp on E2E routing
ACC_ROWS = [
    ("1. Base Model C (AAFF)",
     [91.12, 78.87, 33.80, 61.54], 66.33, None),
    ("2. + Audio Nuance (Pre-Emp + HPSS)",
     [91.72, 80.28, 35.21, 62.13], 67.34, +1.01),
    ("3. + CDA (Pitch Augmentation)",
     [91.72, 84.51, 38.03, 62.72], 69.25, +2.92),
    ("4. + Bayesian Search (Optuna TPE)",
     [92.31, 85.92, 39.44, 63.91], 70.40, +4.07),
    ("5. Full Pipeline\n    (AAFF + Nuance + CDA + Bayesian)",
     [92.31, 85.92, 39.44, 63.91], 70.40, +4.07),
    # Row 6: NEW — Stats LayerWeightedPool (measured Stage 1, estimated E2E)
    ("6. + Stats WavLM Gate\n    (LayerWeightedPool, L=6, 3-seed mean)",
     [96.35, 85.92, 39.44, 65.08], 71.70, +5.37),
]

# ── Macro F1 ──────────────────────────────────────────────────
# Row 6 Stage 1 F1 = 96.29% measured (stats ablation, 3-seed mean)
F1_ROWS = [
    ("1. Base Model C (AAFF)",
     [91.10, 53.43, 25.26, 35.47], 51.32, None),
    ("2. + Audio Nuance (Pre-Emp + HPSS)",
     [91.68, 57.80, 28.45, 37.80], 53.93, +2.61),
    ("3. + CDA (Pitch Augmentation)",
     [91.68, 63.25, 33.87, 41.26], 57.52, +6.20),
    ("4. + Bayesian Search (Optuna TPE)",
     [92.28, 67.73, 35.12, 43.15], 59.57, +8.25),
    ("5. Full Pipeline\n    (AAFF + Nuance + CDA + Bayesian)",
     [92.28, 67.73, 35.12, 43.15], 59.57, +8.25),
    # Row 6: NEW — Stats LayerWeightedPool
    ("6. + Stats WavLM Gate\n    (LayerWeightedPool, L=6, 3-seed mean)",
     [96.29, 67.73, 35.12, 43.72], 60.72, +9.40),
]

# ─────────────────────────────────────────────────────────────
# Colour tokens
# ─────────────────────────────────────────────────────────────
# Row type flags
ROW_NORMAL   = "normal"
ROW_PIPELINE = "pipeline"   # row 5 — full pipeline (amber)
ROW_STATS    = "stats"      # row 6 — new stats implementation (indigo)

def build_table(rows, cols, avg_col_label,
                header_color, avg_header_color,
                row_even, row_odd,
                avg_even, avg_odd,
                text_avg_color,
                title, out_path):

    n_rows = len(rows)
    n_cols = len(cols)
    total_cols = n_cols + 2   # config + data cols + avg

    fig_w = 14.0
    fig_h = 7.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    # ── Title ──
    fig.text(0.5, 0.97, title,
             ha="center", va="top",
             fontsize=14.5, fontweight="bold", color="#1a1a2e")

    # ── Column widths (sum to 1.0) ──
    col_widths = [0.28] + [0.14] * n_cols + [0.16]

    # ── Header ──
    header_labels = ["Ablation Configuration"] + cols + [avg_col_label]
    header_colors = [header_color] * (n_cols + 1) + [avg_header_color]

    # Row-type colour palettes
    PALETTE = {
        ROW_NORMAL:   {"cell": None,       "avg": None,       "edge": "#dddddd",
                       "txt": "#1a1a2e",   "avg_txt": text_avg_color,
                       "lw": 0.5,          "bold_avg": True},
        ROW_PIPELINE: {"cell": "#fef3c7",  "avg": "#fde68a",  "edge": "#d97706",
                       "txt": "#78350f",   "avg_txt": "#92400e",
                       "lw": 1.8,          "bold_avg": True},
        ROW_STATS:    {"cell": "#ede9fe",  "avg": "#c4b5fd",  "edge": "#7c3aed",
                       "txt": "#3b0764",   "avg_txt": "#4c1d95",
                       "lw": 2.2,          "bold_avg": True},
    }

    # Build cell data
    cell_text   = []
    cell_colors = []

    for i, (cfg, vals, avg, delta) in enumerate(rows):
        is_even = (i % 2 == 0)

        # Determine row type
        if i == n_rows - 1:
            rtype = ROW_STATS
        elif i == n_rows - 2:
            rtype = ROW_PIPELINE
        else:
            rtype = ROW_NORMAL

        # Format value cells  (append "*" to Stage-1 col of stats row for footnote)
        row_vals = []
        for j, v in enumerate(vals):
            suffix = "*" if (rtype == ROW_STATS and j == 0) else ""
            row_vals.append(f"{v:.2f}%{suffix}")

        # Avg cell
        if delta is None:
            avg_str = f"{avg:.2f}%"
        else:
            sign = "+" if delta >= 0 else ""
            avg_str = f"{avg:.2f}% ({sign}{delta:.2f}%)"
        if rtype == ROW_STATS:
            avg_str += "  †"

        cell_text.append([cfg] + row_vals + [avg_str])

        pal = PALETTE[rtype]
        if rtype == ROW_NORMAL:
            base_c = row_even if is_even else row_odd
            avg_c  = avg_even if is_even else avg_odd
        else:
            base_c = pal["cell"]
            avg_c  = pal["avg"]
        cell_colors.append([base_c] * (n_cols + 1) + [avg_c])

    # Draw table with explicit bounding box
    table = ax.table(
        cellText=cell_text,
        colLabels=header_labels,
        cellColours=cell_colors,
        colColours=header_colors,
        cellLoc="center",
        colWidths=col_widths,
        loc="center",
        bbox=[0.01, 0.16, 0.98, 0.77],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)

    # Style header
    for col_idx in range(total_cols):
        cell = table[0, col_idx]
        cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)
        cell.set_edgecolor("#cccccc")

    # Style data rows
    for row_idx in range(1, n_rows + 1):
        i = row_idx - 1
        if i == n_rows - 1:
            rtype = ROW_STATS
        elif i == n_rows - 2:
            rtype = ROW_PIPELINE
        else:
            rtype = ROW_NORMAL
        pal = PALETTE[rtype]

        for col_idx in range(total_cols):
            cell = table[row_idx, col_idx]
            cell.set_edgecolor(pal["edge"])
            cell.set_linewidth(pal["lw"])

            if col_idx == 0:
                cell.set_text_props(fontweight="bold", fontsize=9.0, color=pal["txt"])
            elif col_idx == total_cols - 1:
                cell.set_text_props(fontweight="bold", fontsize=9.5, color=pal["avg_txt"])
            else:
                fw = "bold" if rtype != ROW_NORMAL else "normal"
                cell.set_text_props(fontsize=9.2, color=pal["txt"], fontweight=fw)

    # Footnotes (bottom left)
    fig.text(0.02, 0.04,
             "* Stage 1 value = measured (3-seed mean ± 0.37%, val split as held-out eval)\n"
             "† Task 2 / Stage 3 unchanged (same classifiers); E2E estimated from gate improvement",
             ha="left", va="bottom", fontsize=8.0, color="#475569", style="italic")

    # Legend patches (bottom right)
    import matplotlib.patches as mpatches
    leg = [
        mpatches.Patch(facecolor="#fef3c7", edgecolor="#d97706", lw=1.5,
                       label="Full Pipeline (AAFF + Nuance + CDA + Bayesian)"),
        mpatches.Patch(facecolor="#ede9fe", edgecolor="#7c3aed", lw=1.5,
                       label="New: Stats WavLM Gate (LayerWeightedPool, L=6)"),
    ]
    ax.legend(handles=leg, loc="lower right", bbox_to_anchor=(0.99, 0.03),
              fontsize=8.5, framealpha=0.95, edgecolor="#cbd5e1")

    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[SUCCESS] Saved -> {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    base = r"c:\Users\91918\Desktop\Engineering\ENGINEERING S7\Research Code\Research"

    # ── Accuracy Table ──
    build_table(
        rows=ACC_ROWS,
        cols=COLS,
        avg_col_label="Overall Mean\nAccuracy",
        header_color="#1e3a8a",
        avg_header_color="#1e40af",
        row_even="#ffffff",
        row_odd="#f0f4ff",
        avg_even="#dbeafe",
        avg_odd="#bfdbfe",
        text_avg_color="#1d4ed8",
        title="Model C Incremental Ablation Study: Classification Accuracy (%)",
        out_path=os.path.join(base, "model_c_accuracy_table_v2.png"),
    )

    # ── F1 Table ──
    build_table(
        rows=F1_ROWS,
        cols=COLS,
        avg_col_label="Overall Mean\nMacro F1",
        header_color="#065f46",
        avg_header_color="#064e3b",
        row_even="#ffffff",
        row_odd="#f0fdf4",
        avg_even="#d1fae5",
        avg_odd="#a7f3d0",
        text_avg_color="#065f46",
        title="Model C Incremental Ablation Study: Macro F1-Score (%)",
        out_path=os.path.join(base, "model_c_f1_table_v2.png"),
    )
