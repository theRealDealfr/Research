"""
plot_pipeline_ablation_study.py
-------------------------------
Generates a comprehensive scientific ablation study accuracy plot for each stage
of the pediatric speech pathology pipeline:
- Stage 1: Screening Gate Ablation (Acoustic vs Direct Age vs AAFF)
- Stage 2: Disorder Type Ablation (WavLM vs AAFF vs Pitch Augmentation vs Bayesian)
- Stage 3: Disorder Subtypes Ablation (WavLM vs Age vs AAFF vs SMOTE vs Pitch Aug)
- Stage 4: Cascaded Hierarchical Routing Ablation (Flat Model vs Cascaded A/B/C vs Full + CDA)
- Cumulative Step-by-Step Accuracy & F1 Progression Waterfall

Outputs:
- pipeline_ablation_study.png
- results_controlled_baselines/pipeline_ablation_metrics.json
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def generate_ablation_plot():
    # -------------------------------------------------------------
    # Ablation Data for Each Stage
    # -------------------------------------------------------------

    # Stage 1: Binary Screening (TD vs SSD, N=169)
    s1_ablation = {
        "labels": [
            "1. Frozen WavLM\n(No Age)",
            "2. + Direct Age\n(Linear 1025-d)",
            "3. + AAFF Projection\n(Full Model C)"
        ],
        "accuracy": [85.80, 84.62, 91.12],
        "macro_f1": [85.79, 84.60, 91.10]
    }

    # Stage 2: Disorder Type (Articulation vs Phonological, N=71)
    s2_ablation = {
        "labels": [
            "1. WavLM Only\n(Baseline)",
            "2. + Direct Age\n(Linear)",
            "3. + AAFF Proj\n(Non-linear)",
            "4. + Pitch Aug\n(CDA)",
            "5. + Bayesian\n(TPE Tuned)"
        ],
        "accuracy": [83.10, 70.42, 84.51, 84.51, 83.10],
        "macro_f1": [45.38, 54.50, 53.43, 63.25, 67.73]
    }

    # Stage 3: Disorder Symptom Subtypes (4-Way Multiclass, N=71)
    s3_ablation = {
        "labels": [
            "1. WavLM Only\n(No Age)",
            "2. + Direct Age\n(Linear)",
            "3. + AAFF Proj\n(Default)",
            "4. + SMOTE\n(Feature Bal)",
            "5. + Pitch Aug\n(CDA Enhanced)"
        ],
        "accuracy": [28.17, 32.39, 33.80, 33.80, 38.03],
        "macro_f1": [17.52, 23.68, 25.26, 30.30, 33.87]
    }

    # Stage 4: End-to-End Hierarchical Pipeline (5-Way Routing, N=169)
    hier_ablation = {
        "labels": [
            "1. Flat 5-Way\n(No Hierarchy)",
            "2. Hierarchical\n(Model A Gate)",
            "3. Hierarchical\n(Model B Gate)",
            "4. Hierarchical\n(Model C AAFF)",
            "5. Full Pipeline\n(AAFF + CDA)"
        ],
        "accuracy": [54.44, 59.17, 60.95, 61.54, 62.72],
        "macro_f1": [25.20, 29.56, 34.03, 35.47, 41.26]
    }

    # Cumulative Component Value Progression (Hierarchical Macro F1 gain)
    cumulative_progression = {
        "steps": [
            "Baseline\n(Flat WavLM)",
            "+ Hierarchical\nGating",
            "+ Age Fusion\n(Direct Age)",
            "+ AAFF\nProjection",
            "+ Pitch Nuance\nAug (CDA)"
        ],
        "macro_f1": [25.20, 29.56, 34.03, 35.47, 41.26],
        "accuracy": [54.44, 59.17, 60.95, 61.54, 62.72]
    }

    # -------------------------------------------------------------
    # Visualization Layout: 5 Distinct Panels
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0:2])  # Spans 2 columns for hierarchical detail
    ax5 = fig.add_subplot(gs[1, 2])    # Progression waterfall

    # Colors
    c_acc = "#2563eb"
    c_acc_light = "#93c5fd"
    c_f1  = "#10b981"
    c_f1_light  = "#86efac"

    def plot_grouped_bars(ax, data, title, ylim, w=0.35):
        x = np.arange(len(data["labels"]))
        rects1 = ax.bar(x - w/2, data["accuracy"], w, label="Accuracy (%)", color=c_acc, edgecolor="#1e3a8a", alpha=0.9)
        rects2 = ax.bar(x + w/2, data["macro_f1"], w, label="Macro F1 (%)", color=c_f1,  edgecolor="#064e3b", alpha=0.9)

        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_ylabel("Score (%)", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(data["labels"], fontsize=8.5)
        ax.set_ylim(ylim)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

        # Labels
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#065f46")

    # Panel 1: Stage 1 Screening
    plot_grouped_bars(ax1, s1_ablation, "Stage 1: Binary Screening Gate Ablation\n(Typically Developing vs Speech Sound Disorder)", (75, 98))

    # Panel 2: Stage 2 Disorder Type
    plot_grouped_bars(ax2, s2_ablation, "Stage 2: Disorder Type Ablation\n(Articulation vs Phonological Disorder)", (35, 95))

    # Panel 3: Stage 3 Disorder Subtypes
    plot_grouped_bars(ax3, s3_ablation, "Stage 3: 4-Way Symptom Subtypes Ablation\n(Substitution, Stuttering, Omission, Addition)", (10, 48))

    # Panel 4: Hierarchical End-to-End Pipeline
    x_h = np.arange(len(hier_ablation["labels"]))
    w_h = 0.32
    r_h1 = ax4.bar(x_h - w_h/2, hier_ablation["accuracy"], w_h, label="Hierarchical Accuracy (%)", color="#7c3aed", edgecolor="#4c1d95", alpha=0.9)
    r_h2 = ax4.bar(x_h + w_h/2, hier_ablation["macro_f1"], w_h, label="Hierarchical Macro F1 (%)", color="#f59e0b", edgecolor="#78350f", alpha=0.9)

    ax4.set_title("End-to-End Hierarchical Pipeline Ablation (5-Way Diagnostic Routing, N=169 Untouched Patients)", fontsize=12, fontweight="bold", pad=10)
    ax4.set_ylabel("Score (%)", fontsize=10.5, fontweight="bold")
    ax4.set_xticks(x_h)
    ax4.set_xticklabels(hier_ablation["labels"], fontsize=9.5, fontweight="medium")
    ax4.set_ylim(15, 75)
    ax4.grid(axis='y', linestyle='--', alpha=0.35)
    ax4.legend(loc="upper left", fontsize=9.5, framealpha=0.9)

    for rect in r_h1:
        h = rect.get_height()
        ax4.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for rect in r_h2:
        h = rect.get_height()
        ax4.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#78350f")

    # Add annotation arrows showing cumulative jump
    ax4.annotate(f"+8.28% Acc Gain\n+16.06% F1 Gain",
                 xy=(4, hier_ablation["macro_f1"][4]), xytext=(3.4, 20),
                 arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.5),
                 fontsize=9, fontweight="bold", color="#b91c1c", bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec="#ef4444", lw=1))

    # Panel 5: Cumulative Progression Curve (Step-by-Step Pipeline Building)
    steps = cumulative_progression["steps"]
    f1_vals = cumulative_progression["macro_f1"]
    acc_vals = cumulative_progression["accuracy"]
    x_steps = np.arange(len(steps))

    ax5.plot(x_steps, acc_vals, marker='o', linewidth=2.5, markersize=7, color="#7c3aed", label="Accuracy Progression")
    ax5.plot(x_steps, f1_vals,  marker='s', linewidth=2.5, markersize=7, color="#10b981", label="Macro F1 Progression")

    ax5.set_title("Step-by-Step Cumulative Pipeline Gain\n(From Flat Baseline to Full Pipeline)", fontsize=11.5, fontweight="bold", pad=10)
    ax5.set_ylabel("Metric (%)", fontsize=10, fontweight="bold")
    ax5.set_xticks(x_steps)
    ax5.set_xticklabels(steps, fontsize=8)
    ax5.set_ylim(20, 72)
    ax5.grid(True, linestyle='--', alpha=0.35)
    ax5.legend(loc="lower right", fontsize=8.5)

    for i, (a, f) in enumerate(zip(acc_vals, f1_vals)):
        ax5.annotate(f"{a:.1f}%", xy=(i, a), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold", color="#5b21b6")
        ax5.annotate(f"{f:.1f}%", xy=(i, f), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold", color="#047857")

    plt.suptitle("Comprehensive Ablation Study: Pipeline Stages, Model Architectures, and Algorithmic Contributions", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_plot = "pipeline_ablation_study.png"
    plt.savefig(out_plot, dpi=300)
    print(f"[SUCCESS] Saved ablation study plot to: {out_plot}")
    plt.close()

    # Save structured JSON
    ablation_json = {
        "stage1_screening_ablation": s1_ablation,
        "stage2_disorder_type_ablation": s2_ablation,
        "stage3_symptom_subtypes_ablation": s3_ablation,
        "end_to_end_hierarchical_ablation": hier_ablation,
        "cumulative_progression": cumulative_progression
    }
    json_path = "results_controlled_baselines/pipeline_ablation_metrics.json"
    with open(json_path, "w") as f:
        json.dump(ablation_json, f, indent=2)
    print(f"[SUCCESS] Saved ablation metrics to: {json_path}")

if __name__ == "__main__":
    generate_ablation_plot()
