"""
plot_phase18_visualizations.py
------------------------------
Generates all Phase 18 required figures from phase18_master_comparison.json:
1. stage3_confusion_matrix.png
2. per_class_recall_comparison.png
3. macro_f1_comparison.png
4. accuracy_comparison.png
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    output_dir = "./results_controlled_baselines"
    json_path = os.path.join(output_dir, "phase18_master_comparison.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    classes = ["Substitution", "Stuttering", "Omission", "Addition"]
    table_rows = [
        "Baseline",
        "Augmentation",
        "Class-weighted",
        "Balanced Sampling",
        "SMOTE",
        "Augmentation + SMOTE",
        "Fine-tuned WavLM",
        "WavLM + Acoustic",
        "WavLM + Acoustic + Age",
        "Best Combined Model"
    ]

    # -------------------------------------------------------------------------
    # 1. Stage 3 Confusion Matrix (Winning Model: WavLM + Acoustic + Age)
    # -------------------------------------------------------------------------
    winner = data["WavLM + Acoustic + Age"]
    cm = np.array(winner["report"]["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Stage 3: Symptom Confusion Matrix\n(WavLM + Acoustic + Age Fusion, N=71)",
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold", fontsize=12)

    fig.tight_layout()
    cm_path = os.path.join(output_dir, "stage3_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_path}")

    # -------------------------------------------------------------------------
    # 2. Per-Class Recall Graph
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    x_c = np.arange(len(classes))
    bar_w = 0.08
    selected_exps = [
        "Baseline", "Class-weighted", "SMOTE", "Augmentation + SMOTE",
        "Focal Loss", "WavLM + Acoustic + Age", "Best Combined Model"
    ]
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1']

    for idx, (exp_name, col) in enumerate(zip(selected_exps, colors)):
        if exp_name in data:
            rep = data[exp_name]["report"]["report"]
            recalls = [rep[str(c_idx)]["recall"] * 100 for c_idx in range(4)]
            offset = (idx - len(selected_exps)/2 + 0.5) * bar_w
            ax.bar(x_c + offset, recalls, bar_w, label=exp_name, color=col)

    ax.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
    ax.set_title('Stage 3 Per-Class Recall Comparison across Models', fontsize=13, fontweight='bold')
    ax.set_xticks(x_c)
    ax.set_xticklabels([f"{c} (Minority)" if c == "Addition" else c for c in classes], fontsize=11, fontweight='bold')
    ax.legend(frameon=True, fontsize=9, loc='upper right')
    ax.set_ylim(0, 80)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig.tight_layout()
    recall_path = os.path.join(output_dir, "per_class_recall_comparison.png")
    plt.savefig(recall_path, dpi=300)
    plt.close()
    print(f"Saved: {recall_path}")

    # -------------------------------------------------------------------------
    # 3. Macro-F1 Comparison across All 10 Models
    # -------------------------------------------------------------------------
    labels = [r.replace(" + ", "+\n").replace(" Model", "") for r in table_rows]
    val_f1s = [data[r]["val_f1"] * 100 for r in table_rows]
    test_f1s = [data[r]["test_f1"] * 100 for r in table_rows]

    x = np.arange(len(table_rows))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))
    rects1 = ax.bar(x - width/2, val_f1s, width, label='Validation Macro-F1 (Tuning Criterion)', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, test_f1s, width, label='Test Macro-F1 (Out-of-Sample)', color='#d95f02')

    ax.set_ylabel('Macro F1-Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Phase 18: Macro-F1 Comparison across All 10 Experimental Conditions', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, fontweight='bold')
    ax.legend(frameon=True, fontsize=11)
    ax.set_ylim(0, 70)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for rects in [rects1, rects2]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    fig.tight_layout()
    macro_f1_path = os.path.join(output_dir, "macro_f1_comparison.png")
    plt.savefig(macro_f1_path, dpi=300)
    plt.close()
    print(f"Saved: {macro_f1_path}")

    # -------------------------------------------------------------------------
    # 4. Accuracy Comparison across All 10 Models
    # -------------------------------------------------------------------------
    val_accs = [data[r]["val_acc"] * 100 for r in table_rows]
    test_accs = [data[r]["test_acc"] * 100 for r in table_rows]

    fig, ax = plt.subplots(figsize=(14, 6))
    rects1 = ax.bar(x - width/2, val_accs, width, label='Validation Accuracy', color='#388e3c')
    rects2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy (Out-of-Sample)', color='#7b1fa2')

    ax.axhline(60, color='red', linestyle='--', linewidth=1.5, label='Target Benchmark (60%)')

    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Phase 18: Accuracy Comparison across All 10 Experimental Conditions', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, fontweight='bold')
    ax.legend(frameon=True, fontsize=11)
    ax.set_ylim(0, 75)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for rects in [rects1, rects2]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    fig.tight_layout()
    acc_path = os.path.join(output_dir, "accuracy_comparison.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"Saved: {acc_path}")

if __name__ == "__main__":
    main()
