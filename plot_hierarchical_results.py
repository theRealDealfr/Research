"""
plot_hierarchical_results.py
----------------------------
Generates all visualization artifacts for the hierarchical speech classification pipeline:
  1. stage1_confusion_matrix.png
  2. stage2_confusion_matrix.png
  3. stage3_confusion_matrix.png
  4. macro_f1_comparison.png (Side-by-side Macro-F1 across Exp A, B, C, D)
  5. per_class_recall_comparison.png (Per-class recall progression across Exp A, B, C, D)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_cm(cm, class_names, title, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
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
                    fontweight="bold", fontsize=11)

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    output_dir = "./results_controlled_baselines"
    os.makedirs(output_dir, exist_ok=True)

    results_json = os.path.join(output_dir, "hierarchical_experiments_results.json")
    if not os.path.exists(results_json):
        raise FileNotFoundError(f"Missing {results_json}")

    with open(results_json, "r") as f:
        data = json.load(f)

    # -------------------------------------------------------------------------
    # 1. Stage 1 Confusion Matrix (SSD vs TD, N=169)
    # -------------------------------------------------------------------------
    s1_cm = np.array([[73, 4], [11, 81]])
    s1_classes = ["Speech Disorder", "Typically Developing"]
    plot_cm(s1_cm, s1_classes, "Stage 1: Screening Confusion Matrix (N=169)",
            os.path.join(output_dir, "stage1_confusion_matrix.png"))

    # -------------------------------------------------------------------------
    # 2. Stage 2 Confusion Matrix (Best Model: Exp C SMOTE Only)
    # -------------------------------------------------------------------------
    s2_best_cm = np.array(data["stage2_disorder_type"]["experiments"]["Exp_C_SMOTE_Only"]["confusion_matrix"])
    s2_classes = ["Articulation", "Phonological"]
    plot_cm(s2_best_cm, s2_classes, "Stage 2: Disorder Type Confusion Matrix (SMOTE)",
            os.path.join(output_dir, "stage2_confusion_matrix.png"))

    # -------------------------------------------------------------------------
    # 3. Stage 3 Confusion Matrix (Exp D Augmentation + SMOTE)
    # -------------------------------------------------------------------------
    s3_cm = np.array(data["stage3_symptom"]["experiments"]["Exp_D_Augmentation_SMOTE"]["confusion_matrix"])
    s3_classes = ["Substitution", "Stuttering", "Omission", "Addition"]
    plot_cm(s3_cm, s3_classes, "Stage 3: Symptom Confusion Matrix (Aug + SMOTE)",
            os.path.join(output_dir, "stage3_confusion_matrix.png"))

    # -------------------------------------------------------------------------
    # 4. Macro-F1 Comparison Graph across Exp A, B, C, D
    # -------------------------------------------------------------------------
    exp_labels = ["Exp A\nBaseline", "Exp B\nAug Only", "Exp C\nSMOTE Only", "Exp D\nAug+SMOTE"]
    exp_keys = ["Exp_A_Baseline", "Exp_B_Augmentation_Only", "Exp_C_SMOTE_Only", "Exp_D_Augmentation_SMOTE"]

    s2_f1s = [data["stage2_disorder_type"]["experiments"][k]["f1_macro"] * 100 for k in exp_keys]
    s3_f1s = [data["stage3_symptom"]["experiments"][k]["f1_macro"] * 100 for k in exp_keys]

    x = np.arange(len(exp_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    rects1 = ax.bar(x - width/2, s2_f1s, width, label='Stage 2: Disorder Type', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, s3_f1s, width, label='Stage 3: Symptom Type', color='#d95f02')

    ax.set_ylabel('Macro F1-Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Controlled Experimental Comparison: Macro-F1 across 4 Conditions', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=10, fontweight='bold')
    ax.legend(frameon=True, fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    macro_f1_path = os.path.join(output_dir, "macro_f1_comparison.png")
    plt.savefig(macro_f1_path, dpi=300)
    plt.close()
    print(f"Saved: {macro_f1_path}")

    # -------------------------------------------------------------------------
    # 5. Per-Class Recall Graph for Stage 3 (Minority Tracking)
    # -------------------------------------------------------------------------
    s3_classes_display = ["Substitution", "Stuttering", "Omission", "Addition (Minority)"]
    raw_s3_classes = ["substitution", "stuttering", "omission", "addition"]

    recalls_by_exp = []
    for k in exp_keys:
        cr = data["stage3_symptom"]["experiments"][k]["classification_report"]
        recs = [cr[c]["recall"] * 100 for c in raw_s3_classes]
        recalls_by_exp.append(recs)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_c = np.arange(len(s3_classes_display))
    bar_w = 0.18

    colors = ['#7570b3', '#1b9e77', '#d95f02', '#e7298a']
    for i, (k_name, col) in enumerate(zip(exp_labels, colors)):
        vals = recalls_by_exp[i]
        offset = (i - 1.5) * bar_w
        bars = ax.bar(x_c + offset, vals, bar_w, label=k_name.replace('\n', ' - '), color=col)
        for b in bars:
            h = b.get_height()
            ax.annotate(f'{h:.1f}%',
                        xy=(b.get_x() + b.get_width()/2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
    ax.set_title('Stage 3 Per-Class Recall across Experimental Conditions', fontsize=13, fontweight='bold')
    ax.set_xticks(x_c)
    ax.set_xticklabels(s3_classes_display, fontsize=11, fontweight='bold')
    ax.legend(frameon=True, fontsize=10)
    ax.set_ylim(0, 70)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig.tight_layout()
    per_class_path = os.path.join(output_dir, "per_class_recall_comparison.png")
    plt.savefig(per_class_path, dpi=300)
    plt.close()
    print(f"Saved: {per_class_path}")


if __name__ == "__main__":
    main()
