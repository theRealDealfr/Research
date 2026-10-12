import json
import matplotlib.pyplot as plt
import numpy as np

def generate_t2_plot():
    with open("results_controlled_baselines/t2_bayesian_augmented_comparison.json", "r") as f:
        data = json.load(f)

    labels = [
        "1. Unaugmented\nBaseline",
        "2. Augmented\nOnly",
        "3. Bayesian\nTuning Only",
        "4. Combined\nAug + Bayes"
    ]
    
    configs = list(data.keys())
    
    accuracy = [data[c]["metrics"]["accuracy"] * 100 for c in configs]
    macro_f1 = [data[c]["metrics"]["f1_macro"] * 100 for c in configs]
    phon_recall = [data[c]["metrics"]["recall_macro"] * 100 for c in configs]
    disparity = [data[c]["metrics"]["gender_breakdown"]["disparity"] * 100 for c in configs]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    
    rects1 = ax.bar(x - 1.5*width, accuracy, width, label="Test Accuracy (%)", color="#3b82f6")
    rects2 = ax.bar(x - 0.5*width, macro_f1, width, label="Macro F1 (%)", color="#10b981")
    rects3 = ax.bar(x + 0.5*width, phon_recall, width, label="Macro Recall (%)", color="#8b5cf6")
    rects4 = ax.bar(x + 1.5*width, disparity, width, label="Gender Disparity |M-F| (%)", color="#ef4444")

    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title("Task 2 (Disorder Type: Articulation vs Phonological)\nBaseline vs Augmentation vs Bayesian Tuning vs Combined", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Bar labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    add_labels(rects4)

    plt.tight_layout()
    plt.savefig("t2_metrics_comparison.png", dpi=300)
    print("Saved plot to t2_metrics_comparison.png")

if __name__ == "__main__":
    generate_t2_plot()
