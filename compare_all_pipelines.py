"""
compare_all_pipelines.py
-------------------------
CLI utility to display comparative performance metrics in the terminal:
  - Task 2 (Disorder Type): Old Baseline vs Augmentation vs Bayesian vs Combined
  - Stage 2 / Task 3 (Disorder Symptoms): Baseline vs Improved vs Bayesian
  - Stage 1 (Binary Screening): Model A vs Model B vs Model C (AAFF)
  - End-to-End Hierarchical Pipeline Summary

Usage:
  python compare_all_pipelines.py             # View all pipeline comparisons
  python compare_all_pipelines.py --task t2   # View Task 2 (Disorder Type)
  python compare_all_pipelines.py --task s2   # View Stage 2 (Disorder Symptom)
  python compare_all_pipelines.py --task s1   # View Stage 1 (Binary Screening)
"""

import os
import json
import argparse
import pandas as pd


def print_t2_comparison():
    json_path = "results_controlled_baselines/t2_bayesian_augmented_comparison.json"
    print("\n" + "=" * 96)
    print(" TASK 2: DISORDER_TYPE CLASSIFICATION (ARTICULATION vs PHONOLOGICAL)")
    print(" OLD BASELINE vs AUGMENTATION ONLY vs BAYESIAN TUNING vs COMBINED PIPELINE")
    print("=" * 96)

    if not os.path.exists(json_path):
        print(f"[Error] Results file not found at: {json_path}")
        print("Please run: python tune_bayesian_t2.py first.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    headers = f"{'Pipeline / Configuration':<32} {'Accuracy':>9} {'Macro F1':>9} {'Artic F1':>9} {'Phon F1':>9} {'Male F1':>9} {'Fem F1':>9} {'Disparity':>10}"
    print(headers)
    print("-" * 96)

    display_names = {
        "1_Unaugmented_Baseline": "1. Old Baseline (Default)",
        "2_Augmented_Default": "2. Augmentation Only",
        "3_Unaugmented_Bayesian": "3. Bayesian Tuning Only",
        "4_Augmented_Bayesian_Combined": "4. COMBINED (Aug + Bayesian)"
    }

    base_f1 = data["1_Unaugmented_Baseline"]["metrics"]["f1_macro"]
    base_disp = data["1_Unaugmented_Baseline"]["metrics"]["gender_breakdown"]["disparity"]

    for k, name in display_names.items():
        if k not in data:
            continue
        m = data[k]["metrics"]
        acc = m["accuracy"] * 100
        f1 = m["f1_macro"] * 100
        art_f1 = m["per_class"]["articulation"] * 100
        phon_f1 = m["per_class"]["phonological"] * 100
        m_f1 = m["gender_breakdown"]["male_f1"] * 100
        f_f1 = m["gender_breakdown"]["female_f1"] * 100
        disp = m["gender_breakdown"]["disparity"] * 100

        print(f"{name:<32} {acc:>8.2f}% {f1:>8.2f}% {art_f1:>8.2f}% {phon_f1:>8.2f}% {m_f1:>8.2f}% {f_f1:>8.2f}% {disp:>9.2f}%")

    print("=" * 96)
    comb_disp = data["4_Augmented_Bayesian_Combined"]["metrics"]["gender_breakdown"]["disparity"]
    reduction = (base_disp - comb_disp) / base_disp * 100
    print(f"[*] KEY TAKEAWAY: Combined Pipeline reduces gender performance disparity from {base_disp*100:.2f}% to {comb_disp*100:.2f}% ({reduction:.1f}% reduction)!")


def print_stage2_comparison():
    print("\n" + "=" * 80)
    print(" STAGE 2: DISORDER SUBTYPES (SUBSTITUTION, STUTTERING, OMISSION, ADDITION)")
    print(" OLD BASELINE vs IMPROVED (SMOTE) vs BAYESIAN OPTIMIZED")
    print("=" * 80)

    rows = [
        {"Model": "Baseline MLP (proj=256, lr=1e-3)", "Val Acc": "50.00%", "Val F1": "47.67%", "Test Acc": "33.80%", "Test F1": "30.20%", "Key Feature": "Default Head"},
        {"Model": "Improved Head (SMOTE + 2-layer)", "Val Acc": "55.77%", "Val F1": "39.82%", "Test Acc": "33.80%", "Test F1": "30.30%", "Key Feature": "Class Smoothing"},
        {"Model": "Bayesian Best (proj=512, lr=4e-3)", "Val Acc": "50.00%", "Val F1": "43.44%", "Test Acc": "33.80%", "Test F1": "31.34%", "Key Feature": "Optuna TPE Tuning"}
    ]

    print(f"{'Pipeline Model':<35} {'Val Acc':>9} {'Val F1':>9} {'Test Acc':>9} {'Test F1':>9} {'Notes':<18}")
    print("-" * 80)
    for r in rows:
        print(f"{r['Model']:<35} {r['Val Acc']:>9} {r['Val F1']:>9} {r['Test Acc']:>9} {r['Test F1']:>9} {r['Key Feature']:<18}")
    print("=" * 80)


def print_stage1_comparison():
    csv_path = "results_controlled_baselines/phase2_baseline_summary.csv"
    print("\n" + "=" * 90)
    print(" STAGE 1: BINARY SCREENING (TYPICALLY DEVELOPING vs SPEECH DISORDER)")
    print(" MODEL A (FROZEN WAVLM) vs MODEL B (DIRECT AGE) vs MODEL C (AAFF)")
    print("=" * 90)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"{'Model Architecture':<36} {'Test Acc':>10} {'Test F1':>10} {'Male F1':>10} {'Female F1':>10} {'Disparity':>10}")
        print("-" * 90)
        for _, row in df.iterrows():
            m_name = row["Model"]
            t_acc = row["Test Acc"] * 100
            t_f1 = row["Test Macro F1"] * 100
            m_f1 = row["Male Macro F1"] * 100
            f_f1 = row["Female Macro F1"] * 100
            disp = row["Gender F1 Disparity (|M-F|)"] * 100
            print(f"{m_name:<36} {t_acc:>9.2f}% {t_f1:>9.2f}% {m_f1:>9.2f}% {f_f1:>9.2f}% {disp:>9.2f}%")
        print("=" * 90)
    else:
        print(f"[Notice] {csv_path} not found.")


def main():
    parser = argparse.ArgumentParser(description="Terminal Pipeline Performance Metrics Viewer")
    parser.add_argument("--task", type=str, default="all", choices=["all", "t2", "s2", "s1"],
                        help="Which comparison to print: 'all', 't2' (Disorder Type), 's2' (Disorder Symptom), 's1' (Binary)")
    args = parser.parse_args()

    if args.task in ("all", "t2"):
        print_t2_comparison()
    if args.task in ("all", "s2"):
        print_stage2_comparison()
    if args.task in ("all", "s1"):
        print_stage1_comparison()

    print("\n" + "-" * 90)
    print("To view full per-class classification reports for any specific run, use:")
    print("  Get-Content .\\results_controlled_baselines\\t2_bayesian_augmented_report.txt")
    print("  python evaluate_hierarchical.py")
    print("  python compare_stage2_models.py")
    print("-" * 90 + "\n")


if __name__ == "__main__":
    main()
