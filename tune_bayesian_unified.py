"""
tune_bayesian_unified.py
Bayesian Hyperparameter Tuning for T2 (disorder_type) and T3 (disorder_symptom).
Optimizes:
  - learning_rate (log-uniform: 1e-5 to 5e-4)
  - gradient_accumulation ([1, 2, 4, 8])
  - oversample_mult ([1 to 8])

Optimizes for Validation Macro F1 on the validation split (strictly preserving test set isolation).
Once the best hyperparameters are identified, optionally runs final fine-tuning on the full dataset
and evaluates on the held-out test split.
"""

import argparse
import copy
import json
import os
import random
import sys
import numpy as np
import torch

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoModel
import train_unified_models
from train_unified_models import run_unified_training, str2bool, get_parser


def parse_tuning_args():
    parser = argparse.ArgumentParser(description="Bayesian Hyperparameter Tuning for T2 and T3")
    
    parser.add_argument("--target_column", type=str, required=True, 
                        choices=["disorder_type", "disorder_symptom"],
                        help="The column to classify (disorder_type=T2, disorder_symptom=T3)")
    parser.add_argument("--model_architecture", type=str, default="C",
                        choices=["A", "B", "C"],
                        help="Model architecture: A(Frozen WavLM), B(WavLM+Age), C(WavLM+AAFF, default)")
    parser.add_argument("--model_checkpoint", type=str, default="microsoft/wavlm-large")
    
    # Bayesian Optimization Settings
    parser.add_argument("--n_trials", type=int, default=15, 
                        help="Number of Bayesian optimization trials to run")
    parser.add_argument("--trial_epochs", type=int, default=5, 
                        help="Epochs per tuning trial (use fewer epochs for fast exploration)")
    parser.add_argument("--final_epochs", type=int, default=12, 
                        help="Epochs for final fine-tuning run with best hyperparameters")
    parser.add_argument("--run_final_training", type=str2bool, default=True, 
                        help="Automatically run final training with best hyperparameters after search")
    
    # Hyperparameter Search Bounds
    parser.add_argument("--lr_min", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--lr_max", type=float, default=5e-4, help="Maximum learning rate")
    parser.add_argument("--oversample_min", type=int, default=1, help="Min oversample multiplier")
    parser.add_argument("--oversample_max", type=int, default=8, help="Max oversample multiplier")
    parser.add_argument("--grad_accum_choices", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Choices for gradient accumulation steps")
    
    # Common Training Options
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--exclude_healthy", type=str2bool, default=True)
    parser.add_argument("--aug_male", type=str2bool, default=True)
    parser.add_argument("--aug_female", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--wandb_mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_project", type=str, default="SLP-Bayesian-Tuning")
    parser.add_argument("--output_json", type=str, default=None,
                        help="File to save optimal parameters (default: best_params_<task>_<model>.json)")
    
    return parser.parse_args()


def run_bayesian_study(args):
    task_name = "T2 (disorder_type)" if args.target_column == "disorder_type" else "T3 (disorder_symptom)"
    print("=" * 75)
    print(f"BAYESIAN HYPERPARAMETER OPTIMIZATION: {task_name}")
    print(f"Architecture: Model {args.model_architecture} | Checkpoint: {args.model_checkpoint}")
    print(f"Trials: {args.n_trials} | Trial Epochs: {args.trial_epochs} | Final Epochs: {args.final_epochs}")
    print("=" * 75)

    if not OPTUNA_AVAILABLE:
        print("[Error] Optuna is required for Bayesian optimization. Please install with: pip install optuna")
        sys.exit(1)

    # 1. Preload Backbone and Feature Extractor once to reuse across all trials
    print("\n[Optimization] Preloading backbone and feature extractor once for all trials...")
    DATASET_NAME = "SAA-Lab/SLPHelmUltraSuitePlus"
    dataset = load_dataset(DATASET_NAME, split="train")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_checkpoint)
    backbone = AutoModel.from_pretrained(args.model_checkpoint, use_safetensors=True)
    for param in backbone.parameters():
        param.requires_grad = False
    print("Preloading complete. Ready for Bayesian search.\n")

    trial_records = []

    def objective(trial):
        lr = trial.suggest_float("learning_rate", args.lr_min, args.lr_max, log=True)
        grad_accum = trial.suggest_categorical("gradient_accumulation", args.grad_accum_choices)
        oversample = trial.suggest_int("oversample_mult", args.oversample_min, args.oversample_max)

        trial_num = trial.number + 1
        print("\n" + "-" * 60)
        print(f"TRIAL {trial_num}/{args.n_trials}")
        print(f"  Suggested LR:             {lr:.2e}")
        print(f"  Suggested Grad Accum:     {grad_accum} (Effective Batch Size: {args.batch_size * grad_accum})")
        print(f"  Suggested Oversample Mult:{oversample}x")
        print("-" * 60)

        trial_args = copy.deepcopy(args)
        trial_args.learning_rate = lr
        trial_args.gradient_accumulation = grad_accum
        trial_args.oversample_mult = oversample
        trial_args.epochs = args.trial_epochs
        trial_args.tune_mode = True  # Val only, strictly no test evaluation
        trial_args.doTest = False
        trial_args.check_only = False
        trial_args.random_seed = args.random_seed + trial.number

        try:
            val_results = run_unified_training(
                args=trial_args,
                raw_dataset=dataset,
                preloaded_backbone=backbone,
                preloaded_feature_extractor=feature_extractor
            )
            score = val_results["val_f1_macro"]
            val_acc = val_results.get("val_accuracy", 0.0)
        except Exception as e:
            print(f"[Warning] Trial {trial_num} failed with error: {e}. Reporting score 0.0.")
            score = 0.0
            val_acc = 0.0

        trial_records.append({
            "trial": trial_num,
            "learning_rate": lr,
            "gradient_accumulation": grad_accum,
            "oversample_mult": oversample,
            "val_f1_macro": score,
            "val_accuracy": val_acc
        })

        return score

    # Create Optuna study with TPE (Tree-structured Parzen Estimator) Bayesian Sampler
    sampler = TPESampler(seed=args.random_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    # Add the current baseline default as the initial trial for benchmarking
    study.enqueue_trial({
        "learning_rate": 3e-5,
        "gradient_accumulation": 2,
        "oversample_mult": 5
    })

    print("Starting Bayesian search with TPE Sampler...")
    study.optimize(objective, n_trials=args.n_trials)

    best_trial = study.best_trial
    print("\n" + "=" * 75)
    print(f"BAYESIAN OPTIMIZATION COMPLETED FOR {task_name}")
    print("=" * 75)
    print(f"Best Trial #{best_trial.number + 1}:")
    print(f"  Best Val Macro F1:        {best_trial.value:.4f}")
    print(f"  Optimal Learning Rate:    {best_trial.params['learning_rate']:.2e}")
    print(f"  Optimal Grad Accum:       {best_trial.params['gradient_accumulation']}")
    print(f"  Optimal Oversample Mult:  {best_trial.params['oversample_mult']}")
    print("=" * 75)

    # Save optimal parameters to JSON
    default_output_name = f"best_hyperparams_{args.target_column}_Model{args.model_architecture}.json"
    output_path = args.output_json if args.output_json else default_output_name
    
    best_config = {
        "target_column": args.target_column,
        "model_architecture": args.model_architecture,
        "learning_rate": best_trial.params["learning_rate"],
        "gradient_accumulation": best_trial.params["gradient_accumulation"],
        "oversample_mult": best_trial.params["oversample_mult"],
        "best_val_f1_macro": best_trial.value,
        "trial_epochs": args.trial_epochs,
        "n_trials": args.n_trials,
        "trials_summary": trial_records
    }

    with open(output_path, "w") as f:
        json.dump(best_config, f, indent=4)
    print(f"\n[Saved] Optimal configuration saved to: {output_path}")

    # Print summary table of top 5 trials
    print("\n--- TOP TRIALS RANKED BY VAL MACRO F1 ---")
    sorted_trials = sorted(trial_records, key=lambda x: x["val_f1_macro"], reverse=True)
    print(f"{'Trial':<8} {'Val F1':<12} {'Val Acc':<12} {'LR':<12} {'GradAccum':<12} {'Oversample':<12}")
    print("-" * 70)
    for tr in sorted_trials[:min(5, len(sorted_trials))]:
        print(f"{tr['trial']:<8} {tr['val_f1_macro']:<12.4f} {tr['val_accuracy']:<12.4f} {tr['learning_rate']:<12.2e} {tr['gradient_accumulation']:<12} {tr['oversample_mult']:<12}")

    # Optional: Trigger final fine-tuning with optimal parameters
    if args.run_final_training:
        print("\n" + "=" * 75)
        print("STAGE 2: RUNNING FINAL FINE-TUNING WITH OPTIMAL HYPERPARAMETERS")
        print("=" * 75)
        final_args = copy.deepcopy(args)
        final_args.learning_rate = best_trial.params["learning_rate"]
        final_args.gradient_accumulation = best_trial.params["gradient_accumulation"]
        final_args.oversample_mult = best_trial.params["oversample_mult"]
        final_args.epochs = args.final_epochs
        final_args.tune_mode = False
        final_args.doTest = True
        final_args.check_only = False
        final_args.config_json = output_path

        final_metrics = run_unified_training(
            args=final_args,
            raw_dataset=dataset,
            preloaded_backbone=backbone,
            preloaded_feature_extractor=feature_extractor
        )
        print("\n" + "=" * 75)
        print("FINAL FINE-TUNING COMPLETED!")
        print(f"Results Directory: {final_metrics.get('output_dir')}")
        print(f"Final Val Macro F1:  {final_metrics.get('val_f1_macro', 0.0):.4f}")
        print(f"Final Test Acc:      {final_metrics.get('test_accuracy', 0.0):.4f}")
        print(f"Final Test Macro F1: {final_metrics.get('test_f1_macro', 0.0):.4f}")
        print("=" * 75)


if __name__ == "__main__":
    tuning_args = parse_tuning_args()
    run_bayesian_study(tuning_args)
