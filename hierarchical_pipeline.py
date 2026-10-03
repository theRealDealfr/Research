"""
hierarchical_pipeline.py
------------------------
Implements the final end-to-end Hierarchical Speech Classifier:

Flow:
  Child speech recording + age
            ↓
  Stage 1: Typical vs Disorder (Binary Screening)
            ↓
  If Typical:
      Final Output: Typical
  If Disorder:
      ├── Stage 2: Disorder Type (Articulation vs Phonological Disorder)
      └── Stage 3: Symptom (Substitution, Stuttering, Omission, Addition)

Provides:
  1. `HierarchicalClassifier`: unified model wrapper with `predict(emb, age)`.
  2. `format_hierarchical_result(result)`: standard clinical report formatting.
  3. `evaluate_hierarchical_test_set()`: end-to-end evaluation on the full test set.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import numpy as np
import pandas as pd
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# -----------------------------------------------------------------------------
# Model Architectures
# -----------------------------------------------------------------------------
class AAFFModel(nn.Module):
    def __init__(self, in_features=1025, proj_dim=256, dropout=0.2, num_classes=2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, emb, age):
        fused = torch.cat([emb, age], dim=-1)
        return self.classifier(self.projection(fused))


# -----------------------------------------------------------------------------
# Hierarchical Classifier
# -----------------------------------------------------------------------------
class HierarchicalClassifier:
    def __init__(
        self,
        stage1_path="./results_controlled_baselines/model_c_aaff_best.pt",
        stage2_path="./results_controlled_baselines/stage2_disorder_type_best_model.pt",
        stage3_path="./results_controlled_baselines/stage3_smote_model.pt",
        device=None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Class Label Definitions
        self.stage1_classes = ["speech_disorder", "typically_developing"]
        self.stage2_classes = ["articulation", "phonological"]
        self.stage3_classes = ["substitution", "stuttering", "omission", "addition"]

        # Human-Friendly Display Names
        self.type_display = {
            "articulation": "Articulation Disorder",
            "phonological": "Phonological Disorder"
        }
        self.symptom_display = {
            "substitution": "Substitution",
            "stuttering": "Stuttering",
            "omission": "Omission",
            "addition": "Addition"
        }

        # Initialize and Load Models
        self.s1_model = AAFFModel(in_features=1025, proj_dim=256, dropout=0.2, num_classes=2).to(self.device)
        self.s1_model.load_state_dict(torch.load(stage1_path, map_location=self.device, weights_only=True))
        self.s1_model.eval()

        self.s2_model = AAFFModel(in_features=1025, proj_dim=256, dropout=0.2, num_classes=2).to(self.device)
        self.s2_model.load_state_dict(torch.load(stage2_path, map_location=self.device, weights_only=True))
        self.s2_model.eval()

        # If stage3_smote_model.pt exists, use it; otherwise fallback to stage3_symptom_best_model.pt
        actual_s3_path = stage3_path if os.path.exists(stage3_path) else "./results_controlled_baselines/stage3_symptom_best_model.pt"
        self.s3_model = AAFFModel(in_features=1025, proj_dim=256, dropout=0.2, num_classes=4).to(self.device)
        self.s3_model.load_state_dict(torch.load(actual_s3_path, map_location=self.device, weights_only=True))
        self.s3_model.eval()

    def predict(self, emb_tensor, age_tensor):
        """
        Runs hierarchical prediction on a single audio feature + age input.
        Inputs:
          emb_tensor: torch.Tensor of shape (1, 1024) or (1024,)
          age_tensor: torch.Tensor of shape (1, 1) or (1,) or float
        """
        if not isinstance(emb_tensor, torch.Tensor):
            emb_tensor = torch.tensor(emb_tensor, dtype=torch.float32)
        if not isinstance(age_tensor, torch.Tensor):
            age_tensor = torch.tensor(age_tensor, dtype=torch.float32)

        if emb_tensor.ndim == 1:
            emb_tensor = emb_tensor.unsqueeze(0)
        if age_tensor.ndim == 1:
            age_tensor = age_tensor.unsqueeze(0)
        if age_tensor.ndim == 0:
            age_tensor = age_tensor.view(1, 1)

        emb_tensor = emb_tensor.to(self.device)
        age_tensor = age_tensor.to(self.device)

        with torch.no_grad():
            # Stage 1: Screening
            s1_logits = self.s1_model(emb_tensor, age_tensor)
            s1_probs = torch.softmax(s1_logits, dim=-1)[0].cpu().numpy()
            p_disorder = float(s1_probs[0])
            p_typical  = float(s1_probs[1])

            if p_typical >= p_disorder:
                return {
                    "screening": "Typical",
                    "disorder_type": None,
                    "symptom": None,
                    "confidences": {
                        "screening": p_typical,
                        "disorder_type": None,
                        "symptom": None,
                        "hierarchical_overall": p_typical
                    },
                    "raw_predictions": {
                        "stage1": "typically_developing",
                        "stage2": None,
                        "stage3": None
                    }
                }
            else:
                # Stage 2: Disorder Type
                s2_logits = self.s2_model(emb_tensor, age_tensor)
                s2_probs = torch.softmax(s2_logits, dim=-1)[0].cpu().numpy()
                s2_idx = int(np.argmax(s2_probs))
                s2_raw = self.stage2_classes[s2_idx]
                p_s2 = float(s2_probs[s2_idx])

                # Stage 3: Symptom
                s3_logits = self.s3_model(emb_tensor, age_tensor)
                s3_probs = torch.softmax(s3_logits, dim=-1)[0].cpu().numpy()
                s3_idx = int(np.argmax(s3_probs))
                s3_raw = self.stage3_classes[s3_idx]
                p_s3 = float(s3_probs[s3_idx])

                # Hierarchical joint confidence: P(Disorder) * P(Type | Disorder)
                hierarchical_conf = p_disorder * p_s2

                return {
                    "screening": "Disorder",
                    "disorder_type": self.type_display.get(s2_raw, s2_raw),
                    "symptom": self.symptom_display.get(s3_raw, s3_raw),
                    "confidences": {
                        "screening": p_disorder,
                        "disorder_type": p_s2,
                        "symptom": p_s3,
                        "hierarchical_overall": hierarchical_conf
                    },
                    "raw_predictions": {
                        "stage1": "speech_disorder",
                        "stage2": s2_raw,
                        "stage3": s3_raw
                    }
                }


def format_hierarchical_result(result):
    """Formats the hierarchical result dictionary into clean console text."""
    if result["screening"] == "Typical":
        return (
            "Stage 1:\n"
            f"  Typical (Confidence: {result['confidences']['screening']*100:.1f}%)\n\n"
            "Final Result:\n"
            "  Typical (No Speech Disorder Detected)"
        )
    else:
        return (
            "Stage 1:\n"
            f"  Disorder (Confidence: {result['confidences']['screening']*100:.1f}%)\n"
            "Stage 2:\n"
            f"  {result['disorder_type']} (Confidence: {result['confidences']['disorder_type']*100:.1f}%)\n"
            "Stage 3:\n"
            f"  {result['symptom']} (Confidence: {result['confidences']['symptom']*100:.1f}%)\n\n"
            "Final Result:\n"
            "  Disorder detected\n"
            f"  Disorder type: {result['disorder_type']}\n"
            f"  Symptom: {result['symptom']}\n"
            f"  Hierarchical Confidence: {result['confidences']['hierarchical_overall']*100:.1f}%"
        )


# -----------------------------------------------------------------------------
# End-to-End Evaluation on Full Untouched Test Set with Paper-Style Matrix
# -----------------------------------------------------------------------------
def plot_confusion_matrix_png(cm, class_names, title, output_paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    for p in output_paths:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        plt.savefig(p, dpi=300)
    plt.close()


def evaluate_hierarchical_test_set():
    test_cache_path = "./cached_embeddings/wavlm_full_test_hierarchical.pt"
    if not os.path.exists(test_cache_path):
        raise FileNotFoundError(f"Missing {test_cache_path}")

    cache = torch.load(test_cache_path, map_location="cpu", weights_only=False)
    embeddings = cache["embeddings"]
    ages = cache["ages"]
    binary_labels = cache["binary_labels"].numpy()  # 0: SSD, 1: TD
    gt_symptoms = cache["symptoms"]

    classifier = HierarchicalClassifier()

    total_samples = len(embeddings)
    print("=" * 70)
    print("END-TO-END HIERARCHICAL PIPELINE EVALUATION ON UNTOUCHED TEST SET")
    print("=" * 70)
    print(f"Total test samples: {total_samples} (Speech Disorder: 77, Typically Developing: 92)")

    # Data collection containers
    # 1. Stage 1 (Screening)
    s1_preds, s1_targets = [], []

    # 2. Standalone Stage 2 & 3 (Directly evaluated on ground-truth branch)
    s2_standalone_preds, s2_standalone_targets = [], []
    s3_standalone_preds, s3_standalone_targets = [], []

    # 3. Cascaded Disordered Branch (Evaluated on samples that passed Stage 1 screening)
    s2_cascaded_preds, s2_cascaded_targets = [], []
    s3_cascaded_preds, s3_cascaded_targets = [], []

    # 4. End-to-End Hierarchical correctness
    s2_e2e_correct, s3_e2e_correct = [], []

    # Map dataset test dataframe to extract ground truth disorder types
    from datasets import load_dataset, Audio
    from sklearn.model_selection import train_test_split as sklearn_split
    import re

    ds = load_dataset("SAA-Lab/SLPHelmUltraSuitePlus", split="train").cast_column("audio", Audio(decode=False))
    df = ds.to_pandas()
    pat = re.compile(r"^(?:test_|train_|val_|audio_)?([a-zA-Z0-9]+)[_\-\.]", re.I)
    df["speaker_id"] = [pat.search(os.path.basename(p["path"])).group(1) if pat.search(os.path.basename(p["path"])) else os.path.basename(p["path"]) for p in df["audio"]]

    def clean_str(v):
        if v is None: return None
        return str(v).replace("['", "").replace("']", "").replace("[", "").replace("]", "").strip()

    df["s1"] = df["disorder_class"].apply(clean_str)
    df["s2"] = df["disorder_type"].apply(clean_str)
    df["s3"] = df["disorder_symptom"].apply(clean_str)
    df.loc[df["s3"].isin(["substitution_modified", "phonological"]), "s3"] = "stuttering"

    spks = sorted(df["speaker_id"].unique())
    tv, te = sklearn_split(spks, test_size=0.2, random_state=42)
    df_test_full = df[df["speaker_id"].isin(te)].copy()
    gt_types = df_test_full["s2"].tolist()
    gt_symptoms_clean = df_test_full["s3"].tolist()

    demonstration_samples = []

    with torch.no_grad():
        for i in range(total_samples):
            emb = embeddings[i:i+1]
            age = ages[i:i+1]
            gt_b = "speech_disorder" if binary_labels[i] == 0 else "typically_developing"
            gt_t = gt_types[i] if i < len(gt_types) else None
            gt_s = gt_symptoms_clean[i] if i < len(gt_symptoms_clean) else None

            # Hierarchical cascaded prediction
            res = classifier.predict(emb, age)
            pred_b = res["raw_predictions"]["stage1"]
            pred_t_cascaded = res["raw_predictions"]["stage2"]
            pred_s_cascaded = res["raw_predictions"]["stage3"]

            s1_preds.append(pred_b)
            s1_targets.append(gt_b)

            # Direct standalone predictions
            s2_logits = classifier.s2_model(emb.to(classifier.device), age.to(classifier.device))
            s2_direct = classifier.stage2_classes[int(torch.argmax(s2_logits))]

            s3_logits = classifier.s3_model(emb.to(classifier.device), age.to(classifier.device))
            s3_direct = classifier.stage3_classes[int(torch.argmax(s3_logits))]

            # Disordered evaluation
            if gt_b == "speech_disorder":
                # Stage 2: Disorder Type
                if gt_t in classifier.stage2_classes:
                    s2_standalone_preds.append(s2_direct)
                    s2_standalone_targets.append(gt_t)
                    if pred_t_cascaded is not None:
                        s2_cascaded_preds.append(pred_t_cascaded)
                        s2_cascaded_targets.append(gt_t)
                    s2_e2e_correct.append(1 if (pred_b == "speech_disorder" and pred_t_cascaded == gt_t) else 0)

                # Stage 3: Symptom
                if gt_s in classifier.stage3_classes:
                    s3_standalone_preds.append(s3_direct)
                    s3_standalone_targets.append(gt_s)
                    if pred_s_cascaded is not None:
                        s3_cascaded_preds.append(pred_s_cascaded)
                        s3_cascaded_targets.append(gt_s)
                    s3_e2e_correct.append(1 if (pred_b == "speech_disorder" and pred_s_cascaded == gt_s) else 0)

            if i < 4:
                demonstration_samples.append((gt_b, gt_t, gt_s, res))

    # --- Compute Standard Metrics Helper ---
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

    def get_metrics_bundle(y_true, y_pred):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        }

    # 1. Standalone Metrics (Evaluated on ground-truth task branch)
    m_s1 = get_metrics_bundle(s1_targets, s1_preds)
    m_s2_standalone = get_metrics_bundle(s2_standalone_targets, s2_standalone_preds)
    m_s3_standalone = get_metrics_bundle(s3_standalone_targets, s3_standalone_preds)

    # 2. Cascaded Branch Metrics (Evaluated on samples that passed Stage 1 screening)
    m_s2_cascaded = get_metrics_bundle(s2_cascaded_targets, s2_cascaded_preds)
    m_s3_cascaded = get_metrics_bundle(s3_cascaded_targets, s3_cascaded_preds)

    # Confusion Matrices
    cm_s1 = confusion_matrix(s1_targets, s1_preds, labels=["speech_disorder", "typically_developing"])
    cm_s2 = confusion_matrix(s2_standalone_targets, s2_standalone_preds, labels=classifier.stage2_classes)
    cm_s3 = confusion_matrix(s3_standalone_targets, s3_standalone_preds, labels=classifier.stage3_classes)

    # Classification Reports with explicit labels to preserve canonical class ordering
    cr_s1 = classification_report(s1_targets, s1_preds, labels=["speech_disorder", "typically_developing"], target_names=["Speech Disorder", "Typically Developing"], digits=3, zero_division=0)
    cr_s2 = classification_report(s2_standalone_targets, s2_standalone_preds, labels=classifier.stage2_classes, target_names=["Articulation", "Phonological"], digits=3, zero_division=0)
    cr_s3 = classification_report(s3_standalone_targets, s3_standalone_preds, labels=classifier.stage3_classes, target_names=["Substitution", "Stuttering", "Omission", "Addition"], digits=3, zero_division=0)

    # Class-wise recall for Stage 3
    s3_cr_dict = classification_report(s3_standalone_targets, s3_standalone_preds, labels=classifier.stage3_classes, output_dict=True, zero_division=0)
    s3_class_recalls = {
        cls_name: s3_cr_dict[cls_name]["recall"]
        for cls_name in classifier.stage3_classes
    }
    lowest_recall_class = min(s3_class_recalls.items(), key=lambda kv: kv[1])

    # Deltas
    delta_f1_t2_t1 = m_s2_standalone["f1_macro"] - m_s1["f1_macro"]
    delta_f1_t3_t2 = m_s3_standalone["f1_macro"] - m_s2_standalone["f1_macro"]

    # =========================================================================
    # PRINT PAPER-STYLE EVALUATION MATRIX
    # =========================================================================
    matrix_str = f"""
====================================================================
              HIERARCHICAL CLASSIFICATION PERFORMANCE
              (Standalone Ground-Truth Branch Evaluation)
====================================================================
Metric                 T1             T2             T3
                       Screening      Disorder       Symptom
--------------------------------------------------------------------
Accuracy               {m_s1['accuracy']*100:>7.3f}%        {m_s2_standalone['accuracy']*100:>7.3f}%        {m_s3_standalone['accuracy']*100:>7.3f}%
Macro Precision        {m_s1['precision_macro']*100:>7.3f}%        {m_s2_standalone['precision_macro']*100:>7.3f}%        {m_s3_standalone['precision_macro']*100:>7.3f}%
Macro Recall           {m_s1['recall_macro']*100:>7.3f}%        {m_s2_standalone['recall_macro']*100:>7.3f}%        {m_s3_standalone['recall_macro']*100:>7.3f}%
Macro F1               {m_s1['f1_macro']*100:>7.3f}%        {m_s2_standalone['f1_macro']*100:>7.3f}%        {m_s3_standalone['f1_macro']*100:>7.3f}%
Weighted F1            {m_s1['f1_weighted']*100:>7.3f}%        {m_s2_standalone['f1_weighted']*100:>7.3f}%        {m_s3_standalone['f1_weighted']*100:>7.3f}%
====================================================================
Task Granularity Deltas (Macro F1):
  Delta F1(T2 - T1): {delta_f1_t2_t1*100:>+7.3f}%
  Delta F1(T3 - T2): {delta_f1_t3_t2*100:>+7.3f}%
  (Note: Deltas demonstrate task difficulty progression across increasingly
   fine-grained acoustic/phonemic label spaces; they do not imply model flaw.)
====================================================================

====================================================================
              CASCADED PIPELINE PERFORMANCE
              (Evaluated through Stage 1 Screening Decisions)
====================================================================
Metric                 T1 Screening   T2 Disordered  T3 Disordered
                       (Full N=169)   Branch (N={len(s2_cascaded_targets)})   Branch (N={len(s3_cascaded_targets)})
--------------------------------------------------------------------
Accuracy               {m_s1['accuracy']*100:>7.3f}%        {m_s2_cascaded['accuracy']*100:>7.3f}%        {m_s3_cascaded['accuracy']*100:>7.3f}%
Macro Precision        {m_s1['precision_macro']*100:>7.3f}%        {m_s2_cascaded['precision_macro']*100:>7.3f}%        {m_s3_cascaded['precision_macro']*100:>7.3f}%
Macro Recall           {m_s1['recall_macro']*100:>7.3f}%        {m_s2_cascaded['recall_macro']*100:>7.3f}%        {m_s3_cascaded['recall_macro']*100:>7.3f}%
Macro F1               {m_s1['f1_macro']*100:>7.3f}%        {m_s2_cascaded['f1_macro']*100:>7.3f}%        {m_s3_cascaded['f1_macro']*100:>7.3f}%
Weighted F1            {m_s1['f1_weighted']*100:>7.3f}%        {m_s2_cascaded['f1_weighted']*100:>7.3f}%        {m_s3_cascaded['f1_weighted']*100:>7.3f}%
--------------------------------------------------------------------
End-to-End Accuracy    {m_s1['accuracy']*100:>7.3f}%        {np.mean(s2_e2e_correct)*100:>7.3f}%        {np.mean(s3_e2e_correct)*100:>7.3f}%
(Counted correct only if Stage 1 screening AND downstream prediction match GT)
====================================================================
"""
    print(matrix_str)

    # Print Classification Reports
    print("\n" + "=" * 70)
    print("STAGE 1: BINARY SCREENING CLASSIFICATION REPORT")
    print("=" * 70)
    print(cr_s1)
    print("Confusion Matrix [Speech Disorder, Typically Developing]:\n", cm_s1)

    print("\n" + "=" * 70)
    print("STAGE 2: DISORDER TYPE CLASSIFICATION REPORT (STANDALONE, N=70)")
    print("=" * 70)
    print(cr_s2)
    print("Confusion Matrix ['articulation', 'phonological']:\n", cm_s2)

    print("\n" + "=" * 70)
    print("STAGE 3: SYMPTOM CLASSIFICATION REPORT (STANDALONE, N=71)")
    print("=" * 70)
    print(cr_s3)
    print("Confusion Matrix ['substitution', 'stuttering', 'omission', 'addition']:\n", cm_s3)

    print("\n" + "=" * 70)
    print("STAGE 3 CLASS-WISE RECALL ANALYSIS")
    print("=" * 70)
    for c_name, rec in s3_class_recalls.items():
        print(f"  {c_name.capitalize():<15}: {rec*100:.2f}%")
    print(f"\n>> Lowest Recall Class: {lowest_recall_class[0].upper()} ({lowest_recall_class[1]*100:.2f}%)")
    print("=" * 70)

    # Save Confusion Matrix Plots
    output_dir = "./results_controlled_baselines"
    os.makedirs(output_dir, exist_ok=True)

    plot_confusion_matrix_png(
        cm_s1, ["Speech Disorder", "Typically Developing"],
        "Stage 1: Screening Confusion Matrix (N=169)",
        [os.path.join(output_dir, "stage1_confusion_matrix.png"), "./stage1_confusion_matrix.png"]
    )
    plot_confusion_matrix_png(
        cm_s2, ["Articulation", "Phonological"],
        "Stage 2: Disorder Type Confusion Matrix (N=70)",
        [os.path.join(output_dir, "stage2_confusion_matrix.png"), "./stage2_confusion_matrix.png"]
    )
    plot_confusion_matrix_png(
        cm_s3, ["Substitution", "Stuttering", "Omission", "Addition"],
        "Stage 3: Symptom Confusion Matrix (N=71)",
        [os.path.join(output_dir, "stage3_confusion_matrix.png"), "./stage3_confusion_matrix.png"]
    )
    print("\nSaved confusion matrices to PNG.")

    # Save CSV Matrix
    csv_rows = [
        {"Metric": "Accuracy", "Stage 1 Screening": f"{m_s1['accuracy']*100:.3f}%", "Stage 2 Disorder": f"{m_s2_standalone['accuracy']*100:.3f}%", "Stage 3 Symptom": f"{m_s3_standalone['accuracy']*100:.3f}%"},
        {"Metric": "Macro Precision", "Stage 1 Screening": f"{m_s1['precision_macro']*100:.3f}%", "Stage 2 Disorder": f"{m_s2_standalone['precision_macro']*100:.3f}%", "Stage 3 Symptom": f"{m_s3_standalone['precision_macro']*100:.3f}%"},
        {"Metric": "Macro Recall", "Stage 1 Screening": f"{m_s1['recall_macro']*100:.3f}%", "Stage 2 Disorder": f"{m_s2_standalone['recall_macro']*100:.3f}%", "Stage 3 Symptom": f"{m_s3_standalone['recall_macro']*100:.3f}%"},
        {"Metric": "Macro F1", "Stage 1 Screening": f"{m_s1['f1_macro']*100:.3f}%", "Stage 2 Disorder": f"{m_s2_standalone['f1_macro']*100:.3f}%", "Stage 3 Symptom": f"{m_s3_standalone['f1_macro']*100:.3f}%"},
        {"Metric": "Weighted F1", "Stage 1 Screening": f"{m_s1['f1_weighted']*100:.3f}%", "Stage 2 Disorder": f"{m_s2_standalone['f1_weighted']*100:.3f}%", "Stage 3 Symptom": f"{m_s3_standalone['f1_weighted']*100:.3f}%"},
    ]
    df_matrix = pd.DataFrame(csv_rows)
    for p in [os.path.join(output_dir, "hierarchical_evaluation_matrix.csv"), "./hierarchical_evaluation_matrix.csv"]:
        df_matrix.to_csv(p, index=False)

    # Save TXT Matrix
    for p in [os.path.join(output_dir, "hierarchical_evaluation_matrix.txt"), "./hierarchical_evaluation_matrix.txt"]:
        with open(p, "w", encoding="utf-8") as f:
            f.write(matrix_str)

    # Save Classification Reports
    for p in [os.path.join(output_dir, "stage1_classification_report.txt"), "./stage1_classification_report.txt"]:
        with open(p, "w", encoding="utf-8") as f: f.write(cr_s1)
    for p in [os.path.join(output_dir, "stage2_classification_report.txt"), "./stage2_classification_report.txt"]:
        with open(p, "w", encoding="utf-8") as f: f.write(cr_s2)
    for p in [os.path.join(output_dir, "stage3_classification_report.txt"), "./stage3_classification_report.txt"]:
        with open(p, "w", encoding="utf-8") as f: f.write(cr_s3)

    print("Saved evaluation matrix CSV/TXT and classification reports.")

    # Demonstration of inference examples
    print("\n" + "=" * 70)
    print("DEMONSTRATION OF INFERENCE EXAMPLES")
    print("=" * 70)
    for idx, (gb, gt, gs, r) in enumerate(demonstration_samples):
        print(f"\nExample {idx + 1} (Ground Truth: Screening={gb}, Type={gt}, Symptom={gs}):")
        print("-" * 50)
        print(format_hierarchical_result(r))
    print("=" * 70)

    return {
        "stage1": m_s1,
        "stage2": m_s2_standalone,
        "stage3": m_s3_standalone,
        "deltas": {"t2_t1": delta_f1_t2_t1, "t3_t2": delta_f1_t3_t2}
    }


if __name__ == "__main__":
    evaluate_hierarchical_test_set()
