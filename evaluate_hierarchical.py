import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

class ModelC_AAFF(nn.Module):
    """Stage 1 Binary AAFF Classifier"""
    def __init__(self, in_features=1025, proj_dim=256, dropout=0.2, num_classes=2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, emb, age):
        fused = torch.cat([emb, age], dim=1)
        projected = self.projection(fused)
        return self.classifier(projected)

class AAFFDisorderClassifier(nn.Module):
    """Stage 2 Multiclass AAFF Disorder Classifier"""
    def __init__(self, in_features=1025, proj_dim=256, dropout=0.2, num_classes=4):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, emb, age):
        fused = torch.cat([emb, age], dim=1)
        projected = self.projection(fused)
        return self.classifier(projected)

def run_hierarchical_evaluation():
    test_cache_path = "./cached_embeddings/wavlm_full_test_hierarchical.pt"
    stage1_ckpt = "./results_controlled_baselines/model_c_aaff_best.pt"
    stage2_ckpt = "./results_controlled_baselines/model_stage2_disorder_best.pt"

    if not os.path.exists(test_cache_path):
        print(f"Error: {test_cache_path} not found. Please run extract_stage2_embeddings.py first.")
        return
    if not os.path.exists(stage1_ckpt):
        print(f"Error: Stage 1 checkpoint {stage1_ckpt} not found.")
        return
    if not os.path.exists(stage2_ckpt):
        print(f"Error: Stage 2 checkpoint {stage2_ckpt} not found. Please run train_stage2_disorder.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    # Load Models
    stage1_model = ModelC_AAFF(in_features=1025, proj_dim=256, dropout=0.2, num_classes=2).to(device)
    stage1_model.load_state_dict(torch.load(stage1_ckpt, map_location=device, weights_only=True))
    stage1_model.eval()

    stage2_model = AAFFDisorderClassifier(in_features=1025, proj_dim=256, dropout=0.2, num_classes=4).to(device)
    stage2_model.load_state_dict(torch.load(stage2_ckpt, map_location=device, weights_only=True))
    stage2_model.eval()

    # Load Full Test Cache
    test_cache = torch.load(test_cache_path, map_location="cpu", weights_only=False)
    embeddings = test_cache["embeddings"]
    ages = test_cache["ages"]
    binary_labels = test_cache["binary_labels"].numpy() # 0: SSD, 1: TD
    ground_truth_symptoms = test_cache["symptoms"]
    disorder_classes = test_cache["disorder_classes"] # ['substitution', 'stuttering', 'omission', 'addition']

    total_samples = len(embeddings)
    print(f"\nLoaded {total_samples} test samples for comprehensive hierarchical evaluation.")

    # Run Predictions
    stage1_preds = []
    stage1_probs = []
    stage2_preds = []
    stage2_probs = []
    final_hierarchical_preds = []
    final_hierarchical_confidences = []

    with torch.no_grad():
        for i in range(total_samples):
            emb = embeddings[i:i+1].to(device)
            age = ages[i:i+1].to(device)

            # Stage 1: Binary Decision
            s1_logits = stage1_model(emb, age)
            s1_prob = torch.softmax(s1_logits, dim=-1)[0].cpu().numpy()
            p_ssd = float(s1_prob[0])
            p_td  = float(s1_prob[1])

            s1_pred_class = "speech_disorder" if p_ssd >= p_td else "typically_developing"
            s1_conf = max(p_ssd, p_td)
            stage1_preds.append(s1_pred_class)
            stage1_probs.append(s1_prob)

            # Stage 2: Disorder Subtype (Conditional on SSD)
            s2_logits = stage2_model(emb, age)
            s2_prob = torch.softmax(s2_logits, dim=-1)[0].cpu().numpy()
            s2_idx = int(np.argmax(s2_prob))
            predicted_disorder = disorder_classes[s2_idx]
            s2_conf = float(s2_prob[s2_idx])
            stage2_preds.append(predicted_disorder)
            stage2_probs.append(s2_prob)

            # Hierarchical Routing Logic
            if s1_pred_class == "typically_developing":
                final_pred = "typically_developing"
                final_conf = p_td
            else:
                final_pred = predicted_disorder
                # Combined Hierarchical Confidence: P(SSD) * P(Disorder | SSD)
                final_conf = p_ssd * s2_conf

            final_hierarchical_preds.append(final_pred)
            final_hierarchical_confidences.append(final_conf)

    # --- EVALUATION 1: Stage 1 Binary Screening ---
    gt_binary = ["speech_disorder" if b == 0 else "typically_developing" for b in binary_labels]
    s1_acc = accuracy_score(gt_binary, stage1_preds)
    s1_p, s1_r, s1_f1, _ = precision_recall_fscore_support(gt_binary, stage1_preds, average="macro", zero_division=0)
    s1_cm = confusion_matrix(gt_binary, stage1_preds, labels=["speech_disorder", "typically_developing"])

    print("\n" + "="*70)
    print("STAGE 1: BINARY SCREENING EVALUATION (TD vs SSD)")
    print("="*70)
    print(f"Accuracy:        {s1_acc:.4f} ({s1_acc*100:.2f}%)")
    print(f"Macro Precision: {s1_p:.4f}")
    print(f"Macro Recall:    {s1_r:.4f}")
    print(f"Macro F1-Score:  {s1_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(gt_binary, stage1_preds, target_names=["speech_disorder", "typically_developing"], zero_division=0))
    print("Confusion Matrix [speech_disorder, typically_developing]:")
    print(s1_cm)

    # --- EVALUATION 2: Stage 2 Disorder Subtype (on disordered samples) ---
    ssd_mask = [b == 0 and gt_s in disorder_classes for b, gt_s in zip(binary_labels, ground_truth_symptoms)]
    ssd_indices = [i for i, m in enumerate(ssd_mask) if m]

    gt_s2_ssd = [ground_truth_symptoms[i] for i in ssd_indices]
    pred_s2_ssd = [stage2_preds[i] for i in ssd_indices]

    s2_acc = accuracy_score(gt_s2_ssd, pred_s2_ssd)
    s2_p, s2_r, s2_f1, _ = precision_recall_fscore_support(gt_s2_ssd, pred_s2_ssd, average="macro", zero_division=0)
    s2_cm = confusion_matrix(gt_s2_ssd, pred_s2_ssd, labels=disorder_classes)

    print("\n" + "="*70)
    print("STAGE 2: DISORDER CLASSIFICATION EVALUATION (ON DISORDERED TEST SAMPLES)")
    print("="*70)
    print(f"Samples Evaluated: {len(ssd_indices)}")
    print(f"Accuracy:          {s2_acc:.4f} ({s2_acc*100:.2f}%)")
    print(f"Macro Precision:   {s2_p:.4f}")
    print(f"Macro Recall:      {s2_r:.4f}")
    print(f"Macro F1-Score:    {s2_f1:.4f}")
    print("\nPer-Class Breakdown:")
    print(classification_report(gt_s2_ssd, pred_s2_ssd, target_names=disorder_classes, zero_division=0))
    print("Confusion Matrix:")
    print(s2_cm)
    print(f"Classes: {disorder_classes}")

    # --- EVALUATION 3: End-to-End Hierarchical Pipeline ---
    print("\n" + "="*70)
    print("END-TO-END HIERARCHICAL PIPELINE PERFORMANCE")
    print("="*70)
    all_eval_classes = ["typically_developing"] + disorder_classes
    e2e_valid_indices = [i for i, s in enumerate(ground_truth_symptoms) if s in all_eval_classes]
    gt_e2e = [ground_truth_symptoms[i] for i in e2e_valid_indices]
    pred_e2e = [final_hierarchical_preds[i] for i in e2e_valid_indices]

    e2e_acc = accuracy_score(gt_e2e, pred_e2e)
    e2e_p, e2e_r, e2e_f1, _ = precision_recall_fscore_support(gt_e2e, pred_e2e, average="macro", zero_division=0)
    e2e_cm = confusion_matrix(gt_e2e, pred_e2e, labels=all_eval_classes)

    print(f"Total Test Samples:    {len(e2e_valid_indices)}")
    print(f"Hierarchical Accuracy: {e2e_acc:.4f} ({e2e_acc*100:.2f}%)")
    print(f"Macro F1-Score:        {e2e_f1:.4f}")
    print("\nFull Hierarchical 5-Way Classification Report:")
    # Pass labels= to force sklearn to use our exact class order (matches target_names)
    print(classification_report(gt_e2e, pred_e2e, labels=all_eval_classes, target_names=all_eval_classes, zero_division=0))
    print("Hierarchical Confusion Matrix:")
    print(pd.DataFrame(e2e_cm, index=all_eval_classes, columns=all_eval_classes))
    print("="*70)

if __name__ == "__main__":
    run_hierarchical_evaluation()
