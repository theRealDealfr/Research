# Pediatric Speech-Language Pathology AI: Research & Diagnostic System

This repository implements the research pipeline for pediatric Speech Sound Disorder (SSD) assessment using **Age-Aware Feature Fusion (AAFF)**, **Counterfactual Data Augmentation (CDA)**, **Bayesian Hyperparameter Optimization**, and **Hierarchical Gated Inference** on the `SAA-Lab/SLPHelmUltraSuitePlus` dataset.

Based on the research paper: *"Multimodal LLMs are not all you need for Pediatric Speech Language Pathology"* (Fürst, Steindl, Schäfer, arXiv:2604.26568).

---

## 📖 Detailed Documentation

For a comprehensive explanation of all steps, architectural decisions, and techniques taken to improve classification accuracy and demographic fairness, see:

👉 **[Comprehensive Classification Accuracy Report](CLASSIFICATION_ACCURACY_REPORT.md)**

---

## 🚀 Quick Start & CLI Utilities

### 1. View Comparative Performance Metrics
```powershell
python compare_all_pipelines.py
```

### 2. Run the Full Augmented Hierarchical Pipeline
```powershell
.\run_pipeline_chain_augmented.ps1
```

### 3. Launch Clinical Decision Support Web Interface
```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Navigate to `http://127.0.0.1:8000` in your web browser.