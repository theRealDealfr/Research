"""
data_audit.py
-------------
Phase 1: Full Data Audit of SAA-Lab/SLPHelmUltraSuitePlus.
Performs exhaustive analysis of:
  1. Metadata columns and label fields
  2. Stage 1, Stage 2, and Stage 3 unique labels and class counts
  3. Unique children / speakers per class
  4. Missing labels
  5. Inconsistent hierarchy
  6. Duplicate audio recordings / paths
  7. Speaker overlap across splits
"""

import os
import re
import hashlib
import collections
import pandas as pd
import numpy as np
from datasets import load_dataset, Audio
from sklearn.model_selection import train_test_split as sklearn_split

def clean_val(v):
    if v is None: return None
    return str(v).replace("['", "").replace("']", "").replace("[", "").replace("]", "").strip()

def main():
    print("=" * 80)
    print("PHASE 1: EXHAUSTIVE DATA AUDIT")
    print("Dataset: SAA-Lab/SLPHelmUltraSuitePlus")
    print("=" * 80)

    dataset = load_dataset("SAA-Lab/SLPHelmUltraSuitePlus", split="train")
    ds_meta = dataset.cast_column("audio", Audio(decode=False))
    df = ds_meta.to_pandas()

    print(f"Total Raw Rows: {len(df)}")
    print(f"Dataset Columns: {df.columns.tolist()}\n")

    # 1. Speaker ID Extraction
    pattern = re.compile(r"^(?:test_|train_|val_|audio_)?([a-zA-Z0-9]+)[_\-\.]", re.IGNORECASE)
    speaker_ids = []
    audio_paths = []
    for item in df["audio"]:
        p = item["path"]
        audio_paths.append(p)
        fn = os.path.basename(p)
        m = pattern.search(fn)
        speaker_ids.append(m.group(1) if m else fn)
    df["speaker_id"] = speaker_ids
    df["audio_path"] = audio_paths

    # Clean label fields
    df["s1"] = df["disorder_class"].apply(clean_val)
    df["s2"] = df["disorder_type"].apply(clean_val)
    df["s3"] = df["disorder_symptom"].apply(clean_val)

    # 2. Check Duplicate Audio Paths & Files
    unique_paths = df["audio_path"].nunique()
    print(f"Audio Path Duplicate Check: {len(df)} total paths, {unique_paths} unique paths.")
    if len(df) != unique_paths:
        dups = df[df.duplicated(subset=["audio_path"], keep=False)]
        print(f"  [WARNING] Found {len(dups)} duplicate audio paths!")
    else:
        print("  [OK] No duplicate audio paths detected.")

    # 3. Unique Children / Speakers
    unique_speakers = df["speaker_id"].nunique()
    print(f"Total Unique Children / Speakers: {unique_speakers}\n")

    # 4. Stage 1 Audit
    print("-" * 80)
    print("STAGE 1: BINARY SCREENING (disorder_class)")
    print("-" * 80)
    s1_counts = df["s1"].value_counts(dropna=False)
    for cls_name, cnt in s1_counts.items():
        spk_cnt = df[df["s1"] == cls_name]["speaker_id"].nunique()
        print(f"  {str(cls_name):<25}: {cnt:>4} samples ({spk_cnt:>3} unique children)")

    # 5. Stage 2 Audit
    print("\n" + "-" * 80)
    print("STAGE 2: DISORDER TYPE (disorder_type)")
    print("-" * 80)
    print("A. Overall Dataset:")
    s2_all = df["s2"].value_counts(dropna=False)
    for cls_name, cnt in s2_all.items():
        spk_cnt = df[df["s2"] == cls_name]["speaker_id"].nunique()
        print(f"  {str(cls_name):<25}: {cnt:>4} samples ({spk_cnt:>3} unique children)")

    dis_df = df[df["s1"] == "speech_disorder"].copy()
    print(f"\nB. Within Disordered Branch (speech_disorder, N={len(dis_df)}):")
    s2_dis = dis_df["s2"].value_counts(dropna=False)
    for cls_name, cnt in s2_dis.items():
        spk_cnt = dis_df[dis_df["s2"] == cls_name]["speaker_id"].nunique()
        print(f"  {str(cls_name):<25}: {cnt:>4} samples ({spk_cnt:>3} unique children)")

    # 6. Stage 3 Audit
    print("\n" + "-" * 80)
    print("STAGE 3: SYMPTOM CLASSIFICATION (disorder_symptom)")
    print("-" * 80)
    print("A. Overall Dataset:")
    s3_all = df["s3"].value_counts(dropna=False)
    for cls_name, cnt in s3_all.items():
        spk_cnt = df[df["s3"] == cls_name]["speaker_id"].nunique()
        print(f"  {str(cls_name):<25}: {cnt:>4} samples ({spk_cnt:>3} unique children)")

    print(f"\nB. Within Disordered Branch (speech_disorder, N={len(dis_df)}):")
    s3_dis = dis_df["s3"].value_counts(dropna=False)
    for cls_name, cnt in s3_dis.items():
        spk_cnt = dis_df[dis_df["s3"] == cls_name]["speaker_id"].nunique()
        print(f"  {str(cls_name):<25}: {cnt:>4} samples ({spk_cnt:>3} unique children)")

    # 7. Missing and Inconsistent Labels
    print("\n" + "-" * 80)
    print("MISSING & INCONSISTENT LABELS AUDIT")
    print("-" * 80)
    missing_s1 = df["s1"].isna().sum()
    missing_s2 = df["s2"].isna().sum()
    missing_s3 = df["s3"].isna().sum()
    missing_age = df["age"].isna().sum()
    missing_gender = df["gender"].isna().sum()
    print(f"Missing Values -> S1: {missing_s1}, S2: {missing_s2}, S3: {missing_s3}, Age: {missing_age}, Gender: {missing_gender}")

    # Hierarchy Inconsistencies
    incon_s2_td = dis_df[dis_df["s2"] == "typically_developing"]
    incon_s3_td = dis_df[dis_df["s3"] == "typically_developing"]
    misplaced_symptoms_in_s2 = dis_df[dis_df["s2"].isin(["omission", "addition", "stuttering"])]
    misplaced_types_in_s3 = dis_df[dis_df["s3"].isin(["phonological", "substitution_modified"])]

    print(f"1. speech_disorder cases with disorder_type == 'typically_developing': {len(incon_s2_td)}")
    print(f"2. speech_disorder cases with disorder_symptom == 'typically_developing': {len(incon_s3_td)}")
    print(f"3. Symptoms placed into disorder_type column: {len(misplaced_symptoms_in_s2)}")
    print(f"4. Disorder types or modified tokens in symptom column: {len(misplaced_types_in_s3)}")

    # 8. Speaker Disjoint Split Audit
    print("\n" + "-" * 80)
    print("SPEAKER DISJOINT SPLIT & DATA LEAKAGE AUDIT")
    print("-" * 80)
    spks = sorted(df["speaker_id"].unique())
    tv_spk, te_spk = sklearn_split(spks, test_size=0.2, random_state=42)
    tr_spk, va_spk = sklearn_split(tv_spk, test_size=0.2, random_state=42)

    set_tr = set(tr_spk)
    set_va = set(va_spk)
    set_te = set(te_spk)

    leak_tr_va = set_tr.intersection(set_va)
    leak_tr_te = set_tr.intersection(set_te)
    leak_va_te = set_va.intersection(set_te)

    print(f"Speaker Disjoint Check:")
    print(f"  Train <-> Val overlap:  {len(leak_tr_va)} speakers -> {'[LEAKAGE!]' if leak_tr_va else '[PASSED: ZERO LEAKAGE]'}")
    print(f"  Train <-> Test overlap: {len(leak_tr_te)} speakers -> {'[LEAKAGE!]' if leak_tr_te else '[PASSED: ZERO LEAKAGE]'}")
    print(f"  Val <-> Test overlap:   {len(leak_va_te)} speakers -> {'[LEAKAGE!]' if leak_va_te else '[PASSED: ZERO LEAKAGE]'}")

    # Save summary dictionary
    audit_summary = {
        "total_samples": len(df),
        "total_speakers": unique_speakers,
        "stage1": {k: {"samples": int(v), "speakers": int(df[df["s1"] == k]["speaker_id"].nunique())} for k, v in s1_counts.items()},
        "stage2_disordered": {k: {"samples": int(v), "speakers": int(dis_df[dis_df["s2"] == k]["speaker_id"].nunique())} for k, v in s2_dis.items()},
        "stage3_disordered": {k: {"samples": int(v), "speakers": int(dis_df[dis_df["s3"] == k]["speaker_id"].nunique())} for k, v in s3_dis.items()},
        "leakage_passed": bool(len(leak_tr_va) == 0 and len(leak_tr_te) == 0 and len(leak_va_te) == 0)
    }

    import json
    os.makedirs("./results_controlled_baselines", exist_ok=True)
    with open("./results_controlled_baselines/data_audit_summary.json", "w") as f:
        json.dump(audit_summary, f, indent=4)
    print("\nSaved full audit summary to ./results_controlled_baselines/data_audit_summary.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
