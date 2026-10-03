import os
import re
import pandas as pd
from datasets import load_dataset, Audio

def main():
    print("=" * 70)
    print("INSPECTING HIERARCHICAL LABELS IN SAA-Lab/SLPHelmUltraSuitePlus")
    print("=" * 70)

    dataset = load_dataset("SAA-Lab/SLPHelmUltraSuitePlus", split="train")
    ds_meta = dataset.cast_column("audio", Audio(decode=False))
    df = ds_meta.to_pandas()

    pattern = re.compile(r"^(?:test_|train_|val_|audio_)?([a-zA-Z0-9]+)[_\-\.]", re.IGNORECASE)
    speaker_ids = []
    for item in df["audio"]:
        fn = os.path.basename(item["path"])
        m = pattern.search(fn)
        speaker_ids.append(m.group(1) if m else fn)
    df["speaker_id"] = speaker_ids

    def clean_val(v):
        if v is None:
            return None
        return str(v).replace("['", "").replace("']", "").replace("[", "").replace("]", "").strip()

    df["stage1"] = df["disorder_class"].apply(clean_val)
    df["stage2"] = df["disorder_type"].apply(clean_val)
    df["stage3"] = df["disorder_symptom"].apply(clean_val)

    total_rows = len(df)
    unique_speakers = df["speaker_id"].nunique()
    print(f"Total samples: {total_rows}")
    print(f"Total unique children / speakers: {unique_speakers}\n")

    print("--- Stage 1 (Screening: disorder_class) ---")
    s1_counts = df["stage1"].value_counts()
    for label, count in s1_counts.items():
        spk_cnt = df[df["stage1"] == label]["speaker_id"].nunique()
        print(f"  {label:<25}: {count:>4} samples ({spk_cnt:>3} speakers)")

    dis_df = df[df["stage1"] == "speech_disorder"]
    td_df = df[df["stage1"] == "typically_developing"]

    print(f"\n--- Stage 2 (Disorder Type: disorder_type) among speech_disorder (N={len(dis_df)}) ---")
    s2_counts = dis_df["stage2"].value_counts()
    for label, count in s2_counts.items():
        spk_cnt = dis_df[dis_df["stage2"] == label]["speaker_id"].nunique()
        print(f"  {label:<25}: {count:>4} samples ({spk_cnt:>3} speakers)")

    print(f"\n--- Stage 3 (Speech Symptom: disorder_symptom) among speech_disorder (N={len(dis_df)}) ---")
    s3_counts = dis_df["stage3"].value_counts()
    for label, count in s3_counts.items():
        spk_cnt = dis_df[dis_df["stage3"] == label]["speaker_id"].nunique()
        print(f"  {label:<25}: {count:>4} samples ({spk_cnt:>3} speakers)")

    print("\n--- Full Cross-Tabulation: Stage 2 vs Stage 3 (within speech_disorder) ---")
    ct = pd.crosstab(dis_df["stage2"], dis_df["stage3"], margins=True)
    print(ct.to_string())

    print("\n--- Label Quality & Inconsistencies ---")
    # 1. Samples where disorder_class is speech_disorder but disorder_type or symptom is typically_developing
    s2_td_in_disorder = dis_df[dis_df["stage2"] == "typically_developing"]
    s3_td_in_disorder = dis_df[dis_df["stage3"] == "typically_developing"]
    print(f"1. speech_disorder samples labeled disorder_type == 'typically_developing': {len(s2_td_in_disorder)}")
    print(f"2. speech_disorder samples labeled disorder_symptom == 'typically_developing': {len(s3_td_in_disorder)}")

    # 2. Symptoms labeled as disorder types
    s2_symptoms = dis_df[dis_df["stage2"].isin(["omission", "addition", "stuttering"])]
    print(f"3. Samples where symptom names were placed in disorder_type column: {len(s2_symptoms)}")
    for idx, r in s2_symptoms.iterrows():
        print(f"   Sample {r['audio']['path']} -> s1={r['stage1']}, s2={r['stage2']}, s3={r['stage3']}")

    # 3. Disorder types labeled as symptoms
    s3_types = dis_df[dis_df["stage3"].isin(["phonological", "substitution_modified"])]
    print(f"4. Samples with irregular symptom labels ('phonological', 'substitution_modified'): {len(s3_types)}")
    for idx, r in s3_types.iterrows():
        print(f"   Sample {r['audio']['path']} -> s1={r['stage1']}, s2={r['stage2']}, s3={r['stage3']}")

    # 4. Check typically_developing group
    td_anom = td_df[(td_df["stage2"] != "typically_developing") | (td_df["stage3"] != "typically_developing")]
    print(f"5. typically_developing samples with non-TD disorder_type or symptom: {len(td_anom)}")

    # 5. Clean, consistent 3-stage hierarchy count
    # Valid disorder types: articulation, phonological
    # Valid symptoms: substitution, stuttering, omission, addition
    clean_disorder = dis_df[
        dis_df["stage2"].isin(["articulation", "phonological"]) &
        dis_df["stage3"].isin(["substitution", "stuttering", "omission", "addition"])
    ]
    print(f"\nSamples with completely clean and valid 3-stage disorder hierarchy: {len(clean_disorder)} / {len(dis_df)}")

    # Print sample hierarchy paths
    print("\n--- Examples of Hierarchical Mapping (Sample Paths) ---")
    for s_id in ["typically_developing", "articulation", "phonological"]:
        sub = df[df["stage2"] == s_id].head(2)
        for _, row in sub.iterrows():
            print(f"Speaker: {row['speaker_id']:<10} | Stage 1: {row['stage1']:<20} | Stage 2: {row['stage2']:<15} | Stage 3: {row['stage3']}")

    from sklearn.model_selection import train_test_split as sklearn_split
    unique_speakers = sorted(df["speaker_id"].unique())
    train_val_spk, test_spk = sklearn_split(unique_speakers, test_size=0.2, random_state=42)
    train_spk, val_spk = sklearn_split(train_val_spk, test_size=0.2, random_state=42)

    print("\n" + "=" * 70)
    print("SPLIT DISTRIBUTIONS (SPEAKER-DISJOINT)")
    print("=" * 70)
    for name, spk_set in [("TRAIN", train_spk), ("VAL", val_spk), ("TEST", test_spk)]:
        sub = df[df["speaker_id"].isin(spk_set)]
        dis = sub[sub["stage1"] == "speech_disorder"]
        td  = sub[sub["stage1"] == "typically_developing"]
        print(f"\n--- {name} SPLIT (Total={len(sub)}, Speakers={len(spk_set)}) ---")
        print(f"Stage 1 -> typically_developing: {len(td)}, speech_disorder: {len(dis)}")
        print("Stage 2 (within speech_disorder):")
        for k, v in dis["stage2"].value_counts().items():
            print(f"   {k}: {v}")
        print("Stage 3 (within speech_disorder):")
        for k, v in dis["stage3"].value_counts().items():
            print(f"   {k}: {v}")

if __name__ == "__main__":
    main()
