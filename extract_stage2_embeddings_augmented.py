"""
extract_stage2_embeddings_augmented.py
--------------------------------------
Extracts and caches frozen WavLM-Large embeddings for Stage 2 (Disorder Symptoms)
with GENDER-STRATIFIED PITCH AUGMENTATION applied strictly to the TRAINING set.

Key Requirements Followed:
1. TRAIN SET ONLY is augmented. Validation and Test splits are NEVER augmented.
2. Augmentation happens strictly AFTER the speaker-disjoint split (64% train, 16% val, 20% test, seed 42).
3. Gender-stratified pitch shift:
   - Male ('M')   -> shifted UP in [+6, +8] semitones
   - Female ('F') -> shifted DOWN in [-8, -6] semitones
   - Extra female variants generated to address the severe 3:1 gender imbalance in training.
4. Audio duration (12.0s max, 16kHz) and sample rate strictly preserved.
5. All original training samples are retained alongside their augmented versions.
6. Original cached embeddings (wavlm_stage2_disorder_symptom.pt) are untouched.
   New cache saved to: cached_embeddings/wavlm_stage2_disorder_symptom_augmented.pt.
7. Prints comprehensive class and gender distributions BEFORE and AFTER augmentation.
8. Streaming memory-efficient extraction (low_cpu_mem_usage=True) prevents memory spikes.
"""

import os
import re
import io
import gc
import sys
import random
import collections
import numpy as np
import pandas as pd
import soundfile as sf
import soxr
import torch
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.model_selection import train_test_split as sklearn_split

from speech_pitch_augmentation import pitch_shift_audio


def print_distribution_report(title: str, labels_list, genders_list, class_names):
    print("=" * 70, flush=True)
    print(f" {title.upper()}", flush=True)
    print("=" * 70, flush=True)
    total_samples = len(labels_list)
    print(f"Total Samples: {total_samples}", flush=True)

    # Gender counts
    gender_counts = collections.Counter(genders_list)
    print(f"Gender Distribution: Male={gender_counts.get('M', 0)} ({gender_counts.get('M', 0)/max(total_samples, 1)*100:.1f}%), "
          f"Female={gender_counts.get('F', 0)} ({gender_counts.get('F', 0)/max(total_samples, 1)*100:.1f}%)", flush=True)

    # Class counts
    class_counts = collections.Counter([class_names[lbl] if isinstance(lbl, int) else lbl for lbl in labels_list])
    print("\nPer-Class Distribution:", flush=True)
    for c in class_names:
        cnt = class_counts.get(c, 0)
        print(f"  - {c:<15}: {cnt:>4} samples ({cnt/max(total_samples, 1)*100:>5.1f}%)", flush=True)

    # Class x Gender Matrix
    print("\nClass x Gender Matrix:", flush=True)
    print(f"  {'Class':<15} {'Male':>8} {'Female':>8} {'Total':>8} {'% Female':>10}", flush=True)
    print("  " + "-" * 54, flush=True)
    for idx, c in enumerate(class_names):
        m_cnt = sum(1 for l, g in zip(labels_list, genders_list) if (l == idx or l == c) and g == 'M')
        f_cnt = sum(1 for l, g in zip(labels_list, genders_list) if (l == idx or l == c) and g == 'F')
        c_tot = m_cnt + f_cnt
        pct_f = (f_cnt / max(c_tot, 1)) * 100
        print(f"  {c:<15} {m_cnt:>8} {f_cnt:>8} {c_tot:>8} {pct_f:>9.1f}%", flush=True)
    print("=" * 70 + "\n", flush=True)


def extract_stage2_embeddings_augmented():
    print("=" * 70, flush=True)
    print("STEP 1: Loading Dataset and Preparing Disjoint Speaker Splits", flush=True)
    print("=" * 70, flush=True)

    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    dataset_name = "SAA-Lab/SLPHelmUltraSuitePlus"
    model_checkpoint = "microsoft/wavlm-large"
    output_dir = "./cached_embeddings"
    os.makedirs(output_dir, exist_ok=True)

    augmented_output_path = os.path.join(output_dir, "wavlm_stage2_disorder_symptom_augmented.pt")
    original_cache_path   = os.path.join(output_dir, "wavlm_stage2_disorder_symptom.pt")

    dataset = load_dataset(dataset_name, split="train")
    print(f"Total samples loaded from HuggingFace: {len(dataset)}", flush=True)

    # Cast to raw audio bytes (decode=False)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    # Extract Speaker IDs from filenames matching reference protocol
    paths = [item["audio"]["path"] for item in dataset]
    pattern = re.compile(r"^(?:test_|train_|val_|audio_)?([a-zA-Z0-9]+)[_\-\.]", re.IGNORECASE)
    speaker_ids = []
    for p in paths:
        filename = os.path.basename(p)
        match = pattern.search(filename)
        speaker_ids.append(match.group(1) if match else filename)
    dataset = dataset.add_column("speaker_id", speaker_ids)

    # Clean binary labels
    def clean_binary(batch):
        val = batch["disorder_class"]
        if val is not None:
            val = str(val).replace("['", "").replace("']", "").replace("[", "").replace("]", "").strip()
        return {"disorder_class": val}
    dataset = dataset.map(clean_binary)
    dataset = dataset.filter(lambda x: x["disorder_class"] is not None)

    # Clean and remap disorder symptoms
    def clean_symptom(batch):
        val = batch["disorder_symptom"]
        if val is not None:
            val = str(val).replace("['", "").replace("']", "").replace("[", "").replace("]", "").strip()
            if val in ['substitution_modified', 'phonological']:
                val = 'stuttering'
        return {"clean_symptom": val}
    dataset = dataset.map(clean_symptom)

    # Speaker disjoint split (64% train, 16% val, 20% test, fixed seed 42)
    df = dataset.to_pandas()
    unique_speakers = sorted(df["speaker_id"].unique())
    print(f"Derived {len(unique_speakers)} unique speaker IDs.", flush=True)

    train_val_spk, test_spk = sklearn_split(unique_speakers, test_size=0.2, random_state=42)
    train_spk, val_spk = sklearn_split(train_val_spk, test_size=0.2, random_state=42)

    assert set(train_spk).isdisjoint(set(val_spk)), "Leakage: Train <-> Val"
    assert set(train_spk).isdisjoint(set(test_spk)), "Leakage: Train <-> Test"
    assert set(val_spk).isdisjoint(set(test_spk)), "Leakage: Val <-> Test"

    ds_train = dataset.filter(lambda x: x["speaker_id"] in train_spk)
    ds_val   = dataset.filter(lambda x: x["speaker_id"] in val_spk)
    ds_test  = dataset.filter(lambda x: x["speaker_id"] in test_spk)

    print(f"Overall Split Sizes -> Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}", flush=True)

    # Age normalization strictly from training split bounds
    train_ages = np.array([float(x) for x in ds_train["age"]])
    age_min = float(train_ages.min())
    age_max = float(train_ages.max())
    print(f"Training age normalization bounds: [{age_min:.2f}, {age_max:.2f}]", flush=True)

    def normalize_age(a):
        return max(0.0, min(1.0, (float(a) - age_min) / (age_max - age_min)))

    # Stage 2 classes: 4 disorder symptoms
    disorder_classes = ["substitution", "stuttering", "omission", "addition"]
    symptom_label2id = {c: i for i, c in enumerate(disorder_classes)}
    symptom_id2label = {i: c for i, c in enumerate(disorder_classes)}

    ds_train_stage2 = ds_train.filter(lambda x: x["disorder_class"] == "speech_disorder" and x["clean_symptom"] in disorder_classes)
    ds_val_stage2   = ds_val.filter(lambda x: x["disorder_class"] == "speech_disorder" and x["clean_symptom"] in disorder_classes)
    ds_test_stage2  = ds_test.filter(lambda x: x["disorder_class"] == "speech_disorder" and x["clean_symptom"] in disorder_classes)

    print("\nStage 2 Disordered Sample Counts (Original Unaugmented):", flush=True)
    print(f"  Train: {len(ds_train_stage2)} samples", flush=True)
    print(f"  Val:   {len(ds_val_stage2)} samples", flush=True)
    print(f"  Test:  {len(ds_test_stage2)} samples", flush=True)

    target_sr = 16000
    max_duration = 12.0
    max_length = int(target_sr * max_duration)

    def load_audio_array(sample_audio_dict):
        raw_bytes = sample_audio_dict.get("bytes")
        if raw_bytes:
            wav, sr = sf.read(io.BytesIO(raw_bytes))
        else:
            path = sample_audio_dict["path"]
            wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1) # Convert to mono
        if sr != target_sr:
            wav = soxr.resample(wav, sr, target_sr)
        return wav.astype(np.float32)

    # -------------------------------------------------------------------------
    # PRINT BEFORE AUGMENTATION DISTRIBUTION
    # -------------------------------------------------------------------------
    orig_train_labels = [symptom_label2id[s["clean_symptom"]] for s in ds_train_stage2]
    orig_train_genders = [s["gender"] for s in ds_train_stage2]
    print_distribution_report(
        title="Training Set Distribution BEFORE Augmentation",
        labels_list=orig_train_labels,
        genders_list=orig_train_genders,
        class_names=disorder_classes
    )

    # -------------------------------------------------------------------------
    # LOAD FROZEN WAVLM-LARGE BACKBONE EARLY WITH low_cpu_mem_usage=True
    # -------------------------------------------------------------------------
    print("=" * 70, flush=True)
    print("STEP 2: Initializing Frozen WavLM Backbone (low_cpu_mem_usage=True)", flush=True)
    print("=" * 70, flush=True)

    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}", flush=True)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    wavlm = AutoModel.from_pretrained(
        model_checkpoint,
        use_safetensors=True,
        low_cpu_mem_usage=True
    )
    wavlm.eval()
    wavlm.to(device)
    for param in wavlm.parameters():
        param.requires_grad = False
    print("WavLM-Large backbone loaded and frozen successfully.", flush=True)

    # -------------------------------------------------------------------------
    # STREAMING GENDER-STRATIFIED PITCH AUGMENTATION & FEATURE EXTRACTION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("STEP 3: Streaming Pitch Augmentation & Embedding Extraction (TRAIN ONLY)", flush=True)
    print("=" * 70, flush=True)

    aug_rng = np.random.RandomState(42)

    # Track augmented distribution metadata
    aug_labels_list = []
    aug_genders_list = []

    all_train_embeddings = []
    all_train_ages = []
    all_train_labels = []
    all_train_symptoms = []
    all_train_genders = []
    all_train_spk_ids = []

    def extract_batch(audio_batch, age_batch, lbl_batch, sym_batch, gen_batch, spk_batch):
        inputs = feature_extractor(
            audio_batch,
            sampling_rate=target_sr,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        input_values = inputs["input_values"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = wavlm(input_values=input_values, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            feature_mask = wavlm._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask
            )
            mask = feature_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        all_train_embeddings.append(pooled.cpu())
        all_train_ages.extend(age_batch)
        all_train_labels.extend(lbl_batch)
        all_train_symptoms.extend(sym_batch)
        all_train_genders.extend(gen_batch)
        all_train_spk_ids.extend(spk_batch)

    # Mini-batch accumulator
    batch_audio, batch_age, batch_lbl, batch_sym, batch_gen, batch_spk = [], [], [], [], [], []
    batch_size = 4

    total_orig = len(ds_train_stage2)
    processed_orig = 0
    total_generated = 0

    for idx, sample in enumerate(ds_train_stage2):
        audio_orig = load_audio_array(sample["audio"])
        age_norm = normalize_age(sample["age"])
        symptom_name = sample["clean_symptom"]
        label_id = symptom_label2id[symptom_name]
        gender = sample["gender"]
        speaker_id = sample["speaker_id"]

        if gender not in ("M", "F"):
            raise ValueError(f"Sample index {idx} has invalid gender '{gender}'. Missing genders must not be imputed.")

        # Samples to extract for this child: (waveform, variant_label)
        sample_variants = []

        # 1. ALWAYS retain original sample
        sample_variants.append(audio_orig)
        aug_labels_list.append(label_id)
        aug_genders_list.append(gender)

        # 2. Gender-Stratified Pitch Augmentation
        if gender == "M":
            # Male: shift pitch UP by [+6.0, +8.0] semitones
            shift_m = aug_rng.uniform(6.0, 8.0)
            sample_variants.append(pitch_shift_audio(audio_orig, target_sr, shift_semitones=shift_m))
            aug_labels_list.append(label_id)
            aug_genders_list.append(gender)
        elif gender == "F":
            # Female: shift pitch DOWN by [-8.0, -6.0] semitones
            # Generate 2 variations to address 3:1 gender imbalance
            shift_f1 = -aug_rng.uniform(6.0, 7.0)
            shift_f2 = -aug_rng.uniform(7.0, 8.0)
            sample_variants.append(pitch_shift_audio(audio_orig, target_sr, shift_semitones=shift_f1))
            sample_variants.append(pitch_shift_audio(audio_orig, target_sr, shift_semitones=shift_f2))
            aug_labels_list.extend([label_id, label_id])
            aug_genders_list.extend([gender, gender])

        for var_audio in sample_variants:
            batch_audio.append(var_audio)
            batch_age.append(age_norm)
            batch_lbl.append(label_id)
            batch_sym.append(symptom_name)
            batch_gen.append(gender)
            batch_spk.append(speaker_id)
            total_generated += 1

            if len(batch_audio) >= batch_size:
                extract_batch(batch_audio, batch_age, batch_lbl, batch_sym, batch_gen, batch_spk)
                batch_audio, batch_age, batch_lbl, batch_sym, batch_gen, batch_spk = [], [], [], [], [], []

        processed_orig += 1
        if processed_orig % 25 == 0 or processed_orig == total_orig:
            print(f"  Processed {processed_orig}/{total_orig} original samples -> {total_generated} augmented samples extracted", flush=True)
            gc.collect()

    # Flush remaining in batch
    if len(batch_audio) > 0:
        extract_batch(batch_audio, batch_age, batch_lbl, batch_sym, batch_gen, batch_spk)
        batch_audio, batch_age, batch_lbl, batch_sym, batch_gen, batch_spk = [], [], [], [], [], []

    train_augmented_embeddings = torch.cat(all_train_embeddings, dim=0)
    train_augmented_ages = torch.tensor(all_train_ages, dtype=torch.float32).view(-1, 1)
    train_augmented_labels = torch.tensor(all_train_labels, dtype=torch.long)

    print(f"\nFinal Augmented Train Split: Embeddings {train_augmented_embeddings.shape}, Ages {train_augmented_ages.shape}, Labels {train_augmented_labels.shape}", flush=True)

    # -------------------------------------------------------------------------
    # PRINT AFTER AUGMENTATION DISTRIBUTION
    # -------------------------------------------------------------------------
    print_distribution_report(
        title="Training Set Distribution AFTER Gender-Stratified Augmentation",
        labels_list=aug_labels_list,
        genders_list=aug_genders_list,
        class_names=disorder_classes
    )

    train_augmented_data = {
        "embeddings": train_augmented_embeddings,
        "ages": train_augmented_ages,
        "labels": train_augmented_labels,
        "symptom_names": all_train_symptoms,
        "genders": all_train_genders,
        "speaker_ids": all_train_spk_ids
    }

    # -------------------------------------------------------------------------
    # VALIDATION AND TEST: VERIFY UNTOUCHED STATUS
    # -------------------------------------------------------------------------
    print("=" * 70, flush=True)
    print("STEP 4: Verifying UNTOUCHED Validation and Test Sets", flush=True)
    print("=" * 70, flush=True)

    if os.path.exists(original_cache_path):
        print(f"Loading reference untouched validation and test sets from: {original_cache_path}", flush=True)
        orig_cache = torch.load(original_cache_path, map_location="cpu", weights_only=False)
        val_data  = orig_cache["val"]
        test_data = orig_cache["test"]
        print(f"  Untouched Val  size: {len(val_data['labels'])} samples (identical to baseline)", flush=True)
        print(f"  Untouched Test size: {len(test_data['labels'])} samples (identical to baseline)", flush=True)
    else:
        raise FileNotFoundError(f"Reference cache {original_cache_path} not found. Baseline must be present.")

    # -------------------------------------------------------------------------
    # SAVE CACHE
    # -------------------------------------------------------------------------
    augmented_cache = {
        "train": train_augmented_data,
        "val": val_data,
        "test": test_data,
        "disorder_classes": disorder_classes,
        "symptom_label2id": symptom_label2id,
        "symptom_id2label": symptom_id2label,
        "age_min": age_min,
        "age_max": age_max,
        "augmentation_strategy": "gender_stratified_pitch_shift",
        "pitch_shift_bounds": {"male_up": [6.0, 8.0], "female_down": [-8.0, -6.0]},
        "female_oversample_variants": 2,
        "male_variants": 1
    }

    torch.save(augmented_cache, augmented_output_path)
    print("\n" + "=" * 70, flush=True)
    print(f"[SUCCESS] Saved Augmented Stage 2 Cache to: {augmented_output_path}", flush=True)
    print(f"File size: {os.path.getsize(augmented_output_path) / (1024 * 1024):.2f} MB", flush=True)
    print(f"Original cache at {original_cache_path} remains completely UNTOUCHED.", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    extract_stage2_embeddings_augmented()
