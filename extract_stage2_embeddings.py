import os
import re
import io
import sys
import random
import numpy as np
import pandas as pd
import soundfile as sf
import soxr
import torch
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.model_selection import train_test_split as sklearn_split

def extract_stage2_embeddings():
    print("="*70, flush=True)
    print("STEP 1: Loading Dataset and Preparing Disjoint Speaker Splits", flush=True)
    print("="*70, flush=True)

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
    stage2_output_path = os.path.join(output_dir, "wavlm_stage2_disorder_symptom.pt")
    full_test_output_path = os.path.join(output_dir, "wavlm_full_test_hierarchical.pt")

    dataset = load_dataset(dataset_name, split="train")
    print(f"Total samples loaded from HuggingFace: {len(dataset)}", flush=True)

    # Cast to raw audio bytes (decode=False) to avoid external decoding dependencies
    dataset = dataset.cast_column("audio", Audio(decode=False))

    # Extract Speaker IDs from filenames
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

    print("\nStage 2 Disordered Sample Counts:", flush=True)
    print(f"  Train: {len(ds_train_stage2)} samples", flush=True)
    print(f"  Val:   {len(ds_val_stage2)} samples", flush=True)
    print(f"  Test:  {len(ds_test_stage2)} samples", flush=True)
    print(f"  Target Classes ({len(disorder_classes)}): {disorder_classes}", flush=True)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
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

    print("\n" + "="*70, flush=True)
    print("STEP 2: Initializing Frozen WavLM Backbone", flush=True)
    print("="*70, flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    wavlm = AutoModel.from_pretrained(model_checkpoint, use_safetensors=True)
    wavlm.eval()
    wavlm.to(device)
    for param in wavlm.parameters():
        param.requires_grad = False

    def extract_split(ds_split, split_name, batch_size=4):
        print(f"\nProcessing {split_name} ({len(ds_split)} samples)...", flush=True)
        all_embeddings = []
        all_ages = []
        all_labels = []
        all_symptom_names = []
        all_genders = []
        all_speaker_ids = []

        total = len(ds_split)
        for i in range(0, total, batch_size):
            batch_samples = [ds_split[j] for j in range(i, min(i + batch_size, total))]
            audio_arrays = [load_audio_array(s["audio"]) for s in batch_samples]
            ages = [normalize_age(s["age"]) for s in batch_samples]
            symptom_names = [s["clean_symptom"] for s in batch_samples]
            labels = [symptom_label2id[s] for s in symptom_names]
            genders = [s["gender"] for s in batch_samples]
            spk_ids = [s["speaker_id"] for s in batch_samples]

            inputs = feature_extractor(
                audio_arrays,
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

            all_embeddings.append(pooled.cpu())
            all_ages.extend(ages)
            all_labels.extend(labels)
            all_symptom_names.extend(symptom_names)
            all_genders.extend(genders)
            all_speaker_ids.extend(spk_ids)

            processed_count = min(i + batch_size, total)
            if (i // batch_size + 1) % 5 == 0 or processed_count >= total:
                print(f"  [{split_name}] Processed {processed_count}/{total} samples", flush=True)

        embeddings_tensor = torch.cat(all_embeddings, dim=0)
        ages_tensor = torch.tensor(all_ages, dtype=torch.float32).view(-1, 1)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)

        print(f"Finished {split_name}: Embeddings {embeddings_tensor.shape}, Ages {ages_tensor.shape}, Labels {labels_tensor.shape}", flush=True)
        return {
            "embeddings": embeddings_tensor,
            "ages": ages_tensor,
            "labels": labels_tensor,
            "symptom_names": all_symptom_names,
            "genders": all_genders,
            "speaker_ids": all_speaker_ids
        }

    train_data = extract_split(ds_train_stage2, "train", batch_size=4)
    val_data   = extract_split(ds_val_stage2, "val", batch_size=4)
    test_data  = extract_split(ds_test_stage2, "test", batch_size=4)

    cache_stage2 = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "disorder_classes": disorder_classes,
        "symptom_label2id": symptom_label2id,
        "symptom_id2label": symptom_id2label,
        "age_min": age_min,
        "age_max": age_max
    }

    torch.save(cache_stage2, stage2_output_path)
    print(f"\nSaved Stage 2 cached embeddings to: {stage2_output_path}", flush=True)

    # Step 3: Extract FULL Test set (both TD and SSD) for end-to-end Hierarchical Evaluation
    print("\n" + "="*70, flush=True)
    print("STEP 3: Extracting Full Test Set for End-to-End Hierarchical Evaluation", flush=True)
    print("="*70, flush=True)
    print(f"Full Test Set: {len(ds_test)} samples (TD: 92, SSD: 77)", flush=True)

    all_test_embeddings = []
    all_test_ages = []
    all_test_binary_labels = [] # 0: speech_disorder, 1: typically_developing
    all_test_symptoms = []
    all_test_genders = []
    all_test_speaker_ids = []

    total_test = len(ds_test)
    for i in range(0, total_test, 4):
        batch_samples = [ds_test[j] for j in range(i, min(i + 4, total_test))]
        audio_arrays = [load_audio_array(s["audio"]) for s in batch_samples]
        ages = [normalize_age(s["age"]) for s in batch_samples]
        binary_lbls = [0 if s["disorder_class"] == "speech_disorder" else 1 for s in batch_samples]
        symptoms = [s["clean_symptom"] if s["disorder_class"] == "speech_disorder" else "typically_developing" for s in batch_samples]

        inputs = feature_extractor(
            audio_arrays,
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
            feature_mask = wavlm._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            mask = feature_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        all_test_embeddings.append(pooled.cpu())
        all_test_ages.extend(ages)
        all_test_binary_labels.extend(binary_lbls)
        all_test_symptoms.extend(symptoms)
        all_test_genders.extend([s["gender"] for s in batch_samples])
        all_test_speaker_ids.extend([s["speaker_id"] for s in batch_samples])

        processed_count = min(i + 4, total_test)
        if (i // 4 + 1) % 5 == 0 or processed_count >= total_test:
            print(f"  [full_test] Processed {processed_count}/{total_test} samples", flush=True)

    cache_full_test = {
        "embeddings": torch.cat(all_test_embeddings, dim=0),
        "ages": torch.tensor(all_test_ages, dtype=torch.float32).view(-1, 1),
        "binary_labels": torch.tensor(all_test_binary_labels, dtype=torch.long),
        "symptoms": all_test_symptoms,
        "genders": all_test_genders,
        "speaker_ids": all_test_speaker_ids,
        "binary_class_names": ["speech_disorder", "typically_developing"],
        "disorder_classes": disorder_classes,
        "age_min": age_min,
        "age_max": age_max
    }

    torch.save(cache_full_test, full_test_output_path)
    print(f"Saved Full Hierarchical Test Cache to: {full_test_output_path}", flush=True)
    print("="*70, flush=True)
    print("ALL EMBEDDINGS SUCCESSFULLY EXTRACTED AND CACHED!", flush=True)
    print("="*70, flush=True)

if __name__ == "__main__":
    extract_stage2_embeddings()
