import os
import re
import io
import random
import numpy as np
import soundfile as sf
import soxr
import torch
from datasets import load_dataset, Audio, ClassLabel
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.model_selection import train_test_split as sklearn_split

def extract_embeddings():
    print("=== Step 1: Loading Dataset and Preparing Splits ===")
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
    output_path = os.path.join(output_dir, "wavlm_large_split_seed42.pt")

    dataset = load_dataset(dataset_name, split="train")
    print(f"Total samples loaded: {len(dataset)}")

    # Cast audio column to decode=False to prevent external decoding dependencies (torchcodec)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    paths = [item["audio"]["path"] for item in dataset]
    pattern = re.compile(r"^(?:test_|train_|val_|audio_)?([a-zA-Z0-9]+)[_\-\.]", re.IGNORECASE)
    speaker_ids = []
    for p in paths:
        filename = os.path.basename(p)
        match = pattern.search(filename)
        speaker_ids.append(match.group(1) if match else filename)
    dataset = dataset.add_column("speaker_id", speaker_ids)

    # Standard clean labels for target_column = disorder_class
    def clean_labels(batch):
        val = batch["disorder_class"]
        if val is None:
            return batch
        val = str(val).replace("['", "").replace("']", "").replace("[", "").replace("]", "")
        return {"disorder_class": val}

    dataset = dataset.map(clean_labels)
    dataset = dataset.filter(lambda x: x["disorder_class"] is not None)

    # Speaker disjoint split (64% train, 16% val, 20% test)
    df = dataset.to_pandas()
    unique_speakers = sorted(df["speaker_id"].unique())
    print(f"Unique speaker IDs derived from filenames: {len(unique_speakers)}")

    train_val_spk, test_spk = sklearn_split(unique_speakers, test_size=0.2, random_state=42)
    train_spk, val_spk = sklearn_split(train_val_spk, test_size=0.2, random_state=42)

    assert set(train_spk).isdisjoint(set(val_spk)), "Leakage: Train <-> Val"
    assert set(train_spk).isdisjoint(set(test_spk)), "Leakage: Train <-> Test"
    assert set(val_spk).isdisjoint(set(test_spk)), "Leakage: Val <-> Test"

    ds_train = dataset.filter(lambda x: x["speaker_id"] in train_spk)
    ds_val   = dataset.filter(lambda x: x["speaker_id"] in val_spk)
    ds_test  = dataset.filter(lambda x: x["speaker_id"] in test_spk)

    unique_labels = sorted(list(set(ds_train["disorder_class"]) | set(ds_val["disorder_class"]) | set(ds_test["disorder_class"])))
    label_feature = ClassLabel(names=unique_labels)
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}

    ds_train = ds_train.map(lambda x: {"labels": label_feature.str2int(x["disorder_class"])})
    ds_val   = ds_val.map(lambda x: {"labels": label_feature.str2int(x["disorder_class"])})
    ds_test  = ds_test.map(lambda x: {"labels": label_feature.str2int(x["disorder_class"])})

    print(f"Labels: {unique_labels} ({label2id})")
    print(f"Split counts -> Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}")

    # Age normalization strictly from training split
    train_ages = np.array([float(x) for x in ds_train["age"]])
    age_min = float(train_ages.min())
    age_max = float(train_ages.max())
    print(f"Training age range: {age_min:.2f} to {age_max:.2f}")

    def normalize_age(a):
        return (float(a) - age_min) / (age_max - age_min)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    target_sr = feature_extractor.sampling_rate
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

    print("\n=== Step 2: Loading WavLM Model ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    wavlm = AutoModel.from_pretrained(model_checkpoint, use_safetensors=True)
    wavlm.eval()
    wavlm.to(device)
    for param in wavlm.parameters():
        param.requires_grad = False

    def process_split(ds_split, split_name, batch_size=4):
        print(f"\nExtracting embeddings for {split_name} ({len(ds_split)} samples)...")
        all_embeddings = []
        all_ages = []
        all_labels = []
        all_genders = []
        all_speaker_ids = []

        total = len(ds_split)
        for i in range(0, total, batch_size):
            batch_samples = [ds_split[j] for j in range(i, min(i + batch_size, total))]
            audio_arrays = [load_audio_array(s["audio"]) for s in batch_samples]
            ages = [normalize_age(s["age"]) for s in batch_samples]
            labels = [s["labels"] for s in batch_samples]
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
            all_genders.extend(genders)
            all_speaker_ids.extend(spk_ids)

            if (i // batch_size + 1) % 10 == 0 or (i + batch_size) >= total:
                print(f"  Processed {min(i + batch_size, total)}/{total} samples")

        embeddings_tensor = torch.cat(all_embeddings, dim=0)
        ages_tensor = torch.tensor(all_ages, dtype=torch.float32).view(-1, 1)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)

        print(f"Finished {split_name}: Embeddings shape {embeddings_tensor.shape}, Ages shape {ages_tensor.shape}")
        return {
            "embeddings": embeddings_tensor,
            "ages": ages_tensor,
            "labels": labels_tensor,
            "genders": all_genders,
            "speaker_ids": all_speaker_ids
        }

    train_data = process_split(ds_train, "train", batch_size=4)
    val_data   = process_split(ds_val, "val", batch_size=4)
    test_data  = process_split(ds_test, "test", batch_size=4)

    cache = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "age_min": age_min,
        "age_max": age_max,
        "label2id": label2id,
        "id2label": id2label,
        "unique_labels": unique_labels,
        "target_column": "disorder_class",
        "split_seed": 42
    }

    torch.save(cache, output_path)
    print(f"\n[SUCCESS] Cached all embeddings to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    extract_embeddings()
