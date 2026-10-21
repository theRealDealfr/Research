import os
import re
import argparse
import random
import numpy as np
import torch
from datasets import load_dataset, Audio, ClassLabel
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.model_selection import train_test_split as sklearn_split


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract WavLM embeddings (last-layer pooled OR multi-layer mean+std)."
    )
    p.add_argument(
        "--mode",
        choices=["last", "stats"],
        default="last",
        help=(
            "Extraction mode. "
            "'last' = existing last-hidden-state masked-mean pool (default, unchanged). "
            "'stats' = per-layer masked mean+std concatenated -> new cache dir."
        ),
    )
    p.add_argument(
        "--stats_layers",
        type=str,
        default="6,9,12,15,18,24",
        help=(
            "Comma-separated 1-indexed transformer layer indices for stats mode "
            "(default: 6,9,12,15,18,24 -> L=6 -> 2048-d per layer)."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Shared dataset loading / splitting / age normalisation
# ---------------------------------------------------------------------------

def load_and_split(model_checkpoint="microsoft/wavlm-large", random_seed=42):
    dataset_name = "SAA-Lab/SLPHelmUltraSuitePlus"
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    dataset = load_dataset(dataset_name, split="train")
    print(f"Total samples loaded: {len(dataset)}")

    ds_meta = dataset.cast_column("audio", Audio(decode=False))
    paths = [item["audio"]["path"] for item in ds_meta]
    pattern = re.compile(r"^(?:test_|train_|val_|audio_)?([a-zA-Z0-9]+)[_\-\.]", re.IGNORECASE)
    speaker_ids = []
    for p_path in paths:
        filename = os.path.basename(p_path)
        match = pattern.search(filename)
        speaker_ids.append(match.group(1) if match else filename)
    dataset = dataset.add_column("speaker_id", speaker_ids)

    def clean_labels(batch):
        val = batch["disorder_class"]
        if val is None:
            return batch
        val = str(val).replace("['", "").replace("']", "").replace("[", "").replace("]", "")
        return {"disorder_class": val}

    dataset = dataset.map(clean_labels)
    dataset = dataset.filter(lambda x: x["disorder_class"] is not None)

    df = dataset.to_pandas()
    unique_speakers = sorted(df["speaker_id"].unique())
    print(f"Unique speaker IDs: {len(unique_speakers)}")

    train_val_spk, test_spk = sklearn_split(unique_speakers, test_size=0.2, random_state=random_seed)
    train_spk, val_spk = sklearn_split(train_val_spk, test_size=0.2, random_state=random_seed)

    assert set(train_spk).isdisjoint(set(val_spk)), "Leakage: Train <-> Val"
    assert set(train_spk).isdisjoint(set(test_spk)), "Leakage: Train <-> Test"
    assert set(val_spk).isdisjoint(set(test_spk)), "Leakage: Val <-> Test"

    ds_train = dataset.filter(lambda x: x["speaker_id"] in train_spk)
    ds_val   = dataset.filter(lambda x: x["speaker_id"] in val_spk)
    ds_test  = dataset.filter(lambda x: x["speaker_id"] in test_spk)

    unique_labels = sorted(
        list(set(ds_train["disorder_class"]) | set(ds_val["disorder_class"]) | set(ds_test["disorder_class"]))
    )
    label_feature = ClassLabel(names=unique_labels)
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}

    ds_train = ds_train.map(lambda x: {"labels": label_feature.str2int(x["disorder_class"])})
    ds_val   = ds_val.map(lambda x: {"labels": label_feature.str2int(x["disorder_class"])})
    ds_test  = ds_test.map(lambda x: {"labels": label_feature.str2int(x["disorder_class"])})

    print(f"Labels: {unique_labels}")
    print(f"Split -> Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}")

    train_ages = np.array([float(x) for x in ds_train["age"]])
    age_min = float(train_ages.min())
    age_max = float(train_ages.max())
    print(f"Training age range: {age_min:.2f} to {age_max:.2f}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    target_sr = feature_extractor.sampling_rate
    max_length = int(target_sr * 12.0)

    ds_train = ds_train.cast_column("audio", Audio(sampling_rate=target_sr))
    ds_val   = ds_val.cast_column("audio",   Audio(sampling_rate=target_sr))
    ds_test  = ds_test.cast_column("audio",  Audio(sampling_rate=target_sr))

    return (
        ds_train, ds_val, ds_test,
        feature_extractor, target_sr, max_length,
        age_min, age_max,
        label2id, id2label, unique_labels,
    )


# ---------------------------------------------------------------------------
# Mode A: existing last-layer masked-mean pooling (UNCHANGED behaviour)
# ---------------------------------------------------------------------------

def extract_embeddings_last():
    print("=== Step 1: Loading Dataset and Preparing Splits ===")
    (
        ds_train, ds_val, ds_test,
        feature_extractor, target_sr, max_length,
        age_min, age_max,
        label2id, id2label, unique_labels,
    ) = load_and_split()

    output_dir = "./cached_embeddings"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "wavlm_large_split_seed42.pt")

    print("\n=== Step 2: Loading WavLM Model ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model_checkpoint = "microsoft/wavlm-large"
    wavlm = AutoModel.from_pretrained(model_checkpoint, use_safetensors=True)
    wavlm.eval()
    wavlm.to(device)
    for param in wavlm.parameters():
        param.requires_grad = False

    def normalize_age(a):
        return (float(a) - age_min) / (age_max - age_min)

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
            audio_arrays = [s["audio"]["array"] for s in batch_samples]
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

        print(f"Finished {split_name}: Embeddings shape {embeddings_tensor.shape}")
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


# ---------------------------------------------------------------------------
# Mode B: multi-layer mean+std pooled (NEW - does NOT touch last-layer cache)
# ---------------------------------------------------------------------------

def _per_layer_stats(hidden_all_layers, feature_mask, selected_indices):
    """
    Compute masked mean and std for selected transformer layers.

    Args:
        hidden_all_layers : tuple of (B, T, D) tensors, one per layer (index 1..N are transformer layers)
        feature_mask      : (B, T) float tensor - 1.0 for valid frames, 0.0 for padding
        selected_indices  : list of 0-based indices into hidden_all_layers

    Returns:
        Tensor of shape (B, L, 2*D) - concatenated [mean, std] for each selected layer.
    """
    mask = feature_mask.float()                           # (B, T)
    n_valid = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)

    layer_stats = []
    for idx in selected_indices:
        h = hidden_all_layers[idx].float()                # (B, T, D)
        m = mask.unsqueeze(-1)                            # (B, T, 1)
        mean = (h * m).sum(dim=1) / n_valid              # (B, D)
        # Unbiased-ish variance; clamp denominator to avoid /0 when n=1
        var  = ((h - mean.unsqueeze(1)) ** 2 * m).sum(dim=1) / n_valid.clamp(min=2.0)
        std  = var.sqrt()                                 # (B, D)
        layer_stats.append(torch.cat([mean, std], dim=-1))  # (B, 2*D)

    return torch.stack(layer_stats, dim=1)                # (B, L, 2*D)


def extract_embeddings_stats(stats_layers_str: str):
    """
    New path: output_hidden_states=True, extract mean+std per selected layer.
    Saves fp16 tensor of shape [L, 2048] per clip.
    Output: ./cached_embeddings/wavlm_stats_layers/wavlm_large_stats_layers_seed42.pt
    Fully resumable: already-saved splits are skipped.
    """
    # 1-indexed layer numbers. In HuggingFace WavLM, hidden_states[0] is the CNN feature projection,
    # and hidden_states[1..24] correspond to transformer layers 1..24.
    selected_layer_indices = [int(x.strip()) for x in stats_layers_str.split(",")]
    L = len(selected_layer_indices)
    D = 1024   # WavLM-large hidden dim
    print(f"=== Stats-layer extraction: L={L} layers {selected_layer_indices} -> output dim per clip = [{L}, {2*D}] ===")

    print("=== Step 1: Loading Dataset and Preparing Splits ===")
    (
        ds_train, ds_val, ds_test,
        feature_extractor, target_sr, max_length,
        age_min, age_max,
        label2id, id2label, unique_labels,
    ) = load_and_split()

    output_dir = "./cached_embeddings/wavlm_stats_layers"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "wavlm_large_stats_layers_seed42.pt")

    # Load partial cache if resuming
    existing_cache = {}
    if os.path.exists(output_path):
        print(f"Found existing stats cache at {output_path}, loading for resumability ...")
        existing_cache = torch.load(output_path, map_location="cpu", weights_only=False)

    print("\n=== Step 2: Loading WavLM Model (output_hidden_states=True) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model_checkpoint = "microsoft/wavlm-large"
    wavlm = AutoModel.from_pretrained(
        model_checkpoint, use_safetensors=True, output_hidden_states=True
    )
    wavlm.eval()
    wavlm.to(device)
    for param in wavlm.parameters():
        param.requires_grad = False

    def normalize_age(a):
        return (float(a) - age_min) / (age_max - age_min)

    def process_split_stats(ds_split, split_name, batch_size=2):
        if (
            split_name in existing_cache
            and isinstance(existing_cache[split_name], dict)
            and "stats_embeddings" in existing_cache[split_name]
        ):
            print(f"  [SKIP] {split_name} already cached - skipping.")
            return existing_cache[split_name]

        print(f"\nExtracting stats embeddings for {split_name} ({len(ds_split)} samples)...")
        all_stats = []
        all_ages = []
        all_labels = []
        all_genders = []
        all_speaker_ids = []

        total = len(ds_split)
        for i in range(0, total, batch_size):
            batch_samples = [ds_split[j] for j in range(i, min(i + batch_size, total))]
            audio_arrays = [s["audio"]["array"] for s in batch_samples]
            ages   = [normalize_age(s["age"]) for s in batch_samples]
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
            input_values   = inputs["input_values"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                outputs = wavlm(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                # hidden_states: tuple of (B, T, D), indices 0..num_layers
                # index 0 = pre-transformer feature projection; 1..24 = transformer layers
                hidden_all = outputs.hidden_states

                # Build frame-level mask in the frame space of transformer hidden states
                feature_mask = wavlm._get_feature_vector_attention_mask(
                    hidden_all[1].shape[1],   # T (number of frames)
                    attention_mask,
                )
                # stats: (B, L, 2*D)
                stats = _per_layer_stats(hidden_all, feature_mask, selected_layer_indices)

            # Store fp16 to halve disk usage; training code casts back to fp32
            all_stats.append(stats.half().cpu())
            all_ages.extend(ages)
            all_labels.extend(labels)
            all_genders.extend(genders)
            all_speaker_ids.extend(spk_ids)

            if (i // batch_size + 1) % 10 == 0 or (i + batch_size) >= total:
                print(f"  Processed {min(i + batch_size, total)}/{total} samples")

        stats_tensor  = torch.cat(all_stats, dim=0)              # (N, L, 2*D) fp16
        ages_tensor   = torch.tensor(all_ages,   dtype=torch.float32).view(-1, 1)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)

        print(
            f"Finished {split_name}: stats shape {stats_tensor.shape} (fp16), "
            f"ages {ages_tensor.shape}"
        )
        return {
            "stats_embeddings": stats_tensor,   # (N, L, 2048) fp16
            "ages":             ages_tensor,
            "labels":           labels_tensor,
            "genders":          all_genders,
            "speaker_ids":      all_speaker_ids,
        }

    train_data = process_split_stats(ds_train, "train", batch_size=4)
    existing_cache["train"] = train_data
    existing_cache.update({
        "age_min": age_min,
        "age_max": age_max,
        "label2id": label2id,
        "id2label": id2label,
        "unique_labels": unique_labels,
        "target_column": "disorder_class",
        "split_seed": 42,
        "stats_layers": selected_layer_indices,
        "embed_dim": L * 2 * D,
    })
    torch.save(existing_cache, output_path)

    val_data = process_split_stats(ds_val, "val", batch_size=4)
    existing_cache["val"] = val_data
    torch.save(existing_cache, output_path)

    test_data = process_split_stats(ds_test, "test", batch_size=4)
    existing_cache["test"] = test_data
    torch.save(existing_cache, output_path)

    print(f"\n[SUCCESS] Cached stats embeddings to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print(f"Embedding shape per clip: [{L}, {2*D}] (fp16), total fused dim = {L*2*D} + 1 = {L*2*D+1}")


# ---------------------------------------------------------------------------
# Entry point (legacy compatibility: calling without args runs 'last' mode)
# ---------------------------------------------------------------------------

def extract_embeddings():
    """Backward-compatible entry point (no CLI args) - runs last-layer mode."""
    extract_embeddings_last()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "stats":
        extract_embeddings_stats(args.stats_layers)
    else:
        extract_embeddings_last()
