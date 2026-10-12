import torch
from datasets import load_dataset, Audio
import re, os, collections
from sklearn.model_selection import train_test_split as sklearn_split

# 1. Dataset setup
dataset = load_dataset("SAA-Lab/SLPHelmUltraSuitePlus", split="train")
dataset = dataset.cast_column("audio", Audio(decode=False))
paths = [item["audio"]["path"] for item in dataset]
pattern = re.compile(r"^(?:test_|train_|val_|audio_)?([a-zA-Z0-9]+)[_\-\.]", re.IGNORECASE)
speaker_ids = [pattern.search(os.path.basename(p)).group(1) if pattern.search(os.path.basename(p)) else os.path.basename(p) for p in paths]
dataset = dataset.add_column("speaker_id", speaker_ids)

def clean_meta(b):
    val_b = b["disorder_class"]
    if val_b is not None:
        val_b = str(val_b).replace("['", "").replace("']", "").replace("[", "").replace("]", "").strip()
    
    val_s = b["disorder_symptom"]
    if val_s is not None:
        val_s = str(val_s).replace("['", "").replace("']", "").replace("[", "").replace("]", "").strip()
        if val_s in ['substitution_modified', 'phonological']: val_s = 'stuttering'
        
    val_t = b["disorder_type"]
    if val_t is not None:
        val_t = str(val_t).replace("['", "").replace("']", "").replace("[", "").replace("]", "").strip()
        if val_t in ['addition', 'omission']: val_t = 'articulation'
        elif val_t == 'stuttering': val_t = 'phonological'
        
    return {"clean_class": val_b, "clean_symptom": val_s, "clean_type": val_t}

dataset = dataset.map(clean_meta)
df = dataset.to_pandas()
unique_speakers = sorted(df["speaker_id"].unique())
train_val_spk, test_spk = sklearn_split(unique_speakers, test_size=0.2, random_state=42)
train_spk, val_spk = sklearn_split(train_val_spk, test_size=0.2, random_state=42)

ds_train = dataset.filter(lambda x: x["speaker_id"] in train_spk)
ds_val   = dataset.filter(lambda x: x["speaker_id"] in val_spk)
ds_test  = dataset.filter(lambda x: x["speaker_id"] in test_spk)

disorder_classes = ["substitution", "stuttering", "omission", "addition"]
ds_train_stage2 = ds_train.filter(lambda x: x["clean_class"] == "speech_disorder" and x["clean_symptom"] in disorder_classes)
ds_val_stage2   = ds_val.filter(lambda x: x["clean_class"] == "speech_disorder" and x["clean_symptom"] in disorder_classes)
ds_test_stage2  = ds_test.filter(lambda x: x["clean_class"] == "speech_disorder" and x["clean_symptom"] in disorder_classes)

# Load caches
c_orig = torch.load("cached_embeddings/wavlm_stage2_disorder_symptom.pt", map_location="cpu", weights_only=False)
c_aug = torch.load("cached_embeddings/wavlm_stage2_disorder_symptom_augmented.pt", map_location="cpu", weights_only=False)

# Labels for T2: 0: articulation, 1: phonological
t2_classes = ["articulation", "phonological"]
t2_label2id = {"articulation": 0, "phonological": 1}

# Test labels
test_clean_types = [s["clean_type"] for s in ds_test_stage2]
test_t2_labels = torch.tensor([t2_label2id[t] for t in test_clean_types], dtype=torch.long)
test_genders = [s["gender"] for s in ds_test_stage2]

# Val labels
val_clean_types = [s["clean_type"] for s in ds_val_stage2]
val_t2_labels = torch.tensor([t2_label2id[t] for t in val_clean_types], dtype=torch.long)
val_genders = [s["gender"] for s in ds_val_stage2]

# Unaugmented Train labels
train_clean_types = [s["clean_type"] for s in ds_train_stage2]
valid_train_idx = [i for i, t in enumerate(train_clean_types) if t in t2_label2id]
train_unaug_embeddings = c_orig["train"]["embeddings"][valid_train_idx]
train_unaug_ages = c_orig["train"]["ages"][valid_train_idx]
train_unaug_labels = torch.tensor([t2_label2id[train_clean_types[i]] for i in valid_train_idx], dtype=torch.long)
train_unaug_genders = [ds_train_stage2[i]["gender"] for i in valid_train_idx]

print(f"Unaugmented Train: {len(train_unaug_labels)} samples: {collections.Counter([t2_classes[l] for l in train_unaug_labels.numpy()])}")
print(f"Val:               {len(val_t2_labels)} samples: {collections.Counter([t2_classes[l] for l in val_t2_labels.numpy()])}")
print(f"Test:              {len(test_t2_labels)} samples: {collections.Counter([t2_classes[l] for l in test_t2_labels.numpy()])}")

# Augmented Train labels: replicate mapping matching extract_stage2_embeddings_augmented.py
# For each sample in ds_train_stage2:
# if M -> 2 entries; if F -> 3 entries
aug_train_labels_list = []
aug_train_valid_mask = []
for i, sample in enumerate(ds_train_stage2):
    ctype = sample["clean_type"]
    g = sample["gender"]
    n_reps = 2 if g == "M" else 3
    is_valid = (ctype in t2_label2id)
    for _ in range(n_reps):
        if is_valid:
            aug_train_labels_list.append(t2_label2id[ctype])
            aug_train_valid_mask.append(True)
        else:
            aug_train_valid_mask.append(False)

aug_mask_tensor = torch.tensor(aug_train_valid_mask, dtype=torch.bool)
train_aug_embeddings = c_aug["train"]["embeddings"][aug_mask_tensor]
train_aug_ages = c_aug["train"]["ages"][aug_mask_tensor]
train_aug_labels = torch.tensor(aug_train_labels_list, dtype=torch.long)
train_aug_genders = [g for i, g in enumerate(c_aug["train"]["genders"]) if aug_train_valid_mask[i]]

print(f"Augmented Train:   {len(train_aug_labels)} samples: {collections.Counter([t2_classes[l] for l in train_aug_labels.numpy()])}")
print(f"Augmented Train Genders: {collections.Counter(train_aug_genders)}")

# Save these clean caches for T2
t2_unaug_cache = {
    "train": {"embeddings": train_unaug_embeddings, "ages": train_unaug_ages, "labels": train_unaug_labels, "genders": train_unaug_genders},
    "val":   {"embeddings": c_orig["val"]["embeddings"], "ages": c_orig["val"]["ages"], "labels": val_t2_labels, "genders": val_genders},
    "test":  {"embeddings": c_orig["test"]["embeddings"], "ages": c_orig["test"]["ages"], "labels": test_t2_labels, "genders": test_genders},
    "classes": t2_classes,
    "label2id": t2_label2id
}
torch.save(t2_unaug_cache, "cached_embeddings/wavlm_t2_disorder_type_unaugmented.pt")

t2_aug_cache = {
    "train": {"embeddings": train_aug_embeddings, "ages": train_aug_ages, "labels": train_aug_labels, "genders": train_aug_genders},
    "val":   {"embeddings": c_aug["val"]["embeddings"], "ages": c_aug["val"]["ages"], "labels": val_t2_labels, "genders": val_genders},
    "test":  {"embeddings": c_aug["test"]["embeddings"], "ages": c_aug["test"]["ages"], "labels": test_t2_labels, "genders": test_genders},
    "classes": t2_classes,
    "label2id": t2_label2id
}
torch.save(t2_aug_cache, "cached_embeddings/wavlm_t2_disorder_type_augmented.pt")
print("Saved both T2 unaugmented and augmented caches successfully!")
