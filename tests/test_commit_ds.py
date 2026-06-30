import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from dataset import CommitDataset
from collections import Counter

# Load the data and build the dataset for the train split
commit_ds = load_dataset("0x404/ccs_dataset")
dataset = CommitDataset(commit_ds["train"])
dataset_test = CommitDataset(commit_ds["test"])

print("Number of examples train :", len(dataset))
print("Number of examples test :", len(dataset_test))
print(Counter(dataset.data["annotated_type"]))
print(Counter(dataset_test.data["annotated_type"]))

item = dataset[0]
print("\nKeys returned:", list(item.keys()))
print("input_ids shape:     ", item["input_ids"].shape)      # expect [64]
print("attention_mask shape:", item["attention_mask"].shape) # expect [64]
print("label:               ", item["label"], "| shape:", item["label"].shape)  # scalar, shape []

# Peek at the actual values so it's not abstract
print("\nFirst 15 token ids:", item["input_ids"][:15])
print("First 15 mask vals:", item["attention_mask"][:15])

# --- Check that batching works ---
loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(loader))
print("\nBatched input_ids shape:     ", batch["input_ids"].shape)      # expect [4, 64]
print("Batched attention_mask shape:", batch["attention_mask"].shape) # expect [4, 64]
print("Batched label shape:         ", batch["label"].shape)          # expect [4]