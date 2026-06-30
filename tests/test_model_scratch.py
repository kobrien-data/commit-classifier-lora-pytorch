import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import CommitDataset
from model_scratch import FromScratchClassifier



# --- Rebuild the data side ---
commit_ds = load_dataset("0x404/ccs_dataset")
dataset = CommitDataset(commit_ds["train"])
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# --- vocab_size must match the tokenizer the Dataset uses ---
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
vocab_size = tokenizer.vocab_size
print("vocab_size:", vocab_size)

# --- Build the model and run one batch through it ---
model = FromScratchClassifier(vocab_size=vocab_size, num_classes=10)

batch = next(iter(loader))
logits = model(batch["input_ids"])

print("input_ids shape:", batch["input_ids"].shape)   # expect [4, 64]
print("logits shape:   ", logits.shape)               # expect [4, 10]
print("\nlogits for first example:\n", logits[0])