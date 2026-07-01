import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

class CommitDataset(Dataset):
    def __init__(self, split):
        super().__init__()
        self.data = split
        self.category_map = {"feat": 0, "fix": 1, "docs": 2, "refactor": 3, "test": 4, "chore": 5, "ci": 5, "build": 5, "style": 5, "perf": 3}
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        text = row["masked_commit_message"]
        label_name = row["annotated_type"]
        label_id = self.category_map[label_name]
        encoded = self.tokenizer(text, padding="max_length", truncation=True, max_length=64, return_tensors="pt",)

        return {
        "input_ids": encoded["input_ids"].squeeze(0),
        "attention_mask": encoded["attention_mask"].squeeze(0),
        "label": torch.tensor(label_id),
    }
