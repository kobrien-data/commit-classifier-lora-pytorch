import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, classification_report
from dataset import CommitDataset
from model_lora import ModelLora

# Rebuild the SAME validation split you trained with (same seed = same rows)
commit_ds = load_dataset("0x404/ccs_dataset")
full_train = commit_ds["train"].train_test_split(test_size=0.15, seed=42)
val_split = full_train["test"]

val_dataset = CommitDataset(val_split)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load the trained LoRA model
model = ModelLora()
model.load_state_dict(torch.load("src/lora_model.pt"))
model.eval()

# Collect predictions and true labels
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        logits = model(batch["input_ids"], batch["attention_mask"])
        all_preds.extend(logits.argmax(dim=1).tolist())
        all_labels.extend(batch["label"].tolist())

names = ["feat", "fix", "docs", "refactor", "test", "chore"]

print(classification_report(all_labels, all_preds, target_names=names, zero_division=0))
print(confusion_matrix(all_labels, all_preds))