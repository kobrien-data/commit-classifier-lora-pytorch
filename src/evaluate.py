import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset import CommitDataset
from datasets import load_dataset
from model_scratch import FromScratchClassifier
from sklearn.metrics import f1_score

commit_ds = load_dataset("0x404/ccs_dataset")
test_dataset = CommitDataset(commit_ds["test"])
loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

vocab_size = test_dataset.tokenizer.vocab_size
model = FromScratchClassifier(vocab_size=vocab_size, num_classes=10)
model.load_state_dict(torch.load("src/scratch_model.pt"))


model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in loader:
        logits = model(batch["input_ids"])
        pred_batch = logits.argmax(dim=1)
        all_preds.extend(pred_batch.tolist())
        all_labels.extend(batch["label"].tolist()) 
        
accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
macro_f1 = f1_score(all_labels, all_preds, average="macro")
print(f"Test accuracy: {accuracy:.4f}")
print(f"Macro-F1:      {macro_f1:.4f}")