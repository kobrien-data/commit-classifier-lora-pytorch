import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset import CommitDataset
from datasets import load_dataset
from model_scratch import FromScratchClassifier
from model_lora import ModelLora

commit_ds = load_dataset("0x404/ccs_dataset")
full_train = commit_ds["train"].train_test_split(test_size=0.15, seed=42)
train_split = full_train["train"]     # ~1190 examples
val_split   = full_train["test"]

train_dataset = CommitDataset(train_split)
val_dataset = CommitDataset(val_split)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

vocab_size = train_dataset.tokenizer.vocab_size
model = FromScratchClassifier(vocab_size=vocab_size, num_classes=10)
#model = ModelLora()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
num_epochs = 40

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["input_ids"])
            #logits = model(batch["input_ids"], batch["attention_mask"])
            preds = logits.argmax(dim=1)
            correct += (preds == batch["label"]).sum().item()
            total += batch["label"].size(0)
    return correct / total

best_val_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        logits = model(batch["input_ids"])
        #logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["label"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() 
    # print average loss for this epoch
    avg_loss = total_loss / len(train_loader)
    val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch:2d} | train loss: {avg_loss:.4f} | val acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "src/scratch_model.pt")
        #torch.save(model.state_dict(), "src/lora_model.pt")
        print(f"           ^ new best, saved (val acc {val_acc:.4f})")

print(f"\nBest validation accuracy: {best_val_acc:.4f}")