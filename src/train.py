import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset import CommitDataset
from datasets import load_dataset
from model_scratch import FromScratchClassifier

commit_ds = load_dataset("0x404/ccs_dataset")
train_dataset = CommitDataset(commit_ds["train"])
loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

vocab_size = train_dataset.tokenizer.vocab_size
model = FromScratchClassifier(vocab_size=vocab_size, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in loader:
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["label"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() 
    # print average loss for this epoch
    avg_loss = total_loss / len(loader)
    print(f"Average of total loss for epoch {epoch}: {avg_loss}")
torch.save(model.state_dict(), "scratch_model.pt")