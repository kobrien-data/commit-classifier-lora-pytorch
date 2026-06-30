import torch.nn as nn

class FromScratchClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=128, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        pooled = embedded.mean(dim=1)       
        x = self.fc1(pooled)                
        x = self.relu(x)                    
        x = self.dropout(x)                 
        logits = self.fc2(x)                
        return logits
