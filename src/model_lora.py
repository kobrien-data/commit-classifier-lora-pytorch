import torch
import torch.nn as nn
from lora import LoRALinear
from transformers import AutoModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from dataset import CommitDataset

class ModelLora(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")

        #loop over each parameter of the model and freeze it
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        for i in range(self.model.config.num_hidden_layers):
            attn = self.model.transformer.layer[i].attention
            attn.q_lin = LoRALinear(attn.q_lin)
            attn.v_lin = LoRALinear(attn.v_lin)

        self.classifier = nn.Linear(self.model.config.dim, 6)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        
        mask = attention_mask.unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        logits = self.classifier(pooled)
        return logits

#model = ModelLora()
#trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#total = sum(p.numel() for p in model.parameters())
##print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
#
#
#commit_ds = load_dataset("0x404/ccs_dataset")
#test_dataset = CommitDataset(commit_ds["test"])
#loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
#batch = next(iter(loader))
#logits = model(batch["input_ids"], batch["attention_mask"])
##print(logits.shape)   # expect [B, 10]