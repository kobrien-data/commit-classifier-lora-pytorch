import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        #freezes the model weights & bias (we don't touch this, we add to its outputs)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        # squuezes the dimensions of in_features down to size r, the 0.01 initializes it to small random values to create small perturbations
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        # opposite of A, this pushes the dimensions back up to the original size. initialized as zeros because the product B·A starts at zero
        # which means that the first training step starts at zero and allows the layer to behave like DistilBERT
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        # controls the magnitude of the LoRA correction and decouples this from choice of rank r, this allows for stable experimentation
        self.scaling = alpha / r

    def forward(self, x):
        """
        returns the frozen layer's outputs PLUS the outputs of the LoRA calculations
        """
        return self.base(x) + self.scaling * (x @ self.A.T @ self.B.T)