import torch 
import torch.nn as nn


class LayerAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.scorer = nn.Linear(d, 1, bias=False)

    def forward(self, x):
        # x: [B, L, D]
        attn = torch.softmax(self.scorer(x).squeeze(-1), dim=-1)  # [B, L]
        return torch.einsum("bl,bld->bd", attn, x)                # [B, D]

class Probe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.layer_pool = LayerAttention(d)
        self.dropout = nn.Dropout(0.1)
        self.clf = nn.Linear(d, 3)

    def forward(self, x):
        x = self.layer_pool(x)  # [B, D]
        x = self.dropout(x)
        return self.clf(x)