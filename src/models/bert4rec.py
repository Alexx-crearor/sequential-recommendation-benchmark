"""Bidirectional Cloze encoder and capacity-matched causal E2 variant."""
import torch
from torch import nn
from torch.nn import functional as F
from src.models.base import SequentialModel, Attention, initialize


class PostNormBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim, drop = config["hidden_dim"], config["dropout"]
        self.attention = Attention(dim, config["num_heads"], config["attention_dropout"])
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-12), nn.LayerNorm(dim, eps=1e-12)
        self.out_dropout = nn.Dropout(drop)
        self.ffn = nn.Sequential(nn.Linear(dim, config["ffn_dim"]), nn.GELU(), nn.Linear(config["ffn_dim"], dim), nn.Dropout(drop))

    def forward(self, hidden, valid, causal):
        hidden = self.norm1(hidden + self.out_dropout(self.attention(hidden, hidden, valid, causal)))
        return self.norm2(hidden + self.ffn(hidden)) * valid.unsqueeze(-1)


class BERT4Rec(SequentialModel):
    masked_inference = True

    def __init__(self, num_items, config):
        super().__init__(num_items, config)
        self.causal = config.get("attention", "bidirectional") == "causal"
        self.masked_inference = config.get("objective", "masked_ce") == "masked_ce"
        self.position = nn.Embedding(self.max_length, self.dim)
        self.input_norm = nn.LayerNorm(self.dim, eps=1e-12)
        self.dropout = nn.Dropout(config["dropout"])
        self.blocks = nn.ModuleList([PostNormBlock(config) for _ in range(config["num_layers"])])
        self.head = nn.Linear(self.dim, self.dim)
        self.head_norm = nn.LayerNorm(self.dim, eps=1e-12)
        self.output_bias = nn.Parameter(torch.zeros(num_items))
        self.apply(initialize)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, tokens):
        valid = tokens.ne(0)
        hidden = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        hidden = self.dropout(self.input_norm(hidden)) * valid.unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, valid, self.causal)
        return hidden

    def project(self, hidden):
        return self.head_norm(F.gelu(self.head(hidden)))

    def logits(self, hidden, item_ids=None):
        output = super().logits(hidden, item_ids)
        return output + (self.output_bias if item_ids is None else self.output_bias[item_ids - 1])
