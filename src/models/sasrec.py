"""PyTorch port of kang205/SASRec's pre-normalized query/residual blocks."""
import math
import torch
from torch import nn
from src.models.base import SequentialModel, Attention


class SASBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim, dropout = config["hidden_dim"], config["dropout"]
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-8), nn.LayerNorm(dim, eps=1e-8)
        self.attention = Attention(dim, config["num_heads"], config["attention_dropout"], False)
        # Original SASRec uses d -> d -> d, not the BERT 4d intermediate size.
        self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, hidden, valid):
        queries = self.norm1(hidden)
        hidden = queries + self.attention(queries, hidden, valid, True)
        normalized = self.norm2(hidden)
        return (normalized + self.ffn(normalized)) * valid.unsqueeze(-1)


class SASRec(SequentialModel):
    def __init__(self, num_items, config):
        super().__init__(num_items, config)
        self.position = nn.Embedding(self.max_length, self.dim)
        self.dropout = nn.Dropout(config["dropout"])
        self.blocks = nn.ModuleList([SASBlock(config) for _ in range(config["num_layers"])])
        self.norm = nn.LayerNorm(self.dim, eps=1e-8)
        # TF dense/embedding defaults are Glorot; positional/item tables too.
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, tokens):
        valid = tokens.ne(0)
        hidden = self.embedding(tokens) * math.sqrt(self.dim)
        hidden = self.dropout(hidden + self.position(torch.arange(tokens.shape[1], device=tokens.device)))
        hidden = hidden * valid.unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, valid)
        return self.norm(hidden) * valid.unsqueeze(-1)
