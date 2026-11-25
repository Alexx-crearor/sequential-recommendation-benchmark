"""Shared scoring and checkpoint interface for neural sequential models."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def pad_histories(histories, max_length, device, mask_id=None):
    tokens = torch.zeros((len(histories), max_length), dtype=torch.long, device=device)
    for row, history in enumerate(histories):
        values = list(history)
        if mask_id is not None:
            values.append(mask_id)
        values = values[-max_length:]
        if values:
            tokens[row, -len(values):] = torch.tensor(values, device=device)
    return tokens


class SequentialModel(nn.Module):
    masked_inference = False

    def __init__(self, num_items, config):
        super().__init__()
        self.num_items, self.config = num_items, dict(config)
        self.max_length = config["max_seq_length"]
        self.dim = config["hidden_dim"]
        self.mask_id = num_items + 1
        self.embedding = nn.Embedding(num_items + 2, self.dim, padding_idx=0)

    @property
    def device(self):
        return self.embedding.weight.device

    def representations(self, histories):
        tokens = pad_histories(histories, self.max_length, self.device,
                               self.mask_id if self.masked_inference else None)
        return self.forward(tokens)[:, -1]

    def project(self, hidden):
        return hidden

    def logits(self, hidden, item_ids=None):
        hidden = self.project(hidden)
        if item_ids is None:
            return F.linear(hidden, self.embedding.weight[1:self.num_items + 1])
        return hidden @ self.embedding(item_ids).T

    @torch.no_grad()
    def score_items(self, history, item_ids):
        self.eval()
        ids = torch.as_tensor(item_ids, dtype=torch.long, device=self.device)
        return self.logits(self.representations([history]), ids)[0].float().cpu().numpy()

    def save(self, path):
        from src.training.checkpoint import atomic_torch_save
        atomic_torch_save(path, {"config": self.config, "num_items": self.num_items, "model": self.state_dict()})

    @classmethod
    def load(cls, path, device="cpu"):
        from src.models.factory import build_model
        state = torch.load(path, map_location=device, weights_only=True)
        model = build_model(state["config"]["model"], state["num_items"], state["config"]).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        return model


class Attention(nn.Module):
    """Explicit per-head attention; padded queries have zero contribution."""
    def __init__(self, dim, heads, dropout, output_projection=True):
        super().__init__()
        if dim % heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.heads, self.head_dim = heads, dim // heads
        self.query, self.key, self.value = (nn.Linear(dim, dim) for _ in range(3))
        self.output = nn.Linear(dim, dim) if output_projection else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, valid, causal):
        batch, length, dim = queries.shape
        def split(x):
            return x.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.query(queries)), split(self.key(keys)), split(self.value(keys))
        with torch.autocast(device_type=keys.device.type, enabled=False):
            scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
            allowed = valid[:, None, None, :].expand(-1, 1, length, -1)
            if causal:
                allowed = allowed & torch.ones(length, length, device=keys.device, dtype=torch.bool).tril()[None, None]
            # Finite fill avoids NaNs for fully padded queries; explicitly zero them afterward.
            probs = scores.masked_fill(~allowed, -1e9).softmax(-1)
        probs = self.dropout(probs).to(v.dtype)
        context = (probs @ v).transpose(1, 2).reshape(batch, length, dim)
        return self.output(context) * valid.unsqueeze(-1)


def initialize(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, std=.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
