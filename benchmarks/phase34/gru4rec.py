"""Explicit GRU4Rec-CE adaptation: user histories, full-softmax CE, tied output."""
import torch
from torch import nn
from benchmarks.phase34.base import SequentialModel


class GRU4RecCE(SequentialModel):
    def __init__(self, num_items, config):
        super().__init__(num_items, config)
        self.dropout = nn.Dropout(config["dropout"])
        self.gru = nn.GRU(self.dim, self.dim, config["num_layers"], batch_first=True,
                          dropout=config["dropout"] if config["num_layers"] > 1 else 0)

    def forward(self, tokens):
        # Compact real tokens before the recurrent pass; padding never updates state.
        lengths = tokens.ne(0).sum(-1)
        compact = torch.zeros_like(tokens)
        for index in range(len(tokens)):
            values = tokens[index][tokens[index].ne(0)]
            compact[index, :len(values)] = values
        embedded = self.dropout(self.embedding(compact))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=tokens.shape[1])
        aligned = torch.zeros_like(output)
        for index, length in enumerate(lengths.tolist()):
            if length:
                aligned[index, tokens[index].ne(0)] = output[index, :length]
        return aligned
