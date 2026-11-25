import numpy as np


class MostPopular:
    def __init__(self, num_items):
        if num_items < 1:
            raise ValueError("Empty catalogue")
        self.num_items = num_items
        self.counts = np.zeros(num_items + 1, dtype=np.int64)

    def fit(self, train):
        ids = train.item_id.to_numpy(dtype=np.int64)
        if len(ids) and (ids.min() < 1 or ids.max() > self.num_items):
            raise ValueError("Training item outside catalogue")
        self.counts = np.bincount(ids, minlength=self.num_items + 1)
        return self

    def score_items(self, history, item_ids):
        return self.counts[item_ids].astype(float)

    def save(self, path):
        # Array-only checkpoint; never unpickle externally supplied objects.
        with open(path, "wb") as stream:
            np.savez(stream, counts=self.counts)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as state:
            counts = state["counts"]
        if counts.ndim != 1 or len(counts) < 2 or (counts < 0).any() or counts[0] != 0:
            raise ValueError("Invalid popularity checkpoint")
        model = cls(len(counts) - 1)
        model.counts = counts
        return model
