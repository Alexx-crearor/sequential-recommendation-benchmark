import numpy as np


class SequenceDataset:
    """Prefix samples from an explicitly supplied training split only."""

    def __init__(self, train, max_seq_length=50):
        if max_seq_length < 1:
            raise ValueError("max_seq_length must be positive")
        self.max_length = max_seq_length
        self.samples = []
        for user, group in train.sort_values(["user_id", "timestamp", "item_id"]).groupby("user_id"):
            items, events = group.item_id.tolist(), group.event_id.tolist()
            for index in range(1, len(items)):
                self.samples.append((int(user), items[max(0, index - max_seq_length):index], int(items[index]), int(events[index])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        user, history, target, event = self.samples[index]
        padded = np.zeros(self.max_length, dtype=np.int64)
        padded[-len(history):] = history
        return {"user_id": user, "sequence": padded, "target": target, "target_event_id": event}
