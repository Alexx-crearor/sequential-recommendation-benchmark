"""Training-only objective-specific samples. No held-out frame is accepted."""
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def left_pad(values, length):
    result = torch.zeros(length, dtype=torch.long)
    values = list(values)[-length:]
    if values:
        result[-len(values):] = torch.tensor(values, dtype=torch.long)
    return result


class TrainingDataset(Dataset):
    def __init__(self, train, num_items, config):
        self.config, self.num_items = config, num_items
        self.length = config["max_seq_length"]
        self.mask_id = num_items + 1
        self.mode = "fenrec" if config["model"] == "fenrec" else config["objective"]
        self.sequences = []
        self.samples, self.by_target = [], {}
        self.train_event_ids = set(map(int, train.event_id))
        for user, group in train.sort_values(["user_id", "timestamp", "item_id"]).groupby("user_id"):
            values = list(map(int, group.item_id))
            events = list(map(int, group.event_id))
            if self.mode == "fenrec":
                for position in range(1, len(values)):
                    index = len(self.samples)
                    self.samples.append({"user_id": int(user), "history": values[max(0, position-self.length):position],
                                         "target": values[position], "future": values[position:position+3],
                                         "future_event_ids": events[position:position+3]})
                    self.by_target.setdefault(values[position], []).append(index)
            elif len(values) >= 2 or self.mode == "masked_ce":
                self.sequences.append((int(user), values, events))
        if not len(self):
            raise ValueError("No training samples after retention")
        self.cached_tokens = None
        self.cached_targets = None
        self.cached_future = None
        self.positive_indices = None
        if config.get("cache_preprocessing", True):
            estimated = len(self) * self.length * 8 * (1 if self.mode in ("fenrec", "masked_ce") else 2)
            if estimated <= config.get("preprocessing_cache_mb", 512) * 1048576:
                if self.mode == "fenrec":
                    self.cached_tokens = torch.stack([left_pad(sample["history"], self.length) for sample in self.samples])
                    self.cached_future = torch.stack([left_pad(sample["future"], 3) for sample in self.samples])
                    self.positive_indices = np.full(len(self.samples), -1, dtype=np.int64)
                    for indices in self.by_target.values():
                        last = indices[-1]
                        last_history = self.samples[last]["history"]
                        different = next((i for i in reversed(indices) if self.samples[i]["history"] != last_history), -1)
                        for index in indices:
                            self.positive_indices[index] = different if self.samples[index]["history"] == last_history else last
                else:
                    self.cached_tokens = torch.stack([left_pad(values if self.mode == "masked_ce" else values[:-1], self.length) for _, values, _ in self.sequences])
                    if self.mode != "masked_ce":
                        self.cached_targets = torch.stack([left_pad(values[1:], self.length) for _, values, _ in self.sequences])

    def __len__(self):
        return len(self.samples) if self.mode == "fenrec" else len(self.sequences)

    def __getitem__(self, index):
        if self.mode == "fenrec":
            sample = self.samples[index]
            candidates = self.by_target[sample["target"]]
            # Pinned source uses the final different sequence found in label group.
            if self.positive_indices is not None:
                other = int(self.positive_indices[index])
                if other < 0:
                    other = random.choice(candidates)
            else:
                positives = [other for other in candidates if self.samples[other]["history"] != sample["history"]]
                other = positives[-1] if positives else random.choice(candidates)
            return {"tokens": self.cached_tokens[index] if self.cached_tokens is not None else left_pad(sample["history"], self.length),
                    "positive": self.cached_tokens[other] if self.cached_tokens is not None else left_pad(self.samples[other]["history"], self.length),
                    "target": torch.tensor(sample["target"]), "future": self.cached_future[index] if self.cached_future is not None else left_pad(sample["future"], 3),
                    "index": torch.tensor(index)}
        user, values, events = self.sequences[index]
        if self.mode == "masked_ce":
            tokens = self.cached_tokens[index].clone() if self.cached_tokens is not None else left_pad(values, self.length)
            selected = torch.zeros_like(tokens, dtype=torch.bool)
            positions = tokens.ne(0).nonzero(as_tuple=True)[0].tolist()
            if random.random() < self.config["last_mask_probability"]:
                selected[positions[-1]] = True
            else:
                count = max(1, round(len(positions) * self.config["mask_ratio"]))
                selected[random.sample(positions, count)] = True
            targets = tokens.clone().masked_fill(~selected, 0)
            tokens = tokens.masked_fill(selected, self.mask_id)
            return {"tokens": tokens, "targets": targets}
        tokens = self.cached_tokens[index] if self.cached_tokens is not None else left_pad(values[:-1], self.length)
        targets = self.cached_targets[index] if self.cached_targets is not None else left_pad(values[1:], self.length)
        result = {"tokens": tokens, "targets": targets}
        if self.mode == "next_bce":
            excluded = set(values)
            if len(excluded) >= self.num_items:
                raise ValueError("No negative candidate outside the user's training history")
            negatives = torch.zeros_like(targets)
            for position in targets.ne(0).nonzero(as_tuple=True)[0].tolist():
                item = random.randint(1, self.num_items)
                while item in excluded:
                    item = random.randint(1, self.num_items)
                negatives[position] = item
            result["negatives"] = negatives
        return result


def verify_training_boundary(dataset, parts):
    heldout = set(parts["validation"].event_id) | set(parts["test"].event_id)
    if dataset.train_event_ids & heldout:
        raise ValueError("Held-out events in training dataset")
    if dataset.mode == "fenrec":
        for sample in dataset.samples:
            if not set(sample["future_event_ids"]) <= dataset.train_event_ids:
                raise ValueError("Held-out event in future labels")
            if len(sample["future"]) > 3 or sample["future"][0] != sample["target"]:
                raise ValueError("Invalid subsequent interaction window")
