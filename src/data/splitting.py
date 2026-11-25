import pandas as pd


def leave_one_out(frame):
    frame = frame.sort_values(["user_id", "timestamp", "item_id"], kind="stable").copy()
    if (frame.groupby("user_id").size() < 3).any():
        raise ValueError("Each user needs at least three events")
    offset = frame.groupby("user_id").cumcount(ascending=False)
    parts = {"train": frame.loc[offset >= 2].copy(), "validation": frame.loc[offset == 1].copy(), "test": frame.loc[offset == 0].copy()}
    validate_splits(parts)
    return parts


def validate_splits(parts):
    events = [set(parts[key].event_id) for key in ("train", "validation", "test")]
    if any(events[i] & events[j] for i in range(3) for j in range(i)):
        raise ValueError("Event leakage across splits")
    users = set(parts["train"].user_id)
    if any(set(parts[key].user_id) != users for key in ("validation", "test")):
        raise ValueError("User coverage differs across splits")
    for key in ("validation", "test"):
        if parts[key].user_id.duplicated().any():
            raise ValueError("Multiple held-out targets per user")
    combined = pd.concat([part.assign(split=key) for key, part in parts.items()])
    if combined.event_id.duplicated().any():
        raise ValueError("Duplicate event IDs")
    ordered = combined.sort_values(["user_id", "timestamp", "item_id"], kind="stable")
    reverse_position = ordered.groupby("user_id").cumcount(ascending=False)
    # Compare arrays because arbitrary input frames may have duplicate row indexes.
    import numpy as np
    expected = np.where(reverse_position.to_numpy() == 0, "test", np.where(reverse_position.to_numpy() == 1, "validation", "train"))
    if not np.array_equal(ordered.split.to_numpy(), expected):
        raise ValueError("Chronological split violation")


def evaluation_cases(parts, split="test"):
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test")
    history = parts["train"]
    if split == "test":
        history = pd.concat([history, parts["validation"]])
    import numpy as np
    ordered = history.sort_values(["user_id", "timestamp", "item_id"])
    users, starts = np.unique(ordered.user_id.to_numpy(), return_index=True)
    item_groups = np.split(ordered.item_id.to_numpy(), starts[1:])
    histories = {user: values.tolist() for user, values in zip(users, item_groups)}
    return [{"user_id": int(row.user_id), "history": histories[row.user_id], "target": int(row.item_id)}
            for row in parts[split].sort_values("user_id").itertuples()]
