"""Amazon 2014 ingestion. No evaluation-derived learning statistics."""
import ast
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def records(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                try:
                    yield ast.literal_eval(line)  # old metadata uses Python literals
                except (ValueError, SyntaxError) as error:
                    raise ValueError(f"Invalid record at {path}:{line_number}") from error


def load_interactions(path):
    rows = []
    for record in records(path):
        rows.append((str(record["reviewerID"]), str(record["asin"]), record["unixReviewTime"], record["overall"]))
    return pd.DataFrame(rows, columns=["user", "item", "timestamp", "rating"])


def preprocess(frame, core=5):
    if core < 3:
        raise ValueError("core must be at least 3 for leave-one-out")
    frame = frame[["user", "item", "timestamp", "rating"]].copy()
    if frame.isna().any().any():
        raise ValueError("Missing interaction fields")
    frame["user"] = frame.user.astype(str)
    frame["item"] = frame.item.astype(str)
    timestamps = pd.to_numeric(frame.timestamp, errors="raise")
    ratings = pd.to_numeric(frame.rating, errors="raise")
    if not np.isfinite(timestamps).all() or not np.isfinite(ratings).all():
        raise ValueError("Nonfinite interaction fields")
    if (timestamps < 0).any() or (timestamps % 1 != 0).any() or not ratings.between(1, 5).all():
        raise ValueError("Invalid timestamp or rating")
    frame["timestamp"] = timestamps.astype("int64")
    frame["rating"] = ratings
    # One event per user/item, earliest occurrence; rating breaks exact-event ties.
    frame = frame.sort_values(["user", "timestamp", "item", "rating"], kind="stable")
    frame = frame.drop_duplicates(["user", "item"], keep="first")
    while len(frame):
        keep = (frame.groupby("user").user.transform("size") >= core) & (frame.groupby("item").item.transform("size") >= core)
        if keep.all():
            break
        frame = frame.loc[keep]
    if frame.empty:
        raise ValueError("No interactions remain after iterative core filtering")
    users = {key: index + 1 for index, key in enumerate(sorted(frame.user.unique()))}
    items = {key: index + 1 for index, key in enumerate(sorted(frame.item.unique()))}
    frame["user_id"] = frame.user.map(users)
    frame["item_id"] = frame.item.map(items)
    frame = frame.sort_values(["user_id", "timestamp", "item_id"], kind="stable").reset_index(drop=True)
    frame["event_id"] = np.arange(len(frame), dtype=np.int64)
    return frame, users, items


def load_metadata(path, items):
    selected = {}
    for record in records(path):
        asin = str(record.get("asin", ""))
        if asin in items:
            value = {key: record[key] for key in ("title", "description", "brand", "categories") if key in record}
            # Deterministic choice among duplicate metadata records.
            encoded = json.dumps(value, sort_keys=True)
            if asin not in selected or encoded < selected[asin]:
                selected[asin] = encoded
    return [{"item_id": index, "asin": asin, "metadata_missing": asin not in selected,
             **json.loads(selected.get(asin, "{}"))} for asin, index in items.items()]
