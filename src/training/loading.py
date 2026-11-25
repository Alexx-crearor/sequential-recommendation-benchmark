"""Recreated epoch loaders make worker RNG streams resumable, including Windows spawn."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, config, device):
    workers = config.get("num_workers", 0)
    kwargs = {"batch_size": config["batch_size"], "shuffle": True, "num_workers": workers,
              "pin_memory": config.get("pin_memory", True) and str(device).startswith("cuda"),
              "worker_init_fn": seed_worker if workers else None}
    if workers:
        kwargs["prefetch_factor"] = config.get("prefetch_factor", 2)
    # Workers are recreated at epoch boundaries. Persistent workers would need
    # their private transform RNG states in the checkpoint to resume exactly.
    return DataLoader(dataset, **kwargs)
