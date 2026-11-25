import os
import random
import numpy as np
import torch


def atomic_torch_save(path, value):
    path = str(path)
    torch.save(value, path + ".partial")
    os.replace(path + ".partial", path)


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def rng_state(device=None):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "active_cuda": torch.cuda.get_rng_state(device) if device is not None and str(device).startswith("cuda") else None}


def restore_rng(state, device=None):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        for index, value in enumerate(state["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(value.cpu(), index)
    if device is not None and str(device).startswith("cuda") and state.get("active_cuda") is not None:
        torch.cuda.set_rng_state(state["active_cuda"].cpu(), device)
