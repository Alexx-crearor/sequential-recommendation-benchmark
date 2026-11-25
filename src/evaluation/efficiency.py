import time
import numpy as np
import torch


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def inference_profile(model, history, repeats=20, warmup=5):
    """Batch-1 full-catalog scoring latency, excluding rank sorting and disk I/O."""
    neural = hasattr(model, "representations")
    device = model.device if neural else "cpu"
    timings = []
    def predict():
        if neural:
            with torch.no_grad():
                rep = model.representations([history])
                for start in range(1, model.num_items+1, model.config["item_batch_size"]):
                    ids = torch.arange(start, min(model.num_items+1, start+model.config["item_batch_size"]), device=device)
                    model.logits(rep, ids)
        else:
            model.score_items(history, np.arange(1, model.num_items+1))
    if neural:
        model.eval()
    for _ in range(warmup):
        predict()
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        predict()
        synchronize(device)
        timings.append(time.perf_counter()-start)
    return {"batch1_mean_seconds": float(np.mean(timings)), "batch1_p95_seconds": float(np.percentile(timings, 95)),
            "repeats": repeats, "warmup": warmup, "scope": "history_encoding_and_full_catalogue_scoring_without_sorting"}
