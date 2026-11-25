"""Short CPU/CUDA operator traces with explicit input/transfer/compute regions."""
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.profiler import profile, ProfilerActivity, record_function
import _bootstrap
from src.utils.configuration import resolve_config
from src.utils.artifacts import verify_bundle, write_json
from src.models.factory import build_model
from src.data.neural_dataset import TrainingDataset
from src.training.loading import make_loader
from src.training.losses import objective_loss
from src.training.checkpoint import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", choices=["sasrec_ce", "fenrec"], default="sasrec_ce")
    parser.add_argument("--centers", type=Path)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Choose a new profiling output directory")
    args.output.mkdir(parents=True)
    verify_bundle(args.bundle)
    seed_everything(7); torch.set_num_threads(4)
    config = resolve_config(args.model, {"batch_size": 128, "pilot": True})
    train = pd.read_parquet(args.bundle / "train.parquet")
    num_items = int(pd.read_parquet(args.bundle / "interactions.parquet").item_id.max())
    dataset = TrainingDataset(train, num_items, config)
    loader = iter(make_loader(dataset, config, args.device))
    model = build_model(args.model, num_items, config).to(args.device).train()
    if args.model == "fenrec":
        if args.centers is None: parser.error("FENRec profiling requires measured training-only --centers")
        model.set_clusters(torch.load(args.centers, weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if args.device.startswith("cuda") else [])
    # Warm kernels and allocator outside trace.
    batch = {key: value.to(args.device) for key, value in next(loader).items()}
    loss, _ = objective_loss(model, batch, 21); loss.backward(); optimizer.step()
    with profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
        for step in range(3):
            with record_function("input_and_host_to_device"):
                batch = {key: value.to(args.device, non_blocking=True) for key, value in next(loader).items()}
            optimizer.zero_grad(set_to_none=True)
            with record_function("forward_backward_full_objective"):
                loss, _ = objective_loss(model, batch, 21)
                loss.backward()
            with record_function("optimizer_step"):
                optimizer.step()
    profiler.export_chrome_trace(str(args.output / "trace.json"))
    (args.output / "operators_cpu.txt").write_text(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=35))
    (args.output / "operators_device.txt").write_text(profiler.key_averages().table(sort_by="self_device_time_total", row_limit=35))
    rows = [{"operator": event.key, "calls": event.count, "cpu_total_us": event.cpu_time_total,
             "device_total_us": getattr(event, "device_time_total", 0)} for event in profiler.key_averages()]
    write_json(args.output / "operators.json", rows)
    print(args.output)


if __name__ == "__main__": main()
