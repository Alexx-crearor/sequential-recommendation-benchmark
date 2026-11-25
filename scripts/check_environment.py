"""Record actual CPU/CUDA tensor execution and software inventory."""
import argparse
import platform
import sys
from importlib.metadata import version
from pathlib import Path
import torch
import _bootstrap
from src.utils.artifacts import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/reports/environment_check.json"))
    args = parser.parse_args()
    result = {"python": sys.version, "platform": platform.platform(), "packages": {key: version(key) for key in ["torch", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "PyYAML", "pytest", "tqdm", "pyarrow"]}, "devices": []}
    for device in ["cpu"] + [f"cuda:{index}" for index in range(torch.cuda.device_count())]:
        x = torch.ones(4, 4, device=device, requires_grad=True)
        (x @ x).sum().backward()
        result["devices"].append({"device": device, "forward_backward_finite": bool(torch.isfinite(x.grad).all()),
                                  "name": torch.cuda.get_device_name(device) if device.startswith("cuda") else platform.processor(),
                                  "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.startswith("cuda") else None})
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
