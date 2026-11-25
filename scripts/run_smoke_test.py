"""Synthetic fixtures are tests only and never enter research results."""
import argparse
import subprocess
import sys
import _bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.device == "cuda":
        try:
            import torch
        except ImportError:
            parser.exit(2, "CUDA smoke unavailable: install CUDA-enabled PyTorch; Phase 3 neural tests are pending.\n")
        if not torch.cuda.is_available():
            parser.exit(2, "CUDA smoke unavailable: no usable CUDA device.\n")
        print(f"CUDA execution: {torch.cuda.get_device_name(0)}", flush=True)
    marker = "cuda" if args.device == "cuda" else "not cuda"
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", marker], cwd=_bootstrap.ROOT)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
