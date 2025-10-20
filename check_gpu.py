"""
GPU readiness smoke test for quantized model experiments.

This script checks CUDA visibility through PyTorch and verifies that the
bitsandbytes wheel bundles the expected CUDA kernels.  Run it with:

    uv run python check_gpu.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def check_torch() -> bool:
    try:
        import torch  # type: ignore
    except Exception as exc:
        print("✗ PyTorch import failed:", exc)
        return False

    print("PyTorch version        :", torch.__version__)
    print("  CUDA available       :", torch.cuda.is_available())

    if not torch.cuda.is_available():
        return False

    print("  CUDA runtime version :", torch.version.cuda)
    print("  CUDA device count    :", torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        print(f"  Device {idx}            :", torch.cuda.get_device_name(idx))

    # Lightweight tensor op to ensure kernels can be launched.
    try:
        x = torch.ones((2, 2), device="cuda")
        print("  Tensor op check      :", (x @ x).sum().item())
    except Exception as exc:  # pragma: no cover - diagnostic path
        print("  ✗ Tensor op failed   :", exc)
        return False

    return True


def check_bitsandbytes() -> bool:
    try:
        import bitsandbytes as bnb  # type: ignore
    except Exception as exc:
        print("✗ bitsandbytes import failed:", exc)
        print("  -> run `uv sync` to install dependencies")
        return False

    print("bitsandbytes version   :", getattr(bnb, "__version__", "unknown"))

    package_dir = Path(bnb.__file__).resolve().parent
    print("  package location     :", package_dir)

    cuda_libs = sorted(p.name for p in package_dir.glob("libbitsandbytes_cuda*.so"))
    if cuda_libs:
        print("  CUDA kernels present :", ", ".join(cuda_libs))
    else:
        print("  ✗ No CUDA kernels found under bitsandbytes package")

    try:
        cuda_ready = bool(bnb.cuda.is_available())
    except Exception as exc:  # pragma: no cover - diagnostic path
        print("  ✗ bnb.cuda.is_available raised:", exc)
        cuda_ready = False
    else:
        print("  cuda.is_available    :", cuda_ready)

    # bitsandbytes exposes troubleshooting via `python -m bitsandbytes`.
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "bitsandbytes"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - diagnostic path
        print("  ✗ `python -m bitsandbytes` failed:", exc.output.strip())
    except subprocess.TimeoutExpired:  # pragma: no cover - diagnostic path
        print("  ! `python -m bitsandbytes` timed out")
    else:
        snapshot = "\n      ".join(output.strip().splitlines())
        print("  diagnostics:\n     ", snapshot)

    return cuda_ready and bool(cuda_libs)


def show_environment():
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        print("CUDA toolkit path      :", cuda_home)
    ld_path = os.environ.get("LD_LIBRARY_PATH")
    if ld_path:
        print("LD_LIBRARY_PATH        :", ld_path)


def main() -> None:
    show_environment()
    torch_ok = check_torch()
    bnb_ok = check_bitsandbytes()

    ready = torch_ok and bnb_ok
    print("\nOverall readiness      :", "✓ Ready" if ready else "✗ Fix required")


if __name__ == "__main__":
    main()

