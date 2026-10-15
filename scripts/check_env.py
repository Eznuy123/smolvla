"""Check the uv environment and execute a small CUDA operation."""

import argparse
from importlib import metadata
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.runtime import configure_runtime

configure_runtime()


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 Python、训练依赖和 GPU")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    print(f"Python: {sys.executable}")
    for name in ("lerobot", "torch", "torchvision", "gym-aloha"):
        print(f"{name}: {metadata.version(name)}")

    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401
    from lerobot.scripts.lerobot_train import train  # noqa: F401

    print(f"SmolVLA / training imports: OK; CUDA build: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("CUDA: 不可用；检查驱动和当前进程的 GPU 访问权限")
        if args.require_cuda:
            raise SystemExit(1)
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
    value = torch.ones((32, 32), device="cuda")
    assert (value @ value).sum().item() == 32768
    print("CUDA tensor / matmul: OK")


if __name__ == "__main__":
    main()
