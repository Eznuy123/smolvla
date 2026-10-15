import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_runtime() -> None:
    # Call before importing Hugging Face, LeRobot or MuJoCo.
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data/hf_cache"))
    os.environ.setdefault("HF_LEROBOT_HOME", str(PROJECT_ROOT / "data/lerobot"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def require_device(device: str) -> None:
    import torch

    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "请求了 CUDA，但当前进程无法访问 NVIDIA GPU。"
            "请先在宿主机运行 uv run python scripts/check_env.py；"
            "若使用容器或沙箱，需要开放 GPU 设备访问。"
        )
