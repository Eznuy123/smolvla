"""Configure BF16 autocast before entering the upstream LeRobot trainer."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.runtime import configure_runtime

configure_runtime()


if __name__ == "__main__":
    import torch

    # SmolVLA checkpoints contain BF16 parameters. FP16 GradScaler cannot
    # unscale their BF16 gradients; BF16 autocast avoids that scaler entirely.
    torch.set_autocast_dtype("cuda", torch.bfloat16)

    from lerobot.scripts.lerobot_train import main

    main()
