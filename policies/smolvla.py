"""Local SmolVLA inference using the checkpoint's saved processors."""

import numpy as np

from .base import Policy
from utils.runtime import configure_runtime, require_device


class SmolVLAPolicy(Policy):
    def __init__(self, checkpoint: str, device: str = "cuda", prompt: str = "",
                 n_action_steps: int = 10):
        if n_action_steps < 1:
            raise ValueError("n_action_steps 必须大于 0")
        configure_runtime()
        require_device(device)
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as LeRobotSmolVLA

        self._torch = torch
        self._device = torch.device(device)
        self._prompt = prompt
        config = PreTrainedConfig.from_pretrained(checkpoint)
        if config.type != "smolvla":
            raise ValueError(f"需要 SmolVLA checkpoint，实际为 {config.type}")
        config.device = device
        config.load_vlm_weights = False
        if n_action_steps > config.chunk_size:
            raise ValueError(f"执行长度 {n_action_steps} 不能超过预测长度 {config.chunk_size}")
        # Set before constructing the model: reset() allocates its action queue
        # using this value, so old 50-step checkpoints also replan every 10 steps.
        config.n_action_steps = n_action_steps
        self.n_action_steps = config.n_action_steps
        self.chunk_size = config.chunk_size
        state = config.robot_state_feature
        action = config.action_feature
        if state is None or tuple(state.shape) != (14,) or action is None or tuple(action.shape) != (14,):
            raise ValueError("需要已适配 ALOHA 14 维状态/动作的 checkpoint；请先运行 scripts/train.py 微调")
        self._policy = LeRobotSmolVLA.from_pretrained(checkpoint, config=config).to(device).eval()
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self._use_amp = config.use_amp and self._device.type == "cuda"

    def reset(self) -> None:
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    def infer(self, obs: dict) -> np.ndarray:
        torch = self._torch
        batch = {
            "observation.state": torch.from_numpy(np.asarray(obs["state"], dtype=np.float32)),
            "task": self._prompt,
        }
        camera_map = {
            "cam_high": "observation.images.top",
            "cam_left_wrist": "observation.images.left_wrist",
            "cam_right_wrist": "observation.images.right_wrist",
        }
        for camera, image in obs["images"].items():
            key = camera_map.get(camera, f"observation.images.{camera}")
            batch[key] = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
        required = {k for k in self._policy.config.image_features if ".empty_camera_" not in k}
        if missing := required.difference(batch):
            raise ValueError(f"checkpoint 要求的相机缺失: {sorted(missing)}")
        with torch.inference_mode(), torch.autocast(
            self._device.type, dtype=torch.bfloat16, enabled=self._use_amp
        ):
            batch = self._preprocessor(batch)
            action = self._postprocessor(self._policy.select_action(batch))
        result = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if result.shape != (14,) or not np.isfinite(result).all():
            raise ValueError(f"策略返回了无效动作: shape={result.shape}")
        return result
