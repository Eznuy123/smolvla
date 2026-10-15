import numpy as np

# gym-aloha 返回的 pixels 键名 -> 统一的相机名
# gym-aloha 0.1.5 的 Gym 观测目前只返回 top。
_PIXEL_KEY_MAP = {
    "cam_high": "top",
    "cam_left_wrist": "left_wrist",
    "cam_right_wrist": "right_wrist",
}

DEFAULT_CAMERAS = ("cam_high",)


class AlohaEnv:
    """gym-aloha 的薄封装,暴露 VLA 策略需要的观测格式。

    obs = {
        "images": {"cam_high": (H, W, 3) uint8, ...},
        "state":  (14,) float32,   # 双臂关节位置,左臂 7 + 右臂 7
    }
    action = (14,) float32,        # 每臂 6 个绝对关节目标 + 1 个归一化夹爪目标
    """

    def __init__(
        self,
        task: str = "gym_aloha/AlohaTransferCube-v0",
        seed: int = 0,
        cameras: tuple = DEFAULT_CAMERAS,
        max_episode_steps: int | None = None,
    ):
        try:
            import gym_aloha  # noqa: F401
        except ImportError as e:
            raise ImportError("缺少 gym-aloha,请先运行: pip install gym-aloha") from e
        import gymnasium

        kwargs = {} if max_episode_steps is None else {"max_episode_steps": max_episode_steps}
        self._gym = gymnasium.make(task, obs_type="pixels_agent_pos", **kwargs)
        self._seed = seed
        self._cameras = cameras
        self.episode_max_reward = 0.0

    @property
    def action_dim(self) -> int:
        return int(np.prod(self._gym.action_space.shape))

    @property
    def physics(self):
        """Expose the existing dm_control physics for a MuJoCo passive viewer."""
        return self._gym.unwrapped._env.physics

    @property
    def control_timestep(self) -> float:
        return float(self._gym.unwrapped._env.control_timestep())

    def reset(self, seed: int | None = None) -> dict:
        gym_obs, _ = self._gym.reset(seed=seed if seed is not None else self._seed)
        self.episode_max_reward = 0.0
        return self._convert(gym_obs)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError(f"动作必须是有限的 ({self.action_dim},) 向量，实际为 {action.shape}")
        gym_obs, reward, terminated, truncated, info = self._gym.step(action)
        reward = float(reward)
        self.episode_max_reward = max(self.episode_max_reward, reward)
        info = {**info, "terminated": bool(terminated), "truncated": bool(truncated)}
        return self._convert(gym_obs), reward, bool(terminated or truncated), info

    def close(self) -> None:
        self._gym.close()

    def _convert(self, gym_obs: dict) -> dict:
        pixels = gym_obs["pixels"]
        images = {}
        for name in self._cameras:
            key = _PIXEL_KEY_MAP[name]
            if key not in pixels:
                raise ValueError(f"相机 {name} 不可用；底层提供: {list(pixels)}")
            images[name] = np.asarray(pixels[key])
        return {
            "images": images,
            "state": np.asarray(gym_obs["agent_pos"], dtype=np.float32),
        }
