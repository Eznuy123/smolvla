import numpy as np


class Policy:
    """obs -> action 的统一接口。

    obs: {"images": {name: (H, W, 3) uint8}, "state": (D,) float32}
    返回: (action_dim,) float32
    """

    def reset(self) -> None:
        pass

    def infer(self, obs: dict) -> np.ndarray:
        raise NotImplementedError


class RandomPolicy(Policy):
    """仅用于打通闭环的占位策略。"""

    def __init__(self, action_dim: int = 14, scale: float = 0.05, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self.action_dim = action_dim
        self.scale = scale

    def infer(self, obs: dict) -> np.ndarray:
        return self._rng.uniform(-self.scale, self.scale, size=self.action_dim).astype(np.float32)
