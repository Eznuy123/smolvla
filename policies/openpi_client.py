import numpy as np

from .base import Policy


class OpenPIPolicy(Policy):
    """openpi policy server 的 websocket 客户端(阶段3:本地或云端 GPU 上跑 serve_policy.py)。"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        image_size: int = 224,
        prompt: str = "",
        image_key_map: dict | None = None,
    ):
        from openpi_client import image_tools
        from openpi_client import websocket_client_policy

        self._image_tools = image_tools
        self._client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self._image_size = image_size
        self._prompt = prompt
        # 图像键名必须与所服务策略的训练配置(如 AlohaInputs)匹配
        self._image_key_map = image_key_map or {}

    def infer(self, obs: dict) -> np.ndarray:
        observation = {}
        for name, img in obs["images"].items():
            img = self._image_tools.convert_to_uint8(
                self._image_tools.resize_with_pad(img, self._image_size, self._image_size)
            )
            key = self._image_key_map.get(name, f"observation/image/{name}")
            observation[key] = np.transpose(img, (2, 0, 1))
        observation["observation/state"] = obs["state"]
        observation["prompt"] = self._prompt
        result = self._client.infer(observation)
        return np.asarray(result["actions"][0], dtype=np.float32)
