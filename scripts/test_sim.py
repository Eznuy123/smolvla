import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.runtime import configure_runtime

configure_runtime()

import imageio.v3 as iio

from envs.aloha_env import AlohaEnv


def main():
    parser = argparse.ArgumentParser(description="阶段0:仿真冒烟测试 —— 相机出图 + 物理稳定")
    parser.add_argument("--task", default="gym_aloha/AlohaTransferCube-v0")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out/smoke_test"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    env = AlohaEnv(task=args.task)
    obs = env.reset()
    print(f"state: {obs['state'].shape} {obs['state'].dtype}")
    print(f"action_dim: {env.action_dim}")

    for name, img in obs["images"].items():
        path = args.out / f"{name}.png"
        iio.imwrite(path, img)
        print(f"saved {path}  shape={img.shape} dtype={img.dtype}")
    if not obs["images"]:
        print("警告: 没有渲染出任何相机图像,请检查 MUJOCO_GL 设置")

    for i in range(args.steps):
        action = np.random.uniform(-0.05, 0.05, size=env.action_dim).astype(np.float32)
        obs, reward, done, _ = env.step(action)
        if np.isnan(obs["state"]).any():
            raise RuntimeError(f"step {i}: 状态出现 NaN,物理发散")
        if i % 50 == 0:
            print(f"step {i:4d}  reward={reward:+.3f}  max|qpos|={np.abs(obs['state']).max():.2f}")
        if done:
            print(f"episode 结束于 step {i} (max_reward={env.episode_max_reward:.3f}),reset")
            obs = env.reset()

    print("冒烟测试通过:物理稳定,无 NaN")


if __name__ == "__main__":
    main()
