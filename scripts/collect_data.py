import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from envs.aloha_env import AlohaEnv
from policies.base import Policy, RandomPolicy


class ScriptedPolicy(Policy):
    """占位:转移方块任务的脚本策略,用于自动生成演示数据。

    TODO: 基于关键点位移 + 简单夹爪时序实现,参考 ACT 的 sim_transfer_cube_scripted。
    也可以直接从 HuggingFace 下载现成数据集: lerobot/aloha_sim_transfer_cube_scripted
    """

    def infer(self, obs: dict) -> np.ndarray:
        raise NotImplementedError("脚本策略待实现,先用 --policy random 验证录制流程")


def make_policy(name: str, env) -> Policy:
    if name == "random":
        return RandomPolicy(action_dim=env.action_dim)
    if name == "scripted":
        return ScriptedPolicy()
    raise ValueError(f"未知策略: {name}")


def record_episode(env: AlohaEnv, policy: Policy, max_steps: int) -> dict:
    obs = env.reset()
    policy.reset()
    data = {f"image/{k}": [] for k in obs["images"]}
    data["state"] = []
    data["action"] = []
    for _ in range(max_steps):
        action = policy.infer(obs)
        for k, v in obs["images"].items():
            data[f"image/{k}"].append(v)
        data["state"].append(obs["state"])
        data["action"].append(action)
        obs, _, done, _ = env.step(action)
        if done:
            break
    return {k: np.stack(v) for k, v in data.items() if len(v) > 0}, done


def main():
    parser = argparse.ArgumentParser(description="阶段2:录制演示数据(npz 格式)")
    parser.add_argument("--policy", choices=["random", "scripted"], default="scripted")
    parser.add_argument("--task", default="gym_aloha/AlohaTransferCube-v0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/episodes"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    env = AlohaEnv(task=args.task)
    policy = make_policy(args.policy, env)

    meta = {"task": args.task, "policy": args.policy, "episodes": []}
    for ep in range(args.episodes):
        episode, done = record_episode(env, policy, args.max_steps)
        path = args.out / f"episode_{ep:04d}.npz"
        np.savez_compressed(path, **episode)
        meta["episodes"].append({"file": path.name, "steps": len(episode["state"]), "finished": bool(done)})
        print(f"episode {ep}: {len(episode['state'])} 步, 已保存 {path}")

    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"共 {args.episodes} 条,元数据: {args.out / 'meta.json'}")


if __name__ == "__main__":
    main()
