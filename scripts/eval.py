import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.runtime import configure_runtime

configure_runtime()

import imageio

from envs.aloha_env import AlohaEnv
from policies.base import RandomPolicy
from policies.openpi_client import OpenPIPolicy


def make_policy(name: str, args, env) -> "object":
    if name == "random":
        return RandomPolicy(action_dim=env.action_dim, seed=args.seed)
    if name == "openpi":
        return OpenPIPolicy(host=args.host, port=args.port, prompt=args.prompt)
    if name == "smolvla":
        from policies.smolvla import SmolVLAPolicy

        return SmolVLAPolicy(checkpoint=args.checkpoint, device=args.device, prompt=args.prompt,
                            n_action_steps=args.n_action_steps)
    raise ValueError(f"未知策略: {name}")


def main():
    parser = argparse.ArgumentParser(description="闭环评测:策略 ↔ 仿真")
    parser.add_argument("--policy", choices=["random", "openpi", "smolvla"], default="random")
    parser.add_argument("--checkpoint", help="微调后的 pretrained_model 目录或 HF 模型 ID")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--n-action-steps", type=int, default=10,
                        help="SmolVLA 每次连续执行的动作数，默认 10（0.2 秒仿真时间）")
    parser.add_argument("--task", default="gym_aloha/AlohaTransferCube-v0")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="localhost", help="openpi policy server 地址")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="Pick up the cube with the right arm and transfer it to the left arm.",
                        help="语言指令，建议与训练数据一致")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out/eval"))
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1 or args.n_action_steps < 1:
        parser.error("episodes、max-steps 和 n-action-steps 必须大于 0")
    if args.policy == "smolvla" and not args.checkpoint:
        parser.error("--policy smolvla 必须指定 --checkpoint")

    # Load the model before creating rendering resources.
    np.random.seed(args.seed)
    if args.policy == "smolvla":
        import torch

        torch.manual_seed(args.seed)
    env = None
    results = []
    try:
        if args.policy == "smolvla":
            policy = make_policy(args.policy, args, None)
        env = AlohaEnv(task=args.task, seed=args.seed, max_episode_steps=args.max_steps)
        if args.policy != "smolvla":
            policy = make_policy(args.policy, args, env)
        args.out.mkdir(parents=True, exist_ok=True)
        for ep in range(args.episodes):
            obs = env.reset(seed=args.seed + ep)
            policy.reset()
            video_path = args.out / f"episode_{ep:03d}.mp4"
            with imageio.get_writer(video_path, fps=50) as writer:
                for step in range(args.max_steps):
                    action = policy.infer(obs)
                    obs, reward, done, info = env.step(action)
                    writer.append_data(obs["images"]["cam_high"])
                    if done:
                        break
            result = {
                "episode": ep,
                "seed": args.seed + ep,
                "steps": step + 1,
                "max_reward": env.episode_max_reward,
                "is_success": bool(info.get("is_success", False)),
                "terminated": info["terminated"],
                "truncated": info["truncated"],
                "video": video_path.name,
            }
            results.append(result)
            print(f"episode {ep}: {result['steps']} 步, max_reward={result['max_reward']:.3f}, "
                  f"is_success={result['is_success']}; saved {video_path}")
        summary = {
            "policy": args.policy,
            "checkpoint": args.checkpoint,
            "n_action_steps": policy.n_action_steps if args.policy == "smolvla" else None,
            "chunk_size": policy.chunk_size if args.policy == "smolvla" else None,
            "task": args.task,
            "prompt": args.prompt,
            "max_steps": args.max_steps,
            "success_rate": sum(r["is_success"] for r in results) / len(results),
            "episodes": results,
        }
        (args.out / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"success_rate={summary['success_rate']:.1%}; results: {args.out / 'results.json'}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
