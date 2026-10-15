"""Launch LeRobot SmolVLA fine-tuning with defaults for an 8 GiB GPU."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.runtime import configure_runtime, require_device

configure_runtime()


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("必须大于 0")
    return result


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).with_name("_train_backend.py")),
        f"--dataset.repo_id={args.dataset_repo_id}",
        "--dataset.video_backend=pyav",
        f"--policy.path={args.pretrained}",
        f"--policy.device={args.device}",
        f"--policy.use_amp={str(args.device.startswith('cuda')).lower()}",
        "--policy.push_to_hub=false",
        # Derive the 14-D state/action and top-camera keys from this dataset,
        # replacing the base checkpoint's 6-D robot and camera1/2/3 schema.
        # LeRobot merges dict overrides, so {} retains the base schema.
        # null clears it and lets make_policy infer features from the dataset.
        "--policy.input_features=null",
        "--policy.output_features=null",
        "--policy.empty_cameras=0",
        # The full SmolVLA checkpoint already includes its VLM weights.
        "--policy.load_vlm_weights=false",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--num_workers={args.num_workers}",
        f"--output_dir={args.out}",
        "--job_name=smolvla_aloha",
        f"--seed={args.seed}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        "--env_eval_freq=0",
        "--wandb.enable=false",
    ]
    if args.dataset_root is not None:
        command.append(f"--dataset.root={args.dataset_root}")
    if args.episodes is not None:
        command.append(f"--dataset.episodes={json.dumps(args.episodes)}")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="用 LeRobot 微调 SmolVLA（默认单卡 8GB）")
    parser.add_argument("--dataset-repo-id", default="lerobot/aloha_sim_transfer_cube_scripted")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--pretrained", default="lerobot/smolvla_base")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--steps", type=positive_int, default=20_000)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--n-action-steps", type=positive_int, default=10,
                        help="每次预测后连续执行的动作数，默认 10；预测长度仍为 50")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--episodes", nargs="+", type=int, help="只使用指定 episode，例如 0 1")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--save-freq", type=positive_int, default=1000)
    parser.add_argument("--log-freq", type=positive_int, default=10)
    parser.add_argument("--out", type=Path, default=Path("out/train/smolvla_aloha"))
    parser.add_argument("--log-file", type=Path, help="训练日志；默认保存到输出目录旁的同名 .log")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不下载或训练")
    args = parser.parse_args()
    if args.num_workers < 0 or (args.episodes is not None and any(ep < 0 for ep in args.episodes)):
        parser.error("num-workers 和 episode 索引不能为负数")
    if not args.dry_run and args.out.exists():
        parser.error("输出目录已存在；请用 --out 指定新目录，或按 README 使用 LeRobot 断点续训")
    command = build_command(args)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    require_device(args.device)
    log_path = args.log_file or args.out.parent / f"{args.out.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Training log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
