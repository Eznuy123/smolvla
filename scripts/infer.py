"""Run a policy online in an interactive MuJoCo window."""

import argparse
import math
import os
from pathlib import Path
import sys
from threading import Event
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Select the window backend before MuJoCo/dm_control are imported.
os.environ.setdefault("MUJOCO_GL", "glfw")

from utils.runtime import configure_runtime

configure_runtime()

import numpy as np

from envs.aloha_env import AlohaEnv
from scripts.eval import make_policy


class LiveControls:
    """Callbacks signal the main thread; only the main thread changes physics."""

    def __init__(self):
        self.paused = Event()
        self.reset_requested = Event()
        self.next_requested = Event()
        self.quit_requested = Event()

    def on_key(self, keycode: int) -> None:
        if keycode == 32:  # Space
            if self.paused.is_set():
                self.paused.clear()
            else:
                self.paused.set()
        elif keycode in (ord("R"), ord("r")):
            self.reset_requested.set()
        elif keycode in (ord("N"), ord("n")):
            self.next_requested.set()
        elif keycode == 256:  # GLFW Escape
            self.quit_requested.set()


def run_live_loop(env, policy, viewer, controls, *, seed=0, episodes=0, speed=1.0):
    """Advance the same physics shown in the viewer; pace toward real time.

    Inference is synchronous. If prediction/rendering is slow, simulation slows
    down rather than skipping actions or changing the trained control timestep.
    """
    results = []
    steps = 0
    obs = None
    reset_pending = True
    started = time.perf_counter()
    while viewer.is_running() and not controls.quit_requested.is_set():
        tick = time.perf_counter()
        if controls.next_requested.is_set():
            controls.next_requested.clear()
            seed += 1
            reset_pending = True
        if controls.reset_requested.is_set():
            controls.reset_requested.clear()
            reset_pending = True
        if reset_pending:
            with viewer.lock():
                obs = env.reset(seed=seed)
            policy.reset()  # Discard actions predicted for the previous scene.
            steps = 0
            started = time.perf_counter()
            reset_pending = False
            status = "paused" if controls.paused.is_set() else "running"
            print(f"scene seed={seed}; {status}", flush=True)
            viewer.sync()
        if controls.paused.is_set():
            viewer.sync()
            time.sleep(0.02)
            continue

        action = policy.infer(obs)
        # The window may have closed while the GPU was predicting a chunk.
        if not viewer.is_running() or controls.quit_requested.is_set():
            break
        with viewer.lock():
            obs, reward, done, info = env.step(action)
        steps += 1
        viewer.sync()
        if steps % 50 == 0 or done:
            elapsed = max(time.perf_counter() - started, 1e-9)
            factor = steps * env.control_timestep / elapsed
            print(f"step={steps} reward={reward:.0f} max_reward={env.episode_max_reward:.0f} "
                  f"sim/wall={factor:.2f}x", flush=True)
        if done:
            result = {"seed": seed, "steps": steps, "max_reward": env.episode_max_reward,
                      "is_success": bool(info.get("is_success", False))}
            results.append(result)
            print(f"episode {len(results)}: {result}", flush=True)
            if episodes and len(results) >= episodes:
                break
            seed += 1
            reset_pending = True

        delay = env.control_timestep / speed - (time.perf_counter() - tick)
        if delay > 0:
            time.sleep(delay)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="在 MuJoCo 窗口中在线推理")
    parser.add_argument("--policy", choices=["smolvla", "random", "openpi"], default="smolvla")
    parser.add_argument("--checkpoint", default="out/train/smolvla_aloha_20k_a10/checkpoints/last/pretrained_model")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--task", default="gym_aloha/AlohaTransferCube-v0")
    parser.add_argument("--prompt", default="Pick up the cube with the right arm and transfer it to the left arm.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--episodes", type=int, default=0, help="完成几次后退出；0 表示持续运行直到关窗")
    parser.add_argument("--speed", type=float, default=1.0, help="目标播放速度，例如 0.5 为慢速")
    parser.add_argument("--camera", default="free", help="free 为可旋转视角，也可选 top、angle")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.max_steps < 1 or args.n_action_steps < 1 or args.episodes < 0:
        parser.error("max-steps 和 n-action-steps 必须大于 0，episodes 不能为负数")
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("speed 必须是有限正数")
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        parser.error("没有桌面显示会话；请在本机桌面终端运行，离屏评测用 scripts/eval.py")
    if args.policy == "smolvla":
        if not Path(args.checkpoint).is_dir() and (
            args.checkpoint.startswith(("out/", "/", "./", "../"))
        ):
            parser.error(f"checkpoint 目录不存在: {args.checkpoint}")
        import torch

        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import mujoco
    import mujoco.viewer

    env = None
    try:
        if args.policy == "smolvla":
            print(f"Loading checkpoint: {args.checkpoint}; n_action_steps={args.n_action_steps}", flush=True)
            policy = make_policy(args.policy, args, None)
        env = AlohaEnv(task=args.task, seed=args.seed, max_episode_steps=args.max_steps)
        if args.policy != "smolvla":
            policy = make_policy(args.policy, args, env)
        controls = LiveControls()
        physics = env.physics
        camera_id = -1
        if args.camera != "free":
            camera_id = mujoco.mj_name2id(physics.model.ptr, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
            if camera_id < 0:
                parser.error(f"场景中不存在相机: {args.camera}")
        print("Space: 暂停/继续 | R: 重置当前场景 | N: 下一个种子 | Esc/关窗/Ctrl+C: 退出", flush=True)
        with mujoco.viewer.launch_passive(physics.model.ptr, physics.data.ptr,
                                         key_callback=controls.on_key) as viewer:
            with viewer.lock():
                if camera_id >= 0:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = camera_id
                else:
                    viewer.cam.lookat[:] = [0.0, 0.6, 0.1]
                    viewer.cam.distance = 1.5
                    viewer.cam.azimuth = 90
                    viewer.cam.elevation = -35
            results = run_live_loop(env, policy, viewer, controls, seed=args.seed,
                                    episodes=args.episodes, speed=args.speed)
        if results:
            success = sum(result["is_success"] for result in results)
            print(f"已完成 {len(results)} 次，成功 {success} 次", flush=True)
    except KeyboardInterrupt:
        print("在线推理已停止", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
