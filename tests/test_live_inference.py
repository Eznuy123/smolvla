from contextlib import nullcontext
import unittest
from unittest.mock import patch

import numpy as np

from scripts.infer import LiveControls, run_live_loop


class FakeEnv:
    control_timestep = 0.02

    def __init__(self):
        self.seeds = []
        self.total_steps = 0

    def reset(self, seed):
        self.seeds.append(seed)
        self.steps = 0
        self.episode_max_reward = 0.0
        return {"state": np.zeros(14)}

    def step(self, action):
        self.steps += 1
        self.total_steps += 1
        return {"state": np.full(14, self.steps)}, 0.0, self.steps == 3, {"is_success": False}


class FakePolicy:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def infer(self, obs):
        return np.zeros(14)


class FakeViewer:
    def __init__(self):
        self.running = True
        self.on_sync = lambda: None

    def lock(self):
        return nullcontext()

    def is_running(self):
        return self.running

    def sync(self):
        self.on_sync()


class LiveInferenceTests(unittest.TestCase):
    @patch("scripts.infer.time.sleep")
    def test_pause_holds_physics_and_next_scene_resets_before_resuming(self, sleep):
        env, policy, viewer, controls = FakeEnv(), FakePolicy(), FakeViewer(), LiveControls()
        controls.on_key(32)
        syncs = []

        def resume_on_paused_sync():
            syncs.append(True)
            if len(syncs) == 2:
                self.assertEqual(env.total_steps, 0)
                controls.on_key(ord("N"))
                controls.on_key(32)

        viewer.on_sync = resume_on_paused_sync
        results = run_live_loop(env, policy, viewer, controls, seed=7, episodes=1)
        self.assertEqual(env.seeds, [7, 8])
        self.assertEqual(policy.resets, 2)
        self.assertEqual(results[0]["seed"], 8)
        self.assertEqual(env.total_steps, 3)

    @patch("scripts.infer.time.sleep")
    def test_completed_episodes_advance_seed_and_reset_policy_queue(self, sleep):
        env, policy, viewer = FakeEnv(), FakePolicy(), FakeViewer()
        results = run_live_loop(env, policy, viewer, LiveControls(), seed=7, episodes=2)
        self.assertEqual(env.seeds, [7, 8])
        self.assertEqual(env.total_steps, 6)
        self.assertEqual(policy.resets, 2)
        self.assertEqual([result["steps"] for result in results], [3, 3])

    @patch("scripts.infer.time.sleep")
    def test_manual_reset_discards_previous_actions_in_same_scene(self, sleep):
        env, policy, viewer, controls = FakeEnv(), FakePolicy(), FakeViewer(), LiveControls()
        requested = []

        def reset_once():
            if env.total_steps == 1 and not requested:
                controls.on_key(ord("R"))
                requested.append(True)

        viewer.on_sync = reset_once
        run_live_loop(env, policy, viewer, controls, seed=7, episodes=1)
        self.assertEqual(env.seeds, [7, 7])
        self.assertEqual(policy.resets, 2)
        self.assertEqual(env.total_steps, 4)

    def test_closing_during_prediction_does_not_step_physics(self):
        env, policy, viewer = FakeEnv(), FakePolicy(), FakeViewer()

        def predict_then_close(obs):
            viewer.running = False
            return np.zeros(14)

        policy.infer = predict_then_close
        results = run_live_loop(env, policy, viewer, LiveControls(), episodes=1)
        self.assertEqual(results, [])
        self.assertEqual(env.total_steps, 0)


if __name__ == "__main__":
    unittest.main()
