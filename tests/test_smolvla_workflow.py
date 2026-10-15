"""Offline checks for the schema and normalization boundaries of this workflow."""

import argparse
from pathlib import Path
import tempfile
from types import MethodType
import unittest

import numpy as np
import torch

from utils.runtime import configure_runtime

configure_runtime()

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as LeRobotSmolVLA
from lerobot.processor import make_default_policy_processor_steps, make_policy_processor_pipelines

from policies.smolvla import SmolVLAPolicy
from scripts.train import build_command
from scripts.summarize_train import read_metrics


class WorkflowTests(unittest.TestCase):
    def test_ten_step_execution_replans_and_reset_discards_previous_actions(self):
        # Exercise LeRobot's real queue with a deterministic 50-action predictor.
        # No model weights, tokenizers or GPU are needed for this boundary check.
        model = LeRobotSmolVLA.__new__(LeRobotSmolVLA)
        torch.nn.Module.__init__(model)
        model.config = SmolVLAConfig(device="cpu", chunk_size=50, n_action_steps=10)
        calls = []

        def predict(model_self, batch, noise=None):
            calls.append(len(calls))
            return (torch.arange(50) + 100 * calls[-1]).float().reshape(1, 50, 1).expand(1, 50, 14)

        model._get_action_chunk = MethodType(predict, model)
        model.reset()
        actions = [model.select_action({})[0, 0].item() for _ in range(21)]
        self.assertEqual(actions, list(range(10)) + list(range(100, 110)) + [200])
        self.assertEqual(len(calls), 3)
        model.reset()
        self.assertEqual(model.select_action({})[0, 0].item(), 300)

    def test_metrics_keep_exact_steps_when_lerobot_rounds_to_thousands(self):
        rows = read_metrics(
            "'log_freq': 10,\n"
            "Training: 78%|bar| 1570/2000 [03:20<00:52] INFO step:2K loss:0.151 lr:1.3e-05\n"
            "Training: 79%|bar| 1579/2000 [03:20<00:52] INFO step:2K loss:0.140 lr:1.3e-05\n"
            "INFO step:2K loss:0.120 lr:2.5e-06\n"
        )
        self.assertEqual([row["step"] for row in rows], [1570, 1580, 2000])
        self.assertEqual(rows[0]["loss"], 0.151)

    def test_training_clears_base_robot_schema(self):
        base = SmolVLAConfig(
            device="cpu",
            input_features={"observation.state": PolicyFeature(FeatureType.STATE, (6,))},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
        )
        args = argparse.Namespace(
            dataset_repo_id="test/aloha", pretrained="test/base", device="cpu",
            batch_size=1, steps=2, num_workers=0, out=Path("unused"), seed=0,
            save_freq=2, log_freq=1, dataset_root=None, episodes=[0],
            n_action_steps=10,
        )
        command = build_command(args)
        self.assertIn("--policy.n_action_steps=10", command)
        overrides = [arg.replace("--policy.", "--", 1) for arg in command
                     if arg.startswith(("--policy.input_features=", "--policy.output_features="))]
        with tempfile.TemporaryDirectory() as tmp:
            base.save_pretrained(tmp)
            restored = SmolVLAConfig.from_pretrained(tmp, cli_overrides=overrides)
        # Empty dicts would be merged with the 6-D schema instead of clearing it.
        self.assertIsNone(restored.input_features)
        self.assertIsNone(restored.output_features)

    def make_adapter(self):
        config = SmolVLAConfig(
            device="cpu",
            input_features={
                "observation.state": PolicyFeature(FeatureType.STATE, (14,)),
                "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 2, 2)),
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (14,))},
        )
        stats = {
            "observation.state": {"mean": np.ones(14), "std": np.full(14, 2.0)},
            "action": {"mean": np.arange(14, dtype=np.float32), "std": np.full(14, 3.0)},
        }
        steps = make_default_policy_processor_steps(config, stats)
        pre, post = make_policy_processor_pipelines(
            input_steps=[steps.add_batch_dim, steps.to_device, steps.normalize],
            output_steps=[steps.unnormalize, steps.to_cpu],
        )

        class Model:
            def select_action(model_self, batch):
                torch.testing.assert_close(batch["observation.state"], torch.ones(1, 14))
                torch.testing.assert_close(batch["observation.images.top"], torch.ones(1, 3, 2, 2))
                return torch.zeros(1, 14)

        model = Model()
        model.config = config
        adapter = SmolVLAPolicy.__new__(SmolVLAPolicy)
        adapter._torch = torch
        adapter._device = torch.device("cpu")
        adapter._prompt = "transfer cube"
        adapter._policy = model
        adapter._preprocessor, adapter._postprocessor = pre, post
        adapter._use_amp = False
        return adapter

    def test_inference_uses_training_stats_and_camera_layout(self):
        action = self.make_adapter().infer({
            "state": np.full(14, 3.0, dtype=np.float32),
            "images": {"cam_high": np.full((2, 2, 3), 255, dtype=np.uint8)},
        })
        np.testing.assert_allclose(action, np.arange(14, dtype=np.float32))
        self.assertEqual(action.dtype, np.float32)

    def test_missing_training_camera_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "相机缺失"):
            self.make_adapter().infer({"state": np.zeros(14, dtype=np.float32), "images": {}})


if __name__ == "__main__":
    unittest.main()
