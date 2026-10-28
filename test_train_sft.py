"""训练顺序、数值检查和恢复边界；不创建远程客户端。"""

import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
import pytrio as trio
from pytrio.types import TensorData

import train_sft


class TrainSFTTest(unittest.TestCase):
    def test_checkpoint_schedule_scales_to_2048_trajectories(self):
        self.assertTrue(hasattr(train_sft, "checkpoint_schedule"))
        self.assertEqual(train_sft.checkpoint_schedule(128, 32), [8, 32, 64, 96, 128])
        self.assertEqual(train_sft.checkpoint_schedule(512, 128), [8, 128, 256, 384, 512])

    def test_fixed_epoch_covers_every_sample_once_and_resumes_after_eight_batches(self):
        self.assertTrue(hasattr(train_sft, "batch_order"))
        batches = train_sft.batch_order(512, 4, 1, 42)
        self.assertEqual(len(batches), 128)
        self.assertEqual(sorted(i for b in batches for i in b), list(range(512)))
        self.assertEqual(batches, train_sft.batch_order(512, 4, 1, 42))
        self.assertFalse(set(i for b in batches[:8] for i in b) & set(i for b in batches[8:] for i in b))

    def test_resume_requires_complete_pair_exact_progress_and_no_uncertain_operation(self):
        self.assertTrue(hasattr(train_sft, "resume_checkpoint"))
        root = Path("outputs") / ("test-sft-resume-" + uuid.uuid4().hex)
        root.mkdir(parents=True)
        pair = {"step": 8, "config_sha256": "frozen", "sampler_path": "trio://sampler", "state_path": "trio://state"}
        try:
            (root / "latest-checkpoint.json").write_text(json.dumps(pair))
            (root / "progress.json").write_text(json.dumps({"step": 8}))
            self.assertEqual(train_sft.resume_checkpoint(root, "frozen"), pair)
            with self.assertRaises(ValueError):
                train_sft.resume_checkpoint(root, "different")
            (root / "operations").mkdir()
            (root / "operations/1-update-0009.json").write_text(json.dumps({"step": 9}))
            with self.assertRaises(ValueError):
                train_sft.resume_checkpoint(root, "frozen")
            (root / "operations/1-update-0009.json").unlink()
            (root / "pending-operation.json").write_text("{}")
            with self.assertRaises(RuntimeError):
                train_sft.resume_checkpoint(root, "frozen")
            (root / "pending-operation.json").unlink()
            (root / "progress.json").write_text(json.dumps({"step": 9}))
            with self.assertRaises(ValueError):
                train_sft.resume_checkpoint(root, "frozen")
            (root / "progress.json").write_text(json.dumps({"step": 8}))
            del pair["state_path"]
            (root / "latest-checkpoint.json").write_text(json.dumps(pair))
            with self.assertRaises(ValueError):
                train_sft.resume_checkpoint(root, "frozen")
        finally:
            shutil.rmtree(root)

    def test_uncertain_operations_and_token_budget_prevent_repeat_submission(self):
        root = Path("outputs") / ("test-sft-pending-" + uuid.uuid4().hex)
        (root / "operations").mkdir(parents=True)
        try:
            def read_missing_file():
                return json.loads((root / "missing-response.json").read_text())
            with self.assertRaises(RuntimeError):
                train_sft.remote_operation(root, "forward", 0, read_missing_file, 11, token_budget=10)
            self.assertFalse((root / "requests.jsonl").exists())
            with self.assertRaises(FileNotFoundError):
                train_sft.remote_operation(root, "forward", 0, read_missing_file, 10, token_budget=10)
            self.assertTrue((root / "pending-operation.json").exists())
            before = (root / "requests.jsonl").read_bytes()
            with self.assertRaises(RuntimeError):
                train_sft.remote_operation(root, "forward", 0, read_missing_file, 10, token_budget=10)
            self.assertEqual(before, (root / "requests.jsonl").read_bytes())
        finally:
            shutil.rmtree(root)

    def test_weighted_loss_ignores_masked_positions_and_rejects_nonfinite_values(self):
        self.assertTrue(hasattr(train_sft, "loss_per_token"))
        datum = trio.Datum(model_input=trio.ModelInput.from_ints([1, 2, 3]),
                           loss_fn_inputs={"target_tokens": np.array([0, 2, 3]), "weights": np.array([0., 1., 1.])})
        from pytrio.types.forward.forward_backward_output import ForwardBackwardOutput
        result = ForwardBackwardOutput(loss_fn_outputs=[{"logprobs": TensorData.from_numpy(np.array([-99., -2., -4.]))}], metrics={})
        self.assertEqual(train_sft.loss_per_token(result, [datum]), 3.)
        result.loss_fn_outputs[0]["logprobs"] = TensorData.from_numpy(np.array([-99., float("nan"), -4.]))
        with self.assertRaises(ValueError):
            train_sft.loss_per_token(result, [datum])

    def test_pilot_cannot_be_marked_verified_before_both_live_checks_are_recorded(self):
        self.assertTrue(hasattr(train_sft, "record_pilot"))
        root = Path("outputs") / ("test-sft-pilot-" + uuid.uuid4().hex)
        root.mkdir(parents=True)
        pair = {"step": 8, "diagnostic_loss": 0.7}
        try:
            (root / "initial-loss.json").write_text(json.dumps({"loss": 0.8}))
            with self.assertRaises(FileNotFoundError):
                train_sft.record_pilot(root, pair, 4)
            self.assertFalse((root / "pilot-verification.json").exists())
            (root / "sampler-health-0008.json").write_text(json.dumps({"step": 8, "input_tokens": 100, "output_tokens": 10}))
            (root / "restore-verification-0008.json").write_text(json.dumps({"step": 8, "with_optimizer": False}))
            with self.assertRaises(ValueError):
                train_sft.record_pilot(root, pair, 4)
            self.assertFalse((root / "pilot-verification.json").exists())
        finally:
            shutil.rmtree(root)
