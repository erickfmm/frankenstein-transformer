"""Tests for the supervisor-aware thermal guard in TitanTrainer.

The in-process thermal offload path was replaced by the CPU-side
:class:`GPUTempSupervisor`. These tests verify the trainer's new hooks:
``_enforce_thermal_guard`` is a no-op, the SIGUSR1 checkpoint-request flag
drives ``_maybe_flush_supervisor_checkpoint``, and
``resume_from_latest_checkpoint`` finds/loads the latest rolling checkpoint.
"""
import os
import tempfile
import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import Mock

TORCH_AVAILABLE = find_spec("torch") is not None
_IMPORTS_OK = False
if TORCH_AVAILABLE:
    try:
        from src.training.trainer import TitanTrainer, TrainingConfig
        _IMPORTS_OK = True
    except ImportError:
        pass


@unittest.skipUnless(_IMPORTS_OK, "torch and tqdm required for trainer tests")
class TrainerSupervisorHookTests(unittest.TestCase):
    def _bare_trainer(self):
        trainer = TitanTrainer.__new__(TitanTrainer)
        trainer.training_config = TrainingConfig()
        trainer.gpu_temp_guard = SimpleNamespace(is_active=False)
        trainer._last_guard_temp_c = None
        trainer._pending_thermal_repair_action = "none"
        trainer.global_step = 0
        trainer._checkpoint_request_pending = False
        trainer._checkpoint_dir = "checkpoints"
        trainer.rolling_checkpoints = []
        trainer.model = None
        trainer.optimizer = None
        trainer.scheduler = None
        trainer.scaler = None
        trainer.config = None
        trainer.best_loss = float("inf")
        trainer.storage_manager = SimpleNamespace(register_file=lambda p: True)
        return trainer

    def test_enforce_thermal_guard_is_noop(self):
        trainer = self._bare_trainer()
        # Even with an "active" guard, the in-process hook is a no-op now;
        # the supervisor owns thermal lifecycle.
        self.assertEqual(trainer._enforce_thermal_guard(epoch=0, batch_idx=0), "none")

    def test_maybe_flush_noop_when_no_request(self):
        trainer = self._bare_trainer()
        trainer._checkpoint_request_pending = False
        # Should not raise even though model/optimizer are None (early return).
        trainer._maybe_flush_supervisor_checkpoint(epoch=0)
        self.assertFalse(trainer._checkpoint_request_pending)

    def test_find_latest_rolling_checkpoint_picks_highest_step(self):
        with tempfile.TemporaryDirectory() as d:
            for step in (5, 500, 50):
                open(os.path.join(d, f"titan_rolling_step_{step}.pt"), "w").close()
            latest = TitanTrainer.find_latest_rolling_checkpoint(d)
            self.assertIsNotNone(latest)
            self.assertTrue(latest.endswith("titan_rolling_step_500.pt"))

    def test_find_latest_rolling_checkpoint_none_when_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(TitanTrainer.find_latest_rolling_checkpoint(d))

    def test_resume_from_latest_checkpoint_auto_returns_zero_when_none(self):
        trainer = self._bare_trainer()
        with tempfile.TemporaryDirectory() as d:
            trainer._checkpoint_dir = d
            epoch = trainer.resume_from_latest_checkpoint("auto")
        self.assertEqual(epoch, 0)

    def test_resume_from_latest_checkpoint_none_does_nothing(self):
        trainer = self._bare_trainer()
        epoch = trainer.resume_from_latest_checkpoint(None)
        self.assertEqual(epoch, 0)


if __name__ == "__main__":
    unittest.main()