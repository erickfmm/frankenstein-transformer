"""Tests for the trainer resume-from-checkpoint helper (supervisor integration)."""
import os
import tempfile
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None
if TORCH_AVAILABLE:
    from src.training.trainer import TitanTrainer


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class FindLatestRollingCheckpointTests(unittest.TestCase):
    def test_returns_none_when_dir_missing(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(TitanTrainer.find_latest_rolling_checkpoint(d))

    def test_returns_none_when_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(TitanTrainer.find_latest_rolling_checkpoint(d))

    def test_picks_highest_global_step(self):
        with tempfile.TemporaryDirectory() as d:
            for step in (10, 200, 50, 999):
                open(os.path.join(d, f"titan_rolling_step_{step}.pt"), "w").close()
            # Also a non-matching file that must be ignored.
            open(os.path.join(d, "titan_best_loss_0.5_step_300.pt"), "w").close()
            latest = TitanTrainer.find_latest_rolling_checkpoint(d)
            self.assertIsNotNone(latest)
            self.assertTrue(latest.endswith("titan_rolling_step_999.pt"))

    def test_ignores_unrelated_files(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "README.md"), "w").close()
            open(os.path.join(d, "other_rolling_step_5.pt"), "w").close()
            self.assertIsNone(TitanTrainer.find_latest_rolling_checkpoint(d))


if __name__ == "__main__":
    unittest.main()