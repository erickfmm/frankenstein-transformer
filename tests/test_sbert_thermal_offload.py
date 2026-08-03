"""Tests for the supervisor-aware SBERT thermal guard.

The in-process thermal offload path was replaced by the CPU-side
:class:`GPUTempSupervisor`. These tests verify that the SBERT trainer's
``_handle_thermal_guard_for_batch`` is now a no-op and that the
``_ThermalGuardedDataLoader`` still drives the ``on_thermal_check`` callback
(which the trainer sets to the no-op hook).
"""
import os
import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import Mock

SBERT_TEST_DEPS_AVAILABLE = all(
    find_spec(module_name) is not None
    for module_name in ("torch", "datasets", "sentence_transformers")
)

if SBERT_TEST_DEPS_AVAILABLE:
    from torch.utils.data import DataLoader
    from src.sbert.train_sbert import SBERTTrainer, _ThermalGuardedDataLoader


@unittest.skipUnless(
    SBERT_TEST_DEPS_AVAILABLE,
    "torch + datasets + sentence_transformers are required for SBERT thermal tests",
)
class SbertSupervisorHookTests(unittest.TestCase):
    def test_handle_thermal_guard_for_batch_is_noop(self):
        trainer = SBERTTrainer.__new__(SBERTTrainer)
        trainer._last_guard_temp_c = None
        trainer._checkpoint_request_pending = False
        self.assertEqual(trainer._handle_thermal_guard_for_batch(batch_idx=0), "none")

    def test_dataloader_invokes_on_thermal_check_callback(self):
        dataloader = DataLoader(
            [1, 2],
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )
        guard = SimpleNamespace(is_active=True)
        callback = Mock(side_effect=["none", "none"])
        wrapped = _ThermalGuardedDataLoader(
            dataloader=dataloader,
            guard=guard,
            context_prefix="unit.sbert",
            on_thermal_check=callback,
        )

        _ = list(wrapped)

        self.assertEqual(callback.call_count, 2)
        callback.assert_any_call(0)
        callback.assert_any_call(1)

    def test_supervisor_sigusr1_handler_sets_flag(self):
        trainer = SBERTTrainer.__new__(SBERTTrainer)
        trainer._checkpoint_request_pending = False
        # Simulate the supervisor sending SIGUSR1.
        trainer._on_supervisor_sigusr1(None, None)
        self.assertTrue(trainer._checkpoint_request_pending)


if __name__ == "__main__":
    unittest.main()