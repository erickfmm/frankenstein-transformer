"""Config-generation gates for the ``full_tests`` e2e harness.

Regression tests for the two harness bugs that cost experiment 2 runs before
any training started:

* ``--realonall`` raises ``training.max_length`` from 32 to 64 *after* the
  config makers fixed the SSOG raster, leaving a stale ``1x32`` grid that
  crashed every ``ssog_attn`` combo at the first forward (41 runs + 2
  transversals);
* ``_with_optimizer`` passed ``rank`` to every APOLLO variant although
  ``apollo_mini`` (rank-1 by construction) rejects it, failing ``opt_apollo_mini``.

Importing :mod:`run_e2e` pulls in ``full_tests/_helpers`` (no torch import at
module scope), so these gates run in the plain CPU test environment.
"""
from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FULL_TESTS_DIR = REPO_ROOT / "full_tests"
if str(FULL_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(FULL_TESTS_DIR))

import run_e2e  # noqa: E402  (needs the full_tests dir on sys.path)

from src.model.optimizer.factory import OPTIMIZER_ALLOWED_KEYS  # noqa: E402


def test_optimizer_parameter_keys_are_all_allowed():
    """Every generated `<optimizer>-<param>` key must be accepted by the factory."""
    for optimizer in run_e2e.OPTIMIZERS:
        training = run_e2e._with_optimizer({"task": "mlm"}, optimizer)
        parameters = training["optimizer"]["parameters"]
        assert parameters, optimizer
        allowed = OPTIMIZER_ALLOWED_KEYS[optimizer]
        prefix = f"{optimizer}-"
        unknown = [
            key
            for key in parameters
            if key.startswith(prefix) and key[len(prefix):] not in allowed
        ]
        assert not unknown, f"{optimizer} rejects {unknown}"


def test_apollo_mini_gets_no_rank_key():
    """``apollo_mini`` is rank-1 by construction; ``rank`` is rejected."""
    training = run_e2e._with_optimizer({"task": "mlm"}, "apollo_mini")
    parameters = training["optimizer"]["parameters"]
    assert "apollo_mini-rank" not in parameters
    assert "apollo_mini-update_proj_gap" in parameters


def _ssog_combo(max_length=32):
    model, training = run_e2e._base_config(vocab_size=100, num_layers=2, num_loops=1)
    model = run_e2e._deep_update(model, {"dims": {"layer_pattern": ["ssog_attn", "ssog_attn"]}})
    training["max_length"] = max_length
    model = run_e2e._ensure_ssog_grid(model, training)
    return {
        "model_class": "frankenstein_text_encoder",
        "model": model,
        "training": training,
    }


def test_ensure_ssog_grid_force_overrides_stale_grid():
    """force=True re-syncs an auto-derived grid after max_length grows."""
    config = _ssog_combo(max_length=32)
    ssog = config["model"]["attention"]["ssog"]
    assert (ssog["grid_h"], ssog["grid_w"]) == (1, 32)

    training = dict(config["training"])
    training["max_length"] = 64
    model = run_e2e._ensure_ssog_grid(config["model"], training, force=True)
    ssog = model["attention"]["ssog"]
    assert (ssog["grid_h"], ssog["grid_w"]) == (1, 64)


def test_realonall_re_syncs_ssog_grid():
    """apply_realonall must leave grid_h * grid_w == the raised max_length."""
    config = _ssog_combo(max_length=32)
    bound = run_e2e.apply_realonall("attn_x__ssog_attn", config, real_text_dir=None)
    ssog = bound["model"]["attention"]["ssog"]
    max_length = bound["training"]["max_length"]
    assert max_length == run_e2e.REAL_TEXT_MAX_LENGTH
    assert ssog["grid_h"] * ssog["grid_w"] == max_length


def test_validate_combos_catches_stale_ssog_grid():
    """The pre-flight check reports a grid that no longer matches max_length."""
    config = _ssog_combo(max_length=32)
    config["training"]["max_length"] = 64  # grid left behind at 1x32
    errors = run_e2e.validate_combos([("attn_x__ssog_attn", config)])
    assert any("ssog grid" in error for error in errors), errors


def test_validate_combos_catches_unknown_optimizer_key():
    config = _ssog_combo(max_length=32)
    config["model"]["dims"]["layer_pattern"] = ["standard_attn", "standard_attn"]
    config["training"]["optimizer"] = {
        "optimizer_class": "apollo_mini",
        "parameters": {"apollo_mini-rank": 8},
    }
    errors = run_e2e.validate_combos([("opt_apollo_mini", config)])
    assert any("apollo_mini" in error for error in errors), errors


def test_validate_combos_passes_a_healthy_config():
    config = _ssog_combo(max_length=32)
    assert run_e2e.validate_combos([("attn_x__ssog_attn", config)]) == []
