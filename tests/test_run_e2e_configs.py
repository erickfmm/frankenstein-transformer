"""Config-generation gates for the ``full_tests`` e2e harness.

Regression tests for the two harness bugs that cost experiment 2 runs before
any training started:

* ``--realonall`` raises ``training.max_length`` from 32 to 64 *after* the
  config makers fixed the SSOG raster, leaving a stale ``1x32`` grid that
  crashed every ``ssog`` combo at the first forward (41 runs + 2
  transversals);
* ``_with_optimizer`` passed ``rank`` to every APOLLO variant although
  ``apollo_mini`` (rank-1 by construction) rejects it, failing ``opt_apollo_mini``.

Importing :mod:`run_e2e` pulls in ``full_tests/_helpers`` (no torch import at
module scope), so these gates run in the plain CPU test environment.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

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
    model = run_e2e._deep_update(model, {"dims": {"layer_pattern": ["ssog", "ssog"]}})
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
    bound = run_e2e.apply_realonall("attn_x__ssog", config, real_text_dir=None)
    ssog = bound["model"]["attention"]["ssog"]
    max_length = bound["training"]["max_length"]
    assert max_length == run_e2e.REAL_TEXT_MAX_LENGTH
    assert ssog["grid_h"] * ssog["grid_w"] == max_length


def test_validate_combos_catches_stale_ssog_grid():
    """The pre-flight check reports a grid that no longer matches max_length."""
    config = _ssog_combo(max_length=32)
    config["training"]["max_length"] = 64  # grid left behind at 1x32
    errors = run_e2e.validate_combos([("attn_x__ssog", config)])
    assert any("ssog grid" in error for error in errors), errors


def test_validate_combos_catches_unknown_optimizer_key():
    config = _ssog_combo(max_length=32)
    config["model"]["dims"]["layer_pattern"] = ["standard", "standard"]
    config["training"]["optimizer"] = {
        "optimizer_class": "apollo_mini",
        "parameters": {"apollo_mini-rank": 8},
    }
    errors = run_e2e.validate_combos([("opt_apollo_mini", config)])
    assert any("apollo_mini" in error for error in errors), errors


def test_validate_combos_passes_a_healthy_config():
    config = _ssog_combo(max_length=32)
    assert run_e2e.validate_combos([("attn_x__ssog", config)]) == []


# ---------------------------------------------------------------------------
# CPU/env plumbing (``--workers``, ``--omp-threads``, ``--env``)
# ---------------------------------------------------------------------------
def test_worker_and_thread_defaults():
    """The harness defaults to 28 workers with 2 threads each."""
    args = run_e2e.build_parser().parse_args([])
    assert args.workers == run_e2e.DEFAULT_WORKERS == 28
    assert str(args.omp_threads) == run_e2e.DEFAULT_OMP_THREADS == "2"
    assert args.env == []


def test_effective_threads_never_clamps_explicit_values():
    """``--omp-threads`` is honored as given, even past the core count."""
    assert run_e2e._effective_threads("1", 28) == 1
    assert run_e2e._effective_threads("2", 28) == 2
    assert run_e2e._effective_threads("8", 4) == 8
    assert run_e2e._effective_threads("4", 999) == 4
    assert run_e2e._effective_threads("auto", 4) is not None
    assert run_e2e._effective_threads("nope", 4) is None
    assert run_e2e._effective_threads("0", 4) is None


def test_effective_threads_auto_divides_the_core_count():
    """``auto`` splits the cores across workers, never below one thread."""
    import os as _os

    cores = _os.cpu_count() or 1
    assert run_e2e._effective_threads("auto", 1) == cores
    assert run_e2e._effective_threads("auto", cores) == 1
    assert run_e2e._effective_threads("auto", cores * 4) == 1


def test_worker_env_sets_the_whole_openmp_family():
    """Every OpenMP/BLAS pool is pinned, not only OMP_NUM_THREADS."""
    parser = run_e2e.build_parser()
    args = parser.parse_args(["--workers", "28", "--omp-threads", "4"])
    env, threads, notes = run_e2e._resolve_worker_env(args, parser)
    assert threads == 4
    for name in run_e2e.THREAD_ENV_VARS:
        assert env[name] == "4"
    assert any("oversubscribed" in note for note in notes)


def test_worker_env_extra_wins_over_the_thread_default():
    """``--env`` is applied after ``--omp-threads``, so it overrides it."""
    parser = run_e2e.build_parser()
    args = parser.parse_args(
        ["--omp-threads", "2", "--env", "OMP_NUM_THREADS=7", "--env", "FOO=bar"]
    )
    env, threads, _ = run_e2e._resolve_worker_env(args, parser)
    assert threads == 2
    assert env["OMP_NUM_THREADS"] == "7"
    assert env["FOO"] == "bar"
    assert env["MKL_NUM_THREADS"] == "2"


def test_worker_env_rejects_reserved_and_malformed_pairs():
    """The determinism pair cannot be overridden and ``KEY=VALUE`` is enforced."""
    parser = run_e2e.build_parser()
    with pytest.raises(SystemExit):
        run_e2e._resolve_worker_env(
            parser.parse_args(["--env", "PYTHONHASHSEED=9"]), parser
        )
    with pytest.raises(SystemExit):
        run_e2e._resolve_worker_env(parser.parse_args(["--env", "NOEQUALS"]), parser)


def test_run_banner_prints_command_and_defaults(capsys):
    """The startup banner shows the command and every value, defaults included."""
    parser = run_e2e.build_parser()
    argv = ["--category", "opt"]
    args = parser.parse_args(argv)
    run_e2e._emit_run_banner(parser, args, argv)
    out = capsys.readouterr().out
    assert "run_e2e: effective parameters" in out
    assert "--category opt" in out
    assert "--workers" in out and "28" in out
    assert "--omp-threads" in out and "OMP_NUM_THREADS=2" in out
    assert "[default]" in out and "[explicit]" in out


# ---------------------------------------------------------------------------
# Nested-shape migration (the flattener is gone; flat keys are silently dropped)
# ---------------------------------------------------------------------------
def test_default_model_cfg_is_schema_clean():
    """Every generated ``model:`` key must exist in the assembled schema.

    A flat legacy key (``use_bitnet``, ``ffn_hidden_size``, ``top_k_experts`` …)
    is logged and dropped by the runtime, which silently changes the model the
    sweep claims to test. The preflight escalates those to errors, so the base
    config has to stay clean.
    """
    model = run_e2e.helpers.default_model_cfg(vocab_size=256)
    document = {
        "model_class": "frankenstein_text_encoder",
        "model": model,
        "training": {"task": "mlm"},
    }
    assert run_e2e._unknown_schema_keys(document) == []


def test_default_model_cfg_disables_the_default_on_components():
    """BitNet, MoE and MoD are on by default in the schema; the harness opts out."""
    model = run_e2e.helpers.default_model_cfg(vocab_size=256)
    assert model["bitnet"]["enabled"] is False
    assert model["moe"]["enabled"] is False
    assert model["mixture_of_depths"]["enabled"] is False
    assert model["ffn"] == {"hidden_size": 96, "activation": "gelu"}
    for flat_key in (
        "use_bitnet",
        "bitnet_routers",
        "use_moe",
        "top_k_experts",
        "ffn_hidden_size",
        "ffn_activation",
        "retention_heads",
    ):
        assert flat_key not in model
        assert flat_key not in model["dims"]


def test_preflight_rejects_a_flat_model_key():
    """A flat model key would be dropped at runtime; the preflight must fail."""
    config = _ssog_combo(max_length=32)
    config["model"].pop("bitnet", None)
    config["model"]["use_bitnet"] = False
    errors = run_e2e.validate_combos([("trans_bitnet_off", config)])
    assert any("unknown config key" in error for error in errors), errors


def test_transversal_switches_use_the_nested_shape():
    """The toggle sweep must write ``bitnet``/``moe``/``ffn``/``numerics`` blocks."""
    configs = dict(run_e2e.make_transversal_configs(vocab_size=256))
    bitnet = configs["trans_bitnet_on"]["model"]["bitnet"]
    assert bitnet["enabled"] is True
    assert configs["trans_moe"]["model"]["moe"]["enabled"] is True
    assert configs["trans_mod"]["model"]["mixture_of_depths"]["capacity_ratio"] == 0.5
    assert configs["trans_ffn_relu"]["model"]["ffn"]["activation"] == "relu"
    assert configs["trans_fixed_softmax"]["model"]["numerics"]["fixed_softmax"] is True
    assert configs["trans_ssmax"]["model"]["ssmax"]["s_init"] == 1.0
    for combo_id, config in configs.items():
        assert run_e2e._unknown_schema_keys(config) == [], combo_id
