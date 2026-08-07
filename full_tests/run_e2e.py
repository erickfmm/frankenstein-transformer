"""Exhaustive end-to-end smoke tests for frankenstein-transformer.

This script drives the real CLI with tiny synthetic data and a tiny on-the-fly
SentencePiece tokenizer, exercising:

* every trainable attention mixer in unordered pairs,
* every optimizer,
* every normalization type,
* major transversal toggles (BitNet, MoE, MoD, mHC, embeddings, residuals, etc.),
* encoder/MLM and decoder/causal-LM tasks,
* deploy, inference, quantization, transformers-export and bitnet-gguf smoke tests.

It is intentionally NOT collected by ``pytest tests/`` (it lives in ``full_tests/``
at the repository root) and is meant to be left running for several hours.

Run from the repo root inside the ``frankenstein`` conda env:

    conda run -n frankenstein python full_tests/run_e2e.py

To run only a subset:

    conda run -n frankenstein python full_tests/run_e2e.py --category optimizers --limit 3

To run on a specific device (default is cpu):

    conda run -n frankenstein python full_tests/run_e2e.py --device cuda

To enable the GPU thermal guard during training (with optional thresholds):

    conda run -n frankenstein python full_tests/run_e2e.py --device cuda --gpu-temp-guard

To resume from existing tmp/results directory (skip cleanup):

    conda run -n frankenstein python full_tests/run_e2e.py --resume
"""
from __future__ import annotations

import argparse
import itertools
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the helpers importable whether this file is run as a script or module.
_FULL_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FULL_TESTS_DIR.parent
sys.path.insert(0, str(_FULL_TESTS_DIR))

import _helpers as helpers  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Canonical lists from the schema
# ---------------------------------------------------------------------------
# fasa_attn and sparge_attn are training-free/eval-only; skip them here.
ATTENTIONS = [
    "retnet", "retnet_attn", "mamba", "ode", "titan_attn", "standard_attn",
    "sigmoid_attn", "sparse_transformer_attn", "longformer_attn", "bigbird_attn",
    "sparsek_attn", "nsa_attn", "gla_attn", "deltanet_attn", "gated_deltanet_attn",
    "gated_deltanet2_attn", "hgrn2_attn", "fox_attn", "gated_softmax_attn",
    "engram_attn", "gqa_attn", "mla_attn", "gqla_attn", "mlra_attn",
    "tucker_attn", "iha_attn", "gta_attn", "mtla_attn", "cca_attn", "ccgqa_attn",
    "msa_attn", "sparda_attn", "kda_attn",
]

OPTIMIZERS = [
    "sgd_momentum", "adamw", "adafactor", "galore_adamw", "prodigy", "lion",
    "sophia", "muon", "turbo_muon", "radam", "adan", "adopt", "ademamix",
    "mars_adamw", "cautious_adamw", "lamb", "schedulefree_adamw", "shampoo",
    "soap", "anon", "apollo", "apollo_mini", "q_apollo",
]

NORMS = ["layer_norm", "dynamic_tanh", "derf", "rms_norm", "prms_norm", "flash_norm"]


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------
def _deep_update(base: dict, updates: dict) -> dict:
    """Return a new dict merging ``updates`` into ``base`` recursively."""
    out = {}
    for key in base:
        if key in updates and isinstance(base[key], dict) and isinstance(updates[key], dict):
            out[key] = _deep_update(base[key], updates[key])
        else:
            out[key] = base[key]
    for key in updates:
        if key not in base:
            out[key] = updates[key]
    return out


def _base_config(vocab_size: int, num_layers: int, num_loops: int) -> Tuple[dict, dict]:
    model = helpers.default_model_cfg(
        vocab_size=vocab_size, num_layers=num_layers, num_loops=num_loops
    )
    training = helpers.default_training_cfg(
        helpers.ensure_toy_parquet(), batch_size=1, num_epochs=1
    )
    return model, training


def _with_optimizer(training: dict, optimizer_class: str) -> dict:
    """Build a minimal per-group parameter block for the given optimizer class."""
    prefix = f"{optimizer_class}-"
    params: Dict[str, Any] = {
        f"{prefix}lr_embeddings": 1e-4,
        f"{prefix}lr_norms": 1e-4,
        f"{prefix}lr_attention": 1e-4,
        f"{prefix}lr_other": 1e-4,
        f"{prefix}wd_embeddings": 0.0,
        f"{prefix}wd_norms": 0.0,
        f"{prefix}wd_attention": 0.0,
        f"{prefix}wd_other": 0.0,
    }
    # Adam-style families also accept betas/eps per group.  Providing them is
    # harmless for non-Adam optimizers because the factory simply ignores keys
    # it does not understand, while it satisfies the schema's minProperties.
    if optimizer_class in {
        "adamw", "galore_adamw", "prodigy", "radam", "adan", "adopt", "ademamix",
        "mars_adamw", "cautious_adamw", "schedulefree_adamw", "soap", "lion",
    }:
        for group in ["embeddings", "norms", "attention", "other"]:
            params[f"{prefix}betas_{group}"] = [0.9, 0.999]
            params[f"{prefix}eps_{group}"] = 1e-8
    # A few optimizers have documented specific globals that are safer to set.
    if optimizer_class == "prodigy":
        params[f"{prefix}d_coef"] = 0.8
    if optimizer_class in {"muon", "turbo_muon"}:
        params[f"{prefix}momentum"] = 0.9
        params[f"{prefix}ns_steps"] = 5
    if optimizer_class == "sophia":
        params[f"{prefix}rho"] = 0.05
        params[f"{prefix}update_k"] = 10
    if optimizer_class == "cautious_adamw":
        params[f"{prefix}cautious_clip"] = 1.0
    if optimizer_class == "anon":
        params[f"{prefix}gamma"] = 0.5
    if optimizer_class == "galore_adamw":
        params[f"{prefix}rank"] = 8
        params[f"{prefix}update_proj_gap"] = 50
    if optimizer_class in {"apollo", "apollo_mini", "q_apollo"}:
        params[f"{prefix}rank"] = 8
        params[f"{prefix}update_proj_gap"] = 50
        params[f"{prefix}scale"] = 1.0
        params[f"{prefix}proj_type"] = "std"

    training = training.copy()
    training["optimizer"] = {
        "optimizer_class": optimizer_class,
        "parameters": params,
    }
    return training


def make_attention_pair_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []
    for a, b in itertools.combinations(ATTENTIONS, 2):
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {
            "dims": {
                "layer_pattern": [a, b],
            },
        })
        combo_id = f"attn_{a}__{b}"
        configs.append((combo_id, {"model_class": "frankenstein", "model": model, "training": training}))
    return configs


def make_single_attention_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []
    for a in ATTENTIONS:
        model, training = _base_config(vocab_size, num_layers=1, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": [a]}})
        configs.append((f"attn_single_{a}", {"model_class": "frankenstein", "model": model, "training": training}))
    return configs


def make_optimizer_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []
    for opt in OPTIMIZERS:
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": ["standard_attn", "titan_attn"]}})
        training = _with_optimizer(training, opt)
        configs.append((f"opt_{opt}", {"model_class": "frankenstein", "model": model, "training": training}))
    return configs


def make_norm_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []
    for norm in NORMS:
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": ["standard_attn", "titan_attn"]}})
        norm_block: Dict[str, Any] = {"type": norm}
        if norm == "prms_norm":
            norm_block["partial_ratio"] = 0.5
        if norm == "flash_norm":
            norm_block["flashnorm_partial_ratio"] = 0.0
        model = _deep_update(model, {"norm": norm_block})
        configs.append((f"norm_{norm}", {"model_class": "frankenstein", "model": model, "training": training}))
    return configs


def make_transversal_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []

    def add(cid: str, overrides: dict) -> None:
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": ["standard_attn", "titan_attn"]}})
        model = _deep_update(model, overrides)
        configs.append((cid, {"model_class": "frankenstein", "model": model, "training": training}))

    add("trans_bitnet_on", {"use_bitnet": True})
    add("trans_bitnet_routers", {"use_bitnet": True, "bitnet_routers": True, "use_moe": True, "num_experts": 2, "top_k_experts": 1})
    add("trans_bitnet_conv", {"use_bitnet": True, "use_bitnet_conv": True, "embedding": {"conv": {"enabled": True, "kernel": 3}}})
    add("trans_factorized_emb", {"embedding": {"factorized": {"enabled": True, "dim": 32}}})
    add("trans_embedding_conv", {"embedding": {"conv": {"enabled": True, "kernel": 3}}})
    add("trans_moe", {"use_moe": True, "num_experts": 2, "top_k_experts": 1})
    add("trans_mod", {"use_mixture_of_depths": True, "mixture_of_depths_capacity_ratio": 0.5})
    add("trans_residuals_none", {"residuals": {"type": "none"}})
    add("trans_residuals_full_attn", {"residuals": {"type": "full_attn", "full_attn": {"init_query_zero": True, "use_rmsnorm_keys": True}}})
    add("trans_residuals_block_attn", {"residuals": {"type": "block_attn", "block_attn": {"num_blocks": 2, "init_query_zero": True, "use_rmsnorm_keys": True}}})
    add("trans_pos_rope", {"attention": {"titan": {"positional_encoding": "rope"}}})
    add("trans_num_loops_2", {"dims": {"num_layers": 1, "num_loops": 2, "layer_pattern": ["standard_attn"]}})
    add("trans_ffn_relu", {"ffn_activation": "relu"})
    add("trans_ffn_swiglu", {"ffn_activation": "swiglu"})
    add("trans_mhc", {
        "mhc": {"enabled": True, "expansion_rate": 2, "sinkhorn_iters": 5, "gating_init": 0.01, "checkpoint": False, "full_prec_under_bitnet": True},
    })
    return configs


def make_task_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []

    # Encoder MLM (default path)
    model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
    model = _deep_update(model, {"dims": {"layer_pattern": ["standard_attn", "titan_attn"]}})
    configs.append(("task_mlm_encoder", {"model_class": "frankenstein", "model": model, "training": training}))

    # Decoder causal LM
    model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
    model = _deep_update(model, {"dims": {"mode": "decoder", "layer_pattern": ["standard_attn", "titan_attn"]}})
    training = training.copy()
    training["task"] = "causal_lm"
    configs.append(("task_causal_lm_decoder", {"model_class": "frankensteindecoder", "model": model, "training": training}))
    return configs


def make_batch_size_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []
    for bs in [1, 2]:
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": ["standard_attn", "titan_attn"]}})
        training = training.copy()
        training["batch_size"] = bs
        configs.append((f"batch_size_{bs}", {"model_class": "frankenstein", "model": model, "training": training}))
    return configs


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exhaustive end-to-end smoke tests for frankenstein-transformer.",
    )
    parser.add_argument(
        "--category",
        choices=["all", "attn", "opt", "norm", "transversal", "task", "batch_size", "deploy"],
        default="all",
        help="Which sweep to run (default: all).",
    )
    parser.add_argument(
        "--logging-level",
        choices=["none", "info", "debug", "only-errors"],
        default="info",
        help="How much progress logging to emit to stdout (default: info).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N training combos (useful for a quick smoke run).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-combo training timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--skip-attn-pairs",
        action="store_true",
        help="Skip the expensive attention-pair sweep.",
    )
    parser.add_argument(
        "--skip-singles",
        action="store_true",
        help="Skip the single-attention sweep.",
    )
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Skip deploy/infer/quantize/export/gguf smoke tests.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=helpers.TOY_VOCAB_SIZE,
        help="Toy tokenizer vocab size (default: 256).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="cpu",
        help="Device to run training/deploy/infer on (default: cpu).",
    )
    parser.add_argument(
        "--gpu-temp-guard",
        action="store_true",
        help="Enable the GPU thermal guard during training (only meaningful on cuda).",
    )
    parser.add_argument(
        "--gpu-temp-pause-threshold-c",
        type=float,
        default=None,
        help="GPU temp (C) at which training pauses (default: CLI default).",
    )
    parser.add_argument(
        "--gpu-temp-resume-threshold-c",
        type=float,
        default=None,
        help="GPU temp (C) at which training resumes (default: CLI default).",
    )
    parser.add_argument(
        "--gpu-temp-critical-threshold-c",
        type=float,
        default=None,
        help="GPU temp (C) considered critical (default: CLI default).",
    )
    parser.add_argument(
        "--gpu-temp-poll-interval-seconds",
        type=float,
        default=None,
        help="GPU temp polling interval in seconds (default: CLI default).",
    )
    parser.add_argument(
        "--gpu-temp-checkpoint-grace-seconds",
        type=float,
        default=None,
        help="Grace period in seconds before a thermal checkpoint (default: CLI default).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing tmp/results directory instead of cleaning up first.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    log_level = {
        "none": logging.CRITICAL + 10,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "only-errors": logging.ERROR,
    }[args.logging_level]
    helpers.setup_logging(log_level)

    # Clean up tmp directory unless --resume is specified
    if not args.resume and helpers.TMP_DIR.exists():
        logging.info("Cleaning up tmp directory: %s", helpers.TMP_DIR)
        shutil.rmtree(helpers.TMP_DIR)

    helpers.TMP_DIR.mkdir(parents=True, exist_ok=True)
    helpers.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    helpers.write_sitecustomize(helpers.TMP_DIR)
    tokenizer_path = helpers.ensure_toy_tokenizer(args.vocab_size)
    parquet_path = helpers.ensure_toy_parquet()
    runner, env_extra = helpers.resolve_runner()

    logging.info("=" * 60)
    logging.info("Frankenstein exhaustive end-to-end smoke tests")
    logging.info("Tokenizer: %s", tokenizer_path)
    logging.info("Parquet:   %s", parquet_path)
    logging.info("Runner:    %s", " ".join(runner))
    logging.info("=" * 60)

    combos: List[Tuple[str, dict]] = []

    if args.category in {"all", "attn"}:
        if not args.skip_attn_pairs:
            combos.extend(make_attention_pair_configs(args.vocab_size))
        if not args.skip_singles:
            combos.extend(make_single_attention_configs(args.vocab_size))
    if args.category in {"all", "opt"}:
        combos.extend(make_optimizer_configs(args.vocab_size))
    if args.category in {"all", "norm"}:
        combos.extend(make_norm_configs(args.vocab_size))
    if args.category in {"all", "transversal"}:
        combos.extend(make_transversal_configs(args.vocab_size))
    if args.category in {"all", "task"}:
        combos.extend(make_task_configs(args.vocab_size))
    if args.category in {"all", "batch_size"}:
        combos.extend(make_batch_size_configs(args.vocab_size))

    if args.limit is not None:
        combos = combos[: args.limit]

    logging.info("Planning to run %d training combos", len(combos))
    logging.debug("Category=%s limit=%s skip_attn_pairs=%s skip_singles=%s skip_deploy=%s",
                  args.category, args.limit, args.skip_attn_pairs, args.skip_singles, args.skip_deploy)

    results: List[helpers.RunResult] = []
    successful_training: helpers.RunResult | None = None

    train_kwargs = dict(
        runner=runner,
        env_extra=env_extra,
        timeout=args.timeout,
        device=args.device,
        gpu_temp_guard=args.gpu_temp_guard,
        gpu_temp_pause_threshold_c=args.gpu_temp_pause_threshold_c,
        gpu_temp_resume_threshold_c=args.gpu_temp_resume_threshold_c,
        gpu_temp_critical_threshold_c=args.gpu_temp_critical_threshold_c,
        gpu_temp_poll_interval_seconds=args.gpu_temp_poll_interval_seconds,
        gpu_temp_checkpoint_grace_seconds=args.gpu_temp_checkpoint_grace_seconds,
    )

    def write_results_incremental() -> None:
        """Write results to disk immediately (for crash recovery)."""
        helpers.write_results(results)

    def find_metrics_csv(combo_id: str) -> Optional[str]:
        """Find the training_metrics.csv file for a given combo."""
        run_dir = helpers.RUNS_DIR / combo_id
        metrics_path = run_dir / "training_metrics.csv"
        if metrics_path.exists():
            return str(metrics_path)
        return None

    for idx, (combo_id, config) in enumerate(combos, start=1):
        logging.info("[%d/%d] Running %s", idx, len(combos), combo_id)
        result = helpers.run_training(
            config=config,
            combo_id=combo_id,
            batch_size=config["training"].get("batch_size", 1),
            **train_kwargs,
        )
        # Capture metrics CSV path if training produced one
        if result.status in ("OK", "GRAD_EXPLODED"):
            metrics_path = find_metrics_csv(combo_id)
            if metrics_path:
                result.metrics_path = metrics_path
                logging.info("  Metrics saved to: %s", metrics_path)
        results.append(result)
        logging.info("[%d/%d] %s -> %s (%.1fs) %s", idx, len(combos), combo_id, result.status, result.duration_sec, result.notes)

        # Write results incrementally after each combo
        write_results_incremental()

        if result.status == "OK" and successful_training is None:
            successful_training = result

    # Deploy / infer / export smoke tests, using the first successful training run.
    if not args.skip_deploy and successful_training is not None:
        checkpoint_path = Path(successful_training.checkpoint_path) if successful_training.checkpoint_path else None
        if checkpoint_path and checkpoint_path.exists():
            base_id = successful_training.combo_id
            vocab_size = args.vocab_size

            # standard deploy
            deploy_std_dir = helpers.TMP_DIR / "deploy" / f"{base_id}_standard"
            logging.info("Running deploy (standard) %s -> %s", base_id, deploy_std_dir)
            res = helpers.run_deploy(checkpoint_path, deploy_std_dir, f"{base_id}_deploy_std", runner, env_extra, fmt="standard", device=args.device)
            results.append(res)
            write_results_incremental()
            logging.info("%s -> %s (%.1fs) %s", res.combo_id, res.status, res.duration_sec, res.notes)
            if res.status == "OK":
                helpers.copy_tokenizer_to_deploy_dir(deploy_std_dir, vocab_size)
                results.append(helpers.run_infer(deploy_std_dir, f"{base_id}_infer", runner, env_extra, device=args.device))
                write_results_incremental()
                logging.info("%s -> %s (%.1fs) %s", results[-1].combo_id, results[-1].status, results[-1].duration_sec, results[-1].notes)

            # quantized deploy
            deploy_q_dir = helpers.TMP_DIR / "deploy" / f"{base_id}_quantized"
            logging.info("Running deploy (quantized) %s -> %s", base_id, deploy_q_dir)
            results.append(helpers.run_deploy(checkpoint_path, deploy_q_dir, f"{base_id}_deploy_quantized", runner, env_extra, fmt="quantized", device=args.device))
            write_results_incremental()
            logging.info("%s -> %s (%.1fs) %s", results[-1].combo_id, results[-1].status, results[-1].duration_sec, results[-1].notes)

            # transformers-export (needs the original YAML)
            yaml_path = helpers.RUNS_DIR / base_id / "config.yaml"
            if yaml_path.exists():
                export_dir = helpers.TMP_DIR / "transformers-export" / base_id
                logging.info("Running transformers-export %s -> %s", base_id, export_dir)
                results.append(helpers.run_transformers_export(checkpoint_path, yaml_path, export_dir, f"{base_id}_export", runner, env_extra))
                write_results_incremental()
                logging.info("%s -> %s (%.1fs) %s", results[-1].combo_id, results[-1].status, results[-1].duration_sec, results[-1].notes)

    # Dedicated BitNet+standard_attn training + bitnet-gguf export, if not already covered.
    if not args.skip_deploy and (args.category in {"all", "transversal"}):
        model, training = _base_config(args.vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {
            "dims": {"layer_pattern": ["standard_attn", "standard_attn"]},
            "use_bitnet": True,
        })
        bitnet_id = "bitnet_std_attn_for_gguf"
        logging.info("Running dedicated %s training", bitnet_id)
        bitnet_result = helpers.run_training(
            config={"model_class": "frankenstein", "model": model, "training": training},
            combo_id=bitnet_id,
            batch_size=1,
            **train_kwargs,
        )
        # Capture metrics for bitnet training too
        if bitnet_result.status in ("OK", "GRAD_EXPLODED"):
            metrics_path = find_metrics_csv(bitnet_id)
            if metrics_path:
                bitnet_result.metrics_path = metrics_path
                logging.info("  Metrics saved to: %s", metrics_path)
        results.append(bitnet_result)
        write_results_incremental()
        if bitnet_result.status == "OK" and bitnet_result.checkpoint_path:
            yaml_path = helpers.RUNS_DIR / bitnet_id / "config.yaml"
            gguf_path = helpers.TMP_DIR / "bitnet-gguf" / f"{bitnet_id}.gguf"
            logging.info("Running bitnet-gguf %s -> %s", bitnet_id, gguf_path)
            results.append(helpers.run_bitnet_gguf(
                Path(bitnet_result.checkpoint_path),
                yaml_path,
                gguf_path,
                f"{bitnet_id}_gguf",
                runner,
                env_extra,
            ))
            write_results_incremental()
            logging.info("%s -> %s (%.1fs) %s", results[-1].combo_id, results[-1].status, results[-1].duration_sec, results[-1].notes)

    # Final write (ensures everything is flushed)
    helpers.write_results(results)
    return helpers.print_summary(results)


if __name__ == "__main__":
    raise SystemExit(main())
