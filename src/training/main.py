#!/usr/bin/env python3
"""Training pipeline entry point for TORMENTED-BERT-Frankenstein.

Orchestrates the full training workflow: YAML config loading, model
construction (custom Frankenstein or HuggingFace base model), tokenizer
setup, streaming MLM dataset preparation, and the training loop with
GPU thermal guard integration. Supports both MLM and SBERT tasks.
"""

import argparse
import logging
import os
from typing import Any, Dict, Optional, Tuple

# Tesla P40 (sm_61) and other legacy GPUs are often unstable with Triton/Inductor paths.
# Set FRANKENSTEIN_DISABLE_TRITON=0 to re-enable explicitly.
if os.environ.get("FRANKENSTEIN_DISABLE_TRITON", "1").strip().lower() in {"1", "true", "yes", "on"}:
    os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
    os.environ.setdefault("USE_TRITON", "0")

import torch
from torch.utils.data import DataLoader

try:
    from ..tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer
    from .streaming_mlm_dataset import StreamingMLMDataset
    from .trainer import TitanTrainer, TrainingConfig
    from .config_loader import LoadedTrainingConfig, load_training_config, list_config_paths
    from ..model.tormented_bert_frankestein import (
        FrankensteinDecoder,
        TormentedBertFrankenstein,
        TormentedBertMini,
        UltraConfig,
    )
    from ..utils.device import SUPPORTED_DEVICE_CHOICES, resolve_torch_device
except ImportError:
    from tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer
    from training.streaming_mlm_dataset import StreamingMLMDataset
    from training.trainer import TitanTrainer, TrainingConfig
    from training.config_loader import LoadedTrainingConfig, load_training_config, list_config_paths
    from model.tormented_bert_frankestein import (
        FrankensteinDecoder,
        TormentedBertFrankenstein,
        TormentedBertMini,
        UltraConfig,
    )
    from utils.device import SUPPORTED_DEVICE_CHOICES, resolve_torch_device


def _load_base_model_and_tokenizer(
    loaded: LoadedTrainingConfig,
) -> Tuple[torch.nn.Module, Any]:
    """Load a HuggingFace masked-LM model and tokenizer for base-model training.

    Args:
        loaded: Validated :class:`LoadedTrainingConfig` with ``base_model`` set.

    Returns:
        Tuple of ``(model, tokenizer)`` where ``model`` is an
        ``AutoModelForMaskedLM`` instance and ``tokenizer`` is an
        ``AutoTokenizer`` instance.

    Raises:
        RuntimeError: If ``transformers`` is not installed.
        ValueError: If ``tokenizer.name_or_path`` is missing, or the
            tokenizer lacks a pad token or mask token.
    """
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for base_model MLM training. "
            "Install project dependencies before using base_model."
        ) from exc

    tokenizer_cfg = loaded.tokenizer_config or {}
    tokenizer_name_or_path = str(tokenizer_cfg.get("name_or_path", "")).strip()
    if not tokenizer_name_or_path:
        raise ValueError("tokenizer.name_or_path is required for base_model MLM training")

    trust_remote_code = bool(tokenizer_cfg.get("trust_remote_code", False))
    use_fast = bool(tokenizer_cfg.get("use_fast", True))

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("Loaded tokenizer has no pad token and no eos/unk fallback")
    if tokenizer.mask_token_id is None:
        raise ValueError(
            "Loaded tokenizer has no mask token. Provide a compatible tokenizer for MLM training."
        )

    model = AutoModelForMaskedLM.from_pretrained(
        loaded.base_model,
        trust_remote_code=trust_remote_code,
    )

    tokenizer_vocab_size = len(tokenizer)
    embedding_vocab_size = model.get_input_embeddings().num_embeddings
    if tokenizer_vocab_size != embedding_vocab_size:
        logging.info(
            "Resizing token embeddings from %s to %s to match tokenizer vocabulary",
            embedding_vocab_size,
            tokenizer_vocab_size,
        )
        model.resize_token_embeddings(tokenizer_vocab_size)

    return model, tokenizer


def _load_legacy_tormented_model(loaded: LoadedTrainingConfig) -> Tuple[torch.nn.Module, SpanishSPMTokenizer, UltraConfig]:
    """Load a custom TormentedBert model with SentencePiece tokenizer.

    Trains or loads an SPM tokenizer, then constructs the model based on
    ``model_class`` (``"mini"``, ``"frankesteindecoder"``, or default
    ``"frankenstein"``).

    Args:
        loaded: Validated :class:`LoadedTrainingConfig` with ``model_config``.

    Returns:
        Tuple of ``(model, tokenizer, config)``.

    Raises:
        ValueError: If ``model_config`` is ``None``.
    """
    config = loaded.model_config
    if config is None:
        raise ValueError("model config is required when base_model is not provided")

    logging.info("\n" + "=" * 60)
    logging.info("Step 1: Training/Loading SPM tokenizer (%s vocab)", config.vocab_size)
    logging.info("=" * 60)

    vocab_size = config.vocab_size
    model_prefix = "es_redpajama_50k" if vocab_size == 50_000 else f"es_redpajama_{vocab_size}"
    model_path = f"{model_prefix}.model"

    if os.path.exists(model_path):
        logging.info("Loading existing tokenizer...")
        tokenizer = SpanishSPMTokenizer(vocab_size=vocab_size, model_path=model_path)
    else:
        logging.info("Training new tokenizer with maximum data (100GB RAM target)...")
        tokenizer = SpanishSPMTokenizer(vocab_size=vocab_size)
        tokenizer.train(
            model_prefix=model_prefix,
            max_training_samples=50_000_000,
            target_ram_gb=100.0,
        )

    logging.info("Tokenizer loaded with %s tokens", len(tokenizer.vocab))
    logging.info("Tokenizer model path: %s", tokenizer.model_path)

    logging.info("\n" + "=" * 60)
    logging.info("Step 2: Creating TORMENTED-BERT-Frankenstein model")
    logging.info("=" * 60)

    stable_layer_pattern = [
        "retnet",
        "titan_attn",
        "retnet",
        "mamba",
        "titan_attn",
        "ode",
    ]
    if not config.layer_pattern:
        config.layer_pattern = stable_layer_pattern

    if loaded.model_class == "mini":
        model = TormentedBertMini(config)
    elif loaded.model_class == "frankesteindecoder":
        config.mode = "decoder"
        model = FrankensteinDecoder(config)
    else:
        model = TormentedBertFrankenstein(config)

    logging.info("Model Config:")
    logging.info("  - Model Class: %s", loaded.model_class)
    logging.info("  - Hidden Size: %s", config.hidden_size)
    logging.info(
        "  - Layers: %s x %s = %s logical",
        config.num_layers,
        config.num_loops,
        config.num_layers * config.num_loops,
    )
    logging.info("  - Layer Pattern: %s", config.layer_pattern)
    logging.info("  - BitNet: %s (bitnet_routers: %s)", config.use_bitnet, getattr(config, "bitnet_routers", False))
    logging.info("  - ODE Solver: %s (%s steps)", config.ode_solver, config.ode_steps)
    logging.info("  - Norm Type: %s", config.norm_type)

    return model, tokenizer, config


def _build_dataloader(
    tokenizer: Any,
    training_runtime: Dict[str, Any],
    resolved_device: str,
    cli_batch_size: Optional[int],
) -> Tuple[DataLoader, StreamingMLMDataset, Dict[str, Any], int]:
    """Build the streaming MLM dataset and DataLoader.

    Args:
        tokenizer: Tokenizer instance (SPM or HuggingFace).
        training_runtime: Runtime configuration dictionary from YAML.
        resolved_device: Resolved PyTorch device string.
        cli_batch_size: Optional CLI batch size override.

    Returns:
        Tuple of ``(dataloader, dataset, stats, batch_size)``.

    Raises:
        ValueError: If ``cli_batch_size`` is <= 0.
    """
    logging.info("\n" + "=" * 60)
    logging.info("Step 3: Preparing MLM dataset with resilient caching")
    logging.info("=" * 60)

    max_length = int(training_runtime.get("max_length", 512))
    mlm_probability = float(training_runtime.get("mlm_probability", 0.15))
    max_samples = int(training_runtime.get("max_samples", 20_000_000))
    dataset_batch_size = int(training_runtime.get("dataset_batch_size", 25_000))
    dataset_num_workers = int(training_runtime.get("num_workers", 8))
    cache_dir = training_runtime.get("cache_dir", "./temp_data/v2_dataset_cache")
    local_parquet_dir = training_runtime.get(
        "local_parquet_dir",
        "/home/erickfmm/.cache/huggingface/hub/"
        "datasets--erickfmm--red_pajama_es_hq_35/"
        "snapshots/bd7286c289a95dc3803c375bc36aaaeb138b1eab/"
        "train/",
    )
    prefer_local_cache = bool(training_runtime.get("prefer_local_cache", True))
    stream_local_parquet = bool(training_runtime.get("stream_local_parquet", True))
    join_context_window = int(training_runtime.get("join_temp_data_context_window", 0))
    join_min_remainder = int(training_runtime.get("join_temp_data_min_remainder_tokens", 128))

    dataset = StreamingMLMDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        mlm_probability=mlm_probability,
        max_samples=max_samples,
        batch_size=dataset_batch_size,
        num_workers=dataset_num_workers,
        cache_dir=cache_dir,
        local_parquet_dir=local_parquet_dir,
        prefer_local_cache=prefer_local_cache,
        stream_local_parquet=stream_local_parquet,
        join_temp_data_context_window=join_context_window,
        join_temp_data_min_remainder_tokens=join_min_remainder,
    )

    stats = dataset.get_stats()
    logging.info("Dataset Statistics:")
    logging.info("  - Total examples: %s", stats["total_examples"])
    logging.info("  - Completed batches: %s", stats["completed_batches"])
    logging.info("  - Samples processed: %s", stats["total_samples_processed"])
    logging.info("  - Parallel workers: %s", stats["num_workers"])
    logging.info("  - Cache directory: %s", stats["cache_dir"])
    logging.info("  - Join context window: %s", stats.get("join_temp_data_context_window", 0))

    batch_size = training_runtime.get("batch_size", None)
    if cli_batch_size is not None:
        if cli_batch_size <= 0:
            raise ValueError("--batch-size must be > 0")
        batch_size = cli_batch_size
    if batch_size is None:
        batch_size = 1

    dataloader_workers = int(training_runtime.get("dataloader_workers", 2))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_workers,
        pin_memory=resolved_device.startswith("cuda"),
        drop_last=True,
    )

    logging.info("Dataset size: %s examples", len(dataset))
    logging.info("Batch size: %s", batch_size)
    logging.info("Steps per epoch: %s", len(dataloader))

    return dataloader, dataset, stats, int(batch_size)


def _resolve_vocab_size(model: torch.nn.Module, fallback: int = 50_000) -> int:
    """Resolve the vocabulary size from a model.

    Checks ``model.config.vocab_size`` first, then ``get_input_embeddings()``,
    falling back to the provided default.

    Args:
        model: PyTorch model.
        fallback: Default vocab size if neither source is available.

    Returns:
        Resolved vocabulary size as an integer.
    """
    if hasattr(model, "config") and getattr(model.config, "vocab_size", None):
        return int(model.config.vocab_size)
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None and getattr(emb, "num_embeddings", None):
            return int(emb.num_embeddings)
    return fallback


def _validate_gpu_temp_guard_config(training_config: TrainingConfig):
    """Validate GPU thermal guard threshold constraints.

    Args:
        training_config: :class:`TrainingConfig` to validate.

    Raises:
        ValueError: If pause/resume thresholds are <= 0, resume >= pause,
            poll interval <= 0, critical threshold <= 0 when provided, or
            checkpoint grace seconds <= 0.
    """
    pause_threshold = float(training_config.gpu_temp_pause_threshold_c)
    resume_threshold = float(training_config.gpu_temp_resume_threshold_c)
    poll_interval = float(training_config.gpu_temp_poll_interval_seconds)
    critical_threshold = training_config.gpu_temp_critical_threshold_c
    grace_seconds = float(training_config.gpu_temp_checkpoint_grace_seconds)
    if pause_threshold <= 0:
        raise ValueError("gpu_temp_pause_threshold_c must be > 0")
    if resume_threshold <= 0:
        raise ValueError("gpu_temp_resume_threshold_c must be > 0")
    if resume_threshold >= pause_threshold:
        raise ValueError("gpu_temp_resume_threshold_c must be < gpu_temp_pause_threshold_c")
    if poll_interval <= 0:
        raise ValueError("gpu_temp_poll_interval_seconds must be > 0")
    if grace_seconds <= 0:
        raise ValueError("gpu_temp_checkpoint_grace_seconds must be > 0")
    if critical_threshold is not None and float(critical_threshold) <= 0:
        raise ValueError("gpu_temp_critical_threshold_c must be > 0 when provided")


def _apply_gpu_temp_guard_overrides(
    training_config: TrainingConfig,
    args: argparse.Namespace,
    resolved_device: str,
):
    """Apply CLI overrides and device-based adjustments to GPU thermal guard config.

    Args:
        training_config: :class:`TrainingConfig` to modify in-place.
        args: Parsed CLI arguments with optional thermal guard overrides.
        resolved_device: Resolved PyTorch device string. Disables thermal
            guard and switching for non-CUDA devices.
    """
    if getattr(args, "gpu_temp_guard", None) is not None:
        training_config.gpu_temp_guard_enabled = bool(args.gpu_temp_guard)
    if getattr(args, "gpu_temp_pause_threshold_c", None) is not None:
        training_config.gpu_temp_pause_threshold_c = float(args.gpu_temp_pause_threshold_c)
    if getattr(args, "gpu_temp_resume_threshold_c", None) is not None:
        training_config.gpu_temp_resume_threshold_c = float(args.gpu_temp_resume_threshold_c)
    if getattr(args, "gpu_temp_critical_threshold_c", None) is not None:
        training_config.gpu_temp_critical_threshold_c = float(args.gpu_temp_critical_threshold_c)
    if getattr(args, "gpu_temp_poll_interval_seconds", None) is not None:
        training_config.gpu_temp_poll_interval_seconds = float(args.gpu_temp_poll_interval_seconds)
    if getattr(args, "gpu_temp_checkpoint_grace_seconds", None) is not None:
        training_config.gpu_temp_checkpoint_grace_seconds = float(args.gpu_temp_checkpoint_grace_seconds)
    if getattr(args, "resume_from_checkpoint", None) is not None:
        training_config.resume_from_checkpoint = str(args.resume_from_checkpoint)
    if getattr(args, "switch_on_thermal", None) is not None:
        training_config.switch_on_thermal = bool(args.switch_on_thermal)

    if not str(resolved_device).startswith("cuda"):
        if training_config.gpu_temp_guard_enabled:
            logging.info(
                "Disabling GPU temperature guard because resolved device is '%s'",
                resolved_device,
            )
        training_config.gpu_temp_guard_enabled = False
        if training_config.switch_on_thermal:
            logging.info(
                "Disabling switch_on_thermal because resolved device is '%s'",
                resolved_device,
            )
            training_config.switch_on_thermal = False

    _validate_gpu_temp_guard_config(training_config)

    if training_config.gpu_temp_guard_enabled and str(resolved_device).startswith("cuda"):
        if training_config.gpu_temp_critical_threshold_c is not None and not training_config.switch_on_thermal:
            logging.info(
                "Enabling switch_on_thermal automatically because gpu_temp_critical_threshold_c is configured."
            )
            training_config.switch_on_thermal = True
        if training_config.gpu_temp_critical_threshold_c is None:
            logging.info(
                "GPU thermal critical action disabled (gpu_temp_critical_threshold_c is null); pause-only thermal supervisor remains active."
            )
            if training_config.switch_on_thermal:
                logging.info(
                    "switch_on_thermal=true but gpu_temp_critical_threshold_c is null; no GPU->CPU thermal switching will occur."
                )
        else:
            logging.info(
                "GPU thermal critical action configured (critical>=%.1fC, switch_on_thermal=%s).",
                float(training_config.gpu_temp_critical_threshold_c),
                bool(training_config.switch_on_thermal),
            )
        logging.info(
            "GPU thermal switch_on_thermal=%s",
            bool(training_config.switch_on_thermal),
        )


def _build_supervisor_child_argv(
    args: argparse.Namespace,
    training_config: TrainingConfig,
    config_path: str,
) -> list:
    """Build the argv used to (re)launch the training child under the supervisor.

    The child is a fresh invocation of the training entrypoint with the same
    config, its own guard disabled (``--no-gpu-temp-guard`` — the supervisor
    owns thermal monitoring), and asked to resume from the latest rolling
    checkpoint (``--resume-from-checkpoint auto``). The supervisor prepends
    the Python interpreter.
    """
    # Determine the entrypoint. Prefer the installed console script module
    # path; fall back to running the package's main CLI module.
    argv = []
    # Use ``python -m src.cli train ...`` so it works in editable installs
    # without relying on the console-script shim being on PATH.
    argv.extend(["-m", "src.cli", "train"])
    argv.extend(["--config", str(config_path)])
    argv.extend(["--device", str(args.device)])
    # Disable the child's own guard to avoid nesting; the supervisor monitors.
    argv.append("--no-gpu-temp-guard")
    # Pass the resume spec so each re-launch resumes from the latest checkpoint.
    argv.extend(["--resume-from-checkpoint", "auto"])
    # Forward threshold overrides so the child logs consistent config.
    if getattr(args, "batch_size", None) is not None:
        argv.extend(["--batch-size", str(args.batch_size)])
    if getattr(args, "model_mode", None):
        argv.extend(["--model-mode", str(args.model_mode)])
    if getattr(args, "gpu_temp_pause_threshold_c", None) is not None:
        argv.extend(["--gpu-temp-pause-threshold-c", str(args.gpu_temp_pause_threshold_c)])
    if getattr(args, "gpu_temp_resume_threshold_c", None) is not None:
        argv.extend(["--gpu-temp-resume-threshold-c", str(args.gpu_temp_resume_threshold_c)])
    if getattr(args, "gpu_temp_critical_threshold_c", None) is not None:
        argv.extend(["--gpu-temp-critical-threshold-c", str(args.gpu_temp_critical_threshold_c)])
    if getattr(args, "gpu_temp_poll_interval_seconds", None) is not None:
        argv.extend(["--gpu-temp-poll-interval-seconds", str(args.gpu_temp_poll_interval_seconds)])
    if bool(training_config.switch_on_thermal):
        argv.append("--switch-on-thermal")
    return argv


def _build_sbert_supervisor_child_argv(
    loaded: "LoadedTrainingConfig",
    training_config: TrainingConfig,
    resolved_device: str,
) -> list:
    """Build the supervisor child argv for the SBERT task.

    Mirrors :func:`_run_sbert_task` argument construction but re-runs the
    SBERT entrypoint as a subprocess so the supervisor can kill/restart it.
    """
    sbert_cfg = loaded.training_runtime.get("sbert", {}) or {}
    if not isinstance(sbert_cfg, dict):
        raise ValueError("training.sbert must be an object")
    argv = ["-m", "src.sbert.train_sbert"]
    argv.extend(["--base-model", str(loaded.base_model)])
    argv.extend(["--output_dir", str(sbert_cfg.get("output_dir", "./output/sbert_base_model"))])
    argv.extend(["--batch_size", str(int(sbert_cfg.get("batch_size", 16)))])
    argv.extend(["--epochs", str(int(sbert_cfg.get("epochs", 4)))])
    argv.extend(["--learning_rate", str(float(sbert_cfg.get("learning_rate", 2e-5)))])
    argv.extend(["--max_eval_samples", str(int(sbert_cfg.get("max_eval_samples", 10000)))])
    argv.extend(["--pooling_mode", str(sbert_cfg.get("pooling_mode", "mean"))])
    argv.extend(["--resample_std", str(float(sbert_cfg.get("resample_std", 0.3)))])
    argv.extend(["--device", str(resolved_device)])
    argv.append("--no-gpu-temp-guard")
    argv.append("--resume_from_checkpoint")
    gradient_accumulation_steps = sbert_cfg.get("gradient_accumulation_steps")
    if gradient_accumulation_steps is not None:
        argv.extend(["--gradient_accumulation_steps", str(int(gradient_accumulation_steps))])
    max_grad_norm = sbert_cfg.get("max_grad_norm")
    if max_grad_norm is not None:
        argv.extend(["--max_grad_norm", str(float(max_grad_norm))])
    dataset_name = str(sbert_cfg.get("dataset_name", "")).strip()
    if dataset_name:
        argv.extend(["--dataset_name", dataset_name])
    dataset_type = str(sbert_cfg.get("dataset_type", "")).strip()
    if dataset_type:
        argv.extend(["--dataset_type", dataset_type])
    max_train_samples = sbert_cfg.get("max_train_samples")
    if max_train_samples is not None:
        argv.extend(["--max_train_samples", str(int(max_train_samples))])
    warmup_steps = sbert_cfg.get("warmup_steps")
    if warmup_steps is not None:
        argv.extend(["--warmup_steps", str(int(warmup_steps))])
    evaluation_steps = sbert_cfg.get("evaluation_steps")
    if evaluation_steps is not None:
        argv.extend(["--evaluation_steps", str(int(evaluation_steps))])
    checkpoint_save_steps = sbert_cfg.get("checkpoint_save_steps")
    if checkpoint_save_steps is not None:
        argv.extend(["--checkpoint_save_steps", str(int(checkpoint_save_steps))])
    max_seq_length = sbert_cfg.get("max_seq_length")
    if max_seq_length is not None:
        argv.extend(["--max_seq_length", str(int(max_seq_length))])
    if not bool(sbert_cfg.get("use_amp", True)):
        argv.append("--no_amp")
    if not bool(sbert_cfg.get("resample_balanced", True)):
        argv.append("--no_resample")
    if bool(sbert_cfg.get("standardize_scores", False)):
        argv.append("--standardize_scores")
    if bool(sbert_cfg.get("trust_remote_code", False)):
        argv.append("--trust_remote_code")
    columns_cfg = sbert_cfg.get("columns", {}) or {}
    if isinstance(columns_cfg, dict):
        column_arg_map = {
            "sentence1": "--col_sentence1",
            "sentence2": "--col_sentence2",
            "similarity": "--col_similarity",
            "query": "--col_query",
            "positive": "--col_positive",
            "negatives": "--col_negatives",
            "question": "--col_question",
            "answer": "--col_answer",
        }
        for key, cli_flag in column_arg_map.items():
            value = columns_cfg.get(key)
            if value is None:
                continue
            text_value = str(value).strip()
            if text_value:
                argv.extend([cli_flag, text_value])
    query_prefix = sbert_cfg.get("query_prefix")
    if query_prefix is not None:
        argv.extend(["--query_prefix", str(query_prefix)])
    document_prefix = sbert_cfg.get("document_prefix")
    if document_prefix is not None:
        argv.extend(["--document_prefix", str(document_prefix)])
    sbert_optimizer = sbert_cfg.get("optimizer")
    if isinstance(sbert_optimizer, dict):
        optimizer_class = sbert_optimizer.get("optimizer_class")
        if isinstance(optimizer_class, str) and optimizer_class.strip():
            argv.extend(["--optimizer_class", str(optimizer_class).strip()])
        optimizer_params = sbert_optimizer.get("parameters", {})
        if isinstance(optimizer_params, dict):
            import json as _json
            for key, value in optimizer_params.items():
                argv.extend(["--optimizer_param", f"{key}={_json.dumps(value)}"])
    return argv


def _run_under_supervisor(
    args: argparse.Namespace,
    training_config: TrainingConfig,
    resolved_device: str,
    config_path: str,
    task: str,
) -> int:
    """Hand off training to the CPU-side GPU temperature supervisor.

    Args:
        args: Parsed CLI arguments (used to build the child argv).
        training_config: Resolved :class:`TrainingConfig` (thermal fields).
        resolved_device: Resolved PyTorch device string.
        config_path: Absolute path to the YAML config.
        task: ``"mlm"`` or ``"sbert"``.

    Returns:
        The supervisor's exit code.
    """
    from ..utils.gpu_temp_guard import GPUTempSupervisor
    import sys

    if task == "sbert":
        try:
            from ..training.config_loader import LoadedTrainingConfig  # noqa: F401
        except ImportError:
            from training.config_loader import LoadedTrainingConfig  # noqa: F401
        # Re-load to pass a LoadedTrainingConfig to the SBERT argv builder.
        from .config_loader import load_training_config
        loaded = load_training_config(config_path)
        child_argv = _build_sbert_supervisor_child_argv(loaded, training_config, resolved_device)
        checkpoint_dir = os.path.join(
            str(loaded.training_runtime.get("sbert", {}).get("output_dir", "./output/sbert_base_model")),
            "checkpoints",
        )
    else:
        child_argv = _build_supervisor_child_argv(args, training_config, config_path)
        checkpoint_dir = "checkpoints"

    logging.info(
        "🌡️ Launching GPU temperature supervisor (pause>=%.1fC, resume<=%.1fC, critical=%s, "
        "poll=%.1fs, grace=%.1fs, switch_on_thermal=%s)",
        float(training_config.gpu_temp_pause_threshold_c),
        float(training_config.gpu_temp_resume_threshold_c),
        (
            f"{float(training_config.gpu_temp_critical_threshold_c):.1f}C"
            if training_config.gpu_temp_critical_threshold_c is not None
            else "disabled"
        ),
        float(training_config.gpu_temp_poll_interval_seconds),
        float(training_config.gpu_temp_checkpoint_grace_seconds),
        bool(training_config.switch_on_thermal),
    )

    supervisor = GPUTempSupervisor(
        enabled=True,
        device=resolved_device,
        nvml_device_index=int(training_config.nvml_device_index),
        pause_threshold_c=float(training_config.gpu_temp_pause_threshold_c),
        resume_threshold_c=float(training_config.gpu_temp_resume_threshold_c),
        critical_threshold_c=training_config.gpu_temp_critical_threshold_c,
        poll_interval_seconds=float(training_config.gpu_temp_poll_interval_seconds),
        checkpoint_dir=checkpoint_dir,
        checkpoint_grace_seconds=float(training_config.gpu_temp_checkpoint_grace_seconds),
        switch_on_thermal=bool(training_config.switch_on_thermal),
        child_argv=child_argv,
    )
    return int(supervisor.run())


def _run_sbert_task(
    loaded: LoadedTrainingConfig,
    resolved_device: str,
    training_config: TrainingConfig,
) -> int:
    """Dispatch SBERT fine-tuning from the training pipeline.

    Translates the YAML config's ``training.sbert`` block into CLI
    arguments for :func:`sbert.train_sbert.main`.

    Args:
        loaded: Validated :class:`LoadedTrainingConfig` with ``task="sbert"``.
        resolved_device: Resolved PyTorch device string.
        training_config: :class:`TrainingConfig` for thermal guard settings.

    Returns:
        Exit code (0 for success, non-zero for failure).

    Raises:
        ValueError: If ``base_model`` is not set or ``training.sbert``
            is not a valid object.
    """
    if not loaded.base_model:
        raise ValueError("training.task=sbert requires top-level base_model")

    try:
        from ..sbert.train_sbert import main as sbert_train_main
    except ImportError:
        from sbert.train_sbert import main as sbert_train_main

    sbert_cfg = loaded.training_runtime.get("sbert", {}) or {}
    if not isinstance(sbert_cfg, dict):
        raise ValueError("training.sbert must be an object")

    argv = [
        "--base-model",
        loaded.base_model,
        "--output_dir",
        str(sbert_cfg.get("output_dir", "./output/sbert_base_model")),
        "--batch_size",
        str(int(sbert_cfg.get("batch_size", 16))),
        "--epochs",
        str(int(sbert_cfg.get("epochs", 4))),
        "--learning_rate",
        str(float(sbert_cfg.get("learning_rate", 2e-5))),
        "--max_eval_samples",
        str(int(sbert_cfg.get("max_eval_samples", 10000))),
        "--pooling_mode",
        str(sbert_cfg.get("pooling_mode", "mean")),
        "--resample_std",
        str(float(sbert_cfg.get("resample_std", 0.3))),
        "--device",
        resolved_device,
    ]
    gradient_accumulation_steps = sbert_cfg.get("gradient_accumulation_steps")
    if gradient_accumulation_steps is not None:
        argv.extend(
            [
                "--gradient_accumulation_steps",
                str(int(gradient_accumulation_steps)),
            ]
        )
    max_grad_norm = sbert_cfg.get("max_grad_norm")
    if max_grad_norm is not None:
        argv.extend(["--max_grad_norm", str(float(max_grad_norm))])

    dataset_name = str(sbert_cfg.get("dataset_name", "")).strip()
    if dataset_name:
        argv.extend(["--dataset_name", dataset_name])
    dataset_type = str(sbert_cfg.get("dataset_type", "")).strip()
    if dataset_type:
        argv.extend(["--dataset_type", dataset_type])

    max_train_samples = sbert_cfg.get("max_train_samples")
    if max_train_samples is not None:
        argv.extend(["--max_train_samples", str(int(max_train_samples))])

    warmup_steps = sbert_cfg.get("warmup_steps")
    if warmup_steps is not None:
        argv.extend(["--warmup_steps", str(int(warmup_steps))])

    evaluation_steps = sbert_cfg.get("evaluation_steps")
    if evaluation_steps is not None:
        argv.extend(["--evaluation_steps", str(int(evaluation_steps))])
    checkpoint_save_steps = sbert_cfg.get("checkpoint_save_steps")
    if checkpoint_save_steps is not None:
        argv.extend(["--checkpoint_save_steps", str(int(checkpoint_save_steps))])
    if bool(sbert_cfg.get("resume_from_checkpoint", False)):
        argv.append("--resume_from_checkpoint")

    max_seq_length = sbert_cfg.get("max_seq_length")
    if max_seq_length is not None:
        argv.extend(["--max_seq_length", str(int(max_seq_length))])

    if not bool(sbert_cfg.get("use_amp", True)):
        argv.append("--no_amp")
    if not bool(sbert_cfg.get("resample_balanced", True)):
        argv.append("--no_resample")
    if bool(sbert_cfg.get("standardize_scores", False)):
        argv.append("--standardize_scores")
    if bool(sbert_cfg.get("trust_remote_code", False)):
        argv.append("--trust_remote_code")

    columns_cfg = sbert_cfg.get("columns", {}) or {}
    if not isinstance(columns_cfg, dict):
        raise ValueError("training.sbert.columns must be an object when provided")
    column_arg_map = {
        "sentence1": "--col_sentence1",
        "sentence2": "--col_sentence2",
        "similarity": "--col_similarity",
        "query": "--col_query",
        "positive": "--col_positive",
        "negatives": "--col_negatives",
        "question": "--col_question",
        "answer": "--col_answer",
    }
    for key, cli_flag in column_arg_map.items():
        value = columns_cfg.get(key)
        if value is None:
            continue
        text_value = str(value).strip()
        if text_value:
            argv.extend([cli_flag, text_value])

    query_prefix = sbert_cfg.get("query_prefix")
    if query_prefix is not None:
        argv.extend(["--query_prefix", str(query_prefix)])

    document_prefix = sbert_cfg.get("document_prefix")
    if document_prefix is not None:
        argv.extend(["--document_prefix", str(document_prefix)])

    if bool(training_config.gpu_temp_guard_enabled):
        argv.append("--gpu-temp-guard")
    else:
        argv.append("--no-gpu-temp-guard")
    if bool(training_config.switch_on_thermal):
        argv.append("--switch-on-thermal")
    else:
        argv.append("--no-switch-on-thermal")
    argv.extend(
        [
            "--gpu-temp-pause-threshold-c",
            str(float(training_config.gpu_temp_pause_threshold_c)),
            "--gpu-temp-resume-threshold-c",
            str(float(training_config.gpu_temp_resume_threshold_c)),
            "--gpu-temp-poll-interval-seconds",
            str(float(training_config.gpu_temp_poll_interval_seconds)),
            "--gpu-temp-checkpoint-grace-seconds",
            str(float(training_config.gpu_temp_checkpoint_grace_seconds)),
            "--nvml-device-index",
            str(int(training_config.nvml_device_index)),
        ]
    )
    if training_config.gpu_temp_critical_threshold_c is not None:
        argv.extend(
            [
                "--gpu-temp-critical-threshold-c",
                str(float(training_config.gpu_temp_critical_threshold_c)),
            ]
        )
    argv.extend(
        [
            "--csv-log-path",
            str(training_config.csv_log_path),
            "--telemetry-log-interval",
            str(int(training_config.telemetry_log_interval)),
            "--gpu-metrics-backend",
            str(training_config.gpu_metrics_backend),
        ]
    )
    if bool(training_config.csv_rotate_on_schema_change):
        argv.append("--csv-rotate-on-schema-change")
    else:
        argv.append("--no-csv-rotate-on-schema-change")
    if bool(training_config.enable_block_grad_norms):
        argv.append("--enable-block-grad-norms")
    else:
        argv.append("--no-enable-block-grad-norms")

    # Pass custom optimizer if specified in sbert config
    sbert_optimizer = sbert_cfg.get("optimizer")
    if isinstance(sbert_optimizer, dict):
        optimizer_class = sbert_optimizer.get("optimizer_class")
        if isinstance(optimizer_class, str) and optimizer_class.strip():
            argv.extend(["--optimizer_class", str(optimizer_class).strip()])
        optimizer_params = sbert_optimizer.get("parameters", {})
        if isinstance(optimizer_params, dict):
            for key, value in optimizer_params.items():
                import json as _json
                argv.extend(["--optimizer_param", f"{key}={_json.dumps(value)}"])

    wandb_project = sbert_cfg.get("wandb_project")
    if wandb_project is not None:
        argv.extend(["--wandb_project", str(wandb_project)])

    logging.info("Dispatching SBERT finetuning with base_model=%s", loaded.base_model)
    result = sbert_train_main(argv)
    return int(result) if isinstance(result, int) else 0


# ==================== MAIN EXECUTION ====================
def main(argv=None):
    """Main training pipeline for TORMENTED-BERT-Frankenstein and base-model finetuning."""
    parser = argparse.ArgumentParser(description="Train models from YAML configs")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument(
        "--config-name",
        type=str,
        default=os.environ.get("CONFIG_NAME", "mini"),
        help="Config name under configs (without extension)",
    )
    parser.add_argument("--list-configs", action="store_true", help="List available configs and exit")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from YAML")
    parser.add_argument(
        "--model-mode",
        choices=["frankenstein", "mini", "frankesteindecoder"],
        default=None,
        help="Deprecated: use --config-name instead",
    )
    parser.add_argument(
        "--device",
        choices=SUPPORTED_DEVICE_CHOICES,
        default="auto",
        help="Device to run training on (default: auto)",
    )
    gpu_temp_group = parser.add_mutually_exclusive_group()
    gpu_temp_group.add_argument(
        "--gpu-temp-guard",
        dest="gpu_temp_guard",
        action="store_true",
        help="Enable GPU thermal guard for CUDA training.",
    )
    gpu_temp_group.add_argument(
        "--no-gpu-temp-guard",
        dest="gpu_temp_guard",
        action="store_false",
        help="Disable GPU thermal guard.",
    )
    parser.set_defaults(gpu_temp_guard=None)
    switch_on_thermal_group = parser.add_mutually_exclusive_group()
    switch_on_thermal_group.add_argument(
        "--switch-on-thermal",
        dest="switch_on_thermal",
        action="store_true",
        help="Enable GPU->CPU thermal switching (critical threshold) with resume back to GPU.",
    )
    switch_on_thermal_group.add_argument(
        "--no-switch-on-thermal",
        dest="switch_on_thermal",
        action="store_false",
        help="Disable thermal GPU->CPU switching and keep pause-only guard behavior.",
    )
    parser.set_defaults(switch_on_thermal=None)
    parser.add_argument(
        "--gpu-temp-pause-threshold-c",
        type=float,
        default=None,
        help="Pause training when GPU temperature is above this value.",
    )
    parser.add_argument(
        "--gpu-temp-resume-threshold-c",
        type=float,
        default=None,
        help="Resume training when GPU temperature drops to this value or lower.",
    )
    parser.add_argument(
        "--gpu-temp-critical-threshold-c",
        type=float,
        default=None,
        help="Optional critical threshold used by switch_on_thermal to trigger GPU->CPU training fallback.",
    )
    parser.add_argument(
        "--gpu-temp-poll-interval-seconds",
        type=float,
        default=None,
        help="Polling interval in seconds while paused for temperature cooldown.",
    )
    parser.add_argument(
        "--gpu-temp-checkpoint-grace-seconds",
        type=float,
        default=None,
        help="Seconds the supervisor waits for the child to flush a checkpoint after SIGUSR1 before SIGKILL.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume training from a checkpoint ('auto' = latest rolling checkpoint, or a path).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("⚡ Starting training pipeline ⚡")
    logging.info("Current directory: %s", os.getcwd())
    resolved_device = resolve_torch_device(args.device)
    logging.info("Training device requested='%s', resolved='%s'", args.device, resolved_device)

    try:
        import psutil

        logging.info("Available storage: %.2fGB", psutil.disk_usage(".").free / 1024**3)
    except ImportError:
        logging.warning("psutil not installed, storage monitoring limited")

    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs")
    available_configs = list_config_paths(config_dir)
    if args.list_configs:
        logging.info("Available configs:")
        for name, path in available_configs.items():
            logging.info("  - %s: %s", name, path)
        return

    if args.config:
        config_path = args.config
    else:
        config_name = args.config_name
        if args.model_mode and args.model_mode not in ("", None):
            config_name = args.model_mode
        config_path = available_configs.get(config_name)

    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    loaded = load_training_config(config_path)
    training_config = loaded.training_config
    training_runtime = loaded.training_runtime
    _apply_gpu_temp_guard_overrides(training_config, args, resolved_device)
    logging.info("Using config: %s", config_path)
    logging.info("Training task: %s", loaded.task)

    # When the GPU temperature supervisor is enabled on CUDA and this process
    # is NOT already a supervisor child, hand off to the CPU-side supervisor.
    # The supervisor re-executes the training command as a subprocess and
    # manages kill/checkpoint/resume on thermal events.
    if (
        bool(training_config.gpu_temp_guard_enabled)
        and str(resolved_device).startswith("cuda")
        and os.environ.get("FRANKESTEIN_SUPERVISOR_CHILD") != "1"
    ):
        return _run_under_supervisor(args, training_config, resolved_device, config_path, loaded.task)

    if loaded.task == "sbert":
        return _run_sbert_task(loaded, resolved_device, training_config)

    if loaded.base_model:
        logging.info("\n" + "=" * 60)
        logging.info("Step 1: Loading base MLM model + external tokenizer")
        logging.info("=" * 60)
        model, tokenizer = _load_base_model_and_tokenizer(loaded)
        model_descriptor = loaded.base_model
        runtime_config = getattr(model, "config", None)
    else:
        model, tokenizer, runtime_config = _load_legacy_tormented_model(loaded)
        model_descriptor = loaded.model_class or "frankenstein"

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Model Descriptor: %s", model_descriptor)
    logging.info("Total Parameters: %.2fM", total_params / 1e6)
    logging.info("Trainable Parameters: %.2fM", trainable_params / 1e6)

    dataloader, dataset, stats, _ = _build_dataloader(
        tokenizer=tokenizer,
        training_runtime=training_runtime,
        resolved_device=resolved_device,
        cli_batch_size=args.batch_size,
    )

    logging.info("\n" + "=" * 60)
    logging.info("Step 4: MLM training (%s)", model_descriptor)
    logging.info("=" * 60)

    trainer = TitanTrainer(
        model,
        runtime_config,
        training_config=training_config,
        device=resolved_device,
    )

    # Resume from the latest rolling checkpoint when requested (supervisor
    # re-launches pass --resume-from-checkpoint auto).
    resume_epoch = 0
    resume_spec = training_config.resume_from_checkpoint
    if resume_spec:
        resume_epoch = trainer.resume_from_latest_checkpoint(resume_spec)

    num_epochs = int(training_runtime.get("num_epochs", 5))
    nan_detected = False

    try:
        for epoch in range(resume_epoch, num_epochs):
            logging.info("\n🚀 Starting Epoch %s/%s", epoch + 1, num_epochs)
            try:
                avg_loss, should_stop = trainer.train_epoch(dataloader, epoch)
                if should_stop:
                    logging.error("❌ Training stopped due to NaN/instability at epoch %s", epoch + 1)
                    nan_detected = True
                    break

                logging.info("✅ Epoch %s completed - Average Loss: %.4f", epoch + 1, avg_loss)
                checkpoint_path = trainer.save_checkpoint(epoch, suffix="_epoch_end")
                logging.info("💾 Epoch checkpoint saved: %s", checkpoint_path)

                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_cached = torch.cuda.memory_reserved() / 1024**3
                    logging.info(
                        "GPU Memory - Allocated: %.2fGB, Cached: %.2fGB",
                        memory_allocated,
                        memory_cached,
                    )

                storage_used = trainer.storage_manager.used_bytes / 1024**3
                logging.info("Storage used: %.2fGB / 300GB", storage_used)
                if storage_used > 250:
                    logging.warning("Approaching storage limit, stopping training")
                    break

            except Exception as exc:
                logging.error("Error in epoch %s: %s", epoch + 1, exc)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    emergency_path = trainer.save_checkpoint(epoch, suffix="_emergency")
                    logging.info("Emergency checkpoint saved: %s", emergency_path)
                except Exception:
                    logging.error("Failed to save emergency checkpoint")
                raise
    finally:
        trainer.close()

    if nan_detected:
        logging.error("\n" + "=" * 60)
        logging.error("🚨 TRAINING TERMINATED DUE TO NaN/INF")
        logging.error("Check training_metrics.csv for progression leading to failure")
        logging.error("=" * 60)
    else:
        logging.info("\n" + "=" * 60)
        logging.info("🎉 Training completed successfully")
        logging.info("=" * 60)

        model.eval()
        with torch.no_grad():
            vocab_size = _resolve_vocab_size(model)
            seq_len = int(training_runtime.get("max_length", 512))
            test_input = torch.randint(0, vocab_size, (1, seq_len), device=resolved_device)

            logging.info("🔍 Testing final model forward pass...")
            try:
                test_output = model(input_ids=test_input)
            except TypeError:
                test_output = model(test_input)

            logits = test_output
            if hasattr(test_output, "logits"):
                logits = test_output.logits
            elif isinstance(test_output, dict) and "logits" in test_output:
                logits = test_output["logits"]

            logging.info("✅ Model output shape: %s", tuple(logits.shape))
            logging.info(
                "Output range: [%.3f, %.3f]",
                logits.min().item(),
                logits.max().item(),
            )

    if loaded.base_model and hasattr(model, "save_pretrained"):
        hf_output_dir = str(training_runtime.get("hf_output_dir", "checkpoints/hf_final"))
        trainer.save_pretrained_artifacts(hf_output_dir, tokenizer=tokenizer)

    logging.info("\n🧹 Cleaning up temporary files...")
    if hasattr(tokenizer, "storage_manager") and tokenizer.storage_manager is not None:
        tokenizer.storage_manager.cleanup()
    trainer.storage_manager.cleanup()

    logging.info("💡 Dataset cache preserved for fault recovery")
    logging.info("   Location: %s", stats["cache_dir"])

    logging.info("\n📁 Checkpoint Summary:")
    logging.info("  Rolling checkpoints kept: %s", len(trainer.rolling_checkpoints))
    for checkpoint_path in trainer.rolling_checkpoints:
        logging.info("    - %s", checkpoint_path)
    logging.info("  Best model checkpoints: %s", len(trainer.best_checkpoints))
    for neg_loss, checkpoint_path in sorted(trainer.best_checkpoints, reverse=True):
        logging.info("    - %s (loss=%.6f)", checkpoint_path, -neg_loss)

    logging.info("\n📊 Training metrics saved to: %s", training_config.csv_log_path)
    logging.info("✨ Training pipeline completed!")


if __name__ == "__main__":
    main()
