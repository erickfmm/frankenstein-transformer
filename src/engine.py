#!/usr/bin/env python3
"""Non-CLI engine API for Frankenstein Transformer.

Thin, reusable façade over model construction, tokenizer setup, dataset
wiring, and the training loop — without argparse or supervisor subprocess
spawn. This is the entry point the DashAI ``dashai-frankenstein`` plugin (and
notebooks / other embedding hosts) drives to train, save, and load models
in-process. The CLI (:mod:`src.training.main`) delegates here.

The functions here are pure wrappers around the same builders the CLI used;
no behavior is changed — ``supervisor="auto"`` reproduces the legacy CLI path,
``supervisor="off"`` runs the loop in the current process.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

try:
    from .training.streaming_mlm_dataset import StreamingMLMDataset
    from .training.trainer import TitanTrainer, TrainingConfig
    from .training.config_loader import LoadedTrainingConfig, load_training_config
    from .model.config import FrankensteinModelConfig
    from .model.frankenstein_decoder import FrankensteinDecoder
    from .model.frankenstein_encoder import FrankensteinEncoder
    from .model.frankenstein_vit import FrankensteinViT
    from .utils.device import SUPPORTED_DEVICE_CHOICES, resolve_torch_device
except ImportError:
    from training.streaming_mlm_dataset import StreamingMLMDataset
    from training.trainer import TitanTrainer, TrainingConfig
    from training.config_loader import LoadedTrainingConfig, load_training_config
    from model.config import FrankensteinModelConfig
    from model.frankenstein_decoder import FrankensteinDecoder
    from model.frankenstein_encoder import FrankensteinEncoder
    from model.frankenstein_vit import FrankensteinViT
    from utils.device import SUPPORTED_DEVICE_CHOICES, resolve_torch_device


__all__ = [
    "TrainResult",
    "build_model",
    "build_tokenizer",
    "build_base_model_and_tokenizer",
    "build_dataloader",
    "train_from_config",
    "save_checkpoint",
    "load_checkpoint",
    "SUPPORTED_DEVICE_CHOICES",
    "resolve_torch_device",
]


@dataclass
class TrainResult:
    """Result of an in-process training run.

    Attributes:
        model: The trained ``nn.Module`` (on the training device).
        loaded: The validated :class:`LoadedTrainingConfig`.
        tokenizer: The tokenizer used during training (SPM or HF).
        final_epoch: The epoch number training stopped at (0-indexed).
        best_loss: Best (lowest) loss observed during training.
        checkpoint_path: Optional path to the final epoch-end checkpoint.
    """

    model: torch.nn.Module
    loaded: LoadedTrainingConfig
    tokenizer: Any
    final_epoch: int = 0
    best_loss: Optional[float] = None
    checkpoint_path: Optional[str] = None


def _load_base_model_and_tokenizer(
    loaded: LoadedTrainingConfig,
) -> Tuple[torch.nn.Module, Any]:
    """Load a HuggingFace masked-LM model and tokenizer for base-model training.

    Args:
        loaded: Validated :class:`LoadedTrainingConfig` with ``base_model`` set.

    Returns:
        Tuple of ``(model, tokenizer)``.

    Raises:
        RuntimeError: If ``transformers`` is not installed.
        ValueError: If ``tokenizer.name_or_path`` is missing, or the tokenizer
            lacks a pad token or mask token.
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


def build_tokenizer(loaded: LoadedTrainingConfig) -> Any:
    """Build (or load) the custom Spanish SPM tokenizer for a Frankenstein model.

    Args:
        loaded: Validated :class:`LoadedTrainingConfig` with ``model_config``.

    Returns:
        A :class:`SpanishSPMTokenizer` instance.

    Raises:
        ValueError: If ``model_config`` is ``None``.
    """
    config = loaded.model_config
    if config is None:
        raise ValueError("model config is required when base_model is not provided")

    from .tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer

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
    return tokenizer


def build_model(
    model_class: Optional[str],
    config: FrankensteinModelConfig,
    *,
    num_labels: Optional[int] = None,
) -> torch.nn.Module:
    """Build a Frankenstein model from a model class name and config.

    Args:
        model_class: ``"frankensteindecoder"``, ``"frankenstein_vit"``, or any
            other value (``"frankenstein"``) for the encoder.
        config: :class:`FrankensteinModelConfig`.
        num_labels: Optional number of classes to enable the encoder's
            sequence-level classification head (DashAI Strategy A). Only
            applies to the ``FrankensteinEncoder`` path.

    Returns:
        The constructed ``nn.Module``.
    """
    if num_labels is not None:
        config.num_labels = int(num_labels)
        if int(num_labels) >= 1:
            config.classification_head = True

    if model_class == "frankensteindecoder":
        config.mode = "decoder"
        model = FrankensteinDecoder(config)
    elif model_class == "frankenstein_vit":
        model = FrankensteinViT(config)
    else:
        model = FrankensteinEncoder(config)

    logging.info("Model Config:")
    logging.info("  - Model Class: %s", model_class)
    logging.info("  - Hidden Size: %s", config.hidden_size)
    logging.info(
        "  - Layers: %s x %s = %s logical",
        config.num_layers,
        config.num_loops,
        config.num_layers * config.num_loops,
    )
    logging.info("  - Layer Pattern: %s", config.layer_pattern)
    logging.info(
        "  - BitNet: %s (bitnet_routers: %s)", config.use_bitnet, getattr(config, "bitnet_routers", False)
    )
    logging.info("  - ODE Solver: %s (%s steps)", config.ode_solver, config.ode_steps)
    logging.info("  - Norm Type: %s", config.norm_type)
    return model


def build_base_model_and_tokenizer(
    loaded: LoadedTrainingConfig,
) -> Tuple[torch.nn.Module, Any, Any]:
    """Build an HF base model + tokenizer for the ``base_model`` training path.

    Args:
        loaded: Validated :class:`LoadedTrainingConfig` with ``base_model``.

    Returns:
        Tuple of ``(model, tokenizer, runtime_config)``.
    """
    model, tokenizer = _load_base_model_and_tokenizer(loaded)
    runtime_config = getattr(model, "config", None)
    return model, tokenizer, runtime_config


def build_dataloader(
    tokenizer: Any,
    training_runtime: Dict[str, Any],
    resolved_device: str,
    cli_batch_size: Optional[int],
    task: str = "mlm",
) -> Tuple[DataLoader, StreamingMLMDataset, Dict[str, Any], int]:
    """Build the streaming MLM/CLM dataset and DataLoader.

    Args:
        tokenizer: Tokenizer instance (SPM or HuggingFace).
        training_runtime: Runtime configuration dictionary from YAML.
        resolved_device: Resolved PyTorch device string.
        cli_batch_size: Optional batch size override.
        task: Training task identifier. ``"causal_lm"`` disables MLM masking
            so cached sequences are stored unmasked (``labels == input_ids``)
            for autoregressive next-token loss. Any other value keeps the
            default BERT-style MLM masking.

    Returns:
        Tuple of ``(dataloader, dataset, stats, batch_size)``.

    Raises:
        ValueError: If ``cli_batch_size`` is <= 0.
    """
    task = str(task or "mlm").strip().lower()
    apply_mlm_mask = task != "causal_lm"
    logging.info("\n" + "=" * 60)
    logging.info(
        "Step 3: Preparing %s dataset with resilient caching",
        "CLM (unmasked)" if not apply_mlm_mask else "MLM",
    )
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
        apply_mlm_mask=apply_mlm_mask,
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


def train_from_config(
    config: str | Dict[str, Any],
    *,
    dataset: Any = None,
    device: str = "auto",
    supervisor: str = "auto",
    metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    batch_size: Optional[int] = None,
    num_epochs: Optional[int] = None,
    config_path_hint: Optional[str] = None,
    training_config: Optional[TrainingConfig] = None,
) -> TrainResult:
    """Train a model from a Frankenstein YAML config in-process.

    This drives the full MLM / causal-LM / base-model / vision training loop
    without spawning a supervisor subprocess. ``supervisor="off"`` is required
    when embedding training in an external job-queue host (e.g. a DashAI Huey
    worker); ``supervisor="auto"`` preserves the legacy behavior.

    Args:
        config: Path to a YAML config, or a dict already parsed from YAML.
        dataset: Optional pre-built dataset to train on. If ``None``, a
            streaming MLM dataset is built from ``training_runtime``.
        device: Target device string (default ``"auto"``).
        supervisor: ``"auto"`` (legacy) or ``"off"`` (in-process). Note: this
            function runs in-process regardless; ``"auto"`` is accepted for
            API symmetry with the CLI but does not spawn a subprocess here.
        metrics_callback: Optional callable invoked with a per-step dict
            (``{"level": "step", ...}``) and a per-epoch dict
            (``{"level": "epoch", ...}``) mirroring the CSV metrics.
        batch_size: Optional batch-size override.
        num_epochs: Optional epoch-count override.
        config_path_hint: Optional directory hint for relative output paths.
        training_config: Optional pre-resolved :class:`TrainingConfig` to use
            instead of reloading it from the config file. Enables callers that
            have already applied CLI overrides (e.g. GPU thermal guard) to
            inject them into the training run.

    Returns:
        A :class:`TrainResult` with the trained model, loaded config, and
        tokenizer.

    Raises:
        ValueError: If the config is invalid.
    """
    if isinstance(config, dict):
        # Loader expects a file path; serialize the dict to a temp YAML.
        import tempfile
        import yaml

        fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="frankenstein_engine_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
            return _train_from_config_path(
                tmp_path,
                dataset=dataset,
                device=device,
                supervisor=supervisor,
                metrics_callback=metrics_callback,
                batch_size=batch_size,
                num_epochs=num_epochs,
                config_path_hint=config_path_hint,
                training_config=training_config,
            )
        finally:
            os.remove(tmp_path)

    return _train_from_config_path(
        config,
        dataset=dataset,
        device=device,
        supervisor=supervisor,
        metrics_callback=metrics_callback,
        batch_size=batch_size,
        num_epochs=num_epochs,
        config_path_hint=config_path_hint,
        training_config=training_config,
    )


def _train_from_config_path(
    config_path: str,
    *,
    dataset: Any = None,
    device: str = "auto",
    supervisor: str = "auto",
    metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    batch_size: Optional[int] = None,
    num_epochs: Optional[int] = None,
    config_path_hint: Optional[str] = None,
    training_config: Optional[TrainingConfig] = None,
) -> TrainResult:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    loaded = load_training_config(config_path)
    if training_config is not None:
        loaded.training_config = training_config
    training_config = loaded.training_config
    training_runtime = loaded.training_runtime
    if config_path_hint:
        os.chdir(config_path_hint)

    resolved_device = resolve_torch_device(device)
    if batch_size is not None:
        training_runtime["batch_size"] = batch_size
    if num_epochs is not None:
        training_runtime["num_epochs"] = num_epochs

    task = loaded.task

    # SBERT fine-tuning: dispatch to the in-process SBERT entrypoint.
    if task == "sbert":
        return _train_sbert(loaded, resolved_device, training_config, metrics_callback)

    # Vision task (frankenstein_vit).
    vision_tasks = {"patch_prediction", "classification", "segmentation"}
    if task in vision_tasks:
        return _train_vision(loaded, resolved_device, training_config, batch_size, num_epochs, metrics_callback)

    if loaded.base_model:
        logging.info("\n" + "=" * 60)
        logging.info("Step 1: Loading base MLM model + external tokenizer")
        logging.info("=" * 60)
        model, tokenizer, runtime_config = build_base_model_and_tokenizer(loaded)
        model_descriptor = loaded.base_model
    else:
        model, tokenizer = _build_legacy(loaded)
        runtime_config = loaded.model_config
        model_descriptor = loaded.model_class or "frankenstein"

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Model Descriptor: %s", model_descriptor)
    logging.info("Total Parameters: %.2fM", total_params / 1e6)
    logging.info("Trainable Parameters: %.2fM", trainable_params / 1e6)

    dataloader, mlm_dataset, stats, _ = build_dataloader(
        tokenizer=tokenizer,
        training_runtime=training_runtime,
        resolved_device=resolved_device,
        cli_batch_size=batch_size,
        task=task,
    )

    logging.info("\n" + "=" * 60)
    logging.info("Step 4: %s training (%s)", task.upper(), model_descriptor)
    logging.info("=" * 60)

    trainer = TitanTrainer(
        model,
        runtime_config,
        training_config=training_config,
        device=resolved_device,
        task=task,
        metrics_callback=metrics_callback,
    )

    resume_epoch = 0
    resume_spec = training_config.resume_from_checkpoint
    if resume_spec:
        resume_epoch = trainer.resume_from_latest_checkpoint(resume_spec)

    num_epochs_effective = int(training_runtime.get("num_epochs", 5))
    if num_epochs is not None:
        num_epochs_effective = int(num_epochs)
    nan_detected = False
    best_loss = trainer.best_loss

    try:
        for epoch in range(resume_epoch, num_epochs_effective):
            logging.info("\n🚀 Starting Epoch %s/%s", epoch + 1, num_epochs_effective)
            try:
                avg_loss, should_stop = trainer.train_epoch(dataloader, epoch)
                if should_stop:
                    logging.error("❌ Training stopped due to NaN/instability at epoch %s", epoch + 1)
                    nan_detected = True
                    best_loss = trainer.best_loss
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
                    best_loss = trainer.best_loss
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
        else:
            best_loss = trainer.best_loss
    finally:
        trainer.close()

    if nan_detected:
        logging.error("\n" + "=" * 60)
        logging.error("🚨 TRAINING TERMINATED DUE TO NaN/INF")
        logging.error("Check training_metrics.csv for progression leading to failure")
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

    return TrainResult(
        model=model,
        loaded=loaded,
        tokenizer=tokenizer,
        final_epoch=resume_epoch + num_epochs_effective,
        best_loss=best_loss,
    )


def _build_legacy(loaded: LoadedTrainingConfig) -> Tuple[torch.nn.Module, Any]:
    """Build a custom Frankenstein model + SPM tokenizer (legacy path)."""
    tokenizer = build_tokenizer(loaded)
    config = loaded.model_config
    if config is None:
        raise ValueError("model config is required when base_model is not provided")
    model = build_model(loaded.model_class, config)
    return model, tokenizer


def _train_vision(
    loaded: LoadedTrainingConfig,
    resolved_device: str,
    training_config: TrainingConfig,
    batch_size: Optional[int],
    num_epochs: Optional[int],
    metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> TrainResult:
    """Run a vision task (patch_prediction / classification / segmentation)."""
    from .training.vision_dataset import DummyImageDataset

    logging.info("\n" + "=" * 60)
    logging.info("Vision task: %s (model_class=%s)", loaded.task, loaded.model_class)
    logging.info("=" * 60)

    config = loaded.model_config
    model = FrankensteinViT(config)

    # Vision tasks reuse the top-level training.batch_size / training.num_epochs
    # (same fields as mlm / causal_lm). The task sub-block (e.g. training.classification)
    # only carries task-specific knobs (label_smoothing, seg_loss_* weights, etc.).
    batch_size_eff = loaded.training_runtime.get("batch_size", batch_size or 32)
    num_epochs_eff = loaded.training_runtime.get("num_epochs", num_epochs or 1)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total Parameters: %.2fM", total_params / 1e6)
    logging.info("Trainable Parameters: %.2fM", trainable_params / 1e6)

    dataset_name = loaded.dataset_config.get("dataset_name")
    dataset_dir = loaded.dataset_config.get("dataset_dir")
    if dataset_name or dataset_dir:
        from .training.vision_dataset import ImageDataset

        dataset = ImageDataset(loaded.dataset_config, loaded.image_config, loaded.task)
    else:
        logging.info("No dataset_name/dataset_dir — using DummyImageDataset for smoke test")
        dataset = DummyImageDataset(
            task=loaded.task,
            num_samples=64,
            image_height=config.image_height,
            image_width=config.image_width,
            in_channels=config.in_channels,
            patch_size=config.patch_size,
            num_classes=config.num_classes,
            num_seg_classes=config.num_seg_classes,
            mask_ratio=config.mask_ratio,
            prediction_target=config.prediction_target,
        )

    dataloader = DataLoader(dataset, batch_size=batch_size_eff, shuffle=True)

    logging.info("Step 4: %s training", loaded.task.upper())
    trainer = TitanTrainer(
        model,
        config,
        training_config=training_config,
        device=resolved_device,
        task=loaded.task,
        metrics_callback=metrics_callback,
    )

    avg_loss = 0.0
    try:
        for epoch in range(int(num_epochs_eff)):
            avg_loss, should_stop = trainer.train_epoch(dataloader, epoch)
            if should_stop:
                break
    finally:
        trainer.close()

    logging.info("Vision training complete.")
    return TrainResult(
        model=model,
        loaded=loaded,
        tokenizer=None,
        final_epoch=int(num_epochs_eff),
        best_loss=avg_loss,
    )


def _train_sbert(
    loaded: LoadedTrainingConfig,
    resolved_device: str,
    training_config: TrainingConfig,
    metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> TrainResult:
    """Run SBERT fine-tuning via the in-process ``sbert.train_sbert`` entrypoint.

    Translates the YAML config's ``training.sbert`` block into the argv
    expected by :func:`sbert.train_sbert.main` and invokes it in-process
    (no subprocess). The SBERT trainer writes its own checkpoints/logs to
    ``sbert.output_dir`` and manages its own metrics, so ``metrics_callback``
    is accepted for signature symmetry but not wired into the SBERT loop.

    Args:
        loaded: Validated :class:`LoadedTrainingConfig` with ``task="sbert"``
            and a non-empty ``base_model``.
        resolved_device: Resolved PyTorch device string.
        training_config: :class:`TrainingConfig` for thermal-guard settings.
        metrics_callback: Accepted for API symmetry; not used by SBERT today.

    Returns:
        A :class:`TrainResult` (``model=None`` — SBERT persists to its own
        ``output_dir``; the caller may reload the saved checkpoint).

    Raises:
        ValueError: If ``base_model`` is not set or ``training.sbert`` is
            not a valid object.
        RuntimeError: If the SBERT trainer returns a non-zero exit code.
    """
    if not loaded.base_model:
        raise ValueError("training.task=sbert requires top-level base_model")

    try:
        from .sbert.train_sbert import main as sbert_train_main
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
            import json as _json

            for key, value in optimizer_params.items():
                argv.extend(["--optimizer_param", f"{key}={_json.dumps(value)}"])

    wandb_project = sbert_cfg.get("wandb_project")
    if wandb_project is not None:
        argv.extend(["--wandb_project", str(wandb_project)])

    logging.info("Dispatching SBERT finetuning with base_model=%s", loaded.base_model)
    result = sbert_train_main(argv)
    exit_code = int(result) if isinstance(result, int) else 0
    if exit_code != 0:
        raise RuntimeError(f"SBERT training failed with exit code {exit_code}")

    return TrainResult(
        model=None,
        loaded=loaded,
        tokenizer=None,
        final_epoch=int(sbert_cfg.get("epochs", 4)),
    )


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    loaded: LoadedTrainingConfig,
    tokenizer: Any = None,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a model + its validated config + tokenizer + extra metadata.

    This produces a single artifact bundle that can be reloaded with
    :func:`load_checkpoint`. The model state dict is stored alongside the
    validated Frankenstein YAML config and the tokenizer (if it supports
    persistence), plus an arbitrary ``extra`` dict (e.g. ``num_labels``).

    Args:
        path: Directory to save the artifacts into.
        model: The ``nn.Module`` to persist.
        loaded: The validated :class:`LoadedTrainingConfig`.
        tokenizer: Optional tokenizer to persist.
        extra: Optional extra metadata dict (e.g. ``{"num_labels": N}``).

    Returns:
        The path to the saved state-dict file.
    """
    import json
    import yaml

    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, "model.pt")
    torch.save({"state_dict": model.state_dict()}, model_path)

    config_yaml = os.path.join(path, "config.yaml")
    config_dict = loaded.config_dict if hasattr(loaded, "config_dict") else {}
    with open(config_yaml, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config_dict, handle, sort_keys=False)

    meta = {
        "task": loaded.task,
        "model_class": loaded.model_class,
        "base_model": loaded.base_model,
        "extra": extra or {},
    }
    with open(os.path.join(path, "dashai_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    if tokenizer is not None:
        tokenizer_dir = os.path.join(path, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        if hasattr(tokenizer, "save_pretrained"):
            try:
                tokenizer.save_pretrained(tokenizer_dir)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Could not save tokenizer: %s", exc)
        elif hasattr(tokenizer, "model_path") and tokenizer.model_path and os.path.exists(tokenizer.model_path):
            try:
                import shutil

                shutil.copy2(tokenizer.model_path, os.path.join(tokenizer_dir, "tokenizer.model"))
            except Exception as exc:  # noqa: BLE001
                logging.warning("Could not copy SPM tokenizer: %s", exc)

    return model_path


def load_checkpoint(path: str) -> Tuple[torch.nn.Module, LoadedTrainingConfig, Any, Dict[str, Any]]:
    """Load a model bundle saved by :func:`save_checkpoint`.

    Rebuilds the ``nn.Module`` from ``config.yaml`` (via
    :func:`load_training_config` + :func:`build_model`) and restores weights.
    ``extra`` metadata (including ``num_labels`` for the classification head)
    is applied to the config before building the model so the head is
    reconstructed correctly.

    Args:
        path: Directory written by :func:`save_checkpoint`.

    Returns:
        Tuple of ``(model, loaded, tokenizer, extra)``.
    """
    config_path = os.path.join(path, "config.yaml")
    model_path = os.path.join(path, "model.pt")

    loaded = load_training_config(config_path)

    meta = {}
    meta_path = os.path.join(path, "dashai_meta.json")
    if os.path.exists(meta_path):
        import json

        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)

    extra = meta.get("extra") or {}
    num_labels = extra.get("num_labels")
    if loaded.model_config is not None and num_labels is not None:
        loaded.model_config.num_labels = int(num_labels)
        loaded.model_config.classification_head = bool(int(num_labels) >= 1)

    if loaded.model_config is not None:
        model = build_model(loaded.model_class, loaded.model_config)
        tokenizer = None
    else:
        # base_model path: rebuild the HF model + tokenizer from config.
        model, tokenizer, _ = build_base_model_and_tokenizer(loaded)

    tokenizer_dir = os.path.join(path, "tokenizer")
    if os.path.isdir(tokenizer_dir):
        if os.path.exists(os.path.join(tokenizer_dir, "config.json")):
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Could not load HF tokenizer: %s", exc)
        elif os.path.exists(os.path.join(tokenizer_dir, "tokenizer.model")):
            try:
                from .tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer

                vocab_size = int(getattr(loaded.model_config, "vocab_size", 50_000))
                tokenizer = SpanishSPMTokenizer(vocab_size=vocab_size, model_path=os.path.join(tokenizer_dir, "tokenizer.model"))
            except Exception as exc:  # noqa: BLE001
                logging.warning("Could not load SPM tokenizer: %s", exc)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, loaded, tokenizer, extra
