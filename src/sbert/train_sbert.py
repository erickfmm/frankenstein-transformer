#!/usr/bin/env python3
"""SBERT fine-tuning for TORMENTED-BERT-Frankenstein.

Fine-tunes a TormentedBert model (or any HuggingFace base model) on
Spanish sentence similarity using cosine similarity loss and Siamese
training. Supports paired-similarity, triplets, and QA dataset formats
with balanced resampling and score standardization.
"""

import os
import json
import csv
import re
import time
import subprocess
import inspect

# Tesla P40 (sm_61) and other legacy GPUs are often unstable with Triton/Inductor paths.
# Set FRANKENSTEIN_DISABLE_TRITON=0 to re-enable explicitly.
if os.environ.get("FRANKENSTEIN_DISABLE_TRITON", "1").strip().lower() in {"1", "true", "yes", "on"}:
    os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
    os.environ.setdefault("USE_TRITON", "0")

import torch
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    models,
    losses,
    evaluation,
    InputExample
)
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm_progress
import numpy as np

try:
    from ..model.tormented_bert_frankestein import TormentedBertFrankenstein, UltraConfig
    from ..model.optimizer.factory import OPTIMIZER_REGISTRY
    from ..utils.device import SUPPORTED_DEVICE_CHOICES, resolve_torch_device
    from ..utils.gpu_temp_guard import GPUTemperatureGuard
except ImportError:
    from model.tormented_bert_frankestein import TormentedBertFrankenstein, UltraConfig
    from model.optimizer.factory import OPTIMIZER_REGISTRY
    from utils.device import SUPPORTED_DEVICE_CHOICES, resolve_torch_device
    from utils.gpu_temp_guard import GPUTemperatureGuard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SUPPORTED_SBERT_DATASET_TYPES = ("paired_similarity", "triplets", "qa")

DEFAULT_SBERT_COLUMNS = {
    "paired_similarity": {
        "sentence1": "sentence1",
        "sentence2": "sentence2",
        "similarity": "score",
    },
    "triplets": {
        "query": "Q",
        "positive": "POS",
        "negatives": "NEGs",
    },
    "qa": {
        "question": "Question",
        "answer": "Answer",
    },
}


def _extract_primary_score(score: Any) -> Optional[float]:
    """Extract a representative scalar score from evaluator outputs."""
    if score is None:
        return None
    if isinstance(score, (int, float)):
        return float(score)
    if isinstance(score, dict):
        preferred_keys = (
            "cosine_spearman",
            "spearman_cosine",
            "spearman",
            "eval_spearman",
            "score",
        )
        for key in preferred_keys:
            value = score.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        for value in score.values():
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _format_score_for_log(score: Any) -> str:
    primary = _extract_primary_score(score)
    if primary is not None:
        return f"{primary:.4f}"
    if isinstance(score, dict):
        return str({k: v for k, v in score.items() if isinstance(v, (int, float))})
    return str(score)


class TormentedBertSentenceTransformer:
    """Wrapper to adapt TormentedBert for Sentence-BERT training.

    Builds a :class:`SentenceTransformer` model by wrapping a
    :class:`TormentedBertFrankenstein` (or any HuggingFace base model)
    with a pooling layer and L2 normalization. Supports loading from
    pretrained checkpoints or initializing from scratch.

    Attributes:
        max_seq_length: Maximum sequence length for tokenization.
        pooling_mode: Pooling strategy (``"mean"``, ``"cls"``, ``"max"``).
        trust_remote_code: Whether to allow remote code for HF models.
        device: Resolved PyTorch device string.
        base_model: The underlying TormentedBert model (or ``None`` for HF).
        config: :class:`UltraConfig` for the model.
        model: The constructed :class:`SentenceTransformer` instance.
    """

    def __init__(
        self,
        model_config: Optional[UltraConfig] = None,
        pretrained_path: Optional[str] = None,
        base_model_name_or_path: Optional[str] = None,
        max_seq_length: int = 512,
        pooling_mode: str = "mean",
        trust_remote_code: bool = False,
        device: str = "auto"
    ):
        """Initialize the SBERT model wrapper.

        Args:
            model_config: :class:`UltraConfig` for training from scratch.
            pretrained_path: Path to a pretrained TormentedBert checkpoint.
            base_model_name_or_path: HuggingFace model ID or path for
                base-model SBERT fine-tuning.
            max_seq_length: Maximum sequence length.
            pooling_mode: Pooling strategy (``"mean"``, ``"cls"``, or
                ``"max"``).
            trust_remote_code: Whether to allow remote code execution for
                HuggingFace models/tokenizers.
            device: Device string for model placement.
        """
        self.max_seq_length = max_seq_length
        self.pooling_mode = pooling_mode
        self.trust_remote_code = bool(trust_remote_code)
        self.device = resolve_torch_device(device)

        if base_model_name_or_path:
            logger.info(f"Loading base model for SBERT from {base_model_name_or_path}")
            self.base_model = None
            self.config = None
            self.model = self._build_hf_sentence_transformer(base_model_name_or_path)
            return
        
        # Initialize or load base model
        if pretrained_path and os.path.exists(pretrained_path):
            logger.info(f"Loading pretrained model from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Extract config if available
            if 'config' in checkpoint:
                config = checkpoint['config']
            else:
                config = model_config or self._get_default_config()
            
            self.base_model = TormentedBertFrankenstein(config)
            
            # Load weights (handle potential key mismatches)
            if 'model_state_dict' in checkpoint:
                self.base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'state_dict' in checkpoint:
                self.base_model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.base_model.load_state_dict(checkpoint, strict=False)
        else:
            logger.info("Initializing model from scratch")
            config = model_config or self._get_default_config()
            self.base_model = TormentedBertFrankenstein(config)

        self.base_model.to(self.device)
        
        self.config = config
        self.model = self._build_sentence_transformer()
    
    def _get_default_config(self) -> UltraConfig:
        """Get default config optimized for P40 24GB"""
        return UltraConfig(
            vocab_size=50000,
            hidden_size=768,        # Reduced for SBERT training
            num_layers=12,
            num_loops=1,            # Less recursion for faster training
            num_heads=12,
            ode_steps=1,            # Minimal ODE steps for speed
            dropout=0.1,
            use_bitnet=True
        )
    
    def _build_sentence_transformer(self) -> SentenceTransformer:
        """Build SentenceTransformer model with custom base"""
        
        # Create a custom transformer model wrapper
        class TormentedBertWrapper(torch.nn.Module):
            def __init__(self, tormented_model, hidden_size):
                super().__init__()
                self.tormented_model = tormented_model
                self.config_keys = ['max_seq_length']
                self.max_seq_length = 512
                
                # Remove language modeling head if exists
                if hasattr(self.tormented_model, 'head'):
                    self.tormented_model.head = torch.nn.Identity()
            
            def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
                """Forward pass compatible with sentence-transformers"""
                input_ids = features['input_ids']
                attention_mask = features.get('attention_mask', None)
                
                # Get embeddings from base model
                # TormentedBert returns [batch, seq, hidden]
                output = self.tormented_model(input_ids)
                
                # Apply attention mask if provided
                if attention_mask is not None:
                    output = output * attention_mask.unsqueeze(-1)
                
                features['token_embeddings'] = output
                features['attention_mask'] = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
                
                return features
        
        # Create wrapper
        word_embedding_model = TormentedBertWrapper(
            self.base_model,
            self.config.hidden_size
        )
        
        # Add pooling layer
        pooling_model = models.Pooling(
            self.config.hidden_size,
            pooling_mode_mean_tokens=(self.pooling_mode == "mean"),
            pooling_mode_cls_token=(self.pooling_mode == "cls"),
            pooling_mode_max_tokens=(self.pooling_mode == "max")
        )
        
        # Add normalization layer (important for cosine similarity)
        normalize = models.Normalize()
        
        # Build sentence transformer
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model, normalize],
            device=self.device,
        )
        
        return model

    def _build_hf_sentence_transformer(self, base_model_name_or_path: str) -> SentenceTransformer:
        """Build SentenceTransformer from a base HF model identifier/path."""
        transformer_kwargs = {
            "max_seq_length": self.max_seq_length,
        }

        # sentence-transformers changed this constructor across versions.
        try:
            word_embedding_model = models.Transformer(
                base_model_name_or_path,
                max_seq_length=self.max_seq_length,
                model_args={"trust_remote_code": self.trust_remote_code},
                tokenizer_args={"trust_remote_code": self.trust_remote_code},
            )
        except TypeError:
            word_embedding_model = models.Transformer(
                base_model_name_or_path,
                **transformer_kwargs,
            )

        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=(self.pooling_mode == "mean"),
            pooling_mode_cls_token=(self.pooling_mode == "cls"),
            pooling_mode_max_tokens=(self.pooling_mode == "max")
        )

        normalize = models.Normalize()
        return SentenceTransformer(
            modules=[word_embedding_model, pooling_model, normalize],
            device=self.device,
        )
    
    def get_model(self) -> SentenceTransformer:
        """Get the SentenceTransformer model"""
        return self.model


class SBERTTrainer:
    """Trainer for SBERT fine-tuning with cosine similarity loss.

    Manages dataset loading, balanced resampling, score standardization,
    training loop execution via ``sentence-transformers``, GPU thermal
    guard integration, CSV telemetry logging, and checkpoint management.

    Attributes:
        model: The :class:`SentenceTransformer` model.
        output_dir: Directory for checkpoints and final model.
        batch_size: Training batch size.
        gradient_accumulation_steps: Accumulation steps before optimizer update.
        max_grad_norm: Gradient clipping max norm (0 disables).
        epochs: Number of training epochs.
        warmup_steps: Learning rate warmup steps.
        evaluation_steps: Steps between evaluations.
        checkpoint_save_steps: Step interval for rolling checkpoints.
        resume_from_checkpoint: Whether to resume from latest checkpoint.
        learning_rate: Base learning rate.
        use_amp: Whether to use automatic mixed precision.
        device: Resolved PyTorch device string.
        gpu_temp_guard: :class:`GPUTemperatureGuard` for thermal safety.
        global_step: Current global training step counter.
        dataset_type: Active dataset format type.
        train_examples: List of :class:`InputExample` for training.
        eval_examples: List of :class:`InputExample` for evaluation.
        evaluator: Optional :class:`EmbeddingSimilarityEvaluator`.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        output_dir: str = "./output/sbert_tormented_v2",
        batch_size: int = 16,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        epochs: int = 4,
        warmup_steps: int = 1000,
        evaluation_steps: int = 5000,
        checkpoint_save_steps: int = 1000,
        resume_from_checkpoint: bool = False,
        learning_rate: float = 2e-5,
        use_amp: bool = True,
        device: str = "auto",
        gpu_temp_guard_enabled: bool = True,
        switch_on_thermal: bool = False,
        gpu_temp_pause_threshold_c: float = 90.0,
        gpu_temp_resume_threshold_c: float = 80.0,
        gpu_temp_critical_threshold_c: Optional[float] = None,
        gpu_temp_poll_interval_seconds: float = 30.0,
        nvml_device_index: int = 0,
        csv_log_path: str = "training_metrics.csv",
        csv_rotate_on_schema_change: bool = True,
        gpu_metrics_backend: str = "nvml",
        enable_block_grad_norms: bool = True,
        telemetry_log_interval: int = 1,
        optimizer_class: Optional[str] = None,
        optimizer_parameters: Optional[Dict[str, Any]] = None,
        wandb_project: Optional[str] = None,
    ):
        """Initialize the SBERT trainer.

        Args:
            model: :class:`SentenceTransformer` model to fine-tune.
            output_dir: Directory to save checkpoints and final model.
            batch_size: Training batch size.
            gradient_accumulation_steps: Number of accumulation steps
                before an optimizer update.
            max_grad_norm: Gradient clipping max norm (0 disables clipping).
            epochs: Number of training epochs.
            warmup_steps: Warmup steps for learning rate scheduler.
            evaluation_steps: Steps between validation evaluations.
            checkpoint_save_steps: Step interval for rolling checkpoint
                saves (<= 0 disables).
            resume_from_checkpoint: If ``True``, resume from the latest
                checkpoint in ``output_dir/checkpoints``.
            learning_rate: Base learning rate.
            use_amp: Whether to use automatic mixed precision.
            device: Device string for training.
            gpu_temp_guard_enabled: Whether GPU thermal guard is active.
            switch_on_thermal: Whether GPU→CPU thermal switching is enabled.
            gpu_temp_pause_threshold_c: Pause threshold in Celsius.
            gpu_temp_resume_threshold_c: Resume threshold in Celsius.
            gpu_temp_critical_threshold_c: Optional critical threshold.
            gpu_temp_poll_interval_seconds: Poll interval while paused.
            nvml_device_index: NVML device index for multi-GPU.
            csv_log_path: Path for CSV telemetry log.
            csv_rotate_on_schema_change: Whether to rotate CSV on schema
                mismatch.
            gpu_metrics_backend: GPU telemetry backend (``"nvml"`` or
                ``"none"``).
            enable_block_grad_norms: Whether to collect per-block gradient
                norms.
            telemetry_log_interval: Steps between telemetry log writes.
            optimizer_class: Optional custom optimizer class name from
                the registry.
            optimizer_parameters: Optional custom optimizer parameters dict.
            wandb_project: Optional Weights & Biases project name.
        """
        self.model = model
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.gradient_accumulation_steps = max(int(gradient_accumulation_steps), 1)
        self.max_grad_norm = max(float(max_grad_norm), 0.0)
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.evaluation_steps = evaluation_steps
        self.checkpoint_save_steps = int(checkpoint_save_steps)
        self.resume_from_checkpoint = bool(resume_from_checkpoint)
        self.checkpoint_save_total_limit = 3
        self.learning_rate = learning_rate
        self.use_amp = use_amp
        self.device = resolve_torch_device(device)
        self._switch_on_thermal = bool(switch_on_thermal)
        self._last_guard_temp_c: Optional[float] = None
        self._checkpoint_request_pending = False
        self._is_supervisor_child = os.environ.get("FRANKESTEIN_SUPERVISOR_CHILD") == "1"
        self.global_step = 0
        self.csv_log_path = str(csv_log_path)
        self.csv_rotate_on_schema_change = bool(csv_rotate_on_schema_change)
        self.gpu_metrics_backend = str(gpu_metrics_backend)
        self.enable_block_grad_norms = bool(enable_block_grad_norms)
        self.telemetry_log_interval = max(int(telemetry_log_interval), 1)
        self.csv_columns = self._get_csv_columns()
        self.csv_file = None
        self.csv_writer = None
        self._nvml_module = None
        self._nvml_handle = None
        self._nvml_disabled = False
        self._nvml_warning_logged = False
        self.gpu_temp_guard = GPUTemperatureGuard(
            enabled=bool(gpu_temp_guard_enabled),
            device=self.device,
            nvml_device_index=int(nvml_device_index),
            pause_threshold_c=float(gpu_temp_pause_threshold_c),
            resume_threshold_c=float(gpu_temp_resume_threshold_c),
            critical_threshold_c=gpu_temp_critical_threshold_c,
            poll_interval_seconds=float(gpu_temp_poll_interval_seconds),
        )
        if self._is_supervisor_child and self._is_main_thread():
            self._install_supervisor_sigusr1_handler()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        self._init_csv_logger()

        # Optimizer configuration
        self.optimizer_class = str(optimizer_class).strip().lower() if optimizer_class else None
        self.optimizer_parameters = dict(optimizer_parameters) if optimizer_parameters else {}
        self.wandb_project = wandb_project

        # Training metadata
        self.training_metadata = {
            'start_time': datetime.now().isoformat(),
            'batch_size': batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'effective_batch_size': int(self.batch_size) * int(self.gradient_accumulation_steps),
            'max_grad_norm': self.max_grad_norm,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'checkpoint_save_steps': self.checkpoint_save_steps,
            'checkpoint_save_total_limit': self.checkpoint_save_total_limit,
            'resume_from_checkpoint': self.resume_from_checkpoint,
            'switch_on_thermal': self._switch_on_thermal,
            'csv_log_path': self.csv_log_path,
            'telemetry_log_interval': self.telemetry_log_interval,
            'optimizer_class': self.optimizer_class or 'default',
            'wandb_project': self.wandb_project,
        }
        if self.gpu_temp_guard.is_active:
            logger.info(
                "GPU thermal supervisor enabled for SBERT (pause>=%.1fC, resume<=%.1fC, poll=%.1fs, critical=%s)",
                float(gpu_temp_pause_threshold_c),
                float(gpu_temp_resume_threshold_c),
                float(gpu_temp_poll_interval_seconds),
                (
                    f"{float(gpu_temp_critical_threshold_c):.1f}C"
                    if gpu_temp_critical_threshold_c is not None
                    else "disabled"
                ),
            )
        else:
            logger.info("GPU thermal guard disabled for SBERT")
        logger.info("SBERT switch_on_thermal=%s", self._switch_on_thermal)

    def _get_csv_columns(self) -> List[str]:
        return [
            'timestamp', 'epoch', 'step', 'global_step',
            'loss', 'accuracy', 'learning_rate', 'grad_norm',
            'scaler_scale', 'gpu_memory_gb', 'gpu_cached_gb',
            'has_nan', 'has_inf', 'has_zero', 'repair_action',
            'gpu_temp_c', 'gpu_power_w', 'gpu_util_pct', 'gpu_mem_used_mib',
            'grad_norm_embeddings', 'grad_norm_attention', 'grad_norm_ffn',
            'grad_norm_experts', 'grad_norm_router', 'grad_norm_ode',
            'grad_norm_retnet', 'grad_norm_mamba', 'grad_norm_norms',
            'grad_norm_head', 'grad_norm_other',
            'step_time_ms', 'tokens_per_sec', 'clip_ratio', 'effective_batch_size'
        ]

    def _read_csv_header(self, csv_path: str) -> Optional[List[str]]:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            return None
        with open(csv_path, 'r', newline='', encoding='utf-8') as handle:
            reader = csv.reader(handle)
            return next(reader, None)

    def _init_csv_logger(self):
        csv_path = self.csv_log_path
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        file_exists = os.path.exists(csv_path)
        if file_exists:
            existing_header = self._read_csv_header(csv_path)
            if existing_header is not None and existing_header != self.csv_columns:
                if self.csv_rotate_on_schema_change:
                    base, ext = os.path.splitext(csv_path)
                    if not ext:
                        ext = ".csv"
                    rotated_path = f"{base}.{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
                    os.replace(csv_path, rotated_path)
                    logger.info("CSV schema changed. Rotated previous log to: %s", rotated_path)
                    file_exists = False
                else:
                    raise RuntimeError(
                        "CSV schema mismatch for existing metrics file. "
                        "Enable csv_rotate_on_schema_change or remove old CSV."
                    )

        self.csv_file = open(csv_path, 'a', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        should_write_header = (not file_exists) or os.path.getsize(csv_path) == 0
        if should_write_header:
            self.csv_writer.writerow(self.csv_columns)
            self.csv_file.flush()

    def _parse_smi_value(self, raw: str) -> float:
        value = (raw or "").strip()
        if not value or value.lower() in {"n/a", "[not supported]"}:
            return 0.0
        match = re.search(r"[-+]?\d*\.?\d+", value)
        if match is None:
            return 0.0
        try:
            return float(match.group(0))
        except ValueError:
            return 0.0

    def _get_default_gpu_telemetry(self) -> Dict[str, float]:
        return {
            "gpu_temp_c": 0.0,
            "gpu_power_w": 0.0,
            "gpu_util_pct": 0.0,
            "gpu_mem_used_mib": 0.0,
        }

    def _get_gpu_telemetry_from_nvidia_smi(self) -> Dict[str, float]:
        telemetry = self._get_default_gpu_telemetry()
        cmd = [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            str(int(self.gpu_temp_guard._nvml_device_index)),  # best-effort parity with guard
        ]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            line = result.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                telemetry["gpu_temp_c"] = self._parse_smi_value(parts[0])
                telemetry["gpu_power_w"] = self._parse_smi_value(parts[1])
                telemetry["gpu_util_pct"] = self._parse_smi_value(parts[2])
                telemetry["gpu_mem_used_mib"] = self._parse_smi_value(parts[3])
        except Exception:
            pass
        return telemetry

    def _get_gpu_telemetry(self) -> Dict[str, float]:
        telemetry = self._get_default_gpu_telemetry()
        if not torch.cuda.is_available():
            return telemetry
        backend = self.gpu_metrics_backend.strip().lower()
        if backend == "none":
            return telemetry
        if backend != "nvml":
            return telemetry

        if self._nvml_disabled:
            return self._get_gpu_telemetry_from_nvidia_smi()

        if self._nvml_module is None or self._nvml_handle is None:
            try:
                import pynvml  # type: ignore

                pynvml.nvmlInit()
                idx = int(self.gpu_temp_guard._nvml_device_index)
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                self._nvml_module = pynvml
            except Exception as exc:
                self._nvml_disabled = True
                if not self._nvml_warning_logged:
                    logger.warning("NVML telemetry unavailable, falling back to nvidia-smi: %s", exc)
                    self._nvml_warning_logged = True
                return self._get_gpu_telemetry_from_nvidia_smi()

        try:
            pynvml = self._nvml_module
            handle = self._nvml_handle
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            telemetry["gpu_temp_c"] = float(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            )
            telemetry["gpu_power_w"] = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
            telemetry["gpu_util_pct"] = float(utilization.gpu)
            telemetry["gpu_mem_used_mib"] = float(memory_info.used) / (1024.0 ** 2)
        except Exception as exc:
            if not self._nvml_warning_logged:
                logger.warning("Failed reading NVML telemetry, falling back to nvidia-smi: %s", exc)
                self._nvml_warning_logged = True
            return self._get_gpu_telemetry_from_nvidia_smi()
        return telemetry

    def _get_default_block_grad_norms(self) -> Dict[str, float]:
        return {
            "embeddings": 0.0,
            "attention": 0.0,
            "ffn": 0.0,
            "experts": 0.0,
            "router": 0.0,
            "norms": 0.0,
            "head": 0.0,
            "other": 0.0,
        }

    def _log_step_to_csv(
        self,
        *,
        epoch: int,
        step: int,
        loss: float = 0.0,
        accuracy: float = 0.0,
        lr: float = 0.0,
        grad_norm: float = 0.0,
        has_nan: bool = False,
        has_inf: bool = False,
        has_zero: bool = False,
        repair_action: str = "none",
        step_time_ms: float = 0.0,
        tokens_per_sec: float = 0.0,
        clip_ratio: float = 0.0,
        effective_batch_size: int = 0,
    ):
        gpu_memory = 0.0
        gpu_cached = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_cached = torch.cuda.memory_reserved() / 1024**3

        telemetry = self._get_gpu_telemetry()
        if self._last_guard_temp_c is not None:
            telemetry["gpu_temp_c"] = float(self._last_guard_temp_c)
        grad_buckets = self._get_default_block_grad_norms()

        self.csv_writer.writerow([
            datetime.now().isoformat(),
            epoch,
            step,
            self.global_step,
            f'{float(loss):.6f}',
            f'{float(accuracy):.6f}',
            f'{float(lr):.8e}',
            f'{float(grad_norm):.6f}',
            f'{1.0:.2f}',
            f'{gpu_memory:.4f}',
            f'{gpu_cached:.4f}',
            int(has_nan),
            int(has_inf),
            int(has_zero),
            repair_action,
            f"{float(telemetry['gpu_temp_c']):.2f}",
            f"{float(telemetry['gpu_power_w']):.2f}",
            f"{float(telemetry['gpu_util_pct']):.2f}",
            f"{float(telemetry['gpu_mem_used_mib']):.2f}",
            f"{float(grad_buckets['embeddings']):.6f}",
            f"{float(grad_buckets['attention']):.6f}",
            f"{float(grad_buckets['ffn']):.6f}",
            f"{float(grad_buckets['experts']):.6f}",
            f"{float(grad_buckets['router']):.6f}",
            f"{float(grad_buckets['norms']):.6f}",
            f"{float(grad_buckets['head']):.6f}",
            f"{float(grad_buckets['other']):.6f}",
            f"{float(step_time_ms):.2f}",
            f"{float(tokens_per_sec):.2f}",
            f"{float(clip_ratio):.6f}",
            int(effective_batch_size),
        ])
        self.csv_file.flush()

    @staticmethod
    def _is_main_thread() -> bool:
        import threading
        return threading.current_thread() is threading.main_thread()

    def _install_supervisor_sigusr1_handler(self):
        """Register SIGUSR1 to request an immediate checkpoint flush from the supervisor.

        The supervisor sends SIGUSR1 before SIGKILL on a thermal event. Because
        sentence-transformers owns the training loop, the handler only records
        the request; the checkpoint flush is performed opportunistically via
        :meth:`_on_sbert_step` (best-effort, since SBERT's rolling checkpoints
        are managed by sentence-transformers itself at ``checkpoint_save_steps``
        intervals).
        """
        try:
            import signal as _signal
            _signal.signal(_signal.SIGUSR1, self._on_supervisor_sigusr1)
            logger.info("Supervisor SIGUSR1 handler installed (FRANKESTEIN_SUPERVISOR_CHILD=1)")
        except (ValueError, OSError) as exc:
            logger.warning("Could not install SIGUSR1 supervisor handler: %s", exc)

    def _on_supervisor_sigusr1(self, signum, frame):
        self._checkpoint_request_pending = True

    def _handle_thermal_guard_for_batch(self, batch_idx: int) -> str:
        """Thermal guard hook (no-op under the supervisor model).

        The CPU-side :class:`GPUTempSupervisor` owns the training subprocess
        lifecycle and handles thermal events by checkpointing + killing this
        process. In-process thermal offload is therefore no longer performed
        here. SBERT rolling checkpoints are flushed by sentence-transformers at
        ``checkpoint_save_steps`` intervals, so the supervisor relies on those
        (plus a best-effort checkpoint save attempt when SIGUSR1 is received).

        Args:
            batch_idx: Current batch index.

        Returns:
            Always ``"none"``.
        """
        return "none"

    def _on_sbert_step(self, epoch_idx: int, batch_idx: int, repair_action: str):
        self.global_step += 1
        if (self.global_step % self.telemetry_log_interval) != 0:
            return
        self._log_step_to_csv(
            epoch=int(epoch_idx),
            step=int(batch_idx),
            loss=0.0,
            accuracy=0.0,
            lr=float(self.learning_rate),
            grad_norm=0.0,
            has_nan=False,
            has_inf=False,
            has_zero=False,
            repair_action=str(repair_action or "none"),
            step_time_ms=0.0,
            tokens_per_sec=0.0,
            clip_ratio=0.0,
            effective_batch_size=int(self.batch_size) * int(self.gradient_accumulation_steps),
        )

    def close(self):
        if self.csv_file:
            self.csv_file.close()
            logger.info("CSV log closed: %s", self.csv_log_path)
    
    def load_dataset(
        self,
        dataset_name: str = "erickfmm/agentlans__multilingual-sentences__paired_10_sts",
        dataset_type: str = "paired_similarity",
        columns: Optional[Dict[str, str]] = None,
        query_prefix: str = "",
        document_prefix: str = "",
        max_train_samples: Optional[int] = None,
        max_eval_samples: Optional[int] = 10000,
        resample_balanced: bool = True,
        standardize_scores: bool = False,
        target_std: float = 0.3
    ):
        """
        Load and prepare the STS dataset.
        
        Scores are kept in [-1, 1] range to preserve sign:
        - Negative scores remain negative (dissimilar sentences)
        - Zero remains zero (neutral similarity)
        - Positive scores remain positive (similar sentences)
        
        Args:
            dataset_name: HuggingFace dataset name
            dataset_type: Dataset format type (paired_similarity, triplets, qa)
            columns: Optional dataset column mapping
            query_prefix: Optional prefix prepended to query/question text
            document_prefix: Optional prefix prepended to answer/positive/negative text
            max_train_samples: Maximum training samples
            max_eval_samples: Maximum evaluation samples
            resample_balanced: Resample to get normal distribution centered at 0
            standardize_scores: Standardize score column instead of undersampling
            target_std: Target standard deviation for normal distribution (default: 0.3)
        """
        dataset_type = str(dataset_type).strip().lower()
        if dataset_type not in SUPPORTED_SBERT_DATASET_TYPES:
            raise ValueError(
                f"Unsupported dataset_type='{dataset_type}'. "
                f"Supported values: {', '.join(SUPPORTED_SBERT_DATASET_TYPES)}"
            )
        self.dataset_type = dataset_type
        self.query_prefix = str(query_prefix or "")
        self.document_prefix = str(document_prefix or "")
        self.dataset_columns = self._resolve_dataset_columns(dataset_type, columns or {})

        logger.info(f"Loading dataset: {dataset_name}")
        logger.info(
            "SBERT dataset configuration: type=%s, columns=%s, query_prefix=%r, document_prefix=%r",
            self.dataset_type,
            self.dataset_columns,
            self.query_prefix,
            self.document_prefix,
        )
        
        # Load dataset
        dataset = load_dataset(dataset_name, split="train")
        
        logger.info(f"Dataset loaded: {len(dataset)} total samples")
        self._validate_required_columns(dataset)
        
        # Resample to balance distribution (avoid imbalance as recommended)
        if (resample_balanced or standardize_scores) and self.dataset_type != "paired_similarity":
            logger.warning(
                "Score balancing options are only supported for paired_similarity datasets; disabling for dataset_type='%s'",
                self.dataset_type,
            )
        elif standardize_scores:
            similarity_column = self.dataset_columns["similarity"]
            dataset = self._standardize_scores(
                dataset,
                score_column=similarity_column,
                target_std=target_std,
            )
            logger.info(f"After score standardization: {len(dataset)} samples")
        elif resample_balanced:
            similarity_column = self.dataset_columns["similarity"]
            dataset = self._resample_balanced(
                dataset,
                score_column=similarity_column,
                target_std=target_std,
            )
            logger.info(f"After resampling: {len(dataset)} samples")
        
        # Shuffle and split
        dataset = dataset.shuffle(seed=42)
        
        if max_train_samples:
            split_point = max_train_samples
        else:
            split_point = int(len(dataset) * 0.95)  # 95% train, 5% eval
        
        train_dataset = dataset.select(range(split_point))
        eval_dataset = dataset.select(range(split_point, len(dataset)))
        
        if max_eval_samples and len(eval_dataset) > max_eval_samples:
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        
        logger.info(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
        
        # Convert to InputExample format according to dataset type
        self.train_examples = self._dataset_to_examples(train_dataset, split_name="train")
        self.eval_examples = self._dataset_to_examples(eval_dataset, split_name="eval")
        
        # Create evaluator for paired similarity only
        self.evaluator = self._create_evaluator() if self.dataset_type == "paired_similarity" else None
        
        return self.train_examples, self.eval_examples
    
    def _resolve_dataset_columns(self, dataset_type: str, columns: Dict[str, Any]) -> Dict[str, str]:
        defaults = dict(DEFAULT_SBERT_COLUMNS[dataset_type])
        resolved = dict(defaults)
        for key in defaults:
            value = columns.get(key)
            if value is None:
                continue
            text_value = str(value).strip()
            if not text_value:
                raise ValueError(
                    f"Column mapping for key '{key}' cannot be empty for dataset_type='{dataset_type}'"
                )
            resolved[key] = text_value
        return resolved

    def _validate_required_columns(self, dataset):
        available = set(dataset.column_names)
        required = set(self.dataset_columns.values())
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                f"Missing required dataset columns for dataset_type='{self.dataset_type}': {missing}. "
                f"Available columns: {sorted(available)}"
            )

    def _with_prefix(self, text: Any, prefix: str) -> str:
        base = str(text)
        return f"{prefix}{base}" if prefix else base

    def _resample_balanced(
        self,
        dataset,
        score_column: str,
        bins: int = 20,
        target_std: float = 0.3,
    ):
        """
        Resample dataset to create a normal distribution centered at 0.
        
        This undersamples to avoid imbalanced data as recommended by the dataset authors.
        The final distribution will approximate N(0, target_std) in the [-1, 1] range.
        
        Args:
            dataset: The dataset to resample
            bins: Number of bins for score distribution (more bins = finer control)
            target_std: Target standard deviation for the normal distribution
        """
        scores = np.array(dataset[score_column], dtype=np.float32)

        logger.info(f"Original distribution: mean={scores.mean():.4f}, std={scores.std():.4f}")
        logger.info(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")

        # Create bins
        hist, bin_edges = np.histogram(scores, bins=bins)
        # Target: Normal distribution centered at 0 with specified std
        # Calculate expected count per bin based on normal distribution using PyTorch.
        bin_edges_t = torch.tensor(bin_edges, dtype=torch.float64)
        bin_probs = (
            torch.special.ndtr(bin_edges_t[1:] / target_std)
            - torch.special.ndtr(bin_edges_t[:-1] / target_std)
        ).numpy()

        bin_probs_sum = bin_probs.sum()
        if bin_probs_sum > 0:
            bin_probs = bin_probs / bin_probs_sum  # Normalize
        else:
            bin_probs = np.ones(bins) / bins

        # Calculate target samples per bin
        # Use undersampling: limit by the most restrictive bin
        samples_if_all_filled = hist / bin_probs
        samples_if_all_filled[bin_probs < 1e-6] = np.inf  # Ignore near-zero probability bins

        # Total samples is limited by the bin with lowest samples/probability ratio
        total_target = int(np.min(samples_if_all_filled[np.isfinite(samples_if_all_filled)]))

        # Now calculate per-bin targets
        target_per_bin = (bin_probs * total_target).astype(int)

        logger.info(f"Undersampling from {len(scores)} to ~{total_target} samples")

        indices_to_keep = []
        for i in range(bins):
            # Get indices in this bin
            if i < bins - 1:
                in_bin = np.where((scores >= bin_edges[i]) & (scores < bin_edges[i + 1]))[0]
            else:
                in_bin = np.where((scores >= bin_edges[i]) & (scores <= bin_edges[i + 1]))[0]

            target = target_per_bin[i]

            # Sample from bin (undersample if needed)
            if len(in_bin) > target and target > 0:
                sampled = np.random.choice(in_bin, target, replace=False)
            elif target > 0:
                sampled = in_bin
            else:
                continue

            indices_to_keep.extend(sampled.tolist())

        resampled = dataset.select(indices_to_keep)
        resampled_scores = np.array(
            [float(resampled[i][score_column]) for i in range(len(resampled))],
            dtype=np.float32,
        )

        logger.info(f"Resampled distribution: mean={resampled_scores.mean():.4f}, std={resampled_scores.std():.4f}")
        logger.info(f"Final sample count: {len(resampled)}")

        return resampled

    def _standardize_scores(
        self,
        dataset,
        score_column: str,
        target_std: float = 0.3,
    ):
        """
        Standardize score magnitudes while preserving original sign and keeping all rows.
        """
        scores = np.array(dataset[score_column], dtype=np.float32)
        source_mean = float(scores.mean())
        source_std = float(scores.std())
        logger.info(
            "Standardizing scores from mean=%.4f, std=%.4f to target std=%.4f with sign preservation",
            source_mean,
            source_std,
            float(target_std),
        )
        if source_std < 1e-8:
            logger.warning(
                "Score standard deviation is near zero (%.8f); skipping standardization.",
                source_std,
            )
            return dataset

        eps = 1e-12

        def _map_batch(batch):
            values = np.array(batch[score_column], dtype=np.float32)
            standardized = ((values - source_mean) / (source_std + eps)) * float(target_std)
            # Preserve original sign after standardization.
            signed = np.where(values == 0.0, 0.0, np.sign(values) * np.abs(standardized))
            clipped = np.clip(signed, -1.0, 1.0)
            return {score_column: clipped.tolist()}

        standardized = dataset.map(
            _map_batch,
            batched=True,
            desc=f"Standardizing '{score_column}'",
        )
        standardized_scores = np.array(standardized[score_column], dtype=np.float32)
        logger.info(
            "Standardized distribution: mean=%.4f, std=%.4f, range=[%.4f, %.4f]",
            float(standardized_scores.mean()),
            float(standardized_scores.std()),
            float(standardized_scores.min()),
            float(standardized_scores.max()),
        )
        return standardized
    
    def _dataset_to_examples(self, dataset, split_name: str):
        """
        Convert dataset split to InputExample list according to dataset_type.
        """
        if self.dataset_type == "paired_similarity":
            return self._dataset_to_examples_paired_similarity(dataset)
        if self.dataset_type == "triplets":
            return self._dataset_to_examples_triplets(dataset, split_name=split_name)
        if self.dataset_type == "qa":
            return self._dataset_to_examples_qa(dataset)
        raise ValueError(f"Unsupported dataset_type='{self.dataset_type}'")

    def _dataset_to_examples_paired_similarity(self, dataset):
        examples = []
        similarity_column = self.dataset_columns["similarity"]
        sentence1_column = self.dataset_columns["sentence1"]
        sentence2_column = self.dataset_columns["sentence2"]
        
        for item in dataset:
            # Keep score in [-1, 1] range to preserve sign
            score = float(item[similarity_column])
            
            example = InputExample(
                texts=[
                    self._with_prefix(item[sentence1_column], self.query_prefix),
                    self._with_prefix(item[sentence2_column], self.document_prefix),
                ],
                label=score
            )
            examples.append(example)
        
        return examples

    def _dataset_to_examples_triplets(self, dataset, split_name: str):
        examples: List[InputExample] = []
        skipped_rows = 0
        query_column = self.dataset_columns["query"]
        positive_column = self.dataset_columns["positive"]
        negatives_column = self.dataset_columns["negatives"]

        for item in dataset:
            negatives = item[negatives_column]
            if not isinstance(negatives, list):
                skipped_rows += 1
                continue

            query_text = self._with_prefix(item[query_column], self.query_prefix)
            positive_text = self._with_prefix(item[positive_column], self.document_prefix)

            valid_negative_count = 0
            for negative in negatives:
                if not isinstance(negative, str) or not negative.strip():
                    continue
                negative_text = self._with_prefix(negative, self.document_prefix)
                examples.append(InputExample(texts=[query_text, positive_text, negative_text]))
                valid_negative_count += 1
            if valid_negative_count == 0:
                skipped_rows += 1

        if skipped_rows > 0:
            logger.warning(
                "Skipped %d rows in %s split for triplets due to malformed or empty negatives list",
                skipped_rows,
                split_name,
            )
        return examples

    def _dataset_to_examples_qa(self, dataset):
        examples: List[InputExample] = []
        question_column = self.dataset_columns["question"]
        answer_column = self.dataset_columns["answer"]
        for item in dataset:
            examples.append(
                InputExample(
                    texts=[
                        self._with_prefix(item[question_column], self.query_prefix),
                        self._with_prefix(item[answer_column], self.document_prefix),
                    ]
                )
            )
        return examples
    
    def _create_evaluator(self):
        """Create evaluator for validation"""
        # Extract sentences and scores
        sentences1 = [ex.texts[0] for ex in self.eval_examples]
        sentences2 = [ex.texts[1] for ex in self.eval_examples]
        scores = [ex.label for ex in self.eval_examples]
        
        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            sentences1,
            sentences2,
            scores,
            name='sts_eval'
        )
        
        return evaluator

    def _read_checkpoint_global_step(self, checkpoint_dir: str) -> Optional[int]:
        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        if not os.path.exists(trainer_state_path):
            return None
        try:
            with open(trainer_state_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
            value = state.get("global_step")
            if isinstance(value, (int, float)):
                return int(value)
        except Exception as exc:
            logger.warning("Failed reading global_step from %s: %s", trainer_state_path, exc)
        return None

    def _find_latest_rolling_checkpoint(self) -> Optional[Dict[str, Any]]:
        checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        if not os.path.isdir(checkpoint_root):
            return None

        latest_step = -1
        latest_path = None
        for entry in os.listdir(checkpoint_root):
            full_path = os.path.join(checkpoint_root, entry)
            if not os.path.isdir(full_path):
                continue
            match = re.fullmatch(r"checkpoint-(\d+)", entry)
            if match is None:
                continue
            step_from_name = int(match.group(1))
            step_from_state = self._read_checkpoint_global_step(full_path)
            step = step_from_state if step_from_state is not None else step_from_name
            if step > latest_step:
                latest_step = step
                latest_path = full_path

        if latest_path is None:
            return None
        return {
            "path": latest_path,
            "global_step": latest_step,
        }
    
    def train(self):
        """Run training loop"""
        logger.info("Starting SBERT training...")
        logger.info(f"Device: {self.model.device}")
        logger.info(f"Training samples: {len(self.train_examples)}")
        logger.info(f"Batch size: {self.batch_size}, Epochs: {self.epochs}")
        logger.info(
            "Gradient accumulation: %d, effective batch size: %d",
            self.gradient_accumulation_steps,
            int(self.batch_size) * int(self.gradient_accumulation_steps),
        )
        logger.info("Gradient clipping max_grad_norm: %.4f", self.max_grad_norm)
        logger.info(f"Dataset type: {self.dataset_type}")
        
        # Create data loader
        train_dataloader = DataLoader(
            self.train_examples,
            # Keep order deterministic for resume line skipping. We already shuffled the dataset split once.
            shuffle=False,
            batch_size=self.batch_size,
            # sentence-transformers fit() in this environment expects list[InputExample].
            collate_fn=lambda batch: batch,
        )

        resume_checkpoint_path = None
        resume_skip_batches = 0
        if self.resume_from_checkpoint:
            latest = self._find_latest_rolling_checkpoint()
            if latest is None:
                logger.warning(
                    "resume_from_checkpoint is enabled but no rolling checkpoint was found under %s",
                    os.path.join(self.output_dir, "checkpoints"),
                )
            else:
                resume_checkpoint_path = str(latest["path"])
                resume_global_step = int(latest["global_step"])
                steps_per_epoch = max(len(train_dataloader), 1)
                resume_skip_batches = resume_global_step % steps_per_epoch
                approx_skipped_lines = min(
                    len(self.train_examples),
                    resume_skip_batches * self.batch_size,
                )
                logger.info(
                    "Resuming from rolling checkpoint %s (global_step=%d); skipping %d batches (~%d lines) already processed in current epoch",
                    resume_checkpoint_path,
                    resume_global_step,
                    resume_skip_batches,
                    approx_skipped_lines,
                )

        train_dataloader = _ThermalGuardedDataLoader(
            dataloader=train_dataloader,
            guard=self.gpu_temp_guard,
            context_prefix="sbert.fit",
            skip_batches_once=resume_skip_batches,
            on_thermal_check=self._handle_thermal_guard_for_batch,
            on_step=self._on_sbert_step,
        )
        
        # Define loss function according to dataset type.
        if self.dataset_type == "paired_similarity":
            train_loss = losses.CosineSimilarityLoss(self.model)
        else:
            train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        # Calculate total steps
        num_train_steps = len(train_dataloader) * self.epochs
        effective_remaining_steps = max(num_train_steps - resume_skip_batches, 0)
        logger.info(
            "Total training steps: %d (effective remaining after resume skip: %d)",
            num_train_steps,
            effective_remaining_steps,
        )
        
        # Build optimizer configuration
        custom_optimizer_cls = None
        custom_optimizer_kwargs = {}
        if self.optimizer_class and self.optimizer_class in OPTIMIZER_REGISTRY:
            custom_optimizer_cls = OPTIMIZER_REGISTRY[self.optimizer_class]
            custom_optimizer_kwargs = {"lr": self.learning_rate}
            # Extract optimizer-specific params from optimizer_parameters
            params = self.optimizer_parameters or {}
            # Collect global (non-prefixed) optimizer parameters
            skip_prefixes = ("lr_", "wd_", "betas_", "eps_")
            for key, value in params.items():
                # Skip per-group parameters (lr_*, wd_*, betas_*, eps_*)
                if any(key.startswith(p) for p in skip_prefixes):
                    continue
                custom_optimizer_kwargs[key] = value
            logger.info(
                "Using custom optimizer: %s with params: %s",
                self.optimizer_class,
                custom_optimizer_kwargs,
            )

        # Train
        fit_kwargs = dict(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            warmup_steps=self.warmup_steps,
            evaluation_steps=self.evaluation_steps,
            output_path=self.output_dir,
            save_best_model=True,
            optimizer_params={'lr': self.learning_rate},
            use_amp=self.use_amp,
            show_progress_bar=True
        )
        # Override optimizer if custom optimizer is specified
        if custom_optimizer_cls is not None:
            fit_kwargs["optimizer_params"] = custom_optimizer_kwargs
            # sentence-transformers v3+ supports optimizer_cls parameter
            fit_kwargs["optimizer_class"] = custom_optimizer_cls
        if self.checkpoint_save_steps > 0 or self.resume_from_checkpoint:
            fit_kwargs["checkpoint_path"] = os.path.join(self.output_dir, "checkpoints")
        if self.checkpoint_save_steps > 0:
            fit_kwargs["checkpoint_save_steps"] = self.checkpoint_save_steps
            fit_kwargs["checkpoint_save_total_limit"] = self.checkpoint_save_total_limit
        if self.resume_from_checkpoint:
            fit_kwargs["resume_from_checkpoint"] = (
                resume_checkpoint_path if resume_checkpoint_path else True
            )
        if self.evaluator is not None:
            fit_kwargs["evaluator"] = self.evaluator
        try:
            fit_signature = inspect.signature(self.model.fit)
            fit_params = fit_signature.parameters
            accepts_var_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in fit_params.values()
            )
        except (TypeError, ValueError):
            fit_params = {}
            accepts_var_kwargs = False

        if self.gradient_accumulation_steps > 1:
            if "gradient_accumulation_steps" in fit_params or accepts_var_kwargs:
                fit_kwargs["gradient_accumulation_steps"] = int(self.gradient_accumulation_steps)
            else:
                logger.warning(
                    "sentence-transformers fit() in this environment does not expose "
                    "gradient_accumulation_steps; continuing without accumulation support."
                )
        if self.max_grad_norm > 0.0:
            if "max_grad_norm" in fit_params or accepts_var_kwargs:
                fit_kwargs["max_grad_norm"] = float(self.max_grad_norm)
            else:
                logger.warning(
                    "sentence-transformers fit() in this environment does not expose "
                    "max_grad_norm; continuing without explicit gradient clipping."
                )
        try:
            self.model.fit(**fit_kwargs)
        except RuntimeError as exc:
            # In-process thermal CPU fallback is no longer used; the
            # supervisor owns the subprocess lifecycle and handles GPU
            # runtime errors by killing this child. Re-raise so the
            # supervisor can react (and re-launch on CPU if
            # switch_on_thermal is enabled).
            raise
        
        logger.info(f"Training completed! Model saved to {self.output_dir}")
        
        final_score = None
        if self.evaluator is not None:
            # Final evaluation
            logger.info("Running final evaluation...")
            final_score = self.evaluator(self.model)
            logger.info("Final Spearman correlation: %s", _format_score_for_log(final_score))
        else:
            logger.info("Skipping final evaluator for dataset type '%s'", self.dataset_type)
        
        self.training_metadata['end_time'] = datetime.now().isoformat()
        self.training_metadata['final_score'] = final_score
        self.training_metadata['final_score_value'] = _extract_primary_score(final_score)
        
        return final_score


class _ThermalGuardedDataLoader:
    """Dataloader wrapper that blocks when the GPU is too hot."""

    def __init__(
        self,
        dataloader: DataLoader,
        guard: GPUTemperatureGuard,
        context_prefix: str,
        skip_batches_once: int = 0,
        on_thermal_check: Optional[Callable[[int], str]] = None,
        on_step: Optional[Callable[[int, int, str], None]] = None,
        on_critical_temperature: Optional[Callable[[int, float], str]] = None,
    ):
        self._dataloader = dataloader
        self._guard = guard
        self._context_prefix = context_prefix
        self._skip_batches_once = max(int(skip_batches_once), 0)
        self._skip_batches_applied = False
        self._on_thermal_check = on_thermal_check
        self._on_step = on_step
        self._on_critical_temperature = on_critical_temperature
        self._epoch_counter = 0

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        batch_iter = iter(self._dataloader)
        epoch_idx = self._epoch_counter
        self._epoch_counter += 1
        start_batch_idx = 0
        progress_started = time.perf_counter()

        if (not self._skip_batches_applied) and self._skip_batches_once > 0:
            skipped = 0
            while skipped < self._skip_batches_once:
                try:
                    next(batch_iter)
                    skipped += 1
                except StopIteration:
                    break
            self._skip_batches_applied = True
            start_batch_idx = skipped
            logger.info(
                "Skipped %d already-processed batches before resuming SBERT dataloader iteration",
                skipped,
            )

        for batch_idx, batch in enumerate(batch_iter, start=start_batch_idx):
            repair_action = "none"
            if self._on_thermal_check is not None:
                repair_action = self._on_thermal_check(batch_idx)
                if repair_action != "none":
                    logger.warning(
                        "[ThermalGuard][SBERT] thermal action: %s",
                        repair_action,
                    )
            elif self._guard.is_active:
                context = f"{self._context_prefix} batch={batch_idx + 1}"
                temp_c = float(self._guard.read_temperature_c())
                critical_threshold = self._guard.critical_threshold_c
                if critical_threshold is not None and temp_c >= float(critical_threshold):
                    if self._on_critical_temperature is not None:
                        repair_action = self._on_critical_temperature(batch_idx, temp_c)
                        logger.warning(
                            "[ThermalGuard][SBERT] critical offload event: %s (temp=%.1fC)",
                            repair_action,
                            temp_c,
                        )
                    else:
                        result = self._guard.wait_until_safe(context=context)
                        if result.paused:
                            logger.warning(
                                "[ThermalGuard][SBERT] pause event: %s (temp=%.1fC)",
                                result.repair_action,
                                result.temp_c,
                            )
                elif temp_c > float(self._guard.pause_threshold_c):
                    result = self._guard.wait_until_safe(context=context)
                    if result.paused:
                        logger.warning(
                            "[ThermalGuard][SBERT] pause event: %s (temp=%.1fC)",
                            result.repair_action,
                            result.temp_c,
                        )
            if batch_idx > 0 and (batch_idx % 100 == 0):
                progress_snapshot = tqdm_progress.format_meter(
                    batch_idx + 1,
                    len(self._dataloader),
                    time.perf_counter() - progress_started,
                )
                logger.info("Progress %s", progress_snapshot)
            if self._on_step is not None:
                self._on_step(epoch_idx, batch_idx, repair_action)
            yield batch


def main(argv=None):
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SBERT on TormentedBert")
    parser.add_argument("--base-model", type=str, default=None, help="HF model id/path for base-model SBERT finetuning")
    parser.add_argument("--pretrained", type=str, help="Path to pretrained checkpoint")
    parser.add_argument("--output_dir", type=str, default="./output/sbert_tormented_v2")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="erickfmm/agentlans__multilingual-sentences__paired_10_sts",
        help="Hugging Face dataset name for SBERT training",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="paired_similarity",
        choices=SUPPORTED_SBERT_DATASET_TYPES,
        help="Dataset layout used for SBERT training",
    )
    parser.add_argument("--col_sentence1", type=str, default=None, help="Column name for sentence1 in paired_similarity datasets")
    parser.add_argument("--col_sentence2", type=str, default=None, help="Column name for sentence2 in paired_similarity datasets")
    parser.add_argument("--col_similarity", type=str, default=None, help="Column name for similarity score in paired_similarity datasets")
    parser.add_argument("--col_query", type=str, default=None, help="Column name for query in triplets datasets")
    parser.add_argument("--col_positive", type=str, default=None, help="Column name for positive text in triplets datasets")
    parser.add_argument("--col_negatives", type=str, default=None, help="Column name for negatives list in triplets datasets")
    parser.add_argument("--col_question", type=str, default=None, help="Column name for question in qa datasets")
    parser.add_argument("--col_answer", type=str, default=None, help="Column name for answer in qa datasets")
    parser.add_argument("--query_prefix", type=str, default="", help="Optional prefix for query/question text")
    parser.add_argument("--document_prefix", type=str, default="", help="Optional prefix for document/answer/positive/negative text")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--evaluation_steps", type=int, default=5000)
    parser.add_argument(
        "--checkpoint_save_steps",
        type=int,
        default=1000,
        help="Rolling checkpoint save interval in steps (<=0 disables rolling checkpoint saves)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume SBERT training from latest checkpoint in output_dir/checkpoints",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=10000)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--pooling_mode", type=str, default="mean", choices=["mean", "cls", "max"])
    parser.add_argument("--trust_remote_code", action="store_true", help="Allow remote code for HF models/tokenizers")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--no_resample", action="store_true", 
                       help="Don't resample dataset (default: undersample to normal distribution centered at 0)")
    parser.add_argument(
        "--standardize_scores",
        action="store_true",
        help="Standardize similarity scores (mean=0, target std via --resample_std) without undersampling",
    )
    parser.add_argument("--resample_std", type=float, default=0.3,
                       help="Target std for resampled normal distribution (default: 0.3)")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=SUPPORTED_DEVICE_CHOICES,
        help="Device to run SBERT training on"
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
    thermal_switch_group = parser.add_mutually_exclusive_group()
    thermal_switch_group.add_argument(
        "--switch-on-thermal",
        dest="switch_on_thermal",
        action="store_true",
        help="Enable GPU->CPU thermal switching and resume to GPU when temperature recovers.",
    )
    thermal_switch_group.add_argument(
        "--no-switch-on-thermal",
        dest="switch_on_thermal",
        action="store_false",
        help="Disable thermal GPU->CPU switching and keep pause-only behavior.",
    )
    parser.set_defaults(switch_on_thermal=None)
    parser.add_argument("--gpu-temp-pause-threshold-c", type=float, default=90.0)
    parser.add_argument("--gpu-temp-resume-threshold-c", type=float, default=80.0)
    parser.add_argument("--gpu-temp-critical-threshold-c", type=float, default=None)
    parser.add_argument("--gpu-temp-poll-interval-seconds", type=float, default=30.0)
    parser.add_argument("--gpu-temp-checkpoint-grace-seconds", type=float, default=30.0)
    parser.add_argument("--nvml-device-index", type=int, default=0)
    parser.add_argument("--csv-log-path", type=str, default="training_metrics.csv")
    csv_rotate_group = parser.add_mutually_exclusive_group()
    csv_rotate_group.add_argument(
        "--csv-rotate-on-schema-change",
        dest="csv_rotate_on_schema_change",
        action="store_true",
    )
    csv_rotate_group.add_argument(
        "--no-csv-rotate-on-schema-change",
        dest="csv_rotate_on_schema_change",
        action="store_false",
    )
    parser.set_defaults(csv_rotate_on_schema_change=True)
    parser.add_argument("--telemetry-log-interval", type=int, default=1)
    parser.add_argument("--gpu-metrics-backend", type=str, default="nvml", choices=["nvml", "none"])
    block_grad_group = parser.add_mutually_exclusive_group()
    block_grad_group.add_argument(
        "--enable-block-grad-norms",
        dest="enable_block_grad_norms",
        action="store_true",
    )
    block_grad_group.add_argument(
        "--no-enable-block-grad-norms",
        dest="enable_block_grad_norms",
        action="store_false",
    )
    parser.set_defaults(enable_block_grad_norms=True)
    parser.add_argument(
        "--optimizer_class",
        type=str,
        default=None,
        help="Optimizer class name from the registry (e.g., turbo_muon, adamw, lion). "
             "When not set, sentence-transformers default optimizer is used.",
    )
    parser.add_argument(
        "--optimizer_param",
        type=str,
        action="append",
        default=None,
        help="Optimizer parameter in key=value format (repeatable). "
             "Example: --optimizer_param momentum=0.95 --optimizer_param ns_steps=4",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name for telemetry tracking.",
    )
    
    args = parser.parse_args(argv)
    resolved_device = resolve_torch_device(args.device)
    logger.info(f"SBERT train device requested='{args.device}', resolved='{resolved_device}'")
    gpu_temp_guard_enabled = True if args.gpu_temp_guard is None else bool(args.gpu_temp_guard)
    switch_on_thermal_enabled = False if args.switch_on_thermal is None else bool(args.switch_on_thermal)
    if not resolved_device.startswith("cuda"):
        gpu_temp_guard_enabled = False
        switch_on_thermal_enabled = False
    if args.gpu_temp_pause_threshold_c <= 0:
        raise ValueError("gpu_temp_pause_threshold_c must be > 0")
    if args.gpu_temp_resume_threshold_c <= 0:
        raise ValueError("gpu_temp_resume_threshold_c must be > 0")
    if args.gpu_temp_resume_threshold_c >= args.gpu_temp_pause_threshold_c:
        raise ValueError("gpu_temp_resume_threshold_c must be < gpu_temp_pause_threshold_c")
    if args.gpu_temp_poll_interval_seconds <= 0:
        raise ValueError("gpu_temp_poll_interval_seconds must be > 0")
    if (
        args.gpu_temp_critical_threshold_c is not None
        and args.gpu_temp_critical_threshold_c <= 0
    ):
        raise ValueError("gpu_temp_critical_threshold_c must be > 0 when provided")
    if args.checkpoint_save_steps < 0:
        raise ValueError("checkpoint_save_steps must be >= 0")
    if args.telemetry_log_interval <= 0:
        raise ValueError("telemetry_log_interval must be > 0")
    
    # Setup
    logger.info("=" * 80)
    logger.info("SBERT Training on TORMENTED-BERT-Frankenstein v2")
    logger.info("=" * 80)

    if args.base_model and args.pretrained:
        raise ValueError("--base-model and --pretrained are mutually exclusive")
    
    # Create model
    if args.base_model:
        model_wrapper = TormentedBertSentenceTransformer(
            base_model_name_or_path=args.base_model,
            max_seq_length=args.max_seq_length,
            pooling_mode=args.pooling_mode,
            trust_remote_code=args.trust_remote_code,
            device=resolved_device,
        )
    elif args.pretrained:
        model_wrapper = TormentedBertSentenceTransformer(
            pretrained_path=args.pretrained,
            max_seq_length=args.max_seq_length,
            pooling_mode=args.pooling_mode,
            device=resolved_device,
        )
    else:
        config = UltraConfig(
            vocab_size=50000,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_loops=1,
            ode_steps=1,
            use_bitnet=True
        )
        model_wrapper = TormentedBertSentenceTransformer(
            model_config=config,
            max_seq_length=args.max_seq_length,
            pooling_mode=args.pooling_mode,
            device=resolved_device,
        )
    
    model = model_wrapper.get_model()
    logger.info(f"Model initialized with pooling mode: {args.pooling_mode}")
    
    # Parse optimizer parameters from key=value format
    optimizer_parameters = None
    if args.optimizer_param:
        import json as _json
        optimizer_parameters = {}
        for param_str in args.optimizer_param:
            if "=" not in param_str:
                logger.warning("Skipping malformed optimizer parameter: %s (expected key=value)", param_str)
                continue
            key, value_str = param_str.split("=", 1)
            key = key.strip()
            value_str = value_str.strip()
            # Try to parse as JSON (handles numbers, booleans, lists)
            try:
                optimizer_parameters[key] = _json.loads(value_str)
            except (_json.JSONDecodeError, ValueError):
                # Fallback: keep as string
                optimizer_parameters[key] = value_str

    # Create trainer
    trainer = SBERTTrainer(
        model=model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        evaluation_steps=args.evaluation_steps,
        checkpoint_save_steps=args.checkpoint_save_steps,
        resume_from_checkpoint=bool(args.resume_from_checkpoint),
        learning_rate=args.learning_rate,
        use_amp=not args.no_amp,
        device=resolved_device,
        gpu_temp_guard_enabled=gpu_temp_guard_enabled,
        switch_on_thermal=switch_on_thermal_enabled,
        gpu_temp_pause_threshold_c=args.gpu_temp_pause_threshold_c,
        gpu_temp_resume_threshold_c=args.gpu_temp_resume_threshold_c,
        gpu_temp_critical_threshold_c=args.gpu_temp_critical_threshold_c,
        gpu_temp_poll_interval_seconds=args.gpu_temp_poll_interval_seconds,
        nvml_device_index=args.nvml_device_index,
        csv_log_path=args.csv_log_path,
        csv_rotate_on_schema_change=bool(args.csv_rotate_on_schema_change),
        telemetry_log_interval=int(args.telemetry_log_interval),
        gpu_metrics_backend=args.gpu_metrics_backend,
        enable_block_grad_norms=bool(args.enable_block_grad_norms),
        optimizer_class=args.optimizer_class,
        optimizer_parameters=optimizer_parameters,
        wandb_project=args.wandb_project,
    )
    
    # Load dataset
    column_overrides = {
        "sentence1": args.col_sentence1,
        "sentence2": args.col_sentence2,
        "similarity": args.col_similarity,
        "query": args.col_query,
        "positive": args.col_positive,
        "negatives": args.col_negatives,
        "question": args.col_question,
        "answer": args.col_answer,
    }
    column_overrides = {
        key: value for key, value in column_overrides.items()
        if isinstance(value, str) and value.strip()
    }
    trainer.load_dataset(
        dataset_name=args.dataset_name,
        dataset_type=args.dataset_type,
        columns=column_overrides,
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        resample_balanced=not args.no_resample,
        standardize_scores=bool(args.standardize_scores),
        target_std=args.resample_std
    )
    
    # Train
    try:
        final_score = trainer.train()
    finally:
        trainer.close()
    
    logger.info("=" * 80)
    if final_score is not None:
        logger.info("Training complete! Final score: %s", _format_score_for_log(final_score))
    else:
        logger.info("Training complete! Final score: n/a (no evaluator configured for this dataset type)")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
