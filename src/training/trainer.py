"""Training loop with stability controls for TORMENTED-BERT-Frankenstein.

Provides :class:`TitanTrainer`, a production-grade training loop with
gradient clipping, NaN detection and retry, rolling/best checkpointing,
GPU thermal guard integration, CSV telemetry logging, GaLore low-rank
gradient projection, and automatic mixed precision support.
"""

import os
import csv
import torch
import logging
import math
import heapq
import re
import signal
import subprocess
import glob
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional, Dict
from tqdm import tqdm
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

try:
    from ..utils.storage_manager import StorageManager
    from ..utils.gpu_temp_guard import (
        CHECKPOINT_DONE_SENTINEL,
        GPUTelemetryError,
        GPUTemperatureGuard,
        RESUME_AUTO,
    )
    from ..model.optimizer.factory import build_optimizer
except ImportError:
    from utils.storage_manager import StorageManager
    from utils.gpu_temp_guard import (
        CHECKPOINT_DONE_SENTINEL,
        GPUTelemetryError,
        GPUTemperatureGuard,
        RESUME_AUTO,
    )
    from model.optimizer.factory import build_optimizer


# ==================== TRAINING CONFIGURATION ====================
@dataclass
class TrainingConfig:
    """Configuration for training behavior, checkpointing, and stability controls.

    Attributes:
        csv_log_path: Path for CSV training metrics log.
        csv_rotate_on_schema_change: If ``True``, rotate the CSV file when
            the column schema changes instead of raising an error.
        gpu_metrics_backend: GPU telemetry backend (``"nvml"`` or ``"none"``).
        nvml_device_index: NVML device index for multi-GPU systems.
        enable_block_grad_norms: Whether to collect per-block gradient norms.
        telemetry_log_interval: Steps between heavy telemetry collection.
        gpu_temp_guard_enabled: Whether GPU thermal guard is enabled.
        switch_on_thermal: Whether GPU→CPU thermal switching is enabled.
        gpu_temp_pause_threshold_c: Temperature in Celsius above which
            training is paused.
        gpu_temp_resume_threshold_c: Temperature in Celsius below which
            training resumes.
        gpu_temp_critical_threshold_c: Optional critical temperature for
            emergency GPU→CPU offload.
        gpu_temp_poll_interval_seconds: Seconds between temperature polls
            while paused.
        checkpoint_every_n_steps: Save a rolling checkpoint every N steps.
        max_rolling_checkpoints: Maximum number of rolling checkpoints to keep.
        num_best_checkpoints: Number of best-loss checkpoints to retain.
        nan_check_interval: Check for NaN loss every N steps.
        log_gradient_stats: Whether to log gradient statistics.
        gradient_log_interval: Steps between gradient stat logging.
        gradient_accumulation_steps: Number of steps to accumulate gradients
            before an optimizer update.
        optimizer_class: Optimizer class name (e.g. ``"adamw"``).
        optimizer_parameters: Dictionary of optimizer-specific parameters.
        scheduler_total_steps: Total steps for the learning rate scheduler.
        scheduler_warmup_ratio: Fraction of total steps used for warmup.
        scheduler_type: Scheduler type (``"cosine"``, ``"constant"``, or
            ``"linear_warmup_then_constant"``).
        max_nan_retries: Maximum retries per batch when NaN is detected.
        grad_clip_max_norm: Maximum gradient norm for clipping.
        inf_post_clip_threshold: Gradient norm above which the step is
            considered to have exploding gradients.
        min_grad_norm_for_signal: Minimum gradient norm below which
            gradients are considered zero.
        inf_epoch_patience: Consecutive inf-epochs before early stop.
        zero_grad_plateau_patience: Consecutive zero-gradient plateau
            epochs before early stop.
        use_galore: Whether to enable GaLore low-rank gradient projection.
        galore_rank: Rank for GaLore SVD projection.
        galore_update_interval: Steps between GaLore projection updates.
        galore_scale: Scaling factor for GaLore projected gradients.
        galore_max_dim: Maximum dimension for GaLore projection.
        use_amp: Whether to use automatic mixed precision (FP16/BF16).
    """
    # CSV Logging
    csv_log_path: str = "training_metrics.csv"
    csv_rotate_on_schema_change: bool = True
    gpu_metrics_backend: str = "nvml"
    nvml_device_index: int = 0
    enable_block_grad_norms: bool = True
    telemetry_log_interval: int = 1
    gpu_temp_guard_enabled: bool = True
    switch_on_thermal: bool = False
    gpu_temp_pause_threshold_c: float = 90.0
    gpu_temp_resume_threshold_c: float = 80.0
    gpu_temp_critical_threshold_c: Optional[float] = None
    gpu_temp_poll_interval_seconds: float = 30.0
    gpu_temp_checkpoint_grace_seconds: float = 30.0
    resume_from_checkpoint: Optional[str] = None
    
    # Rolling Checkpoints
    checkpoint_every_n_steps: int = 500  # Save every N steps
    max_rolling_checkpoints: int = 3     # Keep only last N rolling checkpoints
    
    # Best Model Checkpoints
    num_best_checkpoints: int = 2        # Keep top K best models
    
    # NaN Detection
    nan_check_interval: int = 10         # Check for NaN every N steps
    
    # Gradient monitoring
    log_gradient_stats: bool = True
    gradient_log_interval: int = 100
    gradient_accumulation_steps: int = 4

    # Optimizer chooser (hard-migrated format from YAML)
    optimizer_class: str = "adamw"
    optimizer_parameters: Dict[str, Any] = field(default_factory=dict)

    # Scheduler configuration
    scheduler_total_steps: int = 10000
    scheduler_warmup_ratio: float = 0.1
    scheduler_type: str = "cosine"

    # Post-backward stability
    max_nan_retries: int = 3
    grad_clip_max_norm: float = 5.0  # Increased from 1.0 - too aggressive clipping was causing issues
    inf_post_clip_threshold: float = 100.0  # Increased from 10.0 - more conservative threshold
    min_grad_norm_for_signal: float = 1e-10
    inf_epoch_patience: int = 5
    zero_grad_plateau_patience: int = 5

    # GaLore (low-rank gradient projection)
    use_galore: bool = False
    galore_rank: int = 64
    galore_update_interval: int = 1
    galore_scale: float = 1.0
    galore_max_dim: int = 4096

    # Precision control
    # NOTE: Default False because hybrid ODE/SSM blocks can become unstable in fp16 on older GPUs (e.g., Tesla P40).
    use_amp: bool = False

# ==================== TITAN TRAINER ====================
class TitanTrainer:
    """Advanced trainer for TORMENTED-BERT-Frankenstein with stability controls.

    Implements a production-grade training loop with gradient accumulation,
    automatic mixed precision (AMP), gradient clipping, NaN detection with
    automatic retry and learning rate scaling, rolling and best-loss
    checkpointing, GPU thermal guard integration, GaLore low-rank gradient
    projection, and CSV telemetry logging.

    Attributes:
        model: The PyTorch model being trained.
        config: Model configuration (:class:`UltraConfig` or HF config).
        training_config: :class:`TrainingConfig` with stability parameters.
        device: Resolved PyTorch device string.
        optimizer: Configured optimizer with per-component parameter groups.
        scaler: AMP gradient scaler.
        scheduler: Learning rate scheduler with warmup.
        storage_manager: :class:`StorageManager` for disk usage tracking.
        global_step: Current global training step counter.
        best_loss: Best (lowest) loss observed so far.
        gradient_accumulation_steps: Number of steps between optimizer updates.
        gpu_temp_guard: :class:`GPUTemperatureGuard` for thermal safety.
    """

    def __init__(self, model: torch.nn.Module, config: Optional[Any],
                 training_config: TrainingConfig = None,
                 device: str = None): #torch.device = None):
        """Initialize the trainer.

        Args:
            model: PyTorch model to train.
            config: Model configuration object (e.g. :class:`UltraConfig`).
            training_config: :class:`TrainingConfig` with stability and
                checkpointing parameters. Uses defaults if ``None``.
            device: Target device string. Defaults to CUDA if available,
                otherwise CPU.
        """
        self.model = model
        self.config = config
        self.training_config = training_config or TrainingConfig()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self._primary_device = str(self.device)
        self._configured_use_amp = bool(self.training_config.use_amp)
        self.use_amp = (
            self._configured_use_amp
            and torch.cuda.is_available()
            and self._primary_device.startswith("cuda")
        )
        self.amp_dtype = torch.float16
        if self.use_amp and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        self.model.to(self.device)
        
        # Specialized optimizer for hybrid architecture
        self.optimizer = self._setup_optimizer()
        
        # Mixed precision with aggressive scaling for BitNet
        self.scaler = GradScaler(
            self.device,
            init_scale=2.**10,  # Lower initial scale for BitNet stability
            growth_factor=1.5,  # Conservative growth
            backoff_factor=0.8,
            growth_interval=100
        )
        
        # Learning rate scheduler with warmup for ODE dynamics
        self.scheduler = self._setup_scheduler()
        
        # Storage monitoring
        self.storage_manager = StorageManager()
        
        # Training state tracking
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Gradient accumulation for effective larger batches
        self.gradient_accumulation_steps = max(int(self.training_config.gradient_accumulation_steps), 1)
        
        # === NEW: CSV Logger ===
        self.csv_file = None
        self.csv_writer = None
        self.csv_columns = self._get_csv_columns()

        # Optional NVML state (best-effort telemetry)
        self._nvml_module = None
        self._nvml_handle = None
        self._nvml_disabled = False
        self._nvml_warning_logged = False
        self._last_guard_temp_c: Optional[float] = None
        self._pending_thermal_repair_action = "none"
        self.gpu_temp_guard = GPUTemperatureGuard(
            enabled=bool(self.training_config.gpu_temp_guard_enabled),
            device=str(self.device),
            nvml_device_index=int(self.training_config.nvml_device_index),
            pause_threshold_c=float(self.training_config.gpu_temp_pause_threshold_c),
            resume_threshold_c=float(self.training_config.gpu_temp_resume_threshold_c),
            critical_threshold_c=self.training_config.gpu_temp_critical_threshold_c,
            poll_interval_seconds=float(self.training_config.gpu_temp_poll_interval_seconds),
        )

        # Supervisor integration: when this process is a supervisor child, the
        # supervisor sends SIGUSR1 to request an immediate rolling-checkpoint
        # flush before SIGKILL. The training loop checks the flag each step.
        self._is_supervisor_child = os.environ.get("FRANKESTEIN_SUPERVISOR_CHILD") == "1"
        self._checkpoint_request_pending = False
        self._checkpoint_dir = "checkpoints"
        if self._is_supervisor_child and self._is_main_thread():
            self._install_supervisor_sigusr1_handler()

        self._init_csv_logger()
        
        # === NEW: Rolling checkpoint tracking ===
        self.rolling_checkpoints: List[str] = []  # FIFO queue of checkpoint paths
        
        # === NEW: Best checkpoints tracking (min-heap by negative loss for max extraction) ===
        # Format: [(-loss, checkpoint_path), ...]
        self.best_checkpoints: List[Tuple[float, str]] = []
        
        # === NEW: NaN detection state ===
        self.nan_detected = False
        self.last_valid_state = None  # For debugging

        # Stability / early-stop state
        self.current_epoch_had_inf = False
        self.consecutive_inf_epochs = 0
        self.consecutive_zero_grad_plateau_epochs = 0
        self.non_improving_epochs = 0
        
        self._log_setup()

    @staticmethod
    def _is_main_thread() -> bool:
        import threading
        return threading.current_thread() is threading.main_thread()

    def _install_supervisor_sigusr1_handler(self):
        """Register SIGUSR1 to request an immediate rolling-checkpoint flush.

        The supervisor sends SIGUSR1 to this process (its direct child) before
        SIGKILL on a thermal event. The handler only sets a flag; the actual
        checkpoint write happens in the training loop
        (:meth:`_maybe_flush_supervisor_checkpoint`) to stay on the main thread
        and avoid torch re-entrancy issues.
        """
        try:
            signal.signal(signal.SIGUSR1, self._on_supervisor_sigusr1)
            logging.info("Supervisor SIGUSR1 handler installed (FRANKESTEIN_SUPERVISOR_CHILD=1)")
        except (ValueError, OSError) as exc:
            logging.warning("Could not install SIGUSR1 supervisor handler: %s", exc)

    def _on_supervisor_sigusr1(self, signum, frame):
        self._checkpoint_request_pending = True

    def _maybe_flush_supervisor_checkpoint(self, epoch: int) -> None:
        """Flush a rolling checkpoint immediately if the supervisor requested one.

        Called every training step from the main thread. After flushing, the
        sentinel file ``.supervisor_checkpoint_done`` is written into the
        checkpoint directory so the supervisor knows the flush completed and
        can proceed to SIGKILL.
        """
        if not self._checkpoint_request_pending:
            return
        try:
            self._save_rolling_checkpoint(epoch)
            os.makedirs(self._checkpoint_dir, exist_ok=True)
            sentinel = os.path.join(self._checkpoint_dir, CHECKPOINT_DONE_SENTINEL)
            with open(sentinel, "w") as fh:
                fh.write(str(self.global_step))
            logging.info("Supervisor checkpoint flush acknowledged (sentinel written)")
        except Exception as exc:
            logging.error("Supervisor checkpoint flush failed: %s", exc)
        finally:
            self._checkpoint_request_pending = False

    @staticmethod
    def find_latest_rolling_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[str]:
        """Return the path of the most recent rolling checkpoint by global_step.

        Rolling checkpoints are named ``titan_rolling_step_<n>.pt``. Returns
        ``None`` if none exist.
        """
        if not os.path.isdir(checkpoint_dir):
            return None
        pattern = os.path.join(checkpoint_dir, "titan_rolling_step_*.pt")
        best_path = None
        best_step = -1
        for path in glob.glob(pattern):
            match = re.search(r"titan_rolling_step_(\d+)\.pt$", os.path.basename(path))
            if match is None:
                continue
            step = int(match.group(1))
            if step > best_step:
                best_step = step
                best_path = path
        return best_path

    def resume_from_latest_checkpoint(self, checkpoint_spec: Optional[str]) -> int:
        """Load the specified (or latest) checkpoint and return its epoch.

        Args:
            checkpoint_spec: ``"auto"`` to pick the latest rolling checkpoint,
                an explicit path, or ``None`` to do nothing.

        Returns:
            The epoch number from the loaded checkpoint, or ``0`` if no
            checkpoint was loaded.
        """
        if not checkpoint_spec:
            return 0
        if checkpoint_spec == RESUME_AUTO:
            path = self.find_latest_rolling_checkpoint(self._checkpoint_dir)
            if path is None:
                logging.info("No rolling checkpoint found in %s; starting fresh", self._checkpoint_dir)
                return 0
        else:
            path = checkpoint_spec
        if not os.path.exists(path):
            logging.warning("Resume checkpoint not found: %s; starting fresh", path)
            return 0
        return int(self.load_checkpoint(path))

    def _get_csv_columns(self) -> List[str]:
        """CSV schema for step-level training metrics."""
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

    def _get_default_gpu_telemetry(self) -> Dict[str, float]:
        return {
            "gpu_temp_c": 0.0,
            "gpu_power_w": 0.0,
            "gpu_util_pct": 0.0,
            "gpu_mem_used_mib": 0.0,
        }

    def _get_default_block_grad_norms(self) -> Dict[str, float]:
        return {
            "embeddings": 0.0,
            "attention": 0.0,
            "ffn": 0.0,
            "experts": 0.0,
            "router": 0.0,
            "ode": 0.0,
            "retnet": 0.0,
            "mamba": 0.0,
            "norms": 0.0,
            "head": 0.0,
            "other": 0.0,
        }

    def _append_repair_action(self, current: str, extra: str) -> str:
        cur = str(current or "none").strip() or "none"
        add = str(extra or "none").strip() or "none"
        if add == "none":
            return cur
        if cur == "none":
            return add
        existing = set(cur.split("+"))
        if add in existing:
            return cur
        return f"{cur}+{add}"

    def _enforce_thermal_guard(self, epoch: int, batch_idx: int) -> str:
        """Thermal guard hook (no-op under the supervisor model).

        The CPU-side :class:`GPUTempSupervisor` owns the training subprocess
        lifecycle and handles thermal events by checkpointing + killing this
        process. In-process thermal offload is therefore no longer performed
        here. This method is kept as a no-op so the training loop remains
        unchanged and reads ``"none"`` for the CSV ``repair_action`` column.

        Args:
            epoch: Current epoch (0-indexed).
            batch_idx: Current batch index within the epoch.

        Returns:
            Always ``"none"``.
        """
        return "none"

    def _merge_guard_temperature(self, telemetry: Dict[str, float]) -> Dict[str, float]:
        merged = dict(telemetry)
        if self.gpu_temp_guard.is_active and self._last_guard_temp_c is not None:
            merged["gpu_temp_c"] = float(self._last_guard_temp_c)
        return merged

    def _read_csv_header(self, csv_path: str) -> Optional[List[str]]:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            return None
        with open(csv_path, 'r', newline='', encoding='utf-8') as handle:
            reader = csv.reader(handle)
            return next(reader, None)
    
    def _init_csv_logger(self):
        """Initialize CSV file for metrics logging"""
        csv_path = self.training_config.csv_log_path
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        file_exists = os.path.exists(csv_path)
        if file_exists:
            existing_header = self._read_csv_header(csv_path)
            if existing_header is not None and existing_header != self.csv_columns:
                if self.training_config.csv_rotate_on_schema_change:
                    base, ext = os.path.splitext(csv_path)
                    if not ext:
                        ext = ".csv"
                    rotated_path = f"{base}.{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
                    os.replace(csv_path, rotated_path)
                    logging.info(
                        f"CSV schema changed. Rotated previous log to: {rotated_path}"
                    )
                    file_exists = False
                else:
                    raise RuntimeError(
                        "CSV schema mismatch for existing metrics file. "
                        "Enable csv_rotate_on_schema_change or remove the old CSV."
                    )

        self.csv_file = open(csv_path, 'a', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)

        # Write header for new or empty file
        should_write_header = (not file_exists) or os.path.getsize(csv_path) == 0
        if should_write_header:
            self.csv_writer.writerow(self.csv_columns)
            self.csv_file.flush()

    def _get_gpu_telemetry(self) -> Dict[str, float]:
        """Collect GPU telemetry with best-effort NVML fallback."""
        telemetry = self._get_default_gpu_telemetry()
        if not torch.cuda.is_available():
            return telemetry

        backend = str(self.training_config.gpu_metrics_backend).strip().lower()
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
                idx = int(self.training_config.nvml_device_index)
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                self._nvml_module = pynvml
            except Exception as exc:
                self._nvml_disabled = True
                if not self._nvml_warning_logged:
                    logging.warning(f"NVML telemetry unavailable, falling back to nvidia-smi: {exc}")
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
                logging.warning(f"Failed reading NVML telemetry, falling back to nvidia-smi: {exc}")
                self._nvml_warning_logged = True
            return self._get_gpu_telemetry_from_nvidia_smi()
        return telemetry

    def _parse_smi_value(self, raw: str) -> float:
        value = raw.strip()
        if not value or value.lower() in {"n/a", "[not supported]"}:
            return 0.0
        match = re.search(r"[-+]?\d*\.?\d+", value)
        if match is None:
            return 0.0
        try:
            return float(match.group(0))
        except ValueError:
            return 0.0

    def _get_gpu_telemetry_from_nvidia_smi(self) -> Dict[str, float]:
        telemetry = self._get_default_gpu_telemetry()
        cmd = [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            str(int(self.training_config.nvml_device_index)),
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
            # Best-effort fallback: keep zeros if nvidia-smi is unavailable/unreadable.
            pass
        return telemetry

    def _bucket_for_param_name(self, param_name: str) -> str:
        name = param_name.lower()
        if "router" in name:
            return "router"
        if "experts" in name:
            return "experts"
        if "embed" in name or ".emb." in name or "factorized_embedding" in name:
            return "embeddings"
        if "ode" in name:
            return "ode"
        if "retnet" in name or "retention" in name:
            return "retnet"
        if "mamba" in name:
            return "mamba"
        if "ffn" in name:
            return "ffn"
        if "norm" in name:
            return "norms"
        if name.startswith("head") or ".head." in name:
            return "head"
        if (
            "attn" in name
            or "attention" in name
            or "q_proj" in name
            or "k_proj" in name
            or "v_proj" in name
            or "out_proj" in name
        ):
            return "attention"
        return "other"

    def _collect_block_grad_norms(self) -> Dict[str, float]:
        """Collect L2 gradient norms by fixed logical component buckets."""
        bucket_sq_sums = {k: 0.0 for k in self._get_default_block_grad_norms().keys()}
        if not self.training_config.enable_block_grad_norms:
            return self._get_default_block_grad_norms()

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            bucket = self._bucket_for_param_name(name)
            bucket_sq_sums[bucket] += float((grad.float().pow(2).sum()).item())

        return {key: math.sqrt(max(val, 0.0)) for key, val in bucket_sq_sums.items()}
    
    def _log_step_to_csv(self, epoch: int, step: int, loss: float, accuracy: float,
                         lr: float, grad_norm: float = 0.0,
                         has_nan: bool = False, has_inf: bool = False,
                         has_zero: bool = False, repair_action: str = "none",
                         gpu_telemetry: Optional[Dict[str, float]] = None,
                         block_grad_norms: Optional[Dict[str, float]] = None,
                         step_time_ms: float = 0.0, tokens_per_sec: float = 0.0,
                         clip_ratio: float = 0.0, effective_batch_size: int = 0):
        """Log a single training step to CSV"""
        gpu_memory = 0.0
        gpu_cached = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_cached = torch.cuda.memory_reserved() / 1024**3

        telemetry = gpu_telemetry or self._get_default_gpu_telemetry()
        grad_buckets = block_grad_norms or self._get_default_block_grad_norms()
        
        self.csv_writer.writerow([
            datetime.now().isoformat(),
            epoch,
            step,
            self.global_step,
            f'{loss:.6f}',
            f'{accuracy:.6f}',
            f'{lr:.8e}',
            f'{grad_norm:.6f}',
            f'{self.scaler.get_scale():.2f}',
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
            f"{float(grad_buckets['ode']):.6f}",
            f"{float(grad_buckets['retnet']):.6f}",
            f"{float(grad_buckets['mamba']):.6f}",
            f"{float(grad_buckets['norms']):.6f}",
            f"{float(grad_buckets['head']):.6f}",
            f"{float(grad_buckets['other']):.6f}",
            f"{float(step_time_ms):.2f}",
            f"{float(tokens_per_sec):.2f}",
            f"{float(clip_ratio):.6f}",
            int(effective_batch_size),
        ])
        self.csv_file.flush()
    
    def _check_for_nan(self, loss: torch.Tensor, step: int, batch: dict) -> bool:
        """Check if loss is NaN and log debug information if detected"""
        if not torch.isnan(loss) and not torch.isinf(loss):
            # Save last valid state for debugging
            if step % 50 == 0:  # Don't save every step for memory efficiency
                self.last_valid_state = {
                    'step': step,
                    'loss': loss.item(),
                    'scaler_scale': self.scaler.get_scale()
                }
            return False
        
        # NaN detected! Log comprehensive debug information
        self.nan_detected = True
        logging.error("=" * 80)
        logging.error("🚨 NaN/Inf DETECTED IN LOSS - STOPPING TRAINING 🚨")
        logging.error("=" * 80)
        
        logging.error(f"Step: {self.global_step}, Batch index: {step}")
        logging.error(f"Loss value: {loss.item()}")
        logging.error(f"Scaler scale: {self.scaler.get_scale()}")
        
        # Log last valid state
        if self.last_valid_state:
            logging.error(f"Last valid state: step={self.last_valid_state['step']}, "
                         f"loss={self.last_valid_state['loss']:.6f}, "
                         f"scale={self.last_valid_state['scaler_scale']}")
        
        # Log learning rates for each param group
        logging.error("Learning rates by parameter group:")
        for i, group in enumerate(self.optimizer.param_groups):
            name = group.get('name', f'group_{i}')
            logging.error(f"  {name}: lr={group['lr']:.8e}")
        
        # Check gradients
        logging.error("Gradient statistics by parameter:")
        nan_params = []
        inf_params = []
        large_grad_params = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                if torch.isnan(grad).any():
                    nan_params.append(name)
                elif torch.isinf(grad).any():
                    inf_params.append(name)
                else:
                    grad_max = grad.abs().max().item()
                    if grad_max > 1000:
                        large_grad_params.append((name, grad_max))
        
        if nan_params:
            logging.error(f"Parameters with NaN gradients ({len(nan_params)}):")
            for p in nan_params[:10]:  # Limit output
                logging.error(f"  - {p}")
        
        if inf_params:
            logging.error(f"Parameters with Inf gradients ({len(inf_params)}):")
            for p in inf_params[:10]:
                logging.error(f"  - {p}")
        
        if large_grad_params:
            logging.error(f"Parameters with large gradients (>1000) ({len(large_grad_params)}):")
            for name, val in sorted(large_grad_params, key=lambda x: -x[1])[:10]:
                logging.error(f"  - {name}: {val:.2f}")
        
        # Check model weights
        logging.error("Model weight statistics:")
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                logging.error(f"  NaN weights in: {name}")
            elif torch.isinf(param).any():
                logging.error(f"  Inf weights in: {name}")
        
        # Log input batch statistics
        logging.error("Input batch statistics:")
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                logging.error(f"  {key}: shape={val.shape}, dtype={val.dtype}, "
                            f"min={val.min().item()}, max={val.max().item()}")
        
        # GPU memory state
        if torch.cuda.is_available():
            logging.error(f"GPU Memory - Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB, "
                         f"Cached: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
        
        # Log config that might affect stability
        logging.error("Model configuration (stability-relevant):")
        logging.error(f"  - BitNet enabled: {getattr(self.config, 'use_bitnet', 'n/a')}")
        logging.error(
            "  - ODE solver: %s, steps: %s",
            getattr(self.config, "ode_solver", "n/a"),
            getattr(self.config, "ode_steps", "n/a"),
        )
        logging.error(f"  - Layer pattern: {getattr(self.config, 'layer_pattern', 'n/a')}")
        logging.error(f"  - Num loops: {getattr(self.config, 'num_loops', 'n/a')}")
        logging.error(f"  - Dropout: {getattr(self.config, 'dropout', 'n/a')}")
        
        logging.error("=" * 80)
        
        return True

    def _inspect_gradients(self) -> Dict[str, float]:
        """Inspect gradients right after backward for NaN/Inf/zero and global norm validity."""
        has_nan = False
        has_inf = False
        has_zero = False
        total_norm = 0.0

        for param in self.model.parameters():
            if param.grad is None:
                continue

            grad = param.grad
            if torch.isnan(grad).any():
                has_nan = True
            if torch.isinf(grad).any():
                has_inf = True
            if (grad == 0).all():
                has_zero = True

            grad_norm = grad.norm().item()
            total_norm += grad_norm

        if not torch.isfinite(torch.tensor(total_norm, device=self.device)):
            has_inf = True

        if total_norm < self.training_config.min_grad_norm_for_signal:
            has_zero = True

        return {
            "has_nan": has_nan,
            "has_inf": has_inf,
            "has_zero": has_zero,
            "total_norm": float(total_norm),
        }

    def _apply_galore_projection(self):
        """Apply low-rank projection to gradients (GaLore-style) if enabled."""
        if not self.training_config.use_galore:
            return
        if self.global_step % max(int(self.training_config.galore_update_interval), 1) != 0:
            return

        rank = max(int(self.training_config.galore_rank), 1)
        max_dim = int(self.training_config.galore_max_dim)
        scale = float(self.training_config.galore_scale)

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if param.grad.dim() != 2:
                continue
            if "bias" in name.lower() or "embed" in name.lower():
                continue

            grad = param.grad
            rows, cols = grad.shape
            if min(rows, cols) < rank:
                continue
            if max(rows, cols) > max_dim:
                continue

            try:
                u, s, vh = torch.linalg.svd(grad.float(), full_matrices=False)
            except RuntimeError:
                continue

            u = u[:, :rank]
            s = s[:rank]
            vh = vh[:rank, :]
            approx = (u * s) @ vh
            grad.copy_(approx.to(dtype=grad.dtype) * scale)

    def _scale_learning_rate(self, factor: float):
        """Scale all optimizer param-group learning rates by a multiplicative factor."""
        for group in self.optimizer.param_groups:
            group['lr'] = max(group['lr'] * factor, 1e-9)  # Lower floor to allow more aggressive reduction
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with parameter groups for different components"""
        # Separate parameters by component type
        titan_attn_params = []
        norm_params = []
        embed_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if ('titan_attn' in name.lower() or 'attention' in name.lower()
                    or 'ode' in name.lower() or 'retention' in name.lower()
                    or 'retnet' in name.lower() or 'mamba' in name.lower()):
                titan_attn_params.append(param)
            elif 'norm' in name.lower() or 'layer_norm' in name.lower():
                norm_params.append(param)
            elif 'embed' in name.lower():
                embed_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {
                'params': embed_params,
                'lr': 1e-6,
                'weight_decay': 0.01,
                'betas': (0.9, 0.95),
                'eps': 1e-8,
                'name': 'embeddings'
            },
            {
                'params': norm_params,
                'lr': 5e-6,
                'weight_decay': 0.001,
                'betas': (0.9, 0.95),
                'eps': 1e-8,
                'name': 'norms'
            },
            {
                'params': titan_attn_params,
                'lr': 3e-6,
                'weight_decay': 0.01,
                'betas': (0.9, 0.95),
                'eps': 1e-8,
                'name': 'attention'
            },
            {
                'params': other_params,
                'lr': 2e-6,
                'weight_decay': 0.01,
                'betas': (0.9, 0.95),
                'eps': 1e-8,
                'name': 'other'
            }
        ]
        
        # Filter out empty groups
        param_groups = [group for group in param_groups if group['params']]
        
        # Log parameter distribution
        for group in param_groups:
            param_count = sum(p.numel() for p in group['params'])
            logging.info(f"Parameter group '{group['name']}': {param_count:,} parameters, lr={group['lr']}")
        
        optimizer = build_optimizer(
            optimizer_class=self.training_config.optimizer_class,
            param_groups=param_groups,
            parameters=self.training_config.optimizer_parameters,
        )
        logging.info(f"Using optimizer: {self.training_config.optimizer_class}")
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup"""
        total_steps = int(self.training_config.scheduler_total_steps)
        if total_steps <= 0:
            total_steps = 10000  # Estimated total steps
        warmup_ratio = float(self.training_config.scheduler_warmup_ratio)
        warmup_ratio = min(max(warmup_ratio, 0.0), 1.0)
        warmup_steps = int(warmup_ratio * total_steps)
        
        scheduler_type = str(self.training_config.scheduler_type).strip().lower()

        def lr_lambda(step):
            if scheduler_type == "constant":
                return 1.0
            if scheduler_type == "linear_warmup_then_constant":
                if warmup_steps == 0:
                    return 1.0
                if step < warmup_steps:
                    return (step + 1) / warmup_steps
                return 1.0
            # cosine (default)
            if warmup_steps == 0:
                progress = step / max(total_steps, 1)
                progress = min(max(progress, 0.0), 1.0)
                return 0.5 * (1 + math.cos(math.pi * progress))
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _log_setup(self):
        """Log training setup information"""
        logging.info(f"🎯 Training on device: {self.device}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logging.info(f"🔥 GPU: {gpu_name}")
            logging.info(f"💾 GPU Memory: {gpu_memory:.2f}GB")
            
            # Enable optimizations for P40
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Model-specific logging
        logging.info("🧠 Architecture: %s", self.model.__class__.__name__)
        logging.info(f"⚡ BitNet Quantization: {getattr(self.config, 'use_bitnet', 'n/a')}")
        logging.info(
            "🌊 ODE Solver: %s (%s steps)",
            getattr(self.config, "ode_solver", "n/a"),
            getattr(self.config, "ode_steps", "n/a"),
        )
        logging.info(f"🔄 Recursion Loops: {getattr(self.config, 'num_loops', 'n/a')}")
        logging.info(f"📐 Gradient Accumulation: {self.gradient_accumulation_steps} steps")
        if self.gpu_temp_guard.is_active:
            logging.info(
                "🌡️ GPU thermal supervisor enabled (pause>=%.1fC, resume<=%.1fC, poll=%.1fs, critical=%s, grace=%.1fs)",
                float(self.training_config.gpu_temp_pause_threshold_c),
                float(self.training_config.gpu_temp_resume_threshold_c),
                float(self.training_config.gpu_temp_poll_interval_seconds),
                (
                    f"{float(self.training_config.gpu_temp_critical_threshold_c):.1f}C"
                    if self.training_config.gpu_temp_critical_threshold_c is not None
                    else "disabled"
                ),
                float(self.training_config.gpu_temp_checkpoint_grace_seconds),
            )
            if self._is_supervisor_child:
                logging.info("🌡️ Running as supervisor child (SIGUSR1 checkpoint handler active)")
            if self.training_config.gpu_temp_critical_threshold_c is not None:
                logging.info(
                    "🌡️ Critical threshold: %s (switch_on_thermal=%s)",
                    f"{float(self.training_config.gpu_temp_critical_threshold_c):.1f}C",
                    "CPU-continuation" if bool(self.training_config.switch_on_thermal) else "permanent-abort",
                )
            logging.info(
                "🌡️ switch_on_thermal=%s",
                bool(self.training_config.switch_on_thermal),
            )
        else:
            logging.info("🌡️ GPU thermal guard disabled")

    def _forward_mlm_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass compatible with custom models and HF masked-LM outputs."""
        try:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        except TypeError:
            try:
                outputs = self.model(input_ids, attention_mask=attention_mask)
            except TypeError:
                outputs = self.model(input_ids)

        if torch.is_tensor(outputs):
            return outputs
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
            if logits is not None:
                return logits
        if hasattr(outputs, "logits"):
            return outputs.logits

        raise RuntimeError(
            "Model forward output does not include MLM logits. "
            "Expected tensor, dict['logits'], or object.logits."
        )
    
    def compute_mlm_loss(self, input_ids, attention_mask, labels):
        """Compute masked language modeling loss with shape safety checks.

        Handles both HF-style labels (``-100`` on unmasked tokens) and
        raw-sequence labels (infers masked positions by comparing input
        and label token IDs). Includes a fallback that forces at least one
        valid supervised token when no MLM targets exist.

        Args:
            input_ids: Token ID tensor of shape ``(B, S)``.
            attention_mask: Attention mask tensor of shape ``(B, S)``.
            labels: Label tensor of shape ``(B, S)``.

        Returns:
            Tuple of ``(loss, accuracy)`` where ``loss`` is a scalar tensor
            and ``accuracy`` is a scalar tensor in ``[0, 1]``.

        Raises:
            RuntimeError: If non-finite logits are detected, or if tensor
                shapes are mismatched.
        """
        # Forward pass
        logits = self._forward_mlm_logits(input_ids, attention_mask)
        auxiliary_losses = getattr(self.model, "last_auxiliary_losses", None)

        # Safety: avoid propagating non-finite logits into CE (would yield NaN loss)
        if not torch.isfinite(logits).all():
            bad = (~torch.isfinite(logits)).sum().item()
            raise RuntimeError(f"Non-finite logits detected in forward pass: {bad} elements")
        
        # Reshape for loss computation
        batch_size, seq_len, vocab_size = logits.shape
        
        # CRITICAL: Ensure all tensors have matching shapes before flattening
        # Variable sequence lengths can cause shape mismatches
        if labels.shape != input_ids.shape:
            raise RuntimeError(f"Shape mismatch: labels {labels.shape} != input_ids {input_ids.shape}")
        if attention_mask.shape != input_ids.shape:
            raise RuntimeError(f"Shape mismatch: attention_mask {attention_mask.shape} != input_ids {input_ids.shape}")
        
        logits = logits.view(-1, vocab_size)
        labels = labels.view(-1)
        input_flat = input_ids.view(-1)
        attn_flat = attention_mask.view(-1)
        
        # Support both label conventions:
        # 1) labels already contain -100 on unmasked tokens (HF-style)
        # 2) labels contain full original sequence; infer masked positions by input!=label
        if (labels == -100).any():
            mlm_labels = labels.clone()
        else:
            mlm_labels = labels.clone()
            mlm_labels[labels == input_flat] = -100

        valid_mask = (mlm_labels != -100)

        # Rare fallback: if no valid MLM targets exist, force one valid supervised token
        # from attended positions to keep CE finite and training alive.
        if valid_mask.sum().item() == 0:
            candidate_positions = torch.nonzero(attn_flat > 0, as_tuple=False)
            if candidate_positions.numel() > 0:
                first_idx = int(candidate_positions[0].item())
                mlm_labels[first_idx] = labels[first_idx]
                valid_mask = (mlm_labels != -100)

        # Compute CE in float32 for numerical stability
        if valid_mask.sum().item() > 0:
            loss = F.cross_entropy(logits[valid_mask].float(), mlm_labels[valid_mask])
        else:
            # Defensive fallback (should almost never happen after forced target above)
            loss = logits.new_zeros((), requires_grad=True)

        if isinstance(auxiliary_losses, dict):
            for aux_name, aux_loss in auxiliary_losses.items():
                if torch.is_tensor(aux_loss) and aux_loss.requires_grad:
                    if not torch.isfinite(aux_loss):
                        raise RuntimeError(
                            f"Non-finite auxiliary loss detected for {aux_name} during forward pass"
                        )
                    loss = loss + aux_loss
        
        # Compute accuracy on masked tokens
        with torch.no_grad():
            mask = valid_mask
            if mask.any():
                predictions = logits.argmax(dim=-1)
                correct = (predictions == mlm_labels) & mask
                accuracy = correct.sum().float() / mask.sum().float()
            else:
                accuracy = torch.tensor(0.0, device=logits.device)
        
        return loss, accuracy
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Tuple[float, bool]:
        """
        Train for one epoch with advanced monitoring.
        
        Returns:
            Tuple[float, bool]: (average_loss, should_stop)
            should_stop is True if NaN was detected
        """
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        grad_norm = 0.0
        self.current_epoch_had_inf = False
        current_epoch_had_zero = False
        
        # Progress bar with detailed metrics
        progress_bar = tqdm(
            dataloader, 
            desc=f"🚀 Epoch {epoch+1}",
            leave=True,
            ncols=120
        )
        epoch_started = time.perf_counter()
        
        self.optimizer.zero_grad(set_to_none=True)
        optimizer_window_start = time.perf_counter()
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                thermal_action = self._enforce_thermal_guard(epoch, batch_idx)
                if thermal_action != "none":
                    self._pending_thermal_repair_action = self._append_repair_action(
                        self._pending_thermal_repair_action,
                        thermal_action,
                    )
                # Move to device
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                repair_action = self._pending_thermal_repair_action
                retry_count = 0
                batch_done = False
                amp_enabled = self.use_amp and str(self.device).startswith("cuda")

                while retry_count <= self.training_config.max_nan_retries and not batch_done:
                    # Forward pass (AMP if enabled)
                    with autocast(self.device, enabled=amp_enabled, dtype=self.amp_dtype):
                        loss, accuracy = self.compute_mlm_loss(
                            batch['input_ids'],
                            batch['attention_mask'],
                            batch['labels']
                        )

                        if self._check_for_nan(loss * self.gradient_accumulation_steps, batch_idx, batch):
                            repair_action = self._append_repair_action(
                                repair_action,
                                f"nan_loss_retry_{retry_count + 1}",
                            )
                            self.optimizer.zero_grad(set_to_none=True)
                            self._scale_learning_rate(0.5)
                            retry_count += 1

                            if retry_count > self.training_config.max_nan_retries:
                                try:
                                    emergency_path = self.save_checkpoint(epoch, suffix="_nan_emergency")
                                    logging.error(f"Emergency checkpoint saved: {emergency_path}")
                                except Exception as e:
                                    logging.error(f"Failed to save emergency checkpoint: {e}")
                                return total_loss / max(num_batches, 1), True
                            continue

                        raw_loss = loss.item()
                        scaled_loss = loss / self.gradient_accumulation_steps

                    # Backward pass
                    if amp_enabled:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    grad_stats = self._inspect_gradients()
                    current_epoch_had_zero = current_epoch_had_zero or grad_stats["has_zero"]

                    # Repair policy for NaN gradients: skip step, zero grad, LR*0.75 retry up to N times
                    if grad_stats["has_nan"]:
                        repair_action = self._append_repair_action(
                            repair_action,
                            f"nan_grad_retry_{retry_count + 1}",
                        )
                        self.optimizer.zero_grad(set_to_none=True)
                        self._scale_learning_rate(0.75)  # Less aggressive reduction (was 0.5)
                        retry_count += 1

                        if retry_count > self.training_config.max_nan_retries:
                            logging.error(
                                "NaN gradients persisted after retries. Saving checkpoint and stopping training."
                            )
                            try:
                                emergency_path = self.save_checkpoint(epoch, suffix="_nan_grad_emergency")
                                logging.error(f"Emergency checkpoint saved: {emergency_path}")
                            except Exception as e:
                                logging.error(f"Failed to save emergency checkpoint: {e}")
                            return total_loss / max(num_batches, 1), True
                        continue

                    # Update every gradient_accumulation_steps
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        if amp_enabled:
                            self.scaler.unscale_(self.optimizer)

                        self._apply_galore_projection()

                        grad_norm = clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.training_config.grad_clip_max_norm
                        )
                        grad_norm_value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                        clip_ratio = grad_norm_value / max(float(self.training_config.grad_clip_max_norm), 1e-12)

                        telemetry_interval = max(int(self.training_config.telemetry_log_interval), 1)
                        should_log_heavy = ((self.global_step + 1) % telemetry_interval == 0)
                        gpu_telemetry = (
                            self._get_gpu_telemetry()
                            if should_log_heavy
                            else self._get_default_gpu_telemetry()
                        )
                        gpu_telemetry = self._merge_guard_temperature(gpu_telemetry)
                        block_grad_norms = (
                            self._collect_block_grad_norms()
                            if should_log_heavy
                            else self._get_default_block_grad_norms()
                        )
                        step_time_ms = (time.perf_counter() - optimizer_window_start) * 1000.0
                        effective_batch_size = int(batch['input_ids'].shape[0]) * self.gradient_accumulation_steps
                        token_count = effective_batch_size * int(batch['input_ids'].shape[1])
                        tokens_per_sec = float(token_count) / max(step_time_ms / 1000.0, 1e-9)

                        post_clip_bad = (not math.isfinite(grad_norm_value)) or (
                            grad_norm_value > self.training_config.inf_post_clip_threshold
                        )

                        # Repair policy for Inf/exploding gradients
                        if grad_stats["has_inf"] or post_clip_bad:
                            self.current_epoch_had_inf = True
                            repair_action = self._append_repair_action(
                                repair_action,
                                "inf_skip_step_lr_half",
                            )
                            self._scale_learning_rate(0.5)

                            self.global_step += 1
                            current_lr = self.scheduler.get_last_lr()[0]
                            self._log_step_to_csv(
                                epoch=epoch,
                                step=batch_idx,
                                loss=raw_loss,
                                accuracy=accuracy.item(),
                                lr=current_lr,
                                grad_norm=grad_norm_value,
                                has_nan=grad_stats["has_nan"],
                                has_inf=True,
                                has_zero=grad_stats["has_zero"],
                                repair_action=repair_action,
                                gpu_telemetry=gpu_telemetry,
                                block_grad_norms=block_grad_norms,
                                step_time_ms=step_time_ms,
                                tokens_per_sec=tokens_per_sec,
                                clip_ratio=clip_ratio,
                                effective_batch_size=effective_batch_size,
                            )
                            self._pending_thermal_repair_action = "none"
                            self.optimizer.zero_grad(set_to_none=True)
                            optimizer_window_start = time.perf_counter()
                            batch_done = True
                            continue

                        # Normal optimizer step
                        if amp_enabled:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()

                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

                        self.global_step += 1
                        current_lr = self.scheduler.get_last_lr()[0]
                        self._log_step_to_csv(
                            epoch=epoch,
                            step=batch_idx,
                            loss=raw_loss,
                            accuracy=accuracy.item(),
                            lr=current_lr,
                            grad_norm=grad_norm_value,
                            has_nan=grad_stats["has_nan"],
                            has_inf=grad_stats["has_inf"],
                            has_zero=grad_stats["has_zero"],
                            repair_action=repair_action,
                            gpu_telemetry=gpu_telemetry,
                            block_grad_norms=block_grad_norms,
                            step_time_ms=step_time_ms,
                            tokens_per_sec=tokens_per_sec,
                            clip_ratio=clip_ratio,
                            effective_batch_size=effective_batch_size,
                        )
                        self._pending_thermal_repair_action = "none"
                        optimizer_window_start = time.perf_counter()

                        if self.global_step % self.training_config.checkpoint_every_n_steps == 0:
                            self._save_rolling_checkpoint(epoch)

                        # Flush an on-demand checkpoint if the supervisor
                        # sent SIGUSR1 (thermal kill is imminent).
                        self._maybe_flush_supervisor_checkpoint(epoch)

                        self._maybe_save_best_checkpoint(epoch, raw_loss)

                    # Accumulate metrics
                    total_loss += raw_loss
                    total_accuracy += accuracy.item()
                    num_batches += 1
                    batch_done = True

                if not batch_done:
                    return total_loss / max(num_batches, 1), True
                
                # Update progress bar
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{(total_loss / max(num_batches, 1)):.4f}',
                    'acc': f'{accuracy.item():.3f}',
                    'lr': f'{current_lr:.2e}',
                    'step': self.global_step
                })
                
                # Memory management every 50 steps
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Detailed logging every 100 steps
                if batch_idx % 100 == 0 and batch_idx > 0:
                    avg_loss = total_loss / num_batches
                    avg_acc = total_accuracy / num_batches
                    progress_snapshot = tqdm.format_meter(
                        batch_idx + 1,
                        len(dataloader),
                        time.perf_counter() - epoch_started,
                    )
                    logging.info("Progress %s", progress_snapshot)
                    
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / 1024**3
                        memory_cached = torch.cuda.memory_reserved() / 1024**3
                        logging.info(
                            f"Step {self.global_step}: Loss={avg_loss:.4f}, "
                            f"Acc={avg_acc:.3f}, GPU={memory_used:.2f}GB"
                        )
                
            except RuntimeError as e:
                if isinstance(e, GPUTelemetryError):
                    telemetry = self._merge_guard_temperature(self._get_default_gpu_telemetry())
                    self._log_step_to_csv(
                        epoch=epoch,
                        step=batch_idx,
                        loss=float("nan"),
                        accuracy=0.0,
                        lr=self.scheduler.get_last_lr()[0],
                        grad_norm=0.0,
                        has_nan=False,
                        has_inf=True,
                        has_zero=False,
                        repair_action="telemetry_fatal_stop",
                        gpu_telemetry=telemetry,
                        block_grad_norms=self._get_default_block_grad_norms(),
                        step_time_ms=0.0,
                        tokens_per_sec=0.0,
                        clip_ratio=0.0,
                        effective_batch_size=0,
                    )
                    logging.error(
                        "Fatal GPU telemetry error at batch %s: %s. Stopping training immediately.",
                        batch_idx,
                        e,
                    )
                    raise
                # In-process thermal CPU fallback is no longer used; the
                # supervisor owns the subprocess lifecycle and handles GPU
                # runtime errors by killing this child. Re-raise so the
                # supervisor can react (and re-launch on CPU if
                # switch_on_thermal is enabled).
                if "out of memory" in str(e):
                    telemetry = self._merge_guard_temperature(self._get_default_gpu_telemetry())
                    self._log_step_to_csv(
                        epoch=epoch,
                        step=batch_idx,
                        loss=float("nan"),
                        accuracy=0.0,
                        lr=self.scheduler.get_last_lr()[0],
                        grad_norm=0.0,
                        has_nan=False,
                        has_inf=False,
                        has_zero=False,
                        repair_action="oom_skip_batch",
                        gpu_telemetry=telemetry,
                        block_grad_norms=self._get_default_block_grad_norms(),
                        step_time_ms=0.0,
                        tokens_per_sec=0.0,
                        clip_ratio=0.0,
                        effective_batch_size=0,
                    )
                    logging.error(f"OOM at batch {batch_idx}, clearing cache and continuing...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                else:
                    raise e
        
        # Final epoch metrics
        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)
        
        # Update best loss / plateau tracking
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.non_improving_epochs = 0
            logging.info(f"🎉 New best loss: {self.best_loss:.4f}")
        else:
            self.non_improving_epochs += 1

        if self.current_epoch_had_inf:
            self.consecutive_inf_epochs += 1
        else:
            self.consecutive_inf_epochs = 0

        if current_epoch_had_zero and self.non_improving_epochs >= self.training_config.zero_grad_plateau_patience:
            self.consecutive_zero_grad_plateau_epochs += 1
        else:
            self.consecutive_zero_grad_plateau_epochs = 0
        
        logging.info(
            f"📊 Epoch {epoch+1} Summary: "
            f"Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.3f}, "
            f"Best Loss={self.best_loss:.4f}, "
            f"InfEpochs={self.consecutive_inf_epochs}, "
            f"ZeroGradPlateauEpochs={self.consecutive_zero_grad_plateau_epochs}"
        )

        # Early stop policy: unrecoverable exploding gradients across epochs
        if self.consecutive_inf_epochs > self.training_config.inf_epoch_patience:
            logging.error(
                "Early stop: Inf/exploding gradients persisted for too many consecutive epochs."
            )
            return avg_loss, True

        # Early stop policy: vanishing gradients + plateau persistence
        if self.consecutive_zero_grad_plateau_epochs >= 1:
            logging.error(
                "Early stop: persistent near-zero gradients with plateaued loss detected."
            )
            return avg_loss, True
        
        return avg_loss, False  # No NaN detected
    
    def _save_rolling_checkpoint(self, epoch: int, path: str = "checkpoints"):
        """Save a rolling checkpoint and manage the queue"""
        os.makedirs(path, exist_ok=True)
        
        filename = f"titan_rolling_step_{self.global_step}.pt"
        checkpoint_path = os.path.join(path, filename)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss,
            'model_class': self.model.__class__.__name__
        }
        
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"💾 Rolling checkpoint saved: {checkpoint_path}")
        
        # Add to queue
        self.rolling_checkpoints.append(checkpoint_path)
        
        # Remove old checkpoints if exceeding max
        while len(self.rolling_checkpoints) > self.training_config.max_rolling_checkpoints:
            old_checkpoint = self.rolling_checkpoints.pop(0)
            try:
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    logging.info(f"🗑️ Removed old rolling checkpoint: {old_checkpoint}")
            except Exception as e:
                logging.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")
    
    def _maybe_save_best_checkpoint(self, epoch: int, current_loss: float, path: str = "checkpoints"):
        """Save checkpoint if it's among the top K best models"""
        os.makedirs(path, exist_ok=True)
        
        k = self.training_config.num_best_checkpoints
        
        # Use negative loss because heapq is a min-heap
        # We want to keep the K smallest losses (best models)
        should_save = False
        
        if len(self.best_checkpoints) < k:
            should_save = True
        elif current_loss < -self.best_checkpoints[0][0]:  # Compare with worst of the best
            should_save = True
        
        if should_save:
            filename = f"titan_best_loss_{current_loss:.6f}_step_{self.global_step}.pt"
            checkpoint_path = os.path.join(path, filename)
            
            checkpoint = {
                'epoch': epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'config': self.config,
                'best_loss': current_loss,
                'model_class': self.model.__class__.__name__
            }
            
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"⭐ Best model checkpoint saved: {checkpoint_path} (loss={current_loss:.6f})")
            
            # Add to heap (using negative loss for min-heap to act as max-heap)
            heapq.heappush(self.best_checkpoints, (-current_loss, checkpoint_path))
            
            # Remove worst of the best if exceeding K
            if len(self.best_checkpoints) > k:
                _, removed_path = heapq.heappop(self.best_checkpoints)
                try:
                    if os.path.exists(removed_path):
                        os.remove(removed_path)
                        logging.info(f"🗑️ Removed replaced best checkpoint: {removed_path}")
                except Exception as e:
                    logging.warning(f"Failed to remove old best checkpoint {removed_path}: {e}")
    
    def close(self):
        """Cleanup resources"""
        if self.csv_file:
            self.csv_file.close()
            logging.info(f"📊 CSV log closed: {self.training_config.csv_log_path}")
    
    def save_checkpoint(self, epoch: int, suffix: str = "", path: str = "checkpoints") -> str:
        """Save a comprehensive training checkpoint.

        Includes model state, optimizer state, scheduler state, AMP scaler
        state, config, global step, and best loss.

        Args:
            epoch: Current epoch number.
            suffix: Optional filename suffix (e.g. ``"_epoch_end"``).
            path: Directory to save the checkpoint in.

        Returns:
            Absolute path to the saved checkpoint file.
        """
        os.makedirs(path, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss,
            'model_class': self.model.__class__.__name__
        }
        
        filename = f"titan_checkpoint_epoch_{epoch}{suffix}.pt"
        checkpoint_path = os.path.join(path, filename)
        
        torch.save(checkpoint, checkpoint_path)
        
        # Register with storage manager
        if self.storage_manager.register_file(checkpoint_path):
            logging.info(f"💾 Checkpoint saved: {checkpoint_path}")
        else:
            logging.warning(f"⚠️ Checkpoint saved but storage limit warning")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint and resume training state.

        Restores model weights, optimizer state, scheduler state, AMP
        scaler state, global step, and best loss.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            The epoch number from the loaded checkpoint.
        """
        logging.info(f"📂 Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        logging.info(f"✅ Checkpoint loaded - Global Step: {self.global_step}, Best Loss: {self.best_loss:.4f}")
        
        return checkpoint['epoch']

    def save_pretrained_artifacts(self, output_dir: str, tokenizer: Any = None):
        """Save HuggingFace-compatible model and tokenizer artifacts.

        Args:
            output_dir: Directory to save artifacts in.
            tokenizer: Optional tokenizer with ``save_pretrained`` support.

        Raises:
            RuntimeError: If the model does not support ``save_pretrained``.
        """
        if not hasattr(self.model, "save_pretrained"):
            raise RuntimeError("Current model does not support save_pretrained")

        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        logging.info("💾 Saved model with save_pretrained: %s", output_dir)

        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(output_dir)
            logging.info("💾 Saved tokenizer with save_pretrained: %s", output_dir)
