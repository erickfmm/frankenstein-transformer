"""DashAI telemetry sink for Frankenstein training telemetry.

Bridges Frankenstein's :class:`~src.training.trainer.TitanTrainer`
``metrics_callback`` payloads into DashAI's native metric store.

DashAI models persist metrics through ``BaseModel._save_metrics`` (a
``@final`` method that writes to the ``Metric`` SQL table keyed by
``(run_id, split, level, name, value, step)``). This adapter receives the
flat per-step dict emitted at ``trainer.py`` (34 fields mirroring the CSV
columns) and the per-epoch summary dict, and writes:

- **STEP level**: every scalar telemetry field of the step payload as its
  own metric row (loss, accuracy, learning_rate, grad_norm, all 11
  per-block grad-norm buckets, GPU temperature/power/utilization/memory,
  throughput, AMP scaler state, stability flags...). The step index is the
  trainer's ``global_step`` (one row per optimizer update).
- **EPOCH level**: the epoch summary fields (loss, accuracy, best_loss)
  plus DashAI metric components (accuracy, F1, ...) computed through
  ``calculate_metrics`` on the train/validation splits — the same behavior
  as the historical ``EpochMetricsHook``, unified here.

Non-numeric fields (``timestamp``, ``repair_action``) are not persisted —
the ``Metric`` table only holds floats; they remain in the trainer's CSV. Boolean fields are
coerced to ``0.0/1.0``.

The callback is invoked synchronously inside the training loop; it must be
fast. A failure to persist telemetry must never kill a training run, so all
DB errors are swallowed with a warning log.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

from DashAI.back.core.enums.metrics import LevelEnum, SplitEnum

log = logging.getLogger(__name__)

#: Per-step numeric telemetry fields written at LevelEnum.STEP (mirrors the
#: CSV schema in ``TitanTrainer._get_csv_columns`` minus timestamp/repair).
STEP_FIELDS = (
    "loss",
    "accuracy",
    "learning_rate",
    "grad_norm",
    "scaler_scale",
    "gpu_memory_gb",
    "gpu_cached_gb",
    "has_nan",
    "has_inf",
    "has_zero",
    "gpu_temp_c",
    "gpu_power_w",
    "gpu_util_pct",
    "gpu_mem_used_mib",
    "grad_norm_embeddings",
    "grad_norm_attention",
    "grad_norm_ffn",
    "grad_norm_experts",
    "grad_norm_router",
    "grad_norm_ode",
    "grad_norm_retnet",
    "grad_norm_mamba",
    "grad_norm_norms",
    "grad_norm_head",
    "grad_norm_other",
    "step_time_ms",
    "tokens_per_sec",
    "clip_ratio",
    "effective_batch_size",
)

#: Epoch-summary numeric fields written at LevelEnum.EPOCH.
EPOCH_FIELDS = ("loss", "accuracy", "best_loss", "global_step")


def _coerce(value: Any) -> Optional[float]:
    """Coerce a telemetry field to a finite float, or ``None``.

    Booleans become ``1.0/0.0``; NaN/Inf and non-numeric values are dropped
    (DashAI's ``calculate_metrics`` skips non-finite scores as well).
    """
    try:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


class DashAITelemetryCallback:
    """Stream Frankenstein training telemetry into the DashAI metric store.

    Instantiated by a DashAI model component (which carries ``run_id`` and
    the split datasets) and passed as the ``metrics_callback`` argument of
    :func:`src.engine.train_from_config`.

    Parameters
    ----------
    model_instance : Any
        The DashAI ``BaseModel`` instance (needs ``run_id`` and the
        ``_save_metrics``/``calculate_metrics`` API injected by
        ``ModelFactory``).
    x_train, y_train : Any
        Training split (DashAIDataset) for epoch-level scored metrics.
    x_val, y_val : Any, optional
        Validation split. When ``None``, validation scored metrics are
        skipped.
    split : SplitEnum
        Split tag for step/epoch telemetry rows (default ``TRAIN``).
    log_every_n_steps : int
        Write step rows every N optimizer updates (default ``1`` — every
        step, per DashAI-MLOps full-resolution requirement).
    """

    def __init__(
        self,
        model_instance: Any,
        x_train: Any = None,
        y_train: Any = None,
        x_val: Any = None,
        y_val: Any = None,
        *,
        split: SplitEnum = SplitEnum.TRAIN,
        log_every_n_steps: int = 1,
    ) -> None:
        self.model_instance = model_instance
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.split = split
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self._last_epoch = -1

    # -- metrics_callback contract -------------------------------------------

    def __call__(self, data: Dict[str, Any]) -> None:
        """Handle one step or epoch payload from TitanTrainer.

        Parameters
        ----------
        data : dict
            ``{"level": "step", ...}`` flat telemetry (see
            ``TitanTrainer._log_step_to_csv``) or ``{"level": "epoch", ...}``
            summary.
        """
        level = str(data.get("level", "step")).lower()
        try:
            if level == "epoch":
                self._on_epoch(data)
            else:
                self._on_step(data)
        except Exception as exc:  # noqa: BLE001 — telemetry must never kill training
            log.warning(
                "DashAI telemetry write failed (%s level): %s", level, exc
            )

    # -- internals ------------------------------------------------------------

    def _on_step(self, data: Dict[str, Any]) -> None:
        global_step = int(data.get("global_step", 0) or 0)
        if global_step % self.log_every_n_steps != 0:
            return
        results: Dict[str, float] = {}
        for field in STEP_FIELDS:
            coerced = _coerce(data.get(field))
            if coerced is not None:
                results[field] = coerced
        if not results:
            return
        self.model_instance._save_metrics(
            split=self.split,
            level=LevelEnum.STEP,
            results=results,
            log_index=global_step,
        )

    def _on_epoch(self, data: Dict[str, Any]) -> None:
        epoch = int(data.get("epoch", 0) or 0)
        if epoch <= self._last_epoch:
            return
        self._last_epoch = epoch

        results: Dict[str, float] = {}
        for field in EPOCH_FIELDS:
            coerced = _coerce(data.get(field))
            if coerced is not None:
                results[field] = coerced
        if results:
            # Epochs are 1-based in the store: DashAI's step counter starts
            # at 0 and reindexes log_index <= current_max, so a 0-based
            # epoch would collide with the initial counter state.
            self.model_instance._save_metrics(
                split=self.split,
                level=LevelEnum.EPOCH,
                results=results,
                log_index=epoch + 1,
            )

        # DashAI metric components (accuracy/F1/...) computed by predict+score.
        self._score_split(
            SplitEnum.TRAIN, self.x_train, self.y_train, epoch
        )
        if self.x_val is not None and self.y_val is not None:
            self._score_split(SplitEnum.VALIDATION, self.x_val, self.y_val, epoch)

    def _score_split(
        self, split: SplitEnum, x_data: Any, y_data: Any, epoch: int
    ) -> None:
        if x_data is None or y_data is None:
            return
        try:
            self.model_instance.calculate_metrics(
                split=split,
                level=LevelEnum.EPOCH,
                x_data=x_data,
                y_data=y_data,
                log_index=epoch + 1,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "calculate_metrics failed at epoch %s (%s): %s",
                epoch, split.value, exc,
            )