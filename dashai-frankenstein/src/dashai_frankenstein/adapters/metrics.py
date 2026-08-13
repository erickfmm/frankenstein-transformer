"""Metrics adapter: stream per-epoch metrics into DashAI's metric store.

DashAI models expose ``calculate_metrics(split, level, x_data, y_data, ...)``,
which calls ``self.predict(x_data)`` and writes scores to the database. This
module provides a tiny callback the training loop invokes at the end of each
epoch, mirroring the HuggingFace ``MetricsCallback`` pattern.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from DashAI.back.core.enums.metrics import LevelEnum, SplitEnum

log = logging.getLogger(__name__)


class EpochMetricsHook:
    """Invoke ``model_instance.calculate_metrics`` once per epoch.

    Parameters
    ----------
    model_instance : BaseModel
        The DashAI model component (carries ``run_id``, ``train_metrics`` ...).
    x_train, y_train : DashAIDataset
        Training split.
    x_val, y_val : DashAIDataset, optional
        Validation split. When ``None``, validation metrics are skipped.
    log_every_n_epochs : int
        Log frequency (epochs). ``1`` logs every epoch.
    """

    def __init__(
        self,
        model_instance: Any,
        x_train: Any,
        y_train: Any,
        x_val: Any = None,
        y_val: Any = None,
        *,
        log_every_n_epochs: int = 1,
    ) -> None:
        self.model_instance = model_instance
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.log_every_n_epochs = max(1, int(log_every_n_epochs))
        self._last_epoch = -1

    def __call__(self, epoch: int, **_: Any) -> None:
        if epoch <= self._last_epoch:
            return
        self._last_epoch = epoch
        if epoch % self.log_every_n_epochs != 0:
            return
        try:
            self.model_instance.calculate_metrics(
                split=SplitEnum.TRAIN,
                level=LevelEnum.EPOCH,
                x_data=self.x_train,
                y_data=self.y_train,
                log_index=epoch,
            )
            if self.x_val is not None and self.y_val is not None:
                self.model_instance.calculate_metrics(
                    split=SplitEnum.VALIDATION,
                    level=LevelEnum.EPOCH,
                    x_data=self.x_val,
                    y_data=self.y_val,
                    log_index=epoch,
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("calculate_metrics failed at epoch %s: %s", epoch, exc)
