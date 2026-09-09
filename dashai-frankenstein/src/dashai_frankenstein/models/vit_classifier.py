"""FrankensteinViTClassifier — DashAI image-classification component.

Wraps :class:`FrankensteinViT` (which already exposes a classification head,
``classification_head = Linear(hidden_size, num_classes)``) and fine-tunes it
on a DashAI image dataset. The image and label columns are extracted from the
``DashAIDataset``; images are resized to the model's ``image_height``/``image_width``
and normalized with ImageNet statistics (mirroring the torchvision DashAI
classifiers).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
from sklearn.exceptions import NotFittedError

from DashAI.back.core.utils import MultilingualString
from DashAI.back.models.base_model import BaseModel

from dashai_frankenstein.config import FrankensteinClassifierSchema
from dashai_frankenstein.engine import (
    build_model_from_json,
    resolve_device,
    validate_training_json,
)
from dashai_frankenstein.models.base import resolve_json


class FrankensteinViTClassifier(BaseModel):
    """Frankenstein Vision Transformer for image classification.

    Builds a :class:`FrankensteinViT` backbone (all attention mixers / norms /
    activations available via the passthrough config) and fine-tunes its built-in
    classification head. Compatible with ``ImageClassificationTask``.

    The number of classes is derived from the DashAI dataset's categorical
    label column and injected into the model config before construction. The
    image size defaults to 224 (override via ``model.image_height`` /
    ``model.image_width`` in the config).
    """

    COMPATIBLE_COMPONENTS = ["ImageClassificationTask"]
    DISPLAY_NAME: str = MultilingualString(
        en="Frankenstein ViT",
        es="ViT Frankenstein",
        pt="ViT Frankenstein",
        de="Frankenstein ViT",
        zh="Frankenstein ViT",
    )
    DESCRIPTION: str = MultilingualString(
        en=(
            "Frankenstein Vision Transformer for image classification. Supports "
            "the full mixer/norm/activation catalog via the passthrough YAML."
        ),
        es=(
            "Vision Transformer Frankenstein para clasificación de imágenes. "
            "Catálogo completo de mixers/normas/activaciones vía YAML."
        ),
        pt="Vision Transformer Frankenstein para classificação de imagens.",
        de="Frankenstein Vision Transformer für Bildklassifikation.",
        zh="用于图像分类的 Frankenstein 视觉 Transformer。",
    )
    COLOR: str = "#1565C0"
    ICON: str = "Image"
    SCHEMA = FrankensteinClassifierSchema

    def __init__(self, **kwargs) -> None:
        kwargs = self.validate_and_transform(kwargs)
        self.frankenstein_json = kwargs.get("frankenstein_json", "")

        self.num_classes = None
        self.fitted = False
        self.idx_to_label: dict = {}
        self.label_to_idx: dict = {}
        self._frank_model = None
        self._loaded_config = None
        self._device = "cpu"
        self._batch_size = 32
        self._image_size = 224
        self.x_data = None
        self.y_data = None

    def train(self, x_train, y_train, x_validation=None, y_validation=None):
        """Fine-tune the ViT classification head on the DashAI image dataset.

        Delegates to :func:`src.engine.train_from_config` (``task=classification``)
        with a pre-built dict-batch DataLoader; the engine's ``TitanTrainer``
        streams all step/epoch telemetry into DashAI via
        :class:`~dashai_frankenstein.adapters.telemetry.DashAITelemetryCallback`.
        """
        from DashAI.back.core.enums.metrics import SplitEnum

        from dashai_frankenstein.adapters.dataset import image_dataloader, image_dataloader_dict
        from dashai_frankenstein.adapters.telemetry import DashAITelemetryCallback
        from dashai_frankenstein.engine import train_from_config
        from dashai_frankenstein.models.base import _inject_engine_overrides, _ConcatLoader

        json_text = resolve_json(self)
        validate_training_json(json_text)

        # Resolve label mapping first (needed for model num_classes override).
        # image_dataloader returns (loader, label_to_idx, num_classes); we use
        # a throwaway loader on CPU just to extract the label mapping.
        _, label_to_idx, num_classes = image_dataloader(
            x_train, y_dataset=y_train, image_size=self._image_size,
            batch_size=1, device="cpu", shuffle=False,
        )
        self.label_to_idx = label_to_idx
        self.idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
        self.num_classes = int(num_classes)

        overrides = {
            "model": {
                "num_classes": self.num_classes,
                "image_height": self._image_size,
                "image_width": self._image_size,
            }
        }
        model, loaded, _ = build_model_from_json(
            json_text, model_class_override="frankenstein_vit", overrides=overrides,
        )

        runtime = getattr(loaded, "training_runtime", {}) or {}
        device = resolve_device(runtime.get("device", "auto"))
        batch_size = int(runtime.get("batch_size", 32) or 32)
        num_epochs = int(runtime.get("num_epochs", 3) or 3)

        self._device = device
        self._batch_size = batch_size
        self._frank_model = model.to(device)
        self._loaded_config = loaded

        # Build the real dict-format loaders with the resolved device/batch_size.
        train_loader, _, _ = image_dataloader_dict(
            x_train, y_dataset=y_train, image_size=self._image_size,
            batch_size=batch_size, device=device, shuffle=True,
            label_to_idx=label_to_idx,
        )

        val_loader = None
        if x_validation is not None and y_validation is not None:
            val_loader, _, _ = image_dataloader_dict(
                x_validation, y_dataset=y_validation, image_size=self._image_size,
                batch_size=batch_size, device=device, shuffle=False,
                label_to_idx=label_to_idx,
            )
        engine_dataset = train_loader if val_loader is None else _ConcatLoader(train_loader, val_loader)

        engine_cfg = _inject_engine_overrides(
            json_text,
            task="classification",
            num_epochs=num_epochs,
            batch_size=batch_size,
            extra={"model": {
                "num_classes": self.num_classes,
                "image_height": self._image_size,
                "image_width": self._image_size,
            }},
        )

        callback = DashAITelemetryCallback(
            self, x_train, y_train, x_validation, y_validation,
            split=SplitEnum.TRAIN, log_every_n_steps=1,
        )

        result = train_from_config(
            engine_cfg,
            dataset=engine_dataset,
            device=device,
            supervisor="off",
            metrics_callback=callback,
        )
        if getattr(result, "model", None) is not None:
            self._frank_model = result.model.to(device)

        self.fitted = True
        self.x_data = {"train": x_train}
        self.y_data = {"train": y_train}
        if x_validation is not None:
            self.x_data["validation"] = x_validation
            self.y_data["validation"] = y_validation
        return self

    def predict(self, x_pred):
        """Return an ``(N, num_classes)`` softmax probability matrix."""
        if not self.fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet."
            )
        import torch

        from dashai_frankenstein.adapters.dataset import image_dataloader

        loader, _, _ = image_dataloader(
            x_pred, y_dataset=None, image_size=self._image_size,
            batch_size=int(getattr(self, "_batch_size", 32) or 32),
            device=self._device, shuffle=False,
        )
        model = self._frank_model
        model.eval()
        probs = []
        with torch.no_grad():
            for images in loader:
                logits = model(images.to(self._device), task="classification")
                probs.append(logits.softmax(dim=-1).detach().cpu().numpy())
        if not probs:
            return np.zeros((0, int(self.num_classes)))
        return np.concatenate(probs, axis=0)

    def save(self, filename: Union[str, Path]) -> None:
        from dashai_frankenstein.models.base import persistence_save

        persistence_save(self, str(filename))

    @classmethod
    def load(cls, filename: Union[str, Path]) -> "FrankensteinViTClassifier":
        from dashai_frankenstein.models.base import persistence_load

        return persistence_load(cls, str(filename))
