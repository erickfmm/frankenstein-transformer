"""FrankensteinViTSegmenter — DashAI image-segmentation component.

Wraps :class:`FrankensteinViT` with its segmentation head
(``seg_head = Linear(hidden_size, num_seg_classes)`` + ViTDet-style upsampler)
and trains it on a DashAI image dataset with mask targets. The forward returns
per-pixel class logits of shape ``(B, num_seg_classes, H, W)``; ``predict``
returns the argmax class map ``(B, H, W)``.
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


class FrankensteinViTSegmenter(BaseModel):
    """Frankenstein Vision Transformer for semantic image segmentation.

    Builds a :class:`FrankensteinViT` backbone and fine-tunes its segmentation
    head. Compatible with :class:`SegmentationTask`. The number of segmentation
    classes (``num_seg_classes``) is taken from the Frankenstein config.
    """

    COMPATIBLE_COMPONENTS = ["SegmentationTask"]
    DISPLAY_NAME: str = MultilingualString(
        en="Frankenstein ViT Segmenter",
        es="Segmentador ViT Frankenstein",
        pt="Segmentador ViT Frankenstein",
        de="Frankenstein ViT Segmentierer",
        zh="Frankenstein ViT 分割器",
    )
    DESCRIPTION: str = MultilingualString(
        en=(
            "Frankenstein Vision Transformer for semantic image segmentation "
            "(per-pixel class prediction) via the ViTDet-style upsampler head."
        ),
        es=(
            "Vision Transformer Frankenstein para segmentación semántica de "
            "imágenes (predicción de clases por píxel)."
        ),
        pt="Vision Transformer Frankenstein para segmentação semântica.",
        de="Frankenstein Vision Transformer für semantische Segmentierung.",
        zh="用于语义图像分割的 Frankenstein 视觉 Transformer。",
    )
    COLOR: str = "#00695C"
    ICON: str = "GridOn"
    SCHEMA = FrankensteinClassifierSchema

    def __init__(self, **kwargs) -> None:
        kwargs = self.validate_and_transform(kwargs)
        self.frankenstein_json = kwargs.get("frankenstein_json", "")
        self.use_dashai_dataset = bool(kwargs.get("use_dashai_dataset", True))

        self.num_seg_classes = None
        self.fitted = False
        self._frank_model = None
        self._loaded_config = None
        self._device = "cpu"
        self._batch_size = 8
        self._image_size = 224
        self.x_data = None
        self.y_data = None

    def train(self, x_train, y_train, x_validation=None, y_validation=None):
        """Fine-tune the ViT segmentation head on a DashAI image+mask dataset.

        Delegates to :func:`src.engine.train_from_config`
        (``task=segmentation``) with a pre-built dict-batch DataLoader; the
        engine's ``TitanTrainer`` streams all step/epoch telemetry into
        DashAI via
        :class:`~dashai_frankenstein.adapters.telemetry.DashAITelemetryCallback`.

        Masks are expected as an image column in ``y_train``; each mask is
        converted to a per-pixel class-index map (``(H, W)``) via its luminance
        quantized to ``num_seg_classes`` levels when no explicit palette exists.
        """
        from DashAI.back.core.enums.metrics import SplitEnum

        from dashai_frankenstein.adapters.dataset import segmentation_dataloader_dict
        from dashai_frankenstein.adapters.telemetry import DashAITelemetryCallback
        from dashai_frankenstein.engine import train_from_config
        from dashai_frankenstein.models.base import _inject_engine_overrides

        json_text = resolve_json(self)
        validate_training_json(json_text)

        img_size = self._image_size
        overrides = {
            "model": {
                "image_height": img_size,
                "image_width": img_size,
            }
        }
        model, loaded, _ = build_model_from_json(
            json_text, model_class_override="frankenstein_vit", overrides=overrides,
        )
        cfg = loaded.model_config
        self.num_seg_classes = int(getattr(cfg, "num_seg_classes", 2))

        runtime = getattr(loaded, "training_runtime", {}) or {}
        device = resolve_device(runtime.get("device", "auto"))
        batch_size = int(runtime.get("batch_size", 8) or 8)
        num_epochs = int(runtime.get("num_epochs", 3) or 3)

        self._device = device
        self._batch_size = batch_size
        self._frank_model = model.to(device)
        self._loaded_config = loaded

        train_loader = segmentation_dataloader_dict(
            x_train, image_size=img_size,
            batch_size=batch_size, device=device, shuffle=True,
            num_seg_classes=self.num_seg_classes,
        )

        engine_cfg = _inject_engine_overrides(
            json_text,
            task="segmentation",
            num_epochs=num_epochs,
            batch_size=batch_size,
            extra={"model": {
                "image_height": img_size,
                "image_width": img_size,
            }},
        )

        callback = DashAITelemetryCallback(
            self, x_train, y_train,
            split=SplitEnum.TRAIN, log_every_n_steps=1,
        )

        if self.use_dashai_dataset:
            result = train_from_config(
                engine_cfg,
                dataset=train_loader,
                device=device,
                supervisor="off",
                metrics_callback=callback,
            )
        else:
            # JSON-dataset mode: the engine resolves the corpus from the
            # config's ``vision_dataset`` block (or a dummy smoke dataset).
            result = train_from_config(
                engine_cfg,
                dataset=None,
                device=device,
                supervisor="off",
                metrics_callback=callback,
            )
        if getattr(result, "model", None) is not None:
            self._frank_model = result.model.to(device)

        self.fitted = True
        self.x_data = {"train": x_train}
        self.y_data = {"train": y_train}
        return self

    def predict(self, x_pred):
        """Return per-pixel class maps of shape ``(N, H, W)`` (argmax)."""
        if not self.fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet."
            )
        import torch

        from dashai_frankenstein.adapters.dataset import image_dataloader

        loader, _, _ = image_dataloader(
            x_pred, y_dataset=None, image_size=self._image_size,
            batch_size=int(getattr(self, "_batch_size", 8) or 8),
            device=self._device, shuffle=False,
        )
        model = self._frank_model
        model.eval()
        maps = []
        with torch.no_grad():
            for images in loader:
                logits = model(images.to(self._device), task="segmentation")
                maps.append(logits.argmax(dim=1).detach().cpu().numpy())
        if not maps:
            return np.zeros((0, self._image_size, self._image_size), dtype=np.int64)
        return np.concatenate(maps, axis=0)

    def save(self, filename: Union[str, Path]) -> None:
        from dashai_frankenstein.models.base import persistence_save

        persistence_save(self, str(filename))

    @classmethod
    def load(cls, filename: Union[str, Path]) -> "FrankensteinViTSegmenter":
        from dashai_frankenstein.models.base import persistence_load

        return persistence_load(cls, str(filename))
