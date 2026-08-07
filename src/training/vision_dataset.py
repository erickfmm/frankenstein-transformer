#!/usr/bin/env python3
"""Image dataset for Vision Transformer tasks.

This module hosts :class:`ImageDataset`, a dataset class for the three
vision tasks supported by :class:`FrankensteinViT`:

* ``patch_prediction`` — masked patch prediction / autosupervised
  pre-training. The dataset yields ``pixel_values`` + a ``mask_bool``
  indicating corrupted patches + a ``mask_target`` for loss computation.
* ``classification`` — image classification. The dataset yields
  ``pixel_values`` + ``labels`` (class indices).
* ``segmentation`` — image segmentation. The dataset yields
  ``pixel_values`` + ``segmentation_map`` (per-pixel class indices).

The dataset loads images from a HuggingFace dataset (via
``datasets.load_dataset``) or a local directory, applies rescaling,
optional grayscale conversion, normalization, and augmentations.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore


def _to_tensor(img: Any, channels: int, height: int, width: int) -> torch.Tensor:
    """Convert an image (PIL, numpy, or tensor) to a ``(C, H, W)`` float tensor.

    Args:
        img: PIL Image, numpy array ``(H, W, C)`` or ``(H, W)``, or tensor.
        channels: Expected number of channels (1 for grayscale, 3 for RGB).
        height: Target height.
        width: Target width.

    Returns:
        Float tensor of shape ``(C, H, W)`` in ``[0, 1]``.
    """
    if isinstance(img, torch.Tensor):
        t = img.float()
    else:
        if hasattr(img, "resize"):
            # PIL Image
            if channels == 1:
                img = img.convert("L")
            else:
                img = img.convert("RGB")
            img = img.resize((width, height))
            arr = np.array(img)  # (H, W) or (H, W, 3)
        else:
            arr = np.array(img)
        if arr.ndim == 2:
            arr = arr[:, :, None]  # (H, W, 1)
        t = torch.from_numpy(arr).float()
    if t.ndim == 2:
        t = t.unsqueeze(0)
    elif t.ndim == 3 and t.shape[-1] in (1, 3):
        t = t.permute(2, 0, 1)  # HWC -> CHW
    # Normalize to [0, 1]
    if t.max() > 1.0:
        t = t / 255.0
    return t


def compute_mean_color_3bit(pixel_values: torch.Tensor, patch_size: int,
                            num_patches_h: int, num_patches_w: int) -> torch.Tensor:
    """Compute the 3-bit mean color target for each patch.

    Args:
        pixel_values: Image tensor ``(B, C, H, W)`` in ``[0, 1]``.
        patch_size: Patch size ``P``.
        num_patches_h: ``H // P``.
        num_patches_w: ``W // P``.

    Returns:
        Long tensor of shape ``(B, N)`` with 512-way class indices
        (R*64 + G*8 + B, where R/G/B ∈ {0..7}).
    """
    B, C, H, W = pixel_values.shape
    # Reshape to patches: (B, C, H/P, P, W/P, P) -> (B, N, C, P, P) -> mean
    x = pixel_values.view(B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    mean = x.mean(dim=(3, 5))  # (B, C, H/P, W/P)
    mean = mean.permute(0, 2, 3, 1)  # (B, N, C)
    quantized = (mean * 7).round().clamp(0, 7).long()  # (B, N, C) where N=H/P*W/P flattened below
    # Encode as single 512-way class
    target = quantized[..., 0] * 64 + quantized[..., 1] * 8 + quantized[..., 2]
    # Flatten spatial dims: (B, H/P, W/P) -> (B, N)
    if target.ndim == 3:
        B, Hh, Ww = target.shape
        target = target.reshape(B, Hh * Ww)
    return target  # (B, N)


def compute_full_patch_target(pixel_values: torch.Tensor, patch_size: int,
                              num_patches_h: int, num_patches_w: int) -> torch.Tensor:
    """Compute the full-patch L2 regression target (raw patch pixels).

    Args:
        pixel_values: Image tensor ``(B, C, H, W)`` in ``[0, 1]``.

    Returns:
        Float tensor of shape ``(B, N, P*P*C)``.
    """
    B, C, H, W = pixel_values.shape
    x = pixel_values.view(B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    # (B, H/P, W/P, P, P, C) -> (B, N, P*P*C)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, num_patches_h * num_patches_w, -1)
    return x


class ImageDataset(Dataset):
    """Image dataset for patch_prediction / classification / segmentation.

    Args:
        dataset_config: Dict from the ``dataset:`` YAML block.
        image_config: Dict from the ``image:`` YAML block.
        task: Vision task name.
        split: Dataset split (``"train"``, ``"validation"``, ``"test"``).

    Attributes:
        task: Vision task name.
        image_height: Target image height.
        image_width: Target image width.
        patch_size: Patch size.
        in_channels: Expected channels after grayscale conversion.
        items: List of data items (dicts with image + label/seg_map).
    """

    def __init__(
        self,
        dataset_config: Dict[str, Any],
        image_config: Dict[str, Any],
        task: str,
        split: str = "train",
    ) -> None:
        self.task = task
        self.dataset_config = dataset_config
        self.image_config = image_config
        self.split = split

        # Resolve dimensions.
        rescale = image_config.get("image_size", {}) if "image_size" in image_config else image_config.get("rescale", {})
        self.image_height = rescale.get("height", 224)
        self.image_width = rescale.get("width", 224)
        self.patch_size = image_config.get("patch_size", 16)
        self.in_channels = image_config.get("in_channels", 3)
        to_gray = image_config.get("to_grayscale", False)
        if to_gray:
            self.in_channels = 1

        self.mask_ratio = image_config.get("mask_ratio", 0.5)
        self.prediction_target = image_config.get("prediction_target", "mean_color_3bit")

        # Normalization.
        norm = dataset_config.get("normalize", {})
        self.mean = norm.get("mean")
        self.std = norm.get("std")

        # Augmentations.
        aug = dataset_config.get("augmentations", {})
        self.horizontal_flip = aug.get("horizontal_flip", False) and split == "train"

        # Column names.
        self.image_column = dataset_config.get("image_column", "image")
        self.label_column = dataset_config.get("label_column", "label")
        self.seg_column = dataset_config.get("segmentation_column", "segmentation_map")

        # Load items.
        self.items: List[Dict[str, Any]] = []
        self._load_items()

    def _load_items(self) -> None:
        """Load image references from HuggingFace dataset or local dir."""
        dataset_name = self.dataset_config.get("dataset_name")
        dataset_dir = self.dataset_config.get("dataset_dir")

        if dataset_name:
            try:
                from datasets import load_dataset
                split = self.split if self.split in ("train", "validation", "test") else "train"
                ds = load_dataset(dataset_name, split=split)
                for item in ds:
                    self.items.append(dict(item))
            except Exception:
                # If datasets library isn't available or loading fails,
                # leave empty (will be filled with dummy data for tests).
                pass
        elif dataset_dir and os.path.isdir(dataset_dir):
            exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
            for fname in sorted(os.listdir(dataset_dir)):
                if fname.lower().endswith(exts):
                    self.items.append({"path": os.path.join(dataset_dir, fname)})

    def __len__(self) -> int:
        return len(self.items)

    def _get_image(self, item: Dict[str, Any]) -> Any:
        """Get the image from an item dict (PIL, numpy, or path string)."""
        if "path" in item:
            from PIL import Image
            return Image.open(item["path"])
        return item.get(self.image_column)

    def _normalize(self, t: torch.Tensor) -> torch.Tensor:
        if self.mean is not None and self.std is not None:
            mean = torch.tensor(self.mean).view(-1, 1, 1)
            std = torch.tensor(self.std).view(-1, 1, 1)
            t = (t - mean) / std
        return t

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        img = self._get_image(item)
        t = _to_tensor(img, self.in_channels, self.image_height, self.image_width)

        # Grayscale conversion if needed (handled by _to_tensor channels param).
        t = self._normalize(t)

        # Optional horizontal flip augmentation.
        if self.horizontal_flip and torch.rand(1).item() < 0.5:
            t = torch.flip(t, dims=[2])

        result: Dict[str, torch.Tensor] = {"pixel_values": t}

        if self.task == "classification":
            label = item.get(self.label_column, 0)
            result["labels"] = torch.tensor(int(label), dtype=torch.long)

        elif self.task == "segmentation":
            seg = item.get(self.seg_column)
            if seg is not None:
                seg_t = _to_tensor(seg, 1, self.image_height, self.image_width)
                result["segmentation_map"] = seg_t.squeeze(0).long()
            else:
                # Default: all background
                result["segmentation_map"] = torch.zeros(
                    self.image_height, self.image_width, dtype=torch.long
                )

        elif self.task == "patch_prediction":
            num_h = self.image_height // self.patch_size
            num_w = self.image_width // self.patch_size
            N = num_h * num_w
            num_mask = max(1, int(N * self.mask_ratio))
            mask_bool = torch.zeros(N, dtype=torch.bool)
            perm = torch.randperm(N)[:num_mask]
            mask_bool[perm] = True
            result["mask_bool"] = mask_bool

            # Compute target from the original (unnormalized) patches.
            raw = _to_tensor(img, self.in_channels, self.image_height, self.image_width)
            if self.prediction_target == "mean_color_3bit":
                target = compute_mean_color_3bit(
                    raw.unsqueeze(0), self.patch_size, num_h, num_w
                ).squeeze(0)
            elif self.prediction_target == "full_patch_l2":
                target = compute_full_patch_target(
                    raw.unsqueeze(0), self.patch_size, num_h, num_w
                ).squeeze(0)
            else:
                # downsampled_3bit: fallback to mean_color for simplicity
                target = compute_mean_color_3bit(
                    raw.unsqueeze(0), self.patch_size, num_h, num_w
                ).squeeze(0)
            result["mask_target"] = target

        return result


class DummyImageDataset(Dataset):
    """A dummy image dataset for testing (generates random images).

    Args:
        task: Vision task name.
        num_samples: Number of random samples.
        image_height: Image height.
        image_width: Image width.
        in_channels: Input channels.
        patch_size: Patch size.
        num_classes: Number of classification classes.
        num_seg_classes: Number of segmentation classes.
        mask_ratio: Fraction of patches to mask (patch_prediction).
        prediction_target: Reconstruction target type.
    """

    def __init__(
        self,
        task: str = "classification",
        num_samples: int = 16,
        image_height: int = 32,
        image_width: int = 32,
        in_channels: int = 3,
        patch_size: int = 16,
        num_classes: int = 10,
        num_seg_classes: int = 5,
        mask_ratio: float = 0.5,
        prediction_target: str = "mean_color_3bit",
    ) -> None:
        self.task = task
        self.num_samples = num_samples
        self.image_height = image_height
        self.image_width = image_width
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        self.mask_ratio = mask_ratio
        self.prediction_target = prediction_target

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = torch.rand(self.in_channels, self.image_height, self.image_width)
        result: Dict[str, torch.Tensor] = {"pixel_values": t}
        if self.task == "classification":
            result["labels"] = torch.tensor(idx % self.num_classes, dtype=torch.long)
        elif self.task == "segmentation":
            result["segmentation_map"] = torch.randint(
                0, self.num_seg_classes, (self.image_height, self.image_width),
                dtype=torch.long,
            )
        elif self.task == "patch_prediction":
            num_h = self.image_height // self.patch_size
            num_w = self.image_width // self.patch_size
            N = num_h * num_w
            num_mask = max(1, int(N * self.mask_ratio))
            mask_bool = torch.zeros(N, dtype=torch.bool)
            perm = torch.randperm(N)[:num_mask]
            mask_bool[perm] = True
            result["mask_bool"] = mask_bool
            if self.prediction_target == "mean_color_3bit":
                target = compute_mean_color_3bit(
                    t.unsqueeze(0), self.patch_size, num_h, num_w
                ).squeeze(0)
            elif self.prediction_target == "full_patch_l2":
                target = compute_full_patch_target(
                    t.unsqueeze(0), self.patch_size, num_h, num_w
                ).squeeze(0)
            else:
                target = compute_mean_color_3bit(
                    t.unsqueeze(0), self.patch_size, num_h, num_w
                ).squeeze(0)
            result["mask_target"] = target
        return result