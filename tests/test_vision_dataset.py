#!/usr/bin/env python3
"""Tests for the vision dataset (ImageDataset + DummyImageDataset)."""

from __future__ import annotations

import pytest

try:
    import torch
    from src.training.vision_dataset import DummyImageDataset, compute_mean_color_3bit, compute_full_patch_target
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestDummyImageDataset:
    def test_classification(self):
        ds = DummyImageDataset(task="classification", num_samples=16, image_height=32, image_width=32, num_classes=10)
        assert len(ds) == 16
        item = ds[0]
        assert "pixel_values" in item
        assert item["pixel_values"].shape == (3, 32, 32)
        assert "labels" in item
        assert item["labels"].dtype == torch.long

    def test_segmentation(self):
        ds = DummyImageDataset(task="segmentation", num_samples=8, image_height=32, image_width=32, num_seg_classes=5)
        item = ds[0]
        assert item["pixel_values"].shape == (3, 32, 32)
        assert item["segmentation_map"].shape == (32, 32)
        assert item["segmentation_map"].dtype == torch.long

    def test_patch_prediction(self):
        ds = DummyImageDataset(task="patch_prediction", num_samples=8, image_height=32, image_width=32, patch_size=8, mask_ratio=0.5, prediction_target="mean_color_3bit")
        item = ds[0]
        assert item["pixel_values"].shape == (3, 32, 32)
        assert "mask_bool" in item
        assert "mask_target" in item
        N = (32 // 8) * (32 // 8)
        assert item["mask_bool"].shape == (N,)
        assert item["mask_bool"].dtype == torch.bool

    def test_full_patch_l2_target(self):
        ds = DummyImageDataset(task="patch_prediction", num_samples=4, image_height=32, image_width=32, patch_size=8, prediction_target="full_patch_l2")
        item = ds[0]
        assert item["mask_target"].shape[0] == (32 // 8) * (32 // 8)
        assert item["mask_target"].shape[1] == 8 * 8 * 3


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestComputeTargets:
    def test_mean_color_3bit(self):
        x = torch.rand(1, 3, 32, 32)
        target = compute_mean_color_3bit(x, patch_size=8, num_patches_h=4, num_patches_w=4)
        assert target.shape == (1, 16)
        assert target.dtype == torch.long
        assert target.min() >= 0
        assert target.max() < 512

    def test_full_patch_target(self):
        x = torch.rand(1, 3, 32, 32)
        target = compute_full_patch_target(x, patch_size=8, num_patches_h=4, num_patches_w=4)
        assert target.shape == (1, 16, 8 * 8 * 3)
        assert target.dtype == torch.float32