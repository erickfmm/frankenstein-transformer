#!/usr/bin/env python3
"""Tests for vision task loss methods (patch_prediction, classification, segmentation)."""

from __future__ import annotations

import pytest

try:
    import torch
    from src.model.frankenstein_vit import FrankensteinViT
    from src.training.vision_dataset import DummyImageDataset
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _make_trainer(model, task):
    """Create a minimal TitanTrainer-like object for loss testing."""
    from src.training.trainer import TitanTrainer
    cfg = model.config
    return TitanTrainer(model, cfg, training_config=None, device="cpu", task=task)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestClassificationLoss:
    def test_loss_and_accuracy(self):
        cfg = FrankensteinViT.build_vit_config(
            hidden_size=64, num_layers=2, num_heads=4, patch_size=8,
            image_height=32, image_width=32, num_classes=10,
        )
        model = FrankensteinViT(cfg)
        trainer = _make_trainer(model, "classification")
        ds = DummyImageDataset(task="classification", num_samples=4, image_height=32, image_width=32, num_classes=10)
        batch = next(iter(DataLoader(ds, batch_size=4)))
        loss, acc = trainer.compute_classification_loss(batch)
        assert loss.dim() == 0
        assert loss.item() > 0
        assert 0.0 <= acc.item() <= 1.0
        loss.backward()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestSegmentationLoss:
    def test_loss_and_accuracy(self):
        cfg = FrankensteinViT.build_vit_config(
            hidden_size=64, num_layers=2, num_heads=4, patch_size=8,
            image_height=32, image_width=32, num_seg_classes=5,
        )
        model = FrankensteinViT(cfg)
        trainer = _make_trainer(model, "segmentation")
        ds = DummyImageDataset(task="segmentation", num_samples=4, image_height=32, image_width=32, num_seg_classes=5)
        batch = next(iter(DataLoader(ds, batch_size=4)))
        loss, acc = trainer.compute_segmentation_loss(batch)
        assert loss.dim() == 0
        assert loss.item() > 0
        assert 0.0 <= acc.item() <= 1.0
        loss.backward()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestPatchPredictionLoss:
    def test_mean_color_3bit_loss(self):
        cfg = FrankensteinViT.build_vit_config(
            hidden_size=64, num_layers=2, num_heads=4, patch_size=8,
            image_height=32, image_width=32, prediction_target="mean_color_3bit",
        )
        model = FrankensteinViT(cfg)
        trainer = _make_trainer(model, "patch_prediction")
        ds = DummyImageDataset(task="patch_prediction", num_samples=4, image_height=32, image_width=32, patch_size=8, prediction_target="mean_color_3bit")
        batch = next(iter(DataLoader(ds, batch_size=4)))
        loss, acc = trainer.compute_patch_prediction_loss(batch)
        assert loss.dim() == 0
        assert loss.item() > 0
        loss.backward()

    def test_full_patch_l2_loss(self):
        cfg = FrankensteinViT.build_vit_config(
            hidden_size=64, num_layers=2, num_heads=4, patch_size=8,
            image_height=32, image_width=32, prediction_target="full_patch_l2",
        )
        model = FrankensteinViT(cfg)
        trainer = _make_trainer(model, "patch_prediction")
        ds = DummyImageDataset(task="patch_prediction", num_samples=4, image_height=32, image_width=32, patch_size=8, prediction_target="full_patch_l2")
        batch = next(iter(DataLoader(ds, batch_size=4)))
        loss, acc = trainer.compute_patch_prediction_loss(batch)
        assert loss.dim() == 0
        assert loss.item() > 0
        loss.backward()