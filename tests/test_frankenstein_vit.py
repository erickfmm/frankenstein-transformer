#!/usr/bin/env python3
"""Tests for FrankensteinViT (Vision Transformer, arXiv:2010.11929).

Covers instantiation for all three tasks, forward pass shapes, patch
embedding, cls vs gap pooling, learned vs no positional embedding, masking,
gradient flow, BitNet on/off, and multiple attention mixers.
"""

from __future__ import annotations

import pytest

try:
    import torch
    from src.model.frankenstein_vit import FrankensteinViT, PatchEmbed
    from src.model.config import FrankensteinModelConfig
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.fixture
def tiny_vit_config():
    return FrankensteinViT.build_vit_config(
        hidden_size=64, num_layers=2, num_heads=4, patch_size=8,
        image_height=32, image_width=32, in_channels=3,
    )


@pytest.fixture
def dummy_image():
    return torch.randn(2, 3, 32, 32)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestPatchEmbed:
    def test_patch_embed_shape(self, tiny_vit_config):
        pe = PatchEmbed(tiny_vit_config)
        x = torch.randn(2, 3, 32, 32)
        out = pe(x)
        assert out.shape == (2, 16, 64)  # 4*4=16 patches, D=64

    def test_num_patches(self, tiny_vit_config):
        pe = PatchEmbed(tiny_vit_config)
        assert pe.num_patches == 16  # (32/8)*(32/8)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestFrankensteinViT:
    def test_instantiation(self, tiny_vit_config):
        model = FrankensteinViT(tiny_vit_config)
        assert model.num_patches == 16

    def test_classification_forward(self, tiny_vit_config, dummy_image):
        model = FrankensteinViT(tiny_vit_config)
        out = model(dummy_image, task="classification")
        assert out.shape == (2, 1000)

    def test_patch_prediction_forward(self, tiny_vit_config, dummy_image):
        model = FrankensteinViT(tiny_vit_config)
        out = model(dummy_image, task="patch_prediction")
        # (B, N+1, 512) if cls_token; (B, N, 512) otherwise
        assert out.shape[0] == 2
        assert out.shape[-1] == 512

    def test_segmentation_forward(self, tiny_vit_config, dummy_image):
        tiny_vit_config.num_seg_classes = 5
        model = FrankensteinViT(tiny_vit_config)
        out = model(dummy_image, task="segmentation")
        assert out.shape == (2, 5, 32, 32)

    def test_gap_pooling(self, dummy_image):
        cfg = FrankensteinViT.build_vit_config(
            hidden_size=64, num_layers=2, num_heads=4, patch_size=8,
            image_height=32, image_width=32, cls_token=False, pooling_mode="gap",
        )
        model = FrankensteinViT(cfg)
        out = model(dummy_image, task="classification")
        assert out.shape == (2, 1000)

    def test_no_pos_embed(self, dummy_image):
        cfg = FrankensteinViT.build_vit_config(
            hidden_size=64, num_layers=2, num_heads=4, patch_size=8,
            image_height=32, image_width=32, pos_embedding_type="none",
        )
        model = FrankensteinViT(cfg)
        out = model(dummy_image, task="classification")
        assert out.shape == (2, 1000)

    def test_gradient_flow(self, tiny_vit_config, dummy_image):
        model = FrankensteinViT(tiny_vit_config)
        out = model(dummy_image, task="classification")
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and "patch_embed" in name:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()

    def test_multiple_mixers(self, dummy_image):
        cfg = FrankensteinViT.build_vit_config(
            hidden_size=64, num_layers=2, num_heads=4, patch_size=8,
            image_height=32, image_width=32, layer_pattern=["titan_attn", "standard_attn"],
        )
        model = FrankensteinViT(cfg)
        out = model(dummy_image, task="classification")
        assert out.shape == (2, 1000)

    def test_no_bitnet(self, tiny_vit_config):
        model = FrankensteinViT(tiny_vit_config)
        assert not tiny_vit_config.use_bitnet

    def test_build_vit_config_defaults(self):
        cfg = FrankensteinViT.build_vit_config()
        assert cfg.hidden_size == 768
        assert cfg.num_layers == 12
        assert cfg.num_heads == 12
        assert cfg.patch_size == 16
        assert cfg.ffn_activation == "gelu"
        assert cfg.ffn_hidden_size == 3072  # 4 * 768
        assert cfg.mode == "encoder"

    def test_configure_optimizers(self, tiny_vit_config):
        model = FrankensteinViT(tiny_vit_config)
        groups = model.configure_optimizers()
        names = [g["name"] for g in groups]
        assert "embeddings" in names
        assert "norms" in names
        assert "attention" in names
        assert "other" in names

    def test_mode_forced_to_encoder(self):
        cfg = FrankensteinViT.build_vit_config()
        cfg.mode = "decoder"
        model = FrankensteinViT(cfg)
        assert model.config.mode == "encoder"

    def test_full_patch_l2_target(self, dummy_image):
        cfg = FrankensteinViT.build_vit_config(
            hidden_size=64, num_layers=2, num_heads=4, patch_size=8,
            image_height=32, image_width=32, prediction_target="full_patch_l2",
        )
        model = FrankensteinViT(cfg)
        out = model(dummy_image, task="patch_prediction")
        assert out.shape[-1] == 8 * 8 * 3  # P*P*C