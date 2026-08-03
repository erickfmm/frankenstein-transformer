"""Unit tests for gated attention modules."""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from src.model.tormented_bert_frankestein import UltraConfig
    from src.model.attention.gated import (
        GatedLinearAttention,
        DeltaNetAttention,
        GatedDeltaNetAttention,
        RetNetAttention,
        HGRN2Attention,
        ForgettingAttention,
        GatedSoftmaxAttention,
        KDAAttention,
    )


def _cfg(**overrides):
    base = dict(
        vocab_size=64,
        hidden_size=48,
        num_layers=1,
        num_loops=1,
        num_heads=6,
        retention_heads=6,
        num_experts=2,
        top_k_experts=1,
        dropout=0.0,
        norm_type="layer_norm",
        use_bitnet=False,
        layer_pattern=["standard_attn"],
        use_moe=False,
        ode_solver="rk4",
        ode_steps=1,
        ffn_hidden_size=96,
        ffn_activation="gelu",
        use_hope=True,
        mode="encoder",
    )
    base.update(overrides)
    return UltraConfig(**base)


BSZ, SEQ, DIM = 2, 8, 48


def _x():
    return torch.randn(BSZ, SEQ, DIM)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GatedLinearAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = GatedLinearAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = GatedLinearAttention(_cfg(mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = GatedLinearAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_hidden_raises(self):
        with self.assertRaises(ValueError):
            GatedLinearAttention(_cfg(hidden_size=50, num_heads=6))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class DeltaNetAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = DeltaNetAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = DeltaNetAttention(_cfg(mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = DeltaNetAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GatedDeltaNetAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = GatedDeltaNetAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = GatedDeltaNetAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class RetNetAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = RetNetAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_logical_layer_idx_accepted(self):
        attn = RetNetAttention(_cfg())
        y = attn(_x(), logical_layer_idx=2)
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = RetNetAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class HGRN2AttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = HGRN2Attention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = HGRN2Attention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ForgettingAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = ForgettingAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = ForgettingAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GatedSoftmaxAttentionTests(unittest.TestCase):
    def test_encoder_output_shape(self):
        attn = GatedSoftmaxAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_output_shape(self):
        attn = GatedSoftmaxAttention(_cfg(mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = GatedSoftmaxAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class KDAAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = KDAAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = KDAAttention(_cfg(mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = KDAAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_hidden_raises(self):
        with self.assertRaises(ValueError):
            KDAAttention(_cfg(hidden_size=50, num_heads=6))


if __name__ == "__main__":
    unittest.main()
