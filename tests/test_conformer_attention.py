"""Unit tests for the Conformer mixer (arXiv:2005.08100).

Covers the macaron FFN₁ + MHSA + convolution-module block, its causal
padding in decoder mode, the PE and fixed-softmax dispatch, BitNet-aware
projections and a manual-formula parity check of the convolution module.
"""

import unittest
from importlib.util import find_spec
from unittest import mock

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F

    from src.model.attention.common import BitLinear
    from src.model.attention.conformer import ConformerAttention, ConformerConvModule
    from src.model.config import FrankensteinModelConfig
    from src.model.embeddings import RoPE


def _cfg(**overrides):
    base = dict(
        vocab_size=64,
        hidden_size=48,
        num_layers=1,
        num_loops=1,
        num_heads=6,
        num_kv_heads=2,
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
        positional_encoding="rope",
        mode="encoder",
    )
    base.update(overrides)
    return FrankensteinModelConfig(**base)


BSZ, SEQ, DIM = 2, 8, 48


def _x():
    return torch.randn(BSZ, SEQ, DIM)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ConformerAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = ConformerAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = ConformerAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_decoder_output_shape(self):
        attn = ConformerAttention(_cfg(mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_causality(self):
        """Future tokens must not change past outputs in decoder mode."""
        attn = ConformerAttention(
            _cfg(mode="decoder", conformer_conv_norm="layer")
        )
        attn.eval()
        x = _x()
        x_future = x.clone()
        x_future[:, 5:] = torch.randn_like(x_future[:, 5:]) * 3.0
        with torch.no_grad():
            y = attn(x)
            y_future = attn(x_future)
        torch.testing.assert_close(y[:, :5], y_future[:, :5])

    def test_encoder_is_bidirectional(self):
        """Encoder mode keeps the convolution symmetric (no causal trim)."""
        attn = ConformerAttention(_cfg(conformer_conv_norm="layer"))
        attn.eval()
        x = _x()
        x_future = x.clone()
        x_future[:, 5:] = torch.randn_like(x_future[:, 5:]) * 3.0
        with torch.no_grad():
            y = attn(x)
            y_future = attn(x_future)
        self.assertFalse(torch.allclose(y, y_future))

    def test_macaron_on_off(self):
        on = ConformerAttention(_cfg(conformer_macaron=True))
        off = ConformerAttention(_cfg(conformer_macaron=False))
        self.assertTrue(on.macaron)
        self.assertFalse(off.macaron)

        x = _x()
        self.assertEqual(on(x).shape, (BSZ, SEQ, DIM))
        self.assertEqual(off(x).shape, (BSZ, SEQ, DIM))

        on.zero_grad()
        on(x).sum().backward()
        self.assertIsNotNone(on.ff_in.weight.grad)

        off.zero_grad()
        off(x).sum().backward()
        self.assertIsNone(off.ff_in.weight.grad)

    def test_pe_on_off(self):
        rope = RoPE(head_dim=DIM // 6)
        on = ConformerAttention(_cfg(), pos_encoder=rope)
        off = ConformerAttention(
            _cfg(conformer_attn_use_pe=False), pos_encoder=rope
        )
        off.load_state_dict(on.state_dict())
        on.eval()
        off.eval()
        self.assertTrue(on.use_pe)
        self.assertFalse(off.use_pe)
        x = _x()
        with torch.no_grad():
            y_on = on(x)
            y_off = off(x)
        self.assertEqual(y_on.shape, y_off.shape)
        self.assertFalse(torch.allclose(y_on, y_off))

    def test_conv_norm_variants(self):
        for conv_norm in ("batch", "group", "layer"):
            attn = ConformerAttention(_cfg(conformer_conv_norm=conv_norm))
            self.assertEqual(attn.conv.conv_norm, conv_norm)
            self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_bitnet_projections(self):
        attn = ConformerAttention(_cfg(use_bitnet=True))
        self.assertIsInstance(attn.q_proj, BitLinear)
        self.assertIsInstance(attn.k_proj, BitLinear)
        self.assertIsInstance(attn.v_proj, BitLinear)
        self.assertIsInstance(attn.out_proj, BitLinear)
        self.assertIsInstance(attn.ff_in, BitLinear)
        self.assertIsInstance(attn.ff_out, BitLinear)
        # Convolutions stay full-precision by design.
        self.assertNotIsInstance(attn.conv.pointwise_in, BitLinear)
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_fixed_softmax_attribute_dispatch(self):
        import src.model.attention.conformer as conformer_module

        attn = ConformerAttention(_cfg())
        attn.eval()
        self.assertFalse(attn.use_fixed_softmax)
        x = _x()
        with torch.no_grad():
            base = attn(x)

        calls = []
        original = conformer_module.module_softmax

        def spy(module, logits, dim=-1):
            calls.append(bool(getattr(module, "use_fixed_softmax", False)))
            return original(module, logits, dim=dim)

        with mock.patch.object(conformer_module, "module_softmax", spy):
            with torch.no_grad():
                fixed = attn(x)
        self.assertEqual(calls, [False])
        torch.testing.assert_close(base, fixed)

        attn.use_fixed_softmax = True
        with mock.patch.object(conformer_module, "module_softmax", spy):
            with torch.no_grad():
                fixed_on = attn(x)
        self.assertEqual(calls, [False, True])
        torch.testing.assert_close(base, fixed_on, atol=1e-6, rtol=1e-5)

    def test_dense_forward_alias(self):
        attn = ConformerAttention(_cfg())
        attn.eval()
        x = _x()
        with torch.no_grad():
            torch.testing.assert_close(attn(x), attn.dense_forward(x))

    def test_invalid_kv_heads_raises(self):
        with self.assertRaises(ValueError):
            ConformerAttention(_cfg(num_kv_heads=5))  # 5 does not divide 6


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ConformerConfigValidationTests(unittest.TestCase):
    def test_invalid_kernel_size_raises(self):
        with self.assertRaises(ValueError):
            _cfg(conformer_kernel_size=0)

    def test_invalid_conv_norm_raises(self):
        with self.assertRaises(ValueError):
            _cfg(conformer_conv_norm="instance")

    def test_conv_module_rejects_bad_kernel(self):
        with self.assertRaises(ValueError):
            ConformerConvModule(16, kernel_size=0)

    def test_conv_module_rejects_bad_norm(self):
        with self.assertRaises(ValueError):
            ConformerConvModule(16, conv_norm="instance")


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ConformerConvModuleTests(unittest.TestCase):
    def _reference(self, module, x):
        """Explicit reference computation of the Conformer conv module."""
        y = x.transpose(1, 2)
        y = module.pointwise_in(y)
        a, b = y.chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        k = module.kernel_size
        if module.causal:
            y = F.pad(y, (k - 1, 0))
        else:
            left = k // 2
            right = k - 1 - left
            if left or right:
                y = F.pad(y, (left, right))
        y = module.depthwise(y)
        if module.conv_norm == "layer":
            y = module.norm(y.transpose(1, 2)).transpose(1, 2)
        else:
            y = module.norm(y)
        y = F.silu(y)
        y = module.pointwise_out(y)
        y = module.dropout(y)
        return y.transpose(1, 2)

    def test_manual_formula_parity_encoder(self):
        torch.manual_seed(0)
        module = ConformerConvModule(
            hidden_size=16, kernel_size=5, conv_norm="batch", causal=False
        )
        module.eval()
        x = torch.randn(2, 7, 16)
        with torch.no_grad():
            out = module(x)
            ref = self._reference(module, x)
        torch.testing.assert_close(out, ref)

    def test_manual_formula_parity_decoder_causal(self):
        torch.manual_seed(0)
        module = ConformerConvModule(
            hidden_size=16, kernel_size=4, conv_norm="group", causal=True
        )
        module.eval()
        x = torch.randn(2, 7, 16)
        with torch.no_grad():
            out = module(x)
            ref = self._reference(module, x)
        torch.testing.assert_close(out, ref)

    def test_causal_conv_ignores_future(self):
        torch.manual_seed(0)
        module = ConformerConvModule(
            hidden_size=8, kernel_size=5, conv_norm="layer", causal=True
        )
        module.eval()
        x = torch.randn(1, 6, 8)
        x_future = x.clone()
        x_future[:, 4:] = torch.randn_like(x_future[:, 4:]) * 5.0
        with torch.no_grad():
            y = module(x)
            y_future = module(x_future)
        torch.testing.assert_close(y[:, :4], y_future[:, :4])
        self.assertEqual(y.shape, x.shape)

    def test_layer_norm_feature_axis_shape(self):
        module = ConformerConvModule(
            hidden_size=12, kernel_size=3, conv_norm="layer"
        )
        x = torch.randn(2, 9, 12)
        self.assertEqual(module(x).shape, x.shape)

    def test_odd_kernel_encoder_is_symmetric_padding(self):
        module = ConformerConvModule(
            hidden_size=8, kernel_size=5, conv_norm="layer", causal=False
        )
        x = torch.randn(1, 7, 8)
        y = module._pad(x.transpose(1, 2))
        # (B, C, S + k - 1) with equal left/right margins.
        self.assertEqual(y.shape[-1], 7 + 5 - 1)


if __name__ == "__main__":
    unittest.main()
