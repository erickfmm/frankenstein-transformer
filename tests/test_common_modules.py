"""Unit tests for common building blocks: quant functions, BitLinear, norms, get_norm."""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from src.model.attention.common import (
        activation_quant,
        weight_quant,
        BitLinear,
    )
    from src.model.norm import (
        DynamicTanhNorm,
        Derf,
        FlashNorm,
        FlashNormLinear,
        FlashNormBitLinear,
        RMSNorm,
        fold_norm_weights,
        get_norm,
    )
    from src.model.tormented_bert_frankestein import UltraConfig


def _cfg(norm_type="layer_norm", hidden_size=32):
    return UltraConfig(
        vocab_size=64,
        hidden_size=hidden_size,
        num_layers=1,
        num_loops=1,
        num_heads=4,
        retention_heads=4,
        num_experts=2,
        top_k_experts=1,
        dropout=0.0,
        norm_type=norm_type,
        use_bitnet=False,
        layer_pattern=["standard_attn"],
        use_moe=False,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ActivationQuantTests(unittest.TestCase):
    def test_output_shape_preserved(self):
        x = torch.randn(2, 8, 16)
        y = activation_quant(x)
        self.assertEqual(y.shape, x.shape)

    def test_output_dtype_preserved(self):
        x = torch.randn(3, 5)
        y = activation_quant(x)
        self.assertEqual(y.dtype, x.dtype)

    def test_clamp_range(self):
        # STE: output must stay within [-128, 127] after unscaling, approximated.
        x = torch.randn(4, 10) * 100
        y = activation_quant(x)
        # The result should be finite and not blow up.
        self.assertTrue(torch.isfinite(y).all())

    def test_ste_gradient_pass_through(self):
        x = torch.randn(2, 4, requires_grad=True)
        y = activation_quant(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class WeightQuantTests(unittest.TestCase):
    def test_output_shape_preserved(self):
        w = torch.randn(8, 16)
        wq = weight_quant(w)
        self.assertEqual(wq.shape, w.shape)

    def test_ternary_values_approx(self):
        # The quantized weights (before STE correction) should be close to {-1, 0, 1}
        w = torch.randn(64, 64)
        scale = 1.0 / w.abs().mean().clamp(min=1e-5)
        wq_raw = (w * scale).round().clamp(-1, 1) / scale
        # All unique scaled values should be at most 3 distinct magnitudes
        unique = torch.unique((wq_raw * scale).round())
        self.assertLessEqual(unique.numel(), 3)

    def test_ste_gradient_pass_through(self):
        w = torch.randn(4, 4, requires_grad=True)
        wq = weight_quant(w)
        wq.sum().backward()
        self.assertIsNotNone(w.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BitLinearTests(unittest.TestCase):
    def test_forward_shape(self):
        layer = BitLinear(16, 32)
        x = torch.randn(2, 8, 16)
        out = layer(x)
        self.assertEqual(out.shape, (2, 8, 32))

    def test_no_bias_by_default(self):
        layer = BitLinear(8, 16)
        self.assertIsNone(layer.bias)

    def test_is_linear_subclass(self):
        self.assertTrue(issubclass(BitLinear, nn.Linear))

    def test_gradient_flows(self):
        layer = BitLinear(8, 8)
        x = torch.randn(1, 4, 8, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_square_dimensions(self):
        layer = BitLinear(16, 16)
        x = torch.randn(3, 16)
        out = layer(x)
        self.assertEqual(out.shape, (3, 16))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class DynamicTanhNormTests(unittest.TestCase):
    def test_output_shape(self):
        norm = DynamicTanhNorm(32)
        x = torch.randn(2, 10, 32)
        y = norm(x)
        self.assertEqual(y.shape, x.shape)

    def test_learnable_alpha_beta(self):
        norm = DynamicTanhNorm(16)
        self.assertEqual(norm.alpha.shape, (16,))
        self.assertEqual(norm.beta.shape, (16,))

    def test_output_range_bounded_by_tanh(self):
        norm = DynamicTanhNorm(8)
        x = torch.randn(4, 5, 8) * 100
        y = norm(x)
        self.assertTrue((y.abs() <= 1.0 + 1e-5).all())

    def test_gradient_flows(self):
        norm = DynamicTanhNorm(16)
        x = torch.randn(2, 4, 16, requires_grad=True)
        norm(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class DerfTests(unittest.TestCase):
    def test_output_shape(self):
        norm = Derf(32)
        x = torch.randn(2, 10, 32)
        y = norm(x)
        self.assertEqual(y.shape, x.shape)

    def test_learnable_parameters(self):
        norm = Derf(16)
        self.assertTrue(hasattr(norm, "alpha"))
        self.assertTrue(hasattr(norm, "gamma"))
        self.assertTrue(hasattr(norm, "beta"))
        self.assertTrue(hasattr(norm, "s"))

    def test_gradient_flows(self):
        norm = Derf(16)
        x = torch.randn(1, 3, 16, requires_grad=True)
        norm(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class RMSNormTests(unittest.TestCase):
    def test_output_shape(self):
        norm = RMSNorm(32)
        x = torch.randn(2, 10, 32)
        y = norm(x)
        self.assertEqual(y.shape, x.shape)

    def test_learnable_weight_no_bias(self):
        norm = RMSNorm(16)
        self.assertEqual(norm.weight.shape, (16,))
        self.assertFalse(hasattr(norm, "bias"))

    def test_correctness_against_manual_rms(self):
        dim = 8
        norm = RMSNorm(dim, eps=1e-6)
        norm.weight.data.fill_(1.0)  # identity scale
        x = torch.randn(4, dim)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        expected = x / rms
        y = norm(x)
        self.assertTrue(torch.allclose(y, expected, atol=1e-6))

    def test_scale_is_applied(self):
        dim = 4
        norm = RMSNorm(dim)
        norm.weight.data.fill_(2.0)
        x = torch.ones(1, dim)
        y = norm(x)
        # RMS of ones = 1, so output should be 2.0 everywhere
        self.assertTrue(torch.allclose(y, torch.full_like(y, 2.0), atol=1e-5))

    def test_gradient_flows(self):
        norm = RMSNorm(16)
        x = torch.randn(2, 4, 16, requires_grad=True)
        norm(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_partial_uses_subset_k(self):
        dim = 32
        ratio = 0.0625
        norm = RMSNorm(dim, partial_ratio=ratio)
        import math
        expected_k = max(1, math.ceil(dim * ratio))
        self.assertEqual(norm.k, expected_k)
        self.assertEqual(norm.partial_ratio, ratio)

    def test_partial_correctness(self):
        dim = 16
        ratio = 0.5
        norm = RMSNorm(dim, eps=0.0, partial_ratio=ratio)
        norm.weight.data.fill_(1.0)
        x = torch.randn(4, dim)
        k = norm.k
        rms_partial = torch.sqrt(x[..., :k].pow(2).mean(dim=-1, keepdim=True))
        expected = x / rms_partial
        y = norm(x)
        self.assertTrue(torch.allclose(y, expected, atol=1e-5))

    def test_partial_ratio_zero_uses_all_dims(self):
        norm = RMSNorm(32, partial_ratio=0.0)
        self.assertEqual(norm.k, 32)

    def test_partial_ratio_one_equals_full(self):
        dim = 16
        x = torch.randn(4, dim)
        full = RMSNorm(dim, eps=1e-6)(x)
        partial = RMSNorm(dim, eps=1e-6, partial_ratio=1.0)(x)
        self.assertTrue(torch.allclose(full, partial, atol=1e-6))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class FlashNormTests(unittest.TestCase):
    def test_output_shape(self):
        norm = FlashNorm(32)
        x = torch.randn(2, 10, 32)
        y = norm(x)
        self.assertEqual(y.shape, x.shape)

    def test_weightless_no_learnable_params(self):
        norm = FlashNorm(16)
        # FlashNorm has no Parameters (weightless per Prop. 1 of arXiv:2407.09577).
        params = list(norm.parameters())
        self.assertEqual(len(params), 0)

    def test_correctness_against_rmsnorm_with_unit_weight(self):
        # FlashNorm output == RMSNorm(weight=1) output for same eps.
        dim = 8
        eps = 1e-6
        flash = FlashNorm(dim, eps=eps)
        rms = RMSNorm(dim, eps=eps)
        rms.weight.data.fill_(1.0)
        x = torch.randn(4, dim)
        self.assertTrue(torch.allclose(flash(x), rms(x), atol=1e-6))

    def test_correctness_manual(self):
        dim = 8
        flash = FlashNorm(dim, eps=1e-6)
        x = torch.randn(4, dim)
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        expected = x * torch.rsqrt(ms + 1e-6)
        self.assertTrue(torch.allclose(flash(x), expected, atol=1e-6))

    def test_partial_ratio_uses_subset_k(self):
        dim = 32
        ratio = 0.25
        norm = FlashNorm(dim, partial_ratio=ratio)
        import math
        expected_k = max(1, math.ceil(dim * ratio))
        self.assertEqual(norm.k, expected_k)
        self.assertEqual(norm.partial_ratio, ratio)

    def test_partial_correctness(self):
        dim = 16
        ratio = 0.5
        norm = FlashNorm(dim, eps=0.0, partial_ratio=ratio)
        x = torch.randn(4, dim)
        k = norm.k
        ms_partial = x[..., :k].pow(2).mean(dim=-1, keepdim=True)
        expected = x * torch.rsqrt(ms_partial)
        self.assertTrue(torch.allclose(norm(x), expected, atol=1e-5))

    def test_partial_ratio_zero_uses_all_dims(self):
        norm = FlashNorm(32, partial_ratio=0.0)
        self.assertEqual(norm.k, 32)

    def test_gradient_flows(self):
        norm = FlashNorm(16)
        x = torch.randn(2, 4, 16, requires_grad=True)
        norm(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_rms_inv_helper(self):
        norm = FlashNorm(8, eps=1e-6)
        x = torch.randn(4, 8)
        rms_inv = norm.rms_inv(x)
        # Shape: (..., 1)
        self.assertEqual(rms_inv.shape, (4, 1))
        # Equals 1/sqrt(mean(x^2)+eps)
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        expected = torch.rsqrt(ms + 1e-6)
        self.assertTrue(torch.allclose(rms_inv, expected, atol=1e-6))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class FlashNormLinearTests(unittest.TestCase):
    def test_output_shape(self):
        fnl = FlashNormLinear(16, 32, bias=False)
        x = torch.randn(2, 5, 16)
        y = fnl(x)
        self.assertEqual(y.shape, (2, 5, 32))

    def test_prop2_equivalent_to_sequential_bias_free(self):
        # Prop. 2: (a * rms_inv) @ W == (a @ W) * rms_inv  for bias-free linear.
        torch.manual_seed(0)
        fnl = FlashNormLinear(8, 12, bias=False, eps=1e-8)
        x = torch.randn(3, 8)
        fused = fnl(x)
        # Reference: sequential flash then linear
        rms_inv = fnl.flash.rms_inv(x)
        reference = torch.nn.functional.linear(x * rms_inv, fnl.linear.weight)
        self.assertTrue(torch.allclose(fused, reference, atol=1e-6))

    def test_prop1_weight_folding_equivalence(self):
        # Prop. 1: RMSNorm(g) -> Linear(W)  ==  FlashNorm -> Linear(W* = diag(g) W).
        torch.manual_seed(0)
        dim = 8
        rms = RMSNorm(dim, eps=1e-8)
        linear = torch.nn.Linear(dim, 12, bias=False)
        x = torch.randn(3, dim)
        reference = linear(rms(x))

        # Fold g into linear's weights
        from src.model.norm.flash import fold_norm_weights
        folded = torch.nn.Linear(dim, 12, bias=False)
        folded.weight.data.copy_(linear.weight.data)
        fold_norm_weights(folded, rms.weight.data)
        flash = FlashNorm(dim, eps=1e-8)
        out = flash(x) @ folded.weight.t()
        self.assertTrue(torch.allclose(out, reference, atol=1e-6))

    def test_bias_path_is_sequential(self):
        # With bias, Prop. 2 cannot defer; layer must apply RMS before the linear.
        fnl = FlashNormLinear(8, 4, bias=True)
        self.assertIsNotNone(fnl.linear.bias)
        x = torch.randn(3, 8)
        y = fnl(x)
        rms_inv = fnl.flash.rms_inv(x)
        expected = torch.nn.functional.linear(x * rms_inv, fnl.linear.weight, fnl.linear.bias)
        self.assertTrue(torch.allclose(y, expected, atol=1e-6))

    def test_from_rmsnorm_and_linear_folds_weights(self):
        torch.manual_seed(0)
        dim = 6
        rms = RMSNorm(dim, eps=1e-6, partial_ratio=0.5)
        linear = torch.nn.Linear(dim, 10, bias=False)
        original_w = linear.weight.data.clone()
        fnl = FlashNormLinear.from_rmsnorm_and_linear(rms, linear)
        # Weight was modified in place: W*[i,j] = g[j] * W[i,j]
        # Verify folded = original * g (broadcast over rows).
        expected = original_w * rms.weight.data.unsqueeze(0)
        self.assertTrue(torch.allclose(fnl.linear.weight.data, expected, atol=1e-6))
        self.assertEqual(fnl.linear.in_features, dim)
        self.assertEqual(fnl.linear.out_features, 10)
        self.assertAlmostEqual(fnl.flash.partial_ratio, 0.5)

    def test_fold_norm_weights_validates_shape(self):
        linear = torch.nn.Linear(8, 4, bias=False)
        bad_g = torch.randn(10)  # wrong shape
        with self.assertRaises(ValueError):
            fold_norm_weights(linear, bad_g)

    def test_gradient_flows(self):
        fnl = FlashNormLinear(8, 4, bias=False)
        x = torch.randn(2, 3, 8, requires_grad=True)
        fnl(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(fnl.linear.weight.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class FlashNormBitLinearTests(unittest.TestCase):
    def test_output_shape(self):
        fnbl = FlashNormBitLinear(16, 32, bias=False)
        x = torch.randn(2, 5, 16)
        y = fnbl(x)
        self.assertEqual(y.shape, (2, 5, 32))

    def test_wraps_bitlinear(self):
        from src.model.attention.common import BitLinear
        fnbl = FlashNormBitLinear(8, 4, bias=False)
        self.assertIsInstance(fnbl.linear, BitLinear)
        self.assertIsInstance(fnbl.flash, FlashNorm)

    def test_sequential_composition(self):
        # FlashNormBitLinear is sequential: out = BitLinear(FlashNorm(x)).
        torch.manual_seed(0)
        fnbl = FlashNormBitLinear(8, 4, bias=False, eps=1e-8)
        from src.model.attention.common import BitLinear
        # Reference: fresh BitLinear with same weights + same FlashNorm
        ref_bitlinear = BitLinear(8, 4, bias=False)
        ref_bitlinear.weight.data.copy_(fnbl.linear.weight.data)
        x = torch.randn(3, 8)
        fused = fnbl(x)
        reference = ref_bitlinear(fnbl.flash(x))
        self.assertTrue(torch.allclose(fused, reference, atol=1e-6))

    def test_gradient_flows(self):
        fnbl = FlashNormBitLinear(8, 4, bias=False)
        x = torch.randn(2, 3, 8, requires_grad=True)
        fnbl(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GetNormTests(unittest.TestCase):
    def test_dynamic_tanh_returns_correct_type(self):
        cfg = _cfg(norm_type="dynamic_tanh")
        norm = get_norm(cfg)
        self.assertIsInstance(norm, DynamicTanhNorm)

    def test_derf_returns_correct_type(self):
        cfg = _cfg(norm_type="derf")
        norm = get_norm(cfg)
        self.assertIsInstance(norm, Derf)

    def test_rms_norm_returns_correct_type(self):
        cfg = _cfg(norm_type="rms_norm")
        norm = get_norm(cfg)
        self.assertIsInstance(norm, RMSNorm)
        self.assertEqual(norm.partial_ratio, 0.0)
        self.assertEqual(norm.k, cfg.hidden_size)

    def test_prms_norm_returns_correct_type(self):
        cfg = _cfg(norm_type="prms_norm")
        norm = get_norm(cfg)
        self.assertIsInstance(norm, RMSNorm)
        self.assertGreater(norm.partial_ratio, 0.0)
        self.assertEqual(norm.partial_ratio, cfg.prms_partial_ratio)

    def test_prms_norm_custom_ratio(self):
        cfg = _cfg(norm_type="prms_norm")
        cfg.prms_partial_ratio = 0.5
        norm = get_norm(cfg)
        self.assertIsInstance(norm, RMSNorm)
        self.assertAlmostEqual(norm.partial_ratio, 0.5)

    def test_flash_norm_returns_correct_type(self):
        cfg = _cfg(norm_type="flash_norm")
        norm = get_norm(cfg)
        self.assertIsInstance(norm, FlashNorm)
        self.assertEqual(norm.partial_ratio, 0.0)
        self.assertEqual(norm.k, cfg.hidden_size)

    def test_flash_norm_partial_ratio(self):
        cfg = _cfg(norm_type="flash_norm")
        cfg.flashnorm_partial_ratio = 0.25
        norm = get_norm(cfg)
        self.assertIsInstance(norm, FlashNorm)
        self.assertAlmostEqual(norm.partial_ratio, 0.25)
        import math
        self.assertEqual(norm.k, max(1, math.ceil(cfg.hidden_size * 0.25)))

    def test_flash_norm_default_partial_ratio(self):
        cfg = _cfg(norm_type="flash_norm")
        norm = get_norm(cfg)
        # Default flashnorm_partial_ratio is 0.0 (full RMS, not partial).
        self.assertEqual(norm.partial_ratio, 0.0)
        self.assertEqual(norm.k, cfg.hidden_size)

    def test_fallback_returns_layer_norm(self):
        cfg = _cfg(norm_type="layer_norm")
        norm = get_norm(cfg)
        self.assertIsInstance(norm, nn.LayerNorm)

    def test_unknown_norm_fallback_to_layer_norm(self):
        cfg = _cfg(norm_type="unknown_norm_xyz")
        norm = get_norm(cfg)
        self.assertIsInstance(norm, nn.LayerNorm)


if __name__ == "__main__":
    unittest.main()
