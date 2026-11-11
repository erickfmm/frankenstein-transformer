"""Unit tests for gated attention modules."""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from src.model.config import FrankensteinModelConfig, merge_model_blocks
    from src.model.attention.gated import (
        GatedLinearAttention,
        DeltaNetAttention,
        GatedDeltaNetAttention,
        HGRN2Attention,
        ForgettingAttention,
        FirstPassageAttention,
        GatedSoftmaxAttention,
        KDAAttention,
    )


def _cfg(**overrides):
    base = {"dims": {"vocab_size": 64, "hidden_size": 48, "num_layers": 1, "num_loops": 1, "num_heads": 6, "dropout": 0.0, "layer_pattern": ['standard'], "mode": 'encoder'}, "attention": {"retnet": {"heads": 6}, "ode": {"solver": 'rk4', "steps": 1}}, "moe": {"num_experts": 2, "top_k": 1, "enabled": False}, "norm": {"type": 'layer_norm'}, "bitnet": {"enabled": False}, "ffn": {"hidden_size": 96, "activation": 'gelu'}, "positional_encoding": 'hope'}
    base = merge_model_blocks(base, overrides)
    return FrankensteinModelConfig.from_dict({**base})


BSZ, SEQ, DIM = 2, 8, 48


def _x():
    return torch.randn(BSZ, SEQ, DIM)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GatedLinearAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = GatedLinearAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = GatedLinearAttention(_cfg(dims={'mode': 'decoder'}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = GatedLinearAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_hidden_raises(self):
        with self.assertRaises(ValueError):
            GatedLinearAttention(_cfg(dims={'hidden_size': 50, 'num_heads': 6}))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class DeltaNetAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = DeltaNetAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = DeltaNetAttention(_cfg(dims={'mode': 'decoder'}))
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
class FirstPassageAttentionTests(unittest.TestCase):
    def test_encoder_output_shape(self):
        attn = FirstPassageAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_output_shape(self):
        attn = FirstPassageAttention(_cfg(dims={'mode': 'decoder'}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_output_is_finite(self):
        attn = FirstPassageAttention(_cfg(dims={'mode': 'decoder'})).eval()
        out = attn(_x())
        self.assertTrue(torch.isfinite(out).all())

    def test_gradient_flows(self):
        attn = FirstPassageAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_knobs_honored(self):
        attn = FirstPassageAttention(_cfg(attention={'fpa': {'beta_init': 2.5, 'tau': 0.5}}))
        self.assertTrue(torch.allclose(attn.beta, torch.full((6,), 2.5)))
        self.assertEqual(attn.tau, 0.5)
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_bitnet_on(self):
        attn = FirstPassageAttention(_cfg(bitnet={'enabled': True}))
        out = attn(_x())
        self.assertEqual(out.shape, (BSZ, SEQ, DIM))
        self.assertTrue(torch.isfinite(out).all())

    def test_diagonal_bias_is_zero(self):
        attn = FirstPassageAttention(_cfg()).eval()
        phi = torch.randn(BSZ, 6, SEQ)
        psi = torch.randn(BSZ, 6, SEQ)
        bias = attn._bias_matrix(phi, psi)
        self.assertTrue(torch.allclose(bias.diagonal(dim1=-2, dim2=-1), torch.zeros(BSZ, 6, SEQ)))

    def test_encoder_span_minimum_is_bidirectional(self):
        attn = FirstPassageAttention(_cfg()).eval()
        phi = torch.randn(BSZ, 6, SEQ)
        psi = torch.randn(BSZ, 6, SEQ)
        bias = attn._bias_matrix(phi, psi)
        # Encoder extension: the open-span minimum runs in either direction;
        # each key keeps its own level psi_s (so the bias is directional).
        span = attn._hard_range_min(phi)
        span_sym = torch.minimum(span, span.transpose(-1, -2))
        expected = attn.beta.view(1, -1, 1, 1) * torch.clamp(
            span_sym - psi.unsqueeze(-2), max=0.0
        )
        self.assertTrue(torch.allclose(bias, expected))

    def test_special_case_fox_like(self):
        """Prop. 2.2(b): a non-rising walk reproduces the FoX sum-of-logs bias."""
        attn = FirstPassageAttention(_cfg()).eval()
        # g = h = 0, delta_u = -1  =>  W_t = -t, phi = psi = W.
        t = torch.arange(SEQ, dtype=torch.float32)
        phi = (-t).view(1, 1, SEQ).expand(BSZ, 6, SEQ)
        psi = phi.clone()
        bias = attn._bias_matrix(phi, psi)
        beta = attn.beta.view(1, -1, 1, 1)
        # FoX closed form: b_ts = beta * sum_{s<u<=t} delta_u (delta = -1).
        s = t.view(1, SEQ)
        tt = t.view(SEQ, 1)
        expected = beta * torch.clamp(-(tt - s), max=0.0)
        expected = expected.expand(BSZ, 6, SEQ, SEQ)
        mask = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool))
        self.assertTrue(torch.allclose(bias[mask.expand_as(bias)], expected[mask.expand_as(bias)]))

    def test_field_parameters_receive_gradients(self):
        attn = FirstPassageAttention(_cfg())
        # Deterministic active bias: delta = -1, g = h = 0 (decreasing walk).
        with torch.no_grad():
            attn.field_proj.weight.zero_()
            bias_vec = torch.zeros(3 * 6)
            bias_vec[:6] = -1.0
            attn.field_proj.bias.copy_(bias_vec)
        attn(_x()).sum().backward()
        self.assertIsNotNone(attn.beta.grad)
        self.assertGreater(float(attn.beta.grad.abs().sum()), 0.0)
        self.assertIsNotNone(attn.field_proj.bias.grad)
        self.assertGreater(float(attn.field_proj.bias.grad.abs().sum()), 0.0)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GatedSoftmaxAttentionTests(unittest.TestCase):
    def test_encoder_output_shape(self):
        attn = GatedSoftmaxAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_output_shape(self):
        attn = GatedSoftmaxAttention(_cfg(dims={'mode': 'decoder'}))
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
        attn = KDAAttention(_cfg(dims={'mode': 'decoder'}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = KDAAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_hidden_raises(self):
        with self.assertRaises(ValueError):
            KDAAttention(_cfg(dims={'hidden_size': 50, 'num_heads': 6}))


if __name__ == "__main__":
    unittest.main()
