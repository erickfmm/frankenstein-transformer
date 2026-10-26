"""Unit tests for HybridLayer."""
import unittest
import warnings
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from src.model.config import FrankensteinModelConfig
    from src.model.hybrid_layer import HybridLayer


def _cfg(layer_pattern=None, **overrides):
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
        layer_pattern=layer_pattern or ["standard_attn"],
        use_moe=False,
        ode_solver="rk4",
        ode_steps=1,
        ffn_hidden_size=96,
        ffn_activation="gelu",
        positional_encoding="hope",
        mode="encoder",
    )
    base.update(overrides)
    return FrankensteinModelConfig(**base)


BSZ, SEQ, DIM = 2, 8, 48


def _x():
    return torch.randn(BSZ, SEQ, DIM)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class HybridLayerMambaTests(unittest.TestCase):
    def test_mamba_output_shape(self):
        layer = HybridLayer(_cfg(), layer_type="mamba")
        y = layer(_x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_mamba_residual_connection(self):
        # output should not equal input (transformed by mixer+ffn)
        layer = HybridLayer(_cfg(), layer_type="mamba")
        x = _x()
        y = layer(x)
        self.assertFalse(torch.allclose(x, y))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class HybridLayerDenseFfnTests(unittest.TestCase):
    def test_dense_ffn_output_shape(self):
        layer = HybridLayer(_cfg(use_moe=False), layer_type="standard_attn")
        y = layer(_x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_gradient_flows_dense(self):
        layer = HybridLayer(_cfg(use_moe=False), layer_type="standard_attn")
        x = _x().requires_grad_(True)
        layer(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class HybridLayerMoETests(unittest.TestCase):
    def test_moe_output_shape(self):
        cfg = _cfg(use_moe=True, num_experts=4, top_k_experts=2)
        layer = HybridLayer(cfg, layer_type="standard_attn")
        y = layer(_x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_moe_gradient_flows(self):
        cfg = _cfg(use_moe=True, num_experts=4, top_k_experts=2)
        layer = HybridLayer(cfg, layer_type="standard_attn")
        x = _x().requires_grad_(True)
        layer(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_moe_has_router(self):
        cfg = _cfg(use_moe=True)
        layer = HybridLayer(cfg, layer_type="standard_attn")
        self.assertIsNotNone(layer.router)
        self.assertIsNotNone(layer.experts)

    def test_dense_has_no_router(self):
        layer = HybridLayer(_cfg(use_moe=False), layer_type="standard_attn")
        self.assertIsNone(layer.router)
        self.assertIsNone(layer.experts)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class HybridLayerTrainingFreeTests(unittest.TestCase):
    """Training-free mixers train as dense attention, infer sparsely."""

    def _assert_dense_fallback(self, layer_type):
        layer = HybridLayer(_cfg(), layer_type=layer_type)
        layer.train()
        with self.assertWarnsRegex(UserWarning, "training-free"):
            y = layer(_x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))
        # Gradients reach the projections shared with the sparse eval path.
        y.sum().backward()
        self.assertIsNotNone(layer.mixer.q_proj.weight.grad)
        self.assertIsNotNone(layer.mixer.out_proj.weight.grad)
        # The warning is emitted only once per layer instance.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            y2 = layer(_x())
        self.assertEqual(y2.shape, (BSZ, SEQ, DIM))
        self.assertEqual([w for w in caught if "training-free" in str(w.message)], [])

    def test_fasa_dense_fallback_in_train_mode(self):
        self._assert_dense_fallback("fasa_attn")

    def test_sparge_dense_fallback_in_train_mode(self):
        self._assert_dense_fallback("sparge_attn")

    def test_fasa_works_in_eval_mode(self):
        layer = HybridLayer(_cfg(), layer_type="fasa_attn")
        layer.eval()
        with torch.no_grad():
            y = layer(_x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_sparge_works_in_eval_mode(self):
        layer = HybridLayer(_cfg(), layer_type="sparge_attn")
        layer.eval()
        with torch.no_grad():
            y = layer(_x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_dense_fallback_equals_standard_attention(self):
        """Train-mode fallback is exactly standard attention (shared weights)."""
        from src.model.attention.standard import StandardAttention

        layer = HybridLayer(_cfg(), layer_type="fasa_attn")
        standard = StandardAttention(_cfg())
        # Share the projections: the fallback must read the mixer weights.
        standard.q_proj = layer.mixer.q_proj
        standard.k_proj = layer.mixer.k_proj
        standard.v_proj = layer.mixer.v_proj
        standard.out_proj = layer.mixer.out_proj
        standard.pe_type = layer.mixer.pe_type
        standard.use_pe = layer.mixer.use_pe
        standard.pos_encoder = layer.mixer.pos_encoder

        x = _x()
        layer.train()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = layer.mixer.dense_forward(x, pos_encoder=layer.pos_encoder)
        expected = standard(x)
        self.assertTrue(torch.allclose(got, expected, atol=1e-6))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class HybridLayerMiscTests(unittest.TestCase):
    def test_unknown_layer_type_raises(self):
        with self.assertRaises(ValueError):
            HybridLayer(_cfg(), layer_type="unicorn_attn")

    def test_logical_layer_idx_passed(self):
        layer = HybridLayer(_cfg(), layer_type="standard_attn")
        y = layer(_x(), logical_layer_idx=7)
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_silu_activation(self):
        cfg = _cfg(ffn_activation="silu")
        layer = HybridLayer(cfg, layer_type="standard_attn")
        y = layer(_x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_ode_layer_type(self):
        layer = HybridLayer(_cfg(), layer_type="ode")
        y = layer(_x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_retnet_layer_type(self):
        layer = HybridLayer(_cfg(), layer_type="retnet")
        y = layer(_x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))


if __name__ == "__main__":
    unittest.main()
