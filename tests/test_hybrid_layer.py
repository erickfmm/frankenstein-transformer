"""Unit tests for HybridLayer."""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from src.model.tormented_bert_frankestein import HybridLayer, UltraConfig


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
        use_hope=True,
        mode="encoder",
    )
    base.update(overrides)
    return UltraConfig(**base)


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
    def test_fasa_raises_in_train_mode(self):
        layer = HybridLayer(_cfg(), layer_type="fasa_attn")
        layer.train()
        with self.assertRaisesRegex(ValueError, "training-free"):
            layer(_x())

    def test_sparge_raises_in_train_mode(self):
        layer = HybridLayer(_cfg(), layer_type="sparge_attn")
        layer.train()
        with self.assertRaisesRegex(ValueError, "training-free"):
            layer(_x())

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
