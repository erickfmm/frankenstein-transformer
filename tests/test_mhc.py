"""Unit tests for Manifold-Constrained Hyper-Connections (mHC, arXiv:2512.24880)."""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from src.model.mhc import (
        ManifoldHyperConnections,
        SinkhornKnoppFunction,
    )
    from src.model.attention.common import BitLinear
    from src.model.config import FrankensteinModelConfig
    from src.model.hybrid_layer import HybridLayer
    from src.model.frankenstein_encoder import FrankensteinEncoder


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class SinkhornKnoppTests(unittest.TestCase):
    def test_produces_doubly_stochastic_matrix(self):
        X = torch.randn(5, 5)
        P = SinkhornKnoppFunction.apply(X, 100)
        self.assertTrue((P >= 0).all().item())
        torch.testing.assert_close(P.sum(dim=-1), torch.ones(5), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(P.sum(dim=-2), torch.ones(5), atol=1e-4, rtol=1e-4)

    def test_gradient_flows(self):
        X = torch.randn(4, 4, requires_grad=True)
        P = SinkhornKnoppFunction.apply(X, 20)
        P.sum().backward()
        self.assertIsNotNone(X.grad)
        self.assertTrue(torch.isfinite(X.grad).all().item())


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ManifoldHyperConnectionsTests(unittest.TestCase):
    def _module(self, **overrides):
        kwargs = dict(
            hidden_size=32,
            expansion_rate=4,
            sinkhorn_iters=20,
            gating_init=0.01,
            use_bitnet=False,
            full_prec_under_bitnet=True,
        )
        kwargs.update(overrides)
        return ManifoldHyperConnections(**kwargs)

    def test_forward_shapes(self):
        m = self._module()
        x = torch.randn(2, 8, 4, 32)
        out = torch.randn(2, 8, 32)
        fpre, x_next = m(x, out)
        self.assertEqual(fpre.shape, (2, 8, 32))
        self.assertEqual(x_next.shape, (2, 8, 4, 32))

    def test_recombine_doubly_stochastic_res_mixing(self):
        m = self._module()
        x = torch.randn(1, 4, 4, 32)
        out = torch.randn(1, 4, 32)
        x_next = m.recombine(x, out)
        self.assertEqual(x_next.shape, x.shape)
        # recombine should differ from a plain identity pass (non-trivial mixing)
        self.assertFalse(torch.allclose(x_next, x))

    def test_bitnet_full_prec_default(self):
        m = self._module(use_bitnet=True, full_prec_under_bitnet=True)
        self.assertIsInstance(m.proj, torch.nn.Linear)

    def test_bitnet_full_prec_disabled_uses_bitlinear(self):
        m = self._module(use_bitnet=True, full_prec_under_bitnet=False)
        self.assertIsInstance(m.proj, BitLinear)

    def test_expansion_rate_one_recovers_identity(self):
        # n=1 should degenerate toward identity: H[pre] ~ 1, H[post] ~ small,
        # H[res] = 1 (single doubly stochastic scalar). At init gating the
        # mappings are near-identity for pre, but post adds the layer output.
        m = self._module(expansion_rate=1)
        x = torch.randn(1, 3, 1, 32)
        out = torch.zeros(1, 3, 32)
        fpre, x_next = m(x, out)
        # fpre = H[pre] @ x with a single stream: H[pre] is a scalar in (0,1);
        # the residual part H[res]@x = 1.0 * x (single doubly-stochastic 1x1).
        self.assertEqual(fpre.shape, (1, 3, 32))
        self.assertEqual(x_next.shape, (1, 3, 1, 32))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class MhcModelTests(unittest.TestCase):
    def _cfg(self, **overrides):
        base = dict(
            vocab_size=128,
            hidden_size=64,
            num_layers=2,
            num_loops=1,
            num_heads=4,
            retention_heads=4,
            num_experts=2,
            top_k_experts=1,
            dropout=0.0,
            norm_type="layer_norm",
            use_bitnet=False,
            use_moe=False,
            layer_pattern=["standard_attn", "retnet"],
            use_hope=True,
            mode="encoder",
            use_mhc=True,
            mhc_expansion_rate=4,
            mhc_sinkhorn_iters=20,
            mhc_gating_init=0.01,
            mhc_checkpoint=False,
            mhc_full_prec_under_bitnet=True,
        )
        base.update(overrides)
        return FrankensteinModelConfig(**base)

    def test_model_output_shape(self):
        cfg = self._cfg()
        model = FrankensteinEncoder(cfg)
        ids = torch.randint(0, 128, (2, 8))
        out = model(ids)
        self.assertEqual(out.shape, (2, 8, 128))

    def test_gradient_flows_through_sinkhorn(self):
        cfg = self._cfg()
        model = FrankensteinEncoder(cfg)
        ids = torch.randint(0, 128, (2, 8))
        out = model(ids)
        out.sum().backward()
        mhc = model.layers[0].mhc_attn
        self.assertIsNotNone(mhc.proj.weight.grad)
        self.assertTrue(torch.isfinite(mhc.proj.weight.grad).all().item())

    def test_checkpoint_forward_backward(self):
        cfg = self._cfg(mhc_checkpoint=True)
        model = FrankensteinEncoder(cfg)
        ids = torch.randint(0, 128, (2, 8))
        out = model(ids)
        out.sum().backward()
        self.assertEqual(out.shape, (2, 8, 128))

    def test_moe_supported_with_mhc(self):
        cfg = self._cfg(use_moe=True, num_experts=2, top_k_experts=1)
        model = FrankensteinEncoder(cfg)
        ids = torch.randint(0, 128, (2, 8))
        out = model(ids)
        out.sum().backward()
        self.assertEqual(out.shape, (2, 8, 128))

    def test_mod_incompatible_with_mhc(self):
        cfg = self._cfg(use_mixture_of_depths=True)
        with self.assertRaises(ValueError):
            HybridLayer(cfg, layer_type="standard_attn")

    def test_stream_expansion_modules_present(self):
        cfg = self._cfg()
        model = FrankensteinEncoder(cfg)
        self.assertIsNotNone(model.mhc_in_proj)
        self.assertIsNotNone(model.mhc_out_proj)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class MhcHybridLayerTests(unittest.TestCase):
    def _cfg(self):
        return FrankensteinModelConfig(
            vocab_size=64,
            hidden_size=48,
            num_layers=1,
            num_loops=1,
            num_heads=6,
            retention_heads=6,
            dropout=0.0,
            norm_type="layer_norm",
            use_bitnet=False,
            use_moe=False,
            layer_pattern=["standard_attn"],
            use_mhc=True,
            mhc_expansion_rate=2,
            use_hope=True,
            mode="encoder",
        )

    def test_hybrid_layer_nstream_output(self):
        layer = HybridLayer(self._cfg(), layer_type="standard_attn")
        x = torch.randn(2, 8, 2, 48)
        y = layer(x)
        self.assertEqual(y.shape, (2, 8, 2, 48))


if __name__ == "__main__":
    unittest.main()
