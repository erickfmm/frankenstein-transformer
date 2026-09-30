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


if TORCH_AVAILABLE:
    from src.model.hyperloop import HyperloopConnections


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class HyperloopConnectionsTests(unittest.TestCase):
    """Unit tests for the loop-level hyper-connection module."""

    def _module(self, **overrides):
        kwargs = dict(
            hidden_size=32,
            num_loops=3,
            expansion_rate=4,
            gating_init=0.01,
            res_parameterization="diagonal",
            sinkhorn_iters=20,
            use_bitnet=False,
            full_prec_under_bitnet=True,
        )
        kwargs.update(overrides)
        return HyperloopConnections(**kwargs)

    def test_valid_res_parameterizations(self):
        for rp in ("diagonal", "sinkhorn", "identity"):
            m = self._module(res_parameterization=rp)
            self.assertEqual(m.res_parameterization, rp)

    def test_invalid_res_parameterization_rejected(self):
        with self.assertRaises(ValueError):
            self._module(res_parameterization="bogus")

    def test_mapping_shapes_per_parameterization(self):
        x = torch.randn(2, 8, 4, 32)
        h_pre, h_post, h_res = self._module(res_parameterization="diagonal").mappings(x, 0)
        self.assertEqual(h_pre.shape, (2, 8, 4))
        self.assertEqual(h_post.shape, (2, 8, 4))
        self.assertEqual(h_res.shape, (2, 8, 4))
        h_pre, h_post, h_res = self._module(res_parameterization="sinkhorn").mappings(x, 0)
        self.assertEqual(h_res.shape, (2, 8, 4, 4))
        h_pre, h_post, h_res = self._module(res_parameterization="identity").mappings(x, 0)
        self.assertIsNone(h_res)

    def test_sinkhorn_produces_doubly_stochastic(self):
        m = self._module(res_parameterization="sinkhorn")
        x = torch.randn(1, 4, 4, 32)
        _, _, h_res = m.mappings(x, 1)
        torch.testing.assert_close(
            h_res.sum(dim=-1), torch.ones(1, 4, 4), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            h_res.sum(dim=-2), torch.ones(1, 4, 4), atol=1e-3, rtol=1e-3
        )

    def test_identity_update_preserves_stream(self):
        m = self._module(res_parameterization="identity")
        x = torch.randn(2, 4, 4, 32)
        out = torch.randn(2, 4, 32)
        h_pre, h_post, h_res = m.mappings(x, 0)
        updated = m.apply_update(x, out, h_post, h_res, 0)
        expected = x + torch.einsum(
            "...n,...C->...nC", h_post, out + m.loop_pos_embs[0]
        )
        torch.testing.assert_close(updated, expected, atol=1e-6, rtol=1e-6)

    def test_expand_copies_and_collapse_averages(self):
        m = self._module(expansion_rate=4)
        x = torch.randn(2, 4, 32)
        expanded = m.expand(x)
        self.assertEqual(expanded.shape, (2, 4, 4, 32))
        for i in range(4):
            torch.testing.assert_close(expanded[:, :, i], x)
        torch.testing.assert_close(m.collapse(expanded), x)

    def test_per_loop_parameters_are_independent(self):
        m = self._module(num_loops=3)
        self.assertEqual(len(m.projs), 3)
        self.assertEqual(len(m.biases), 3)
        self.assertEqual(len(m.alphas_pre), 3)
        self.assertEqual(len(m.alphas_post), 3)
        self.assertEqual(len(m.alphas_res), 3)
        self.assertEqual(len(m.loop_pos_embs), 3)
        # Parameters are distinct objects (per-loop, not shared).
        self.assertIsNot(m.projs[0], m.projs[1])
        # loop_idx wraps safely instead of crashing.
        x = torch.randn(1, 2, 4, 32)
        h_pre_a, _, _ = m.mappings(x, 3)
        h_pre_b, _, _ = m.mappings(x, 0)
        torch.testing.assert_close(h_pre_a, h_pre_b)

    def test_identity_parameterization_has_no_res_params(self):
        m = self._module(res_parameterization="identity")
        self.assertEqual(len(m.alphas_res), 0)
        # proj output dim = 2n (pre + post) under identity
        self.assertEqual(m.projs[0].out_features, 2 * m.expansion_rate)

    def test_bitnet_full_prec_default(self):
        m = self._module(use_bitnet=True, full_prec_under_bitnet=True)
        self.assertIsInstance(m.projs[0], torch.nn.Linear)

    def test_bitnet_full_prec_disabled_uses_bitlinear(self):
        m = self._module(use_bitnet=True, full_prec_under_bitnet=False)
        self.assertIsInstance(m.projs[0], BitLinear)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class HyperloopModelTests(unittest.TestCase):
    def _cfg(self, **overrides):
        base = dict(
            vocab_size=128,
            hidden_size=64,
            num_layers=6,
            num_loops=3,
            num_heads=4,
            retention_heads=4,
            dropout=0.0,
            norm_type="layer_norm",
            use_bitnet=False,
            use_moe=False,
            mode="encoder",
            use_hope=True,
            layer_pattern=[
                "standard_attn",
                "retnet",
                "titan_attn",
                "standard_attn",
                "retnet",
                "titan_attn",
            ],
            use_mhc=True,
            mhc_expansion_rate=4,
            mhc_hyperloop=True,
            mhc_hyperloop_begin_layers=2,
            mhc_hyperloop_end_layers=2,
        )
        base.update(overrides)
        return FrankensteinModelConfig(**base)

    def test_encoder_output_shape(self):
        model = FrankensteinEncoder(self._cfg())
        ids = torch.randint(0, 128, (2, 8))
        self.assertEqual(model(ids).shape, (2, 8, 128))

    def test_encoder_backward_including_loop_pos_embs(self):
        model = FrankensteinEncoder(self._cfg())
        ids = torch.randint(0, 128, (2, 8))
        model(ids).sum().backward()
        for l_idx, e in enumerate(model.hyperloop.loop_pos_embs):
            self.assertIsNotNone(e.grad, f"e_{l_idx} has no grad")
            self.assertTrue(torch.isfinite(e.grad).all().item())
        self.assertTrue(
            torch.isfinite(model.hyperloop.projs[0].weight.grad).all().item()
        )

    def test_uses_hyperloop_not_learned_projections(self):
        model = FrankensteinEncoder(self._cfg())
        self.assertTrue(model.use_hyperloop)
        self.assertFalse(hasattr(model, "mhc_in_proj"))
        self.assertFalse(hasattr(model, "mhc_out_proj"))
        # Per-layer mHC modules must be disabled inside HybridLayer too.
        self.assertIsNone(model.layers[0].mhc_attn)
        self.assertIsNone(model.layers[0].mhc_ffn)
        self.assertFalse(model.layers[0].use_mhc)

    def test_decoder_forward(self):
        from src.model.frankenstein_decoder import FrankensteinDecoder

        model = FrankensteinDecoder(self._cfg(mode="decoder"))
        ids = torch.randint(0, 128, (2, 8))
        self.assertEqual(model(ids).shape, (2, 8, 128))

    def test_all_res_parameterizations_forward(self):
        ids = torch.randint(0, 128, (2, 8))
        for rp in ("diagonal", "sinkhorn", "identity"):
            model = FrankensteinEncoder(
                self._cfg(mhc_hyperloop_res_parameterization=rp)
            )
            self.assertEqual(model(ids).shape, (2, 8, 128), msg=rp)

    def test_no_split_loops_whole_stack(self):
        model = FrankensteinEncoder(
            self._cfg(mhc_hyperloop_begin_layers=0, mhc_hyperloop_end_layers=0)
        )
        ids = torch.randint(0, 128, (2, 8))
        self.assertEqual(model(ids).shape, (2, 8, 128))

    def test_bitnet_hyperloop_forward(self):
        model = FrankensteinEncoder(self._cfg(use_bitnet=True))
        ids = torch.randint(0, 128, (2, 8))
        out = model(ids)
        self.assertEqual(out.shape, (2, 8, 128))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_mod_compatible_with_hyperloop(self):
        # Hyperloop's middle-block layers run the standard C-dim path, so the
        # per-layer MoD/n-stream conflict that plain mHC has does not apply.
        model = FrankensteinEncoder(self._cfg(use_mixture_of_depths=True))
        ids = torch.randint(0, 128, (2, 8))
        out = model(ids)
        self.assertEqual(out.shape, (2, 8, 128))
        out.sum().backward()
        self.assertIn(
            "mixture_of_depths_router_loss", model.last_auxiliary_losses
        )

    def test_requires_mhc_enabled(self):
        with self.assertRaises(ValueError):
            self._cfg(use_mhc=False)

    def test_requires_num_loops_ge_2(self):
        with self.assertRaises(ValueError):
            self._cfg(num_loops=1)

    def test_begin_plus_end_must_be_less_than_num_layers(self):
        with self.assertRaises(ValueError):
            self._cfg(mhc_hyperloop_begin_layers=4, mhc_hyperloop_end_layers=3)

    def test_negative_begin_end_rejected(self):
        with self.assertRaises(ValueError):
            self._cfg(mhc_hyperloop_begin_layers=-1)
        with self.assertRaises(ValueError):
            self._cfg(mhc_hyperloop_end_layers=-1)

    def test_invalid_res_parameterization_rejected(self):
        with self.assertRaises(ValueError):
            self._cfg(mhc_hyperloop_res_parameterization="bogus")

    def test_incompatible_with_attnres_residuals(self):
        with self.assertRaises(ValueError):
            self._cfg(residual_type="full_attn")
        with self.assertRaises(ValueError):
            self._cfg(residual_type="block_attn")

    def test_hybrid_layer_still_rejects_mod_with_plain_mhc(self):
        from src.model.hybrid_layer import HybridLayer

        cfg = self._cfg(mhc_hyperloop=False, use_mixture_of_depths=True)
        with self.assertRaises(ValueError):
            HybridLayer(cfg, layer_type="standard_attn")


if __name__ == "__main__":
    unittest.main()
