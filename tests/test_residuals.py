"""Unit tests for Attention Residuals (arXiv:2603.15031).

Covers:
    - ``StandardResidual`` / ``NoResidual`` stateless variants.
    - ``FullAttentionResidual`` (zero-init uniform averaging, gradient
      flow, mHC independent / joint modes, gradient checkpointing).
    - ``BlockAttentionResidual`` (block boundary, partial sums, zero-init
      uniform averaging, mHC modes).
    - ``build_residual`` factory and ``FrankensteinModelConfig`` wiring.
    - End-to-end encoder / decoder forward / backward passes with each
      residual type and with combined mHC + AttnRes.
"""
import unittest
from importlib.util import find_spec

import yaml

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_decoder import FrankensteinDecoder
    from src.model.frankenstein_encoder import FrankensteinEncoder
    from src.model.residuals import (
        BlockAttentionResidual,
        FullAttentionResidual,
        NoResidual,
        ResidualBase,
        StandardResidual,
        build_residual,
    )
    from src.utils.config_flatten import flatten_model_dict


def _base_kwargs(**overrides):
    """Return a minimal valid kwargs dict for FrankensteinModelConfig."""
    base = dict(
        vocab_size=32,
        hidden_size=16,
        num_layers=3,
        num_loops=1,
        num_heads=2,
        retention_heads=2,
        num_experts=2,
        top_k_experts=1,
    )
    base.update(overrides)
    return base


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ResidualBaseTests(unittest.TestCase):
    """Sanity-check the abstract base class lifecycle hooks."""

    def test_set_streams_clamps_to_min_one(self):
        res = StandardResidual(hidden_size=8)
        res.set_streams(0)
        self.assertEqual(res.n_streams, 1)

    def test_register_state_updates_num_layers(self):
        res = StandardResidual(hidden_size=8)
        res.register_state(12)
        self.assertEqual(res.num_layers, 12)

    def test_reset_state_preserves_allocation(self):
        res = FullAttentionResidual(hidden_size=8, num_layers=4)
        res.register_state(4)
        res._layer_outputs.append(torch.randn(1, 2, 8))
        res.reset_state()
        # Allocation stays; buffer clears.
        self.assertEqual(res.num_layers, 4)
        self.assertEqual(res._layer_outputs, [])


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class StandardResidualTests(unittest.TestCase):
    """Standard residual must be a stateless pass-through."""

    def test_passthrough(self):
        res = StandardResidual(hidden_size=8)
        x = torch.randn(2, 4, 8)
        out = res(0, x)
        torch.testing.assert_close(out, x)

    def test_stateless(self):
        res = StandardResidual(hidden_size=8)
        # Calling reset_state multiple times should not fail.
        res.reset_state()
        res.register_state(5)
        res.reset_state()


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class NoResidualTests(unittest.TestCase):
    """No-residual is a stateless pass-through marker."""

    def test_passthrough(self):
        res = NoResidual(hidden_size=8)
        x = torch.randn(2, 4, 8)
        out = res(0, x)
        torch.testing.assert_close(out, x)
        self.assertTrue(res.is_no_residual)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class FullAttentionResidualTests(unittest.TestCase):
    """Mathematical and gradient-flow tests for Full AttnRes."""

    def _build(self, **kwargs):
        defaults = dict(hidden_size=8, num_layers=3)
        defaults.update(kwargs)
        return FullAttentionResidual(**defaults)

    def test_zero_init_uniform_average(self):
        """At zero-init the attention weights are uniform, so the output
        should equal the arithmetic mean of all sources."""
        torch.manual_seed(42)
        res = self._build(init_query_zero=True, use_rmsnorm_keys=False)
        res.register_state(num_layers=3)
        emb = torch.randn(1, 2, 8)
        lo1 = torch.randn(1, 2, 8)
        lo2 = torch.randn(1, 2, 8)
        res.set_embedding(emb)

        out0 = res(0, lo1)
        torch.testing.assert_close(out0, (emb + lo1) / 2, atol=1e-5, rtol=1e-5)

        out1 = res(1, lo2)
        expected = (emb + lo1 + lo2) / 3
        torch.testing.assert_close(out1, expected, atol=1e-5, rtol=1e-5)

    def test_rmsnorm_keys_changes_output(self):
        """RMSNorm on keys should change the output for non-uniform magnitudes."""
        torch.manual_seed(0)
        res_rms = self._build(init_query_zero=False, use_rmsnorm_keys=True)
        res_no = self._build(init_query_zero=False, use_rmsnorm_keys=False)
        for r in (res_rms, res_no):
            r.register_state(num_layers=2)
            r.set_embedding(torch.randn(1, 2, 8))

        lo = torch.randn(1, 2, 8) * 100  # large magnitude to break uniform averaging
        torch.manual_seed(1)
        r1 = res_rms(0, lo)
        torch.manual_seed(1)
        r2 = res_no(0, lo)
        # The two should differ whenever the magnitudes are skewed.
        self.assertFalse(torch.allclose(r1, r2))

    def test_requires_embedding(self):
        res = self._build()
        res.register_state(num_layers=2)
        with self.assertRaises(RuntimeError):
            res(0, torch.zeros(1, 2, 8))

    def test_gradient_flows(self):
        """Backward pass must populate query_weight gradients."""
        torch.manual_seed(0)
        res = self._build(init_query_zero=False, use_rmsnorm_keys=True)
        res.register_state(num_layers=3)
        res.set_embedding(torch.randn(2, 4, 8))
        out = res(0, torch.randn(2, 4, 8, requires_grad=False))
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(res.query_weight.grad)
        self.assertTrue(torch.isfinite(res.query_weight.grad).all().item())

    def test_mhc_independent_stream_mode(self):
        """When mHC is on (n_streams > 1) the independent mode keeps the
        ``n``-stream shape and runs attention per stream."""
        torch.manual_seed(0)
        res = self._build()
        res.set_streams(n_streams=2)
        res.register_state(num_layers=2)
        emb = torch.randn(1, 2, 2, 8)
        lo = torch.randn(1, 2, 2, 8)
        res.set_embedding(emb)
        out = res(0, lo)
        self.assertEqual(out.shape, (1, 2, 2, 8))

    def test_mhc_joint_stream_mode(self):
        """Joint mode also keeps the ``n``-stream shape."""
        torch.manual_seed(0)
        res = self._build(mhc_stream_mode="joint")
        res.set_streams(n_streams=3)
        res.register_state(num_layers=2)
        emb = torch.randn(1, 2, 3, 8)
        lo = torch.randn(1, 2, 3, 8)
        res.set_embedding(emb)
        out = res(0, lo)
        self.assertEqual(out.shape, (1, 2, 3, 8))

    def test_layer_idx_out_of_range(self):
        res = self._build(num_layers=2)
        res.register_state(num_layers=2)
        res.set_embedding(torch.zeros(1, 2, 8))
        with self.assertRaises(IndexError):
            res(5, torch.zeros(1, 2, 8))

    def test_gradient_checkpoint(self):
        """Gradient checkpointing should still let gradients reach the
        query_weight parameter."""
        torch.manual_seed(0)
        res = self._build(init_query_zero=False, gradient_checkpoint=True)
        res.register_state(num_layers=2)
        res.set_embedding(torch.randn(1, 2, 8))
        res.train()
        out = res(0, torch.randn(1, 2, 8))
        out.sum().backward()
        self.assertIsNotNone(res.query_weight.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BlockAttentionResidualTests(unittest.TestCase):
    """Block-wise attention should respect block boundaries."""

    def _build(self, **kwargs):
        defaults = dict(hidden_size=8, num_layers=4, num_blocks=2)
        defaults.update(kwargs)
        return BlockAttentionResidual(**defaults)

    def test_block_size_computation(self):
        res = self._build(num_layers=8, num_blocks=3)
        # ceil(8/3) = 3 layers per block.
        self.assertEqual(res.block_size, 3)

    def test_zero_init_uniform_average(self):
        """At zero-init the per-layer attention is uniform over the
        available sources (block sums + partial sum + embedding)."""
        torch.manual_seed(0)
        res = self._build(init_query_zero=True, use_rmsnorm_keys=False)
        res.register_state(num_layers=4)
        emb = torch.randn(1, 2, 8)
        res.set_embedding(emb)

        # First layer: only the embedding is available as a source (the
        # partial sum is None until after the call). Softmax over a single
        # source trivially picks that source.
        lo1 = torch.randn(1, 2, 8)
        out0 = res(0, lo1)
        torch.testing.assert_close(out0, emb, atol=1e-5, rtol=1e-5)

        # Second layer: sources are [emb, partial_sum (=lo1)] — uniform
        # averaging gives (emb + lo1) / 2.
        lo2 = torch.randn(1, 2, 8)
        out1 = res(1, lo2)
        torch.testing.assert_close(out1, (emb + lo1) / 2, atol=1e-5, rtol=1e-5)

    def test_block_sealing(self):
        """After block_size layers the partial sum should be sealed and
        appended to ``_block_sums``."""
        res = self._build(num_layers=4, num_blocks=2)
        res.register_state(num_layers=4)
        res.set_embedding(torch.zeros(1, 2, 8))
        for i in range(res.block_size):
            res(i, torch.randn(1, 2, 8))
        self.assertEqual(len(res._block_sums), 1)

    def test_invalid_num_blocks(self):
        with self.assertRaises(ValueError):
            BlockAttentionResidual(hidden_size=8, num_layers=2, num_blocks=8)

    def test_gradient_flows(self):
        torch.manual_seed(0)
        res = self._build(init_query_zero=False)
        res.register_state(num_layers=4)
        res.set_embedding(torch.randn(1, 2, 8))
        for i in range(2):
            out = res(i, torch.randn(1, 2, 8))
        out.sum().backward()
        self.assertIsNotNone(res.query_weight.grad)
        self.assertTrue(torch.isfinite(res.query_weight.grad).all().item())

    def test_mhc_nstream(self):
        torch.manual_seed(0)
        res = self._build()
        res.set_streams(n_streams=2)
        res.register_state(num_layers=4)
        emb = torch.randn(1, 2, 2, 8)
        lo = torch.randn(1, 2, 2, 8)
        res.set_embedding(emb)
        out = res(0, lo)
        self.assertEqual(out.shape, (1, 2, 2, 8))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class FactoryTests(unittest.TestCase):
    """``build_residual`` selects the correct class from config."""

    def test_factory_returns_correct_classes(self):
        for rt, expected in [
            ("standard", StandardResidual),
            ("none", NoResidual),
            ("full_attn", FullAttentionResidual),
            ("block_attn", BlockAttentionResidual),
        ]:
            cfg = FrankensteinModelConfig(
                **_base_kwargs(residual_type=rt, block_attn_num_blocks=2)
            )
            res = build_residual(cfg)
            self.assertIsInstance(res, expected)

    def test_invalid_residual_type_raises_in_config(self):
        with self.assertRaises(ValueError):
            FrankensteinModelConfig(**_base_kwargs(residual_type="bogus"))

    def test_block_attn_validation(self):
        with self.assertRaises(ValueError):
            FrankensteinModelConfig(
                **_base_kwargs(
                    residual_type="block_attn", num_layers=2, block_attn_num_blocks=8
                )
            )

    def test_factory_propagates_mhc_nstreams(self):
        cfg = FrankensteinModelConfig(
            **_base_kwargs(
                residual_type="full_attn",
                use_mhc=True,
                mhc_expansion_rate=4,
            )
        )
        res = build_residual(cfg)
        self.assertEqual(res.n_streams, 4)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ConfigFlattenTests(unittest.TestCase):
    """The new ``residuals`` sub-tree must flatten into flat kwargs."""

    def test_full_attn_flatten(self):
        nested = {
            "residuals": {
                "type": "full_attn",
                "full_attn": {
                    "init_query_zero": True,
                    "use_rmsnorm_keys": False,
                },
                "mhc_stream_mode": "joint",
                "gradient_checkpoint": True,
            }
        }
        flat = flatten_model_dict(nested)
        self.assertEqual(flat["residual_type"], "full_attn")
        self.assertTrue(flat["full_attn_init_query_zero"])
        self.assertFalse(flat["full_attn_use_rmsnorm_keys"])
        self.assertEqual(flat["attnres_mhc_stream_mode"], "joint")
        self.assertTrue(flat["attnres_gradient_checkpoint"])

    def test_block_attn_flatten(self):
        nested = {
            "residuals": {
                "type": "block_attn",
                "block_attn": {
                    "num_blocks": 6,
                    "init_query_zero": True,
                    "use_rmsnorm_keys": True,
                },
            }
        }
        flat = flatten_model_dict(nested)
        self.assertEqual(flat["residual_type"], "block_attn")
        self.assertEqual(flat["block_attn_num_blocks"], 6)
        self.assertTrue(flat["block_attn_init_query_zero"])
        self.assertTrue(flat["block_attn_use_rmsnorm_keys"])

    def test_residuals_passthrough_with_other_keys(self):
        nested = {
            "use_bitnet": True,
            "residuals": {"type": "standard"},
        }
        flat = flatten_model_dict(nested)
        self.assertTrue(flat["use_bitnet"])
        self.assertEqual(flat["residual_type"], "standard")


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class EndToEndEncoderTests(unittest.TestCase):
    """Run the full encoder forward/backward for each residual type."""

    def _cfg(self, **overrides):
        kwargs = _base_kwargs(num_layers=3, num_loops=1, block_attn_num_blocks=2)
        kwargs.update(overrides)
        return FrankensteinModelConfig(**kwargs)

    def test_standard_residual_forward_backward(self):
        enc = FrankensteinEncoder(self._cfg(residual_type="standard")).train()
        ids = torch.randint(0, 32, (2, 6))
        out = enc(ids)
        self.assertEqual(out.shape, (2, 6, 32))
        out.sum().backward()
        # Should have at least one gradient in the residual module (none here,
        # but the encoder/layer params should be updated).
        self.assertTrue(any(p.grad is not None for p in enc.parameters() if p.requires_grad))

    def test_no_residual_forward(self):
        enc = FrankensteinEncoder(self._cfg(residual_type="none")).eval()
        ids = torch.randint(0, 32, (2, 6))
        out = enc(ids)
        self.assertEqual(out.shape, (2, 6, 32))

    def test_full_attn_residual_forward_backward(self):
        torch.manual_seed(0)
        enc = FrankensteinEncoder(self._cfg(residual_type="full_attn")).train()
        ids = torch.randint(0, 32, (2, 6))
        out = enc(ids)
        self.assertEqual(out.shape, (2, 6, 32))
        out.sum().backward()
        # Query weights should have gradients.
        self.assertIsNotNone(enc.residual.query_weight.grad)

    def test_block_attn_residual_forward_backward(self):
        torch.manual_seed(0)
        enc = FrankensteinEncoder(self._cfg(residual_type="block_attn")).train()
        ids = torch.randint(0, 32, (2, 6))
        out = enc(ids)
        self.assertEqual(out.shape, (2, 6, 32))
        out.sum().backward()
        self.assertIsNotNone(enc.residual.query_weight.grad)

    def test_mhc_plus_full_attn(self):
        torch.manual_seed(0)
        enc = FrankensteinEncoder(
            self._cfg(
                residual_type="full_attn",
                use_mhc=True,
                mhc_expansion_rate=2,
                attnres_mhc_stream_mode="independent",
            )
        ).train()
        ids = torch.randint(0, 32, (2, 6))
        out = enc(ids)
        self.assertEqual(out.shape, (2, 6, 32))
        out.sum().backward()

    def test_mhc_plus_block_attn(self):
        torch.manual_seed(0)
        enc = FrankensteinEncoder(
            self._cfg(
                residual_type="block_attn",
                use_mhc=True,
                mhc_expansion_rate=2,
                attnres_mhc_stream_mode="joint",
            )
        ).train()
        ids = torch.randint(0, 32, (2, 6))
        out = enc(ids)
        self.assertEqual(out.shape, (2, 6, 32))
        out.sum().backward()

    def test_mod_plus_full_attn(self):
        """MoD + AttnRes: only selected tokens contribute to the layer-output
        buffer / partial sum, but the model should still train end-to-end."""
        torch.manual_seed(0)
        enc = FrankensteinEncoder(
            self._cfg(
                residual_type="full_attn",
                use_mixture_of_depths=True,
                mixture_of_depths_capacity_ratio=0.5,
                num_layers=4,
            )
        ).train()
        ids = torch.randint(0, 32, (2, 8))
        out = enc(ids)
        self.assertEqual(out.shape, (2, 8, 32))
        loss = out.sum()
        loss.backward()
        self.assertIn("mixture_of_depths_router_loss", enc.last_auxiliary_losses)

    def test_residual_module_attribute_present(self):
        enc = FrankensteinEncoder(self._cfg(residual_type="standard"))
        self.assertIsInstance(enc.residual, ResidualBase)
        self.assertFalse(enc.residual.is_attn_res)
        enc2 = FrankensteinEncoder(self._cfg(residual_type="full_attn"))
        self.assertTrue(enc2.residual.is_attn_res)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class EndToEndDecoderTests(unittest.TestCase):
    """Decoder-side forward and backward passes for each residual type."""

    def _cfg(self, **overrides):
        kwargs = _base_kwargs(
            num_layers=4,
            num_loops=1,
            layer_pattern=["titan_attn", "retnet", "titan_attn", "mamba"],
            mode="decoder",
            block_attn_num_blocks=2,
        )
        kwargs.update(overrides)
        return FrankensteinModelConfig(**kwargs)

    def test_full_attn_decoder(self):
        torch.manual_seed(0)
        dec = FrankensteinDecoder(self._cfg(residual_type="full_attn")).train()
        ids = torch.randint(0, 32, (2, 6))
        out = dec(ids)
        self.assertEqual(out.shape, (2, 6, 32))
        out.sum().backward()

    def test_block_attn_decoder(self):
        torch.manual_seed(0)
        dec = FrankensteinDecoder(self._cfg(residual_type="block_attn")).train()
        ids = torch.randint(0, 32, (2, 6))
        out = dec(ids)
        self.assertEqual(out.shape, (2, 6, 32))
        out.sum().backward()

    def test_block_attn_generate(self):
        """Generate should work end-to-end with Block AttnRes."""
        torch.manual_seed(0)
        dec = FrankensteinDecoder(self._cfg(residual_type="block_attn")).eval()
        ids = torch.randint(0, 32, (1, 4))
        out = dec.generate(ids, max_new_tokens=4)
        self.assertEqual(out.shape, (1, 8))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class YAMLExampleTests(unittest.TestCase):
    """The example YAMLs in ``configs/examples/`` must load cleanly."""

    EXAMPLES = [
        "configs/examples/full_attn_res_adamw.yaml",
        "configs/examples/block_attn_res_adamw.yaml",
        "configs/examples/mhc_full_attn_res_adamw.yaml",
    ]

    def test_examples_parse(self):
        from src.training.config_loader import load_training_config

        for path in self.EXAMPLES:
            loaded = load_training_config(path)
            self.assertIsNotNone(loaded.model_config)
            self.assertIn(
                loaded.model_config.residual_type,
                {"standard", "none", "full_attn", "block_attn"},
            )


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class SchemaValidationTests(unittest.TestCase):
    """Schema entry ``_model/_residuals.yaml`` must declare the right shape."""

    def test_schema_yaml_is_valid(self):
        with open("src/schema/_model/_residuals.yaml", "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.assertEqual(data["type"], "object")
        self.assertFalse(data["additionalProperties"])
        self.assertIn("type", data["properties"])
        self.assertEqual(
            set(data["properties"]["type"]["enum"]),
            {"standard", "none", "full_attn", "block_attn"},
        )


if __name__ == "__main__":
    unittest.main()
