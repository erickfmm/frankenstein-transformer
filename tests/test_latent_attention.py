"""Unit tests for the latent / KV-compression attention family.

Covers the nine latent mixers introduced in the latent/ subpackage:
MLA, GQLA, MLRA, Tucker, IHA, GTA, MTLA, CCA, CCGQA (arXiv:2506.09342,
2605.15250, 2603.02188, 2603.30033, 2602.21371, 2506.17286, 2505.13544,
2510.04476).
"""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from src.model.tormented_bert_frankestein import UltraConfig
    from src.model.attention.latent import (
        MLAAttention,
        GQLAAttention,
        MLRAAttention,
        TuckerAttention,
        IHAAttention,
        GTAAttention,
        MTLAAttention,
        CCAAttention,
        CCGQAAttention,
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
class MLAAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = MLAAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = MLAAttention(_cfg(mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = MLAAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_hidden_raises(self):
        with self.assertRaises(ValueError):
            MLAAttention(_cfg(hidden_size=50, num_heads=6))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GQLAAttentionTests(unittest.TestCase):
    def test_output_shape_gqa_path(self):
        attn = GQLAAttention(_cfg(gqla_decode_path="gqa"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_output_shape_mqa_path(self):
        attn = GQLAAttention(_cfg(gqla_decode_path="mqa_absorb"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = GQLAAttention(_cfg(mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = GQLAAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_decode_path_raises(self):
        with self.assertRaises(ValueError):
            GQLAAttention(_cfg(gqla_decode_path="bogus"))

    def test_invalid_num_groups_raises(self):
        with self.assertRaises(ValueError):
            GQLAAttention(_cfg(gqla_num_groups=5))  # 5 does not divide 6


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class MLRAAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = MLRAAttention(_cfg(mlra_num_latent_heads=2))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = MLRAAttention(_cfg(mlra_num_latent_heads=2, mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = MLRAAttention(_cfg(mlra_num_latent_heads=2))
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_latent_heads_raises(self):
        with self.assertRaises(ValueError):
            MLRAAttention(_cfg(mlra_latent_rank=24, mlra_num_latent_heads=5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class TuckerAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = TuckerAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = TuckerAttention(_cfg(mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = TuckerAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_hidden_raises(self):
        with self.assertRaises(ValueError):
            TuckerAttention(_cfg(hidden_size=50, num_heads=6))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class IHAAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = IHAAttention(_cfg(iha_num_pseudo_heads=6))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = IHAAttention(_cfg(iha_num_pseudo_heads=6, mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = IHAAttention(_cfg(iha_num_pseudo_heads=6))
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_hidden_raises(self):
        with self.assertRaises(ValueError):
            IHAAttention(_cfg(hidden_size=50, num_heads=6))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GTAAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = GTAAttention(_cfg(gta_num_shared_groups=3))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = GTAAttention(_cfg(gta_num_shared_groups=3, mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = GTAAttention(_cfg(gta_num_shared_groups=3))
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_groups_raises(self):
        with self.assertRaises(ValueError):
            GTAAttention(_cfg(gta_num_shared_groups=4))  # 4 does not divide 6


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class MTLAAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = MTLAAttention(_cfg(mtla_merge_factor=2, mtla_stride=2))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = MTLAAttention(_cfg(mtla_merge_factor=2, mtla_stride=2, mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = MTLAAttention(_cfg(mtla_merge_factor=2, mtla_stride=2))
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_merge_factor_raises(self):
        with self.assertRaises(ValueError):
            MTLAAttention(_cfg(mtla_merge_factor=0))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class CCAAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = CCAAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_output_shape_no_conv(self):
        attn = CCAAttention(_cfg(cca_num_conv_layers=0))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_output_shape_one_conv(self):
        attn = CCAAttention(_cfg(cca_num_conv_layers=1))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_output_shape_no_qk_mean(self):
        attn = CCAAttention(_cfg(cca_qk_mean=False))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_output_shape_no_value_shift(self):
        attn = CCAAttention(_cfg(cca_value_shift=False))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = CCAAttention(_cfg(mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = CCAAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_latent_rank_raises(self):
        with self.assertRaises(ValueError):
            CCAAttention(_cfg(cca_latent_rank=25))  # 25 not divisible by 6

    def test_invalid_num_conv_layers_raises(self):
        with self.assertRaises(ValueError):
            CCAAttention(_cfg(cca_num_conv_layers=3))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class CCGQAAttentionTests(unittest.TestCase):
    """CCGQA tests.  The test config uses hidden=48, heads=6; we set
    ``ccgqa_query_latent_rank=24`` (d_h=4), ``ccgqa_kv_latent_rank=8``
    (d_h=4), ``ccgqa_num_kv_heads=2`` so the per-head dims match."""

    def test_output_shape(self):
        attn = CCGQAAttention(_cfg(
            ccgqa_query_latent_rank=24, ccgqa_kv_latent_rank=8, ccgqa_num_kv_heads=2,
        ))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_output_shape_no_conv(self):
        attn = CCGQAAttention(_cfg(
            ccgqa_query_latent_rank=24, ccgqa_kv_latent_rank=8, ccgqa_num_kv_heads=2,
            ccgqa_num_conv_layers=0,
        ))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_output_shape_no_value_shift(self):
        attn = CCGQAAttention(_cfg(
            ccgqa_query_latent_rank=24, ccgqa_kv_latent_rank=8, ccgqa_num_kv_heads=2,
            ccgqa_value_shift=False,
        ))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = CCGQAAttention(_cfg(
            ccgqa_query_latent_rank=24, ccgqa_kv_latent_rank=8, ccgqa_num_kv_heads=2,
            mode="decoder",
        ))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = CCGQAAttention(_cfg(
            ccgqa_query_latent_rank=24, ccgqa_kv_latent_rank=8, ccgqa_num_kv_heads=2,
        ))
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_kv_heads_raises(self):
        with self.assertRaises(ValueError):
            CCGQAAttention(_cfg(
                ccgqa_query_latent_rank=24, ccgqa_kv_latent_rank=8, ccgqa_num_kv_heads=4,
            ))  # 4 does not divide 6

    def test_dh_mismatch_raises(self):
        # query_latent/num_heads must equal kv_latent/num_kv_heads
        with self.assertRaises(ValueError):
            CCGQAAttention(_cfg(
                ccgqa_query_latent_rank=48,
                ccgqa_kv_latent_rank=24,
                ccgqa_num_kv_heads=2,
            ))  # 48/6=8 != 24/2=12

    def test_kv_larger_than_query_raises(self):
        with self.assertRaises(ValueError):
            CCGQAAttention(_cfg(
                ccgqa_query_latent_rank=12,
                ccgqa_kv_latent_rank=24,
                ccgqa_num_kv_heads=6,
            ))  # kv > query


if __name__ == "__main__":
    unittest.main()