"""Unit tests for the latent / KV-compression attention family.

Covers the ten latent mixers introduced in the latent/ subpackage:
MLA, GQLA, MLRA, Tucker, IHA, GTA, MTLA, CCA, CCGQA, KPA, TPA
(arXiv:2506.09342, 2605.15250, 2603.02188, 2603.30033, 2602.21371,
2506.17286, 2505.13544, 2510.04476, 2610.12657, 2501.06425).
"""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from src.model.config import FrankensteinModelConfig, merge_model_blocks
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
        KPAAttention,
        TPAAttention,
        MFAAttention,
    )


def _cfg(**overrides):
    base = {"dims": {"vocab_size": 64, "hidden_size": 48, "num_layers": 1, "num_loops": 1, "num_heads": 6, "dropout": 0.0, "layer_pattern": ['standard'], "mode": 'encoder'}, "attention": {"retnet": {"heads": 6}, "ode": {"solver": 'rk4', "steps": 1}}, "moe": {"num_experts": 2, "top_k": 1, "enabled": False}, "norm": {"type": 'layer_norm'}, "bitnet": {"enabled": False}, "ffn": {"hidden_size": 96, "activation": 'gelu'}, "positional_encoding": 'hope'}
    base = merge_model_blocks(base, overrides)
    return FrankensteinModelConfig.from_dict({**base})


BSZ, SEQ, DIM = 2, 8, 48


def _x():
    return torch.randn(BSZ, SEQ, DIM)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class MLAAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = MLAAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = MLAAttention(_cfg(dims={'mode': 'decoder'}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = MLAAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_hidden_raises(self):
        with self.assertRaises(ValueError):
            MLAAttention(_cfg(dims={'hidden_size': 50, 'num_heads': 6}))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GQLAAttentionTests(unittest.TestCase):
    def test_output_shape_gqa_path(self):
        attn = GQLAAttention(_cfg(attention={'gqla': {'decode_path': 'gqa'}}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_output_shape_mqa_path(self):
        attn = GQLAAttention(_cfg(attention={'gqla': {'decode_path': 'mqa_absorb'}}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = GQLAAttention(_cfg(dims={'mode': 'decoder'}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = GQLAAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_decode_path_raises(self):
        with self.assertRaises(ValueError):
            GQLAAttention(_cfg(attention={'gqla': {'decode_path': 'bogus'}}))

    def test_invalid_num_groups_raises(self):
        with self.assertRaises(ValueError):
            GQLAAttention(_cfg(attention={'gqla': {'num_groups': 5}}))  # 5 does not divide 6


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class MLRAAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = MLRAAttention(_cfg(attention={'mlra': {'num_latent_heads': 2}}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = MLRAAttention(_cfg(attention={'mlra': {'num_latent_heads': 2}}, dims={'mode': 'decoder'}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = MLRAAttention(_cfg(attention={'mlra': {'num_latent_heads': 2}}))
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_latent_heads_raises(self):
        with self.assertRaises(ValueError):
            MLRAAttention(_cfg(attention={'mlra': {'latent_rank': 24, 'num_latent_heads': 5}}))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class TuckerAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = TuckerAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = TuckerAttention(_cfg(dims={'mode': 'decoder'}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = TuckerAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_hidden_raises(self):
        with self.assertRaises(ValueError):
            TuckerAttention(_cfg(dims={'hidden_size': 50, 'num_heads': 6}))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class IHAAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = IHAAttention(_cfg(attention={'iha': {'num_pseudo_heads': 6}}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = IHAAttention(_cfg(attention={'iha': {'num_pseudo_heads': 6}}, dims={'mode': 'decoder'}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = IHAAttention(_cfg(attention={'iha': {'num_pseudo_heads': 6}}))
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_hidden_raises(self):
        with self.assertRaises(ValueError):
            IHAAttention(_cfg(dims={'hidden_size': 50, 'num_heads': 6}))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GTAAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = GTAAttention(_cfg(attention={'gta': {'num_shared_groups': 3}}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = GTAAttention(_cfg(attention={'gta': {'num_shared_groups': 3}}, dims={'mode': 'decoder'}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = GTAAttention(_cfg(attention={'gta': {'num_shared_groups': 3}}))
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_groups_raises(self):
        with self.assertRaises(ValueError):
            GTAAttention(_cfg(attention={'gta': {'num_shared_groups': 4}}))  # 4 does not divide 6


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class MTLAAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = MTLAAttention(_cfg(attention={'mtla': {'merge_factor': 2, 'stride': 2}}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = MTLAAttention(_cfg(attention={'mtla': {'merge_factor': 2, 'stride': 2}}, dims={'mode': 'decoder'}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = MTLAAttention(_cfg(attention={'mtla': {'merge_factor': 2, 'stride': 2}}))
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_merge_factor_raises(self):
        with self.assertRaises(ValueError):
            MTLAAttention(_cfg(attention={'mtla': {'merge_factor': 0}}))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class CCAAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = CCAAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_output_shape_no_conv(self):
        attn = CCAAttention(_cfg(attention={'cca': {'num_conv_layers': 0}}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_output_shape_one_conv(self):
        attn = CCAAttention(_cfg(attention={'cca': {'num_conv_layers': 1}}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_output_shape_no_qk_mean(self):
        attn = CCAAttention(_cfg(attention={'cca': {'qk_mean': False}}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_output_shape_no_value_shift(self):
        attn = CCAAttention(_cfg(attention={'cca': {'value_shift': False}}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = CCAAttention(_cfg(dims={'mode': 'decoder'}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = CCAAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_latent_rank_raises(self):
        with self.assertRaises(ValueError):
            CCAAttention(_cfg(attention={'cca': {'latent_rank': 25}}))  # 25 not divisible by 6

    def test_invalid_num_conv_layers_raises(self):
        with self.assertRaises(ValueError):
            CCAAttention(_cfg(attention={'cca': {'num_conv_layers': 3}}))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class CCGQAAttentionTests(unittest.TestCase):
    """CCGQA tests.  The test config uses hidden=48, heads=6; we set
    ``ccgqa_query_latent_rank=24`` (d_h=4), ``ccgqa_kv_latent_rank=8``
    (d_h=4), ``ccgqa_num_kv_heads=2`` so the per-head dims match."""

    def test_output_shape(self):
        attn = CCGQAAttention(_cfg(attention={'ccgqa': {'query_latent_rank': 24, 'kv_latent_rank': 8, 'num_kv_heads': 2}}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_output_shape_no_conv(self):
        attn = CCGQAAttention(_cfg(attention={'ccgqa': {'query_latent_rank': 24, 'kv_latent_rank': 8, 'num_kv_heads': 2, 'num_conv_layers': 0}}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_output_shape_no_value_shift(self):
        attn = CCGQAAttention(_cfg(attention={'ccgqa': {'query_latent_rank': 24, 'kv_latent_rank': 8, 'num_kv_heads': 2, 'value_shift': False}}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = CCGQAAttention(_cfg(attention={'ccgqa': {'query_latent_rank': 24, 'kv_latent_rank': 8, 'num_kv_heads': 2}}, dims={'mode': 'decoder'}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = CCGQAAttention(_cfg(attention={'ccgqa': {'query_latent_rank': 24, 'kv_latent_rank': 8, 'num_kv_heads': 2}}))
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_kv_heads_raises(self):
        with self.assertRaises(ValueError):
            CCGQAAttention(_cfg(attention={'ccgqa': {'query_latent_rank': 24, 'kv_latent_rank': 8, 'num_kv_heads': 4}}))  # 4 does not divide 6

    def test_dh_mismatch_raises(self):
        # query_latent/num_heads must equal kv_latent/num_kv_heads
        with self.assertRaises(ValueError):
            CCGQAAttention(_cfg(attention={'ccgqa': {'query_latent_rank': 48, 'kv_latent_rank': 24, 'num_kv_heads': 2}}))  # 48/6=8 != 24/2=12

    def test_kv_larger_than_query_raises(self):
        with self.assertRaises(ValueError):
            CCGQAAttention(_cfg(attention={'ccgqa': {'query_latent_rank': 12, 'kv_latent_rank': 24, 'num_kv_heads': 6}}))  # kv > query


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class KPAAttentionTests(unittest.TestCase):
    """Kronecker Product Attention (arXiv:2610.12657)."""

    def test_output_shape(self):
        attn = KPAAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = KPAAttention(_cfg(dims={'mode': 'decoder'}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_explicit_knobs(self):
        attn = KPAAttention(_cfg(attention={'kpa': {
            'h1': 2, 'h2': 3, 'd1': 1, 'd2': 8,
            'query_rank': 4, 'key_rank': 2, 'value_rank': 3,
        }}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))
        self.assertEqual((attn.h1, attn.h2, attn.d1, attn.d2), (2, 3, 1, 8))

    def test_auto_split_defaults(self):
        attn = KPAAttention(_cfg())
        self.assertEqual(attn.h1 * attn.h2, attn.num_heads)
        self.assertEqual(attn.d1 * attn.d2, attn.head_dim)
        # 6 heads -> largest proper divisor 3; head_dim 8 -> d1 = 8, d2 = 1
        self.assertEqual((attn.h1, attn.h2), (3, 2))
        self.assertEqual((attn.d1, attn.d2), (8, 1))

    def test_invalid_head_split_raises(self):
        with self.assertRaises(ValueError):
            KPAAttention(_cfg(attention={'kpa': {'h1': 4, 'h2': 4}}))  # 16 != 6

    def test_invalid_feature_split_raises(self):
        with self.assertRaises(ValueError):
            KPAAttention(_cfg(attention={'kpa': {'d1': 3, 'd2': 3}}))  # 9 != 8

    def test_invalid_rank_raises(self):
        with self.assertRaises(ValueError):
            KPAAttention(_cfg(attention={'kpa': {'key_rank': 0}}))

    def test_no_nan_inf(self):
        attn = KPAAttention(_cfg(dims={'mode': 'decoder'}))
        out = attn(_x())
        self.assertTrue(torch.isfinite(out).all())

    def test_gradient_flows(self):
        attn = KPAAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_param_count_formula(self):
        cfg = _cfg()
        attn = KPAAttention(cfg)
        h = cfg.dims.hidden_size
        s = attn.h1 * attn.d1 + attn.h2 * attn.d2
        expected = h * (attn.query_rank + attn.key_rank + attn.value_rank) * s + h * h
        actual = sum(p.numel() for p in attn.parameters())
        self.assertEqual(actual, expected)

    def test_bitnet_enabled_forward(self):
        attn = KPAAttention(_cfg(bitnet={'enabled': True}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class TPAAttentionTests(unittest.TestCase):
    """Tensor Product Attention (arXiv:2501.06425)."""

    def test_output_shape(self):
        attn = TPAAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = TPAAttention(_cfg(dims={'mode': 'decoder'}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_explicit_knobs(self):
        attn = TPAAttention(_cfg(attention={'tpa': {
            'query_rank': 4, 'key_rank': 1, 'value_rank': 3,
        }}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))
        self.assertEqual(
            (attn.query_rank, attn.key_rank, attn.value_rank), (4, 1, 3)
        )

    def test_default_ranks(self):
        attn = TPAAttention(_cfg())
        self.assertEqual(
            (attn.query_rank, attn.key_rank, attn.value_rank), (8, 2, 2)
        )

    def test_invalid_rank_raises(self):
        for knob in ("query_rank", "key_rank", "value_rank"):
            with self.assertRaises(ValueError):
                TPAAttention(_cfg(attention={'tpa': {knob: 0}}))

    def test_invalid_hidden_raises(self):
        with self.assertRaises(ValueError):
            TPAAttention(_cfg(dims={'hidden_size': 50, 'num_heads': 6}))

    def test_no_nan_inf(self):
        attn = TPAAttention(_cfg(dims={'mode': 'decoder'}))
        out = attn(_x())
        self.assertTrue(torch.isfinite(out).all())

    def test_gradient_flows(self):
        attn = TPAAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_param_count_formula(self):
        cfg = _cfg()
        attn = TPAAttention(cfg)
        h = cfg.dims.hidden_size
        d = attn.head_dim
        expected = (
            h * (attn.query_rank + attn.key_rank + attn.value_rank)
            * (attn.num_heads + d)
            + h * attn.num_heads * d
        )
        actual = sum(p.numel() for p in attn.parameters())
        self.assertEqual(actual, expected)

    def test_bitnet_enabled_forward(self):
        attn = TPAAttention(_cfg(bitnet={'enabled': True}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class MFAAttentionTests(unittest.TestCase):
    """Multi-matrix Factorization Attention (arXiv:2412.19255)."""

    def test_output_shape(self):
        attn = MFAAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = MFAAttention(_cfg(dims={'mode': 'decoder'}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_default_factor_rank(self):
        attn = MFAAttention(_cfg())
        self.assertEqual(attn.factor_rank, 128)
        self.assertFalse(attn.key_reuse)
        self.assertEqual(attn.num_kv_heads, 1)

    def test_explicit_knobs(self):
        attn = MFAAttention(_cfg(attention={'mfa': {
            'factor_rank': 16, 'key_reuse': True,
        }}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))
        self.assertEqual(attn.factor_rank, 16)
        self.assertTrue(attn.key_reuse)
        self.assertIsNone(attn.w_v_shared)

    def test_key_reuse_starts_as_identity(self):
        attn = MFAAttention(_cfg(attention={'mfa': {'key_reuse': True}}))
        keys = torch.randn(BSZ, SEQ, attn.factor_rank)
        self.assertTrue(torch.equal(attn._values_from_keys(keys), keys))

    def test_invalid_rank_raises(self):
        with self.assertRaises(ValueError):
            MFAAttention(_cfg(attention={'mfa': {'factor_rank': 0}}))

    def test_head_geometry_decoupled_from_hidden_size(self):
        # Paper Table 7: 18 heads of width 256 at hidden_size 2048 (2048 % 18 != 0).
        attn = MFAAttention(_cfg(dims={'hidden_size': 36, 'num_heads': 5}))
        self.assertEqual(attn.num_heads, 5)
        self.assertEqual(attn.head_dim, 128)
        out = attn(torch.randn(BSZ, SEQ, 36))
        self.assertEqual(out.shape, (BSZ, SEQ, 36))

    def test_no_nan_inf(self):
        attn = MFAAttention(_cfg(dims={'mode': 'decoder'}))
        self.assertTrue(torch.isfinite(attn(_x())).all())

    def test_gradient_flows(self):
        attn = MFAAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_key_reuse_gate_receives_gradient(self):
        attn = MFAAttention(_cfg(attention={'mfa': {'key_reuse': True}}))
        attn(_x()).sum().backward()
        self.assertIsNotNone(attn.key_reuse_gate.grad)
        self.assertGreater(float(attn.key_reuse_gate.grad.abs().sum()), 0.0)

    def test_param_count_formula(self):
        cfg = _cfg()
        attn = MFAAttention(cfg)
        h, m, c = cfg.dims.hidden_size, attn.num_heads, attn.factor_rank
        expected = 3 * h * c + m * c * c + m * c * h
        self.assertEqual(sum(p.numel() for p in attn.parameters()), expected)

    def test_param_count_formula_key_reuse(self):
        cfg = _cfg(attention={'mfa': {'key_reuse': True}})
        attn = MFAAttention(cfg)
        h, m, c = cfg.dims.hidden_size, attn.num_heads, attn.factor_rank
        expected = 2 * h * c + m * c * c + m * c * h + c * c + c
        self.assertEqual(sum(p.numel() for p in attn.parameters()), expected)

    def test_bitnet_enabled_forward(self):
        attn = MFAAttention(_cfg(bitnet={'enabled': True}))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))


if __name__ == "__main__":
    unittest.main()