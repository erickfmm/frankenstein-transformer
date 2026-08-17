"""Unit tests for positional encodings."""
from __future__ import annotations

import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from src.model.embeddings.rope import RoPE
    from src.model.embeddings.hope import HoPE
    from src.model.embeddings.nope import NoPE
    from src.model.embeddings.alibi import ALiBi
    from src.model.embeddings.pape import PaPE
    from src.model.embeddings.pape_efficient import PaPEEfficient
    from src.model.embeddings.pape_ri import PaPERI
    from src.model.embeddings.sinusoidal import SinusoidalAbsolute, SinusoidalRotary
    from src.model.embeddings.learned_absolute import LearnedAbsolutePE
    from src.model.embeddings.factory import build_pos_encoder
    from src.model.config import FrankensteinModelConfig


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class RoPETests(unittest.TestCase):
    def _make_input(self, bsz=2, heads=4, seq=8, head_dim=16):
        return torch.randn(bsz, heads, seq, head_dim)

    def test_output_shape_matches_input(self):
        rope = RoPE(head_dim=16)
        x = self._make_input()
        y = rope(x)
        self.assertEqual(y.shape, x.shape)

    def test_pair_dim_zero_returns_identity(self):
        # head_dim=1 → pair_dim=0, should return unchanged
        rope = RoPE(head_dim=1)
        x = self._make_input(head_dim=1)
        y = rope(x)
        self.assertTrue(torch.equal(y, x))

    def test_single_pair_dim_no_error(self):
        # head_dim=2 → pair_dim=1
        rope = RoPE(head_dim=2)
        x = self._make_input(head_dim=2)
        y = rope(x)
        self.assertEqual(y.shape, x.shape)

    def test_logical_layer_idx_accepted(self):
        rope = RoPE(head_dim=16)
        x = self._make_input()
        y = rope(x, logical_layer_idx=5)
        self.assertEqual(y.shape, x.shape)

    def test_dtype_preserved(self):
        rope = RoPE(head_dim=16)
        x = self._make_input().double()
        y = rope(x)
        self.assertEqual(y.dtype, torch.float64)

    def test_different_base_values(self):
        rope_small = RoPE(head_dim=16, base=1000.0)
        rope_large = RoPE(head_dim=16, base=100_000.0)
        x = self._make_input()
        y_small = rope_small(x)
        y_large = rope_large(x)
        # Different bases produce different outputs
        self.assertFalse(torch.allclose(y_small, y_large))

    def test_scaling_affects_output(self):
        rope1 = RoPE(head_dim=16, scaling=1.0)
        rope2 = RoPE(head_dim=16, scaling=2.0)
        x = self._make_input()
        y1 = rope1(x)
        y2 = rope2(x)
        self.assertFalse(torch.allclose(y1, y2))

    def test_gradient_flows(self):
        rope = RoPE(head_dim=16)
        x = self._make_input().requires_grad_(True)
        rope(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class HoPETests(unittest.TestCase):
    def _make_input(self, bsz=2, heads=4, seq=8, head_dim=16):
        return torch.randn(bsz, heads, seq, head_dim)

    def test_output_shape_matches_input(self):
        hope = HoPE(head_dim=16)
        x = self._make_input()
        y = hope(x)
        self.assertEqual(y.shape, x.shape)

    def test_pair_dim_zero_returns_identity(self):
        hope = HoPE(head_dim=1)
        x = self._make_input(head_dim=1)
        y = hope(x)
        self.assertTrue(torch.equal(y, x))

    def test_single_pair_dim_no_error(self):
        hope = HoPE(head_dim=2)
        x = self._make_input(head_dim=2)
        y = hope(x)
        self.assertEqual(y.shape, x.shape)

    def test_logical_layer_idx_changes_output(self):
        hope = HoPE(head_dim=16)
        x = self._make_input()
        y0 = hope(x, logical_layer_idx=0)
        y3 = hope(x, logical_layer_idx=3)
        # Different layer indices → different scaling → different output
        self.assertFalse(torch.allclose(y0, y3))

    def test_zero_logical_layer_idx(self):
        hope = HoPE(head_dim=16)
        x = self._make_input()
        y = hope(x, logical_layer_idx=0)
        self.assertEqual(y.shape, x.shape)

    def test_damping_affects_output(self):
        hope1 = HoPE(head_dim=16, damping=0.001)
        hope2 = HoPE(head_dim=16, damping=0.5)
        x = self._make_input()
        y1 = hope1(x)
        y2 = hope2(x)
        self.assertFalse(torch.allclose(y1, y2))

    def test_dtype_preserved(self):
        hope = HoPE(head_dim=16)
        x = self._make_input().double()
        y = hope(x)
        self.assertEqual(y.dtype, torch.float64)

    def test_gradient_flows(self):
        hope = HoPE(head_dim=16)
        x = self._make_input().requires_grad_(True)
        hope(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class NoPETests(unittest.TestCase):
    def _make_input(self, bsz=2, heads=4, seq=8, head_dim=16):
        return torch.randn(bsz, heads, seq, head_dim)

    def test_output_shape_matches_input(self):
        nope = NoPE()
        x = self._make_input()
        y = nope(x)
        self.assertEqual(y.shape, x.shape)

    def test_returns_identity(self):
        nope = NoPE()
        x = self._make_input()
        y = nope(x)
        self.assertTrue(torch.equal(y, x))

    def test_logical_layer_idx_accepted(self):
        nope = NoPE()
        x = self._make_input()
        y = nope(x, logical_layer_idx=5)
        self.assertTrue(torch.equal(y, x))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ALiBiTests(unittest.TestCase):
    def test_slopes_shape(self):
        alibi = ALiBi(num_heads=8)
        self.assertEqual(alibi.slopes.shape, (1, 8, 1, 1))

    def test_slopes_are_negative(self):
        alibi = ALiBi(num_heads=4)
        self.assertTrue((alibi.slopes < 0).all())

    def test_bias_shape(self):
        alibi = ALiBi(num_heads=4)
        bias = alibi.bias(seq_len=16)
        self.assertEqual(bias.shape, (1, 4, 16, 16))

    def test_bias_is_negative(self):
        alibi = ALiBi(num_heads=4)
        bias = alibi.bias(seq_len=16)
        self.assertTrue((bias <= 0).all())

    def test_bias_diagonal_is_zero(self):
        alibi = ALiBi(num_heads=4)
        bias = alibi.bias(seq_len=16)
        for i in range(16):
            self.assertAlmostEqual(float(bias[0, 0, i, i]), 0.0, places=5)

    def test_bias_farther_positions_more_negative(self):
        alibi = ALiBi(num_heads=4)
        bias = alibi.bias(seq_len=16)
        self.assertLessEqual(float(bias[0, 0, 0, 1]), float(bias[0, 0, 0, 0]))
        self.assertLessEqual(float(bias[0, 0, 0, 5]), float(bias[0, 0, 0, 1]))

    def test_forward_returns_identity(self):
        alibi = ALiBi(num_heads=4)
        x = torch.randn(2, 4, 8, 16)
        y = alibi(x)
        self.assertTrue(torch.equal(y, x))

    def test_non_power_of_2_heads(self):
        alibi = ALiBi(num_heads=6)
        self.assertEqual(alibi.slopes.shape, (1, 6, 1, 1))
        bias = alibi.bias(seq_len=8)
        self.assertEqual(bias.shape, (1, 6, 8, 8))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class PaPETests(unittest.TestCase):
    def _make_qk(self, bsz=2, heads=4, seq=8, head_dim=16):
        q = torch.randn(bsz, heads, seq, head_dim)
        k = torch.randn(bsz, heads, seq, head_dim)
        return q, k

    def test_encode_qk_output_shape(self):
        pape = PaPE(hidden_size=48, num_heads=4, head_dim=12, num_parabolas=4, num_positions=1)
        q, k = self._make_qk(heads=4, head_dim=12)
        hidden = torch.randn(2, 8, 48)
        positions = pape.default_positions(2, 8, hidden.device, hidden.dtype)
        q_aug, k_aug = pape.encode_qk(hidden, q, k, positions)
        self.assertEqual(q_aug.shape[0], q.shape[0])
        self.assertEqual(q_aug.shape[1], q.shape[1])
        self.assertEqual(q_aug.shape[2], q.shape[2])
        self.assertGreaterEqual(q_aug.shape[3], q.shape[3])

    def test_encode_qk_padded_to_multiple_of_8(self):
        pape = PaPE(hidden_size=48, num_heads=4, head_dim=12, num_parabolas=4, num_positions=1)
        q, k = self._make_qk(heads=4, head_dim=12)
        hidden = torch.randn(2, 8, 48)
        positions = pape.default_positions(2, 8, hidden.device, hidden.dtype)
        q_aug, k_aug = pape.encode_qk(hidden, q, k, positions)
        self.assertEqual(q_aug.shape[3] % 8, 0)
        self.assertEqual(k_aug.shape[3] % 8, 0)

    def test_forward_returns_identity(self):
        pape = PaPE(hidden_size=48, num_heads=4, head_dim=12, num_parabolas=4, num_positions=1)
        x = torch.randn(2, 4, 8, 12)
        y = pape(x)
        self.assertTrue(torch.equal(y, x))

    def test_default_positions_1d(self):
        pape = PaPE(hidden_size=48, num_heads=4, head_dim=12, num_parabolas=4, num_positions=1)
        positions = pape.default_positions(2, 8, torch.device("cpu"), torch.float32)
        self.assertEqual(positions.shape, (2, 8, 1))

    def test_gradient_flows(self):
        pape = PaPE(hidden_size=48, num_heads=4, head_dim=12, num_parabolas=4, num_positions=1)
        q, k = self._make_qk(heads=4, head_dim=12, seq=4)
        hidden = torch.randn(2, 4, 48, requires_grad=True)
        positions = pape.default_positions(2, 4, hidden.device, hidden.dtype)
        q_aug, k_aug = pape.encode_qk(hidden, q, k, positions)
        (q_aug.sum() + k_aug.sum()).backward()
        self.assertIsNotNone(hidden.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class PaPEEfficientTests(unittest.TestCase):
    def _make_qk(self, bsz=2, heads=4, seq=8, head_dim=16):
        q = torch.randn(bsz, heads, seq, head_dim)
        k = torch.randn(bsz, heads, seq, head_dim)
        return q, k

    def test_encode_qk_output_shape(self):
        pape = PaPEEfficient(hidden_size=48, num_heads=4, head_dim=12, num_parabolas=4, num_positions=1)
        q, k = self._make_qk(heads=4, head_dim=12)
        hidden = torch.randn(2, 8, 48)
        positions = pape.default_positions(2, 8, hidden.device, hidden.dtype)
        q_aug, k_aug = pape.encode_qk(hidden, q, k, positions)
        self.assertEqual(q_aug.shape[0], q.shape[0])
        self.assertEqual(q_aug.shape[1], q.shape[1])
        self.assertEqual(q_aug.shape[2], q.shape[2])
        self.assertGreaterEqual(q_aug.shape[3], q.shape[3])

    def test_encode_qk_padded_to_multiple_of_8(self):
        pape = PaPEEfficient(hidden_size=48, num_heads=4, head_dim=12, num_parabolas=4, num_positions=1)
        q, k = self._make_qk(heads=4, head_dim=12)
        hidden = torch.randn(2, 8, 48)
        positions = pape.default_positions(2, 8, hidden.device, hidden.dtype)
        q_aug, k_aug = pape.encode_qk(hidden, q, k, positions)
        self.assertEqual(q_aug.shape[3] % 8, 0)
        self.assertEqual(k_aug.shape[3] % 8, 0)

    def test_forward_returns_identity(self):
        pape = PaPEEfficient(hidden_size=48, num_heads=4, head_dim=12, num_parabolas=4, num_positions=1)
        x = torch.randn(2, 4, 8, 12)
        y = pape(x)
        self.assertTrue(torch.equal(y, x))

    def test_default_positions_1d(self):
        pape = PaPEEfficient(hidden_size=48, num_heads=4, head_dim=12, num_parabolas=4, num_positions=1)
        positions = pape.default_positions(2, 8, torch.device("cpu"), torch.float32)
        self.assertEqual(positions.shape, (2, 8, 1))

    def test_gradient_flows(self):
        pape = PaPEEfficient(hidden_size=48, num_heads=4, head_dim=12, num_parabolas=4, num_positions=1)
        q, k = self._make_qk(heads=4, head_dim=12, seq=4)
        hidden = torch.randn(2, 4, 48, requires_grad=True)
        positions = pape.default_positions(2, 4, hidden.device, hidden.dtype)
        q_aug, k_aug = pape.encode_qk(hidden, q, k, positions)
        (q_aug.sum() + k_aug.sum()).backward()
        self.assertIsNotNone(hidden.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class PaPERITests(unittest.TestCase):
    def _make_qk(self, bsz=2, heads=4, seq=8, head_dim=16):
        q = torch.randn(bsz, heads, seq, head_dim)
        k = torch.randn(bsz, heads, seq, head_dim)
        return q, k

    def test_encode_qk_output_shape(self):
        pape = PaPERI(hidden_size=48, num_heads=4, head_dim=12, num_parabolas=4, num_positions=2)
        q, k = self._make_qk(heads=4, head_dim=12)
        hidden = torch.randn(2, 8, 48)
        positions = pape.default_positions(2, 8, hidden.device, hidden.dtype)
        q_aug, k_aug = pape.encode_qk(hidden, q, k, positions)
        self.assertEqual(q_aug.shape[0], q.shape[0])
        self.assertEqual(q_aug.shape[1], q.shape[1])
        self.assertEqual(q_aug.shape[2], q.shape[2])
        self.assertGreaterEqual(q_aug.shape[3], q.shape[3])

    def test_encode_qk_padded_to_multiple_of_8(self):
        pape = PaPERI(hidden_size=48, num_heads=4, head_dim=12, num_parabolas=4, num_positions=2)
        q, k = self._make_qk(heads=4, head_dim=12)
        hidden = torch.randn(2, 8, 48)
        positions = pape.default_positions(2, 8, hidden.device, hidden.dtype)
        q_aug, k_aug = pape.encode_qk(hidden, q, k, positions)
        self.assertEqual(q_aug.shape[3] % 8, 0)
        self.assertEqual(k_aug.shape[3] % 8, 0)

    def test_forward_returns_identity(self):
        pape = PaPERI(hidden_size=48, num_heads=4, head_dim=12, num_parabolas=4, num_positions=2)
        x = torch.randn(2, 4, 8, 12)
        y = pape(x)
        self.assertTrue(torch.equal(y, x))

    def test_default_positions_2d(self):
        pape = PaPERI(hidden_size=48, num_heads=4, head_dim=12, num_parabolas=4, num_positions=2)
        positions = pape.default_positions(2, 8, torch.device("cpu"), torch.float32)
        self.assertEqual(positions.shape, (2, 8, 2))

    def test_gradient_flows(self):
        pape = PaPERI(hidden_size=48, num_heads=4, head_dim=12, num_parabolas=4, num_positions=2)
        q, k = self._make_qk(heads=4, head_dim=12, seq=4)
        hidden = torch.randn(2, 4, 48, requires_grad=True)
        positions = pape.default_positions(2, 4, hidden.device, hidden.dtype)
        q_aug, k_aug = pape.encode_qk(hidden, q, k, positions)
        (q_aug.sum() + k_aug.sum()).backward()
        self.assertIsNotNone(hidden.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class SinusoidalAbsoluteTests(unittest.TestCase):
    def test_add_shape(self):
        pe = SinusoidalAbsolute(hidden_size=48, max_len=64)
        x = torch.randn(2, 8, 48)
        y = pe.add(x)
        self.assertEqual(y.shape, x.shape)

    def test_add_changes_output(self):
        pe = SinusoidalAbsolute(hidden_size=48, max_len=64)
        x = torch.randn(2, 8, 48)
        y = pe.add(x)
        self.assertFalse(torch.allclose(y, x))

    def test_forward_returns_identity(self):
        pe = SinusoidalAbsolute(hidden_size=48, max_len=64)
        x = torch.randn(2, 4, 8, 12)
        y = pe(x)
        self.assertTrue(torch.equal(y, x))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class SinusoidalRotaryTests(unittest.TestCase):
    def test_output_shape_matches_input(self):
        pe = SinusoidalRotary(head_dim=16)
        x = torch.randn(2, 4, 8, 16)
        y = pe(x)
        self.assertEqual(y.shape, x.shape)

    def test_pair_dim_zero_returns_identity(self):
        pe = SinusoidalRotary(head_dim=1)
        x = torch.randn(2, 4, 8, 1)
        y = pe(x)
        self.assertTrue(torch.equal(y, x))

    def test_gradient_flows(self):
        pe = SinusoidalRotary(head_dim=16)
        x = torch.randn(2, 4, 8, 16, requires_grad=True)
        pe(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class LearnedAbsolutePETests(unittest.TestCase):
    def test_add_shape(self):
        pe = LearnedAbsolutePE(hidden_size=48, max_len=64)
        x = torch.randn(2, 8, 48)
        y = pe.add(x)
        self.assertEqual(y.shape, x.shape)

    def test_add_changes_output(self):
        pe = LearnedAbsolutePE(hidden_size=48, max_len=64)
        x = torch.randn(2, 8, 48)
        y = pe.add(x)
        self.assertFalse(torch.allclose(y, x))

    def test_forward_returns_identity(self):
        pe = LearnedAbsolutePE(hidden_size=48, max_len=64)
        x = torch.randn(2, 4, 8, 12)
        y = pe(x)
        self.assertTrue(torch.equal(y, x))

    def test_gradient_flows(self):
        pe = LearnedAbsolutePE(hidden_size=48, max_len=64)
        x = torch.randn(2, 8, 48)
        y = pe.add(x)
        y.sum().backward()
        self.assertIsNotNone(pe.pos_embed.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildPosEncoderFactoryTests(unittest.TestCase):
    def _cfg(self, pe):
        return FrankensteinModelConfig(
            vocab_size=100, hidden_size=48, num_layers=1, num_loops=1,
            num_heads=4, retention_heads=4, num_experts=2, top_k_experts=1,
            dropout=0.0, layer_pattern=["standard_attn"], use_bitnet=False,
            norm_type="layer_norm", use_moe=False, ffn_activation="gelu",
            positional_encoding=pe,
        )

    def test_build_rope(self):
        from src.model.embeddings.rope import RoPE
        pe = build_pos_encoder(self._cfg("rope"))
        self.assertIsInstance(pe, RoPE)

    def test_build_hope(self):
        from src.model.embeddings.hope import HoPE
        pe = build_pos_encoder(self._cfg("hope"))
        self.assertIsInstance(pe, HoPE)

    def test_build_nope(self):
        from src.model.embeddings.nope import NoPE
        pe = build_pos_encoder(self._cfg("nope"))
        self.assertIsInstance(pe, NoPE)

    def test_build_alibi(self):
        from src.model.embeddings.alibi import ALiBi
        pe = build_pos_encoder(self._cfg("alibi"))
        self.assertIsInstance(pe, ALiBi)

    def test_build_pape(self):
        from src.model.embeddings.pape import PaPE
        pe = build_pos_encoder(self._cfg("pape"))
        self.assertIsInstance(pe, PaPE)

    def test_build_pape_efficient(self):
        from src.model.embeddings.pape_efficient import PaPEEfficient
        pe = build_pos_encoder(self._cfg("pape_efficient"))
        self.assertIsInstance(pe, PaPEEfficient)

    def test_build_pape_ri(self):
        from src.model.embeddings.pape_ri import PaPERI
        pe = build_pos_encoder(self._cfg("pape_ri"))
        self.assertIsInstance(pe, PaPERI)

    def test_build_sinusoidal_absolute(self):
        from src.model.embeddings.sinusoidal import SinusoidalAbsolute
        pe = build_pos_encoder(self._cfg("sinusoidal_absolute"))
        self.assertIsInstance(pe, SinusoidalAbsolute)

    def test_build_sinusoidal_rotary(self):
        from src.model.embeddings.sinusoidal import SinusoidalRotary
        pe = build_pos_encoder(self._cfg("sinusoidal_rotary"))
        self.assertIsInstance(pe, SinusoidalRotary)

    def test_build_learned_absolute(self):
        from src.model.embeddings.learned_absolute import LearnedAbsolutePE
        pe = build_pos_encoder(self._cfg("learned_absolute"))
        self.assertIsInstance(pe, LearnedAbsolutePE)

    def test_build_none(self):
        pe = build_pos_encoder(self._cfg("none"))
        from src.model.embeddings.nope import NoPE
        self.assertIsInstance(pe, NoPE)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            cfg = self._cfg("rope")
            cfg.__dict__["positional_encoding"] = "invalid"
            build_pos_encoder(cfg)


if __name__ == "__main__":
    unittest.main()
