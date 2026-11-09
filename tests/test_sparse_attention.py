"""Unit tests for sparse attention modules."""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from src.model.config import FrankensteinModelConfig
    from src.model.attention.sparse import (
        BigBirdAttention,
        FASAAttention,
        FastConformerAttention,
        LongformerAttention,
        LSAAttention,
        MSAAttention,
        NSAAttention,
        SparseKAttention,
        SparseTransformerAttention,
        SpargeAttention,
        SparDAAttention,
        SlidingWindowAttention,
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
        layer_pattern=["standard"],
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


def _x(seq=SEQ):
    return torch.randn(BSZ, seq, DIM)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class SparseTransformerAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = SparseTransformerAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = SparseTransformerAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_decoder_mode(self):
        attn = SparseTransformerAttention(_cfg(mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class LongformerAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = LongformerAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = LongformerAttention(_cfg(mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = LongformerAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_sliding_window_mask_shape(self):
        attn = LongformerAttention(_cfg())
        mask = attn._build_mask(SEQ, torch.device("cpu"))
        self.assertEqual(mask.shape, (SEQ, SEQ))
        self.assertEqual(mask.dtype, torch.bool)

    def test_sliding_window_global_token_attended_by_all(self):
        attn = LongformerAttention(_cfg())
        # global_tokens=[0] by default
        mask = attn._build_mask(SEQ, torch.device("cpu"))
        # Row 0 (global token) attends to all
        self.assertTrue(mask[0, :].all())
        # Column 0: all tokens attend to global token 0
        self.assertTrue(mask[:, 0].all())


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class SlidingWindowAttentionTests(unittest.TestCase):
    """SWA(w, s) with attention sinks (arXiv:2608.28444)."""

    def _cfg_swa(self, mode="encoder", window=5, sinks=2):
        return _cfg(
            mode=mode,
            swa_window_size=window,
            swa_num_sinks=sinks,
        )

    def test_output_shape(self):
        attn = SlidingWindowAttention(self._cfg_swa())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = SlidingWindowAttention(self._cfg_swa(mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = SlidingWindowAttention(self._cfg_swa())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_mask_shape_and_dtype(self):
        attn = SlidingWindowAttention(self._cfg_swa())
        mask = attn._build_mask(SEQ, torch.device("cpu"))
        self.assertEqual(mask.shape, (SEQ, SEQ))
        self.assertEqual(mask.dtype, torch.bool)

    def test_sinks_attended_by_all(self):
        # Every query from position `num_sinks` onward must see all sinks; in
        # decoder mode earlier rows are causal-clipped (no future leak).
        for mode in ("encoder", "decoder"):
            attn = SlidingWindowAttention(self._cfg_swa(mode=mode, window=5, sinks=2))
            mask = attn._build_mask(SEQ, torch.device("cpu"))
            if mode == "decoder":
                self.assertTrue(mask[2:, :2].all(), msg=mode)
                self.assertTrue(mask[0, 1].item() is False, msg=mode)
            else:
                self.assertTrue(mask[:, :2].all(), msg=mode)

    def test_decoder_window_boundary(self):
        # w=5, s=2 -> sliding part spans 3 tokens: j in [i-2, i].
        attn = SlidingWindowAttention(self._cfg_swa(mode="decoder", window=5, sinks=2))
        mask = attn._build_mask(SEQ, torch.device("cpu"))
        # Row 7: sinks {0, 1} plus recent {5, 6, 7}; 2..4 are out of window.
        expected = {0, 1, 5, 6, 7}
        self.assertEqual(
            set(mask[7].nonzero(as_tuple=False).flatten().tolist()), expected
        )
        # Row 0: only itself (sinks are causal-clipped).
        self.assertEqual(
            set(mask[0].nonzero(as_tuple=False).flatten().tolist()), {0}
        )

    def test_decoder_has_no_future_leak(self):
        attn = SlidingWindowAttention(self._cfg_swa(mode="decoder", window=5, sinks=4))
        mask = attn._build_mask(SEQ, torch.device("cpu"))
        upper = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)
        self.assertFalse((mask & upper).any())

    def test_encoder_symmetric_window(self):
        # w=5, s=2 -> span=3, half=1: |i-j| <= 1 plus sinks {0, 1}.
        attn = SlidingWindowAttention(self._cfg_swa(mode="encoder", window=5, sinks=2))
        mask = attn._build_mask(SEQ, torch.device("cpu"))
        expected = {0, 1, 3, 4, 5}
        self.assertEqual(
            set(mask[4].nonzero(as_tuple=False).flatten().tolist()), expected
        )

    def test_window_larger_than_sequence_is_full_causal(self):
        # w=256, s=4 over a 8-token sequence: decoder mask is plain causal.
        attn = SlidingWindowAttention(
            self._cfg_swa(mode="decoder", window=256, sinks=4)
        )
        mask = attn._build_mask(SEQ, torch.device("cpu"))
        causal = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool))
        self.assertTrue(torch.equal(mask, causal))

    def test_invalid_sink_count_raises(self):
        with self.assertRaisesRegex(ValueError, "swa_num_sinks"):
            SlidingWindowAttention(self._cfg_swa(window=4, sinks=4))

    def test_invalid_window_raises(self):
        with self.assertRaisesRegex(ValueError, "swa_window_size"):
            SlidingWindowAttention(self._cfg_swa(window=1, sinks=1))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class FastConformerAttentionTests(unittest.TestCase):
    """Limited-context attention with a head-pinned global token (arXiv:2305.05084)."""

    def _fc_cfg(self, **overrides):
        base = dict(
            fastconformer_window_size=5,
            fastconformer_global_tokens=2,
            fastconformer_global_tokens_spacing=2,
            fastconformer_global_attn_separate=False,
        )
        base.update(overrides)
        return _cfg(**base)

    def _fc(self, **overrides):
        return FastConformerAttention(self._fc_cfg(**overrides))

    def test_output_shape_encoder(self):
        attn = self._fc()
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_output_shape_decoder(self):
        attn = self._fc(mode="decoder")
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = self._fc()
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_no_nan_or_inf(self):
        for mode in ("encoder", "decoder"):
            attn = self._fc(mode=mode)
            out = attn(_x())
            self.assertTrue(torch.isfinite(out).all(), msg=mode)

    def test_mask_shape_and_dtype(self):
        attn = self._fc()
        mask = attn._build_mask(SEQ, torch.device("cpu"))
        self.assertEqual(mask.shape, (SEQ, SEQ))
        self.assertEqual(mask.dtype, torch.bool)

    def test_global_token_attended_by_all(self):
        # G = {0, 2}: every query sees both global columns (encoder mode).
        attn = self._fc()
        mask = attn._build_mask(SEQ, torch.device("cpu"))
        self.assertTrue(mask[:, 0].all())
        self.assertTrue(mask[:, 2].all())

    def test_global_token_attends_to_all(self):
        attn = self._fc()
        mask = attn._build_mask(SEQ, torch.device("cpu"))
        self.assertTrue(mask[0, :].all())
        self.assertTrue(mask[2, :].all())

    def test_encoder_symmetric_window(self):
        # w=5 -> half=2: |i-j| <= 2, plus globals {0, 2}.
        attn = self._fc()
        mask = attn._build_mask(SEQ, torch.device("cpu"))
        expected = {0, 2, 5, 6, 7}
        self.assertEqual(
            set(mask[7].nonzero(as_tuple=False).flatten().tolist()), expected
        )

    def test_decoder_has_no_future_leak(self):
        attn = self._fc(mode="decoder")
        mask = attn._build_mask(SEQ, torch.device("cpu"))
        upper = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)
        self.assertFalse((mask & upper).any())

    def test_decoder_global_rows_are_causal_clipped(self):
        attn = self._fc(mode="decoder")
        mask = attn._build_mask(SEQ, torch.device("cpu"))
        # Row 0: only itself. Row 2 (global): every past position.
        self.assertEqual(
            set(mask[0].nonzero(as_tuple=False).flatten().tolist()), {0}
        )
        self.assertEqual(
            set(mask[2].nonzero(as_tuple=False).flatten().tolist()), {0, 1, 2}
        )

    def test_window_larger_than_sequence_is_full_causal(self):
        attn = self._fc(mode="decoder", fastconformer_window_size=256,
                        fastconformer_global_tokens=1)
        mask = attn._build_mask(SEQ, torch.device("cpu"))
        causal = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool))
        self.assertTrue(torch.equal(mask, causal))

    def test_shared_projections_are_parameter_neutral(self):
        attn = self._fc()
        params = sum(p.numel() for p in attn.parameters())
        self.assertEqual(params, 4 * DIM * DIM)
        self.assertIsNone(attn.global_q_proj)

    def test_separate_projections_add_three_matrices(self):
        attn = self._fc(fastconformer_global_attn_separate=True)
        params = sum(p.numel() for p in attn.parameters())
        self.assertEqual(params, 7 * DIM * DIM)
        for proj in (attn.global_q_proj, attn.global_k_proj, attn.global_v_proj):
            self.assertIsNotNone(proj)

    def test_separate_variant_forward(self):
        for mode in ("encoder", "decoder"):
            attn = self._fc(mode=mode, fastconformer_global_attn_separate=True)
            out = attn(_x())
            self.assertEqual(out.shape, (BSZ, SEQ, DIM))
            self.assertTrue(torch.isfinite(out).all())

    def test_bitnet_uses_bitlinear(self):
        from src.model.attention.common import BitLinear

        attn = self._fc(use_bitnet=True, fastconformer_global_attn_separate=True)
        for proj in (
            attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj,
            attn.global_q_proj, attn.global_k_proj, attn.global_v_proj,
        ):
            self.assertIsInstance(proj, BitLinear)

    def test_invalid_window_raises(self):
        with self.assertRaisesRegex(ValueError, "fastconformer_window_size"):
            self._fc(fastconformer_window_size=1)

    def test_invalid_global_tokens_raises(self):
        with self.assertRaisesRegex(ValueError, "fastconformer_global_tokens must"):
            self._fc(fastconformer_global_tokens=-1)

    def test_invalid_spacing_raises(self):
        with self.assertRaisesRegex(ValueError, "global_tokens_spacing"):
            self._fc(fastconformer_global_tokens_spacing=0)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BigBirdAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = BigBirdAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = BigBirdAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class SparseKAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = SparseKAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = SparseKAttention(_cfg(mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = SparseKAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class NSAAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = NSAAttention(_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = NSAAttention(_cfg())
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class SpargeAttentionTests(unittest.TestCase):
    def test_output_shape_eval_mode(self):
        attn = SpargeAttention(_cfg())
        attn.eval()
        with torch.no_grad():
            self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode_eval(self):
        attn = SpargeAttention(_cfg(mode="decoder"))
        attn.eval()
        with torch.no_grad():
            self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_seq_len_not_divisible_by_block_size(self):
        # seq=10, block_size=4 → padding required
        attn = SpargeAttention(_cfg())
        attn.eval()
        with torch.no_grad():
            y = attn(_x(seq=10))
        self.assertEqual(y.shape, (BSZ, 10, DIM))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class FASAAttentionTests(unittest.TestCase):
    def test_output_shape_eval_mode(self):
        attn = FASAAttention(_cfg())
        attn.eval()
        with torch.no_grad():
            self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class MSAAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = MSAAttention(_cfg(num_kv_heads=2, msa_block_size=4, msa_topk_blocks=2))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = MSAAttention(_cfg(num_kv_heads=2, msa_block_size=4, msa_topk_blocks=2, mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = MSAAttention(_cfg(num_kv_heads=2, msa_block_size=4, msa_topk_blocks=2))
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_kv_heads_raises(self):
        with self.assertRaises(ValueError):
            MSAAttention(_cfg(num_kv_heads=4))  # 4 does not divide 6


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class SparDAAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = SparDAAttention(_cfg(num_kv_heads=2, sparda_block_size=4, sparda_topk_blocks=2))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = SparDAAttention(_cfg(num_kv_heads=2, sparda_block_size=4, sparda_topk_blocks=2, mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = SparDAAttention(_cfg(num_kv_heads=2, sparda_block_size=4, sparda_topk_blocks=2))
        x = _x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_kv_heads_raises(self):
        with self.assertRaises(ValueError):
            SparDAAttention(_cfg(num_kv_heads=4))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class LSAAttentionTests(unittest.TestCase):
    def _lsa_cfg(self, **overrides):
        base = dict(
            num_kv_heads=1,
            lsa_cluster_block=8,
            lsa_pool_size=2,
            lsa_keys_per_bucket=2,
            lsa_num_buckets=2,
            lsa_num_long_range=4,
            lsa_num_sinks=1,
        )
        base.update(overrides)
        return _cfg(**base)

    def test_output_shape(self):
        attn = LSAAttention(self._lsa_cfg())
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_decoder_mode(self):
        attn = LSAAttention(self._lsa_cfg(mode="decoder"))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        attn = LSAAttention(self._lsa_cfg())
        x = _x().requires_grad_(True)
        out = attn(x)
        self.assertTrue(torch.isfinite(out).all())
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_invalid_kv_heads_raises(self):
        with self.assertRaises(ValueError):
            LSAAttention(_cfg(num_kv_heads=4))  # 4 does not divide 6

    def test_short_sequence_uses_window_only(self):
        # SEQ=8 < default cluster_block=512 -> no complete block: J = sinks + window.
        attn = LSAAttention(_cfg(num_kv_heads=1))
        out = attn(_x())
        self.assertEqual(attn.last_num_buckets, 0)
        self.assertTrue(torch.isfinite(out).all())

    def test_buckets_form_and_select(self):
        # cluster_block=4 with SEQ=8 -> blocks 0 and 1 complete.
        attn = LSAAttention(self._lsa_cfg())
        out = attn(_x())
        self.assertGreater(attn.last_num_buckets, 0)
        self.assertTrue(torch.isfinite(out).all())

    def test_bitnet_on(self):
        attn = LSAAttention(self._lsa_cfg(use_bitnet=True))
        self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM))


if __name__ == "__main__":
    unittest.main()
