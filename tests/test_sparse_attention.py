"""Unit tests for sparse attention modules."""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from src.model.tormented_bert_frankestein import UltraConfig
    from src.model.attention.sparse import (
        BigBirdAttention,
        FASAAttention,
        LongformerAttention,
        MSAAttention,
        NSAAttention,
        SparseKAttention,
        SparseTransformerAttention,
        SpargeAttention,
        SparDAAttention,
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


if __name__ == "__main__":
    unittest.main()
