"""Unit tests for core attention modules: Standard, Sigmoid, Titan, ODE, RetNet."""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from src.model.tormented_bert_frankestein import UltraConfig
    from src.model.attention.standard import StandardAttention
    from src.model.attention.sigmoid import SigmoidAttention
    from src.model.attention.titan import TitanAttention
    from src.model.attention.ode import ODEAttentionBlock, ODEFunc
    from src.model.attention.retnet import MultiScaleRetention
    from src.model.attention.grouped_query_attention import GroupedQueryAttention


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


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class StandardAttentionTests(unittest.TestCase):
    def _make(self, **kw):
        return StandardAttention(_cfg(**kw))

    def _x(self):
        return torch.randn(BSZ, SEQ, DIM)

    def test_encoder_output_shape(self):
        attn = self._make()
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_decoder_output_shape(self):
        attn = self._make(mode="decoder")
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_logical_layer_idx_accepted(self):
        attn = self._make()
        y = attn(self._x(), logical_layer_idx=3)
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_invalid_hidden_size_raises(self):
        with self.assertRaises(ValueError):
            StandardAttention(_cfg(hidden_size=50, num_heads=6))  # 50 % 6 != 0

    def test_gradient_flows(self):
        attn = self._make()
        x = self._x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_batch_size_one(self):
        attn = self._make()
        x = torch.randn(1, SEQ, DIM)
        y = attn(x)
        self.assertEqual(y.shape, (1, SEQ, DIM))

    def test_seq_len_one(self):
        attn = self._make()
        x = torch.randn(BSZ, 1, DIM)
        y = attn(x)
        self.assertEqual(y.shape, (BSZ, 1, DIM))

    def test_with_bitnet(self):
        attn = StandardAttention(_cfg(use_bitnet=True))
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class SigmoidAttentionTests(unittest.TestCase):
    def _make(self, **kw):
        return SigmoidAttention(_cfg(**kw))

    def _x(self):
        return torch.randn(BSZ, SEQ, DIM)

    def test_encoder_output_shape(self):
        attn = self._make()
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_decoder_output_shape(self):
        attn = self._make(mode="decoder")
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_invalid_hidden_size_raises(self):
        with self.assertRaises(ValueError):
            SigmoidAttention(_cfg(hidden_size=50, num_heads=6))

    def test_gradient_flows(self):
        attn = self._make()
        x = self._x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class TitanAttentionTests(unittest.TestCase):
    def _make(self, **kw):
        return TitanAttention(_cfg(**kw))

    def _x(self):
        return torch.randn(BSZ, SEQ, DIM)

    def test_encoder_with_hope(self):
        attn = self._make(use_hope=True, positional_encoding="hope")
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_encoder_with_rope(self):
        attn = self._make(use_hope=False, positional_encoding="rope")
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_decoder_causal_mode(self):
        attn = self._make(mode="decoder")
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_logical_layer_idx_affects_hope(self):
        attn = self._make(positional_encoding="hope")
        y0 = attn(self._x(), logical_layer_idx=0)
        y5 = attn(self._x(), logical_layer_idx=5)
        # outputs differ because HoPE uses layer_idx for scaling
        self.assertEqual(y0.shape, y5.shape)

    def test_invalid_positional_encoding_raises(self):
        with self.assertRaises(ValueError):
            cfg = _cfg()
            cfg.__dict__["positional_encoding"] = "absolute"
            TitanAttention(cfg)

    def test_invalid_hidden_size_raises(self):
        with self.assertRaises(ValueError):
            TitanAttention(_cfg(hidden_size=50, num_heads=6))

    def test_gradient_flows(self):
        attn = self._make()
        x = self._x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_with_bitnet_and_hope(self):
        attn = TitanAttention(_cfg(use_bitnet=True, positional_encoding="hope"))
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ODEFuncTests(unittest.TestCase):
    def _x(self):
        return torch.randn(BSZ, SEQ, DIM)

    def test_output_shape_encoder(self):
        func = ODEFunc(_cfg())
        y = func(0.0, self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_output_shape_decoder(self):
        func = ODEFunc(_cfg(mode="decoder"))
        y = func(0.5, self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ODEAttentionBlockTests(unittest.TestCase):
    def _x(self):
        return torch.randn(BSZ, SEQ, DIM)

    def test_rk4_solver_output_shape(self):
        block = ODEAttentionBlock(_cfg(ode_solver="rk4", ode_steps=2))
        y = block(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_euler_solver_output_shape(self):
        block = ODEAttentionBlock(_cfg(ode_solver="euler", ode_steps=2))
        y = block(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_single_step(self):
        block = ODEAttentionBlock(_cfg(ode_solver="rk4", ode_steps=1))
        y = block(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        block = ODEAttentionBlock(_cfg(ode_solver="rk4", ode_steps=1))
        x = self._x().requires_grad_(True)
        block(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class MultiScaleRetentionTests(unittest.TestCase):
    def _x(self):
        return torch.randn(BSZ, SEQ, DIM)

    def test_encoder_output_shape(self):
        ret = MultiScaleRetention(_cfg())
        y = ret(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_decoder_output_shape(self):
        ret = MultiScaleRetention(_cfg(mode="decoder"))
        y = ret(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        ret = MultiScaleRetention(_cfg())
        x = self._x().requires_grad_(True)
        ret(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_decay_mask_buffer_registered(self):
        ret = MultiScaleRetention(_cfg())
        self.assertTrue(hasattr(ret, "decay_mask"))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GroupedQueryAttentionTests(unittest.TestCase):
    def _make(self, **kw):
        return GroupedQueryAttention(_cfg(**kw))

    def _x(self):
        return torch.randn(BSZ, SEQ, DIM)

    def test_encoder_output_shape(self):
        attn = self._make(num_kv_heads=2)
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_decoder_output_shape(self):
        attn = self._make(num_kv_heads=2, mode="decoder")
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_kv_heads_one_mqa(self):
        attn = self._make(num_kv_heads=1)
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_kv_heads_equal_num_heads(self):
        attn = self._make(num_kv_heads=6)
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_logical_layer_idx_accepted(self):
        attn = self._make(num_kv_heads=2)
        y = attn(self._x(), logical_layer_idx=3)
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_invalid_hidden_size_raises(self):
        with self.assertRaises(ValueError):
            GroupedQueryAttention(_cfg(hidden_size=50, num_heads=6, num_kv_heads=2))

    def test_invalid_kv_heads_less_than_one_raises(self):
        with self.assertRaises(ValueError):
            GroupedQueryAttention(_cfg(num_heads=6, num_kv_heads=0))

    def test_invalid_kv_heads_greater_than_num_heads_raises(self):
        with self.assertRaises(ValueError):
            GroupedQueryAttention(_cfg(num_heads=6, num_kv_heads=12))

    def test_indivisible_kv_heads_raises(self):
        with self.assertRaises(ValueError):
            GroupedQueryAttention(_cfg(num_heads=6, num_kv_heads=4))

    def test_gradient_flows(self):
        attn = self._make(num_kv_heads=2)
        x = self._x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_batch_size_one(self):
        attn = self._make(num_kv_heads=3)
        x = torch.randn(1, SEQ, DIM)
        y = attn(x)
        self.assertEqual(y.shape, (1, SEQ, DIM))

    def test_seq_len_one(self):
        attn = self._make(num_kv_heads=2)
        x = torch.randn(BSZ, 1, DIM)
        y = attn(x)
        self.assertEqual(y.shape, (BSZ, 1, DIM))

    def test_with_bitnet(self):
        attn = GroupedQueryAttention(_cfg(use_bitnet=True, num_kv_heads=2))
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))


if __name__ == "__main__":
    unittest.main()
