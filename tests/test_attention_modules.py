"""Unit tests for core attention modules: Standard, Sigmoid, Titan, ODE, RetNet."""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from src.model.config import FrankensteinModelConfig, merge_model_blocks
    from src.model.attention.standard import StandardAttention
    from src.model.attention.sigmoid import SigmoidAttention
    from src.model.attention.ca import ContinuumAttention, _exp_moments
    from src.model.attention.titan import TitanAttention
    from src.model.attention.ode import ODEAttentionBlock, ODEFunc
    from src.model.attention.retnet import MultiScaleRetention
    from src.model.attention.grouped_query_attention import GroupedQueryAttention
    from src.model.attention.memory import MemoryAttention


def _cfg(**overrides):
    base = {"dims": {"vocab_size": 64, "hidden_size": 48, "num_layers": 1, "num_loops": 1, "num_heads": 6, "dropout": 0.0, "layer_pattern": ['standard'], "mode": 'encoder'}, "attention": {"retnet": {"heads": 6}, "ode": {"solver": 'rk4', "steps": 1}}, "moe": {"num_experts": 2, "top_k": 1, "enabled": False}, "norm": {"type": 'layer_norm'}, "bitnet": {"enabled": False}, "ffn": {"hidden_size": 96, "activation": 'gelu'}, "positional_encoding": 'hope'}
    base = merge_model_blocks(base, overrides)
    return FrankensteinModelConfig.from_dict({**base})


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
        attn = self._make(dims={"mode": 'decoder'})
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_logical_layer_idx_accepted(self):
        attn = self._make()
        y = attn(self._x(), logical_layer_idx=3)
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_invalid_hidden_size_raises(self):
        with self.assertRaises(ValueError):
            StandardAttention(_cfg(dims={'hidden_size': 50, 'num_heads': 6}))  # 50 % 6 != 0

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
        attn = StandardAttention(_cfg(bitnet={'enabled': True}))
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
        attn = self._make(dims={"mode": 'decoder'})
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_invalid_hidden_size_raises(self):
        with self.assertRaises(ValueError):
            SigmoidAttention(_cfg(dims={'hidden_size': 50, 'num_heads': 6}))

    def test_gradient_flows(self):
        attn = self._make()
        x = self._x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ContinuumAttentionTests(unittest.TestCase):
    """Continuum Attention: exact segment integration over linear memory."""

    def _make(self, **kw):
        return ContinuumAttention(_cfg(**kw))

    def _x(self):
        return torch.randn(BSZ, SEQ, DIM)

    def test_encoder_output_shape(self):
        attn = self._make()
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_decoder_output_shape(self):
        attn = self._make(dims={"mode": 'decoder'})
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_decoder_is_causal(self):
        attn = self._make(dims={"mode": 'decoder'})
        attn.eval()
        x = self._x()
        with torch.no_grad():
            y = attn(x)
            x2 = x.clone()
            x2[:, -1, :] += 10.0
            y2 = attn(x2)
        # Perturbing the last token must not change any earlier output.
        self.assertTrue(torch.allclose(y[:, :-1, :], y2[:, :-1, :], atol=1e-5))

    def test_equal_scores_reduce_to_uniform(self):
        attn = self._make(dims={"mode": 'decoder'})
        seq = 6
        scores = torch.zeros(1, 1, seq, seq)
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        with torch.no_grad():
            w = attn._effective_weights(scores)
        for i in range(seq):
            expected = torch.full((i + 1,), 1.0 / (i + 1))
            self.assertTrue(torch.allclose(w[0, 0, i, : i + 1], expected, atol=1e-5))
            self.assertTrue(torch.allclose(w[0, 0, i, :].sum(), torch.tensor(1.0), atol=1e-5))

    def test_first_token_output_is_value(self):
        # Paper section 2.4: for i = 0 the two half caps give o_0 = v_0.
        attn = self._make(dims={"mode": 'decoder'})
        scores = torch.zeros(1, 1, 4, 4).masked_fill(
            torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
            .unsqueeze(0).unsqueeze(0), float("-inf")
        )
        with torch.no_grad():
            w = attn._effective_weights(scores)
        self.assertAlmostEqual(float(w[0, 0, 0, 0]), 1.0, places=6)
        self.assertTrue(torch.allclose(w[0, 0, 0, 1:], torch.zeros(3), atol=1e-6))

    def test_moments_match_quadrature(self):
        u = torch.linspace(0, 1, 20001, dtype=torch.float64)
        for d in (-8.0, -0.9, -1e-4, 0.0, 1e-4, 0.9, 8.0):
            j0, j1, j2 = _exp_moments(torch.tensor([d]))
            e = torch.exp(torch.tensor(d, dtype=torch.float64) * u)
            self.assertTrue(torch.allclose(j0, torch.trapz(e, u).float(), rtol=1e-4, atol=1e-5))
            self.assertTrue(torch.allclose(j1, torch.trapz(u * e, u).float(), rtol=1e-4, atol=1e-5))
            self.assertTrue(torch.allclose(j2, torch.trapz(u * u * e, u).float(), rtol=1e-4, atol=1e-5))

    def test_learned_gain_parameter_count(self):
        attn = self._make(attention={"ca": {"learned_gain": True}})
        n_proj = sum(p.numel() for n, p in attn.named_parameters() if n.endswith("_proj.weight"))
        gain = dict(attn.named_parameters())["log_gain"]
        self.assertEqual(gain.numel(), attn.num_heads)
        self.assertEqual(n_proj + gain.numel(), 4 * DIM * DIM + attn.num_heads)

    def test_fixed_gain_has_no_extra_parameters(self):
        attn = self._make(attention={"ca": {"learned_gain": False}})
        self.assertIsNone(attn.log_gain)
        self.assertEqual(
            sum(p.numel() for p in attn.parameters()), 4 * DIM * DIM
        )

    def test_gradient_flows(self):
        attn = self._make(attention={"ca": {"learned_gain": True}})
        x = self._x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(dict(attn.named_parameters())["log_gain"].grad)

    def test_with_bitnet(self):
        attn = self._make(bitnet={"enabled": True})
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_no_nans_long_sequence(self):
        attn = self._make(dims={"mode": 'decoder'})
        attn.eval()
        x = torch.randn(1, 64, DIM)
        with torch.no_grad():
            y = attn(x)
        self.assertTrue(torch.isfinite(y).all())

    def test_use_pe_flag(self):
        # ALiBi is a score-bias PE: enabling it for `ca` must change the
        # output, and disabling it must reproduce the NoPE run.
        from src.model.embeddings.factory import build_pos_encoder

        cfg_on = _cfg(positional_encoding='alibi', positional_encoding_parameters={'use_pe': {'ca_use_pe': True}})
        cfg_off = _cfg(positional_encoding='alibi', positional_encoding_parameters={'use_pe': {'ca_use_pe': False}})
        cfg_none = _cfg(positional_encoding='nope', positional_encoding_parameters={'use_pe': {'ca_use_pe': False}})
        x = self._x()
        torch.manual_seed(0)
        on = ContinuumAttention(cfg_on, pos_encoder=build_pos_encoder(cfg_on))
        off = ContinuumAttention(cfg_off, pos_encoder=build_pos_encoder(cfg_off))
        none = ContinuumAttention(cfg_none, pos_encoder=build_pos_encoder(cfg_none))
        off.load_state_dict(on.state_dict())
        none.load_state_dict(on.state_dict())
        with torch.no_grad():
            y_on, y_off, y_none = on(x), off(x), none(x)
        self.assertFalse(torch.allclose(y_on, y_off, atol=1e-5))
        self.assertTrue(torch.allclose(y_off, y_none, atol=1e-6))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class MemoryAttentionTests(unittest.TestCase):
    """Memory Attention: V = K + Norm(E[s]) with a token-ID memory table."""

    def _make(self, **kw):
        kw.setdefault("num_kv_heads", 2)
        return MemoryAttention(_cfg(**kw))

    def _x(self):
        return torch.randn(BSZ, SEQ, DIM)

    def _ids(self):
        return torch.randint(0, 64, (BSZ, SEQ))

    def test_encoder_output_shape(self):
        attn = self._make()
        y = attn(self._x(), input_ids=self._ids())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_decoder_output_shape(self):
        attn = self._make(dims={"mode": 'decoder'})
        y = attn(self._x(), input_ids=self._ids())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_missing_input_ids_raises(self):
        attn = self._make()
        with self.assertRaises(ValueError):
            attn(self._x())

    def test_decoder_is_causal(self):
        attn = self._make(dims={"mode": 'decoder'}).eval()
        x, ids = self._x(), self._ids()
        full = attn(x, input_ids=ids)
        prefix = attn(x[:, :4], input_ids=ids[:, :4])
        self.assertTrue(torch.allclose(full[:, :4], prefix, atol=1e-5))

    def test_memory_is_per_head_rms_normalized(self):
        attn = self._make().eval()
        ids = self._ids()
        e = attn.memory(ids).view(BSZ, SEQ, attn.num_kv_heads, attn.head_dim)
        rms = torch.sqrt(e.pow(2).mean(dim=-1, keepdim=True) + attn.memory_eps)
        gamma = attn.memory_norm_weight.view(attn.num_kv_heads, attn.head_dim)
        expected = (e / rms) * gamma
        actual = attn._memory_lookup(ids).view(BSZ, SEQ, attn.num_kv_heads, attn.head_dim)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6))

    def test_gradient_flows_to_memory_and_keys(self):
        attn = self._make()
        x = self._x().requires_grad_(True)
        attn(x, input_ids=self._ids()).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(attn.memory.weight.grad)
        self.assertIsNotNone(attn.memory_norm_weight.grad)
        self.assertIsNotNone(attn.k_proj.weight.grad)

    def test_with_bitnet(self):
        from src.model.attention.common import BitLinear

        attn = self._make(bitnet={"enabled": True})
        self.assertIsInstance(attn.q_proj, BitLinear)
        self.assertIsInstance(attn.k_proj, BitLinear)
        self.assertIsInstance(attn.out_proj, BitLinear)
        self.assertNotIsInstance(attn.memory, BitLinear)
        y = attn(self._x(), input_ids=self._ids())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_no_nans(self):
        for mode in ("encoder", "decoder"):
            attn = self._make(dims={"mode": mode}).eval()
            y = attn(self._x(), input_ids=self._ids())
            self.assertTrue(torch.isfinite(y).all())

    def test_qk_norm_attached_and_runs(self):
        from src.model.norm.qk import QKNorm

        attn = self._make(norm={"qk_norm": {"enabled": True, "method": 'rms_norm'}})
        self.assertIsInstance(attn.qk_norm, QKNorm)
        y = attn(self._x(), input_ids=self._ids())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_use_pe_flag_controls_positional_encoder(self):
        class _AddOne(torch.nn.Module):
            def forward(self, x, logical_layer_idx=None, position_offset=None):
                return x + 1.0

        torch.manual_seed(0)
        on = self._make(positional_encoding_parameters={'use_pe': {'memory_use_pe': True}})
        off = self._make(positional_encoding_parameters={'use_pe': {'memory_use_pe': False}})
        off.load_state_dict(on.state_dict())
        pe = _AddOne()
        x, ids = self._x(), self._ids()
        y_on = on(x, input_ids=ids, pos_encoder=pe)
        y_off = off(x, input_ids=ids, pos_encoder=pe)
        self.assertFalse(torch.allclose(y_on, y_off, atol=1e-6))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class TitanAttentionTests(unittest.TestCase):
    def _make(self, **kw):
        from src.model.embeddings.factory import build_pos_encoder
        cfg = _cfg(**kw)
        pe = build_pos_encoder(cfg)
        return TitanAttention(cfg, pos_encoder=pe)

    def _x(self):
        return torch.randn(BSZ, SEQ, DIM)

    def test_encoder_with_hope(self):
        attn = self._make(positional_encoding='hope')
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_encoder_with_rope(self):
        attn = self._make(positional_encoding='rope')
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_decoder_causal_mode(self):
        attn = self._make(dims={"mode": 'decoder'})
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_logical_layer_idx_affects_hope(self):
        attn = self._make(positional_encoding='hope')
        y0 = attn(self._x(), logical_layer_idx=0)
        y5 = attn(self._x(), logical_layer_idx=5)
        # outputs differ because HoPE uses layer_idx for scaling
        self.assertEqual(y0.shape, y5.shape)

    def test_invalid_positional_encoding_raises(self):
        with self.assertRaises(ValueError):
            FrankensteinModelConfig.from_dict({**{**_cfg().__dict__, 'positional_encoding': 'absolute'}})

    def test_invalid_hidden_size_raises(self):
        with self.assertRaises(ValueError):
            TitanAttention(_cfg(dims={'hidden_size': 50, 'num_heads': 6}))

    def test_gradient_flows(self):
        attn = self._make()
        x = self._x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_with_bitnet_and_hope(self):
        attn = TitanAttention(_cfg(bitnet={'enabled': True}, positional_encoding='hope'))
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_titan_with_nope(self):
        attn = self._make(positional_encoding='nope')
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_titan_with_alibi(self):
        attn = self._make(positional_encoding='alibi')
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_titan_with_pape(self):
        attn = self._make(positional_encoding='pape', positional_encoding_parameters={"pape": {"num_parabolas": 4, "num_positions": 1}})
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
        func = ODEFunc(_cfg(dims={'mode': 'decoder'}))
        y = func(0.5, self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ODEAttentionBlockTests(unittest.TestCase):
    def _x(self):
        return torch.randn(BSZ, SEQ, DIM)

    def test_rk4_solver_output_shape(self):
        block = ODEAttentionBlock(_cfg(attention={'ode': {'solver': 'rk4', 'steps': 2}}))
        y = block(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_euler_solver_output_shape(self):
        block = ODEAttentionBlock(_cfg(attention={'ode': {'solver': 'euler', 'steps': 2}}))
        y = block(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_single_step(self):
        block = ODEAttentionBlock(_cfg(attention={'ode': {'solver': 'rk4', 'steps': 1}}))
        y = block(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_gradient_flows(self):
        block = ODEAttentionBlock(_cfg(attention={'ode': {'solver': 'rk4', 'steps': 1}}))
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
        ret = MultiScaleRetention(_cfg(dims={'mode': 'decoder'}))
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
        attn = self._make(dims={"num_kv_heads": 2})
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_decoder_output_shape(self):
        attn = self._make(dims={"num_kv_heads": 2, "mode": 'decoder'})
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_kv_heads_one_mqa(self):
        attn = self._make(dims={"num_kv_heads": 1})
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_kv_heads_equal_num_heads(self):
        attn = self._make(dims={"num_kv_heads": 6})
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_logical_layer_idx_accepted(self):
        attn = self._make(dims={"num_kv_heads": 2})
        y = attn(self._x(), logical_layer_idx=3)
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))

    def test_invalid_hidden_size_raises(self):
        with self.assertRaises(ValueError):
            GroupedQueryAttention(_cfg(dims={'hidden_size': 50, 'num_heads': 6, 'num_kv_heads': 2}))

    def test_invalid_kv_heads_less_than_one_raises(self):
        with self.assertRaises(ValueError):
            GroupedQueryAttention(_cfg(dims={'num_heads': 6, 'num_kv_heads': 0}))

    def test_invalid_kv_heads_greater_than_num_heads_raises(self):
        with self.assertRaises(ValueError):
            GroupedQueryAttention(_cfg(dims={'num_heads': 6, 'num_kv_heads': 12}))

    def test_indivisible_kv_heads_raises(self):
        with self.assertRaises(ValueError):
            GroupedQueryAttention(_cfg(dims={'num_heads': 6, 'num_kv_heads': 4}))

    def test_gradient_flows(self):
        attn = self._make(dims={"num_kv_heads": 2})
        x = self._x().requires_grad_(True)
        attn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_batch_size_one(self):
        attn = self._make(dims={"num_kv_heads": 3})
        x = torch.randn(1, SEQ, DIM)
        y = attn(x)
        self.assertEqual(y.shape, (1, SEQ, DIM))

    def test_seq_len_one(self):
        attn = self._make(dims={"num_kv_heads": 2})
        x = torch.randn(BSZ, 1, DIM)
        y = attn(x)
        self.assertEqual(y.shape, (BSZ, 1, DIM))

    def test_with_bitnet(self):
        attn = GroupedQueryAttention(_cfg(bitnet={'enabled': True}, dims={'num_kv_heads': 2}))
        y = attn(self._x())
        self.assertEqual(y.shape, (BSZ, SEQ, DIM))


if __name__ == "__main__":
    unittest.main()
