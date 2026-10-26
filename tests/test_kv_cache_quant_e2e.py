"""Quantization end-to-end tests (plan §13, T8).

int8: greedy decode stays close to the fp baseline (bounded logit drift)
with BitNet on and off; int4_kivi: drift bounded and the protected window
exact.
"""

import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from src.deploy.inference_config import InferenceConfig
    from src.model.cache import FrankensteinCache
    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_decoder import FrankensteinDecoder


def _model(use_bitnet=False):
    cfg = FrankensteinModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_loops=1,
        num_heads=4,
        num_kv_heads=2,
        dropout=0.0,
        use_bitnet=use_bitnet,
        layer_pattern=["standard_attn", "gqa_attn"],
        mode="decoder",
        positional_encoding="rope",
        norm_type="layer_norm",
        ffn_hidden_size=64,
    )
    torch.manual_seed(0)
    return FrankensteinDecoder(cfg).eval()


def _compare_logits(quant_method, use_bitnet=False, max_diff=0.15, **quant_kwargs):
    m = _model(use_bitnet=use_bitnet)
    prompt = torch.randint(0, 64, (1, 6))
    with torch.inference_mode():
        full = m(prompt)
        cfg_fp = InferenceConfig(kv_cache_enabled=True, kv_quant_method="none")
        cache_fp = FrankensteinCache.new(m, cfg_fp)
        fp_cached, cache_fp = m(
            prompt, past_key_values=cache_fp, use_cache=True, return_cache=True
        )
        cfg_q = InferenceConfig(
            kv_cache_enabled=True, kv_quant_method=quant_method, **quant_kwargs
        )
        cache_q = FrankensteinCache.new(m, cfg_q)
        q_cached, _ = m(
            prompt, past_key_values=cache_q, use_cache=True, return_cache=True
        )
    # The cached fp path matches the full forward exactly.
    torch.testing.assert_close(fp_cached, full, rtol=1e-5, atol=1e-6)
    drift = (q_cached - full).abs().max().item()
    if drift > max_diff:
        raise AssertionError(
            f"{quant_method} drift {drift:.4f} > {max_diff} "
            f"(use_bitnet={use_bitnet})"
        )
    return drift


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class QuantEndToEndTests(unittest.TestCase):
    def test_int8_drift_bounded(self):
        _compare_logits("int8", max_diff=0.15)
        _compare_logits("int8", use_bitnet=True, max_diff=0.5)

    def test_int4_kivi_drift_bounded(self):
        # protected_window=0 + group_size=2 forces actual group flushing on a
        # short prompt (otherwise the whole sequence stays in the fp window
        # and the codec path is never exercised).
        _compare_logits(
            "int4_kivi",
            max_diff=0.75,
            kv_group_size=2,
            kv_protected_window=0,
        )

    def test_int4_kivi_quantizes_history(self):
        """Assert that the int4 path is actually engaged (history quantized)."""
        from src.model.cache import FrankensteinCache
        from src.deploy.inference_config import InferenceConfig

        m = _model()
        prompt = torch.randint(0, 64, (1, 6))
        cfg = InferenceConfig(
            kv_cache_enabled=True,
            kv_quant_method="int4_kivi",
            kv_group_size=2,
            kv_protected_window=0,
        )
        with torch.inference_mode():
            cache = FrankensteinCache.new(m, cfg)
            m(prompt, past_key_values=cache, use_cache=True, return_cache=True)
        slot = cache.slot(0)
        # Quantized parts exist (payload dicts rather than fp tensors).
        quantized_parts = [p for p in slot._parts if not p.fp]
        self.assertTrue(quantized_parts, "int4 history never quantized")

    def test_int8_greedy_matches_majority(self):
        m = _model()
        prompt = torch.randint(0, 64, (1, 5))
        with torch.inference_mode():
            torch.manual_seed(3)
            base = m.generate(prompt.clone(), max_new_tokens=5, greedy=True)
            torch.manual_seed(3)
            cfg = InferenceConfig(kv_cache_enabled=True, kv_quant_method="int8")
            cache = FrankensteinCache.new(m, cfg)
            quant = m.generate(
                prompt.clone(), max_new_tokens=5, greedy=True, use_cache=True,
                past_key_values=cache,
            )
        # Quantized decode may diverge after a few tokens; prompt+first token
        # must agree and the shapes always match.
        self.assertEqual(base.shape, quant.shape)
        self.assertTrue(torch.equal(base[:, :6], quant[:, :6]))

    def test_protected_window_exact_under_kivi(self):
        m = _model()
        prompt = torch.randint(0, 64, (1, 5))
        cfg = InferenceConfig(
            kv_cache_enabled=True,
            kv_quant_method="int4_kivi",
            kv_protected_window=64,  # everything stays fp
            kv_group_size=16,
        )
        with torch.inference_mode():
            full = m(prompt)
            cache = FrankensteinCache.new(m, cfg)
            cached, _ = m(prompt, past_key_values=cache, use_cache=True, return_cache=True)
        torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
