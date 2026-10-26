"""KV-cache mechanism-interaction tests (plan §13, T5).

Covers ``num_loops>1`` (slot identity per logical layer, D6), Hyperloop,
mHC, Mixture-of-Depths dense-prefill policy (D8), AttnRes fallback and
``layer_pattern`` heterogeneity.
"""

import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from src.model.cache import FrankensteinCache
    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_decoder import FrankensteinDecoder


def _cfg(**overrides):
    base = dict(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_loops=1,
        num_heads=4,
        retention_heads=4,
        num_kv_heads=2,
        dropout=0.0,
        use_bitnet=False,
        layer_pattern=["standard_attn", "gqa_attn"],
        mode="decoder",
        positional_encoding="rope",
        norm_type="layer_norm",
        ffn_hidden_size=64,
    )
    base.update(overrides)
    return FrankensteinModelConfig(**base)


def _paired(**overrides):
    torch.manual_seed(0)
    m1 = FrankensteinDecoder(_cfg(**overrides)).eval()
    m2 = FrankensteinDecoder(_cfg(**overrides)).eval()
    m2.load_state_dict(m1.state_dict())
    return m1, m2


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class NumLoopsTests(unittest.TestCase):
    def test_looped_depth_parity_and_distinct_slots(self):
        m1, m2 = _paired(num_loops=3)
        prompt = torch.randint(0, 64, (1, 5))
        with torch.inference_mode():
            legacy = m1.generate(prompt.clone(), max_new_tokens=3, greedy=True)
            cache = FrankensteinCache.new(m2)
            cached = m2.generate(
                prompt.clone(), max_new_tokens=3, greedy=True, use_cache=True,
                past_key_values=cache,
            )
        self.assertTrue(torch.equal(legacy, cached))
        # 2 physical layers x 3 loops => 6 logical slots (D6).
        self.assertEqual(sorted(cache.slots.keys()), [0, 1, 2, 3, 4, 5])


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class MhcTests(unittest.TestCase):
    def test_mhc_parity(self):
        m1, m2 = _paired(use_mhc=True, mhc_expansion_rate=2)
        prompt = torch.randint(0, 64, (1, 5))
        with torch.inference_mode():
            legacy = m1.generate(prompt.clone(), max_new_tokens=3, greedy=True)
            cached = m2.generate(
                prompt.clone(), max_new_tokens=3, greedy=True, use_cache=True
            )
        self.assertTrue(torch.equal(legacy, cached))

    def test_hyperloop_parity(self):
        m1, m2 = _paired(
            use_mhc=True,
            mhc_hyperloop=True,
            mhc_expansion_rate=2,
            mhc_hyperloop_begin_layers=1,
            mhc_hyperloop_end_layers=1,
            num_layers=3,
            num_loops=2,
            layer_pattern=["standard_attn", "gqa_attn", "standard_attn"],
        )
        prompt = torch.randint(0, 64, (1, 5))
        with torch.inference_mode():
            legacy = m1.generate(prompt.clone(), max_new_tokens=3, greedy=True)
            cached = m2.generate(
                prompt.clone(), max_new_tokens=3, greedy=True, use_cache=True
            )
        self.assertTrue(torch.equal(legacy, cached))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class MixtureOfDepthsTests(unittest.TestCase):
    def test_prefill_is_forced_dense_and_decode_keeps_router(self):
        cfg = dict(
            use_mixture_of_depths=True,
            mixture_of_depths_capacity_ratio=0.5,
        )
        m1, m2 = _paired(**cfg)
        prompt = torch.randint(0, 64, (1, 6))
        with torch.inference_mode():
            # With the cache, prefill runs dense (D8) — it must equal the
            # legacy *dense* forward, not the routed one.
            cache = FrankensteinCache.new(m2)
            prefill_logits, cache = m2(
                prompt, past_key_values=cache, use_cache=True, return_cache=True
            )
            layer = m2.backbone.layers[0]
            self.assertEqual(layer.last_mixture_of_depths_selected_fraction, 1.0)
            self.assertIsNone(layer.last_mixture_of_depths_aux_loss)
            # Decode (1 token) degenerates to dense by construction and runs.
            dec_logits, cache = m2(
                torch.tensor([[7]]),
                past_key_values=cache,
                use_cache=True,
                cache_position=cache.next_position(),
                return_cache=True,
            )
            self.assertEqual(dec_logits.shape, (1, 1, 64))
            self.assertEqual(prefill_logits.shape, (1, 6, 64))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class AttnResFallbackTests(unittest.TestCase):
    def test_attnres_disables_cache_with_warning(self):
        m1, m2 = _paired(residual_type="full_attn", num_layers=2)
        prompt = torch.randint(0, 64, (1, 5))
        with torch.inference_mode():
            import warnings

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cached = m2.generate(
                    prompt.clone(), max_new_tokens=3, greedy=True, use_cache=True
                )
            legacy = m1.generate(prompt.clone(), max_new_tokens=3, greedy=True)
        self.assertTrue(torch.equal(legacy, cached))
        self.assertTrue(any("AttnRes" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
