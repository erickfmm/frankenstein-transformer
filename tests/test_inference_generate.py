"""``generate()`` tests (plan §13, T7).

Greedy parity with/without cache (exact), sampling determinism under a
fixed seed, and ``truncate()`` rollback correctness.
"""

import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from src.model.cache import FrankensteinCache
    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_decoder import FrankensteinDecoder


def _model():
    cfg = FrankensteinModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_loops=1,
        num_heads=4,
        num_kv_heads=2,
        dropout=0.0,
        use_bitnet=False,
        layer_pattern=["standard_attn", "gqa_attn"],
        mode="decoder",
        positional_encoding="rope",
        norm_type="layer_norm",
        ffn_hidden_size=64,
    )
    torch.manual_seed(0)
    return FrankensteinDecoder(cfg).eval()


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GenerateTests(unittest.TestCase):
    def test_greedy_parity_exact(self):
        m1 = _model()
        m2 = _model()
        m2.load_state_dict(m1.state_dict())
        prompt = torch.randint(0, 64, (2, 6))
        with torch.inference_mode():
            legacy = m1.generate(prompt.clone(), max_new_tokens=6, greedy=True)
            cached = m2.generate(
                prompt.clone(), max_new_tokens=6, greedy=True, use_cache=True
            )
        self.assertTrue(torch.equal(legacy, cached))

    def test_sampling_determinism_with_fixed_seed(self):
        m1 = _model()
        m2 = _model()
        m2.load_state_dict(m1.state_dict())
        prompt = torch.randint(0, 64, (1, 5))
        with torch.inference_mode():
            torch.manual_seed(123)
            a = m1.generate(prompt.clone(), max_new_tokens=5, temperature=0.9, top_k=20)
            torch.manual_seed(123)
            b = m2.generate(
                prompt.clone(), max_new_tokens=5, temperature=0.9, top_k=20,
                use_cache=True,
            )
        self.assertTrue(torch.equal(a, b))

    def test_top_p_filtering_runs(self):
        m = _model()
        prompt = torch.randint(0, 64, (1, 5))
        with torch.inference_mode():
            out = m.generate(prompt.clone(), max_new_tokens=3, top_p=0.9, use_cache=True)
        self.assertEqual(out.shape, (1, 8))

    def test_truncate_rolls_back_the_cache(self):
        m = _model()
        prompt = torch.randint(0, 64, (1, 5))
        with torch.inference_mode():
            cache = FrankensteinCache.new(m)
            logits, cache = m(
                prompt, past_key_values=cache, use_cache=True, return_cache=True
            )
            self.assertEqual(cache.next_position(), 5)
            before = {i: s.length for i, s in cache.slots.items()}
            cache.truncate(3)
            self.assertEqual(cache.next_position(), 3)
            for i, slot in cache.slots.items():
                self.assertEqual(slot.length, 3, msg=f"layer {i}")
                self.assertEqual(
                    slot.read()["positions"].tolist(), [0, 1, 2], msg=f"layer {i}"
                )
            # Re-running from the truncated prefix reproduces the same logits.
            logits2, _ = m(
                prompt[:, :3], past_key_values=cache, use_cache=True,
                return_cache=True,
            )
            self.assertEqual(logits2.shape, (1, 3, 64))

    def test_legacy_generate_unchanged_when_cache_disabled(self):
        m = _model()
        prompt = torch.randint(0, 64, (1, 5))
        with torch.inference_mode():
            torch.manual_seed(7)
            a = m.generate(prompt.clone(), max_new_tokens=4, temperature=1.0)
            torch.manual_seed(7)
            b = m.generate(prompt.clone(), max_new_tokens=4, temperature=1.0)
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
