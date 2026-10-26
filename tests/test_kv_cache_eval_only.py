"""Eval-only mixer cache tests (plan §13, T6).

``fasa_attn`` is a decode-time selection method by design: it reads a full
K/V cache and skips computation. Cached decoding must work while the
training-mode guard still raises (unchanged behaviour).
"""

import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_decoder import FrankensteinDecoder


def _build():
    cfg = FrankensteinModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=1,
        num_loops=1,
        num_heads=4,
        num_kv_heads=2,
        dropout=0.0,
        use_bitnet=False,
        layer_pattern=["fasa_attn"],
        mode="decoder",
        positional_encoding="rope",
        norm_type="layer_norm",
        ffn_hidden_size=64,
    )
    torch.manual_seed(0)
    return FrankensteinDecoder(cfg)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class EvalOnlyCacheTests(unittest.TestCase):
    def test_fasa_cached_decode_matches_full(self):
        torch.manual_seed(42)
        m1 = _build().eval()
        m2 = _build().eval()
        m2.load_state_dict(m1.state_dict())
        prompt = torch.randint(0, 64, (1, 5))
        with torch.inference_mode():
            legacy = m1.generate(prompt.clone(), max_new_tokens=3, greedy=True)
            cached = m2.generate(
                prompt.clone(), max_new_tokens=3, greedy=True, use_cache=True
            )
        self.assertTrue(torch.equal(legacy, cached))

    def test_training_guard_still_raises(self):
        model = _build()  # train mode by default
        x = torch.randint(0, 64, (1, 5))
        with self.assertRaises(ValueError):
            model(x)


if __name__ == "__main__":
    unittest.main()
