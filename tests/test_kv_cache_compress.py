"""KV-cache compression / eviction policy tests (plan F4).

StreamingLLM ``window_sinks``: pinned sinks, sliding window, correctness of
the retained region, and composition with quantization.
"""

import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from src.deploy.inference_config import InferenceConfig
    from src.model.cache import FrankensteinCache
    from src.model.cache.compress import WindowSinksPolicy, build_cache_policy
    from src.model.cache.quant import build_cache_codec
    from src.model.cache.slots import KVSlot
    from src.model.cache.base import SlotMeta
    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_decoder import FrankensteinDecoder


def _meta(idx=0):
    return SlotMeta(
        mixer_key="standard_attn",
        logical_layer_idx=idx,
        num_heads=2,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class WindowSinksSlotTests(unittest.TestCase):
    def test_budget_and_eviction(self):
        policy = WindowSinksPolicy(num_sinks=2, window=4)
        self.assertEqual(policy.budget(0), 6)
        slot = KVSlot(_meta(), policy=policy, num_sinks=2)
        for i in range(12):
            slot.append(
                k=torch.full((1, 2, 1, 4), float(i)),
                v=torch.full((1, 2, 1, 4), float(i)),
                positions=torch.tensor([i]),
            )
        out = slot.read()
        # 2 sinks + the 4 most recent positions.
        self.assertEqual(slot.length, 6)
        self.assertEqual(out["positions"].tolist(), [0, 1, 8, 9, 10, 11])
        self.assertEqual(out["k"][0, 0, :, 0].tolist(), [0.0, 1.0, 8.0, 9.0, 10.0, 11.0])

    def test_per_layer_budgets_override(self):
        policy = WindowSinksPolicy(num_sinks=1, window=8, per_layer_budgets=[3, 5])
        self.assertEqual(policy.budget(0), 1 + 3)
        self.assertEqual(policy.budget(1), 1 + 5)
        self.assertEqual(policy.budget(2), 1 + 3)  # wraps

    def test_registry(self):
        self.assertIsNone(build_cache_policy("none"))
        policy = build_cache_policy("window_sinks", num_sinks=2, window=16)
        self.assertIsInstance(policy, WindowSinksPolicy)
        with self.assertRaises(ValueError):
            build_cache_policy("h2o")  # enum-ready, not in v1

    def test_composes_with_quantization(self):
        codec = build_cache_codec("int8")
        policy = WindowSinksPolicy(num_sinks=2, window=4)
        slot = KVSlot(_meta(), codec=codec, policy=policy, num_sinks=2)
        for i in range(10):
            slot.append(
                k=torch.full((1, 2, 1, 4), float(i + 1)),
                v=torch.full((1, 2, 1, 4), float(i + 1)),
                positions=torch.tensor([i]),
            )
        out = slot.read()
        self.assertEqual(slot.length, 6)
        self.assertEqual(out["positions"].tolist(), [0, 1, 6, 7, 8, 9])
        # int8 round-trip of the retained values (each block is its own group).
        self.assertTrue(torch.allclose(out["k"][0, 0, :, 0], torch.tensor([1.0, 2.0, 7.0, 8.0, 9.0, 10.0]), atol=0.1))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class WindowSinksModelTests(unittest.TestCase):
    def test_cached_generation_with_eviction_runs(self):
        cfg = FrankensteinModelConfig(
            vocab_size=64, hidden_size=32, num_layers=1, num_loops=1,
            num_heads=4, num_kv_heads=2, dropout=0.0, use_bitnet=False,
            layer_pattern=["standard_attn"], mode="decoder",
            positional_encoding="rope", norm_type="layer_norm", ffn_hidden_size=64,
        )
        torch.manual_seed(0)
        model = FrankensteinDecoder(cfg).eval()
        infer_cfg = InferenceConfig(
            kv_cache_enabled=True,
            kv_compress_method="window_sinks",
            kv_num_sinks=2,
            kv_window=4,
        )
        prompt = torch.randint(0, 64, (1, 6))
        with torch.inference_mode():
            cache = FrankensteinCache.new(model, infer_cfg)
            out = model.generate(
                prompt, max_new_tokens=10, greedy=True, use_cache=True,
                past_key_values=cache,
            )
        self.assertEqual(out.shape, (1, 16))
        self.assertTrue(torch.isfinite(out).all())
        # After generation the retained cache is bounded by sinks + window.
        slot = cache.slot(0)
        self.assertLessEqual(slot.length, 2 + 4)
        # Pinned sinks stay at the front.
        positions = slot.read()["positions"].tolist()
        self.assertEqual(positions[:2], [0, 1])


if __name__ == "__main__":
    unittest.main()
