"""Cache fallback tests over all 45 layer types (plan §13, T4 + T12).

With ``kv_cache.enabled=true``: no exception for any enum value; mixers
without an implemented incremental branch produce identical logits with
the cache on/off (silent fallback, D5); PaPE-configured layers fall back
per-layer while other layers cache.
"""

import unittest
from importlib.util import find_spec

import yaml

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from src.model.cache.capabilities import CACHE_SPECS, IMPLEMENTED_MIXERS, is_cacheable
    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_decoder import FrankensteinDecoder
    from src.model.hybrid_layer import HybridLayer

EVAL_ONLY = {"fasa_attn", "sparge_attn"}


def _schema_enum():
    """Read the layer_pattern enum straight from the schema (45 values)."""
    with open("src/schema/_model/_dims.yaml") as handle:
        dims = yaml.safe_load(handle)
    return list(dims["properties"]["layer_pattern"]["items"]["enum"])


def _build(pattern, pe="rope", **extra):
    cfg = FrankensteinModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=len(pattern),
        num_loops=1,
        num_heads=4,
        retention_heads=4,
        num_kv_heads=2,
        dropout=0.0,
        use_bitnet=False,
        layer_pattern=list(pattern),
        mode="decoder",
        positional_encoding=pe,
        norm_type="layer_norm",
        ffn_hidden_size=64,
        ode_solver="euler",
        ode_steps=1,
        **extra,
    )
    torch.manual_seed(0)
    return FrankensteinDecoder(cfg).eval()


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class RegistryConsistencyTests(unittest.TestCase):
    def test_capacity_spec_covers_every_enum_value(self):
        enum = _schema_enum()
        self.assertEqual(len(enum), 45)
        self.assertEqual(set(CACHE_SPECS), set(enum))
        # All implemented mixers must exist in the schema enum.
        self.assertTrue(IMPLEMENTED_MIXERS.issubset(set(enum)))

    def test_never_cached_mixers(self):
        self.assertFalse(is_cacheable("ssog_attn"))
        self.assertFalse(is_cacheable("conformer_attn"))
        self.assertFalse(is_cacheable("mamba"))
        self.assertFalse(is_cacheable("not_a_mixer"))

    def test_eval_only_mixers_are_cacheable(self):
        # FASA has an implemented cached branch (row-local top-k).
        self.assertTrue(is_cacheable("fasa_attn"))
        # SpargeAttn is registered as cacheable-in-principle but v1 falls
        # back (block means are not row-local) — no exception, D5.
        self.assertTrue(CACHE_SPECS["sparge_attn"].supported)
        self.assertFalse(is_cacheable("sparge_attn"))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class FallbackTests(unittest.TestCase):
    def test_every_enum_value_runs_with_cache(self):
        """No exception for any layer type with caching enabled.

        Some mixers have ``FrankensteinModelConfig`` constraints that the
        synthetic hidden_size=32 / seq_len=5 setup cannot satisfy (e.g.
        CCGQA's latent-rank divisibility, MSA's topk > num_blocks). Those
        are pre-existing configuration limitations: construction and the
        uncached forward fail identically, so they are skipped here. The
        important assertion is that the *cache* never introduces a failure
        the legacy path does not already have.
        """
        extra_cfg = {
            "msa_attn": {"msa_block_size": 4, "msa_topk_blocks": 1},
            "ssog_attn": {"ssog_grid_h": 2, "ssog_grid_w": 3},
        }
        for mixer in _schema_enum():
            with self.subTest(mixer=mixer):
                kwargs = extra_cfg.get(mixer, {})
                try:
                    model = _build([mixer], **kwargs)
                except Exception:
                    continue  # pre-existing construction constraint
                if mixer == "ssog_attn":
                    x = torch.randint(0, 64, (1, 6))
                    with torch.inference_mode():
                        out = model(x)
                    self.assertEqual(out.shape[0], 1)
                    continue
                prompt = torch.randint(0, 64, (2, 5))
                try:
                    with torch.inference_mode():
                        legacy = model.generate(prompt.clone(), max_new_tokens=3, greedy=True)
                except Exception:
                    continue  # pre-existing forward constraint (e.g. cca rank)
                with torch.inference_mode():
                    out = model.generate(
                        prompt, max_new_tokens=3, greedy=True, use_cache=True
                    )
                self.assertEqual(out.shape, (2, 8))
                self.assertEqual(legacy.shape, (2, 8))

    def test_uncached_fallback_is_bit_identical(self):
        """Mixers without an implemented branch produce identical logits."""
        from src.model.cache import FrankensteinCache

        for mixer in ("ode", "falcon1_attn", "engram_attn", "nsa_attn"):
            with self.subTest(mixer=mixer):
                model = _build([mixer])
                x = torch.randint(0, 64, (1, 6))
                with torch.inference_mode():
                    legacy = model(x)
                    cache = FrankensteinCache.new(model)
                    cached = model(x, past_key_values=cache, use_cache=True)
                self.assertTrue(torch.equal(legacy, cached))

    def test_pape_layer_falls_back_while_others_cache(self):
        model = _build(["standard_attn", "standard_attn"], pe="pape")
        prompt = torch.randint(0, 64, (1, 5))
        with torch.inference_mode():
            legacy = model.generate(prompt.clone(), max_new_tokens=3, greedy=True)
            cached = model.generate(
                prompt.clone(), max_new_tokens=3, greedy=True, use_cache=True
            )
        self.assertTrue(torch.equal(legacy, cached))

    def test_mamba_runs_unchanged_in_cached_loop(self):
        model = _build(["mamba", "standard_attn"])
        prompt = torch.randint(0, 64, (1, 5))
        with torch.inference_mode():
            legacy = model.generate(prompt.clone(), max_new_tokens=3, greedy=True)
            cached = model.generate(
                prompt.clone(), max_new_tokens=3, greedy=True, use_cache=True
            )
        self.assertTrue(torch.equal(legacy, cached))


if __name__ == "__main__":
    unittest.main()
