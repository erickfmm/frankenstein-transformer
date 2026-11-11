"""Unit tests for the nested FrankensteinModelConfig dataclass."""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    from src.model.attention.params import use_pe_for
    from src.model.config import FrankensteinModelConfig, merge_model_blocks


def _minimal_block(**overrides):
    """Return a minimal nested ``model:`` block with overrides applied."""
    base = {
        "dims": {
            "vocab_size": 100,
            "hidden_size": 48,
            "num_layers": 2,
            "num_loops": 1,
            "num_heads": 6,
            "dropout": 0.0,
            "layer_pattern": ["standard"],
        },
        "norm": {"type": "layer_norm"},
        "bitnet": {"enabled": False},
        "moe": {"enabled": False, "num_experts": 2, "top_k": 1},
        "attention": {"retnet": {"heads": 6}, "ode": {"solver": "rk4", "steps": 1}},
        "embedding": {"factorized": {"enabled": False}},
        "ffn": {"activation": "gelu"},
    }
    return merge_model_blocks(base, overrides)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class FrankensteinModelConfigDefaultsTests(unittest.TestCase):
    def test_default_construction_succeeds(self):
        cfg = FrankensteinModelConfig()
        self.assertEqual(cfg.dims.vocab_size, 50000)
        self.assertEqual(cfg.dims.hidden_size, 2048)
        self.assertEqual(cfg.dims.num_layers, 12)
        self.assertEqual(cfg.dims.mode, "encoder")

    def test_ffn_hidden_size_auto_computed(self):
        cfg = FrankensteinModelConfig.from_dict(
            _minimal_block(dims={"hidden_size": 64}, ffn={"hidden_size": None})
        )
        self.assertEqual(cfg.ffn.hidden_size, 128)

    def test_ffn_hidden_size_explicit_respected(self):
        cfg = FrankensteinModelConfig.from_dict(
            _minimal_block(dims={"hidden_size": 64}, ffn={"hidden_size": 200})
        )
        self.assertEqual(cfg.ffn.hidden_size, 200)

    def test_positional_encoding_defaults_to_rope(self):
        cfg = FrankensteinModelConfig.from_dict(_minimal_block(positional_encoding=None))
        self.assertEqual(cfg.positional_encoding, "rope")

    def test_positional_encoding_hope_explicit(self):
        cfg = FrankensteinModelConfig.from_dict(_minimal_block(positional_encoding="hope"))
        self.assertEqual(cfg.positional_encoding, "hope")

    def test_positional_encoding_rope_explicit(self):
        cfg = FrankensteinModelConfig.from_dict(_minimal_block(positional_encoding="rope"))
        self.assertEqual(cfg.positional_encoding, "rope")

    def test_positional_encoding_case_insensitive(self):
        cfg = FrankensteinModelConfig.from_dict(_minimal_block(positional_encoding="HOPE"))
        self.assertEqual(cfg.positional_encoding, "hope")

    def test_invalid_positional_encoding_raises(self):
        with self.assertRaises(ValueError):
            FrankensteinModelConfig.from_dict(_minimal_block(positional_encoding="sinusoidal"))

    def test_nope_positional_encoding_accepted(self):
        cfg = FrankensteinModelConfig.from_dict(_minimal_block(positional_encoding="nope"))
        self.assertEqual(cfg.positional_encoding, "nope")

    def test_alibi_positional_encoding_accepted(self):
        cfg = FrankensteinModelConfig.from_dict(_minimal_block(positional_encoding="alibi"))
        self.assertEqual(cfg.positional_encoding, "alibi")
        self.assertEqual(cfg.alibi_num_heads, cfg.dims.num_heads)

    def test_bam_positional_encoding_accepted(self):
        cfg = FrankensteinModelConfig.from_dict(_minimal_block(positional_encoding="bam"))
        self.assertEqual(cfg.positional_encoding, "bam")
        self.assertFalse(cfg.bam_learn_mu)
        self.assertAlmostEqual(cfg.bam_theta_init, 0.0)
        self.assertAlmostEqual(cfg.bam_eps, 1e-5)

    def test_bam_eps_validation(self):
        with self.assertRaises(ValueError):
            FrankensteinModelConfig.from_dict(
                _minimal_block(
                    positional_encoding="bam",
                    positional_encoding_parameters={"bam": {"eps": 0.0}},
                )
            )

    def test_use_ssmax_accepted(self):
        cfg = FrankensteinModelConfig.from_dict(
            _minimal_block(ssmax={"enabled": True, "s_init": 1.5})
        )
        self.assertTrue(cfg.ssmax.enabled)
        self.assertAlmostEqual(cfg.ssmax.s_init, 1.5)

    def test_ssmax_s_init_validation_when_use_ssmax_true(self):
        with self.assertRaises(ValueError):
            FrankensteinModelConfig.from_dict(
                _minimal_block(ssmax={"enabled": True, "s_init": 0.0})
            )

    def test_pape_positional_encoding_accepted(self):
        cfg = FrankensteinModelConfig.from_dict(_minimal_block(positional_encoding="pape"))
        self.assertEqual(cfg.positional_encoding, "pape")
        self.assertEqual(cfg.pape_num_parabolas, 4)

    def test_pape_efficient_positional_encoding_accepted(self):
        cfg = FrankensteinModelConfig.from_dict(
            _minimal_block(positional_encoding="pape_efficient")
        )
        self.assertEqual(cfg.positional_encoding, "pape_efficient")

    def test_pape_ri_positional_encoding_accepted(self):
        cfg = FrankensteinModelConfig.from_dict(_minimal_block(positional_encoding="pape_ri"))
        self.assertEqual(cfg.positional_encoding, "pape_ri")

    def test_sinusoidal_absolute_accepted(self):
        cfg = FrankensteinModelConfig.from_dict(
            _minimal_block(positional_encoding="sinusoidal_absolute")
        )
        self.assertEqual(cfg.positional_encoding, "sinusoidal_absolute")

    def test_sinusoidal_rotary_accepted(self):
        cfg = FrankensteinModelConfig.from_dict(
            _minimal_block(positional_encoding="sinusoidal_rotary")
        )
        self.assertEqual(cfg.positional_encoding, "sinusoidal_rotary")

    def test_learned_absolute_accepted(self):
        cfg = FrankensteinModelConfig.from_dict(
            _minimal_block(positional_encoding="learned_absolute")
        )
        self.assertEqual(cfg.positional_encoding, "learned_absolute")

    def test_none_positional_encoding_accepted(self):
        cfg = FrankensteinModelConfig.from_dict(_minimal_block(positional_encoding="none"))
        self.assertEqual(cfg.positional_encoding, "none")

    def test_pape_num_parabolas_validation(self):
        with self.assertRaises(ValueError):
            FrankensteinModelConfig.from_dict(
                _minimal_block(
                    positional_encoding="pape",
                    positional_encoding_parameters={"pape": {"num_parabolas": 0}},
                )
            )

    def test_pape_num_positions_validation(self):
        with self.assertRaises(ValueError):
            FrankensteinModelConfig.from_dict(
                _minimal_block(
                    positional_encoding="pape",
                    positional_encoding_parameters={"pape": {"num_positions": 0}},
                )
            )

    def test_per_mixer_use_pe_defaults(self):
        cfg = FrankensteinModelConfig.from_dict(_minimal_block())
        self.assertTrue(use_pe_for(cfg, "standard"))
        self.assertTrue(use_pe_for(cfg, "titan"))
        self.assertFalse(use_pe_for(cfg, "retnet"))
        self.assertFalse(use_pe_for(cfg, "mamba"))
        self.assertFalse(use_pe_for(cfg, "ode"))
        self.assertFalse(use_pe_for(cfg, "gla"))
        self.assertTrue(use_pe_for(cfg, "gated_softmax"))

    def test_per_mixer_use_pe_override_is_honoured(self):
        cfg = FrankensteinModelConfig.from_dict(
            _minimal_block(
                positional_encoding="rope",
                positional_encoding_parameters={
                    "use_pe": {"standard_use_pe": False, "retnet_use_pe": True}
                },
            )
        )
        self.assertFalse(use_pe_for(cfg, "standard"))
        self.assertTrue(use_pe_for(cfg, "retnet"))

    def test_encoder_mode_accepted(self):
        cfg = FrankensteinModelConfig.from_dict(_minimal_block(dims={"mode": "encoder"}))
        self.assertEqual(cfg.dims.mode, "encoder")

    def test_decoder_mode_accepted(self):
        cfg = FrankensteinModelConfig.from_dict(_minimal_block(dims={"mode": "decoder"}))
        self.assertEqual(cfg.dims.mode, "decoder")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            FrankensteinModelConfig.from_dict(_minimal_block(dims={"mode": "bidirectional"}))

    def test_layer_pattern_default_has_expected_types(self):
        cfg = FrankensteinModelConfig()
        self.assertIsInstance(cfg.dims.layer_pattern, list)
        self.assertGreater(len(cfg.dims.layer_pattern), 0)
        for lt in cfg.dims.layer_pattern:
            self.assertIsInstance(lt, str)

    def test_bitnet_routers_default_false(self):
        cfg = FrankensteinModelConfig.from_dict(_minimal_block())
        self.assertFalse(cfg.bitnet.routers)

    def test_bitnet_routers_true_accepted(self):
        cfg = FrankensteinModelConfig.from_dict(
            _minimal_block(bitnet={"enabled": True, "routers": True})
        )
        self.assertTrue(cfg.bitnet.routers)

    def test_use_bitnet_conv_default_false(self):
        cfg = FrankensteinModelConfig.from_dict(_minimal_block())
        self.assertFalse(cfg.bitnet.conv)

    def test_no_flat_constructor_or_aliases(self):
        import src.model.config as config_module

        self.assertFalse(hasattr(FrankensteinModelConfig, "from_flat"))
        self.assertFalse(hasattr(config_module, "_FLAT_ALIASES"))
        with self.assertRaises(AttributeError):
            _ = FrankensteinModelConfig().hidden_size

    def test_use_bitnet_conv_true_accepted(self):
        cfg = FrankensteinModelConfig.from_dict(
            _minimal_block(bitnet={"enabled": True, "conv": True})
        )
        self.assertTrue(cfg.bitnet.conv)


if __name__ == "__main__":
    unittest.main()
