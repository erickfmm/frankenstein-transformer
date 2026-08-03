"""Unit tests for UltraConfig dataclass."""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    from src.model.tormented_bert_frankestein import UltraConfig


def _minimal_dict(**overrides):
    """Return a minimal valid UltraConfig kwarg dict with overrides applied."""
    base = dict(
        vocab_size=100,
        hidden_size=48,
        num_layers=2,
        num_loops=1,
        num_heads=6,
        retention_heads=6,
        num_experts=2,
        top_k_experts=1,
        dropout=0.0,
        layer_pattern=["standard_attn"],
        ode_solver="rk4",
        ode_steps=1,
        use_bitnet=False,
        norm_type="layer_norm",
        use_factorized_embedding=False,
        use_moe=False,
        ffn_activation="gelu",
    )
    base.update(overrides)
    return base


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class UltraConfigDefaultsTests(unittest.TestCase):
    def test_default_construction_succeeds(self):
        cfg = UltraConfig()
        self.assertEqual(cfg.vocab_size, 50000)
        self.assertEqual(cfg.hidden_size, 2048)
        self.assertEqual(cfg.num_layers, 12)
        self.assertEqual(cfg.mode, "encoder")

    def test_ffn_hidden_size_auto_computed(self):
        cfg = UltraConfig(**_minimal_dict(hidden_size=64, ffn_hidden_size=None))
        self.assertEqual(cfg.ffn_hidden_size, 128)

    def test_ffn_hidden_size_explicit_respected(self):
        cfg = UltraConfig(**_minimal_dict(hidden_size=64, ffn_hidden_size=200))
        self.assertEqual(cfg.ffn_hidden_size, 200)

    def test_use_hope_true_sets_positional_encoding_to_hope(self):
        cfg = UltraConfig(**_minimal_dict(use_hope=True, positional_encoding=None))
        self.assertEqual(cfg.positional_encoding, "hope")
        self.assertTrue(cfg.use_hope)

    def test_use_hope_false_sets_positional_encoding_to_rope(self):
        cfg = UltraConfig(**_minimal_dict(use_hope=False, positional_encoding=None))
        self.assertEqual(cfg.positional_encoding, "rope")
        self.assertFalse(cfg.use_hope)

    def test_positional_encoding_hope_explicit(self):
        cfg = UltraConfig(**_minimal_dict(positional_encoding="hope"))
        self.assertEqual(cfg.positional_encoding, "hope")
        self.assertTrue(cfg.use_hope)

    def test_positional_encoding_rope_explicit(self):
        cfg = UltraConfig(**_minimal_dict(positional_encoding="rope"))
        self.assertEqual(cfg.positional_encoding, "rope")
        self.assertFalse(cfg.use_hope)

    def test_positional_encoding_case_insensitive(self):
        cfg = UltraConfig(**_minimal_dict(positional_encoding="HOPE"))
        self.assertEqual(cfg.positional_encoding, "hope")

    def test_invalid_positional_encoding_raises(self):
        with self.assertRaises(ValueError):
            UltraConfig(**_minimal_dict(positional_encoding="sinusoidal"))

    def test_encoder_mode_accepted(self):
        cfg = UltraConfig(**_minimal_dict(mode="encoder"))
        self.assertEqual(cfg.mode, "encoder")

    def test_decoder_mode_accepted(self):
        cfg = UltraConfig(**_minimal_dict(mode="decoder"))
        self.assertEqual(cfg.mode, "decoder")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            UltraConfig(**_minimal_dict(mode="bidirectional"))

    def test_layer_pattern_default_has_expected_types(self):
        cfg = UltraConfig()
        self.assertIsInstance(cfg.layer_pattern, list)
        self.assertGreater(len(cfg.layer_pattern), 0)
        for lt in cfg.layer_pattern:
            self.assertIsInstance(lt, str)

    def test_bitnet_routers_default_false(self):
        cfg = UltraConfig(**_minimal_dict())
        self.assertFalse(cfg.bitnet_routers)

    def test_bitnet_routers_true_accepted(self):
        cfg = UltraConfig(**_minimal_dict(use_bitnet=True, bitnet_routers=True))
        self.assertTrue(cfg.bitnet_routers)

    def test_use_bitnet_conv_default_false(self):
        cfg = UltraConfig(**_minimal_dict())
        self.assertFalse(cfg.use_bitnet_conv)

    def test_use_bitnet_conv_true_accepted(self):
        cfg = UltraConfig(**_minimal_dict(use_bitnet=True, use_bitnet_conv=True))
        self.assertTrue(cfg.use_bitnet_conv)


if __name__ == "__main__":
    unittest.main()
