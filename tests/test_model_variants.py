"""Unit tests for TormentedBertMini and FrankensteinDecoder."""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from src.model.tormented_bert_frankestein import (
        TormentedBertFrankenstein,
        TormentedBertMini,
        FrankensteinDecoder,
        UltraConfig,
    )


def _mini_config(**overrides):
    cfg = TormentedBertMini.build_mini_config(vocab_size=200, use_bitnet=False)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class TormentedBertMiniTests(unittest.TestCase):
    def test_default_construction(self):
        model = TormentedBertMini()
        self.assertIsInstance(model, TormentedBertMini)

    def test_forward_shape_default(self):
        model = TormentedBertMini()
        x = torch.randint(0, model.config.vocab_size, (2, 8))
        y = model(x)
        self.assertEqual(y.shape, (2, 8, model.config.vocab_size))

    def test_custom_config(self):
        cfg = _mini_config()
        model = TormentedBertMini(cfg)
        x = torch.randint(0, cfg.vocab_size, (1, 6))
        y = model(x)
        self.assertEqual(y.shape, (1, 6, cfg.vocab_size))

    def test_forces_factorized_embedding_true(self):
        cfg = _mini_config()
        cfg.use_factorized_embedding = False
        model = TormentedBertMini(cfg)
        self.assertTrue(model.config.use_factorized_embedding)

    def test_build_mini_config_uses_derf_norm(self):
        cfg = TormentedBertMini.build_mini_config()
        self.assertEqual(cfg.norm_type, "derf")

    def test_build_mini_config_layer_pattern_is_stable(self):
        cfg = TormentedBertMini.build_mini_config()
        for lt in cfg.layer_pattern:
            self.assertIn(lt, ["retnet", "titan_attn", "mamba", "ode"])

    def test_gradient_flows(self):
        cfg = _mini_config()
        model = TormentedBertMini(cfg)
        x = torch.randint(0, cfg.vocab_size, (1, 4))
        model(x).sum().backward()


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class FrankensteinDecoderTests(unittest.TestCase):
    def _small_decoder(self, **kw):
        cfg = FrankensteinDecoder.build_decoder_config(
            vocab_size=200,
            hidden_size=48,
            num_layers=2,
            num_loops=1,
            use_bitnet=False,
            **kw,
        )
        return FrankensteinDecoder(cfg)

    def test_forward_shape(self):
        model = self._small_decoder()
        x = torch.randint(0, 200, (2, 6))
        y = model(x)
        self.assertEqual(y.shape, (2, 6, 200))

    def test_mode_forced_to_decoder(self):
        cfg = FrankensteinDecoder.build_decoder_config(vocab_size=200)
        cfg.mode = "encoder"  # override to test that the decoder enforces mode
        model = FrankensteinDecoder(cfg)
        self.assertEqual(model.config.mode, "decoder")

    def test_default_config_is_decoder(self):
        cfg = FrankensteinDecoder.build_decoder_config(vocab_size=200)
        self.assertEqual(cfg.mode, "decoder")

    def test_generate_output_length(self):
        model = self._small_decoder()
        model.eval()
        x = torch.randint(0, 200, (1, 4))
        new_tokens = 3
        out = model.generate(x, max_new_tokens=new_tokens, temperature=1.0, top_k=10)
        self.assertEqual(out.shape, (1, 4 + new_tokens))

    def test_generate_greedy_top_k_zero(self):
        model = self._small_decoder()
        model.eval()
        x = torch.randint(0, 200, (1, 3))
        out = model.generate(x, max_new_tokens=2, temperature=1.0, top_k=0)
        self.assertEqual(out.shape, (1, 5))

    def test_generate_temperature_small(self):
        model = self._small_decoder()
        model.eval()
        x = torch.randint(0, 200, (1, 4))
        out = model.generate(x, max_new_tokens=2, temperature=0.01, top_k=5)
        self.assertEqual(out.shape, (1, 6))

    def test_generate_does_not_modify_input(self):
        model = self._small_decoder()
        model.eval()
        x = torch.randint(0, 200, (1, 4))
        x_clone = x.clone()
        model.generate(x, max_new_tokens=2)
        self.assertTrue(torch.equal(x, x_clone))

    def test_custom_layer_pattern(self):
        model = self._small_decoder(layer_pattern=["titan_attn", "mamba"])
        x = torch.randint(0, 200, (1, 5))
        y = model(x)
        self.assertEqual(y.shape, (1, 5, 200))

    def test_gradient_flows(self):
        model = self._small_decoder()
        x = torch.randint(0, 200, (1, 4))
        model(x).sum().backward()


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class TormentedBertFrankensteinTests(unittest.TestCase):
    def _model(self, layer_pattern=None, **kw):
        cfg = UltraConfig(
            vocab_size=100,
            hidden_size=48,
            num_layers=len(layer_pattern or ["standard_attn"]),
            num_loops=1,
            num_heads=6,
            retention_heads=6,
            num_experts=2,
            top_k_experts=1,
            dropout=0.0,
            norm_type=kw.pop("norm_type", "layer_norm"),
            use_bitnet=False,
            layer_pattern=layer_pattern or ["standard_attn"],
            use_moe=False,
            ode_solver="rk4",
            ode_steps=1,
            ffn_hidden_size=96,
            ffn_activation="gelu",
            use_hope=True,
            **kw,
        )
        return TormentedBertFrankenstein(cfg)

    def test_flat_embedding_forward(self):
        model = self._model(use_factorized_embedding=False)
        x = torch.randint(0, 100, (2, 8))
        y = model(x)
        self.assertEqual(y.shape, (2, 8, 100))

    def test_factorized_embedding_forward(self):
        model = self._model(
            use_factorized_embedding=True,
            factorized_embedding_dim=16,
            use_embedding_conv=True,
        )
        x = torch.randint(0, 100, (2, 8))
        y = model(x)
        self.assertEqual(y.shape, (2, 8, 100))

    def test_multi_loop(self):
        cfg = UltraConfig(
            vocab_size=100,
            hidden_size=48,
            num_layers=2,
            num_loops=3,
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
        )
        model = TormentedBertFrankenstein(cfg)
        x = torch.randint(0, 100, (1, 5))
        y = model(x)
        self.assertEqual(y.shape, (1, 5, 100))

    def test_dynamic_tanh_norm(self):
        model = self._model(norm_type="dynamic_tanh")
        x = torch.randint(0, 100, (1, 4))
        y = model(x)
        self.assertEqual(y.shape, (1, 4, 100))

    def test_rms_norm_forward(self):
        model = self._model(norm_type="rms_norm")
        x = torch.randint(0, 100, (1, 4))
        y = model(x)
        self.assertEqual(y.shape, (1, 4, 100))

    def test_prms_norm_forward(self):
        model = self._model(norm_type="prms_norm", prms_partial_ratio=0.5)
        x = torch.randint(0, 100, (1, 4))
        y = model(x)
        self.assertEqual(y.shape, (1, 4, 100))

    def test_flash_norm_forward(self):
        model = self._model(norm_type="flash_norm")
        x = torch.randint(0, 100, (1, 4))
        y = model(x)
        self.assertEqual(y.shape, (1, 4, 100))

    def test_flash_norm_partial_ratio_forward(self):
        model = self._model(norm_type="flash_norm", flashnorm_partial_ratio=0.25)
        x = torch.randint(0, 100, (1, 4))
        y = model(x)
        self.assertEqual(y.shape, (1, 4, 100))

    def test_flash_norm_default_partial_ratio(self):
        cfg = UltraConfig(
            vocab_size=100,
            hidden_size=48,
            num_layers=1,
            num_loops=1,
            num_heads=6,
            retention_heads=6,
            num_experts=2,
            top_k_experts=1,
            dropout=0.0,
            norm_type="flash_norm",
            use_bitnet=False,
            layer_pattern=["standard_attn"],
            use_moe=False,
            ode_solver="rk4",
            ode_steps=1,
            ffn_hidden_size=96,
            ffn_activation="gelu",
        )
        # Default flashnorm_partial_ratio is 0.0 (full RMS, not partial).
        self.assertAlmostEqual(cfg.flashnorm_partial_ratio, 0.0)

    def test_flashnorm_partial_ratio_validation(self):
        # 0.0 is valid (full RMS).
        cfg = UltraConfig(
            vocab_size=100,
            hidden_size=48,
            num_layers=1,
            num_loops=1,
            num_heads=6,
            retention_heads=6,
            num_experts=2,
            top_k_experts=1,
            dropout=0.0,
            norm_type="flash_norm",
            flashnorm_partial_ratio=0.0,
            use_bitnet=False,
            layer_pattern=["standard_attn"],
            use_moe=False,
            ode_solver="rk4",
            ode_steps=1,
            ffn_hidden_size=96,
            ffn_activation="gelu",
        )
        self.assertAlmostEqual(cfg.flashnorm_partial_ratio, 0.0)
        # 1.0 is valid (upper bound).
        UltraConfig(
            vocab_size=100,
            hidden_size=48,
            num_layers=1,
            num_loops=1,
            num_heads=6,
            retention_heads=6,
            num_experts=2,
            top_k_experts=1,
            dropout=0.0,
            norm_type="flash_norm",
            flashnorm_partial_ratio=1.0,
            use_bitnet=False,
            layer_pattern=["standard_attn"],
            use_moe=False,
            ode_solver="rk4",
            ode_steps=1,
            ffn_hidden_size=96,
            ffn_activation="gelu",
        )
        # Negative is invalid.
        with self.assertRaises(ValueError):
            UltraConfig(
                vocab_size=100,
                hidden_size=48,
                num_layers=1,
                num_loops=1,
                num_heads=6,
                retention_heads=6,
                num_experts=2,
                top_k_experts=1,
                dropout=0.0,
                norm_type="flash_norm",
                flashnorm_partial_ratio=-0.1,
                use_bitnet=False,
                layer_pattern=["standard_attn"],
                use_moe=False,
                ode_solver="rk4",
                ode_steps=1,
                ffn_hidden_size=96,
                ffn_activation="gelu",
            )
        # > 1.0 is invalid.
        with self.assertRaises(ValueError):
            UltraConfig(
                vocab_size=100,
                hidden_size=48,
                num_layers=1,
                num_loops=1,
                num_heads=6,
                retention_heads=6,
                num_experts=2,
                top_k_experts=1,
                dropout=0.0,
                norm_type="flash_norm",
                flashnorm_partial_ratio=1.5,
                use_bitnet=False,
                layer_pattern=["standard_attn"],
                use_moe=False,
                ode_solver="rk4",
                ode_steps=1,
                ffn_hidden_size=96,
                ffn_activation="gelu",
            )

    def test_prms_norm_default_ratio(self):
        cfg = UltraConfig(
            vocab_size=100,
            hidden_size=48,
            num_layers=1,
            num_loops=1,
            num_heads=6,
            retention_heads=6,
            num_experts=2,
            top_k_experts=1,
            dropout=0.0,
            norm_type="prms_norm",
            use_bitnet=False,
            layer_pattern=["standard_attn"],
            use_moe=False,
            ode_solver="rk4",
            ode_steps=1,
            ffn_hidden_size=96,
            ffn_activation="gelu",
        )
        self.assertAlmostEqual(cfg.prms_partial_ratio, 0.0625)

    def test_prms_partial_ratio_validation(self):
        with self.assertRaises(ValueError):
            UltraConfig(
                vocab_size=100,
                hidden_size=48,
                num_layers=1,
                num_loops=1,
                num_heads=6,
                retention_heads=6,
                num_experts=2,
                top_k_experts=1,
                dropout=0.0,
                norm_type="prms_norm",
                prms_partial_ratio=0.0,
                use_bitnet=False,
                layer_pattern=["standard_attn"],
                use_moe=False,
                ode_solver="rk4",
                ode_steps=1,
                ffn_hidden_size=96,
                ffn_activation="gelu",
            )


if __name__ == "__main__":
    unittest.main()
