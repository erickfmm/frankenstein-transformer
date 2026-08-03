"""Unit tests for FrankensteinDecoder and FrankensteinTransformer."""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from src.model.frankenstein_model import (
        FrankensteinTransformer,
        FrankensteinDecoder,
        FrankensteinModelConfig,
    )


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
class FrankensteinTransformerTests(unittest.TestCase):
    def _model(self, layer_pattern=None, **kw):
        cfg = FrankensteinModelConfig(
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
        return FrankensteinTransformer(cfg)

    def test_flat_embedding_forward(self):
        model = self._model(use_factorized_embedding=False)
        x = torch.randint(0, 100, (2, 8))
        y = model(x)
        self.assertEqual(y.shape, (2, 8, 100))

    def test_mhc_forward(self):
        model = self._model(use_mhc=True, mhc_expansion_rate=2)
        x = torch.randint(0, 100, (2, 8))
        y = model(x)
        self.assertEqual(y.shape, (2, 8, 100))
        y.sum().backward()

    def test_mhc_decoder_forward(self):
        cfg = FrankensteinDecoder.build_decoder_config(vocab_size=100, hidden_size=48, num_layers=1, num_loops=1)
        cfg.use_mhc = True
        cfg.mhc_expansion_rate = 2
        model = FrankensteinDecoder(cfg)
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
        cfg = FrankensteinModelConfig(
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
        model = FrankensteinTransformer(cfg)
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
        cfg = FrankensteinModelConfig(
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
        cfg = FrankensteinModelConfig(
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
        FrankensteinModelConfig(
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
            FrankensteinModelConfig(
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
            FrankensteinModelConfig(
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
        cfg = FrankensteinModelConfig(
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
            FrankensteinModelConfig(
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
