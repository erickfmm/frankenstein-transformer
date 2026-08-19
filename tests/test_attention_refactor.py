import unittest

import torch

from src.model.attention import (
    BigBirdAttention,
    DeltaNetAttention,
    EngramLayer,
    FASAAttention,
    ForgettingAttention,
    GatedDeltaNet2Attention,
    GatedDeltaNetAttention,
    GatedLinearAttention,
    GatedSoftmaxAttention,
    GaussianMixtureAttention,
    GroupedQueryAttention,
    HGRN2Attention,
    HoPE,
    LongformerAttention,
    NSAAttention,
    ODEAttentionBlock,
    RetNetAttention,
    RoPE,
    SigmoidAttention,
    SparseKAttention,
    SparseTransformerAttention,
    SpargeAttention,
    SSOGAttention,
    StandardAttention,
    TitanAttention,
)
from src.model.config import FrankensteinModelConfig
from src.model.embeddings.alibi import ALiBi
from src.model.embeddings.factory import build_pos_encoder
from src.model.embeddings.hope import HoPE
from src.model.embeddings.nope import NoPE
from src.model.embeddings.pape import PaPE
from src.model.embeddings.rope import RoPE
from src.model.hybrid_layer import HybridLayer
from src.model.frankenstein_encoder import FrankensteinEncoder


class AttentionRefactorTests(unittest.TestCase):
    def _build_config(self, layer_pattern):
        return FrankensteinModelConfig(
            vocab_size=100,
            hidden_size=48,
            num_layers=1,
            num_loops=1,
            num_heads=6,
            retention_heads=6,
            num_experts=2,
            top_k_experts=1,
            dropout=0.0,
            layer_pattern=layer_pattern,
            ode_solver="rk4",
            ode_steps=1,
            use_bitnet=False,
            norm_type="layer_norm",
            use_factorized_embedding=False,
            factorized_embedding_dim=16,
            use_embedding_conv=False,
            embedding_conv_kernel=3,
            use_hope=True,
            use_moe=False,
            use_mixture_of_depths=False,
            mixture_of_depths_capacity_ratio=0.5,
            mixture_of_depths_router_aux_loss_weight=0.0,
            ffn_hidden_size=96,
            ffn_activation="gelu",
        )

    def test_import_smoke(self):
        self.assertTrue(callable(FrankensteinEncoder))
        self.assertTrue(callable(FrankensteinModelConfig))
        self.assertTrue(callable(EngramLayer))
        self.assertTrue(callable(TitanAttention))
        self.assertTrue(callable(StandardAttention))
        self.assertTrue(callable(SigmoidAttention))
        self.assertTrue(callable(ODEAttentionBlock))
        self.assertTrue(callable(HoPE))
        self.assertTrue(callable(RoPE))
        self.assertTrue(callable(SparseTransformerAttention))
        self.assertTrue(callable(LongformerAttention))
        self.assertTrue(callable(BigBirdAttention))
        self.assertTrue(callable(SparseKAttention))
        self.assertTrue(callable(NSAAttention))
        self.assertTrue(callable(SpargeAttention))
        self.assertTrue(callable(FASAAttention))
        self.assertTrue(callable(GatedLinearAttention))
        self.assertTrue(callable(DeltaNetAttention))
        self.assertTrue(callable(GatedDeltaNetAttention))
        self.assertTrue(callable(GatedDeltaNet2Attention))
        self.assertTrue(callable(RetNetAttention))
        self.assertTrue(callable(HGRN2Attention))
        self.assertTrue(callable(ForgettingAttention))
        self.assertTrue(callable(GatedSoftmaxAttention))
        self.assertTrue(callable(GroupedQueryAttention))
        self.assertTrue(callable(GaussianMixtureAttention))
        self.assertTrue(callable(SSOGAttention))

    def test_default_forward_compat(self):
        config = self._build_config(["titan_attn", "standard_attn"])
        config.num_layers = 2
        model = FrankensteinEncoder(config)
        x = torch.randint(0, config.vocab_size, (2, 8))
        y = model(x)
        self.assertEqual(y.shape, (2, 8, config.vocab_size))

    def test_legacy_use_hope_false_maps_to_rope(self):
        from src.model.embeddings.factory import build_pos_encoder
        config = self._build_config(["titan_attn"])
        config.use_hope = False
        config.positional_encoding = None
        config.__post_init__()
        pe = build_pos_encoder(config)
        attn = TitanAttention(config, pos_encoder=pe)
        from src.model.embeddings.rope import RoPE
        self.assertIsInstance(pe, RoPE)

    def test_positional_encoding_override(self):
        from src.model.embeddings.factory import build_pos_encoder
        from src.model.embeddings.hope import HoPE
        from src.model.embeddings.rope import RoPE
        base = self._build_config(["titan_attn"])
        cfg_hope = FrankensteinModelConfig(**{**base.__dict__, "positional_encoding": "hope"})
        cfg_rope = FrankensteinModelConfig(**{**base.__dict__, "positional_encoding": "rope"})
        pe_hope = build_pos_encoder(cfg_hope)
        pe_rope = build_pos_encoder(cfg_rope)
        self.assertIsInstance(pe_hope, HoPE)
        self.assertIsInstance(pe_rope, RoPE)

    def test_invalid_positional_encoding_raises(self):
        with self.assertRaisesRegex(ValueError, "positional_encoding"):
            FrankensteinModelConfig(**{**self._build_config(["titan_attn"]).__dict__, "positional_encoding": "invalid"})

    def test_layer_type_coverage_trainable(self):
        layer_types = [
            "titan_attn",
            "standard_attn",
            "sigmoid_attn",
            "ode",
            "retnet",
            "retnet_attn",
            "mamba",
            "sparse_transformer_attn",
            "longformer_attn",
            "bigbird_attn",
            "sparsek_attn",
            "nsa_attn",
            "gla_attn",
            "deltanet_attn",
            "gated_deltanet_attn",
            "gated_deltanet2_attn",
            "hgrn2_attn",
            "fox_attn",
            "gated_softmax_attn",
            "engram_attn",
            "gqa_attn",
            "gma_attn",
            "ssog_attn",
        ]
        for layer_type in layer_types:
            config = self._build_config([layer_type])
            if layer_type == "ssog_attn":
                # SSOG needs the seq len to match its grid: use a 1x6
                # degenerate grid for the 6-token smoke input.
                config.ssog_grid_h = 1
                config.ssog_grid_w = 6
            model = FrankensteinEncoder(config)
            x = torch.randint(0, config.vocab_size, (1, 6))
            y = model(x)
            self.assertEqual(y.shape, (1, 6, config.vocab_size), msg=layer_type)

    def test_training_free_sparse_layers_eval_only(self):
        for layer_type in ["fasa_attn", "sparge_attn"]:
            config = self._build_config([layer_type])
            model = FrankensteinEncoder(config)
            model.eval()
            x = torch.randint(0, config.vocab_size, (1, 6))
            with torch.no_grad():
                y = model(x)
            self.assertEqual(y.shape, (1, 6, config.vocab_size), msg=layer_type)

    def test_training_free_sparse_layers_raise_in_train_mode(self):
        for layer_type in ["fasa_attn", "sparge_attn"]:
            config = self._build_config([layer_type])
            model = FrankensteinEncoder(config)
            model.train()
            x = torch.randint(0, config.vocab_size, (1, 6))
            with self.assertRaisesRegex(ValueError, "training-free"):
                _ = model(x)

    def test_invalid_mixture_of_depths_capacity_ratio_raises(self):
        with self.assertRaisesRegex(ValueError, "mixture_of_depths_capacity_ratio"):
            FrankensteinModelConfig(
                **{
                    **self._build_config(["standard_attn"]).__dict__,
                    "use_mixture_of_depths": True,
                    "mixture_of_depths_capacity_ratio": 0.0,
                }
            )

    def test_mixture_of_depths_updates_only_selected_tokens(self):
        config = self._build_config(["mamba"])
        config.use_mixture_of_depths = True
        config.mixture_of_depths_capacity_ratio = 0.5
        layer = HybridLayer(config, "mamba")
        with torch.no_grad():
            layer.depth_router.weight.zero_()
            layer.depth_router.weight[0, 0] = 1.0
        x = torch.tensor(
            [
                [
                    [0.1] + [0.0] * 47,
                    [0.2] + [0.0] * 47,
                    [10.0] + [0.0] * 47,
                    [20.0] + [0.0] * 47,
                ]
            ],
            dtype=torch.float32,
        )
        y = layer(x)
        self.assertTrue(torch.allclose(y[:, :2, :], x[:, :2, :]))
        self.assertEqual(layer.last_mixture_of_depths_capacity, 2)

    def test_mixture_of_depths_collects_auxiliary_stats(self):
        config = self._build_config(["standard_attn"])
        config.use_mixture_of_depths = True
        config.mixture_of_depths_capacity_ratio = 0.5
        config.mixture_of_depths_router_aux_loss_weight = 0.25
        model = FrankensteinEncoder(config)
        x = torch.randint(0, config.vocab_size, (2, 6))
        y = model(x)
        self.assertEqual(y.shape, (2, 6, config.vocab_size))
        self.assertIn("mixture_of_depths_router_loss", model.last_auxiliary_losses)
        self.assertIn("average_selected_fraction", model.last_mixture_of_depths_stats)
        self.assertAlmostEqual(model.last_mixture_of_depths_stats["average_selected_fraction"], 0.5)
        self.assertGreater(
            float(model.last_auxiliary_losses["mixture_of_depths_router_loss"].detach().item()),
            0.0,
        )

    def test_gma_forward_shape_encoder_and_decoder(self):
        # Bidirectional (encoder) and causal (decoder) GMA must preserve
        # the (B, N, hidden_size) shape and produce finite outputs.
        for mode in ("encoder", "decoder"):
            config = self._build_config(["gma_attn"])
            config.mode = mode
            config.gma_num_components = 8
            config.gma_routing_dim = 16
            mixer = GaussianMixtureAttention(config)
            x = torch.randn(2, 7, config.hidden_size)
            with torch.no_grad():
                y = mixer(x)
            self.assertEqual(y.shape, (2, 7, config.hidden_size), msg=mode)
            self.assertTrue(torch.isfinite(y).all(), msg=mode)

    def test_gma_responsibilities_are_probabilistic(self):
        # Each row of the (B, N, H, K) responsibility tensor must sum to 1.
        config = self._build_config(["gma_attn"])
        config.gma_num_components = 5
        config.gma_routing_dim = 12
        mixer = GaussianMixtureAttention(config)
        x = torch.randn(1, 4, config.hidden_size)
        bsz, seq_len, _ = x.shape
        q = mixer.q_proj(x).view(bsz, seq_len, config.num_heads, mixer.routing_dim)
        gamma = mixer._responsibilities(q)
        row_sums = gamma.sum(dim=-1)
        self.assertEqual(gamma.shape, (1, 4, config.num_heads, 5))
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

    def test_ssog_forward_shape_2d_and_1d_grids(self):
        # 2D raster grid and the 1xW degenerate (sequence) grid must both
        # preserve (B, N, hidden_size) and stay finite, in train and eval.
        for grid_h, grid_w in ((2, 3), (1, 6)):
            config = self._build_config(["ssog_attn"])
            config.ssog_grid_h = grid_h
            config.ssog_grid_w = grid_w
            mixer = SSOGAttention(config)
            n = grid_h * grid_w
            x = torch.randn(2, n, config.hidden_size)
            y = mixer(x)
            self.assertEqual(y.shape, (2, n, config.hidden_size), msg=f"{grid_h}x{grid_w}")
            self.assertTrue(torch.isfinite(y).all(), msg=f"{grid_h}x{grid_w}")
            mixer.eval()
            with torch.no_grad():
                y_eval = mixer(x)
            self.assertTrue(torch.isfinite(y_eval).all(), msg=f"{grid_h}x{grid_w}")

    def test_ssog_cold_start_equals_fixed_field(self):
        # Zero-init probes behind ~0 gates: the steered forward at init
        # must reproduce the purely fixed geometric field.
        config = self._build_config(["ssog_attn"])
        config.ssog_grid_h = 2
        config.ssog_grid_w = 2
        config.ssog_num_atoms = 3
        torch.manual_seed(0)
        mixer = SSOGAttention(config)
        mixer.eval()
        x = torch.randn(2, 4, config.hidden_size)
        with torch.no_grad():
            y_steer = mixer(x)
            mixer.lookat = False
            y_fixed = mixer(x)
        self.assertTrue(torch.allclose(y_steer, y_fixed, atol=1e-6))

    def test_ssog_axis_kernels_row_stochastic(self):
        # Softmaxed kernels must sum to 1 over the key axis.
        config = self._build_config(["ssog_attn"])
        config.ssog_grid_h = 3
        config.ssog_grid_w = 4
        mixer = SSOGAttention(config)
        ay, ax = mixer.axis_kernels()
        self.assertEqual(ay.shape, (config.num_heads, mixer.num_atoms, 3, 3))
        self.assertEqual(ax.shape, (config.num_heads, mixer.num_atoms, 4, 4))
        self.assertTrue(torch.allclose(ay.sum(dim=-1), torch.ones_like(ay.sum(dim=-1)), atol=1e-5))
        self.assertTrue(torch.allclose(ax.sum(dim=-1), torch.ones_like(ax.sum(dim=-1)), atol=1e-5))

    def test_ssog_grid_mismatch_raises(self):
        config = self._build_config(["ssog_attn"])
        config.ssog_grid_h = 2
        config.ssog_grid_w = 2
        mixer = SSOGAttention(config)
        x = torch.randn(1, 5, config.hidden_size)
        with self.assertRaisesRegex(ValueError, "ssog_attn"):
            mixer(x)

    def test_ssog_decoder_mode_raises(self):
        base = self._build_config(["ssog_attn"])
        base.ssog_grid_h = 1
        base.ssog_grid_w = 6
        # Config-level check fires at construction time.
        with self.assertRaisesRegex(ValueError, "encoder-only"):
            FrankensteinModelConfig(**{**base.__dict__, "mode": "decoder"})
        # Mixer-level guard also fires (e.g. runtime mode forcing).
        config = FrankensteinModelConfig(**base.__dict__)
        config.mode = "decoder"
        with self.assertRaisesRegex(ValueError, "encoder-only"):
            SSOGAttention(config)

    def test_ssog_bitnet_and_gradient_flow(self):
        config = self._build_config(["ssog_attn"])
        config.ssog_grid_h = 2
        config.ssog_grid_w = 2
        config.use_bitnet = True
        torch.manual_seed(1)
        mixer = SSOGAttention(config)
        x = torch.randn(2, 4, config.hidden_size)
        y = mixer(x)
        self.assertTrue(torch.isfinite(y).all())
        y.sum().backward()
        for param_name in ("mu", "raw_sigma", "log_lambda", "raw_temperature",
                           "v_proj.weight", "out_proj.weight"):
            param = dict(mixer.named_parameters())[param_name]
            self.assertIsNotNone(param.grad, msg=param_name)
        # Steering probes must also receive gradients once gates open.
        probe = dict(mixer.named_parameters())["mu_delta.weight"]
        self.assertIsNotNone(probe.grad)
        self.assertTrue(torch.isfinite(probe.grad).all())

    def test_ssog_steering_changes_output(self):
        # With opened gates and non-zero probes, steered output must
        # diverge from the fixed field.
        config = self._build_config(["ssog_attn"])
        config.ssog_grid_h = 2
        config.ssog_grid_w = 2
        torch.manual_seed(2)
        mixer = SSOGAttention(config)
        mixer.eval()
        with torch.no_grad():
            mixer.raw_mu_delta_scale.fill_(2.0)
            mixer.mu_delta.weight.normal_(0.0, 0.5)
        x = torch.randn(1, 4, config.hidden_size)
        with torch.no_grad():
            y_steer = mixer(x)
            mixer.lookat = False
            y_fixed = mixer(x)
        self.assertFalse(torch.allclose(y_steer, y_fixed, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
