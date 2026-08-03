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
    StandardAttention,
    TitanAttention,
)
from src.model.tormented_bert_frankestein import HybridLayer, TormentedBertFrankenstein, UltraConfig


class AttentionRefactorTests(unittest.TestCase):
    def _build_config(self, layer_pattern):
        return UltraConfig(
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
        self.assertTrue(callable(TormentedBertFrankenstein))
        self.assertTrue(callable(UltraConfig))
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

    def test_default_forward_compat(self):
        config = self._build_config(["titan_attn", "standard_attn"])
        config.num_layers = 2
        model = TormentedBertFrankenstein(config)
        x = torch.randint(0, config.vocab_size, (2, 8))
        y = model(x)
        self.assertEqual(y.shape, (2, 8, config.vocab_size))

    def test_legacy_use_hope_false_maps_to_rope(self):
        config = self._build_config(["titan_attn"])
        config.use_hope = False
        config.positional_encoding = None
        attn = TitanAttention(config)
        self.assertIsInstance(attn.pos_encoder, RoPE)

    def test_positional_encoding_override(self):
        base = self._build_config(["titan_attn"])
        cfg_hope = UltraConfig(**{**base.__dict__, "positional_encoding": "hope"})
        cfg_rope = UltraConfig(**{**base.__dict__, "positional_encoding": "rope"})
        self.assertIsInstance(TitanAttention(cfg_hope).pos_encoder, HoPE)
        self.assertIsInstance(TitanAttention(cfg_rope).pos_encoder, RoPE)

    def test_invalid_positional_encoding_raises(self):
        with self.assertRaisesRegex(ValueError, "positional_encoding"):
            UltraConfig(**{**self._build_config(["titan_attn"]).__dict__, "positional_encoding": "invalid"})

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
        ]
        for layer_type in layer_types:
            config = self._build_config([layer_type])
            model = TormentedBertFrankenstein(config)
            x = torch.randint(0, config.vocab_size, (1, 6))
            y = model(x)
            self.assertEqual(y.shape, (1, 6, config.vocab_size), msg=layer_type)

    def test_training_free_sparse_layers_eval_only(self):
        for layer_type in ["fasa_attn", "sparge_attn"]:
            config = self._build_config([layer_type])
            model = TormentedBertFrankenstein(config)
            model.eval()
            x = torch.randint(0, config.vocab_size, (1, 6))
            with torch.no_grad():
                y = model(x)
            self.assertEqual(y.shape, (1, 6, config.vocab_size), msg=layer_type)

    def test_training_free_sparse_layers_raise_in_train_mode(self):
        for layer_type in ["fasa_attn", "sparge_attn"]:
            config = self._build_config([layer_type])
            model = TormentedBertFrankenstein(config)
            model.train()
            x = torch.randint(0, config.vocab_size, (1, 6))
            with self.assertRaisesRegex(ValueError, "training-free"):
                _ = model(x)

    def test_invalid_mixture_of_depths_capacity_ratio_raises(self):
        with self.assertRaisesRegex(ValueError, "mixture_of_depths_capacity_ratio"):
            UltraConfig(
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
        model = TormentedBertFrankenstein(config)
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


if __name__ == "__main__":
    unittest.main()
