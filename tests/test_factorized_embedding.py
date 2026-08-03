"""Unit tests for FactorizedEmbedding."""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from src.model.embeddings import FactorizedEmbedding
    from src.model.tormented_bert_frankestein import UltraConfig


def _cfg(**overrides):
    base = dict(
        vocab_size=64,
        hidden_size=32,
        num_layers=1,
        num_loops=1,
        num_heads=4,
        retention_heads=4,
        num_experts=2,
        top_k_experts=1,
        dropout=0.0,
        norm_type="layer_norm",
        use_bitnet=False,
        layer_pattern=["standard_attn"],
        use_moe=False,
        ode_solver="rk4",
        ode_steps=1,
        ffn_hidden_size=64,
        ffn_activation="gelu",
        use_hope=True,
        use_factorized_embedding=True,
        factorized_embedding_dim=16,
        use_embedding_conv=False,
        embedding_conv_kernel=3,
    )
    base.update(overrides)
    return UltraConfig(**base)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class FactorizedEmbeddingTests(unittest.TestCase):
    def test_output_shape_no_conv(self):
        cfg = _cfg(use_embedding_conv=False)
        emb = FactorizedEmbedding(cfg)
        ids = torch.randint(0, 64, (2, 10))
        y = emb(ids)
        self.assertEqual(y.shape, (2, 10, 32))

    def test_output_shape_with_conv_odd_kernel(self):
        cfg = _cfg(use_embedding_conv=True, embedding_conv_kernel=3)
        emb = FactorizedEmbedding(cfg)
        ids = torch.randint(0, 64, (2, 10))
        y = emb(ids)
        self.assertEqual(y.shape, (2, 10, 32))

    def test_output_shape_with_conv_even_kernel(self):
        cfg = _cfg(use_embedding_conv=True, embedding_conv_kernel=4)
        emb = FactorizedEmbedding(cfg)
        ids = torch.randint(0, 64, (2, 10))
        y = emb(ids)
        # seq len must be preserved despite padding/clipping
        self.assertEqual(y.shape, (2, 10, 32))

    def test_output_shape_with_conv_kernel_1(self):
        cfg = _cfg(use_embedding_conv=True, embedding_conv_kernel=1)
        emb = FactorizedEmbedding(cfg)
        ids = torch.randint(0, 64, (3, 7))
        y = emb(ids)
        self.assertEqual(y.shape, (3, 7, 32))

    def test_conv_none_when_disabled(self):
        cfg = _cfg(use_embedding_conv=False)
        emb = FactorizedEmbedding(cfg)
        self.assertIsNone(emb.conv)

    def test_conv_not_none_when_enabled(self):
        cfg = _cfg(use_embedding_conv=True)
        emb = FactorizedEmbedding(cfg)
        self.assertIsNotNone(emb.conv)

    def test_seq_len_preserved_for_various_lengths(self):
        cfg = _cfg(use_embedding_conv=True, embedding_conv_kernel=3)
        emb = FactorizedEmbedding(cfg)
        for seq_len in [1, 5, 16, 32]:
            ids = torch.randint(0, 64, (1, seq_len))
            y = emb(ids)
            self.assertEqual(y.shape[1], seq_len, f"seq_len={seq_len}")

    def test_gradient_flows(self):
        cfg = _cfg(use_embedding_conv=True)
        emb = FactorizedEmbedding(cfg)
        ids = torch.randint(0, 64, (2, 8))
        emb(ids).sum().backward()  # should not raise

    def test_with_bitnet(self):
        cfg = _cfg(use_bitnet=True, use_embedding_conv=False)
        emb = FactorizedEmbedding(cfg)
        ids = torch.randint(0, 64, (2, 8))
        y = emb(ids)
        self.assertEqual(y.shape, (2, 8, 32))


if __name__ == "__main__":
    unittest.main()
