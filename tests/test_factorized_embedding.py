"""Unit tests for FactorizedEmbedding."""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from src.model.embeddings import FactorizedEmbedding
    from src.model.config import FrankensteinModelConfig, merge_model_blocks


def _cfg(**overrides):
    base = {"dims": {"vocab_size": 64, "hidden_size": 32, "num_layers": 1, "num_loops": 1, "num_heads": 4, "dropout": 0.0, "layer_pattern": ['standard']}, "attention": {"retnet": {"heads": 4}, "ode": {"solver": 'rk4', "steps": 1}}, "moe": {"num_experts": 2, "top_k": 1, "enabled": False}, "norm": {"type": 'layer_norm'}, "bitnet": {"enabled": False}, "ffn": {"hidden_size": 64, "activation": 'gelu'}, "positional_encoding": 'hope', "embedding": {"factorized": {"enabled": True, "dim": 16}, "conv": {"enabled": False, "kernel": 3}}}
    base = merge_model_blocks(base, overrides)
    return FrankensteinModelConfig.from_dict({**base})


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class FactorizedEmbeddingTests(unittest.TestCase):
    def test_output_shape_no_conv(self):
        cfg = _cfg(embedding={'conv': {'enabled': False}})
        emb = FactorizedEmbedding(cfg)
        ids = torch.randint(0, 64, (2, 10))
        y = emb(ids)
        self.assertEqual(y.shape, (2, 10, 32))

    def test_output_shape_with_conv_odd_kernel(self):
        cfg = _cfg(embedding={'conv': {'enabled': True, 'kernel': 3}})
        emb = FactorizedEmbedding(cfg)
        ids = torch.randint(0, 64, (2, 10))
        y = emb(ids)
        self.assertEqual(y.shape, (2, 10, 32))

    def test_output_shape_with_conv_even_kernel(self):
        cfg = _cfg(embedding={'conv': {'enabled': True, 'kernel': 4}})
        emb = FactorizedEmbedding(cfg)
        ids = torch.randint(0, 64, (2, 10))
        y = emb(ids)
        # seq len must be preserved despite padding/clipping
        self.assertEqual(y.shape, (2, 10, 32))

    def test_output_shape_with_conv_kernel_1(self):
        cfg = _cfg(embedding={'conv': {'enabled': True, 'kernel': 1}})
        emb = FactorizedEmbedding(cfg)
        ids = torch.randint(0, 64, (3, 7))
        y = emb(ids)
        self.assertEqual(y.shape, (3, 7, 32))

    def test_conv_none_when_disabled(self):
        cfg = _cfg(embedding={'conv': {'enabled': False}})
        emb = FactorizedEmbedding(cfg)
        self.assertIsNone(emb.conv)

    def test_conv_not_none_when_enabled(self):
        cfg = _cfg(embedding={'conv': {'enabled': True}})
        emb = FactorizedEmbedding(cfg)
        self.assertIsNotNone(emb.conv)

    def test_seq_len_preserved_for_various_lengths(self):
        cfg = _cfg(embedding={'conv': {'enabled': True, 'kernel': 3}})
        emb = FactorizedEmbedding(cfg)
        for seq_len in [1, 5, 16, 32]:
            ids = torch.randint(0, 64, (1, seq_len))
            y = emb(ids)
            self.assertEqual(y.shape[1], seq_len, f"seq_len={seq_len}")

    def test_gradient_flows(self):
        cfg = _cfg(embedding={'conv': {'enabled': True}})
        emb = FactorizedEmbedding(cfg)
        ids = torch.randint(0, 64, (2, 8))
        emb(ids).sum().backward()  # should not raise

    def test_with_bitnet(self):
        cfg = _cfg(bitnet={'enabled': True}, embedding={'conv': {'enabled': False}})
        emb = FactorizedEmbedding(cfg)
        ids = torch.randint(0, 64, (2, 8))
        y = emb(ids)
        self.assertEqual(y.shape, (2, 8, 32))


if __name__ == "__main__":
    unittest.main()
