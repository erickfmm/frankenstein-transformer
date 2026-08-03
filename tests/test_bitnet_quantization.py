"""Tests for the BitNet ternary weight quantization pipeline.

Covers faithful round-trip packing, respect of ``bitnet_routers`` (float
parameters preserved), that an FP32 model is never silently ternarized,
``bake_ternary_weights`` correctness, and the save/load checkpoint path.
"""
import os
import tempfile
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from src.model.attention.common import BitLinear, weight_quant
    from src.model.tormented_bert_frankestein import UltraConfig, TormentedBertFrankenstein
    from src.deploy.quantization import (
        BitNetQuantizer,
        bake_bitnet_weights,
        save_quantized_checkpoint,
        load_quantized_checkpoint,
    )


def _cfg(bitnet_routers=False):
    return UltraConfig(
        vocab_size=64,
        hidden_size=64,
        num_layers=4,
        num_heads=4,
        num_experts=4,
        top_k_experts=2,
        dropout=0.0,
        use_bitnet=True,
        bitnet_routers=bitnet_routers,
        use_moe=True,
        use_mixture_of_depths=True,
        ffn_hidden_size=128,
        layer_pattern=["standard_attn", "gla_attn", "msa_attn", "sparsek_attn"],
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestBitNetQuantization(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_ternary_round_trip_is_bit_exact(self):
        """dequant(quantize(master)) == weight_quant(master), bit-exact."""
        m = TormentedBertFrankenstein(_cfg())
        m.eval()
        name = "layers.0.mixer.q_proj.weight"
        master = dict(m.named_parameters())[name].detach().clone()
        expected = weight_quant(master).detach()

        q = BitNetQuantizer()
        st = q.quantize_model_weights(m)
        self.assertIn(name, st["quantized_tensors"])

        m2 = TormentedBertFrankenstein(_cfg())
        q.dequantize_model_weights(st, m2)
        deq = dict(m2.named_parameters())[name].detach()
        self.assertTrue(torch.allclose(deq, expected, atol=0.0))

    def test_dequant_values_in_ternary_set(self):
        """Every dequantized BitLinear value is in {-scale, 0, +scale}."""
        m = TormentedBertFrankenstein(_cfg())
        m.eval()
        q = BitNetQuantizer()
        st = q.quantize_model_weights(m)
        m2 = TormentedBertFrankenstein(_cfg())
        q.dequantize_model_weights(st, m2)
        for name in st["quantized_tensors"]:
            w = dict(m2.named_parameters())[name].detach()
            scale = st["scales"][name]
            mask = (w == 0) | (w == scale) | (w == -scale)
            self.assertTrue(bool(mask.all()), f"{name} not ternary")

    def test_float_parameters_preserved(self):
        """Full-precision router weights survive a round-trip unchanged."""
        m = TormentedBertFrankenstein(_cfg(bitnet_routers=False))
        m.eval()
        ref = m.layers[0].router.weight.detach().clone()
        q = BitNetQuantizer()
        st = q.quantize_model_weights(m)
        m2 = TormentedBertFrankenstein(_cfg(bitnet_routers=False))
        q.dequantize_model_weights(st, m2)
        self.assertTrue(torch.allclose(m2.layers[0].router.weight, ref, atol=1e-6))

    def test_fp32_model_not_ternarized(self):
        """A model with use_bitnet=False has zero ternary tensors."""
        cfg = _cfg()
        cfg.use_bitnet = False
        m = TormentedBertFrankenstein(cfg)
        q = BitNetQuantizer()
        st = q.quantize_model_weights(m)
        self.assertEqual(st["quantized_tensors"], [])
        self.assertGreater(len(st["full_precision_tensors"]), 0)

    def test_bake_matches_one_weight_quant(self):
        """bake_bitnet_weights reproduces a single weight_quant application."""
        m = TormentedBertFrankenstein(_cfg())
        m.eval()
        name = "layers.0.mixer.q_proj.weight"
        master = dict(m.named_parameters())[name].detach().clone()
        expected = weight_quant(master).detach()
        n = bake_bitnet_weights(m)
        self.assertGreater(n, 0)
        self.assertTrue(torch.allclose(
            dict(m.named_parameters())[name].detach(), expected, atol=0.0,
        ))

    def test_bake_is_idempotent(self):
        """Baking an already-baked (ternary) weight is a no-op."""
        m = TormentedBertFrankenstein(_cfg())
        m.eval()
        bake_bitnet_weights(m)
        w_after = m.layers[0].mixer.q_proj.weight.detach().clone()
        bake_bitnet_weights(m)
        self.assertTrue(torch.allclose(
            m.layers[0].mixer.q_proj.weight, w_after, atol=0.0,
        ))

    def test_save_load_checkpoint_after_bake(self):
        """save -> load reproduces baked ternary weights bit-exactly."""
        m = TormentedBertFrankenstein(_cfg())
        m.eval()
        bake_bitnet_weights(m)
        path = os.path.join(self.tmpdir, "m.pt")
        save_quantized_checkpoint(m, path)
        m2 = TormentedBertFrankenstein(_cfg())
        load_quantized_checkpoint(path, m2)
        self.assertTrue(torch.allclose(
            m2.layers[0].mixer.q_proj.weight,
            m.layers[0].mixer.q_proj.weight, atol=0.0,
        ))


if __name__ == "__main__":
    unittest.main()