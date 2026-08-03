"""Tests for the HuggingFace Transformers export with BitNet quantization.

Verifies that exported ``config.json`` declares a ``quantization_config``
block when ``use_bitnet=True``, that BitLinear weights are baked to faithful
ternary values in ``pytorch_model.bin``, and that decoder (autoregressive)
forward/backward works end-to-end with BitNet enabled.
"""
import os
import tempfile
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import json
    import torch
    from src.model.tormented_bert_frankestein import (
        FrankensteinDecoder,
        TormentedBertFrankenstein,
        UltraConfig,
    )
    from src.deploy.transformers_export import export_transformers_model


def _yaml_text(model_overrides: str) -> str:
    return (
        "model_class: frankenstein\n"
        "model:\n"
        "  vocab_size: 64\n"
        "  hidden_size: 64\n"
        "  num_layers: 2\n"
        "  num_heads: 4\n"
        "  num_experts: 4\n"
        "  top_k_experts: 2\n"
        "  layer_pattern: [standard_attn]\n"
        "  use_moe: true\n"
        "  ffn_hidden_size: 128\n"
        f"{model_overrides}\n"
        "training:\n"
        "  task: mlm\n"
    )


def _cfg(mode="encoder", bitnet_routers=False):
    return UltraConfig(
        vocab_size=64, hidden_size=64, num_layers=2, num_heads=4,
        num_experts=4, top_k_experts=2, dropout=0.0,
        use_bitnet=True, bitnet_routers=bitnet_routers,
        use_moe=True, ffn_hidden_size=128, mode=mode,
        layer_pattern=["standard_attn"],
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestBitNetHFExport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _save_ckpt(self, model, name="c.pt"):
        path = os.path.join(self.tmpdir, name)
        torch.save({"model_state_dict": model.state_dict()}, path)
        return path

    def _export(self, ckpt, yaml_text):
        ypath = os.path.join(self.tmpdir, "cfg.yaml")
        with open(ypath, "w") as f:
            f.write(yaml_text)
        out = os.path.join(self.tmpdir, "hf")
        return export_transformers_model(ckpt, ypath, out), out

    def test_quantization_config_declared(self):
        m = TormentedBertFrankenstein(_cfg())
        ckpt = self._save_ckpt(m)
        overrides = "  use_bitnet: true\n  bitnet_routers: false"
        res, out = self._export(ckpt, _yaml_text(overrides))
        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["bitnet"]["enabled"])
        self.assertGreater(res["bitnet"]["baked_layers"], 0)
        with open(os.path.join(out, "config.json")) as f:
            conf = json.load(f)
        qc = conf["quantization_config"]
        self.assertEqual(qc["quant_method"], "bitnet")
        self.assertEqual(qc["bits"], 1.58)
        self.assertTrue(qc["ternary_weights"])

    def test_exported_weights_are_ternary(self):
        m = TormentedBertFrankenstein(_cfg())
        ckpt = self._save_ckpt(m)
        overrides = "  use_bitnet: true\n  bitnet_routers: false"
        res, out = self._export(ckpt, _yaml_text(overrides))
        sd = torch.load(os.path.join(out, "pytorch_model.bin"),
                        map_location="cpu", weights_only=False)
        w = sd["layers.0.mixer.q_proj.weight"]
        gamma = w.abs().max().item()
        nz = w[w.abs() > 1e-6]
        self.assertTrue(torch.allclose(nz.abs(), torch.full_like(nz.abs(), gamma),
                                      atol=1e-5))

    def test_no_quantization_config_without_bitnet(self):
        cfg = _cfg()
        cfg.use_bitnet = False
        m = TormentedBertFrankenstein(cfg)
        ckpt = self._save_ckpt(m)
        overrides = "  use_bitnet: false\n  bitnet_routers: false"
        res, out = self._export(ckpt, _yaml_text(overrides))
        self.assertEqual(res["status"], "ok")
        self.assertFalse(res["bitnet"]["enabled"])
        with open(os.path.join(out, "config.json")) as f:
            conf = json.load(f)
        self.assertNotIn("quantization_config", conf)

    def test_decoder_bitnet_forward_backward(self):
        m = FrankensteinDecoder(_cfg(mode="decoder"))
        x = torch.randint(0, 64, (2, 16))
        y = m(x)
        self.assertEqual(y.shape, (2, 16, 64))
        loss = y.float().mean()
        loss.backward()
        # gradients must flow through ternary weights (STE). The decoder
        # wraps a backbone, so the first layer lives under m.backbone.
        first_layer = getattr(m, "backbone", m).layers[0]
        grad = first_layer.mixer.q_proj.weight.grad
        self.assertIsNotNone(grad)
        self.assertFalse(torch.allclose(grad, torch.zeros_like(grad)))


if __name__ == "__main__":
    unittest.main()