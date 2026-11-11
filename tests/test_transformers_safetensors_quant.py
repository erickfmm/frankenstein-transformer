"""End-to-end tests for quantized HuggingFace safetensors exports.

For every storage format (q8_0, q4_0, i2_s) and both load runtimes
(``dequantize_on_load`` = v1, ``quantized_linear`` = v2) the exporter must
emit ``model.safetensors`` only, embed the quantization metadata in
``config.json``, reload through the generated wrapper, and reproduce the
source logits within the format's error budget.
"""
from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file

from src.deploy.safetensors_quant import PACKED_SUFFIX, SCALE_SUFFIX
from src.deploy.transformers_export import export_transformers_model
from src.model.config import FrankensteinModelConfig
from src.model.frankenstein_encoder import FrankensteinEncoder


def _config(use_bitnet: bool = False) -> FrankensteinModelConfig:
    return FrankensteinModelConfig.from_dict({"dims": {"vocab_size": 32, "hidden_size": 64, "num_layers": 2, "num_loops": 1, "layer_pattern": ['standard', 'gqa'], "num_heads": 4, "num_kv_heads": 2, "dropout": 0.0, "mode": 'encoder'}, "attention": {"retnet": {"heads": 4}}, "moe": {"num_experts": 2, "top_k": 1, "enabled": False}, "bitnet": {"enabled": use_bitnet}, "norm": {"type": 'layer_norm'}, "ffn": {"hidden_size": 128}})


def _load_export(export_dir: Path):
    """Load through the artifact's ``load_frankenstein.py`` helper."""
    export_dir = Path(export_dir)
    if str(export_dir) not in sys.path:
        sys.path.insert(0, str(export_dir))

    importlib.invalidate_caches()
    sys.modules.pop("load_frankenstein", None)
    loader = importlib.import_module("load_frankenstein")
    return loader.load_pretrained(str(export_dir))


class TestQuantizedSafetensorsExport(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = _config()
        self.model = FrankensteinEncoder(self.cfg).eval()
        self.checkpoint = self.tmp / "checkpoint.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.cfg.to_jsonable(),
            },
            self.checkpoint,
        )
        self.yaml = self.tmp / "train.yaml"
        self.yaml.write_text(
            yaml.safe_dump(
                {
                    "model_class": "frankenstein_text_encoder",
                    "model": self.cfg.to_jsonable(),
                    "training": {"task": "mlm"},
                }
            ),
            encoding="utf-8",
        )
        self.ids = torch.randint(0, 32, (1, 8))
        with torch.inference_mode():
            self.reference = self.model(self.ids)

    def _export(self, quant: str, runtime: str) -> Path:
        out = self.tmp / f"{quant}_{runtime}"
        result = export_transformers_model(
            str(self.checkpoint),
            str(self.yaml),
            str(out),
            quant=quant,
            quant_runtime=runtime,
        )
        self.assertEqual(result["status"], "ok")
        return out

    def test_safetensors_only_and_metadata(self):
        out = self._export("q8_0", "dequantize_on_load")
        self.assertTrue((out / "model.safetensors").is_file())
        self.assertFalse((out / "pytorch_model.bin").exists())
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        self.assertNotIn(
            "num_labels",
            config,
            "an explicit null num_labels makes transformers 5.x build id2label "
            "with range(None) in PretrainedConfig.__post_init__",
        )
        quant = config["frankenstein_quantization"]
        self.assertEqual(quant["format"], "q8_0")
        self.assertEqual(quant["runtime"], "dequantize_on_load")
        self.assertGreater(quant["bits_per_weight"], 8.0)
        self.assertTrue(quant["tensors"])

        tensors = load_file(str(out / "model.safetensors"))
        packed_keys = [key for key in tensors if key.endswith(PACKED_SUFFIX)]
        self.assertTrue(packed_keys)
        self.assertTrue(all(tensors[key].dtype == torch.uint8 for key in packed_keys))
        for weight_name in quant["tensors"]:
            self.assertNotIn(weight_name, tensors)
            module_name = weight_name[: -len(".weight")]
            self.assertIn(module_name + PACKED_SUFFIX, tensors)

    def test_all_formats_and_runtimes_load(self):
        tolerances = {"q8_0": 0.05, "q4_0": 0.5, "i2_s": 2.5}
        for quant in tolerances:
            for runtime in ("dequantize_on_load", "quantized_linear"):
                with self.subTest(quant=quant, runtime=runtime):
                    out = self._export(quant, runtime)
                    wrapper = _load_export(out)
                    with torch.inference_mode():
                        logits = wrapper.model(self.ids)
                    drift = (logits - self.reference).abs().max().item()
                    self.assertLessEqual(
                        drift,
                        tolerances[quant],
                        f"{quant}/{runtime} drift {drift:.4f}",
                    )

    def test_compression_order(self):
        sizes = {}
        for quant in ("none", "q8_0", "q4_0", "i2_s"):
            out = self._export(quant, "dequantize_on_load")
            sizes[quant] = (out / "model.safetensors").stat().st_size
        self.assertLess(sizes["i2_s"], sizes["q4_0"])
        self.assertLess(sizes["q4_0"], sizes["q8_0"])
        self.assertLess(sizes["q8_0"], sizes["none"])

    def test_bitnet_i2_s_export(self):
        torch.manual_seed(1)
        cfg = _config(use_bitnet=True)
        model = FrankensteinEncoder(cfg).eval()
        checkpoint = self.tmp / "bitnet.pt"
        torch.save(
            {"model_state_dict": model.state_dict(), "config": cfg.to_jsonable()},
            checkpoint,
        )
        yaml_path = self.tmp / "bitnet.yaml"
        yaml_path.write_text(
            yaml.safe_dump(
                {
                    "model_class": "frankenstein_text_encoder",
                    "model": cfg.to_jsonable(),
                    "training": {"task": "mlm"},
                }
            ),
            encoding="utf-8",
        )
        for runtime in ("dequantize_on_load", "quantized_linear"):
            out = self.tmp / f"bitnet_{runtime}"
            result = export_transformers_model(
                str(checkpoint),
                str(yaml_path),
                str(out),
                quant="i2_s",
                quant_runtime=runtime,
            )
            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["bitnet"]["enabled"])
            self.assertGreater(result["bitnet"]["baked_layers"], 0)
            config = json.loads((out / "config.json").read_text(encoding="utf-8"))
            self.assertTrue(config["frankenstein_quantization"]["bitnet_baked"])
            tensors = load_file(str(out / "model.safetensors"))
            self.assertTrue(
                any(key.endswith(SCALE_SUFFIX) for key in tensors),
                "i2_s payload must carry per-tensor scales",
            )
            wrapper = _load_export(out)
            with torch.inference_mode():
                logits = wrapper.model(self.ids)
            self.assertTrue(torch.isfinite(logits).all())

    def test_decoder_q4_export(self):
        from src.model.frankenstein_decoder import FrankensteinDecoder

        cfg = _config()
        cfg.dims.mode = "decoder"
        cfg.positional_encoding = "rope"
        cfg.moe.enabled = False
        model = FrankensteinDecoder(cfg).eval()
        checkpoint = self.tmp / "decoder.pt"
        torch.save(
            {"model_state_dict": model.state_dict(), "config": cfg.to_jsonable()},
            checkpoint,
        )
        yaml_path = self.tmp / "decoder.yaml"
        yaml_path.write_text(
            yaml.safe_dump(
                {
                    "model_class": "frankenstein_text_decoder",
                    "model": cfg.to_jsonable(),
                    "training": {"task": "causal_lm"},
                }
            ),
            encoding="utf-8",
        )
        out = self.tmp / "decoder_q4"
        result = export_transformers_model(
            str(checkpoint), str(yaml_path), str(out), quant="q4_0",
            quant_runtime="quantized_linear",
        )
        self.assertEqual(result["status"], "ok")
        wrapper = _load_export(out)
        ids = torch.randint(0, 32, (1, 6))
        with torch.inference_mode():
            reference = model(ids)
            logits = wrapper.model(ids)
        self.assertEqual(logits.shape, reference.shape)
        self.assertTrue(torch.isfinite(logits).all())

    def test_legacy_export_with_null_num_labels_still_loads(self):
        """Artifacts exported before the null-stripping guard must still load."""
        out = self._export("q8_0", "dequantize_on_load")
        config_path = out / "config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        # Older exports mirrored the whole flat model config at the top level,
        # so `num_labels: None` leaked in next to `frankenstein_config`.
        payload["num_labels"] = None
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        wrapper = _load_export(out)
        with torch.inference_mode():
            logits = wrapper.model(self.ids)
        self.assertEqual(logits.shape, self.reference.shape)
        self.assertTrue(torch.isfinite(logits).all())

    def test_invalid_quant_and_runtime_rejected(self):
        with self.assertRaises(ValueError):
            export_transformers_model(
                str(self.checkpoint), str(self.yaml), str(self.tmp / "x"), quant="q5_0"
            )
        with self.assertRaises(ValueError):
            export_transformers_model(
                str(self.checkpoint),
                str(self.yaml),
                str(self.tmp / "y"),
                quant="q8_0",
                quant_runtime="bitsandbytes",
            )


if __name__ == "__main__":
    unittest.main()
