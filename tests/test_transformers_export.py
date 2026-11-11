from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from src.deploy.transformers_export import (
        check_yaml_export_compatibility,
        export_transformers_model,
    )
    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_encoder import FrankensteinEncoder


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class TransformersExportTests(unittest.TestCase):
    def test_export_creates_transformers_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt = tmp_path / "checkpoint.pt"
            train_yaml = tmp_path / "train.yaml"
            out_dir = tmp_path / "out"

            model_cfg = FrankensteinModelConfig.from_dict({"dims": {"vocab_size": 32, "hidden_size": 16, "num_layers": 1, "num_loops": 1, "layer_pattern": ['standard'], "num_heads": 4, "dropout": 0.1, "mode": 'encoder'}, "attention": {"retnet": {"heads": 4}}, "moe": {"num_experts": 2, "top_k": 1, "enabled": False}, "embedding": {"factorized": {"enabled": False}, "conv": {"enabled": False}}, "norm": {"type": 'layer_norm'}, "ffn": {"hidden_size": 32}})
            model = FrankensteinEncoder(model_cfg)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": model_cfg,
                },
                ckpt,
            )

            yaml_payload = {
                "model_class": "frankenstein_text_encoder",
                "model": {
                    "dims": {
                        "vocab_size": 32,
                        "hidden_size": 16,
                        "num_layers": 1,
                        "num_loops": 1,
                        "layer_pattern": ["standard"],
                        "num_heads": 4,
                        "dropout": 0.1,
                        "mode": "encoder",
                    },
                    "attention": {
                        "retnet": {"heads": 4},
                        "ode": {"solver": "rk4", "steps": 2},
                        "engram": {
                            "max_ngram_size": 3,
                            "n_heads_per_ngram": 2,
                            "embed_dim_per_head": 4,
                            "kernel_size": 4,
                            "seed": 42,
                        },
                    },
                    "norm": {"type": "layer_norm"},
                    "bitnet": {"enabled": True},
                    "moe": {"enabled": False, "num_experts": 2, "top_k": 1},
                    "mixture_of_depths": {
                        "enabled": False,
                        "capacity_ratio": 0.5,
                        "router_aux_loss_weight": 0.0,
                    },
                    "embedding": {
                        "factorized": {"enabled": False, "dim": 8},
                        "conv": {"enabled": False, "kernel": 3},
                    },
                    "positional_encoding": "hope",
                    "positional_encoding_parameters": {
                        "hope": {"base": 10000.0, "damping": 0.01},
                        "rope": {"base": 10000.0, "scaling": 1.0},
                    },
                    "ffn": {"hidden_size": 32, "activation": "silu"},
                },
                "training": {
                    "task": "mlm",
                    "optimizer": {
                        "optimizer_class": "adamw",
                        "parameters": {},
                    },
                },
            }
            train_yaml.write_text(yaml.safe_dump(yaml_payload), encoding="utf-8")

            result = export_transformers_model(
                model_path=str(ckpt),
                yaml_path=str(train_yaml),
                output_dir=str(out_dir),
            )

            self.assertEqual(result["status"], "ok")
            self.assertTrue((out_dir / "config.json").exists())
            self.assertTrue((out_dir / "model.safetensors").exists())
            self.assertFalse((out_dir / "pytorch_model.bin").exists())
            self.assertTrue((out_dir / "quantization_frankenstein.py").exists())

            # transformers >= 5 validates ``rope_scaling`` as a mapping; the
            # Frankenstein config stores a float factor, so it must stay
            # inside ``frankenstein_config`` and never reach PretrainedConfig.
            exported = json.loads((out_dir / "config.json").read_text())
            self.assertNotIn("rope_scaling", exported)
            self.assertNotIn("rope_parameters", exported)
            self.assertEqual(
                exported["frankenstein_config"]["positional_encoding_parameters"]["rope"]["scaling"],
                1.0,
            )
            self.assertTrue((out_dir / "configuration_frankenstein.py").exists())
            self.assertTrue((out_dir / "modeling_frankenstein.py").exists())
            self.assertTrue((out_dir / "compatibility_report.json").exists())
            self.assertTrue((out_dir / "model" / "frankenstein_encoder.py").exists())

            config_data = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config_data["model_type"], "frankenstein")
            self.assertEqual(config_data["architectures"], ["FrankensteinForMaskedLM"])

    def test_sbert_config_is_reported_as_incompatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt = tmp_path / "checkpoint.pt"
            train_yaml = tmp_path / "train.yaml"
            out_dir = tmp_path / "out"

            torch.save({"model_state_dict": {}}, ckpt)
            train_yaml.write_text(
                yaml.safe_dump(
                    {
                        "model_class": "frankenstein_text_encoder",
                        "model": {"layer_pattern": ["standard"]},
                        "training": {"task": "sbert"},
                    }
                ),
                encoding="utf-8",
            )

            result = export_transformers_model(
                model_path=str(ckpt),
                yaml_path=str(train_yaml),
                output_dir=str(out_dir),
            )

            self.assertEqual(result["status"], "incompatible")
            self.assertTrue((out_dir / "compatibility_report.json").exists())

    def test_yaml_precheck_reports_base_model_as_incompatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_yaml = tmp_path / "train.yaml"
            train_yaml.write_text(
                yaml.safe_dump(
                    {
                        "base_model": "answerdotai/ModernBERT-base",
                        "training": {"task": "mlm"},
                        "model": {"layer_pattern": ["standard"]},
                    }
                ),
                encoding="utf-8",
            )
            compatibility = check_yaml_export_compatibility(str(train_yaml))
            self.assertFalse(compatibility["is_compatible"])
            self.assertTrue(any("base_model" in issue for issue in compatibility["issues"]))

    def test_yaml_precheck_reports_missing_model_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_yaml = tmp_path / "train.yaml"
            train_yaml.write_text(
                yaml.safe_dump({"training": {"task": "mlm"}}),
                encoding="utf-8",
            )
            compatibility = check_yaml_export_compatibility(str(train_yaml))
            self.assertFalse(compatibility["is_compatible"])
            self.assertTrue(any("model" in issue for issue in compatibility["issues"]))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class TransformersExportCacheTests(unittest.TestCase):
    """T11: the exported causal-LM wrapper supports past_key_values."""

    def test_modeling_source_has_real_cache_plumbing(self):
        from src.deploy.transformers_export import _MODELING_FILE

        self.assertIn("CausalLMOutputWithPast", _MODELING_FILE)
        self.assertIn("past_key_values", _MODELING_FILE)
        self.assertIn("use_cache", _MODELING_FILE)
        self.assertIn("cache_position", _MODELING_FILE)
        self.assertIn("prepare_inputs_for_generation", _MODELING_FILE)
        # The decoder call must request the cache tuple, not swallow kwargs.
        self.assertIn("return_cache=True", _MODELING_FILE)

    def test_local_equivalent_matches_cached_decode(self):
        """The exported wrapper's call shape == local cached decode."""
        import torch

        from src.model.cache import FrankensteinCache
        from src.model.config import FrankensteinModelConfig
        from src.model.frankenstein_decoder import FrankensteinDecoder

        cfg = FrankensteinModelConfig.from_dict({"dims": {"vocab_size": 32, "hidden_size": 16, "num_layers": 1, "num_loops": 1, "layer_pattern": ['standard'], "num_heads": 4, "dropout": 0.0, "mode": 'decoder'}, "attention": {"retnet": {"heads": 4}}, "bitnet": {"enabled": False}, "positional_encoding": 'rope', "norm": {"type": 'layer_norm'}, "ffn": {"hidden_size": 32}})
        torch.manual_seed(0)
        model = FrankensteinDecoder(cfg).eval()
        prompt = torch.randint(0, 32, (1, 5))
        with torch.inference_mode():
            full = model(prompt)
            cache = FrankensteinCache.new(model)
            logits, cache = model(
                prompt, past_key_values=cache, use_cache=True, return_cache=True
            )
            self.assertIsNotNone(cache)
            next_logits, cache = model(
                torch.tensor([[7]]),
                past_key_values=cache,
                use_cache=True,
                cache_position=cache.next_position(),
                return_cache=True,
            )
        torch.testing.assert_close(logits, full, rtol=1e-5, atol=1e-6)
        self.assertEqual(next_logits.shape, (1, 1, 32))


if __name__ == "__main__":
    unittest.main()
