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

            model_cfg = FrankensteinModelConfig(
                vocab_size=32,
                hidden_size=16,
                num_layers=1,
                num_loops=1,
                layer_pattern=["standard_attn"],
                num_heads=4,
                retention_heads=4,
                num_experts=2,
                top_k_experts=1,
                dropout=0.1,
                use_moe=False,
                use_factorized_embedding=False,
                use_embedding_conv=False,
                norm_type="layer_norm",
                ffn_hidden_size=32,
                mode="encoder",
            )
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
                    "vocab_size": 32,
                    "hidden_size": 16,
                    "num_layers": 1,
                    "num_loops": 1,
                    "layer_pattern": ["standard_attn"],
                    "num_heads": 4,
                    "retention_heads": 4,
                    "num_experts": 2,
                    "top_k_experts": 1,
                    "dropout": 0.1,
                    "ode_solver": "rk4",
                    "ode_steps": 2,
                    "use_bitnet": True,
                    "norm_type": "layer_norm",
                    "use_factorized_embedding": False,
                    "factorized_embedding_dim": 8,
                    "use_embedding_conv": False,
                    "embedding_conv_kernel": 3,
                    "positional_encoding": "hope",
                    "hope_base": 10000.0,
                    "hope_damping": 0.01,
                    "rope_base": 10000.0,
                    "rope_scaling": 1.0,
                    "positional_encoding": "hope",
                    "use_moe": False,
                    "use_mixture_of_depths": False,
                    "mixture_of_depths_capacity_ratio": 0.5,
                    "mixture_of_depths_router_aux_loss_weight": 0.0,
                    "ffn_hidden_size": 32,
                    "ffn_activation": "silu",
                    "mode": "encoder",
                    "engram_max_ngram_size": 3,
                    "engram_n_heads_per_ngram": 2,
                    "engram_embed_dim_per_head": 4,
                    "engram_kernel_size": 4,
                    "engram_seed": 42,
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
            self.assertTrue((out_dir / "pytorch_model.bin").exists())
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
                        "model": {"layer_pattern": ["standard_attn"]},
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
                        "model": {"layer_pattern": ["standard_attn"]},
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

        cfg = FrankensteinModelConfig(
            vocab_size=32, hidden_size=16, num_layers=1, num_loops=1,
            layer_pattern=["standard_attn"], num_heads=4, retention_heads=4,
            dropout=0.0, use_bitnet=False, mode="decoder",
            positional_encoding="rope", norm_type="layer_norm",
            ffn_hidden_size=32,
        )
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
