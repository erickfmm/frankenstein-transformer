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
    from src.model.tormented_bert_frankestein import TormentedBertFrankenstein, UltraConfig


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class TransformersExportTests(unittest.TestCase):
    def test_export_creates_transformers_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt = tmp_path / "checkpoint.pt"
            train_yaml = tmp_path / "train.yaml"
            out_dir = tmp_path / "out"

            model_cfg = UltraConfig(
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
            model = TormentedBertFrankenstein(model_cfg)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": model_cfg,
                },
                ckpt,
            )

            yaml_payload = {
                "model_class": "frankenstein",
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
                    "use_hope": True,
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
            self.assertTrue((out_dir / "configuration_frankestein.py").exists())
            self.assertTrue((out_dir / "modeling_frankestein.py").exists())
            self.assertTrue((out_dir / "compatibility_report.json").exists())
            self.assertTrue((out_dir / "model" / "tormented_bert_frankestein.py").exists())

            config_data = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config_data["model_type"], "frankestein")
            self.assertEqual(config_data["architectures"], ["FrankesteinForMaskedLM"])

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
                        "model_class": "frankenstein",
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


if __name__ == "__main__":
    unittest.main()
