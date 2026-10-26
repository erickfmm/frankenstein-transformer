"""Inference-config resolution tests (plan §13, T9).

Covers YAML loading, ``flatten_inference_dict``, schema-default parity,
CLI > YAML > default precedence, conditional rules (window >=
protected_window; compression implies enabled) and warning emission for
unsupported mixers.
"""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from importlib.util import find_spec

import yaml

TORCH_AVAILABLE = find_spec("torch") is not None

from src.deploy.inference_config import (
    InferenceConfig,
    apply_cli_overrides,
    inference_config_to_dict,
)
from src.training.config_loader import load_inference_config
from src.utils.config_flatten import flatten_inference_dict

#: Schema default <-> dataclass default parity (plan §11.3).
INFERENCE_DEFAULT_PAIRS = [
    (("generation", "max_new_tokens"), "max_new_tokens", 64),
    (("generation", "temperature"), "temperature", 1.0),
    (("generation", "top_k"), "top_k", 50),
    (("generation", "top_p"), "top_p", 1.0),
    (("generation", "greedy"), "greedy", False),
    (("kv_cache", "enabled"), "kv_cache_enabled", False),
    (("kv_cache", "quantization", "method"), "kv_quant_method", "none"),
    (("kv_cache", "quantization", "k_bits"), "kv_k_bits", 4),
    (("kv_cache", "quantization", "v_bits"), "kv_v_bits", 4),
    (("kv_cache", "quantization", "group_size"), "kv_group_size", 64),
    (("kv_cache", "quantization", "protected_window"), "kv_protected_window", 32),
    (("kv_cache", "compression", "method"), "kv_compress_method", "none"),
    (("kv_cache", "compression", "num_sinks"), "kv_num_sinks", 4),
    (("kv_cache", "compression", "window"), "kv_window", 512),
    (("kv_cache", "compression", "budget"), "kv_budget", None),
    (("kv_cache", "compression", "per_layer_budgets"), "kv_per_layer_budgets", []),
]


class SchemaDefaultParityTests(unittest.TestCase):
    def setUp(self):
        with open("src/schema/_inference.yaml") as handle:
            self.schema = yaml.safe_load(handle)

    def _schema_default(self, path):
        node = self.schema
        for key in path:
            node = node["properties"][key]
            if key != path[-1]:
                pass
        if "properties" in node:
            node = node["properties"]
        return node.get("default")

    def test_defaults_match_dataclass(self):
        cfg = InferenceConfig()
        for path, field_name, expected in INFERENCE_DEFAULT_PAIRS:
            node = self.schema
            for key in path:
                node = node["properties"][key]
            self.assertIn("default", node, msg=f"schema default missing at {path}")
            self.assertEqual(node["default"], expected, msg=f"schema default {path}")
            self.assertEqual(
                getattr(cfg, field_name), expected, msg=f"dataclass default {field_name}"
            )


class FlattenTests(unittest.TestCase):
    def test_flatten_nested_block(self):
        data = {
            "generation": {"max_new_tokens": 8, "greedy": True},
            "kv_cache": {
                "enabled": True,
                "quantization": {"method": "int8"},
                "compression": {"method": "window_sinks", "num_sinks": 2, "window": 64},
                "per_mixer": [{"mixer": "standard_attn", "enabled": False}],
            },
        }
        flat = flatten_inference_dict(data)
        self.assertEqual(flat["max_new_tokens"], 8)
        self.assertTrue(flat["greedy"])
        self.assertTrue(flat["kv_cache_enabled"])
        self.assertEqual(flat["kv_quant_method"], "int8")
        self.assertEqual(flat["kv_compress_method"], "window_sinks")
        self.assertEqual(flat["kv_num_sinks"], 2)
        self.assertEqual(flat["kv_window"], 64)
        self.assertEqual(flat["kv_per_mixer"], [{"mixer": "standard_attn", "enabled": False}])

    def test_flatten_absent_section(self):
        self.assertEqual(flatten_inference_dict(None), {})
        self.assertEqual(flatten_inference_dict({}), {})


class LoadAndPrecedenceTests(unittest.TestCase):
    def test_load_from_yaml_path(self):
        doc = {
            "model": {
                "dims": {"layer_pattern": ["standard_attn"], "hidden_size": 32},
                "positional_encoding": "rope",
            },
            "inference": {
                "generation": {"max_new_tokens": 12},
                "kv_cache": {"enabled": True},
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(doc, handle)
            path = handle.name
        cfg = load_inference_config(path)
        self.assertEqual(cfg.max_new_tokens, 12)
        self.assertTrue(cfg.kv_cache_enabled)

    def test_cli_overrides_yaml(self):
        cfg = InferenceConfig(max_new_tokens=12, temperature=0.5, kv_cache_enabled=True)

        class Args:
            max_new_tokens = 4
            temperature = None  # not passed -> keep YAML value
            top_k = None
            top_p = None
            greedy = None
            kv_cache = None
            kv_quant = "int8"
            kv_group_size = None
            kv_protected_window = None
            kv_compress = None
            kv_num_sinks = None
            kv_window = None

        apply_cli_overrides(cfg, Args)
        self.assertEqual(cfg.max_new_tokens, 4)
        self.assertEqual(cfg.temperature, 0.5)
        self.assertEqual(cfg.kv_quant_method, "int8")
        self.assertTrue(cfg.kv_cache_enabled)

    def test_cli_can_disable_cache(self):
        cfg = InferenceConfig(kv_cache_enabled=True)

        class Args:
            max_new_tokens = None
            temperature = None
            top_k = None
            top_p = None
            greedy = None
            kv_cache = False
            kv_quant = None
            kv_group_size = None
            kv_protected_window = None
            kv_compress = None
            kv_num_sinks = None
            kv_window = None

        apply_cli_overrides(cfg, Args)
        self.assertFalse(cfg.kv_cache_enabled)

    def test_conditional_rule_window_ge_protected(self):
        with self.assertRaises(ValueError):
            load_inference_config(
                {
                    "kv_cache": {
                        "enabled": True,
                        "quantization": {"method": "int4_kivi", "protected_window": 128},
                        "compression": {"method": "window_sinks", "window": 64},
                    }
                }
            )

    def test_conditional_rule_compression_requires_enabled(self):
        with self.assertRaises(ValueError):
            load_inference_config(
                {
                    "kv_cache": {
                        "enabled": False,
                        "compression": {"method": "window_sinks"},
                    }
                }
            )

    def test_schema_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            load_inference_config({"kv_cache": {"enabled": True, "bogus": 1}})

    def test_warning_for_unsupported_mixer(self):
        doc = {
            "model": {
                "dims": {"layer_pattern": ["ssog_attn"]},
                "positional_encoding": "rope",
            },
            "inference": {"kv_cache": {"enabled": True}},
        }
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            load_inference_config(doc)
        self.assertIn("ssog_attn", buffer.getvalue())
        self.assertIn("WARNING", buffer.getvalue())


class SerializationTests(unittest.TestCase):
    def test_roundtrip_to_nested_dict(self):
        cfg = InferenceConfig(
            max_new_tokens=9,
            kv_cache_enabled=True,
            kv_quant_method="int4_kivi",
            kv_compress_method="window_sinks",
            kv_budget=128,
            kv_per_layer_budgets=[64, 32],
            kv_per_mixer=[{"mixer": "gqa_attn", "enabled": True}],
        )
        data = inference_config_to_dict(cfg)
        self.assertEqual(data["generation"]["max_new_tokens"], 9)
        self.assertTrue(data["kv_cache"]["enabled"])
        self.assertEqual(data["kv_cache"]["quantization"]["method"], "int4_kivi")
        self.assertEqual(data["kv_cache"]["compression"]["budget"], 128)
        self.assertEqual(data["kv_cache"]["compression"]["per_layer_budgets"], [64, 32])
        # And back through the flattener.
        flat = flatten_inference_dict(data)
        self.assertEqual(InferenceConfig(**flat).kv_quant_method, "int4_kivi")


if __name__ == "__main__":
    unittest.main()
