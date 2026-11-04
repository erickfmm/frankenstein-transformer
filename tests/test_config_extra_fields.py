"""Unknown YAML keys warn instead of aborting the load.

The root schema keeps ``additionalProperties: false`` (the in-repo presets
stay strict via ``tests/test_schema_consistency.py``), but user configs may
carry fields the schema does not document yet while it evolves. Those keys
are logged as ``WARNING`` records and ignored; the loader's existing hard
checks (missing task, bad model_class, invalid optimizer, ...) still raise.
"""
import os
import tempfile
import unittest
from importlib.util import find_spec

import yaml

TORCH_AVAILABLE = find_spec("torch") is not None

_IMPORTS_OK = False
if TORCH_AVAILABLE:
    try:
        from src.training.config_loader import load_inference_config, load_training_config
        _IMPORTS_OK = True
    except ImportError:
        pass


LOGGER_NAME = "frankenstein.config"


def _base_config():
    """Return a minimal schema-valid nested training config."""
    return {
        "model_class": "frankenstein_text_encoder",
        "model": {
            "dims": {
                "vocab_size": 100,
                "hidden_size": 64,
                "num_layers": 1,
                "num_heads": 4,
                "layer_pattern": ["standard_attn"],
                "mode": "encoder",
            },
            "bitnet": {"enabled": False},
            "moe": {"enabled": False},
            "ffn": {"hidden_size": 128, "activation": "gelu"},
        },
        "training": {
            "task": "mlm",
            "optimizer": {"optimizer_class": "adamw", "parameters": {}},
        },
    }


@unittest.skipUnless(_IMPORTS_OK, "torch and training deps required")
class ExtraFieldsWarnTests(unittest.TestCase):
    """Extra keys load with a warning; hard violations still raise."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def _write(self, config):
        path = os.path.join(self._tmpdir, "cfg.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        return path

    def _assert_warns_and_loads(self, config, needle):
        path = self._write(config)
        with self.assertLogs(LOGGER_NAME, level="WARNING") as captured:
            cfg = load_training_config(path)
        self.assertIsNotNone(cfg.model_config)
        self.assertTrue(
            any(needle in message for message in captured.output),
            captured.output,
        )
        return cfg

    def test_unknown_root_key(self):
        config = _base_config()
        config["bogus_root"] = 1
        self._assert_warns_and_loads(config, "bogus_root")

    def test_unknown_model_key(self):
        config = _base_config()
        config["model"]["bogus_model"] = 1
        self._assert_warns_and_loads(config, "bogus_model")

    def test_unknown_attention_mixer(self):
        config = _base_config()
        config["model"]["attention"] = {"bogus_mixer": {"leaf": 1}}
        self._assert_warns_and_loads(config, "bogus_mixer")

    def test_unknown_training_key(self):
        config = _base_config()
        config["training"]["bogus_training"] = 1
        self._assert_warns_and_loads(config, "bogus_training")

    def test_legacy_flat_unknown_key(self):
        config = _base_config()
        config["model"] = {
            "vocab_size": 100,
            "hidden_size": 48,
            "num_layers": 1,
            "num_loops": 1,
            "num_heads": 6,
            "retention_heads": 6,
            "num_experts": 2,
            "top_k_experts": 1,
            "dropout": 0.0,
            "norm_type": "layer_norm",
            "layer_pattern": ["standard_attn"],
            "use_bitnet": False,
            "use_moe": False,
            "ode_solver": "rk4",
            "ode_steps": 1,
            "ffn_hidden_size": 96,
            "ffn_activation": "gelu",
            "bogus_legacy": 1,
        }
        self._assert_warns_and_loads(config, "bogus_legacy")

    def test_hard_loader_error_still_raises(self):
        config = _base_config()
        config["training"]["task"] = "bogus"
        path = self._write(config)
        with self.assertRaises(ValueError):
            load_training_config(path)

    def test_unknown_inference_key_warns(self):
        with self.assertLogs(LOGGER_NAME, level="WARNING") as captured:
            cfg = load_inference_config({"kv_cache": {"enabled": True, "bogus": 1}})
        self.assertTrue(cfg.kv_cache_enabled)
        self.assertTrue(any("bogus" in message for message in captured.output))


if __name__ == "__main__":
    unittest.main()
