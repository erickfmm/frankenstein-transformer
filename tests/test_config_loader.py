"""Unit tests for load_training_config and list_config_paths."""
import os
import tempfile
import textwrap
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

_IMPORTS_OK = False
if TORCH_AVAILABLE:
    try:
        from src.training.config_loader import load_training_config, list_config_paths
        _IMPORTS_OK = True
    except ImportError:
        pass


def _write_yaml(tmp_dir, name, content):
    path = os.path.join(tmp_dir, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(textwrap.dedent(content))
    return path


@unittest.skipUnless(_IMPORTS_OK, "torch and training deps required")
class LoadTrainingConfigMLMTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def _cfg_path(self, content):
        return _write_yaml(self._tmpdir, "cfg.yaml", content)

    def test_minimal_mlm_config(self):
        path = self._cfg_path("""
            model:
              vocab_size: 100
              hidden_size: 48
              num_layers: 1
              num_loops: 1
              num_heads: 6
              retention_heads: 6
              num_experts: 2
              top_k_experts: 1
              dropout: 0.0
              norm_type: layer_norm
              layer_pattern: [standard_attn]
              use_bitnet: false
              use_moe: false
              ode_solver: rk4
              ode_steps: 1
              ffn_hidden_size: 96
              ffn_activation: gelu
            training:
              task: mlm
              optimizer:
                optimizer_class: adamw
                parameters: {}
        """)
        cfg = load_training_config(path)
        self.assertEqual(cfg.task, "mlm")
        self.assertIsNotNone(cfg.model_config)
        self.assertEqual(cfg.model_config.vocab_size, 100)

    def test_mlm_sets_frankenstein_model_class_by_default(self):
        path = self._cfg_path("""
            model:
              vocab_size: 100
              hidden_size: 48
              num_layers: 1
              num_loops: 1
              num_heads: 6
              retention_heads: 6
              num_experts: 2
              top_k_experts: 1
              dropout: 0.0
              norm_type: layer_norm
              layer_pattern: [standard_attn]
              use_bitnet: false
              use_moe: false
              ode_solver: rk4
              ode_steps: 1
              ffn_hidden_size: 96
              ffn_activation: gelu
            training:
              task: mlm
              optimizer:
                optimizer_class: adamw
                parameters: {}
        """)
        cfg = load_training_config(path)
        self.assertEqual(cfg.model_class, "frankenstein")

    def test_mlm_missing_optimizer_raises(self):
        path = self._cfg_path("""
            model:
              vocab_size: 100
              hidden_size: 48
              num_layers: 1
              num_loops: 1
              num_heads: 6
              retention_heads: 6
              num_experts: 2
              top_k_experts: 1
              dropout: 0.0
              norm_type: layer_norm
              layer_pattern: [standard_attn]
              use_bitnet: false
              use_moe: false
              ode_solver: rk4
              ode_steps: 1
            training:
              task: mlm
        """)
        with self.assertRaises(ValueError):
            load_training_config(path)

    def test_mlm_missing_optimizer_class_raises(self):
        path = self._cfg_path("""
            model:
              vocab_size: 100
              hidden_size: 48
              num_layers: 1
              num_loops: 1
              num_heads: 6
              retention_heads: 6
              num_experts: 2
              top_k_experts: 1
              dropout: 0.0
              norm_type: layer_norm
              layer_pattern: [standard_attn]
              use_bitnet: false
              use_moe: false
              ode_solver: rk4
              ode_steps: 1
            training:
              task: mlm
              optimizer:
                parameters: {}
        """)
        with self.assertRaises(ValueError):
            load_training_config(path)

    def test_invalid_task_raises(self):
        path = self._cfg_path("""
            model:
              vocab_size: 100
              hidden_size: 48
              num_layers: 1
              num_loops: 1
              num_heads: 6
              retention_heads: 6
              num_experts: 2
              top_k_experts: 1
              dropout: 0.0
              norm_type: layer_norm
              layer_pattern: [standard_attn]
              use_bitnet: false
              use_moe: false
              ode_solver: rk4
              ode_steps: 1
            training:
              task: autoregressive
              optimizer:
                optimizer_class: adamw
        """)
        with self.assertRaises(ValueError):
            load_training_config(path)

    def test_invalid_model_class_raises(self):
        path = self._cfg_path("""
            model_class: bigbert
            model:
              vocab_size: 100
              hidden_size: 48
              num_layers: 1
              num_loops: 1
              num_heads: 6
              retention_heads: 6
              num_experts: 2
              top_k_experts: 1
              dropout: 0.0
              norm_type: layer_norm
              layer_pattern: [standard_attn]
              use_bitnet: false
              use_moe: false
              ode_solver: rk4
              ode_steps: 1
            training:
              task: mlm
              optimizer:
                optimizer_class: adamw
        """)
        with self.assertRaises(ValueError):
            load_training_config(path)

    def test_no_model_without_base_model_raises(self):
        path = self._cfg_path("""
            training:
              task: mlm
              optimizer:
                optimizer_class: adamw
        """)
        with self.assertRaises(ValueError):
            load_training_config(path)

    def test_base_model_mlm_requires_tokenizer_name_or_path(self):
        path = self._cfg_path("""
            base_model: "bert-base-uncased"
            training:
              task: mlm
              optimizer:
                optimizer_class: adamw
                parameters: {}
        """)
        with self.assertRaises(ValueError):
            load_training_config(path)

    def test_sbert_task_no_model_required_when_base_model_set(self):
        path = self._cfg_path("""
            base_model: "sentence-transformers/paraphrase-MiniLM-L6-v2"
            training:
              task: sbert
        """)
        cfg = load_training_config(path)
        self.assertEqual(cfg.task, "sbert")
        self.assertIsNone(cfg.model_config)

    def test_tokenizer_not_dict_raises(self):
        path = self._cfg_path("""
            model:
              vocab_size: 100
              hidden_size: 48
              num_layers: 1
              num_loops: 1
              num_heads: 6
              retention_heads: 6
              num_experts: 2
              top_k_experts: 1
              dropout: 0.0
              norm_type: layer_norm
              layer_pattern: [standard_attn]
              use_bitnet: false
              use_moe: false
              ode_solver: rk4
              ode_steps: 1
            tokenizer: "some_string_tokenizer"
            training:
              task: mlm
              optimizer:
                optimizer_class: adamw
        """)
        with self.assertRaises(ValueError):
            load_training_config(path)

    def test_optimizer_parameters_accessible(self):
        path = self._cfg_path("""
            model:
              vocab_size: 100
              hidden_size: 48
              num_layers: 1
              num_loops: 1
              num_heads: 6
              retention_heads: 6
              num_experts: 2
              top_k_experts: 1
              dropout: 0.0
              norm_type: layer_norm
              layer_pattern: [standard_attn]
              use_bitnet: false
              use_moe: false
              ode_solver: rk4
              ode_steps: 1
            training:
              task: mlm
              optimizer:
                optimizer_class: lamb
                parameters:
                  lamb-lr_other: 0.001
        """)
        cfg = load_training_config(path)
        self.assertEqual(cfg.training_config.optimizer_class, "lamb")
        self.assertIn("lamb-lr_other", cfg.training_config.optimizer_parameters)

    def test_flash_norm_hierarchical_schema(self):
        # Hierarchical schema: model.norm.{type, flashnorm_partial_ratio}.
        path = self._cfg_path("""
            model:
              dims:
                vocab_size: 100
                hidden_size: 48
                num_layers: 1
                num_loops: 1
                num_heads: 6
                retention_heads: 6
                dropout: 0.0
                layer_pattern: [standard_attn]
                mode: encoder
              norm:
                type: flash_norm
                flashnorm_partial_ratio: 0.25
              use_bitnet: false
              use_moe: false
              ode_solver: rk4
              ode_steps: 1
              ffn_hidden_size: 96
              ffn_activation: gelu
            training:
              task: mlm
              optimizer:
                optimizer_class: adamw
                parameters: {}
        """)
        cfg = load_training_config(path)
        self.assertEqual(cfg.model_config.norm_type, "flash_norm")
        self.assertAlmostEqual(cfg.model_config.flashnorm_partial_ratio, 0.25)

    def test_flash_norm_flat_schema(self):
        # Flat top-level schema still works.
        path = self._cfg_path("""
            model:
              vocab_size: 100
              hidden_size: 48
              num_layers: 1
              num_loops: 1
              num_heads: 6
              retention_heads: 6
              num_experts: 2
              top_k_experts: 1
              dropout: 0.0
              norm_type: flash_norm
              flashnorm_partial_ratio: 0.5
              layer_pattern: [standard_attn]
              use_bitnet: false
              use_moe: false
              ode_solver: rk4
              ode_steps: 1
              ffn_hidden_size: 96
              ffn_activation: gelu
            training:
              task: mlm
              optimizer:
                optimizer_class: adamw
                parameters: {}
        """)
        cfg = load_training_config(path)
        self.assertEqual(cfg.model_config.norm_type, "flash_norm")
        self.assertAlmostEqual(cfg.model_config.flashnorm_partial_ratio, 0.5)


@unittest.skipUnless(_IMPORTS_OK, "torch and training deps required")
class ListConfigPathsTests(unittest.TestCase):
    def test_lists_yaml_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_yaml(tmpdir, "first.yaml", "")
            _write_yaml(tmpdir, "second.yml", "")
            open(os.path.join(tmpdir, "not_yaml.txt"), "w").close()
            result = list_config_paths(tmpdir)
            self.assertIn("first", result)
            self.assertIn("second", result)
            self.assertNotIn("not_yaml", result)

    def test_nonexistent_dir_returns_empty(self):
        result = list_config_paths("/nonexistent/path/xyz")
        self.assertEqual(result, {})

    def test_paths_are_full_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_yaml(tmpdir, "demo.yaml", "")
            result = list_config_paths(tmpdir)
            self.assertTrue(os.path.isabs(result["demo"]))

    def test_sorted_alphabetically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_yaml(tmpdir, "b.yaml", "")
            _write_yaml(tmpdir, "a.yaml", "")
            result = list_config_paths(tmpdir)
            keys = list(result.keys())
            self.assertEqual(keys, sorted(keys))


if __name__ == "__main__":
    unittest.main()
