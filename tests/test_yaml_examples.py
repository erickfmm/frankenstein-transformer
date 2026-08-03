"""
Smoke-tests that load each YAML example from configs/examples/ and
configs/ (root-level files) through load_training_config, verifying
they are schema-valid and produce a well-formed LoadedTrainingConfig.

No torch is required: load_training_config only needs PyYAML and the
pure-Python dataclass layer.  The skipUnless guard is kept for safety
since UltraConfig lives inside the torch-dependent model package.
"""
import os
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

_IMPORTS_OK = False
if TORCH_AVAILABLE:
    try:
        from src.training.config_loader import load_training_config, LoadedTrainingConfig
        _IMPORTS_OK = True
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Locate YAML files
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
_EXAMPLES_DIR = os.path.join(_REPO_ROOT, "configs", "examples")
_CONFIGS_ROOT = os.path.join(_REPO_ROOT, "configs")


def _yaml_files_in(directory, exclude=None):
    """Return sorted list of (name, abs_path) tuples for *.yaml/*.yml files."""
    exclude = set(exclude or [])
    out = []
    if not os.path.isdir(directory):
        return out
    for fname in sorted(os.listdir(directory)):
        if fname.endswith((".yaml", ".yml")):
            name = os.path.splitext(fname)[0]
            if name not in exclude:
                out.append((name, os.path.join(directory, fname)))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@unittest.skipUnless(_IMPORTS_OK, "torch and training deps required")
class YamlExamplesLoadTests(unittest.TestCase):
    """Every file in configs/examples/ must load without raising."""

    pass  # tests are injected below


@unittest.skipUnless(_IMPORTS_OK, "torch and training deps required")
class YamlRootConfigsLoadTests(unittest.TestCase):
    """Root-level config files (configs/*.yaml) must load without raising."""

    pass  # tests are injected below


def _make_load_test(path: str):
    """Factory: return a test method that loads *path* and asserts validity."""
    def test_method(self):
        cfg = load_training_config(path)
        self.assertIsInstance(cfg, LoadedTrainingConfig)
        self.assertIn(cfg.task, {"mlm", "sbert"}, f"Unexpected task in {path}")
        # model_config is present when no base_model is given
        if cfg.base_model is None:
            self.assertIsNotNone(cfg.model_config, f"model_config missing for {path}")
    return test_method


# Inject one test method per YAML file into the appropriate test class
for _name, _path in _yaml_files_in(_EXAMPLES_DIR):
    _method = _make_load_test(_path)
    _method.__name__ = f"test_load_{_name}"
    setattr(YamlExamplesLoadTests, _method.__name__, _method)

for _name, _path in _yaml_files_in(_CONFIGS_ROOT, exclude={"schema"}):
    _method = _make_load_test(_path)
    _method.__name__ = f"test_load_{_name}"
    setattr(YamlRootConfigsLoadTests, _method.__name__, _method)


@unittest.skipUnless(_IMPORTS_OK, "torch and training deps required")
class YamlExamplesContentTests(unittest.TestCase):
    """Spot-checks on specific YAML content."""

    def _load(self, filename):
        path = os.path.join(_EXAMPLES_DIR, filename)
        return load_training_config(path)

    def test_all_gated_muon_layer_pattern_has_all_six_gated_types(self):
        cfg = self._load("es_arch_all_gated_muon.yaml")
        pattern = set(cfg.model_config.layer_pattern)
        for lt in ("gla_attn", "deltanet_attn", "gated_deltanet_attn",
                   "gated_deltanet2_attn",
                   "hgrn2_attn", "fox_attn", "gated_softmax_attn"):
            self.assertIn(lt, pattern, f"{lt} missing from all_gated pattern")

    def test_gla_deltanet_uses_lion_optimizer(self):
        cfg = self._load("es_arch_gla_deltanet_lion.yaml")
        self.assertEqual(cfg.training_config.optimizer_class, "lion")

    def test_bigbird_sparsek_uses_lamb_optimizer(self):
        cfg = self._load("es_arch_bigbird_sparsek_lamb.yaml")
        self.assertEqual(cfg.training_config.optimizer_class, "lamb")

    def test_nsa_retnet_uses_radam_optimizer(self):
        cfg = self._load("es_arch_nsa_retnet_radam.yaml")
        self.assertEqual(cfg.training_config.optimizer_class, "radam")

    def test_apollo_example_uses_apollo_optimizer(self):
        cfg = self._load("es_mlm_bertin_apollo.yaml")
        self.assertEqual(cfg.training_config.optimizer_class, "apollo")

    def test_apollo_mini_example_uses_apollo_mini_optimizer(self):
        cfg = self._load("es_mlm_mbert_apollo_mini.yaml")
        self.assertEqual(cfg.training_config.optimizer_class, "apollo_mini")

    def test_q_apollo_example_uses_q_apollo_optimizer(self):
        cfg = self._load("es_mlm_modernbert_q_apollo.yaml")
        self.assertEqual(cfg.training_config.optimizer_class, "q_apollo")

    def test_sparse_longformer_layer_pattern(self):
        cfg = self._load("es_arch_sparse_longformer_adamw.yaml")
        for lt in cfg.model_config.layer_pattern:
            self.assertIn(lt, {"sparse_transformer_attn", "longformer_attn"})

    def test_decoder_gated_mixed_mode_is_decoder(self):
        cfg = self._load("decoder_gated_mixed_adamw.yaml")
        self.assertEqual(cfg.model_config.mode, "decoder")

    def test_hgrn2_fox_gated_softmax_norm_is_dynamic_tanh(self):
        cfg = self._load("es_arch_hgrn2_fox_gated_softmax_adamw.yaml")
        self.assertEqual(cfg.model_config.norm_type, "dynamic_tanh")

    def test_gpt_like_decoder_mode_decoder(self):
        cfg = self._load("decoder_gpt_like.yaml")
        self.assertEqual(cfg.model_config.mode, "decoder")

    def test_llama_like_decoder_rope(self):
        cfg = self._load("decoder_llama_like.yaml")
        self.assertEqual(cfg.model_config.positional_encoding, "rope")

    def test_mini_cautious_adamw_model_class(self):
        cfg = self._load("es_arch_mini_cautious_adamw.yaml")
        self.assertEqual(cfg.model_class, "mini")

    def test_moe_titan_has_use_moe_true(self):
        cfg = self._load("es_arch_moe_titan_ademamix.yaml")
        self.assertTrue(cfg.model_config.use_moe)

    def test_bitnet_factorized_uses_bitnet(self):
        cfg = self._load("es_arch_bitnet_factorized_mars_adamw.yaml")
        self.assertTrue(cfg.model_config.use_bitnet)

    def test_bitnet_decoder_uses_bitnet_and_routers_false(self):
        cfg = self._load("bitnet_decoder.yaml")
        self.assertTrue(cfg.model_config.use_bitnet)
        self.assertFalse(cfg.model_config.bitnet_routers)
        self.assertFalse(cfg.model_config.use_bitnet_conv)
        self.assertEqual(cfg.model_config.mode, "decoder")
        self.assertEqual(cfg.model_class, "frankesteindecoder")


if __name__ == "__main__":
    unittest.main()
