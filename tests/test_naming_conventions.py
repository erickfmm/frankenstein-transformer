"""Regression tests for the frankenstein nomenclature rename.

These tests guard against accidental reintroduction of the old names
(``tormented``, ``ultra``, misspelled ``frankestein``) in source code,
schema, and config files.
"""
from __future__ import annotations

import unittest
from importlib.util import find_spec
from pathlib import Path

TORCH_AVAILABLE = find_spec("torch") is not None

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class TestRenamedClasses(unittest.TestCase):
    """The new class / module names must be importable."""

    def test_model_module_imports_new_names(self):
        from src.model.config import FrankensteinModelConfig
        from src.model.hybrid_layer import HybridLayer
        from src.model.frankenstein_encoder import FrankensteinEncoder
        from src.model.frankenstein_decoder import FrankensteinDecoder
        self.assertTrue(callable(FrankensteinModelConfig))
        self.assertTrue(callable(FrankensteinEncoder))
        self.assertTrue(callable(FrankensteinDecoder))
        self.assertTrue(callable(HybridLayer))

    def test_package_reexports(self):
        from src import model as model_pkg
        self.assertIn("FrankensteinEncoder", model_pkg.__all__)
        self.assertIn("FrankensteinModelConfig", model_pkg.__all__)
        self.assertTrue(hasattr(model_pkg, "FrankensteinEncoder"))
        self.assertTrue(hasattr(model_pkg, "FrankensteinModelConfig"))

    def test_sbert_class_renamed(self):
        from src.sbert.train_sbert import FrankensteinSentenceTransformer
        self.assertTrue(callable(FrankensteinSentenceTransformer))

    def test_deploy_inference_class_renamed(self):
        from src.deploy.inference import FrankensteinInference
        self.assertTrue(callable(FrankensteinInference))

    def test_config_class_name_string(self):
        from src.model.config import FrankensteinModelConfig
        self.assertEqual(FrankensteinModelConfig.__name__, "FrankensteinModelConfig")

    def test_encoder_class_name_string(self):
        from src.model.frankenstein_encoder import FrankensteinEncoder
        self.assertEqual(FrankensteinEncoder.__name__, "FrankensteinEncoder")


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class TestOldNamesRemoved(unittest.TestCase):
    """Old identifiers must no longer be importable."""

    def test_old_module_path_gone(self):
        import importlib
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("src.model.tormented_bert_frankestein")

    def test_shim_module_removed(self):
        import importlib
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("src.model.frankenstein_model")

    def test_no_ultra_config_in_package(self):
        import src.model as model_pkg
        self.assertFalse(hasattr(model_pkg, "UltraConfig"))


class TestSchemaModelClassEnum(unittest.TestCase):
    """The schema ``model_class`` enum must use well-spelled values."""

    def setUp(self):
        from src.utils.schema_loader import resolve_schema
        schema = resolve_schema(_SRC_DIR / "schema.yaml")
        props = schema["properties"]["model_class"]
        self.enum = set(props.get("enum", []))

    def test_enum_contains_well_spelled_decoder(self):
        self.assertIn("frankensteindecoder", self.enum)

    def test_enum_does_not_contain_misspelled_decoder(self):
        self.assertNotIn("frankesteindecoder", self.enum)

    def test_enum_contains_encoder(self):
        self.assertIn("frankenstein", self.enum)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class TestConfigLoaderAcceptsNewNames(unittest.TestCase):
    """``config_loader`` validation must accept ``frankensteindecoder``."""

    def test_decoder_preset_loads(self):
        from src.training.config_loader import load_training_config
        path = _REPO_ROOT / "configs" / "frankensteindecoder.yaml"
        loaded = load_training_config(str(path))
        self.assertEqual(loaded.model_class, "frankensteindecoder")

    def test_rejects_misspelled_decoder(self):
        from src.training.config_loader import load_training_config
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(
                'model_class: frankesteindecoder\n'
                'model:\n'
                '  dims:\n'
                '    vocab_size: 100\n'
                '    hidden_size: 48\n'
                '    num_heads: 4\n'
                '    num_layers: 1\n'
                '    num_loops: 1\n'
                '    layer_pattern: [standard_attn]\n'
                'training:\n'
                '  task: mlm\n'
            )
            f.flush()
            with self.assertRaises(ValueError):
                load_training_config(f.name)


class TestExportModelType(unittest.TestCase):
    """The HF export ``model_type`` must be ``frankenstein``."""

    def test_model_type_string(self):
        import src.deploy.transformers_export as te
        self.assertIn('model_type = "frankenstein"', te._CONFIGURATION_FILE)
        self.assertNotIn('model_type = "frankestein"', te._CONFIGURATION_FILE)

    def test_exported_template_names(self):
        import src.deploy.transformers_export as te
        self.assertIn("frankenstein_config", te._CONFIGURATION_FILE)
        self.assertNotIn("ultra_config", te._CONFIGURATION_FILE)
        self.assertNotIn("UltraConfig", te._MODELING_FILE)
        self.assertIn("FrankensteinEncoder", te._MODELING_FILE)
        self.assertIn("FrankensteinModelConfig", te._MODELING_FILE)
        self.assertIn("_MODEL_KEYS", te._MODELING_FILE)
        self.assertNotIn("_ULTRA_KEYS", te._MODELING_FILE)


class TestNoOldNomenclatureInSource(unittest.TestCase):
    """Grep guard: no old names in ``src/`` Python files."""

    BANNED_TOKENS = (
        "tormented",
        "TormentedBert",
        "TORMENTED",
        "UltraConfig",
        "ultra_config",
        "_ULTRA_KEYS",
        "frankestein",
        "Frankestein",
        "tormented_bert_frankestein",
    )

    def _scan(self, directory: Path, suffix: str) -> list[str]:
        hits: list[str] = []
        for py in directory.rglob(suffix):
            if py.name == "test_naming_conventions.py":
                continue
            try:
                text = py.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for token in self.BANNED_TOKENS:
                if token in text:
                    hits.append(f"{py}: contains '{token}'")
        return hits

    def test_src_python_clean(self):
        hits = self._scan(_SRC_DIR, "*.py")
        self.assertEqual(hits, [], "Banned tokens found in src/: " + "; ".join(hits))

    def test_tests_python_clean(self):
        hits = self._scan(_REPO_ROOT / "tests", "*.py")
        self.assertEqual(hits, [], "Banned tokens found in tests/: " + "; ".join(hits))

    def test_schema_yaml_clean(self):
        hits = self._scan(_SRC_DIR / "schema", "*.yaml")
        self.assertEqual(hits, [], "Banned tokens found in schema/: " + "; ".join(hits))


class TestPackageMetadata(unittest.TestCase):
    """``pyproject.toml`` must use well-spelled package/script names."""

    def test_pyproject_names(self):
        text = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("transformer_encoder_frankenstein", text)
        self.assertIn("frankenstein-transformer", text)
        self.assertNotIn("transformer_encoder_frankestein", text)
        self.assertNotIn("frankestein-transformer", text)


class TestGGUFKeys(unittest.TestCase):
    """GGUF export must use ``frankenstein`` architecture keys."""

    def test_gguf_architecture_key(self):
        text = (_SRC_DIR / "deploy" / "bitnet_gguf_export.py").read_text(encoding="utf-8")
        self.assertIn('"frankenstein"', text)
        self.assertIn("frankenstein.hidden_size", text)
        self.assertNotIn('"frankestein"', text)
        self.assertNotIn("frankestein.hidden_size", text)


if __name__ == "__main__":
    unittest.main()
