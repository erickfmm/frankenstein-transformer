"""Component registry core: registration, lookup, derived views, isolation."""

from __future__ import annotations

import subprocess
import sys
import unittest
import pathlib

from src.registry import (
    ComponentFamily,
    ComponentSpec,
    MixerSpec,
    registry,
    register_spec,
)
from src.registry.errors import (
    DuplicateComponentError,
    InvalidSpecError,
    UnknownComponentError,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _spec(name="demo", **overrides):
    base = dict(
        name=name,
        builder="src.model.attention.standard:StandardAttention",
        description="Demo component.",
        description_es="Componente de demostración.",
    )
    base.update(overrides)
    return MixerSpec(**base)


class RegistrationTests(unittest.TestCase):
    """Register / duplicate / idempotency contract."""

    def tearDown(self):
        registry.unregister(ComponentFamily.MIXER, "demo")

    def test_register_and_get(self):
        spec = _spec()
        register_spec(spec)
        self.assertIs(registry.get(ComponentFamily.MIXER, "demo"), spec)

    def test_identical_reregistration_is_idempotent(self):
        register_spec(_spec())
        register_spec(_spec())  # must not raise
        self.assertEqual(registry.names(ComponentFamily.MIXER).count("demo"), 1)

    def test_conflicting_reregistration_raises(self):
        register_spec(_spec())
        with self.assertRaises(DuplicateComponentError):
            register_spec(_spec(description="Different component."))

    def test_unknown_component_lists_known_names_and_plugin_hint(self):
        from src.registry import ensure_builtins

        ensure_builtins()
        with self.assertRaises(UnknownComponentError) as ctx:
            registry.get(ComponentFamily.MIXER, "definitely_missing")
        message = str(ctx.exception)
        self.assertIn("standard", message)
        self.assertIn("plugin", message)

    def test_invalid_name_rejected(self):
        with self.assertRaises(InvalidSpecError):
            _spec(name="Bad-Name")

    def test_missing_bilingual_description_rejected(self):
        with self.assertRaises(InvalidSpecError):
            _spec(description_es="")

    def test_invalid_builder_rejected(self):
        with self.assertRaises(InvalidSpecError):
            _spec(builder=123)

    def test_non_spec_rejected(self):
        with self.assertRaises(InvalidSpecError):
            register_spec(object())  # type: ignore[arg-type]


class DerivedViewTests(unittest.TestCase):
    """Capability sets are derived from specs."""

    def test_builtin_counts(self):
        """Built-in spec tuples are plugin-independent (the global registry may
        also hold plugin components loaded by other tests)."""
        from src.registry.builtin.activation import ACTIVATION_SPECS
        from src.registry.builtin.attention import ATTENTION_SPECS
        from src.registry.builtin.embedding import EMBEDDING_SPECS
        from src.registry.builtin.norm import NORM_SPECS
        from src.registry.builtin.optimizer import OPTIMIZER_SPECS
        from src.registry.builtin.residual import RESIDUAL_SPECS

        self.assertEqual(len(ATTENTION_SPECS), 54)
        self.assertEqual(len(OPTIMIZER_SPECS), 38)
        self.assertEqual(len(ACTIVATION_SPECS), 44)
        self.assertEqual(len(NORM_SPECS), 6)
        self.assertEqual(len(RESIDUAL_SPECS), 4)
        self.assertEqual(len(EMBEDDING_SPECS), 3)

    def test_training_free_and_cache_sets(self):
        from src.registry import ensure_builtins
        from src.registry.builtin.attention import ATTENTION_SPECS

        ensure_builtins()
        self.assertEqual(
            registry.property_set(ComponentFamily.MIXER, "training_free"),
            frozenset({"fasa", "sparge"}),
        )
        self.assertIn("standard", registry.property_set(ComponentFamily.MIXER, "cache_implemented"))
        # The cache registry is derived from these specs, so every built-in
        # mixer must declare a CacheSpec (the derivation silently skips None).
        for spec in ATTENTION_SPECS:
            self.assertIsNotNone(spec.cache, f"{spec.name} declares no CacheSpec")

    def test_alias_resolution(self):
        from src.registry import ensure_builtins

        ensure_builtins()
        self.assertEqual(
            registry.resolve_name(ComponentFamily.POSITIONAL_ENCODING, "none"),
            "nope",
        )


class TorchFreeImportTests(unittest.TestCase):
    """The registry package must import without torch or model modules."""

    def test_registry_imports_without_torch(self):
        code = (
            "import sys; sys.modules['torch'] = None\n"
            "import src.registry\n"
            "assert 'src.model' not in sys.modules\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
