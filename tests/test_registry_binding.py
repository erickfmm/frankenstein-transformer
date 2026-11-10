"""Parameter binding: defaults, coercion, unknown keys, None handling."""

from __future__ import annotations

import unittest

from src.registry import ComponentFamily, MixerSpec, bind_params
from src.registry.binding import ComponentParams


_SPEC = MixerSpec(
    name="demo",
    builder="src.model.attention.standard:StandardAttention",
    description="Demo.",
    description_es="Demostración.",
    params_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rank": {"type": ["integer", "null"], "default": None},
            "scale": {"type": "number", "default": 1.0},
            "flag": {"type": "boolean", "default": False},
            "mode": {"type": "string", "default": "a"},
            "tags": {"type": "array", "default": []},
        },
    },
    defaults={"rank": 8},
)


class BindingTests(unittest.TestCase):
    """Bind and coerce component parameters."""

    def test_defaults_are_applied(self):
        params = bind_params(_SPEC, {})
        self.assertEqual(params.rank, 8)
        self.assertEqual(params.scale, 1.0)
        self.assertFalse(params.flag)
        self.assertEqual(params.mode, "a")

    def test_values_are_coerced(self):
        params = bind_params(_SPEC, {"rank": "4", "scale": "0.5", "flag": "true"})
        self.assertEqual(params.rank, 4)
        self.assertEqual(params.scale, 0.5)
        self.assertTrue(params.flag)

    def test_unknown_key_raises_in_strict_mode(self):
        with self.assertRaises(Exception):
            bind_params(_SPEC, {"bogus": 1})
        params = bind_params(_SPEC, {"bogus": 1}, strict=False)
        self.assertEqual(params.rank, 8)

    def test_none_uses_default_and_value_helper(self):
        params = bind_params(_SPEC, {"rank": None})
        self.assertEqual(params.rank, 8)
        self.assertEqual(params.value("rank", 3), 8)

    def test_mapping_protocol(self):
        params = bind_params(_SPEC, {"scale": 2.0})
        self.assertEqual(dict(params)["scale"], 2.0)
        self.assertEqual(len(params), 5)

    def test_component_params_attribute_error(self):
        params = ComponentParams("demo", {"x": 1})
        with self.assertRaises(AttributeError):
            _ = params.missing
