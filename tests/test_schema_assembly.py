"""Registry-assembled schema: every component validates and documents."""

from __future__ import annotations

import pathlib

import pytest
import yaml
from jsonschema import Draft202012Validator, validate

from src.registry import ComponentFamily, ensure_builtins, registry
from src.utils.schema_loader import resolve_schema

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCHEMA_PATH = _REPO_ROOT / "src" / "schema.yaml"


@pytest.fixture(scope="module")
def schema():
    return resolve_schema(_SCHEMA_PATH)


@pytest.fixture(scope="module")
def families():
    ensure_builtins()
    return registry


def _at(document, path):
    node = document
    for part in path.split("."):
        node = node[part]
    return node


def test_mixer_enum_matches_registry(schema, families):
    enum = _at(schema, "properties.model.properties.dims.properties.layer_pattern.items.enum")
    assert list(enum) == list(families.names(ComponentFamily.MIXER))


def test_optimizer_enum_matches_registry(schema, families):
    enum = _at(
        schema,
        "properties.training.properties.optimizer.properties.optimizer_class.enum",
    )
    assert list(enum) == list(families.names(ComponentFamily.OPTIMIZER))


def test_optimizer_prefix_reference_matches_registry(schema, families):
    reference = _at(
        schema,
        "properties.training.properties.optimizer.properties.parameters.x_parameter_reference.by_optimizer",
    )
    assert set(reference) == set(families.names(ComponentFamily.OPTIMIZER))
    for name, entry in reference.items():
        assert entry["prefix"] == name
        spec = families.get(ComponentFamily.OPTIMIZER, name)
        assert entry["specific_global_suffixes"] == sorted(spec.param_bindings)


def test_one_conditional_rule_per_optimizer(schema, families):
    rules = _at(schema, "properties.training").get("allOf") or []
    constants = [
        rule["if"]["properties"]["optimizer"]["properties"]["optimizer_class"]["const"]
        for rule in rules
    ]
    assert sorted(constants) == sorted(families.names(ComponentFamily.OPTIMIZER))


def test_simple_family_enums(schema, families):
    assert list(_at(schema, "properties.model.properties.norm.properties.type.enum")) == list(
        families.names(ComponentFamily.NORM)
    )
    assert list(_at(schema, "properties.model.properties.ffn.properties.activation.enum")) == list(
        families.names(ComponentFamily.ACTIVATION)
    )
    assert list(_at(schema, "properties.model.properties.residuals.properties.type.enum")) == list(
        families.names(ComponentFamily.RESIDUAL)
    )
    assert list(_at(schema, "properties.model.properties.embedding.properties.type.enum")) == list(
        families.names(ComponentFamily.EMBEDDING)
    )


def test_positional_encoding_enum_includes_aliases(schema, families):
    enum = set(_at(schema, "properties.model.properties.positional_encoding.enum"))
    assert set(families.names(ComponentFamily.POSITIONAL_ENCODING)) <= enum
    assert "none" in enum


def test_attention_properties_cover_every_fragment(schema, families):
    properties = _at(schema, "properties.model.properties.attention.properties")
    for spec in families.specs(ComponentFamily.MIXER):
        if spec.params_schema is None:
            continue
        assert spec.name in properties, spec.name


def test_use_pe_flag_per_mixer(schema, families):
    use_pe = _at(
        schema,
        "properties.model.properties.positional_encoding_parameters.properties.use_pe.properties",
    )
    for spec in families.specs(ComponentFamily.MIXER):
        key = f"{spec.name}_use_pe"
        assert key in use_pe
        assert use_pe[key]["default"] == bool(spec.needs_pos_encoder)


def test_plugins_key_present(schema):
    assert "plugins" in schema["properties"]
    Draft202012Validator(schema["properties"]["plugins"]).validate(["./x.py"])


def test_plugin_component_validates_and_renders(tmp_path):
    source = tmp_path / "plugin.py"
    source.write_text(
        "\n".join(
            [
                "import torch.nn as nn",
                "from src.plugins.api import register_norm",
                "",
                "class ScaledLayerNorm(nn.LayerNorm):",
                "    pass",
                "",
                "register_norm(",
                "    name='scaled_layer_norm',",
                "    builder=ScaledLayerNorm,",
                "    description='Scaled LayerNorm plugin.',",
                "    description_es='LayerNorm escalada plugin.',",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    from src.plugins.loader import load_plugins

    load_plugins([str(source)])
    assembled = resolve_schema(_SCHEMA_PATH)
    enum = _at(assembled, "properties.model.properties.norm.properties.type.enum")
    assert "scaled_layer_norm" in enum
    try:
        validate(
            {
                "dims": {
                    "vocab_size": 100,
                    "hidden_size": 64,
                    "num_layers": 1,
                    "num_loops": 1,
                    "num_heads": 4,
                    "layer_pattern": ["standard"],
                },
                "norm": {"type": "scaled_layer_norm"},
            },
            assembled["properties"]["model"],
        )
    finally:
        registry.unregister(ComponentFamily.NORM, "scaled_layer_norm")
