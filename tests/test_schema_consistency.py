"""Schema consistency and parity tests.

These tests lock in the invariants introduced by the schema audit:

* every preset in ``configs/`` and ``configs/examples/`` validates against the
  resolved JSON Schema;
* the ``layer_pattern`` enum matches its bilingual documentation tables;
* optimizer parameter prefixes have exactly one ``if/then`` block;
* the SBERT optimizer enum resolves to the canonical one (no duplication);
* schema defaults mirror ``FrankensteinModelConfig`` defaults for the fields
  where both are declared;
* optional rank fields accept an explicit ``null``;
* legacy flat feature keys and the deprecated titan PE subtree are gone.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import yaml
from jsonschema import Draft202012Validator

from src.model.config import FrankensteinModelConfig
from src.utils.schema_loader import resolve_schema

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCHEMA_ROOT = _REPO_ROOT / "src" / "schema"
_CONFIG_FILES = sorted((_REPO_ROOT / "configs").glob("*.yaml")) + sorted(
    (_REPO_ROOT / "configs" / "examples").glob("*.yaml")
)


@pytest.fixture(scope="module")
def schema():
    return resolve_schema(str(_REPO_ROOT / "src" / "schema.yaml"))


def _at(document, path):
    node = document
    for part in path.split("."):
        node = node[part]
    return node


def test_all_presets_validate(schema):
    validator = Draft202012Validator(schema)
    failures = []
    for path in _CONFIG_FILES:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        errors = sorted(validator.iter_errors(document), key=lambda e: list(e.path))
        if errors:
            failures.append((path.name, errors[0].message))
    assert not failures, failures


def test_schema_examples_validate(schema):
    examples = yaml.safe_load((_SCHEMA_ROOT / "_examples.yaml").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    for index, example in enumerate(examples):
        errors = list(validator.iter_errors(example))
        assert not errors, f"examples[{index}]: {errors[0].message}"


def test_layer_pattern_enum_matches_documentation():
    dims = yaml.safe_load((_SCHEMA_ROOT / "_model" / "_dims.yaml").read_text(encoding="utf-8"))
    layer_pattern = dims["properties"]["layer_pattern"]
    enum = set(layer_pattern["items"]["enum"])
    for language in ("description", "description_es"):
        table = layer_pattern[language]
        missing = sorted(value for value in enum if f"`{value}`" not in table)
        assert not missing, f"{language} table is missing {missing}"


def test_optimizer_prefixes_match_conditional_rules():
    optimizer = yaml.safe_load((_SCHEMA_ROOT / "_optimizer.yaml").read_text(encoding="utf-8"))
    prefixes = {
        entry["prefix"]
        for entry in optimizer["properties"]["parameters"]["x_parameter_reference"]["by_optimizer"].values()
    }
    raw_training = (_SCHEMA_ROOT / "_training.yaml").read_text(encoding="utf-8")
    enforced = {
        f"{match}-"
        for match in re.findall(r"^\s+\^([a-z0-9_]+)-:\s*\{", raw_training, flags=re.MULTILINE)
    }
    assert prefixes == enforced, {
        "missing_rule": sorted(prefixes - enforced),
        "stale_rule": sorted(enforced - prefixes),
    }


def test_sbert_optimizer_enum_resolves_to_canonical(schema):
    canonical = _at(schema, "properties.training.properties.optimizer.properties.optimizer_class.enum")
    sbert = _at(schema, "properties.training.properties.sbert.properties.optimizer.properties.optimizer_class.enum")
    assert sbert == canonical


_DEFAULT_PAIRS = {
    "properties.model.properties.dims.properties.vocab_size": "vocab_size",
    "properties.model.properties.dims.properties.hidden_size": "hidden_size",
    "properties.model.properties.dims.properties.num_layers": "num_layers",
    "properties.model.properties.dims.properties.num_loops": "num_loops",
    "properties.model.properties.dims.properties.num_heads": "num_heads",
    "properties.model.properties.dims.properties.num_kv_heads": "num_kv_heads",
    "properties.model.properties.dims.properties.dropout": "dropout",
    "properties.model.properties.dims.properties.mode": "mode",
    "properties.model.properties.norm.properties.type": "norm_type",
    "properties.model.properties.norm.properties.partial_ratio": "prms_partial_ratio",
    "properties.model.properties.norm.properties.flashnorm_partial_ratio": "flashnorm_partial_ratio",
    "properties.model.properties.embedding.properties.factorized.properties.enabled": "use_factorized_embedding",
    "properties.model.properties.embedding.properties.factorized.properties.dim": "factorized_embedding_dim",
    "properties.model.properties.embedding.properties.conv.properties.enabled": "use_embedding_conv",
    "properties.model.properties.embedding.properties.conv.properties.kernel": "embedding_conv_kernel",
    "properties.model.properties.moe.properties.enabled": "use_moe",
    "properties.model.properties.moe.properties.num_experts": "num_experts",
    "properties.model.properties.moe.properties.top_k": "top_k_experts",
    "properties.model.properties.bitnet.properties.enabled": "use_bitnet",
    "properties.model.properties.bitnet.properties.routers": "bitnet_routers",
    "properties.model.properties.bitnet.properties.conv": "use_bitnet_conv",
    "properties.model.properties.mixture_of_depths.properties.enabled": "use_mixture_of_depths",
    "properties.model.properties.mixture_of_depths.properties.capacity_ratio": "mixture_of_depths_capacity_ratio",
    "properties.model.properties.mixture_of_depths.properties.router_aux_loss_weight": "mixture_of_depths_router_aux_loss_weight",
    "properties.model.properties.ssmax.properties.enabled": "use_ssmax",
    "properties.model.properties.ssmax.properties.s_init": "ssmax_s_init",
    "properties.model.properties.numerics.properties.fixed_softmax": "fixed_softmax",
    "properties.model.properties.numerics.properties.fixed_cross_entropy": "fixed_cross_entropy",
    "properties.model.properties.ffn.properties.activation": "ffn_activation",
    "properties.model.properties.classification_head.properties.enabled": "classification_head",
    "properties.model.properties.classification_head.properties.pooling_mode": "encoder_pooling_mode",
    "properties.model.properties.attention.properties.ode.properties.solver": "ode_solver",
    "properties.model.properties.attention.properties.ode.properties.steps": "ode_steps",
    "properties.model.properties.attention.properties.retnet.properties.heads": "retention_heads",
    "properties.model.properties.attention.properties.swa.properties.window_size": "swa_window_size",
    "properties.model.properties.attention.properties.swa.properties.num_sinks": "swa_num_sinks",
    "properties.model.properties.attention.properties.conformer.properties.kernel_size": "conformer_kernel_size",
    "properties.model.properties.attention.properties.conformer.properties.macaron": "conformer_macaron",
    "properties.model.properties.attention.properties.conformer.properties.conv_norm": "conformer_conv_norm",
}


def test_schema_defaults_match_dataclass(schema):
    config = FrankensteinModelConfig()
    mismatches = []
    for schema_path, attribute in _DEFAULT_PAIRS.items():
        field_schema = _at(schema, schema_path)
        assert "default" in field_schema, f"{schema_path} has no declared default"
        expected = getattr(config, attribute)
        if field_schema["default"] != expected:
            mismatches.append((schema_path, field_schema["default"], expected))
    assert not mismatches, mismatches


def test_optional_ranks_accept_explicit_null(schema):
    validator = Draft202012Validator(_at(schema, "properties.model"))
    model = {
        "dims": {
            "vocab_size": 100,
            "hidden_size": 64,
            "num_layers": 1,
            "num_loops": 1,
            "num_heads": 4,
            "layer_pattern": ["standard_attn"],
        },
        "attention": {
            "mla": {"latent_rank": None},
            "gqla": {"latent_rank": None, "num_groups": None},
            "mlra": {"latent_rank": None},
            "tucker": {"query_rank": None, "key_rank": None, "value_rank": None},
            "iha": {"num_pseudo_heads": None},
            "gta": {"num_shared_groups": None, "value_latent_rank": None},
            "mtla": {"latent_rank": None, "stride": None},
            "cca": {"latent_rank": None},
            "ccgqa": {"query_latent_rank": None, "kv_latent_rank": None, "num_kv_heads": None},
            "gma": {"routing_dim": None},
        },
        "positional_encoding_parameters": {"alibi": {"num_heads": None}},
        "classification_head": {"num_labels": None},
    }
    errors = list(validator.iter_errors(model))
    assert not errors, [error.message for error in errors]


def test_legacy_flat_feature_keys_are_gone():
    model = yaml.safe_load((_SCHEMA_ROOT / "_model.yaml").read_text(encoding="utf-8"))
    properties = set(model["properties"])
    forbidden = {
        "use_moe",
        "num_experts",
        "top_k_experts",
        "use_bitnet",
        "bitnet_routers",
        "use_bitnet_conv",
        "use_mixture_of_depths",
        "mixture_of_depths_capacity_ratio",
        "mixture_of_depths_router_aux_loss_weight",
        "use_ssmax",
        "ssmax_s_init",
        "ffn_hidden_size",
        "ffn_activation",
        "ffn_activation_config",
        "num_labels",
        "encoder_pooling_mode",
        "ode_solver",
        "ode_steps",
        "retention_heads",
        "use_hope",
    }
    assert not (properties & forbidden), sorted(properties & forbidden)


def test_titan_pe_subtree_and_placeholders_are_removed():
    model = yaml.safe_load((_SCHEMA_ROOT / "_model.yaml").read_text(encoding="utf-8"))
    assert "titan" not in model["properties"]["attention"]["properties"]

    training = yaml.safe_load((_SCHEMA_ROOT / "_training.yaml").read_text(encoding="utf-8"))
    assert "patch_prediction" not in training["properties"]

    pe = yaml.safe_load((_SCHEMA_ROOT / "_model" / "_positional_encoding.yaml").read_text(encoding="utf-8"))
    alibi = pe["properties"]["positional_encoding_parameters"]["properties"]["alibi"]
    assert alibi["properties"], "alibi placeholder must expose at least num_heads"

    use_pe = pe["properties"]["positional_encoding_parameters"]["properties"]["use_pe"]["properties"]
    assert "ssog_attn" in use_pe
