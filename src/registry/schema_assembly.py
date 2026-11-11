"""Registry-driven JSON-Schema assembly.

The hand-authored YAML under ``src/schema/`` keeps the structural contract
(required keys, cross-field rules, documentation prose). Component
*enumerations* — mixer names, per-mixer parameter fragments, optimizer
classes and prefixes, norm/activation/residual/PE/embedding variants — are
injected here from the component registry, so adding a component never means
editing a second list by hand.

Assembly runs after ``$ref`` resolution and before any validation, so the
validated document is a closed schema (``additionalProperties: false``)
covering both built-ins and plugins.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from .fragments import resolve_params_schema
from .spec import ComponentFamily

__all__ = ["assemble_schema"]


def _at(schema: Mapping[str, Any], *path: str) -> Any:
    """Navigate a nested schema document.

    Args:
        schema: Root schema mapping.
        *path: Key path.

    Returns:
        The node at ``path``.

    Raises:
        KeyError: If the path does not resolve.
    """
    node: Any = schema
    for part in path:
        node = node[part]
    return node


def _append_missing_rows(
    description: str,
    specs: Iterable[Any],
) -> str:
    """Append documentation rows for components missing from a table.

    Args:
        description: Existing Markdown description (may contain a table).
        specs: Component specs with ``name``/``description`` fields.

    Returns:
        The description with one row appended per missing component.
    """
    missing = [
        spec for spec in specs if f"`{spec.name}`" not in description
    ]
    if not missing:
        return description
    lines = [description.rstrip(), ""]
    for spec in sorted(missing, key=lambda item: item.name):
        lines.append(f"| `{spec.name}` | {spec.description} |")
    return "\n".join(lines) + "\n"


def _inject_mixers(schema: Dict[str, Any]) -> None:
    """Inject mixer names, bilingual rows and per-mixer parameter fragments."""
    from . import ensure_builtins, registry

    ensure_builtins()
    specs = registry.specs(ComponentFamily.MIXER)
    names = [spec.name for spec in specs]
    layer_pattern = _at(
        schema,
        "properties",
        "model",
        "properties",
        "dims",
        "properties",
        "layer_pattern",
    )
    layer_pattern["items"]["enum"] = names
    for field in ("description", "description_es"):
        if field in layer_pattern:
            layer_pattern[field] = _append_missing_rows(layer_pattern[field], specs)

    attention = _at(schema, "properties", "model", "properties", "attention")
    properties: Dict[str, Any] = {}
    for spec in specs:
        fragment = resolve_params_schema(spec)
        if not fragment:
            continue
        properties[spec.name] = dict(fragment)
        # Shared fragment keys (e.g. ``attention.falcon`` serving the six
        # falcon variants) keep working: the pointer leaf differs from the
        # spec name, so expose it under that leaf too.
        ref = getattr(spec, "params_schema", None)
        if isinstance(ref, str) and "#" in ref:
            leaf = ref.rsplit("/", 1)[-1]
            if leaf and leaf != spec.name and leaf not in properties:
                properties[leaf] = dict(fragment)
    attention["properties"] = properties

    use_pe = _at(
        schema,
        "properties",
        "model",
        "properties",
        "positional_encoding_parameters",
        "properties",
        "use_pe",
        "properties",
    )
    for spec in specs:
        key = f"{spec.name}_use_pe"
        if key not in use_pe:
            use_pe[key] = {
                "type": "boolean",
                "default": bool(getattr(spec, "needs_pos_encoder", True)),
                "title": f"{spec.name} positional encoding",
                "title_es": f"Codificación posicional de {spec.name}",
                "description": (
                    f"Whether `{spec.name}` consumes the model-wide positional "
                    "encoding."
                ),
                "description_es": (
                    f"Si `{spec.name}` consume la codificación posicional global."
                ),
            }


def _optimizer_suffixes(spec: Any) -> List[str]:
    """Return the optimizer-specific suffix names of a spec."""
    return sorted(spec.param_bindings)


def _inject_optimizers(schema: Dict[str, Any]) -> None:
    """Inject optimizer names, tables, prefix reference and ``if/then`` rules."""
    from . import ensure_builtins, registry

    ensure_builtins()
    specs = registry.specs(ComponentFamily.OPTIMIZER)
    optimizer = _at(schema, "properties", "training", "properties", "optimizer")
    optimizer_class = optimizer["properties"]["optimizer_class"]
    optimizer_class["enum"] = [spec.name for spec in specs]
    for field in ("description", "description_es"):
        if field in optimizer_class:
            optimizer_class[field] = _append_missing_rows(
                optimizer_class[field], specs
            )

    parameters = optimizer["properties"]["parameters"]
    reference = parameters.setdefault("x_parameter_reference", {})
    sbert_optimizer = (
        optimizer.get("properties", {})
        .get("sbert", {})
    )
    # Sync any sibling enum that $ref-copied the canonical optimizer enum
    # before assembly (e.g. training.sbert.optimizer.optimizer_class).
    try:
        sbert_enum = _at(
            schema,
            "properties",
            "training",
            "properties",
            "sbert",
            "properties",
            "optimizer",
            "properties",
            "optimizer_class",
        )
        sbert_enum["enum"] = [spec.name for spec in specs]
    except KeyError:
        pass
    reference["by_optimizer"] = {
        spec.name: {
            "prefix": spec.name,
            "supports_shared_per_group_suffixes": bool(
                spec.supports_shared_per_group_suffixes
            ),
            "specific_global_suffixes": _optimizer_suffixes(spec),
        }
        for spec in specs
    }

    rules = []
    for spec in specs:
        rules.append(
            {
                "if": {
                    "properties": {
                        "optimizer": {
                            "properties": {
                                "optimizer_class": {"const": spec.name}
                            }
                        }
                    }
                },
                "then": {
                    "properties": {
                        "optimizer": {
                            "properties": {
                                "parameters": {
                                    "additionalProperties": False,
                                    "patternProperties": {
                                        f"^{spec.name}-": {
                                            "type": ["number", "boolean", "array", "string"]
                                        }
                                    },
                                }
                            }
                        }
                    }
                },
            }
        )
    training = _at(schema, "properties", "training")
    # The generated rules are relative to the ``training`` node; always place
    # them there. The hand-authored raw blocks are gone, so there is no root
    # fallback anymore.
    training["allOf"] = rules


def _inject_simple_enum(
    schema: Dict[str, Any],
    path: tuple,
    family: ComponentFamily,
    *,
    include_aliases: bool = False,
    append_tables: bool = True,
) -> None:
    """Inject a registry enum into a schema node.

    Args:
        schema: Root schema mapping.
        path: Key path to the node holding ``enum``/``description``.
        family: Component family to read names from.
        include_aliases: Include registry aliases in the enum.
        append_tables: Append rows for missing documented components.
    """
    from . import ensure_builtins, registry

    ensure_builtins()
    node = _at(schema, *path)
    names = list(registry.names(family))
    if include_aliases:
        names += sorted(registry.alias_index(family))
    node["enum"] = names
    if append_tables:
        specs = registry.specs(family)
        for field in ("description", "description_es"):
            if field in node:
                node[field] = _append_missing_rows(node[field], specs)


def _inject_embeddings(schema: Dict[str, Any]) -> None:
    """Inject the token-embedding type enum (new ``embedding.type`` key)."""
    from . import ensure_builtins, registry

    ensure_builtins()
    embedding = _at(schema, "properties", "model", "properties", "embedding")
    embedding.setdefault("properties", {})["type"] = {
        "type": "string",
        "title": "Token Embedding Type",
        "title_es": "Tipo de Embedding de Token",
        "description": (
            "Token-embedding variant. `standard` uses the full "
            "`vocab_size × hidden_size` lookup table; `factorized` splits it "
            "into two low-rank matrices (`embedding.factorized.*`); "
            "`dynamic` replaces it with a trainable-free hashed n-gram table "
            "(`embedding.dynamic.*`). The legacy `*.enabled` flags are folded "
            "into this value at load time when `type` is left at its default."
        ),
        "description_es": (
            "Variante de embedding de token. `standard` usa la tabla completa "
            "`vocab_size × hidden_size`; `factorized` la divide en dos matrices "
            "de bajo rango (`embedding.factorized.*`); `dynamic` la reemplaza "
            "por una tabla de n-gramas hasheados sin entrenamiento "
            "(`embedding.dynamic.*`). Las banderas `*.enabled` heredadas se "
            "pliegan en este valor al cargar cuando `type` queda en su valor "
            "por defecto."
        ),
        "enum": list(registry.names(ComponentFamily.EMBEDDING)),
        "default": "standard",
        "examples": ["standard"],
    }


def _inject_plugins_key(schema: Dict[str, Any]) -> None:
    """Inject the top-level ``plugins:`` key."""
    if "plugins" in schema.get("properties", {}):
        return
    schema.setdefault("properties", {})["plugins"] = {
        "type": "array",
        "items": {"type": "string"},
        "title": "Plugins",
        "title_es": "Plugins",
        "description": (
            "Local plugin modules to import before schema validation. Each "
            "entry is an importable module path or a `module:object` "
            "reference whose module registers component specs through "
            "`src.plugins.api`. Installed plugins are discovered "
            "automatically through the `frankenstein.plugins` entry-point "
            "group."
        ),
        "description_es": (
            "Módulos de plugin locales a importar antes de la validación del "
            "esquema. Cada entrada es una ruta de módulo importable o una "
            "referencia `modulo:objeto` cuyo módulo registra componentes vía "
            "`src.plugins.api`. Los plugins instalados se descubren "
            "automáticamente mediante el grupo de entry-points "
            "`frankenstein.plugins`."
        ),
        "default": [],
        "examples": [["./plugins/my_mixer.py"]],
    }


def assemble_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inject registry-derived enumerations and fragments into a schema.

    Args:
        schema: The fully ``$ref``-resolved schema document (mutated in place).

    Returns:
        The same mapping, ready for validation.
    """
    _inject_mixers(schema)
    _inject_optimizers(schema)
    _inject_simple_enum(
        schema,
        ("properties", "model", "properties", "norm", "properties", "type"),
        ComponentFamily.NORM,
    )
    _inject_simple_enum(
        schema,
        ("properties", "model", "properties", "ffn", "properties", "activation"),
        ComponentFamily.ACTIVATION,
    )
    _inject_simple_enum(
        schema,
        ("properties", "model", "properties", "residuals", "properties", "type"),
        ComponentFamily.RESIDUAL,
    )
    _inject_simple_enum(
        schema,
        ("properties", "model", "properties", "positional_encoding"),
        ComponentFamily.POSITIONAL_ENCODING,
        include_aliases=True,
    )
    _inject_embeddings(schema)
    _inject_plugins_key(schema)
    return schema
