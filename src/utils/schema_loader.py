from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def resolve_schema(schema_path: str | Path) -> Dict[str, Any]:
    """Load a YAML schema and resolve all ``$ref`` pointers in-place.

    Walks the loaded dict recursively. When a key ``$ref`` is found at any
    level, the referenced file (optionally with a ``#`` JSON-pointer fragment)
    is loaded and its contents are merged into the parent object (replacing
    the ``$ref`` key). References are resolved relative to the directory of
    the file that contains them.

    Supports local JSON-pointer fragments of the form
    ``sub/file.yaml#/path/to/field`` — the loaded document is navigated
    along the slash-separated path and the pointed sub-node is returned.

    Args:
        schema_path: Path to the root schema YAML file.

    Returns:
        Fully resolved schema dictionary with all ``$ref`` pointers inlined.
    """
    schema_path = Path(schema_path)
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = yaml.safe_load(handle) or {}

    return _resolve_refs(schema, schema_path.parent)


def _resolve_refs(node: Any, base_dir: Path) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            ref = node["$ref"]
            file_part, _, fragment = ref.partition("#")
            ref_path = base_dir / file_part
            with ref_path.open("r", encoding="utf-8") as handle:
                resolved = yaml.safe_load(handle) or {}
            resolved = _resolve_refs(resolved, ref_path.parent)
            if fragment:
                target = _navigate_fragment(resolved, fragment)
                return target
            return resolved
        result = {}
        for key, value in node.items():
            result[key] = _resolve_refs(value, base_dir)
        return result
    elif isinstance(node, list):
        return [_resolve_refs(item, base_dir) for item in node]
    return node


def _navigate_fragment(doc: Any, fragment: str) -> Any:
    """Navigate a loaded document along a slash-separated JSON-pointer path."""
    target = doc
    for part in fragment.split("/"):
        if part == "":
            continue
        if isinstance(target, dict):
            target = target[part]
        elif isinstance(target, list):
            target = target[int(part)]
        else:
            raise ValueError(f"Cannot navigate fragment '{fragment}' in non-container node")
    return target
