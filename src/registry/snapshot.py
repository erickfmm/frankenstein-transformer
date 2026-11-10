"""Assembled-schema snapshot export.

The runtime schema is assembled from the hand-authored YAML plus the
component registry (``src/registry/schema_assembly.py``). Tooling consumers
that cannot run Python — the GitHub Pages mirror, CI diffs, plugin authors —
consume the fully resolved snapshot produced here.

Usage::

    python -m src.registry.snapshot --out docs/schema.assembled.yaml
"""

from __future__ import annotations

import argparse
import pathlib

import yaml

__all__ = ["export_schema"]

_DEFAULT_OUTPUT = pathlib.Path("docs/schema.assembled.yaml")


def export_schema(
    schema_path: str | pathlib.Path = "src/schema.yaml",
    out_path: str | pathlib.Path = _DEFAULT_OUTPUT,
) -> pathlib.Path:
    """Write the fully assembled schema to ``out_path``.

    Args:
        schema_path: Root schema entry point.
        out_path: Destination YAML file.

    Returns:
        The destination path.
    """
    from src.utils.schema_loader import resolve_schema

    schema = resolve_schema(schema_path)
    destination = pathlib.Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(schema, sort_keys=False, allow_unicode=True, width=1000),
        encoding="utf-8",
    )
    return destination


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default="src/schema.yaml")
    parser.add_argument("--out", default=str(_DEFAULT_OUTPUT))
    args = parser.parse_args()
    destination = export_schema(args.schema, args.out)
    print(f"wrote {destination} ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
