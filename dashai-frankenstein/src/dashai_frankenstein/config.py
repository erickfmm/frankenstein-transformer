"""Pydantic schemas for the DashAI plugin components (v1: passthrough YAML).

The Frankenstein JSON Schema remains the single source of truth. Each model
component exposes a small pydantic ``BaseSchema`` whose only user-facing field,
``frankenstein_yaml``, is a string holding a full Frankenstein training YAML.
The adapter validates that payload against the Frankenstein JSON Schema (and
the config loader's cross-component rules) before launching train/inference.

Build the YAML with the
`Frankenstein YAML builder <https://erickfmm.github.io/frankenstein-transformer/index.html>`_
and paste it into the field.
"""
from __future__ import annotations

from DashAI.back.core.schema_fields import BaseSchema, schema_field, string_field
from DashAI.back.core.utils import MultilingualString


class FrankensteinPassthroughSchema(BaseSchema):
    """Shared fields for every Frankenstein DashAI model component."""

    frankenstein_yaml: schema_field(
        string_field(),
        placeholder=(
            "model_class: frankenstein\n"
            "model:\n  dims:\n    hidden_size: 128\n    num_layers: 2\n"
        ),
        description=MultilingualString(
            en=(
                "A full Frankenstein training YAML. Build it with the "
                "Frankenstein YAML builder "
                "(https://erickfmm.github.io/frankenstein-transformer/index.html) "
                "and paste it here. Validated against the Frankenstein JSON "
                "Schema before launch."
            ),
            es=(
                "Un YAML completo de entrenamiento Frankenstein. Constrúyelo "
                "con el generador de YAML de Frankenstein "
                "(https://erickfmm.github.io/frankenstein-transformer/index.html) "
                "y pégalo aquí. Se valida contra el esquema JSON de "
                "Frankenstein antes de lanzar."
            ),
        ),
        alias=MultilingualString(en="Frankenstein YAML", es="YAML Frankenstein"),
    )  # type: ignore


class FrankensteinClassifierSchema(FrankensteinPassthroughSchema):
    """Schema for text/image classification components (encoder + ViT cls)."""