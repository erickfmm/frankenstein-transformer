"""Pydantic schemas for the DashAI plugin components (v1: passthrough JSON).

The Frankenstein JSON Schema remains the single source of truth. Each model
component exposes a small pydantic ``BaseSchema`` whose only user-facing field,
``frankenstein_json``, is a string holding a full Frankenstein training config
as a **single-line JSON** string. The adapter validates that payload against
the Frankenstein JSON Schema (and the config loader's cross-component rules)
before launching train/inference.

Build your YAML with the
`Frankenstein YAML builder <https://erickfmm.github.io/frankenstein-transformer/index.html>`_,
convert it to a one-line JSON string, and paste it into the field.
"""
from __future__ import annotations

from DashAI.back.core.schema_fields import BaseSchema, schema_field, string_field
from DashAI.back.core.utils import MultilingualString


class FrankensteinPassthroughSchema(BaseSchema):
    """Shared fields for every Frankenstein DashAI model component."""

    frankenstein_json: schema_field(
        string_field(),
        placeholder=(
            '{"model_class": "frankenstein", "model": {"dims": '
            '{"hidden_size": 128, "num_layers": 2}}}'
        ),
        description=MultilingualString(
            en=(
                "A full Frankenstein training config as a single-line JSON "
                "string. Build it with the Frankenstein YAML builder "
                "(https://erickfmm.github.io/frankenstein-transformer/index.html), "
                "convert the YAML to a one-line JSON string, and paste it here. "
                "Validated against the Frankenstein JSON Schema before launch. "
                "Convert your YAML to a one-line JSON before pasting it (the "
                "DashAI field is single-line)."
            ),
            es=(
                "Un config completo de entrenamiento Frankenstein como una "
                "cadena JSON de una sola línea. Constrúyelo con el generador de "
                "YAML de Frankenstein "
                "(https://erickfmm.github.io/frankenstein-transformer/index.html), "
                "convierte el YAML a un JSON de una sola línea y pégalo aquí. "
                "Se valida contra el esquema JSON de Frankenstein antes de "
                "lanzar. Convierte tu YAML a un JSON de una sola línea antes de "
                "pegarlo (el campo de DashAI es de una sola línea)."
            ),
        ),
        alias=MultilingualString(en="Frankenstein JSON", es="JSON Frankenstein"),
    )  # type: ignore


class FrankensteinClassifierSchema(FrankensteinPassthroughSchema):
    """Schema for text/image classification components (encoder + ViT cls)."""