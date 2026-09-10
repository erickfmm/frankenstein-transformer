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

from DashAI.back.core.schema_fields import (
    BaseSchema,
    bool_field,
    schema_field,
    string_field,
)
from DashAI.back.core.utils import MultilingualString


class FrankensteinPassthroughSchema(BaseSchema):
    """Shared fields for every Frankenstein DashAI model component."""

    use_dashai_dataset: schema_field(
        bool_field(),
        placeholder=True,
        description=MultilingualString(
            en=(
                "If checked (default), train on the DashAI run dataset, "
                "overriding any dataset configured in the Frankenstein JSON. "
                "If unchecked, the dataset configured inside the Frankenstein "
                "JSON is used instead (the `text_dataset` block for NLP "
                "tasks, the `vision_dataset` block for vision tasks)."
            ),
            es=(
                "Si está marcado (por defecto), entrena con el dataset del "
                "run de DashAI, sobrescribiendo cualquier dataset configurado "
                "en el JSON Frankenstein. Si se desmarca, se usa el dataset "
                "configurado dentro del JSON Frankenstein (el bloque "
                "`text_dataset` para NLP, `vision_dataset` para visión)."
            ),
        ),
        alias=MultilingualString(en="Use DashAI dataset", es="Usar dataset DashAI"),
    ) = True  # type: ignore

    use_dashai_dataset_for_tokenizer: schema_field(
        bool_field(),
        placeholder=True,
        description=MultilingualString(
            en=(
                "Only relevant when `tokenizer.source` is "
                "`train_from_dataset` in the Frankenstein JSON. If checked "
                "(default), the tokenizer is trained from the DashAI run "
                "dataset text. If unchecked, the tokenizer is trained from "
                "the corpus configured inside the Frankenstein JSON (the "
                "`text_dataset` block)."
            ),
            es=(
                "Solo relevante cuando `tokenizer.source` es "
                "`train_from_dataset` en el JSON Frankenstein. Si está "
                "marcado (por defecto), el tokenizador se entrena desde el "
                "texto del dataset del run de DashAI. Si se desmarca, se "
                "entrena desde el corpus configurado dentro del JSON "
                "Frankenstein (el bloque `text_dataset`)."
            ),
        ),
        alias=MultilingualString(
            en="Use DashAI dataset for tokenizer",
            es="Usar dataset DashAI para tokenizador",
        ),
    ) = True  # type: ignore

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