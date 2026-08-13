"""Pydantic schemas for the DashAI plugin components (v1: passthrough YAML).

The Frankenstein JSON Schema remains the single source of truth. Each model
component exposes a small pydantic ``BaseSchema`` whose main field,
``frankenstein_yaml``, is a string holding a full Frankenstein training YAML.
The adapter validates that payload through Frankenstein's own config loader on
``train()``. Convenience fields (``preset``, ``device``, ``batch_size``,
``num_epochs``, ``learning_rate``) are merged into the YAML before validation.
"""
from __future__ import annotations

from DashAI.back.core.schema_fields import (
    BaseSchema,
    enum_field,
    float_field,
    int_field,
    none_type,
    schema_field,
    string_field,
)
from DashAI.back.core.utils import MultilingualString

from dashai_frankenstein.presets import preset_names


# Device choices: DashAI surfaces GPU/CPU; the engine facade maps to torch.
_DEVICE_CHOICES = ["GPU", "CPU"]
_DEVICE_PLACEHOLDER = "GPU"


def _preset_enum():
    names = preset_names() or ["frankenstein", "mini", "tinybert", "standard"]
    return enum_field(enum=names)


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
                "A full Frankenstein training YAML payload. The Frankenstein "
                "JSON Schema is the source of truth — this field is validated "
                "by Frankenstein's own config loader on train(). Paste a config "
                "here or pick a 'preset' below to populate it."
            ),
            es=(
                "Un payload YAML completo de entrenamiento Frankenstein. El "
                "esquema JSON de Frankenstein es la fuente de verdad — este "
                "campo se valida con el cargador de Frankenstein al entrenar."
            ),
        ),
        alias=MultilingualString(en="Frankenstein YAML", es="YAML Frankenstein"),
    )  # type: ignore

    preset: schema_field(
        _preset_enum(),
        placeholder="mini",
        description=MultilingualString(
            en=(
                "A bundled Frankenstein preset (from configs/*.yaml). Selecting "
                "one populates the Frankenstein YAML field with the preset "
                "content."
            ),
            es=(
                "Un preset Frankenstein incluido (de configs/*.yaml). Al "
                "seleccionarlo se rellena el campo YAML de Frankenstein."
            ),
        ),
        alias=MultilingualString(en="Preset", es="Preset"),
    )  # type: ignore

    device: schema_field(
        enum_field(enum=_DEVICE_CHOICES),
        placeholder=_DEVICE_PLACEHOLDER,
        description=MultilingualString(
            en="Hardware on which training runs (GPU recommended if available).",
            es="Hardware donde se ejecuta el entrenamiento (GPU recomendada).",
        ),
        alias=MultilingualString(en="Device", es="Dispositivo"),
    )  # type: ignore

    batch_size: schema_field(
        int_field(ge=1),
        placeholder=16,
        description=MultilingualString(
            en="Batch size for the in-process training loop.",
            es="Tamaño de lote para el bucle de entrenamiento en proceso.",
        ),
        alias=MultilingualString(en="Batch size", es="Tamaño de lote"),
    )  # type: ignore

    num_epochs: schema_field(
        int_field(ge=1),
        placeholder=3,
        description=MultilingualString(
            en="Number of training epochs over the DashAI dataset.",
            es="Número de épocas de entrenamiento sobre el dataset DashAI.",
        ),
        alias=MultilingualString(en="Epochs", es="Épocas"),
    )  # type: ignore

    learning_rate: schema_field(
        none_type(float_field(ge=0.0)),
        placeholder=1e-4,
        description=MultilingualString(
            en=(
                "Learning rate for the classifier head optimizer. If None, the "
                "optimizer configuration in the Frankenstein YAML is used."
            ),
            es=(
                "Tasa de aprendizaje para el optimizador de la cabeza. Si es "
                "None, se usa la configuración del YAML Frankenstein."
            ),
        ),
        alias=MultilingualString(en="Learning rate", es="Tasa de aprendizaje"),
    )  # type: ignore


class FrankensteinClassifierSchema(FrankensteinPassthroughSchema):
    """Schema for text/image classification components (encoder + ViT cls)."""
