"""Built-in normalization component specs.

One :class:`~src.registry.spec.NormSpec` per ``model.norm.type`` enum value.
The construction functions live in :mod:`src.model.norm.factory` and are
resolved lazily through the registry.
"""

from __future__ import annotations

from ..spec import NormSpec

__all__ = ["NORM_SPECS", "register"]

_NORM_PARAMS = {
    "type": "object",
    "properties": {
        "partial_ratio": {"type": "number", "default": 0.0625},
        "flashnorm_partial_ratio": {"type": "number", "default": 0.0},
        "eps": {"type": "number", "default": 1e-6},
    },
}

NORM_SPECS = (
    NormSpec(
        name="layer_norm",
        builder="src.model.norm.factory:_build_layer_norm",
        description="Standard Layer Normalization — mean-centering plus variance rescaling (paper: arxiv:1607.06450)",
        description_es="Layer Normalization estándar — centrado de media y reescalado por varianza (paper: arxiv:1607.06450)",
        references=("1607.06450",),
        params_schema=_NORM_PARAMS,
        defaults={"partial_ratio": 0.0625, "flashnorm_partial_ratio": 0.0, "eps": 1e-6},
    ),
    NormSpec(
        name="dynamic_tanh",
        builder="src.model.norm.factory:_build_dynamic_tanh",
        description="Dynamic tanh normalization (DyT) — a learned bounded map replaces explicit normalization (paper: arxiv:2503.10622)",
        description_es="Normalización tanh dinámica (DyT) — un mapa acotado aprendido reemplaza la normalización explícita (paper: arxiv:2503.10622)",
        references=("2503.10622",),
        params_schema=_NORM_PARAMS,
        defaults={"partial_ratio": 0.0625, "flashnorm_partial_ratio": 0.0, "eps": 1e-6},
    ),
    NormSpec(
        name="derf",
        builder="src.model.norm.factory:_build_derf",
        description="Dynamic erf normalization — extends DyT with an erf map plus learnable scale and shift (paper: arxiv:2512.10938)",
        description_es="Normalización erf dinámica — extiende DyT con un mapa erf más escala y desplazamiento aprendibles (paper: arxiv:2512.10938)",
        references=("2512.10938",),
        params_schema=_NORM_PARAMS,
        defaults={"partial_ratio": 0.0625, "flashnorm_partial_ratio": 0.0, "eps": 1e-6},
    ),
    NormSpec(
        name="rms_norm",
        builder="src.model.norm.factory:_build_rms_norm",
        description="Root Mean Square Layer Normalization — RMS rescaling without mean subtraction (paper: arxiv:1910.07467)",
        description_es="Root Mean Square Layer Normalization — reescalado RMS sin resta de media (paper: arxiv:1910.07467)",
        references=("1910.07467",),
        params_schema=_NORM_PARAMS,
        defaults={"partial_ratio": 0.0625, "flashnorm_partial_ratio": 0.0, "eps": 1e-6},
    ),
    NormSpec(
        name="prms_norm",
        builder="src.model.norm.factory:_build_prms_norm",
        description="Partial RMSNorm — RMS estimated from the first `norm.partial_ratio` fraction of dimensions (paper: arxiv:1910.07467, §5)",
        description_es="RMSNorm parcial — RMS estimado con la primera fracción `norm.partial_ratio` de dimensiones (paper: arxiv:1910.07467, §5)",
        references=("1910.07467",),
        params_schema=_NORM_PARAMS,
        defaults={"partial_ratio": 0.0625, "flashnorm_partial_ratio": 0.0, "eps": 1e-6},
        reads_partial_ratio=True,
    ),
    NormSpec(
        name="flash_norm",
        builder="src.model.norm.factory:_build_flash_norm",
        description="FlashNorm — weightless RMSNorm whose scale folds into the following projection (paper: arxiv:2407.09577)",
        description_es="FlashNorm — RMSNorm sin pesos cuya escala se pliega en la proyección siguiente (paper: arxiv:2407.09577)",
        references=("2407.09577",),
        params_schema=_NORM_PARAMS,
        defaults={"partial_ratio": 0.0625, "flashnorm_partial_ratio": 0.0, "eps": 1e-6},
        supports_ffn_fusion=True,
        reads_partial_ratio=True,
    ),
)


def register() -> None:
    """Register every built-in normalization spec (idempotent)."""
    from src.registry import register_spec

    for spec in NORM_SPECS:
        register_spec(spec)
