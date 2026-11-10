"""Built-in residual-connection component specs.

One :class:`~src.registry.spec.ResidualSpec` per ``model.residuals.type``
enum value; construction functions live in :mod:`src.model.residuals.factory`.
"""

from __future__ import annotations

from ..spec import ResidualSpec

__all__ = ["RESIDUAL_SPECS", "register"]

RESIDUAL_SPECS = (
    ResidualSpec(
        name="standard",
        builder="src.model.residuals.factory:_build_standard",
        description="Classical residual connection `h_{l+1} = h_l + f_l(h_l)` — stateless, no extra parameters (paper: arxiv:1512.03385)",
        description_es="Conexión residual clásica `h_{l+1} = h_l + f_l(h_l)` — sin estado ni parámetros extra (paper: arxiv:1512.03385)",
        references=("1512.03385",),
    ),
    ResidualSpec(
        name="none",
        builder="src.model.residuals.factory:_build_none",
        description="No residual connection — ablation probe only, unstable for deep stacks (paper: arxiv:2603.15031)",
        description_es="Sin conexión residual — solo sonda de ablación, inestable en pilas profundas (paper: arxiv:2603.15031)",
        references=("2603.15031",),
    ),
    ResidualSpec(
        name="full_attn",
        builder="src.model.residuals.factory:_build_full_attn",
        description="Full AttnRes — depth-wise softmax attention over every previous layer output (paper: arxiv:2603.15031)",
        description_es="AttnRes completa — atención softmax sobre la profundidad sobre todas las salidas de capa previas (paper: arxiv:2603.15031)",
        references=("2603.15031",),
    ),
    ResidualSpec(
        name="block_attn",
        builder="src.model.residuals.factory:_build_block_attn",
        description="Block AttnRes — attention over N block representations plus the running partial sum (paper: arxiv:2603.15031)",
        description_es="AttnRes por bloques — atención sobre N representaciones de bloque más la suma parcial corriente (paper: arxiv:2603.15031)",
        references=("2603.15031",),
    ),
)


def register() -> None:
    """Register every built-in residual spec (idempotent)."""
    from src.registry import register_spec

    for spec in RESIDUAL_SPECS:
        register_spec(spec)
