"""Built-in activation component specs.

One :class:`~src.registry.spec.ActivationSpec` per ``model.ffn.activation``
enum value. Elementwise builders live in
:mod:`src.model.activation_function.builders`; GLU variants are built by
``GatedFFN`` and flagged with ``is_glu=True``.
"""

from __future__ import annotations

from ..spec import ActivationSpec

__all__ = ["ACTIVATION_SPECS", "register"]

ACTIVATION_SPECS = (
    ActivationSpec(
        name="abs_relu",
        builder="src.model.activation_function.builders:build_abs_relu",
        description="Elementwise FFN activation 'abs_relu' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'abs_relu' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="arctan",
        builder="src.model.activation_function.builders:build_arctan",
        description="Elementwise FFN activation 'arctan' (common family).",
        description_es="Activación elementwise para FFN 'arctan' (familia common).",
    ),
    ActivationSpec(
        name="brelu",
        builder="src.model.activation_function.builders:build_brelu",
        description="Elementwise FFN activation 'brelu' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'brelu' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="celu",
        builder="src.model.activation_function.builders:build_celu",
        description="Elementwise FFN activation 'celu' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'celu' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="dis_relu",
        builder="src.model.activation_function.builders:build_dis_relu",
        description="Elementwise FFN activation 'dis_relu' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'dis_relu' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="eelu",
        builder="src.model.activation_function.builders:build_eelu",
        description="Elementwise FFN activation 'eelu' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'eelu' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="elish",
        builder="src.model.activation_function.builders:build_elish",
        description="Elementwise FFN activation 'elish' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'elish' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="elliott",
        builder="src.model.activation_function.builders:build_elliott",
        description="Elementwise FFN activation 'elliott' (common family).",
        description_es="Activación elementwise para FFN 'elliott' (familia common).",
    ),
    ActivationSpec(
        name="elu",
        builder="src.model.activation_function.builders:build_elu",
        description="Elementwise FFN activation 'elu' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'elu' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="felu",
        builder="src.model.activation_function.builders:build_felu",
        description="Elementwise FFN activation 'felu' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'felu' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="gelu",
        builder="src.model.activation_function.builders:build_gelu",
        description="Elementwise FFN activation 'gelu' (common family).",
        description_es="Activación elementwise para FFN 'gelu' (familia common).",
    ),
    ActivationSpec(
        name="gelu_tanh",
        builder="src.model.activation_function.builders:build_gelu_tanh",
        description="Elementwise FFN activation 'gelu_tanh' (common family).",
        description_es="Activación elementwise para FFN 'gelu_tanh' (familia common).",
    ),
    ActivationSpec(
        name="hardelish",
        builder="src.model.activation_function.builders:build_hardelish",
        description="Elementwise FFN activation 'hardelish' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'hardelish' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="hardswish",
        builder="src.model.activation_function.builders:build_hardswish",
        description="Elementwise FFN activation 'hardswish' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'hardswish' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="hexpo",
        builder="src.model.activation_function.builders:build_hexpo",
        description="Elementwise FFN activation 'hexpo' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'hexpo' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="identity",
        builder="src.model.activation_function.builders:build_identity",
        description="Elementwise FFN activation 'identity' (common family).",
        description_es="Activación elementwise para FFN 'identity' (familia common).",
    ),
    ActivationSpec(
        name="leaky_relu",
        builder="src.model.activation_function.builders:build_leaky_relu",
        description="Elementwise FFN activation 'leaky_relu' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'leaky_relu' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="lisht",
        builder="src.model.activation_function.builders:build_lisht",
        description="Elementwise FFN activation 'lisht' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'lisht' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="maxout",
        builder="src.model.activation_function.builders:build_maxout",
        description="Elementwise FFN activation 'maxout' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'maxout' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="mish",
        builder="src.model.activation_function.builders:build_mish",
        description="Elementwise FFN activation 'mish' (common family).",
        description_es="Activación elementwise para FFN 'mish' (familia common).",
    ),
    ActivationSpec(
        name="mpelu",
        builder="src.model.activation_function.builders:build_mpelu",
        description="Elementwise FFN activation 'mpelu' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'mpelu' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="nl_relu",
        builder="src.model.activation_function.builders:build_nl_relu",
        description="Elementwise FFN activation 'nl_relu' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'nl_relu' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="pdelu",
        builder="src.model.activation_function.builders:build_pdelu",
        description="Elementwise FFN activation 'pdelu' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'pdelu' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="pelu",
        builder="src.model.activation_function.builders:build_pelu",
        description="Elementwise FFN activation 'pelu' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'pelu' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="prelu",
        builder="src.model.activation_function.builders:build_prelu",
        description="Elementwise FFN activation 'prelu' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'prelu' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="preu",
        builder="src.model.activation_function.builders:build_preu",
        description="Elementwise FFN activation 'preu' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'preu' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="ptanh",
        builder="src.model.activation_function.builders:build_ptanh",
        description="Elementwise FFN activation 'ptanh' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'ptanh' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="raf",
        builder="src.model.activation_function.builders:build_raf",
        description="Elementwise FFN activation 'raf' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'raf' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="relu",
        builder="src.model.activation_function.builders:build_relu",
        description="Elementwise FFN activation 'relu' (common family).",
        description_es="Activación elementwise para FFN 'relu' (familia common).",
    ),
    ActivationSpec(
        name="relu6",
        builder="src.model.activation_function.builders:build_relu6",
        description="Elementwise FFN activation 'relu6' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'relu6' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="selu",
        builder="src.model.activation_function.builders:build_selu",
        description="Elementwise FFN activation 'selu' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'selu' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="sigmoid",
        builder="src.model.activation_function.builders:build_sigmoid",
        description="Elementwise FFN activation 'sigmoid' (common family).",
        description_es="Activación elementwise para FFN 'sigmoid' (familia common).",
    ),
    ActivationSpec(
        name="silu",
        builder="src.model.activation_function.builders:build_silu",
        description="Elementwise FFN activation 'silu' (common family).",
        description_es="Activación elementwise para FFN 'silu' (familia common).",
    ),
    ActivationSpec(
        name="softexp",
        builder="src.model.activation_function.builders:build_softexp",
        description="Elementwise FFN activation 'softexp' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'softexp' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="softplus",
        builder="src.model.activation_function.builders:build_softplus",
        description="Elementwise FFN activation 'softplus' (common family).",
        description_es="Activación elementwise para FFN 'softplus' (familia common).",
    ),
    ActivationSpec(
        name="softsign",
        builder="src.model.activation_function.builders:build_softsign",
        description="Elementwise FFN activation 'softsign' (common family).",
        description_es="Activación elementwise para FFN 'softsign' (familia common).",
    ),
    ActivationSpec(
        name="swish",
        builder="src.model.activation_function.builders:build_swish",
        description="Elementwise FFN activation 'swish' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'swish' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="swish_trainable",
        builder="src.model.activation_function.builders:build_swish_trainable",
        description="Elementwise FFN activation 'swish_trainable' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'swish_trainable' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="tanh",
        builder="src.model.activation_function.builders:build_tanh",
        description="Elementwise FFN activation 'tanh' (common family).",
        description_es="Activación elementwise para FFN 'tanh' (familia common).",
    ),
    ActivationSpec(
        name="vrelu",
        builder="src.model.activation_function.builders:build_vrelu",
        description="Elementwise FFN activation 'vrelu' (rectified/exponential/learnable family).",
        description_es="Activación elementwise para FFN 'vrelu' (familia rectified/exponential/learnable).",
    ),
    ActivationSpec(
        name="geglu",
        builder="src.model.activation_function.glu:make_gated_ffn",
        description="GeGLU gated FFN — GELU gate times a linear up-projection (paper: arxiv:2002.05202)",
        description_es="GeGLU FFN con compuerta — puerta GELU por una proyección up lineal (paper: arxiv:2002.05202)",
        references=("2002.05202",),
        is_glu=True,
    ),
    ActivationSpec(
        name="reglu",
        builder="src.model.activation_function.glu:make_gated_ffn",
        description="ReGLU gated FFN — ReLU gate times a linear up-projection (paper: arxiv:2002.05202)",
        description_es="ReGLU FFN con compuerta — puerta ReLU por una proyección up lineal (paper: arxiv:2002.05202)",
        references=("2002.05202",),
        is_glu=True,
    ),
    ActivationSpec(
        name="swiglu",
        builder="src.model.activation_function.glu:make_gated_ffn",
        description="SwiGLU gated FFN — SiLU gate times a linear up-projection (paper: arxiv:2002.05202)",
        description_es="SwiGLU FFN con compuerta — puerta SiLU por una proyección up lineal (paper: arxiv:2002.05202)",
        references=("2002.05202",),
        is_glu=True,
    ),
    ActivationSpec(
        name="situ",
        builder="src.model.activation_function.glu:make_gated_ffn",
        description="SiTU-GLU gated FFN — smooth-capped Swish gate (β₁ tanh(u/β₁)·σ(u)) times a smooth-capped up-projection (β₂ tanh(v/β₂)), bounded by β₁β₂ (paper: arxiv:2607.24653)",
        description_es="FFN con compuerta SiTU-GLU — puerta Swish con tope suave (β₁ tanh(u/β₁)·σ(u)) por una proyección up con tope suave (β₂ tanh(v/β₂)), acotada por β₁β₂ (paper: arxiv:2607.24653)",
        references=("2607.24653",),
        is_glu=True,
    ),
)


def register() -> None:
    """Register every built-in activation spec (idempotent)."""
    from src.registry import register_spec

    for spec in ACTIVATION_SPECS:
        register_spec(spec)
