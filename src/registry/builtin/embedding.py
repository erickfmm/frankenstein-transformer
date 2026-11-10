"""Built-in positional-encoding and token-embedding component specs.

``PE_SPECS`` covers every ``model.positional_encoding`` enum value (plus the
``none`` alias); ``EMBEDDING_SPECS`` covers the new ``model.embedding.type``
enum. Construction functions live in :mod:`src.model.embeddings.factory` and
are resolved lazily through the registry.
"""

from __future__ import annotations

from ..spec import EmbeddingSpec, PESpec

__all__ = ["PE_SPECS", "EMBEDDING_SPECS", "register", "register_embeddings"]

_FACTORY = "src.model.embeddings.factory:"

PE_SPECS = (
    PESpec(
        name="rope",
        builder=_FACTORY + "_build_rope",
        description="Rotary Position Embedding — rotates q/k by position-dependent angles (paper: arxiv:2104.09864)",
        description_es="Embedding Posicional Rotatorio — rota q/k por ángulos dependientes de posición (paper: arxiv:2104.09864)",
        references=("2104.09864",),
        params_schema="schema/_model/_positional_encoding.yaml#/properties/positional_encoding_parameters/properties/rope",
        params_key="rope",
    ),
    PESpec(
        name="hope",
        builder=_FACTORY + "_build_hope",
        description="Higher-Order Position Encoding — hyperbolic rotation with exponential damping (paper: arxiv:2509.05218)",
        description_es="Codificación Posicional de Orden Superior — rotación hiperbólica con amortiguación exponencial (paper: arxiv:2509.05218)",
        references=("2509.05218",),
        params_schema="schema/_model/_positional_encoding.yaml#/properties/positional_encoding_parameters/properties/hope",
        params_key="hope",
    ),
    PESpec(
        name="nope",
        builder=_FACTORY + "_build_nope",
        description="No Position Encoding — identity, position learned implicitly through causal masking (paper: arxiv:2305.19466)",
        description_es="Sin Codificación Posicional — identidad; la posición se aprende implícitamente vía máscara causal (paper: arxiv:2305.19466)",
        references=("2305.19466",),
        aliases=("none",),
    ),
    PESpec(
        name="alibi",
        builder=_FACTORY + "_build_alibi",
        description="Attention with Linear Biases — distance-proportional negative score bias (paper: arxiv:2108.12409)",
        description_es="Atención con Sesgos Lineales — sesgo negativo proporcional a la distancia (paper: arxiv:2108.12409)",
        references=("2108.12409",),
        params_schema="schema/_model/_positional_encoding.yaml#/properties/positional_encoding_parameters/properties/alibi",
        params_key="alibi",
    ),
    PESpec(
        name="bam",
        builder=_FACTORY + "_build_bam",
        description="Bayesian Attention Mechanism — generalized-Gaussian relative-position bias (paper: arxiv:2505.22842)",
        description_es="Mecanismo de Atención Bayesiana — sesgo de posición relativa con Gaussiana generalizada (paper: arxiv:2505.22842)",
        references=("2505.22842",),
        params_schema="schema/_model/_positional_encoding.yaml#/properties/positional_encoding_parameters/properties/bam",
        params_key="bam",
    ),
    PESpec(
        name="pape",
        builder=_FACTORY + "_build_pape",
        description="Parabolic Position Encoding — learnable parabola terms over distance, direction and semantics (paper: arxiv:2602.01418)",
        description_es="Codificación Posicional Parabólica — términos parabólicos aprendibles sobre distancia, dirección y semántica (paper: arxiv:2602.01418)",
        references=("2602.01418",),
        params_schema="schema/_model/_positional_encoding.yaml#/properties/positional_encoding_parameters/properties/pape",
        params_key="pape",
    ),
    PESpec(
        name="pape_efficient",
        builder=_FACTORY + "_build_pape_efficient",
        description="Efficient PaPE — same math as `pape` with a pre-computed layout (paper: arxiv:2602.01418)",
        description_es="PaPE Eficiente — misma matemática que `pape` con layout pre-calculado (paper: arxiv:2602.01418)",
        references=("2602.01418",),
        params_schema="schema/_model/_positional_encoding.yaml#/properties/positional_encoding_parameters/properties/pape",
        params_key="pape",
    ),
    PESpec(
        name="pape_ri",
        builder=_FACTORY + "_build_pape_ri",
        description="Rotation-Invariant PaPE — distance-only parabola terms for 3D vision (paper: arxiv:2602.01418)",
        description_es="PaPE Invariante a Rotación — términos de distancia únicamente para visión 3D (paper: arxiv:2602.01418)",
        references=("2602.01418",),
        params_schema="schema/_model/_positional_encoding.yaml#/properties/positional_encoding_parameters/properties/pape",
        params_key="pape",
    ),
    PESpec(
        name="sinusoidal_absolute",
        builder=_FACTORY + "_build_sinusoidal_absolute",
        description="Fixed sinusoidal absolute positional encoding added to token embeddings (paper: arxiv:1706.03762)",
        description_es="Codificación posicional absoluta sinusoidal fija añadida a los embeddings de token (paper: arxiv:1706.03762)",
        references=("1706.03762",),
        params_schema="schema/_model/_positional_encoding.yaml#/properties/positional_encoding_parameters/properties/sinusoidal",
        params_key="sinusoidal",
    ),
    PESpec(
        name="sinusoidal_rotary",
        builder=_FACTORY + "_build_sinusoidal_rotary",
        description="Sinusoidal rotary positional encoding — absolute sin/cos rotation applied to q/k (paper: arxiv:1906.08237)",
        description_es="Codificación posicional rotatoria sinusoidal — rotación sin/cos absoluta aplicada a q/k (paper: arxiv:1906.08237)",
        references=("1906.08237",),
        params_schema="schema/_model/_positional_encoding.yaml#/properties/positional_encoding_parameters/properties/sinusoidal",
        params_key="sinusoidal",
    ),
    PESpec(
        name="learned_absolute",
        builder=_FACTORY + "_build_learned_absolute",
        description="Learnable absolute positional encoding added to token embeddings (paper: arxiv:2010.11929)",
        description_es="Codificación posicional absoluta aprendible añadida a los embeddings de token (paper: arxiv:2010.11929)",
        references=("2010.11929",),
        params_schema="schema/_model/_positional_encoding.yaml#/properties/positional_encoding_parameters/properties/learned",
        params_key="learned",
    ),
)

EMBEDDING_SPECS = (
    EmbeddingSpec(
        name="standard",
        builder=_FACTORY + "_build_embedding_standard",
        description="Standard token embedding lookup table `vocab_size × hidden_size`",
        description_es="Tabla de búsqueda de embeddings de token estándar `vocab_size × hidden_size`",
    ),
    EmbeddingSpec(
        name="factorized",
        builder=_FACTORY + "_build_embedding_factorized",
        description="Factorized low-rank embedding — two smaller matrices replace the full lookup table (paper: arxiv:2004.02984)",
        description_es="Embedding factorizado de bajo rango — dos matrices menores reemplazan la tabla completa (paper: arxiv:2004.02984)",
        references=("2004.02984",),
    ),
    EmbeddingSpec(
        name="dynamic",
        builder=_FACTORY + "_build_embedding_dynamic",
        description="Dynamic (hash) embedding — no trainable lookup table; tokens are hashed n-grams (paper: arxiv:2310.20144)",
        description_es="Embedding dinámico (hash) — sin tabla aprendible; los tokens son n-gramas hasheados (paper: arxiv:2310.20144)",
        references=("2310.20144",),
    ),
)


def register() -> None:
    """Register positional-encoding and token-embedding specs (idempotent)."""
    from src.registry import register_spec

    for spec in PE_SPECS:
        register_spec(spec)
    for spec in EMBEDDING_SPECS:
        register_spec(spec)
