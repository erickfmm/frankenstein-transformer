#!/usr/bin/env python3
"""Frankenstein Transformer: model configuration dataclass.

This module hosts :class:`FrankensteinModelConfig`, the single-source-of-truth
dataclass for all model hyperparameters, along with the helper validator
:func:`_validate_ffn_activation_config`.

The configuration is validated in ``FrankensteinModelConfig.__post_init__`` and
enforced by ``src/schema.yaml`` for YAML-based training configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .activation_function import ALL_ACTIVATIONS

# Allowed keys for the nested ``ffn_activation_config`` mapping. Each entry
# maps to the validation rule applied by ``_validate_ffn_activation_config``.
_FFN_ACTIVATION_CONFIG_KEYS = {
    "raf_degrees", "raf_version", "raf_approx_func", "raf_trainable",
    "raf_input_scaling", "prelu_init", "elu_alpha", "celu_alpha",
    "swish_beta", "leaky_relu_slope", "pelu_alpha", "mpelu_alpha",
    "mpelu_beta", "felu_alpha", "eelu_alpha", "eelu_beta", "pdelu_alpha",
    "preu_alpha", "preu_beta", "softexp_alpha", "maxout_pieces",
}
_RAF_VERSIONS = {"A", "B", "C", "D", "N"}
_RAF_APPROX_FUNCS = {
    "gelu", "relu", "leaky_relu", "leaky_relu_0.1", "sigmoid", "tanh",
    "swish", "silu", "identity",
}


def _validate_ffn_activation_config(activation: str, cfg: Dict[str, Any]) -> None:
    """Validate the nested ``ffn_activation_config`` mapping.

    Args:
        activation: The lower-cased ``ffn_activation`` name.
        cfg: The nested config mapping.

    Raises:
        ValueError: If an unknown key is present, a value has the wrong type,
            or a RAF-specific constraint (degrees, version, approx_func) is
            violated.
    """
    unknown = set(cfg) - _FFN_ACTIVATION_CONFIG_KEYS
    if unknown:
        raise ValueError(
            f"Unknown ffn_activation_config keys: {sorted(unknown)}. "
            f"Allowed: {sorted(_FFN_ACTIVATION_CONFIG_KEYS)}"
        )
    # Type checks for the boolean keys.
    for bk in ("raf_trainable", "raf_input_scaling"):
        if bk in cfg and not isinstance(cfg[bk], bool):
            raise ValueError(f"ffn_activation_config.{bk} must be a boolean")
    # Float keys.
    for fk in (
        "prelu_init", "elu_alpha", "celu_alpha", "swish_beta", "leaky_relu_slope",
        "pelu_alpha", "mpelu_alpha", "mpelu_beta", "felu_alpha", "eelu_alpha",
        "eelu_beta", "pdelu_alpha", "preu_alpha", "preu_beta", "softexp_alpha",
    ):
        if fk in cfg and not isinstance(cfg[fk], (int, float)):
            raise ValueError(f"ffn_activation_config.{fk} must be a number")
    if "maxout_pieces" in cfg:
        if not isinstance(cfg["maxout_pieces"], int) or cfg["maxout_pieces"] < 1:
            raise ValueError("ffn_activation_config.maxout_pieces must be an int >= 1")
    # RAF degrees.
    if "raf_degrees" in cfg:
        d = cfg["raf_degrees"]
        if (
            not isinstance(d, (list, tuple))
            or len(d) != 2
            or not all(isinstance(v, int) and v >= 1 for v in d)
        ):
            raise ValueError(
                "ffn_activation_config.raf_degrees must be a [m, n] pair of ints >= 1"
            )
    if "raf_version" in cfg and cfg["raf_version"] not in _RAF_VERSIONS:
        raise ValueError(
            f"ffn_activation_config.raf_version must be one of "
            f"{sorted(_RAF_VERSIONS)}, got {cfg['raf_version']!r}"
        )
    if "raf_approx_func" in cfg and cfg["raf_approx_func"] not in _RAF_APPROX_FUNCS:
        raise ValueError(
            f"ffn_activation_config.raf_approx_func must be one of "
            f"{sorted(_RAF_APPROX_FUNCS)}, got {cfg['raf_approx_func']!r}"
        )


@dataclass
class FrankensteinModelConfig:
    """Single-source-of-truth configuration for all Frankenstein model variants.

    Every hyperparameter lives here. The schema is validated in
    ``__post_init__`` and enforced by ``configs/schema.yaml`` for YAML-based
    training configs.

    Attributes:
        vocab_size: Vocabulary size for token embeddings. Default: 50000.
        hidden_size: Dimensionality of hidden states throughout the model.
            Must be divisible by ``num_heads``. Default: 2048.
        num_layers: Number of physical :class:`HybridLayer` blocks in the
            stack. Default: 12.
        num_loops: Number of times the layer stack is repeated (looped depth).
            Logical depth = ``num_layers * num_loops``. Default: 2.
        layer_pattern: Ordered list of mixer types assigned to each physical
            layer. The pattern is cycled modulo its length when
            ``num_layers`` exceeds the pattern length. Supported values:
            ``"ode"``, ``"retnet"``, ``"retnet_attn"``, ``"titan_attn"``,
            ``"standard_attn"``, ``"sigmoid_attn"``, ``"mamba"``,
            ``"sparse_transformer_attn"``, ``"longformer_attn"``,
            ``"bigbird_attn"``, ``"sparsek_attn"``, ``"nsa_attn"``,
            ``"sparge_attn"``, ``"fasa_attn"``, ``"gla_attn"``,
            ``"deltanet_attn"``, ``"gated_deltanet_attn"``,
            ``"gated_deltanet2_attn"``,
            ``"hgrn2_attn"``, ``"fox_attn"``, ``"gated_softmax_attn"``,
            ``"kda_attn"``,
            ``"engram_attn"``, ``"gqa_attn"``,
            ``"mla_attn"``, ``"gqla_attn"``, ``"mlra_attn"``,
            ``"tucker_attn"``, ``"iha_attn"``, ``"gta_attn"``,
            ``"mtla_attn"``, ``"cca_attn"``, ``"ccgqa_attn"``,
            ``"msa_attn"``, ``"sparda_attn"``.
            Default: ``["retnet", "ode", "mamba", "titan_attn"] * 3``.
        ode_solver: ODE solver for ``ode`` mixer layers. One of ``"rk4"``
            (Runge-Kutta 4th order) or ``"euler"``. Default: ``"rk4"``.
        ode_steps: Number of ODE integration steps per ``ode`` layer.
            Default: 2.
        retention_heads: Number of retention heads for ``retnet`` /
            ``retnet_attn`` layers. Default: 8.
        num_heads: Number of attention heads for standard / sparse / gated
            attention mixers. Default: 16.
        num_experts: Total number of FFN experts when ``use_moe`` is True.
            Default: 8.
        top_k_experts: Number of experts activated per token in MoE routing.
            Default: 2.
        dropout: Dropout probability applied after embeddings and within
            attention layers. Default: 0.1.
        use_bitnet: If True, replace all primary and gate ``nn.Linear``
            layers with :class:`BitLinear` (ternary weight quantization,
            BitNet b1.58). Routing/scoring projections are governed
            separately by ``bitnet_routers``. Default: True.
        bitnet_routers: If True (and ``use_bitnet`` is True), also quantize
            routing/scoring projections (MoE router, Mixture-of-Depths
            router, sparse block-index/forecast, top-k score nets) to
            :class:`BitLinear`. Default ``False`` keeps them full-precision
            for routing stability. Default: False.
        use_bitnet_conv: If True (and ``use_bitnet`` is True), replace the
            factorized-embedding Conv1d pre-projection with
            :class:`BitConv1d` (ternary weights). Default ``False`` keeps
            the Conv1d full-precision; the convolution operates over the
            reduced embedding stream where ternary quantization can be noisy.
            No effect when ``use_bitnet`` is False or the embedding conv is
            disabled. Default: False.
        norm_type: Normalization layer type. One of ``"layer_norm"``,
            ``"dynamic_tanh"`` (DyT), ``"derf"`` (Dynamic Erf),
            ``"rms_norm"`` (RMSNorm), ``"prms_norm"`` (partial RMSNorm),
            or ``"flash_norm"`` (FlashNorm — weightless RMSNorm; the
            per-dim scale ``g`` is dropped per Prop. 1 of arXiv:2407.09577
            and is meant to be absorbed by the subsequent linear layer).
            Default: ``"dynamic_tanh"``.
        prms_partial_ratio: Fraction of hidden dimensions used for RMS
            estimation when ``norm_type="prms_norm"``. The paper default is
            6.25%. Must be in ``(0, 1]``. Ignored for other ``norm_type``
            values. Default: ``0.0625``.
        flashnorm_partial_ratio: Fraction of hidden dimensions used for
            RMS estimation when ``norm_type="flash_norm"``. ``0.0``
            (default) uses the full RMS — standard FlashNorm. Values in
            ``(0, 1]`` activate the partial-RMS variant (composing the
            pRMSNorm trick with Prop. 1 of FlashNorm). Must be in
            ``[0, 1]``. Ignored for other ``norm_type`` values.
            Default: ``0.0``.
        use_factorized_embedding: If True, use :class:`FactorizedEmbedding`
            with reduced embedding dimension + projection. Default: False.
        factorized_embedding_dim: Embedding dimension when factorization is
            enabled. Default: 128.
        use_embedding_conv: If True, apply a Conv1d over the factorized
            embedding stream before projection. Default: True.
        hope_base: Base frequency for HoPE (Hybrid Positional Encoding).
            Default: 10000.0.
        hope_damping: Damping factor for HoPE high-frequency components.
            Default: 0.01.
        rope_base: Base frequency for RoPE (Rotary Position Embedding).
            Default: 10000.0.
        rope_scaling: Scaling factor applied to RoPE frequencies.
            Default: 1.0.
        use_hope: Legacy toggle for HoPE. Automatically aligned with
            ``positional_encoding`` in ``__post_init__``. Default: True.
        positional_encoding: Explicit positional encoding scheme. One of
            ``"rope"`` (Rotary Position Embedding), ``"hope"`` (Hybrid
            Positional Encoding), ``"nope"`` (No Position Embedding),
            ``"alibi"`` (Attention with Linear Biases), ``"pape"``
            (Parabola Attention Positional Encoding), ``"pape_efficient"``
            (efficient PaPE variant), ``"pape_ri"`` (rotation-invariant
            PaPE), ``"sinusoidal_absolute"`` (absolute sinusoidal), 
            ``"sinusoidal_rotary"`` (rotary sinusoidal),
            ``"learned_absolute"`` (learned absolute position embedding), or
            ``"none"`` (explicitly disabled). If None, inferred from
            ``use_hope``. Default: None.
        alibi_num_heads: Number of heads used by ALiBi biases. Defaults to
            ``num_heads`` in ``__post_init__`` when None. Default: None.
        pape_num_parabolas: Number of parabola segments used by PaPE and
            its variants (``pape``, ``pape_efficient``, ``pape_ri``). Must
            be >= 1. Default: 4.
        pape_num_positions: Number of positional dimensions for PaPE. Use
            ``1`` for 1-D language sequences and ``2`` for 2-D vision
            (row/column). Must be >= 1. Default: 1.
        pape_rotation_invariant: If True, use the rotation-invariant PaPE
            variant regardless of the ``positional_encoding`` value.
            Default: False.
        sinusoidal_max_len: Maximum sequence length for precomputed
            sinusoidal positional encodings. Default: 512.
        sinusoidal_base: Base wavelength of the sinusoidal encoding
            (``10000`` in the original Transformer paper). Default: 10000.0.
        sinusoidal_scale: Global scaling factor applied to sinusoidal
            frequencies. Default: 1.0.
        learned_max_len: Maximum sequence length for the learned absolute
            positional embedding table. Default: 512.
        learned_init_std: Initialisation standard deviation for the
            learned absolute positional embedding. Default: 0.02.
        standard_attn_use_pe: Whether ``standard_attn`` layers consume the
            model-wide positional encoding. Default: True.
        titan_attn_use_pe: Whether ``titan_attn`` layers consume the
            model-wide positional encoding. Default: True.
        sigmoid_attn_use_pe: Whether ``sigmoid_attn`` layers consume the
            model-wide positional encoding. Default: True.
        gated_softmax_attn_use_pe: Whether ``gated_softmax_attn`` layers
            consume the model-wide positional encoding. Default: True.
        gqa_attn_use_pe: Whether ``gqa_attn`` layers consume the
            model-wide positional encoding. Default: True.
        mla_attn_use_pe: Whether ``mla_attn`` layers consume the
            model-wide positional encoding. Default: True.
        gqla_attn_use_pe: Whether ``gqla_attn`` layers consume the
            model-wide positional encoding. Default: True.
        mlra_attn_use_pe: Whether ``mlra_attn`` layers consume the
            model-wide positional encoding. Default: True.
        tucker_attn_use_pe: Whether ``tucker_attn`` layers consume the
            model-wide positional encoding. Default: True.
        iha_attn_use_pe: Whether ``iha_attn`` layers consume the
            model-wide positional encoding. Default: True.
        gta_attn_use_pe: Whether ``gta_attn`` layers consume the
            model-wide positional encoding. Default: True.
        mtla_attn_use_pe: Whether ``mtla_attn`` layers consume the
            model-wide positional encoding. Disabled by default because
            MTLA is a recurrent/decay mixer. Default: False.
        cca_attn_use_pe: Whether ``cca_attn`` layers consume the
            model-wide positional encoding. Default: True.
        ccgqa_attn_use_pe: Whether ``ccgqa_attn`` layers consume the
            model-wide positional encoding. Default: True.
        msa_attn_use_pe: Whether ``msa_attn`` layers consume the
            model-wide positional encoding. Default: True.
        sparda_attn_use_pe: Whether ``sparda_attn`` layers consume the
            model-wide positional encoding. Default: True.
        gma_attn_use_pe: Whether ``gma_attn`` layers consume the
            model-wide positional encoding. Default: True.
        longformer_attn_use_pe: Whether ``longformer_attn`` layers consume
            the model-wide positional encoding. Default: True.
        bigbird_attn_use_pe: Whether ``bigbird_attn`` layers consume the
            model-wide positional encoding. Default: True.
        sparse_transformer_attn_use_pe: Whether ``sparse_transformer_attn``
            layers consume the model-wide positional encoding.
            Default: True.
        sparsek_attn_use_pe: Whether ``sparsek_attn`` layers consume the
            model-wide positional encoding. Default: True.
        nsa_attn_use_pe: Whether ``nsa_attn`` layers consume the
            model-wide positional encoding. Default: True.
        fasa_attn_use_pe: Whether ``fasa_attn`` layers consume the
            model-wide positional encoding. Default: True.
        sparge_attn_use_pe: Whether ``sparge_attn`` layers consume the
            model-wide positional encoding. Default: True.
        engram_attn_use_pe: Whether ``engram_attn`` layers consume the
            model-wide positional encoding. Default: True.
        retnet_use_pe: Whether ``retnet`` layers consume the model-wide
            positional encoding. Disabled by default because RetNet uses
            decay-based positional information. Default: False.
        retnet_attn_use_pe: Whether ``retnet_attn`` layers consume the
            model-wide positional encoding. Disabled by default because
            RetNet-Attn uses decay-based positional information.
            Default: False.
        mamba_use_pe: Whether ``mamba`` layers consume the model-wide
            positional encoding. Disabled by default because Mamba is a
            state-space mixer with implicit positional information.
            Default: False.
        ode_use_pe: Whether ``ode`` layers consume the model-wide
            positional encoding. Disabled by default because ODE layers
            integrate over the hidden state directly. Default: False.
        gla_attn_use_pe: Whether ``gla_attn`` layers consume the
            model-wide positional encoding. Disabled by default because
            GLA is a gated linear attention mixer. Default: False.
        deltanet_attn_use_pe: Whether ``deltanet_attn`` layers consume
            the model-wide positional encoding. Disabled by default
            because DeltaNet is a gated linear attention mixer.
            Default: False.
        gated_deltanet_attn_use_pe: Whether ``gated_deltanet_attn``
            layers consume the model-wide positional encoding. Disabled
            by default because gated DeltaNet is a gated linear attention
            mixer. Default: False.
        gated_deltanet2_attn_use_pe: Whether ``gated_deltanet2_attn``
            layers consume the model-wide positional encoding. Disabled
            by default because gated DeltaNet-2 is a gated linear
            attention mixer. Default: False.
        hgrn2_attn_use_pe: Whether ``hgrn2_attn`` layers consume the
            model-wide positional encoding. Disabled by default because
            HGRN-2 is a gated linear attention mixer. Default: False.
        fox_attn_use_pe: Whether ``fox_attn`` layers consume the
            model-wide positional encoding. Disabled by default because
            Fox is a gated linear attention mixer. Default: False.
        kda_attn_use_pe: Whether ``kda_attn`` layers consume the
            model-wide positional encoding. Disabled by default because
            KDA is a gated linear attention mixer. Default: False.
        use_moe: If True, replace the dense FFN with a Mixture-of-Experts
            FFN block. Default: True.
        use_mixture_of_depths: If True, apply per-layer token routing where
            only the top-capacity tokens are updated; remaining tokens are
            passed through unchanged. Default: False.
        mixture_of_depths_capacity_ratio: Fraction of tokens selected per
            layer when Mixture-of-Depths is active. Must be in (0, 1].
            Default: 0.5.
        mixture_of_depths_router_aux_loss_weight: Weight for the auxiliary
            load-balancing loss in Mixture-of-Depths routing. Must be >= 0.
            Default: 0.0.
        ffn_hidden_size: Hidden size of the FFN intermediate layer. If None,
            defaults to ``hidden_size * 2``. Default: None.
        ffn_activation: FFN activation function. One of the elementwise
            activations (e.g. ``"silu"`` (default, SiLU / Swish), ``"gelu"``,
            ``"relu"``, ``"mish"``, ``"prelu"``, ``"raf"`` (Rational Activation
            Function, learnable), ...) or a gated FFN variant
            (``"swiglu"``, ``"geglu"``, ``"reglu"``). See
            ``src/model/activation_function/factory.py`` for the full enum.
        ffn_activation_config: Optional nested mapping of learnable-activation
            parameters (e.g. ``raf_degrees``, ``raf_version``,
            ``raf_approx_func``, ``raf_trainable``, ``raf_input_scaling``,
            ``prelu_init``, ``elu_alpha``, ``swish_beta``). Ignored for
            stateless activations. Default: ``None``.
        embedding_conv_kernel: Kernel size for the embedding Conv1d when
            ``use_embedding_conv`` is True. Default: 3.
        mode: Model mode. ``"encoder"`` for bidirectional (MLM) or
            ``"decoder"`` for autoregressive causal generation. The
            ``model_class=frankensteindecoder`` preset forces ``mode=decoder``
            at runtime. Default: ``"encoder"``.
        engram_max_ngram_size: Highest N-gram order for Engram memory layers
            (range 2..max). Default: 3.
        engram_n_heads_per_ngram: Number of hash heads per N-gram order in
            Engram layers. Default: 4.
        engram_embed_dim_per_head: Embedding dimension per Engram hash head.
            Default: 32.
        engram_kernel_size: ShortConv kernel width for Engram layers.
            Default: 4.
        engram_seed: RNG seed for Engram hash multipliers. Default: 42.

    Raises:
        ValueError: If ``positional_encoding`` is not one of
            ``"rope"``, ``"hope"``, ``"nope"``, ``"alibi"``, ``"pape"``,
            ``"pape_efficient"``, ``"pape_ri"``, ``"sinusoidal_absolute"``,
            ``"sinusoidal_rotary"``, ``"learned_absolute"``, or ``"none"``.
        ValueError: If ``pape_num_parabolas`` or ``pape_num_positions`` is
            less than 1.
        ValueError: If ``mode`` is not ``"encoder"`` or ``"decoder"``.
        ValueError: If ``mixture_of_depths_capacity_ratio`` is not in (0, 1].
        ValueError: If ``mixture_of_depths_router_aux_loss_weight`` is < 0.
        ValueError: If ``pos_embedding_type`` is not one of the legacy
            values (``"learned_1d"``, ``"none"``) or the unified positional
            encoding enum values.
    """

    vocab_size: int = 50000
    hidden_size: int = 2048
    num_layers: int = 12
    num_loops: int = 2

    layer_pattern: List[str] = field(default_factory=lambda: ["retnet", "ode", "mamba", "titan_attn"] * 3)

    ode_solver: str = "rk4"
    ode_steps: int = 2

    retention_heads: int = 8

    num_heads: int = 16
    num_experts: int = 8
    top_k_experts: int = 2
    dropout: float = 0.1

    use_bitnet: bool = True
    bitnet_routers: bool = False
    use_bitnet_conv: bool = False
    norm_type: str = "dynamic_tanh"
    prms_partial_ratio: float = 0.0625
    flashnorm_partial_ratio: float = 0.0
    use_factorized_embedding: bool = False
    factorized_embedding_dim: int = 128
    use_embedding_conv: bool = True

    hope_base: float = 10_000.0
    hope_damping: float = 0.01
    rope_base: float = 10_000.0
    rope_scaling: float = 1.0

    use_hope: bool = True
    positional_encoding: Optional[str] = None  # defaults to "rope" in __post_init__

    alibi_num_heads: Optional[int] = None

    pape_num_parabolas: int = 4
    pape_num_positions: int = 1
    pape_rotation_invariant: bool = False

    sinusoidal_max_len: int = 512
    sinusoidal_base: float = 10000.0
    sinusoidal_scale: float = 1.0

    learned_max_len: int = 512
    learned_init_std: float = 0.02

    standard_attn_use_pe: bool = True
    titan_attn_use_pe: bool = True
    sigmoid_attn_use_pe: bool = True
    gated_softmax_attn_use_pe: bool = True
    gqa_attn_use_pe: bool = True
    mla_attn_use_pe: bool = True
    gqla_attn_use_pe: bool = True
    mlra_attn_use_pe: bool = True
    tucker_attn_use_pe: bool = True
    iha_attn_use_pe: bool = True
    gta_attn_use_pe: bool = True
    mtla_attn_use_pe: bool = False
    cca_attn_use_pe: bool = True
    ccgqa_attn_use_pe: bool = True
    msa_attn_use_pe: bool = True
    sparda_attn_use_pe: bool = True
    gma_attn_use_pe: bool = True
    longformer_attn_use_pe: bool = True
    bigbird_attn_use_pe: bool = True
    sparse_transformer_attn_use_pe: bool = True
    sparsek_attn_use_pe: bool = True
    nsa_attn_use_pe: bool = True
    fasa_attn_use_pe: bool = True
    sparge_attn_use_pe: bool = True
    engram_attn_use_pe: bool = True
    retnet_use_pe: bool = False
    retnet_attn_use_pe: bool = False
    mamba_use_pe: bool = False
    ode_use_pe: bool = False
    gla_attn_use_pe: bool = False
    deltanet_attn_use_pe: bool = False
    gated_deltanet_attn_use_pe: bool = False
    gated_deltanet2_attn_use_pe: bool = False
    hgrn2_attn_use_pe: bool = False
    fox_attn_use_pe: bool = False
    kda_attn_use_pe: bool = False

    use_moe: bool = True
    use_mixture_of_depths: bool = False
    mixture_of_depths_capacity_ratio: float = 0.5
    mixture_of_depths_router_aux_loss_weight: float = 0.0
    ffn_hidden_size: Optional[int] = None
    ffn_activation: str = "silu"
    ffn_activation_config: Optional[Dict[str, Any]] = None
    embedding_conv_kernel: int = 3
    mode: str = "encoder"

    engram_max_ngram_size: int = 3
    engram_n_heads_per_ngram: int = 4
    engram_embed_dim_per_head: int = 32
    engram_kernel_size: int = 4
    engram_seed: int = 42

    # ---- mHC: Manifold-Constrained Hyper-Connections (arXiv:2512.24880) ----
    # Expands the residual stream to width ``n * hidden_size`` and constrains
    # the stream-mixing matrix ``H[res]`` to the Birkhoff polytope (doubly
    # stochastic) via Sinkhorn-Knopp, restoring the identity-mapping property.
    use_mhc: bool = False
    mhc_expansion_rate: int = 4
    mhc_sinkhorn_iters: int = 20
    mhc_gating_init: float = 0.01
    mhc_checkpoint: bool = False
    # Keep ``φ_l`` full-precision under BitNet (avoids ternary-quantisation
    # noise on the small mHC coefficients). Defaults to True.
    mhc_full_prec_under_bitnet: bool = True

    # ---- Attention Residuals (AttnRes, arXiv:2603.15031) ----
    # Depth-wise softmax attention replaces the fixed residual coefficient of 1.
    # ``residual_type`` selects the strategy:
    #   - "standard": h_l = h_{l-1} + f_l(h_{l-1})  [default, backwards-compatible]
    #   - "none": h_l = f_l(h_{l-1})                 [experimental ablation]
    #   - "full_attn": softmax attention over all previous layer outputs.
    #   - "block_attn": softmax attention over N block representations + partial sum.
    # AttnRes variants add O(num_layers * hidden_size) parameters (one query
    # vector per layer) and may be combined with mHC via ``attnres_mhc_stream_mode``.
    residual_type: str = "standard"
    # Full AttnRes: zero-init query vectors so initial attention is uniform.
    full_attn_init_query_zero: bool = True
    # Full AttnRes: RMSNorm on keys prevents large-magnitude layers dominating.
    full_attn_use_rmsnorm_keys: bool = True
    # Block AttnRes: zero-init query vectors (paper §3.2 recommendation).
    block_attn_init_query_zero: bool = True
    # Block AttnRes: RMSNorm on keys (paper §3.2 recommendation).
    block_attn_use_rmsnorm_keys: bool = True
    # Block AttnRes: number of block representations N (paper sweet spot ≈ 8).
    block_attn_num_blocks: int = 8
    # AttnRes: how to combine with mHC's n-stream residual:
    #   - "independent": per-stream depth-wise attention (n parallel attentions).
    #   - "joint":       single attention over the flattened nC projection.
    attnres_mhc_stream_mode: str = "independent"
    # AttnRes: wrap the attention computation in gradient checkpointing to save
    # memory at the cost of one extra forward pass during backprop.
    attnres_gradient_checkpoint: bool = False

    num_kv_heads: int = 1

    # ---- MLA (arXiv:2506.09342) ----
    mla_latent_rank: Optional[int] = None

    # ---- GQLA (arXiv:2605.15250) ----
    gqla_latent_rank: Optional[int] = None
    gqla_num_groups: Optional[int] = None
    gqla_decode_path: str = "gqa"

    # ---- MLRA (arXiv:2603.02188) ----
    mlra_latent_rank: Optional[int] = None
    mlra_num_latent_heads: int = 4

    # ---- Tucker Attention (arXiv:2603.30033) ----
    tucker_query_rank: Optional[int] = None
    tucker_key_rank: Optional[int] = None
    tucker_value_rank: Optional[int] = None

    # ---- IHA (arXiv:2602.21371) ----
    iha_num_pseudo_heads: Optional[int] = None

    # ---- GTA (arXiv:2506.17286) ----
    gta_num_shared_groups: Optional[int] = None
    gta_value_latent_rank: Optional[int] = None

    # ---- MTLA (arXiv:2505.13544) ----
    mtla_latent_rank: Optional[int] = None
    mtla_merge_factor: int = 2
    mtla_stride: Optional[int] = None

    # ---- GMA / Gaussian Mixture Attention (arXiv:2606.18283) ----
    # Number K of learned Gaussian mixture components per head. Each
    # component routes values into one latent memory slot.
    gma_num_components: int = 8
    # Routing dimension d_r used to compute Gaussian responsibilities.
    # None -> head_dim (hidden_size // num_heads). Decoupled from d_v.
    gma_routing_dim: Optional[int] = None
    # Numerical-stability constant for the read-step normaliser
    # Gamma^Q Z + epsilon. Must be > 0.
    gma_epsilon: float = 1e-6
    # Lower bound for the diagonal variances: sigma^2 = softplus(omega)
    # + sigma_eps. Guarantees strict positive-definiteness. Must be > 0.
    gma_sigma_eps: float = 1e-4
    # Initialisation std for the component means mu ~ N(0, init_mean_std^2).
    gma_init_mean_std: float = 1.0

    # ---- CCA / CCGQA (arXiv:2510.04476) ----
    cca_latent_rank: Optional[int] = None
    cca_num_conv_layers: int = 2
    cca_conv_kernel_seq: int = 4
    cca_conv_kernel_ch: int = 3
    cca_qk_mean: bool = True
    cca_value_shift: bool = True

    ccgqa_query_latent_rank: Optional[int] = None
    ccgqa_kv_latent_rank: Optional[int] = None
    ccgqa_num_kv_heads: Optional[int] = None
    ccgqa_num_conv_layers: int = 2
    ccgqa_conv_kernel_seq: int = 4
    ccgqa_conv_kernel_ch: int = 3
    ccgqa_qk_mean: bool = True
    ccgqa_value_shift: bool = True

    # ---- MSA / MiniMax Sparse Attention (arXiv:2606.13392) ----
    msa_block_size: int = 128
    msa_topk_blocks: int = 16
    msa_index_dim: int = 64
    msa_kl_loss_weight: float = 0.0

    # ---- SparDA (arXiv:2606.04511) ----
    sparda_block_size: int = 128
    sparda_topk_blocks: int = 16
    sparda_forecast_dim: int = 64

    # ---- Vision Transformer (frankenstein_vit, arXiv:2010.11929) ----
    # Image dimensions and patch config for the Vision Transformer model class.
    # When ``model_class == "frankenstein_vit"``, the model splits an image
    # into non-overlapping ``patch_size``×``patch_size`` patches, linearly
    # projects each patch to ``hidden_size`` dimensions, and processes the
    # resulting sequence through the standard HybridLayer stack.
    image_height: int = 224
    image_width: int = 224
    patch_size: int = 16
    in_channels: int = 3
    to_grayscale: bool = False
    # Positional encoding for ViT: ``"learned_1d"`` adds a learnable
    # ``nn.Parameter`` of shape ``(1, N+cls, D)`` after patch embedding
    # (faithful to the ViT paper). ``"none"`` relies on the existing
    # RoPE/HoPE inside attention mixers (treats patches as a sequence).
    pos_embedding_type: str = "learned_1d"
    # Whether to prepend a learnable [CLS] token (ViT paper default).
    cls_token: bool = True
    # Pooling mode for classification: ``"cls"`` reads the [CLS] token
    # output at position 0; ``"gap"`` averages over all patch tokens.
    pooling_mode: str = "cls"
    # Masked patch prediction (autosupervised): fraction of patches to
    # corrupt during ``task == "patch_prediction"``. Paper default: 0.5.
    mask_ratio: float = 0.5
    # Masking strategy: ``"bert"`` = 80% mask-token / 10% random / 10%
    # keep (paper recipe); ``"mask_only"`` = 100% mask-token;
    # ``"random_only"`` = 100% random patch.
    mask_token_strategy: str = "bert"
    # Reconstruction target for patch prediction: ``"mean_color_3bit"``
    # (512-way CE, paper best), ``"downsampled_3bit"`` (16×512 CE),
    # ``"full_patch_l2"`` (MSE on raw patch pixels, MAE-style).
    prediction_target: str = "mean_color_3bit"
    # Segmentation head type: ``"pixel"`` = per-pixel linear head
    # (supports 1D grayscale + multicolor); ``"eomt"`` = Encoder-only
    # Mask Transformer (arXiv:2503.19108, query-based with Hungarian
    # matching and mask annealing).
    seg_head_type: str = "pixel"
    # Number of classes for image classification.
    num_classes: int = 1000
    # Number of segmentation classes (including background).
    num_seg_classes: int = 21
    # EoMT: number of learnable object queries.
    seg_num_queries: int = 100
    # EoMT: number of L₂ blocks that process patches + queries jointly.
    seg_l2_blocks: int = 3
    # EoMT: mask annealing (polynomial decay of P_mask to 0 at inference).
    seg_mask_annealing: bool = True
    # Optional sequence-level classification head on the NLP encoder (DashAI
    # integration, Strategy A). When ``classification_head=True`` and
    # ``num_labels`` is set, ``FrankensteinEncoder.forward`` returns
    # ``(B, num_labels)`` class logits instead of ``(B, S, vocab_size)`` MLM
    # logits. The head is a full-precision ``nn.Linear`` (NOT BitNet-quantized)
    # over a pooled representation. Disabled by default so the MLM CLI path is
    # unchanged.
    classification_head: bool = False
    # Number of target classes for the encoder classification head. Ignored
    # unless ``classification_head=True``.
    num_labels: Optional[int] = None
    # Pooling for the encoder classification head: ``"cls"`` reads the first
    # token output; ``"gap"`` averages over all tokens. Ignored unless
    # ``classification_head=True``. Mirrors the ViT ``pooling_mode``.
    encoder_pooling_mode: str = "cls"

    def __post_init__(self):
        """Validate and derive dependent configuration fields after dataclass init.

        Derives ``ffn_hidden_size``, ``positional_encoding``, and aligns the
        legacy ``use_hope`` flag. Validates ``mode``, ``positional_encoding``,
        and Mixture-of-Depths parameter ranges.

        Raises:
            ValueError: If any field fails validation constraints.
        """
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size * 2

        # ---- Resolve latent-family ranks that default to hidden_size // 2 ----
        half = max(1, self.hidden_size // 2)
        if self.mla_latent_rank is None:
            self.mla_latent_rank = half
        if self.gqla_latent_rank is None:
            self.gqla_latent_rank = half
        if self.gqla_num_groups is None:
            self.gqla_num_groups = max(1, self.num_heads // 4)
        if self.mlra_latent_rank is None:
            self.mlra_latent_rank = half
        if self.tucker_query_rank is None:
            self.tucker_query_rank = self.hidden_size
        if self.tucker_key_rank is None:
            self.tucker_key_rank = half
        if self.tucker_value_rank is None:
            self.tucker_value_rank = half
        if self.iha_num_pseudo_heads is None:
            self.iha_num_pseudo_heads = self.num_heads
        if self.gta_num_shared_groups is None:
            self.gta_num_shared_groups = max(1, self.num_heads // 4)
        if self.gta_value_latent_rank is None:
            self.gta_value_latent_rank = half
        if self.mtla_latent_rank is None:
            self.mtla_latent_rank = half
        if self.mtla_stride is None:
            self.mtla_stride = self.mtla_merge_factor

        # ---- Resolve CCA / CCGQA defaults ----
        if self.cca_latent_rank is None:
            self.cca_latent_rank = max(1, self.hidden_size // 4)
        if self.ccgqa_query_latent_rank is None:
            self.ccgqa_query_latent_rank = half
        if self.ccgqa_kv_latent_rank is None:
            self.ccgqa_kv_latent_rank = max(1, self.hidden_size // 8)
        if self.ccgqa_num_kv_heads is None:
            self.ccgqa_num_kv_heads = max(1, self.num_heads // 4)

        # ---- Resolve GMA / Gaussian Mixture Attention defaults ----
        # routing_dim d_r defaults to the per-head dimension.
        if self.gma_routing_dim is None:
            self.gma_routing_dim = max(1, self.hidden_size // self.num_heads)
        else:
            self.gma_routing_dim = int(self.gma_routing_dim)
            if self.gma_routing_dim < 1:
                raise ValueError(
                    f"gma_routing_dim must be >= 1, got {self.gma_routing_dim}"
                )
        if self.gma_num_components < 1:
            raise ValueError(
                f"gma_num_components must be >= 1, got {self.gma_num_components}"
            )
        if self.gma_epsilon <= 0:
            raise ValueError(
                f"gma_epsilon must be > 0, got {self.gma_epsilon}"
            )
        if self.gma_sigma_eps <= 0:
            raise ValueError(
                f"gma_sigma_eps must be > 0, got {self.gma_sigma_eps}"
            )

        _VALID_POSITIONAL_ENCODINGS = {
            "rope", "hope", "nope", "alibi", "pape", "pape_efficient", "pape_ri",
            "sinusoidal_absolute", "sinusoidal_rotary", "learned_absolute", "none",
        }
        if self.positional_encoding is None:
            self.positional_encoding = "hope" if bool(self.use_hope) else "rope"
        else:
            self.positional_encoding = str(self.positional_encoding).lower()
            if self.positional_encoding not in _VALID_POSITIONAL_ENCODINGS:
                raise ValueError(
                    f"positional_encoding must be one of {sorted(_VALID_POSITIONAL_ENCODINGS)}, "
                    f"got {self.positional_encoding!r}"
                )

        self.use_hope = self.positional_encoding == "hope"

        if self.alibi_num_heads is None:
            self.alibi_num_heads = self.num_heads

        if self.pape_num_parabolas < 1:
            raise ValueError(f"pape_num_parabolas must be >= 1, got {self.pape_num_parabolas}")
        if self.pape_num_positions < 1:
            raise ValueError(f"pape_num_positions must be >= 1, got {self.pape_num_positions}")

        if self.mode not in {"encoder", "decoder"}:
            raise ValueError("mode must be one of {'encoder', 'decoder'}")

        if not 0.0 < float(self.mixture_of_depths_capacity_ratio) <= 1.0:
            raise ValueError("mixture_of_depths_capacity_ratio must be in the range (0, 1]")

        if float(self.mixture_of_depths_router_aux_loss_weight) < 0.0:
            raise ValueError("mixture_of_depths_router_aux_loss_weight must be >= 0")

        if not 0.0 < float(self.prms_partial_ratio) <= 1.0:
            raise ValueError("prms_partial_ratio must be in the range (0, 1]")

        if not 0.0 <= float(self.flashnorm_partial_ratio) <= 1.0:
            raise ValueError("flashnorm_partial_ratio must be in the range [0, 1]")

        # ---- Validate mHC (Manifold-Constrained Hyper-Connections) ----
        if int(self.mhc_expansion_rate) < 1:
            raise ValueError("mhc_expansion_rate must be >= 1")
        if int(self.mhc_sinkhorn_iters) < 1:
            raise ValueError("mhc_sinkhorn_iters must be >= 1")
        if float(self.mhc_gating_init) <= 0.0:
            raise ValueError("mhc_gating_init must be > 0")

        # ---- Validate Attention Residuals (arXiv:2603.15031) ----
        valid_residual_types = {"standard", "none", "full_attn", "block_attn"}
        rt = str(self.residual_type).lower()
        if rt not in valid_residual_types:
            raise ValueError(
                f"residual_type must be one of {sorted(valid_residual_types)}, "
                f"got {self.residual_type!r}"
            )
        self.residual_type = rt
        if int(self.block_attn_num_blocks) < 1:
            raise ValueError("block_attn_num_blocks must be >= 1")
        max_blocks = int(self.num_layers) * int(self.num_loops)
        if rt == "block_attn" and int(self.block_attn_num_blocks) > max_blocks:
            raise ValueError(
                f"block_attn_num_blocks ({self.block_attn_num_blocks}) cannot "
                f"exceed total logical depth ({max_blocks})"
            )
        if self.attnres_mhc_stream_mode not in {"independent", "joint"}:
            raise ValueError(
                "attnres_mhc_stream_mode must be 'independent' or 'joint', "
                f"got {self.attnres_mhc_stream_mode!r}"
            )

        # ---- Validate FFN activation ----
        ffn_act = str(self.ffn_activation).lower()
        if ffn_act not in ALL_ACTIVATIONS:
            raise ValueError(
                f"ffn_activation must be one of {sorted(ALL_ACTIVATIONS)}, "
                f"got {self.ffn_activation!r}"
            )
        self.ffn_activation = ffn_act
        if self.ffn_activation_config is not None:
            if not isinstance(self.ffn_activation_config, dict):
                raise ValueError(
                    "ffn_activation_config must be a mapping (dict) when provided"
                )
            _validate_ffn_activation_config(self.ffn_activation, self.ffn_activation_config)

        # ---- Validate Vision Transformer fields ----
        _VALID_VIT_PE = {
            "learned_1d", "none",
            "learned_absolute", "sinusoidal_absolute", "sinusoidal_rotary",
            "rope", "hope", "nope", "alibi", "pape", "pape_efficient", "pape_ri",
        }
        if self.pos_embedding_type not in _VALID_VIT_PE:
            raise ValueError(
                f"pos_embedding_type must be one of {sorted(_VALID_VIT_PE)}, "
                f"got {self.pos_embedding_type!r}"
            )
        if self.pos_embedding_type == "learned_1d":
            self.pos_embedding_type = "learned_absolute"
        if self.pooling_mode not in {"cls", "gap"}:
            raise ValueError(
                f"pooling_mode must be 'cls' or 'gap', got {self.pooling_mode!r}"
            )
        if self.mask_token_strategy not in {"bert", "mask_only", "random_only"}:
            raise ValueError(
                "mask_token_strategy must be 'bert', 'mask_only', or 'random_only', "
                f"got {self.mask_token_strategy!r}"
            )
        if self.prediction_target not in {
            "mean_color_3bit", "downsampled_3bit", "full_patch_l2"
        }:
            raise ValueError(
                "prediction_target must be 'mean_color_3bit', 'downsampled_3bit', "
                f"or 'full_patch_l2', got {self.prediction_target!r}"
            )
        if self.seg_head_type not in {"pixel", "eomt"}:
            raise ValueError(
                f"seg_head_type must be 'pixel' or 'eomt', got {self.seg_head_type!r}"
            )
        if int(self.patch_size) < 1:
            raise ValueError(f"patch_size must be >= 1, got {self.patch_size}")
        if int(self.image_height) % int(self.patch_size) != 0:
            raise ValueError(
                f"image_height ({self.image_height}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )
        if int(self.image_width) % int(self.patch_size) != 0:
            raise ValueError(
                f"image_width ({self.image_width}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )
        if not 0.0 < float(self.mask_ratio) < 1.0:
            raise ValueError(
                f"mask_ratio must be in the range (0, 1), got {self.mask_ratio}"
            )
        if int(self.num_classes) < 1:
            raise ValueError(f"num_classes must be >= 1, got {self.num_classes}")
        if int(self.num_seg_classes) < 1:
            raise ValueError(f"num_seg_classes must be >= 1, got {self.num_seg_classes}")
        if int(self.seg_num_queries) < 1:
            raise ValueError(f"seg_num_queries must be >= 1, got {self.seg_num_queries}")
        if int(self.seg_l2_blocks) < 1:
            raise ValueError(f"seg_l2_blocks must be >= 1, got {self.seg_l2_blocks}")

        # Classification-head validation (NLP encoder, DashAI Strategy A).
        if bool(self.classification_head) and self.num_labels is None:
            raise ValueError(
                "num_labels is required when classification_head=True"
            )
        if self.num_labels is not None and int(self.num_labels) < 1:
            raise ValueError(
                f"num_labels must be >= 1, got {self.num_labels}"
            )
        if str(self.encoder_pooling_mode) not in ("cls", "gap"):
            raise ValueError(
                f"encoder_pooling_mode must be 'cls' or 'gap', "
                f"got {self.encoder_pooling_mode}"
            )
