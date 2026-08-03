#!/usr/bin/env python3
"""TORMENTED-BERT-Frankenstein: The Ultimate Hybrid Transformer (SOTA 2026).

**TORMENTED** = **T**ernary **O**DE **R**etention **M**amba **E**xperts **N**eural **T**anh **E**ncoder **D**epth.

This module implements a mixed-architecture Transformer encoder/decoder that
integrates 20+ attention mixer families, Mixture-of-Experts (MoE) FFN routing,
BitNet b1.58 ternary weight quantization, factorized embeddings, looped depth
for parameter-efficient recursion, Mixture-of-Depths token routing, and Engram
conditional memory layers.

Core components:

* :class:`UltraConfig` — single-source-of-truth dataclass for all model
  hyperparameters.
* :class:`TormentedBertFrankenstein` — full hybrid encoder with looped depth,
  MoE, BitNet, factorized embeddings, and Mixture-of-Depths.
* :class:`TormentedBertMini` — simplified encoder variant preset for
  constrained GPU training.
* :class:`FrankensteinDecoder` — autoregressive causal decoder for LLM-style
  text generation.
* :class:`HybridLayer` — per-layer dispatcher that routes to the configured
  attention mixer and FFN (dense or MoE).
* :class:`FactorizedEmbedding` — reduced-dimension embedding with optional
  Conv1d pre-projection.

Supported attention mixer families (20+):

* ``standard_attn`` — scaled dot-product self-attention.
* ``sigmoid_attn`` — sigmoid-based linear attention.
* ``retnet`` / ``retnet_attn`` — multi-scale retention.
* ``ode`` — Neural ODE attention block (RK4 / Euler solvers).
* ``titan_attn`` — Titan memory-augmented attention.
* ``mamba`` — state-space model placeholder.
* Sparse family (7 variants): ``sparse_transformer_attn``, ``longformer_attn``,
  ``bigbird_attn``, ``sparsek_attn``, ``nsa_attn``, ``sparge_attn``,
  ``fasa_attn``.
* Gated family (8 variants): ``gla_attn``, ``deltanet_attn``,
  ``gated_deltanet_attn``, ``gated_deltanet2_attn``, ``hgrn2_attn``,
  ``fox_attn``, ``gated_softmax_attn``.
* ``engram_attn`` — Engram conditional memory via scalable lookup.
* ``gqa_attn`` — Grouped-Query Attention (GQA; Ainslie et al. 2023).
* Latent family (9 variants): ``mla_attn`` (Multi-Head Latent
  Attention + RoPE, arXiv:2506.09342), ``gqla_attn`` (Group-Query
  Latent Attention, arXiv:2605.15250), ``mlra_attn`` (Multi-Head
  Low-Rank Attention, arXiv:2603.02188), ``tucker_attn`` (Tucker
  Attention, arXiv:2603.30033), ``iha_attn`` (Interleaved Head
  Attention, arXiv:2602.21371), ``gta_attn`` (Grouped-head laTenT
  Attention, arXiv:2506.17286), ``mtla_attn`` (Multi-head Temporal
  Latent Attention, arXiv:2505.13544), ``cca_attn`` (Compressed
  Convolutional Attention, arXiv:2510.04476), ``ccgqa_attn``
  (Compressed Convolutional Grouped Query Attention, arXiv:2510.04476).
* Extended sparse family: ``msa_attn`` (MiniMax Sparse Attention,
  arXiv:2606.13392), ``sparda_attn`` (SparDA, arXiv:2606.04511).
* Extended gated family: ``kda_attn`` (Kimi Delta Attention,
  arXiv:2510.26692).

Training-free policy: ``fasa_attn`` and ``sparge_attn`` raise a RuntimeError
when called in training mode; they are eval/inference-only blocks.

Hardware Target: Dual Xeon E5-2680v4 + Nvidia Tesla P40 (24GB).
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention.common import BitLinear
from .norm import FlashNormBitLinear, FlashNormLinear, get_norm
from .activation_function import (
    ALL_ACTIVATIONS,
    GLU_VARIANTS,
    get_activation,
    make_gated_ffn,
)
from .attention.engram import EngramLayer
from .attention.grouped_query_attention import GroupedQueryAttention
from .attention.gated import (
    DeltaNetAttention,
    ForgettingAttention,
    GatedDeltaNet2Attention,
    GatedDeltaNetAttention,
    GatedLinearAttention,
    GatedSoftmaxAttention,
    HGRN2Attention,
    RetNetAttention,
)
from .attention.ode import ODEAttentionBlock
from .attention.retnet import MultiScaleRetention
from .attention.sigmoid import SigmoidAttention
from .attention.sparse import (
    BigBirdAttention,
    FASAAttention,
    LongformerAttention,
    MSAAttention,
    NSAAttention,
    SparseKAttention,
    SparseTransformerAttention,
    SpargeAttention,
    SparDAAttention,
)
from .attention.standard import StandardAttention
from .attention.titan import TitanAttention
from .attention.gated import KDAAttention
from .attention.latent import (
    CCAAttention,
    CCGQAAttention,
    GTAAttention,
    GQLAAttention,
    IHAAttention,
    MLAAttention,
    MLRAAttention,
    MTLAAttention,
    TuckerAttention,
)
from .embeddings import FactorizedEmbedding

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
class UltraConfig:
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
            ``"hope"`` or ``"rope"``. If None, inferred from ``use_hope``.
            Default: None.
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
            ``model_class=frankesteindecoder`` preset forces ``mode=decoder``
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
        ValueError: If ``positional_encoding`` is not ``"hope"`` or
            ``"rope"``.
        ValueError: If ``mode`` is not ``"encoder"`` or ``"decoder"``.
        ValueError: If ``mixture_of_depths_capacity_ratio`` is not in (0, 1].
        ValueError: If ``mixture_of_depths_router_aux_loss_weight`` is < 0.
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
    positional_encoding: Optional[str] = None
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

        if self.positional_encoding is None:
            self.positional_encoding = "hope" if bool(self.use_hope) else "rope"
        else:
            self.positional_encoding = str(self.positional_encoding).lower()
            if self.positional_encoding not in {"hope", "rope"}:
                raise ValueError("positional_encoding must be one of {'hope', 'rope'}")

        self.use_hope = self.positional_encoding == "hope"

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


class HybridLayer(nn.Module):
    """Per-layer dispatcher: attention mixer + FFN (dense or MoE) + optional Mixture-of-Depths.

    Each :class:`HybridLayer` instantiates the attention mixer specified by
    ``layer_type``, a normalization layer, and a feed-forward block. The FFN
    can be a standard dense MLP or a Mixture-of-Experts (MoE) block with
    top-k expert routing. When ``use_mixture_of_depths`` is enabled, only the
    top-capacity tokens are passed through the full layer; remaining tokens
    are scattered back unchanged.

    Training-free layers (``fasa_attn``, ``sparge_attn``) raise a
    RuntimeError if called in training mode.

    Attributes:
        layer_type: The mixer type string (e.g. ``"retnet"``, ``"ode"``).
        norm1: Pre-attention normalization layer.
        norm2: Pre-FFN normalization layer.
        mixer: The instantiated attention mixer module.
        router: MoE router linear layer (None if dense FFN).
        experts: ModuleList of MoE expert FFNs (None if dense FFN).
        top_k: Number of experts activated per token in MoE mode.
        ffn: Dense FFN sequential block (None if MoE mode).
        depth_router: Mixture-of-Depths token router (None if disabled).
        use_moe: Whether MoE FFN is active.
        use_mixture_of_depths: Whether Mixture-of-Depths routing is active.
        mixture_of_depths_capacity_ratio: Fraction of tokens selected.
        mixture_of_depths_router_aux_loss_weight: Aux loss weight.
        last_mixture_of_depths_aux_loss: Aux loss from the most recent
            forward pass (None if MoD disabled).
        last_mixture_of_depths_selected_fraction: Fraction of tokens
            selected in the most recent forward pass.
        last_mixture_of_depths_capacity: Token capacity used in the most
            recent forward pass.
    """

    TRAINING_FREE_LAYERS = {"fasa_attn", "sparge_attn"}

    def __init__(self, config, layer_type):
        """Initialize a hybrid layer for the given mixer type.

        Args:
            config: :class:`UltraConfig` instance with model hyperparameters.
            layer_type: String identifying the attention mixer. Must be one
                of the keys in the internal mixer registry or ``"mamba"``.

        Raises:
            ValueError: If ``layer_type`` is not recognized.
        """
        super().__init__()
        self.layer_type = layer_type
        self.norm1 = get_norm(config)
        self.use_moe = bool(config.use_moe)
        self.use_mixture_of_depths = bool(getattr(config, "use_mixture_of_depths", False))
        self.mixture_of_depths_capacity_ratio = float(
            getattr(config, "mixture_of_depths_capacity_ratio", 1.0)
        )
        self.mixture_of_depths_router_aux_loss_weight = float(
            getattr(config, "mixture_of_depths_router_aux_loss_weight", 0.0)
        )
        self.last_mixture_of_depths_aux_loss: Optional[torch.Tensor] = None
        self.last_mixture_of_depths_selected_fraction: float = 1.0
        self.last_mixture_of_depths_capacity: Optional[int] = None

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        router_cls = BitLinear if (config.use_bitnet and getattr(config, "bitnet_routers", False)) else nn.Linear

        mixer_registry = {
            "ode": ODEAttentionBlock,
            "retnet": MultiScaleRetention,
            "retnet_attn": RetNetAttention,
            "titan_attn": TitanAttention,
            "standard_attn": StandardAttention,
            "sigmoid_attn": SigmoidAttention,
            "sparse_transformer_attn": SparseTransformerAttention,
            "longformer_attn": LongformerAttention,
            "bigbird_attn": BigBirdAttention,
            "sparsek_attn": SparseKAttention,
            "nsa_attn": NSAAttention,
            "sparge_attn": SpargeAttention,
            "fasa_attn": FASAAttention,
            "gla_attn": GatedLinearAttention,
            "deltanet_attn": DeltaNetAttention,
            "gated_deltanet_attn": GatedDeltaNetAttention,
            "gated_deltanet2_attn": GatedDeltaNet2Attention,
            "hgrn2_attn": HGRN2Attention,
            "fox_attn": ForgettingAttention,
            "gated_softmax_attn": GatedSoftmaxAttention,
            "kda_attn": KDAAttention,
            "engram_attn": EngramLayer,
            "gqa_attn": GroupedQueryAttention,
            "mla_attn": MLAAttention,
            "gqla_attn": GQLAAttention,
            "mlra_attn": MLRAAttention,
            "tucker_attn": TuckerAttention,
            "iha_attn": IHAAttention,
            "gta_attn": GTAAttention,
            "mtla_attn": MTLAAttention,
            "cca_attn": CCAAttention,
            "ccgqa_attn": CCGQAAttention,
            "msa_attn": MSAAttention,
            "sparda_attn": SparDAAttention,
        }

        if layer_type == "mamba":
            self.mixer = proj_cls(config.hidden_size, config.hidden_size)
        elif layer_type in mixer_registry:
            self.mixer = mixer_registry[layer_type](config)
        else:
            supported_layers = sorted(list(mixer_registry.keys()) + ["mamba"])
            raise ValueError(
                f"Unknown layer_type '{layer_type}'. Supported values: {supported_layers}"
            )

        self.norm2 = get_norm(config)
        is_glu = config.ffn_activation in GLU_VARIANTS

        # FlashNorm fusion (Prop. 2 of arXiv:2407.09577): when norm_type is
        # flash_norm and the FFN is the simple elementwise non-MoE path,
        # fold norm2 into the FFN's first linear projection. The router in
        # MoE needs a normalized input, and GLU FFNs have two parallel
        # projections (gate + up) that would each need their own fusion;
        # those cases keep norm2 as a standalone FlashNorm.
        fuse_ffn_input = (
            config.norm_type == "flash_norm"
            and not is_glu
            and not self.use_moe
        )
        flashnorm_partial_ratio = float(getattr(config, "flashnorm_partial_ratio", 0.0))

        def _build_ffn():
            """Return a fresh FFN block (gated or elementwise activation)."""
            if is_glu:
                # Gated FFN units own their projections; use the same
                # proj_cls (BitLinear when BitNet is on) for BitNet parity.
                return make_gated_ffn(
                    config.ffn_activation,
                    hidden_size=config.hidden_size,
                    intermediate_size=config.ffn_hidden_size,
                    bias=False,
                    dropout=float(getattr(config, "dropout", 0.0) or 0.0),
                    proj_factory=proj_cls,
                )
            if fuse_ffn_input:
                # Fused first projection: FlashNorm + Linear in one module.
                # The bias-free path applies Prop. 2 (deferred scalar RMS).
                if config.use_bitnet:
                    first_proj = FlashNormBitLinear(
                        config.hidden_size,
                        config.ffn_hidden_size,
                        bias=False,
                        partial_ratio=flashnorm_partial_ratio,
                    )
                else:
                    first_proj = FlashNormLinear(
                        config.hidden_size,
                        config.ffn_hidden_size,
                        bias=False,
                        partial_ratio=flashnorm_partial_ratio,
                    )
            else:
                first_proj = proj_cls(config.hidden_size, config.ffn_hidden_size)
            return nn.Sequential(
                first_proj,
                get_activation(config, dim=config.ffn_hidden_size),
                proj_cls(config.ffn_hidden_size, config.hidden_size),
            )

        if fuse_ffn_input:
            # norm2 is absorbed into the FFN's first projection.
            self.norm2 = nn.Identity()

        if self.use_moe:
            self.router = router_cls(config.hidden_size, config.num_experts, bias=False)
            # One FFN per expert (fresh learnable activations per expert).
            self.experts = nn.ModuleList([_build_ffn() for _ in range(config.num_experts)])
            self.top_k = config.top_k_experts
        else:
            self.router = None
            self.experts = None
            self.top_k = 0
            self.ffn = _build_ffn()
        self.depth_router = (
            router_cls(config.hidden_size, 1, bias=False) if self.use_mixture_of_depths else None
        )

    def _forward_dense(
        self,
        x,
        logical_layer_idx: Optional[int] = None,
        input_ids: Optional[torch.Tensor] = None,
    ):
        """Run the full attention + FFN path for all tokens (no MoD routing).

        Args:
            x: Input tensor of shape ``(B, S, hidden_size)``.
            logical_layer_idx: Global logical layer index (0-based across
                loops). Passed to mixers that need positional awareness.
            input_ids: Original token IDs, required by Engram layers.

        Returns:
            Output tensor of shape ``(B, S, hidden_size)``.

        Raises:
            ValueError: If the layer is training-free and called in training
                mode.
        """
        residual = x
        x = self.norm1(x)

        if self.training and self.layer_type in self.TRAINING_FREE_LAYERS:
            raise ValueError(
                f"Layer '{self.layer_type}' is training-free and only supported in eval/inference mode."
            )

        if self.layer_type == "mamba":
            x = x + self.mixer(x)
        elif self.layer_type in {"ode", "retnet"}:
            x = self.mixer(x)
        elif self.layer_type == "engram_attn":
            x = self.mixer(x, input_ids=input_ids, logical_layer_idx=logical_layer_idx)
        else:
            x = self.mixer(x, logical_layer_idx=logical_layer_idx)

        x = residual + x

        residual = x
        x = self.norm2(x)

        if self.use_moe:
            logits = self.router(x)
            weights, indices = torch.topk(F.softmax(logits, dim=-1), self.top_k, dim=-1)

            batch_size, seq_len, dim = x.shape
            flat_x = x.view(-1, dim)
            out = torch.zeros_like(flat_x)

            for k in range(self.top_k):
                expert_indices = indices[:, :, k].flatten()
                expert_weights = weights[:, :, k].flatten().unsqueeze(1)

                for i, expert in enumerate(self.experts):
                    mask = expert_indices == i
                    if mask.any():
                        selected_x = flat_x[mask]
                        expert_out = expert(selected_x)
                        out[mask] += expert_out * expert_weights[mask]

            x = residual + out.view(batch_size, seq_len, dim)
            return x

        x = residual + self.ffn(x)
        return x

    def _mixture_of_depths_capacity(self, seq_len: int) -> int:
        """Compute the token capacity for Mixture-of-Depths routing.

        Args:
            seq_len: Sequence length of the current batch.

        Returns:
            Integer token capacity, at least 1.
        """
        return max(1, int(math.ceil(seq_len * self.mixture_of_depths_capacity_ratio)))

    def forward(
        self,
        x,
        logical_layer_idx: Optional[int] = None,
        input_ids: Optional[torch.Tensor] = None,
    ):
        """Forward pass with optional Mixture-of-Depths token routing.

        When MoD is disabled, delegates directly to :meth:`_forward_dense`.
        When MoD is enabled, computes per-token router scores, selects the
        top-capacity tokens, runs the dense path on those tokens only, and
        scatters the updated tokens back into the original sequence. An
        auxiliary load-balancing loss is computed and stored in
        ``last_mixture_of_depths_aux_loss``.

        Args:
            x: Input tensor of shape ``(B, S, hidden_size)``.
            logical_layer_idx: Global logical layer index (0-based across
                loops).
            input_ids: Original token IDs, required by Engram layers.

        Returns:
            Output tensor of shape ``(B, S, hidden_size)``.

        Raises:
            ValueError: If MoD is active and the sequence length is 0.
        """
        if not self.use_mixture_of_depths:
            self.last_mixture_of_depths_aux_loss = None
            self.last_mixture_of_depths_selected_fraction = 1.0
            self.last_mixture_of_depths_capacity = x.size(1)
            return self._forward_dense(
                x,
                logical_layer_idx=logical_layer_idx,
                input_ids=input_ids,
            )

        batch_size, seq_len, hidden_size = x.shape
        if seq_len == 0:
            raise ValueError("Mixture-of-Depths requires a non-empty token sequence")
        capacity = self._mixture_of_depths_capacity(seq_len)
        self.last_mixture_of_depths_capacity = capacity
        self.last_mixture_of_depths_selected_fraction = capacity / seq_len

        router_logits = self.depth_router(x).squeeze(-1)
        router_probs = torch.sigmoid(router_logits)
        self.last_mixture_of_depths_aux_loss = (
            (router_probs.mean(dim=1) - self.mixture_of_depths_capacity_ratio).pow(2).mean()
        )

        if capacity >= seq_len:
            return self._forward_dense(x, logical_layer_idx=logical_layer_idx)

        selected_indices = torch.topk(router_logits, k=capacity, dim=1).indices
        selected_indices, _ = torch.sort(selected_indices, dim=1)
        gather_index = selected_indices.unsqueeze(-1).expand(batch_size, capacity, hidden_size)
        selected_tokens = torch.gather(x, dim=1, index=gather_index)
        selected_input_ids = None
        if input_ids is not None:
            selected_input_ids = torch.gather(input_ids, dim=1, index=selected_indices)
        updated_tokens = self._forward_dense(
            selected_tokens,
            logical_layer_idx=logical_layer_idx,
            input_ids=selected_input_ids,
        )
        return torch.scatter(x, dim=1, index=gather_index, src=updated_tokens)


class TormentedBertFrankenstein(nn.Module):
    """TORMENTED-BERT-Frankenstein: Hybrid mixed-architecture Transformer encoder.

    This is the flagship model. It stacks ``num_layers`` :class:`HybridLayer`
    blocks, each configured by ``layer_pattern``, and repeats the entire
    stack ``num_loops`` times (looped depth). The logical depth is
    ``num_layers * num_loops``.

    Key architectural features:

    * **17+ attention mixer families** dispatched per-layer via
      :class:`HybridLayer`.
    * **Looped depth**: the physical layer stack is iterated ``num_loops``
      times, sharing parameters across loops for parameter-efficient deep
      computation.
    * **Mixture-of-Experts (MoE) FFN**: per-token top-k expert routing with
      weighted expert outputs.
    * **BitNet b1.58**: ternary weight quantization via :class:`BitLinear`
      when ``use_bitnet`` is True.
    * **Factorized embeddings**: reduced-dimension embedding lookup +
      projection via :class:`FactorizedEmbedding`.
    * **Mixture-of-Depths**: per-layer token routing where only the
      top-capacity tokens are updated; auxiliary load-balancing loss is
      accumulated and exposed via ``last_auxiliary_losses``.
    * **Normalization**: ``layer_norm``, ``dynamic_tanh`` (DyT), or
      ``derf`` (Dynamic Erf).
    * **Positional encoding**: RoPE or HoPE, applied inside attention
      mixers.

    Attributes:
        config: The :class:`UltraConfig` used to build the model.
        emb: Token embedding layer (:class:`FactorizedEmbedding` or
            ``nn.Embedding``).
        dropout: Embedding dropout layer.
        layers: ModuleList of ``num_layers`` :class:`HybridLayer` blocks.
        final_norm: Final normalization before the output head.
        head: Output projection to vocabulary logits (Linear or BitLinear).
        last_auxiliary_losses: Dict of auxiliary losses from the most recent
            forward pass (e.g. ``"mixture_of_depths_router_loss"``).
        last_mixture_of_depths_stats: Dict of MoD statistics from the most
            recent forward pass (``"average_selected_fraction"``,
            ``"raw_router_aux_loss"``).
    """

    def __init__(self, config):
        """Build the Frankenstein encoder from an UltraConfig.

        Args:
            config: :class:`UltraConfig` instance with all model
                hyperparameters.
        """
        super().__init__()
        self.config = config
        self.last_auxiliary_losses = {}
        self.last_mixture_of_depths_stats = {}

        if config.use_factorized_embedding:
            self.emb = FactorizedEmbedding(config)
        else:
            self.emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [
                HybridLayer(config, layer_type=config.layer_pattern[i % len(config.layer_pattern)])
                for i in range(config.num_layers)
            ]
        )

        self.final_norm = get_norm(config)
        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.head = proj_cls(config.hidden_size, config.vocab_size)

    def forward(self, input_ids):
        """Run the full looped-depth encoder forward pass.

        Iterates the physical layer stack ``num_loops`` times, tracking a
        global ``logical_layer_idx``. Accumulates Mixture-of-Depths auxiliary
        losses across all layers and stores them in
        ``last_auxiliary_losses``.

        Args:
            input_ids: Integer token indices of shape ``(B, S)``.

        Returns:
            Logits tensor of shape ``(B, S, vocab_size)``.
        """
        x = self.emb(input_ids)
        x = self.dropout(x)

        logical_layer_idx = 0
        mixture_of_depths_aux_losses = []
        mixture_of_depths_selected_fractions = []
        for _ in range(self.config.num_loops):
            for layer in self.layers:
                x = layer(x, logical_layer_idx=logical_layer_idx, input_ids=input_ids)
                if layer.use_mixture_of_depths and layer.last_mixture_of_depths_aux_loss is not None:
                    mixture_of_depths_aux_losses.append(layer.last_mixture_of_depths_aux_loss)
                    mixture_of_depths_selected_fractions.append(
                        layer.last_mixture_of_depths_selected_fraction
                    )
                logical_layer_idx += 1

        x = self.final_norm(x)
        if mixture_of_depths_aux_losses:
            raw_aux_loss = torch.stack(mixture_of_depths_aux_losses).mean()
            weighted_aux_loss = (
                raw_aux_loss * float(self.config.mixture_of_depths_router_aux_loss_weight)
            )
            self.last_auxiliary_losses = {
                "mixture_of_depths_router_loss": weighted_aux_loss,
            }
            self.last_mixture_of_depths_stats = {
                "average_selected_fraction": sum(mixture_of_depths_selected_fractions)
                / len(mixture_of_depths_selected_fractions),
                "raw_router_aux_loss": float(raw_aux_loss.detach().item()),
            }
        else:
            self.last_auxiliary_losses = {}
            self.last_mixture_of_depths_stats = {}
        return self.head(x)


class TormentedBertMini(nn.Module):
    """Simplified encoder variant preset for stable training on constrained GPUs.

    Wraps a :class:`TormentedBertFrankenstein` backbone with a compact preset
    configuration: ``hidden_size=384``, ``num_layers=6``, ``num_loops=2``,
    ``num_heads=6``, ``num_experts=4``, ``norm_type="derf"``,
    ``use_factorized_embedding=True``, and a stable layer pattern of
    ``["retnet", "titan_attn", "retnet", "mamba", "titan_attn", "ode"]``.

    Factorized embeddings are forced on even if the provided config disables
    them.

    Attributes:
        config: The :class:`UltraConfig` (built from preset or user-provided).
        backbone: The underlying :class:`TormentedBertFrankenstein` model.
        last_auxiliary_losses: Mirrored from the backbone after each forward
            pass.
        last_mixture_of_depths_stats: Mirrored from the backbone after each
            forward pass.
    """

    @staticmethod
    def build_mini_config(vocab_size: int = 50_000, use_bitnet: bool = True) -> UltraConfig:
        """Build the default Mini preset configuration.

        Args:
            vocab_size: Vocabulary size. Default: 50000.
            use_bitnet: Whether to use BitNet ternary quantization.
                Default: True.

        Returns:
            A pre-configured :class:`UltraConfig` with compact dimensions.
        """
        stable_layer_pattern = [
            "retnet",
            "titan_attn",
            "retnet",
            "mamba",
            "titan_attn",
            "ode",
        ]
        return UltraConfig(
            vocab_size=vocab_size,
            hidden_size=384,
            num_layers=6,
            num_loops=2,
            num_heads=6,
            retention_heads=6,
            num_experts=4,
            top_k_experts=2,
            dropout=0.1,
            ode_solver="rk4",
            ode_steps=2,
            use_bitnet=use_bitnet,
            norm_type="derf",
            layer_pattern=stable_layer_pattern,
            use_factorized_embedding=True,
            factorized_embedding_dim=128,
            use_embedding_conv=True,
        )

    def __init__(self, config: Optional[UltraConfig] = None):
        """Initialize the Mini model.

        Args:
            config: Optional :class:`UltraConfig`. If None, the default Mini
                preset is used. Factorized embeddings are forced on.
        """
        super().__init__()
        self.config = config or self.build_mini_config()
        self.last_auxiliary_losses = {}
        self.last_mixture_of_depths_stats = {}

        if self.config.use_factorized_embedding is False:
            self.config.use_factorized_embedding = True

        self.backbone = TormentedBertFrankenstein(self.config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Mini backbone.

        Args:
            input_ids: Integer token indices of shape ``(B, S)``.

        Returns:
            Logits tensor of shape ``(B, S, vocab_size)``.
        """
        output = self.backbone(input_ids)
        self.last_auxiliary_losses = dict(getattr(self.backbone, "last_auxiliary_losses", {}))
        self.last_mixture_of_depths_stats = dict(
            getattr(self.backbone, "last_mixture_of_depths_stats", {})
        )
        return output


class FrankensteinDecoder(nn.Module):
    """Autoregressive causal decoder variant for LLM-style text generation.

    Wraps a :class:`TormentedBertFrankenstein` backbone with ``mode='decoder'``
    so every attention layer applies causal (autoregressive) masking. Supports
    the same hybrid architecture features as the encoder: 17+ mixer families,
    MoE, BitNet, factorized embeddings, looped depth, Mixture-of-Depths, and
    Engram memory.

    The ``model_class=frankesteindecoder`` preset forces ``mode=decoder`` at
    runtime, overriding any user-provided mode.

    Attributes:
        config: The :class:`UltraConfig` (built from preset or user-provided).
        backbone: The underlying :class:`TormentedBertFrankenstein` model.
        last_auxiliary_losses: Mirrored from the backbone after each forward
            pass.
        last_mixture_of_depths_stats: Mirrored from the backbone after each
            forward pass.
    """

    @staticmethod
    def build_decoder_config(
        vocab_size: int = 50_000,
        hidden_size: int = 2048,
        num_layers: int = 12,
        num_loops: int = 1,
        use_bitnet: bool = True,
        layer_pattern: Optional[List[str]] = None,
    ) -> UltraConfig:
        """Build the default decoder preset configuration.

        Args:
            vocab_size: Vocabulary size. Default: 50000.
            hidden_size: Hidden state dimensionality. Default: 2048.
            num_layers: Number of physical HybridLayer blocks. Default: 12.
            num_loops: Number of loop iterations. Default: 1.
            use_bitnet: Whether to use BitNet ternary quantization.
                Default: True.
            layer_pattern: Optional custom layer pattern. If None, defaults
                to ``["titan_attn", "retnet", "titan_attn", "mamba"] * 3``.

        Returns:
            A pre-configured :class:`UltraConfig` with ``mode='decoder'``.
        """
        if layer_pattern is None:
            layer_pattern = [
                "titan_attn",
                "retnet",
                "titan_attn",
                "mamba",
            ] * 3
        return UltraConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_loops=num_loops,
            num_heads=16,
            retention_heads=8,
            num_experts=8,
            top_k_experts=2,
            dropout=0.1,
            ode_solver="rk4",
            ode_steps=2,
            use_bitnet=use_bitnet,
            norm_type="dynamic_tanh",
            layer_pattern=layer_pattern,
            use_factorized_embedding=False,
            mode="decoder",
        )

    def __init__(self, config: Optional[UltraConfig] = None):
        """Initialize the decoder model.

        Args:
            config: Optional :class:`UltraConfig`. If None, the default
                decoder preset is used. ``mode`` is forced to ``"decoder"``
                if not already set.
        """
        super().__init__()
        self.config = config or self.build_decoder_config()
        self.last_auxiliary_losses = {}
        self.last_mixture_of_depths_stats = {}

        if self.config.mode != "decoder":
            self.config.mode = "decoder"

        self.backbone = TormentedBertFrankenstein(self.config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder backbone (causal masking).

        Args:
            input_ids: Integer token indices of shape ``(B, S)``.

        Returns:
            Logits tensor of shape ``(B, S, vocab_size)``.
        """
        output = self.backbone(input_ids)
        self.last_auxiliary_losses = dict(getattr(self.backbone, "last_auxiliary_losses", {}))
        self.last_mixture_of_depths_stats = dict(
            getattr(self.backbone, "last_mixture_of_depths_stats", {})
        )
        return output

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive token generation with top-k sampling.

        Runs the decoder forward pass repeatedly, sampling one token at a
        time from the top-k filtered softmax distribution. Runs under
        ``torch.inference_mode()`` for efficiency.

        Args:
            input_ids: Prompt token indices of shape ``(B, S_prompt)``.
            max_new_tokens: Maximum number of tokens to generate.
                Default: 128.
            temperature: Softmax temperature for sampling. Lower values
                make output more deterministic. Default: 1.0.
            top_k: Number of highest-probability tokens to keep for
                sampling. Set to 0 to disable top-k filtering.
                Default: 50.

        Returns:
            Tensor of shape ``(B, S_prompt + max_new_tokens)`` containing
            the prompt followed by generated tokens.
        """
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# ==================== PRUEBA DE ESTRES ====================
if __name__ == "__main__":
    config = UltraConfig(
        hidden_size=1536,
        num_layers=16,
        num_loops=2,
        ode_solver="rk4",
        ode_steps=2,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n⚡ TORMENTED-BERT-Frankenstein INITIALIZING ⚡")
    model = TormentedBertFrankenstein(config).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model Params: {params / 1e6:.2f}M")
    print(f"Architecture Pattern: {config.layer_pattern}")
    print("Weights: Ternary (BitNet b1.58)")
    print("Dynamics: Neural ODE (RK4)")

    x = torch.randint(0, 50000, (4, 512), device=device)

    print("\n[...] Running Forward Pass with ODE Dynamics & RetNet...")
    y = model(x)
    print(f"Output Shape: {y.shape}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    loss = y.mean()
    print("[...] Running Backward Pass (Training ODE through time)...")
    loss.backward()
    print("Gradients computed successfully.")
