#!/usr/bin/env python3
"""Nested, schema-aligned model configuration for all Frankenstein variants.

The runtime shape mirrors the YAML ``model:`` block one-to-one — there is no
nested-to-flat flattening step. Structural blocks are typed dataclasses;
per-component knobs live in per-family mappings (``attention``,
``positional_encoding_parameters``) and are bound through the component
registry (:mod:`src.registry`) by the family factories.

Attribute access is nested only: structural blocks are dataclasses and
per-component knobs live in their mappings, bound through the component
registry (:mod:`src.registry`). Positional-encoding knobs keep small
read-only properties (``config.rope_base`` …) that read the nested mapping.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "FrankensteinModelConfig",
    "DimsConfig",
    "NormConfig",
    "QKNormConfig",
    "EmbeddingConfig",
    "MoEConfig",
    "BitnetConfig",
    "MixtureOfDepthsConfig",
    "SsmaxConfig",
    "NumericsConfig",
    "FfnConfig",
    "ClassificationHeadConfig",
    "MhcConfig",
    "ResidualsConfig",
    "ImageConfig",
    "AudioConfig",
    "merge_model_blocks",
]

_RAF_VERSIONS = {"A", "B", "C", "D", "N"}
_RAF_APPROX_FUNCS = {
    "gelu", "relu", "leaky_relu", "leaky_relu_0.1", "sigmoid", "tanh",
    "swish", "silu", "identity",
}
_FFN_ACTIVATION_CONFIG_KEYS = {
    "prelu_init", "elu_alpha", "celu_alpha", "swish_beta", "leaky_relu_slope",
    "pelu_alpha", "mpelu_alpha", "mpelu_beta", "felu_alpha", "eelu_alpha",
    "eelu_beta", "pdelu_alpha", "preu_alpha", "preu_beta", "softexp_alpha",
    "raf_degrees", "raf_version", "raf_approx_func", "raf_trainable",
    "raf_input_scaling", "maxout_pieces", "situ_beta_gate", "situ_beta_up",
}


def _validate_ffn_activation_config(activation: str, cfg: Dict[str, Any]) -> None:
    """Validate the nested ``ffn.activation_config`` mapping.

    Args:
        activation: The lower-cased ``ffn.activation`` name.
        cfg: The nested config mapping.

    Raises:
        ValueError: If an unknown key is present, a value has the wrong type,
            or a RAF-specific constraint is violated.
    """
    unknown = set(cfg) - _FFN_ACTIVATION_CONFIG_KEYS
    if unknown:
        raise ValueError(
            f"Unknown ffn_activation_config keys: {sorted(unknown)}. "
            f"Allowed: {sorted(_FFN_ACTIVATION_CONFIG_KEYS)}"
        )
    for bk in ("raf_trainable", "raf_input_scaling"):
        if bk in cfg and not isinstance(cfg[bk], bool):
            raise ValueError(f"ffn_activation_config.{bk} must be a boolean")
    for fk in (
        "prelu_init", "elu_alpha", "celu_alpha", "swish_beta", "leaky_relu_slope",
        "pelu_alpha", "mpelu_alpha", "mpelu_beta", "felu_alpha", "eelu_alpha",
        "eelu_beta", "pdelu_alpha", "preu_alpha", "preu_beta", "softexp_alpha",
        "situ_beta_gate", "situ_beta_up",
    ):
        if fk in cfg and not isinstance(cfg[fk], (int, float)):
            raise ValueError(f"ffn_activation_config.{fk} must be a number")
        if fk in cfg and cfg[fk] <= 0:
            raise ValueError(f"ffn_activation_config.{fk} must be > 0")
    if "maxout_pieces" in cfg:
        if not isinstance(cfg["maxout_pieces"], int) or cfg["maxout_pieces"] < 1:
            raise ValueError("ffn_activation_config.maxout_pieces must be an int >= 1")
    if "raf_degrees" in cfg:
        degrees = cfg["raf_degrees"]
        if (
            not isinstance(degrees, (list, tuple))
            or len(degrees) != 2
            or not all(isinstance(v, int) and v >= 1 for v in degrees)
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


# ---------------------------------------------------------------------------
# structural blocks
# ---------------------------------------------------------------------------


@dataclass
class DimsConfig:
    """Core dimensions, dropout and the per-layer mixer pattern."""

    vocab_size: int = 50000
    hidden_size: int = 2048
    num_layers: int = 12
    num_loops: int = 2
    num_heads: int = 16
    num_kv_heads: int = 1
    dropout: float = 0.1
    layer_pattern: List[str] = field(
        default_factory=lambda: ["retnet", "ode", "mamba", "titan"] * 3
    )
    mode: str = "encoder"

    @property
    def head_dim(self) -> int:
        """Per-head dimension (``hidden_size // num_heads``)."""
        return self.hidden_size // self.num_heads


@dataclass
class QKNormConfig:
    """Query–key normalization / logit-control block (``norm.qk_norm``)."""

    enabled: bool = False
    method: str = "rms_norm"
    clip_tau: float = 100.0
    clip_alpha: float = 0.5
    quack_tau: float = 1.0
    eps: float = 1e-6


@dataclass
class NormConfig:
    """Normalization block (``model.norm``)."""

    type: str = "dynamic_tanh"
    partial_ratio: float = 0.0625
    flashnorm_partial_ratio: float = 0.0
    eps: float = 1e-6
    qk_norm: QKNormConfig = field(default_factory=QKNormConfig)


@dataclass
class FactorizedEmbeddingConfig:
    """Factorized (low-rank) embedding knobs."""

    enabled: bool = False
    dim: int = 128


@dataclass
class DynamicEmbeddingConfig:
    """Dynamic (hash) embedding knobs."""

    enabled: bool = False
    ngram_max: int = 3
    bucket: int = 1_000_000_007
    seed: int = 42
    precompute: bool = True


@dataclass
class EmbeddingConvConfig:
    """Causal depth-wise conv frontend on the embedding stream."""

    enabled: bool = True
    kernel: int = 3


@dataclass
class EmbeddingConfig:
    """Token-embedding block (``model.embedding``)."""

    type: str = "standard"
    factorized: FactorizedEmbeddingConfig = field(default_factory=FactorizedEmbeddingConfig)
    dynamic: DynamicEmbeddingConfig = field(default_factory=DynamicEmbeddingConfig)
    conv: EmbeddingConvConfig = field(default_factory=EmbeddingConvConfig)


@dataclass
class MoERouterConfig:
    """MoE router block (``moe.router``)."""

    scoring: str = "softmax"
    pre_softmax: bool = False
    renormalize: bool = True
    routed_scaling_factor: float = 1.0
    dtype: str = "float32"
    jitter_eps: float = 0.0
    z_loss_coeff: float = 0.0


@dataclass
class MoERoutingConfig:
    """MoE routing block (``moe.routing``)."""

    type: str = "token_choice"
    capacity_factor: float = 0.0
    hash_layers: int = 0
    hash_table: str = "random"
    hash_seed: int = 42


@dataclass
class MoEBalancingConfig:
    """MoE balancing block (``moe.balancing``)."""

    aux_loss: str = "batch"
    aux_loss_coeff: float = 0.01
    bias_update: str = "none"
    bias_update_rate: float = 0.001


@dataclass
class MoESharedConfig:
    """Shared-expert block (``moe.shared``)."""

    count: int = 0
    hidden_size: Optional[int] = None
    gate: str = "none"


@dataclass
class MoELatentConfig:
    """Latent MoE block (``moe.latent``)."""

    enabled: bool = False
    dim: Optional[int] = None
    norm_before_up_proj: bool = False


@dataclass
class MoELayoutConfig:
    """Per-layer MoE layout (``moe.layout``)."""

    dense_layers: int = 0
    every_n_layers: int = 1


@dataclass
class MoEExpertsConfig:
    """Expert block (``moe.experts``)."""

    activation_clamp: float = 0.0


@dataclass
class MoEConfig:
    """Mixture-of-Experts block (``model.moe``)."""

    enabled: bool = True
    num_experts: int = 8
    top_k: int = 2
    expert_hidden_size: Optional[int] = None
    router: MoERouterConfig = field(default_factory=MoERouterConfig)
    routing: MoERoutingConfig = field(default_factory=MoERoutingConfig)
    balancing: MoEBalancingConfig = field(default_factory=MoEBalancingConfig)
    shared: MoESharedConfig = field(default_factory=MoESharedConfig)
    latent: MoELatentConfig = field(default_factory=MoELatentConfig)
    layout: MoELayoutConfig = field(default_factory=MoELayoutConfig)
    experts: MoEExpertsConfig = field(default_factory=MoEExpertsConfig)


@dataclass
class BitnetConfig:
    """BitNet 1.58-bit quantization block (``model.bitnet``)."""

    enabled: bool = True
    routers: bool = False
    conv: bool = False


@dataclass
class MixtureOfDepthsConfig:
    """Mixture-of-Depths token routing (``model.mixture_of_depths``)."""

    enabled: bool = False
    capacity_ratio: float = 0.5
    router_aux_loss_weight: float = 0.0


@dataclass
class SsmaxConfig:
    """Scalable softmax (``model.ssmax``)."""

    enabled: bool = False
    s_init: float = 1.0


@dataclass
class NumericsConfig:
    """Fixed-numerics opt-ins (``model.numerics``)."""

    fixed_softmax: bool = False
    fixed_cross_entropy: bool = False


@dataclass
class FfnConfig:
    """Feed-forward block (``model.ffn``)."""

    hidden_size: Optional[int] = None
    activation: str = "silu"
    activation_config: Optional[Dict[str, Any]] = None


@dataclass
class ClassificationHeadConfig:
    """Sequence-level classification head (``model.classification_head``)."""

    enabled: bool = False
    num_labels: Optional[int] = None
    pooling_mode: str = "cls"


@dataclass
class MhcConfig:
    """Manifold-constrained hyper-connections (``model.mhc``)."""

    enabled: bool = False
    expansion_rate: int = 4
    sinkhorn_iters: int = 20
    gating_init: float = 0.01
    checkpoint: bool = False
    full_prec_under_bitnet: bool = True
    hyperloop: bool = False
    hyperloop_begin_layers: int = 0
    hyperloop_end_layers: int = 0
    hyperloop_res_parameterization: str = "diagonal"


@dataclass
class FullAttnResConfig:
    """Full AttnRes knobs (``residuals.full_attn``)."""

    init_query_zero: bool = True
    use_rmsnorm_keys: bool = True


@dataclass
class BlockAttnResConfig:
    """Block AttnRes knobs (``residuals.block_attn``)."""

    num_blocks: int = 8
    init_query_zero: bool = True
    use_rmsnorm_keys: bool = True


@dataclass
class ResidualsConfig:
    """Residual-connection block (``model.residuals``)."""

    type: str = "standard"
    full_attn: FullAttnResConfig = field(default_factory=FullAttnResConfig)
    block_attn: BlockAttnResConfig = field(default_factory=BlockAttnResConfig)
    mhc_stream_mode: str = "independent"
    gradient_checkpoint: bool = False


@dataclass
class ImageConfig:
    """Vision-Transformer block (top-level ``image:``)."""

    height: int = 224
    width: int = 224
    patch_size: int = 16
    in_channels: int = 3
    to_grayscale: bool = False
    pos_embedding_type: str = "learned_1d"
    cls_token: bool = True
    pooling_mode: str = "cls"
    mask_ratio: float = 0.5
    mask_token_strategy: str = "bert"
    prediction_target: str = "mean_color_3bit"
    seg_head_type: str = "pixel"
    num_classes: int = 1000
    num_seg_classes: int = 21
    seg_num_queries: int = 100
    seg_l2_blocks: int = 3
    seg_mask_annealing: bool = True


@dataclass
class SpecAugmentConfig:
    """SpecAugment time/frequency masking (``audio.spec_augment``)."""

    enabled: bool = False
    freq_mask_width: int = 27
    time_masks: int = 10
    time_mask_ratio: float = 0.05


@dataclass
class AudioConfig:
    """Audio frontend + ASR head block (top-level ``audio:``)."""

    sample_rate: int = 16000
    n_mels: int = 80
    window_ms: float = 25.0
    stride_ms: float = 10.0
    n_fft: int = 400
    feature: str = "logmel"
    in_channels: int = 1
    conv_channels: int = 32
    num_classes: int = 2
    pooling_mode: str = "gap"
    asr_loss: str = "ctc"
    asr_decoder_dim: int = 320
    asr_decoder_layers: int = 1
    asr_decoder_dropout: float = 0.1
    spec_augment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)


def _build(cls, raw: Optional[Mapping[str, Any]]):
    """Instantiate a dataclass block from a YAML mapping, ignoring unknown keys.

    Args:
        cls: Block dataclass type.
        raw: Raw YAML mapping (``None`` when absent).

    Returns:
        The populated dataclass instance.
    """
    if isinstance(raw, cls):
        return raw
    if not isinstance(raw, Mapping):
        return cls()
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{key: value for key, value in raw.items() if key in names})


def _build_with_nested(cls, raw, nested: Mapping[str, Any]) -> Any:
    """Build a block whose attribute ``name`` holds a nested dataclass.

    Args:
        cls: Block dataclass type.
        raw: Raw YAML mapping.
        nested: Mapping of attribute name -> nested dataclass type.

    Returns:
        The populated dataclass instance.
    """
    if not isinstance(raw, Mapping):
        return cls()
    names = {f.name for f in dataclasses.fields(cls)}
    kwargs: Dict[str, Any] = {}
    for key, value in raw.items():
        if key not in names:
            continue
        if key in nested:
            kwargs[key] = _build(nested[key], value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# top-level configuration
# ---------------------------------------------------------------------------


@dataclass
class FrankensteinModelConfig:
    """Nested configuration shared by every Frankenstein model variant.

    Attributes:
        dims: Core dimensions, dropout, ``layer_pattern`` and ``mode``.
        norm: Normalization type/parameters and the ``qk_norm`` sub-block.
        embedding: Token-embedding variant (``type``) and its knobs.
        attention: Per-mixer knobs, keyed by registry mixer name.
        positional_encoding: Model-wide positional encoding name.
        positional_encoding_parameters: Per-encoding knobs plus the per-mixer
            ``use_pe`` flags.
        moe: Mixture-of-Experts block.
        bitnet: BitNet 1.58-bit quantization block.
        mixture_of_depths: Mixture-of-Depths block.
        ssmax: Scalable-softmax block.
        numerics: Fixed-numerics opt-ins.
        ffn: Feed-forward width/activation block.
        classification_head: Sequence-level classification head.
        mhc: Manifold-constrained hyper-connections.
        residuals: Residual-connection strategy and knobs.
        image: Vision-Transformer block (``image:`` YAML key).
        audio: Audio frontend + ASR block (``audio:`` YAML key).
        model_class: Recorded model class for deployment dispatch.
        token_vocab: Id-ordered token strings for dynamic embeddings.
        num_classes: Task class count (image or audio, resolved at load).
        pooling_mode: Task pooling mode (image or audio, resolved at load).
    """

    dims: DimsConfig = field(default_factory=DimsConfig)
    norm: NormConfig = field(default_factory=NormConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    attention: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    positional_encoding: str = "rope"
    positional_encoding_parameters: Dict[str, Any] = field(default_factory=dict)
    moe: MoEConfig = field(default_factory=MoEConfig)
    bitnet: BitnetConfig = field(default_factory=BitnetConfig)
    mixture_of_depths: MixtureOfDepthsConfig = field(
        default_factory=MixtureOfDepthsConfig
    )
    ssmax: SsmaxConfig = field(default_factory=SsmaxConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    ffn: FfnConfig = field(default_factory=FfnConfig)
    classification_head: ClassificationHeadConfig = field(
        default_factory=ClassificationHeadConfig
    )
    mhc: MhcConfig = field(default_factory=MhcConfig)
    residuals: ResidualsConfig = field(default_factory=ResidualsConfig)
    image: Optional[ImageConfig] = None
    audio: Optional[AudioConfig] = None
    model_class: Optional[str] = None
    token_vocab: Optional[List[str]] = None
    num_classes: int = 1000
    pooling_mode: str = "cls"

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(
        cls,
        model_block: Optional[Mapping[str, Any]] = None,
        *,
        image_block: Optional[Mapping[str, Any]] = None,
        audio_block: Optional[Mapping[str, Any]] = None,
        model_class: Optional[str] = None,
    ) -> "FrankensteinModelConfig":
        """Build the configuration from the nested YAML blocks.

        Args:
            model_block: The ``model:`` mapping (already schema-validated).
            image_block: The top-level ``image:`` mapping, when present.
            audio_block: The top-level ``audio:`` mapping, when present.
            model_class: The resolved ``model_class`` value.

        Returns:
            A fully populated :class:`FrankensteinModelConfig`.
        """
        raw = dict(model_block or {})
        if image_block is None and isinstance(raw.get("image"), Mapping):
            image_block = dict(raw.pop("image"))
        if audio_block is None and isinstance(raw.get("audio"), Mapping):
            audio_block = dict(raw.pop("audio"))
        if model_class is None and raw.get("model_class") is not None:
            model_class = str(raw.pop("model_class"))
        else:
            raw.pop("model_class", None)
        embedding_raw = raw.get("embedding") or {}
        embedding = _build_with_nested(
            EmbeddingConfig,
            embedding_raw,
            {
                "factorized": FactorizedEmbeddingConfig,
                "dynamic": DynamicEmbeddingConfig,
                "conv": EmbeddingConvConfig,
            },
        )
        # ``embedding.type`` is authoritative; the legacy ``*.enabled`` flags
        # are folded in when ``type`` is left at its default.
        if embedding.type == "standard":
            if embedding.dynamic.enabled:
                embedding.type = "dynamic"
            elif embedding.factorized.enabled:
                embedding.type = "factorized"

        norm = _build_with_nested(NormConfig, raw.get("norm") or {}, {"qk_norm": QKNormConfig})
        moe = _build_with_nested(
            MoEConfig,
            raw.get("moe") or {},
            {
                "router": MoERouterConfig,
                "routing": MoERoutingConfig,
                "balancing": MoEBalancingConfig,
                "shared": MoESharedConfig,
                "latent": MoELatentConfig,
                "layout": MoELayoutConfig,
                "experts": MoEExpertsConfig,
            },
        )
        residuals = _build_with_nested(
            ResidualsConfig,
            raw.get("residuals") or {},
            {"full_attn": FullAttnResConfig, "block_attn": BlockAttnResConfig},
        )
        image = None
        if image_block is not None:
            image_raw = dict(image_block)
            size = image_raw.pop("image_size", None)
            if isinstance(size, Mapping):
                image_raw.setdefault("height", size.get("height"))
                image_raw.setdefault("width", size.get("width"))
            image_raw.pop("height", None) if image_raw.get("height") is None else None
            image_raw.pop("width", None) if image_raw.get("width") is None else None
            image = _build(ImageConfig, image_raw)
        audio = None
        if audio_block is not None:
            audio = _build_with_nested(
                AudioConfig, audio_block, {"spec_augment": SpecAugmentConfig}
            )

        explicit_num_classes = raw.get("num_classes")
        explicit_pooling_mode = raw.get("pooling_mode")
        attention = {
            str(name): dict(values or {})
            for name, values in (raw.get("attention") or {}).items()
        }
        config = cls(
            dims=_build(DimsConfig, raw.get("dims")),
            norm=norm,
            embedding=embedding,
            attention=attention,
            positional_encoding=str(raw.get("positional_encoding") or "rope"),
            positional_encoding_parameters=dict(
                raw.get("positional_encoding_parameters") or {}
            ),
            moe=moe,
            bitnet=_build(BitnetConfig, raw.get("bitnet")),
            mixture_of_depths=_build(MixtureOfDepthsConfig, raw.get("mixture_of_depths")),
            ssmax=_build(SsmaxConfig, raw.get("ssmax")),
            numerics=_build(NumericsConfig, raw.get("numerics")),
            ffn=_build(FfnConfig, raw.get("ffn")),
            classification_head=_build(ClassificationHeadConfig, raw.get("classification_head")),
            mhc=_build(MhcConfig, raw.get("mhc")),
            residuals=residuals,
            image=image,
            audio=audio,
            model_class=model_class,
            token_vocab=raw.get("token_vocab"),
        )
        if explicit_num_classes is not None:
            config.num_classes = int(explicit_num_classes)
        elif image is not None:
            config.num_classes = int(image.num_classes)
        elif audio is not None:
            config.num_classes = int(audio.num_classes)
        if explicit_pooling_mode is not None:
            config.pooling_mode = str(explicit_pooling_mode)
        elif image is not None:
            config.pooling_mode = str(image.pooling_mode)
        elif audio is not None:
            config.pooling_mode = str(audio.pooling_mode)
        return config

    def to_jsonable(self) -> Dict[str, Any]:
        """Return a JSON-serializable nested mapping (checkpoints / export)."""
        return dataclasses.asdict(self)

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Derive dependent values and validate cross-field invariants.

        Raises:
            ValueError: If a cross-field invariant is violated.
        """
        dims = self.dims
        if self.ffn.hidden_size is None:
            self.ffn.hidden_size = dims.hidden_size * 2

        self.ffn.activation = str(self.ffn.activation).lower()


        # ---- MoE anatomy ----
        moe = self.moe
        if moe.num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {moe.num_experts}")
        if moe.top_k < 1:
            raise ValueError(f"top_k_experts must be >= 1, got {moe.top_k}")
        if moe.top_k > moe.num_experts:
            raise ValueError(
                f"top_k_experts must be <= num_experts, got top_k_experts={moe.top_k}, "
                f"num_experts={moe.num_experts}"
            )
        if moe.router.scoring not in {"softmax", "sigmoid", "softplus_sqrt"}:
            raise ValueError(
                "moe_router_scoring must be 'softmax', 'sigmoid' or "
                f"'softplus_sqrt', got {moe.router.scoring!r}"
            )
        if moe.router.dtype not in {"float32", "bfloat16"}:
            raise ValueError(
                f"moe_router_dtype must be 'float32' or 'bfloat16', got {moe.router.dtype!r}"
            )
        if moe.routing.type not in {"token_choice", "expert_choice"}:
            raise ValueError(
                "moe_routing_type must be 'token_choice' or 'expert_choice', got "
                f"{moe.routing.type!r}"
            )
        if moe.routing.hash_table not in {"random", "round_robin"}:
            raise ValueError(
                f"moe_hash_table must be 'random' or 'round_robin', got {moe.routing.hash_table!r}"
            )
        if moe.balancing.aux_loss not in {"none", "batch", "sequence"}:
            raise ValueError(
                "moe_aux_loss_type must be 'none', 'batch' or 'sequence', got "
                f"{moe.balancing.aux_loss!r}"
            )
        if moe.balancing.bias_update not in {
            "none",
            "sign",
            "centered_sign",
            "quantile",
        }:
            raise ValueError(
                "moe_bias_update must be 'none', 'sign', 'centered_sign' or "
                f"'quantile', got {moe.balancing.bias_update!r}"
            )
        if moe.shared.gate not in {"none", "sigmoid"}:
            raise ValueError(
                f"moe_shared_expert_gate must be 'none' or 'sigmoid', got {moe.shared.gate!r}"
            )
        if moe.balancing.bias_update != "none" and moe.router.scoring not in {
            "sigmoid",
            "softplus_sqrt",
        }:
            raise ValueError(
                "moe.balancing.bias_update requires router scoring 'sigmoid' or "
                "'softplus_sqrt'"
            )
        if moe.routing.type == "expert_choice" and moe.balancing.bias_update != "none":
            raise ValueError(
                "moe.routing.type='expert_choice' is self-balancing: it cannot "
                "be combined with moe.balancing.bias_update."
            )
        if moe.routing.type == "expert_choice" and moe.balancing.aux_loss != "none":
            raise ValueError(
                "moe.routing.type='expert_choice' is self-balancing: it cannot "
                "be combined with moe.balancing.aux_loss."
            )
        if moe.routing.hash_layers and moe.routing.type != "token_choice":
            raise ValueError("moe.routing.hash_layers requires routing.type='token_choice'.")
        if moe.routing.hash_layers and moe.balancing.bias_update != "none":
            raise ValueError(
                "moe.routing.hash_layers selects experts from a fixed table: the "
                "expert bias cannot rebalance it, so bias_update must stay 'none'."
            )
        if moe.routing.hash_layers < 0:
            raise ValueError(f"moe_hash_layers must be >= 0, got {moe.routing.hash_layers}")
        if moe.layout.dense_layers < 0:
            raise ValueError(f"moe_dense_layers must be >= 0, got {moe.layout.dense_layers}")
        if moe.layout.dense_layers >= dims.num_layers:
            raise ValueError(
                "moe_dense_layers must be < num_layers, got "
                f"dense_layers={moe.layout.dense_layers}, num_layers={dims.num_layers}"
            )
        if moe.layout.every_n_layers < 1:
            raise ValueError(
                f"moe_layer_frequency must be >= 1, got {moe.layout.every_n_layers}"
            )
        if moe.routing.hash_layers > self.moe_layer_count():
            raise ValueError(
                "moe_hash_layers cannot exceed the number of MoE layers "
                f"(moe_hash_layers={moe.routing.hash_layers}, "
                f"moe_layers={self.moe_layer_count()})."
            )
        if moe.routing.capacity_factor < 0.0:
            raise ValueError(
                f"moe_capacity_factor must be >= 0, got {moe.routing.capacity_factor}"
            )
        if moe.router.jitter_eps < 0.0:
            raise ValueError(
                f"moe_router_jitter_eps must be >= 0, got {moe.router.jitter_eps}"
            )
        if moe.router.z_loss_coeff < 0.0:
            raise ValueError(
                f"moe_router_z_loss_coeff must be >= 0, got {moe.router.z_loss_coeff}"
            )
        if moe.balancing.bias_update_rate < 0.0:
            raise ValueError(
                f"moe_bias_update_rate must be >= 0, got {moe.balancing.bias_update_rate}"
            )
        if moe.shared.count < 0:
            raise ValueError(
                f"moe_shared_experts_count must be >= 0, got {moe.shared.count}"
            )
        if moe.experts.activation_clamp < 0.0:
            raise ValueError(
                f"moe_activation_clamp must be >= 0, got {moe.experts.activation_clamp}"
            )
        if moe.expert_hidden_size is not None and int(moe.expert_hidden_size) < 1:
            raise ValueError(
                f"moe_expert_hidden_size must be >= 1 when set, got {moe.expert_hidden_size}"
            )
        if moe.shared.hidden_size is not None and int(moe.shared.hidden_size) < 1:
            raise ValueError(
                "moe_shared_expert_hidden_size must be >= 1 when set, got "
                f"{moe.shared.hidden_size}"
            )
        if moe.latent.enabled:
            if moe.latent.dim is None:
                raise ValueError("moe.latent.enabled=true requires moe.latent.dim to be set.")
            if not 0 < int(moe.latent.dim) < int(dims.hidden_size):
                raise ValueError(
                    "moe_latent_dim must be in (0, hidden_size) when latent MoE "
                    f"is enabled, got latent_dim={moe.latent.dim}, "
                    f"hidden_size={dims.hidden_size}"
                )
        if dims.mode not in {"encoder", "decoder"}:
            raise ValueError(
                f"mode must be 'encoder' or 'decoder', got {dims.mode!r}"
            )
        if self.mixture_of_depths.capacity_ratio <= 0 or self.mixture_of_depths.capacity_ratio > 1:
            raise ValueError(
                "mixture_of_depths_capacity_ratio (mixture_of_depths.capacity_ratio) must be in (0, 1], got "
                f"{self.mixture_of_depths.capacity_ratio}"
            )
        if self.mixture_of_depths.router_aux_loss_weight < 0:
            raise ValueError(
                "mixture_of_depths_router_aux_loss_weight (mixture_of_depths.router_aux_loss_weight) must be >= 0, got "
                f"{self.mixture_of_depths.router_aux_loss_weight}"
            )

        from ._validation import validate_config

        validate_config(self)

    # ------------------------------------------------------------------
    # MoE layout helpers
    # ------------------------------------------------------------------
    def is_moe_layer(self, layer_idx: int) -> bool:
        """Return whether physical layer ``layer_idx`` runs the MoE FFN.

        Args:
            layer_idx: Zero-based physical layer index.

        Returns:
            ``True`` when the layer uses the MoE FFN.
        """
        return self.moe_layer_mask()[layer_idx]

    def moe_layer_mask(self) -> List[bool]:
        """Return the per-physical-layer MoE mask.

        Returns:
            A list of booleans with one entry per physical layer.
        """
        mask = [False] * int(self.dims.num_layers)
        if not self.moe.enabled:
            return mask
        dense = int(self.moe.layout.dense_layers)
        frequency = max(int(self.moe.layout.every_n_layers), 1)
        for idx in range(dense, len(mask)):
            mask[idx] = (idx - dense) % frequency == 0
        return mask

    def moe_layer_count(self) -> int:
        """Return the number of physical layers that run the MoE FFN."""
        return sum(self.moe_layer_mask())

    # ------------------------------------------------------------------
    # historical flat-name accessors (structural scalars only)
    # ------------------------------------------------------------------
    def use_pe_for(self, mixer: str, default: bool = True) -> bool:
        """Return whether ``mixer`` consumes the model-wide positional encoding.

        Args:
            mixer: Mixer registry name.
            default: Fallback when neither the config nor the spec decides.

        Returns:
            ``True`` when the mixer should apply the positional encoding.
        """
        from src.model.attention.params import use_pe_for

        return use_pe_for(self, mixer, default)

    def attention_params(self, mixer: str) -> Dict[str, Any]:
        """Return the raw ``model.attention.<mixer>`` mapping.

        Args:
            mixer: Mixer registry name (or alias).

        Returns:
            The raw parameter mapping (empty when absent).
        """
        return dict(self.attention.get(mixer, {}) or {})

    @property
    def head_dim(self) -> int:
        """Per-head dimension (``dims.hidden_size // dims.num_heads``)."""
        return self.dims.head_dim


    # ------------------------------------------------------------------
    # positional-encoding parameters (nested ``positional_encoding_parameters``)
    # ------------------------------------------------------------------
    def _pe_param(self, key: str, leaf: str, default: Any) -> Any:
        """Read one positional-encoding knob with a fallback default."""
        block = self.positional_encoding_parameters.get(key) or {}
        value = block.get(leaf) if isinstance(block, Mapping) else None
        return default if value is None else value

    @property
    def rope_base(self) -> float:
        """RoPE base frequency (``positional_encoding_parameters.rope.base``)."""
        return float(self._pe_param("rope", "base", 10_000.0))

    @property
    def rope_scaling(self) -> float:
        """RoPE scaling factor (``positional_encoding_parameters.rope.scaling``)."""
        return float(self._pe_param("rope", "scaling", 1.0))

    @property
    def hope_base(self) -> float:
        """HoPE base frequency."""
        return float(self._pe_param("hope", "base", 10_000.0))

    @property
    def hope_damping(self) -> float:
        """HoPE exponential damping coefficient."""
        return float(self._pe_param("hope", "damping", 0.01))

    @property
    def alibi_num_heads(self) -> int:
        """ALiBi head count (defaults to ``dims.num_heads``)."""
        return int(self._pe_param("alibi", "num_heads", self.dims.num_heads))

    @property
    def bam_learn_mu(self) -> bool:
        """BAM: learn the location parameter."""
        return bool(self._pe_param("bam", "learn_mu", False))

    @property
    def bam_theta_init(self) -> float:
        """BAM: initial shape parameter."""
        return float(self._pe_param("bam", "theta_init", 0.0))

    @property
    def bam_mu_init(self) -> float:
        """BAM: initial location parameter."""
        return float(self._pe_param("bam", "mu_init", 0.0))

    @property
    def bam_eps(self) -> float:
        """BAM: numerical epsilon."""
        return float(self._pe_param("bam", "eps", 1e-5))

    @property
    def pape_num_parabolas(self) -> int:
        """PaPE: number of learnable parabola terms."""
        return int(self._pe_param("pape", "num_parabolas", 4))

    @property
    def pape_num_positions(self) -> int:
        """PaPE: positions per token."""
        return int(self._pe_param("pape", "num_positions", 1))

    @property
    def pape_rotation_invariant(self) -> bool:
        """PaPE: rotation-invariant variant flag."""
        return bool(self._pe_param("pape", "rotation_invariant", False))

    @property
    def sinusoidal_max_len(self) -> int:
        """Sinusoidal PE: maximum sequence length."""
        return int(self._pe_param("sinusoidal", "max_len", 512))

    @property
    def sinusoidal_base(self) -> float:
        """Sinusoidal PE: base frequency."""
        return float(self._pe_param("sinusoidal", "base", 10_000.0))

    @property
    def sinusoidal_scale(self) -> float:
        """Sinusoidal PE: output scale."""
        return float(self._pe_param("sinusoidal", "scale", 1.0))

    @property
    def learned_max_len(self) -> int:
        """Learned absolute PE: maximum sequence length."""
        return int(self._pe_param("learned", "max_len", 512))

    @property
    def learned_init_std(self) -> float:
        """Learned absolute PE: initialization std."""
        return float(self._pe_param("learned", "init_std", 0.02))

def merge_model_blocks(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-merge nested ``model:`` block mappings.

    Builder helpers use this to let callers override a nested block without
    restating its defaults (``image={"height": 32}`` keeps the default width,
    channels and patch size).

    Args:
        base: Nested model block with defaults.
        overrides: Nested overrides.

    Returns:
        A new merged mapping.
    """
    merged: Dict[str, Any] = {
        key: dict(value) if isinstance(value, Mapping) else value
        for key, value in base.items()
    }
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            merged[key] = merge_model_blocks(current, value)
        else:
            merged[key] = value
    return merged
