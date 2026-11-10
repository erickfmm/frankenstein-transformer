#!/usr/bin/env python3
"""Nested, schema-aligned model configuration for all Frankenstein variants.

The runtime shape mirrors the YAML ``model:`` block one-to-one — there is no
nested-to-flat flattening step. Structural blocks are typed dataclasses;
per-component knobs live in per-family mappings (``attention``,
``positional_encoding_parameters``) and are bound through the component
registry (:mod:`src.registry`) by the family factories.

Historical flat attribute names remain readable through ``__getattr__``
aliases (``config.hidden_size`` → ``config.dims.hidden_size``) because the
model, training, deployment and export code reads them in thousands of
places; per-component knobs are deliberately *not* re-exposed — components
read their bound parameters.
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
    "raf_input_scaling", "maxout_pieces",
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
    ):
        if fk in cfg and not isinstance(cfg[fk], (int, float)):
            raise ValueError(f"ffn_activation_config.{fk} must be a number")
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
        nested_markers = {
            "dims", "norm", "embedding", "attention", "positional_encoding",
            "positional_encoding_parameters", "moe", "bitnet",
            "mixture_of_depths", "ssmax", "numerics", "ffn",
            "classification_head", "mhc", "residuals",
        }
        if raw and not (nested_markers & set(raw)):
            config = cls.from_flat(**raw)
            if image_block is not None:
                config.image = _build(ImageConfig, image_block)
            if audio_block is not None:
                config.audio = _build_with_nested(
                    AudioConfig, audio_block, {"spec_augment": SpecAugmentConfig}
                )
            if config.image is not None:
                config.num_classes = int(config.image.num_classes)
                config.pooling_mode = str(config.image.pooling_mode)
            elif config.audio is not None:
                config.num_classes = int(config.audio.num_classes)
                config.pooling_mode = str(config.audio.pooling_mode)
            config.model_class = model_class
            return config
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

    @classmethod
    def from_flat(cls, **kwargs: Any) -> "FrankensteinModelConfig":
        """Build from the historical flat keyword arguments.

        This is the migration path for legacy checkpoints and for the test
        suite; production code builds the nested shape directly. Flat
        structural names are routed through :data:`_FLAT_ALIASES` and
        per-mixer names (``mla_latent_rank``, ``falcon_beta``, …) are routed
        into ``model.attention.<mixer>``.

        Args:
            **kwargs: Historical flat configuration fields, or nested blocks
                (``dims=…``, ``attention={…}``).

        Returns:
            A fully populated nested configuration.

        Raises:
            TypeError: If a keyword is neither a nested field nor a known
                flat alias.
        """
        from src.registry import ensure_builtins

        ensure_builtins()
        nested_names = {f.name for f in dataclasses.fields(cls)}
        component_alias = _component_flat_aliases()
        model: Dict[str, Any] = {}
        attention: Dict[str, Dict[str, Any]] = dict(kwargs.pop("attention", {}) or {})
        image_block = dict(kwargs.pop("image", None) or {}) or None
        audio_block = dict(kwargs.pop("audio", None) or {}) or None
        model_class = kwargs.pop("model_class", None)
        positional = kwargs.pop("positional_encoding_parameters", None)
        if positional:
            model["positional_encoding_parameters"] = dict(positional)

        def put(path: Tuple[str, ...], value: Any) -> None:
            node = model
            for part in path[:-1]:
                child = node.get(part)
                if dataclasses.is_dataclass(child):
                    child = dataclasses.asdict(child)
                    node[part] = child
                elif not isinstance(child, dict):
                    child = {}
                    node[part] = child
                node = child
            node[path[-1]] = value

        for key, value in kwargs.items():
            if key in _FLAT_ALIASES and not dataclasses.is_dataclass(value) and not (
                key in nested_names and isinstance(value, Mapping)
            ):
                path = _FLAT_ALIASES[key]
                if path[0] == "image":
                    image_block = dict(image_block or {})
                    image_block[path[-1]] = value
                elif path[0] == "audio":
                    audio_block = dict(audio_block or {})
                    node = audio_block
                    for part in path[1:-1]:
                        node = node.setdefault(part, {})
                    node[path[-1]] = value
                else:
                    put(path, value)
                continue
            if key in nested_names or key == "positional_encoding":
                if key in {"dims", "norm", "embedding", "moe", "bitnet",
                           "mixture_of_depths", "ssmax", "numerics", "ffn",
                           "classification_head", "mhc", "residuals"} and dataclasses.is_dataclass(value):
                    model[key] = value
                elif key == "positional_encoding":
                    model[key] = value
                elif key == "token_vocab":
                    model[key] = value
                elif key in {"num_classes", "pooling_mode"}:
                    put(("__top__", key), value)
                else:
                    model[key] = value
                continue
            if key.endswith("_use_pe"):
                mixer = key[: -len("_use_pe")]
                put(("positional_encoding_parameters", "use_pe", key), value)
                continue
            if key in _FFN_ACTIVATION_CONFIG_KEYS:
                put(("ffn", "activation_config", key), value)
                continue
            if key in component_alias:
                for mixer, leaf in component_alias[key]:
                    attention.setdefault(mixer, {})[leaf] = value
                continue
            # Unknown flat keys are warned about and dropped by the loader
            # (the strict whole-document gate lives in the schema consistency
            # test); programmatic callers get the same tolerant behavior.
            continue
        if attention:
            model["attention"] = attention
        top = model.pop("__top__", {})
        config = cls.from_dict(
            model,
            image_block=image_block,
            audio_block=audio_block,
            model_class=model_class,
        )
        for key, value in top.items():
            setattr(config, key, value)
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
        if moe.balancing.bias_update not in {"none", "sign", "centered_sign"}:
            raise ValueError(
                "moe_bias_update must be 'none', 'sign' or 'centered_sign', got "
                f"{moe.balancing.bias_update!r}"
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


    def __setattr__(self, name: str, value: Any) -> None:
        """Route historical flat-name assignments into the nested blocks.

        Args:
            name: Attribute name.
            value: Assigned value.
        """
        if name.startswith("_") or name in _FIELD_NAMES:
            if name == "classification_head" and isinstance(value, bool):
                object.__setattr__(
                    self, name, ClassificationHeadConfig(enabled=bool(value))
                )
                self.__dict__.get("_alias_cache", {}).pop(name, None)
                return
            object.__setattr__(self, name, value)
            self.__dict__.get("_alias_cache", {}).pop(name, None)
            return
        alias = _FLAT_ALIASES.get(name)
        if alias is not None:
            node: Any = self
            for part in alias[:-1]:
                child = getattr(node, part)
                if child is None:
                    child = _default_block_for(part)
                    object.__setattr__(node, part, child)
                node = child
            setattr(node, alias[-1], value)
            self.__dict__.get("_alias_cache", {}).pop(name, None)
            return
        component = _COMPONENT_ALIASES.get(name)
        if component is not None:
            mixer, leaf = component
            self.attention.setdefault(mixer, {})[leaf] = value
            self.__dict__.get("_alias_cache", {}).pop(name, None)
            return
        object.__setattr__(self, name, value)
        self.__dict__.get("_alias_cache", {}).pop(name, None)

    def __getattr__(self, name: str) -> Any:
        """Resolve historical flat attribute names (structural blocks only).

        Args:
            name: Attribute name requested by legacy code.

        Returns:
            The corresponding nested value.

        Raises:
            AttributeError: If ``name`` has no nested mapping.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        if name in {
            "dims", "norm", "embedding", "attention", "moe", "bitnet",
            "mixture_of_depths", "ssmax", "numerics", "ffn",
            "classification_head", "mhc", "residuals", "image", "audio",
            "positional_encoding_parameters",
        }:
            raise AttributeError(name)
        alias = _FLAT_ALIASES.get(name)
        if alias is not None:
            cache = self.__dict__.setdefault("_alias_cache", {})
            if name in cache:
                return cache[name]
            node: Any = self
            parts = list(alias)
            # Modality blocks are optional; fall back to their defaults so
            # legacy scalar reads never explode on a text-only config.
            image = self.__dict__.get("image")
            audio = self.__dict__.get("audio")
            if parts[0] == "image" and image is None:
                node = ImageConfig()
                parts = parts[1:]
            elif parts[0] == "audio" and audio is None:
                node = AudioConfig()
                parts = parts[1:]
            for part in parts:
                if isinstance(node, Mapping):
                    node = node.get(part)
                else:
                    node = getattr(node, part)
            cache[name] = node
            return node
        component = _COMPONENT_ALIASES.get(name)
        if component is not None:
            cache = self.__dict__.setdefault("_alias_cache", {})
            if name in cache:
                return cache[name]
            mixer, leaf = component
            raw = (self.__dict__.get("attention") or {}).get(mixer) or {}
            value = raw.get(leaf)
            if value is None:
                from src.registry import ComponentFamily, ensure_builtins, registry

                ensure_builtins()
                value = dict(
                    registry.get(ComponentFamily.MIXER, mixer).defaults
                ).get(leaf)
            cache[name] = value
            return value
        if name.endswith("_use_pe"):
            mixer = name[: -len("_use_pe")]
            try:
                return self.use_pe_for(mixer, True)
            except Exception as exc:  # noqa: BLE001 - unknown mixers fall back
                raise AttributeError(name) from exc
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


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

def _component_flat_aliases() -> Dict[str, Tuple[Tuple[str, str], ...]]:
    """Map historical flat mixer fields to ``(mixer, param)`` pairs.

    Returns:
        A mapping such as ``"mla_latent_rank" -> (("mla", "latent_rank"),)``
        and ``"falcon_beta" -> ((six falcon variants…), …)``.
    """
    from src.registry import ComponentFamily, ensure_builtins, registry

    ensure_builtins()
    aliases: Dict[str, Tuple[Tuple[str, str], ...]] = {}
    for spec in registry.specs(ComponentFamily.MIXER):
        prefix = "falcon" if spec.name.startswith("falcon") else spec.name
        for leaf in spec.defaults:
            if spec.name == "retnet" and leaf == "heads":
                flat = "retention_heads"
            elif spec.name == "ode":
                flat = f"ode_{leaf}"
            else:
                flat = f"{prefix}_{leaf}"
            aliases.setdefault(flat, [])
            entry = (spec.name, leaf)
            if entry not in aliases[flat]:
                aliases[flat].append(entry)
    return {key: tuple(value) for key, value in aliases.items()}


_FLAT_ALIASES: Dict[str, Tuple[str, ...]] = {
    'alibi_num_heads': ('positional_encoding_parameters', 'alibi', 'num_heads'),
    'asr_decoder_dim': ('audio', 'asr_decoder_dim'),
    'asr_decoder_dropout': ('audio', 'asr_decoder_dropout'),
    'asr_decoder_layers': ('audio', 'asr_decoder_layers'),
    'asr_loss': ('audio', 'asr_loss'),
    'attnres_gradient_checkpoint': ('residuals', 'gradient_checkpoint'),
    'attnres_mhc_stream_mode': ('residuals', 'mhc_stream_mode'),
    'audio_conv_channels': ('audio', 'conv_channels'),
    'audio_feature': ('audio', 'feature'),
    'audio_in_channels': ('audio', 'in_channels'),
    'audio_n_fft': ('audio', 'n_fft'),
    'audio_n_mels': ('audio', 'n_mels'),
    'audio_sample_rate': ('audio', 'sample_rate'),
    'audio_spec_augment_enabled': ('audio', 'spec_augment', 'enabled'),
    'audio_spec_augment_freq_mask_width': ('audio', 'spec_augment', 'freq_mask_width'),
    'audio_spec_augment_time_mask_ratio': ('audio', 'spec_augment', 'time_mask_ratio'),
    'audio_spec_augment_time_masks': ('audio', 'spec_augment', 'time_masks'),
    'audio_stride_ms': ('audio', 'stride_ms'),
    'audio_window_ms': ('audio', 'window_ms'),
    'bam_eps': ('positional_encoding_parameters', 'bam', 'eps'),
    'bam_learn_mu': ('positional_encoding_parameters', 'bam', 'learn_mu'),
    'bam_mu_init': ('positional_encoding_parameters', 'bam', 'mu_init'),
    'bam_theta_init': ('positional_encoding_parameters', 'bam', 'theta_init'),
    'bitnet_routers': ('bitnet', 'routers'),
    'block_attn_init_query_zero': ('residuals', 'block_attn', 'init_query_zero'),
    'block_attn_num_blocks': ('residuals', 'block_attn', 'num_blocks'),
    'block_attn_use_rmsnorm_keys': ('residuals', 'block_attn', 'use_rmsnorm_keys'),
    'classification_head': ('classification_head', 'enabled'),
    'cls_token': ('image', 'cls_token'),
    'dropout': ('dims', 'dropout'),
    'dynamic_embedding_bucket': ('embedding', 'dynamic', 'bucket'),
    'dynamic_embedding_ngram_max': ('embedding', 'dynamic', 'ngram_max'),
    'dynamic_embedding_precompute': ('embedding', 'dynamic', 'precompute'),
    'dynamic_embedding_seed': ('embedding', 'dynamic', 'seed'),
    'embedding_conv_kernel': ('embedding', 'conv', 'kernel'),
    'encoder_pooling_mode': ('classification_head', 'pooling_mode'),
    'factorized_embedding_dim': ('embedding', 'factorized', 'dim'),
    'ffn_activation': ('ffn', 'activation'),
    'ffn_activation_config': ('ffn', 'activation_config'),
    'ffn_hidden_size': ('ffn', 'hidden_size'),
    'fixed_cross_entropy': ('numerics', 'fixed_cross_entropy'),
    'fixed_softmax': ('numerics', 'fixed_softmax'),
    'flashnorm_partial_ratio': ('norm', 'flashnorm_partial_ratio'),
    'full_attn_init_query_zero': ('residuals', 'full_attn', 'init_query_zero'),
    'full_attn_use_rmsnorm_keys': ('residuals', 'full_attn', 'use_rmsnorm_keys'),
    'hidden_size': ('dims', 'hidden_size'),
    'hope_base': ('positional_encoding_parameters', 'hope', 'base'),
    'hope_damping': ('positional_encoding_parameters', 'hope', 'damping'),
    'image_height': ('image', 'height'),
    'image_width': ('image', 'width'),
    'in_channels': ('image', 'in_channels'),
    'layer_pattern': ('dims', 'layer_pattern'),
    'learned_init_std': ('positional_encoding_parameters', 'learned', 'init_std'),
    'learned_max_len': ('positional_encoding_parameters', 'learned', 'max_len'),
    'mask_ratio': ('image', 'mask_ratio'),
    'mask_token_strategy': ('image', 'mask_token_strategy'),
    'mhc_checkpoint': ('mhc', 'checkpoint'),
    'mhc_expansion_rate': ('mhc', 'expansion_rate'),
    'mhc_full_prec_under_bitnet': ('mhc', 'full_prec_under_bitnet'),
    'mhc_gating_init': ('mhc', 'gating_init'),
    'mhc_hyperloop': ('mhc', 'hyperloop'),
    'mhc_hyperloop_begin_layers': ('mhc', 'hyperloop_begin_layers'),
    'mhc_hyperloop_end_layers': ('mhc', 'hyperloop_end_layers'),
    'mhc_hyperloop_res_parameterization': ('mhc', 'hyperloop_res_parameterization'),
    'mhc_sinkhorn_iters': ('mhc', 'sinkhorn_iters'),
    'mixture_of_depths_capacity_ratio': ('mixture_of_depths', 'capacity_ratio'),
    'mixture_of_depths_router_aux_loss_weight': ('mixture_of_depths', 'router_aux_loss_weight'),
    'mode': ('dims', 'mode'),
    'moe_activation_clamp': ('moe', 'experts', 'activation_clamp'),
    'moe_aux_loss_coeff': ('moe', 'balancing', 'aux_loss_coeff'),
    'moe_aux_loss_type': ('moe', 'balancing', 'aux_loss'),
    'moe_bias_update': ('moe', 'balancing', 'bias_update'),
    'moe_bias_update_rate': ('moe', 'balancing', 'bias_update_rate'),
    'moe_capacity_factor': ('moe', 'routing', 'capacity_factor'),
    'moe_dense_layers': ('moe', 'layout', 'dense_layers'),
    'moe_expert_hidden_size': ('moe', 'expert_hidden_size'),
    'moe_hash_layers': ('moe', 'routing', 'hash_layers'),
    'moe_hash_seed': ('moe', 'routing', 'hash_seed'),
    'moe_hash_table': ('moe', 'routing', 'hash_table'),
    'moe_latent_dim': ('moe', 'latent', 'dim'),
    'moe_latent_enabled': ('moe', 'latent', 'enabled'),
    'moe_latent_norm_before_up_proj': ('moe', 'latent', 'norm_before_up_proj'),
    'moe_layer_frequency': ('moe', 'layout', 'every_n_layers'),
    'moe_routed_scaling_factor': ('moe', 'router', 'routed_scaling_factor'),
    'moe_router_dtype': ('moe', 'router', 'dtype'),
    'moe_router_jitter_eps': ('moe', 'router', 'jitter_eps'),
    'moe_router_pre_softmax': ('moe', 'router', 'pre_softmax'),
    'moe_router_renormalize': ('moe', 'router', 'renormalize'),
    'moe_router_scoring': ('moe', 'router', 'scoring'),
    'moe_router_z_loss_coeff': ('moe', 'router', 'z_loss_coeff'),
    'moe_routing_type': ('moe', 'routing', 'type'),
    'moe_shared_expert_gate': ('moe', 'shared', 'gate'),
    'moe_shared_expert_hidden_size': ('moe', 'shared', 'hidden_size'),
    'moe_shared_experts_count': ('moe', 'shared', 'count'),
    'norm_type': ('norm', 'type'),
    'num_experts': ('moe', 'num_experts'),
    'num_heads': ('dims', 'num_heads'),
    'num_kv_heads': ('dims', 'num_kv_heads'),
    'num_labels': ('classification_head', 'num_labels'),
    'num_layers': ('dims', 'num_layers'),
    'num_loops': ('dims', 'num_loops'),
    'num_seg_classes': ('image', 'num_seg_classes'),
    'pape_num_parabolas': ('positional_encoding_parameters', 'pape', 'num_parabolas'),
    'pape_num_positions': ('positional_encoding_parameters', 'pape', 'num_positions'),
    'pape_rotation_invariant': ('positional_encoding_parameters', 'pape', 'rotation_invariant'),
    'patch_size': ('image', 'patch_size'),
    'pos_embedding_type': ('image', 'pos_embedding_type'),
    'prediction_target': ('image', 'prediction_target'),
    'prms_partial_ratio': ('norm', 'partial_ratio'),
    'qk_clip_alpha': ('norm', 'qk_norm', 'clip_alpha'),
    'qk_clip_tau': ('norm', 'qk_norm', 'clip_tau'),
    'qk_norm_eps': ('norm', 'qk_norm', 'eps'),
    'qk_norm_method': ('norm', 'qk_norm', 'method'),
    'quack_tau': ('norm', 'qk_norm', 'quack_tau'),
    'residual_type': ('residuals', 'type'),
    'rope_base': ('positional_encoding_parameters', 'rope', 'base'),
    'rope_scaling': ('positional_encoding_parameters', 'rope', 'scaling'),
    'seg_head_type': ('image', 'seg_head_type'),
    'seg_l2_blocks': ('image', 'seg_l2_blocks'),
    'seg_mask_annealing': ('image', 'seg_mask_annealing'),
    'seg_num_queries': ('image', 'seg_num_queries'),
    'sinusoidal_base': ('positional_encoding_parameters', 'sinusoidal', 'base'),
    'sinusoidal_max_len': ('positional_encoding_parameters', 'sinusoidal', 'max_len'),
    'sinusoidal_scale': ('positional_encoding_parameters', 'sinusoidal', 'scale'),
    'ssmax_s_init': ('ssmax', 's_init'),
    'to_grayscale': ('image', 'to_grayscale'),
    'top_k_experts': ('moe', 'top_k'),
    'use_bitnet': ('bitnet', 'enabled'),
    'use_bitnet_conv': ('bitnet', 'conv'),
    'use_dynamic_embedding': ('embedding', 'dynamic', 'enabled'),
    'use_embedding_conv': ('embedding', 'conv', 'enabled'),
    'use_factorized_embedding': ('embedding', 'factorized', 'enabled'),
    'use_mhc': ('mhc', 'enabled'),
    'use_mixture_of_depths': ('mixture_of_depths', 'enabled'),
    'use_moe': ('moe', 'enabled'),
    'use_qk_norm': ('norm', 'qk_norm', 'enabled'),
    'use_ssmax': ('ssmax', 'enabled'),
    'vocab_size': ('dims', 'vocab_size'),
}


_COMPONENT_ALIASES_CACHE: Optional[Dict[str, Tuple[str, str]]] = None


def _component_alias_map() -> Dict[str, Tuple[str, str]]:
    """Return the legacy flat mixer-name -> ``(mixer, param)`` map.

    Returns:
        A mapping resolved once from the mixer specs.
    """
    global _COMPONENT_ALIASES_CACHE
    if _COMPONENT_ALIASES_CACHE is None:
        _COMPONENT_ALIASES_CACHE = {
            key: value[0] for key, value in _component_flat_aliases().items()
        }
    return _COMPONENT_ALIASES_CACHE


class _LazyComponentAliases(dict):
    """Dict view that materialises the component alias map on first access."""

    def _ensure(self) -> None:
        if not super().__len__():
            super().update(_component_alias_map())

    def __getitem__(self, key):
        self._ensure()
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._ensure()
        return super().get(key, default)

    def __contains__(self, key):
        self._ensure()
        return super().__contains__(key)


_COMPONENT_ALIASES: Dict[str, Tuple[str, str]] = _LazyComponentAliases()

_FIELD_NAMES = {f.name for f in dataclasses.fields(FrankensteinModelConfig)}


def _default_classification_head() -> "ClassificationHeadConfig":
    """Return a fresh default classification-head block."""
    return ClassificationHeadConfig()


def _default_block_for(name: str) -> Any:
    """Return a fresh default block for an optional modality attribute.

    Args:
        name: Attribute name on the config (``image`` / ``audio``).

    Returns:
        A default block instance.
    """
    return ImageConfig() if name == "image" else AudioConfig()


def flat_name_set() -> frozenset:
    """Return every accepted historical flat configuration name.

    Returns:
        A frozenset of flat names that :meth:`FrankensteinModelConfig.from_flat`
        accepts (structural aliases, nested fields and per-component knobs).
    """
    return frozenset(
        set(_FLAT_ALIASES)
        | set(_component_flat_aliases())
        | set(_FIELD_NAMES)
        | {"num_classes", "pooling_mode", "model_class", "token_vocab"}
        | {f"{name}_use_pe" for name in ()}
    )
