"""Declarative component specifications for the plugin registry.

A :class:`ComponentSpec` is the single source of truth for one pluggable
component: its registry name, lazy builder, parameter schema fragment,
runtime parameter defaults, bilingual documentation metadata and the
capability flags that used to live in separate hand-maintained lists
(``TRAINING_FREE_LAYERS``, ``CACHE_SPECS``, ``IMPLEMENTED_MIXERS``,
``QK_NORM_MIXERS``, ``OPTIMIZER_ALLOWED_KEYS``, …).

Every family has a frozen subclass with family-specific declarative fields.
The classes are intentionally data-only: they never import torch or any
model package, so schema assembly, plugin discovery and documentation
generation stay CPU- and torch-free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Tuple

__all__ = [
    "ComponentFamily",
    "CacheSpec",
    "ParamBinding",
    "ComponentSpec",
    "MixerSpec",
    "OptimizerSpec",
    "NormSpec",
    "ActivationSpec",
    "ResidualSpec",
    "PESpec",
    "EmbeddingSpec",
    "SPEC_CLASSES",
]

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
"""Registry names are snake_case and double as optimizer key prefixes."""


class ComponentFamily(str, Enum):
    """Families of pluggable components supported by the registry."""

    MIXER = "mixer"
    OPTIMIZER = "optimizer"
    NORM = "norm"
    ACTIVATION = "activation"
    RESIDUAL = "residual"
    POSITIONAL_ENCODING = "positional_encoding"
    EMBEDDING = "embedding"


@dataclass(frozen=True)
class CacheSpec:
    """Static KV-cache capability description for one mixer key.

    Moved verbatim from :mod:`src.model.cache.capabilities`; the registry is
    now the source of truth and the cache package re-exports derived views.

    Attributes:
        supported: Whether the mixer's source formulation admits a cache at
            all. ``False`` means the mixer always runs uncached (with a
            warning from ``config_loader`` when the cache is enabled).
        slot_type: Primary slot kind (``""`` when not supported).
        slot_types: Additional slot kinds (e.g. ``("state", "conv_tail")``
            for falcon-style mixers with a causal short conv).
        extras: Per-token metadata the mixer will store (documentation).
        notes: Paper-based rationale, surfaced in ``config_loader`` warnings
            and docs.
    """

    supported: bool
    slot_type: str = ""
    slot_types: Tuple[str, ...] = ()
    extras: Tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class ParamBinding:
    """Declarative constructor-argument binding for one optimizer suffix.

    Replaces one branch of the old ``build_optimizer`` ``if/elif`` chain.

    Attributes:
        coerce: ``(raw_value, default) -> typed_value`` coercion helper
            (e.g. :func:`src.model.optimizer.base.to_float`).
        default: Fallback used when the suffix is absent or uncoercible.
    """

    coerce: Callable[[Any, Any], Any]
    default: Any = None


@dataclass(frozen=True)
class ComponentSpec:
    """Declarative registration record for one pluggable component.

    Attributes:
        name: Registry key (snake_case; also the optimizer config prefix).
        family: Component family (fixed by the concrete subclass).
        builder: Lazy builder reference — a ``"module:attribute"`` string or
            a zero-argument callable returning the class/factory.
        description: English one-row description used by the generated schema
            tables and documentation.
        description_es: Spanish translation of ``description``.
        references: arXiv identifiers cited by the component
            (``"2604.09967"`` becomes ``(paper: arxiv:2604.09967)``).
        params_schema: JSON-Schema fragment describing the component's
            parameters — either an inline mapping or a
            ``"schema/_model/<file>.yaml#/json/pointer"`` reference resolved
            relative to the ``src/`` directory.
        defaults: Runtime parameter defaults, applied when the YAML block
            omits a key. Schema fragments may declare their own defaults too;
            the two are cross-checked by the consistency gate.
    """

    name: str
    family: ComponentFamily
    builder: Any
    description: str
    description_es: str
    references: Tuple[str, ...] = ()
    params_schema: Any = None
    defaults: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the declarative record.

        Raises:
            InvalidSpecError: If the name, builder or documentation metadata
                is malformed.
        """
        from .errors import InvalidSpecError

        if not isinstance(self.name, str) or not _NAME_RE.match(self.name):
            raise InvalidSpecError(
                f"Invalid component name {self.name!r}: registry names must "
                "match ^[a-z][a-z0-9_]*$"
            )
        if not isinstance(self.family, ComponentFamily):
            raise InvalidSpecError(
                f"Component {self.name!r}: family must be a ComponentFamily, "
                f"got {type(self.family).__name__}"
            )
        if not (isinstance(self.builder, str) or callable(self.builder)):
            raise InvalidSpecError(
                f"Component {self.name!r}: builder must be a 'module:attr' "
                "string or a callable"
            )
        if not str(self.description or "").strip():
            raise InvalidSpecError(
                f"Component {self.name!r}: description is required"
            )
        if not str(self.description_es or "").strip():
            raise InvalidSpecError(
                f"Component {self.name!r}: description_es is required"
            )
        if self.params_schema is not None and not (
            isinstance(self.params_schema, Mapping)
            or isinstance(self.params_schema, str)
        ):
            raise InvalidSpecError(
                f"Component {self.name!r}: params_schema must be a mapping, a "
                "'file.yaml#/pointer' string, or None"
            )
        if not isinstance(self.defaults, Mapping):
            raise InvalidSpecError(
                f"Component {self.name!r}: defaults must be a mapping"
            )


@dataclass(frozen=True)
class MixerSpec(ComponentSpec):
    """Sequence-mixer component (``model.dims.layer_pattern``).

    Attributes:
        training_free: Mixer defines only an inference-time heuristic; in
            training it runs the dense surrogate (``fasa``, ``sparge``).
        needs_pos_encoder: Default for the per-mixer ``use_pe`` opt-out flag.
        cache: Static KV-cache capability (``None`` = never cached).
        cache_implemented: Incremental branch implemented and parity-tested.
        qk_norm_supported: Supports forward-time QK normalization / QuacK.
        qk_clip_supported: Forms per-head logits (supports QK-Clip).
        aliases: Alternative names accepted for the same mixer.
    """

    family: ComponentFamily = field(default=ComponentFamily.MIXER, init=False)
    training_free: bool = False
    needs_pos_encoder: bool = True
    cache: Optional[CacheSpec] = None
    cache_implemented: bool = False
    qk_norm_supported: bool = False
    qk_clip_supported: bool = False
    aliases: Tuple[str, ...] = ()


@dataclass(frozen=True)
class OptimizerSpec(ComponentSpec):
    """Optimizer component (``training.optimizer.optimizer_class``).

    Attributes:
        param_bindings: Optimizer-specific suffix bindings (without the
            ``"<name>-"`` prefix), each a :class:`ParamBinding`.
        supports_shared_per_group_suffixes: Accepts the shared
            ``lr_*``/``wd_*``/``betas_*``/``eps_*`` suffixes.
        group_policy: ``"adaptive"`` adds per-group ``betas``/``eps``;
            ``"sgd"`` only routes ``lr``/``wd``.
        closure_based: Trainer must supply a closure (``eve``, ``ea4llm``).
        sbert_allowed: Selectable for SBERT fine-tuning.
        requires_grad_accum_one: Requires ``gradient_accumulation_steps: 1``.
        custom_build: Optional escape hatch
            ``(spec, groups, scoped, kwargs) -> Optimizer`` for genuinely
            special constructors.
    """

    family: ComponentFamily = field(default=ComponentFamily.OPTIMIZER, init=False)
    param_bindings: Mapping[str, ParamBinding] = field(default_factory=dict)
    supports_shared_per_group_suffixes: bool = True
    group_policy: str = "adaptive"
    closure_based: bool = False
    sbert_allowed: bool = True
    requires_grad_accum_one: bool = False
    custom_build: Optional[Callable[..., Any]] = None


@dataclass(frozen=True)
class NormSpec(ComponentSpec):
    """Normalization component (``model.norm.type``).

    Attributes:
        supports_ffn_fusion: The norm can be folded into the FFN's first
            projection (FlashNorm Prop. 2 pattern).
        reads_partial_ratio: The builder consumes ``norm.partial_ratio`` /
            ``norm.flashnorm_partial_ratio``.
    """

    family: ComponentFamily = field(default=ComponentFamily.NORM, init=False)
    supports_ffn_fusion: bool = False
    reads_partial_ratio: bool = False


@dataclass(frozen=True)
class ActivationSpec(ComponentSpec):
    """Activation component (``model.ffn.activation``).

    Attributes:
        is_glu: Gated FFN variant — built by ``GatedFFN`` instead of
            :func:`get_activation`.
    """

    family: ComponentFamily = field(default=ComponentFamily.ACTIVATION, init=False)
    is_glu: bool = False


@dataclass(frozen=True)
class ResidualSpec(ComponentSpec):
    """Residual-connection strategy (``model.residuals.type``)."""

    family: ComponentFamily = field(default=ComponentFamily.RESIDUAL, init=False)


@dataclass(frozen=True)
class PESpec(ComponentSpec):
    """Positional-encoding component (``model.positional_encoding``).

    Attributes:
        aliases: Alternative names (``"none"`` for ``nope``).
        params_key: Key under ``model.positional_encoding_parameters`` that
            holds this encoding's knobs.
    """

    family: ComponentFamily = field(
        default=ComponentFamily.POSITIONAL_ENCODING, init=False
    )
    aliases: Tuple[str, ...] = ()
    params_key: str = ""


@dataclass(frozen=True)
class EmbeddingSpec(ComponentSpec):
    """Token-embedding variant (``model.embedding.type``)."""

    family: ComponentFamily = field(default=ComponentFamily.EMBEDDING, init=False)


SPEC_CLASSES: Mapping[ComponentFamily, type] = {
    ComponentFamily.MIXER: MixerSpec,
    ComponentFamily.OPTIMIZER: OptimizerSpec,
    ComponentFamily.NORM: NormSpec,
    ComponentFamily.ACTIVATION: ActivationSpec,
    ComponentFamily.RESIDUAL: ResidualSpec,
    ComponentFamily.POSITIONAL_ENCODING: PESpec,
    ComponentFamily.EMBEDDING: EmbeddingSpec,
}
"""Canonical spec class per family, used by the registration helpers."""
