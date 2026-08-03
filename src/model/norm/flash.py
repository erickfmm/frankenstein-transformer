"""FlashNorm: weightless RMSNorm and fused norm-then-linear layers.

Implements the three identities of FlashNorm (Graef et al., 2026,
arXiv:2407.09577):

* **Proposition 1 (weight folding).** Given a preceding RMSNorm with
  per-dimension scale ``g`` and a subsequent ``nn.Linear`` with weight
  ``W``, the scale can be absorbed into the linear by setting
  ``W* = diag(g) W`` — the normalization becomes *weightless*.

* **Proposition 2 (deferred normalization).** For a bias-free linear,
  ``(a / RMS(a)) W* = (a W*) / RMS(a)`` — the matrix multiplication and
  the RMS reduction are independent and can execute in parallel on
  matrix and vector units; only a single vector--scalar multiply is
  needed afterwards.

* **Proposition 3 (RMS cancellation).** When an ``RMSNorm -> Linear ->
  RMSNorm`` pattern exists (e.g., QKV-normalization, MLA latent norm),
  the first RMSNorm is redundant under exact scale invariance and can
  be dropped entirely.

This module provides:

* :class:`FlashNorm` — standalone weightless RMSNorm (drop-in via
  ``get_norm``). Optional ``partial_ratio`` composes the pRMSNorm trick
  with Prop. 1.
* :class:`FlashNormLinear` — fused ``FlashNorm + nn.Linear`` pair that
  applies Prop. 2 (deferred scalar RMS) on the bias-free path.
* :class:`FlashNormBitLinear` — BitLinear-compatible variant. Folding
  floating-point ``g`` into ternary weights would break quantization,
  so this layer falls back to the sequential ``FlashNorm -> BitLinear``
  composition (matmul and RMS still run in parallel; only the deferred
  rewrite is skipped).
* :func:`fold_norm_weights` — utility that performs post-hoc Prop. 1
  weight folding on a constructed ``nn.Linear``/``BitLinear``.

References:
    Graef, Makraduli, Wasielewski & Clapp (2026). "FlashNorm: Fast
    Normalization for Transformers." arXiv:2407.09577.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashNorm(nn.Module):
    """Weightless Root Mean Square Layer Normalization.

    Applies RMS normalization without any learnable per-dimension scale
    (Prop. 1 of FlashNorm — the scale ``g`` is meant to be folded into
    the subsequent linear layer)::

        y_i = x_i / RMS(x),   RMS(x) = sqrt( (1/n) * sum_j x_j^2 + eps )

    When ``partial_ratio`` is greater than 0, the RMS is estimated from
    only the first ``k = ceil(dim * partial_ratio)`` elements (the
    pRMSNorm trick composed with the FlashNorm weightless property).

    Args:
        dim: Number of features in the input (normalized dimension).
        eps: Small constant for numerical stability. Defaults to ``1e-6``.
        partial_ratio: Fraction of dimensions used for RMS estimation.
            ``0.0`` (default) uses all dimensions (standard FlashNorm).
            Values in ``(0, 1]`` activate the partial-RMS variant.

    Attributes:
        eps: Epsilon value for numerical stability.
        partial_ratio: Fraction of dimensions used for RMS estimation.
        k: Number of dimensions used for RMS estimation.

    References:
        Graef et al. (2026), "FlashNorm", arXiv:2407.09577.
        Zhang & Sennrich (2019), "RMSNorm", arXiv:1910.07467.
    """

    def __init__(self, dim: int, eps: float = 1e-6, partial_ratio: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.partial_ratio = float(partial_ratio)
        if self.partial_ratio > 0.0:
            self.k = max(1, math.ceil(self.dim * self.partial_ratio))
        else:
            self.k = self.dim

    def extra_repr(self) -> str:
        parts = [f"dim={self.dim}", f"eps={self.eps}"]
        if self.partial_ratio > 0.0:
            parts.append(f"partial_ratio={self.partial_ratio}")
        return ", ".join(parts)

    def rms_inv(self, x: torch.Tensor) -> torch.Tensor:
        """Return the per-token reciprocal RMS scalar ``1 / sqrt(mean(x^2) + eps)``.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Tensor of shape ``(..., 1)`` containing the reciprocal RMS
            for each token. Independent of any subsequent matmul — this
            is the quantity that Prop. 2 allows to execute in parallel
            with the matrix multiplication.
        """
        if self.partial_ratio > 0.0:
            ms = x[..., : self.k].pow(2).mean(dim=-1, keepdim=True)
        else:
            ms = x.pow(2).mean(dim=-1, keepdim=True)
        return torch.rsqrt(ms + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply weightless RMS normalization.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Tensor of same shape as ``x``, rescaled by the reciprocal
            RMS (no learnable parameters).
        """
        return x * self.rms_inv(x)


def fold_norm_weights(linear: nn.Linear, weight: torch.Tensor) -> None:
    """Fold an RMSNorm per-dim scale into a linear layer's weights (Prop. 1).

    Performs the in-place rewrite ``W* = diag(g) W`` so that the
    preceding RMSNorm's ``g`` becomes redundant (can be set to 1 or
    removed). For ``nn.Linear``, ``weight`` has shape ``(out, in)`` and
    the scale ``g`` has shape ``(in,)``; each column ``i`` of ``W`` is
    multiplied by ``g[i]``. If ``linear`` has a bias, the bias is left
    unchanged (Prop. 1 holds with or without bias).

    Args:
        linear: A ``nn.Linear`` (or subclass such as ``BitLinear``) whose
            ``weight`` tensor will be modified in place.
        weight: The RMSNorm per-dimension scale ``g`` of shape
            ``(in_features,)``. Must match ``linear.in_features``.

    Raises:
        ValueError: If ``weight`` shape is incompatible with
            ``linear.in_features``.
    """
    if weight.shape[0] != linear.in_features:
        raise ValueError(
            f"weight (g) has shape {tuple(weight.shape)} but linear.in_features="
            f"{linear.in_features}; expected shape ({linear.in_features},)."
        )
    g = weight.to(linear.weight.device, dtype=linear.weight.dtype).view(1, -1)
    with torch.no_grad():
        linear.weight.mul_(g)


class FlashNormLinear(nn.Module):
    """Fused ``FlashNorm + nn.Linear`` applying Prop. 1 and Prop. 2.

    The module owns a :class:`FlashNorm` and an ``nn.Linear``. On the
    bias-free path, the forward pass applies Proposition 2 (deferred
    scalar normalization): the matmul of the *un-normalized* input with
    the linear weight is computed, and the per-token ``1/RMS`` scalar is
    applied to the matmul output. On hardware with distinct matrix and
    vector units, the matmul and the RMS reduction can execute in
    parallel; only a single vector--scalar multiply synchronizes them.

    When the wrapped linear has a bias, Prop. 2 does not hold exactly
    (the bias must be added after the RMS scalar — see Remark 1 in the
    paper), so the layer applies RMS to the input first, then runs the
    linear as normal. Matmul and RMS still parallelize, but the bias is
    added after the scalar multiply.

    The module is constructed *without* an RMSNorm per-dim scale ``g``
    (it is weightless by design). To fold an existing RMSNorm's ``g``
    into the wrapped linear, use :func:`fold_norm_weights` before
    wrapping, or :meth:`from_rmsnorm_and_linear`.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If ``True``, adds a learnable bias to the output. Defaults
            to ``False`` (matches the FlashNorm paper's bias-free
            assumption for full Prop. 2 deferral).
        partial_ratio: Optional pRMSNorm partial ratio forwarded to
            :class:`FlashNorm`. ``0.0`` (default) uses the full RMS.
        eps: Epsilon for the RMS reduction. Defaults to ``1e-6``.
        linear_cls: Linear layer class to wrap. Defaults to
            :class:`torch.nn.Linear`.

    Attributes:
        flash: The :class:`FlashNorm` instance.
        linear: The wrapped linear layer.

    References:
        Graef et al. (2026), "FlashNorm", arXiv:2407.09577, §3.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        partial_ratio: float = 0.0,
        eps: float = 1e-6,
        linear_cls: type = nn.Linear,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.flash = FlashNorm(self.in_features, eps=eps, partial_ratio=partial_ratio)
        self.linear = linear_cls(self.in_features, self.out_features, bias=bias)

    @classmethod
    def from_rmsnorm_and_linear(
        cls,
        rmsnorm: nn.Module,
        linear: nn.Linear,
        eps: Optional[float] = None,
        partial_ratio: Optional[float] = None,
    ) -> "FlashNormLinear":
        """Fold an existing RMSNorm's ``g`` into a linear's ``W`` (Prop. 1).

        Constructs a :class:`FlashNormLinear` that wraps the supplied
        linear (with its weights folded in place) and a fresh
        :class:`FlashNorm` carrying no learnable scale. The caller is
        responsible for detaching the original RMSNorm from the model
        graph after this folding.

        Args:
            rmsnorm: An RMSNorm-like module exposing a ``weight``
                parameter of shape ``(in_features,)`` and optionally
                ``eps`` / ``partial_ratio`` attributes (e.g.,
                :class:`src.model.norm.rms.RMSNorm`).
            linear: The ``nn.Linear`` (or subclass) whose weights will
                be folded in place.
            eps: Optional epsilon override. If ``None``, read from
                ``rmsnorm.eps`` (falling back to ``1e-6``).
            partial_ratio: Optional partial-ratio override. If ``None``,
                read from ``rmsnorm.partial_ratio`` (falling back to
                ``0.0``).

        Returns:
            A new :class:`FlashNormLinear` wrapping the folded linear.
        """
        if not hasattr(rmsnorm, "weight"):
            raise ValueError("rmsnorm must expose a `weight` parameter (the g vector).")
        fold_norm_weights(linear, rmsnorm.weight.data)
        eps_val = float(eps if eps is not None else getattr(rmsnorm, "eps", 1e-6))
        pr_val = float(
            partial_ratio if partial_ratio is not None else getattr(rmsnorm, "partial_ratio", 0.0)
        )
        out = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            partial_ratio=pr_val,
            eps=eps_val,
            linear_cls=type(linear),
        )
        # Reuse the folded linear's parameters instead of allocating new ones.
        out.linear = linear
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fused forward applying Prop. 2 (deferred RMS) when possible.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        rms_inv = self.flash.rms_inv(x)
        bias = getattr(self.linear, "bias", None)
        if bias is None and isinstance(self.linear, nn.Linear):
            # Prop. 2: matmul on un-normalized x, then apply scalar.
            out = F.linear(x, self.linear.weight)
            return out * rms_inv
        # Bias present or non-Linear: apply scalar first, then the linear.
        # (Matmul and RMS still parallelize; bias is added by the linear.)
        return self.linear(x * rms_inv)


class FlashNormBitLinear(nn.Module):
    """Fused ``FlashNorm + BitLinear`` pair (BitNet-aware variant).

    BitNet's ``BitLinear`` quantizes weights to *ternary* values
    ``{-1, 0, +1}`` (arXiv:2402.17764). Folding a floating-point
    per-dimension scale ``g`` into ternary weights would destroy the
    quantization, so this module does **not** apply Prop. 1 weight
    folding. Prop. 2 (deferred scalar) is also skipped because
    ``BitLinear`` applies its own internal LayerNorm before activation
    quantization, which needs the normalized input.

    The composition is therefore sequential: ``FlashNorm`` (model-level
    pre-norm) is applied first, then ``BitLinear`` runs its own
    internal LayerNorm + quantization pipeline. On hardware with
    distinct matrix/vector units the matmul inside ``BitLinear`` and
    the FlashNorm RMS reduction still execute in parallel; what is
    sacrificed is the *post-matmul* scalar multiply of Prop. 2.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If ``True``, adds a learnable bias. Defaults to ``False``
            (matches ``BitLinear`` default).
        partial_ratio: Optional pRMSNorm partial ratio forwarded to
            :class:`FlashNorm`. ``0.0`` (default) uses the full RMS.
        eps: Epsilon for the RMS reduction. Defaults to ``1e-6``.
        bitlinear_cls: ``BitLinear`` class to wrap. Defaults to the
            lazy import of ``src.model.attention.common.BitLinear``.

    Attributes:
        flash: The :class:`FlashNorm` instance.
        linear: The wrapped ``BitLinear``.

    Raises:
        ImportError: If ``bitlinear_cls`` is not provided and
            :mod:`src.model.attention.common` cannot be imported.

    References:
        Graef et al. (2026), "FlashNorm", arXiv:2407.09577.
        Ma et al. (2024), "The Era of 1-bit LLMs", arXiv:2402.17764.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        partial_ratio: float = 0.0,
        eps: float = 1e-6,
        bitlinear_cls: Optional[type] = None,
    ):
        super().__init__()
        if bitlinear_cls is None:
            try:
                from src.model.attention.common import BitLinear as _BitLinear

                bitlinear_cls = _BitLinear
            except ImportError as exc:  # pragma: no cover - defensive
                raise ImportError(
                    "FlashNormBitLinear requires `src.model.attention.common.BitLinear`; "
                    "either pass bitlinear_cls explicitly or ensure the package is importable."
                ) from exc
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.flash = FlashNorm(self.in_features, eps=eps, partial_ratio=partial_ratio)
        self.linear = bitlinear_cls(self.in_features, self.out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sequential FlashNorm → BitLinear forward.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        return self.linear(self.flash(x))


__all__ = [
    "FlashNorm",
    "FlashNormLinear",
    "FlashNormBitLinear",
    "fold_norm_weights",
]
