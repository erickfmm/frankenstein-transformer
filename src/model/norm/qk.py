"""Query–key normalization and attention-logit stabilization.

Three families, all selected through ``model.norm.qk_norm.method``:

* **Forward-time per-head QK normalization** — ``rms_norm``, ``layer_norm``
  and ``l2_norm``. Queries and keys are normalized along the per-head feature
  dimension before the dot product (Henry et al. 2020, arXiv:2010.04245).
  The RMS variant is the exact formulation analyzed for Multi-Head Latent
  Attention by Han et al. (2026, arXiv:2606.16310).
* **QK-Clip** — ``qk_clip``. After every optimizer step the query/key
  projection weights are rescaled per attention head by
  ``γ_h = min(1, τ / S_max^h)``, where ``S_max^h`` is the maximum pre-softmax
  logit of head ``h`` in that step (Kimi Team 2025, arXiv:2507.20534, §2.1 and
  Algorithm 1, "MuonClip").
* **QuacK** — ``quack``. Parameter-dependent learning rates for the query/key
  weights, derived from μP-style control of the *change* in attention logits
  (Anson & Aitchison 2025, arXiv:2511.21377).

Only mixers that materialize explicit per-head query/key projections can
support these mechanisms; :data:`QK_NORM_MIXERS` is the capability set used by
the configuration validator.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Forward-time normalization methods (add a module and, for the first two,
#: learnable parameters).
FORWARD_METHODS = frozenset({"rms_norm", "layer_norm", "l2_norm"})

#: Training-time methods (no forward module; act on weights or learning rates).
TRAINING_METHODS = frozenset({"qk_clip", "quack"})

#: Every method accepted by ``model.norm.qk_norm.method``.
QK_NORM_METHODS = FORWARD_METHODS | TRAINING_METHODS

#: Mixers that expose explicit per-head Q/K projections and therefore support
#: every QK stabilization method. Kept in sync with the mixers that call
#: :func:`attach_qk_norm` and record per-head max logits.
QK_NORM_MIXERS = frozenset(
    {
        "standard_attn",
        "sigmoid_attn",
        "gqa_attn",
        "mla_attn",
        "swa_attn",
        "longformer_attn",
        "fastconformer_attn",
        "bigbird_attn",
        "sparse_transformer_attn",
        "gta_attn",
        "iha_attn",
        "mtla_attn",
        "gma_attn",
        "nsa_attn",
        "sparsek_attn",
        "msa_attn",
        "sparda_attn",
        "conformer_attn",
        "fox_attn",
        "gated_softmax_attn",
        "titan_attn",
        "gqla_attn",
        "mlra_attn",
        "memory_attn",
        "lsa_attn",
    }
)

#: Mixers that additionally form per-head dot-product logits and therefore
#: support QK-Clip (which needs the per-head maximum pre-softmax logit).
#: ``gma_attn`` routes tokens through Gaussian mixture responsibilities rather
#: than a QK dot product, so QK-Clip is not meaningful there.
QK_CLIP_MIXERS = QK_NORM_MIXERS - {"gma_attn"}


class QKNorm(nn.Module):
    """Per-head query/key normalization applied before the dot product.

    Normalizes the last dimension of ``q`` and ``k`` (shape ``(B, H, S, D)``
    or ``(B, S, H, D)`` — anything ending in the head dimension):

    * ``rms_norm`` (arXiv:2010.04245, arXiv:2606.16310)::

          y = γ ⊙ x / sqrt(mean(x²) + ε)

    * ``layer_norm`` (arXiv:2010.04245, "SAN" variant): mean/variance
      normalization with learnable gain ``γ`` and bias ``β``.

    * ``l2_norm`` (arXiv:2010.04245): weightless ``y = x / max(‖x‖, ε)``.

    The per-dimension affine parameters are shared across heads and broadcast
    over the head axis, matching the common RMSNorm implementation (and
    Eq. 1 of arXiv:2606.16310). Reductions run in float32 for mixed-precision
    safety, mirroring :func:`falcon_qk_norm`; the output keeps the input dtype.

    Args:
        head_dim: Per-head feature dimension ``D``.
        method: One of ``"rms_norm"``, ``"layer_norm"``, ``"l2_norm"``.
        eps: Numerical stabilizer. Defaults to ``1e-6``.

    Raises:
        ValueError: If ``method`` is not a forward-time method, or ``eps`` is
            not positive.

    Attributes:
        method: The normalization method string.
        q_weight: Learnable query scale ``(head_dim,)`` (``None`` for
            ``l2_norm``).
        k_weight: Learnable key scale ``(head_dim,)`` (``None`` for
            ``l2_norm``).
        q_bias: Learnable query bias ``(head_dim,)`` (only ``layer_norm``).
        k_bias: Learnable key bias ``(head_dim,)`` (only ``layer_norm``).

    References:
        Henry, Dachapally, Pawar & Chen (2020). "Query-Key Normalization for
        Transformers." arXiv:2010.04245.
        Han, Zhao, Zhou, Li & Sun (2026). "QK-Normed MLA: QK Normalization
        Without Full Key Caching." arXiv:2606.16310.
    """

    def __init__(self, head_dim: int, method: str = "rms_norm", eps: float = 1e-6):
        super().__init__()
        method = str(method).lower()
        if method not in FORWARD_METHODS:
            raise ValueError(
                f"QKNorm only implements forward-time methods {sorted(FORWARD_METHODS)}, "
                f"got {method!r}"
            )
        if float(eps) <= 0.0:
            raise ValueError(f"QKNorm eps must be > 0, got {eps!r}")
        self.head_dim = int(head_dim)
        self.method = method
        self.eps = float(eps)

        if method == "l2_norm":
            # Weightless: register None so state_dict/parameter counts stay clean.
            self.register_parameter("q_weight", None)
            self.register_parameter("k_weight", None)
            self.register_parameter("q_bias", None)
            self.register_parameter("k_bias", None)
        else:
            self.q_weight = nn.Parameter(torch.ones(self.head_dim))
            self.k_weight = nn.Parameter(torch.ones(self.head_dim))
            if method == "layer_norm":
                self.q_bias = nn.Parameter(torch.zeros(self.head_dim))
                self.k_bias = nn.Parameter(torch.zeros(self.head_dim))
            else:
                self.register_parameter("q_bias", None)
                self.register_parameter("k_bias", None)

    def _normalize(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Normalize ``x`` along its last dimension, preserving the dtype."""
        x32 = x.float()
        if self.method == "l2_norm":
            return F.normalize(x32, dim=-1, eps=self.eps).to(x.dtype)
        if self.method == "rms_norm":
            y = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return (y * weight.float()).to(x.dtype)
        # layer_norm
        y = F.layer_norm(
            x32,
            (self.head_dim,),
            weight.float(),
            None if bias is None else bias.float(),
            self.eps,
        )
        return y.to(x.dtype)

    def normalize_q(self, q: torch.Tensor) -> torch.Tensor:
        """Normalize queries ``q`` of shape ``(..., head_dim)``."""
        return self._normalize(q, self.q_weight, self.q_bias)

    def normalize_k(self, k: torch.Tensor) -> torch.Tensor:
        """Normalize keys ``k`` of shape ``(..., head_dim)``."""
        return self._normalize(k, self.k_weight, self.k_bias)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize a ``(q, k)`` pair.

        Args:
            q: Queries ``(..., head_dim)``.
            k: Keys ``(..., head_dim)`` (same layout as ``q``).

        Returns:
            Tuple ``(q_norm, k_norm)`` with the input dtype and shape.
        """
        return self.normalize_q(q), self.normalize_k(k)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, method={self.method!r}, eps={self.eps}"


def make_qk_norm(config, head_dim: int) -> Optional[QKNorm]:
    """Build the forward-time QK-normalization module, or ``None``.

    Returns ``None`` when QK stabilization is disabled or when the selected
    method acts during training rather than in the forward pass.

    Args:
        config: Model configuration with ``use_qk_norm``, ``qk_norm_method``
            and ``qk_norm_eps`` attributes.
        head_dim: Per-head feature dimension.

    Returns:
        A :class:`QKNorm` instance for ``rms_norm`` / ``layer_norm`` /
        ``l2_norm``, otherwise ``None``.
    """
    if not bool(getattr(config, "use_qk_norm", False)):
        return None
    method = str(getattr(config, "qk_norm_method", "rms_norm")).lower()
    if method not in FORWARD_METHODS:
        return None
    return QKNorm(
        head_dim,
        method=method,
        eps=float(getattr(config, "qk_norm_eps", 1e-6)),
    )


def attach_qk_norm(module: nn.Module, config, head_dim: int) -> None:
    """Attach QK-norm state to a mixer module.

    Sets three attributes on ``module``:

    * ``qk_norm`` — a :class:`QKNorm` instance or ``None`` (forward methods).
    * ``_record_qk_max_logits`` — whether to capture per-head maximum
      pre-softmax logits for QK-Clip.
    * ``last_qk_max_logits`` — the most recent per-head maxima (``(H,)``) or
      ``None``.

    Args:
        module: Mixer module being initialized.
        config: Model configuration.
        head_dim: Per-head feature dimension.
    """
    enabled = bool(getattr(config, "use_qk_norm", False))
    method = str(getattr(config, "qk_norm_method", "rms_norm")).lower()
    module.qk_norm_capable = True
    module.qk_norm = make_qk_norm(config, head_dim)
    module._record_qk_max_logits = enabled and method == "qk_clip"
    module.last_qk_max_logits = None


def record_qk_max_logits(mixer: nn.Module, scores: torch.Tensor) -> None:
    """Record per-head maximum pre-softmax logits for QK-Clip.

    No-op unless the mixer enabled recording in :func:`attach_qk_norm`.
    Masked entries (``-inf``) never win the maximum, so causal masks are
    harmless.

    Args:
        mixer: Mixer module owning the scores.
        scores: Attention logits ``(B, H, S, S_kv)`` (already scaled, before
            masking/softmax).
    """
    if not getattr(mixer, "_record_qk_max_logits", False):
        return
    with torch.no_grad():
        new_max = scores.detach().amax(dim=(0, 2, 3)).float()
        previous = getattr(mixer, "last_qk_max_logits", None)
        # Accumulate across sub-paths (e.g. NSA's compressed/selected/window
        # branches) so no branch under-reports the step maximum. The value is
        # cleared by :func:`apply_qk_clip` after each optimizer step.
        mixer.last_qk_max_logits = (
            new_max if previous is None else torch.maximum(previous, new_max)
        )


# ---------------------------------------------------------------------------
# QK-Clip (arXiv:2507.20534)
# ---------------------------------------------------------------------------


def _qk_weight_pair(module: nn.Module) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return ``(W_q, W_k)`` for a QK-capable mixer, or ``None``.

    ``W_k`` is ``k_proj.weight`` for materialized-key attention and
    ``uk_proj.weight`` for latent attention (MLA/MLRA/GQLA), whose keys are
    decompressed from the cached latent every step.
    """
    q_proj = getattr(module, "q_proj", None)
    k_proj = getattr(module, "k_proj", None) or getattr(module, "uk_proj", None)
    if q_proj is None or k_proj is None:
        return None
    q_weight = getattr(q_proj, "weight", None)
    k_weight = getattr(k_proj, "weight", None)
    if q_weight is None or k_weight is None:
        return None
    return q_weight, k_weight


def _rescale_rows_per_head(
    weight: torch.Tensor, factors: torch.Tensor, num_heads: int, head_dim: int
) -> None:
    """Multiply ``(num_heads * head_dim, in)`` weight rows in place per head."""
    view = weight.detach().view(num_heads, head_dim, -1)
    view.mul_(factors.detach().to(view.dtype).view(num_heads, 1, 1))


def _kv_head_factors(gamma_q: torch.Tensor, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    """Map per-query-head ``γ`` to per-KV-head ``γ`` (grouped-query attention).

    A shared KV head bounds every query head that reads it, so the most
    restrictive (smallest) factor among the group is used.
    """
    if num_kv_heads == num_heads:
        return gamma_q
    group_size = num_heads // num_kv_heads
    return gamma_q.view(num_kv_heads, group_size).amin(dim=1)


@torch.no_grad()
def apply_qk_clip(
    model: nn.Module, tau: float = 100.0, alpha: float = 0.5
) -> int:
    """Rescale query/key weights per head after an optimizer step.

    Implements QK-Clip (arXiv:2507.20534, Algorithm 1): for every QK-capable
    mixer that recorded per-head maxima ``S_max^h`` during the step,

    .. math::

        \\gamma_h = \\min(1, \\tau / S_\\max^h), \\qquad
        W_Q \\leftarrow \\gamma_h^{\\alpha} W_Q, \\qquad
        W_K \\leftarrow \\gamma_h^{1-\\alpha} W_K,

    applied to the per-head row blocks of the projection matrices. Heads
    already below ``tau`` get ``γ_h = 1`` and are untouched, so the mechanism
    deactivates itself once training stabilizes. The recorded maxima are
    cleared after use so a stale value is never reapplied.

    For grouped-query attention the key projection has fewer heads than the
    query projection; each KV head receives the smallest factor of the query
    heads that read it, which bounds every logit in the group.

    Args:
        model: Model whose mixer modules hold ``last_qk_max_logits``.
        tau: Per-head logit threshold. Must be positive.
        alpha: Balancing exponent in ``[0, 1]``.

    Returns:
        Number of heads that were actually rescaled (``γ_h < 1``).

    Raises:
        ValueError: If ``tau <= 0`` or ``alpha`` is outside ``[0, 1]``.
    """
    if float(tau) <= 0.0:
        raise ValueError(f"qk_clip tau must be > 0, got {tau!r}")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"qk_clip alpha must be in [0, 1], got {alpha!r}")

    rescaled_heads = 0
    for module in model.modules():
        max_logits = getattr(module, "last_qk_max_logits", None)
        if max_logits is None:
            continue
        weights = _qk_weight_pair(module)
        if weights is None:
            module.last_qk_max_logits = None
            continue
        q_weight, k_weight = weights
        num_heads = int(getattr(module, "num_heads", 0))
        head_dim = int(getattr(module, "head_dim", 0))
        if num_heads <= 0 or head_dim <= 0:
            module.last_qk_max_logits = None
            continue

        gamma = torch.clamp(float(tau) / max_logits.clamp_min(1e-12), max=1.0)
        rescaled_heads += int((gamma < 1.0).sum().item())
        if bool((gamma < 1.0).any()):
            num_kv_heads = int(getattr(module, "num_kv_heads", num_heads))
            _rescale_rows_per_head(
                q_weight, gamma.pow(float(alpha)), num_heads, head_dim
            )
            kv_gamma = _kv_head_factors(gamma, num_heads, num_kv_heads)
            _rescale_rows_per_head(
                k_weight, kv_gamma.pow(1.0 - float(alpha)), num_kv_heads, head_dim
            )
        module.last_qk_max_logits = None
    return rescaled_heads


# ---------------------------------------------------------------------------
# QuacK (arXiv:2511.21377)
# ---------------------------------------------------------------------------


def _frobenius_norm(weight: torch.Tensor) -> float:
    """Frobenius norm of a weight matrix in float32."""
    return float(weight.detach().float().norm())


class QuacKController:
    """Per-parameter query/key learning-rate modulation (arXiv:2511.21377).

    Stores the initial Frobenius norms ``‖W_Q‖_init`` and ``‖W_K‖_init`` and,
    before each optimizer step, rewrites the learning rates of the dedicated
    query/key parameter groups::

        lr(W_Q) = τ · η · ‖W_K‖_init / ‖W_K‖
        lr(W_K) = τ · η · ‖W_Q‖_init / ‖W_Q‖

    where ``η`` is the scheduler's current base learning rate for the group
    (read from ``group["lr"]`` immediately before the step, because the
    scheduler rewrites it after every step).

    The controller **splits** the existing optimizer parameter groups so the
    query and key weights of each mixer live in their own groups, preserving
    every other group option (weight decay, betas, eps, ...). It must be
    constructed after the optimizer and before its scheduler, so
    ``LambdaLR`` initializes ``initial_lr`` for the split groups.

    Args:
        model: Model with QK-capable mixers.
        optimizer: Optimizer whose parameter groups are split and updated.
        tau: Initial relative learning rate. Must be positive.

    Raises:
        ValueError: If ``tau <= 0``.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, tau: float = 1.0):
        if float(tau) <= 0.0:
            raise ValueError(f"QuacK tau must be > 0, got {tau!r}")
        self.tau = float(tau)
        self.optimizer = optimizer
        self._entries: List[Dict[str, object]] = []
        self._split_groups(model, optimizer)

    def _split_groups(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Move query/key weights into dedicated groups and cache init norms."""
        # Map every parameter to the group that currently holds it.
        param_group_index = {}
        for index, group in enumerate(optimizer.param_groups):
            for param in group["params"]:
                param_group_index[param] = index

        # Collect (q_param, k_param) pairs of QK-capable mixers.
        pairs = []
        for module in model.modules():
            if not getattr(module, "qk_norm_capable", False):
                continue
            weights = _qk_weight_pair(module)
            if weights is None:
                continue
            q_weight, k_weight = weights
            pairs.append((q_weight, k_weight))

        if not pairs:
            return

        q_params = {id(pair[0]) for pair in pairs}
        k_params = {id(pair[1]) for pair in pairs}
        special = q_params | k_params

        new_groups: List[Dict[str, object]] = []
        q_param_to_group: Dict[int, Dict[str, object]] = {}
        k_param_to_group: Dict[int, Dict[str, object]] = {}

        for group in optimizer.param_groups:
            options = {key: value for key, value in group.items() if key != "params"}
            rest = [p for p in group["params"] if id(p) not in special]
            if rest:
                new_groups.append({**options, "params": rest})
            for param in group["params"]:
                if id(param) in q_params:
                    q_group = {**options, "params": [param], "name": f"{options.get('name', 'group')}_qk_q"}
                    new_groups.append(q_group)
                    q_param_to_group[id(param)] = q_group
                elif id(param) in k_params:
                    k_group = {**options, "params": [param], "name": f"{options.get('name', 'group')}_qk_k"}
                    new_groups.append(k_group)
                    k_param_to_group[id(param)] = k_group

        optimizer.param_groups = new_groups

        for q_weight, k_weight in pairs:
            q_group = q_param_to_group.get(id(q_weight))
            k_group = k_param_to_group.get(id(k_weight))
            if q_group is None or k_group is None:
                continue
            self._entries.append(
                {
                    "q": q_weight,
                    "k": k_weight,
                    "q_group": q_group,
                    "k_group": k_group,
                    "init_q": _frobenius_norm(q_weight),
                    "init_k": _frobenius_norm(k_weight),
                }
            )

    @property
    def num_pairs(self) -> int:
        """Number of query/key weight pairs under modulation."""
        return len(self._entries)

    def step(self) -> None:
        """Update the query/key learning rates for the next optimizer step."""
        with torch.no_grad():
            for entry in self._entries:
                q_group = entry["q_group"]
                k_group = entry["k_group"]
                # ``group["lr"]`` is the scheduler's base rate: the scheduler
                # rewrites it at the end of every step, before this hook runs.
                q_base = float(q_group["lr"])  # type: ignore[index]
                k_base = float(k_group["lr"])  # type: ignore[index]
                norm_q = max(_frobenius_norm(entry["q"]), 1e-12)  # type: ignore[arg-type]
                norm_k = max(_frobenius_norm(entry["k"]), 1e-12)  # type: ignore[arg-type]
                q_group["lr"] = self.tau * q_base * (float(entry["init_k"]) / norm_k)  # type: ignore[index]
                k_group["lr"] = self.tau * k_base * (float(entry["init_q"]) / norm_q)  # type: ignore[index]
