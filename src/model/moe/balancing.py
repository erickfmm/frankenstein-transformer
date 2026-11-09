"""Load-balancing losses, metrics and expert-bias updates for MoE layers.

Two independent balancing families live here:

* the auxiliary loss ``E · Σ_e f_e · P_e`` (Switch / DeepSeekMoE), computable
  over a micro-batch or per sequence, and
* the auxiliary-loss-free expert bias (DeepSeek-V3 ``noaux_tc``), which is
  applied by :class:`~src.model.moe.router.MoERouter`.

The module also defines ``MaxVio`` — how far the busiest expert's load sits
above the mean, ``(max_e c_e − mean) / mean`` — and the helpers that aggregate
per-layer MoE statistics and losses for the model forward.

References:
    Lepikhin et al., *GShard*, arXiv:2006.16668.
    Fedus et al., *Switch Transformers*, arXiv:2101.03961.
    DeepSeek-AI, *DeepSeekMoE*, arXiv:2401.06066.
    Wang et al., *Auxiliary-Loss-Free Load Balancing*, arXiv:2408.15664.
    Qiu et al., *Demons in the Detail*, arXiv:2501.11873.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch


def load_balancing_loss(
    probs: torch.Tensor,
    indices: torch.Tensor,
    num_experts: int,
    scope: str = "batch",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the auxiliary load-balancing loss.

    Args:
        probs: Router probabilities of shape ``(B, S, E)`` (the per-expert
            distribution used for balancing).
        indices: Selected expert indices of shape ``(B, S, k)``.
        num_experts: Number of experts ``E``.
        scope: ``"batch"`` for one micro-batch or ``"sequence"`` for one loss
            per sequence averaged over the batch.

    Returns:
        Tuple ``(loss, counts)`` where ``counts`` is the total
        ``(E,)`` selection count tensor (detached).

    Raises:
        ValueError: If ``scope`` is not ``"batch"`` or ``"sequence"``.
    """
    if scope not in {"batch", "sequence"}:
        raise ValueError(
            f"Unknown MoE aux-loss scope {scope!r}; expected 'batch' or 'sequence'."
        )
    if scope == "sequence" and probs.dim() == 3:
        bsz, seq_len, _ = probs.shape
        top_k = indices.shape[-1]
        idx = indices.reshape(bsz, -1)
        counts = torch.zeros(
            bsz, num_experts, device=probs.device, dtype=probs.dtype
        ).scatter_add_(
            1, idx, torch.ones_like(idx, dtype=probs.dtype)
        )
        f = counts / max(1, seq_len * top_k)
        p = probs.mean(dim=1)
        per_sequence = num_experts * (f * p).sum(dim=-1)
        return per_sequence.mean(), counts.sum(dim=0).detach()

    flat_probs = probs.reshape(-1, num_experts)
    flat_idx = indices.reshape(-1)
    counts = torch.bincount(
        flat_idx, minlength=num_experts
    ).to(flat_probs.dtype)
    f = counts / max(1, flat_idx.numel())
    loss = num_experts * (f * flat_probs.mean(dim=0)).sum()
    return loss, counts.detach()


def max_vio(counts: torch.Tensor) -> float:
    """Return the maximal violation of a balanced expert load.

    Args:
        counts: Per-expert token counts of shape ``(E,)``.

    Returns:
        ``(max_e c_e − mean) / mean``; ``0`` means perfectly even.
    """
    counts = counts.to(torch.float64)
    mean = counts.mean().clamp_min(1e-12)
    return float(((counts.max() - mean) / mean).item())


def aggregate_moe_aux_losses(
    layers: Iterable[torch.nn.Module],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Aggregate per-layer MoE auxiliary losses and scalar statistics.

    Args:
        layers: The transformer layers; MoE layers expose ``moe_layer`` with
            ``last_auxiliary_losses`` / ``last_moe_stats``.

    Returns:
        Tuple ``(losses, stats)``. ``losses`` maps each loss name to the mean
        over the MoE layers that produced it; ``stats`` averages the scalar
        statistics (MaxVio, entropy, expert-norm ratio, ...).
    """
    losses: Dict[str, list] = {}
    stats: Dict[str, list] = {}
    for layer in layers:
        moe = getattr(layer, "moe_layer", None)
        if moe is None:
            continue
        for name, value in (getattr(moe, "last_auxiliary_losses", None) or {}).items():
            if torch.is_tensor(value):
                losses.setdefault(name, []).append(value)
        for name, value in (getattr(moe, "last_moe_stats", None) or {}).items():
            if isinstance(value, (int, float)):
                stats.setdefault(name, []).append(float(value))
    aggregated_losses = {
        name: torch.stack(values).mean() for name, values in losses.items()
    }
    aggregated_stats = {
        name: sum(values) / len(values) for name, values in stats.items()
    }
    return aggregated_losses, aggregated_stats


@torch.no_grad()
def update_moe_expert_bias(model: torch.nn.Module) -> bool:
    """Apply the once-per-optimizer-step expert-bias update to every MoE layer.

    Args:
        model: Model that may contain
            :class:`~src.model.moe.layer.MoELayer` modules.

    Returns:
        ``True`` when at least one layer updated its bias.
    """
    from .layer import MoELayer  # local import avoids a cycle

    updated = False
    for module in model.modules():
        if isinstance(module, MoELayer):
            updated = module.update_expert_bias() or updated
    return updated


__all__ = [
    "load_balancing_loss",
    "max_vio",
    "aggregate_moe_aux_losses",
    "update_moe_expert_bias",
]
