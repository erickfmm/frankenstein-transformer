"""Router scoring for Mixture-of-Experts layers.

The router turns each token into one logit per expert and then into the
weights that mix the selected experts' outputs. This module implements the
four choices that sit between the logits and the weights:

* the gating function (``softmax``, ``sigmoid`` or ``sqrt(softplus)``),
* the output scale (top-k renormalisation and ``routed_scaling_factor``),
* the sharpness controls (Switch-style input jitter and the router precision),
* the router z-loss that keeps the logits from drifting.

It also owns the auxiliary-loss-free expert bias used for load balancing: the
bias is added to the scores *before* selection only and is never part of the
mixing weights.

References:
    Fedus et al., *Switch Transformers*, arXiv:2101.03961.
    Zoph et al., *ST-MoE*, arXiv:2202.08906.
    DeepSeek-AI, *DeepSeek-V3*, arXiv:2412.19437.
    Wang et al., *Auxiliary-Loss-Free Load Balancing*, arXiv:2408.15664.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _score(logits: torch.Tensor, scoring: str) -> torch.Tensor:
    """Apply the configured gating function to router logits.

    Args:
        logits: Router logits of shape ``(..., num_experts)``.
        scoring: One of ``"softmax"``, ``"sigmoid"`` or ``"softplus_sqrt"``.

    Returns:
        Scores with the same shape as ``logits``.
    """
    if scoring == "softmax":
        return torch.softmax(logits, dim=-1)
    if scoring == "sigmoid":
        return torch.sigmoid(logits)
    if scoring == "softplus_sqrt":
        return torch.sqrt(F.softplus(logits))
    raise ValueError(
        f"Unknown MoE router scoring {scoring!r}; expected 'softmax', "
        "'sigmoid' or 'softplus_sqrt'."
    )


class MoERouter(nn.Module):
    """Score experts for each token and select the top-k.

    Args:
        hidden_size: Width of the layer input.
        num_experts: Number of routed experts.
        top_k: Number of experts selected per token.
        scoring: Gating function (see :func:`_score`).
        pre_softmax: Apply the softmax over all experts before top-k selection
            (keeps raw probabilities, skips top-k renormalisation).
        renormalize: Rescale the selected scores so they sum to one.
        routed_scaling_factor: Constant multiplied onto the mixing weights.
        dtype: ``"float32"`` (default) or ``"bfloat16"`` router math.
        jitter_eps: Switch-style uniform multiplicative input jitter.
        z_loss_coeff: Weight of the ``logsumexp`` z-loss (0 disables it).
        bias_update: ``"none"``, ``"sign"`` or ``"centered_sign"``.
        bias_update_rate: Fixed step ``γ`` of the expert bias.
        linear_factory: Optional projection factory (e.g. ``BitLinear``).
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        scoring: str = "softmax",
        pre_softmax: bool = False,
        renormalize: bool = True,
        routed_scaling_factor: float = 1.0,
        dtype: str = "float32",
        jitter_eps: float = 0.0,
        z_loss_coeff: float = 0.0,
        bias_update: str = "none",
        bias_update_rate: float = 0.0,
        linear_factory: Optional[Callable[[int, int, bool], nn.Module]] = None,
    ):
        super().__init__()
        build = linear_factory or (lambda i, o, b: nn.Linear(i, o, bias=b))
        self.linear = build(hidden_size, num_experts, False)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.scoring = str(scoring)
        self.pre_softmax = bool(pre_softmax)
        self.renormalize = bool(renormalize)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.dtype = str(dtype)
        self.jitter_eps = float(jitter_eps)
        self.z_loss_coeff = float(z_loss_coeff)
        self.bias_update = str(bias_update)
        self.bias_update_rate = float(bias_update_rate)

        # Auxiliary-loss-free bias (DeepSeek-V3 ``noaux_tc``). Kept in fp32 and
        # non-persistent: it is training state, not an inference parameter.
        if self.bias_update != "none":
            self.register_buffer(
                "expert_bias", torch.zeros(self.num_experts), persistent=False
            )
            self.register_buffer(
                "_pending_counts", torch.zeros(self.num_experts), persistent=False
            )
        else:
            self.expert_bias = None
            self._pending_counts = None

    @property
    def weight(self) -> torch.Tensor:
        """Router projection weight (legacy ``nn.Linear``-style accessor)."""
        return self.linear.weight

    @property
    def bias(self):
        """Router projection bias (``None``: routers are bias-free)."""
        return getattr(self.linear, "bias", None)

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    def score(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Return selection scores, auxiliary probabilities and z-loss.

        Args:
            x: Router input of shape ``(N, hidden_size)``.

        Returns:
            Tuple ``(scores, aux_probs, z_loss)``. ``scores`` drives selection
            and mixing; ``aux_probs`` is the per-expert distribution used by
            the load-balancing loss; ``z_loss`` is the coefficient-weighted
            z-loss tensor (or ``None`` when disabled).
        """
        if self.jitter_eps > 0.0 and self.training:
            x = x * (1.0 + self.jitter_eps * (torch.rand_like(x) * 2.0 - 1.0))
        logits = self.linear(x)
        if self.dtype == "float32":
            logits = logits.float()
        scores = _score(logits, self.scoring)
        z_loss = None
        if self.z_loss_coeff > 0.0:
            z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()
            z_loss = z_loss * self.z_loss_coeff
        return scores, scores, z_loss

    def mix_weights(
        self, scores: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Return the mixing weights for already-selected experts.

        Args:
            scores: Selection scores of shape ``(N, num_experts)``.
            indices: Selected expert indices of shape ``(N, top_k)``.

        Returns:
            Weights of shape ``(N, top_k)``, including the routed scaling
            factor.
        """
        selected = scores.gather(-1, indices)
        if self.scoring == "softmax":
            if self.pre_softmax:
                # Raw probabilities from the full softmax (no renormalisation).
                weights = selected
            else:
                # Softmax over the selected logits == renormalised probabilities.
                weights = torch.softmax(selected, dim=-1)
        else:
            weights = selected
            if self.renormalize:
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        return weights * self.routed_scaling_factor

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        indices_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Route a batch of tokens.

        Args:
            x: Router input of shape ``(N, hidden_size)``.
            indices_override: Optional pre-computed expert indices of shape
                ``(N, top_k)`` (hash layers). The learned router is then used
                for the weights only.

        Returns:
            Tuple ``(weights, indices, aux_probs, z_loss)`` with shapes
            ``(N, top_k)``, ``(N, top_k)``, ``(N, num_experts)`` and a scalar
            tensor (or ``None``).
        """
        scores, aux_probs, z_loss = self.score(x)

        if indices_override is not None:
            indices = indices_override.to(device=scores.device, dtype=torch.long)
        else:
            selection = scores
            if self.expert_bias is not None:
                selection = scores + self.expert_bias.to(scores.dtype)
            indices = torch.topk(selection, self.top_k, dim=-1).indices

        weights = self.mix_weights(scores, indices)

        # Accumulate selection counts for the once-per-step bias update.
        if self.training and self._pending_counts is not None:
            counts = torch.bincount(
                indices.reshape(-1), minlength=self.num_experts
            ).to(self._pending_counts.dtype)
            self._pending_counts += counts

        return weights, indices, aux_probs, z_loss

    # ------------------------------------------------------------------
    # Load-balancing bias
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_expert_bias(
        self, counts: Optional[torch.Tensor] = None
    ) -> bool:
        """Move the expert bias once, from the accumulated selection counts.

        Overloaded experts (count above the mean) step down and underloaded
        ones step up. ``centered_sign`` subtracts the mean step so the bias
        mean does not drift.

        Args:
            counts: Optional counts to use instead of the accumulated ones.

        Returns:
            ``True`` when the bias was updated.
        """
        if self.expert_bias is None:
            return False
        source = counts if counts is not None else self._pending_counts
        if source is None or float(source.sum().item()) <= 0.0:
            return False
        counts_f = source.to(self.expert_bias.dtype)
        mean = counts_f.mean()
        step = self.bias_update_rate * torch.sign(mean - counts_f)
        if self.bias_update == "centered_sign":
            step = step - step.mean()
        self.expert_bias.add_(step)
        if counts is None and self._pending_counts is not None:
            self._pending_counts.zero_()
        return True


__all__ = ["MoERouter", "_score"]
