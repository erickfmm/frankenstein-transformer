"""A single Mixture-of-Experts feed-forward layer.

The layer owns the routed experts, the always-on shared experts, the optional
latent projections (LatentMoE), the router, and the balancing bookkeeping. It
returns the layer output and exposes:

* ``last_auxiliary_losses`` — coefficient-weighted tensors that the model adds
  to the task loss (``moe_load_balancing_loss``, ``moe_router_z_loss``).
* ``last_moe_stats`` — scalar diagnostics for logging (MaxVio, routing entropy,
  top-1 weight, expert output-norm ratio, dropped fraction).

References:
    Shazeer et al., *Sparsely-Gated Mixture-of-Experts*, arXiv:1701.06538.
    Fedus et al., *Switch Transformers*, arXiv:2101.03961.
    Lewis et al., *BASE Layers*, arXiv:2103.16716.
    Roller et al., *Hash Layers*, arXiv:2106.04426.
    Zoph et al., *ST-MoE*, arXiv:2202.08906.
    Zhou et al., *Expert Choice Routing*, arXiv:2202.09368.
    DeepSeek-AI, *DeepSeekMoE*, arXiv:2401.06066.
    DeepSeek-AI, *DeepSeek-V3*, arXiv:2412.19437.
    Elango et al., *LatentMoE*, arXiv:2601.18089.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .balancing import load_balancing_loss, max_vio
from .router import MoERouter

ExpertFactory = Callable[[int, int, float], nn.Module]


class _RMSNorm(nn.Module):
    """Minimal RMSNorm used before the latent up-projection."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class MoELayer(nn.Module):
    """Routed mixture-of-experts feed-forward layer.

    Args:
        config: ``FrankensteinModelConfig`` (the schema-backed contract).
        expert_factory: Callable ``(intermediate_size, in_out_size, clamp)``
            returning a fresh expert FFN. ``in_out_size`` is the latent width
            when latent MoE is enabled, ``hidden_size`` otherwise.
        router_factory: Optional projection factory for the router (e.g.
            ``BitLinear`` when ``bitnet_routers`` is on).
        linear_factory: Optional projection factory for the latent
            projections (e.g. ``BitLinear`` when BitNet is on).
        use_hash: Whether this layer selects experts from the fixed token-id
            table (first ``moe.hash_layers`` MoE layers).
    """

    def __init__(
        self,
        config,
        expert_factory: ExpertFactory,
        router_factory: Optional[Callable[[int, int, bool], nn.Module]] = None,
        linear_factory: Optional[Callable[[int, int, bool], nn.Module]] = None,
        use_hash: bool = False,
    ):
        super().__init__()
        self.config = config
        self.num_experts = int(config.moe.num_experts)
        self.top_k = int(config.moe.top_k)
        self.hidden_size = int(config.dims.hidden_size)
        self.expert_hidden_size = int(
            config.moe.expert_hidden_size or config.ffn.hidden_size
        )
        self.routing_type = str(config.moe.routing.type)
        self.capacity_factor = float(config.moe.routing.capacity_factor)
        self.aux_loss_type = str(config.moe.balancing.aux_loss)
        self.aux_loss_coeff = float(config.moe.balancing.aux_loss_coeff)
        self.activation_clamp = float(config.moe.experts.activation_clamp)
        self.latent_enabled = bool(config.moe.latent.enabled)
        self.latent_dim = (
            int(config.moe.latent.dim) if config.moe.latent.dim else self.hidden_size
        )
        self.latent_norm_before_up_proj = bool(
            config.moe.latent.norm_before_up_proj
        )
        self.shared_count = int(config.moe.shared.count)
        self.shared_hidden_size = int(
            config.moe.shared.hidden_size or self.expert_hidden_size
        )
        self.shared_gate = str(config.moe.shared.gate)
        self.routed_scaling_factor = float(config.moe.router.routed_scaling_factor)
        self.use_hash = bool(use_hash) and int(config.moe.routing.hash_layers) > 0

        inner = self.latent_dim if self.latent_enabled else self.hidden_size
        self.inner_size = inner

        # The learned router (also provides the weights for hash layers).
        self.router = MoERouter(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            scoring=str(config.moe.router.scoring),
            pre_softmax=bool(config.moe.router.pre_softmax),
            renormalize=bool(config.moe.router.renormalize),
            routed_scaling_factor=self.routed_scaling_factor,
            dtype=str(config.moe.router.dtype),
            jitter_eps=float(config.moe.router.jitter_eps),
            z_loss_coeff=float(config.moe.router.z_loss_coeff),
            bias_update=(
                "none" if self.use_hash else str(config.moe.balancing.bias_update)
            ),
            bias_update_rate=float(config.moe.balancing.bias_update_rate),
            linear_factory=router_factory,
        )

        # Routed experts.
        self.experts = nn.ModuleList(
            [
                expert_factory(
                    self.expert_hidden_size, inner, self.activation_clamp
                )
                for _ in range(self.num_experts)
            ]
        )

        # Always-on shared experts (outside the router, full hidden width).
        self.shared_experts = nn.ModuleList(
            [
                expert_factory(
                    self.shared_hidden_size, self.hidden_size, self.activation_clamp
                )
                for _ in range(self.shared_count)
            ]
        )
        if self.shared_count > 0 and self.shared_gate == "sigmoid":
            self.shared_gate_proj = nn.Linear(self.hidden_size, 1)
        else:
            self.shared_gate_proj = None

        # Latent projections (LatentMoE, arXiv:2601.18089).
        build = linear_factory or (lambda i, o, b: nn.Linear(i, o, bias=b))
        if self.latent_enabled:
            self.latent_down_proj = build(
                self.hidden_size, self.latent_dim, False
            )
            self.latent_up_proj = build(self.latent_dim, self.hidden_size, False)
            self.up_norm = (
                _RMSNorm(self.latent_dim)
                if self.latent_norm_before_up_proj
                else None
            )
        else:
            self.latent_down_proj = None
            self.latent_up_proj = None
            self.up_norm = None

        # Fixed token-id → expert table (Hash Layers, arXiv:2106.04426).
        if self.use_hash:
            table = self._build_hash_table(config)
            self.register_buffer("tid2eid", table, persistent=False)
        else:
            self.tid2eid = None

        self.last_auxiliary_losses: Dict[str, torch.Tensor] = {}
        self.last_moe_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_hash_table(self, config) -> torch.Tensor:
        """Build the ``(vocab_size, top_k)`` hash table.

        Args:
            config: Model config providing ``vocab_size`` and the hash keys.

        Returns:
            Long tensor mapping each token id to ``top_k`` expert indices.
        """
        vocab = int(config.dims.vocab_size)
        k = self.top_k
        e = self.num_experts
        if str(config.moe.routing.hash_table) == "round_robin":
            rows = torch.arange(vocab).unsqueeze(1)
            cols = torch.arange(k).unsqueeze(0)
            return ((rows + cols) % e).long()

        generator = torch.Generator().manual_seed(int(config.moe.routing.hash_seed))
        chunks: List[torch.Tensor] = []
        chunk_size = 65_536
        for start in range(0, vocab, chunk_size):
            size = min(chunk_size, vocab - start)
            scores = torch.rand(size, e, generator=generator)
            chunks.append(torch.topk(scores, k, dim=-1).indices)
        return torch.cat(chunks, dim=0).long()

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------
    def _capacity(self, num_tokens: int) -> Optional[int]:
        """Return the per-expert assignment cap, or ``None`` when dropless."""
        if self.capacity_factor <= 0.0:
            return None
        return max(
            1,
            int(
                math.ceil(
                    self.capacity_factor * num_tokens * self.top_k / self.num_experts
                )
            ),
        )

    def _apply_capacity(
        self, indices: torch.Tensor, weights: torch.Tensor, capacity: int
    ) -> torch.Tensor:
        """Drop overflowing assignments, keeping the highest-probability ones.

        Args:
            indices: ``(N, top_k)`` selected expert indices.
            weights: ``(N, top_k)`` mixing weights (used for ranking).
            capacity: Maximum assignments kept per expert.

        Returns:
            Boolean keep mask of shape ``(N, top_k)``.
        """
        flat_experts = indices.reshape(-1)
        flat_weights = weights.reshape(-1)
        keep = torch.ones_like(flat_experts, dtype=torch.bool)
        for expert in range(self.num_experts):
            selected = (flat_experts == expert).nonzero(as_tuple=False).flatten()
            if selected.numel() <= capacity:
                continue
            order = torch.argsort(flat_weights[selected], descending=True)
            keep[selected[order[capacity:]]] = False
        return keep.view_as(indices)

    def _dispatch_token_choice(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        keep: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run the routed experts for token-choice routing.

        Args:
            inputs: Expert input of shape ``(N, inner_size)``.
            weights: Mixing weights ``(N, top_k)``.
            indices: Expert indices ``(N, top_k)``.
            keep: Optional capacity mask ``(N, top_k)``.

        Returns:
            Combined routed output of shape ``(N, inner_size)``.
        """
        num_tokens = inputs.shape[0]
        out = torch.zeros_like(inputs)
        norms = torch.zeros(self.num_experts, device=inputs.device)
        norm_counts = torch.zeros(self.num_experts, device=inputs.device)
        for slot in range(self.top_k):
            slot_indices = indices[:, slot]
            slot_weights = weights[:, slot]
            slot_keep = None if keep is None else keep[:, slot]
            for expert_idx, expert in enumerate(self.experts):
                mask = slot_indices == expert_idx
                if slot_keep is not None:
                    mask = mask & slot_keep
                if not bool(mask.any()):
                    continue
                expert_out = expert(inputs[mask])
                out[mask] += expert_out * slot_weights[mask].unsqueeze(-1)
                norms[expert_idx] += expert_out.norm(dim=-1).sum()
                norm_counts[expert_idx] += expert_out.shape[0]
        self._last_expert_norms = (norms, norm_counts, num_tokens)
        return out

    def _dispatch_expert_choice(
        self, inputs: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        """Run the routed experts for expert-choice routing.

        Each expert keeps its best ``N · k / E`` tokens. Weights come from the
        router scores and are renormalised per token over that token's selected
        experts when ``router.renormalize`` is on.

        Args:
            inputs: Expert input of shape ``(N, inner_size)``.
            scores: Router scores ``(N, num_experts)``.

        Returns:
            Combined routed output of shape ``(N, inner_size)``.
        """
        num_tokens = inputs.shape[0]
        per_expert = max(1, (num_tokens * self.top_k) // self.num_experts)
        top_scores, top_tokens = torch.topk(
            scores.transpose(0, 1), min(per_expert, num_tokens), dim=-1
        )
        out = torch.zeros_like(inputs)
        weight_sum = torch.zeros(num_tokens, 1, device=inputs.device)
        norms = torch.zeros(self.num_experts, device=inputs.device)
        norm_counts = torch.zeros(self.num_experts, device=inputs.device)
        for expert_idx, expert in enumerate(self.experts):
            tokens = top_tokens[expert_idx]
            weights = top_scores[expert_idx]
            if tokens.numel() == 0:
                continue
            expert_out = expert(inputs[tokens])
            out.index_add_(0, tokens, expert_out * weights.unsqueeze(-1))
            weight_sum.index_add_(0, tokens, weights.unsqueeze(-1))
            norms[expert_idx] += expert_out.norm(dim=-1).sum()
            norm_counts[expert_idx] += expert_out.shape[0]
        if self.config.moe.router.renormalize:
            out = out / weight_sum.clamp_min(1e-12)
        self._last_expert_norms = (norms, norm_counts, num_tokens)
        self._last_expert_choice_counts = torch.full(
            (self.num_experts,),
            float(per_expert),
            device=inputs.device,
        )
        return out * self.routed_scaling_factor

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the MoE feed-forward layer.

        Args:
            x: Input of shape ``(B, S, hidden_size)``.
            input_ids: Token ids, required by hash layers.

        Returns:
            Output of shape ``(B, S, hidden_size)``.

        Raises:
            ValueError: If hash routing is active and ``input_ids`` is missing.
        """
        batch, seq_len, hidden = x.shape
        flat = x.reshape(-1, hidden)
        num_tokens = flat.shape[0]

        expert_input = (
            self.latent_down_proj(flat) if self.latent_enabled else flat
        )

        self._last_expert_norms = None
        self._last_expert_choice_counts = None

        # ---- Routing -------------------------------------------------
        if self.routing_type == "expert_choice":
            scores, aux_probs, z_loss = self.router.score(flat)
            routed = self._dispatch_expert_choice(expert_input, scores)
            indices = None
            weights = None
            keep = None
            counts = self._last_expert_choice_counts
        else:
            indices_override = None
            if self.use_hash:
                if input_ids is None:
                    raise ValueError(
                        "MoE hash layers require input_ids; this model path did "
                        "not provide them (hash routing is only available for "
                        "token-id models, not ViT/audio inputs)."
                    )
                ids = input_ids.reshape(-1)
                indices_override = self.tid2eid[ids].reshape(-1, self.top_k)
            weights, indices, aux_probs, z_loss = self.router(
                flat, indices_override=indices_override
            )
            capacity = self._capacity(num_tokens)
            keep = (
                self._apply_capacity(indices, weights, capacity)
                if capacity is not None
                else None
            )
            routed = self._dispatch_token_choice(
                expert_input, weights, indices, keep
            )
            counts = torch.bincount(
                indices.reshape(-1), minlength=self.num_experts
            ).to(flat.dtype)

        # ---- Latent up-projection ------------------------------------
        if self.latent_enabled:
            if self.up_norm is not None:
                routed = self.up_norm(routed)
            routed = self.latent_up_proj(routed)

        # ---- Shared experts ------------------------------------------
        if self.shared_count > 0:
            shared = torch.zeros_like(flat)
            for expert in self.shared_experts:
                shared = shared + expert(flat)
            if self.shared_gate_proj is not None:
                shared = shared * torch.sigmoid(self.shared_gate_proj(flat))
            routed = routed + shared

        output = routed.view(batch, seq_len, hidden)
        self._record_statistics(
            aux_probs=aux_probs,
            indices=indices,
            weights=weights,
            counts=counts,
            keep=keep,
            z_loss=z_loss,
            seq_len=seq_len,
        )
        return output

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _record_statistics(
        self,
        aux_probs: torch.Tensor,
        indices: Optional[torch.Tensor],
        weights: Optional[torch.Tensor],
        counts: torch.Tensor,
        keep: Optional[torch.Tensor],
        z_loss: Optional[torch.Tensor],
        seq_len: int,
    ) -> None:
        """Compute and store per-layer losses and scalar diagnostics."""
        losses: Dict[str, torch.Tensor] = {}

        # Auxiliary load-balancing loss (token choice only; expert choice is
        # self-balancing and the config forbids the combination).
        if (
            self.routing_type == "token_choice"
            and self.aux_loss_type != "none"
            and indices is not None
        ):
            probs = aux_probs.reshape(-1, seq_len, self.num_experts)
            shaped_indices = indices.reshape(-1, seq_len, self.top_k)
            aux_loss, _ = load_balancing_loss(
                probs, shaped_indices, self.num_experts, self.aux_loss_type
            )
            losses["moe_load_balancing_loss"] = aux_loss * self.aux_loss_coeff
            self._last_aux_loss_value = float(aux_loss.detach().item())
        else:
            self._last_aux_loss_value = 0.0

        if z_loss is not None:
            losses["moe_router_z_loss"] = z_loss

        stats: Dict[str, float] = {
            "max_vio": max_vio(counts),
            "load_balancing_loss": self._last_aux_loss_value,
            "z_loss": (
                float(z_loss.detach().item()) if z_loss is not None else 0.0
            ),
        }

        counts_list = counts.detach().to("cpu").tolist()
        stats["num_tokens"] = float(sum(counts_list))
        stats["tokens_per_expert"] = counts_list  # type: ignore[assignment]
        if keep is not None:
            stats["dropped_fraction"] = float(
                (1.0 - keep.float().mean()).item()
            )
        else:
            stats["dropped_fraction"] = 0.0

        # Routing sharpness: entropy and top-1 weight over the selected experts.
        if weights is not None and self.routed_scaling_factor != 0.0:
            normalised = weights / self.routed_scaling_factor
            stats["routing_entropy"] = float(
                (
                    -(normalised * torch.log(normalised.clamp_min(1e-12)))
                    .sum(dim=-1)
                    .mean()
                )
                .detach()
                .item()
            )
            stats["top1_weight"] = float(
                normalised[:, 0].mean().detach().item()
            )

        # Rogue / dead expert signal: max over median of per-expert output norms.
        if self._last_expert_norms is not None:
            norms, norm_counts, _ = self._last_expert_norms
            mean_norms = norms / norm_counts.clamp_min(1.0)
            used = mean_norms[mean_norms > 0]
            if used.numel() > 0:
                median = used.median().clamp_min(1e-12)
                stats["expert_norm_ratio"] = float(
                    (used.max() / median).detach().item()
                )
            else:
                stats["expert_norm_ratio"] = 1.0

        self.last_auxiliary_losses = losses
        self.last_moe_stats = stats

    # ------------------------------------------------------------------
    # Bias update
    # ------------------------------------------------------------------
    def update_expert_bias(self) -> bool:
        """Apply the accumulated expert-bias update (see ``MoERouter``)."""
        return self.router.update_expert_bias()


__all__ = ["MoELayer"]
