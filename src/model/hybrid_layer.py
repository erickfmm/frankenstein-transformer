#!/usr/bin/env python3
"""Frankenstein Transformer: per-layer dispatcher module.

This module hosts :class:`HybridLayer`, the per-layer block that routes to
the configured attention mixer and FFN (dense or MoE), with optional
Mixture-of-Depths token routing and Manifold-constrained Hyper-Connections
(mHC).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FrankensteinModelConfig
from .attention.common import BitLinear
from .norm import FlashNormBitLinear, FlashNormLinear, get_norm
from .activation_function import (
    GLU_VARIANTS,
    get_activation,
    make_gated_ffn,
)
from .mhc import ManifoldHyperConnections
from .attention.engram import EngramLayer
from .attention.grouped_query_attention import GroupedQueryAttention
from .attention.gated import (
    DeltaNetAttention,
    Falcon1Attention,
    Falcon1AAttention,
    Falcon2Attention,
    Falcon2AAttention,
    Falcon3Attention,
    Falcon3AAttention,
    ForgettingAttention,
    GatedDeltaNet2Attention,
    GatedDeltaNetAttention,
    GatedLinearAttention,
    GatedSoftmaxAttention,
    HGRN2Attention,
    KDAAttention,
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
from .attention.ssog import SSOGAttention
from .attention.titan import TitanAttention
from .attention.latent import (
    CCAAttention,
    CCGQAAttention,
    GTAAttention,
    GQLAAttention,
    GaussianMixtureAttention,
    IHAAttention,
    MLAAttention,
    MLRAAttention,
    MTLAAttention,
    TuckerAttention,
)


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

    def __init__(self, config: FrankensteinModelConfig, layer_type: str, pos_encoder=None):
        """Initialize a hybrid layer for the given mixer type.

        Args:
            config: :class:`FrankensteinModelConfig` instance with model hyperparameters.
            layer_type: String identifying the attention mixer. Must be one
                of the keys in the internal mixer registry or ``"mamba"``.
            pos_encoder: Optional shared positional encoding module forwarded
                to attention mixers that accept it. ``None`` leaves mixers
                to build/use their own PE.

        Raises:
            ValueError: If ``layer_type`` is not recognized.
        """
        super().__init__()
        self.layer_type = layer_type
        self.pos_encoder = pos_encoder
        self.use_mhc = bool(getattr(config, "use_mhc", False))
        self.use_mixture_of_depths = bool(getattr(config, "use_mixture_of_depths", False))
        if self.use_mhc and self.use_mixture_of_depths:
            raise ValueError(
                "mHC (use_mhc) and Mixture-of-Depths (use_mixture_of_depths) "
                "are mutually exclusive: MoD token routing operates on a single "
                "C-dimensional stream, which conflicts with the n-stream residual."
            )
        self.norm1 = get_norm(config)
        self.use_moe = bool(config.use_moe)
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

        # Residual-connection strategy (arXiv:2603.15031). Decides whether
        # to add the layer output to the residual (standard / none / AttnRes
        # managed externally by the encoder).
        self.residual_type = str(getattr(config, "residual_type", "standard")).lower()
        if self.residual_type not in {"standard", "none", "full_attn", "block_attn"}:
            raise ValueError(
                f"Unknown residual_type {self.residual_type!r}; expected one of "
                "'standard', 'none', 'full_attn', 'block_attn'."
            )
        # AttnRes variants are handled by an external ResidualBase module
        # injected by FrankensteinEncoder — the layer itself never touches
        # the residual sum (the encoder passes the attended ``x`` in).

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
            "falcon1_attn": Falcon1Attention,
            "falcon2_attn": Falcon2Attention,
            "falcon3_attn": Falcon3Attention,
            "falcon1a_attn": Falcon1AAttention,
            "falcon2a_attn": Falcon2AAttention,
            "falcon3a_attn": Falcon3AAttention,
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
            "gma_attn": GaussianMixtureAttention,
            "ssog_attn": SSOGAttention,
        }

        if layer_type == "mamba":
            self.mixer = proj_cls(config.hidden_size, config.hidden_size)
        elif layer_type in mixer_registry:
            self.mixer = mixer_registry[layer_type](config, pos_encoder=pos_encoder)
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

        # mHC: one module per layer function (attention, then FFN). The n-stream
        # residual is shared and read/written by both sub-functions.
        if self.use_mhc:
            self.mhc_attn = ManifoldHyperConnections(
                hidden_size=config.hidden_size,
                expansion_rate=config.mhc_expansion_rate,
                sinkhorn_iters=config.mhc_sinkhorn_iters,
                gating_init=config.mhc_gating_init,
                use_bitnet=config.use_bitnet,
                full_prec_under_bitnet=config.mhc_full_prec_under_bitnet,
            )
            self.mhc_ffn = ManifoldHyperConnections(
                hidden_size=config.hidden_size,
                expansion_rate=config.mhc_expansion_rate,
                sinkhorn_iters=config.mhc_sinkhorn_iters,
                gating_init=config.mhc_gating_init,
                use_bitnet=config.use_bitnet,
                full_prec_under_bitnet=config.mhc_full_prec_under_bitnet,
            )
        else:
            self.mhc_attn = None
            self.mhc_ffn = None

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
        # ---- mHC path: the stream is (B, S, n, C) across the whole layer.
        # Attention and FFN each act as a layer function on the C-dim pre-
        # projection, writing back onto the shared n-stream residual.
        if self.use_mhc:
            return self._forward_dense_mhc(
                x,
                logical_layer_idx=logical_layer_idx,
                input_ids=input_ids,
            )

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
            x = self.mixer(x, input_ids=input_ids, logical_layer_idx=logical_layer_idx, pos_encoder=self.pos_encoder)
        else:
            x = self.mixer(x, logical_layer_idx=logical_layer_idx, pos_encoder=self.pos_encoder)

        # Residual merge: standard adds the input, none drops it,
        # AttnRes variants receive the attended ``x`` from the encoder
        # (the encoder has already applied depth-wise attention via
        # the ResidualBase module).
        if self.residual_type == "none":
            # x is already the layer output; do not add the residual back.
            pass
        else:
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

            if self.residual_type == "none":
                x = out.view(batch_size, seq_len, dim)
            else:
                x = residual + out.view(batch_size, seq_len, dim)
            return x

        if self.residual_type == "none":
            x = self.ffn(x)
        else:
            x = residual + self.ffn(x)
        return x

    def _forward_dense_mhc(
        self,
        x,
        logical_layer_idx: Optional[int] = None,
        input_ids: Optional[torch.Tensor] = None,
    ):
        """Run the attention + FFN path over an n-stream mHC residual.

        The input ``x`` is the ``(B, S, n, C)`` n-stream residual. The
        attention mixer and FFN each act as a layer function ``F`` on the
        C-dimensional pre-projection ``Fpre = H[pre] @ x``, writing their
        output back onto the shared stream via ``H[post]`` / ``H[res]``.

        Args:
            x: Input stream of shape ``(B, S, n, C)``.
            logical_layer_idx: Global logical layer index (0-based across
                loops). Passed to mixers that need positional awareness.
            input_ids: Original token IDs, required by Engram layers.

        Returns:
            Updated stream of shape ``(B, S, n, C)``.

        Raises:
            ValueError: If the layer is training-free and called in training
                mode.
        """
        if self.training and self.layer_type in self.TRAINING_FREE_LAYERS:
            raise ValueError(
                f"Layer '{self.layer_type}' is training-free and only supported in eval/inference mode."
            )

        # Attention sub-function.
        fpre = self.mhc_attn.fpre(x)
        attn_in = self.norm1(fpre)
        if self.layer_type == "mamba":
            attn_out = self.mixer(attn_in)
        elif self.layer_type in {"ode", "retnet"}:
            attn_out = self.mixer(attn_in)
        elif self.layer_type == "engram_attn":
            attn_out = self.mixer(
                attn_in,
                input_ids=input_ids,
                logical_layer_idx=logical_layer_idx,
                pos_encoder=self.pos_encoder,
            )
        else:
            attn_out = self.mixer(attn_in, logical_layer_idx=logical_layer_idx, pos_encoder=self.pos_encoder)
        x = self.mhc_attn.recombine(x, attn_out)

        # FFN sub-function.
        fpre = self.mhc_ffn.fpre(x)
        ffn_in = self.norm2(fpre)
        if self.use_moe:
            logits = self.router(ffn_in)
            weights, indices = torch.topk(F.softmax(logits, dim=-1), self.top_k, dim=-1)
            batch_size, seq_len, dim = ffn_in.shape
            flat_x = ffn_in.view(-1, dim)
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
            ffn_out = out.view(batch_size, seq_len, dim)
        else:
            ffn_out = self.ffn(ffn_in)

        return self.mhc_ffn.recombine(x, ffn_out)

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
