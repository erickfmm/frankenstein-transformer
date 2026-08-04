#!/usr/bin/env python3
"""Frankenstein Transformer: hybrid mixed-architecture Transformer encoder.

This module hosts :class:`FrankensteinEncoder`, the flagship encoder model.
It stacks ``num_layers`` :class:`HybridLayer` blocks, each configured by
``layer_pattern``, and repeats the entire stack ``num_loops`` times (looped
depth). The logical depth is ``num_layers * num_loops``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import FrankensteinModelConfig
from .hybrid_layer import HybridLayer
from .attention.common import BitLinear
from .norm import get_norm
from .embeddings import FactorizedEmbedding


class FrankensteinEncoder(nn.Module):
    """Frankenstein Encoder: Hybrid mixed-architecture Transformer encoder.

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
        config: The :class:`FrankensteinModelConfig` used to build the model.
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

    def __init__(self, config: FrankensteinModelConfig):
        """Build the Frankenstein encoder from a FrankensteinModelConfig.

        Args:
            config: :class:`FrankensteinModelConfig` instance with all model
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

        # mHC stream expansion / collapse. The embedding is C-dimensional; the
        # n-stream residual lives at (B, S, n, C). We expand the single stream
        # into ``n`` copies on entry and collapse back to ``C`` before the head.
        self.use_mhc = bool(getattr(config, "use_mhc", False))
        if self.use_mhc:
            n = int(config.mhc_expansion_rate)
            self.mhc_in_proj = nn.Linear(config.hidden_size, n * config.hidden_size)
            self.mhc_out_proj = nn.Linear(n * config.hidden_size, config.hidden_size)

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

        if self.use_mhc:
            n = int(self.config.mhc_expansion_rate)
            bsz, seq_len, dim = x.shape
            # Expand the C-dim stream to (B, S, n, C).
            x = self.mhc_in_proj(x).view(bsz, seq_len, n, dim)

        logical_layer_idx = 0
        mixture_of_depths_aux_losses = []
        mixture_of_depths_selected_fractions = []
        mhc_checkpoint = bool(getattr(self.config, "mhc_checkpoint", False))
        for _ in range(self.config.num_loops):
            for layer in self.layers:
                if mhc_checkpoint and layer.use_mhc:
                    x = torch.utils.checkpoint.checkpoint(
                        layer,
                        x,
                        logical_layer_idx,
                        input_ids,
                        use_reentrant=False,
                    )
                else:
                    x = layer(x, logical_layer_idx=logical_layer_idx, input_ids=input_ids)
                if layer.use_mixture_of_depths and layer.last_mixture_of_depths_aux_loss is not None:
                    mixture_of_depths_aux_losses.append(layer.last_mixture_of_depths_aux_loss)
                    mixture_of_depths_selected_fractions.append(
                        layer.last_mixture_of_depths_selected_fraction
                    )
                logical_layer_idx += 1

        if self.use_mhc:
            bsz, seq_len, n, dim = x.shape
            x = self.mhc_out_proj(x.reshape(bsz, seq_len, n * dim))

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
