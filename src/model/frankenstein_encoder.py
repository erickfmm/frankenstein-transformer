#!/usr/bin/env python3
"""Frankenstein Transformer: hybrid mixed-architecture Transformer encoder.

This module hosts :class:`FrankensteinEncoder`, the flagship encoder model.
It stacks ``num_layers`` :class:`HybridLayer` blocks, each configured by
``layer_pattern``, and repeats the entire stack ``num_loops`` times (looped
depth). The logical depth is ``num_layers * num_loops``.

The encoder additionally owns the cross-layer residual state used by the
Attention Residuals variants (``full_attn``, ``block_attn``). For those
strategies each :class:`HybridLayer` produces a layer output that is then
aggregated by an externally-managed :class:`ResidualBase` module before
the next layer consumes the resulting stream.
"""

from __future__ import annotations

import torch
import torch.nn as nn
# Explicit submodule import: torch>=2.7 no longer guarantees that
# ``torch.utils.checkpoint`` is materialized as an attribute by an indirect
# import elsewhere, and gradient checkpointing below relies on it.
import torch.utils.checkpoint

from .config import FrankensteinModelConfig
from .hybrid_layer import HybridLayer
from .hyperloop import HyperloopConnections
from .attention.common import BitLinear
from .norm import get_norm
from .embeddings import FactorizedEmbedding, build_pos_encoder
from .residuals import ResidualBase, build_residual


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
     * **Hyperloop (arXiv:2604.21254)**: ``mhc_hyperloop=True`` (requires
       ``use_mhc=True``) reorganises the stack as a middle-cycle looped
       Transformer — begin block (runs once) → looped middle block → end
       block (runs once) — wrapped in **loop-level** hyper-connections with
       per-loop parameters ``{W_l, b_l, α_l, e_l}``. The stream is expanded
       by copying ``mhc_expansion_rate`` times and collapsed by averaging,
       replacing the learned in/out projections of per-sublayer mHC.
    * **Mixture-of-Experts (MoE) FFN**: per-token top-k expert routing with
      weighted expert outputs.
    * **BitNet b1.58**: ternary weight quantization via :class:`BitLinear`
      when ``use_bitnet`` is True.
    * **Factorized embeddings**: reduced-dimension embedding lookup +
      projection via :class:`FactorizedEmbedding`.
    * **Mixture-of-Depths**: per-layer token routing where only the
      top-capacity tokens are updated; auxiliary load-balancing loss is
      accumulated and exposed via ``last_auxiliary_losses``.
    * **Residual connections** (this module wires the
      :class:`ResidualBase` strategy):
        - ``standard`` (default): fixed unit-weight sum.
        - ``none`` (experimental): no skip connection.
        - ``full_attn``: depth-wise softmax attention over all previous
          layer outputs (arXiv:2603.15031).
        - ``block_attn``: block-wise attention over ``N`` block
          representations + intra-block partial sum.
    * **Normalization**: ``layer_norm``, ``dynamic_tanh`` (DyT),
      ``derf``, ``rms_norm``, ``prms_norm``, or ``flash_norm``.
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
        residual: :class:`ResidualBase` module owning the cross-layer
            residual state. Stateless for ``standard`` / ``none``;
            stateful for AttnRes variants.
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

        self.pos_encoder = build_pos_encoder(config)

        self.layers = nn.ModuleList(
            [
                HybridLayer(config, layer_type=config.layer_pattern[i % len(config.layer_pattern)], pos_encoder=self.pos_encoder)
                for i in range(config.num_layers)
            ]
        )

        self.final_norm = get_norm(config)
        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.head = proj_cls(config.hidden_size, config.vocab_size)

        # Optional sequence-level classification head (DashAI Strategy A).
        # Full-precision (NOT BitNet-quantized) so downstream fine-tuning is
        # stable regardless of the backbone's quantization flags. Disabled by
        # default; when active, ``forward`` returns ``(B, num_labels)``.
        self.classification_head = bool(getattr(config, "classification_head", False))
        self.cls_num_labels = getattr(config, "num_labels", None)
        self.encoder_pooling_mode = str(getattr(config, "encoder_pooling_mode", "cls"))
        self.cls_head = None
        if self.classification_head and self.cls_num_labels is not None:
            self.cls_head = nn.Linear(config.hidden_size, int(self.cls_num_labels))

        # ---- Residual-connection module (AttnRes / standard) ----
        # Built unconditionally so the encoder can dispatch on
        # ``self.residual.is_attn_res`` without checking config flags.
        self.residual: ResidualBase = build_residual(config)
        # mHC stream expansion / collapse. The embedding is C-dimensional; the
        # n-stream residual lives at (B, S, n, C). We expand the single stream
        # into ``n`` copies on entry and collapse back to ``C`` before the head.
        #
        # Hyperloop (arXiv:2604.21254): ``mhc_hyperloop=True`` swaps the learned
        # in/out projections for loop-level hyper-connections with per-loop
        # parameters — the stream is created by copying and collapsed by
        # averaging, and hyper-connections fire once per loop iteration over
        # the middle-cycle partitioning of the layer stack.
        self.use_mhc = bool(getattr(config, "use_mhc", False))
        self.use_hyperloop = self.use_mhc and bool(getattr(config, "mhc_hyperloop", False))
        if self.use_hyperloop:
            self.hyperloop = HyperloopConnections(
                hidden_size=config.hidden_size,
                num_loops=int(config.num_loops),
                expansion_rate=config.mhc_expansion_rate,
                gating_init=config.mhc_gating_init,
                res_parameterization=config.mhc_hyperloop_res_parameterization,
                sinkhorn_iters=config.mhc_sinkhorn_iters,
                use_bitnet=config.use_bitnet,
                full_prec_under_bitnet=config.mhc_full_prec_under_bitnet,
            )
        elif self.use_mhc:
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
            Logits tensor of shape ``(B, S, vocab_size)``, or ``(B, num_labels)``
            when the optional classification head is enabled
            (``classification_head=True``).
        """
        x = self.emb(input_ids)
        if self.pos_encoder is not None and hasattr(self.pos_encoder, "add"):
            x = self.pos_encoder.add(x)
        x = self.dropout(x)

        # ---- Middle-cycle partitioning (Hyperloop, arXiv:2604.21254) ----
        # begin layers | looped middle layers | end layers. Without Hyperloop
        # the whole stack is the middle block and both guards are 0.
        n_begin = (
            int(getattr(self.config, "mhc_hyperloop_begin_layers", 0))
            if self.use_hyperloop
            else 0
        )
        n_end = (
            int(getattr(self.config, "mhc_hyperloop_end_layers", 0))
            if self.use_hyperloop
            else 0
        )
        n_middle = self.config.num_layers - n_begin - n_end

        if self.use_mhc and not self.use_hyperloop:
            n = int(self.config.mhc_expansion_rate)
            bsz, seq_len, dim = x.shape
            # Expand the C-dim stream to (B, S, n, C).
            x = self.mhc_in_proj(x).view(bsz, seq_len, n, dim)

        # ---- Residual state init for AttnRes variants ----
        logical_depth = int(self.config.num_layers) * int(self.config.num_loops)
        self.residual.register_state(num_layers=logical_depth)
        self.residual.reset_state()
        if self.residual.is_attn_res:
            self.residual.set_embedding(x)

        logical_layer_idx = 0
        mixture_of_depths_aux_losses = []
        mixture_of_depths_selected_fractions = []
        mhc_checkpoint = bool(getattr(self.config, "mhc_checkpoint", False))

        def _run_layers(x_in, layers, idx0):
            """Run a layer slice and dispatch per-layer mHC checkpointing.

            Args:
                x_in: Input stream for the first layer in the slice.
                layers: The :class:`HybridLayer` objects to run in order.
                idx0: The logical layer index of the first layer in the slice.

            Returns:
                Tuple ``(x_out, next_idx)`` with the updated stream and the
                next logical layer index.
            """
            x = x_in
            idx = idx0
            for layer in layers:
                if mhc_checkpoint and layer.use_mhc:
                    x = torch.utils.checkpoint.checkpoint(
                        layer,
                        x,
                        idx,
                        input_ids,
                        use_reentrant=False,
                    )
                else:
                    x = layer(x, logical_layer_idx=idx, input_ids=input_ids)
                # For AttnRes variants, the depth-wise attention is the
                # post-layer residual update. The layer has already applied
                # its internal residual merge (``standard`` semantics);
                # we now overwrite ``x`` with the attended aggregation over
                # all previous layer outputs / block sums.
                if self.residual.is_attn_res:
                    x = self.residual(idx, x)
                if layer.use_mixture_of_depths and layer.last_mixture_of_depths_aux_loss is not None:
                    mixture_of_depths_aux_losses.append(layer.last_mixture_of_depths_aux_loss)
                    mixture_of_depths_selected_fractions.append(
                        layer.last_mixture_of_depths_selected_fraction
                    )
                idx += 1
            return x, idx

        # ---- Hyperloop middle-cycle forward ----
        if self.use_hyperloop:
            hl = self.hyperloop
            # Begin block: plain C-dim layers, run exactly once.
            x, logical_layer_idx = _run_layers(x, self.layers[:n_begin], logical_layer_idx)

            # Expand by copying: (B, S, C) → (B, S, n, C).
            x = hl.expand(x)

            # Looped middle block wrapped in loop-level hyper-connections.
            # All coefficient matrices are computed from the *input* stream
            # y^(l) (Eq. 4 of arXiv:2604.21254):
            #   y^(l+1) = H^res y^(l) + H^post ⊗ ( F(H^pre y^(l)) + e_l )
            middle = self.layers[n_begin : n_begin + n_middle]
            for loop_idx in range(self.config.num_loops):
                h_pre, h_post, h_res = hl.mappings(x, loop_idx)
                block_in = hl.apply_fpre(x, h_pre)
                block_out, logical_layer_idx = _run_layers(
                    block_in, middle, logical_layer_idx
                )
                x = hl.apply_update(x, block_out, h_post, h_res, loop_idx)

            # Collapse by averaging: (B, S, n, C) → (B, S, C).
            x = hl.collapse(x)

            # End block: plain C-dim layers, run exactly once.
            x, logical_layer_idx = _run_layers(
                x, self.layers[n_begin + n_middle :], logical_layer_idx
            )
        else:
            for _ in range(self.config.num_loops):
                x, logical_layer_idx = _run_layers(x, self.layers, logical_layer_idx)

        x = self.residual.finalize(x)

        if self.use_mhc and not self.use_hyperloop:
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
        if self.classification_head and self.cls_head is not None:
            # Sequence-level classification: pool the final hidden states and
            # project to ``(B, num_labels)``.
            if self.encoder_pooling_mode == "gap":
                pooled = x.mean(dim=1)
            else:
                pooled = x[:, 0]
            return self.cls_head(pooled)
        return self.head(x)
