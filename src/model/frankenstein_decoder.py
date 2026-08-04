#!/usr/bin/env python3
"""Frankenstein Transformer: autoregressive causal decoder variant.

This module hosts :class:`FrankensteinDecoder`, which wraps a
:class:`FrankensteinEncoder` backbone with ``mode='decoder'`` so every
attention layer applies causal (autoregressive) masking. It supports the
same hybrid architecture features as the encoder: 17+ mixer families, MoE,
BitNet, factorized embeddings, looped depth, Mixture-of-Depths, and Engram
memory.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FrankensteinModelConfig
from .frankenstein_encoder import FrankensteinEncoder


class FrankensteinDecoder(nn.Module):
    """Autoregressive causal decoder variant for LLM-style text generation.

    Wraps a :class:`FrankensteinEncoder` backbone with ``mode='decoder'``
    so every attention layer applies causal (autoregressive) masking. Supports
    the same hybrid architecture features as the encoder: 17+ mixer families,
    MoE, BitNet, factorized embeddings, looped depth, Mixture-of-Depths, and
    Engram memory.

    The ``model_class=frankensteindecoder`` preset forces ``mode=decoder`` at
    runtime, overriding any user-provided mode.

    Attributes:
        config: The :class:`FrankensteinModelConfig` (built from preset or user-provided).
        backbone: The underlying :class:`FrankensteinEncoder` model.
        last_auxiliary_losses: Mirrored from the backbone after each forward
            pass.
        last_mixture_of_depths_stats: Mirrored from the backbone after each
            forward pass.
    """

    @staticmethod
    def build_decoder_config(
        vocab_size: int = 50_000,
        hidden_size: int = 2048,
        num_layers: int = 12,
        num_loops: int = 1,
        use_bitnet: bool = True,
        layer_pattern: Optional[List[str]] = None,
    ) -> FrankensteinModelConfig:
        """Build the default decoder preset configuration.

        Args:
            vocab_size: Vocabulary size. Default: 50000.
            hidden_size: Hidden state dimensionality. Default: 2048.
            num_layers: Number of physical HybridLayer blocks. Default: 12.
            num_loops: Number of loop iterations. Default: 1.
            use_bitnet: Whether to use BitNet ternary quantization.
                Default: True.
            layer_pattern: Optional custom layer pattern. If None, defaults
                to ``["titan_attn", "retnet", "titan_attn", "mamba"] * 3``.

        Returns:
            A pre-configured :class:`FrankensteinModelConfig` with ``mode='decoder'``.
        """
        if layer_pattern is None:
            layer_pattern = [
                "titan_attn",
                "retnet",
                "titan_attn",
                "mamba",
            ] * 3
        return FrankensteinModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_loops=num_loops,
            num_heads=16,
            retention_heads=8,
            num_experts=8,
            top_k_experts=2,
            dropout=0.1,
            ode_solver="rk4",
            ode_steps=2,
            use_bitnet=use_bitnet,
            norm_type="dynamic_tanh",
            layer_pattern=layer_pattern,
            use_factorized_embedding=False,
            mode="decoder",
        )

    def __init__(self, config: Optional[FrankensteinModelConfig] = None):
        """Initialize the decoder model.

        Args:
            config: Optional :class:`FrankensteinModelConfig`. If None, the default
                decoder preset is used. ``mode`` is forced to ``"decoder"``
                if not already set.
        """
        super().__init__()
        self.config = config or self.build_decoder_config()
        self.last_auxiliary_losses = {}
        self.last_mixture_of_depths_stats = {}

        if self.config.mode != "decoder":
            self.config.mode = "decoder"

        self.backbone = FrankensteinEncoder(self.config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder backbone (causal masking).

        Args:
            input_ids: Integer token indices of shape ``(B, S)``.

        Returns:
            Logits tensor of shape ``(B, S, vocab_size)``.
        """
        output = self.backbone(input_ids)
        self.last_auxiliary_losses = dict(getattr(self.backbone, "last_auxiliary_losses", {}))
        self.last_mixture_of_depths_stats = dict(
            getattr(self.backbone, "last_mixture_of_depths_stats", {})
        )
        return output

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive token generation with top-k sampling.

        Runs the decoder forward pass repeatedly, sampling one token at a
        time from the top-k filtered softmax distribution. Runs under
        ``torch.inference_mode()`` for efficiency.

        Args:
            input_ids: Prompt token indices of shape ``(B, S_prompt)``.
            max_new_tokens: Maximum number of tokens to generate.
                Default: 128.
            temperature: Softmax temperature for sampling. Lower values
                make output more deterministic. Default: 1.0.
            top_k: Number of highest-probability tokens to keep for
                sampling. Set to 0 to disable top-k filtering.
                Default: 50.

        Returns:
            Tensor of shape ``(B, S_prompt + max_new_tokens)`` containing
            the prompt followed by generated tokens.
        """
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
