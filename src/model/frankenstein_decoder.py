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
from .numerics import module_softmax

from .config import FrankensteinModelConfig
from .frankenstein_encoder import FrankensteinEncoder


class FrankensteinDecoder(nn.Module):
    """Autoregressive causal decoder variant for LLM-style text generation.

    Wraps a :class:`FrankensteinEncoder` backbone with ``mode='decoder'``
    so every attention layer applies causal (autoregressive) masking. Supports
    the same hybrid architecture features as the encoder: 17+ mixer families,
    MoE, BitNet, factorized embeddings, looped depth, Mixture-of-Depths, and
    Engram memory.

    The ``model_class=frankenstein_text_decoder`` preset forces ``mode=decoder`` at
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
        self.use_fixed_softmax = bool(getattr(self.config, "fixed_softmax", False))
        self.last_auxiliary_losses = {}
        self.last_mixture_of_depths_stats = {}

        if self.config.mode != "decoder":
            self.config.mode = "decoder"

        self.backbone = FrankensteinEncoder(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
        cache_position: Optional[int] = None,
        return_cache: bool = False,
    ):
        """Forward pass through the decoder backbone (causal masking).

        Args:
            input_ids: Integer token indices of shape ``(B, S)``.
            past_key_values: Optional
                :class:`~src.model.cache.manager.FrankensteinCache` manager
                (only consumed when ``use_cache`` is true).
            use_cache: Opt-in incremental decoding flag (D4). ``False``
                (default) runs the legacy path bit-identically.
            cache_position: Absolute index of the first token in
                ``input_ids`` (``None`` during prefill).
            return_cache: When true (and ``use_cache``), return the tuple
                ``(logits, cache)`` for generation loops.

        Returns:
            Logits tensor of shape ``(B, S, vocab_size)`` — or a
            ``(logits, cache)`` tuple when ``return_cache`` and ``use_cache``.
        """
        output = self.backbone(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            return_cache=return_cache,
        )
        if return_cache and use_cache and isinstance(output, tuple):
            logits, cache = output
            self.last_auxiliary_losses = dict(getattr(self.backbone, "last_auxiliary_losses", {}))
            self.last_mixture_of_depths_stats = dict(
                getattr(self.backbone, "last_mixture_of_depths_stats", {})
            )
            return logits, cache
        self.last_auxiliary_losses = dict(getattr(self.backbone, "last_auxiliary_losses", {}))
        self.last_mixture_of_depths_stats = dict(
            getattr(self.backbone, "last_mixture_of_depths_stats", {})
        )
        return output

    def _sample_next(self, logits, temperature: float, top_k: int, top_p: float, greedy: bool):
        """Sample one token from ``(B, vocab)`` logits.

        Order of operations: greedy argmax ⇒ temperature scaling ⇒ top-k
        filtering ⇒ top-p (nucleus) filtering ⇒ multinomial sampling.
        """
        if greedy:
            return torch.argmax(logits, dim=-1, keepdim=True)
        next_logits = logits / max(temperature, 1e-8)
        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = float("-inf")
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
            probs = module_softmax(self, sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            # Remove tokens with cumulative probability above the threshold,
            # always keeping the first token above it.
            remove = cumulative - probs > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            next_logits = torch.full_like(next_logits, float("-inf"))
            next_logits = next_logits.scatter(-1, sorted_idx, sorted_logits)
        probs = module_softmax(self, next_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        greedy: bool = False,
        past_key_values=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Autoregressive token generation.

        With ``use_cache=True`` this runs a **prefill + decode** loop: one
        full forward over the prompt builds the
        :class:`~src.model.cache.manager.FrankensteinCache`, then each step
        feeds only the last sampled token (O(S) per token instead of the
        legacy O(S²)). With ``use_cache=False`` (default) the legacy
        full-recompute loop runs — bit-identical behavior (D12).

        Args:
            input_ids: Prompt token indices of shape ``(B, S_prompt)``.
            max_new_tokens: Maximum number of tokens to generate.
                Default: 128.
            temperature: Softmax temperature for sampling. Lower values
                make output more deterministic. Default: 1.0.
            top_k: Number of highest-probability tokens to keep for
                sampling. Set to 0 to disable top-k filtering.
                Default: 50.
            top_p: Nucleus-sampling probability mass. ``1.0`` (default)
                disables top-p filtering.
            greedy: When true, always take the argmax token (overrides
                sampling). Default: ``False``.
            past_key_values: Optional pre-built
                :class:`~src.model.cache.manager.FrankensteinCache`; a fresh
                one is created when ``None`` and ``use_cache`` is true.
            use_cache: Enable the KV-cache path. Default: ``False``.

        Returns:
            Tensor of shape ``(B, S_prompt + max_new_tokens)`` containing
            the prompt followed by generated tokens.
        """
        if not use_cache:
            # Legacy O(S^2) full-recompute loop (byte-identical to the
            # pre-cache implementation, D12).
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids)
                next_logits = logits[:, -1, :]
                next_token = self._sample_next(next_logits, temperature, top_k, top_p, greedy)
                input_ids = torch.cat([input_ids, next_token], dim=1)
            return input_ids

        from .cache import FrankensteinCache

        cache = past_key_values
        if cache is None:
            cache = FrankensteinCache.new(self, None, device=input_ids.device)

        # 1) PREFILL: one full forward over the prompt, building the cache.
        logits, cache = self.forward(
            input_ids,
            past_key_values=cache,
            use_cache=True,
            return_cache=True,
        )
        if cache is None:
            # An incompatible mechanism (AttnRes residuals) disabled caching
            # for this forward — fall back to the legacy full-recompute loop
            # (D5: never fail because of the cache).
            return self.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy,
                use_cache=False,
            )
        next_logits = logits[:, -1, :]
        tokens = []
        for step in range(max_new_tokens):
            next_token = self._sample_next(next_logits, temperature, top_k, top_p, greedy)
            tokens.append(next_token)
            if step == max_new_tokens - 1:
                break
            cache_position = cache.next_position()  # absolute index to write
            logits, cache = self.forward(
                next_token,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                return_cache=True,
            )
            next_logits = logits[:, -1, :]
        return torch.cat([input_ids, torch.cat(tokens, dim=1)], dim=1)
