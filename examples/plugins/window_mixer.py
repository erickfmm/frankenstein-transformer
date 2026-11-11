"""Example Frankenstein plugin: local windowed self-attention.

The mixer is a compact, dependency-free demonstration of the plugin API:

* it is registered declaratively through :mod:`src.plugins.api`;
* its knobs live in a strict inline JSON-Schema fragment, so
  ``model.attention.local_window`` validates like a built-in block;
* the constructor follows the family contract
  ``(config, pos_encoder=None, params=None)`` and reads ``params`` (bound by
  the registry) rather than flat configuration attributes;
* it declares no KV-cache capability, so inference falls back to the legacy
  full-recompute path (D5: caching never fails an inference run).

Use it from a config with::

    plugins:
      - ./examples/plugins/window_mixer.py
    model:
      attention:
        local_window:
          window_size: 128
      dims:
        layer_pattern: [local_window]

and install it as a package by advertising the ``frankenstein.plugins``
entry-point group.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.plugins.api import register_mixer
except ImportError:  # legacy flat sys.path layout (src/ on sys.path)
    from plugins.api import register_mixer

_PARAMS = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "window_size": {
            "type": "integer",
            "minimum": 1,
            "default": 128,
            "title": "Window Size",
            "title_es": "Tamaño de Ventana",
            "description": "Maximum distance (in tokens) between a query and its keys.",
            "description_es": "Distancia máxima (en tokens) entre una query y sus keys.",
        },
        "causal": {
            "type": ["boolean", "null"],
            "default": None,
            "title": "Causal Mask",
            "title_es": "Máscara Causal",
            "description": "Override the causal mask (defaults to decoder mode).",
            "description_es": "Fuerza la máscara causal (por defecto, modo decoder).",
        },
    },
}


class LocalWindowAttention(nn.Module):
    """Sliding-window multi-head self-attention.

    Args:
        config: Nested model configuration.
        pos_encoder: Optional shared positional encoder (unused).
        params: Bound plugin parameters (``window_size``, ``causal``).
    """

    def __init__(self, config, pos_encoder=None, params=None):
        """Initialize the QKV/out projections."""
        super().__init__()
        self.hidden_size = int(config.dims.hidden_size)
        self.num_heads = int(config.dims.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.window = int(params.window_size if params is not None else 128)
        causal = params.causal if params is not None else None
        self.causal = (
            str(config.dims.mode) == "decoder"
            if causal is None
            else bool(causal)
        )
        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(float(config.dims.dropout or 0.0))

    def forward(self, x: torch.Tensor, logical_layer_idx=None, pos_encoder=None) -> torch.Tensor:
        """Run windowed attention.

        Args:
            x: Input tensor ``(B, S, H)``.
            logical_layer_idx: Logical layer index (unused).
            pos_encoder: Per-call positional encoder override (unused).

        Returns:
            Output tensor ``(B, S, H)``.
        """
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(seq_len, device=x.device)
        distance = (positions[:, None] - positions[None, :]).abs()
        mask = distance > self.window
        if self.causal:
            mask = mask | (positions[None, :] > positions[:, None])
        scores = scores.masked_fill(mask, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        return self.out_proj(out)


register_mixer(
    name="local_window",
    builder=LocalWindowAttention,
    description=(
        "Local windowed self-attention (example plugin) — each query attends "
        "to at most `attention.local_window.window_size` neighbouring keys."
    ),
    description_es=(
        "Autoatención local por ventana (plugin de ejemplo) — cada query "
        "atiende como máximo `attention.local_window.window_size` claves vecinas."
    ),
    params_schema=_PARAMS,
    defaults={"window_size": 128, "causal": None},
    needs_pos_encoder=True,
)
