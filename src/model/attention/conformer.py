"""Conformer: convolution-augmented self-attention (Gulati et al. 2020).

Implements the Conformer block of arXiv:2005.08100 as a repository mixer.
The original block stacks four residual units::

    x~ = x + ½·FFN₁(x)          # macaron half-step FFN (pre-norm)
    x' = x~ + MHSA(LN(x~))     # multi-head self-attention (pre-norm)
    x'' = x' + Conv(LN(x'))    # convolution module (pre-norm)
    y  = LN(x'' + ½·FFN₂(x'')) # second macaron FFN + final LayerNorm

Repository integration (documented deviation, justified by Table 5 of the
paper, where half-step vs. full-step residuals differ by 0.0–0.1 WER):

* The shared :class:`~src.model.hybrid_layer.HybridLayer` already supplies
  ``norm1(x)`` and owns the outer residual plus ``norm2`` + FFN; that outer
  FFN plays the role of FFN₂. This mixer therefore owns only the macaron
  FFN₁, the MHSA and the convolution module, with *internal* pre-norm
  residuals, and returns the delta ``m + v + w`` so that the outer
  ``HybridLayer`` residual yields exactly ``x + m + v + w``.
* The paper's ½ residual scaling is dropped (full-step integration), which
  the paper's own ablation shows is equivalent in quality.

References:
    Gulati et al. (2020), "Conformer: Convolution-augmented Transformer for
    Speech Recognition", arXiv:2005.08100.
    Lu et al. (2019), "Understanding and Improving Transformer From a
    Multi-Particle Dynamic System Point of View", arXiv:1906.02762
    (macaron half-step FFN).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activation_function import get_activation
from ..norm import get_norm
from ..numerics import module_softmax
from .common import BitLinear, apply_pe_to_qk, apply_pe_to_scores

__all__ = ["ConformerAttention", "ConformerConvModule"]


class ConformerConvModule(nn.Module):
    """Conformer convolution module (arXiv:2005.08100, Sec. 2.2, Fig. 2).

    Applies the paper's convolution block over a ``(B, S, hidden_size)``
    sequence::

        pointwise Conv1d (d → 2d) → GLU (→ d) → depthwise Conv1d (kernel k)
        → BatchNorm1d → Swish (SiLU) → pointwise Conv1d (d → d) → dropout.

    In decoder mode the depthwise convolution uses causal (left-only)
    padding so that a token never attends to the future; otherwise the
    padding is symmetric ("same" padding trimmed to the input length).

    The ``nn.Conv1d`` layers are intentionally *full-precision*: BitNet
    b1.58 ternary quantization in this repository targets the attention and
    FFN projections (``BitLinear``) and the embedding convolution
    (``BitConv1d``), not the Conformer convolution module, so enabling
    ``use_bitnet`` leaves this block unquantized.

    Args:
        hidden_size: Number of input/output channels ``d``.
        kernel_size: Depthwise convolution kernel size ``k`` (paper best:
            32; odd values give perfectly symmetric encoder padding).
        conv_norm: Normalization after the depthwise convolution:
            ``"batch"`` (paper default), ``"group"`` or ``"layer"``.
        causal: If True, left-pad the depthwise convolution (decoder mode).
        dropout: Dropout probability applied after the output projection.

    Raises:
        ValueError: If ``kernel_size`` is < 1 or ``conv_norm`` is unknown.

    Attributes:
        pointwise_in: Pointwise ``d → 2d`` convolution.
        depthwise: Depthwise ``d → d`` convolution (``groups=d``).
        norm: BatchNorm1d / GroupNorm / LayerNorm applied channel-wise.
        pointwise_out: Pointwise ``d → d`` convolution.
        dropout: Dropout layer.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 32,
        conv_norm: str = "batch",
        causal: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.kernel_size = int(kernel_size)
        if self.kernel_size < 1:
            raise ValueError(
                f"conformer_kernel_size must be >= 1, got {self.kernel_size}"
            )
        self.causal = bool(causal)

        self.conv_norm = str(conv_norm).lower()
        if self.conv_norm not in {"batch", "group", "layer"}:
            raise ValueError(
                "conformer_conv_norm must be one of {'batch', 'group', 'layer'}, "
                f"got {self.conv_norm!r}"
            )

        self.pointwise_in = nn.Conv1d(self.hidden_size, 2 * self.hidden_size, 1)
        self.depthwise = nn.Conv1d(
            self.hidden_size,
            self.hidden_size,
            self.kernel_size,
            groups=self.hidden_size,
            bias=False,
        )
        if self.conv_norm == "batch":
            self.norm = nn.BatchNorm1d(
                self.hidden_size, eps=1e-5, momentum=0.1
            )
        elif self.conv_norm == "group":
            self.norm = nn.GroupNorm(self._num_groups(), self.hidden_size)
        else:
            self.norm = nn.LayerNorm(self.hidden_size)
        self.pointwise_out = nn.Conv1d(self.hidden_size, self.hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def _num_groups(self) -> int:
        """Return the largest group count <= 8 that divides ``hidden_size``."""
        for groups in range(min(8, self.hidden_size), 0, -1):
            if self.hidden_size % groups == 0:
                return groups
        return 1

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad a ``(B, C, S)`` tensor for the depthwise convolution."""
        if self.causal:
            return F.pad(x, (self.kernel_size - 1, 0))
        left = self.kernel_size // 2
        right = self.kernel_size - 1 - left
        if left == 0 and right == 0:
            return x
        return F.pad(x, (left, right))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the convolution module.

        Args:
            x: Input tensor of shape ``(batch, seq_len, hidden_size)``.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        y = x.transpose(1, 2)  # (B, C, S)
        y = self.pointwise_in(y)
        a, b = y.chunk(2, dim=1)
        y = a * torch.sigmoid(b)  # GLU
        y = self._pad(y)
        y = self.depthwise(y)
        if self.conv_norm == "layer":
            # LayerNorm normalizes the feature axis, so move it last.
            y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        else:
            y = self.norm(y)
        y = F.silu(y)
        y = self.pointwise_out(y)
        y = self.dropout(y)
        return y.transpose(1, 2)


class ConformerAttention(nn.Module):
    """Conformer mixer: macaron FFN₁ + MHSA + convolution module.

    Args:
        config: Model configuration object exposing ``hidden_size``,
            ``num_heads``, ``num_kv_heads``, ``ffn_hidden_size``,
            ``dropout``, ``use_bitnet``, ``positional_encoding``, ``mode``
            and the ``conformer_*`` knobs.
        pos_encoder: Optional positional encoding module shared by the
            encoder stack.

    Raises:
        ValueError: If ``hidden_size`` is not divisible by ``num_heads``.
        ValueError: If ``num_kv_heads`` is invalid for ``num_heads``.

    Attributes:
        hidden_size: Input/output dimensionality ``d``.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA).
        head_dim: Per-head dimensionality.
        scale: ``1 / sqrt(head_dim)`` applied to the attention logits.
        mode: ``"encoder"`` (bidirectional) or ``"decoder"`` (causal).
        pe_type: Model-wide positional-encoding name.
        use_pe: Whether this mixer consumes the model-wide PE.
        macaron: Whether the macaron FFN₁ branch is active.
        norm_ffn1 / norm_attn / norm_conv: Internal pre-norm layers.
        q_proj / k_proj / v_proj / out_proj: Attention projections
            (``BitLinear`` when BitNet is enabled).
        ff_in / ff_out: Macaron FFN₁ projections.
        conv: :class:`ConformerConvModule`.
    """

    def __init__(self, config, pos_encoder=None) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_heads)
        self.num_kv_heads = int(getattr(config, "num_kv_heads", 1))
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads for ConformerAttention"
            )
        if not 1 <= self.num_kv_heads <= self.num_heads:
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) must be in "
                f"[1, num_heads ({self.num_heads})]"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads "
                f"({self.num_kv_heads})"
            )
        self.num_groups = self.num_heads // self.num_kv_heads

        self.mode = getattr(config, "mode", "encoder")
        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(getattr(config, "conformer_attn_use_pe", True))
        # Set by HybridLayer (transversal fixed-numerics opt-in). Standalone
        # use keeps the classic softmax.
        self.use_fixed_softmax = False

        self.macaron = bool(getattr(config, "conformer_macaron", True))
        self.kernel_size = int(getattr(config, "conformer_kernel_size", 32))
        self.conv_norm = str(getattr(config, "conformer_conv_norm", "batch")).lower()

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = proj_cls(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        ffn_hidden = int(getattr(config, "ffn_hidden_size", self.hidden_size * 2))
        self.ff_in = proj_cls(self.hidden_size, ffn_hidden)
        self.activation = get_activation(config, dim=ffn_hidden)
        self.ff_out = proj_cls(ffn_hidden, self.hidden_size)
        self.ffn_dropout = nn.Dropout(config.dropout)

        self.norm_ffn1 = get_norm(config)
        self.norm_attn = get_norm(config)
        self.norm_conv = get_norm(config)

        self.conv = ConformerConvModule(
            hidden_size=self.hidden_size,
            kernel_size=self.kernel_size,
            conv_norm=self.conv_norm,
            causal=self.mode == "decoder",
            dropout=config.dropout,
        )

    def _macaron_ffn(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the macaron FFN₁ (pre-norm residual is handled by caller).

        Args:
            x: Input tensor of shape ``(B, S, hidden_size)``.

        Returns:
            Delta tensor of shape ``(B, S, hidden_size)``.
        """
        y = self.ff_in(x)
        y = self.activation(y)
        y = self.ffn_dropout(y)
        y = self.ff_out(y)
        return self.ffn_dropout(y)

    def _attention(
        self,
        x: torch.Tensor,
        logical_layer_idx: Optional[int],
        pos_encoder,
    ) -> torch.Tensor:
        """Run GQA self-attention over ``x`` (no cache; full recompute).

        Args:
            x: Input tensor of shape ``(B, S, hidden_size)``.
            logical_layer_idx: Logical layer index for layer-dependent PEs.
            pos_encoder: Resolved positional-encoding module for this call.

        Returns:
            Attention output of shape ``(B, S, hidden_size)``.
        """
        bsz, seq_len, hidden = x.shape
        q = self.q_proj(x).view(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        idx = logical_layer_idx or 0
        q, k = apply_pe_to_qk(
            pos_encoder, self.pe_type, q, k, x, idx, self.use_pe
        )
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_scores = apply_pe_to_scores(
            pos_encoder, self.pe_type, attn_scores, q, self.use_pe
        )
        if self.mode == "decoder":
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
        attn_weights = module_softmax(self, attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (
            attn_weights @ v
        ).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)

    def forward(
        self,
        x: torch.Tensor,
        logical_layer_idx: Optional[int] = None,
        pos_encoder=None,
        cache=None,
        cache_position: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute the Conformer mixer delta for a pre-normalized input.

        Args:
            x: Pre-normalized input ``norm1(h)`` of shape
                ``(batch, seq_len, hidden_size)``.
            logical_layer_idx: Logical layer index forwarded to the
                positional encoder.
            pos_encoder: Optional positional encoding override for this
                call.
            cache: Accepted for interface uniformity. ``conformer_attn`` has
                no implemented incremental branch (see
                ``src/model/cache/capabilities.py``), so the full-recompute
                path always runs.
            cache_position: Absolute index of the first token in ``x``
                (unused without a cache).

        Returns:
            Delta tensor ``m + v + w`` of shape
            ``(batch, seq_len, hidden_size)``. The outer ``HybridLayer``
            residual adds the pre-norm input, yielding
            ``x + m + v + w``.
        """
        del cache, cache_position  # unsupported: legacy full-recompute path
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder

        if self.macaron:
            m = self._macaron_ffn(self.norm_ffn1(x))
        else:
            m = torch.zeros_like(x)

        v = self._attention(self.norm_attn(x + m), logical_layer_idx, pe)
        w = self.conv(self.norm_conv(x + m + v))
        return m + v + w

    def dense_forward(
        self,
        x: torch.Tensor,
        logical_layer_idx: Optional[int] = None,
        pos_encoder=None,
    ) -> torch.Tensor:
        """Alias for :meth:`forward` (uniform mixer interface).

        ``conformer_attn`` is not a training-free mixer, but exposing the
        dense entry point keeps the mixer uniform with the rest of the
        registry and lets callers use the same dispatch.

        Args:
            x: Input tensor of shape ``(B, S, hidden_size)``.
            logical_layer_idx: Logical layer index forwarded to the PE.
            pos_encoder: Optional positional encoding override.

        Returns:
            Delta tensor of shape ``(B, S, hidden_size)``.
        """
        return self.forward(
            x, logical_layer_idx=logical_layer_idx, pos_encoder=pos_encoder
        )
