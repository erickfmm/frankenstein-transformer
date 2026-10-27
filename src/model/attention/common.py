"""Shared utilities for attention modules.

Provides quantization helpers (BitNet b1.58 ternary weights, 8-bit activations)
and the ``BitLinear`` layer.

Normalization layers (``DynamicTanhNorm``, ``Derf``) and the ``get_norm``
factory have been moved to ``src/model/norm/``.

References:
    Ma et al. (2024), "The Era of 1-bit LLMs: All Large Language Models are in
    1.58 Bits", arXiv:2402.17764.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..numerics import module_softmax


def activation_quant(x):
    """Quantize activations to 8-bit via straight-through estimator (STE).

    Scales activations to the range [-128, 127], rounds to integer, then
    rescales back. The STE passes gradients through the rounding operation
    unchanged, enabling end-to-end training with quantized activations.

    Args:
        x: Input tensor of any shape. The last dimension is used for
            per-token scaling.

    Returns:
        Tensor of same shape as ``x``, with values quantized to 8-bit
        precision. Gradients flow through the quantization via STE.
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    y = (x * scale).round().clamp(-128, 127) / scale
    return x + (y - x).detach()  # STE


def weight_quant(w):
    """Quantize weights to ternary values {-1, 0, 1} via straight-through estimator.

    Scales weights by the inverse of their mean absolute value, rounds to the
    nearest integer in [-1, 1], then rescales. The STE passes gradients through
    the rounding operation unchanged.

    Args:
        w: Weight tensor of any shape.

    Returns:
        Tensor of same shape as ``w``, with values quantized to ternary
        precision. Gradients flow through the quantization via STE.
    """
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    w_quant = (w * scale).round().clamp(-1, 1) / scale
    return w + (w_quant - w).detach()  # STE


class BitLinear(nn.Linear):
    """BitNet b1.58 linear layer with ternary weight quantization.

    A drop-in replacement for ``nn.Linear`` that reduces VRAM usage by 3-4x
    through ternary weight quantization and 8-bit activation quantization.
    Applies LayerNorm to inputs before quantization for stability, as
    prescribed by the BitNet b1.58 formulation.

    Reference:
        Ma et al. (2024), "The Era of 1-bit LLMs", arXiv:2402.17764.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If ``True``, adds a learnable bias to the output. Defaults to
            ``False``.
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias)

    def forward(self, x):
        """Forward pass with activation and weight quantization.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        x_norm = F.layer_norm(x, (x.shape[-1],))

        w_q = weight_quant(self.weight)
        x_q = activation_quant(x_norm)

        output = F.linear(x_q, w_q, self.bias)
        return output

    @torch.no_grad()
    def bake_ternary_weights(self):
        """Replace the master weight with its ternary-quantized value.

        Applies :func:`weight_quant` once and stores the result
        (``{-1, 0, 1} * scale``) as the learnable parameter, removing the
        straight-through estimator (STE). After baking, the weight holds the
        faithful ternary values and is no longer a full-precision master.

        Idempotent: if the weight is already ternary (all nonzero elements
        equal ``+/-max(|w|)``), the call is a no-op. This prevents the
        per-tensor absmean scale from drifting on repeated baking.

        Used at export/deployment time to produce compact, self-describing
        ternary checkpoints. Training should not continue after baking
        (the STE gradient path is gone).

        The bias (if any) is left untouched.
        """
        w = self.weight
        gamma = w.abs().max().item()
        if gamma < 1e-5:
            return  # all-zero (or near-zero) weight; nothing to bake
        nonzero = w.abs() > 1e-5
        if nonzero.any():
            rel_err = (w[nonzero].abs() - gamma).abs().max().item() / gamma
            if rel_err < 1e-5:
                # Already ternary; baking again would recompute the absmean
                # scale and shift the magnitude. Skip to stay idempotent.
                return
        w_quant = weight_quant(w)
        self.weight.copy_(w_quant)


class BitConv1d(nn.Conv1d):
    """BitNet b1.58 1D convolution with ternary weight quantization.

    A drop-in replacement for ``nn.Conv1d`` that applies the same ternary
    weight (``{-1, 0, 1}``) + int8 activation quantization as
    :class:`BitLinear`, via the straight-through estimator (STE). Inputs are
    LayerNorm-normalized over the channel dimension before activation
    quantization, mirroring the :class:`BitLinear` formulation.

    Used by :class:`FactorizedEmbedding` so the embedding Conv1d pre-projection
    also becomes ternary when ``use_bitnet`` is enabled.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Default: 1.
        padding: Padding added to both sides. Default: 0.
        groups: Number of blocked connections. Default: 1.
        bias: If ``True``, adds a learnable bias. Defaults to ``False``.

    Reference:
        Ma et al. (2024), "The Era of 1-bit LLMs", arXiv:2402.17764.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """Forward pass with activation and weight quantization.

        Args:
            x: Input tensor of shape ``(B, C_in, L)``.

        Returns:
            Output tensor of shape ``(B, C_out, L_out)``.
        """
        x_norm = F.layer_norm(x, (x.shape[1],))

        w_q = weight_quant(self.weight)
        x_q = activation_quant(x_norm)

        return F.conv1d(
            x_q,
            w_q,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    @torch.no_grad()
    def bake_ternary_weights(self):
        """Replace the master weight with its ternary-quantized value.

        Idempotent: a no-op if the weight is already ternary. See
        :meth:`BitLinear.bake_ternary_weights` for details.
        """
        w = self.weight
        gamma = w.abs().max().item()
        if gamma < 1e-5:
            return
        nonzero = w.abs() > 1e-5
        if nonzero.any():
            rel_err = (w[nonzero].abs() - gamma).abs().max().item() / gamma
            if rel_err < 1e-5:
                return
        w_quant = weight_quant(w)
        self.weight.copy_(w_quant)


def is_bitlinear_module(module: nn.Module) -> bool:
    """Return ``True`` if ``module`` is a BitNet-quantized layer.

    Covers both :class:`BitLinear` and :class:`BitConv1d` (the ternary
    Conv1d used by the factorized embedding pre-projection).

    Args:
        module: Any ``nn.Module``.

    Returns:
        Whether the module performs BitNet b1.58 ternary quantization.
    """
    return isinstance(module, (BitLinear, BitConv1d))


def apply_pe_to_qk(
    pos_encoder,
    pe_type: str,
    q: "torch.Tensor",
    k: "torch.Tensor",
    hidden: "torch.Tensor",
    logical_layer_idx: int,
    use_pe: bool,
    position_offset: Optional[int] = None,
) -> tuple:
    """Apply positional encoding to query/key tensors (rotation or augmentation).

    Call this AFTER q/k projection and BEFORE computing attention scores.
    Handles rope, hope, sinusoidal_rotary (rotation) and pape, pape_efficient,
    pape_ri (q/k augmentation). No-op for alibi, bam, nope, none,
    sinusoidal_absolute, learned_absolute (these inject via
    ``apply_pe_to_scores`` as additive biases).

    Args:
        pos_encoder: PE module or ``None``.
        pe_type: PE type string (lowercased).
        q: Query tensor of shape ``(B, H, S, D)``.
        k: Key tensor of shape ``(B, H, S, D)``.
        hidden: Hidden state of shape ``(B, S, H_hidden)`` used by
            content-dependent PEs (PaPE).
        logical_layer_idx: Logical layer index for layer-dependent PE.
        use_pe: Whether PE is enabled for this mixer.
        position_offset: Absolute index of the FIRST token in ``q``/``k``.
            ``None`` positions from 0 (legacy behavior, bit-identical);
            incremental decoding passes the cache write position so keys
            rotate on the same absolute schedule as the cached post-RoPE
            keys (blocker B1).

    Returns:
        Tuple ``(q, k)`` with PE applied.
    """
    if not use_pe or pos_encoder is None:
        return q, k

    pe_type = str(pe_type).lower()
    if pe_type in ("rope", "hope", "sinusoidal_rotary"):
        q = pos_encoder(q, logical_layer_idx=logical_layer_idx, position_offset=position_offset)
        k = pos_encoder(k, logical_layer_idx=logical_layer_idx, position_offset=position_offset)
    elif pe_type in ("pape", "pape_efficient", "pape_ri"):
        bsz, seq_len, _ = hidden.shape
        positions = pos_encoder.default_positions(bsz, seq_len, hidden.device, hidden.dtype)
        q, k = pos_encoder.encode_qk(hidden, q, k, positions)
    # alibi, bam, nope, none, sinusoidal_absolute, learned_absolute: no-op here
    return q, k


def apply_pe_to_scores(
    pos_encoder,
    pe_type: str,
    attn_scores: "torch.Tensor",
    q: "torch.Tensor",
    use_pe: bool,
    kv_len: Optional[int] = None,
    q_offset: int = 0,
    kv_positions: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Apply positional encoding bias to attention scores (ALiBi / BAM + SSMax).

    Call this AFTER computing attention scores and BEFORE softmax/causal mask.

    Handles two additive-bias positional encodings (``alibi``, ``bam``) and
    the transversal Scalable Softmax (SSMax) logit rescale, which is
    independent of the PE type and applies whenever ``pos_encoder.ssmax`` is
    not None.

    Args:
        pos_encoder: PE module or ``None``.
        pe_type: PE type string (lowercased).
        attn_scores: Attention scores of shape ``(B, H, S, kv_len)``.
        q: Query tensor of shape ``(B, H, S, D)`` used to infer
            query length, device and dtype.
        use_pe: Whether PE is enabled for this mixer.
        kv_len: Total number of attended keys (cached + new). ``None``
            defaults to the query count (legacy full-square behavior).
            SSMax uses ``ln(kv_len)`` — its rescale depends on total
            length, so cached decoding must pass the accumulated value
            (blocker B6).
        q_offset: Absolute index of the FIRST query position (blocker B2).
        kv_positions: Optional absolute int64 positions of the cached keys
            ``(kv_len,)``; enables exact ALiBi/BAM distances after
            eviction (gaps in the position sequence).

    Returns:
        Attention scores with PE bias applied (or unchanged).
    """
    if not use_pe or pos_encoder is None:
        return attn_scores

    pe_type = str(pe_type).lower()
    seq_len = q.size(-2)
    effective_kv_len = int(kv_len) if kv_len is not None else seq_len
    if pe_type in ("alibi", "bam"):
        bias = pos_encoder.bias(
            seq_len,
            effective_kv_len,
            q_offset,
            kv_positions=kv_positions,
            device=q.device,
            dtype=q.dtype,
        )
        attn_scores = attn_scores + bias

    # Transversal Scalable Softmax (SSMax) logit rescale. Applies to every PE
    # type when ``use_ssmax=True`` (the factory attaches ``pos_encoder.ssmax``
    # as a child submodule). Rescales logits by ``s * ln(kv_len)``.
    ssmax = getattr(pos_encoder, "ssmax", None)
    if ssmax is not None:
        attn_scores = attn_scores * ssmax.scale(
            effective_kv_len, device=q.device, dtype=q.dtype
        )

    return attn_scores


def dense_attention_forward(
    module,
    x: "torch.Tensor",
    logical_layer_idx: Optional[int] = None,
    pos_encoder=None,
) -> "torch.Tensor":
    """Compute standard dense multi-head attention with ``module``'s weights.

    Training-free mixers (FASA, SpargeAttn) only define their sparse
    selection heuristic for inference. Their enclosing ``HybridLayer`` calls
    this helper in training mode so the sparse mixers are trained as standard
    dense attention (Vaswani et al., 2017) over the *same* projections that
    the sparse forward uses at eval time. Sharing the projections is what
    makes the dense surrogate meaningful: gradients update exactly the
    weights the inference path will read.

    The computation mirrors ``StandardAttention.forward`` (no cache):
    q/k/v projection, positional encoding on q/k, optional additive score
    bias (ALiBi/BAM + SSMax), causal mask in decoder mode, softmax, dropout
    on the attention weights, value aggregation and output projection.

    Args:
        module: Mixer instance exposing ``q_proj``, ``k_proj``, ``v_proj``,
            ``out_proj``, ``dropout``, ``scale``, ``mode``, ``pe_type`` and
            ``use_pe`` (both training-free mixers satisfy this).
        x: Input tensor of shape ``(batch, seq_len, hidden_size)``.
        logical_layer_idx: Logical layer index forwarded to the positional
            encoder.
        pos_encoder: Optional positional encoding module overriding the
            mixer's own ``pos_encoder`` for this call.

    Returns:
        Output tensor of shape ``(batch, seq_len, hidden_size)``.
    """
    pe = pos_encoder if pos_encoder is not None else getattr(module, "pos_encoder", None)
    bsz, seq_len, hidden = x.shape
    num_heads = module.num_heads
    head_dim = module.head_dim
    idx = logical_layer_idx or 0

    q = module.q_proj(x).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    k = module.k_proj(x).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    v = module.v_proj(x).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

    q, k = apply_pe_to_qk(pe, module.pe_type, q, k, x, idx, module.use_pe)

    attn_scores = (q @ k.transpose(-2, -1)) * module.scale
    attn_scores = apply_pe_to_scores(pe, module.pe_type, attn_scores, q, module.use_pe)
    if module.mode == "decoder":
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn_weights = module_softmax(module, attn_scores, dim=-1)
    attn_weights = module.dropout(attn_weights)

    out = (attn_weights @ v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
    return module.out_proj(out)



