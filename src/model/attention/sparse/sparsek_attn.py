"""SparseK attention (Lou et al., 2024; arXiv:2406.16747).

Implements a differentiable top-k selection mechanism for sparse attention.
Instead of using fixed or random patterns, SparseK learns to select the most
relevant key-value pairs for each query using a learned scoring network and a
differentiable top-k operator. This achieves linear time complexity during
training and constant memory usage during autoregressive generation, as only
the top-k keys and values are retained.

Reference:
    Lou, C., Jia, Z., & Tu, Z. (2024).
    "SparseK: Differentiable Top-k Sparse Attention for Long-Context LLMs."
    arXiv:2406.16747.
"""

from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class SparseKOperator(torch.autograd.Function):
    """Differentiable top-k projection operator for SparseK attention.

    Implements a differentiable approximation of the top-k selection operation
    using the method described in Lou et al. (2024). The forward pass computes
    a threshold such that exactly ``k`` elements survive after clamping, and
    the backward pass propagates gradients only through the selected elements
    with a mean-subtraction correction to preserve gradient flow.

    This is a ``torch.autograd.Function`` subclass and should be invoked via
    ``SparseKOperator.apply(scores, k)``.
    """

    @staticmethod
    def forward(ctx, scores: torch.Tensor, k: int):
        """Forward pass: compute thresholded top-k projection.

        Sorts scores descending, computes cumulative sums, and determines a
        per-row threshold ``tau`` such that ``(scores - tau).clamp(min=0)``
        yields exactly ``k`` non-zero elements per row.

        Args:
            ctx: Autograd context for saving tensors.
            scores (torch.Tensor): Input scores of shape ``(batch, heads, seq_len, seq_len)``.
            k (int): Number of elements to retain per row.

        Returns:
            torch.Tensor: Thresholded scores of the same shape as input,
                with at most ``k`` non-zero entries per row.
        """
        sorted_scores, _ = scores.sort(dim=-1, descending=True)
        cumsum = sorted_scores.cumsum(dim=-1)
        arange = torch.arange(1, scores.shape[-1] + 1, device=scores.device, dtype=scores.dtype)
        threshold = (cumsum - float(k)) / arange
        support = sorted_scores > threshold
        rho = support.sum(dim=-1, keepdim=True).clamp(min=1)
        tau = (cumsum.gather(-1, rho - 1) - float(k)) / rho.to(scores.dtype)
        out = (scores - tau).clamp(min=0)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass: propagate gradients through selected elements only.

        Gradients flow only through positions where the forward output was
        positive. A mean-subtraction correction is applied within each row's
        support set to ensure unbiased gradient estimates.

        Args:
            ctx: Autograd context with saved tensors.
            grad_output (torch.Tensor): Gradient of the loss with respect to
                the forward output.

        Returns:
            tuple: ``(grad_scores, None)`` where ``grad_scores`` has the same
                shape as the forward input and ``None`` is the gradient for
                the non-differentiable ``k`` parameter.
        """
        (out,) = ctx.saved_tensors
        support = (out > 0).to(grad_output.dtype)
        n_support = support.sum(dim=-1, keepdim=True).clamp(min=1)
        grad = support * (grad_output - (grad_output * support).sum(dim=-1, keepdim=True) / n_support)
        return grad, None


class SparseKAttention(nn.Module):
    """Differentiable top-k sparse attention with learned key-value selection.

    Uses a small feed-forward scoring network to assign importance scores to
    each key position. A differentiable top-k operator selects the ``k`` most
    important keys, and scaled dot-product attention is computed only over the
    selected subset. This yields O(n * k) complexity where ``k`` is a fixed
    constant, independent of sequence length.

    In decoder mode, future positions among the selected keys are masked out
    using position-based causal filtering.

    Attributes:
        hidden_size (int): Dimensionality of the input and output embeddings.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        scale (float): Scaling factor for dot-product scores (1 / sqrt(head_dim)).
        k (int): Number of key-value pairs to select per query (top-k).
        q_proj (nn.Module): Query projection (BitLinear or nn.Linear).
        k_proj (nn.Module): Key projection (BitLinear or nn.Linear).
        v_proj (nn.Module): Value projection (BitLinear or nn.Linear).
        out_proj (nn.Module): Output projection (BitLinear or nn.Linear).
        score_net (nn.Sequential): Two-layer MLP that scores each key position
            for selection importance.
        dropout (nn.Dropout): Dropout applied to attention weights.
        mode (str): ``"encoder"`` or ``"decoder"``. Decoder mode applies causal
            masking on the selected key subset.

    Reference:
        Lou, C., Jia, Z., & Tu, Z. (2024).
        "SparseK: Differentiable Top-k Sparse Attention for Long-Context LLMs."
        arXiv:2406.16747.
    """

    def __init__(self, config):
        """Initialize SparseKAttention.

        Args:
            config: Configuration object with the following expected attributes:
                hidden_size (int): Model hidden dimension.
                num_heads (int): Number of attention heads. Must divide
                    ``hidden_size`` evenly.
                dropout (float): Dropout probability for attention weights.
                use_bitnet (bool): If True, use BitLinear projections.
                sparsek_k (int, optional): Number of keys to select per query.
                    Defaults to 128.
                mode (str, optional): ``"encoder"`` or ``"decoder"``.
                    Defaults to ``"encoder"``.

        Raises:
            ValueError: If ``hidden_size`` is not divisible by ``num_heads``.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for SparseKAttention")

        self.k = max(1, int(getattr(config, "sparsek_k", 128)))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        router_cls = BitLinear if (config.use_bitnet and getattr(config, "bitnet_routers", False)) else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)

        self.score_net = nn.Sequential(
            router_cls(self.head_dim, self.head_dim),
            nn.ReLU(),
            router_cls(self.head_dim, 1),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute SparseK attention with differentiable top-k selection.

        Projects queries, keys, and values. Scores each key position using
        ``score_net``, applies the differentiable top-k operator to select
        the most important keys, gathers the corresponding key-value pairs,
        and computes scaled dot-product attention over the selected subset.

        In decoder mode, selected keys whose original positions are ahead of
        the query position are masked out.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch, seq_len, hidden_size)``.
            logical_layer_idx (Optional[int]): Logical layer index for
                potential layer-specific behavior. Not used by this
                implementation.

        Returns:
            torch.Tensor: Output tensor of shape ``(batch, seq_len, hidden_size)``.
        """
        bsz, seq_len, hidden = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_scores = self.score_net(k).squeeze(-1)
        selection = SparseKOperator.apply(kv_scores, self.k)

        k_actual = min(self.k, seq_len)
        _, top_idx = selection.topk(k_actual, dim=-1)
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)

        k_sel = k.gather(2, top_idx_exp)
        v_sel = v.gather(2, top_idx_exp)

        scores = torch.matmul(q, k_sel.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.mode == "decoder":
            # Mask future positions: top_idx holds key positions per head
            q_pos = torch.arange(seq_len, device=x.device).view(1, 1, seq_len, 1)
            k_pos = top_idx.unsqueeze(2).expand_as(scores)
            scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v_sel).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
