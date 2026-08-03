"""Native Sparse Attention (Yuan et al., 2025; arXiv:2502.11089).

Implements a hardware-aligned sparse attention mechanism with three
complementary branches: compressed token blocks, selected fine-grained tokens,
and a local sliding window. Learned gating weights dynamically blend the
outputs of all three branches, enabling the model to adaptively trade off
between coarse long-range context and fine-grained local detail. The design
is aligned with modern GPU hardware characteristics for efficient training
and inference.

Reference:
    Yuan, J., Gao, H., Shi, D., Li, X., Liu, B., Chen, Z., Li, Z., Zhao, H.,
    & Li, Z. (2025).
    "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention."
    arXiv:2502.11089.
"""

from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class NSAAttention(nn.Module):
    """Three-branch native sparse attention with learned gating.

    Attention is computed through three parallel branches:

    1. **Compress branch**: Keys and values are aggregated into coarse blocks
       via strided mean-pooling, providing a compressed long-range context.
    2. **Select branch**: The most important blocks (identified from compress
       attention scores) are expanded to fine-grained token-level selection,
       giving precise access to relevant distant tokens.
    3. **Window branch**: A local sliding window of recent tokens captures
       fine-grained short-range patterns.

    A learned gating network (MLP + sigmoid) outputs three scalar weights per
    head per position, dynamically blending the three branch outputs. The
    architecture is designed to be hardware-friendly, with block-aligned
    operations that map efficiently to GPU tensor cores.

    Attributes:
        hidden_size (int): Dimensionality of the input and output embeddings.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        scale (float): Scaling factor for dot-product scores (1 / sqrt(head_dim)).
        block_size (int): Block size for the compress branch.
        stride (int): Stride between consecutive compress blocks.
        select_block_size (int): Block size for fine-grained token selection.
        n_select (int): Number of top blocks to select for fine-grained attention.
        window_size (int): Size of the local sliding window.
        q_proj (nn.Module): Query projection (BitLinear or nn.Linear).
        k_proj (nn.Module): Key projection (BitLinear or nn.Linear).
        v_proj (nn.Module): Value projection (BitLinear or nn.Linear).
        out_proj (nn.Module): Output projection (BitLinear or nn.Linear).
        compress_k (nn.Linear): Linear layer to compress key blocks.
        compress_v (nn.Linear): Linear layer to compress value blocks.
        gate (nn.Sequential): MLP + sigmoid gating network producing 3 weights
            per head per position.
        dropout (nn.Dropout): Dropout applied to the blended output.
        mode (str): ``"encoder"`` or ``"decoder"``. Decoder mode applies causal
            masking to the compress branch.

    Reference:
        Yuan, J., Gao, H., Shi, D., Li, X., Liu, B., Chen, Z., Li, Z., Zhao, H.,
        & Li, Z. (2025).
        "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention."
        arXiv:2502.11089.
    """

    def __init__(self, config):
        """Initialize NSAAttention.

        Args:
            config: Configuration object with the following expected attributes:
                hidden_size (int): Model hidden dimension.
                num_heads (int): Number of attention heads. Must divide
                    ``hidden_size`` evenly.
                dropout (float): Dropout probability for the blended output.
                use_bitnet (bool): If True, use BitLinear projections.
                nsa_block_size (int, optional): Block size for compress branch.
                    Defaults to 32.
                nsa_stride (int, optional): Stride between compress blocks.
                    Defaults to 16.
                nsa_select_block_size (int, optional): Block size for select
                    branch. Defaults to 64.
                nsa_n_select (int, optional): Number of top blocks to select.
                    Defaults to 16.
                nsa_window_size (int, optional): Local window size.
                    Defaults to 512.
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
            raise ValueError("hidden_size must be divisible by num_heads for NSAAttention")

        self.block_size = max(1, int(getattr(config, "nsa_block_size", 32)))
        self.stride = max(1, int(getattr(config, "nsa_stride", 16)))
        self.select_block_size = max(1, int(getattr(config, "nsa_select_block_size", 64)))
        self.n_select = max(1, int(getattr(config, "nsa_n_select", 16)))
        self.window_size = max(1, int(getattr(config, "nsa_window_size", 512)))

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        router_cls = BitLinear if (config.use_bitnet and getattr(config, "bitnet_routers", False)) else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)

        self.compress_k = router_cls(self.block_size * self.head_dim, self.head_dim)
        self.compress_v = router_cls(self.block_size * self.head_dim, self.head_dim)
        self.gate = nn.Sequential(router_cls(self.head_dim, 3), nn.Sigmoid())
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

    def _compress(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress keys and values into coarse-grained blocks.

        Partitions the key and value sequences into blocks of size
        ``block_size`` with stride ``stride``, then projects each flattened
        block down to ``head_dim`` using learned linear layers. If the
        sequence is shorter than ``block_size``, it is padded and compressed
        into a single block.

        Args:
            k (torch.Tensor): Key tensor of shape ``(batch, heads, seq_len, head_dim)``.
            v (torch.Tensor): Value tensor of shape ``(batch, heads, seq_len, head_dim)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Compressed keys and values,
                each of shape ``(batch, heads, n_blocks, head_dim)``.
        """
        bsz, heads, seq_len, dim = k.shape

        if seq_len <= self.block_size:
            pad_len = self.block_size - seq_len
            if pad_len > 0:
                k_pad = F.pad(k, (0, 0, 0, pad_len))
                v_pad = F.pad(v, (0, 0, 0, pad_len))
            else:
                k_pad = k
                v_pad = v

            comp_k = self.compress_k(k_pad.reshape(bsz, heads, -1)).unsqueeze(2)
            comp_v = self.compress_v(v_pad.reshape(bsz, heads, -1)).unsqueeze(2)
            return comp_k, comp_v

        n_blocks = (seq_len - self.block_size) // self.stride + 1
        comp_k_list = []
        comp_v_list = []
        for i in range(n_blocks):
            start = i * self.stride
            end = start + self.block_size
            k_block = k[:, :, start:end, :].reshape(bsz, heads, -1)
            v_block = v[:, :, start:end, :].reshape(bsz, heads, -1)
            comp_k_list.append(self.compress_k(k_block))
            comp_v_list.append(self.compress_v(v_block))

        comp_k = torch.stack(comp_k_list, dim=2)
        comp_v = torch.stack(comp_v_list, dim=2)
        return comp_k, comp_v

    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None) -> torch.Tensor:
        """Compute three-branch native sparse attention.

        Executes all three branches in parallel:

        1. **Compress**: Compresses keys/values into blocks, computes attention
           over compressed representations.
        2. **Select**: Uses compress attention scores to identify top blocks,
           then gathers fine-grained tokens from those blocks for precise
           attention.
        3. **Window**: Computes attention over a local sliding window of the
           most recent tokens.

        The three outputs are blended via learned per-head per-position gating
        weights from ``self.gate``.

        In decoder mode, the compress branch is causally masked.

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

        comp_k, comp_v = self._compress(k, v)
        comp_scores = torch.matmul(q, comp_k.transpose(-2, -1)) * self.scale
        if self.mode == "decoder":
            c_len = comp_k.shape[2]
            causal_c = torch.triu(torch.ones(seq_len, c_len, device=x.device, dtype=torch.bool), diagonal=1)
            comp_scores = comp_scores.masked_fill(causal_c.unsqueeze(0).unsqueeze(0), float("-inf"))
        comp_attn = F.softmax(comp_scores, dim=-1)
        out_comp = torch.matmul(comp_attn, comp_v)

        block_importance = comp_attn.mean(dim=1).mean(dim=1)
        n_sel = min(self.n_select, block_importance.shape[-1])
        _, top_blocks = block_importance.topk(n_sel, dim=-1)

        sel_indices = []
        for b_idx in top_blocks.unbind(-1):
            start = b_idx * self.stride
            idx = torch.arange(self.select_block_size, device=x.device).unsqueeze(0) + start.unsqueeze(-1)
            sel_indices.append(idx.clamp(max=seq_len - 1))

        sel_idx = torch.cat(sel_indices, dim=-1)
        sel_idx = sel_idx.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, self.head_dim)
        k_sel = k.gather(2, sel_idx)
        v_sel = v.gather(2, sel_idx)

        sel_scores = torch.matmul(q, k_sel.transpose(-2, -1)) * self.scale
        sel_attn = F.softmax(sel_scores, dim=-1)
        out_sel = torch.matmul(sel_attn, v_sel)

        win_size = min(self.window_size, seq_len)
        k_win = k[:, :, -win_size:, :]
        v_win = v[:, :, -win_size:, :]
        win_scores = torch.matmul(q, k_win.transpose(-2, -1)) * self.scale
        win_attn = F.softmax(win_scores, dim=-1)
        out_win = torch.matmul(win_attn, v_win)

        gates = self.gate(q.mean(dim=2))
        g_comp = gates[..., 0:1].unsqueeze(2)
        g_sel = gates[..., 1:2].unsqueeze(2)
        g_win = gates[..., 2:3].unsqueeze(2)

        out = g_comp * out_comp + g_sel * out_sel + g_win * out_win
        out = self.dropout(out)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.out_proj(out)
