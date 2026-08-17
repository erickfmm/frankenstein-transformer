"""Parabolic Position Encoding (PaPE), efficient formulation.

Reformulates the naive PaPE augmentation so that the parabolic attention terms
are computed via matrix products instead of explicit per-element
augmentation. This reduces the size of the augmented query/key tensors and
avoids the redundant ``ones`` columns, yielding a lower-memory and
lower-compute implementation with identical attention scores.

Reference:
    Oehrstroem et al. (2026), "Parabolic Position Encoding for
    Transformers", arXiv:2602.01418. This is the efficient formulation
    (pure-PyTorch, no Triton kernels).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaPEEfficient(nn.Module):
    """Parabolic Position Encoding (efficient formulation).

    Precomputes the position-weight matrix ``W_p`` from the learnable position
    projection and expresses the parabolic terms through matrix products
    against the positions tensor. The augmentation layout is::

        query <- [query, -squares, -a_mat.flatten, right_side, dot_b_pos, b_p]
        key   <- [key, 1, outer(positions, positions), 2*positions, 1, -positions]

    where ``a_mat`` and ``b_p`` are derived from the ``ab`` projection of the
    hidden state and ``W_p``.

    Reference:
        Oehrstroem et al. (2026), "Parabolic Position Encoding for
        Transformers", arXiv:2602.01418.

    Args:
        hidden_size: Hidden dimensionality of the model.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        num_parabolas: Number of learnable parabolic basis functions per head.
        num_positions: Dimensionality of the input position tensor (``1`` for
            1D language).

    Attributes:
        hidden_size: Hidden dimensionality of the model.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        num_parabolas: Number of learnable parabolic basis functions per head.
        num_positions: Dimensionality of the input position tensor.
        num_pad: Number of padding elements added to align the augmented head
            size to a multiple of ``8`` (``0`` when already aligned).
        should_pad: Whether padding is required.
        position: Linear projection from positions to per-head parabolic
            features.
        ab: Linear projection from the hidden state to the parabolic
            coefficients ``a`` and ``b``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_parabolas: int = 4,
        num_positions: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_parabolas = num_parabolas
        self.num_positions = num_positions

        total_head_size = head_dim + num_positions ** 2 + 2 * num_positions + 2
        self.num_pad = 8 * math.ceil(total_head_size / 8) - total_head_size
        self.should_pad = self.num_pad != 0

        self.position = nn.Linear(num_positions, num_heads * num_parabolas, bias=False)
        self.ab = nn.Linear(hidden_size, 2 * num_heads * num_parabolas, bias=False)

    def default_positions(
        self,
        batch_size: int,
        seq_len: int,
        device,
        dtype,
    ) -> torch.Tensor:
        """Build the default 1D position tensor.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            device: Target device.
            dtype: Target data type.

        Returns:
            Tensor of shape ``(batch_size, seq_len, 1)`` with positions
            ``0, 1, ..., seq_len - 1`` replicated across the batch.
        """
        pos = torch.arange(seq_len, device=device, dtype=dtype)
        return pos.view(1, seq_len, 1).expand(batch_size, -1, -1)

    def encode_qk(
        self,
        hidden_state: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
    ):
        """Augment query and key with efficient parabolic positional features.

        Args:
            hidden_state: Hidden tensor of shape ``(B, S, hidden_size)``.
            query: Query tensor of shape ``(B, H, S, head_dim)``.
            key: Key tensor of shape ``(B, H, S, head_dim)``.
            positions: Position tensor of shape ``(B, S, num_positions)``.

        Returns:
            Tuple ``(query_aug, key_aug)`` of tensors with augmented last
            dimension padded to a multiple of ``8`` when required.
        """
        positions = positions.float()
        batch_size, seq_length, _ = positions.size()
        num_heads = self.num_heads
        num_parabolas = self.num_parabolas
        num_positions = self.num_positions

        W_p = self.position.weight.reshape(num_heads, num_parabolas, num_positions)

        ab = self.ab(hidden_state)
        ab = ab.view(batch_size, seq_length, 2, num_heads, num_parabolas)
        ab = ab.permute(2, 0, 3, 1, 4)

        a_pre = ab[0]
        b_pre = ab[1]

        a_softplus = F.softplus(a_pre)

        b_p = torch.einsum("bhsm,hmp->bhsp", b_pre, W_p)

        a_mat = torch.einsum(
            "bhsm,hmp,hmq->bhspq",
            a_softplus,
            W_p,
            W_p,
        )

        positions_h = positions.permute(0, 2, 1)
        right_side = torch.einsum("bhspq,bqs->bhsp", a_mat, positions_h)
        squares = torch.einsum("bhsp,bsp->bhs", right_side, positions).unsqueeze(-1)
        dot_b_pos = torch.einsum("bhsp,bsp->bhs", b_p, positions).unsqueeze(-1)

        ones = torch.ones(
            (batch_size, num_heads, seq_length, 1),
            device=positions.device,
            dtype=positions.dtype,
        )

        a_mat_flat = a_mat.reshape(batch_size, num_heads, seq_length, -1)
        right_side = right_side.reshape(batch_size, num_heads, seq_length, -1)

        query = torch.cat(
            [query, -squares, -a_mat_flat, right_side, dot_b_pos, b_p],
            dim=-1,
        )

        pos_pair = torch.einsum("bsp,bsq->bspq", positions, positions)
        pos_pair = pos_pair.reshape(batch_size, seq_length, -1)
        pos_pair = pos_pair.unsqueeze(1).expand(-1, num_heads, -1, -1)
        pos_2 = positions.unsqueeze(1).expand(-1, num_heads, -1, -1)
        pos_neg = -pos_2

        key = torch.cat(
            [key, ones, pos_pair, 2 * pos_2, ones, pos_neg],
            dim=-1,
        )

        if self.should_pad:
            query = F.pad(query, (0, self.num_pad), "constant", 0)
            key = F.pad(key, (0, self.num_pad), "constant", 0)
        return query, key

    def forward(self, x: torch.Tensor, logical_layer_idx: int = 0) -> torch.Tensor:
        """Return the input tensor unchanged.

        Args:
            x: Input tensor of shape ``(batch, heads, seq_len, head_dim)``.
            logical_layer_idx: Logical layer index (unused; accepted for
                interface compatibility with other positional encodings).

        Returns:
            The input tensor ``x`` unchanged.
        """
        return x