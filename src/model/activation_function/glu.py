"""Gated linear unit (GLU) feed-forward blocks.

SwiGLU, GEGLU, and ReGLU are gated FFN units that replace the standard
``Linear -> activation -> Linear`` block with two projections whose elementwise
product forms the gate::

    FFN(x) = (act(x W_gate + b_gate)) * (x W_up + b_up),  then down-project.

They are not pure elementwise activations (they own linear weights), so they
are implemented here as drop-in FFN modules consumed by ``HybridLayer``.

Reference: Shazeer (2020), *GLU Variants Improve Transformer*, arXiv:2002.05202.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFFN(nn.Module):
    """Gated feed-forward network (SwiGLU / GEGLU / ReGLU).

    Computes::

        y = dropout( down( act(gate_proj(x)) * up_proj(x) ) )

    where ``gate_proj`` and ``up_proj`` map ``hidden -> intermediate`` and
    ``down_proj`` maps ``intermediate -> hidden``.

    Args:
        hidden_size: Input/output feature dimension.
        intermediate_size: Gated intermediate dimension.
        gate_fn: Elementwise gating function applied to the gate projection.
            One of ``F.silu`` (SwiGLU), ``F.gelu`` (GEGLU), or ``F.relu``
            (ReGLU). Default: ``F.silu`` (SwiGLU).
        bias: Whether to include bias in the projections. Default: ``False``.
        dropout: Dropout probability after gating. Default: ``0.0``.
        proj_factory: Optional callable ``(in, out, bias) -> nn.Module`` used
            to build the projections (e.g. :class:`BitLinear` for BitNet
            compatibility). When ``None``, plain :class:`nn.Linear` is used.

    Attributes:
        gate_proj: Linear ``hidden -> intermediate`` feeding the gate function.
        up_proj: Linear ``hidden -> intermediate`` value pathway.
        down_proj: Linear ``intermediate -> hidden`` output projection.

    Reference: Shazeer (2020), arXiv:2002.05202.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        gate_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        bias: bool = False,
        dropout: float = 0.0,
        proj_factory: Callable[[int, int, bool], nn.Module] | None = None,
    ):
        super().__init__()
        build = proj_factory or (lambda i, o, b: nn.Linear(i, o, bias=b))
        self.gate_proj = build(hidden_size, intermediate_size, bias)
        self.up_proj = build(hidden_size, intermediate_size, bias)
        self.down_proj = build(intermediate_size, hidden_size, bias)
        self.gate_fn = gate_fn
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated FFN.

        Args:
            x: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Tensor of shape ``(..., hidden_size)``.
        """
        gate = self.gate_fn(self.gate_proj(x))
        value = self.up_proj(x)
        return self.dropout(self.down_proj(gate * value))


def make_gated_ffn(
    kind: str,
    hidden_size: int,
    intermediate_size: int,
    bias: bool = False,
    dropout: float = 0.0,
    proj_factory: Callable[[int, int, bool], nn.Module] | None = None,
) -> GatedFFN:
    """Build a :class:`GatedFFN` for the requested GLU variant.

    Args:
        kind: One of ``"swiglu"``, ``"geglu"``, ``"reglu"``.
        hidden_size: Input/output feature dimension.
        intermediate_size: Gated intermediate dimension.
        bias: Whether projections include bias. Default: ``False``.
        dropout: Dropout probability after gating. Default: ``0.0``.
        proj_factory: Optional projection factory (e.g. BitLinear).

    Returns:
        A :class:`GatedFFN` configured for the requested variant.

    Raises:
        ValueError: If ``kind`` is not a recognized GLU variant.
    """
    gate_map = {
        "swiglu": F.silu,
        "geglu": F.gelu,
        "reglu": F.relu,
    }
    gate_fn = gate_map.get(kind)
    if gate_fn is None:
        raise ValueError(
            f"Unknown GLU variant {kind!r}. Expected one of {sorted(gate_map)}."
        )
    return GatedFFN(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        gate_fn=gate_fn,
        bias=bias,
        dropout=dropout,
        proj_factory=proj_factory,
    )


__all__ = [
    "GatedFFN",
    "make_gated_ffn",
]
