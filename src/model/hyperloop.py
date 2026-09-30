#!/usr/bin/env python3
"""Hyperloop Transformers: loop-level hyper-connections (arXiv:2604.21254).

Implements the *Hyper-Connected Looped Transformer* (Hyperloop Transformer) of
Zeitoun, Torroba-Hennigen & Kim (arXiv:2604.21254). The model is organised as
a **middle-cycle** looped Transformer — begin block → middle block (looped) →
end block — and the looped middle block is wrapped in *hyper-connections*
applied **only at the loop level** (once per loop iteration), instead of after
every sub-layer as in per-layer mHC (arXiv:2512.24880,
:mod:`src.model.mhc`). This adds minimal new parameters and compute while
letting the looped representations deviate more flexibly across iterations.

Per loop ``l``, with loop-specific parameters ``{W_l, b_l, α_l, e_l}`` for
``τ ∈ {pre, post, res}``, and ``z_t = RMSNorm(flatten(y_t^{(l)}))``:

    y_t^{(l+1)} = H_l^{res} · y_t^{(l)} + H_l^{post} ⊗ ( F(H_l^{pre} · y_t^{(l)}) + e_l )

with the input-dependent coefficients

    H_l^{pre}  = σ(α_pre  · (W_pre  z_t) + b_pre)     # read  (n streams → C)
    H_l^{post} = 2σ(α_post · (W_post z_t) + b_post)   # write (C → n streams)
    H_l^{res}  = diag(σ(α_res · (W_res z_t) + b_res)) # mix   (diagonal, paper default)
              | sinkhorn(·)                            # doubly stochastic (mHC-style)
              | identity                               # stateless ablation

Key differences vs per-layer mHC (``use_mhc`` without ``mhc_hyperloop``):

1. Hyper-connections fire **once per loop**, not after every attention/FFN
   sub-layer (the paper's ablations show every-loop is also the most
   performant placement, Table 5).
2. The stream is created by **copying** the begin-block output ``n`` times
   (not a learned projection) and collapsed by **averaging** the streams
   (not a learned projection) before the end block.
3. Parameters are **loop-specific** — each loop owns ``W/b/λ/α/e`` — relaxing
   the strict parameter sharing of ordinary looped Transformers.
4. ``H^{res}`` uses a **diagonal sigmoid** parameterization by default (the
   paper's best ablation, Table 6) instead of the Sinkhorn-Knopp doubly
   stochastic projection; ``identity`` and full mHC-style ``sinkhorn`` remain
   available for ablations.
5. Mixture-of-Depths is compatible: the middle-block layers run the standard
   C-dimensional path (no per-layer n-stream residual).

Reference: Zeitoun, Torroba-Hennigen & Kim, "Hyperloop Transformers",
arXiv:2604.21254. Hyper-connections: Zhu et al. (ICLR 2026) and
mHC: Xie et al., arXiv:2512.24880.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .attention.common import BitLinear
from .mhc import SinkhornKnoppFunction


class HyperloopConnections(nn.Module):
    """Loop-level hyper-connections for a looped middle block.

    Expands the C-dimensional stream into ``n`` parallel streams (by
    copying), then — once per loop iteration — reads a C-dimensional layer
    input from the stream, runs the middle block externally, and writes the
    block output back into the mixed stream together with a loop position
    embedding ``e_l``.

    All parameters are **per loop**: ``projs[l]``, ``biases[l]``, the three
    gating scalars ``α`` (``alphas_pre/alphas_post/alphas_res``) and the loop
    position embedding ``loop_pos_embs[l]``.

    Attributes:
        expansion_rate: Stream expansion factor ``n``.
        num_loops: Number of loop iterations (parameter groups).
        res_parameterization: ``"diagonal"`` (paper default), ``"sinkhorn"``
            (doubly stochastic, mHC-style) or ``"identity"``.
        sinkhorn_iters: Sinkhorn-Knopp rounds for the ``"sinkhorn"`` mode.
        gating_init: Initial value of the learnable gating scalars ``α``.
        projs: Per-loop linear maps ``φ_l`` from ``R^{nC}`` to the coefficient
            vector (size depends on ``res_parameterization``).
        biases: Per-loop static mappings ``b_l``.
        alphas_pre / alphas_post / alphas_res: Per-loop gating scalars
            (``alphas_res`` is empty under ``"identity"``).
        loop_pos_embs: Per-loop position embeddings ``e_l ∈ R^C``
            (zero-initialised so the recurrence starts as close to the plain
            looped Transformer as possible).
    """

    VALID_RES_PARAMETERIZATIONS = ("diagonal", "sinkhorn", "identity")

    def __init__(
        self,
        hidden_size: int,
        num_loops: int,
        expansion_rate: int = 4,
        gating_init: float = 0.01,
        res_parameterization: str = "diagonal",
        sinkhorn_iters: int = 20,
        use_bitnet: bool = True,
        full_prec_under_bitnet: bool = True,
    ):
        """Initialise the Hyperloop hyper-connection module.

        Args:
            hidden_size: Internal (per-stream) hidden dimension ``C``.
            num_loops: Number of loop iterations over the middle block. Each
                iteration owns its own ``W/b/α/e`` parameters.
            expansion_rate: Stream expansion factor ``n``. Defaults to ``4``
                (paper recommendation, inherited from mHC).
            gating_init: Initial value of the learnable gating scalars.
                Defaults to ``0.01``.
            res_parameterization: ``"diagonal"`` (paper default), ``"sinkhorn"``
                or ``"identity"``.
            sinkhorn_iters: Sinkhorn-Knopp normalisation rounds used by the
                ``"sinkhorn"`` mode. Defaults to ``20``.
            use_bitnet: Whether the backbone is in BitNet mode.
            full_prec_under_bitnet: If ``True`` (default), keep ``φ_l`` a
                full-precision ``nn.Linear`` even under BitNet to avoid
                ternary-quantisation noise on the small coefficients. If
                ``False`` and ``use_bitnet`` is ``True``, use ``BitLinear``.

        Raises:
            ValueError: If ``num_loops < 1``, ``expansion_rate < 1`` or
                ``res_parameterization`` is not one of the supported values.
        """
        super().__init__()
        if int(num_loops) < 1:
            raise ValueError(f"num_loops must be >= 1, got {num_loops}")
        if int(expansion_rate) < 1:
            raise ValueError(f"expansion_rate must be >= 1, got {expansion_rate}")
        res_parameterization = str(res_parameterization).lower()
        if res_parameterization not in self.VALID_RES_PARAMETERIZATIONS:
            raise ValueError(
                f"res_parameterization must be one of "
                f"{self.VALID_RES_PARAMETERIZATIONS}, got {res_parameterization!r}"
            )

        n = int(expansion_rate)
        self.hidden_size = int(hidden_size)
        self.expansion_rate = n
        self.num_loops = int(num_loops)
        self.res_parameterization = res_parameterization
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.gating_init = float(gating_init)

        # Coefficient vector layout: [pre (n) | post (n) | res (n | n² | —)].
        if res_parameterization == "diagonal":
            out_dim = 3 * n
        elif res_parameterization == "sinkhorn":
            out_dim = n * n + 2 * n
        else:  # identity — no res head at all
            out_dim = 2 * n

        in_dim = n * hidden_size
        proj_cls = BitLinear if (use_bitnet and not full_prec_under_bitnet) else nn.Linear
        self.projs = nn.ModuleList(
            [proj_cls(in_dim, out_dim, bias=False) for _ in range(self.num_loops)]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, out_dim)) for _ in range(self.num_loops)]
        )
        self.alphas_pre = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(gating_init))) for _ in range(self.num_loops)]
        )
        self.alphas_post = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(gating_init))) for _ in range(self.num_loops)]
        )
        if res_parameterization == "identity":
            self.alphas_res = nn.ParameterList([])
        else:
            self.alphas_res = nn.ParameterList(
                [nn.Parameter(torch.tensor(float(gating_init))) for _ in range(self.num_loops)]
            )
        # Loop position embedding e_l (zero-init: at initialisation the update
        # reduces to the plain looped-Transformer recurrence).
        self.loop_pos_embs = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_size)) for _ in range(self.num_loops)]
        )

    def mappings(self, x: torch.Tensor, loop_idx: int):
        """Compute ``H[pre]``, ``H[post]`` and ``H[res]`` for loop ``loop_idx``.

        Args:
            x: Stream of shape ``(..., n, C)``.
            loop_idx: Zero-based loop iteration (selects the parameter group).

        Returns:
            Tuple ``(h_pre, h_post, h_res)`` with shapes ``(..., n)``,
            ``(..., n)`` and — per parameterization — ``(..., n)`` (diagonal
            coefficients), ``(..., n, n)`` (sinkhorn) or ``None`` (identity).
        """
        n = self.expansion_rate
        loop_idx = int(loop_idx) % self.num_loops
        shape = x.shape[:-2]
        x_flat = x.reshape(*shape, -1)  # (..., n*C)
        h = self.projs[loop_idx](x_flat)
        r = x_flat.norm(dim=-1, keepdim=True) / float(x_flat.shape[-1]) ** 0.5
        r = r.clamp_min(1e-6)
        h = (1.0 / r) * h

        h_pre = torch.sigmoid(
            h[..., :n] * self.alphas_pre[loop_idx] + self.biases[loop_idx][..., :n]
        )
        h_post = 2.0 * torch.sigmoid(
            h[..., n : 2 * n] * self.alphas_post[loop_idx] + self.biases[loop_idx][..., n : 2 * n]
        )

        if self.res_parameterization == "identity":
            h_res = None
        elif self.res_parameterization == "diagonal":
            sl = self.biases[loop_idx].shape[-1] - n
            h_res = torch.sigmoid(
                h[..., sl:] * self.alphas_res[loop_idx] + self.biases[loop_idx][..., sl:]
            )
        else:  # sinkhorn — full doubly stochastic mixing (mHC-style)
            sl = self.biases[loop_idx].shape[-1] - n * n
            h_res = h[..., sl:] * self.alphas_res[loop_idx] + self.biases[loop_idx][..., sl:]
            h_res = SinkhornKnoppFunction.apply(h_res.reshape(*shape, n, n), self.sinkhorn_iters)
        return h_pre, h_post, h_res

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """Copy the C-dim stream into ``n`` parallel streams.

        Args:
            x: Stream of shape ``(..., C)``.

        Returns:
            Stream of shape ``(..., n, C)`` (all ``n`` copies identical).
        """
        n = self.expansion_rate
        return x.unsqueeze(-2).expand(*x.shape[:-1], n, x.shape[-1]).contiguous()

    def collapse(self, x: torch.Tensor) -> torch.Tensor:
        """Average the ``n`` parallel streams back to one C-dim stream.

        Args:
            x: Stream of shape ``(..., n, C)``.

        Returns:
            Stream of shape ``(..., C)``.
        """
        return x.mean(dim=-2)

    def apply_fpre(self, x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        """Project the stream down to a middle-block input using ``H[pre]``.

        Args:
            x: Input stream of shape ``(..., n, C)``.
            h_pre: ``H[pre]`` of shape ``(..., n)``.

        Returns:
            Middle-block input of shape ``(..., C)``.
        """
        return torch.einsum("...n,...nC->...C", h_pre, x)

    def apply_update(
        self,
        x: torch.Tensor,
        loop_out: torch.Tensor,
        h_post: torch.Tensor,
        h_res: Optional[torch.Tensor],
        loop_idx: int,
    ) -> torch.Tensor:
        """Write the middle-block output back into the n-stream residual.

        Implements ``y' = H[res] · y + H[post] ⊗ (loop_out + e_l)``.

        Args:
            x: Input stream of shape ``(..., n, C)``.
            loop_out: Middle-block output of shape ``(..., C)``.
            h_post: ``H[post]`` of shape ``(..., n)``.
            h_res: Diagonal coefficients ``(..., n)``, full matrix
                ``(..., n, n)``, or ``None`` for the identity mixing.
            loop_idx: Zero-based loop iteration (selects ``e_l``).

        Returns:
            Updated stream of shape ``(..., n, C)``.
        """
        loop_idx = int(loop_idx) % self.num_loops
        out = loop_out + self.loop_pos_embs[loop_idx]
        post_part = torch.einsum("...n,...C->...nC", h_post, out)
        if h_res is None:
            res_part = x
        elif h_res.dim() == x.dim() - 1:  # diagonal coefficients (..., n)
            res_part = x * h_res.unsqueeze(-1)
        else:  # full matrix (..., n, n) — sinkhorn mode
            res_part = torch.einsum("...nm,...mC->...nC", h_res, x)
        return res_part + post_part

    def extra_state(self) -> dict:
        """Return non-parameter state useful for debugging / logging.

        Returns:
            A dict of named state entries.
        """
        return {
            "type": type(self).__name__,
            "expansion_rate": self.expansion_rate,
            "num_loops": self.num_loops,
            "res_parameterization": self.res_parameterization,
        }


__all__ = ["HyperloopConnections"]