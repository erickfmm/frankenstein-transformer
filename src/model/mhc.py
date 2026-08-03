#!/usr/bin/env python3
"""Manifold-Constrained Hyper-Connections (mHC).

Implements arXiv:2512.24880 — a residual-connection framework that expands the
residual stream width by a factor ``n`` and constrains the stream-mixing
mapping ``H[res]`` to the Birkhoff polytope (the set of doubly stochastic
matrices) via the Sinkhorn-Knopp projection. This restores the identity-mapping
property lost by unconstrained Hyper-Connections (HC), stabilising training at
scale.

Given an ``n``-stream residual ``x_l ∈ R^{n×C}``, a layer computes:

    x̃_l      = vec(x_l) ∈ R^{1×nC}
    H̃        = (1/r)·(α ⊙ (x̃_l φ_l)) + b_l          # r = RMS(x̃_l)
    H_l[pre]  = σ(H̃_pre)
    H_l[post] = 2σ(H̃_post)
    H_l[res]  = SinkhornKnopp(exp(H̃_res))            # doubly stochastic
    Fpre      = H_l[pre] @ x_l                        # [1, C]
    x_{l+1}   = H_l[res] @ x_l + H_l[post]ᵀ ⊗ F(Fpre, W_l)

When ``n = 1`` the doubly stochastic constraint degenerates to scalar 1,
recovering the exact identity mapping.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention.common import BitLinear


def _sinkhorn_iter(P: torch.Tensor, iters: int) -> torch.Tensor:
    """Run Sinkhorn-Knopp row/column normalisations on an exp-activated matrix."""
    for _ in range(iters):
        P = P / P.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        P = P / P.sum(dim=-2, keepdim=True).clamp_min(1e-6)
    return P


class SinkhornKnoppFunction(torch.autograd.Function):
    """Differentiable Sinkhorn-Knopp projection onto the Birkhoff polytope.

    ``forward`` normalises ``exp(X)`` to a doubly stochastic matrix. The
    ``backward`` recomputes the full iteration under ``torch.enable_grad`` and
    differentiates through it to obtain the exact Jacobian-vector product (the
    recompute-on-chip strategy described in the paper).
    """

    @staticmethod
    def forward(ctx, X: torch.Tensor, iters: int) -> torch.Tensor:
        """Project ``exp(X)`` to a doubly stochastic matrix.

        Args:
            X: Input of shape ``(..., n, n)``.
            iters: Number of Sinkhorn-Knopp row/column normalisation rounds.

        Returns:
            Doubly stochastic matrix of shape ``(..., n, n)``.
        """
        ctx.iters = int(iters)
        ctx.save_for_backward(X)
        P = _sinkhorn_iter(torch.exp(X), ctx.iters)
        return P

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple:
        """Backpropagate through the Sinkhorn-Knopp iteration.

        Args:
            grad_out: Gradient of the output, shape ``(..., n, n)``.

        Returns:
            Tuple ``(grad_input, None)`` with ``grad_input`` of the same shape.
        """
        (X,) = ctx.saved_tensors
        with torch.enable_grad():
            Xr = X.detach().requires_grad_(True)
            P = _sinkhorn_iter(torch.exp(Xr), ctx.iters)
            (grad_X,) = torch.autograd.grad(P, Xr, grad_out)
        return grad_X, None


class ManifoldHyperConnections(nn.Module):
    """mHC residual-stream read/write/update module.

    Expands the residual stream to width ``n × hidden_size`` and maintains it
    across layers. The per-layer coefficients ``H[pre]``, ``H[post]``,
    ``H[res]`` are computed from the flattened input via a learned linear
    projection ``φ_l``, a learned bias ``b_l`` and three learnable gating
    scalars ``α``.

    Attributes:
        expansion_rate: Stream expansion factor ``n``.
        sinkhorn_iters: Sinkhorn-Knopp normalisation rounds for ``H[res]``.
        gating_init: Initial value of the learnable gating scalars ``α``.
        proj: Linear map ``φ_l`` from ``R^{nC}`` to ``R^{n²+2n}``.
        bias: Learnable static mapping ``b_l`` of shape ``(1, n²+2n)``.
        alpha_pre: Learnable gating scalar for ``H[pre]``.
        alpha_post: Learnable gating scalar for ``H[post]``.
        alpha_res: Learnable gating scalar for ``H[res]``.
    """

    def __init__(
        self,
        hidden_size: int,
        expansion_rate: int = 4,
        sinkhorn_iters: int = 20,
        gating_init: float = 0.01,
        use_bitnet: bool = True,
        full_prec_under_bitnet: bool = True,
    ):
        """Initialise the mHC residual module.

        Args:
            hidden_size: Internal (per-stream) hidden dimension ``C``.
            expansion_rate: Stream expansion factor ``n``. ``1`` recovers the
                identity mapping. Defaults to ``4``.
            sinkhorn_iters: Sinkhorn-Knopp normalisation rounds. Defaults to
                ``20``.
            gating_init: Initial value of the learnable gating scalars.
                Defaults to ``0.01``.
            use_bitnet: Whether the backbone is in BitNet mode.
            full_prec_under_bitnet: If ``True`` (default), keep ``φ_l`` a
                full-precision ``nn.Linear`` even under BitNet to avoid
                ternary-quantisation noise on the small mHC coefficients. If
                ``False`` and ``use_bitnet`` is ``True``, use ``BitLinear``.
        """
        super().__init__()
        n = int(expansion_rate)
        self.expansion_rate = n
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.gating_init = float(gating_init)

        in_dim = n * hidden_size
        out_dim = n * n + 2 * n
        if use_bitnet and not full_prec_under_bitnet:
            self.proj = BitLinear(in_dim, out_dim, bias=False)
        else:
            self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(1, out_dim))
        self.alpha_pre = nn.Parameter(torch.tensor(float(gating_init)))
        self.alpha_post = nn.Parameter(torch.tensor(float(gating_init)))
        self.alpha_res = nn.Parameter(torch.tensor(float(gating_init)))

    def mappings(self, x: torch.Tensor):
        """Compute ``H[pre]``, ``H[post]`` and ``H[res]`` from the stream.

        Args:
            x: Stream of shape ``(..., n, C)``.

        Returns:
            Tuple ``(h_pre, h_post, h_res)`` with shapes ``(..., n)``,
            ``(..., n)`` and ``(..., n, n)``.
        """
        n = self.expansion_rate
        shape = x.shape[:-2]
        x_flat = x.reshape(*shape, -1)  # (..., n*C)
        h = self.proj(x_flat)           # (..., n² + 2n)
        r = x_flat.norm(dim=-1, keepdim=True) / float(x_flat.shape[-1]) ** 0.5
        r = r.clamp_min(1e-6)

        # Eq. 16: \hat{H} = (1/r)(α · \tilde{H}) + b
        h = (1.0 / r) * h

        h_pre = h[..., :n] * self.alpha_pre + self.bias[..., :n]
        h_post = h[..., n : 2 * n] * self.alpha_post + self.bias[..., n : 2 * n]
        h_res = h[..., 2 * n :] * self.alpha_res + self.bias[..., 2 * n :]

        h_pre = torch.sigmoid(h_pre)
        h_post = 2.0 * torch.sigmoid(h_post)
        h_res = h_res.reshape(*shape, n, n)
        h_res = SinkhornKnoppFunction.apply(h_res, self.sinkhorn_iters)
        return h_pre, h_post, h_res

    def apply_fpre(self, x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        """Project the stream down to a layer input using a computed ``H[pre]``.

        Args:
            x: Input stream of shape ``(..., n, C)``.
            h_pre: ``H[pre]`` of shape ``(..., n)``.

        Returns:
            Layer input of shape ``(..., C)``.
        """
        return torch.einsum("...n,...nC->...C", h_pre, x)

    def apply_recombine(
        self,
        x: torch.Tensor,
        layer_out: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
    ) -> torch.Tensor:
        """Update the stream using computed ``H[post]`` and ``H[res]``.

        Args:
            x: Input stream of shape ``(..., n, C)``.
            layer_out: Layer function output of shape ``(..., C)``.
            h_post: ``H[post]`` of shape ``(..., n)``.
            h_res: ``H[res]`` of shape ``(..., n, n)``.

        Returns:
            Updated stream of shape ``(..., n, C)``.
        """
        res_part = torch.einsum("...nm,...mC->...nC", h_res, x)
        post_part = torch.einsum("...n,...C->...nC", h_post, layer_out)
        return res_part + post_part

    def fpre(self, x: torch.Tensor) -> torch.Tensor:
        """Project the stream down to a layer input.

        Args:
            x: Input stream of shape ``(..., n, C)``.

        Returns:
            Layer input of shape ``(..., C)``.
        """
        h_pre, _, _ = self.mappings(x)
        return self.apply_fpre(x, h_pre)

    def recombine(self, x: torch.Tensor, layer_out: torch.Tensor) -> torch.Tensor:
        """Update the stream with a layer function output.

        Args:
            x: Input stream of shape ``(..., n, C)``.
            layer_out: Layer function output of shape ``(..., C)``.

        Returns:
            Updated stream of shape ``(..., n, C)``.
        """
        _, h_post, h_res = self.mappings(x)
        return self.apply_recombine(x, layer_out, h_post, h_res)

    def forward(self, x: torch.Tensor, layer_out: torch.Tensor):
        """Project the stream to a layer input and update the stream.

        Args:
            x: Input stream of shape ``(..., n, C)``.
            layer_out: Layer function output of shape ``(..., C)``.

        Returns:
            Tuple ``(fpre, x_next)`` where ``fpre`` has shape ``(..., C)`` and
            ``x_next`` has shape ``(..., n, C)``.
        """
        return self.fpre(x), self.recombine(x, layer_out)
