"""Bayesian Attention Mechanism (BAM).

Implements a Generalized-Gaussian relative-position bias added directly to
attention scores. BAM reframes positional encoding as a probabilistic prior
over positions and unifies existing methods: with ``beta = 1, mu = 0,
alpha = 1/m`` the bias recovers ALiBi (Laplace prior); ``beta = 2`` gives a
Normal prior; ``0 < beta < 1`` yields heavier tails than Laplace; ``beta < 0``
is a relaxed regime that suppresses local context and acts as a long-range
"retrieval head". The per-head shape (``theta_beta``) and scale (``theta_alpha``)
parameters are learned during training. Initialising both to zero yields the
Uniform prior (zero bias at init), which the paper's ablation H.1 found best.

The ``forward`` method is an identity pass-through so the module can be
dropped into ``pos_encoder(q)`` / ``pos_encoder(k)`` call sites; the additive
bias is exposed through :meth:`bias` for use by attention layers via
``apply_pe_to_scores``.

Reference:
    Bianchessi et al. (2025), "Bayesian Attention Mechanism: A Probabilistic
    Framework for Positional Encoding and Context Length Extrapolation",
    arXiv:2505.22842.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class BAM(nn.Module):
    """Bayesian Attention Mechanism with a Generalized-Gaussian position prior.

    Stores per-head learnable scalars ``theta_alpha``, ``theta_beta`` (and
    optionally ``theta_mu``) reparametrised so that ``alpha = exp(-theta_beta *
    theta_alpha) >= 0``. The bias matrix is::

        B[h, i, j] = -(|j - alpha[h] * i - mu[h]| + eps) ^ beta[h]

    added to the attention logits alongside the causal mask, exactly like
    ALiBi. When ``theta_init = 0`` (Uniform prior) the bias is zero at init and
    the per-head shape/scale are learned from scratch.

    Reference:
        Bianchessi et al. (2025), "Bayesian Attention Mechanism: A
        Probabilistic Framework for Positional Encoding and Context Length
        Extrapolation", arXiv:2505.22842.

    Args:
        num_heads: Number of attention heads.
        learn_mu: If True, ``theta_mu`` is a learnable parameter; otherwise it
            is a fixed buffer at ``mu_init``. Default: False.
        theta_init: Initial value for ``theta_alpha`` and ``theta_beta``.
            ``0.0`` = Uniform prior (paper's best); ``1.0`` = Laplace/ALiBi.
            Default: 0.0.
        mu_init: Initial value for ``theta_mu``. Default: 0.0.
        eps: Numerical stability floor added to ``|.|`` before the power when
            ``beta < 0``. Default: 1e-5.

    Attributes:
        num_heads: Number of attention heads.
        learn_mu: Whether ``theta_mu`` is learnable.
        eps: Numerical stability floor.
        theta_alpha: Per-head scale parameter of shape ``(num_heads,)``.
        theta_beta: Per-head shape parameter of shape ``(num_heads,)``.
        theta_mu: Per-head location parameter (learnable or fixed buffer).
    """

    def __init__(
        self,
        num_heads: int,
        learn_mu: bool = False,
        theta_init: float = 0.0,
        mu_init: float = 0.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.learn_mu = bool(learn_mu)
        self.eps = float(eps)

        self.theta_alpha = nn.Parameter(torch.full((num_heads,), float(theta_init)))
        self.theta_beta = nn.Parameter(torch.full((num_heads,), float(theta_init)))
        if self.learn_mu:
            self.theta_mu = nn.Parameter(torch.full((num_heads,), float(mu_init)))
        else:
            self.register_buffer("theta_mu", torch.full((num_heads,), float(mu_init)), persistent=False)

    def bias(
        self,
        seq_len: int,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        """Compute the BAM additive attention bias.

        Args:
            seq_len: Sequence length for which to compute the bias.
            device: Device on which to materialise the bias tensor. Defaults
                to the ``theta_alpha`` parameter device.
            dtype: Data type of the returned bias. Defaults to the
                ``theta_alpha`` parameter dtype.

        Returns:
            Tensor of shape ``(1, num_heads, seq_len, seq_len)`` containing
            the additive bias ``- (|j - alpha*i - mu| + eps) ^ beta``.
        """
        if device is None:
            device = self.theta_alpha.device
        if dtype is None:
            dtype = self.theta_alpha.dtype

        pos = torch.arange(seq_len, device=device, dtype=dtype)
        i = pos[:, None]  # [S, 1] query index
        j = pos[None, :]  # [1, S] key index

        theta_alpha = self.theta_alpha.to(device=device, dtype=dtype)  # [H]
        theta_beta = self.theta_beta.to(device=device, dtype=dtype)  # [H]
        theta_mu = self.theta_mu.to(device=device, dtype=dtype)  # [H]

        alpha = torch.exp(-theta_beta * theta_alpha)  # [H], >= 0
        beta = theta_beta  # [H]
        mu = theta_mu  # [H]

        # relative position contribution per head: j - alpha[h] * i - mu[h]
        # shape [H, S, S] via broadcasting
        contrib = j[None, :, :] - alpha[:, None, None] * i[None, :, :] - mu[:, None, None]
        # B[h, i, j] = -(|contrib| + eps) ^ beta[h]
        B = -((contrib.abs() + self.eps) ** beta[:, None, None])
        return B.unsqueeze(0)  # [1, H, S, S]

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