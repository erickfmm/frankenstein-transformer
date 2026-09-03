"""Falcon: Fast Weight Attention for Continual Learning.

Implements the six fast-weight attention mixers of arXiv:2608.27763
(Zhang et al., 2026). The framework treats the recurrent fast-weight
state transition ``S_t`` (one ``d x d_v`` matrix per head) as an
*online continual learning rule* under read-after-write (RAW)
autoregressive semantics: the local fast-memory example revealed at
step ``t`` is the **prefix-aligned** pair
``(x_t, y_t) = (phi(k_{t-1}), v_t)`` -- one step shifted relative to
DeltaNet's same-step association. The internal fast-memory prediction
is ``y_hat_t = S_{t-1}^T x_t`` and the model readout is the
read-after-write ``o_t = S_t^T phi(q_t)``.

Two objectives give two families of normalized first-order updates:

* **Squared-error regression** (delta-rule family):
  ``l_t(S) = 1/2 ||v_t - S^T x_t||^2 + lambda_t/2 ||S||_F^2``
* **Negative inner product** (additive / linear-attention family):
  ``l_t(S) = -<S^T x_t, v_t> + lambda_t/2 ||S||_F^2``

and three step-size structures give the six variants:

===== ========== ============================== ===========================
Name  Family     Update (``gamma_t = 1-alpha_t``) Step size
===== ========== ============================== ===========================
F-1   regression ``S_t = gS + n x_t r_t^T``      ``n = b/(|x|^2+l+e)``
F-2   regression per-column ``n`` vector         per-column ``n_{j,t}``
F-3   regression sliding-window minibatch        ``n = b/(mu^(B)+l+e)``
F-1A  additive   ``S_t = gS + n x_t v_t^T``      ``n = b/(|x|^2+l+e)``
F-2A  additive   per-column ``n`` vector         per-column ``n_{j,t}``
F-3A  additive   sliding-window ``S_t = gS + n Nbar^(B)``  ``n = b/(Ebar^(B)+l+e)``
===== ========== ============================== ===========================

with the NLMS gain ``beta in (0, 2)``, the ridge ``lambda >= 0``, the
stabilizer ``eps > 0``, the positive-decay clamp
``alpha_t = min(eta_t lambda_t, 1 - eps_gamma)`` and the boundary
sentinels ``x_1 = 0``, ``eta_1 = 0``, ``(alpha_1, gamma_1) = (0, 1)``.

Every variant offers two computation forms:

* **Recurrent** -- a per-timestep scan (O(N) time, O(d^2) state),
  the ground-truth reference of the paper's Algorithm 5 / Algorithm 2.
* **Chunk-parallel** -- SSD-style kernels (chunk size
  ``falcon_chunk_size``, default 64): decay-mask linear attention
  (Falcon-1A/2A/3A, Fig. 7/8/10), the WY representation with
  single-residual triangular solves (Falcon-1/2, Alg. 6/1/8) and the
  rank-B block forward-substitution solve of the windowed regression
  recurrence (Falcon-3, Alg. 3). All chunk kernels use the numerically
  stable positive-decay renormalization: fp32 ``log1p(-alpha)``
  chunk-local log-prefixes ``u_i = cumsum(log gamma)`` with
  ``delta_i = exp(u_i)`` bounding every exponential by O(C) (Sec. 4.4
  of the paper), never forming global cumulative products.

Shared configuration (``model.attention.falcon.*``): the gate
parameterizations ``falcon_beta_mode`` (``static`` / ``ctx_beta`` /
``ctx_eta``, default ``ctx_eta``), ``falcon_lambda_mode``
(``static`` / ``ctx``, default ``ctx``), QK normalization
(``falcon_qk_norm``: ``rms_norm`` default or ``l2_norm``), the paper's
default causal short convolutions on the attention projections
(``falcon_short_conv``, kernel ``falcon_conv_kernel``), the sliding
window ``falcon_window`` (Falcon-3/3A only, paper default 4) and the
stabilizers ``falcon_eps`` / ``falcon_eps_gamma``.

Reference:
    Zhang et al. (2026). "Fast Weight Attention for Continual
    Learning". arXiv:2608.27763.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..common import BitLinear
from .falcon_common import (
    FalconGates,
    FalconShortConv,
    _stable_exp,
    falcon_qk_norm,
)


class _FalconBase(nn.Module):
    """Shared machinery of the six Falcon fast-weight attention mixers.

    Holds the q/k/v/out projections (BitLinear-aware), the causal short
    convolutions, the QK normalization, the gate module and the
    read-after-write shifted feature stream ``x_t = phi(k_{t-1})`` with
    the feature-space boundary condition ``x_1 = 0``. Subclasses
    implement :meth:`_recurrent_forward` and :meth:`_chunk_forward`.

    Attributes:
        variant: Registry stem (e.g. ``"falcon1"`` for
            ``falcon1_attn``); selects the per-mixer ``use_pe`` knob.
        per_column: Whether the plasticity gate emits one numerator per
            value channel (Falcon-2 / Falcon-2A).
        windowed: Whether the variant uses the sliding window
            (Falcon-3 / Falcon-3A).
    """

    variant: str = "falcon1"
    per_column: bool = False
    windowed: bool = False

    def __init__(self, config, pos_encoder=None):
        """Initialize projections, convolutions, gates and knobs.

        Args:
            config: Model configuration object with ``hidden_size``,
                ``num_heads``, ``dropout``, ``use_bitnet``, ``mode``
                and the ``falcon_*`` / ``<variant>_attn_use_pe`` knobs.
            pos_encoder: Optional shared positional encoding module.

        Raises:
            ValueError: If ``hidden_size`` is not divisible by
                ``num_heads``.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.total_dim = self.head_dim * self.num_heads

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads for {type(self).__name__}"
            )

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.q_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.total_dim, bias=False)
        self.out_proj = proj_cls(self.total_dim, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

        chunk_size = getattr(config, "falcon_chunk_size", 64)
        self.chunk_size = int(chunk_size) if chunk_size is not None else None
        if self.chunk_size is not None and self.chunk_size < 1:
            raise ValueError("falcon_chunk_size must be >= 1 or null")
        self.qk_norm = str(getattr(config, "falcon_qk_norm", "rms_norm")).lower()
        self.eps = float(getattr(config, "falcon_eps", 1e-6))
        self.eps_gamma = float(getattr(config, "falcon_eps_gamma", 1e-4))
        self.window = int(getattr(config, "falcon_window", 4))
        if self.window < 1:
            raise ValueError("falcon_window must be >= 1")

        if bool(getattr(config, "falcon_short_conv", True)):
            self.short_conv = FalconShortConv(
                self.total_dim, int(getattr(config, "falcon_conv_kernel", 4))
            )
        else:
            self.short_conv = None

        self.gates = FalconGates(config, per_column=self.per_column)

        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(getattr(config, f"{self.variant}_attn_use_pe", False))

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, logical_layer_idx: Optional[int] = None, pos_encoder=None
    ) -> torch.Tensor:
        """Run the Falcon fast-weight attention over the input sequence.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Optional logical layer index forwarded to
                the positional encoding.
            pos_encoder: Optional positional encoding module overriding
                ``self.pos_encoder``.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder
        bsz, seq_len, _ = x.shape
        n_heads, head_dim = self.num_heads, self.head_dim

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if self.short_conv is not None:
            q, k, v = self.short_conv(q, k, v)

        q = q.view(bsz, seq_len, n_heads, head_dim)
        k = k.view(bsz, seq_len, n_heads, head_dim)
        v = v.view(bsz, seq_len, n_heads, head_dim)

        q, k = falcon_qk_norm(q, k, self.qk_norm, self.eps)
        if self.use_pe and pe is not None:
            from ..common import apply_pe_to_qk

            q, k = apply_pe_to_qk(
                pe, self.pe_type, q, k, x, logical_layer_idx or 0, self.use_pe
            )
            q, k = falcon_qk_norm(q, k, self.qk_norm, self.eps)

        # Read-after-write shifted feature stream: x_t = phi(k_{t-1}),
        # x_1 = 0 (feature-space boundary condition, Sec. 3.1).
        x_feat = torch.zeros_like(k)
        x_feat[:, 1:] = k[:, :-1]

        num, lam_bar = self.gates(x)

        if self.chunk_size is not None:
            out = self._chunk_forward(q, x_feat, v, num, lam_bar)
        else:
            out = self._recurrent_forward(q, x_feat, v, num, lam_bar)

        out = out.reshape(bsz, seq_len, self.total_dim)
        out = self.dropout(out)
        return self.out_proj(out)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _step_sizes(
        self,
        num: torch.Tensor,
        lam_bar: torch.Tensor,
        statistic: torch.Tensor,
        first_step_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalized step sizes and decay factors for a chunk.

        Args:
            num: Step-size numerator ``(B, S, H)`` or ``(B, S, H, D)``.
            lam_bar: Per-head base ridge ``(B, S, H)``.
            statistic: Objective-matched curvature statistic matching
                ``num``'s trailing shape (energy, window spectral
                statistic or window mean energy).
            first_step_mask: Optional ``(S,)`` mask that is 0 at the
                global first step (enforces ``eta_1 = 0``).

        Returns:
            Tuple ``(eta, alpha, gamma, delta, log_gamma)`` where
            ``eta`` matches ``num``'s shape and ``delta`` /
            ``log_gamma`` are the fp32 chunk-local decay prefix and its
            logarithm.
        """
        lam = self.gates.resolve_lambda(lam_bar, statistic)
        denom = (statistic + lam + self.eps).clamp_min(self.eps)
        eta = num / denom
        if first_step_mask is not None:
            shape = [1] * eta.dim()
            shape[-2 if eta.dim() == 4 else -1] = first_step_mask.shape[0]
            eta = eta * first_step_mask.to(eta.dtype).view(shape)
        alpha = torch.clamp_max(eta * lam, 1.0 - self.eps_gamma)
        log_gamma = torch.log1p(-alpha.to(torch.float32))
        delta = _stable_exp(
            torch.cumsum(log_gamma, dim=-2 if num.dim() == 4 else -1)
        )
        return eta, alpha, gamma_of(alpha), delta, log_gamma

    def _read(self, state: torch.Tensor, q_t: torch.Tensor) -> torch.Tensor:
        """Read-after-write output ``o_t = S_t^T q_t``.

        Args:
            state: Fast-weight state ``(B, H, D, D)``.
            q_t: Query at the current step ``(B, H, D)``.

        Returns:
            Output ``(B, H, D)``.
        """
        return (state * q_t.unsqueeze(-1)).sum(-2)

    def _chunk_slices(self, seq_len: int) -> List[int]:
        """Chunk boundaries for the configured chunk size.

        Args:
            seq_len: Total sequence length.

        Returns:
            List of chunk start positions.
        """
        if self.chunk_size is None:
            return [0]
        return list(range(0, seq_len, self.chunk_size))

    def _slice_chunk(
        self,
        q: torch.Tensor,
        x_feat: torch.Tensor,
        v: torch.Tensor,
        num: torch.Tensor,
        lam_bar: torch.Tensor,
        a: int,
        end: int,
    ) -> Tuple[torch.Tensor, ...]:
        """Slice one chunk and transpose to head-major layout.

        Args:
            q: Queries ``(B, S, H, D)``.
            x_feat: Shifted features ``(B, S, H, D)``.
            v: Values ``(B, S, H, D)``.
            num: Numerator gates ``(B, S, H)`` or ``(B, S, H, D)``.
            lam_bar: Base ridge ``(B, S, H)``.
            a: Chunk start (inclusive).
            end: Chunk end (exclusive; may exceed ``S``).

        Returns:
            ``(q_c, x_c, v_c, num_c, lam_c)`` in ``(B, H, ...)`` layout.
        """
        return (
            q[:, a:end].transpose(1, 2),
            x_feat[:, a:end].transpose(1, 2),
            v[:, a:end].transpose(1, 2),
            num[:, a:end].transpose(1, 2),
            lam_bar[:, a:end].transpose(1, 2),
        )

    # ------------------------------------------------------------------
    # Variant-specific kernels (overridden by subclasses)
    # ------------------------------------------------------------------
    def _recurrent_forward(
        self, q: torch.Tensor, x_feat: torch.Tensor, v: torch.Tensor, num: torch.Tensor, lam_bar: torch.Tensor
    ) -> torch.Tensor:
        """Recurrent reference scan. Must be overridden."""
        raise NotImplementedError

    def _chunk_forward(
        self, q: torch.Tensor, x_feat: torch.Tensor, v: torch.Tensor, num: torch.Tensor, lam_bar: torch.Tensor
    ) -> torch.Tensor:
        """Chunk-parallel kernel. Must be overridden."""
        raise NotImplementedError


def gamma_of(alpha: torch.Tensor) -> torch.Tensor:
    """Carry factor ``gamma = 1 - alpha`` (alpha already clamped)."""
    return 1.0 - alpha


# ============================================================================
# Regression family (squared-error objective, delta-rule updates)
# ============================================================================


class Falcon1Attention(_FalconBase):
    """Falcon-1: scalar NLMS regression (delta-rule) fast-weight update.

    State update (read-after-write, ``r_t = v_t - S_{t-1}^T x_t``)::

        eta_t   = beta_t / (||x_t||^2 + lambda_t + eps)
        S_t     = (1 - eta_t*lambda_t) S_{t-1} + eta_t x_t r_t^T
        o_t     = S_t^T q_t

    With ``eps = 0``, ``lambda_t = 0`` and the unshifted assignment
    ``x_t = phi(k_t)`` this recovers DeltaNet's delta rule; the paper's
    contribution is the one-step-shifted write stream, the
    objective-matched NLMS normalization and the explicit ridge decay.

    Chunk form: WY representation ``L = tril(G, -1) + I`` with the
    single-residual triangular solve of Alg. 6/8, plus the
    positive-decay renormalization (Alg. 8).

    Reference:
        Zhang et al. (2026). arXiv:2608.27763, Fig. 3/4, Alg. 5-8.
    """

    variant = "falcon1"
    per_column = False
    windowed = False

    def _recurrent_forward(self, q, x_feat, v, num, lam_bar):
        bsz, seq_len, n_heads, head_dim = q.shape
        state = q.new_zeros(bsz, n_heads, head_dim, head_dim)
        outputs = []
        for t in range(seq_len):
            x_t = x_feat[:, t]
            v_t = v[:, t]
            energy = (x_t * x_t).sum(-1)
            lam = self.gates.resolve_lambda(lam_bar[:, t], energy)
            denom = (energy + lam + self.eps).clamp_min(self.eps)
            eta = num[:, t] / denom
            if t == 0:
                eta = torch.zeros_like(eta)
            alpha = torch.clamp_max(eta * lam, 1.0 - self.eps_gamma)
            pred = (state * x_t.unsqueeze(-1)).sum(-2)
            resid = v_t - pred
            state = (
                state * gamma_of(alpha)[..., None, None]
                + x_t.unsqueeze(-1) * (eta.unsqueeze(-1) * resid).unsqueeze(-2)
            )
            outputs.append(self._read(state, q[:, t]))
        return torch.stack(outputs, dim=1)

    def _chunk_forward(self, q, x_feat, v, num, lam_bar):
        bsz, seq_len, n_heads, head_dim = q.shape
        dtype = q.dtype
        state = q.new_zeros(bsz, n_heads, head_dim, head_dim)
        outputs = []
        for a in self._chunk_slices(seq_len):
            end = a + self.chunk_size
            qc, xc, vc, num_c, lam_c = self._slice_chunk(q, x_feat, v, num, lam_bar, a, end)
            length = qc.shape[2]
            energy = xc.float().pow(2).sum(-1)
            first = torch.ones(length, device=q.device, dtype=torch.float32)
            if a == 0:
                first[0] = 0.0
            eta, _alpha, _gamma, delta, log_gamma = self._step_sizes(num_c, lam_c, energy, first)
            # Positive-decay renormalization (Alg. 8): rescale drivers to
            # the no-ridge WY kernel, run it, then restore the scale.
            # Feature drivers only carry sqrt(eta_tilde); the decay
            # rescaling applies to the value targets (Alg. 8). The tiny
            # floor keeps the sqrt/solve backward well-defined at the
            # boundary step (eta_1 = 0).
            eta_tilde = eta * _stable_exp(-log_gamma)
            u_prefix = torch.cumsum(log_gamma, dim=-1)
            prev_u = u_prefix - log_gamma
            sqrt_eta_tilde = torch.sqrt(torch.clamp_min(eta_tilde, 1e-12))
            u = xc * sqrt_eta_tilde.to(dtype)[..., None]
            v_tilde = vc * (sqrt_eta_tilde * _stable_exp(-prev_u)).to(dtype)[..., None]

            gram = torch.einsum("bhqd,bhkd->bhqk", u, u)
            eye = torch.eye(length, device=q.device, dtype=dtype)
            tri = torch.ones(length, length, device=q.device, dtype=dtype).tril(0)
            lower = gram.tril(-1) + eye

            proj_state = torch.einsum("bhqd,bhde->bhqe", u, state)
            rhs = v_tilde - proj_state
            b_coef = torch.linalg.solve_triangular(
                lower.float(), rhs.float(), upper=False
            ).to(dtype)

            hist = torch.einsum("bhqd,bhde->bhqe", qc, state)
            mask = torch.einsum("bhqd,bhkd->bhqk", qc, u).tril(0)
            out_c = delta.to(dtype)[..., None] * (
                hist + torch.einsum("bhqk,bhke->bhqe", mask, b_coef)
            )
            outputs.append(out_c)

            state = delta.to(dtype)[:, :, -1][..., None, None] * (
                state + torch.einsum("bhqd,bhqe->bhde", u, b_coef)
            )
        return torch.cat(outputs, dim=2).transpose(1, 2)


class Falcon2Attention(_FalconBase):
    """Falcon-2: per-column (per-value-channel) NLMS regression update.

    The ridge loss decomposes across value coordinates, so each value
    channel ``j`` gets its own step size ``eta_{j,t}`` sharing the NLMS
    normalizer ``||x_t||^2 + lambda_t + eps`` across columns::

        eta_{j,t} = beta_{j,t} / (||x_t||^2 + lambda_t + eps)
        S_t       = S_{t-1} (I - lambda_t Diag(eta_t)) + x_t (eta_t r_t)^T

    Chunk form: shared key Gram with per-channel unit-lower-triangular
    systems ``L_j = I + tril(G * (Sigma_j Sigma_j^T), -1)`` solved as a
    batch (Alg. 1), where ``Sigma = sqrt(eta_tilde)``.

    Reference:
        Zhang et al. (2026). arXiv:2608.27763, Fig. 3/5, Alg. 1.
    """

    variant = "falcon2"
    per_column = True
    windowed = False

    def _recurrent_forward(self, q, x_feat, v, num, lam_bar):
        bsz, seq_len, n_heads, head_dim = q.shape
        state = q.new_zeros(bsz, n_heads, head_dim, head_dim)
        outputs = []
        for t in range(seq_len):
            x_t = x_feat[:, t]
            v_t = v[:, t]
            energy = (x_t * x_t).sum(-1)
            lam = self.gates.resolve_lambda(lam_bar[:, t], energy)
            denom = (energy + lam + self.eps).clamp_min(self.eps)
            eta = num[:, t] / denom[..., None]
            if t == 0:
                eta = torch.zeros_like(eta)
            alpha = torch.clamp_max(eta * lam[..., None], 1.0 - self.eps_gamma)
            pred = (state * x_t.unsqueeze(-1)).sum(-2)
            resid = v_t - pred
            state = (
                state * gamma_of(alpha)[:, :, None, :]
                + x_t.unsqueeze(-1) * (eta * resid).unsqueeze(-2)
            )
            outputs.append(self._read(state, q[:, t]))
        return torch.stack(outputs, dim=1)

    def _chunk_forward(self, q, x_feat, v, num, lam_bar):
        bsz, seq_len, n_heads, head_dim = q.shape
        dtype = q.dtype
        state = q.new_zeros(bsz, n_heads, head_dim, head_dim)
        outputs = []
        for a in self._chunk_slices(seq_len):
            end = a + self.chunk_size
            qc, xc, vc, num_c, lam_c = self._slice_chunk(q, x_feat, v, num, lam_bar, a, end)
            length = qc.shape[2]
            energy = xc.float().pow(2).sum(-1)
            first = torch.ones(length, device=q.device, dtype=torch.float32)
            if a == 0:
                first[0] = 0.0
            eta, _alpha, _gamma, delta, log_gamma = self._step_sizes(
                num_c, lam_c[..., None], energy[..., None], first
            )
            # Per-channel positive-decay renormalization. The sqrt floor
            # keeps the batched solve backward well-defined at the
            # boundary step (eta_1 = 0).
            eta_tilde = eta * _stable_exp(-log_gamma)
            sigma = torch.sqrt(torch.clamp_min(eta_tilde, 1e-12))
            u_prefix = torch.cumsum(log_gamma, dim=-2)
            prev_u = u_prefix - log_gamma
            v_resc = vc * _stable_exp(-prev_u).to(dtype)

            gram = torch.einsum("bhqd,bhkd->bhqk", xc, xc)
            eye = torch.eye(length, device=q.device, dtype=dtype)
            tri = torch.ones(length, length, device=q.device, dtype=dtype).tril(0)

            x_state = torch.einsum("bhqd,bhde->bhqe", xc, state)
            rhs = sigma.to(dtype) * (v_resc - x_state)

            sigma_outer = sigma[:, :, :, None, :] * sigma[:, :, None, :, :]
            lower_c = gram.tril(-1)[..., None] * sigma_outer.to(dtype) + eye[..., None]
            lower_b = lower_c.permute(0, 1, 4, 2, 3)
            rhs_b = rhs.permute(0, 1, 3, 2)[..., None]
            b_tilde = torch.linalg.solve_triangular(
                lower_b.float(), rhs_b.float(), upper=False
            )[..., 0].permute(0, 1, 3, 2)
            b_coef = sigma.to(dtype) * b_tilde

            hist = torch.einsum("bhqd,bhde->bhqe", qc, state)
            mask = torch.einsum("bhqd,bhkd->bhqk", qc, xc).tril(0)
            out_c = delta.to(dtype) * (
                hist + torch.einsum("bhqk,bhke->bhqe", mask, b_coef)
            )
            outputs.append(out_c)

            state = (state + torch.einsum("bhqd,bhqe->bhde", xc, b_coef)) * delta[
                :, :, -1, :
            ].to(dtype).unsqueeze(2)
        return torch.cat(outputs, dim=2).transpose(1, 2)


class Falcon3Attention(_FalconBase):
    """Falcon-3: sliding-window minibatch regression update.

    Each step regresses the fast memory on the last ``B_t <= B``
    prefix-aligned pairs with **all window residuals evaluated at the
    pre-update state** ``S_{t-1}`` (bounded rehearsal)::

        I_t    = {max(2, t-B+1), ..., t}
        mu_t   = lambda_max(X_t^T X_t) / B_t
        eta_t  = beta_t / (mu_t + lambda_t + eps)
        S_t    = (1 - eta_t*lambda_t) S_{t-1}
                  + (eta_t/B_t) * sum_{j in I_t} x_j (v_j - S_{t-1}^T x_j)^T

    The window-averaged injection is invariant to ``B`` (each pair is
    replayed in exactly ``B`` consecutive updates at weight ``1/B``).
    ``S_t`` alone is not Markov: exact continuation requires the FIFO
    tail of the last ``B-1`` pairs (Alg. 2 returns ``(O, S_T, W_T)``).

    Chunk form: rank-B affine recurrence solved by block forward
    substitution over the ``(time x rank)`` extended index space --
    same-time rank components uncoupled, exactly the ParallelFlow
    ``tensorInv`` structure of Alg. 3 with the positive-decay
    renormalization folded into the drivers.

    Reference:
        Zhang et al. (2026). arXiv:2608.27763, Fig. 3/9, Alg. 2-3.
    """

    variant = "falcon3"
    per_column = False
    windowed = True

    def _window_tensors(self, x_feat, v, win_x, win_v, t):
        """Push the current pair into the FIFO window and stack it.

        Args:
            x_feat: Shifted feature stream ``(B, S, H, D)``.
            v: Value stream ``(B, S, H, D)``.
            win_x/win_v: Running FIFO lists (mutated in place).
            t: Current 0-indexed step (pairs are pushed for ``t >= 1``).

        Returns:
            Tuple ``(Xw, Vw, B_t)`` with stacked windows
            ``(B, H, B_t, D)`` and the realized window size.
        """
        if t >= 1:
            win_x.append(x_feat[:, t])
            win_v.append(v[:, t])
            if len(win_x) > self.window:
                win_x.pop(0)
                win_v.pop(0)
        size = len(win_x)
        if size == 0:
            return None, None, 0
        return torch.stack(win_x, dim=2), torch.stack(win_v, dim=2), size

    def _recurrent_forward(self, q, x_feat, v, num, lam_bar):
        bsz, seq_len, n_heads, head_dim = q.shape
        state = q.new_zeros(bsz, n_heads, head_dim, head_dim)
        win_x: List[torch.Tensor] = []
        win_v: List[torch.Tensor] = []
        outputs = []
        for t in range(seq_len):
            xw, vw, size = self._window_tensors(x_feat, v, win_x, win_v, t)
            if size == 0:
                outputs.append(q.new_zeros(bsz, n_heads, head_dim))
                continue
            gram = torch.einsum("bhjd,bhkd->bhjk", xw, xw)
            mu = (
                torch.linalg.eigvalsh(gram.float())[..., -1].clamp_min(0.0) / size
            ).to(q.dtype)
            lam = self.gates.resolve_lambda(lam_bar[:, t], mu)
            denom = (mu + lam + self.eps).clamp_min(self.eps)
            eta = num[:, t] / denom
            if t == 0:
                eta = torch.zeros_like(eta)
            alpha = torch.clamp_max(eta * lam, 1.0 - self.eps_gamma)
            pred = torch.einsum("bhjd,bhde->bhje", xw, state)
            resid = vw - pred
            update = torch.einsum("bhjd,bhje->bhde", xw, resid)
            state = (
                state * gamma_of(alpha)[..., None, None]
                + (eta / size)[..., None, None] * update
            )
            outputs.append(self._read(state, q[:, t]))
        return torch.stack(outputs, dim=1)

    def _chunk_forward(self, q, x_feat, v, num, lam_bar):
        bsz, seq_len, n_heads, head_dim = q.shape
        dtype = q.dtype
        window = self.window
        state = q.new_zeros(bsz, n_heads, head_dim, head_dim)
        outputs = []
        for a in self._chunk_slices(seq_len):
            end = a + self.chunk_size
            length = min(self.chunk_size, seq_len - a)
            qc, xc, vc, num_c, lam_c = self._slice_chunk(q, x_feat, v, num, lam_bar, a, end)

            # Extended slice: the B-1 FIFO tail from previous chunks plus
            # the current chunk (|J| <= C + B - 1, Sec. C.7).
            ext_start = max(0, a - (window - 1))
            xe = x_feat[:, ext_start:end].transpose(1, 2)  # [B, H, E, D]
            ve = v[:, ext_start:end].transpose(1, 2)
            ext_len = xe.shape[2]

            # Zero-padded windows Xw[i, r], Vw[i, r]: members
            # j in [max(1, i-B+1), i] (0-indexed), oldest first.
            i_abs = torch.arange(a, a + length, device=q.device)  # [L]
            j_start = torch.clamp_min(i_abs - window + 1, 1)
            size = (i_abs - j_start + 1).clamp_min(0)  # realized B_t
            size_pos = size.clamp_min(1)
            slots = torch.arange(window, device=q.device)  # [W]
            j_idx = j_start[:, None] + slots[None, :]  # [L, W]
            member_mask = slots[None, :] < size[:, None]  # [L, W]
            gather_idx = (j_idx - ext_start).clamp(0, ext_len - 1).long()
            pad = member_mask[None, None, :, :, None].to(dtype)
            xw = xe[:, :, gather_idx] * pad  # [B, H, L, W, D]
            vw = ve[:, :, gather_idx] * pad

            # Per-step spectral statistic mu_t = lambda_max(X^T X)/B_t
            # (zero padding is inert for the top eigenvalue).
            blocks = torch.einsum("bhxrd,bhxsd->bhxrs", xw, xw)  # [B,H,L,W,W]
            mu = (
                torch.linalg.eigvalsh(blocks.float())[..., -1].clamp_min(0.0)
                / size_pos.to(torch.float32)
            ).to(dtype)  # [B, H, L]

            first = torch.ones(length, device=q.device, dtype=torch.float32)
            if a == 0:
                first[0] = 0.0
            eta, _alpha, _gamma, delta, log_gamma = self._step_sizes(
                num_c, lam_c, mu, first
            )

            # Positive-decay renormalization: drivers in the no-decay
            # rank-B domain (eta_tilde = eta/gamma, values /c_{i-1}).
            eta_tilde = eta * _stable_exp(-log_gamma)
            w_step = eta_tilde / size_pos.to(torch.float32)[None, None, :]
            u_prefix = torch.cumsum(log_gamma, dim=-1)
            prev_u = u_prefix - log_gamma
            v_resc = vw * _stable_exp(-prev_u).to(dtype)[..., None, None]

            # Block forward substitution: Z_i + sum_{m<i} WW[i,m] Z_m =
            # w_i (Vr_i - Xw_i^T S_in); same-time ranks uncoupled.
            gram_w = torch.einsum("bhird,bhjsd->bhijrs", xw, xw).to(dtype)
            proj = torch.einsum("bhlrd,bhde->bhlre", xw, state)  # [B,H,L,W,D]
            z_list: List[torch.Tensor] = []
            for i in range(length):
                rhs = w_step[:, :, i].to(dtype)[..., None, None] * (
                    v_resc[:, :, i] - proj[:, :, i]
                )
                if i > 0:
                    acc = torch.einsum(
                        "bhkrs,bhksc->bhrc",
                        gram_w[:, :, i, :i],
                        torch.stack(z_list[:i], dim=2),
                    )
                    rhs = rhs - w_step[:, :, i].to(dtype)[..., None, None] * acc
                z_list.append(rhs)
            z = torch.stack(z_list, dim=2)  # [B, H, L, W, D]

            # Outputs: read-after-write (inclusive m <= i), rescaled.
            cross = torch.einsum("bhid,bhmrd->bhimr", qc, xw)  # [B,H,L,L,W]
            tri = torch.ones(length, length, device=q.device, dtype=dtype).tril(0)
            intra = torch.einsum(
                "bhimr,bhmrc->bhic", cross * tri[None, None, ..., None], z
            )
            hist = torch.einsum("bhqd,bhde->bhqe", qc, state)
            outputs.append(delta.to(dtype)[..., None] * (hist + intra))

            state = delta.to(dtype)[:, :, -1][..., None, None] * (
                state + torch.einsum("bhmrd,bhmrc->bhdc", xw, z)
            )
        return torch.cat(outputs, dim=2).transpose(1, 2)


# ============================================================================
# Inner-product family (additive / linear-attention updates)
# ============================================================================


class Falcon1AAttention(_FalconBase):
    """Falcon-1A: scalar inner-product (additive) fast-weight update.

    One gradient step on ``l_t(S) = -<S^T x_t, v_t> + lambda/2 ||S||^2``
    gives a purely additive (Hebbian) write with energy-normalized gain::

        eta_t = beta_t / (||x_t||^2 + lambda_t + eps)
        S_t   = gamma_t S_{t-1} + eta_t x_t v_t^T
        o_t   = S_t^T q_t

    With ``lambda_t = 0`` and the unshifted assignment this is standard
    linear attention / Mamba-2 accumulation; the shifted write stream
    makes it the one-step-shifted next-latent variant (Eq. 2.3).

    Chunk form: the decay-mask linear-attention kernel
    ``M_{t,i} = eta_i * prod_{r=i+1}^{t} gamma_r`` (Fig. 7B).

    Reference:
        Zhang et al. (2026). arXiv:2608.27763, Fig. 6/7, Eq. 4.8.
    """

    variant = "falcon1a"
    per_column = False
    windowed = False

    def _recurrent_forward(self, q, x_feat, v, num, lam_bar):
        bsz, seq_len, n_heads, head_dim = q.shape
        state = q.new_zeros(bsz, n_heads, head_dim, head_dim)
        outputs = []
        for t in range(seq_len):
            x_t = x_feat[:, t]
            v_t = v[:, t]
            energy = (x_t * x_t).sum(-1)
            lam = self.gates.resolve_lambda(lam_bar[:, t], energy)
            denom = (energy + lam + self.eps).clamp_min(self.eps)
            eta = num[:, t] / denom
            if t == 0:
                eta = torch.zeros_like(eta)
            alpha = torch.clamp_max(eta * lam, 1.0 - self.eps_gamma)
            state = (
                state * gamma_of(alpha)[..., None, None]
                + x_t.unsqueeze(-1) * (eta.unsqueeze(-1) * v_t).unsqueeze(-2)
            )
            outputs.append(self._read(state, q[:, t]))
        return torch.stack(outputs, dim=1)

    def _chunk_forward(self, q, x_feat, v, num, lam_bar):
        bsz, seq_len, n_heads, head_dim = q.shape
        dtype = q.dtype
        state = q.new_zeros(bsz, n_heads, head_dim, head_dim)
        outputs = []
        for a in self._chunk_slices(seq_len):
            end = a + self.chunk_size
            qc, xc, vc, num_c, lam_c = self._slice_chunk(q, x_feat, v, num, lam_bar, a, end)
            length = qc.shape[2]
            energy = xc.float().pow(2).sum(-1)
            first = torch.ones(length, device=q.device, dtype=torch.float32)
            if a == 0:
                first[0] = 0.0
            eta, _alpha, _gamma, delta, log_gamma = self._step_sizes(
                num_c, lam_c, energy, first
            )
            tri = torch.ones(length, length, device=q.device, dtype=torch.float32).tril(0)
            u_prefix = torch.cumsum(log_gamma, dim=-1)
            decay = _stable_exp(u_prefix[:, :, :, None] - u_prefix[:, :, None, :]) * tri
            mask = eta[:, :, None, :] * decay

            scores = torch.einsum("bhqd,bhkd->bhqk", qc, xc)
            hist = torch.einsum("bhqd,bhde->bhqe", qc, state)
            out_c = delta.to(dtype)[..., None] * hist + torch.einsum(
                "bhqk,bhqk,bhke->bhqe", scores, mask.to(dtype), vc
            )
            outputs.append(out_c)

            w_state = eta * _stable_exp(u_prefix[:, :, -1:] - u_prefix)
            state = delta.to(dtype)[:, :, -1][..., None, None] * state + torch.einsum(
                "bhkd,bhk,bhke->bhde", xc, w_state.to(dtype), vc
            )
        return torch.cat(outputs, dim=2).transpose(1, 2)


class Falcon2AAttention(_FalconBase):
    """Falcon-2A: per-column inner-product fast-weight update.

    The additive analogue of Falcon-2 -- each value channel ``j``
    carries its own step size and decay::

        eta_{j,t} = beta_{j,t} / (||x_t||^2 + lambda_t + eps)
        S_t       = S_{t-1} Diag(gamma_t) + x_t (eta_t v_t)^T

    Chunk form: per-channel decay masks
    ``M_{t,i}^{(c)} = eta_{c,i} * prod_{r=i+1}^{t} gamma_{c,r}``
    (Fig. 8B).

    Reference:
        Zhang et al. (2026). arXiv:2608.27763, Fig. 6/8, Eq. 4.9.
    """

    variant = "falcon2a"
    per_column = True
    windowed = False

    def _recurrent_forward(self, q, x_feat, v, num, lam_bar):
        bsz, seq_len, n_heads, head_dim = q.shape
        state = q.new_zeros(bsz, n_heads, head_dim, head_dim)
        outputs = []
        for t in range(seq_len):
            x_t = x_feat[:, t]
            v_t = v[:, t]
            energy = (x_t * x_t).sum(-1)
            lam = self.gates.resolve_lambda(lam_bar[:, t], energy)
            denom = (energy + lam + self.eps).clamp_min(self.eps)
            eta = num[:, t] / denom[..., None]
            if t == 0:
                eta = torch.zeros_like(eta)
            alpha = torch.clamp_max(eta * lam[..., None], 1.0 - self.eps_gamma)
            state = (
                state * gamma_of(alpha)[:, :, None, :]
                + x_t.unsqueeze(-1) * (eta * v_t).unsqueeze(-2)
            )
            outputs.append(self._read(state, q[:, t]))
        return torch.stack(outputs, dim=1)

    def _chunk_forward(self, q, x_feat, v, num, lam_bar):
        bsz, seq_len, n_heads, head_dim = q.shape
        dtype = q.dtype
        state = q.new_zeros(bsz, n_heads, head_dim, head_dim)
        outputs = []
        for a in self._chunk_slices(seq_len):
            end = a + self.chunk_size
            qc, xc, vc, num_c, lam_c = self._slice_chunk(q, x_feat, v, num, lam_bar, a, end)
            length = qc.shape[2]
            energy = xc.float().pow(2).sum(-1)
            first = torch.ones(length, device=q.device, dtype=torch.float32)
            if a == 0:
                first[0] = 0.0
            eta, _alpha, _gamma, delta, log_gamma = self._step_sizes(
                num_c, lam_c[..., None], energy[..., None], first
            )
            tri = torch.ones(length, length, device=q.device, dtype=torch.float32).tril(0)
            u_prefix = torch.cumsum(log_gamma, dim=-2)
            decay = (
                _stable_exp(u_prefix[:, :, :, None, :] - u_prefix[:, :, None, :, :])
                * tri[:, :, None]
            )  # [B, H, L, L, D]
            mask = eta[:, :, None, :, :] * decay

            scores = torch.einsum("bhqd,bhkd->bhqk", qc, xc)
            hist = torch.einsum("bhqd,bhde->bhqe", qc, state)
            intra = torch.einsum(
                "bhqk,bhqkc,bhkc->bhqc", scores, mask.to(dtype), vc
            )
            out_c = delta.to(dtype) * hist + intra
            outputs.append(out_c)

            w_state = eta * _stable_exp(u_prefix[:, :, -1:, :] - u_prefix)
            state = (
                state * delta[:, :, -1, :].to(dtype).unsqueeze(2)
                + torch.einsum("bhkd,bhkc,bhkc->bhdc", xc, w_state.to(dtype), vc)
            )
        return torch.cat(outputs, dim=2).transpose(1, 2)


class Falcon3AAttention(_FalconBase):
    """Falcon-3A: sliding-window inner-product fast-weight update.

    The additive windowed variant -- best length extrapolation of the
    paper (Falcon-3A.3, Table 3)::

        Nbar_t^{(B)} = (1/B_t) sum_{j in I_t} x_j v_j^T
        Ebar_t^{(B)} = (1/B_t) sum_{j in I_t} ||x_j||^2
        eta_t        = beta_t / (Ebar_t^{(B)} + lambda_t + eps)
        S_t          = gamma_t S_{t-1} + eta_t Nbar_t^{(B)}

    Chunk form: the window-banded decay-mask kernel
    ``M = D * Diag(eta) * A`` over the extended slice ``J`` with
    ``|J| <= C + B - 1`` (Fig. 10B, Alg. 4) where ``A`` is the B-banded
    window operator and ``D`` the causal decay kernel.

    Reference:
        Zhang et al. (2026). arXiv:2608.27763, Fig. 6/10, Alg. 4,
        Eq. 4.17-4.18.
    """

    variant = "falcon3a"
    per_column = False
    windowed = True

    def _recurrent_forward(self, q, x_feat, v, num, lam_bar):
        bsz, seq_len, n_heads, head_dim = q.shape
        state = q.new_zeros(bsz, n_heads, head_dim, head_dim)
        win_x: List[torch.Tensor] = []
        win_v: List[torch.Tensor] = []
        outputs = []
        for t in range(seq_len):
            if t >= 1:
                win_x.append(x_feat[:, t])
                win_v.append(v[:, t])
                if len(win_x) > self.window:
                    win_x.pop(0)
                    win_v.pop(0)
            size = len(win_x)
            if size == 0:
                outputs.append(q.new_zeros(bsz, n_heads, head_dim))
                continue
            xw = torch.stack(win_x, dim=2)
            vw = torch.stack(win_v, dim=2)
            n_bar = torch.einsum("bhjd,bhje->bhde", xw, vw) / size
            e_bar = (xw * xw).sum(-1).sum(-1) / size
            lam = self.gates.resolve_lambda(lam_bar[:, t], e_bar)
            denom = (e_bar + lam + self.eps).clamp_min(self.eps)
            eta = num[:, t] / denom
            if t == 0:
                eta = torch.zeros_like(eta)
            alpha = torch.clamp_max(eta * lam, 1.0 - self.eps_gamma)
            state = (
                state * gamma_of(alpha)[..., None, None]
                + eta[..., None, None] * n_bar
            )
            outputs.append(self._read(state, q[:, t]))
        return torch.stack(outputs, dim=1)

    def _chunk_forward(self, q, x_feat, v, num, lam_bar):
        bsz, seq_len, n_heads, head_dim = q.shape
        dtype = q.dtype
        window = self.window
        state = q.new_zeros(bsz, n_heads, head_dim, head_dim)
        outputs = []
        for a in self._chunk_slices(seq_len):
            end = a + self.chunk_size
            length = min(self.chunk_size, seq_len - a)
            qc, xc, vc, num_c, lam_c = self._slice_chunk(q, x_feat, v, num, lam_bar, a, end)

            ext_start = max(0, a - (window - 1))
            xe = x_feat[:, ext_start:end].transpose(1, 2)  # [B, H, E, D]
            ve = v[:, ext_start:end].transpose(1, 2)
            ext_len = xe.shape[2]

            arange_l = torch.arange(a, a + length, device=q.device)
            arange_e = torch.arange(ext_start, ext_start + ext_len, device=q.device)
            valid = (
                (arange_e[None, :] >= 1)
                & (arange_e[None, :] >= arange_l[:, None] - window + 1)
                & (arange_e[None, :] <= arange_l[:, None])
            ).to(torch.float32)  # [L, E]
            size = valid.sum(-1)  # [L]
            size_pos = size.clamp_min(1.0)

            energies = xe.float().pow(2).sum(-1)  # [B, H, E]
            e_bar = (energies.unsqueeze(2) * valid[None, None]).sum(-1) / size_pos.to(torch.float32)[None, None]

            first = torch.ones(length, device=q.device, dtype=torch.float32)
            if a == 0:
                first[0] = 0.0
            eta, _alpha, _gamma, delta, log_gamma = self._step_sizes(
                num_c, lam_c, e_bar, first
            )

            # M = D @ (Diag(eta) @ A) with A = valid / B^+ (window
            # operator) and D the inclusive causal decay kernel.
            a_op = valid / size_pos[:, None]  # [L, E]
            tri = torch.ones(length, length, device=q.device, dtype=torch.float32).tril(0)
            u_prefix = torch.cumsum(log_gamma, dim=-1)
            decay = _stable_exp(u_prefix[:, :, :, None] - u_prefix[:, :, None, :]) * tri
            mask = torch.einsum(
                "bhts,bhs,se->bhte", decay, eta, a_op
            )  # [B, H, L, E]

            scores = torch.einsum("bhqd,bhed->bhqe", qc, xe)
            hist = torch.einsum("bhqd,bhde->bhqe", qc, state)
            out_c = delta.to(dtype)[..., None] * hist + torch.einsum(
                "bhqe,bhqe,bhec->bhqc", scores, mask.to(dtype), ve
            )
            outputs.append(out_c)

            m_last = mask[:, :, -1, :]
            state = delta.to(dtype)[:, :, -1][..., None, None] * state + torch.einsum(
                "bhed,bhec,bhe->bhdc", xe, ve, m_last.to(dtype)
            )
        return torch.cat(outputs, dim=2).transpose(1, 2)
