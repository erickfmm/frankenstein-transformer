"""Shared utilities for the Falcon fast-weight attention family.

Implements the shared machinery of the six Falcon attention mixers
(arXiv:2608.27763, "Fast Weight Attention for Continual Learning",
Zhang et al., 2026):

* :class:`FalconGates` -- context-conditioned (or static) plasticity
  (``beta`` / ``eta`` numerator) and forgetting (``lambda`` ridge)
  gates shared by all variants.
* :func:`falcon_qk_norm` -- weightless RMSNorm (paper default,
  ``QK-RMSNorm``) or L2 normalization (``QK-l2-norm`` ablation) of the
  query/key projections.
* :class:`FalconShortConv` -- lightweight causal depth-wise short
  convolutions on the attention projections (paper default: enabled).
* :func:`positive_decay` -- the numerically stable positive-decay
  renormalization primitive: clamps the per-step shrinkage
  ``alpha_t = min(eta_t * lambda_t, 1 - eps_gamma)`` so the carry
  ``gamma_t = 1 - alpha_t >= eps_gamma > 0`` stays strictly positive,
  and returns the fp32 log-carry used by the chunk-parallel kernels.

The framework treats the recurrent fast-weight state transition as an
online learning rule under read-after-write (RAW) autoregressive
semantics: the local fast-memory example revealed at step ``t`` is the
prefix-aligned pair ``(x_t, y_t) = (phi(k_{t-1}), v_t)``. All variants
share this one-step-shifted write stream, the boundary sentinels
(``x_1 = 0``, ``eta_1 = 0``, ``alpha_1 = 0``, ``gamma_1 = 1``) and the
normalized first-order step size
``eta_t = beta_t / (statistic_t + lambda_t + eps)`` where
``statistic_t`` is the objective-matched local curvature estimate
(``||x_t||^2`` for Falcon-1/1A/2/2A, the window spectral statistic
``mu_t^(B)`` for Falcon-3 and the window mean energy ``E_t^(B)`` for
Falcon-3A).

Reference:
    Zhang et al. (2026). "Fast Weight Attention for Continual
    Learning". arXiv:2608.27763.
"""

from __future__ import annotations

from typing import Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear

# Bound on any positive exponent used by the decay rescaling kernels.
# exp(80) ~= 5.5e34 stays finite in fp32 and only engages in extreme
# forgetting regimes where the carried state is numerically zero.
_EXP_CLAMP = 80.0


def _stable_exp(t: torch.Tensor) -> torch.Tensor:
    """Exponential with the positive exponent clamped for fp32 safety.

    Args:
        t: Exponent tensor (any shape, typically fp32).

    Returns:
        ``exp(t)`` with positive exponents clamped at 80.
    """
    return torch.exp(torch.clamp(t, max=_EXP_CLAMP))


def positive_decay(
    eta: torch.Tensor, lam: torch.Tensor, eps_gamma: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clamped positive-decay factors of the Falcon family.

    Computes ``alpha_t = min(eta_t * lambda_t, 1 - eps_gamma)`` so the
    carry ``gamma_t = 1 - alpha_t >= eps_gamma`` is strictly positive
    (the positive-decay surrogate of arXiv:2608.27763 Sec. 4), plus the
    fp32 ``log(gamma_t)`` used by the chunk-parallel kernels.

    Args:
        eta: Step sizes, any shape.
        lam: Ridge coefficients, broadcastable to ``eta``.
        eps_gamma: Decay floor in ``(0, 1)``.

    Returns:
        Tuple ``(alpha, gamma, log_gamma)`` where ``log_gamma`` is
        computed in float32 via ``log1p``.
    """
    alpha = torch.clamp_max(eta * lam, 1.0 - eps_gamma)
    gamma = 1.0 - alpha
    log_gamma = torch.log1p(-alpha.to(torch.float32))
    return alpha, gamma, log_gamma


def falcon_qk_norm(
    q: torch.Tensor, k: torch.Tensor, mode: str = "rms_norm", eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize query/key projections (paper Sec. 5, App. B.1).

    ``QK-RMSNorm`` (the paper default, computed in fp32 for mixed
    precision safety) keeps ``||x_t||^2 ~= d`` which stabilizes the
    NLMS denominator; ``QK-l2-norm`` is the DeltaNet-style ablation
    (``||x_t|| = 1``).

    Args:
        q: Queries of shape ``(B, S, H, D)``.
        k: Keys of shape ``(B, S, H, D)``.
        mode: ``"rms_norm"`` or ``"l2_norm"``.
        eps: Numerical stabilizer for the RMS / L2 denominators.

    Returns:
        Normalized ``(q, k)`` with the input dtype preserved.

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    if mode == "l2_norm":
        return F.normalize(q, dim=-1, eps=eps), F.normalize(k, dim=-1, eps=eps)
    if mode == "rms_norm":
        q32 = q.to(torch.float32)
        k32 = k.to(torch.float32)
        q32 = q32 * torch.rsqrt(q32.pow(2).mean(dim=-1, keepdim=True) + eps)
        k32 = k32 * torch.rsqrt(k32.pow(2).mean(dim=-1, keepdim=True) + eps)
        return q32.to(q.dtype), k32.to(k.dtype)
    raise ValueError(f"Unknown falcon_qk_norm mode {mode!r}; expected 'rms_norm' or 'l2_norm'")


class FalconShortConv(nn.Module):
    """Causal depth-wise short convolutions on the attention projections.

    A lightweight H3/SLConv-style inductive bias applied independently
    to the projected q, k and v streams. Each stream gets its own
    depth-wise ``nn.Conv1d`` (``groups = total_dim``, no bias) with left
    padding so the convolutions are causal and safe in decoder
    (autoregressive) mode. The paper notes this component is orthogonal
    to the fast-memory update and not required by the theory; it is on
    by default and can be disabled with ``falcon_short_conv=false``.

    Args:
        total_dim: Total projection width (``num_heads * head_dim``).
        kernel_size: Convolution kernel size. Default: 4.

    Attributes:
        conv_q/conv_k/conv_v: Depth-wise causal convolutions.
    """

    def __init__(self, total_dim: int, kernel_size: int = 4):
        """Initialize the three depth-wise causal convolutions."""
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv_q = nn.Conv1d(
            total_dim, total_dim, self.kernel_size, groups=total_dim,
            padding=self.kernel_size - 1, bias=False,
        )
        self.conv_k = nn.Conv1d(
            total_dim, total_dim, self.kernel_size, groups=total_dim,
            padding=self.kernel_size - 1, bias=False,
        )
        self.conv_v = nn.Conv1d(
            total_dim, total_dim, self.kernel_size, groups=total_dim,
            padding=self.kernel_size - 1, bias=False,
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply causal short convolutions to projected streams.

        Args:
            q: Queries of shape ``(B, S, total_dim)``.
            k: Keys of shape ``(B, S, total_dim)``.
            v: Values of shape ``(B, S, total_dim)``.

        Returns:
            Convoluted ``(q, k, v)`` of the same shapes.
        """
        seq_len = q.shape[1]
        pad = self.kernel_size - 1

        def _conv(stream: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
            out = conv(stream.transpose(1, 2))
            if pad > 0:
                out = out[:, :, :seq_len]
            return out.transpose(1, 2)

        return _conv(q, self.conv_q), _conv(k, self.conv_k), _conv(v, self.conv_v)


class FalconGates(nn.Module):
    """Plasticity (beta / eta numerator) and forgetting (lambda) gates.

    Implements the gate parameterizations of arXiv:2608.27763 App. B.1:

    * ``beta_mode="static"`` -- fixed scalar numerator ``beta``
      (dimensionless NLMS gain, used directly as ``eta`` numerator).
    * ``beta_mode="ctx_beta"`` -- the network emits the bounded gain
      ``beta_t = 2 * sigmoid(proj)`` in the descent-safe interval
      ``(0, 2)`` (paper naming: ``ctx_beta``).
    * ``beta_mode="ctx_eta"`` -- the network emits the (unbounded)
      step-size numerator via ``softplus`` (paper naming: ``ctx_eta``;
      allows the sign-flip regime ``beta > 2`` useful for state
      tracking).
    * ``lambda_mode="static"`` -- fixed scalar ridge ``lambda >= 0``.
    * ``lambda_mode="ctx"`` -- per-head base ridge
      ``lambda_bar_t = softplus(proj)`` combined with the detached
      objective statistic (scale-coupled ridge of Sec. 4.1:
      ``lambda_t = lambda_bar_t * statistic_t.detach()``) computed
      inside each variant's scan.

    Gate projections are initialized so the effective values at
    initialization are ``beta ~= 1`` (NLMS gain 1, an orthogonal
    projection along the write direction) and ``lambda_bar ~= 0.018``
    (nearly no forgetting until learned).

    Args:
        config: Model configuration object with ``hidden_size``,
            ``num_heads``, ``use_bitnet`` and the ``falcon_*`` knobs.
        per_column: If True, the plasticity gate emits one numerator
            per value channel ``(H, head_dim)`` (Falcon-2 / Falcon-2A);
            otherwise one scalar per head (Falcon-1/3/1A/3A).

    Attributes:
        beta_mode: One of ``"static"``, ``"ctx_beta"``, ``"ctx_eta"``.
        lambda_mode: One of ``"static"``, ``"ctx"``.
        beta_static: Static numerator used when ``beta_mode="static"``.
        lambda_static: Static ridge used when ``lambda_mode="static"``.
        beta_proj: Bounded-gain projection (``ctx_beta`` mode).
        eta_proj: Numerator projection (``ctx_eta`` mode).
        lambda_proj: Per-head base-ridge projection (``ctx`` mode).
    """

    def __init__(self, config, per_column: bool):
        """Initialize gate projections from the config knobs."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.per_column = bool(per_column)
        out_dim = self.num_heads * self.head_dim if per_column else self.num_heads

        self.beta_mode = str(getattr(config, "falcon_beta_mode", "ctx_eta")).lower()
        self.lambda_mode = str(getattr(config, "falcon_lambda_mode", "ctx")).lower()
        self.beta_static = float(getattr(config, "falcon_beta", 1.0))
        self.lambda_static = float(getattr(config, "falcon_lambda", 0.0))

        if self.beta_mode not in {"static", "ctx_beta", "ctx_eta"}:
            raise ValueError(
                f"Unknown falcon_beta_mode {self.beta_mode!r}; expected 'static', 'ctx_beta' or 'ctx_eta'"
            )
        if self.lambda_mode not in {"static", "ctx"}:
            raise ValueError(
                f"Unknown falcon_lambda_mode {self.lambda_mode!r}; expected 'static' or 'ctx'"
            )

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        gate_cls = BitLinear if (config.use_bitnet and getattr(config, "bitnet_routers", False)) else nn.Linear

        if self.beta_mode == "ctx_beta":
            self.beta_proj = gate_cls(self.hidden_size, out_dim, bias=True)
            nn.init.zeros_(self.beta_proj.bias)
        elif self.beta_mode == "ctx_eta":
            self.eta_proj = gate_cls(self.hidden_size, out_dim, bias=True)
            nn.init.constant_(self.eta_proj.bias, math.log(math.e - 1.0))
        if self.lambda_mode == "ctx":
            self.lambda_proj = gate_cls(self.hidden_size, self.num_heads, bias=True)
            nn.init.constant_(self.lambda_proj.bias, -4.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the per-position gate values.

        Args:
            x: Input tensor of shape ``(B, S, hidden_size)``.

        Returns:
            Tuple ``(num, lam_bar)``:

            * ``num``: step-size numerator of shape
              ``(B, S, H)`` (scalar variants) or ``(B, S, H, D)``
              (per-column variants).
            * ``lam_bar``: per-head base ridge of shape ``(B, S, H)``
              (only meaningful when ``lambda_mode="ctx"``; zeros
              otherwise).
        """
        bsz, seq_len, _ = x.shape
        head_shape = (bsz, seq_len, self.num_heads, self.head_dim)
        if self.beta_mode == "static":
            if self.per_column:
                num = x.new_full(head_shape, self.beta_static)
            else:
                num = x.new_full((bsz, seq_len, self.num_heads), self.beta_static)
        elif self.beta_mode == "ctx_beta":
            num = 2.0 * torch.sigmoid(self.beta_proj(x))
            if self.per_column:
                num = num.view(head_shape)
        else:  # ctx_eta
            num = F.softplus(self.eta_proj(x))
            if self.per_column:
                num = num.view(head_shape)

        if self.lambda_mode == "ctx":
            lam_bar = F.softplus(self.lambda_proj(x))
        else:
            lam_bar = x.new_zeros(bsz, seq_len, self.num_heads)
        return num, lam_bar

    def resolve_lambda(self, lam_bar: torch.Tensor, statistic: torch.Tensor) -> torch.Tensor:
        """Combine the base ridge with the objective statistic.

        In ``ctx`` mode this applies the scale-coupled ridge
        ``lambda_t = lam_bar_t * statistic_t`` with the statistic
        detached (statistics-only multiplier, Sec. 4.1 of the paper);
        in ``static`` mode the scalar ridge is returned unchanged.

        Args:
            lam_bar: Base ridge from :meth:`forward`.
            statistic: Objective-matched curvature statistic
                (``||x_t||^2``, ``mu_t^(B)`` or ``E_t^(B)``).

        Returns:
            Effective per-step ridge, broadcastable to ``statistic``.
        """
        if self.lambda_mode == "ctx":
            return lam_bar * statistic.detach()
        return statistic.new_full((), self.lambda_static)
