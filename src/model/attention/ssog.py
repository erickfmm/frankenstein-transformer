"""SSOG: Separable Sum of Gaussians attention.

Replaces content-scored attention (QK^T) with a learned geometric
field over relative position. Each head owns a handful of Gaussian
atoms — five numbers per atom: a center offset
``(mu_y, mu_x)``, a width per axis ``(sigma_y, sigma_x)`` and a
mixture weight ``lambda``. The attention weight from token ``p`` to
token ``q`` is the softmax-tempered log-sum-exp of the atom
log-weights evaluated at the displacement ``p - q``:

    s(p, q) = logsumexp_r( log lambda_r
                           + log N(p - q; mu_r, sigma_r) )

Because a 2D Gaussian factorizes into a product of two 1D Gaussians,
applying the field is two 1D filter passes per atom (rows, then
columns) plus a lambda-mix contraction. The induced ``N x N``
attention matrix never exists, so the attention-specific cost is
``O(R * N * sqrt(N) * d)`` for ``R`` atoms on a
``grid_h x grid_w = sqrt(N) x sqrt(N)`` token grid, instead of the
``O(N^2 * d)`` of scaled dot-product attention. SSOG also skips the
query and key projections entirely (``2 d^2`` of projections instead
of ``4 d^2``).

Content never scores; it only *steers*. With ``lookat`` enabled
(default), zero-initialized linear probes predict bounded per-token
residuals on the field parameters behind cold-started softplus gates:

    mu     <- mu_0 + s_mu * max_offset * tanh(W_mu x)
    sigma  <- sigma_0 * exp(s_sigma * tanh(W_sigma x))
    lambda <- softmax( log lambda_0 + s_lambda * tanh(W_lambda x) )

The gates start at ``softplus(-8) ~= 3e-4``, so the model begins as a
frozen geometric field and learns how far to open each content tap.
Positional encodings are not applicable — the field *is* positional.

Tokens must be the row-major raster of a ``grid_h x grid_w`` grid
(the ViT patch embedding produces exactly this). A ``grid_h = 1``
configuration degenerates naturally to a 1D positional-field
attention for non-vision sequences.

Encoder-only: the separable factorization has no causal formulation,
so ``mode="decoder"`` raises at construction time.

Reference:
    Pisoni, R. (2026), "A Few Gaussians Is All You Need: SSOG-Attention
    That Steers Instead of Scores",
    https://www.pisoni.ai/posts/ssog/ (reference implementation:
    https://github.com/4rtemi5/ssog, AGPL-3.0; this module is an
    independent MIT-licensed implementation of the published
    formulation).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BitLinear

_EPS = 1e-4


def _softplus_std(raw: torch.Tensor, floor: float = 0.0) -> torch.Tensor:
    """Map an unconstrained parameter to a strictly positive width/scale.

    Args:
        raw: Unconstrained tensor (any shape).
        floor: Optional lower bound added after the softplus.

    Returns:
        Tensor of the same shape with values in ``(floor + eps, inf)``.
    """
    return F.softplus(raw) + _EPS + floor


def _log_kernel(d: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Evaluate ``log N(d; mu, sigma^2)`` for every pair along one axis.

    Args:
        d: Pairwise displacements (broadcastable against ``mu``/``sigma``).
        mu: Gaussian centers (broadcastable against ``d``).
        sigma: Gaussian widths (broadcastable against ``d``).

    Returns:
        Tensor of the broadcast shape holding ``log N(d; mu, sigma^2)``.
    """
    return (
        -0.5 * math.log(2.0 * math.pi)
        - torch.log(sigma)
        - (d - mu).square() / (2.0 * sigma.square())
    )


class SSOGAttention(nn.Module):
    """Gaussian-mixture attention field over relative position.

    Tokens must arrive as the row-major raster of a
    ``grid_h x grid_w`` grid. Each head owns ``num_atoms`` Gaussian
    atoms; the shared field is identical for every input, and the
    optional ``lookat`` steering lets each token nudge the field with
    bounded residuals predicted from its own content.

    Complexity:
        Training/eval: ``O(R * N * sqrt(N) * d)`` — the ``N x N``
        attention matrix is never materialized.

    Reference:
        Pisoni (2026), https://www.pisoni.ai/posts/ssog/

    Args:
        config: Model configuration object with attributes
            ``hidden_size``, ``num_heads``, ``dropout``, ``use_bitnet``,
            ``mode`` and the SSOG knobs ``ssog_num_atoms`` (default 4),
            ``ssog_lookat`` (default True), ``ssog_max_offset``
            (default 4.0), ``ssog_cold_init`` (default True),
            ``ssog_sigma_floor`` (default 0.25), ``ssog_grid_h`` and
            ``ssog_grid_w`` (default: derived from the image/patch
            dimensions; must be set explicitly for non-vision grids).
        pos_encoder: Unused — accepted for interface compatibility with
            the other attention mixers. SSOG defines its own positional
            field, so external positional encodings are ignored.

    Attributes:
        hidden_size: Dimensionality of the input and output embeddings.
        num_heads: Number of parallel heads; each owns ``num_atoms``
            Gaussian atoms.
        head_dim: Dimensionality per head (``hidden_size // num_heads``).
        num_atoms: Gaussian atoms per head.
        grid_h: Rows of the token grid.
        grid_w: Columns of the token grid.
        lookat: Whether content-conditioned steering is enabled.
        max_offset: Bound on per-token center travel, in grid cells.
        sigma_floor: Minimum atom width in grid cells.
        mu: Learnable atom centers, shape ``(H, R, 2)``.
        raw_sigma: Unconstrained atom widths (softplus-reparametrized),
            shape ``(H, R, 2)``.
        log_lambda: Unconstrained atom mixture logits, shape ``(H, R)``.
        raw_temperature: Unconstrained softmax temperature scalar.
        dy: Buffer of pairwise row displacements ``(grid_h, grid_h)``.
        dx: Buffer of pairwise column displacements ``(grid_w, grid_w)``.
        v_proj: Linear (or BitLinear) value projection.
        out_proj: Linear (or BitLinear) output projection.

    Raises:
        ValueError: If ``hidden_size`` is not divisible by
            ``num_heads``, if ``mode == "decoder"`` (no causal
            formulation exists), or if the grid dimensions are invalid.
    """

    def __init__(self, config, pos_encoder=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads for SSOGAttention"
            )
        if str(getattr(config, "mode", "encoder")) == "decoder":
            raise ValueError(
                "ssog_attn is encoder-only: the separable Gaussian field has "
                "no causal formulation. Set model.dims.mode to 'encoder' and "
                "remove ssog_attn from layer_pattern for decoder models."
            )

        self.num_atoms = int(getattr(config, "ssog_num_atoms", 4))
        self.grid_h = int(getattr(config, "ssog_grid_h", 8))
        self.grid_w = int(getattr(config, "ssog_grid_w", 8))
        self.lookat = bool(getattr(config, "ssog_lookat", True))
        self.max_offset = float(getattr(config, "ssog_max_offset", 4.0))
        self.sigma_floor = float(getattr(config, "ssog_sigma_floor", 0.25))
        self.pos_encoder = pos_encoder

        if self.num_atoms < 1:
            raise ValueError(f"ssog_num_atoms must be >= 1, got {self.num_atoms}")
        if self.grid_h < 1 or self.grid_w < 1:
            raise ValueError(
                f"ssog grid must be >= 1x1, got {self.grid_h}x{self.grid_w}"
            )

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = proj_cls(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # Shared field: per-head Gaussian atoms.
        self.mu = nn.Parameter(torch.empty(self.num_heads, self.num_atoms, 2))
        nn.init.normal_(self.mu, mean=0.0, std=0.5)
        self.raw_sigma = nn.Parameter(
            torch.full((self.num_heads, self.num_atoms, 2), -0.5)
        )
        self.log_lambda = nn.Parameter(
            torch.zeros(self.num_heads, self.num_atoms)
        )
        # Softmax temperature, initialized slightly sharp:
        # softplus(-1) + 0.5 ~= 0.81.
        self.raw_temperature = nn.Parameter(torch.tensor(-1.0))

        # Pairwise displacements along each grid axis — constant.
        ys = torch.arange(self.grid_h, dtype=torch.float32)
        xs = torch.arange(self.grid_w, dtype=torch.float32)
        self.register_buffer("dy", ys[:, None] - ys[None, :], persistent=False)
        self.register_buffer("dx", xs[:, None] - xs[None, :], persistent=False)

        if self.lookat:
            h, r = self.num_heads, self.num_atoms
            gate0 = -8.0 if bool(getattr(config, "ssog_cold_init", True)) else -2.0
            # Plain nn.Linear (never BitLinear): the probes must stay
            # exactly zero-initialized for the cold start, and their
            # bounded residuals are too small to benefit from ternary
            # quantization.
            self.mu_delta = nn.Linear(self.hidden_size, h * r * 2)
            self.sigma_delta = nn.Linear(self.hidden_size, h * r * 2)
            self.lambda_gate = nn.Linear(self.hidden_size, h * r)
            nn.init.zeros_(self.mu_delta.weight)
            nn.init.zeros_(self.mu_delta.bias)
            nn.init.zeros_(self.sigma_delta.weight)
            nn.init.zeros_(self.sigma_delta.bias)
            nn.init.zeros_(self.lambda_gate.weight)
            nn.init.zeros_(self.lambda_gate.bias)
            # softplus(-8) + eps ~= 3e-4, so steering is off at init.
            self.raw_mu_delta_scale = nn.Parameter(torch.tensor(gate0))
            self.raw_sigma_delta_scale = nn.Parameter(torch.tensor(gate0))
            self.raw_lambda_gate_scale = nn.Parameter(torch.tensor(gate0))

    def _sigma(self) -> torch.Tensor:
        """Positive atom widths ``(H, R, 2)`` from the raw parameter."""
        return _softplus_std(self.raw_sigma, floor=self.sigma_floor)

    def _temperature(self) -> torch.Tensor:
        """Positive softmax temperature scalar."""
        return _softplus_std(self.raw_temperature) + 0.5

    def forward(
        self,
        x: torch.Tensor,
        logical_layer_idx: Optional[int] = None,
        pos_encoder=None,
    ) -> torch.Tensor:
        """Apply the Gaussian attention field to the token raster.

        Args:
            x: Input tensor of shape ``(batch_size, grid_h * grid_w,
                hidden_size)`` — the row-major raster of the token grid.
            logical_layer_idx: Logical layer index (unused; accepted for
                interface compatibility with the other mixers).
            pos_encoder: Unused positional encoder (the field is
                positional by construction).

        Returns:
            Output tensor of shape ``(batch_size, grid_h * grid_w,
            hidden_size)``.

        Raises:
            ValueError: If the sequence length is not
                ``grid_h * grid_w``.
        """
        bsz, seq_len, _ = x.shape
        if seq_len != self.grid_h * self.grid_w:
            raise ValueError(
                f"ssog_attn received seq_len={seq_len} but its field is "
                f"defined on a {self.grid_h}x{self.grid_w} raster "
                f"({self.grid_h * self.grid_w} tokens). Set attention.ssog."
                f"grid_h/grid_w to match (ViT configs derive it from "
                f"image_size/patch_size; use grid_h=1, grid_w=max_len for "
                f"non-vision sequences)."
            )

        gh, gw, hd = self.grid_h, self.grid_w, self.head_dim
        v = self.v_proj(x).view(bsz, gh, gw, self.num_heads, hd)
        sigma = self._sigma()
        temperature = self._temperature()

        if self.lookat:
            y = self._steered_apply(x, v, sigma, temperature)
        else:
            y = self._fixed_apply(v, sigma, temperature)
        return self.out_proj(y.reshape(bsz, seq_len, self.hidden_size))

    def _fixed_apply(
        self,
        v: torch.Tensor,
        sigma: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the shared (content-blind) field: same attention everywhere.

        Args:
            v: Values reshaped to ``(B, grid_h, grid_w, H, head_dim)``.
            sigma: Positive atom widths ``(H, R, 2)``.
            temperature: Positive softmax temperature scalar.

        Returns:
            Field-mixed values of shape ``(B, grid_h, grid_w, H, head_dim)``.
        """
        sy, sx = sigma[:, :, 0], sigma[:, :, 1]
        mu_y, mu_x = self.mu[:, :, 0], self.mu[:, :, 1]

        # (H, R, L, L) kernels over row / column displacements: the
        # row kernel is [query_row, key_row], the column kernel
        # [query_col, key_col].
        log_ay = _log_kernel(
            self.dy[None, None], mu_y[:, :, None, None], sy[:, :, None, None]
        )
        log_ax = _log_kernel(
            self.dx[None, None], mu_x[:, :, None, None], sx[:, :, None, None]
        )
        ay = self.dropout(torch.softmax(log_ay / temperature, dim=-1))
        ax = self.dropout(torch.softmax(log_ax / temperature, dim=-1))
        lam = torch.softmax(self.log_lambda, dim=-1)

        # Two 1D filter passes per atom, then the lambda mix — the
        # N x N matrix never exists.
        y = torch.einsum("prij,bjwpd->biwpdr", ay, v)
        y = torch.einsum("prjk,bikpdr->bijpdr", ax, y)
        return torch.einsum("pr,bijpdr->bijpd", lam, y)

    def _steered_apply(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        sigma: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the field with bounded per-query residuals on mu/sigma/lambda.

        Args:
            x: Raw input tokens ``(B, grid_h * grid_w, hidden_size)``.
            v: Values reshaped to ``(B, grid_h, grid_w, H, head_dim)``.
            sigma: Positive atom widths ``(H, R, 2)``.
            temperature: Positive softmax temperature scalar.

        Returns:
            Steered field-mixed values of shape
            ``(B, grid_h, grid_w, H, head_dim)``.
        """
        bsz = x.shape[0]
        gh, gw = self.grid_h, self.grid_w
        h, r = self.num_heads, self.num_atoms

        # mu: shift where each atom looks, bounded to +/- max_offset cells.
        mu_scale = _softplus_std(self.raw_mu_delta_scale)
        dmu = self.mu_delta(x).view(bsz, gh, gw, h, r, 2)
        mu_y = self.mu[None, None, None, :, :, 0] + mu_scale * self.max_offset * torch.tanh(dmu[..., 0])
        mu_x = self.mu[None, None, None, :, :, 1] + mu_scale * self.max_offset * torch.tanh(dmu[..., 1])

        # sigma: widen / tighten each atom, bounded log-space multiplier.
        sig_scale = _softplus_std(self.raw_sigma_delta_scale)
        dsig = self.sigma_delta(x).view(bsz, gh, gw, h, r, 2)
        sy = sigma[None, None, None, :, :, 0] * torch.exp(sig_scale * torch.tanh(dsig[..., 0]))
        sx = sigma[None, None, None, :, :, 1] * torch.exp(sig_scale * torch.tanh(dsig[..., 1]))

        # Per-query per-axis kernels: (B, gh, gw, H, R, L). The row
        # kernel keys on key rows j (displacement dy[i, j] for the
        # token's own row i); the column kernel on key cols k.
        log_ay = _log_kernel(self.dy[None, :, None, None, None, :], mu_y[..., None], sy[..., None])
        log_ax = _log_kernel(self.dx[None, None, :, None, None, :], mu_x[..., None], sx[..., None])
        ay = self.dropout(torch.softmax(log_ay / temperature, dim=-1))
        ax = self.dropout(torch.softmax(log_ax / temperature, dim=-1))

        y = torch.einsum("biwprj,bjwpd->biwpdr", ay, v)
        y = torch.einsum("biwprk,bikpdr->biwpdr", ax, y)

        # lambda: re-weight which atoms matter, per query.
        lam_scale = _softplus_std(self.raw_lambda_gate_scale)
        gate = self.lambda_gate(x).view(bsz, gh, gw, h, r)
        lam_q = torch.softmax(
            self.log_lambda[None, None, None] + lam_scale * torch.tanh(gate),
            dim=-1,
        )
        return torch.einsum("biwpr,biwpdr->biwpd", lam_q, y)

    def axis_kernels(self) -> tuple:
        """Fixed-field kernels for plotting — not used in ``forward``.

        Returns:
            Tuple ``(ay, ax)`` of the shared softmaxed kernels, with
            shapes ``(H, R, grid_h, grid_h)`` and
            ``(H, R, grid_w, grid_w)``. Call this in eager mode; it is
            not part of the compiled training graph.
        """
        sigma = self._sigma()
        temperature = self._temperature()
        sy, sx = sigma[:, :, 0], sigma[:, :, 1]
        mu_y, mu_x = self.mu[:, :, 0], self.mu[:, :, 1]
        ay = torch.softmax(
            _log_kernel(self.dy[None, None], mu_y[:, :, None, None], sy[:, :, None, None])
            / temperature,
            dim=-1,
        )
        ax = torch.softmax(
            _log_kernel(self.dx[None, None], mu_x[:, :, None, None], sx[:, :, None, None])
            / temperature,
            dim=-1,
        )
        return ay, ax
