"""Gaussian Mixture Attention (GMA) for linear-time sequence mixing.

Implements the probabilistic latent-routing attention mixer of Huang &
Raza (2026), arXiv:2606.18283. GMA replaces the explicit pairwise
``Q K^T`` interaction of standard dot-product attention with routing
through ``K`` learned Gaussian-mixture components. Queries and keys are
mapped to posterior *responsibility* vectors over a shared latent routing
space; their overlap defines an implicit responsibility-space affinity,
while values are written into and read from a ``K``-slot latent memory.

By exploiting the associativity of matrix multiplication, GMA avoids
materialising the induced ``N x N`` affinity matrix and instead uses two
``N x K`` responsibility matrices whose dominant activation storage
scales as ``O(NK)`` rather than ``O(N^2)`` for fixed ``K``. The
attention-specific routing cost is

    C_GMA ~= 2 * N * K * d_r + 2 * N * K * d_v + 4 * N * K + N * d_v
           = O(N * K * d_r + N * K * d_v)

which is linear in the sequence length ``N`` for fixed number of
mixture components ``K``, routing dimension ``d_r`` and value dimension
``d_v``.

Formulation (single head, bidirectional):

    Q_X = X W_Q in R^{N x d_r}
    K_X = X W_K in R^{N x d_r}
    V_X = X W_V in R^{N x d_v}

    gamma_{i,k} = p(z=k | x_i)
                = pi_k * N(x_i | mu_k, Sigma_k)
                  / sum_l pi_l * N(x_i | mu_l, Sigma_l)        (Eq. 2)

    Gamma^Q, Gamma^K in R^{N x K}                         (posterior)

    V_tilde = (Gamma^K)^T V_X  in R^{K x d_v}              (write)
    Z       = (Gamma^K)^T 1_N  in R^{K}                   (normaliser)

    O = Gamma^Q V_tilde / (Gamma^Q Z + eps)               (read)

The implicit normalised affinity ``A^GMA = diag(d)^{-1} Gamma^Q
(Gamma^K)^T`` (with ``d = Gamma^Q Z + eps 1_N``) is never materialised;
only the associative ``V_tilde = (Gamma^K)^T V_X`` then
``Gamma^Q V_tilde`` path is computed.

Causal variant (autoregressive decoding / ``mode == "decoder"``): the
global write statistics are replaced by prefix-restricted cumulative
sums along the sequence dimension,

    V_tilde^(i)_k = sum_{j<=i} gamma^K_{j,k} V_{X,j}
    Z^(i)_k       = sum_{j<=i} gamma^K_{j,k}

so position ``i`` only reads from its prefix. Causality is therefore
enforced while preserving the same fixed-``K`` linear scaling.

End-to-end parameter learning: the Gaussian mixture parameters of every
head --- the component means ``mu``, the diagonal-covariance
reparameterisation ``omega`` (with ``sigma^2 = softplus(omega) +
eps_sigma``) and the mixture-prior logits ``alpha`` (with
``pi = softmax(alpha)``) --- are fully differentiable learnable
parameters optimised jointly with the rest of the model by standard
backpropagation. No EM loop or auxiliary clustering loss is required.

Reference:
    Huang, Y. & Raza, H. (2026). "Gaussian Mixture Attention:
    Linear-Time Sequence Mixing via Probabilistic Latent Routing",
    arXiv:2606.18283.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import BitLinear


class GaussianMixtureAttention(nn.Module):
    """Gaussian Mixture Attention with per-head learned GMM routing.

    Replaces the pairwise ``Q K^T`` interaction of standard attention with
    probabilistic routing through ``K`` learned Gaussian-mixture
    components per head. Values are written into a ``K``-slot latent
    memory via key responsibilities and read back via query
    responsibilities, giving ``O(NK)`` activation storage for fixed ``K``.

    Args:
        config: Model configuration object with attributes
            ``hidden_size``, ``num_heads``, ``dropout``, ``use_bitnet``,
            ``mode`` and the optional GMA knobs ``gma_num_components``
            (default 8), ``gma_routing_dim`` (default ``head_dim``),
            ``gma_epsilon`` (default 1e-6), ``gma_sigma_eps`` (default
            1e-4) and ``gma_init_mean_std`` (default 1.0).
        pos_encoder: Shared positional encoding module applied to the
            routing query/key tensors. If ``None``, no PE is applied.
            Note: GMA does not materialise an ``(S, S)`` attention matrix,
            so score-bias PEs (e.g. ALiBi) are not applicable; only
            rotation-type PEs (rope/hope) affect the routing space.

    Attributes:
        hidden_size: Input embedding dimensionality.
        num_heads: Number of attention heads.
        head_dim: Dimensionality per head (``hidden_size // num_heads``).
            Used as the value dimension ``d_v``.
        num_components: Number ``K`` of Gaussian mixture components per
            head.
        routing_dim: Routing dimension ``d_r`` used to compute Gaussian
            responsibilities.
        epsilon: Numerical-stability constant for the read normaliser.
        sigma_eps: Lower bound for the diagonal variances.
        init_mean_std: Initialisation standard deviation for the
            component means ``mu``.
        q_proj: Query projection ``hidden_size -> num_heads * d_r``.
        k_proj: Key projection ``hidden_size -> num_heads * d_r``.
        v_proj: Value projection ``hidden_size -> num_heads * d_v``.
        out_proj: Output projection ``num_heads * d_v -> hidden_size``.
        mu: Component means, shape ``(num_heads, K, d_r)``.
        omega: Diagonal-covariance reparameterisation,
            shape ``(num_heads, K, d_r)``.
        alpha: Mixture-prior logits, shape ``(num_heads, K)``.
        dropout: Dropout layer applied to the responsibilities.
        mode: ``"encoder"`` (bidirectional) or ``"decoder"`` (causal via
            prefix cumsums).
        pos_encoder: Shared positional encoding module (or ``None``).
        pe_type: Positional encoding type string (lowercased).
        use_pe: Whether PE is enabled for this mixer.

    Raises:
        ValueError: If ``hidden_size`` is not divisible by ``num_heads``,
            if ``gma_num_components < 1``, or if any stability constant
            is non-positive.
    """

    def __init__(self, config, pos_encoder=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads for "
                "GaussianMixtureAttention"
            )
        self.head_dim = self.hidden_size // self.num_heads

        # GMA knobs (flattened from model.attention.gma.* by config_flatten).
        self.num_components = int(getattr(config, "gma_num_components", 8))
        if self.num_components < 1:
            raise ValueError(
                f"gma_num_components must be >= 1, got {self.num_components}"
            )
        routing_dim = getattr(config, "gma_routing_dim", None)
        if routing_dim is None:
            routing_dim = self.head_dim
        self.routing_dim = int(routing_dim)
        if self.routing_dim < 1:
            raise ValueError(
                f"gma_routing_dim must be >= 1, got {self.routing_dim}"
            )
        self.epsilon = float(getattr(config, "gma_epsilon", 1e-6))
        if self.epsilon <= 0:
            raise ValueError(
                f"gma_epsilon must be > 0, got {self.epsilon}"
            )
        self.sigma_eps = float(getattr(config, "gma_sigma_eps", 1e-4))
        if self.sigma_eps <= 0:
            raise ValueError(
                f"gma_sigma_eps must be > 0, got {self.sigma_eps}"
            )
        self.init_mean_std = float(getattr(config, "gma_init_mean_std", 1.0))

        self.mode = getattr(config, "mode", "encoder")
        proj_cls = BitLinear if config.use_bitnet else nn.Linear

        # Q/K project into the *routing* space (d_r); V projects into the
        # value space (d_v = head_dim). out_proj maps the concatenated
        # heads back to the hidden dimension.
        self.q_proj = proj_cls(self.hidden_size, self.num_heads * self.routing_dim, bias=False)
        self.k_proj = proj_cls(self.hidden_size, self.num_heads * self.routing_dim, bias=False)
        self.v_proj = proj_cls(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.out_proj = proj_cls(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # Per-head Gaussian mixture parameters (arXiv:2606.18283, Sec. 3.7).
        #   mu    : component means          (H, K, d_r)
        #   omega : diagonal-cov reparam      (H, K, d_r)  -> sigma^2 = softplus(omega) + eps_sigma
        #   alpha : mixture-prior logits       (H, K)      -> pi      = softmax(alpha)
        self.mu = nn.Parameter(
            torch.randn(self.num_heads, self.num_components, self.routing_dim) * self.init_mean_std
        )
        self.omega = nn.Parameter(
            torch.zeros(self.num_heads, self.num_components, self.routing_dim)
        )
        self.alpha = nn.Parameter(
            torch.zeros(self.num_heads, self.num_components)
        )

        # Precompute the constant part of the diagonal-Gaussian log-density
        # log N(x | mu, diag(sigma^2)) = -0.5 * (d_r * log(2*pi)
        #   + sum_m log(sigma^2_m) + sum_m (x-mu)^2_m / sigma^2_m).
        self.register_buffer(
            "_log_2pi",
            torch.tensor(math.log(2.0 * math.pi), dtype=torch.float32),
            persistent=False,
        )
        self.pos_encoder = pos_encoder
        self.pe_type = str(getattr(config, "positional_encoding", "rope")).lower()
        self.use_pe = bool(getattr(config, "gma_attn_use_pe", True))

    # ------------------------------------------------------------------
    # Core GMA primitives
    # ------------------------------------------------------------------
    def _log_density_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the pre-normalised log-density scores ``s_{h,i,k}``.

        For a routing tensor ``x`` of shape ``(B, N, H, d_r)`` this returns
        the log of ``pi_k * N(x | mu_k, diag(sigma^2_k))`` (up to the
        shared per-token normaliser that cancels under softmax), as a
        tensor of shape ``(B, N, H, K)``.

        Args:
            x: Routing tensor of shape ``(B, N, H, d_r)``.

        Returns:
            Log-density scores of shape ``(B, N, H, K)``.
        """
        # sigma^2 = softplus(omega) + eps_sigma  -> (H, K, d_r)
        sigma_sq = F.softplus(self.omega) + self.sigma_eps
        log_sigma_sq = torch.log(sigma_sq)  # (H, K, d_r)

        # Expand routing tensor: x   -> (B, N, H, 1, d_r)
        # Expand mixture params:  mu  -> (1, 1, H, K, d_r)
        x_exp = x.unsqueeze(-2)                              # (B, N, H, 1, d_r)
        mu_exp = self.mu.unsqueeze(0).unsqueeze(0)           # (1, 1, H, K, d_r)
        sigma_sq_exp = sigma_sq.unsqueeze(0).unsqueeze(0)    # (1, 1, H, K, d_r)
        log_sigma_sq_exp = log_sigma_sq.unsqueeze(0).unsqueeze(0)

        diff = x_exp - mu_exp                                 # (B, N, H, K, d_r)
        mahalanobis = (diff * diff / sigma_sq_exp).sum(dim=-1)  # (B, N, H, K)

        # s_{i,k} = log pi_k - 0.5 * (d_r * log(2 pi) + sum_m log sigma^2_m
        #                            + sum_m (x-mu)^2_m / sigma^2_m)
        log_pi = F.log_softmax(self.alpha, dim=-1)            # (H, K)
        log_pi_exp = log_pi_exp = log_pi.unsqueeze(0).unsqueeze(0)  # (1, 1, H, K)

        const = -0.5 * (
            self.routing_dim * self._log_2pi
            + log_sigma_sq_exp.sum(dim=-1)                    # (1, 1, H, K)
        )
        scores = log_pi_exp + const - 0.5 * mahalanobis        # (B, N, H, K)
        return scores

    def _responsibilities(self, x: torch.Tensor) -> torch.Tensor:
        """Compute posterior responsibilities ``Gamma`` for a routing tensor.

        Args:
            x: Routing tensor of shape ``(B, N, H, d_r)``.

        Returns:
            Responsibilities of shape ``(B, N, H, K)``; each row over
            ``K`` sums to 1.
        """
        scores = self._log_density_scores(x)                  # (B, N, H, K)
        return scores.softmax(dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, logical_layer_idx: Optional[int] = None, pos_encoder=None) -> torch.Tensor:
        """Compute Gaussian Mixture Attention.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            logical_layer_idx: Logical layer index passed to the positional
                encoder. Defaults to ``0`` if ``None``.
            pos_encoder: Optional positional encoding module overriding
                ``self.pos_encoder`` for this forward call. Only rotation-type
                PEs (rope/hope) affect the routing space; score-bias PEs
                (e.g. ALiBi) are no-ops since GMA does not materialise an
                attention matrix.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, seq_len, _ = x.shape
        H, K = self.num_heads, self.num_components
        d_r, d_v = self.routing_dim, self.head_dim
        pe = pos_encoder if pos_encoder is not None else self.pos_encoder

        # --- Projections ---------------------------------------------------
        q = self.q_proj(x).view(bsz, seq_len, H, d_r)          # (B, N, H, d_r)
        k = self.k_proj(x).view(bsz, seq_len, H, d_r)          # (B, N, H, d_r)
        v = self.v_proj(x).view(bsz, seq_len, H, d_v)          # (B, N, H, d_v)

        # --- Positional encoding on routing tensors (rotation only) -------
        from ..common import apply_pe_to_qk
        q_t = q.transpose(1, 2)                               # (B, H, N, d_r)
        k_t = k.transpose(1, 2)
        q_t, k_t = apply_pe_to_qk(pe, self.pe_type, q_t, k_t, x, logical_layer_idx or 0, self.use_pe)
        q = q_t.transpose(1, 2)
        k = k_t.transpose(1, 2)

        # --- Responsibilities ----------------------------------------------
        gamma_q = self._responsibilities(q)                    # (B, N, H, K)
        gamma_k = self._responsibilities(k)                    # (B, N, H, K)
        gamma_q = self.dropout(gamma_q)
        gamma_k = self.dropout(gamma_k)

        if self.mode == "decoder":
            # Causal GMA: prefix-restricted write statistics via cumsum
            # along the sequence axis (arXiv:2606.18283, Sec. 3.6).
            # V_tilde^(i)_k = sum_{j<=i} gamma^K_{j,k} V_{X,j}
            # Z^(i)_k       = sum_{j<=i} gamma^K_{j,k}
            weighted_v = gamma_k.unsqueeze(-1) * v.unsqueeze(-2)  # (B, N, H, K, d_v)
            v_tilde = torch.cumsum(weighted_v, dim=1)             # (B, N, H, K, d_v)
            z_prefix = torch.cumsum(gamma_k, dim=1)                # (B, N, H, K)
        else:
            # Bidirectional GMA: global write statistics.
            # V_tilde = (Gamma^K)^T V_X  in R^{K x d_v} per head.
            # Z       = (Gamma^K)^T 1_N  in R^{K}      per head.
            v_tilde = torch.einsum("bnhk,bnhd->bhkd", gamma_k, v)  # (B, H, K, d_v)
            z_prefix = gamma_k.sum(dim=1)                         # (B, H, K)

        # --- Read ----------------------------------------------------------
        # numerator_i = sum_k gamma^Q_{i,k} * V_tilde_k        -> (B, N, H, d_v)
        # denom_i     = sum_k gamma^Q_{i,k} * Z_k + epsilon   -> (B, N, H)
        if self.mode == "decoder":
            numerator = torch.einsum("bnhk,bnhkd->bnhd", gamma_q, v_tilde)
            denominator = (gamma_q * z_prefix).sum(dim=-1) + self.epsilon  # (B, N, H)
        else:
            numerator = torch.einsum("bnhk,bhkd->bnhd", gamma_q, v_tilde)
            denominator = torch.einsum("bnhk,bhk->bnh", gamma_q, z_prefix) + self.epsilon

        out = numerator / denominator.unsqueeze(-1)               # (B, N, H, d_v)
        out = out.reshape(bsz, seq_len, H * d_v)                   # (B, N, H*d_v)
        return self.out_proj(out)