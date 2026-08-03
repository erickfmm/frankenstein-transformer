"""ODE-style continuous-depth attention block.

Models the attention transformation as an ordinary differential equation
initial value problem::

    z(1) = z(0) + integral_0^1 f(z(t), t) dt

where ``f`` is a self-attention derivative function. The ODE is solved via
explicit numerical integration (Euler or RK4) with a configurable number of
steps. Weights are shared across all integration steps, providing a
continuous-depth parameterization.

Reference:
    Zhang et al. (2021), "ODE Transformer: An Ordinary Differential Equation
    Inspired Transformer", ACL 2022.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BitLinear
from ..norm import get_norm


class ODEFunc(nn.Module):
    """Derivative function ``dx/dt`` modeled as multi-head self-attention.

    This module represents the right-hand side ``f(z(t), t)`` of the ODE.
    It applies normalization, a fused QKV projection, scaled dot-product
    attention with softmax, and an output projection. The same weights are
    reused at every integration step, enabling a continuous-depth
    parameterization with constant parameter count regardless of the number
    of solver steps.

    Reference:
        Zhang et al. (2021), "ODE Transformer", ACL 2022.

    Args:
        config: Model configuration object with attributes ``hidden_size``,
            ``num_heads``, ``dropout``, ``use_bitnet``, ``norm_type``, and
            optionally ``mode`` (``"encoder"`` or ``"decoder"``).

    Attributes:
        dim: Dimensionality of the input and output embeddings.
        num_heads: Number of parallel attention heads.
        head_dim: Dimensionality of each attention head (``dim // num_heads``).
        scale: Scaling factor ``1 / sqrt(head_dim)`` applied to dot products.
        qkv: Fused linear (or BitLinear) projection for queries, keys, and
            values (outputs ``dim * 3`` features).
        out_proj: Linear (or BitLinear) output projection.
        norm: Normalization layer applied before the QKV projection.
        dropout: Dropout layer applied to attention weights.
        mode: ``"encoder"`` for bidirectional, ``"decoder"`` for causal.
    """

    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        proj_cls = BitLinear if config.use_bitnet else nn.Linear
        self.qkv = proj_cls(self.dim, self.dim * 3, bias=False)
        self.out_proj = proj_cls(self.dim, self.dim, bias=False)
        self.norm = get_norm(config)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = getattr(config, "mode", "encoder")

    def forward(self, t, x):
        """Evaluate the derivative function at time ``t``.

        Args:
            t: Current integration time (float; unused in the attention
                computation but accepted for ODE solver interface).
            x: State tensor of shape ``(batch_size, seq_len, dim)``.

        Returns:
            Derivative tensor ``dx/dt`` of shape
            ``(batch_size, seq_len, dim)``.
        """
        bsz, seq_len, dim = x.shape

        h = self.norm(x)
        qkv = self.qkv(h).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.mode == "decoder":
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, dim)
        out = self.out_proj(out)
        return out


class ODEAttentionBlock(nn.Module):
    """Continuous-depth attention block via ODE integration.

    Solves the initial value problem::

        z(1) = z(0) + integral_0^1 f(z(t), t) dt

    using an explicit numerical ODE solver (Euler or classical RK4). The
    number of integration steps is configurable. Weights in ``ODEFunc`` are
    shared across all steps, so the parameter count is independent of the
    number of solver steps.

    Reference:
        Zhang et al. (2021), "ODE Transformer", ACL 2022.

    Args:
        config: Model configuration object with attributes ``hidden_size``,
            ``num_heads``, ``dropout``, ``use_bitnet``, ``norm_type``,
            ``ode_solver`` (``"euler"`` or ``"rk4"``), ``ode_steps``, and
            optionally ``mode``.

    Attributes:
        ode_func: The ``ODEFunc`` module representing ``f(z, t)``.
        solver: Integration method (``"euler"`` or ``"rk4"``).
        steps: Number of integration steps from ``t=0`` to ``t=1``.
    """

    def __init__(self, config):
        super().__init__()
        self.ode_func = ODEFunc(config)
        self.solver = config.ode_solver
        self.steps = config.ode_steps

    def forward(self, x):
        """Integrate the ODE from ``t=0`` to ``t=1``.

        Args:
            x: Initial state tensor of shape
                ``(batch_size, seq_len, hidden_size)``.

        Returns:
            Final state ``z(1)`` of shape
            ``(batch_size, seq_len, hidden_size)``.
        """
        dt = 1.0 / self.steps
        t = 0.0
        z = x

        for _ in range(self.steps):
            if self.solver == "euler":
                dz = self.ode_func(t, z)
                z = z + dz * dt
            elif self.solver == "rk4":
                k1 = self.ode_func(t, z)
                k2 = self.ode_func(t + dt / 2, z + k1 * dt / 2)
                k3 = self.ode_func(t + dt / 2, z + k2 * dt / 2)
                k4 = self.ode_func(t + dt, z + k3 * dt)
                z = z + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6.0)
            t += dt

        return z
