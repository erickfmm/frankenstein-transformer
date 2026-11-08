"""NAMO-D optimizer: diagonal extension of NAMO with neuron-wise adaptation.

NAMO-D (Zhang, Liu & Schaeffer 2026, arXiv:2602.17080, Algorithm 2) replaces
NAMO's single adaptive scalar with a diagonal matrix that right-multiplies the
orthogonalized momentum.  A per-column (per-neuron) second-moment EMA of the
gradient yields the neuron-wise stepsizes

    d_t = sqrt(1 - mu_2^t) / (1 - mu_1^t) * N^c(M_t) / (sqrt(v_t) + eps),

where the operator ``N^c(M)`` returns the Euclidean norm of each column of
``M``.  The entries are clamped toward their average,

    d_tilde_t = min(max(d_t, c * d_bar_t * 1), d_bar_t / c * 1),
    d_bar_t = ||d_t||_1 / n,

so the diagonal stays well conditioned (its condition number is at most
``1/c^2``); the parameter update is then

    Theta_t = Theta_{t-1} - eta * (O_t + lambda * Theta_{t-1}) * diag(d_tilde_t).

Column-wise scaling enables finer-grained noise adaptation aligned with the
near block-diagonal Hessian of neural networks, at the price of no longer
preserving strict orthogonality of the update direction.  The additional
state over NAMO is one vector of length ``n`` per matrix (the paper's "no
longer free" overhead; NAMO itself needs only a scalar).

Following the Muon hybrid recipe, only hidden-layer 2-D matrices take the
NAMO-D path; embedding tables, learned positional encodings, output heads and
every non-matrix parameter (biases, norm scales, convolution kernels) receive
decoupled AdamW updates configured by the shared per-group ``betas``/``eps``
suffixes.

Reference:
    Zhang, M., Liu, Y., & Schaeffer, H. (2026). Adam Improves Muon: Adaptive
    Moment Estimation with Orthogonalized Momentum. arXiv:2602.17080.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from .base import ADAMW_FALLBACK_GROUPS, adamw_fallback_update

_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
"""Quintic Newton-Schulz coefficients shared with Muon."""


def _newton_schulz(
    matrix: torch.Tensor,
    steps: int,
    eps: float,
    coefficients: tuple[float, float, float] = _COEFFICIENTS,
) -> torch.Tensor:
    """Approximate the orthogonal polar factor of a matrix.

    Implements Algorithm 2, line 8 of the NAMO-D paper (the same
    Newton-Schulz routine used by Muon): the input is normalized by its
    Frobenius norm, transposed so that ``rows <= cols`` when necessary, and
    refined with ``steps`` quintic iterations before being transposed back.
    Iterations run in bfloat16, matching the other Muon-family modules in this
    package.

    Args:
        matrix: A 2-D tensor whose orthogonal factor is sought.
        steps: Number of Newton-Schulz iterations (the paper uses five).
        eps: Numerical-stability term of the Frobenius normalization.
        coefficients: Quintic polynomial ``(a, b, c)``; defaults to the Muon
            triple.

    Returns:
        The orthogonalized matrix in bfloat16, in the input orientation.
    """
    x = matrix.bfloat16()
    x = x / (x.norm() + eps)

    transposed = False
    if x.size(0) > x.size(1):
        x = x.T
        transposed = True

    a, b, c = coefficients
    for _ in range(int(steps)):
        gram = x @ x.T
        polynomial = b * gram + c * (gram @ gram)
        x = a * x + polynomial @ x

    if transposed:
        x = x.T
    return x


class NAMO_D(Optimizer):
    r"""NAMO-D optimizer with neuron-wise adaptive orthogonal updates.

    Accumulates momentum on matrix-shaped gradients, orthogonalizes it with
    Newton-Schulz iterations, and right-multiplies the orthogonalized
    direction by a clamped diagonal matrix whose entries are neuron-wise
    adaptive stepsizes:

    .. math::

        M_t &= \mu_1 M_{t-1} + (1-\mu_1) G_t \\
        [v_t]_j &= \mu_2 [v_{t-1}]_j + (1-\mu_2) \lVert [G_t]_{:j} \rVert^2 \\
        d_t &= \frac{\sqrt{1-\mu_2^t}}{1-\mu_1^t}\,
               N^c(M_t) \oslash \bigl(\sqrt{v_t} + \epsilon\bigr) \\
        \bar d_t &= \lVert d_t \rVert_1 / n, \qquad
        \tilde d_t = \min\{\max\{d_t, c\,\bar d_t \mathbf 1\},\,
                           \bar d_t/c\,\mathbf 1\} \\
        O_t &= \mathrm{NS}_K(M_t) \\
        \Theta_t &= \Theta_{t-1} - \eta\,
                    (O_t + \lambda \Theta_{t-1})\,\operatorname{diag}(\tilde d_t)

    Hidden-layer 2-D parameters take this path; parameters in the embedding,
    positional-encoding and head groups, as well as any parameter that is not a
    2-D matrix, take a decoupled AdamW update using the group's ``betas`` and
    ``eps``.

    Reference:
        Zhang, M., Liu, Y., & Schaeffer, H. (2026). Adam Improves Muon:
        Adaptive Moment Estimation with Orthogonalized Momentum.
        arXiv:2602.17080.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate (default: 1e-3).  NAMO-D is not RMS-aligned, so the
            paper's optimal learning rates for hidden matrices are roughly an
            order of magnitude above AdamW's.
        betas: ``(mu_1, mu_2)`` momenta: the decay of the momentum buffer and
            of the column-wise norm EMA (default: ``(0.95, 0.99)``, the
            paper's GPT-2 values).
        ns_steps: Number of Newton-Schulz iterations (default: 5).
        ns_eps: Epsilon added during the Frobenius normalization before
            Newton-Schulz (default: 1e-7).
        clamp: Clamping constant ``c`` in ``(0, 1]`` of the diagonal stepsize
            (default: 0.1, the paper's GPT-2 124M value; the 355M sweep selects
            0.9 and the reference implementation defaults to 0.75).  Smaller
            values clamp harder toward the average stepsize.
        weight_decay: Decoupled weight decay coefficient (default: 0.0),
            right-multiplied by the diagonal exactly as in the paper's update.

    Attributes:
        defaults (dict): Default hyper-parameter values for each parameter
            group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state (momentum buffer,
            column-wise second moment and step counter for matrix parameters;
            first and second moments for the AdamW fallback).
        betas (tuple): The ``(mu_1, mu_2)`` momenta.  Stored as an instance
            attribute rather than in ``defaults`` because the factory injects
            per-group ``betas`` keys for the AdamW fallback, which would
            otherwise shadow the global momenta.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.95, 0.99),
        ns_steps: int = 5,
        ns_eps: float = 1e-7,
        clamp: float = 0.1,
        weight_decay: float = 0.0,
    ) -> None:
        """Initializes the NAMO-D optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default: 1e-3).
            betas: ``(mu_1, mu_2)`` momentum decays (default: ``(0.95, 0.99)``).
            ns_steps: Newton-Schulz iteration count (default: 5).
            ns_eps: Frobenius-normalization epsilon (default: 1e-7).
            clamp: Clamping constant ``c`` of the diagonal stepsize; must lie
                in ``(0, 1]`` (default: 0.1).
            weight_decay: Decoupled weight decay coefficient (default: 0.0).

        Raises:
            ValueError: If ``lr``, ``betas[i]``, ``ns_steps``, ``ns_eps``,
                ``clamp`` or ``weight_decay`` is out of its valid range.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 1 <= ns_steps:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")
        if not 0.0 <= ns_eps:
            raise ValueError(f"Invalid ns_eps value: {ns_eps}")
        if not 0.0 < clamp <= 1.0:
            raise ValueError(f"Invalid clamp value (must be in (0, 1]): {clamp}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.betas = (float(betas[0]), float(betas[1]))
        defaults = dict(
            lr=lr,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            clamp=clamp,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure: A closure that re-evaluates the model and returns the
                loss.  Optional for most use cases.

        Returns:
            The loss value if ``closure`` is provided, otherwise ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        mu1, mu2 = self.betas
        for group in self.param_groups:
            use_adamw = str(group.get("name", "other")) in ADAMW_FALLBACK_GROUPS
            eps = float(group.get("eps", 1e-8))
            lr = float(group["lr"])
            clamp = float(group.get("clamp", 0.1))
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Paper hybrid recipe: embeddings, positional encodings,
                # output heads and every non-matrix tensor take AdamW; only
                # hidden-layer 2-D matrices are orthogonalized.
                if use_adamw or p.ndim != 2:
                    adamw_fallback_update(
                        p,
                        p.grad,
                        self.state[p],
                        lr=lr,
                        betas=group.get("betas", (0.9, 0.95)),
                        eps=eps,
                        weight_decay=group["weight_decay"],
                    )
                    continue

                if p.numel() == 0:
                    continue

                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                    state["column_second_moment"] = torch.zeros(
                        grad.size(1), dtype=torch.float32, device=grad.device
                    )
                    state["step"] = 0
                state["step"] += 1
                t = state["step"]

                # Algorithm 2, line 4: M_t = mu_1 * M_{t-1} + (1 - mu_1) * G_t.
                # The (1 - mu_1) factor is kept: d_t reads N^c(M_t) and the
                # bias corrections directly, so the constant scale matters.
                buffer = state["momentum_buffer"]
                buffer.mul_(mu1).add_(grad, alpha=1.0 - mu1)

                # Algorithm 2, line 5:
                #   v_t = mu_2 * v_{t-1}
                #         + (1 - mu_2) * N^c(G_t) * N^c(G_t),
                # where [N^c(M)]_j = ||M_{:,j}|| is the column-norm operator.
                second = state["column_second_moment"]
                col_norms_grad = grad.norm(dim=0)
                second.mul_(mu2).add_(
                    col_norms_grad.square().to(second.dtype), alpha=1.0 - mu2
                )

                # Algorithm 2, lines 6-7:
                #   d_t = sqrt(1 - mu_2^t) / (1 - mu_1^t)
                #         * N^c(M_t) / (sqrt(v_t) + eps),
                #   d_bar_t = ||d_t||_1 / n.
                bias_correction1 = 1.0 - mu1**t
                bias_correction2 = 1.0 - mu2**t
                col_norms_momentum = buffer.norm(dim=0)
                d = (
                    math.sqrt(bias_correction2)
                    / bias_correction1
                    * col_norms_momentum
                    / (second.sqrt() + eps)
                )
                d_bar = float(d.sum()) / d.numel()

                # Algorithm 2, line 9: clamp toward the average,
                #   d_tilde = min(max(d_t, c * d_bar * 1), d_bar / c * 1).
                d_tilde = d.clamp(min=clamp * d_bar, max=d_bar / clamp)

                # Algorithm 2, line 8: O_t = Orth(M_t) ~ NS_K(M_t).
                orth = _newton_schulz(
                    buffer, int(group["ns_steps"]), float(group["ns_eps"])
                )

                # Paper section 4.1: the decoupled weight decay is also
                # right-multiplied by the diagonal:
                #   Theta_t = Theta_{t-1} - eta * (O_t + lambda * Theta_{t-1}) * diag(d_tilde).
                if group["weight_decay"] > 0:
                    p.mul_(
                        1.0
                        - lr
                        * float(group["weight_decay"])
                        * d_tilde.to(p.dtype).unsqueeze(0)
                    )

                # Algorithm 2, line 10: Theta_t = Theta_{t-1} - eta * O_t * D_t.
                scaled = orth.to(p.dtype) * d_tilde.to(p.dtype).unsqueeze(0)
                p.add_(scaled, alpha=-lr)

        return loss
