"""ROOT optimizer: robust orthogonalized updates.

ROOT (He et al. 2025, arXiv:2511.20626) extends Muon with two robustness
mechanisms.  First, *AdaNewton* replaces the fixed Newton-Schulz coefficients
with per-matrix-shape calibrated triples, removing the precision loss that the
one-size-fits-all iteration exhibits on square and near-square shapes.
Second, *soft-thresholding* separates the momentum matrix into a sparse outlier
component ``O_t`` and a robust base component ``B_t = M_t - O_t`` and
orthogonalizes only ``B_t``, with the threshold set dynamically from the
percentile ``eps_t = Quantile(|M_t|, p)``.

As in Muon, the orthogonalized path is reserved for hidden-layer matrices;
embedding tables, learned positional encodings, output heads and every
non-matrix tensor receive decoupled AdamW updates instead of being skipped.

Reference:
    He, W., Han, K., Zhou, H., Chen, H., Liu, Z., Chen, X., & Wang, Y.
    (2025). ROOT: Robust Orthogonalized Optimizer for Neural Network
    Training. arXiv:2511.20626.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from .base import ADAMW_FALLBACK_GROUPS, adamw_fallback_update

_MUON_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
"""Fixed Muon coefficients, used as the fallback for uncalibrated shapes."""

_ADA_NEWTON_COEFFICIENTS = {
    (2048, 2048): (3.3334, -4.2591, 1.7791),
    (4096, 4096): (3.3732, -4.6134, 2.0576),
    (8192, 8192): (3.3886, -4.9026, 2.3105),
    (2048, 3072): (2.9091, -3.8108, 1.8600),
    (2048, 4096): (2.7739, -3.4911, 1.6992),
    (2048, 8192): (2.5925, -3.0010, 1.4138),
    (2048, 16384): (2.7045, -3.2335, 1.5336),
}
"""Per-shape Newton-Schulz coefficients from Table 1 of arXiv:2511.20626.

The table is the Mixed (1:3) calibration strategy.  Keys are the
shape-canonical pair ``(min(m, n), max(m, n))`` of the working matrix, i.e.
the shape seen after the rows<=cols transpose used by the iteration.
"""

_QUANTILE_ELEMENT_LIMIT = 16_777_216
"""Element cap below which ``torch.quantile`` is safe to call (2**24)."""


def _shape_coefficients(rows: int, cols: int, ada_newton: bool) -> tuple[float, float, float]:
    """Return the Newton-Schulz coefficients for a matrix shape.

    Args:
        rows: Number of rows of the working matrix (after the transpose).
        cols: Number of columns of the working matrix.
        ada_newton: When ``True``, look the shape up in the calibrated
            Table 1 dictionary; when ``False``, use Muon's fixed triples.

    Returns:
        The ``(a, b, c)`` coefficient triple.  Shapes outside the calibrated
        table fall back to Muon's fixed coefficients.
    """
    if not ada_newton:
        return _MUON_COEFFICIENTS
    key = (min(int(rows), int(cols)), max(int(rows), int(cols)))
    return _ADA_NEWTON_COEFFICIENTS.get(key, _MUON_COEFFICIENTS)


def _threshold_eps(tensor: torch.Tensor, quantile: float) -> torch.Tensor:
    """Compute the dynamic soft-threshold ``eps_t = Quantile(|M_t|, p)``.

    ``torch.quantile`` rejects tensors above ``2**24`` elements on some
    builds, and a wide projection can exceed that (e.g. ``2048 x 16384``).
    The ``kthvalue`` path is mathematically the same percentile and works for
    any size.

    Args:
        tensor: The momentum matrix ``M_t``.
        quantile: The percentile ``p`` in ``(0, 1]``.

    Returns:
        A scalar tensor holding the threshold value.
    """
    flat = tensor.abs().flatten()
    p = min(max(float(quantile), 0.0), 1.0)
    if flat.numel() <= _QUANTILE_ELEMENT_LIMIT:
        return torch.quantile(flat, p)
    k = max(1, min(flat.numel(), int(math.ceil(flat.numel() * p))))
    return flat.kthvalue(k).values


class ROOT(Optimizer):
    """ROOT optimizer with shape-calibrated orthogonalization.

    Accumulates momentum on matrix-shaped gradients, separates outliers from
    the momentum via soft-thresholding, and applies Newton-Schulz iterations
    with coefficients calibrated for the matrix shape.  Hidden-layer 2-D
    parameters take the orthogonalized path; parameters in the embedding,
    positional-encoding and head groups, as well as any parameter that is not
    a 2-D matrix, take a decoupled AdamW update using the group's ``betas``
    and ``eps``.

    Reference:
        He, W., Han, K., Zhou, H., Chen, H., Liu, Z., Chen, X., & Wang, Y.
        (2025). ROOT: Robust Orthogonalized Optimizer for Neural Network
        Training. arXiv:2511.20626.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate (default: 0.02).
        momentum: Momentum coefficient for the gradient buffer
            (default: 0.95).
        nesterov: Whether to use Nesterov-style momentum, where the gradient
            is combined with the momentum buffer before orthogonalization
            (default: True).
        ns_steps: Number of Newton-Schulz iterations for orthogonalization
            (default: 5).
        ns_eps: Epsilon added for numerical stability during normalization
            (default: 1e-7).
        quantile: Percentile ``p`` of ``|M_t|`` used for the dynamic
            soft-threshold (default: 0.90).
        softthresh: Whether to apply the soft-threshold outlier suppression
            (default: True).  Disabling it yields pure AdaNewton.
        ada_newton: Whether to use the shape-calibrated coefficients
            (default: True).  Disabling it restores Muon's fixed
            coefficients, which together with ``softthresh=True`` reproduces
            the paper's "ROOT (SoftThresh)" ablation.
        weight_decay: Decoupled weight decay coefficient (default: 0.0).
        betas: ``(beta1, beta2)`` coefficients of the AdamW fallback
            (default: ``(0.9, 0.95)``).
        eps: Epsilon of the AdamW fallback (default: 1e-8).

    Attributes:
        defaults (dict): Default hyper-parameter values for each parameter
            group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state (momentum buffer for
            matrix parameters; first/second moments for the AdamW fallback).
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        ns_eps=1e-7,
        quantile=0.90,
        softthresh=True,
        ada_newton=True,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
    ):
        """Initializes the ROOT optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default: 0.02).
            momentum: Momentum coefficient (default: 0.95).
            nesterov: Use Nesterov momentum (default: True).
            ns_steps: Newton-Schulz iteration count (default: 5).
            ns_eps: Normalization epsilon (default: 1e-7).
            quantile: Dynamic soft-threshold percentile (default: 0.90).
            softthresh: Enable soft-threshold outlier suppression
                (default: True).
            ada_newton: Use shape-calibrated coefficients (default: True).
            weight_decay: Decoupled weight decay coefficient (default: 0.0).
            betas: AdamW-fallback moment coefficients (default: ``(0.9, 0.95)``).
            eps: AdamW-fallback epsilon (default: 1e-8).
        """
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            quantile=quantile,
            softthresh=softthresh,
            ada_newton=ada_newton,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Implements Algorithm 2 of arXiv:2511.20626: momentum accumulation
        (Eq. 1), soft-threshold outlier separation (Eqs. 7-10), shape-
        calibrated Newton-Schulz orthogonalization (Eqs. 2-3), and the
        parameter update.

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

        for group in self.param_groups:
            use_adamw = str(group.get("name", "other")) in ADAMW_FALLBACK_GROUPS
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
                        lr=group["lr"],
                        betas=group.get("betas", (0.9, 0.95)),
                        eps=group.get("eps", 1e-8),
                        weight_decay=group["weight_decay"],
                    )
                    continue

                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]
                # Eq. (1): M_t = mu * M_{t-1} + G_t.
                buf.mul_(group["momentum"]).add_(grad)

                if group["nesterov"]:
                    g = grad.add(buf, alpha=group["momentum"])
                else:
                    g = buf

                # Eqs. (9)-(10): O_t = T_eps(M_t), B_t = M_t - O_t.  Only
                # the robust base component is orthogonalized.
                if group["softthresh"]:
                    eps_t = _threshold_eps(g, group["quantile"])
                    outlier = torch.sign(g) * (g.abs() - eps_t).clamp(min=0.0)
                    g = g - outlier

                if group["weight_decay"] > 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                x = g.bfloat16()
                # Eq. (2): X_0 = B_t / ||B_t||_F.
                x = x / (x.norm() + group["ns_eps"])

                transposed = False
                if x.size(0) > x.size(1):
                    x = x.T
                    transposed = True

                # Eq. (3): shape-calibrated quintic Newton-Schulz iteration.
                a, b, c = _shape_coefficients(
                    x.size(0), x.size(1), bool(group["ada_newton"])
                )

                for _ in range(int(group["ns_steps"])):
                    a_mat = x @ x.T
                    b_mat = b * a_mat + c * (a_mat @ a_mat)
                    x = a * x + b_mat @ x

                if transposed:
                    x = x.T

                p.add_(x.to(p.dtype), alpha=-group["lr"])

        return loss
