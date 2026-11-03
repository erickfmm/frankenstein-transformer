"""Dion3 optimizer with fractional Gram-Newton-Schulz orthogonalization.

Dion3 (Amsel et al. 2026, arXiv:2608.11612) is a revision of Muon that reduces the
cost of the orthogonalization step.  It replaces standard Newton-Schulz with *Gram
Newton-Schulz*, an iteration on the small symmetric Gram matrix ``X X^T`` that is
mathematically equivalent to the standard iteration, and it orthogonalizes only a
fraction ``f`` of the momentum rows at each step: the ``k = ceil(f * n)`` rows with
the largest l1 norm.  Only those rows of the weight matrix are updated, and *error
feedback* decays only the selected rows of the momentum buffer so that ignored rows
accumulate and are eventually selected.  With ``f = 1`` the update reduces to Muon
(or to NorMuon when the optional row-wise normalization is enabled).

Reference:
    Amsel, N., Zhang, J., Ahn, K., Naeimi, A., Feng, A., Chen, B., Dao, T., &
    Langford, J. (2026). Dion3: Full-stack orthogonal updates. arXiv:2608.11612.
"""

from __future__ import annotations

import math

import torch

from torch.optim import Optimizer


# Polar Express coefficients for five Newton-Schulz iterations (Amsel et al.,
# "The Polar Express", ICLR 2026), as shipped by the reference
# ``gram-newton-schulz`` package.  Each triple ``(a, b, c)`` defines the odd
# polynomial ``p(x) = a x + b x^3 + c x^5``, i.e. ``h(y) = a + b y + c y^2`` with
# ``y = x^2``.  The five-step table is only used when ``ns_steps == 5``; other
# step counts fall back to the constant Muon triple, which is a valid (if slower
# to converge) Newton-Schulz polynomial.
_POLAR_EXPRESS_COEFFICIENTS: tuple[tuple[float, float, float], ...] = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)
_MUON_COEFFICIENT: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
_NS_STEPS_WITH_POLAR_EXPRESS = 5
# The paper restarts Gram Newton-Schulz after the second iteration: resets the
# spurious negative eigenvalues that half precision introduces in the Gram matrix.
_RESTART_AFTER_ITERATION = 2


def _resolve_coefficients(
    ns_steps: int,
) -> tuple[tuple[float, float, float], ...]:
    """Return one ``(a, b, c)`` triple per Newton-Schulz iteration.

    Args:
        ns_steps: Number of Gram Newton-Schulz iterations.

    Returns:
        A tuple of ``(a, b, c)`` polynomial coefficients.  The Polar Express
        table is used for the canonical five-step schedule; every other step
        count reuses the constant Muon triple, which defines a valid
        Newton-Schulz iteration for singular values in ``[0, 1]``.
    """
    if ns_steps == _NS_STEPS_WITH_POLAR_EXPRESS:
        return _POLAR_EXPRESS_COEFFICIENTS
    return tuple(_MUON_COEFFICIENT for _ in range(ns_steps))


def _gram_newton_schulz(
    matrix: torch.Tensor,
    coefficients: tuple[tuple[float, float, float], ...],
    eps: float,
) -> torch.Tensor:
    """Orthogonalize a matrix with stabilized Gram Newton-Schulz.

    Implements Algorithm 3 of the Dion3 paper (arXiv:2608.11612): the iteration
    runs on the symmetric Gram matrix ``X X^T`` instead of on ``X`` itself, which
    is mathematically equivalent to standard Newton-Schulz but cheaper.  A
    restart after iteration 2 resets the spurious negative eigenvalues produced
    by half-precision arithmetic.

    Args:
        matrix: A 2-D tensor with ``rows <= cols`` whose polar factor is sought.
        coefficients: One ``(a, b, c)`` triple per iteration.
        eps: Frobenius-normalization epsilon (the paper uses ``1e-7``).

    Returns:
        The orthogonalized matrix, in the input dtype.
    """
    if matrix.numel() == 0:
        return matrix

    dtype = matrix.dtype
    x = (matrix / (matrix.norm() + eps)).to(torch.float16)
    rows = x.shape[0]
    q = torch.eye(rows, dtype=x.dtype, device=x.device)
    r = x @ x.transpose(0, 1)

    restart_at = _RESTART_AFTER_ITERATION + 1
    for step, (a, b, c) in enumerate(coefficients, start=1):
        if step == restart_at:
            # Restart: re-form the Gram matrix from the current iterate.
            x = q @ x
            r = x @ x.transpose(0, 1)
            q = torch.eye(rows, dtype=x.dtype, device=x.device)

        z = b * r + c * (r @ r)
        q_next = q @ z + a * q
        rz = r @ z + a * r
        r = z @ rz + a * rz
        q = q_next

    return (q @ x).to(dtype)


class Dion3(Optimizer):
    r"""Dion3: fractional orthogonalized updates with error feedback.

    Implements the Dion3 update rule (Algorithm 4 of arXiv:2608.11612) for
    matrix-shaped parameters.  Gradients are accumulated into a momentum buffer;
    the ``k = ceil(fraction * n)`` rows with the largest l1 norm are selected,
    orthogonalized with Gram Newton-Schulz, and used to update the matching rows
    of the parameter.  Only the selected momentum rows are decayed by
    ``momentum`` (error feedback), so rows that were not selected keep
    accumulating.  When ``normuon`` is enabled, a row-wise second moment of the
    orthogonalized update rescales it before the Frobenius-norm renormalization
    (NorMuon flavor).

    Parameters with fewer than two dimensions (biases, norm scales) are left
    unchanged, matching the other orthogonalized optimizers in this package.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter
            groups.
        lr: Learning rate (default: 0.02).
        fraction: Fraction of the smaller dimension to select at each step,
            ``f in (0, 1]`` (default: 0.25).  ``f = 1`` reduces to Muon.
        momentum: Momentum coefficient used for the selected rows' error
            feedback (default: 0.95).
        ns_steps: Number of Gram Newton-Schulz iterations (default: 5).
        ns_eps: Frobenius-normalization epsilon (default: 1e-7).
        normuon: Enable the NorMuon flavor with row-wise second-moment
            normalization and Frobenius rescaling (default: False).
        betas: Coefficients for the first/second moment estimates, per group.
            The second coefficient is the decay of the NorMuon row-wise second
            moment (default: ``(0.9, 0.95)``).
        eps: Numerical-stability term, also used by the NorMuon row
            normalization (default: 1e-8).
        weight_decay: Decoupled weight decay applied to every row before the
            selected-row update (default: 0.0).

    Attributes:
        defaults (dict): Default hyper-parameter values for each parameter
            group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state (momentum buffer and, when
            ``normuon`` is enabled, the row-wise second moment).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        fraction: float = 0.25,
        momentum: float = 0.95,
        ns_steps: int = 5,
        ns_eps: float = 1e-7,
        normuon: bool = False,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """Initializes the Dion3 optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default: 0.02).
            fraction: Row-selection fraction in ``(0, 1]`` (default: 0.25).
            momentum: Momentum coefficient / error-feedback decay
                (default: 0.95).
            ns_steps: Gram Newton-Schulz iteration count (default: 5).
            ns_eps: Frobenius-normalization epsilon (default: 1e-7).
            normuon: Enable the NorMuon row-normalization flavor
                (default: False).
            betas: First/second moment coefficients; the second is the NorMuon
                row-moment decay (default: ``(0.9, 0.95)``).
            eps: Numerical-stability epsilon (default: 1e-8).
            weight_decay: Decoupled weight decay (default: 0.0).

        Raises:
            ValueError: If any hyperparameter is outside its valid range.
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"Invalid fraction: {fraction} (expected (0, 1])")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if int(ns_steps) < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps} (expected >= 1)")
        if not 0.0 < ns_eps:
            raise ValueError(f"Invalid ns_eps: {ns_eps}")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(
            lr=lr,
            fraction=fraction,
            momentum=momentum,
            ns_steps=int(ns_steps),
            ns_eps=ns_eps,
            normuon=bool(normuon),
            betas=betas,
            eps=eps,
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

        Raises:
            RuntimeError: If any parameter has sparse gradients.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            fraction = float(group["fraction"])
            momentum_coefficient = float(group["momentum"])
            ns_steps = int(group["ns_steps"])
            ns_eps = float(group["ns_eps"])
            normuon = bool(group["normuon"])
            beta2 = float(group["betas"][1])
            eps = float(group["eps"])
            coefficients = _resolve_coefficients(ns_steps)

            for p in group["params"]:
                if p.grad is None or p.ndim < 2 or p.numel() == 0:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Dion3 does not support sparse gradients")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                if normuon and "row_second_moment" not in state:
                    state["row_second_moment"] = torch.zeros(
                        min(p.shape), device=p.device, dtype=torch.float32
                    )

                momentum = state["momentum_buffer"]
                momentum.add_(grad)

                # Algorithm 4, Selection: operate along the smaller dimension
                # (the paper selects rows when unsharded and picks the smaller
                # dimension).  ``work`` has shape (rows, cols), rows <= cols.
                transposed = momentum.size(0) > momentum.size(1)
                work = momentum.transpose(0, 1) if transposed else momentum
                rows = work.size(0)
                k = min(rows, max(1, math.ceil(fraction * rows)))
                if k < rows:
                    selected = torch.topk(
                        work.abs().sum(dim=1), k, largest=True, sorted=False
                    ).indices
                    submatrix = work[selected]
                else:
                    selected = None
                    submatrix = work

                # Algorithm 4, Orthogonalization (Gram Newton-Schulz).
                orth = _gram_newton_schulz(submatrix, coefficients, ns_eps)

                # Algorithm 4, optional NorMuon steps (float32 normalization).
                if normuon:
                    second_moment = state["row_second_moment"]
                    orth_f32 = orth.to(torch.float32)
                    row_moment = orth_f32.square().mean(dim=1)
                    if selected is None:
                        second_moment.mul_(beta2).add_(row_moment, alpha=1.0 - beta2)
                        scale_rows = second_moment
                    else:
                        updated = beta2 * second_moment[selected] + (
                            1.0 - beta2
                        ) * row_moment
                        second_moment[selected] = updated
                        scale_rows = updated
                    norm_before = orth_f32.norm()
                    orth_hat = orth_f32 / (scale_rows.sqrt().unsqueeze(1) + eps)
                    orth_hat = orth_hat * (norm_before / (orth_hat.norm() + 1e-12))
                    orth = orth_hat.to(orth.dtype)

                # Algorithm 4, Weight decay: applied to every row before the
                # selected-row update.
                if group["weight_decay"] > 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                # Algorithm 4, Update: only the selected rows (all rows at f=1).
                update = orth.to(p.dtype)
                if transposed:
                    if selected is None:
                        p.sub_(group["lr"] * update.transpose(0, 1))
                    else:
                        p[:, selected] = p[:, selected] - group["lr"] * update.transpose(0, 1)
                else:
                    if selected is None:
                        p.sub_(group["lr"] * update)
                    else:
                        p[selected, :] = p[selected, :] - group["lr"] * update

                # Algorithm 4, Error feedback: decay only the selected rows.
                if selected is not None:
                    if transposed:
                        momentum[:, selected] *= momentum_coefficient
                    else:
                        momentum[selected, :] *= momentum_coefficient

        return loss
