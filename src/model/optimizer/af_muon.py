"""AF-Muon: an AdamW-free Muon optimizer for tied-embedding models.

AF-Muon (Lagzian et al. 2026, arXiv:2610.01395) assigns every parameter class
the geometry of the role it plays instead of delegating non-matrix blocks to
AdamW:

* **hidden matrices** keep Muon's spectral update — bias-corrected momentum,
  a Nesterov signal, Newton-Schulz orthogonalization and the Moonlight
  ``sqrt(max(1, m/n))`` shape scale;
* **the vocabulary / embedding table** receives a support-aware finite-cap
  linear-minimization-oracle (LMO) update solved per row as
  ``sign(b_j) * min(c, tau * |b_j|)``, where ``tau`` is obtained by bisection
  so that each row reaches the smaller of the row-l2 radius ``sqrt(d)`` and
  the cap-limited support capacity ``c**2 * |supp(b)|``;
* **one-dimensional auxiliary tensors** receive an RMS-normalized update.

Only one first-moment buffer is stored per parameter, so no second-moment
state is allocated at all.

Role routing in this toolkit (parameter groups are produced by
:func:`src.model.optimizer.base.categorize_param_groups`):

* ``ndim <= 1`` (any group)              → vector RMS update (``vector_lr``);
* ``ndim == 2`` and group ``embeddings`` → tied-table clipped-row LMO;
* any other tensor with ``ndim >= 2``    → Muon matrix update.

Tied-table and vector parameters receive no weight decay, matching the
paper's matched protocol. ``tied_cap = 1`` reproduces the SCION-style Sign
endpoint and ``tied_cap >= sqrt(d)`` reproduces the row-RMS endpoint.

Reference:
    Lagzian, A., Halvachi, P., Zhang, J., Lin, Z., & Liu, D. (2026).
    AF-Muon: An AdamW-Free Muon Optimizer for Tied-Embedding Models.
    arXiv:2610.01395.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

__all__ = ["AFMuon"]


def _validate_momentum(beta: float) -> None:
    """Raise when the momentum coefficient is outside ``[0, 1)``.

    Args:
        beta: Momentum coefficient.

    Raises:
        ValueError: If ``beta`` is not in ``[0, 1)``.
    """
    if not 0.0 <= float(beta) < 1.0:
        raise ValueError(f"momentum must lie in [0, 1), got {beta}")


def _newton_schulz(signal: torch.Tensor, ns_steps: int, ns_eps: float) -> torch.Tensor:
    """Approximate the polar factor of a 2D matrix with Newton-Schulz steps.

    Args:
        signal: Two-dimensional momentum signal to orthogonalize.
        ns_steps: Number of Newton-Schulz iterations.
        ns_eps: Epsilon added to the Frobenius norm for numerical stability.

    Returns:
        A tensor with the same shape as ``signal`` (bfloat16 internally).
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = signal.bfloat16()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + ns_eps)
    for _ in range(int(ns_steps)):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return x


@torch.no_grad()
def _muon_matrix_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    *,
    beta: float,
    nesterov: bool,
    ns_steps: int,
    ns_eps: float,
) -> torch.Tensor:
    """Muon hidden-matrix update: EMA momentum + Nesterov signal + NS map.

    Args:
        grad: Gradient of the matrix parameter.
        momentum: Persistent first-moment buffer, updated in place.
        beta: Momentum coefficient.
        nesterov: Whether to combine the gradient with the momentum buffer.
        ns_steps: Number of Newton-Schulz iterations.
        ns_eps: Numerical stability epsilon for the NS normalization.

    Returns:
        The orthogonalized update, shaped like ``grad``.
    """
    momentum.lerp_(grad, 1.0 - beta)
    signal = grad.lerp(momentum, beta) if nesterov else momentum
    if signal.ndim > 2:
        signal = signal.reshape(signal.shape[0], -1)
    update = _newton_schulz(signal, ns_steps=ns_steps, ns_eps=ns_eps)
    update = update * math.sqrt(max(1.0, update.size(-2) / update.size(-1)))
    return update.reshape_as(grad)


@torch.no_grad()
def _vector_rms_update(
    param: torch.Tensor,
    grad: torch.Tensor,
    state: dict,
    *,
    beta: float,
    eps: float,
) -> torch.Tensor:
    """Unnormalized first momentum followed by the stabilized RMS oracle.

    Args:
        param: One-dimensional (or scalar) parameter being updated.
        grad: Gradient with the same shape as ``param``.
        state: Per-parameter optimizer state dictionary.
        beta: Momentum coefficient.
        eps: Positive floor for the RMS normalization.

    Returns:
        The RMS-normalized update direction in the parameter dtype.

    Raises:
        ValueError: If the shapes differ, the parameter is empty, or
            ``eps`` is not positive.
    """
    _validate_momentum(beta)
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if param.shape != grad.shape:
        raise ValueError(
            f"parameter/gradient shape mismatch: {tuple(param.shape)} vs {tuple(grad.shape)}"
        )
    if param.numel() == 0:
        raise ValueError("AF-Muon does not support empty vector parameters")

    buf = state.setdefault("momentum_buffer", torch.zeros_like(param))
    buf.mul_(beta).add_(grad)

    direction = buf.detach().float()
    rms = direction.square().mean().sqrt().clamp_min(float(eps))
    return (direction / rms).to(dtype=param.dtype)


def _row_rms_oracle_fp32(x: torch.Tensor) -> torch.Tensor:
    """Exact row-RMS direction for nonzero fp32 rows, with zero rows mapped to zero.

    Args:
        x: Two-dimensional momentum rows in fp32.

    Returns:
        A tensor with unit row RMS on every nonzero row and zeros elsewhere.
    """
    max_abs = x.abs().amax(dim=1, keepdim=True)
    nonzero = max_abs > 0
    safe_max = torch.where(nonzero, max_abs, torch.ones_like(max_abs))
    scaled = x / safe_max
    scaled_rms = scaled.square().mean(dim=1, keepdim=True).sqrt()
    safe_rms = torch.where(nonzero, scaled_rms, torch.ones_like(scaled_rms))
    return torch.where(nonzero, scaled / safe_rms, torch.zeros_like(scaled))


def _capped_row_sq(q: torch.Tensor, tau: torch.Tensor, cap: float) -> torch.Tensor:
    """Squared row norm of ``min(cap, tau * q)`` for each row of ``q``.

    Args:
        q: Non-negative row magnitudes (positive scale invariance applied).
        tau: Per-row scale (broadcastable column vector).
        cap: Entrywise magnitude cap ``c``.

    Returns:
        A ``(rows, 1)`` tensor with the squared row norms.
    """
    capped = torch.minimum(tau * q, torch.full_like(q, float(cap)))
    return capped.square().sum(dim=1, keepdim=True)


def _finite_cap_oracle_fp32(
    x: torch.Tensor,
    *,
    cap: float,
    bisection_steps: int,
    max_bracket_steps: int,
) -> torch.Tensor:
    """Support-aware finite-cap row oracle on an fp32 row chunk.

    For each row ``b`` this approximates ``sign(b_j) min(c, tau |b_j|)`` with
    the squared row target ``min(d, c**2 * |supp(b)|)``. Rows whose support
    cannot fill the row-l2 radius saturate every useful coordinate at the cap.
    Radius-limited rows use a dynamically bracketed scalar bisection; the
    returned iterate is the feasible lower endpoint, so finite solver
    tolerance cannot violate either the row radius or the coordinate cap.

    Args:
        x: Two-dimensional momentum rows in fp32.
        cap: Entrywise magnitude cap ``c`` with ``1 <= c < sqrt(d)``.
        bisection_steps: Number of bisection iterations per row.
        max_bracket_steps: Maximum bracket expansions before failing.

    Returns:
        The clipped-row LMO update for the chunk.

    Raises:
        RuntimeError: If a row cannot be bracketed because its momentum row
            exceeds the supported floating-point dynamic range.
    """
    rows, d = int(x.shape[0]), int(x.shape[1])
    if rows == 0:
        return torch.empty_like(x)

    abs_x = x.abs()
    signs = x.sign()

    # The oracle is invariant to positive row scaling. Normalizing by the
    # maximum magnitude avoids overflow/underflow in the root solve.
    max_abs = abs_x.amax(dim=1, keepdim=True)
    nonzero_rows = max_abs > 0
    safe_max = torch.where(nonzero_rows, max_abs, torch.ones_like(max_abs))
    q = abs_x / safe_max
    support = q > 0
    support_size = support.sum(dim=1, keepdim=True)

    cap_sq = float(cap) * float(cap)
    support_capacity = support_size.to(dtype=x.dtype) * cap_sq
    support_limited = (support_capacity <= float(d)).squeeze(1)
    radius_limited = ~support_limited

    result = torch.zeros_like(x)

    # This includes zero rows. Since sign(0) = 0, they remain exactly zero.
    if bool(support_limited.any().item()):
        result[support_limited] = signs[support_limited] * float(cap)

    if not bool(radius_limited.any().item()):
        return result

    q_radius = q[radius_limited]
    sign_radius = signs[radius_limited]
    row_norm_target = math.sqrt(float(d))

    # The uncapped row-RMS scale is a lower bracket: clipping can only reduce
    # its squared norm from d. Expansion continues until every row is bracketed
    # or fails explicitly rather than silently returning an infeasible update.
    q_norm = q_radius.square().sum(dim=1, keepdim=True).sqrt()
    lo = torch.zeros_like(q_norm)
    hi = row_norm_target / q_norm
    target_sq = float(d)

    row_sq = _capped_row_sq(q_radius, hi, cap)
    need_expansion = row_sq < target_sq
    for _ in range(int(max_bracket_steps)):
        if not bool(need_expansion.any().item()):
            break
        lo = torch.where(need_expansion, hi, lo)
        candidate = hi * 2.0
        if not bool(torch.isfinite(candidate[need_expansion]).all().item()):
            raise RuntimeError(
                "clipped-row LMO bracket overflowed; the momentum row has "
                "unsupported floating-point dynamic range"
            )
        hi = torch.where(need_expansion, candidate, hi)
        row_sq = _capped_row_sq(q_radius, hi, cap)
        need_expansion = row_sq < target_sq

    if bool(need_expansion.any().item()):
        count = int(need_expansion.sum().item())
        raise RuntimeError(
            "failed to bracket the clipped-row LMO root for "
            f"{count} row(s) after {max_bracket_steps} expansions"
        )

    for _ in range(int(bisection_steps)):
        mid = (lo + hi) * 0.5
        row_sq = _capped_row_sq(q_radius, mid, cap)
        below_target = row_sq < target_sq
        lo = torch.where(below_target, mid, lo)
        hi = torch.where(below_target, hi, mid)

    # lo is feasible by construction. It approaches the exact root from below
    # and preserves the KKT form without a post-clipping renormalization.
    magnitude = torch.minimum(lo * q_radius, torch.full_like(q_radius, float(cap)))
    result[radius_limited] = sign_radius * magnitude
    return result


@torch.no_grad()
def _tied_table_update(
    param: torch.Tensor,
    grad: torch.Tensor,
    state: dict,
    *,
    beta: float,
    cap: float,
    chunk_rows: int,
    bisection_steps: int,
    max_bracket_steps: int,
) -> torch.Tensor:
    """Support-aware, complete-row chunked tied-table clipped LMO.

    The persistent state is one parameter-dtype first-moment buffer. Solver
    intermediates are chunked by complete rows in fp32; the returned update is
    a full parameter-dtype tensor.

    Args:
        param: Two-dimensional vocabulary table parameter.
        grad: Merged gradient (input-lookup + output-softmax contributions).
        state: Per-parameter optimizer state dictionary.
        beta: Momentum coefficient.
        cap: Entrywise magnitude cap ``c >= 1`` (``inf`` selects row-RMS).
        chunk_rows: Number of complete rows processed per solver chunk.
        bisection_steps: Bisection iterations per row.
        max_bracket_steps: Maximum bracket expansions per row.

    Returns:
        The LMO update for the whole table.

    Raises:
        ValueError: On invalid momentum, cap, chunking or tensor shapes.
    """
    _validate_momentum(beta)
    if param.ndim != 2 or grad.ndim != 2:
        raise ValueError(
            "the AF-Muon tied-table oracle requires 2D parameter and gradient tensors"
        )
    if param.shape != grad.shape:
        raise ValueError(
            f"parameter/gradient shape mismatch: {tuple(param.shape)} vs {tuple(grad.shape)}"
        )
    if int(param.shape[1]) <= 0:
        raise ValueError("the tied table must have a positive model width")
    if int(chunk_rows) <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    if int(bisection_steps) <= 0:
        raise ValueError(f"bisection_steps must be positive, got {bisection_steps}")
    if int(max_bracket_steps) <= 0:
        raise ValueError(
            f"max_bracket_steps must be positive, got {max_bracket_steps}"
        )

    cap = float(cap)
    if math.isnan(cap) or cap < 1.0:
        raise ValueError(f"cap must satisfy c >= 1 or c = inf, got {cap}")

    buf = state.setdefault("momentum_buffer", torch.zeros_like(param))
    buf.mul_(beta).add_(grad)

    rows, d = int(param.shape[0]), int(param.shape[1])
    row_norm_target = math.sqrt(float(d))
    update = torch.empty_like(param)

    for start in range(0, rows, int(chunk_rows)):
        end = min(rows, start + int(chunk_rows))
        x = buf[start:end].detach().float()

        if cap == 1.0:
            direction = x.sign()
        elif math.isinf(cap) or cap >= row_norm_target:
            direction = _row_rms_oracle_fp32(x)
        else:
            direction = _finite_cap_oracle_fp32(
                x,
                cap=cap,
                bisection_steps=int(bisection_steps),
                max_bracket_steps=int(max_bracket_steps),
            )

        update[start:end].copy_(direction.to(dtype=param.dtype))

    return update


class AFMuon(Optimizer):
    """AF-Muon: AdamW-free Muon for models with a vocabulary-table group.

    Applies the Muon spectral update to hidden matrices, the support-aware
    finite-cap clipped-row LMO to two-dimensional parameters of the
    ``embeddings`` group (the vocabulary / token table), and RMS-normalized
    updates to one-dimensional auxiliary tensors. Only a single first-moment
    buffer is kept per parameter.

    Reference:
        Lagzian, A., Halvachi, P., Zhang, J., Lin, Z., & Liu, D. (2026).
        AF-Muon: An AdamW-Free Muon Optimizer for Tied-Embedding Models.
        arXiv:2610.01395.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups. Groups must carry a ``name`` from
            :data:`src.model.optimizer.base.GROUP_NAMES` for the tied-table
            role to activate.
        lr: Base learning rate applied to hidden matrices and used as the
            tied-table base (default: 0.02).
        momentum: Momentum coefficient for the first-moment buffer
            (default: 0.95).
        nesterov: Whether to use a Nesterov signal for hidden matrices
            (default: True).
        ns_steps: Number of Newton-Schulz iterations (default: 5).
        ns_eps: Numerical stability epsilon for Newton-Schulz
            (default: 1e-7).
        weight_decay: Default decoupled weight decay for hidden matrices
            (default: 0.0). Per-group ``weight_decay`` overrides it.
        vector_lr: Learning rate for one-dimensional auxiliary parameters
            (default: 3e-4).
        tied_cap: Entrywise magnitude cap ``c`` for the tied-table LMO;
            ``c = 1`` is Sign and ``c >= sqrt(d)`` is row-RMS
            (default: 3.0).
        tied_scale: Scale ``s`` in ``eta_tie = lr * (rho_output / rho_hidden)
            * s / d`` (default: 0.5).
        rho_hidden: Hidden-matrix radius used in the tied-table step size
            (default: 50.0).
        rho_output: Output radius used in the tied-table step size
            (default: 3000.0).
        chunk_rows: Complete rows processed per tied-table solver chunk
            (default: 2048).
        bisection_steps: Bisection iterations per tied-table row
            (default: 32).
        max_bracket_steps: Maximum bracket expansions per tied-table row
            (default: 128).
        eps: Default RMS-normalization floor for vector parameters
            (default: 1e-8). Per-group ``eps`` overrides it.

    Attributes:
        defaults (dict): Default hyper-parameter values for each parameter
            group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state (first-moment buffer).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_eps: float = 1e-7,
        weight_decay: float = 0.0,
        vector_lr: float = 3e-4,
        tied_cap: float = 3.0,
        tied_scale: float = 0.5,
        rho_hidden: float = 50.0,
        rho_output: float = 3000.0,
        chunk_rows: int = 2048,
        bisection_steps: int = 32,
        max_bracket_steps: int = 128,
        eps: float = 1e-8,
    ):
        """Initializes the AF-Muon optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Base learning rate (default: 0.02).
            momentum: Momentum coefficient (default: 0.95).
            nesterov: Nesterov signal for hidden matrices (default: True).
            ns_steps: Newton-Schulz iterations (default: 5).
            ns_eps: Newton-Schulz epsilon (default: 1e-7).
            weight_decay: Default hidden-matrix weight decay (default: 0.0).
            vector_lr: One-dimensional parameter learning rate (default: 3e-4).
            tied_cap: Tied-table entrywise cap (default: 3.0).
            tied_scale: Tied-table update scale (default: 0.5).
            rho_hidden: Hidden-matrix radius (default: 50.0).
            rho_output: Output radius (default: 3000.0).
            chunk_rows: Tied-table chunk size in rows (default: 2048).
            bisection_steps: Tied-table bisection iterations (default: 32).
            max_bracket_steps: Tied-table bracket expansions (default: 128).
            eps: Vector RMS floor (default: 1e-8).

        Raises:
            ValueError: If any hyper-parameter is out of its valid range.
        """
        _validate_momentum(momentum)
        if float(lr) < 0.0 or float(vector_lr) < 0.0:
            raise ValueError("learning rates must be nonnegative")
        if float(weight_decay) < 0.0:
            raise ValueError("weight_decay must be nonnegative")
        if float(rho_hidden) <= 0.0:
            raise ValueError(f"rho_hidden must be positive, got {rho_hidden}")
        if float(rho_output) < 0.0:
            raise ValueError(f"rho_output must be nonnegative, got {rho_output}")
        if float(tied_scale) < 0.0:
            raise ValueError(f"tied_scale must be nonnegative, got {tied_scale}")
        if int(chunk_rows) <= 0:
            raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
        if int(ns_steps) <= 0:
            raise ValueError(f"ns_steps must be positive, got {ns_steps}")
        if int(bisection_steps) <= 0:
            raise ValueError(f"bisection_steps must be positive, got {bisection_steps}")
        if int(max_bracket_steps) <= 0:
            raise ValueError(
                f"max_bracket_steps must be positive, got {max_bracket_steps}"
            )
        if float(eps) <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        tied_cap = float(tied_cap)
        if math.isnan(tied_cap) or tied_cap < 1.0:
            raise ValueError(f"tied_cap must satisfy c >= 1 or c = inf, got {tied_cap}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            weight_decay=weight_decay,
            vector_lr=vector_lr,
            tied_cap=tied_cap,
            tied_scale=tied_scale,
            rho_hidden=rho_hidden,
            rho_output=rho_output,
            chunk_rows=chunk_rows,
            bisection_steps=bisection_steps,
            max_bracket_steps=max_bracket_steps,
            eps=eps,
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

        for group in self.param_groups:
            group_name = str(group.get("name", "other")).lower()
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_steps = int(group["ns_steps"])
            ns_eps = float(group["ns_eps"])
            weight_decay = float(group.get("weight_decay", 0.0))
            vector_lr = float(group["vector_lr"])
            eps = float(group.get("eps", 1e-8))

            _validate_momentum(momentum)
            if lr < 0.0:
                raise ValueError(f"learning rate must be nonnegative, got {lr}")
            if weight_decay < 0.0:
                raise ValueError(
                    f"weight decay must be nonnegative, got {weight_decay}"
                )

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if p.ndim <= 1:
                    update = _vector_rms_update(
                        p, p.grad, state, beta=momentum, eps=eps
                    )
                    p.add_(update, alpha=-vector_lr)

                elif p.ndim == 2 and group_name == "embeddings":
                    d_model = int(p.shape[1])
                    eta_tie = (
                        lr
                        * float(group["rho_output"])
                        / float(group["rho_hidden"])
                        * float(group["tied_scale"])
                        / float(d_model)
                    )
                    update = _tied_table_update(
                        p,
                        p.grad,
                        state,
                        beta=momentum,
                        cap=float(group["tied_cap"]),
                        chunk_rows=int(group["chunk_rows"]),
                        bisection_steps=int(group["bisection_steps"]),
                        max_bracket_steps=int(group["max_bracket_steps"]),
                    )
                    p.add_(update, alpha=-eta_tie)

                else:
                    buf = state.setdefault("momentum_buffer", torch.zeros_like(p))
                    update = _muon_matrix_update(
                        p.grad,
                        buf,
                        beta=momentum,
                        nesterov=nesterov,
                        ns_steps=ns_steps,
                        ns_eps=ns_eps,
                    )
                    if weight_decay > 0:
                        p.mul_(1.0 - lr * weight_decay)
                    p.add_(update, alpha=-lr)

        return loss
