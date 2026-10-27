"""Numerically fixed softmax, log-softmax and cross-entropy.

Implements the drop-in replacements from Barron, Gnanapragasam & Tschannen
(2026), "A Fix for Softmax and Cross-Entropy", arXiv:2610.31642.

Standard implementations subtract the maximum logit before exponentiating
(Eq. 1 of the paper). That prevents overflow, but the shifted target/argmax
logit lands at exactly ``0`` while the remaining mass ``e^{-m}`` is added to
``1.0`` in the denominator. Once the logit margin

    m = z_target - logsumexp(z_non_target)

crosses the floating-point precision cliff (≈5.5 in bf16, 16.6 in fp32,
36.7 in fp64) the ``e^{-m}`` term is swallowed and the target probability
rounds to one. The forward loss then rounds to zero *and* the target gradient
``q_target - 1`` rounds to zero, while the non-target gradients stay positive:
the gradient no longer sums to zero, which produces the "Slingshot" loss
spikes reported for adaptive optimizers.

The fixes implemented here:

* :func:`fixed_log_softmax` / :func:`fixed_softmax` exclude the argmax logit
  from the log-sum-exp and combine the resulting margin with a numerically
  stable ``softplus``. The forward values are mathematically identical to
  ``torch.log_softmax`` / ``torch.softmax``; the backward pass differentiates
  the ``softplus`` graph, so the argmax entry of the softmax Jacobian keeps its
  true magnitude ``~e^{-m}`` past the cliff instead of rounding to zero.
* :func:`fixed_cross_entropy` uses the (hard) target logit as the reference,
  ``L = softplus(logsumexp(z - z_t))`` over the non-target logits. Because
  ``z_t`` enters every shifted logit with coefficient ``-1``, autograd returns
  a target gradient equal to minus the sum of the non-target gradients: the
  gradient sums to zero by construction, at any margin.

Soft-target (distillation / KL) cross-entropy is out of scope: this codebase
has no distillation loss path. :func:`module_softmax` and
:func:`module_log_softmax` dispatch on a per-module ``use_fixed_softmax``
attribute (attached centrally by ``HybridLayer``), mirroring the way SSMax
attaches itself to the positional encoder.

References:
    Barron, J. T., Gnanapragasam, D., & Tschannen, M. (2026). A Fix for
    Softmax and Cross-Entropy. arXiv:2610.31642.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _fixed_softmax_parts(logits: torch.Tensor, dim: int):
    """Shared forward quantities of the fixed softmax / log-softmax.

    Returns:
        Tuple ``(shifted, s, one_minus_qmax, mask)`` where ``shifted`` is
        ``logits - max``, ``s = softplus(-m) = -log q_max``, ``one_minus_qmax =
        1 - q_max`` computed through ``-expm1(-s)`` (accurate for tiny ``s``),
        and ``mask`` marks the (single) argmax position excluded from the
        log-sum-exp.
    """
    z_max = logits.max(dim=dim, keepdim=True).values
    argmax = logits.argmax(dim=dim, keepdim=True)
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(dim, argmax, True)

    shifted = logits - z_max
    lse_excl_shifted = torch.logsumexp(
        shifted.masked_fill(mask, float("-inf")), dim=dim, keepdim=True
    )
    s = F.softplus(lse_excl_shifted)
    one_minus_qmax = -torch.expm1(-s)
    return shifted, s, one_minus_qmax, mask


class _FixedLogSoftmax(torch.autograd.Function):
    """Log-softmax with the paper's accurate backward (Eq. 2-3).

    The value is ``shifted - softplus(-m)``; the custom vjp computes, at the
    argmax position, ``1 - q_max`` through ``-expm1(-s)`` instead of the
    cancellation-prone ``1 + (something ~ -1)`` of the autograd graph.
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, dim: int):
        shifted, s, one_minus_qmax, mask = _fixed_softmax_parts(logits, dim)
        log_q = shifted - s
        ctx.save_for_backward(log_q.exp(), one_minus_qmax, mask)
        ctx.dim = dim
        return log_q

    @staticmethod
    def backward(ctx, grad_output):
        q, one_minus_qmax, mask = ctx.saved_tensors
        dim = ctx.dim
        total = grad_output.sum(dim=dim, keepdim=True)
        sum_excl = grad_output.masked_fill(mask, 0).sum(dim=dim, keepdim=True)
        # Jacobian of log-softmax: dL/dz_j = g_j - q_j * sum(g). At the argmax
        # rewrite it as -sum_{j != max} g_j + (1 - q_max) * sum(g) so that the
        # nearly-equal g_max and sum(g) are never subtracted from each other.
        grad = grad_output - q * total
        grad = torch.where(mask, -sum_excl + one_minus_qmax * total, grad)
        return grad, None


class _FixedSoftmax(torch.autograd.Function):
    """Softmax with the paper's accurate backward (Eq. 2-3).

    Values are ``exp(fixed_log_softmax)`` (mathematically identical to
    ``torch.softmax``). The vjp ``q * (g - <g, q>)`` is rewritten at the argmax
    position as ``q_max * (g_max * (1 - q_max) - sum_{j != max} g_j q_j)``, so
    the true Jacobian magnitude ``~e^{-m}`` survives the precision cliff.
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, dim: int):
        shifted, s, one_minus_qmax, mask = _fixed_softmax_parts(logits, dim)
        q = (shifted - s).exp()
        ctx.save_for_backward(q, one_minus_qmax, mask)
        ctx.dim = dim
        return q

    @staticmethod
    def backward(ctx, grad_output):
        q, one_minus_qmax, mask = ctx.saved_tensors
        dim = ctx.dim
        gq = grad_output * q
        total = gq.sum(dim=dim, keepdim=True)
        sum_excl = gq.masked_fill(mask, 0).sum(dim=dim, keepdim=True)
        g_max = grad_output.masked_fill(~mask, 0)
        grad = q * (grad_output - total)
        grad = torch.where(
            mask, q * (g_max * one_minus_qmax - sum_excl), grad
        )
        return grad, None


def fixed_log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Cancellation-free log-softmax (arXiv:2610.31642 §2).

    The argmax logit is removed from the log-sum-exp, so the log-normalizer is
    ``z_max + softplus(-m)`` with ``m = z_max - logsumexp(z_excl)``. The
    ``softplus`` term is evaluated as ``log1p(exp(-m))``, which retains the
    tiny ``e^{-m}`` contribution arbitrarily far past the standard precision
    cliff (down to exponent underflow). The returned value at the argmax
    position is therefore ``-softplus(-m)`` rather than a rounded ``log(1)``,
    and the custom vjp keeps the argmax Jacobian entry accurate.

    Args:
        logits: Input tensor of arbitrary shape.
        dim: Dimension along which the log-softmax is computed.

    Returns:
        Tensor with the same shape and dtype as ``logits``. Mathematically
        equal to ``torch.log_softmax(logits, dim)``.
    """
    return _FixedLogSoftmax.apply(logits, dim)


def fixed_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Cancellation-free softmax (arXiv:2610.31642 §2).

    Forward values are mathematically identical to ``torch.softmax``; the vjp
    uses the paper's accurate expression for the argmax entry, so the softmax
    Jacobian keeps its true magnitude ``q_max * (1 - q_max) ~ e^{-m}`` past the
    precision cliff instead of rounding to zero.

    Args:
        logits: Input tensor of arbitrary shape.
        dim: Dimension along which the softmax is computed.

    Returns:
        Tensor with the same shape and dtype as ``logits``.
    """
    return _FixedSoftmax.apply(logits, dim)


def fixed_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> torch.Tensor:
    """Hard-target cross-entropy with a zero-sum gradient by construction.

    Implements arXiv:2610.31642 §2: the target logit ``z_t`` is used as the
    reference and the loss is ``softplus(logsumexp(z - z_t))`` over the
    non-target logits, i.e. ``log(1 + sum_{j != t} exp(z_j - z_t))``. Writing
    ``z_t`` through differences makes autograd return
    ``dL/dz_t = -sum_{j != t} dL/dz_j`` exactly, so the gradient sums to zero
    even when the margin exceeds the dtype's precision cliff. The target
    gradient magnitude ``e^{-m} / (1 + e^{-m})`` is preserved down to exponent
    underflow.

    ``ignore_index`` rows contribute zero loss and zero gradient (matching
    ``torch.nn.functional.cross_entropy``); the ``"mean"`` reduction divides
    by the number of valid rows.

    Args:
        logits: Logits of shape ``(..., vocab_size)``.
        targets: Integer targets of shape ``(...)``. ``ignore_index`` entries
            are excluded from the loss.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.
        ignore_index: Target value excluded from the loss.

    Returns:
        Scalar loss for ``"mean"``/``"sum"``, or a tensor shaped like
        ``targets`` for ``"none"``.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"Unsupported reduction '{reduction}'")

    vocab_size = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)

    valid = flat_targets != ignore_index
    safe_targets = flat_targets.clone()
    safe_targets[~valid] = 0

    z_t = flat_logits.gather(1, safe_targets.unsqueeze(1))
    shifted = flat_logits - z_t
    excl_mask = torch.zeros_like(shifted, dtype=torch.bool).scatter_(1, safe_targets.unsqueeze(1), True)
    lse_excl = torch.logsumexp(shifted.masked_fill(excl_mask, float("-inf")), dim=1)
    per_row = F.softplus(lse_excl)
    per_row = torch.where(valid, per_row, torch.zeros_like(per_row))

    if reduction == "none":
        return per_row.reshape(targets.shape)
    if reduction == "sum":
        return per_row.sum()
    denom = valid.sum().clamp(min=1).to(per_row.dtype)
    return per_row.sum() / denom


def module_softmax(module, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax dispatch honoring a module's ``use_fixed_softmax`` attribute.

    Args:
        module: Mixer/residual/router module that may carry
            ``use_fixed_softmax`` (set centrally by ``HybridLayer``).
        logits: Attention logits (or any tensor).
        dim: Dimension along which the softmax is computed.

    Returns:
        ``fixed_softmax`` when the module opted in, else ``F.softmax``.
    """
    if getattr(module, "use_fixed_softmax", False):
        return fixed_softmax(logits, dim=dim)
    return F.softmax(logits, dim=dim)


def module_log_softmax(module, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Log-softmax dispatch honoring a module's ``use_fixed_softmax`` attribute.

    Args:
        module: Module that may carry ``use_fixed_softmax``.
        logits: Input tensor.
        dim: Dimension along which the log-softmax is computed.

    Returns:
        ``fixed_log_softmax`` when the module opted in, else ``F.log_softmax``.
    """
    if getattr(module, "use_fixed_softmax", False):
        return fixed_log_softmax(logits, dim=dim)
    return F.log_softmax(logits, dim=dim)
