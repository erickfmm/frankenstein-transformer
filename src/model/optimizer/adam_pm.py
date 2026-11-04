"""AdamPM optimizer: Adam with pooled second moments.

AdamPM (Ji et al. 2026, arXiv:2610.24598) is a minimal, model-agnostic
variant of Adam that compresses the second-moment estimate by grouping
adjacent entries along the row (default) or column dimension and
representing each group by the mean of its squared gradients.  Because
the group mean is taken *before* the exponential moving average, only
the compressed statistic is persisted, reducing second-moment state
memory to ``1/s`` of Adam's requirement (``mn(1 + 1/s)`` optimizer-state
elements instead of ``2mn``).  The first moment stays element-wise, so
Adam's hyperparameters (learning rate, betas, epsilon) transfer
directly.

Weight decay is not defined by the paper's Algorithm 1; this
implementation follows the repository convention and applies decoupled
AdamW-style decay through the shared ``wd_*`` suffixes.

Reference:
    Ji, X., Xu, L., An, J., Luo, J., Wu, W., & Huang, L. (2026). AdamPM:
    Memory-Efficient LLM Optimization via Moment Pooling.
    *arXiv:2610.24598*. https://arxiv.org/abs/2610.24598
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.optim import Optimizer


def _pool_matrix(param: Tensor, pool_dim: str) -> Tensor:
    """Return the 2-D view whose last axis is the pooling axis.

    Row pooling partitions each row (the last dimension) into groups of
    adjacent entries; column pooling partitions each column instead, so
    the matrix is transposed to make the column axis last.  Scalars and
    1-D tensors are treated as single-row or single-column matrices,
    matching the paper's rule that "one-dimensional parameters ... are
    pooled as single-row or single-column matrices".

    Args:
        param: The tensor whose squared gradient is pooled (a parameter
            or a squared gradient).
        pool_dim: Pooling direction, ``"row"`` or ``"col"``.

    Returns:
        A 2-D tensor ``(outer, axis_length)`` whose last axis is the
        dimension partitioned into groups.
    """
    if param.dim() == 0:
        return param.reshape(1, 1)
    if pool_dim == "row":
        return param.reshape(-1, int(param.shape[-1]))
    return param.reshape(int(param.shape[0]), -1).transpose(0, 1)


def _group_mean(squared_grad: Tensor, group_size: int) -> Tensor:
    """Average every ``group_size`` adjacent entries along the last axis.

    Implements the paper's padding rule: "when ``n`` is not divisible by
    ``s``, the final group of squared gradients is padded with its mean",
    which is equivalent to averaging over the entries actually present in
    the final group (a partial mean, not a sum divided by ``s``).

    Args:
        squared_grad: 2-D tensor ``(outer, axis_length)`` of squared
            gradients.
        group_size: Number of adjacent entries per group (``s``).

    Returns:
        A tensor ``(outer, ceil(axis_length / group_size))`` of group
        means.
    """
    outer, axis_length = squared_grad.shape
    n_groups = math.ceil(axis_length / group_size)
    padded_length = n_groups * group_size
    if padded_length != axis_length:
        squared_grad = torch.nn.functional.pad(squared_grad, (0, padded_length - axis_length))
    grouped = squared_grad.reshape(outer, n_groups, group_size)
    counts = torch.full(
        (n_groups,),
        float(group_size),
        dtype=grouped.dtype,
        device=grouped.device,
    )
    counts[-1] = float(axis_length - (n_groups - 1) * group_size)
    return grouped.sum(dim=-1) / counts


def _broadcast_groups(pooled: Tensor, group_size: int, axis_length: int) -> Tensor:
    """Replicate each group statistic across its member entries.

    Corresponds to the Kronecker product with ``1_{1 x s}`` in the paper's
    Algorithm 1, truncated back to the original axis length so incomplete
    final groups broadcast only over the entries they own.

    Args:
        pooled: 2-D tensor ``(outer, n_groups)`` of group statistics.
        group_size: Number of adjacent entries per group (``s``).
        axis_length: Length of the original pooling axis.

    Returns:
        A tensor ``(outer, axis_length)`` with each group statistic
        repeated over its entries.
    """
    return pooled.repeat_interleave(group_size, dim=-1)[:, :axis_length]


class AdamPM(Optimizer):
    r"""Adam with pooled second moments (AdamPM).

    Groups every :math:`s` adjacent second-moment entries within each row
    (default) or column and keeps a single shared mean per group.  The
    compressed statistic is bias-corrected and broadcast back over its
    group before the parameter update, so the update rule mirrors Adam
    while the persistent second-moment buffer shrinks to ``1/s`` of the
    parameter size.  With ``group_size=1`` the optimizer is exactly Adam
    (plus the repository's decoupled weight decay); with
    ``group_size`` greater than or equal to the pooling-axis length every
    row (or column) shares one scalar, the paper's ``AdamPM-inf`` limit.

    State buffers (per parameter):
        ``step``: Update counter (scalar).
        ``exp_avg``: First-moment estimate :math:`m_t` (full parameter
            shape, ``mn`` elements).
        ``exp_avg_pooled``: Compressed second-moment estimate
            :math:`\bar v_t` (``m`` x ``ceil(n/s)`` for row pooling,
            ``ceil(m/s)`` x ``n`` for column pooling).

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate (default: ``1e-3``).
        betas: Coefficients used for computing the first and second
            moment running averages (default: ``(0.9, 0.95)``, matching
            the paper's pre-training settings).
        eps: Term added to the denominator for numerical stability
            (default: ``1e-8``).
        weight_decay: Decoupled weight-decay coefficient (default:
            ``0.0``; the paper's Algorithm 1 defines no weight decay).
        group_size: Number of adjacent second-moment entries pooled into
            one statistic (default: ``8``; the paper reports robust
            behaviour for group sizes between 2 and 8).
        pool_dim: Pooling direction, ``"row"`` (within each row, the
            paper's default) or ``"col"`` (within each column, the
            paper's ``AdamPM-col`` variant).

    Attributes:
        defaults (dict): Default hyper-parameter values for each
            parameter group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state (step counter, first
            moment, compressed second moment).

    Reference:
        Ji, X., Xu, L., An, J., Luo, J., Wu, W., & Huang, L. (2026).
        AdamPM: Memory-Efficient LLM Optimization via Moment Pooling.
        *arXiv:2610.24598*.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        group_size: int = 8,
        pool_dim: str = "row",
    ) -> None:
        """Initializes the AdamPM optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default: ``1e-3``).
            betas: ``(beta1, beta2)`` moment coefficients
                (default: ``(0.9, 0.95)``).
            eps: Numerical-stability constant added inside the square
                root's denominator (default: ``1e-8``).
            weight_decay: Decoupled weight-decay coefficient (default:
                ``0.0``).
            group_size: Second-moment pooling group size ``s``
                (default: ``8``, must be >= 1).
            pool_dim: Pooling direction, ``"row"`` or ``"col"``
                (default: ``"row"``).

        Raises:
            ValueError: If ``lr``, ``eps``, or ``betas`` are out of
                range, if ``group_size`` is not a positive integer, or
                if ``pool_dim`` is not ``"row"`` or ``"col"``.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if int(group_size) < 1:
            raise ValueError(f"Invalid group_size value: {group_size}")

        pool_dim = str(pool_dim).strip().lower()
        if pool_dim not in {"row", "col"}:
            raise ValueError(f"Invalid pool_dim value: {pool_dim!r} (expected 'row' or 'col')")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            group_size=int(group_size),
            pool_dim=pool_dim,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Applies Algorithm 1 of the paper: first-moment EMA, squared
        gradient, within-group mean, compressed second-moment EMA, bias
        corrections, broadcast of the pooled statistic, and the adaptive
        update (with decoupled weight decay applied first when
        ``weight_decay > 0``).

        Args:
            closure: A closure that re-evaluates the model and returns
                the loss.  Optional for most use cases.

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
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            group_size = int(group["group_size"])
            pool_dim = str(group["pool_dim"]).strip().lower()

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamPM does not support sparse gradients")

                state = self.state[param]
                pooled_view = _pool_matrix(param, pool_dim)
                outer, axis_length = pooled_view.shape
                n_groups = math.ceil(axis_length / group_size)

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["exp_avg_pooled"] = torch.zeros(
                        (outer, n_groups),
                        dtype=param.dtype,
                        device=param.device,
                    )

                exp_avg = state["exp_avg"]
                exp_avg_pooled = state["exp_avg_pooled"]
                state["step"] += 1
                step = state["step"]

                # M_t = beta1 * M_{t-1} + (1 - beta1) * G_t  (first moment,
                # element-wise: the paper never pools it).
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # S_t = G_t . G_t, then the mean of every s adjacent entries
                # along the pooling axis (the final partial group is averaged
                # over the entries it actually contains).
                squared_grad = grad * grad
                pooled = _group_mean(_pool_matrix(squared_grad, pool_dim), group_size)

                # V_t = beta2 * V_{t-1} + (1 - beta2) * S_t  (compressed EMA:
                # only the (outer, n_groups) statistic is persisted).
                exp_avg_pooled.mul_(beta2).add_(pooled, alpha=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                # Decoupled weight decay (repository extension; the paper's
                # Algorithm 1 does not define one).
                if weight_decay != 0:
                    param.mul_(1.0 - lr * weight_decay)

                # Broadcast the bias-corrected pooled statistic back over its
                # group: denominator = sqrt(V_hat) + eps.
                broadcasted = _broadcast_groups(
                    exp_avg_pooled / bias_correction2,
                    group_size,
                    axis_length,
                )
                if pool_dim == "col":
                    broadcasted = broadcasted.transpose(0, 1)
                denominator = broadcasted.reshape(param.shape)
                denominator = denominator.sqrt_().add_(eps)

                # W_t = W_{t-1} - eta_t * M_hat_t / (sqrt(V_hat_t) + eps).
                param.addcdiv_(exp_avg, denominator, value=-lr / bias_correction1)

        return loss
