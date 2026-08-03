"""APOLLO optimizer with structured scaling from projected moments.

APOLLO (Zhu et al. 2025, arXiv:2502.xxxxx) projects 2D gradients into a
low-rank subspace via Gaussian random matrices, computes Adam-style first and
second moments in the compressed space, and then scales the original gradient
element-wise or channel-wise using the ratio of the normalized low-rank
moment to the low-rank gradient norm.  This provides structured adaptive
scaling with sublinear memory overhead.

Reference:
    Zhu, Z., Li, Y., Wang, Z., & Anandkumar, A. (2025). APOLLO: SGD-like
    Memory, Adam-like Performance. arXiv:2502.xxxxx.
"""

from __future__ import annotations

import math
import torch
from torch.optim import Optimizer

LCG_MULTIPLIER = 1103515245
LCG_INCREMENT = 12345
LCG_MODULUS = 2**31 - 1
NORM_LIMITER_THRESHOLD = 1.01
EPS = 1e-8


class Apollo(Optimizer):
    """APOLLO optimizer with structured scaling from projected moments.

    Projects 2D gradients into a low-rank subspace via Gaussian random
    projection matrices, maintains Adam-style first and second moments in the
    compressed space, and scales the original gradient using the ratio of the
    normalized low-rank moment to the low-rank gradient norm.  Supports
    channel-wise and tensor-wise scaling, optional norm-growth limiting, and
    configurable projection side.

    Reference:
        Zhu, Z., Li, Y., Wang, Z., & Anandkumar, A. (2025). APOLLO: SGD-like
        Memory, Adam-like Performance. arXiv:2502.xxxxx.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate (default: 1e-3).
        rank: Rank of the Gaussian random projection subspace (default: 128).
        update_proj_gap: Number of steps between projector resampling
            (default: 200).
        scale: Global scale multiplier applied to the structured update
            (default: 1.0).
        scale_type: Scaling granularity, either ``"channel"`` (per-row or
            per-column) or ``"tensor"`` (single scalar per tensor)
            (default: ``"channel"``).
        proj_type: Projection side strategy.  ``"std"`` projects from the
            larger dimension, ``"reverse_std"`` from the smaller,
            ``"left"`` and ``"right"`` force a specific side
            (default: ``"std"``).
        betas: Coefficients for first and second moment running averages
            (default: ``(0.9, 0.999)``).
        eps: Term added for numerical stability (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 0.0).
        correct_bias: Whether to apply Adam-style bias correction
            (default: True).
        scale_front: Whether to apply the global scale multiplier before
            norm-growth limiting instead of after (default: False).
        disable_nl: Whether to disable the Fira-style norm-growth limiter
            (default: False).

    Attributes:
        defaults (dict): Default hyper-parameter values for each parameter
            group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state (step counter, projector
            matrix, projector seed, low-rank moment buffers, scaled gradient
            norm).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        rank=128,
        update_proj_gap=200,
        scale=1.0,
        scale_type="channel",
        proj_type="std",
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        correct_bias=True,
        scale_front=False,
        disable_nl=False,
    ):
        """Initializes the APOLLO optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default: 1e-3).
            rank: Projection rank (default: 128).
            update_proj_gap: Projector resampling interval (default: 200).
            scale: Global scale multiplier (default: 1.0).
            scale_type: ``"channel"`` or ``"tensor"`` (default: ``"channel"``).
            proj_type: ``"std"``, ``"reverse_std"``, ``"left"``, or
                ``"right"`` (default: ``"std"``).
            betas: Momentum decay coefficients (default: ``(0.9, 0.999)``).
            eps: Numerical stability term (default: 1e-8).
            weight_decay: Decoupled weight decay coefficient (default: 0.0).
            correct_bias: Apply bias correction (default: True).
            scale_front: Apply scale before norm limiter (default: False).
            disable_nl: Disable norm-growth limiter (default: False).

        Raises:
            ValueError: If ``scale_type`` is not ``"channel"`` or
                ``"tensor"``, or ``proj_type`` is not one of the valid
                options.
        """
        if scale_type not in {"channel", "tensor"}:
            raise ValueError("scale_type must be one of: channel, tensor")
        if proj_type not in {"std", "reverse_std", "left", "right"}:
            raise ValueError("proj_type must be one of: std, reverse_std, left, right")

        defaults = dict(
            lr=lr,
            rank=rank,
            update_proj_gap=update_proj_gap,
            scale=scale,
            scale_type=scale_type,
            proj_type=proj_type,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
            scale_front=scale_front,
            disable_nl=disable_nl,
        )
        super().__init__(params, defaults)

    def _load_moments(self, state, grad_like):
        """Loads or initializes the first and second moment buffers.

        Args:
            state: Per-parameter optimizer state dict.
            grad_like: Tensor whose shape and dtype are used to initialize
                missing or shape-mismatched buffers.

        Returns:
            Tuple of ``(exp_avg, exp_avg_sq)`` tensors matching the shape
            and dtype of ``grad_like``.
        """
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg is None or exp_avg_sq is None or exp_avg.shape != grad_like.shape:
            exp_avg = torch.zeros_like(grad_like)
            exp_avg_sq = torch.zeros_like(grad_like)
        return exp_avg, exp_avg_sq

    def _store_moments(self, state, exp_avg, exp_avg_sq):
        """Stores the first and second moment buffers in the optimizer state.

        Args:
            state: Per-parameter optimizer state dict.
            exp_avg: First moment (exponential moving average of gradients).
            exp_avg_sq: Second moment (exponential moving average of squared
                gradients).
        """
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq

    @staticmethod
    def _sample_projector(shape, rank, side, *, device, dtype, seed):
        """Samples a Gaussian random projection matrix.

        Args:
            shape: Shape ``(m, n)`` of the gradient tensor.
            rank: Rank of the projection subspace.
            side: ``"left"`` to produce an ``(m, rank)`` matrix, ``"right"``
                for ``(rank, n)``.
            device: Torch device for the projector.
            dtype: Torch dtype for the projector.
            seed: Integer seed for the random generator.

        Returns:
            A Gaussian random matrix scaled by ``1 / sqrt(rank)``.
        """
        generator = torch.Generator(device=device).manual_seed(int(seed))
        if side == "left":
            mat = torch.randn((shape[0], rank), generator=generator, device=device, dtype=dtype)
        else:
            mat = torch.randn((rank, shape[1]), generator=generator, device=device, dtype=dtype)
        return mat / math.sqrt(max(rank, 1))

    def _project_grad(self, grad, state, group):
        """Projects a 2D gradient into a low-rank subspace.

        Determines the projection side based on ``proj_type``, samples or
        reuses a Gaussian random projector, and applies the projection.
        The projector is resampled every ``update_proj_gap`` steps or when
        the gradient shape changes.

        Args:
            grad: 2D gradient tensor of shape ``(m, n)``.
            state: Per-parameter optimizer state dict.
            group: Parameter group dict with ``rank``, ``update_proj_gap``,
                and ``proj_type``.

        Returns:
            Tuple of ``(low_grad, norm_dim)`` where ``low_grad`` is the
            projected gradient and ``norm_dim`` is the dimension index (0 or
            1) along which norms should be computed for channel-wise scaling.
        """
        rank = max(1, min(int(group["rank"]), min(grad.shape)))
        step = int(state.get("step", 0))
        gap = max(1, int(group["update_proj_gap"]))
        proj_type = str(group["proj_type"])

        if proj_type == "std":
            side = "right" if grad.shape[0] >= grad.shape[1] else "left"
        elif proj_type == "reverse_std":
            side = "left" if grad.shape[0] >= grad.shape[1] else "right"
        elif proj_type in {"left", "right"}:
            side = proj_type
        else:
            side = "right"

        if "projector_seed" not in state:
            state["projector_seed"] = int(torch.randint(1, 2**31 - 1, (1,), device="cpu").item())

        needs_update = (
            "projector" not in state
            or state.get("projector_side") != side
            or step % gap == 0
            or state["projector"].shape[0] != (grad.shape[0] if side == "left" else rank)
            or state["projector"].shape[1] != (rank if side == "left" else grad.shape[1])
        )
        if needs_update:
            state["projector"] = self._sample_projector(
                grad.shape,
                rank,
                side,
                device=grad.device,
                dtype=grad.dtype,
                seed=state["projector_seed"],
            )
            state["projector_side"] = side
            state["projector_seed"] = int(
                (state["projector_seed"] * LCG_MULTIPLIER + LCG_INCREMENT) % LCG_MODULUS
            )

        proj = state["projector"]
        if side == "right":
            low_grad = grad @ proj.T
        else:
            low_grad = proj.T @ grad
        norm_dim = 0 if low_grad.shape[0] < low_grad.shape[1] else 1
        return low_grad, norm_dim

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
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Apollo does not support sparse gradients")

                state = self.state[p]
                state["step"] = int(state.get("step", 0)) + 1
                step_num = state["step"]

                projected = grad.ndim == 2 and int(group["rank"]) > 0
                if projected:
                    low_grad, norm_dim = self._project_grad(grad, state, group)
                else:
                    low_grad = grad
                    norm_dim = None

                exp_avg, exp_avg_sq = self._load_moments(state, low_grad)
                exp_avg.mul_(beta1).add_(low_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(low_grad, low_grad, value=1.0 - beta2)
                self._store_moments(state, exp_avg, exp_avg_sq)

                denom = exp_avg_sq.sqrt().add_(eps)
                norm_grad = exp_avg / denom

                step_size = lr
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1**step_num
                    bias_correction2 = 1.0 - beta2**step_num
                    step_size = lr * math.sqrt(bias_correction2) / max(bias_correction1, 1e-16)

                if projected:
                    if group["scale_type"] == "channel":
                        numer = torch.norm(norm_grad, dim=norm_dim)
                        denom_scale = torch.norm(low_grad, dim=norm_dim).clamp(min=1e-8)
                        scaling = numer / denom_scale
                        if norm_dim == 1:
                            scaling = scaling.unsqueeze(1)
                    else:
                        numer = torch.norm(norm_grad)
                        denom_scale = torch.norm(low_grad).clamp(min=1e-8)
                        scaling = numer / denom_scale

                    update = grad * scaling
                    scale_coeff = math.sqrt(max(float(group["scale"]), 0.0))
                    if group["scale_front"]:
                        update = update * scale_coeff

                    if not group["disable_nl"]:
                        update_norm = torch.norm(update)
                        prev_norm = state.get("scaled_grad_norm")
                        if prev_norm is not None:
                            limiter = (
                                max(float(update_norm / (prev_norm + EPS)), NORM_LIMITER_THRESHOLD)
                                / NORM_LIMITER_THRESHOLD
                            )
                            update = update / limiter
                            state["scaled_grad_norm"] = update_norm / limiter
                        else:
                            state["scaled_grad_norm"] = update_norm

                    if not group["scale_front"]:
                        update = update * scale_coeff
                else:
                    update = norm_grad

                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-step_size)

        return loss
