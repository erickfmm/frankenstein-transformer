"""APOLLO-Mini optimizer with rank-1 tensor-wise scaling.

APOLLO-Mini is a memory-efficient variant of APOLLO that uses rank-1 Gaussian
random projection and tensor-wise (single scalar) scaling, achieving extreme
memory efficiency while retaining structured adaptive gradient scaling.

Reference:
    Zhu, Z., Li, Y., Wang, Z., & Anandkumar, A. (2025). APOLLO: SGD-like
    Memory, Adam-like Performance. arXiv:2502.xxxxx.
"""

from __future__ import annotations

from .apollo import Apollo


class ApolloMini(Apollo):
    """APOLLO-Mini: rank-1 tensor-wise scaled APOLLO variant.

    Configures the base APOLLO optimizer with ``rank=1`` and
    ``scale_type="tensor"`` for extreme memory efficiency.  All other
    hyper-parameters are forwarded to the parent class.

    Reference:
        Zhu, Z., Li, Y., Wang, Z., & Anandkumar, A. (2025). APOLLO: SGD-like
        Memory, Adam-like Performance. arXiv:2502.xxxxx.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate (default: 1e-3).
        update_proj_gap: Number of steps between projector resampling
            (default: 200).
        scale: Global scale multiplier (default: 128.0).
        proj_type: Projection side strategy (default: ``"std"``).
        betas: Coefficients for first and second moment running averages
            (default: ``(0.9, 0.999)``).
        eps: Term added for numerical stability (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 0.0).
        correct_bias: Whether to apply bias correction (default: True).
        scale_front: Whether to apply scale before norm limiter
            (default: False).
        disable_nl: Whether to disable norm-growth limiter (default: False).

    Attributes:
        Inherits all attributes from :class:`Apollo`.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        update_proj_gap=200,
        scale=128.0,
        proj_type="std",
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        correct_bias=True,
        scale_front=False,
        disable_nl=False,
    ):
        """Initializes the APOLLO-Mini optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default: 1e-3).
            update_proj_gap: Projector resampling interval (default: 200).
            scale: Global scale multiplier (default: 128.0).
            proj_type: Projection side strategy (default: ``"std"``).
            betas: Momentum decay coefficients (default: ``(0.9, 0.999)``).
            eps: Numerical stability term (default: 1e-8).
            weight_decay: Decoupled weight decay coefficient (default: 0.0).
            correct_bias: Apply bias correction (default: True).
            scale_front: Apply scale before norm limiter (default: False).
            disable_nl: Disable norm-growth limiter (default: False).
        """
        super().__init__(
            params=params,
            lr=lr,
            rank=1,
            update_proj_gap=update_proj_gap,
            scale=scale,
            scale_type="tensor",
            proj_type=proj_type,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
            scale_front=scale_front,
            disable_nl=disable_nl,
        )
