"""Shampoo optimizer with Kronecker-factored preconditioners.

Shampoo (Gupta et al. 2018, arXiv:1802.09568) approximates the full-matrix
AdaGrad preconditioner by Kronecker-factoring it into per-dimension
preconditioners, reducing memory from O(n^2) to O(n^(2/d)) for d-dimensional
tensors.  This implementation provides a Shampoo-compatible interface using
AdamW stepping as a lightweight stand-in.

Reference:
    Gupta, V., Koren, T., & Singer, Y. (2018). Shampoo: Preconditioned
    Stochastic Tensor Optimization. arXiv:1802.09568.
"""

from __future__ import annotations

from torch.optim import AdamW


class Shampoo(AdamW):
    """Shampoo-compatible interface using AdamW stepping.

    Provides the Shampoo optimizer API surface backed by AdamW.  Full
    Kronecker-factored preconditioning is not implemented; this class exists
    as a configuration-compatible placeholder.

    Reference:
        Gupta, V., Koren, T., & Singer, Y. (2018). Shampoo: Preconditioned
        Stochastic Tensor Optimization. arXiv:1802.09568.

    Attributes:
        defaults (dict): Default hyper-parameter values inherited from AdamW.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state inherited from AdamW.
    """

    pass
