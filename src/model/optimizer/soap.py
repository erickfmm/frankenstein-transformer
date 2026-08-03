"""SOAP optimizer with Shampoo + Adam in eigenbasis.

SOAP (Vyas et al. 2024, arXiv:2409.11321) combines Shampoo-style
Kronecker-factored preconditioning with Adam-style adaptive learning rates
applied in the eigenbasis of the preconditioners.  This implementation
provides a SOAP-compatible interface using AdamW stepping as a lightweight
stand-in.

Reference:
    Vyas, N., Morwani, D., Zhao, R., Shapira, I., Brandfonbrener, D., Janson,
    L., & Kakade, S. (2024). SOAP: Improving and Stabilizing Shampoo using
    Adam. arXiv:2409.11321.
"""

from __future__ import annotations

from torch.optim import AdamW


class SOAP(AdamW):
    """SOAP-compatible interface using AdamW stepping.

    Provides the SOAP optimizer API surface backed by AdamW.  Full
    eigenbasis-based preconditioning is not implemented; this class exists as
    a configuration-compatible placeholder.

    Reference:
        Vyas, N., Morwani, D., Zhao, R., Shapira, I., Brandfonbrener, D.,
        Janson, L., & Kakade, S. (2024). SOAP: Improving and Stabilizing
        Shampoo using Adam. arXiv:2409.11321.

    Attributes:
        defaults (dict): Default hyper-parameter values inherited from AdamW.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state inherited from AdamW.
    """

    pass
