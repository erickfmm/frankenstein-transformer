"""Mixture-of-Experts building blocks.

Public surface:

* :class:`~src.model.moe.layer.MoELayer` — routed experts + shared experts +
  optional latent projections.
* :class:`~src.model.moe.router.MoERouter` — logits → weights and the
  auxiliary-loss-free expert bias.
* Load-balancing helpers from :mod:`src.model.moe.balancing`.
"""

from __future__ import annotations

from .balancing import (
    aggregate_moe_aux_losses,
    load_balancing_loss,
    max_vio,
    update_moe_expert_bias,
)
from .layer import MoELayer
from .router import MoERouter

__all__ = [
    "MoELayer",
    "MoERouter",
    "aggregate_moe_aux_losses",
    "load_balancing_loss",
    "max_vio",
    "update_moe_expert_bias",
]
