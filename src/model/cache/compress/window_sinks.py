"""StreamingLLM-style window-with-sinks eviction policy.

Implements the attention-sinks recipe (StreamingLLM, arXiv:2309.17453;
"Sliding Window Beats Linear Attention", arXiv:2608.28444): pin the first
``num_sinks`` leading positions and keep only the last ``window``
positions, dropping everything in between. Removing the sinks collapses
performance catastrophically; the retained region must stay exact.

Slots honor the stored ``positions`` metadata, so eviction gaps produce
correct relative distances for ALiBi/BAM-style biases and for
position-based sparse patterns (plan §5.6 / blocker B5).

Constraints (plan §10): policies apply only to position-indexed slots
(``KVSlot`` / ``RingSlot`` / ``LatentKVSlot``) and never run on
``mtla_attn`` slots (D13 — MTLA manages its own merging schedule).
"""

from __future__ import annotations

from typing import List, Optional

__all__ = ["WindowSinksPolicy"]


class WindowSinksPolicy:
    """Pin ``num_sinks`` leading tokens; keep the last ``window`` tokens.

    Args:
        num_sinks: Number of leading sink positions pinned forever
            (paper default: 4).
        window: Size of the sliding window over the most recent positions.
        per_layer_budgets: Optional per-layer budget vector overriding the
            scalar ``window`` (PyramidKV-style depth allocation,
            arXiv:2406.02069); index ``i`` is ``logical_layer_idx % len``.

    Attributes:
        name: Policy identifier (``"window_sinks"``).
    """

    name = "window_sinks"

    def __init__(
        self,
        num_sinks: int = 4,
        window: int = 512,
        per_layer_budgets: Optional[List[int]] = None,
    ):
        if window < 1:
            raise ValueError("window_sinks policy requires window >= 1")
        if num_sinks < 0:
            raise ValueError("num_sinks must be non-negative")
        self.num_sinks = int(num_sinks)
        self.window = int(window)
        self.per_layer_budgets = list(per_layer_budgets or [])

    def budget(self, logical_layer_idx: int) -> Optional[int]:
        """Max retained tokens for this layer (sinks + window)."""
        window = self.window
        if self.per_layer_budgets:
            window = int(self.per_layer_budgets[logical_layer_idx % len(self.per_layer_budgets)])
        return self.num_sinks + max(1, window)

    def on_append(self, slot, cache_position: int) -> None:  # noqa: ARG002
        """Evict immediately when over budget (slots honor sinks)."""
        self._evict(slot)

    def on_step_end(self, slot, cache_position: int) -> None:  # noqa: ARG002
        """Safety net: enforce the budget at each step boundary too."""
        self._evict(slot)

    def _evict(self, slot) -> None:
        budget = self.budget(getattr(getattr(slot, "meta", None), "logical_layer_idx", 0) or 0)
        protected = max(self.num_sinks, int(getattr(slot, "num_sinks", 0) or 0))
        overflow = int(getattr(slot, "length", 0)) - max(budget, protected)
        if overflow > 0:
            slot.evict_oldest(overflow)
