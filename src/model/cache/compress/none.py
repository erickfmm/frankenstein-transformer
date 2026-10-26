"""Keep-everything policy (default). No eviction ever happens."""

from __future__ import annotations

from typing import Optional

__all__ = ["NonePolicy"]


class NonePolicy:
    """Identity policy: retain the full cache across decode steps.

    Attributes:
        name: Policy identifier (``"none"``).
    """

    name = "none"

    def on_append(self, slot, cache_position: int) -> None:  # noqa: ARG002
        """No-op."""

    def on_step_end(self, slot, cache_position: int) -> None:  # noqa: ARG002
        """No-op."""

    def budget(self, logical_layer_idx: int) -> Optional[int]:  # noqa: ARG002
        """Unlimited retention."""
        return None
