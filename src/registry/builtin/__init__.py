"""Built-in component specs (first-party plugins).

Every built-in component ships a spec in this package and registers through
the same :func:`src.registry.register_spec` API that third-party plugins use.
The modules import only :mod:`src.registry`; builders are lazy
``"module:attribute"`` strings, so importing this package never imports torch
or any model module.
"""

from __future__ import annotations

__all__ = ["ensure_builtins", "reset_builtins"]

_REGISTERED = False


def ensure_builtins() -> None:
    """Register every built-in component spec exactly once (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    from . import activation, attention, embedding, norm, optimizer, residual

    for module in (attention, optimizer, norm, activation, residual, embedding):
        module.register()
    _REGISTERED = True


def reset_builtins() -> None:
    """Allow a subsequent :func:`ensure_builtins` call (test utility)."""
    global _REGISTERED
    _REGISTERED = False
