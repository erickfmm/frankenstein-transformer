"""Device resolution utilities for PyTorch training and inference.

Provides a single entry point for resolving user-requested device strings
(``auto``, ``cpu``, ``cuda``, ``mps``) into concrete PyTorch device identifiers,
with availability checks for CUDA and Apple Metal Performance Shaders (MPS).
"""

from __future__ import annotations

import torch


SUPPORTED_DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


def is_mps_available() -> bool:
    """Check whether Apple Metal Performance Shaders (MPS) backend is available.

    Returns:
        ``True`` if ``torch.backends.mps`` is available and built, ``False`` otherwise.
    """
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def resolve_torch_device(requested: str = "auto") -> str:
    """Resolve a user-requested device string to a concrete PyTorch device.

    Args:
        requested: One of ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
            ``"auto"`` selects CUDA if available, then MPS, falling back to CPU.

    Returns:
        The resolved device string (``"cuda"``, ``"mps"``, or ``"cpu"``).

    Raises:
        ValueError: If ``requested`` is not a supported choice, or if the
            requested device is not available on the current machine.
    """
    requested_normalized = (requested or "auto").strip().lower()
    if requested_normalized not in SUPPORTED_DEVICE_CHOICES:
        raise ValueError(
            f"Unsupported device '{requested}'. Supported values: {', '.join(SUPPORTED_DEVICE_CHOICES)}"
        )

    if requested_normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if is_mps_available():
            return "mps"
        return "cpu"

    if requested_normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available on this machine")

    if requested_normalized == "mps" and not is_mps_available():
        raise ValueError("MPS was requested but is not available on this machine")

    return requested_normalized
