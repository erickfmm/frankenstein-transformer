"""Global RNG seeding helpers for reproducible training runs.

``training.seed`` (default ``42``) drives every stochastic source in a run:
the Python, NumPy, PyTorch and CUDA random generators, model weight
initialization (the entrypoints seed *before* constructing the model), the
DataLoader shuffle order and worker seeding, the held-out test-split
permutation, and the SBERT dataset shuffle.

CUDA runs also enable deterministic cuDNN (``cudnn.deterministic = True``,
``cudnn.benchmark = False``). This can cost some throughput, but it makes
repeated runs on the same hardware and software stack reproducible.
``torch.use_deterministic_algorithms`` is intentionally not enabled: several
CUDA kernels raise on it, which would turn a reproducibility knob into a
runtime failure.

The module is import-safe without CUDA and without NumPy (NumPy is part of the
project dependencies, but the guard keeps the helper usable in minimal
environments).
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict

import torch

try:  # pragma: no cover - NumPy ships with the project dependencies
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def set_global_seed(seed: int) -> None:
    """Seed every RNG used by the training stack and enable deterministic cuDNN.

    Seeds ``random``, NumPy (when available), ``torch``, and every CUDA device
    (when CUDA is available). Also exports ``PYTHONHASHSEED`` for child
    processes (the thermal supervisor re-launches training as a subprocess) and
    sets ``cudnn.deterministic = True`` / ``cudnn.benchmark = False`` so cuDNN
    picks deterministic kernels.

    Args:
        seed: Non-negative integer seed (``training.seed``).

    Raises:
        ValueError: If ``seed`` is negative.
    """
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be >= 0")

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    if np is not None:
        np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed: int) -> torch.Generator:
    """Return a CPU ``torch.Generator`` seeded with ``seed``.

    Pass the result to ``DataLoader(..., generator=...)`` so ``shuffle=True``
    draws its permutation from a dedicated, reproducible stream.

    Args:
        seed: Non-negative integer seed (``training.seed``).

    Returns:
        A seeded :class:`torch.Generator`.
    """
    return torch.Generator().manual_seed(int(seed))


def seed_worker(worker_id: int) -> None:
    """``DataLoader`` ``worker_init_fn`` that seeds each worker reproducibly.

    Each worker derives its seed from the loader's base seed
    (``torch.initial_seed()``), so a run with the same ``training.seed`` and
    ``num_workers`` shuffles and augments data identically.

    Args:
        worker_id: Worker index provided by PyTorch (unused directly; the
            per-worker seed is derived from the loader's base seed).
    """
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    if np is not None:
        np.random.seed(worker_seed)


def dataloader_kwargs(seed: int) -> Dict[str, Any]:
    """Return the ``DataLoader`` kwargs needed for reproducible shuffling.

    Args:
        seed: Non-negative integer seed (``training.seed``).

    Returns:
        ``{"generator": <seeded generator>, "worker_init_fn": seed_worker}``.
    """
    return {"generator": make_generator(seed), "worker_init_fn": seed_worker}
