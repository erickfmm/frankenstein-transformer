"""TUON optimizer: tensor-aware orthogonalized updates.

TUON (Tang et al. 2026, arXiv:2610.35130) generalizes Muon's layer-wise
orthogonalization to cross-layer tensors using a transform-based tensor
algebra derived from the Tensor-SVD.  For each compatible group of ``K``
same-shape matrices it stacks the momentum tensors along the layer mode,
mixes the slices with a unitary transform ``Q`` (the default ``decorq``
takes the eigenvectors of the ``K x K`` layer Gram matrix), polarizes every
transformed slice independently with Newton-Schulz, inverts the transform,
and rescales the whole stack to RMS ``1 / sqrt(max(m, n))`` before applying a
decoupled weight-decay update.

Following the Muon hybrid recipe, only hidden-layer 2-D matrices take the
TUON path; embedding tables, learned positional encodings, output heads and
every non-matrix parameter (biases, norm scales, convolution kernels) receive
decoupled AdamW updates instead of being skipped.

Implementation notes:
    * One ``compatible group`` maps to one ``(parameter group, matrix shape)``
      bucket: same-shape 2-D parameters of a parameter group are stacked in
      registration order.  ``stack: full`` stacks every such matrix (the
      paper's default); ``stack: pairs`` restricts the stack to consecutive
      pairs (``K = 2``).
    * The paper's Algorithm 1 writes ``M_t = beta M_{t-1} + (1 - beta) G_t``
      while ``muon.py`` accumulates ``M_t = beta M_{t-1} + G_t``.  The two
      recursions differ by a constant ``(1 - beta)`` scale, and both the
      Frobenius normalization that starts Newton-Schulz and the shared RMS
      rescale ``gamma_t`` are scale-invariant, so the update is unchanged.
      This module implements Algorithm 1 verbatim.
    * For ``K = 1`` the transform cancels and TUON reduces to Muon
      (paper Proposition 2).
    * The cached layer basis lives in ``state[first_param]["tuon_basis"]``
      and therefore serializes with the optimizer state dict.

Reference:
    Tang, A., Yang, X., Wang, J., & Lin, Z. (2026). TUON: A Tensor-Aware
    Optimizer Utilizing Tensor-SVD and Polarization for Large Language Model
    Pretraining. arXiv:2610.35130.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.optim import Optimizer

from .base import ADAMW_FALLBACK_GROUPS, adamw_fallback_update

_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
"""Quintic Newton-Schulz coefficients shared with Muon."""

_TRANSFORMS = frozenset({"decorq", "dct", "haar", "dft"})
"""Supported layer transforms ``Q`` (paper Appendix E)."""

_STACK_MODES = frozenset({"full", "pairs"})
"""Supported stacking strategies (paper Table 8: ``K = full`` and ``K = 2``)."""


def _dct_matrix(size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Build the orthonormal DCT-II matrix (paper Appendix E).

    Args:
        size: Matrix dimension ``K`` (any positive integer).
        device: Device of the returned tensor.
        dtype: Real dtype of the returned tensor.

    Returns:
        A ``(size, size)`` orthonormal matrix ``Q`` with
        ``Q[j, k] = alpha_j cos(pi j (k + 1/2) / K)``, ``alpha_0 = K^{-1/2}``,
        ``alpha_j = (2/K)^{1/2}`` for ``j > 0``.
    """
    index = torch.arange(size, device=device, dtype=torch.float64)
    rows = index[:, None]
    cols = index[None, :]
    matrix = torch.cos(math.pi / size * (cols + 0.5) * rows)
    matrix = matrix * math.sqrt(2.0 / size)
    matrix[0] = matrix[0] / math.sqrt(2.0)
    return matrix.to(dtype=dtype)


def _haar_matrix(size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Build the orthonormal Haar wavelet matrix (paper Appendix E, Eq. 90).

    The recursive construction ``H_{2n} = 2^{-1/2} [[H_n (x) [1, 1]],
    [I_n (x) [1, -1]]]`` is defined for powers of two.  For other sizes the
    largest power-of-two block is embedded block-diagonally with an identity
    tail, which keeps the transform unitary.

    Args:
        size: Matrix dimension ``K``.
        device: Device of the returned tensor.
        dtype: Real dtype of the returned tensor.

    Returns:
        A ``(size, size)`` orthonormal matrix.
    """
    power = 1
    while power * 2 <= size:
        power *= 2

    ones = torch.ones(1, 2, device=device, dtype=torch.float64)
    ones_minus = torch.tensor([[1.0, -1.0]], device=device, dtype=torch.float64)
    block = torch.ones(1, 1, device=device, dtype=torch.float64)
    current = 1
    while current < power:
        identity = torch.eye(current, device=device, dtype=torch.float64)
        upper = torch.kron(block, ones)
        lower = torch.kron(identity, ones_minus)
        block = torch.cat([upper, lower], dim=0) / math.sqrt(2.0)
        current *= 2

    if power == size:
        return block.to(dtype=dtype)
    matrix = torch.eye(size, device=device, dtype=torch.float64)
    matrix[:power, :power] = block
    return matrix.to(dtype=dtype)


def _dft_matrix(size: int, device: torch.device) -> Tensor:
    """Build the unitary DFT matrix (paper Appendix E).

    Args:
        size: Matrix dimension ``K``.
        device: Device of the returned tensor.

    Returns:
        A ``(size, size)`` unitary complex matrix with
        ``Q[j, k] = K^{-1/2} exp(-2 pi i j k / K)``.
    """
    index = torch.arange(size, device=device, dtype=torch.float64)
    phase = -2.0 * math.pi * torch.outer(index, index) / size
    real = torch.cos(phase) / math.sqrt(size)
    imag = torch.sin(phase) / math.sqrt(size)
    return torch.complex(real, imag).to(torch.complex64)


def _build_transform(
    name: str,
    size: int,
    flat: Tensor,
    device: torch.device,
) -> Tensor:
    """Build the layer transform ``Q`` (paper Algorithm 1, line 5).

    Args:
        name: One of ``decorq``, ``dct``, ``haar`` or ``dft``.
        size: Number of stacked slices ``K``.
        flat: Mode-3 unfolding of the momentum tensor, shape ``(K, m * n)``;
            only read by ``decorq``.
        device: Device of the returned tensor.

    Returns:
        The ``(K, K)`` transform.  Real for ``decorq``/``dct``/``haar`` and
        complex64 for ``dft``.

    Raises:
        ValueError: If ``name`` is not a supported transform.
    """
    if name == "decorq":
        gram = flat @ flat.transpose(0, 1)
        _, eigenvectors = torch.linalg.eigh(gram)
        return eigenvectors.transpose(0, 1).to(dtype=flat.dtype)
    if name == "dct":
        return _dct_matrix(size, device, flat.dtype)
    if name == "haar":
        return _haar_matrix(size, device, flat.dtype)
    if name == "dft":
        return _dft_matrix(size, device)
    raise ValueError(f"Unknown TUON transform: {name}")


def _newton_schulz_slices(
    slices: Tensor,
    steps: int,
    eps: float,
    coefficients: tuple[float, float, float] = _COEFFICIENTS,
) -> Tensor:
    """Approximate the orthogonal polar factor of every stacked slice.

    Implements Algorithm 1, lines 7-9 of the TUON paper (Appendix F): each
    ``m x n`` slice is normalized by its Frobenius norm, transposed so that
    ``rows <= cols`` when necessary, and refined with ``steps`` quintic
    Newton-Schulz iterations.  Real slices iterate in bfloat16 like the other
    Muon-family modules; complex slices (``dft``) iterate in complex64 with
    the Hermitian Gram matrix ``X X^H``.

    Args:
        slices: Stack of matrices, shape ``(m, n, K)``.
        steps: Number of Newton-Schulz iterations (the paper uses five).
        eps: Numerical-stability term of the Frobenius normalization.
        coefficients: Quintic polynomial ``(a, b, c)``; defaults to the Muon
            triple.

    Returns:
        The orthogonalized stack in the input orientation, in bfloat16 for
        real inputs and complex64 for complex inputs.
    """
    transposed = slices.size(0) > slices.size(1)
    work = slices.transpose(0, 1) if transposed else slices
    a, b, c = coefficients

    if work.is_complex():
        work = work.to(torch.complex64)
        norms = work.abs().pow(2).sum(dim=(0, 1)).sqrt()
        work = work / (norms + eps)
        batch = work.permute(2, 0, 1)
        for _ in range(int(steps)):
            gram = batch @ batch.mH
            batch = a * batch + (b * gram + c * (gram @ gram)) @ batch
        result = batch.permute(1, 2, 0)
    else:
        work = work.bfloat16()
        norms = work.norm(dim=(0, 1))
        work = work / (norms + eps)
        batch = work.permute(2, 0, 1)
        for _ in range(int(steps)):
            gram = batch @ batch.transpose(-2, -1)
            batch = a * batch + (b * gram + c * (gram @ gram)) @ batch
        result = batch.permute(1, 2, 0)

    return result.transpose(0, 1) if transposed else result


class TUON(Optimizer):
    r"""Tensor-aware optimizer with T-SVD transforms and slice polarization.

    Accumulates momentum on matrix-shaped gradients, stacks the momenta of
    each compatible group along the layer mode, mixes the slices with a
    unitary transform ``Q``, polarizes every transformed slice with
    Newton-Schulz, inverts the transform and rescales the whole stack to RMS
    ``1 / sqrt(max(m, n))``.  The default ``decorq`` transform takes the
    eigenvectors of the ``K x K`` layer Gram matrix and is refreshed every
    ``refresh_interval`` steps; ``dct``, ``haar`` and ``dft`` are fixed
    alternatives.

    Hidden-layer 2-D parameters take this path; parameters in the embedding,
    positional-encoding and head groups, as well as any parameter that is not
    a 2-D matrix, take a decoupled AdamW update using the group's ``betas``
    and ``eps``.

    Reference:
        Tang, A., Yang, X., Wang, J., & Lin, Z. (2026). TUON: A Tensor-Aware
        Optimizer Utilizing Tensor-SVD and Polarization for Large Language
        Model Pretraining. arXiv:2610.35130.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate (default: 0.02, matching the Muon-family default;
            the RMS stack scaling keeps the update magnitude at Muon's
            level).
        momentum: Momentum coefficient ``beta`` of the momentum tensor
            (default: 0.95).
        ns_steps: Number of Newton-Schulz iterations per transformed slice
            (default: 5).
        ns_eps: Epsilon added during the Frobenius normalization before
            Newton-Schulz (default: 1e-7).
        refresh_interval: DecorQ basis refresh period ``T`` in optimizer
            steps (default: 200, the paper's Algorithm 1).
        rms_eps: Epsilon of the stack RMS scaling
            ``gamma_t = s / (RMS(P_t) + eps)`` (default: 1e-8, the paper's
            implementation setting).
        transform: Layer transform ``Q``: ``decorq`` (default), ``dct``,
            ``haar`` or ``dft``.
        stack: Stacking strategy: ``full`` (every same-shape matrix of the
            group; default) or ``pairs`` (consecutive pairs, ``K = 2``).
        weight_decay: Decoupled weight decay coefficient (default: 0.0).
        betas: ``(beta1, beta2)`` coefficients of the AdamW fallback
            (default: ``(0.9, 0.95)``).
        eps: Epsilon of the AdamW fallback (default: 1e-8).

    Attributes:
        defaults (dict): Default hyper-parameter values for each parameter
            group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state (momentum buffer and, for
            the first parameter of each stack, the cached layer basis; first
            and second moments for the AdamW fallback).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        ns_eps: float = 1e-7,
        refresh_interval: int = 200,
        rms_eps: float = 1e-8,
        transform: str = "decorq",
        stack: str = "full",
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        """Initializes the TUON optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default: 0.02).
            momentum: Momentum coefficient (default: 0.95).
            ns_steps: Newton-Schulz iteration count (default: 5).
            ns_eps: Frobenius-normalization epsilon (default: 1e-7).
            refresh_interval: DecorQ basis refresh period (default: 200).
            rms_eps: Stack-RMS scaling epsilon (default: 1e-8).
            transform: Layer transform (default: ``decorq``).
            stack: Stacking strategy (default: ``full``).
            weight_decay: Decoupled weight decay coefficient (default: 0.0).
            betas: AdamW-fallback moment coefficients (default: ``(0.9,
                0.95)``).
            eps: AdamW-fallback epsilon (default: 1e-8).

        Raises:
            ValueError: If any argument is out of its valid range.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 1 <= ns_steps:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")
        if not 0.0 <= ns_eps:
            raise ValueError(f"Invalid ns_eps value: {ns_eps}")
        if not 1 <= refresh_interval:
            raise ValueError(f"Invalid refresh_interval value: {refresh_interval}")
        if not 0.0 <= rms_eps:
            raise ValueError(f"Invalid rms_eps value: {rms_eps}")
        transform = str(transform).strip().lower()
        if transform not in _TRANSFORMS:
            raise ValueError(
                f"Invalid transform '{transform}'; expected one of {sorted(_TRANSFORMS)}"
            )
        stack = str(stack).strip().lower()
        if stack not in _STACK_MODES:
            raise ValueError(
                f"Invalid stack '{stack}'; expected one of {sorted(_STACK_MODES)}"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            refresh_interval=refresh_interval,
            rms_eps=rms_eps,
            transform=transform,
            stack=stack,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def _update_stack(self, group: dict, params: list[Tensor]) -> None:
        """Apply one TUON update to a bucket of same-shape matrices.

        Maps Algorithm 1 lines 3-11: layer Gram + eigendecomposition (with the
        ``refresh_interval`` cache), mode-3 transform, parallel slice
        polarization, inverse transform, shared RMS scaling and the
        decoupled weight-decay update.

        Args:
            group: The parameter group holding the tuned hyperparameters.
            params: Non-empty list of same-shape 2-D parameters with
                gradients.
        """
        momentum = torch.stack(
            [self.state[p]["momentum_buffer"] for p in params], dim=-1
        )
        slices, columns = momentum.size(0), momentum.size(1)
        count = momentum.size(-1)
        transform = str(group["transform"]).lower()

        # Algorithm 1, lines 3-5: C_t = M M^T, eigendecomposition, Q = U^T,
        # refreshed every `refresh_interval` steps.
        head = self.state[params[0]]
        head["tuon_step"] = int(head.get("tuon_step", 0)) + 1
        step_index = head["tuon_step"]
        basis = head.get("tuon_basis")
        if (
            basis is None
            or tuple(basis.shape) != (count, count)
            or step_index % int(group["refresh_interval"]) == 0
        ):
            # Mode-3 unfolding: row k is vec(M(:, :, k))^T (paper Eq. 80).
            flat = momentum.permute(2, 0, 1).reshape(count, slices * columns)
            basis = _build_transform(transform, count, flat, momentum.device)
            head["tuon_basis"] = basis

        # Algorithm 1, line 6: M_tilde = M x_3 Q.
        if transform == "dft":
            q = basis.to(torch.complex64)
            mixed = torch.einsum("jk,mnk->mnj", q, momentum.to(torch.complex64))
        else:
            q = basis.to(dtype=momentum.dtype)
            mixed = torch.einsum("jk,mnk->mnj", q, momentum)

        # Algorithm 1, lines 7-9: parallel Newton-Schulz slice polarization.
        polarized = _newton_schulz_slices(
            mixed, int(group["ns_steps"]), float(group["ns_eps"])
        )

        # Algorithm 1, line 10: P = P_tilde x_3 Q^H.
        if transform == "dft":
            update = torch.einsum("jk,mnj->mnk", q.conj(), polarized).real
        else:
            update = torch.einsum("jk,mnj->mnk", q, polarized.to(q.dtype))

        # Algorithm 1, lines 11 and Eqs. 100-103: shared RMS scaling and the
        # decoupled update W <- (1 - eta * lambda) W - eta * gamma * P.
        rms = update.float().pow(2).mean().sqrt()
        target = 1.0 / math.sqrt(max(slices, columns))
        gamma = target / (float(rms) + float(group["rms_eps"]))
        step_size = float(group["lr"]) * gamma
        if float(group["weight_decay"]) > 0:
            decay = 1.0 - float(group["lr"]) * float(group["weight_decay"])
            for param in params:
                param.mul_(decay)
        for index, param in enumerate(params):
            param.add_(update[:, :, index].to(param.dtype), alpha=-step_size)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure: A closure that re-evaluates the model and returns the
                loss.  Optional for most use cases.

        Returns:
            The loss value if ``closure`` is provided, otherwise ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_adamw = str(group.get("name", "other")) in ADAMW_FALLBACK_GROUPS
            buckets: dict[tuple[int, ...], list[Tensor]] = {}
            for param in group["params"]:
                if param.grad is None:
                    continue

                # Paper hybrid recipe: embeddings, positional encodings,
                # output heads and every non-matrix tensor take AdamW; only
                # hidden-layer 2-D matrices are orthogonalized.
                if use_adamw or param.ndim != 2 or param.numel() == 0:
                    adamw_fallback_update(
                        param,
                        param.grad,
                        self.state[param],
                        lr=group["lr"],
                        betas=group.get("betas", (0.9, 0.95)),
                        eps=group.get("eps", 1e-8),
                        weight_decay=group["weight_decay"],
                    )
                    continue

                # Algorithm 1, line 2: M_t = beta M_{t-1} + (1 - beta) G_t.
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param.grad)
                buffer = state["momentum_buffer"]
                beta = float(group["momentum"])
                buffer.mul_(beta).add_(param.grad, alpha=1.0 - beta)
                buckets.setdefault(tuple(param.shape), []).append(param)

            for params in buckets.values():
                if str(group["stack"]).lower() == "pairs":
                    stacks = [params[i : i + 2] for i in range(0, len(params), 2)]
                else:
                    stacks = [params]
                for stack in stacks:
                    self._update_stack(group, stack)

        return loss
