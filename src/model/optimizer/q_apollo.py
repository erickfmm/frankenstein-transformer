"""Q-APOLLO optimizer with quantized low-rank state.

Q-APOLLO extends APOLLO by storing the first and second moment buffers in
quantized uint8 form, dramatically reducing memory consumption for the
optimizer state.  Quantization uses uniform affine mapping with per-tensor
scale and offset, supporting 2 to 8 bits per element.

Reference:
    Zhu, Z., Li, Y., Wang, Z., & Anandkumar, A. (2025). APOLLO: SGD-like
    Memory, Adam-like Performance. arXiv:2502.xxxxx.
"""

from __future__ import annotations

import warnings

import torch

from .apollo import Apollo

MIN_QUANT_BITS = 2
MAX_QUANT_BITS = 8
MIN_SCALE_EPSILON = 1e-8


class QApollo(Apollo):
    """Quantized APOLLO: stores first/second moments in quantized form.

    Overrides the moment loading and storing methods of :class:`Apollo` to
    quantize the first and second moment buffers into uint8 tensors with
    per-tensor scale and offset.  Dequantization is performed on-the-fly
    before each update computation.

    Reference:
        Zhu, Z., Li, Y., Wang, Z., & Anandkumar, A. (2025). APOLLO: SGD-like
        Memory, Adam-like Performance. arXiv:2502.xxxxx.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate (default: 1e-3).
        rank: Rank of the Gaussian random projection subspace (default: 128).
        update_proj_gap: Number of steps between projector resampling
            (default: 200).
        scale: Global scale multiplier (default: 1.0).
        scale_type: Scaling granularity (default: ``"channel"``).
        proj_type: Projection side strategy (default: ``"std"``).
        betas: Coefficients for first and second moment running averages
            (default: ``(0.9, 0.999)``).
        eps: Term added for numerical stability (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 0.0).
        correct_bias: Whether to apply bias correction (default: True).
        scale_front: Whether to apply scale before norm limiter
            (default: False).
        disable_nl: Whether to disable norm-growth limiter (default: False).
        quant_bits: Number of bits for quantized state storage, clamped to
            [2, 8] (default: 8).

    Attributes:
        quant_bits (int): Effective number of quantization bits, clamped to
            the range [2, 8].
        Inherits all other attributes from :class:`Apollo`.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        rank=128,
        update_proj_gap=200,
        scale=1.0,
        scale_type="channel",
        proj_type="std",
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        correct_bias=True,
        scale_front=False,
        disable_nl=False,
        quant_bits=8,
    ):
        """Initializes the Q-APOLLO optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate (default: 1e-3).
            rank: Projection rank (default: 128).
            update_proj_gap: Projector resampling interval (default: 200).
            scale: Global scale multiplier (default: 1.0).
            scale_type: ``"channel"`` or ``"tensor"`` (default: ``"channel"``).
            proj_type: Projection side strategy (default: ``"std"``).
            betas: Momentum decay coefficients (default: ``(0.9, 0.999)``).
            eps: Numerical stability term (default: 1e-8).
            weight_decay: Decoupled weight decay coefficient (default: 0.0).
            correct_bias: Apply bias correction (default: True).
            scale_front: Apply scale before norm limiter (default: False).
            disable_nl: Disable norm-growth limiter (default: False).
            quant_bits: Quantization bit-width, clamped to [2, 8]
                (default: 8).
        """
        super().__init__(
            params=params,
            lr=lr,
            rank=rank,
            update_proj_gap=update_proj_gap,
            scale=scale,
            scale_type=scale_type,
            proj_type=proj_type,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
            scale_front=scale_front,
            disable_nl=disable_nl,
        )
        try:
            bits = int(quant_bits)
        except (TypeError, ValueError):
            warnings.warn(
                f"Invalid quant_bits={quant_bits!r}; falling back to {MAX_QUANT_BITS}.",
                RuntimeWarning,
            )
            bits = MAX_QUANT_BITS
        self.quant_bits = max(MIN_QUANT_BITS, min(bits, MAX_QUANT_BITS))

    def _quantize(self, value):
        """Quantizes a tensor to uint8 using uniform affine mapping.

        Args:
            value: Floating-point tensor to quantize.

        Returns:
            Tuple of ``(q, scale, v_min)`` where ``q`` is the uint8 quantized
            tensor, ``scale`` is the per-element step size, and ``v_min`` is
            the minimum value of the original tensor.
        """
        max_int = float((1 << self.quant_bits) - 1)
        v_min = value.min()
        v_max = value.max()
        scale = (v_max - v_min).clamp(min=MIN_SCALE_EPSILON) / max_int
        q = torch.clamp(torch.round((value - v_min) / scale), 0, max_int).to(torch.uint8)
        return q, scale, v_min

    @staticmethod
    def _dequantize(q_value, scale, v_min, dtype):
        """Dequantizes a uint8 tensor back to a floating-point tensor.

        Args:
            q_value: Quantized uint8 tensor.
            scale: Per-element step size from quantization.
            v_min: Minimum value of the original tensor.
            dtype: Target floating-point dtype.

        Returns:
            Dequantized floating-point tensor.
        """
        return q_value.to(dtype=dtype) * scale + v_min

    def _load_moments(self, state, grad_like):
        """Loads or initializes quantized first and second moment buffers.

        Dequantizes stored uint8 moment buffers if they exist and match the
        expected shape; otherwise returns zero-initialized buffers.

        Args:
            state: Per-parameter optimizer state dict.
            grad_like: Tensor whose shape and dtype are used for
                initialization.

        Returns:
            Tuple of ``(exp_avg, exp_avg_sq)`` dequantized floating-point
            tensors.
        """
        if "exp_avg_q" in state and "exp_avg_sq_q" in state:
            exp_avg = self._dequantize(
                state["exp_avg_q"],
                state["exp_avg_scale"],
                state["exp_avg_min"],
                grad_like.dtype,
            )
            exp_avg_sq = self._dequantize(
                state["exp_avg_sq_q"],
                state["exp_avg_sq_scale"],
                state["exp_avg_sq_min"],
                grad_like.dtype,
            )
            if exp_avg.shape == grad_like.shape and exp_avg_sq.shape == grad_like.shape:
                return exp_avg, exp_avg_sq

        return torch.zeros_like(grad_like), torch.zeros_like(grad_like)

    def _store_moments(self, state, exp_avg, exp_avg_sq):
        """Quantizes and stores the first and second moment buffers.

        Args:
            state: Per-parameter optimizer state dict.
            exp_avg: First moment tensor to quantize and store.
            exp_avg_sq: Second moment tensor to quantize and store.
        """
        q_avg, s_avg, m_avg = self._quantize(exp_avg)
        q_avg_sq, s_avg_sq, m_avg_sq = self._quantize(exp_avg_sq)
        state["exp_avg_q"] = q_avg
        state["exp_avg_scale"] = s_avg
        state["exp_avg_min"] = m_avg
        state["exp_avg_sq_q"] = q_avg_sq
        state["exp_avg_sq_scale"] = s_avg_sq
        state["exp_avg_sq_min"] = m_avg_sq
        state.pop("exp_avg", None)
        state.pop("exp_avg_sq", None)
