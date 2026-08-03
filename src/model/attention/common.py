"""Shared utilities for attention modules.

Provides quantization helpers (BitNet b1.58 ternary weights, 8-bit activations)
and the ``BitLinear`` layer.

Normalization layers (``DynamicTanhNorm``, ``Derf``) and the ``get_norm``
factory have been moved to ``src/model/norm/``.

References:
    Ma et al. (2024), "The Era of 1-bit LLMs: All Large Language Models are in
    1.58 Bits", arXiv:2402.17764.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_quant(x):
    """Quantize activations to 8-bit via straight-through estimator (STE).

    Scales activations to the range [-128, 127], rounds to integer, then
    rescales back. The STE passes gradients through the rounding operation
    unchanged, enabling end-to-end training with quantized activations.

    Args:
        x: Input tensor of any shape. The last dimension is used for
            per-token scaling.

    Returns:
        Tensor of same shape as ``x``, with values quantized to 8-bit
        precision. Gradients flow through the quantization via STE.
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    y = (x * scale).round().clamp(-128, 127) / scale
    return x + (y - x).detach()  # STE


def weight_quant(w):
    """Quantize weights to ternary values {-1, 0, 1} via straight-through estimator.

    Scales weights by the inverse of their mean absolute value, rounds to the
    nearest integer in [-1, 1], then rescales. The STE passes gradients through
    the rounding operation unchanged.

    Args:
        w: Weight tensor of any shape.

    Returns:
        Tensor of same shape as ``w``, with values quantized to ternary
        precision. Gradients flow through the quantization via STE.
    """
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    w_quant = (w * scale).round().clamp(-1, 1) / scale
    return w + (w_quant - w).detach()  # STE


class BitLinear(nn.Linear):
    """BitNet b1.58 linear layer with ternary weight quantization.

    A drop-in replacement for ``nn.Linear`` that reduces VRAM usage by 3-4x
    through ternary weight quantization and 8-bit activation quantization.
    Applies LayerNorm to inputs before quantization for stability, as
    prescribed by the BitNet b1.58 formulation.

    Reference:
        Ma et al. (2024), "The Era of 1-bit LLMs", arXiv:2402.17764.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If ``True``, adds a learnable bias to the output. Defaults to
            ``False``.
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias)

    def forward(self, x):
        """Forward pass with activation and weight quantization.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        x_norm = F.layer_norm(x, (x.shape[-1],))

        w_q = weight_quant(self.weight)
        x_q = activation_quant(x_norm)

        output = F.linear(x_q, w_q, self.bias)
        return output

    @torch.no_grad()
    def bake_ternary_weights(self):
        """Replace the master weight with its ternary-quantized value.

        Applies :func:`weight_quant` once and stores the result
        (``{-1, 0, 1} * scale``) as the learnable parameter, removing the
        straight-through estimator (STE). After baking, the weight holds the
        faithful ternary values and is no longer a full-precision master.

        Idempotent: if the weight is already ternary (all nonzero elements
        equal ``+/-max(|w|)``), the call is a no-op. This prevents the
        per-tensor absmean scale from drifting on repeated baking.

        Used at export/deployment time to produce compact, self-describing
        ternary checkpoints. Training should not continue after baking
        (the STE gradient path is gone).

        The bias (if any) is left untouched.
        """
        w = self.weight
        gamma = w.abs().max().item()
        if gamma < 1e-5:
            return  # all-zero (or near-zero) weight; nothing to bake
        nonzero = w.abs() > 1e-5
        if nonzero.any():
            rel_err = (w[nonzero].abs() - gamma).abs().max().item() / gamma
            if rel_err < 1e-5:
                # Already ternary; baking again would recompute the absmean
                # scale and shift the magnitude. Skip to stay idempotent.
                return
        w_quant = weight_quant(w)
        self.weight.copy_(w_quant)


class BitConv1d(nn.Conv1d):
    """BitNet b1.58 1D convolution with ternary weight quantization.

    A drop-in replacement for ``nn.Conv1d`` that applies the same ternary
    weight (``{-1, 0, 1}``) + int8 activation quantization as
    :class:`BitLinear`, via the straight-through estimator (STE). Inputs are
    LayerNorm-normalized over the channel dimension before activation
    quantization, mirroring the :class:`BitLinear` formulation.

    Used by :class:`FactorizedEmbedding` so the embedding Conv1d pre-projection
    also becomes ternary when ``use_bitnet`` is enabled.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Default: 1.
        padding: Padding added to both sides. Default: 0.
        groups: Number of blocked connections. Default: 1.
        bias: If ``True``, adds a learnable bias. Defaults to ``False``.

    Reference:
        Ma et al. (2024), "The Era of 1-bit LLMs", arXiv:2402.17764.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """Forward pass with activation and weight quantization.

        Args:
            x: Input tensor of shape ``(B, C_in, L)``.

        Returns:
            Output tensor of shape ``(B, C_out, L_out)``.
        """
        x_norm = F.layer_norm(x, (x.shape[1],))

        w_q = weight_quant(self.weight)
        x_q = activation_quant(x_norm)

        return F.conv1d(
            x_q,
            w_q,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    @torch.no_grad()
    def bake_ternary_weights(self):
        """Replace the master weight with its ternary-quantized value.

        Idempotent: a no-op if the weight is already ternary. See
        :meth:`BitLinear.bake_ternary_weights` for details.
        """
        w = self.weight
        gamma = w.abs().max().item()
        if gamma < 1e-5:
            return
        nonzero = w.abs() > 1e-5
        if nonzero.any():
            rel_err = (w[nonzero].abs() - gamma).abs().max().item() / gamma
            if rel_err < 1e-5:
                return
        w_quant = weight_quant(w)
        self.weight.copy_(w_quant)


def is_bitlinear_module(module: nn.Module) -> bool:
    """Return ``True`` if ``module`` is a BitNet-quantized layer.

    Covers both :class:`BitLinear` and :class:`BitConv1d` (the ternary
    Conv1d used by the factorized embedding pre-projection).

    Args:
        module: Any ``nn.Module``.

    Returns:
        Whether the module performs BitNet b1.58 ternary quantization.
    """
    return isinstance(module, (BitLinear, BitConv1d))



