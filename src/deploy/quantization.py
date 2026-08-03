#!/usr/bin/env python3
"""BitNet b1.58 quantization and compression utilities for TORMENTED-BERT.

Provides faithful ternary weight packing (``{-1, 0, 1}`` at ~1.58 bits per
weight), INT8 activation quantization, model size estimation, and fast
serialization/deserialization for quantized checkpoints.

Design contract
---------------
Only ``BitLinear`` modules (instances of
:class:`src.model.attention.common.BitLinear`) are packed to ternary. All
other parameters — full-precision ``nn.Linear`` weights (e.g. routing and
scoring projections when ``bitnet_routers`` is False), biases, embeddings,
LayerNorm/normalization parameters — are stored verbatim as float. This
correctly honours the ``use_bitnet`` / ``bitnet_routers`` schema flags because
the model itself decides at construction time which layers become
``BitLinear``.

Reference:
    Ma et al. (2024), "The Era of 1-bit LLMs: All Large Language Models are in
    1.58 Bits", arXiv:2402.17764.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Late import of BitLinear/is_bitlinear_module. Kept lazy so this module can be
# imported in contexts where the full model package is on sys.path either as a
# package (``src.model...``) or flat (``model...``).
# ---------------------------------------------------------------------------
def _get_bitlinear():
    try:
        from ..model.attention.common import BitLinear, is_bitlinear_module
        return BitLinear, is_bitlinear_module
    except ImportError:
        from model.attention.common import BitLinear, is_bitlinear_module  # type: ignore
        return BitLinear, is_bitlinear_module


def _map_param_to_module(model: nn.Module) -> Dict[str, nn.Module]:
    """Map each ``weight``/``bias`` parameter name to its owning module.

    Args:
        model: Root model to traverse.

    Returns:
        Dictionary mapping fully-qualified parameter names (e.g.
        ``"layers.0.mixer.q_proj.weight"``) to the ``nn.Module`` that owns
        them. Parameters without an owning module are omitted.
    """
    mapping: Dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        for param_name, _ in module.named_parameters(recurse=False):
            full = f"{name}.{param_name}" if name else param_name
            mapping[full] = module
    return mapping


def _is_bitnet_module(module: nn.Module) -> bool:
    """Return True if ``module`` is any BitNet-quantized layer.

    Covers both :class:`BitLinear` and :class:`BitConv1d` (the ternary
    Conv1d used by the factorized embedding pre-projection).
    """
    try:
        from ..model.attention.common import BitLinear, BitConv1d
    except ImportError:
        from model.attention.common import BitLinear, BitConv1d  # type: ignore
    return isinstance(module, (BitLinear, BitConv1d))


class BitNetQuantizer:
    """Quantization manager for BitNet ternary weight models.

    Handles packing/unpacking of ternary weights (``{-1, 0, 1}``) into
    2-bit-per-element byte arrays (4 weights per byte) and dequantization
    back to float tensors. Only ``BitLinear`` modules are quantized; every
    other parameter is preserved at full precision.

    Attributes:
        quantization_config: Dictionary describing the quantization scheme
            (``weight_bits`` = 1.58 ternary, ``activation_bits`` = 8).
    """

    def __init__(self):
        """Initialize the quantizer with default BitNet b1.58 config."""
        self.quantization_config = {
            'weight_bits': 1.58,  # Ternary: {-1, 0, 1}
            'activation_bits': 8,   # INT8 for activations
        }

    @staticmethod
    def quantize_ternary_weights(weight: torch.Tensor) -> Tuple[np.ndarray, float]:
        """Quantize weights to ternary {-1, 0, 1} and pack for storage.

        Uses the BitNet b1.58 per-tensor absmean scale, unless the weight is
        already ternary (baked), in which case the existing scale
        (``max(|w|)``) is preserved so the representation is reproduced
        bit-exactly.

        Args:
            weight: Float tensor to quantize (master or already-baked).

        Returns:
            Tuple of ``(packed_weights, scale)`` where ``packed_weights`` is a
            ``uint8`` array (2 bits per weight, 4 weights per byte) and
            ``scale`` is the scaling factor for dequantization.
        """
        if BitNetQuantizer._is_already_ternary(weight):
            scale = weight.abs().max().item()
            if scale < 1e-5:
                scale = 1.0
            ternary = torch.sign(weight).clamp(-1, 1).cpu().numpy()
        else:
            scale = weight.abs().mean().item()
            if scale < 1e-5:
                scale = 1.0
            w_scaled = weight / scale
            ternary = torch.round(w_scaled).clamp(-1, 1).cpu().numpy()
        packed = BitNetQuantizer._pack_ternary(ternary.flatten())
        return packed, scale

    @staticmethod
    def _is_already_ternary(weight: torch.Tensor, atol: float = 1e-6) -> bool:
        """Return True if every nonzero element equals ±max(|weight|).

        A baked BitLinear weight has values in ``{-gamma, 0, gamma}`` where
        ``gamma`` is the per-tensor scale. Detecting this lets the quantizer
        reproduce the representation bit-exactly instead of recomputing the
        absmean scale (which would shift ``gamma`` by the nonzero fraction).
        """
        w = weight.detach()
        if w.numel() == 0:
            return True
        gamma = w.abs().max().item()
        if gamma < atol:
            return True  # all-zero tensor
        nonzero = w[w.abs() > atol]
        if nonzero.numel() == 0:
            return True
        rel_err = (nonzero.abs() - gamma).abs().max().item() / gamma
        return rel_err < atol

    @staticmethod
    def _pack_ternary(ternary_array: np.ndarray) -> np.ndarray:
        """Pack ternary values into 2 bits each (4 values per byte).

        Mapping: ``-1 -> 0b00``, ``0 -> 0b01``, ``1 -> 0b10``.
        ``0b11`` is unused and reserved.
        """
        values = (ternary_array.astype(np.int32) + 1).astype(np.uint8)
        n = len(values)
        n_bytes = (n + 3) // 4  # Round up
        packed = np.zeros(n_bytes, dtype=np.uint8)
        # Place 4 fields per byte at bit offsets 0, 2, 4, 6.
        full = n - (n % 4)
        if full:
            reshaped = values[:full].reshape(-1, 4)
            byte = (
                reshaped[:, 0]
                | (reshaped[:, 1] << 2)
                | (reshaped[:, 2] << 4)
                | (reshaped[:, 3] << 6)
            ).astype(np.uint8)
            packed[: full // 4] = byte
        for i in range(full, n):
            byte_idx = i // 4
            bit_offset = (i % 4) * 2
            packed[byte_idx] |= np.uint8((values[i] & 0b11) << bit_offset)
        return packed

    @staticmethod
    def _unpack_ternary(packed: np.ndarray, original_size: int) -> np.ndarray:
        """Unpack ternary values from the 2-bit packed format."""
        packed = np.asarray(packed, dtype=np.uint8)
        n_bytes = (original_size + 3) // 4
        padded = np.zeros(max(n_bytes, len(packed)), dtype=np.uint8)
        padded[: len(packed)] = packed
        full = original_size - (original_size % 4)
        out = np.zeros(original_size, dtype=np.int32)
        if full:
            block = padded[: full // 4].astype(np.int32)
            fields = np.empty((full // 4, 4), dtype=np.int32)
            fields[:, 0] = block & 0b11
            fields[:, 1] = (block >> 2) & 0b11
            fields[:, 2] = (block >> 4) & 0b11
            fields[:, 3] = (block >> 6) & 0b11
            out[:full] = fields.reshape(-1) - 1
        for i in range(full, original_size):
            byte_idx = i // 4
            bit_offset = (i % 4) * 2
            value = int((padded[byte_idx] >> bit_offset) & 0b11)
            out[i] = value - 1  # Convert back to -1, 0, 1
        return out.astype(np.int8)

    @staticmethod
    def dequantize_ternary_weights(
        packed: np.ndarray,
        scale: float,
        original_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Dequantize packed ternary weights back to a float tensor.

        Args:
            packed: Packed ternary weights (``uint8``).
            scale: The absmean scaling factor.
            original_shape: Original tensor shape.

        Returns:
            Dequantized float tensor of ``original_shape``.
        """
        n_elements = int(np.prod(original_shape))
        ternary = BitNetQuantizer._unpack_ternary(packed, n_elements)
        weights = torch.from_numpy(ternary.astype(np.float32)) * scale
        return weights.reshape(original_shape)

    def quantize_model_weights(self, model: nn.Module) -> Dict[str, Any]:
        """Quantize all ``BitLinear`` weights in the model to ternary format.

        Only ``BitLinear`` modules are packed; every other parameter is stored
        verbatim as a float numpy array. This honours ``use_bitnet`` and
        ``bitnet_routers`` because the model itself decided which layers are
        ``BitLinear`` at construction time.

        Args:
            model: PyTorch model to quantize.

        Returns:
            Dictionary with ``weights``, ``scales``, ``shapes``,
            ``quantized_tensors`` (names of packed tensors), and ``config``.
        """
        quantized_state: Dict[str, Any] = {
            'weights': {},
            'scales': {},
            'shapes': {},
            'quantized_tensors': [],
            'full_precision_tensors': [],
            'config': self.quantization_config,
        }

        param_to_module = _map_param_to_module(model)

        total_original_size = 0
        total_compressed_size = 0

        for name, param in model.named_parameters():
            owner = param_to_module.get(name)
            if _is_bitnet_module(owner) and name.endswith('weight'):
                packed, scale = self.quantize_ternary_weights(param.data)
                quantized_state['weights'][name] = packed
                quantized_state['scales'][name] = scale
                quantized_state['shapes'][name] = tuple(param.shape)
                quantized_state['quantized_tensors'].append(name)
                original_size = param.numel() * 4  # FP32 = 4 bytes
                compressed_size = len(packed)
                total_original_size += original_size
                total_compressed_size += compressed_size
                logger.debug(
                    f"[ternary] {name}: {original_size / 1024:.2f}KB -> "
                    f"{compressed_size / 1024:.2f}KB"
                )
            else:
                quantized_state['weights'][name] = param.data.cpu().numpy()
                quantized_state['full_precision_tensors'].append(name)
                total_original_size += param.numel() * 4
                total_compressed_size += param.numel() * 4

        if total_compressed_size > 0:
            compression_ratio = total_original_size / total_compressed_size
            logger.info(
                f"Model packed: {total_original_size / (1024**2):.2f}MB -> "
                f"{total_compressed_size / (1024**2):.2f}MB "
                f"(effective compression {compression_ratio:.2f}x, "
                f"{len(quantized_state['quantized_tensors'])} ternary tensors, "
                f"{len(quantized_state['full_precision_tensors'])} full-precision tensors)"
            )
        return quantized_state

    def dequantize_model_weights(
        self,
        quantized_state: Dict[str, Any],
        model: nn.Module,
    ) -> None:
        """Load quantized weights back into a model.

        Ternary-packed tensors are dequantized; full-precision tensors are
        loaded verbatim. Missing keys are ignored (``strict=False``).

        Args:
            quantized_state: Dictionary produced by
                :meth:`quantize_model_weights`.
            model: PyTorch model to load weights into.
        """
        state_dict: Dict[str, torch.Tensor] = {}
        scales = quantized_state.get('scales', {})
        shapes = quantized_state.get('shapes', {})

        for name, payload in quantized_state['weights'].items():
            if name in scales:
                weight = self.dequantize_ternary_weights(
                    payload, scales[name], shapes[name]
                )
                state_dict[name] = weight
            else:
                state_dict[name] = torch.from_numpy(
                    np.asarray(payload)
                ).float() if not isinstance(payload, torch.Tensor) else payload

        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded {len(state_dict)} parameters into model")


class ActivationQuantizer:
    """Runtime activation quantization for inference optimization.

    Provides static methods for quantizing activations to INT8 and
    dequantizing back to float, enabling efficient integer-arithmetic
    computation during inference.
    """

    @staticmethod
    def quantize_activation_int8(x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Quantize activations to INT8 range ``[-128, 127]``.

        Args:
            x: Activation tensor of any shape.

        Returns:
            Tuple of ``(quantized_tensor, scale)`` where ``scale`` is the
            factor used for quantization (for later dequantization).
        """
        scale = 127.0 / x.abs().max().clamp(min=1e-5)
        x_q = (x * scale).round().clamp(-128, 127)
        return x_q, scale.item()

    @staticmethod
    def dequantize_activation_int8(
        x_q: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Dequantize INT8 activations back to float.

        Args:
            x_q: Quantized INT8 tensor.
            scale: The scale factor from :meth:`quantize_activation_int8`.

        Returns:
            Dequantized float tensor.
        """
        return x_q.float() / scale


def bake_bitnet_weights(model: nn.Module) -> int:
    """Bake ternary weights into every BitNet module in ``model``.

    Applies ``bake_ternary_weights()`` once to each BitNet-quantized layer
    (``BitLinear`` and ``BitConv1d``) so the stored master weight becomes the
    faithful ``{-1, 0, 1} * scale`` value (no STE). Used at export/deploy
    time to produce compact, self-describing ternary checkpoints.

    Args:
        model: Root model whose BitNet layers should be baked.

    Returns:
        Number of BitNet modules that were baked.
    """
    count = 0
    for module in model.modules():
        if _is_bitnet_module(module):
            module.bake_ternary_weights()
            count += 1
    logger.info(f"Baked {count} BitNet layers to faithful ternary weights")
    return count


def save_quantized_checkpoint(
    model: nn.Module,
    save_path: str,
    additional_data: Dict[str, Any] = None,
) -> None:
    """Save a model as a quantized (ternary-packed) checkpoint.

    Only ``BitLinear`` weights are packed; other parameters are stored as
    float. Additional metadata (config, tokenizer info, etc.) is merged in.

    Args:
        model: PyTorch model to save.
        save_path: Path to save the checkpoint.
        additional_data: Additional metadata to merge into the checkpoint.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Quantizing and saving model to {save_path}")

    quantizer = BitNetQuantizer()
    quantized_state = quantizer.quantize_model_weights(model)

    checkpoint: Dict[str, Any] = {
        'quantized_weights': quantized_state,
        'model_class': model.__class__.__name__,
    }
    if additional_data:
        checkpoint.update(additional_data)

    torch.save(checkpoint, save_path, pickle_protocol=4)

    file_size_mb = save_path.stat().st_size / (1024**2)
    logger.info(f"Checkpoint saved: {file_size_mb:.2f}MB")


def load_quantized_checkpoint(
    load_path: str,
    model: nn.Module,
) -> Dict[str, Any]:
    """Load a quantized checkpoint into a model.

    Args:
        load_path: Path to a quantized checkpoint.
        model: Model to load weights into.

    Returns:
        Dictionary with additional metadata from the checkpoint (everything
        except ``quantized_weights``).
    """
    logger.info(f"Loading quantized checkpoint from {load_path}")

    checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)
    quantized_state = checkpoint['quantized_weights']

    quantizer = BitNetQuantizer()
    quantizer.dequantize_model_weights(quantized_state, model)

    logger.info("Model weights loaded successfully")

    metadata = {k: v for k, v in checkpoint.items() if k != 'quantized_weights'}
    return metadata


def estimate_model_size(model: nn.Module) -> Dict[str, float]:
    """Estimate model size in different formats.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with size estimates in MB for fp32, fp16, and BitNet 1.58.
    """
    total_params = sum(p.numel() for p in model.parameters())
    sizes = {
        'fp32_mb': total_params * 4 / (1024**2),
        'fp16_mb': total_params * 2 / (1024**2),
        'bitnet_158_mb': total_params * 1.58 / 8 / (1024**2),  # 1.58 bits per param
    }
    return sizes


if __name__ == "__main__":
    logger.info("Testing BitNet quantization...")
    test_weight = torch.randn(1024, 1024)
    logger.info(f"Original weight size: {test_weight.numel() * 4 / 1024:.2f}KB")

    quantizer = BitNetQuantizer()
    packed, scale = quantizer.quantize_ternary_weights(test_weight)
    logger.info(f"Compressed size: {len(packed) / 1024:.2f}KB")

    reconstructed = quantizer.dequantize_ternary_weights(packed, scale, test_weight.shape)
    error = (test_weight - reconstructed).abs().mean()
    logger.info(f"Reconstruction error: {error:.6f}")
    logger.info(f"Compression ratio: {test_weight.numel() * 4 / len(packed):.2f}x")
