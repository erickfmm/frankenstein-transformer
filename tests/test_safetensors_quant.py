"""Tests for the safetensors block-quantization codecs.

Covers the exact Q8_0/Q4_0 block layouts (34/18 bytes per 32 weights),
reference-exact rounding/nibble order, i2_s ternary packing, payload sizing
and dequantization error bounds.
"""
import math
import unittest

import torch

from src.deploy.safetensors_quant import (
    BLOCK_SIZE,
    I2_S,
    PACKED_SUFFIX,
    Q4_0,
    Q8_0,
    SCALE_SUFFIX,
    bits_per_weight,
    dequantize_weight,
    packed_numel,
    quantize_weight,
)


class SafetensorsQuantCodecTests(unittest.TestCase):
    def test_block_sizes_and_bits_per_weight(self):
        self.assertEqual(BLOCK_SIZE, 32)
        self.assertAlmostEqual(bits_per_weight(Q8_0), 8.5)
        self.assertAlmostEqual(bits_per_weight(Q4_0), 4.5)
        self.assertAlmostEqual(bits_per_weight(I2_S), 2.0)

    def test_packed_numel(self):
        self.assertEqual(packed_numel((4, 64), Q8_0), 4 * 2 * 34)
        self.assertEqual(packed_numel((4, 64), Q4_0), 4 * 2 * 18)
        self.assertEqual(packed_numel((4, 64), I2_S), 256 // 4)

    def test_q8_0_block_layout_and_scale(self):
        weight = (torch.arange(32, dtype=torch.float32) - 16).reshape(1, 32)
        payload = quantize_weight(weight, Q8_0)
        packed = payload[PACKED_SUFFIX]
        self.assertEqual(packed.numel(), 34)
        self.assertNotIn(SCALE_SUFFIX, payload)
        scale = packed[:2].view(torch.float16).item()
        self.assertAlmostEqual(scale, 16.0 / 127.0, places=4)
        quants = packed[2:].view(torch.int8)
        self.assertEqual(int(quants[0]), -127)
        self.assertEqual(int(quants[16]), 0)
        self.assertEqual(int(quants[-1]), 119)

    def test_q8_0_round_trip_error_is_bounded(self):
        torch.manual_seed(0)
        weight = torch.randn(8, 64)
        payload = quantize_weight(weight, Q8_0)
        restored = dequantize_weight(
            payload[PACKED_SUFFIX], None, weight.shape, Q8_0
        )
        # Per-block max/127; allow the fp16 scale rounding on top.
        row_max = weight.reshape(-1, BLOCK_SIZE).abs().max(dim=1).values
        bound = (row_max / 127.0 * 1.01).max().item()
        self.assertLessEqual((restored - weight).abs().max().item(), bound)

    def test_q4_0_nibble_order_and_signed_scale(self):
        weight = torch.tensor(
            [[0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0,
              6.0, -6.0, 7.0, -7.0, 8.0] + [0.0] * 16]
        )
        payload = quantize_weight(weight, Q4_0)
        packed = payload[PACKED_SUFFIX]
        self.assertEqual(packed.numel(), 18)
        self.assertAlmostEqual(packed[:2].view(torch.float16).item(), -1.0, places=6)
        quants = packed[2:]
        # Low nibble is element j, high nibble is element j + 16.
        self.assertEqual(int(quants[0]) & 0xF, 8)  # 0 -> bin 8
        self.assertEqual(int(quants[0]) >> 4, 8)   # element 16 is 0
        self.assertEqual(int(quants[1]) & 0xF, 7)  # 1 -> 7 because d = -1
        restored = dequantize_weight(packed, None, weight.shape, Q4_0)
        torch.testing.assert_close(restored, weight, rtol=0.0, atol=1e-6)

    def test_q4_0_round_trip_error_is_bounded(self):
        torch.manual_seed(1)
        weight = torch.randn(8, 64)
        payload = quantize_weight(weight, Q4_0)
        restored = dequantize_weight(
            payload[PACKED_SUFFIX], None, weight.shape, Q4_0
        )
        row_max = weight.reshape(-1, BLOCK_SIZE).abs().max(dim=1).values
        # Q4_0 uses the signed maximum; the opposite-sign extreme can land
        # half a bin off (max/8), not the ideal max/16.
        bound = (row_max / 8.0 * 1.01).max().item()
        self.assertLessEqual((restored - weight).abs().max().item(), bound)

    def test_i2_s_packs_ternary_with_per_tensor_scale(self):
        weight = torch.tensor([[1.0, -1.0, 0.0, 0.5]])
        payload = quantize_weight(weight, I2_S)
        self.assertEqual(payload[PACKED_SUFFIX].numel(), 1)
        self.assertAlmostEqual(payload[SCALE_SUFFIX].item(), 0.625, places=6)
        restored = dequantize_weight(
            payload[PACKED_SUFFIX],
            payload[SCALE_SUFFIX],
            weight.shape,
            I2_S,
        )
        torch.testing.assert_close(
            restored, torch.tensor([[0.625, -0.625, 0.0, 0.625]])
        )

    def test_i2_s_keeps_baked_ternary_scale_exact(self):
        gamma = 2.0
        weight = torch.tensor([[gamma, -gamma, 0.0, gamma]])
        payload = quantize_weight(weight, I2_S)
        self.assertAlmostEqual(payload[SCALE_SUFFIX].item(), gamma, places=6)
        restored = dequantize_weight(
            payload[PACKED_SUFFIX],
            payload[SCALE_SUFFIX],
            weight.shape,
            I2_S,
        )
        torch.testing.assert_close(restored, weight)

    def test_zero_block_and_padding(self):
        weight = torch.zeros(1, 32)
        for fmt in (Q8_0, Q4_0):
            payload = quantize_weight(weight, fmt)
            restored = dequantize_weight(
                payload[PACKED_SUFFIX], None, weight.shape, fmt
            )
            self.assertEqual(float(restored.abs().sum()), 0.0)

        # i2_s pads the final byte when numel is not a multiple of four.
        odd = torch.ones(1, 6)
        payload = quantize_weight(odd, I2_S)
        self.assertEqual(payload[PACKED_SUFFIX].numel(), math.ceil(6 / 4))
        restored = dequantize_weight(
            payload[PACKED_SUFFIX], payload[SCALE_SUFFIX], odd.shape, I2_S
        )
        torch.testing.assert_close(restored, odd)

    def test_rejects_bad_shape_and_unknown_format(self):
        with self.assertRaises(ValueError):
            quantize_weight(torch.randn(2, 30), Q8_0)
        with self.assertRaises(ValueError):
            quantize_weight(torch.randn(2, 32), "q5_0")
        with self.assertRaises(ValueError):
            dequantize_weight(torch.zeros(34, dtype=torch.uint8), None, (1, 32), "q5_0")


if __name__ == "__main__":
    unittest.main()
