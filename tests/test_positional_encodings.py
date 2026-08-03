"""Unit tests for RoPE and HoPE positional encodings."""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from src.model.embeddings.rope import RoPE
    from src.model.embeddings.hope import HoPE


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class RoPETests(unittest.TestCase):
    def _make_input(self, bsz=2, heads=4, seq=8, head_dim=16):
        return torch.randn(bsz, heads, seq, head_dim)

    def test_output_shape_matches_input(self):
        rope = RoPE(head_dim=16)
        x = self._make_input()
        y = rope(x)
        self.assertEqual(y.shape, x.shape)

    def test_pair_dim_zero_returns_identity(self):
        # head_dim=1 → pair_dim=0, should return unchanged
        rope = RoPE(head_dim=1)
        x = self._make_input(head_dim=1)
        y = rope(x)
        self.assertTrue(torch.equal(y, x))

    def test_single_pair_dim_no_error(self):
        # head_dim=2 → pair_dim=1
        rope = RoPE(head_dim=2)
        x = self._make_input(head_dim=2)
        y = rope(x)
        self.assertEqual(y.shape, x.shape)

    def test_logical_layer_idx_accepted(self):
        rope = RoPE(head_dim=16)
        x = self._make_input()
        y = rope(x, logical_layer_idx=5)
        self.assertEqual(y.shape, x.shape)

    def test_dtype_preserved(self):
        rope = RoPE(head_dim=16)
        x = self._make_input().double()
        y = rope(x)
        self.assertEqual(y.dtype, torch.float64)

    def test_different_base_values(self):
        rope_small = RoPE(head_dim=16, base=1000.0)
        rope_large = RoPE(head_dim=16, base=100_000.0)
        x = self._make_input()
        y_small = rope_small(x)
        y_large = rope_large(x)
        # Different bases produce different outputs
        self.assertFalse(torch.allclose(y_small, y_large))

    def test_scaling_affects_output(self):
        rope1 = RoPE(head_dim=16, scaling=1.0)
        rope2 = RoPE(head_dim=16, scaling=2.0)
        x = self._make_input()
        y1 = rope1(x)
        y2 = rope2(x)
        self.assertFalse(torch.allclose(y1, y2))

    def test_gradient_flows(self):
        rope = RoPE(head_dim=16)
        x = self._make_input().requires_grad_(True)
        rope(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class HoPETests(unittest.TestCase):
    def _make_input(self, bsz=2, heads=4, seq=8, head_dim=16):
        return torch.randn(bsz, heads, seq, head_dim)

    def test_output_shape_matches_input(self):
        hope = HoPE(head_dim=16)
        x = self._make_input()
        y = hope(x)
        self.assertEqual(y.shape, x.shape)

    def test_pair_dim_zero_returns_identity(self):
        hope = HoPE(head_dim=1)
        x = self._make_input(head_dim=1)
        y = hope(x)
        self.assertTrue(torch.equal(y, x))

    def test_single_pair_dim_no_error(self):
        hope = HoPE(head_dim=2)
        x = self._make_input(head_dim=2)
        y = hope(x)
        self.assertEqual(y.shape, x.shape)

    def test_logical_layer_idx_changes_output(self):
        hope = HoPE(head_dim=16)
        x = self._make_input()
        y0 = hope(x, logical_layer_idx=0)
        y3 = hope(x, logical_layer_idx=3)
        # Different layer indices → different scaling → different output
        self.assertFalse(torch.allclose(y0, y3))

    def test_zero_logical_layer_idx(self):
        hope = HoPE(head_dim=16)
        x = self._make_input()
        y = hope(x, logical_layer_idx=0)
        self.assertEqual(y.shape, x.shape)

    def test_damping_affects_output(self):
        hope1 = HoPE(head_dim=16, damping=0.001)
        hope2 = HoPE(head_dim=16, damping=0.5)
        x = self._make_input()
        y1 = hope1(x)
        y2 = hope2(x)
        self.assertFalse(torch.allclose(y1, y2))

    def test_dtype_preserved(self):
        hope = HoPE(head_dim=16)
        x = self._make_input().double()
        y = hope(x)
        self.assertEqual(y.dtype, torch.float64)

    def test_gradient_flows(self):
        hope = HoPE(head_dim=16)
        x = self._make_input().requires_grad_(True)
        hope(x).sum().backward()
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
