"""Unit tests for resolve_torch_device."""
import unittest
from importlib.util import find_spec
from unittest.mock import patch

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    from src.utils.device import resolve_torch_device, SUPPORTED_DEVICE_CHOICES


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ResolveDeviceTests(unittest.TestCase):
    def test_cpu_always_works(self):
        result = resolve_torch_device("cpu")
        self.assertEqual(result, "cpu")

    def test_auto_returns_cpu_when_no_gpu(self):
        with (
            patch("src.utils.device.torch.cuda.is_available", return_value=False),
            patch("src.utils.device.is_mps_available", return_value=False),
        ):
            result = resolve_torch_device("auto")
        self.assertEqual(result, "cpu")

    def test_auto_returns_cuda_when_available(self):
        with patch("src.utils.device.torch.cuda.is_available", return_value=True):
            result = resolve_torch_device("auto")
        self.assertEqual(result, "cuda")

    def test_auto_returns_mps_when_cuda_unavailable_but_mps_available(self):
        with (
            patch("src.utils.device.torch.cuda.is_available", return_value=False),
            patch("src.utils.device.is_mps_available", return_value=True),
        ):
            result = resolve_torch_device("auto")
        self.assertEqual(result, "mps")

    def test_cuda_raises_when_unavailable(self):
        with patch("src.utils.device.torch.cuda.is_available", return_value=False):
            with self.assertRaises(ValueError):
                resolve_torch_device("cuda")

    def test_mps_raises_when_unavailable(self):
        with patch("src.utils.device.is_mps_available", return_value=False):
            with self.assertRaises(ValueError):
                resolve_torch_device("mps")

    def test_unsupported_device_raises(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_torch_device("tpu")
        self.assertIn("tpu", str(ctx.exception))

    def test_none_treated_as_auto(self):
        # None -> "auto" -> falls through to cpu when nothing available
        with (
            patch("src.utils.device.torch.cuda.is_available", return_value=False),
            patch("src.utils.device.is_mps_available", return_value=False),
        ):
            result = resolve_torch_device(None)
        self.assertEqual(result, "cpu")

    def test_supported_choices_contains_expected(self):
        for choice in ("auto", "cpu", "cuda", "mps"):
            self.assertIn(choice, SUPPORTED_DEVICE_CHOICES)

    def test_case_insensitive_cpu(self):
        result = resolve_torch_device("CPU")
        self.assertEqual(result, "cpu")

    def test_cuda_accepted_when_available(self):
        with patch("src.utils.device.torch.cuda.is_available", return_value=True):
            result = resolve_torch_device("cuda")
        self.assertEqual(result, "cuda")

    def test_mps_accepted_when_available(self):
        with patch("src.utils.device.is_mps_available", return_value=True):
            result = resolve_torch_device("mps")
        self.assertEqual(result, "mps")


if __name__ == "__main__":
    unittest.main()
