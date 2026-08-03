"""Tests for the best-effort GGUF (BitNet i2_s) exporter.

Covers the compatibility check (standard_attn-only accepted, hybrids and
non-bitnet rejected), the GGUF binary header (magic, version, tensor count),
and the i2_s ternary packing helper.
"""
import os
import struct
import tempfile
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import numpy as np
    import torch
    from src.model.tormented_bert_frankestein import (
        FrankensteinDecoder,
        TormentedBertFrankenstein,
        UltraConfig,
    )
    from src.deploy.bitnet_gguf_export import (
        GGUF_MAGIC,
        GGUFExportError,
        check_gguf_compatibility,
        export_bitnet_gguf,
        pack_i2_s,
    )


def _write_yaml(path, layer_pattern, use_bitnet=True):
    import yaml
    data = {
        "model": {
            "vocab_size": 64,
            "hidden_size": 64,
            "num_layers": 2,
            "num_heads": 4,
            "layer_pattern": layer_pattern,
            "use_bitnet": use_bitnet,
            "use_moe": False,
            "ffn_hidden_size": 128,
        }
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestBitNetGGUFExport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _make_checkpoint(self, mode="encoder"):
        cfg = UltraConfig(
            vocab_size=64, hidden_size=64, num_layers=2, num_heads=4,
            use_bitnet=True, use_moe=False, ffn_hidden_size=128,
            layer_pattern=["standard_attn"], mode=mode,
        )
        model = (FrankensteinDecoder(cfg) if mode == "decoder"
                 else TormentedBertFrankenstein(cfg))
        path = os.path.join(self.tmpdir, "c.pt")
        torch.save({"model_state_dict": model.state_dict()}, path)
        return path

    def test_compatibility_standard_attn_ok(self):
        yp = os.path.join(self.tmpdir, "ok.yaml")
        _write_yaml(yp, ["standard_attn"])
        res = check_gguf_compatibility(yp)
        self.assertTrue(res["is_compatible"])

    def test_compatibility_hybrid_rejected(self):
        yp = os.path.join(self.tmpdir, "hyb.yaml")
        _write_yaml(yp, ["standard_attn", "retnet"])
        res = check_gguf_compatibility(yp)
        self.assertFalse(res["is_compatible"])
        self.assertIn("standard_attn", res["reason"])

    def test_compatibility_non_bitnet_rejected(self):
        yp = os.path.join(self.tmpdir, "fp.yaml")
        _write_yaml(yp, ["standard_attn"], use_bitnet=False)
        res = check_gguf_compatibility(yp)
        self.assertFalse(res["is_compatible"])

    def test_export_header_and_tensors(self):
        ckpt = self._make_checkpoint()
        yp = os.path.join(self.tmpdir, "ok.yaml")
        _write_yaml(yp, ["standard_attn"])
        out = os.path.join(self.tmpdir, "m.gguf")
        res = export_bitnet_gguf(ckpt, yp, out)
        self.assertEqual(res["status"], "ok")
        self.assertGreater(res["n_tensors"], 0)
        with open(out, "rb") as f:
            head = f.read(12)
        magic, version, nt = struct.unpack("<III", head)
        self.assertEqual(magic, GGUF_MAGIC)
        self.assertEqual(version, 3)
        self.assertEqual(nt, res["n_tensors"])
        self.assertGreater(res["size_bytes"], 100)

    def test_export_decoder(self):
        ckpt = self._make_checkpoint(mode="decoder")
        yp = os.path.join(self.tmpdir, "ok.yaml")
        _write_yaml(yp, ["standard_attn"])
        out = os.path.join(self.tmpdir, "dec.gguf")
        res = export_bitnet_gguf(ckpt, yp, out)
        self.assertEqual(res["status"], "ok")

    def test_export_hybrid_raises(self):
        ckpt = self._make_checkpoint()
        yp = os.path.join(self.tmpdir, "hyb.yaml")
        _write_yaml(yp, ["standard_attn", "mamba"])
        out = os.path.join(self.tmpdir, "x.gguf")
        with self.assertRaises(GGUFExportError):
            export_bitnet_gguf(ckpt, yp, out)

    def test_pack_i2_s_round_trip(self):
        w = torch.tensor([[1.0, -1.0, 0.0, 1.0], [-1.0, 0.0, 0.0, 1.0]])
        packed, scale = pack_i2_s(w)
        self.assertAlmostEqual(scale, w.abs().mean().item(), places=5)
        # packed: 8 values -> 2 bytes
        self.assertEqual(len(packed), 2)
        # decode back
        vals = ((packed.astype(np.int32) & 0xFF) )
        # simple manual decode of first byte
        b = int(packed[0])
        fields = [(b >> (2 * i)) & 0b11 for i in range(4)]
        expected = [2, 0, 1, 2]  # 1->2, -1->0, 0->1, 1->2
        self.assertEqual(fields, expected)


if __name__ == "__main__":
    unittest.main()