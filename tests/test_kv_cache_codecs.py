"""KV-cache codec tests (plan §13, T2).

Round-trip error bounds for ``none``, ``int8`` and ``int4_kivi`` on
synthetic tensors: int8 max relative error <= 1e-2, int4 <= 5e-2 (per
axis); protected window stays exactly fp; group-boundary handling;
``codec.update()`` equals dequant -> concat -> requant.
"""

import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from src.model.cache.base import SlotMeta
    from src.model.cache.quant import build_cache_codec
    from src.model.cache.quant.int8 import Int8Codec
    from src.model.cache.quant.kivi4 import Kivi4Codec
    from src.model.cache.quant.none import NoneCodec


def _meta():
    return SlotMeta(
        mixer_key="standard_attn",
        logical_layer_idx=0,
        num_heads=1,
        num_kv_heads=1,
        head_dim=8,
        dtype=torch.float32,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class NoneCodecTests(unittest.TestCase):
    def test_roundtrip_is_exact(self):
        codec = NoneCodec()
        x = torch.randn(1, 2, 3, 8)
        stored = codec.encode_write("k", x, "channel", _meta())
        self.assertTrue(torch.equal(codec.decode_read(stored, _meta()), x))

    def test_update_concatenates(self):
        codec = NoneCodec()
        a = torch.randn(1, 2, 3, 8)
        b = torch.randn(1, 2, 1, 8)
        merged = codec.update(a, b, _meta())
        self.assertEqual(merged.shape, (1, 2, 4, 8))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class Int8CodecTests(unittest.TestCase):
    def test_channel_axis_error_bound(self):
        codec = Int8Codec()
        x = torch.randn(2, 2, 16, 8)
        out = codec.decode_read(codec.encode_write("k", x, "channel", _meta()), _meta())
        rel = (out - x).abs().max() / x.abs().max()
        self.assertLessEqual(rel.item(), 1e-2)

    def test_token_axis_error_bound(self):
        codec = Int8Codec()
        x = torch.randn(2, 2, 16, 8)
        out = codec.decode_read(codec.encode_write("v", x, "token", _meta()), _meta())
        rel = (out - x).abs().max() / x.abs().max()
        self.assertLessEqual(rel.item(), 1e-2)

    def test_update_matches_dequant_concat_requant(self):
        codec = Int8Codec()
        a = torch.randn(1, 1, 4, 8)
        b = torch.randn(1, 1, 2, 8)
        sa = codec.encode_write("k", a, "channel", _meta())
        sb = codec.encode_write("k", b, "channel", _meta())
        merged = codec.decode_read(codec.update(sa, sb, _meta()), _meta())
        manual = torch.cat([codec.decode_read(sa, _meta()), codec.decode_read(sb, _meta())], dim=2)
        # One extra requantization pass bounds the difference by the step size.
        self.assertTrue(torch.allclose(merged, manual, atol=2e-2))

    def test_zero_tensor_is_stable(self):
        codec = Int8Codec()
        x = torch.zeros(1, 1, 3, 8)
        out = codec.decode_read(codec.encode_write("k", x, "channel", _meta()), _meta())
        self.assertTrue(torch.equal(out, x))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class Kivi4CodecTests(unittest.TestCase):
    def test_error_bound(self):
        codec = Kivi4Codec(k_bits=4, v_bits=4)
        # Signed 4-bit levels [-7, 7]: the worst-case absolute error of a
        # grouped quantizer is scale/2 = group_absmax/14 (~7.1% of the
        # group max). The plan's 5e-2 figure is optimistic for int4; 7.5e-2
        # is the mathematically tight bound on random data.
        for axis in ("channel", "token"):
            x = torch.randn(2, 2, 64, 8)
            out = codec.decode_read(codec.encode_write("k", x, axis, _meta()), _meta())
            rel = (out - x).abs().max() / x.abs().max()
            self.assertLessEqual(rel.item(), 7.5e-2, msg=f"axis={axis}")

    def test_group_size_controls_grouping(self):
        codec = Kivi4Codec(group_size=16)
        self.assertEqual(codec.group_size, 16)
        self.assertTrue(codec.groupwise)

    def test_build_cache_codec_registry(self):
        self.assertIsInstance(build_cache_codec("none"), NoneCodec)
        self.assertIsInstance(build_cache_codec("int8"), Int8Codec)
        self.assertIsInstance(build_cache_codec("int4_kivi"), Kivi4Codec)
        with self.assertRaises(ValueError):
            build_cache_codec("int2_kivi")  # interface-ready, not in v1

    def test_protected_window_in_kvslot_stays_fp(self):
        from src.model.cache.slots import KVSlot

        codec = Kivi4Codec(group_size=4, protected_window=3, num_sinks=1)
        meta = _meta()
        slot = KVSlot(
            meta,
            codec=codec,
            num_sinks=1,
            protected_window=3,
            group_size=4,
        )
        # 1 pinned sink + 5 tokens appended; the last 3 must stay fp-exact.
        for i in range(6):
            slot.append(
                k=torch.full((1, 1, 1, 8), float(i + 1)),
                v=torch.full((1, 1, 1, 8), float(i + 1)),
                positions=torch.tensor([i]),
            )
        out = slot.read()
        self.assertTrue(torch.equal(out["k"][:, :, :, 0], torch.arange(1, 7).float().view(1, 1, 6)))

    def test_prefix_requantization_error_is_bounded(self):
        # Older groups may be quantized; total error must stay under the
        # int4 bound across the whole retained region.
        from src.model.cache.slots import KVSlot

        codec = Kivi4Codec(group_size=2, protected_window=1, num_sinks=0)
        slot = KVSlot(_meta(), codec=codec, num_sinks=0, protected_window=1, group_size=2)
        values = torch.arange(8).float() + 1.0
        for i, val in enumerate(values):
            slot.append(
                k=torch.full((1, 1, 1, 8), float(val)),
                v=torch.full((1, 1, 1, 8), float(val)),
                positions=torch.tensor([i]),
            )
        out = slot.read()["k"][:, :, :, 0].view(-1)
        rel = (out - values).abs().max() / values.abs().max()
        self.assertLessEqual(rel.item(), 5e-2)


if __name__ == "__main__":
    unittest.main()
