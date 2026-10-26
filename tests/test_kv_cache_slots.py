"""KV-cache slot tests (plan §13, T1).

Covers every slot type: append/read/truncate/to(device), ring wrap-around
with ``positions`` metadata, ``StateSlot`` in-place update, conv-tail FIFO
order and token-tail contents.
"""

import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from src.model.cache.base import SlotMeta
    from src.model.cache.slots import (
        ConvTailSlot,
        KVSlot,
        LatentKVSlot,
        RingSlot,
        StateSlot,
        TokenTailSlot,
    )


def _meta(mixer="standard_attn", idx=0):
    return SlotMeta(
        mixer_key=mixer,
        logical_layer_idx=idx,
        num_heads=2,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class KVSlotTests(unittest.TestCase):
    def test_append_read_roundtrip(self):
        slot = KVSlot(_meta())
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        pos = torch.arange(3)
        slot.append(k=k, v=v, positions=pos)
        out = slot.read()
        self.assertEqual(slot.length, 3)
        self.assertTrue(torch.equal(out["k"], k))
        self.assertTrue(torch.equal(out["v"], v))
        self.assertTrue(torch.equal(out["positions"], pos))

    def test_append_incremental(self):
        slot = KVSlot(_meta())
        slot.append(k=torch.randn(1, 2, 2, 4), v=torch.randn(1, 2, 2, 4), positions=torch.arange(2))
        slot.append(k=torch.randn(1, 2, 1, 4), v=torch.randn(1, 2, 1, 4), positions=torch.tensor([2]))
        out = slot.read()
        self.assertEqual(out["k"].shape, (1, 2, 3, 4))
        self.assertEqual(slot.length, 3)

    def test_truncate_keeps_prefix(self):
        slot = KVSlot(_meta())
        k = torch.randn(1, 2, 5, 4)
        slot.append(k=k, v=torch.randn(1, 2, 5, 4), positions=torch.arange(5))
        slot.truncate(3)
        out = slot.read()
        self.assertEqual(slot.length, 3)
        self.assertTrue(torch.equal(out["k"], k[:, :, :3]))
        self.assertTrue(torch.equal(out["positions"], torch.arange(3)))

    def test_to_device_is_noop_on_cpu(self):
        slot = KVSlot(_meta())
        slot.append(k=torch.randn(1, 2, 2, 4), v=torch.randn(1, 2, 2, 4), positions=torch.arange(2))
        slot.to(torch.device("cpu"))
        self.assertEqual(slot.read()["k"].device.type, "cpu")

    def test_extras_are_stored_and_truncated(self):
        slot = KVSlot(_meta("fox_attn"))
        c = torch.randn(1, 2, 3)  # (B, H, T) FoX cumulative log-forget
        slot.append(
            k=torch.randn(1, 2, 3, 4),
            v=torch.randn(1, 2, 3, 4),
            positions=torch.arange(3),
            extras={"cum_log_f": c},
        )
        out = slot.read()
        self.assertTrue(torch.equal(out["extras"]["cum_log_f"], c))
        slot.truncate(2)
        out = slot.read()
        self.assertTrue(torch.equal(out["extras"]["cum_log_f"], c[:, :, :2]))

    def test_evict_oldest_keeps_sinks(self):
        slot = KVSlot(_meta(), num_sinks=2)
        for i in range(6):
            slot.append(
                k=torch.full((1, 2, 1, 4), float(i)),
                v=torch.full((1, 2, 1, 4), float(i)),
                positions=torch.tensor([i]),
            )
        slot.evict_oldest(2)
        out = slot.read()
        self.assertEqual(slot.length, 4)
        # First two (sinks) preserved; positions 2,3 evicted.
        self.assertEqual(out["positions"].tolist(), [0, 1, 4, 5])


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class RingSlotTests(unittest.TestCase):
    def test_wraparound_keeps_sinks_and_recent_positions(self):
        slot = RingSlot(_meta("swa_attn"), capacity=4, num_sinks=2)
        for i in range(6):
            slot.append(
                k=torch.full((1, 2, 1, 4), float(i)),
                v=torch.full((1, 2, 1, 4), float(i)),
                positions=torch.tensor([i]),
            )
        out = slot.read()
        self.assertEqual(slot.length, 4)
        # Pinned sinks 0,1 + the most recent positions.
        self.assertEqual(out["positions"].tolist(), [0, 1, 4, 5])
        self.assertEqual(out["k"][0, 0, :, 0].tolist(), [0.0, 1.0, 4.0, 5.0])

    def test_no_wrap_below_capacity(self):
        slot = RingSlot(_meta(), capacity=8)
        slot.append(k=torch.randn(1, 2, 3, 4), v=torch.randn(1, 2, 3, 4), positions=torch.arange(3))
        self.assertEqual(slot.read()["positions"].tolist(), [0, 1, 2])


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class LatentKVSlotTests(unittest.TestCase):
    def test_named_streams(self):
        slot = LatentKVSlot(_meta("mla_attn"), axes={"c_kv": "token"})
        c = torch.randn(1, 3, 6)  # (B, T, r)
        slot.append(positions=torch.arange(3), c_kv=c)
        slot.append(positions=torch.tensor([3]), c_kv=torch.randn(1, 1, 6))
        out = slot.read()
        self.assertEqual(out["c_kv"].shape, (1, 4, 6))
        self.assertTrue(torch.equal(out["c_kv"][:, :3], c))

    def test_heterogeneous_4d_stream(self):
        slot = LatentKVSlot(_meta("gta_attn"), axes={"k_g": "channel", "v_lat": "token"})
        slot.append(
            positions=torch.arange(2),
            k_g=torch.randn(1, 2, 2, 4),   # (B, G, T, D)
            v_lat=torch.randn(1, 2, 6),    # (B, T, r)
        )
        slot.append(
            positions=torch.tensor([2]),
            k_g=torch.randn(1, 2, 1, 4),
            v_lat=torch.randn(1, 1, 6),
        )
        out = slot.read()
        self.assertEqual(out["k_g"].shape, (1, 2, 3, 4))
        self.assertEqual(out["v_lat"].shape, (1, 3, 6))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class StateSlotTests(unittest.TestCase):
    def test_in_place_update(self):
        slot = StateSlot(_meta("gla_attn"))
        s1 = torch.randn(1, 2, 4, 4)
        slot.update(state=s1)
        self.assertTrue(torch.equal(slot.read()["state"], s1))
        s2 = s1 + 1
        slot.update(state=s2)
        self.assertTrue(torch.equal(slot.read()["state"], s2))
        self.assertEqual(slot.length, 0)

    def test_truncate_is_noop(self):
        slot = StateSlot(_meta())
        slot.update(state=torch.randn(1, 2, 4, 4))
        before = slot.read()["state"].clone()
        slot.truncate(0)
        self.assertTrue(torch.equal(slot.read()["state"], before))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ConvTailSlotTests(unittest.TestCase):
    def test_fifo_order(self):
        slot = ConvTailSlot(_meta("falcon1_attn"), capacity=3)
        for i in range(5):
            slot.append(torch.full((1, 1, 2), float(i)))
        out = slot.read()
        # Last three entries, in order.
        self.assertEqual(out.shape, (1, 3, 2))
        self.assertEqual(out[0, :, 0].tolist(), [2.0, 3.0, 4.0])

    def test_empty_read_is_none(self):
        slot = ConvTailSlot(_meta(), capacity=2)
        self.assertIsNone(slot.read())


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class TokenTailSlotTests(unittest.TestCase):
    def test_fifo_contents(self):
        slot = TokenTailSlot(capacity=4)
        slot.append(torch.tensor([[1, 2]]))
        slot.append(torch.tensor([[3]]))
        self.assertEqual(slot.read().tolist(), [[1, 2, 3]])
        for i in range(6):
            slot.append(torch.tensor([[10 + i]]))
        self.assertEqual(slot.read().tolist(), [[12, 13, 14, 15]])
        self.assertLessEqual(slot.read().size(-1), 4)


if __name__ == "__main__":
    unittest.main()
