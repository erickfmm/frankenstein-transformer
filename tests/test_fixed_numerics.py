"""Tests for the fixed softmax / cross-entropy numerics (arXiv:2610.31642).

Covers:
  * forward parity with ``torch`` for softmax, log-softmax and cross-entropy;
  * the two bugs the paper fixes: Softmax Collapse (target/argmax gradient
    rounding to zero past the precision cliff, breaking the zero-sum gradient)
    and the softmax Jacobian in the same regime;
  * ``ignore_index`` / reduction semantics of the fixed cross-entropy;
  * per-module dispatch (``HybridLayer`` attaches ``use_fixed_softmax``);
  * full-model forward parity (fixed vs standard is mathematically identical);
  * KV-cache prefill+decode parity with the fixed softmax enabled;
  * the trainer's loss dispatch.

All tensors are tiny so the suite stays fast on CPU.
"""

import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F

    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_decoder import FrankensteinDecoder
    from src.model.frankenstein_encoder import FrankensteinEncoder
    from src.model.hybrid_layer import HybridLayer
    from src.model.numerics import (
        fixed_cross_entropy,
        fixed_log_softmax,
        fixed_softmax,
        module_log_softmax,
        module_softmax,
    )
    from src.training.trainer import TitanTrainer


def _cfg(**overrides):
    """Minimal encoder config; overrides win."""
    base = dict(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_loops=1,
        num_heads=4,
        num_kv_heads=2,
        dropout=0.0,
        norm_type="layer_norm",
        use_bitnet=False,
        use_moe=False,
        ffn_hidden_size=64,
        layer_pattern=["standard_attn", "titan_attn"],
        mode="encoder",
    )
    base.update(overrides)
    return FrankensteinModelConfig(**base)


class _DispatchModule(torch.nn.Module):
    """Tiny module carrying the opt-in attribute checked by the dispatchers."""

    def __init__(self, use_fixed_softmax):
        super().__init__()
        self.use_fixed_softmax = use_fixed_softmax


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ForwardParityTests(unittest.TestCase):
    def test_fixed_log_softmax_matches_torch(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                torch.manual_seed(0)
                logits = torch.randn(3, 5, 7, dtype=dtype) * 2.0
                torch.testing.assert_close(
                    fixed_log_softmax(logits, dim=-1),
                    F.log_softmax(logits, dim=-1),
                    rtol=1e-5 if dtype == torch.float32 else 1e-12,
                    atol=1e-6 if dtype == torch.float32 else 1e-13,
                )

    def test_fixed_softmax_matches_torch(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                torch.manual_seed(1)
                logits = torch.randn(2, 4, 9, dtype=dtype) * 2.0
                torch.testing.assert_close(
                    fixed_softmax(logits, dim=-1),
                    F.softmax(logits, dim=-1),
                    rtol=1e-5 if dtype == torch.float32 else 1e-12,
                    atol=1e-6 if dtype == torch.float32 else 1e-13,
                )

    def test_matches_torch_with_neg_inf_mask(self):
        logits = torch.randn(2, 3, 6)
        logits[0, 0, 3:] = float("-inf")
        torch.testing.assert_close(
            fixed_log_softmax(logits, dim=-1),
            F.log_softmax(logits, dim=-1),
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            fixed_softmax(logits, dim=-1), F.softmax(logits, dim=-1), rtol=1e-5, atol=1e-6
        )

    def test_single_key_softmax_is_one(self):
        logits = torch.tensor([[[2.5]]])
        out = fixed_softmax(logits, dim=-1)
        self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(out, torch.ones_like(out))

    def test_fixed_cross_entropy_matches_torch(self):
        torch.manual_seed(2)
        logits = torch.randn(6, 11) * 1.5
        targets = torch.randint(0, 11, (6,))
        for reduction in ("mean", "sum", "none"):
            with self.subTest(reduction=reduction):
                torch.testing.assert_close(
                    fixed_cross_entropy(logits, targets, reduction=reduction),
                    F.cross_entropy(logits, targets, reduction=reduction),
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_fixed_cross_entropy_ignore_index(self):
        torch.manual_seed(3)
        logits = torch.randn(2 * 4, 7)
        targets = torch.randint(0, 7, (2 * 4,))
        targets[5:] = -100
        torch.testing.assert_close(
            fixed_cross_entropy(logits, targets),
            F.cross_entropy(logits, targets, ignore_index=-100),
            rtol=1e-5,
            atol=1e-6,
        )
        # Ignored rows carry no gradient at all.
        logits = logits.clone().requires_grad_(True)
        fixed_cross_entropy(logits, targets).backward()
        self.assertEqual(int((logits.grad[5:] != 0).sum()), 0)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class SoftmaxCollapseTests(unittest.TestCase):
    """The paper's core regression: gradient magnitude past the precision cliff."""

    def _grads(self, fn, logits):
        x = logits.clone().requires_grad_(True)
        loss = fn(x)
        (grad,) = torch.autograd.grad(loss, x)
        return grad

    def test_cross_entropy_target_gradient_survives_cliff_fp32(self):
        # Margin 25 > fp32 cliff (~16.6). Standard CE zeroes the target
        # gradient while keeping the non-target one, so the gradient sum is
        # no longer zero; the fixed CE preserves both sum-zero and magnitude.
        logits = torch.tensor([[0.0, 25.0]])
        target = torch.tensor([1])
        std = self._grads(lambda z: F.cross_entropy(z, target), logits)
        fix = self._grads(lambda z: fixed_cross_entropy(z, target), logits)

        self.assertEqual(std[0, 1].item(), 0.0)
        self.assertGreater(std.sum().item(), 0.0)
        self.assertNotEqual(fix[0, 1].item(), 0.0)
        self.assertAlmostEqual(fix.sum().item(), 0.0, places=15)
        # True target gradient magnitude ~ e^-25 in fp32 (not rounded away).
        self.assertAlmostEqual(abs(fix[0, 1].item()), 1.3888e-11, delta=1e-13)

    def test_cross_entropy_target_gradient_survives_cliff_fp64(self):
        # Margin 40 > fp64 cliff (~36.7).
        logits = torch.tensor([[0.0, 40.0]], dtype=torch.float64)
        target = torch.tensor([1])
        std = self._grads(lambda z: F.cross_entropy(z, target), logits)
        fix = self._grads(lambda z: fixed_cross_entropy(z, target), logits)
        self.assertEqual(std[0, 1].item(), 0.0)
        self.assertNotEqual(fix[0, 1].item(), 0.0)
        self.assertAlmostEqual(fix.sum().item(), 0.0, places=30)

    def test_softmax_jacobian_survives_cliff_fp32(self):
        logits = torch.tensor([[0.0, 20.0]])
        target = torch.tensor([1])
        std = self._grads(lambda z: F.softmax(z, dim=-1)[0, 1], logits)
        fix = self._grads(lambda z: fixed_softmax(z, dim=-1)[0, 1], logits)
        self.assertEqual(std[0, 1].item(), 0.0)
        self.assertGreater(fix[0, 1].item(), 0.0)
        # d q_max / d z_max ~ e^-20 (fp32).
        self.assertAlmostEqual(fix[0, 1].item(), 2.0612e-9, delta=1e-12)

    def test_cross_entropy_loss_survives_cliff_forward(self):
        # Standard CE rounds the loss to 0 past the cliff; the fixed CE keeps
        # the analytic value log1p(e^-25).
        logits = torch.tensor([[0.0, 25.0]])
        target = torch.tensor([1])
        std = F.cross_entropy(logits, target).item()
        fix = fixed_cross_entropy(logits, target).item()
        self.assertEqual(std, 0.0)
        self.assertAlmostEqual(fix, 1.3888e-11, delta=1e-13)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class DispatchTests(unittest.TestCase):
    def test_module_dispatch_switches_implementation(self):
        logits = torch.tensor([[0.0, 20.0]])
        on = _DispatchModule(True)
        off = _DispatchModule(False)
        # Dispatch chooses by module attribute; compare against the explicit fns.
        torch.testing.assert_close(module_softmax(on, logits), fixed_softmax(logits))
        torch.testing.assert_close(module_softmax(off, logits), F.softmax(logits, dim=-1))
        torch.testing.assert_close(
            module_log_softmax(on, logits), fixed_log_softmax(logits)
        )
        torch.testing.assert_close(
            module_log_softmax(off, logits), F.log_softmax(logits, dim=-1)
        )

    def test_hybrid_layer_attaches_flag_to_mixers(self):
        for flag in (True, False):
            with self.subTest(fixed_softmax=flag):
                layer = HybridLayer(_cfg(fixed_softmax=flag), "standard_attn")
                self.assertEqual(layer.mixer.use_fixed_softmax, flag)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ModelIntegrationTests(unittest.TestCase):
    def _models(self):
        torch.manual_seed(7)
        plain = _cfg()
        torch.manual_seed(7)
        fixed = _cfg(fixed_softmax=True, fixed_cross_entropy=True)
        m_plain = FrankensteinEncoder(plain).eval()
        m_fixed = FrankensteinEncoder(fixed).eval()
        m_fixed.load_state_dict(m_plain.state_dict())
        return m_plain, m_fixed

    def test_encoder_outputs_are_mathematically_identical(self):
        m_plain, m_fixed = self._models()
        ids = torch.randint(0, 64, (2, 8))
        with torch.inference_mode():
            out_plain = m_plain(ids)
            out_fixed = m_fixed(ids)
        out_plain = out_plain[0] if isinstance(out_plain, (tuple, list)) else out_plain
        out_fixed = out_fixed[0] if isinstance(out_fixed, (tuple, list)) else out_fixed
        torch.testing.assert_close(out_fixed, out_plain, rtol=1e-5, atol=1e-5)

    def test_backward_is_finite_with_fixed_flags(self):
        _, m_fixed = self._models()
        ids = torch.randint(0, 64, (2, 8))
        logits = m_fixed(ids)
        loss = fixed_cross_entropy(logits.reshape(-1, logits.size(-1)), ids.reshape(-1))
        loss.backward()
        for name, param in m_fixed.named_parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(), msg=name)

    def test_gradient_sums_to_zero_over_vocab(self):
        _, m_fixed = self._models()
        ids = torch.randint(0, 64, (1, 4))
        logits = m_fixed(ids)
        logits = logits.detach().requires_grad_(True)
        # Push the logits towards the precision cliff, then check the
        # zero-sum property of the fixed CE gradient (paper Eq. 3).
        peaked = logits * 40.0
        loss = fixed_cross_entropy(peaked.reshape(-1, peaked.size(-1)), ids.reshape(-1))
        (grad,) = torch.autograd.grad(loss, peaked)
        self.assertLess(grad.sum().item(), 1e-6)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class TrainerDispatchTests(unittest.TestCase):
    def _bare_trainer(self, fixed):
        trainer = TitanTrainer.__new__(TitanTrainer)
        trainer.config = _cfg(fixed_cross_entropy=fixed)
        trainer.model = torch.nn.Identity()
        return trainer

    def test_trainer_uses_fixed_cross_entropy_when_enabled(self):
        logits = torch.tensor([[0.0, 25.0]])
        target = torch.tensor([1])
        on = self._bare_trainer(True)
        off = self._bare_trainer(False)
        self.assertTrue(on._use_fixed_cross_entropy())
        self.assertFalse(off._use_fixed_cross_entropy())
        self.assertEqual(off._task_cross_entropy(logits, target).item(), 0.0)
        self.assertGreater(on._task_cross_entropy(logits, target).item(), 0.0)

    def test_trainer_falls_back_to_model_config(self):
        trainer = TitanTrainer.__new__(TitanTrainer)
        trainer.config = None
        trainer.model = torch.nn.Module()
        trainer.model.config = _cfg(fixed_cross_entropy=True)
        self.assertTrue(trainer._use_fixed_cross_entropy())


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class CacheParityTests(unittest.TestCase):
    """Prefill + decode must equal the full forward with fixed softmax on."""

    def _build(self, pattern):
        cfg = FrankensteinModelConfig(
            vocab_size=64,
            hidden_size=32,
            num_layers=1,
            num_loops=1,
            num_heads=4,
            retention_heads=4,
            num_kv_heads=2,
            dropout=0.0,
            use_bitnet=False,
            layer_pattern=[pattern],
            mode="decoder",
            positional_encoding="rope",
            norm_type="layer_norm",
            ffn_hidden_size=64,
            fixed_softmax=True,
        )
        torch.manual_seed(0)
        return FrankensteinDecoder(cfg).eval()

    def test_standard_attn_cache_parity(self):
        for pattern in ("standard_attn", "mla_attn", "titan_attn"):
            with self.subTest(pattern=pattern):
                torch.manual_seed(42)
                m1 = self._build(pattern)
                m2 = self._build(pattern)
                m2.load_state_dict(m1.state_dict())
                prompt = torch.randint(0, 64, (1, 6))
                with torch.inference_mode():
                    full = m1.generate(prompt.clone(), max_new_tokens=3, greedy=True)
                    cached = m2.generate(
                        prompt.clone(), max_new_tokens=3, greedy=True, use_cache=True
                    )
                if not torch.equal(full, cached):
                    with torch.inference_mode():
                        torch.testing.assert_close(
                            m1(full), m2(cached), rtol=1e-4, atol=1e-5
                        )


if __name__ == "__main__":
    unittest.main()
