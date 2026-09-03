"""Unit tests for the Falcon fast-weight attention family (arXiv:2608.27763)."""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from src.model.config import FrankensteinModelConfig
    from src.model.attention.gated import (
        Falcon1Attention,
        Falcon1AAttention,
        Falcon2Attention,
        Falcon2AAttention,
        Falcon3Attention,
        Falcon3AAttention,
    )


def _cfg(**overrides):
    base = dict(
        vocab_size=64,
        hidden_size=48,
        num_layers=1,
        num_loops=1,
        num_heads=6,
        retention_heads=6,
        num_experts=2,
        top_k_experts=1,
        dropout=0.0,
        norm_type="layer_norm",
        use_bitnet=False,
        layer_pattern=["standard_attn"],
        use_moe=False,
        ode_solver="rk4",
        ode_steps=1,
        ffn_hidden_size=96,
        ffn_activation="gelu",
        use_hope=True,
        mode="encoder",
    )
    base.update(overrides)
    return FrankensteinModelConfig(**base)


BSZ, SEQ, DIM = 2, 8, 48

FALCON_CLASSES = [
    ("falcon1", Falcon1Attention),
    ("falcon2", Falcon2Attention),
    ("falcon3", Falcon3Attention),
    ("falcon1a", Falcon1AAttention),
    ("falcon2a", Falcon2AAttention),
    ("falcon3a", Falcon3AAttention),
]


def _x():
    return torch.randn(BSZ, SEQ, DIM)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class FalconBasicTests(unittest.TestCase):
    """Shape / decoder / gradient / validation template for each variant."""

    def test_output_shapes(self):
        for name, cls in FALCON_CLASSES:
            attn = cls(_cfg(falcon_chunk_size=8))
            self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM), msg=name)

    def test_recurrent_output_shapes(self):
        for name, cls in FALCON_CLASSES:
            attn = cls(_cfg(falcon_chunk_size=None))
            self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM), msg=name)

    def test_decoder_mode(self):
        for name, cls in FALCON_CLASSES:
            attn = cls(_cfg(mode="decoder", falcon_chunk_size=8))
            self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM), msg=name)

    def test_gradient_flows(self):
        for name, cls in FALCON_CLASSES:
            for chunk in (None, 8):
                attn = cls(_cfg(falcon_chunk_size=chunk))
                x = _x().requires_grad_(True)
                attn(x).sum().backward()
                self.assertIsNotNone(x.grad, msg=f"{name} chunk={chunk}")
                self.assertTrue(
                    torch.isfinite(x.grad).all(), msg=f"{name} chunk={chunk}"
                )

    def test_invalid_hidden_raises(self):
        for name, cls in FALCON_CLASSES:
            with self.assertRaises(ValueError, msg=name):
                cls(_cfg(hidden_size=50, num_heads=6))

    def test_bitnet_projection_swap(self):
        from src.model.attention.common import BitLinear

        for name, cls in FALCON_CLASSES:
            attn = cls(_cfg(use_bitnet=True))
            self.assertIsInstance(attn.q_proj, BitLinear, msg=name)
            out = attn(_x())
            self.assertEqual(out.shape, (BSZ, SEQ, DIM), msg=name)

    def test_short_conv_toggle(self):
        for name, cls in FALCON_CLASSES:
            attn = cls(_cfg(falcon_short_conv=True))
            self.assertIsNotNone(attn.short_conv, msg=name)
            attn = cls(_cfg(falcon_short_conv=False))
            self.assertIsNone(attn.short_conv, msg=name)
            self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM), msg=name)

    def test_qk_norm_variants(self):
        for name, cls in FALCON_CLASSES:
            for norm in ("rms_norm", "l2_norm"):
                attn = cls(_cfg(falcon_qk_norm=norm))
                self.assertEqual(attn(_x()).shape, (BSZ, SEQ, DIM),
                                msg=f"{name}/{norm}")

    def test_gate_mode_combinations(self):
        combos = [
            dict(falcon_beta_mode="static", falcon_beta=1.2,
                 falcon_lambda_mode="static", falcon_lambda=0.3),
            dict(falcon_beta_mode="ctx_beta", falcon_lambda_mode="static",
                 falcon_lambda=0.1),
            dict(falcon_beta_mode="ctx_eta", falcon_lambda_mode="ctx"),
        ]
        for name, cls in FALCON_CLASSES:
            for combo in combos:
                attn = cls(_cfg(**combo))
                out = attn(_x())
                self.assertTrue(torch.isfinite(out).all(),
                                msg=f"{name}/{combo}")

    def test_window_validation(self):
        for name, cls in FALCON_CLASSES:
            with self.assertRaises(ValueError, msg=name):
                cls(_cfg(falcon_window=0))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class FalconChunkEquivalenceTests(unittest.TestCase):
    """The chunk-parallel kernels must match the recurrent reference."""

    def test_chunk_matches_recurrent(self):
        torch.manual_seed(0)
        x = _x()
        for name, cls in FALCON_CLASSES:
            for chunk in (3, 8, 64):
                for window in ((3,) if name in ("falcon3", "falcon3a") else (4,)):
                    attn = cls(_cfg(falcon_chunk_size=None, falcon_window=window))
                    attn.eval()
                    with torch.no_grad():
                        ref = attn(x)
                    attn.chunk_size = chunk
                    with torch.no_grad():
                        out = attn(x)
                    self.assertTrue(
                        torch.allclose(out, ref, atol=1e-4, rtol=1e-4),
                        msg=f"{name} chunk={chunk}",
                    )

    def test_chunk_matches_recurrent_with_decay(self):
        """Static gates with lambda > 0 exercise the decay rescaling."""
        torch.manual_seed(1)
        x = _x()
        for name, cls in FALCON_CLASSES:
            cfg = _cfg(
                falcon_chunk_size=None,
                falcon_beta_mode="static",
                falcon_beta=1.5,
                falcon_lambda_mode="static",
                falcon_lambda=0.7,
            )
            attn = cls(cfg)
            attn.eval()
            with torch.no_grad():
                ref = attn(x)
            attn.chunk_size = 3
            with torch.no_grad():
                out = attn(x)
            self.assertTrue(
                torch.allclose(out, ref, atol=1e-4, rtol=1e-4),
                msg=f"{name} decay chunk=3",
            )

    def test_long_sequence_chunking(self):
        """Multi-chunk processing with non-divisible lengths."""
        torch.manual_seed(2)
        for name, cls in FALCON_CLASSES:
            attn = cls(_cfg(falcon_chunk_size=None))
            attn.eval()
            x = torch.randn(2, 21, DIM)
            with torch.no_grad():
                ref = attn(x)
            attn.chunk_size = 5
            with torch.no_grad():
                out = attn(x)
            self.assertTrue(
                torch.allclose(out, ref, atol=1e-4, rtol=1e-4),
                msg=f"{name} long chunk=5",
            )


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class FalconSemanticsTests(unittest.TestCase):
    """Read-after-write semantics, causality and boundary sentinels."""

    def test_causality(self):
        """Perturbing future tokens must not change past outputs."""
        torch.manual_seed(3)
        for name, cls in FALCON_CLASSES:
            for chunk in (None, 4):
                attn = cls(_cfg(falcon_chunk_size=chunk)).eval()
                x = _x()
                with torch.no_grad():
                    a = attn(x.clone())
                    x2 = x.clone()
                    x2[:, 5:] += 10.0
                    b = attn(x2)
                self.assertTrue(
                    torch.allclose(a[:, :5], b[:, :5], atol=1e-5),
                    msg=f"{name} causality chunk={chunk}",
                )

    def test_first_token_output_is_zero(self):
        """At t=1 the fresh state is zero (x_1 = 0, eta_1 = 0, S_0 = 0):
        the read-after-write output o_1 = S_1^T q_1 must vanish before the
        output projection is applied."""
        torch.manual_seed(4)
        for name, cls in FALCON_CLASSES:
            for chunk in (None, 4):
                attn = cls(_cfg(falcon_chunk_size=chunk)).eval()
                with torch.no_grad():
                    q = attn.q_proj(_x()).view(BSZ, SEQ, 6, 8)
                    k = attn.k_proj(_x()).view(BSZ, SEQ, 6, 8)
                    v = attn.v_proj(_x()).view(BSZ, SEQ, 6, 8)
                    from src.model.attention.gated.falcon_common import (
                        falcon_qk_norm,
                    )

                    q, k = falcon_qk_norm(q, k, attn.qk_norm, attn.eps)
                    xf = torch.zeros_like(k)
                    xf[:, 1:] = k[:, :-1]
                    num, lam_bar = attn.gates(_x())
                    if chunk is None:
                        out = attn._recurrent_forward(q, xf, v, num, lam_bar)
                    else:
                        out = attn._chunk_forward(q, xf, v, num, lam_bar)
                self.assertTrue(
                    torch.allclose(out[:, 0], torch.zeros_like(out[:, 0]), atol=1e-6),
                    msg=f"{name} first-token sentinel chunk={chunk}",
                )

    def test_falcon1a_lambda_zero_is_additive(self):
        """With lambda = 0, Falcon-1A reduces to shifted additive linear
        attention: S_t = S_{t-1} + eta_t x_t v_t^T with o_t = S_t^T q_t.
        A manual reference scan must match the module output."""
        torch.manual_seed(5)
        attn = Falcon1AAttention(
            _cfg(falcon_chunk_size=None, falcon_short_conv=False,
                 falcon_beta_mode="static", falcon_beta=1.0,
                 falcon_lambda_mode="static", falcon_lambda=0.0)
        )
        attn.eval()
        x = _x()
        with torch.no_grad():
            q = attn.q_proj(x).view(BSZ, SEQ, 6, 8)
            k = attn.k_proj(x).view(BSZ, SEQ, 6, 8)
            v = attn.v_proj(x).view(BSZ, SEQ, 6, 8)
            from src.model.attention.gated.falcon_common import falcon_qk_norm

            q, k = falcon_qk_norm(q, k, "rms_norm", attn.eps)
            xf = torch.zeros_like(k)
            xf[:, 1:] = k[:, :-1]
            num = torch.ones(BSZ, SEQ, 6)
            lam = torch.zeros(BSZ, SEQ, 6)
            module_out = attn._recurrent_forward(q, xf, v, num, lam)

            state = torch.zeros(BSZ, 6, 8, 8)
            manual = []
            for t in range(SEQ):
                x_t = xf[:, t]
                energy = (x_t * x_t).sum(-1)
                eta = torch.zeros_like(energy) if t == 0 else num[:, t] / (
                    energy + attn.eps
                ).clamp_min(attn.eps)
                state = state + x_t.unsqueeze(-1) * (eta.unsqueeze(-1) * v[:, t]).unsqueeze(-2)
                manual.append((state * q[:, t].unsqueeze(-1)).sum(-2))
            manual = torch.stack(manual, dim=1)
        self.assertTrue(
            torch.allclose(module_out, manual, atol=1e-5),
            msg="Falcon-1A lambda=0 additive reduction",
        )


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class FalconConfigTests(unittest.TestCase):
    """Config-level validation of the falcon_* knobs."""

    def test_defaults(self):
        cfg = _cfg()
        self.assertEqual(cfg.falcon_chunk_size, 64)
        self.assertEqual(cfg.falcon_qk_norm, "rms_norm")
        self.assertEqual(cfg.falcon_beta_mode, "ctx_eta")
        self.assertEqual(cfg.falcon_lambda_mode, "ctx")
        self.assertEqual(cfg.falcon_beta, 1.0)
        self.assertEqual(cfg.falcon_lambda, 0.0)
        self.assertEqual(cfg.falcon_window, 4)
        self.assertTrue(cfg.falcon_short_conv)
        self.assertEqual(cfg.falcon_conv_kernel, 4)
        self.assertEqual(cfg.falcon_eps, 1e-6)
        self.assertEqual(cfg.falcon_eps_gamma, 1e-4)
        self.assertFalse(cfg.falcon1_attn_use_pe)
        self.assertFalse(cfg.falcon3a_attn_use_pe)

    def test_invalid_values_raise(self):
        for kwargs, msg in [
            (dict(falcon_chunk_size=0), "chunk_size"),
            (dict(falcon_qk_norm="bogus"), "qk_norm"),
            (dict(falcon_beta_mode="bogus"), "beta_mode"),
            (dict(falcon_lambda_mode="bogus"), "lambda_mode"),
            (dict(falcon_beta=2.5), "beta"),
            (dict(falcon_lambda=-0.1), "lambda"),
            (dict(falcon_window=0), "window"),
            (dict(falcon_conv_kernel=0), "conv_kernel"),
            (dict(falcon_eps=0.0), "eps"),
            (dict(falcon_eps_gamma=1.5), "eps_gamma"),
        ]:
            with self.assertRaises(ValueError, msg=msg):
                _cfg(**kwargs)


if __name__ == "__main__":
    unittest.main()