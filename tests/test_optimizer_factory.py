"""Unit tests for the optimizer factory (build_optimizer)."""
import math
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from src.model.optimizer.factory import build_optimizer, OPTIMIZER_REGISTRY


def _param_groups(lr=1e-3, wd=0.01):
    """Build a minimal single-group param list for a tiny linear layer."""
    p = nn.Parameter(torch.randn(4, 4))
    return [{"params": [p], "lr": lr, "weight_decay": wd, "name": "other"}]


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class OptimizerRegistryTests(unittest.TestCase):
    def test_registry_is_nonempty(self):
        self.assertGreater(len(OPTIMIZER_REGISTRY), 0)

    def test_registry_contains_adamw(self):
        self.assertIn("adamw", OPTIMIZER_REGISTRY)

    def test_registry_contains_all_required(self):
        required = {"adamw", "sgd_momentum", "lion", "muon", "af_muon", "lamb", "anon", "apollo", "apollo_mini", "q_apollo"}
        for name in required:
            self.assertIn(name, OPTIMIZER_REGISTRY, f"Missing optimizer: {name}")


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerUnknownTests(unittest.TestCase):
    def test_unknown_class_raises(self):
        with self.assertRaises(ValueError) as ctx:
            build_optimizer("totally_unknown", _param_groups(), {})
        self.assertIn("totally_unknown", str(ctx.exception))

    def test_empty_class_raises(self):
        with self.assertRaises(ValueError):
            build_optimizer("", _param_groups(), {})

    def test_unknown_parameters_raise(self):
        with self.assertRaises(ValueError) as ctx:
            build_optimizer("adamw", _param_groups(), {"adamw-bad_param": 1.0})
        self.assertIn("bad_param", str(ctx.exception))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerAdamWTests(unittest.TestCase):
    def test_builds_without_error(self):
        opt = build_optimizer("adamw", _param_groups(), {})
        self.assertIsNotNone(opt)

    def test_step_runs(self):
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 1e-3, "name": "other"}]
        opt = build_optimizer("adamw", groups, {})
        loss = p.sum()
        loss.backward()
        opt.step()

    def test_per_group_lr_override(self):
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 1e-4, "name": "embeddings"}]
        params = {"adamw-lr_embeddings": 5e-5}
        opt = build_optimizer("adamw", groups, params)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 5e-5)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerSGDTests(unittest.TestCase):
    def test_builds_without_error(self):
        opt = build_optimizer("sgd_momentum", _param_groups(), {})
        self.assertIsNotNone(opt)

    def test_step_runs(self):
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 1e-2, "name": "other"}]
        opt = build_optimizer("sgd_momentum", groups, {})
        loss = p.sum()
        loss.backward()
        opt.step()

    def test_custom_momentum(self):
        groups = _param_groups()
        opt = build_optimizer("sgd_momentum", groups, {"sgd_momentum-momentum": 0.8})
        self.assertAlmostEqual(opt.defaults.get("momentum", opt.param_groups[0].get("momentum")), 0.8)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerLionTests(unittest.TestCase):
    def test_builds_without_error(self):
        opt = build_optimizer("lion", _param_groups(), {})
        self.assertIsNotNone(opt)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerLAMBTests(unittest.TestCase):
    def test_builds_without_error(self):
        opt = build_optimizer("lamb", _param_groups(), {})
        self.assertIsNotNone(opt)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerMuonTests(unittest.TestCase):
    def test_builds_without_error(self):
        opt = build_optimizer("muon", _param_groups(), {})
        self.assertIsNotNone(opt)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerRAdamTests(unittest.TestCase):
    def test_builds_without_error(self):
        opt = build_optimizer("radam", _param_groups(), {})
        self.assertIsNotNone(opt)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerProdigyTests(unittest.TestCase):
    def test_builds_without_error(self):
        opt = build_optimizer("prodigy", _param_groups(), {})
        self.assertIsNotNone(opt)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerAdafactorTests(unittest.TestCase):
    def test_builds_without_error(self):
        opt = build_optimizer("adafactor", _param_groups(), {})
        self.assertIsNotNone(opt)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerApolloTests(unittest.TestCase):
    def test_builds_without_error(self):
        opt = build_optimizer("apollo", _param_groups(), {})
        self.assertIsNotNone(opt)

    def test_step_runs(self):
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 1e-3, "name": "other"}]
        opt = build_optimizer("apollo", groups, {})
        loss = p.square().mean()
        loss.backward()
        opt.step()

    def test_integer_params_accept_string_and_invalid_fallback(self):
        opt = build_optimizer(
            "apollo",
            _param_groups(),
            {
                "apollo-rank": "16",
                "apollo-update_proj_gap": "50",
            },
        )
        self.assertEqual(opt.defaults["rank"], 16)
        self.assertEqual(opt.defaults["update_proj_gap"], 50)

        opt_bad = build_optimizer(
            "apollo",
            _param_groups(),
            {
                "apollo-rank": "bad_rank",
                "apollo-update_proj_gap": None,
            },
        )
        self.assertEqual(opt_bad.defaults["rank"], 128)
        self.assertEqual(opt_bad.defaults["update_proj_gap"], 200)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerApolloMiniTests(unittest.TestCase):
    def test_builds_without_error(self):
        opt = build_optimizer("apollo_mini", _param_groups(), {})
        self.assertIsNotNone(opt)

    def test_step_runs(self):
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 1e-3, "name": "other"}]
        opt = build_optimizer("apollo_mini", groups, {})
        loss = p.square().mean()
        loss.backward()
        opt.step()


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerQApolloTests(unittest.TestCase):
    def test_builds_without_error(self):
        opt = build_optimizer("q_apollo", _param_groups(), {})
        self.assertIsNotNone(opt)

    def test_step_runs(self):
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 1e-3, "name": "other"}]
        opt = build_optimizer("q_apollo", groups, {"q_apollo-quant_bits": 8})
        loss = p.square().mean()
        loss.backward()
        opt.step()

    def test_quant_bits_accepts_string_and_invalid_fallback(self):
        opt = build_optimizer("q_apollo", _param_groups(), {"q_apollo-quant_bits": "6"})
        self.assertEqual(opt.quant_bits, 6)

        opt_bad = build_optimizer("q_apollo", _param_groups(), {"q_apollo-quant_bits": "bad"})
        self.assertEqual(opt_bad.quant_bits, 8)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerAdanTests(unittest.TestCase):
    def test_builds_without_error(self):
        opt = build_optimizer("adan", _param_groups(), {})
        self.assertIsNotNone(opt)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerSophiaTests(unittest.TestCase):
    def test_builds_without_error(self):
        opt = build_optimizer("sophia", _param_groups(), {})
        self.assertIsNotNone(opt)

    def test_custom_rho(self):
        opt = build_optimizer("sophia", _param_groups(), {"sophia-rho": 0.02})
        self.assertIsNotNone(opt)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerAnonTests(unittest.TestCase):
    def test_builds_without_error(self):
        opt = build_optimizer("anon", _param_groups(), {})
        self.assertIsNotNone(opt)

    def test_step_runs(self):
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 1e-3, "name": "other"}]
        opt = build_optimizer("anon", groups, {})
        loss = p.square().mean()
        loss.backward()
        opt.step()

    def test_step_runs_negative_gamma(self):
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 1e-3, "name": "other"}]
        opt = build_optimizer("anon", groups, {"anon-gamma": -0.5})
        loss = p.square().mean()
        loss.backward()
        opt.step()

    def test_custom_gamma(self):
        opt = build_optimizer("anon", _param_groups(), {"anon-gamma": 1.5})
        self.assertAlmostEqual(opt.defaults["gamma"], 1.5)

    def test_gamma_default_zero(self):
        opt = build_optimizer("anon", _param_groups(), {})
        self.assertAlmostEqual(opt.defaults["gamma"], 0.0)

    def test_string_gamma_fallback(self):
        opt = build_optimizer("anon", _param_groups(), {"anon-gamma": "0.75"})
        self.assertAlmostEqual(opt.defaults["gamma"], 0.75)

    def test_non_convertible_gamma_fallback(self):
        opt = build_optimizer("anon", _param_groups(), {"anon-gamma": "bad"})
        self.assertAlmostEqual(opt.defaults["gamma"], 0.0)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerAFMuonTests(unittest.TestCase):
    """AF-Muon role routing, defaults and oracle invariants."""

    def _role_groups(self, lr=0.02, wd=0.1):
        emb = nn.Parameter(torch.randn(16, 8))
        matrix = nn.Parameter(torch.randn(8, 8))
        bias = nn.Parameter(torch.randn(8))
        groups = [
            {"params": [emb], "lr": lr, "weight_decay": wd, "name": "embeddings"},
            {"params": [matrix, bias], "lr": lr, "weight_decay": wd, "name": "attention"},
        ]
        return groups, emb, matrix, bias

    def _eta_tie(self, lr=0.02, d_model=8):
        return lr * 3000.0 / 50.0 * 0.5 / d_model

    def test_builds_without_error(self):
        groups, *_ = self._role_groups()
        opt = build_optimizer("af_muon", groups, {})
        self.assertIsNotNone(opt)

    def test_defaults_match_paper_protocol(self):
        groups, *_ = self._role_groups()
        d = build_optimizer("af_muon", groups, {}).defaults
        self.assertAlmostEqual(d["momentum"], 0.95)
        self.assertAlmostEqual(d["vector_lr"], 3e-4)
        self.assertAlmostEqual(d["tied_cap"], 3.0)
        self.assertAlmostEqual(d["tied_scale"], 0.5)
        self.assertAlmostEqual(d["rho_hidden"], 50.0)
        self.assertAlmostEqual(d["rho_output"], 3000.0)
        self.assertEqual(d["ns_steps"], 5)
        self.assertEqual(d["chunk_rows"], 2048)
        self.assertEqual(d["bisection_steps"], 32)
        self.assertEqual(d["max_bracket_steps"], 128)

    def test_step_runs_all_three_roles(self):
        groups, emb, matrix, bias = self._role_groups()
        opt = build_optimizer("af_muon", groups, {})
        before = [p.detach().clone() for p in (emb, matrix, bias)]
        loss = emb.square().mean() + matrix.square().mean() + bias.square().mean()
        loss.backward()
        opt.step()
        for param, prior in zip((emb, matrix, bias), before):
            self.assertFalse(torch.allclose(param.detach(), prior))

    def test_only_first_moment_state_is_stored(self):
        groups, emb, matrix, bias = self._role_groups()
        opt = build_optimizer("af_muon", groups, {})
        (emb.square().mean() + matrix.square().mean() + bias.square().mean()).backward()
        opt.step()
        for param in (emb, matrix, bias):
            self.assertEqual(set(opt.state[param]), {"momentum_buffer"})

    def test_tied_table_sign_endpoint(self):
        """``tied_cap=1`` saturates every coordinate at ``eta_tie``."""
        groups, emb, _, _ = self._role_groups()
        opt = build_optimizer("af_muon", groups, {"af_muon-tied_cap": 1.0})
        emb.grad = torch.randn_like(emb)
        before = emb.detach().clone()
        opt.step()
        delta = (emb.detach() - before).abs()
        eta = self._eta_tie(d_model=emb.shape[1])
        self.assertTrue(torch.allclose(delta, torch.full_like(delta, eta), atol=1e-6))

    def test_tied_table_row_rms_endpoint(self):
        """``tied_cap >= sqrt(d)`` yields unit-RMS rows scaled by ``eta_tie``."""
        groups, emb, _, _ = self._role_groups()
        opt = build_optimizer("af_muon", groups, {"af_muon-tied_cap": float("inf")})
        emb.grad = torch.randn_like(emb)
        before = emb.detach().clone()
        opt.step()
        row_rms = (emb.detach() - before).square().mean(dim=1).sqrt()
        eta = self._eta_tie(d_model=emb.shape[1])
        self.assertTrue(torch.allclose(row_rms, torch.full_like(row_rms, eta), atol=1e-6))

    def test_tied_table_cap_feasibility(self):
        """Finite caps bound both the row l2 radius and every coordinate."""
        groups, emb, _, _ = self._role_groups()
        cap = 3.0
        opt = build_optimizer("af_muon", groups, {"af_muon-tied_cap": cap})
        emb.grad = torch.randn_like(emb)
        before = emb.detach().clone()
        opt.step()
        delta = emb.detach() - before
        eta = self._eta_tie(d_model=emb.shape[1])
        self.assertTrue(
            (delta.norm(dim=1) <= eta * math.sqrt(emb.shape[1]) + 1e-6).all()
        )
        self.assertTrue((delta.abs() <= eta * cap + 1e-6).all())

    def test_tied_table_zero_rows_stay_zero(self):
        groups, emb, _, _ = self._role_groups()
        opt = build_optimizer("af_muon", groups, {})
        emb.grad = torch.zeros_like(emb)
        before = emb.detach().clone()
        opt.step()
        self.assertTrue(torch.equal(emb.detach(), before))

    def test_vector_update_is_rms_normalized(self):
        groups, _, _, bias = self._role_groups()
        opt = build_optimizer("af_muon", groups, {"af_muon-vector_lr": 1e-3})
        bias.grad = torch.randn_like(bias)
        before = bias.detach().clone()
        opt.step()
        delta = bias.detach() - before
        self.assertAlmostEqual(delta.square().mean().sqrt().item(), 1e-3, places=7)

    def test_tied_and_vector_params_skip_weight_decay(self):
        def step_with(wd):
            gen = torch.Generator().manual_seed(0)
            emb = nn.Parameter(torch.randn(16, 8, generator=gen))
            matrix = nn.Parameter(torch.randn(8, 8, generator=gen))
            bias = nn.Parameter(torch.randn(8, generator=gen))
            groups = [
                {"params": [emb], "lr": 0.02, "weight_decay": wd, "name": "embeddings"},
                {"params": [matrix], "lr": 0.02, "weight_decay": wd, "name": "attention"},
                {"params": [bias], "lr": 0.02, "weight_decay": wd, "name": "norms"},
            ]
            opt = build_optimizer("af_muon", groups, {})
            emb.grad = torch.ones_like(emb)
            matrix.grad = torch.ones_like(matrix)
            bias.grad = torch.ones_like(bias)
            before = [p.detach().clone() for p in (emb, matrix, bias)]
            opt.step()
            return [p.detach() - b for p, b in zip((emb, matrix, bias), before)]

        tied_lo, matrix_lo, vec_lo = step_with(0.0)
        tied_hi, matrix_hi, vec_hi = step_with(0.5)
        self.assertTrue(torch.allclose(tied_lo, tied_hi))
        self.assertTrue(torch.allclose(vec_lo, vec_hi))
        self.assertFalse(torch.allclose(matrix_lo, matrix_hi))

    def test_per_group_lr_for_new_block_groups(self):
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 1e-3, "name": "ffn"}]
        opt = build_optimizer("af_muon", groups, {"af_muon-lr_ffn": 0.02})
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.02)

    def test_unknown_group_suffix_raises(self):
        groups, *_ = self._role_groups()
        with self.assertRaises(ValueError) as ctx:
            build_optimizer("af_muon", groups, {"af_muon-lr_bogus": 1.0})
        self.assertIn("lr_bogus", str(ctx.exception))

    def test_string_coercion_and_invalid_fallback(self):
        groups, *_ = self._role_groups()
        opt = build_optimizer(
            "af_muon",
            groups,
            {
                "af_muon-tied_cap": "2",
                "af_muon-vector_lr": "1e-3",
                "af_muon-chunk_rows": "64",
                "af_muon-bisection_steps": "8",
            },
        )
        self.assertAlmostEqual(opt.defaults["tied_cap"], 2.0)
        self.assertAlmostEqual(opt.defaults["vector_lr"], 1e-3)
        self.assertEqual(opt.defaults["chunk_rows"], 64)
        self.assertEqual(opt.defaults["bisection_steps"], 8)

        opt_bad = build_optimizer(
            "af_muon",
            groups,
            {
                "af_muon-tied_cap": "bad",
                "af_muon-chunk_rows": "bad",
            },
        )
        self.assertAlmostEqual(opt_bad.defaults["tied_cap"], 3.0)
        self.assertEqual(opt_bad.defaults["chunk_rows"], 2048)

    def test_invalid_tied_cap_raises(self):
        groups, *_ = self._role_groups()
        with self.assertRaises(ValueError):
            build_optimizer("af_muon", groups, {"af_muon-tied_cap": 0.5})


if __name__ == "__main__":
    unittest.main()
