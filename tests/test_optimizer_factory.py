"""Unit tests for the optimizer factory (build_optimizer)."""
import math
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from src.model.optimizer.factory import build_optimizer, OPTIMIZER_REGISTRY
    from src.model.optimizer.root import _shape_coefficients


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
        required = {"adamw", "adam_pm", "sgd_momentum", "lion", "muon", "muon2", "af_muon", "ada_muon", "lamb", "anon", "apollo", "apollo_mini", "q_apollo", "dion3", "eve", "ea4llm", "root"}
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

    def test_default_betas_and_eps_present(self):
        opt = build_optimizer("muon", _param_groups(), {})
        self.assertEqual(opt.defaults["betas"], (0.9, 0.95))
        self.assertEqual(opt.defaults["eps"], 1e-8)

    def test_hidden_matrix_uses_muon_state(self):
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 0.02, "weight_decay": 0.0, "name": "attention"}]
        opt = build_optimizer("muon", groups, {})
        p.sum().backward()
        opt.step()
        state = opt.state[p]
        self.assertIn("momentum_buffer", state)
        self.assertNotIn("exp_avg", state)

    def test_embedding_group_uses_adamw_fallback(self):
        p = nn.Parameter(torch.randn(8, 4))
        groups = [{"params": [p], "lr": 1e-3, "weight_decay": 0.0, "name": "embeddings"}]
        opt = build_optimizer("muon", groups, {})
        p.sum().backward()
        opt.step()
        state = opt.state[p]
        self.assertIn("exp_avg", state)
        self.assertIn("exp_avg_sq", state)
        self.assertNotIn("momentum_buffer", state)

    def test_head_group_uses_adamw_fallback(self):
        p = nn.Parameter(torch.randn(8, 4))
        groups = [{"params": [p], "lr": 1e-3, "weight_decay": 0.0, "name": "head"}]
        opt = build_optimizer("muon", groups, {})
        p.sum().backward()
        opt.step()
        self.assertIn("exp_avg", opt.state[p])

    def test_vector_param_is_not_frozen(self):
        p = nn.Parameter(torch.ones(6))
        groups = [{"params": [p], "lr": 1e-3, "weight_decay": 0.0, "name": "norms"}]
        opt = build_optimizer("muon", groups, {})
        before = p.detach().clone()
        p.sum().backward()
        opt.step()
        self.assertFalse(torch.equal(before, p.detach()))
        self.assertIn("exp_avg", opt.state[p])
        self.assertFalse(torch.isnan(p).any())

    def test_high_rank_param_uses_adamw_fallback(self):
        kernel = nn.Parameter(torch.randn(4, 4, 3, 3))
        groups = [{"params": [kernel], "lr": 1e-3, "weight_decay": 0.0, "name": "attention"}]
        opt = build_optimizer("muon", groups, {})
        kernel.sum().backward()
        opt.step()
        self.assertIn("exp_avg", opt.state[kernel])
        self.assertFalse(torch.isnan(kernel).any())

    def test_fallback_matches_reference_adamw(self):
        gen = torch.Generator().manual_seed(0)
        ref = nn.Parameter(torch.randn(6, generator=gen))
        ours = nn.Parameter(ref.detach().clone())
        grad = torch.randn(6, generator=gen)

        ref.grad = grad.clone()
        ours.grad = grad.clone()
        reference = torch.optim.AdamW(
            [ref], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
        )
        groups = [{"params": [ours], "lr": 1e-3, "weight_decay": 0.01, "name": "norms"}]
        opt = build_optimizer("muon", groups, {})

        reference.step()
        opt.step()
        # First step has bias correction; both implementations must agree.
        self.assertTrue(torch.allclose(ours, ref, atol=1e-6, rtol=1e-5))

    def test_weight_decay_still_applies_to_muon_path(self):
        p = nn.Parameter(torch.ones(4, 4))
        groups = [{"params": [p], "lr": 0.02, "weight_decay": 0.1, "name": "ffn"}]
        opt = build_optimizer("muon", groups, {})
        opt.zero_grad()
        p.grad = torch.zeros_like(p)
        opt.step()
        self.assertTrue(torch.all(p < 1.0))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerTurboMuonTests(unittest.TestCase):
    def test_builds_without_error(self):
        opt = build_optimizer("turbo_muon", _param_groups(), {})
        self.assertIsNotNone(opt)

    def test_default_ns_steps_is_four(self):
        opt = build_optimizer("turbo_muon", _param_groups(), {})
        self.assertEqual(opt.defaults["ns_steps"], 4)

    def test_hidden_matrix_uses_muon_state(self):
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 0.02, "weight_decay": 0.0, "name": "attention"}]
        opt = build_optimizer("turbo_muon", groups, {})
        p.sum().backward()
        opt.step()
        self.assertIn("momentum_buffer", opt.state[p])
        self.assertNotIn("exp_avg", opt.state[p])

    def test_embedding_group_uses_adamw_fallback(self):
        p = nn.Parameter(torch.randn(8, 4))
        groups = [{"params": [p], "lr": 1e-3, "weight_decay": 0.0, "name": "embeddings"}]
        opt = build_optimizer("turbo_muon", groups, {})
        p.sum().backward()
        opt.step()
        self.assertIn("exp_avg", opt.state[p])
        self.assertNotIn("momentum_buffer", opt.state[p])

    def test_vector_param_is_not_frozen(self):
        p = nn.Parameter(torch.ones(6))
        groups = [{"params": [p], "lr": 1e-3, "weight_decay": 0.0, "name": "norms"}]
        opt = build_optimizer("turbo_muon", groups, {})
        before = p.detach().clone()
        p.sum().backward()
        opt.step()
        self.assertFalse(torch.equal(before, p.detach()))
        self.assertFalse(torch.isnan(p).any())


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerROOTTests(unittest.TestCase):
    def test_builds_without_error(self):
        opt = build_optimizer("root", _param_groups(), {})
        self.assertIsNotNone(opt)

    def test_step_runs(self):
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 1e-3, "name": "other"}]
        opt = build_optimizer("root", groups, {})
        loss = p.sum()
        loss.backward()
        opt.step()

    def test_default_betas_and_eps_present(self):
        opt = build_optimizer("root", _param_groups(), {})
        self.assertEqual(opt.defaults["betas"], (0.9, 0.95))
        self.assertEqual(opt.defaults["eps"], 1e-8)
        self.assertEqual(opt.defaults["quantile"], 0.90)
        self.assertTrue(opt.defaults["softthresh"])
        self.assertTrue(opt.defaults["ada_newton"])

    def test_hidden_matrix_uses_momentum_state(self):
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 0.02, "weight_decay": 0.0, "name": "attention"}]
        opt = build_optimizer("root", groups, {})
        p.sum().backward()
        opt.step()
        self.assertIn("momentum_buffer", opt.state[p])
        self.assertNotIn("exp_avg", opt.state[p])

    def test_embedding_group_uses_adamw_fallback(self):
        p = nn.Parameter(torch.randn(8, 4))
        groups = [{"params": [p], "lr": 1e-3, "weight_decay": 0.0, "name": "embeddings"}]
        opt = build_optimizer("root", groups, {})
        p.sum().backward()
        opt.step()
        self.assertIn("exp_avg", opt.state[p])
        self.assertNotIn("momentum_buffer", opt.state[p])

    def test_specific_suffixes_parsed(self):
        params = {
            "root-quantile": 0.95,
            "root-softthresh": False,
            "root-ada_newton": False,
            "root-ns_steps": 3,
        }
        opt = build_optimizer("root", _param_groups(), params)
        self.assertAlmostEqual(opt.defaults["quantile"], 0.95)
        self.assertFalse(opt.defaults["softthresh"])
        self.assertFalse(opt.defaults["ada_newton"])
        self.assertEqual(opt.defaults["ns_steps"], 3)

    def test_shape_coefficients_match_table1(self):
        self.assertEqual(_shape_coefficients(2048, 8192, True), (2.5925, -3.0010, 1.4138))
        self.assertEqual(_shape_coefficients(8192, 2048, True), (2.5925, -3.0010, 1.4138))
        self.assertEqual(_shape_coefficients(64, 64, False), (3.4445, -4.7750, 2.0315))
        self.assertEqual(_shape_coefficients(64, 64, True), (3.4445, -4.7750, 2.0315))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerMuon2Tests(unittest.TestCase):
    """Muon2 defaults, paper invariants and hybrid routing (arXiv:2604.09967)."""

    def test_builds_without_error(self):
        opt = build_optimizer("muon2", _param_groups(), {})
        self.assertIsNotNone(opt)

    def test_step_runs(self):
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 1e-3, "name": "other"}]
        opt = build_optimizer("muon2", groups, {})
        p.square().mean().backward()
        opt.step()
        self.assertFalse(torch.isnan(p).any())

    def test_step_runs_factored_flavor(self):
        p = nn.Parameter(torch.randn(4, 8))
        groups = [{"params": [p], "lr": 1e-2, "name": "attention"}]
        opt = build_optimizer("muon2", groups, {"muon2-factored": True})
        p.square().mean().backward()
        opt.step()
        self.assertIn("row_stats", opt.state[p])
        self.assertIn("col_stats", opt.state[p])
        self.assertNotIn("exp_avg_sq", opt.state[p])
        self.assertFalse(torch.isnan(p).any())

    def test_defaults_match_paper(self):
        opt = build_optimizer("muon2", _param_groups(), {})
        self.assertEqual(opt.defaults["ns_steps"], 5)
        self.assertAlmostEqual(opt.defaults["ns_eps"], 1e-7)
        self.assertFalse(opt.defaults["factored"])
        self.assertEqual(opt.defaults["betas"], (0.9, 0.95))

    def test_hidden_matrix_uses_muon2_state(self):
        # Eqs. 2-3: exp_avg (M) and exp_avg_sq (V); no plain momentum_buffer.
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 0.02, "weight_decay": 0.0, "name": "attention"}]
        opt = build_optimizer("muon2", groups, {})
        p.sum().backward()
        opt.step()
        state = opt.state[p]
        self.assertIn("exp_avg", state)
        self.assertIn("exp_avg_sq", state)
        self.assertNotIn("momentum_buffer", state)

    def test_embedding_group_uses_adamw_fallback(self):
        p = nn.Parameter(torch.randn(8, 4))
        groups = [{"params": [p], "lr": 1e-3, "weight_decay": 0.0, "name": "embeddings"}]
        opt = build_optimizer("muon2", groups, {})
        p.sum().backward()
        opt.step()
        self.assertIn("exp_avg", opt.state[p])
        self.assertIn("exp_avg_sq", opt.state[p])

    def test_update_scale_uses_sqrt_m_over_n(self):
        """Eq. 6: transposed shapes differ only by the eta*sqrt(m/n) scale."""
        gen = torch.Generator().manual_seed(0)
        tall = nn.Parameter(torch.randn(8, 2, generator=gen))
        grad = torch.randn(8, 2, generator=gen)
        wide = nn.Parameter(tall.detach().t().contiguous().clone())
        tall.grad = grad.clone()
        wide.grad = grad.t().contiguous().clone()

        groups_tall = [
            {"params": [tall], "lr": 1e-2, "weight_decay": 0.0, "name": "attention"}
        ]
        groups_wide = [
            {"params": [wide], "lr": 1e-2, "weight_decay": 0.0, "name": "attention"}
        ]
        opt_tall = build_optimizer("muon2", groups_tall, {})
        opt_wide = build_optimizer("muon2", groups_wide, {})

        before_tall = tall.detach().clone()
        before_wide = wide.detach().clone()
        opt_tall.step()
        opt_wide.step()

        delta_tall = (tall.detach() - before_tall).norm().item()
        delta_wide = (wide.detach() - before_wide).norm().item()
        # scale(tall) = sqrt(2/8), scale(wide) = sqrt(8/2); their ratio is 1/4.
        self.assertAlmostEqual(delta_tall / delta_wide, 0.25, delta=0.02)

    def test_specific_suffixes_parsed(self):
        opt = build_optimizer(
            "muon2",
            _param_groups(),
            {"muon2-ns_steps": "3", "muon2-ns_eps": "1e-6", "muon2-factored": True},
        )
        self.assertEqual(opt.defaults["ns_steps"], 3)
        self.assertAlmostEqual(opt.defaults["ns_eps"], 1e-6)
        self.assertTrue(opt.defaults["factored"])

        opt_bad = build_optimizer(
            "muon2",
            _param_groups(),
            {"muon2-ns_steps": "bad", "muon2-ns_eps": "bad"},
        )
        self.assertEqual(opt_bad.defaults["ns_steps"], 5)
        self.assertAlmostEqual(opt_bad.defaults["ns_eps"], 1e-7)

    def test_unknown_group_suffix_raises(self):
        with self.assertRaises(ValueError) as ctx:
            build_optimizer("muon2", _param_groups(), {"muon2-lr_bogus": 1.0})
        self.assertIn("lr_bogus", str(ctx.exception))

    def test_per_group_lr_routes(self):
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 1e-3, "name": "ffn"}]
        opt = build_optimizer("muon2", groups, {"muon2-lr_ffn": 0.02})
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.02)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerAdamPMTests(unittest.TestCase):
    """AdamPM pooled second moments (arXiv:2610.24598)."""

    def test_builds_without_error(self):
        opt = build_optimizer("adam_pm", _param_groups(), {})
        self.assertIsNotNone(opt)
        self.assertEqual(opt.defaults["group_size"], 8)
        self.assertEqual(opt.defaults["pool_dim"], "row")
        self.assertEqual(opt.defaults["betas"], (0.9, 0.95))

    def test_step_runs(self):
        p = nn.Parameter(torch.randn(8, 16))
        v = nn.Parameter(torch.randn(16))
        groups = [{"params": [p, v], "lr": 1e-3, "name": "other"}]
        opt = build_optimizer("adam_pm", groups, {})
        for _ in range(3):
            opt.zero_grad()
            (p.square().mean() + v.square().mean()).backward()
            opt.step()
        self.assertFalse(torch.isnan(p).any())
        self.assertFalse(torch.isnan(v).any())

    def test_second_moment_state_is_pooled(self):
        # Fig. 1 / Sec. 3.3: the persistent second-moment buffer is (m, ceil(n/s)).
        p = nn.Parameter(torch.randn(4, 16))
        groups = [{"params": [p], "lr": 1e-3, "name": "other"}]
        opt = build_optimizer("adam_pm", groups, {"adam_pm-group_size": 8})
        p.sum().backward()
        opt.step()
        state = opt.state[p]
        self.assertIn("exp_avg", state)
        self.assertIn("exp_avg_pooled", state)
        self.assertNotIn("exp_avg_sq", state)
        self.assertEqual(tuple(state["exp_avg"].shape), (4, 16))
        self.assertEqual(tuple(state["exp_avg_pooled"].shape), (4, 2))

    def test_column_pooling_state_shapes(self):
        # AdamPM-col pools within each column: state is (ceil(m/s), n).
        p = nn.Parameter(torch.randn(16, 4))
        groups = [{"params": [p], "lr": 1e-3, "name": "attention"}]
        opt = build_optimizer("adam_pm", groups, {"adam_pm-group_size": 4, "adam_pm-pool_dim": "col"})
        p.sum().backward()
        opt.step()
        self.assertEqual(tuple(opt.state[p]["exp_avg_pooled"].shape), (4, 4))

    def test_specific_suffixes_parsed(self):
        opt = build_optimizer(
            "adam_pm",
            _param_groups(),
            {"adam_pm-group_size": "4", "adam_pm-pool_dim": "COL"},
        )
        self.assertEqual(opt.defaults["group_size"], 4)
        self.assertEqual(opt.defaults["pool_dim"], "col")

    def test_invalid_group_size_falls_back(self):
        # The factory coerces unparsable ints to the documented default.
        opt = build_optimizer("adam_pm", _param_groups(), {"adam_pm-group_size": "bad"})
        self.assertEqual(opt.defaults["group_size"], 8)

    def test_invalid_pool_dim_raises(self):
        with self.assertRaises(ValueError):
            build_optimizer("adam_pm", _param_groups(), {"adam_pm-pool_dim": "diag"})

    def test_s1_matches_vanilla_adam(self):
        # s = 1 reduces AdamPM to Adam (paper: "AdamPM and Adam (s = 1)").
        torch.manual_seed(0)
        base = torch.randn(4, 6)
        grad = torch.randn(4, 6)

        p_ref = nn.Parameter(base.clone())
        ref = torch.optim.Adam([p_ref], lr=1e-2, betas=(0.9, 0.95), eps=1e-8)
        p_ref.grad = grad.clone()
        ref.step()

        p_pm = nn.Parameter(base.clone())
        groups = [{"params": [p_pm], "lr": 1e-2, "weight_decay": 0.0, "name": "other"}]
        opt = build_optimizer("adam_pm", groups, {"adam_pm-group_size": 1})
        p_pm.grad = grad.clone()
        opt.step()

        torch.testing.assert_close(p_pm, p_ref, atol=1e-6, rtol=1e-6)

    def test_partial_group_mean_pads_with_mean(self):
        # Sec. 3.3 padding rule: the final incomplete group is averaged over
        # the entries it actually contains (sum / count, not sum / s).
        p = nn.Parameter(torch.zeros(1, 13))
        groups = [{"params": [p], "lr": 0.0, "weight_decay": 0.0, "name": "other"}]
        opt = build_optimizer("adam_pm", groups, {"adam_pm-group_size": 8})
        grad = torch.arange(1.0, 14.0).reshape(1, 13)
        p.grad = grad.clone()
        opt.step()

        pooled = opt.state[p]["exp_avg_pooled"]
        self.assertEqual(tuple(pooled.shape), (1, 2))
        squared = grad.square()
        scale = 1.0 - 0.95  # first step of the compressed second-moment EMA
        self.assertAlmostEqual(
            pooled[0, 0].item(), scale * squared[0, :8].mean().item(), places=5
        )
        # Last group holds 5 entries: mean of 9^2..13^2, not sum / 8.
        self.assertAlmostEqual(
            pooled[0, 1].item(), scale * squared[0, 8:].mean().item(), places=5
        )

    def test_unknown_group_suffix_raises(self):
        with self.assertRaises(ValueError) as ctx:
            build_optimizer("adam_pm", _param_groups(), {"adam_pm-lr_bogus": 1.0})
        self.assertIn("lr_bogus", str(ctx.exception))


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

    def test_step_runs_matrix_and_vector(self):
        matrix = nn.Parameter(torch.randn(4, 4))
        vector = nn.Parameter(torch.randn(4))
        groups = [{"params": [matrix, vector], "lr": 1e-3, "name": "other"}]
        opt = build_optimizer("adafactor", groups, {})
        (matrix.square().mean() + vector.square().mean()).backward()
        opt.step()
        self.assertFalse(torch.isnan(matrix).any())
        self.assertFalse(torch.isnan(vector).any())

    def test_default_group_eps_is_tuple(self):
        opt = build_optimizer("adafactor", _param_groups(), {})
        self.assertEqual(opt.param_groups[0]["eps"], (1e-30, 1e-3))

    def test_eps_suffixes_reach_groups_as_tuple(self):
        opt = build_optimizer(
            "adafactor",
            _param_groups(),
            {"adafactor-eps1": 1e-25, "adafactor-eps2": 1e-2},
        )
        self.assertEqual(opt.param_groups[0]["eps"], (1e-25, 1e-2))


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


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerAdaMuonTests(unittest.TestCase):
    """AdaMuon defaults, paper invariants and per-group routing."""

    def test_builds_without_error(self):
        opt = build_optimizer("ada_muon", _param_groups(), {})
        self.assertIsNotNone(opt)

    def test_step_runs(self):
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 1e-3, "name": "other"}]
        opt = build_optimizer("ada_muon", groups, {})
        loss = p.square().mean()
        loss.backward()
        opt.step()

    def test_defaults_match_paper(self):
        opt = build_optimizer("ada_muon", _param_groups(), {})
        self.assertAlmostEqual(opt.defaults["momentum"], 0.95)
        self.assertEqual(opt.defaults["ns_steps"], 5)
        self.assertAlmostEqual(opt.defaults["ns_eps"], 1e-7)

    def test_step_skips_vector_params(self):
        p = nn.Parameter(torch.randn(8))
        groups = [{"params": [p], "lr": 1e-3, "name": "norms"}]
        opt = build_optimizer("ada_muon", groups, {})
        before = p.detach().clone()
        p.square().mean().backward()
        opt.step()
        self.assertTrue(torch.equal(p.detach(), before))

    def test_update_norm_matches_rms_alignment(self):
        """Lemma 2: ||gamma * O_hat||_F = 0.2 * sqrt(mn)."""
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 1e-3, "weight_decay": 0.0, "name": "other"}]
        opt = build_optimizer("ada_muon", groups, {})
        before = p.detach().clone()
        p.square().mean().backward()
        opt.step()
        delta_norm = (p.detach() - before).norm().item()
        expected = 1e-3 * 0.2 * math.sqrt(4 * 4)
        self.assertAlmostEqual(delta_norm, expected, places=6)

    def test_per_group_lr_routes_new_blocks(self):
        p = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [p], "lr": 1e-3, "name": "mhc"}]
        opt = build_optimizer("ada_muon", groups, {"ada_muon-lr_mhc": 2e-4})
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 2e-4)

        router = nn.Parameter(torch.randn(4, 4))
        groups = [{"params": [router], "lr": 1e-3, "name": "router"}]
        opt = build_optimizer("ada_muon", groups, {"ada_muon-lr_router": 3e-4})
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 3e-4)

    def test_string_coercion_and_invalid_fallback(self):
        opt = build_optimizer(
            "ada_muon",
            _param_groups(),
            {"ada_muon-momentum": "0.9", "ada_muon-ns_steps": "4"},
        )
        self.assertAlmostEqual(opt.defaults["momentum"], 0.9)
        self.assertEqual(opt.defaults["ns_steps"], 4)

        opt_bad = build_optimizer(
            "ada_muon",
            _param_groups(),
            {"ada_muon-momentum": "bad", "ada_muon-ns_steps": "bad"},
        )
        self.assertAlmostEqual(opt_bad.defaults["momentum"], 0.95)
        self.assertEqual(opt_bad.defaults["ns_steps"], 5)

    def test_unknown_group_suffix_raises(self):
        with self.assertRaises(ValueError) as ctx:
            build_optimizer("ada_muon", _param_groups(), {"ada_muon-lr_bogus": 1.0})
        self.assertIn("lr_bogus", str(ctx.exception))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerDion3Tests(unittest.TestCase):
    """Dion3 defaults, fractional selection and error feedback."""

    def test_builds_without_error(self):
        opt = build_optimizer("dion3", _param_groups(), {})
        self.assertIsNotNone(opt)

    def test_step_runs(self):
        p = nn.Parameter(torch.randn(4, 8))
        groups = [{"params": [p], "lr": 1e-2, "name": "attention"}]
        opt = build_optimizer("dion3", groups, {})
        p.square().mean().backward()
        opt.step()
        self.assertFalse(torch.isnan(p).any())

    def test_step_runs_normuon_flavor(self):
        p = nn.Parameter(torch.randn(4, 8))
        groups = [{"params": [p], "lr": 1e-2, "name": "attention"}]
        opt = build_optimizer(
            "dion3",
            groups,
            {"dion3-normuon": True, "dion3-fraction": 0.5},
        )
        p.square().mean().backward()
        opt.step()
        self.assertIn("row_second_moment", opt.state[p])
        self.assertFalse(torch.isnan(p).any())

    def test_defaults_match_paper(self):
        opt = build_optimizer("dion3", _param_groups(), {})
        self.assertAlmostEqual(opt.defaults["fraction"], 0.25)
        self.assertAlmostEqual(opt.defaults["momentum"], 0.95)
        self.assertEqual(opt.defaults["ns_steps"], 5)
        self.assertAlmostEqual(opt.defaults["ns_eps"], 1e-7)
        self.assertFalse(opt.defaults["normuon"])

    def test_custom_fraction_suffix(self):
        opt = build_optimizer("dion3", _param_groups(), {"dion3-fraction": 0.5})
        self.assertAlmostEqual(opt.defaults["fraction"], 0.5)

    def test_unselected_rows_unchanged(self):
        p = nn.Parameter(torch.randn(4, 8))
        groups = [{"params": [p], "lr": 1e-2, "weight_decay": 0.0, "name": "attention"}]
        opt = build_optimizer("dion3", groups, {"dion3-fraction": 0.5})
        before = p.detach().clone()
        p.square().mean().backward()
        grad = p.grad.detach().clone()
        opt.step()
        selected = torch.topk(grad.abs().sum(dim=1), 2).indices
        mask = torch.ones(4, dtype=torch.bool)
        mask[selected] = False
        self.assertTrue(torch.equal(p.detach()[mask], before[mask]))

    def test_tall_matrix_updates_only_selected_columns(self):
        p = nn.Parameter(torch.randn(8, 4))
        groups = [{"params": [p], "lr": 1e-2, "weight_decay": 0.0, "name": "attention"}]
        opt = build_optimizer("dion3", groups, {"dion3-fraction": 0.5})
        before = p.detach().clone()
        p.square().mean().backward()
        grad = p.grad.detach().clone()
        opt.step()
        selected = torch.topk(grad.abs().sum(dim=0), 2).indices
        mask = torch.ones(4, dtype=torch.bool)
        mask[selected] = False
        self.assertTrue(torch.equal(p.detach()[:, mask], before[:, mask]))

    def test_error_feedback_decays_only_selected_rows(self):
        p = nn.Parameter(torch.randn(4, 8))
        groups = [{"params": [p], "lr": 1e-2, "weight_decay": 0.0, "name": "attention"}]
        opt = build_optimizer("dion3", groups, {"dion3-fraction": 0.5})
        p.square().mean().backward()
        grad = p.grad.detach().clone()
        opt.step()
        momentum = opt.state[p]["momentum_buffer"]
        selected = torch.topk(grad.abs().sum(dim=1), 2).indices
        mask = torch.ones(4, dtype=torch.bool)
        mask[selected] = False
        self.assertTrue(
            torch.allclose(momentum[selected], grad[selected] * 0.95, atol=1e-6)
        )
        self.assertTrue(torch.allclose(momentum[mask], grad[mask], atol=1e-6))

    def test_fraction_one_update_norm(self):
        p = nn.Parameter(torch.randn(8, 4))
        groups = [{"params": [p], "lr": 1e-2, "weight_decay": 0.0, "name": "attention"}]
        opt = build_optimizer("dion3", groups, {"dion3-fraction": 1.0})
        before = p.detach().clone()
        p.square().mean().backward()
        opt.step()
        delta = (p.detach() - before).norm().item()
        expected = 1e-2 * math.sqrt(4)
        self.assertAlmostEqual(delta / expected, 1.0, delta=0.05)

    def test_string_coercion_and_invalid_fallback(self):
        opt = build_optimizer(
            "dion3",
            _param_groups(),
            {"dion3-fraction": "0.5", "dion3-ns_steps": "4"},
        )
        self.assertAlmostEqual(opt.defaults["fraction"], 0.5)
        self.assertEqual(opt.defaults["ns_steps"], 4)

        opt_bad = build_optimizer(
            "dion3",
            _param_groups(),
            {"dion3-fraction": "bad", "dion3-ns_steps": "bad"},
        )
        self.assertAlmostEqual(opt_bad.defaults["fraction"], 0.25)
        self.assertEqual(opt_bad.defaults["ns_steps"], 5)

    def test_unknown_group_suffix_raises(self):
        with self.assertRaises(ValueError) as ctx:
            build_optimizer("dion3", _param_groups(), {"dion3-lr_bogus": 1.0})
        self.assertIn("lr_bogus", str(ctx.exception))

    def test_vector_params_skipped(self):
        p = nn.Parameter(torch.randn(8))
        groups = [{"params": [p], "lr": 1e-2, "name": "norms"}]
        opt = build_optimizer("dion3", groups, {})
        before = p.detach().clone()
        p.square().mean().backward()
        opt.step()
        self.assertTrue(torch.equal(p.detach(), before))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerEveTests(unittest.TestCase):
    """EvE defaults, closure contract, population state and selection rules."""

    @staticmethod
    def _closure_for(p):
        """Deterministic quadratic closure: optimum 0.25 is box-interior."""
        def closure():
            p.grad = None
            loss = ((p - 0.25) ** 2).sum()
            loss.backward()
            return float(loss.detach())

        return closure

    @staticmethod
    def _tiny_param():
        return nn.Parameter(torch.linspace(-0.5, 0.5, 16).reshape(4, 4))

    def test_builds_without_error(self):
        opt = build_optimizer("eve", _param_groups(), {})
        self.assertIsNotNone(opt)

    def test_step_requires_closure(self):
        p = self._tiny_param()
        groups = [{"params": [p], "lr": 1e-2, "name": "other"}]
        opt = build_optimizer("eve", groups, {})
        with self.assertRaises(RuntimeError) as ctx:
            opt.step()
        self.assertIn("closure", str(ctx.exception))

    def test_step_runs_with_closure(self):
        p = self._tiny_param()
        groups = [{"params": [p], "lr": 1e-2, "name": "other"}]
        opt = build_optimizer("eve", groups, {"eve-seed": 7})
        closure = self._closure_for(p)
        history = [opt.step(closure)]
        for _ in range(7):
            history.append(opt.step(closure))
        self.assertTrue(all(math.isfinite(v) for v in history))
        # Proposition 1: greedy selection keeps the best-so-far monotone.
        for prev, curr in zip(history, history[1:]):
            self.assertLessEqual(curr, prev + 1e-12)
        self.assertFalse(torch.isnan(p).any())

    def test_defaults_match_paper(self):
        opt = build_optimizer("eve", _param_groups(), {})
        self.assertAlmostEqual(opt.defaults["de_weight"], 0.5)
        self.assertAlmostEqual(opt.defaults["de_jitter"], 1e-4)
        self.assertAlmostEqual(opt.defaults["init_sigma"], 0.1)
        self.assertEqual(opt.defaults["adam_steps"], 3)
        self.assertFalse(opt.defaults["use_pm"])
        self.assertAlmostEqual(opt.defaults["pm_eta"], 20.0)

    def test_custom_suffixes_and_coercion(self):
        opt = build_optimizer(
            "eve",
            _param_groups(),
            {"eve-adam_steps": "5", "eve-de_weight": "0.7"},
        )
        self.assertEqual(opt.defaults["adam_steps"], 5)
        self.assertAlmostEqual(opt.defaults["de_weight"], 0.7)

        opt_bad = build_optimizer(
            "eve",
            _param_groups(),
            {"eve-adam_steps": "bad", "eve-de_weight": "bad"},
        )
        self.assertEqual(opt_bad.defaults["adam_steps"], 3)
        self.assertAlmostEqual(opt_bad.defaults["de_weight"], 0.5)

    def test_unknown_group_suffix_raises(self):
        with self.assertRaises(ValueError) as ctx:
            build_optimizer("eve", _param_groups(), {"eve-lr_bogus": 1.0})
        self.assertIn("lr_bogus", str(ctx.exception))

    def test_per_group_lr_routes(self):
        p = self._tiny_param()
        groups = [{"params": [p], "lr": 1e-3, "name": "attention"}]
        opt = build_optimizer("eve", groups, {"eve-lr_attention": 0.02})
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.02)

    def test_population_state_layout(self):
        p = self._tiny_param()
        groups = [{"params": [p], "lr": 1e-2, "name": "other"}]
        opt = build_optimizer("eve", groups, {"eve-seed": 3})
        opt.step(self._closure_for(p))
        state = opt.state[p]
        self.assertEqual(state["slots"].shape[0], 4)
        self.assertEqual(state["m"].shape, state["slots"].shape)
        self.assertEqual(state["v"].shape, state["slots"].shape)
        self.assertIn("slot_losses", state)
        self.assertEqual(len(state["slot_losses"]), 4)
        self.assertTrue(all(math.isfinite(v) for v in state["slot_losses"]))

    def test_use_pm_flavor_runs(self):
        p = self._tiny_param()
        groups = [{"params": [p], "lr": 1e-2, "name": "other"}]
        opt = build_optimizer("eve", groups, {"eve-use_pm": True, "eve-seed": 11})
        closure = self._closure_for(p)
        history = [opt.step(closure)]
        for _ in range(3):
            history.append(opt.step(closure))
        self.assertTrue(all(math.isfinite(v) for v in history))
        for prev, curr in zip(history, history[1:]):
            self.assertLessEqual(curr, prev + 1e-12)
        self.assertFalse(torch.isnan(p).any())


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildOptimizerEA4LLMTests(unittest.TestCase):
    """EA4LLM: gradient-free evolution strategies (arXiv:2510.10603)."""

    @staticmethod
    def _closure_for(p):
        """Deterministic quadratic closure: optimum 0.25 is interior."""
        def closure():
            p.grad = None
            loss = ((p - 0.25) ** 2).sum()
            loss.backward()
            return float(loss.detach())

        return closure

    @staticmethod
    def _tiny_param():
        return nn.Parameter(torch.linspace(-0.5, 0.5, 16).reshape(4, 4))

    def test_builds_without_error(self):
        opt = build_optimizer("ea4llm", _param_groups(), {})
        self.assertIsNotNone(opt)

    def test_step_requires_closure(self):
        p = self._tiny_param()
        groups = [{"params": [p], "lr": 1e-2, "name": "other"}]
        opt = build_optimizer("ea4llm", groups, {})
        with self.assertRaises(RuntimeError) as ctx:
            opt.step()
        self.assertIn("closure", str(ctx.exception))

    def test_step_runs_with_closure(self):
        p = self._tiny_param()
        groups = [{"params": [p], "lr": 5e-3, "name": "other"}]
        opt = build_optimizer(
            "ea4llm", groups, {"ea4llm-seed": 7, "ea4llm-population": 16}
        )
        closure = self._closure_for(p)
        initial = float((((p - 0.25) ** 2).sum()).detach())
        for _ in range(40):
            value = opt.step(closure)
            self.assertTrue(math.isfinite(value))
        final = float((((p - 0.25) ** 2).sum()).detach())
        # The ES direction descends the negated fitness, i.e. the loss.
        self.assertLess(final, initial)
        self.assertFalse(torch.isnan(p).any())

    def test_defaults_match_paper(self):
        opt = build_optimizer("ea4llm", _param_groups(), {})
        self.assertEqual(opt.defaults["population"], 30)
        self.assertAlmostEqual(opt.defaults["sigma"], 1e-3)
        self.assertFalse(opt.defaults["antithetic"])
        self.assertFalse(opt.defaults["rank_shaping"])
        self.assertFalse(opt.defaults["nes_scaling"])

    def test_custom_suffixes_and_coercion(self):
        opt = build_optimizer(
            "ea4llm",
            _param_groups(),
            {"ea4llm-population": "8", "ea4llm-sigma": "0.01"},
        )
        self.assertEqual(opt.defaults["population"], 8)
        self.assertAlmostEqual(opt.defaults["sigma"], 0.01)

        opt_bad = build_optimizer(
            "ea4llm",
            _param_groups(),
            {"ea4llm-population": "bad", "ea4llm-sigma": "bad"},
        )
        self.assertEqual(opt_bad.defaults["population"], 30)
        self.assertAlmostEqual(opt_bad.defaults["sigma"], 1e-3)

    def test_unknown_group_suffix_raises(self):
        with self.assertRaises(ValueError) as ctx:
            build_optimizer("ea4llm", _param_groups(), {"ea4llm-lr_bogus": 1.0})
        self.assertIn("lr_bogus", str(ctx.exception))

    def test_per_group_lr_routes(self):
        p = self._tiny_param()
        groups = [{"params": [p], "lr": 1e-3, "name": "attention"}]
        opt = build_optimizer("ea4llm", groups, {"ea4llm-lr_attention": 0.02})
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.02)

    def test_state_has_no_per_parameter_buffers(self):
        p = self._tiny_param()
        groups = [{"params": [p], "lr": 1e-2, "name": "other"}]
        opt = build_optimizer(
            "ea4llm", groups, {"ea4llm-seed": 3, "ea4llm-population": 4}
        )
        opt.step(self._closure_for(p))
        state = opt.state[p]
        tensor_values = [value for value in state.values() if torch.is_tensor(value)]
        self.assertEqual(tensor_values, [])
        self.assertEqual(state["step"], 1)

    def test_rank_shaping_flavor_runs(self):
        p = self._tiny_param()
        groups = [{"params": [p], "lr": 5e-3, "name": "other"}]
        opt = build_optimizer(
            "ea4llm",
            groups,
            {"ea4llm-rank_shaping": True, "ea4llm-seed": 11, "ea4llm-population": 6},
        )
        closure = self._closure_for(p)
        for _ in range(3):
            value = opt.step(closure)
        self.assertTrue(math.isfinite(value))
        self.assertFalse(torch.isnan(p).any())

    def test_antithetic_flavor_runs(self):
        p = self._tiny_param()
        groups = [{"params": [p], "lr": 5e-3, "name": "other"}]
        opt = build_optimizer(
            "ea4llm",
            groups,
            {"ea4llm-antithetic": True, "ea4llm-seed": 5, "ea4llm-population": 6},
        )
        closure = self._closure_for(p)
        for _ in range(3):
            value = opt.step(closure)
        self.assertTrue(math.isfinite(value))
        self.assertFalse(torch.isnan(p).any())

    def test_antithetic_odd_population_raises(self):
        p = self._tiny_param()
        groups = [{"params": [p], "lr": 1e-2, "name": "other"}]
        with self.assertRaises(ValueError) as ctx:
            build_optimizer(
                "ea4llm",
                groups,
                {"ea4llm-antithetic": True, "ea4llm-population": 7},
            )
        self.assertIn("population", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
