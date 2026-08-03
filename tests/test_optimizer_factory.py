"""Unit tests for the optimizer factory (build_optimizer)."""
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
        required = {"adamw", "sgd_momentum", "lion", "muon", "lamb", "anon", "apollo", "apollo_mini", "q_apollo"}
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


if __name__ == "__main__":
    unittest.main()
