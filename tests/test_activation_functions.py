"""Unit tests for activation functions, the factory, and the GLU FFN blocks.

Mirrors the structure of ``tests/test_common_modules.py`` (norm tests). Covers:
  * every elementwise activation (shape, finiteness, gradient flow)
  * the ``get_activation`` factory dispatch (incl. backward-compat silu/gelu)
  * the Rational Activation Function (RAF) variants and init presets
  * the GatedFFN (SwiGLU/GEGLU/ReGLU) shapes and gradients
  * FrankensteinModelConfig validation of ``ffn_activation`` / ``ffn_activation_config``
"""
import math
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

    from src.model.activation_function import (
        ALL_ACTIVATIONS,
        ELEMENTWISE_ACTIVATIONS,
        GLU_VARIANTS,
        GatedFFN,
        PReLU,
        RationalActivation,
        get_activation,
        make_gated_ffn,
    )
    from src.model.activation_function.learnable import _APPROX_FUNCTIONS
    from src.model.config import FrankensteinModelConfig, merge_model_blocks


def _cfg(ffn_activation="silu", hidden_size=8, ffn_hidden_size=16, **overrides):
    base = {
        "dims": {
            "vocab_size": 64,
            "hidden_size": hidden_size,
            "num_layers": 1,
            "num_loops": 1,
            "num_heads": 4,
            "dropout": 0.0,
            "layer_pattern": ["standard"],
        },
        "ffn": {"hidden_size": ffn_hidden_size, "activation": ffn_activation},
        "attention": {"retnet": {"heads": 4}},
        "moe": {"num_experts": 2, "top_k": 1, "enabled": False},
        "norm": {"type": "layer_norm"},
        "bitnet": {"enabled": False},
    }
    return FrankensteinModelConfig.from_dict(merge_model_blocks(base, overrides))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ElementwiseActivationTests(unittest.TestCase):
    """Each elementwise activation must preserve shape and stay finite."""

    def test_all_elementwise_preserve_shape_and_finite(self):
        x = torch.randn(3, 5, 8)
        for name in sorted(ELEMENTWISE_ACTIVATIONS):
            with self.subTest(activation=name):
                act = get_activation(_cfg(name), dim=8)
                y = act(x)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.isfinite(y).all())

    def test_learnable_activations_have_parameters(self):
        for name in ("prelu", "raf", "swish_trainable", "maxout", "pelu", "mpelu"):
            with self.subTest(activation=name):
                act = get_activation(_cfg(name), dim=16)
                params = list(act.parameters())
                self.assertGreater(len(params), 0, f"{name} should have params")

    def test_stateless_activations_have_no_parameters(self):
        for name in ("silu", "gelu", "relu", "tanh", "mish", "elu", "selu"):
            with self.subTest(activation=name):
                act = get_activation(_cfg(name), dim=8)
                self.assertEqual(list(act.parameters()), [])

    def test_backward_compat_silu_resolves(self):
        act = get_activation(_cfg("silu"))
        self.assertIsInstance(act, nn.Module)

    def test_backward_compat_gelu_resolves(self):
        act = get_activation(_cfg("gelu"))
        self.assertIsInstance(act, nn.Module)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GradientFlowTests(unittest.TestCase):
    """Gradients must flow to the input for learnable activations."""

    def test_learnable_grad_flows_to_input(self):
        for name in ("prelu", "raf", "swish_trainable", "maxout"):
            with self.subTest(activation=name):
                act = get_activation(_cfg(name), dim=16)
                x = torch.randn(2, 4, 16, requires_grad=True)
                act(x).sum().backward()
                self.assertIsNotNone(x.grad)
                self.assertEqual(x.grad.shape, x.shape)

    def test_grad_flows_to_parameters(self):
        act = get_activation(_cfg('raf', ffn={'activation_config': {'raf_trainable': True}}), dim=16)
        x = torch.randn(2, 4, 16)
        act(x).sum().backward()
        self.assertIsNotNone(act.numerator.grad)
        self.assertIsNotNone(act.denominator.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class RangePropertyTests(unittest.TestCase):
    """Spot-check output ranges for bounded activations."""

    def test_sigmoid_in_unit_interval(self):
        act = get_activation(_cfg("sigmoid"), dim=8)
        y = act(torch.randn(4, 8) * 50)
        self.assertTrue(((y >= 0) & (y <= 1)).all())

    def test_tanh_in_unit_ball(self):
        act = get_activation(_cfg("tanh"), dim=8)
        y = act(torch.randn(4, 8) * 50)
        self.assertTrue((y.abs() <= 1.0 + 1e-5).all())

    def test_relu6_bounded(self):
        act = get_activation(_cfg("relu6"), dim=8)
        y = act(torch.randn(4, 8) * 50)
        self.assertTrue(((y >= 0) & (y <= 6.0 + 1e-5)).all())

    def test_brelu_bounded(self):
        act = get_activation(_cfg("brelu"), dim=8)
        y = act(torch.randn(4, 8) * 50)
        self.assertTrue(((y >= 0) & (y <= 1.0 + 1e-5)).all())

    def test_relu_non_negative(self):
        act = get_activation(_cfg("relu"), dim=8)
        y = act(torch.randn(4, 8) * 50)
        self.assertTrue((y >= 0).all())

    def test_vrelu_non_negative(self):
        act = get_activation(_cfg("vrelu"), dim=8)
        y = act(torch.randn(4, 8) * 50)
        self.assertTrue((y >= 0).all())


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class CorrectnessTests(unittest.TestCase):
    """Numerical correctness against reference implementations."""

    def test_silu_matches_torch(self):
        act = get_activation(_cfg("silu"), dim=8)
        x = torch.randn(4, 8)
        self.assertTrue(torch.allclose(act(x), torch.nn.functional.silu(x), atol=1e-6))

    def test_gelu_matches_torch(self):
        act = get_activation(_cfg("gelu"), dim=8)
        x = torch.randn(4, 8)
        self.assertTrue(torch.allclose(act(x), torch.nn.functional.gelu(x), atol=1e-6))

    def test_mish_matches_torch(self):
        act = get_activation(_cfg("mish"), dim=8)
        x = torch.randn(4, 8)
        self.assertTrue(torch.allclose(act(x), torch.nn.functional.mish(x), atol=1e-6))

    def test_prelu_correctness(self):
        prelu = PReLU(8, init=0.25)
        prelu.weight.data.fill_(0.25)
        x = torch.randn(4, 8)
        y = prelu(x)
        expected = torch.relu(x) + (x - torch.relu(x)) * 0.25
        self.assertTrue(torch.allclose(y, expected, atol=1e-6))

    def test_swish_beta_matches_manual(self):
        act = get_activation(_cfg('swish', ffn={'activation_config': {'swish_beta': 2.0}}), dim=8)
        x = torch.randn(4, 8)
        expected = x * torch.sigmoid(2.0 * x)
        self.assertTrue(torch.allclose(act(x), expected, atol=1e-6))

    def test_identity_returns_input(self):
        act = get_activation(_cfg("identity"), dim=8)
        x = torch.randn(4, 8)
        self.assertTrue(torch.equal(act(x), x))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class RationalActivationTests(unittest.TestCase):
    """The RAF (arXiv:2208.14111): variants, init, scaling, freezing."""

    def test_all_versions_build_and_forward(self):
        for v in ("A", "B", "C", "D", "N"):
            with self.subTest(version=v):
                raf = RationalActivation(version=v)
                x = torch.randn(3, 5, 8)
                y = raf(x)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.isfinite(y).all())

    def test_invalid_version_raises(self):
        with self.assertRaises(ValueError):
            RationalActivation(version="Z")

    def test_invalid_degrees_raises(self):
        with self.assertRaises(ValueError):
            RationalActivation(degrees=(0, 4))
        with self.assertRaises(ValueError):
            RationalActivation(degrees=(5, 0))

    def test_invalid_approx_func_raises(self):
        with self.assertRaises(ValueError):
            RationalActivation(approx_func="bogus")

    def test_gelu_init_approximates_gelu_on_range(self):
        # On [-3, 3] the GELU-fit rational should track GELU closely.
        raf = RationalActivation(approx_func="gelu")
        x = torch.linspace(-3, 3, 61)
        target = torch.nn.functional.gelu(x)
        self.assertTrue(torch.allclose(raf(x), target, atol=1e-2))

    def test_input_scaling_keeps_range(self):
        raf = RationalActivation(input_scaling=True)
        # Huge inputs are rescaled into [-3, 3] before the rational.
        x = torch.randn(4, 16) * 1e4
        y = raf(x)
        self.assertTrue(torch.isfinite(y).all())

    def test_freezing_disables_grad(self):
        raf = RationalActivation(trainable=False)
        self.assertFalse(raf.numerator.requires_grad)
        self.assertFalse(raf.denominator.requires_grad)

    def test_all_approx_funcs_supported(self):
        for name in _APPROX_FUNCTIONS:
            with self.subTest(approx_func=name):
                raf = RationalActivation(approx_func=name)
                self.assertEqual(raf.numerator.shape[0], 6)
                self.assertEqual(raf.denominator.shape[0], 4)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GatedFFNTests(unittest.TestCase):
    """SwiGLU / GEGLU / ReGLU gated feed-forward blocks."""

    def test_all_variants_build_and_forward(self):
        for kind in sorted(GLU_VARIANTS):
            with self.subTest(variant=kind):
                ffn = make_gated_ffn(kind, hidden_size=16, intermediate_size=32)
                x = torch.randn(2, 5, 16)
                y = ffn(x)
                self.assertEqual(y.shape, (2, 5, 16))

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            make_gated_ffn("boglu", hidden_size=16, intermediate_size=32)

    def test_grad_flows(self):
        ffn = make_gated_ffn("swiglu", hidden_size=16, intermediate_size=32)
        x = torch.randn(2, 5, 16, requires_grad=True)
        ffn(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_has_three_projections(self):
        ffn = make_gated_ffn("geglu", hidden_size=16, intermediate_size=32)
        self.assertTrue(hasattr(ffn, "gate_proj"))
        self.assertTrue(hasattr(ffn, "up_proj"))
        self.assertTrue(hasattr(ffn, "down_proj"))

    def test_proj_factory_used(self):
        calls = []
        def fac(i, o, b):
            calls.append((i, o, b))
            return nn.Linear(i, o, bias=b)
        make_gated_ffn("reglu", 8, 16, bias=True, proj_factory=fac)
        self.assertEqual(len(calls), 3)

    def test_is_module_subclass(self):
        self.assertTrue(issubclass(GatedFFN, nn.Module))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class FactoryDispatchTests(unittest.TestCase):
    """get_activation dispatch and error handling."""

    def test_glu_variant_raises_in_factory(self):
        for kind in sorted(GLU_VARIANTS):
            with self.subTest(variant=kind):
                with self.assertRaises(ValueError):
                    get_activation(_cfg(kind), dim=8)

    def test_unknown_activation_raises(self):
        with self.assertRaises(ValueError):
            get_activation(_cfg("bogus"), dim=8)

    def test_factory_passes_nested_params(self):
        cfg = _cfg('prelu', ffn={'activation_config': {'prelu_init': 0.5}})
        act = get_activation(cfg, dim=8)
        self.assertTrue(torch.allclose(act.weight.data, torch.full((8,), 0.5)))

    def test_factory_reads_ffn_hidden_size_dim(self):
        cfg = _cfg("prelu")
        act = get_activation(cfg, dim=cfg.ffn.hidden_size)
        self.assertEqual(act.weight.shape[0], cfg.ffn.hidden_size)

    def test_enum_covers_factory(self):
        # Every enum name must be buildable except the GLU variants.
        for name in ALL_ACTIVATIONS:
            with self.subTest(name=name):
                if name in GLU_VARIANTS:
                    continue
                act = get_activation(_cfg(name), dim=8)
                self.assertIsInstance(act, nn.Module)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class FrankensteinModelConfigValidationTests(unittest.TestCase):
    """FrankensteinModelConfig.__post_init__ enforces activation constraints."""

    def test_default_silu(self):
        cfg = _cfg()
        self.assertEqual(cfg.ffn.activation, "silu")

    def test_unknown_activation_rejected(self):
        with self.assertRaises(ValueError):
            FrankensteinModelConfig.from_dict({"dims": {"vocab_size": 64, "hidden_size": 8, "num_layers": 1, "num_loops": 1, "num_heads": 4, "layer_pattern": ['standard']}, "attention": {"retnet": {"heads": 4}}, "moe": {"num_experts": 2, "top_k": 1}, "ffn": {"activation": 'bogus'}})

    def test_activation_lowercased(self):
        cfg = _cfg("MISH")
        self.assertEqual(cfg.ffn.activation, "mish")

    def test_config_must_be_dict(self):
        with self.assertRaises(ValueError):
            FrankensteinModelConfig.from_dict({"dims": {"vocab_size": 64, "hidden_size": 8, "num_layers": 1, "num_loops": 1, "num_heads": 4, "layer_pattern": ['standard']}, "attention": {"retnet": {"heads": 4}}, "moe": {"num_experts": 2, "top_k": 1}, "ffn": {"activation": 'raf', "activation_config": [1, 2, 3]}})

    def test_unknown_config_key_rejected(self):
        with self.assertRaises(ValueError):
            FrankensteinModelConfig.from_dict({"dims": {"vocab_size": 64, "hidden_size": 8, "num_layers": 1, "num_loops": 1, "num_heads": 4, "layer_pattern": ['standard']}, "attention": {"retnet": {"heads": 4}}, "moe": {"num_experts": 2, "top_k": 1}, "ffn": {"activation": 'raf', "activation_config": {'bogus_key': 1}}})

    def test_bad_raf_version_rejected(self):
        with self.assertRaises(ValueError):
            FrankensteinModelConfig.from_dict({"dims": {"vocab_size": 64, "hidden_size": 8, "num_layers": 1, "num_loops": 1, "num_heads": 4, "layer_pattern": ['standard']}, "attention": {"retnet": {"heads": 4}}, "moe": {"num_experts": 2, "top_k": 1}, "ffn": {"activation": 'raf', "activation_config": {'raf_version': 'Z'}}})

    def test_bad_raf_degrees_rejected(self):
        with self.assertRaises(ValueError):
            FrankensteinModelConfig.from_dict({"dims": {"vocab_size": 64, "hidden_size": 8, "num_layers": 1, "num_loops": 1, "num_heads": 4, "layer_pattern": ['standard']}, "attention": {"retnet": {"heads": 4}}, "moe": {"num_experts": 2, "top_k": 1}, "ffn": {"activation": 'raf', "activation_config": {'raf_degrees': [5, 4, 3]}}})


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class SituGluTests(unittest.TestCase):
    """SiTU-GLU (Kimi K3, arXiv:2607.24653 §2.3.2)."""

    def test_registered_as_glu(self):
        self.assertIn("situ", GLU_VARIANTS)
        self.assertNotIn("situ", ELEMENTWISE_ACTIVATIONS)

    def test_gate_and_up_branches_are_smooth_capped(self):
        beta1, beta2 = 4.0, 25.0
        ffn = make_gated_ffn("situ", 4, 8, bias=False)
        with torch.no_grad():
            ffn.gate_proj.weight.copy_(torch.eye(8, 4))
            ffn.up_proj.weight.copy_(torch.eye(8, 4))
            ffn.down_proj.weight.copy_(torch.eye(4, 8))
        x = torch.tensor([[10.0, -10.0, 0.5, 0.0]])
        u = ffn.gate_proj(x)
        v = ffn.up_proj(x)
        gate = beta1 * torch.tanh(u / beta1) * torch.sigmoid(u)
        up = beta2 * torch.tanh(v / beta2)
        expected = ffn.down_proj(gate * up)
        torch.testing.assert_close(ffn(x), expected)
        self.assertTrue(bool(((gate * up).abs() <= beta1 * beta2 + 1e-5).all()))

    def test_matches_swiglu_near_origin(self):
        torch.manual_seed(0)
        situ = make_gated_ffn("situ", 4, 8, beta_gate=1e6, beta_up=1e6, bias=False)
        swiglu = make_gated_ffn("swiglu", 4, 8, bias=False)
        with torch.no_grad():
            for a, b in (
                (situ.gate_proj, swiglu.gate_proj),
                (situ.up_proj, swiglu.up_proj),
                (situ.down_proj, swiglu.down_proj),
            ):
                b.weight.copy_(a.weight)
        x = torch.randn(2, 3, 4) * 0.1
        torch.testing.assert_close(situ(x), swiglu(x), rtol=1e-4, atol=1e-5)
