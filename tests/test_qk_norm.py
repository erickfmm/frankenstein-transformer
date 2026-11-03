"""Query–key normalization and attention-logit control tests.

Covers the three mechanisms selected by ``model.norm.qk_norm.method``:

* forward-time per-head QK normalization (``rms_norm``, ``layer_norm``,
  ``l2_norm``; arXiv:2010.04245, arXiv:2606.16310) against manual
  references;
* QK-Clip per-head post-step weight rescaling (arXiv:2507.20534);
* QuacK per-parameter query/key learning-rate modulation (arXiv:2511.21377);
* schema/config validation of the supported-mixer capability check;
* full-vs-cached decode parity with QK normalization enabled.
"""

import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

    from src.model.attention.grouped_query_attention import GroupedQueryAttention
    from src.model.attention.latent.mla_attn import MLAAttention
    from src.model.attention.standard import StandardAttention
    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_decoder import FrankensteinDecoder
    from src.model.norm import (
        QK_CLIP_MIXERS,
        QK_NORM_MIXERS,
        QKNorm,
        QuacKController,
        apply_qk_clip,
        attach_qk_norm,
        make_qk_norm,
        record_qk_max_logits,
    )


def _config(**overrides):
    base = dict(
        vocab_size=64,
        hidden_size=32,
        num_layers=1,
        num_loops=1,
        num_heads=4,
        retention_heads=4,
        num_kv_heads=2,
        dropout=0.0,
        use_bitnet=False,
        layer_pattern=["standard_attn"],
        mode="decoder",
        positional_encoding="rope",
        norm_type="layer_norm",
        ffn_hidden_size=64,
    )
    base.update(overrides)
    return FrankensteinModelConfig(**base)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class QKNormUnitTests(unittest.TestCase):
    def test_rms_matches_manual_reference(self):
        torch.manual_seed(0)
        q = torch.randn(2, 3, 5, 8)
        k = torch.randn(2, 3, 5, 8)
        norm = QKNorm(8, method="rms_norm", eps=1e-6)
        q_n, k_n = norm(q, k)
        expected_q = q / torch.sqrt(q.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        expected_k = k / torch.sqrt(k.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(q_n, expected_q, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(k_n, expected_k, rtol=1e-5, atol=1e-6)

    def test_layer_norm_matches_torch(self):
        torch.manual_seed(0)
        q = torch.randn(2, 3, 5, 8)
        norm = QKNorm(8, method="layer_norm", eps=1e-5)
        q_n = norm.normalize_q(q)
        expected = nn.functional.layer_norm(
            q.float(), (8,), norm.q_weight.float(), norm.q_bias.float(), 1e-5
        )
        torch.testing.assert_close(q_n, expected, rtol=1e-5, atol=1e-6)

    def test_l2_norm_is_weightless(self):
        norm = QKNorm(8, method="l2_norm")
        self.assertEqual(sum(p.numel() for p in norm.parameters()), 0)
        q = torch.randn(2, 3, 5, 8)
        q_n = norm.normalize_q(q)
        expected = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        torch.testing.assert_close(q_n, expected, rtol=1e-5, atol=1e-6)

    def test_learnable_scale_is_shared_across_heads(self):
        norm = QKNorm(8, method="rms_norm")
        self.assertEqual(tuple(norm.q_weight.shape), (8,))
        self.assertEqual(tuple(norm.k_weight.shape), (8,))
        self.assertIsNone(norm.q_bias)

    def test_gradients_flow(self):
        norm = QKNorm(4, method="rms_norm")
        q = torch.randn(1, 2, 3, 4, requires_grad=True)
        k = torch.randn(1, 2, 3, 4, requires_grad=True)
        q_n, k_n = norm(q, k)
        (q_n.sum() + k_n.sum()).backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(norm.q_weight.grad)

    def test_dtype_preserved(self):
        norm = QKNorm(4, method="rms_norm")
        q = torch.randn(1, 2, 3, 4, dtype=torch.float16)
        q_n, _ = norm(q, q)
        self.assertEqual(q_n.dtype, torch.float16)

    def test_unknown_method_rejected(self):
        with self.assertRaises(ValueError):
            QKNorm(4, method="qk_clip")

    def test_make_qk_norm_respects_config(self):
        self.assertIsNone(make_qk_norm(_config(use_qk_norm=False), 8))
        self.assertIsNone(make_qk_norm(_config(use_qk_norm=True, qk_norm_method="qk_clip"), 8))
        self.assertIsNone(make_qk_norm(_config(use_qk_norm=True, qk_norm_method="quack"), 8))
        self.assertIsInstance(
            make_qk_norm(_config(use_qk_norm=True, qk_norm_method="l2_norm"), 8), QKNorm
        )


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class MixerIntegrationTests(unittest.TestCase):
    def test_forward_methods_run_on_supported_mixers(self):
        for mixer_cls in (StandardAttention, GroupedQueryAttention, MLAAttention):
            for method in ("rms_norm", "layer_norm", "l2_norm"):
                with self.subTest(mixer=mixer_cls.__name__, method=method):
                    torch.manual_seed(0)
                    config = _config(use_qk_norm=True, qk_norm_method=method)
                    mixer = mixer_cls(config)
                    self.assertIsInstance(mixer.qk_norm, QKNorm)
                    x = torch.randn(2, 5, config.hidden_size)
                    out = mixer(x)
                    self.assertEqual(out.shape, x.shape)

    def test_disabled_attaches_no_module(self):
        mixer = StandardAttention(_config(use_qk_norm=False))
        self.assertIsNone(mixer.qk_norm)

    def test_qk_clip_records_per_head_max_logits(self):
        config = _config(use_qk_norm=True, qk_norm_method="qk_clip")
        mixer = StandardAttention(config)
        mixer(torch.randn(2, 6, config.hidden_size))
        self.assertIsNotNone(mixer.last_qk_max_logits)
        self.assertEqual(tuple(mixer.last_qk_max_logits.shape), (config.num_heads,))

    def test_forward_methods_do_not_record(self):
        config = _config(use_qk_norm=True, qk_norm_method="rms_norm")
        mixer = StandardAttention(config)
        mixer(torch.randn(2, 6, config.hidden_size))
        self.assertIsNone(mixer.last_qk_max_logits)

    def test_attach_marks_capability(self):
        module = nn.Module()
        attach_qk_norm(module, _config(use_qk_norm=True, qk_norm_method="quack"), 4)
        self.assertTrue(module.qk_norm_capable)
        self.assertIsNone(module.qk_norm)
        self.assertFalse(module._record_qk_max_logits)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class QKClipTests(unittest.TestCase):
    def test_per_head_rescaling_matches_gamma(self):
        torch.manual_seed(0)
        config = _config(use_qk_norm=True, qk_norm_method="qk_clip")
        mixer = StandardAttention(config)
        mixer(torch.randn(2, 6, config.hidden_size))
        tau = 0.5
        gamma = torch.clamp(tau / mixer.last_qk_max_logits, max=1.0)
        w_q = mixer.q_proj.weight.detach().clone()
        w_k = mixer.k_proj.weight.detach().clone()
        apply_qk_clip(mixer, tau=tau, alpha=0.5)
        heads, dim = mixer.num_heads, mixer.head_dim
        expected_q = w_q.view(heads, dim, -1) * gamma.pow(0.5).view(heads, 1, 1)
        expected_k = w_k.view(heads, dim, -1) * gamma.pow(0.5).view(heads, 1, 1)
        torch.testing.assert_close(mixer.q_proj.weight.view(heads, dim, -1), expected_q)
        torch.testing.assert_close(mixer.k_proj.weight.view(heads, dim, -1), expected_k)

    def test_alpha_split(self):
        torch.manual_seed(0)
        config = _config(use_qk_norm=True, qk_norm_method="qk_clip")
        mixer = StandardAttention(config)
        mixer(torch.randn(2, 6, config.hidden_size))
        tau, alpha = 0.5, 0.25
        gamma = torch.clamp(tau / mixer.last_qk_max_logits, max=1.0)
        w_q = mixer.q_proj.weight.detach().clone()
        w_k = mixer.k_proj.weight.detach().clone()
        apply_qk_clip(mixer, tau=tau, alpha=alpha)
        heads, dim = mixer.num_heads, mixer.head_dim
        expected_q = w_q.view(heads, dim, -1) * gamma.pow(alpha).view(heads, 1, 1)
        expected_k = w_k.view(heads, dim, -1) * gamma.pow(1.0 - alpha).view(heads, 1, 1)
        torch.testing.assert_close(mixer.q_proj.weight.view(heads, dim, -1), expected_q)
        torch.testing.assert_close(mixer.k_proj.weight.view(heads, dim, -1), expected_k)

    def test_heads_below_tau_are_untouched(self):
        torch.manual_seed(0)
        config = _config(use_qk_norm=True, qk_norm_method="qk_clip")
        mixer = StandardAttention(config)
        mixer(torch.randn(2, 6, config.hidden_size))
        w_q = mixer.q_proj.weight.detach().clone()
        rescaled = apply_qk_clip(mixer, tau=1e9, alpha=0.5)
        self.assertEqual(rescaled, 0)
        torch.testing.assert_close(mixer.q_proj.weight, w_q)

    def test_max_logits_cleared_after_use(self):
        mixer = StandardAttention(_config(use_qk_norm=True, qk_norm_method="qk_clip"))
        mixer(torch.randn(2, 6, 32))
        apply_qk_clip(mixer, tau=0.5)
        self.assertIsNone(mixer.last_qk_max_logits)

    def test_gqa_maps_query_heads_to_kv_heads(self):
        config = _config(use_qk_norm=True, qk_norm_method="qk_clip", num_kv_heads=2)
        mixer = GroupedQueryAttention(config)
        mixer(torch.randn(2, 6, config.hidden_size))
        w_k = mixer.k_proj.weight.detach().clone()
        apply_qk_clip(mixer, tau=0.1, alpha=0.5)
        # Key projection has num_kv_heads row blocks; it must have changed.
        self.assertFalse(torch.allclose(w_k, mixer.k_proj.weight))
        self.assertEqual(
            mixer.k_proj.weight.shape[0],
            config.num_kv_heads * mixer.head_dim,
        )

    def test_invalid_parameters_rejected(self):
        mixer = StandardAttention(_config(use_qk_norm=True, qk_norm_method="qk_clip"))
        with self.assertRaises(ValueError):
            apply_qk_clip(mixer, tau=0.0)
        with self.assertRaises(ValueError):
            apply_qk_clip(mixer, tau=1.0, alpha=1.5)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class QuacKTests(unittest.TestCase):
    def _model(self):
        return FrankensteinDecoder(_config(use_qk_norm=True, qk_norm_method="quack"))

    def test_splits_query_key_groups_and_caches_init_norms(self):
        from torch.optim import AdamW

        model = self._model()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        controller = QuacKController(model, optimizer, tau=1.0)
        self.assertEqual(controller.num_pairs, 1)
        names = [group.get("name", "") for group in optimizer.param_groups]
        self.assertTrue(any(name.endswith("_qk_q") for name in names))
        self.assertTrue(any(name.endswith("_qk_k") for name in names))

    def test_learning_rate_follows_counterpart_norm(self):
        from torch.optim import AdamW

        model = self._model()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        controller = QuacKController(model, optimizer, tau=1.0)
        entry = controller._entries[0]  # noqa: SLF001 - white-box formula check
        q_weight = entry["q"]
        k_weight = entry["k"]
        with torch.no_grad():
            k_weight.mul_(2.0)  # ||W_K|| doubles
            entry["init_k"] = float(k_weight.norm()) / 2.0
        controller.step()
        q_lr = float(entry["q_group"]["lr"])
        k_lr = float(entry["k_group"]["lr"])
        # lr(W_Q) = tau * eta * ||W_K||_init / ||W_K|| = eta / 2
        self.assertAlmostEqual(q_lr, 5e-4, places=7)
        # lr(W_K) depends on ||W_Q||/||W_Q||_init = 1 -> eta
        self.assertAlmostEqual(k_lr, 1e-3, places=7)
        self.assertGreater(q_weight.norm(), 0.0)

    def test_invalid_tau_rejected(self):
        from torch.optim import AdamW

        model = self._model()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        with self.assertRaises(ValueError):
            QuacKController(model, optimizer, tau=0.0)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class QKNormCacheParityTests(unittest.TestCase):
    def _build(self, pattern, pe, method="rms_norm"):
        config = _config(
            layer_pattern=[pattern],
            positional_encoding=pe,
            use_qk_norm=True,
            qk_norm_method=method,
        )
        torch.manual_seed(0)
        return FrankensteinDecoder(config).eval()

    def _parity(self, pattern, pe, method="rms_norm"):
        model_a = self._build(pattern, pe, method)
        model_b = self._build(pattern, pe, method)
        model_b.load_state_dict(model_a.state_dict())
        prompt = torch.randint(0, 64, (1, 6))
        with torch.inference_mode():
            full = model_a.generate(prompt.clone(), max_new_tokens=4, greedy=True)
            cached = model_b.generate(
                prompt.clone(), max_new_tokens=4, greedy=True, use_cache=True
            )
        return torch.equal(full, cached)

    def test_rope_parity_for_key_cached_mixers(self):
        for pattern in ("standard_attn", "gqa_attn", "swa_attn"):
            with self.subTest(pattern=pattern):
                self.assertTrue(self._parity(pattern, "rope"))

    def test_latent_mixers_parity(self):
        for pattern in ("mla_attn", "gqla_attn", "mlra_attn", "gta_attn"):
            for pe in ("nope", "rope"):
                with self.subTest(pattern=pattern, pe=pe):
                    self.assertTrue(self._parity(pattern, pe))

    def test_layer_norm_and_l2_parity(self):
        for method in ("layer_norm", "l2_norm"):
            with self.subTest(method=method):
                self.assertTrue(self._parity("mla_attn", "rope", method))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class QKNormConfigValidationTests(unittest.TestCase):
    def test_capability_sets(self):
        self.assertIn("standard_attn", QK_NORM_MIXERS)
        self.assertIn("mla_attn", QK_NORM_MIXERS)
        self.assertIn("gma_attn", QK_NORM_MIXERS)
        self.assertNotIn("gma_attn", QK_CLIP_MIXERS)
        self.assertNotIn("retnet", QK_NORM_MIXERS)

    def test_loader_accepts_supported_pattern(self):
        import pathlib
        import tempfile

        import yaml

        from src.training.config_loader import load_training_config

        document = {
            "model_class": "frankenstein_text_decoder",
            "model": {
                "dims": {
                    "hidden_size": 64,
                    "num_layers": 2,
                    "num_heads": 4,
                    "layer_pattern": ["standard_attn", "mla_attn"],
                    "mode": "decoder",
                },
                "norm": {
                    "type": "rms_norm",
                    "qk_norm": {"enabled": True, "method": "rms_norm"},
                },
                "bitnet": {"enabled": False},
            },
            "training": {
                "task": "causal_lm",
                "optimizer": {"optimizer_class": "adamw"},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            loaded = load_training_config(str(path))
        self.assertTrue(loaded.model_config.use_qk_norm)
        self.assertEqual(loaded.model_config.qk_norm_method, "rms_norm")

    def test_loader_rejects_unsupported_mixer(self):
        import pathlib
        import tempfile

        import yaml

        from src.training.config_loader import load_training_config

        document = {
            "model_class": "frankenstein_text_decoder",
            "model": {
                "dims": {
                    "hidden_size": 64,
                    "num_layers": 2,
                    "num_heads": 4,
                    "layer_pattern": ["retnet", "standard_attn"],
                    "mode": "decoder",
                },
                "norm": {
                    "type": "rms_norm",
                    "qk_norm": {"enabled": True, "method": "rms_norm"},
                },
                "bitnet": {"enabled": False},
            },
            "training": {
                "task": "causal_lm",
                "optimizer": {"optimizer_class": "adamw"},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not supported by mixer"):
                load_training_config(str(path))

    def test_loader_rejects_qk_clip_with_gma(self):
        import pathlib
        import tempfile

        import yaml

        from src.training.config_loader import load_training_config

        document = {
            "model_class": "frankenstein_text_decoder",
            "model": {
                "dims": {
                    "hidden_size": 64,
                    "num_layers": 1,
                    "num_heads": 4,
                    "layer_pattern": ["gma_attn"],
                    "mode": "decoder",
                },
                "norm": {
                    "type": "rms_norm",
                    "qk_norm": {"enabled": True, "method": "qk_clip"},
                },
                "bitnet": {"enabled": False},
            },
            "training": {
                "task": "causal_lm",
                "optimizer": {"optimizer_class": "adamw"},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not supported by mixer"):
                load_training_config(str(path))

    def test_dataclass_validates_parameters(self):
        with self.assertRaisesRegex(ValueError, "qk_norm_method"):
            _config(qk_norm_method="bogus")
        with self.assertRaisesRegex(ValueError, "qk_clip_tau"):
            _config(qk_clip_tau=0.0)
        with self.assertRaisesRegex(ValueError, "qk_clip_alpha"):
            _config(qk_clip_alpha=1.5)
        with self.assertRaisesRegex(ValueError, "quack_tau"):
            _config(quack_tau=0.0)
        with self.assertRaisesRegex(ValueError, "qk_norm_eps"):
            _config(qk_norm_eps=0.0)


if __name__ == "__main__":
    unittest.main()
