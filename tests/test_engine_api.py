"""Tests for the reusable non-CLI engine API (``src/engine.py``).

These cover the Phase 0 DashAI-integration exit criterion (see
``docs/dashai-plugin-audit.md`` §6 Phase 0): build a model from a config,
exercise the encoder classification head (Strategy A), round-trip a
checkpoint through ``save_checkpoint``/``load_checkpoint``, and verify the
SBERT task routes through the engine — all in-process on CPU, without
touching ``src/cli.py``.
"""
import os
import tempfile
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from src.engine import (
        SUPPORTED_DEVICE_CHOICES,
        TrainResult,
        _train_sbert,
        build_model,
        load_checkpoint,
        save_checkpoint,
    )
    from src.model.attention.common import BitLinear
    from src.model.config import FrankensteinModelConfig
    from src.training.config_loader import load_training_config
    from src.training.trainer import TrainingConfig

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINI_CONFIG = os.path.join(_REPO_ROOT, "configs", "mini.yaml")

VOCAB, SEQ, BSZ, HEADS = 64, 8, 2, 6


def _mini_cfg(**overrides):
    """Return a minimal valid :class:`FrankensteinModelConfig` for fast CPU tests."""
    base = dict(
        vocab_size=VOCAB,
        hidden_size=48,
        num_layers=1,
        num_loops=1,
        num_heads=HEADS,
        retention_heads=HEADS,
        num_experts=2,
        top_k_experts=1,
        dropout=0.0,
        layer_pattern=["standard_attn"],
        ode_solver="rk4",
        ode_steps=1,
        use_bitnet=False,
        norm_type="layer_norm",
        use_factorized_embedding=False,
        use_moe=False,
        ffn_activation="gelu",
    )
    base.update(overrides)
    return FrankensteinModelConfig(**base)


def _ids():
    return torch.randint(0, VOCAB, (BSZ, SEQ))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BuildModelTests(unittest.TestCase):
    """``engine.build_model`` dispatch and default behavior."""

    def test_build_encoder_returns_mlm_logits(self):
        model = build_model(None, _mini_cfg())
        self.assertIsInstance(model, nn.Module)
        model.eval()
        with torch.no_grad():
            out = model(_ids())
        self.assertEqual(tuple(out.shape), (BSZ, SEQ, VOCAB))

    def test_build_decoder_forces_decoder_mode(self):
        cfg = _mini_cfg(mode="encoder")
        build_model("frankensteindecoder", cfg)
        self.assertEqual(cfg.mode, "decoder")

    def test_build_model_is_in_engine_all(self):
        # Public API sanity — the DashAI plugin imports these names directly.
        from src import engine

        for name in (
            "build_model",
            "train_from_config",
            "save_checkpoint",
            "load_checkpoint",
            "TrainResult",
            "resolve_torch_device",
            "SUPPORTED_DEVICE_CHOICES",
        ):
            self.assertIn(name, engine.__all__, f"{name} missing from engine.__all__")


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ClassificationHeadTests(unittest.TestCase):
    """DashAI Strategy A — optional sequence-level classification head."""

    def test_head_off_by_default_returns_mlm_logits(self):
        model = build_model(None, _mini_cfg())
        self.assertFalse(model.classification_head)
        self.assertIsNone(model.cls_head)
        model.eval()
        with torch.no_grad():
            out = model(_ids())
        self.assertEqual(tuple(out.shape), (BSZ, SEQ, VOCAB))

    def test_num_labels_enables_head_and_returns_class_logits(self):
        model = build_model(None, _mini_cfg(), num_labels=5)
        self.assertTrue(model.classification_head)
        self.assertEqual(model.cls_num_labels, 5)
        self.assertIsNotNone(model.cls_head)
        model.eval()
        with torch.no_grad():
            out = model(_ids())
        self.assertEqual(tuple(out.shape), (BSZ, 5))

    def test_head_is_full_precision_even_under_bitnet(self):
        # Strategy A guarantee: the classification head is a plain nn.Linear
        # (NOT BitLinear) even when use_bitnet=True quantizes the backbone.
        cfg = _mini_cfg(use_bitnet=True)
        model = build_model(None, cfg, num_labels=3)
        self.assertIsNotNone(model.cls_head)
        self.assertIs(
            type(model.cls_head), nn.Linear, "cls_head must be plain nn.Linear"
        )
        self.assertNotIsInstance(model.cls_head, BitLinear)
        # The MLM head, by contrast, IS quantized under BitNet.
        self.assertIsInstance(model.head, BitLinear)

    def test_pooling_modes_cls_and_gap(self):
        for mode in ("cls", "gap"):
            cfg = _mini_cfg(encoder_pooling_mode=mode, classification_head=True, num_labels=4)
            model = build_model(None, cfg)
            model.eval()
            with torch.no_grad():
                out = model(_ids())
            self.assertEqual(tuple(out.shape), (BSZ, 4), f"pooling={mode}")


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class CheckpointRoundTripTests(unittest.TestCase):
    """``save_checkpoint`` -> ``load_checkpoint`` rebuilds the model faithfully."""

    def _loaded_mini(self):
        return load_training_config(MINI_CONFIG)

    def test_round_trip_preserves_state_dict(self):
        loaded = self._loaded_mini()
        model = build_model(loaded.model_class, loaded.model_config)
        # Mutate one parameter so we can detect reload (not accidental re-init).
        first_param = next(p for p in model.parameters() if p.requires_grad)
        with torch.no_grad():
            first_param.add_(0.123)
        original = first_param.detach().clone()

        with tempfile.TemporaryDirectory() as tmp:
            model_path = save_checkpoint(tmp, model, loaded)
            self.assertTrue(os.path.isfile(model_path))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "config.yaml")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "dashai_meta.json")))
            model2, loaded2, _tok, extra = load_checkpoint(tmp)

        self.assertEqual(loaded2.task, loaded.task)
        self.assertEqual(loaded2.model_class, loaded.model_class)
        first_param2 = next(p for p in model2.parameters() if p.requires_grad)
        self.assertTrue(torch.equal(first_param2, original))

    def test_round_trip_rebuilds_classification_head_from_extra(self):
        loaded = self._loaded_mini()
        # Build with the classification head enabled and persist num_labels.
        model = build_model(loaded.model_class, loaded.model_config, num_labels=7)
        self.assertIsNotNone(model.cls_head)

        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(tmp, model, loaded, extra={"num_labels": 7})
            model2, loaded2, _tok, extra2 = load_checkpoint(tmp)

        self.assertEqual(extra2.get("num_labels"), 7)
        self.assertTrue(loaded2.model_config.classification_head)
        self.assertEqual(loaded2.model_config.num_labels, 7)
        self.assertIsNotNone(model2.cls_head, "load_checkpoint must rebuild the head")
        self.assertEqual(model2.cls_num_labels, 7)

    def test_meta_records_task_and_model_class(self):
        import json

        loaded = self._loaded_mini()
        model = build_model(loaded.model_class, loaded.model_config)
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(tmp, model, loaded)
            with open(os.path.join(tmp, "dashai_meta.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
        self.assertEqual(meta["task"], loaded.task)
        self.assertEqual(meta["model_class"], loaded.model_class)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class SbertDispatchTests(unittest.TestCase):
    """The SBERT task now routes through the engine (``_train_sbert``)."""

    def _make_loaded(self, sbert_cfg=None):
        from src.training.config_loader import LoadedTrainingConfig

        sbert_cfg = {
            "output_dir": "./output/sbert_test",
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 1e-5,
            "pooling_mode": "mean",
            **(sbert_cfg or {}),
        }
        return LoadedTrainingConfig(
            task="sbert",
            model_class="base_model",
            model_config=None,
            base_model="sentence-transformers/MiniLM-L6-v2",
            tokenizer_config={"name_or_path": "sentence-transformers/MiniLM-L6-v2"},
            training_config=TrainingConfig(),
            training_runtime={"sbert": sbert_cfg},
            image_config={},
            dataset_config={},
            config_dict=None,
        )

    def _patch_sbert_main(self, return_value=0):
        """Monkeypatch ``src.sbert.train_sbert.main`` with a capturing stub."""
        captured = {"argv": None, "calls": 0}

        def _fake_main(argv):
            captured["argv"] = list(argv)
            captured["calls"] += 1
            return return_value

        import src.sbert.train_sbert as sbert_module

        original = sbert_module.main
        sbert_module.main = _fake_main
        self.addCleanup(setattr, sbert_module, "main", original)
        return captured

    def test_sbert_routes_through_engine(self):
        captured = self._patch_sbert_main(return_value=0)
        loaded = self._make_loaded()

        result = _train_sbert(loaded, "cpu", TrainingConfig())

        self.assertEqual(captured["calls"], 1)
        self.assertIn("--base-model", captured["argv"])
        self.assertIn("sentence-transformers/MiniLM-L6-v2", captured["argv"])
        self.assertIn("--device", captured["argv"])
        self.assertIn("cpu", captured["argv"])
        self.assertIn("--pooling_mode", captured["argv"])
        self.assertIsInstance(result, TrainResult)
        self.assertIsNone(result.model, "SBERT persists to its own output_dir")

    def test_sbert_requires_base_model(self):
        loaded = self._make_loaded()
        loaded.base_model = None
        with self.assertRaises(ValueError):
            _train_sbert(loaded, "cpu", TrainingConfig())

    def test_sbert_nonzero_exit_raises(self):
        self._patch_sbert_main(return_value=2)
        loaded = self._make_loaded()
        with self.assertRaises(RuntimeError):
            _train_sbert(loaded, "cpu", TrainingConfig())

    def test_sbert_invalid_runtime_config_raises(self):
        loaded = self._make_loaded()
        loaded.training_runtime = {"sbert": "not-a-dict"}
        with self.assertRaises(ValueError):
            _train_sbert(loaded, "cpu", TrainingConfig())


if __name__ == "__main__":
    unittest.main()
