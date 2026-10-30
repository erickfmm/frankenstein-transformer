"""Tests for the held-out test split (``training.test_percentage``)."""

from __future__ import annotations

import copy
import json
import pathlib

import pytest
import yaml
from jsonschema import Draft202012Validator

from src.utils.schema_loader import resolve_schema

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

try:
    import torch
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

    from src.model.frankenstein_vit import FrankensteinViT
    from src.training.test_split import (
        TEST_METRICS_FILENAME,
        TestSplit as HeldOutSplit,
        build_test_metrics,
        evaluate_and_write,
        maybe_split_dataloader,
        resolve_test_percentage,
    )
    from src.training.trainer import TitanTrainer
    from src.training.vision_dataset import DummyImageDataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is part of the test extras
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------
class TestTestPercentageSchema:
    @pytest.fixture(scope="class")
    def schema(self):
        return resolve_schema(str(_REPO_ROOT / "src" / "schema.yaml"))

    def test_schema_declares_the_field(self):
        training = yaml.safe_load(
            (_REPO_ROOT / "src" / "schema" / "_training.yaml").read_text(
                encoding="utf-8"
            )
        )
        field = training["properties"]["test_percentage"]
        assert field["type"] == "number"
        assert field["default"] == 0
        assert field["minimum"] == 0
        assert field["maximum"] == 100
        for key in ("title", "title_es", "description", "description_es"):
            assert field[key], f"missing {key}"

    def test_valid_value_accepted_and_out_of_range_rejected(self, schema):
        base = yaml.safe_load(
            (_REPO_ROOT / "configs" / "frankenstein_text_encoder.yaml").read_text(
                encoding="utf-8"
            )
        )
        validator = Draft202012Validator(schema)

        valid = copy.deepcopy(base)
        valid.setdefault("training", {})["test_percentage"] = 20
        assert not list(validator.iter_errors(valid))

        for bad_value in (-1, 101):
            invalid = copy.deepcopy(base)
            invalid.setdefault("training", {})["test_percentage"] = bad_value
            assert list(validator.iter_errors(invalid)), bad_value


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------
def _tiny_classification_setup(num_samples: int = 10):
    """Build a tiny ViT classifier, its dataset and a DataLoader."""
    cfg = FrankensteinViT.build_vit_config(
        hidden_size=64,
        num_layers=1,
        num_heads=4,
        patch_size=8,
        image_height=32,
        image_width=32,
        num_classes=10,
    )
    model = FrankensteinViT(cfg)
    dataset = DummyImageDataset(
        task="classification",
        num_samples=num_samples,
        image_height=32,
        image_width=32,
        num_classes=10,
    )
    return cfg, model, dataset


class TestResolveTestPercentage:
    def test_default_is_disabled(self):
        assert resolve_test_percentage({}) == 0.0

    def test_clamped_to_range(self):
        assert resolve_test_percentage({"test_percentage": 33}) == 33.0
        assert resolve_test_percentage({"test_percentage": -5}) == 0.0
        assert resolve_test_percentage({"test_percentage": 250}) == 100.0

    def test_invalid_value_is_ignored(self):
        assert resolve_test_percentage({"test_percentage": "not-a-number"}) == 0.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestMaybeSplitDataloader:
    def test_zero_percentage_keeps_the_original_loader(self):
        _, _, dataset = _tiny_classification_setup()
        loader = DataLoader(dataset, batch_size=2)
        split = maybe_split_dataloader(loader, {"test_percentage": 0})
        assert split.train_loader is loader
        assert split.test_loader is None
        assert split.num_test == 0

    def test_partition_sizes_and_determinism(self):
        _, _, dataset = _tiny_classification_setup(num_samples=10)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)

        first = maybe_split_dataloader(loader, {"test_percentage": 30})
        second = maybe_split_dataloader(loader, {"test_percentage": 30})

        assert first.num_test == 3
        assert first.num_train == 7
        assert len(first.train_loader.dataset) == 7
        assert len(first.test_loader.dataset) == 3
        assert sorted(first.test_loader.dataset.indices) == sorted(
            second.test_loader.dataset.indices
        )
        assert not set(first.test_loader.dataset.indices) & set(
            first.train_loader.dataset.indices
        )
        assert first.test_loader.batch_size == loader.batch_size
        assert isinstance(first.test_loader.sampler, SequentialSampler)
        assert isinstance(first.train_loader.sampler, RandomSampler)

    def test_tiny_dataset_keeps_at_least_one_test_sample(self):
        _, _, dataset = _tiny_classification_setup(num_samples=4)
        split = maybe_split_dataloader(
            DataLoader(dataset, batch_size=1), {"test_percentage": 20}
        )
        assert split.num_test == 1
        assert split.num_train == 3


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestEvaluateAndWrite:
    def test_writes_test_metrics_json(self, tmp_path):
        cfg, model, dataset = _tiny_classification_setup(num_samples=8)
        trainer = TitanTrainer(
            model, cfg, training_config=None, device="cpu", task="classification"
        )
        try:
            split = maybe_split_dataloader(
                DataLoader(dataset, batch_size=2), {"test_percentage": 25}
            )
            payload = evaluate_and_write(
                trainer, split, "classification", out_dir=tmp_path
            )
        finally:
            trainer.close()

        assert payload is not None
        written = json.loads(
            (tmp_path / TEST_METRICS_FILENAME).read_text(encoding="utf-8")
        )
        assert written == payload
        assert payload["task"] == "classification"
        assert payload["metric"] == "accuracy"
        assert payload["test_percentage"] == 25.0
        assert payload["num_test_samples"] == split.num_test
        assert payload["num_train_samples"] == split.num_train
        assert 0.0 <= payload["test_accuracy"] <= 1.0
        assert payload["num_batches"] >= 1


class TestBuildTestMetrics:
    def test_asr_metric_is_labelled_cer(self):
        class _FakeTrainer:
            def evaluate(self, loader):
                return {
                    "loss": 1.5,
                    "accuracy": 0.25,
                    "num_batches": 1.0,
                    "num_samples": 4.0,
                }

        split = HeldOutSplit(
            train_loader=object(),
            test_loader=object(),
            num_train=12,
            num_test=4,
            test_percentage=25.0,
        )
        payload = build_test_metrics(_FakeTrainer(), split, "asr")
        assert payload is not None
        assert payload["metric"] == "cer"
        assert payload["test_loss"] == 1.5
        assert payload["test_accuracy"] == 0.25
        assert payload["num_train_samples"] == 12
        assert payload["num_test_samples"] == 4

    def test_disabled_split_returns_none(self):
        split = HeldOutSplit(
            train_loader=object(),
            test_loader=None,
            num_train=8,
            num_test=0,
            test_percentage=0.0,
        )
        assert build_test_metrics(object(), split, "mlm") is None
