"""Tests for the held-out test split (``training.test_percentage``)."""

from __future__ import annotations

import copy
import json
import logging
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
        TEST_SPLIT_FILENAME,
        TestSplit as HeldOutSplit,
        build_test_metrics,
        evaluate_and_write,
        maybe_split_dataloader,
        resolve_test_percentage,
        resolve_test_stratify,
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


class _ImbalancedClassDataset:
    """Tiny map-style dataset with a fixed, imbalanced class list."""

    def __init__(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "x": torch.tensor(float(index), dtype=torch.float32),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


class TestResolveTestPercentage:
    def test_default_is_disabled(self):
        assert resolve_test_percentage({}) == 0.0

    def test_clamped_to_range(self):
        assert resolve_test_percentage({"test_percentage": 33}) == 33.0
        assert resolve_test_percentage({"test_percentage": -5}) == 0.0
        assert resolve_test_percentage({"test_percentage": 250}) == 100.0

    def test_invalid_value_is_ignored(self):
        assert resolve_test_percentage({"test_percentage": "not-a-number"}) == 0.0


class TestResolveTestStratify:
    def test_default_is_enabled(self):
        assert resolve_test_stratify({}) is True

    def test_explicit_values(self):
        assert resolve_test_stratify({"test_stratify": False}) is False
        assert resolve_test_stratify({"test_stratify": "false"}) is False
        assert resolve_test_stratify({"test_stratify": "yes"}) is True
        assert resolve_test_stratify({"test_stratify": None}) is True


class TestTestStratifySchema:
    @pytest.fixture(scope="class")
    def schema(self):
        return resolve_schema(str(_REPO_ROOT / "src" / "schema.yaml"))

    def test_schema_declares_the_field(self):
        training = yaml.safe_load(
            (_REPO_ROOT / "src" / "schema" / "_training.yaml").read_text(
                encoding="utf-8"
            )
        )
        field = training["properties"]["test_stratify"]
        assert field["type"] == "boolean"
        assert field["default"] is True
        for key in ("title", "title_es", "description", "description_es"):
            assert field[key], f"missing {key}"

    def test_valid_value_accepted(self, schema):
        base = yaml.safe_load(
            (_REPO_ROOT / "configs" / "frankenstein_text_encoder.yaml").read_text(
                encoding="utf-8"
            )
        )
        validator = Draft202012Validator(schema)
        valid = copy.deepcopy(base)
        valid.setdefault("training", {})["test_stratify"] = False
        assert not list(validator.iter_errors(valid))


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestStratifiedSplit:
    def _split(self, labels, percentage=20, seed=7, task="classification", extra=None):
        dataset = _ImbalancedClassDataset(labels)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        runtime = {"test_percentage": percentage}
        runtime.update(extra or {})
        return maybe_split_dataloader(loader, runtime, seed=seed, task=task)

    def test_keeps_class_proportions_and_rare_classes(self):
        labels = [0] * 80 + [1] * 15 + [2] * 5
        split = self._split(labels)
        assert split.stratified is True
        assert split.num_test == 20
        assert split.class_counts_train == {"0": 64, "1": 12, "2": 4}
        assert split.class_counts_test == {"0": 16, "1": 3, "2": 1}
        assert set(split.train_indices).isdisjoint(split.test_indices)

    def test_is_deterministic_for_a_fixed_seed(self):
        labels = [0] * 8 + [1] * 4 + [2] * 2
        first = self._split(labels, seed=11)
        second = self._split(labels, seed=11)
        different = self._split(labels, seed=12)
        assert first.test_indices == second.test_indices
        assert first.train_indices == second.train_indices
        # The per-class permutation changes with the seed.
        assert first.test_indices != different.test_indices or first.train_indices != different.train_indices

    def test_can_be_disabled(self):
        labels = [0] * 8 + [1] * 4 + [2] * 2
        split = self._split(labels, extra={"test_stratify": False})
        assert split.stratified is False
        assert split.num_test == 3
        assert split.class_counts_test

    def test_non_classification_task_falls_back_to_random(self):
        labels = [0] * 8 + [1] * 4 + [2] * 2
        split = self._split(labels, task="mlm")
        assert split.stratified is False
        assert split.num_test == 3
        assert split.num_classes == 0

    def test_singleton_classes_stay_in_train(self):
        labels = [0] * 10 + [1, 2]
        split = self._split(labels, percentage=20)
        assert split.stratified is True
        assert 1 not in split.test_indices
        assert 2 not in split.test_indices
        assert split.class_counts_test.get("1", 0) == 0
        assert split.class_counts_test.get("2", 0) == 0

    def test_warns_when_a_splittable_class_is_absent(self, caplog):
        labels = [0] * 100 + [1] * 2
        with caplog.at_level(logging.WARNING):
            split = self._split(labels, percentage=1)
        assert split.stratified is True
        assert split.class_counts_test.get("1", 0) == 0
        assert "no test examples" in caplog.text


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
        cfg, model, dataset = _tiny_classification_setup(num_samples=16)
        trainer = TitanTrainer(
            model, cfg, training_config=None, device="cpu", task="classification"
        )
        try:
            split = maybe_split_dataloader(
                DataLoader(dataset, batch_size=2),
                {"test_percentage": 25},
                task="classification",
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
        assert payload["primary_metric"] == "macro_f1"
        assert payload["test_percentage"] == 25.0
        assert payload["num_test_samples"] == split.num_test
        assert payload["num_train_samples"] == split.num_train
        assert 0.0 <= payload["test_accuracy"] <= 1.0
        assert payload["num_batches"] >= 1
        assert "test_macro_f1" in payload
        assert "test_balanced_accuracy" in payload
        assert "test_weighted_f1" in payload
        assert payload["stratified"] is True
        assert payload["num_classes"] == 10
        assert "confusion_matrix" in payload
        manifest = json.loads(
            (tmp_path / TEST_SPLIT_FILENAME).read_text(encoding="utf-8")
        )
        assert manifest["test_indices"] == list(split.test_indices)
        assert manifest["train_indices"] == list(split.train_indices)


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
        assert payload["primary_metric"] == "cer"
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
