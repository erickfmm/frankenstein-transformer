"""Reproducibility contract tests for ``training.seed``.

Locks in the single-knob reproducibility design:

* the schema declares ``training.seed`` with default ``42`` and mirrors the
  :class:`~src.training.trainer.TrainingConfig` default;
* every preset/example YAML declares it explicitly;
* the loader forwards explicit seeds and falls back to ``42``;
* ``set_global_seed`` resets the torch stream and enables deterministic cuDNN;
* model initialization and DataLoader shuffling are reproducible per seed;
* the held-out test split follows ``training.seed``.
"""

from __future__ import annotations

import pathlib
import random

import pytest
import yaml

from src.training.config_loader import load_training_config
from src.training.test_split import maybe_split_dataloader
from src.training.trainer import TrainingConfig
from src.utils.reproducibility import (
    dataloader_kwargs,
    make_generator,
    seed_worker,
    set_global_seed,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_TRAINING_SCHEMA = _REPO_ROOT / "src" / "schema" / "_training.yaml"
_CONFIG_FILES = sorted((_REPO_ROOT / "configs").glob("*.yaml")) + sorted(
    (_REPO_ROOT / "configs" / "examples").glob("*.yaml")
)

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_encoder import FrankensteinEncoder

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch-free schema runs
    TORCH_AVAILABLE = False


def _tiny_config() -> "FrankensteinModelConfig":
    return FrankensteinModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_loops=1,
        num_heads=4,
        layer_pattern=["standard_attn", "standard_attn"],
    )


def test_schema_seed_default_matches_training_config():
    training_schema = yaml.safe_load(_TRAINING_SCHEMA.read_text(encoding="utf-8"))
    seed_schema = training_schema["properties"]["seed"]
    assert seed_schema["type"] == "integer"
    assert seed_schema["default"] == 42
    assert seed_schema["minimum"] == 0
    assert TrainingConfig().seed == seed_schema["default"]


def test_every_preset_declares_seed():
    assert _CONFIG_FILES, "no preset YAML files found"
    missing = []
    for path in _CONFIG_FILES:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        training = data.get("training") or {}
        if training.get("seed") != 42:
            missing.append(str(path.relative_to(_REPO_ROOT)))
    assert not missing, f"presets without explicit training.seed: 42 -> {missing}"


def test_loader_forwards_explicit_seed(tmp_path):
    source = _CONFIG_FILES[0]
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["training"]["seed"] = 123
    target = tmp_path / "seed_override.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    loaded = load_training_config(str(target))
    assert loaded.training_config.seed == 123


def test_loader_defaults_seed_when_absent(tmp_path):
    source = _CONFIG_FILES[0]
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["training"].pop("seed", None)
    target = tmp_path / "seed_default.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    loaded = load_training_config(str(target))
    assert loaded.training_config.seed == 42


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_set_global_seed_resets_torch_stream():
    set_global_seed(7)
    first = torch.rand(8)
    set_global_seed(7)
    second = torch.rand(8)
    assert torch.equal(first, second)

    set_global_seed(8)
    third = torch.rand(8)
    assert not torch.equal(first, third)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_model_init_is_reproducible_per_seed():
    set_global_seed(11)
    model_a = FrankensteinEncoder(_tiny_config())
    set_global_seed(11)
    model_b = FrankensteinEncoder(_tiny_config())
    state_a, state_b = model_a.state_dict(), model_b.state_dict()
    assert state_a.keys() == state_b.keys()
    assert all(torch.equal(state_a[key], state_b[key]) for key in state_a)

    set_global_seed(12)
    model_c = FrankensteinEncoder(_tiny_config())
    state_c = model_c.state_dict()
    assert any(
        not torch.equal(state_a[key], state_c[key])
        for key in state_a
        if torch.is_floating_point(state_a[key])
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_dataloader_shuffle_order_is_reproducible_per_seed():
    dataset = TensorDataset(torch.arange(10))

    def batches(seed: int):
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            **dataloader_kwargs(seed),
        )
        return [batch[0].tolist() for batch in loader]

    assert batches(3) == batches(3)
    assert batches(3) != batches(4)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_test_split_follows_seed():
    dataset = TensorDataset(torch.arange(20))
    loader = DataLoader(dataset, batch_size=4, shuffle=True, **dataloader_kwargs(42))
    runtime = {"test_percentage": 25}

    split_a = maybe_split_dataloader(loader, runtime, seed=1)
    split_b = maybe_split_dataloader(loader, runtime, seed=1)
    split_c = maybe_split_dataloader(loader, runtime, seed=2)

    indices_a = split_a.test_loader.dataset.indices
    indices_b = split_b.test_loader.dataset.indices
    indices_c = split_c.test_loader.dataset.indices
    assert indices_a == indices_b
    assert indices_a != indices_c
    assert split_a.num_train == 15
    assert split_a.num_test == 5


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_seed_worker_seeds_from_loader_base_seed():
    def first_draw() -> float:
        torch.manual_seed(5)
        seed_worker(0)
        return random.random()

    assert first_draw() == first_draw()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_make_generator_is_deterministic():
    a = torch.randperm(16, generator=make_generator(9)).tolist()
    b = torch.randperm(16, generator=make_generator(9)).tolist()
    c = torch.randperm(16, generator=make_generator(10)).tolist()
    assert a == b
    assert a != c
