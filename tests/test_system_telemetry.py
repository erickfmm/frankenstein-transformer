"""Host CPU/RAM telemetry contract for the trainer and the e2e harness.

The trainer appends :data:`SYSTEM_TELEMETRY_COLUMNS` to its metrics CSV on every
logged step, on CPU and CUDA alike, and the harness writes a 1 Hz
``system_telemetry.csv`` next to ``results.json``. These gates pin the column
list, the zero fallbacks and the CSV round-trip so a future refactor cannot
silently drop a column or shift the header/row alignment.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.system_telemetry import (  # noqa: E402
    SYSTEM_TELEMETRY_COLUMNS,
    SYSTEM_TELEMETRY_DEFAULTS,
    SystemTelemetrySampler,
    default_system_telemetry,
    format_thread_usage,
)

try:
    import torch  # noqa: F401

    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_encoder import FrankensteinEncoder
    from src.training.trainer import TitanTrainer, TrainingConfig

    TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001
    TORCH_AVAILABLE = False

EXPECTED_COLUMNS = (
    "cpu_percent",
    "cpu_cores",
    "cpu_threads",
    "cpu_rss_mb",
    "cpu_rss_peak_mb",
    "cpu_thread_hot_pct",
    "sys_cpu_percent",
    "sys_ram_used_pct",
    "sys_ram_used_gb",
    "sys_ram_avail_gb",
    "sys_swap_used_pct",
    "sys_load_1m",
)


def test_columns_are_stable_and_ordered():
    """The CSV column order is part of the artifact contract."""
    assert SYSTEM_TELEMETRY_COLUMNS == EXPECTED_COLUMNS
    assert set(SYSTEM_TELEMETRY_DEFAULTS) == set(EXPECTED_COLUMNS)
    assert all(value == 0.0 for value in SYSTEM_TELEMETRY_DEFAULTS.values())


def test_default_payload_is_a_fresh_copy():
    """Callers mutate the payload, so each call must return a new dict."""
    first = default_system_telemetry()
    first["cpu_rss_mb"] = 123.0
    assert default_system_telemetry()["cpu_rss_mb"] == 0.0


def test_disabled_sampler_returns_zeros():
    """``system_metrics_enabled: false`` must keep the schema and emit zeros."""
    sampler = SystemTelemetrySampler(enabled=False)
    assert sampler.sample() == default_system_telemetry()
    assert sampler.thread_usage() == []


def test_sampler_reports_every_column():
    """An enabled sampler returns a numeric value for every declared column."""
    sampler = SystemTelemetrySampler(enabled=True, thread_breakdown=True)
    sample = sampler.sample()
    assert set(sample) == set(EXPECTED_COLUMNS)
    for name, value in sample.items():
        assert isinstance(value, float), name
    assert sample["cpu_rss_mb"] >= 0.0
    assert sample["cpu_rss_peak_mb"] >= sample["cpu_rss_mb"]
    assert sample["cpu_threads"] >= 1.0


def test_sampler_survives_a_missing_backend():
    """A stripped environment degrades to zeros instead of failing training."""
    sampler = SystemTelemetrySampler(enabled=False)
    sampler._psutil = None  # noqa: SLF001 - exercise the guard directly
    sampler._process = None  # noqa: SLF001
    sampler.enabled = True
    assert set(sampler.sample()) == set(EXPECTED_COLUMNS)


def test_format_thread_usage_is_compact():
    """The detailed per-thread view renders as ``tid:pct|tid:pct``."""
    assert format_thread_usage([]) == ""
    assert format_thread_usage([(12, 99.6), (13, 0.4)]) == "12:100|13:0"
    assert format_thread_usage([(1, 10.0), (2, 5.0), (3, 1.0)], limit=2) == "1:10|2:5"


def _tiny_encoder() -> "FrankensteinEncoder":
    """Build a tiny encoder whose metrics CSV can be written on CPU."""
    config = FrankensteinModelConfig.from_dict(
        {
            "dims": {
                "vocab_size": 64,
                "hidden_size": 48,
                "num_layers": 1,
                "num_loops": 1,
                "num_heads": 6,
                "dropout": 0.0,
                "layer_pattern": ["standard"],
                "mode": "encoder",
            },
            "norm": {"type": "layer_norm"},
            "bitnet": {"enabled": False},
            "ffn": {"hidden_size": 96, "activation": "gelu"},
            "use_moe": False,
        }
    )
    return FrankensteinEncoder(config)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_csv_header_and_row_carry_system_telemetry(tmp_path):
    """The trainer CSV header carries the host columns and rows line up."""
    model = _tiny_encoder()
    trainer = TitanTrainer(
        model,
        model.config,
        training_config=TrainingConfig(
            csv_log_path=str(tmp_path / "metrics.csv"),
            gpu_metrics_backend="none",
            gpu_temp_guard_enabled=False,
        ),
        device="cpu",
        task="mlm",
    )
    columns = trainer._get_csv_columns()  # noqa: SLF001 - schema under test
    for name in EXPECTED_COLUMNS:
        assert name in columns
    assert columns.index("cpu_percent") == columns.index("gpu_mem_used_mib") + 1

    trainer._init_csv_logger()  # noqa: SLF001
    try:
        trainer._log_step_to_csv(  # noqa: SLF001
            epoch=0,
            step=0,
            loss=1.0,
            accuracy=0.0,
            lr=1e-3,
            grad_norm=1.0,
            system_telemetry=SystemTelemetrySampler().sample(),
        )
    finally:
        trainer.close()

    with open(tmp_path / "metrics.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == trainer._get_csv_columns()  # noqa: SLF001
    assert len(rows[1]) == len(rows[0])
    header = dict(zip(rows[0], rows[1]))
    assert float(header["cpu_threads"]) >= 1.0
    assert float(header["sys_ram_used_pct"]) >= 0.0
