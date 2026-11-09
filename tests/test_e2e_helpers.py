"""Tests for the e2e harness result classifier.

Real failures must surface as ``FAILED``; only unambiguous numerical
(NaN/Inf/exploding-gradient) messages may be tolerated as ``GRAD_EXPLODED``.
"""

from __future__ import annotations

import csv
import json

import pytest

from full_tests import _helpers
from full_tests._helpers import looks_like_gradient_explosion


BENIGN_OUTPUTS = [
    "14:00:00 | INFO     | Starting training (timeout=600s)",
    "INFO: Gradient Accumulation: 4 steps",
    "RuntimeError: Size does not match at line 3 of trainer.py",
    "KeyError: 'segmentation_map'",
    "Traceback (most recent call last):",
    "    self.best_loss = float('inf')",
    "    if math.isnan(loss):",
    "ValueError: audio_dataset requires a label_column",
]


NUMERICAL_FAILURES = [
    "🚨 NaN/Inf DETECTED IN LOSS - STOPPING TRAINING 🚨",
    "Early stop: Inf/exploding gradients persisted for too many consecutive epochs.",
    "RuntimeError: Function 'CtcLossBackward0' returned nan values in its 0th output.",
    "loss became NaN at step 12",
    "Non-finite loss detected for aux loss 'moe'",
    "gradient explosion detected at layer 3",
    "numerical overflow in fp16 attention",
    "parameters with Inf gradients",
]


@pytest.mark.parametrize("text", BENIGN_OUTPUTS)
def test_benign_failures_are_not_tolerated(text: str) -> None:
    assert not looks_like_gradient_explosion(text, "")


@pytest.mark.parametrize("text", NUMERICAL_FAILURES)
def test_numerical_failures_are_tolerated(text: str) -> None:
    assert looks_like_gradient_explosion(text, "")


def test_marker_must_be_word_bounded() -> None:
    # "inf" inside INFO/information/infinite must not match.
    assert not looks_like_gradient_explosion("INFO: infinite loop guard", "")
    assert not looks_like_gradient_explosion("information about the run", "")
    assert looks_like_gradient_explosion("loss is inf", "")
    assert looks_like_gradient_explosion("loss went to infinity", "")
    assert looks_like_gradient_explosion("", "RuntimeError: output contained inf")


def test_indented_traceback_source_is_ignored() -> None:
    source = "Traceback (most recent call last):\n    logits = logits / float('inf')\n"
    assert not looks_like_gradient_explosion(source, "")


def test_inference_accuracy_aggregates_quantized_variants(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_helpers, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        _helpers, "RESULTS_INFERENCE_ACCURACY_CSV_PATH", tmp_path / "accuracy.csv"
    )
    rows = [
        {
            "combo_id": "c1",
            "model_class": "frankenstein_text_encoder",
            "task": "mlm",
            "deploy_format": "hf_none",
            "metric_name": "mlm_top1",
            "metric_value": 1.0,
            "status": "OK",
        },
        {
            "combo_id": "c1",
            "model_class": "frankenstein_text_encoder",
            "task": "mlm",
            "deploy_format": "hf_none",
            "metric_name": "mlm_top1",
            "metric_value": 0.0,
            "status": "OK",
        },
        {
            "combo_id": "c1",
            "model_class": "frankenstein_text_encoder",
            "task": "mlm",
            "deploy_format": "hf_q4_0_v2",
            "metric_name": "mlm_top1",
            "metric_value": "",
            "status": "SKIPPED_NO_DATASET",
        },
    ]
    path = _helpers.write_inference_accuracy_csv(rows)
    assert path == tmp_path / "accuracy.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    by_format = {record["deploy_format"]: record for record in records}
    assert by_format["hf_none"]["ok_samples"] == "2"
    assert float(by_format["hf_none"]["accuracy"]) == pytest.approx(0.5)
    assert by_format["hf_q4_0_v2"]["ok_samples"] == "0"
    assert by_format["hf_q4_0_v2"]["accuracy"] == ""
    assert json.loads(by_format["hf_q4_0_v2"]["status_counts"]) == {
        "SKIPPED_NO_DATASET": 1
    }
