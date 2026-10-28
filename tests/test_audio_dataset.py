#!/usr/bin/env python3
"""Tests for the audio dataset layer (``src/training/audio_dataset.py``).

Covers the character tokenizer, the synthetic dummy dataset for both audio
tasks, the HuggingFace-backed dataset (``datasets.load_dataset`` is mocked, no
network), resampling fallback, duration clipping and the padding collate.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.training.audio_dataset import (  # noqa: E402
    AudioDataset,
    CharacterTokenizer,
    DummyAudioDataset,
    audio_collate_fn,
)


class TestCharacterTokenizer:
    def test_vocabulary_is_sorted_and_reserves_blank_and_unk(self):
        tokenizer = CharacterTokenizer(["ba", "abc"])
        assert tokenizer.blank_id == 0
        assert tokenizer.unk_id == 1
        assert tokenizer.chars == ["a", "b", "c"]
        assert tokenizer.char_to_id == {"a": 2, "b": 3, "c": 4}
        assert tokenizer.vocab_size == 5
        assert len(tokenizer) == 5

    def test_encode_decode_round_trip(self):
        tokenizer = CharacterTokenizer(["hola mundo"])
        encoded = tokenizer.encode("ho")
        assert encoded == [tokenizer.char_to_id["h"], tokenizer.char_to_id["o"]]
        assert tokenizer.decode(encoded) == "ho"

    def test_unknown_characters_map_to_unk(self):
        tokenizer = CharacterTokenizer(["ab"])
        encoded = tokenizer.encode("az")
        assert encoded[1] == tokenizer.unk_id
        assert tokenizer.decode(encoded) == "a"

    def test_empty_corpus_is_minimal(self):
        tokenizer = CharacterTokenizer([])
        assert tokenizer.vocab_size == 2
        assert tokenizer.encode("x") == [tokenizer.unk_id]


class TestDummyAudioDataset:
    def test_asr_shapes_and_types(self):
        dataset = DummyAudioDataset(
            length=4, audio_config={"feature": "logmel"}, task="asr",
            vocab_size=16, max_frames=800,
        )
        assert len(dataset) == 4
        item = dataset[1]
        assert item["waveform"].dtype == torch.float32
        assert item["waveform"].shape == (800,)
        assert item["labels"].dtype == torch.long
        assert item["labels"].numel() >= 1
        assert int(item["labels"].min()) >= 1
        assert int(item["labels"].max()) < 16
        assert int(item["input_lengths"]) == 800
        assert int(item["target_lengths"]) == item["labels"].numel()
        assert item["task"] == "asr"
        assert dataset.vocab_size == 16
        assert dataset.tokenizer is None

    def test_classification_labels(self):
        dataset = DummyAudioDataset(
            length=6, audio_config={"feature": "raw", "in_channels": 8},
            task="audio_classification", num_classes=3, max_frames=256,
        )
        item = dataset[4]
        assert item["waveform"].shape == (256, 8)
        assert item["labels"].dtype == torch.long
        assert int(item["labels"]) == 4 % 3
        assert int(item["target_lengths"]) == 0

    def test_deterministic_per_index(self):
        dataset = DummyAudioDataset(length=2, task="asr", max_frames=128)
        assert torch.equal(dataset[0]["waveform"], dataset[0]["waveform"])
        assert not torch.equal(dataset[0]["waveform"], dataset[1]["waveform"])
        assert torch.equal(dataset[0]["labels"], dataset[0]["labels"])


class TestAudioDataset:
    @staticmethod
    def _fake_dataset():
        """In-memory dataset with decoded audio dicts at 8 kHz."""
        from datasets import Dataset

        array = np.sin(np.arange(1600) / 10.0).astype(np.float32)
        return Dataset.from_dict(
            {
                "audio": [
                    {"array": array, "sampling_rate": 8000},
                    {"array": array[:400], "sampling_rate": 8000},
                ],
                "text": ["hola", "chao"],
                "label": [0, 1],
            }
        )

    def test_asr_builds_tokenizer_and_resamples(self):
        fake = self._fake_dataset()
        config = {
            "dataset_name": "fake/audio",
            "split": "test",
            "audio_column": "audio",
            "text_column": "text",
            "max_duration_sec": 10.0,
            "resample": True,
        }
        with mock.patch(
            "datasets.load_dataset", return_value=fake
        ) as loader:
            dataset = AudioDataset(config, {"sample_rate": 16000}, "asr")

        loader.assert_called_once_with("fake/audio", split="test")
        assert len(dataset) == 2
        assert dataset.tokenizer is not None
        assert dataset.vocab_size == dataset.tokenizer.vocab_size

        item = dataset[0]
        # 1600 samples at 8 kHz → 3200 at 16 kHz (linear/torchaudio resample).
        assert abs(item["waveform"].size(0) - 3200) <= 2
        assert item["labels"].tolist() == dataset.tokenizer.encode("hola")
        assert int(item["target_lengths"]) == 4
        assert int(item["input_lengths"]) == item["waveform"].size(0)

    def test_duration_clipping(self):
        fake = self._fake_dataset()
        config = {
            "dataset_name": "fake/audio",
            "text_column": "text",
            "max_duration_sec": 0.05,
            "resample": True,
        }
        with mock.patch("datasets.load_dataset", return_value=fake):
            dataset = AudioDataset(config, {"sample_rate": 16000}, "asr")
        assert dataset[0]["waveform"].size(0) == int(0.05 * 16000)

    def test_classification_requires_label_column(self):
        fake = self._fake_dataset()
        with mock.patch("datasets.load_dataset", return_value=fake):
            with pytest.raises(ValueError, match="label_column"):
                AudioDataset({"dataset_name": "fake/audio"}, {}, "audio_classification")

    def test_classification_returns_class_index(self):
        fake = self._fake_dataset()
        config = {
            "dataset_name": "fake/audio",
            "label_column": "label",
            "max_duration_sec": 10.0,
        }
        with mock.patch("datasets.load_dataset", return_value=fake):
            dataset = AudioDataset(config, {"sample_rate": 16000}, "audio_classification")
        assert int(dataset[1]["labels"]) == 1
        assert int(dataset[1]["target_lengths"]) == 0

    def test_raw_feature_keeps_channels(self):
        from datasets import Dataset

        stereo = np.stack(
            [np.sin(np.arange(800) / 10.0), np.cos(np.arange(800) / 10.0)], axis=-1
        ).astype(np.float32)
        fake = Dataset.from_dict(
            {"audio": [{"array": stereo, "sampling_rate": 16000}], "label": [2]}
        )
        config = {"dataset_name": "fake/audio", "label_column": "label"}
        with mock.patch("datasets.load_dataset", return_value=fake):
            dataset = AudioDataset(
                config, {"feature": "raw", "in_channels": 2}, "audio_classification"
            )
        assert dataset[0]["waveform"].shape == (800, 2)

    def test_missing_source_raises(self):
        with pytest.raises(ValueError, match="dataset_name"):
            AudioDataset({}, {}, "asr")

    def test_missing_text_column_raises(self):
        from datasets import Dataset

        fake = Dataset.from_dict({"audio": [{"array": np.zeros(80, dtype=np.float32), "sampling_rate": 16000}]})
        with mock.patch("datasets.load_dataset", return_value=fake):
            with pytest.raises(ValueError, match="text_column"):
                AudioDataset({"dataset_name": "fake/audio"}, {}, "asr")


class TestCollate:
    def test_pads_waveforms_and_targets(self):
        dataset = DummyAudioDataset(length=3, task="asr", vocab_size=10, max_frames=100)
        batch = [dataset[0], dataset[1], dataset[2]]
        collated = audio_collate_fn(batch)
        assert collated["waveform"].shape == (3, 100)
        assert collated["labels"].shape[0] == 3
        assert collated["labels"].shape[1] == max(
            int(item["labels"].numel()) for item in batch
        )
        assert collated["input_lengths"].tolist() == [100, 100, 100]
        assert collated["target_lengths"].tolist() == [
            int(item["target_lengths"]) for item in batch
        ]
        assert "task" not in collated

    def test_pads_multichannel_waveforms(self):
        short = {
            "waveform": torch.zeros(10, 4),
            "labels": torch.tensor([1, 2]),
            "input_lengths": torch.tensor(10),
            "target_lengths": torch.tensor(2),
            "task": "asr",
        }
        long = {
            "waveform": torch.ones(17, 4),
            "labels": torch.tensor([3]),
            "input_lengths": torch.tensor(17),
            "target_lengths": torch.tensor(1),
            "task": "asr",
        }
        collated = audio_collate_fn([short, long])
        assert collated["waveform"].shape == (2, 17, 4)
        # Zero padding after the short sample.
        assert torch.count_nonzero(collated["waveform"][0, 10:]) == 0
        assert collated["labels"].tolist() == [[1, 2], [3, 0]]
        assert collated["input_lengths"].tolist() == [10, 17]
