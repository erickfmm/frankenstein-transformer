#!/usr/bin/env python3
"""Audio datasets for the ``asr`` and ``audio_classification`` tasks.

This module hosts the data layer of :class:`~src.model.frankenstein_audio`:

* :class:`CharacterTokenizer` — a deterministic character-level tokenizer used
  as the ASR target vocabulary (the model's CTC/transducer head is sized from
  it at runtime).
* :class:`AudioDataset` — loads waveforms + transcripts/labels from a
  HuggingFace dataset (``datasets.load_dataset``) or a local saved dataset
  (``datasets.load_from_disk``), resamples them to ``audio.sample_rate``,
  clips them to ``audio_dataset.max_duration_sec`` and tokenises the target
  text for ASR.
* :class:`DummyAudioDataset` — synthetic waveforms / targets for tests and
  smoke configs, mirroring :class:`~src.training.vision_dataset.DummyImageDataset`.
* :func:`audio_collate_fn` / :func:`build_audio_dataloader_from_config` —
  padding collate (waveforms zero-filled, targets blank-padded) and a
  config-driven DataLoader builder.

SpecAugment lives in the model frontend (``src/model/frankenstein_audio.py``,
``audio.spec_augment``), not here, so augmentation follows ``model.train()``
automatically.

``torchaudio`` is optional: resampling falls back to
``torch.nn.functional.interpolate`` when it is unavailable, keeping CPU CI
torchaudio-free (D7).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

try:  # pragma: no cover - exercised only when the extra is installed
    import torchaudio as _torchaudio

    _HAS_TORCHAUDIO = True
except Exception:  # noqa: BLE001 - optional dependency
    _torchaudio = None
    _HAS_TORCHAUDIO = False

__all__ = [
    "CharacterTokenizer",
    "AudioDataset",
    "DummyAudioDataset",
    "audio_collate_fn",
    "build_audio_dataloader_from_config",
    "HAS_TORCHAUDIO",
]

#: Whether ``torchaudio`` is importable (native resampling fast path).
HAS_TORCHAUDIO = _HAS_TORCHAUDIO


def _audio_value(audio_config: Optional[Dict[str, Any]], key: str, default: Any) -> Any:
    """Read ``audio.<key>`` from either the raw block or the flat config dict.

    Args:
        audio_config: The raw ``audio:`` YAML block, a flat
            ``FrankensteinModelConfig``-style dict, or ``None``.
        key: Leaf name inside the ``audio:`` block (``sample_rate``,
            ``feature``, ``in_channels``, ...).
        default: Value returned when neither spelling is present.

    Returns:
        The resolved configuration value.
    """
    if not isinstance(audio_config, dict):
        return default
    flat_key = f"audio_{key}"
    if flat_key in audio_config:
        return audio_config[flat_key]
    return audio_config.get(key, default)


class CharacterTokenizer:
    """Deterministic character-level tokenizer for ASR targets.

    The vocabulary is built from the sorted set of characters in the training
    corpus: ``0`` is the CTC/RNN-T blank, ``1`` is the unknown token and the
    characters occupy ids ``2..N+1``. Sorting keeps the mapping reproducible
    across runs (the model's vocabulary size is persisted, so checkpoints stay
    loadable as long as the same corpus is used).

    Args:
        texts: Iterable of corpus strings used to build the vocabulary. Empty
            or ``None`` yields a minimal ``<blank>/<unk>`` vocabulary.
        unk_token: Name of the unknown token (id ``1``).
        blank_id: Id reserved for the blank symbol (must be ``0``).

    Attributes:
        blank_id: Blank token id (``0``).
        unk_id: Unknown token id (``1``).
        unk_token: Unknown token string.
        chars: Sorted list of corpus characters.
    """

    def __init__(
        self,
        texts: Optional[Sequence[str]] = None,
        unk_token: str = "<unk>",
        blank_id: int = 0,
    ) -> None:
        if int(blank_id) != 0:
            raise ValueError("CharacterTokenizer reserves id 0 for the blank symbol")
        self.blank_id = 0
        self.unk_id = 1
        self.unk_token = str(unk_token)
        corpus = "".join(str(text) for text in (texts or []))
        self.chars: List[str] = sorted(set(corpus))
        self.char_to_id: Dict[str, int] = {
            char: index + 2 for index, char in enumerate(self.chars)
        }
        self.id_to_char: Dict[int, str] = {
            index: char for char, index in self.char_to_id.items()
        }

    @property
    def vocab_size(self) -> int:
        """Vocabulary size (blank + unk + observed characters)."""
        return len(self.chars) + 2

    def __len__(self) -> int:
        """Alias for :attr:`vocab_size`."""
        return self.vocab_size

    def encode(self, text: Any) -> List[int]:
        """Tokenise a string into ids (unknown characters map to ``unk_id``).

        Args:
            text: Input text.

        Returns:
            A list of token ids.
        """
        return [self.char_to_id.get(char, self.unk_id) for char in str(text)]

    def decode(self, ids: Sequence[int]) -> str:
        """Detokenise ids, skipping blanks and unknown tokens.

        Args:
            ids: Token ids.

        Returns:
            The decoded string.
        """
        return "".join(
            self.id_to_char.get(int(token), "")
            for token in ids
            if int(token) not in (self.blank_id, self.unk_id)
        )


def _as_waveform_tensor(value: Any) -> tuple[torch.Tensor, Optional[int]]:
    """Convert a dataset audio value into a ``(T[, C])`` float32 tensor.

    Args:
        value: A HuggingFace audio dict (``{"array", "sampling_rate"}``), a
            ``(array, sampling_rate)`` tuple, a numpy array, a torch tensor or
            a path string.

    Returns:
        Tuple ``(waveform, sampling_rate)``; the rate is ``None`` when the
        value does not carry one.

    Raises:
        ValueError: If a file path is given while ``torchaudio`` is missing,
            or the value type is unsupported.
    """
    if isinstance(value, dict):
        array = value.get("array", value.get("path"))
        sample_rate = value.get("sampling_rate")
    elif (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and isinstance(value[1], (int, float))
    ):
        array, sample_rate = value[0], int(value[1])
    else:
        array, sample_rate = value, None

    if isinstance(array, str):
        if not _HAS_TORCHAUDIO:  # pragma: no cover - optional dependency
            raise ValueError(
                "Audio values are file paths but torchaudio is not installed; "
                "install the `audio` extra or pre-decode the dataset"
            )
        waveform, sample_rate = _torchaudio.load(array)  # pragma: no cover
        return waveform.squeeze(0).float(), int(sample_rate)

    if isinstance(array, torch.Tensor):
        wave = array.detach().float()
    else:
        wave = torch.as_tensor(array, dtype=torch.float32)
    if wave.dim() == 1:
        pass
    elif wave.dim() == 2:
        # Convention: (T, C). A (C, T) layout with very few rows and many
        # columns is transposed so the long axis is time.
        if wave.size(0) <= 16 and wave.size(1) > wave.size(0):
            wave = wave.transpose(0, 1)
    else:
        raise ValueError(
            "Audio arrays must be 1-D or 2-D, got shape "
            f"{tuple(wave.shape)}"
        )
    return wave.contiguous(), (int(sample_rate) if sample_rate else None)


def _resample_waveform(
    waveform: torch.Tensor, source_rate: int, target_rate: int
) -> torch.Tensor:
    """Resample a ``(T[, C])`` waveform to ``target_rate``.

    Uses ``torchaudio.functional.resample`` when available and a linear
    ``F.interpolate`` fallback otherwise (CPU CI stays torchaudio-free).

    Args:
        waveform: Waveform tensor of shape ``(T,)`` or ``(T, C)``.
        source_rate: Original sample rate in Hz.
        target_rate: Target sample rate in Hz.

    Returns:
        The resampled waveform with the same rank.
    """
    source_rate = int(source_rate)
    target_rate = int(target_rate)
    if source_rate <= 0 or target_rate <= 0 or source_rate == target_rate:
        return waveform

    target_len = max(1, int(round(waveform.size(0) * target_rate / source_rate)))
    if _HAS_TORCHAUDIO:  # pragma: no cover - optional dependency
        resampled = _torchaudio.functional.resample(
            waveform, source_rate, target_rate
        )
        return resampled

    if waveform.dim() == 1:
        data = waveform.view(1, 1, -1)
        out = torch.nn.functional.interpolate(
            data, size=target_len, mode="linear", align_corners=False
        )
        return out.view(-1)
    # (T, C) → (1, C, T) → interpolate → (T, C)
    data = waveform.transpose(0, 1).unsqueeze(0)
    out = torch.nn.functional.interpolate(
        data, size=target_len, mode="linear", align_corners=False
    )
    return out.squeeze(0).transpose(0, 1).contiguous()


class AudioDataset(Dataset):
    """Waveform + transcript/label dataset for audio tasks.

    Args:
        dataset_config: The ``audio_dataset:`` YAML block (data source,
            columns, split, duration clipping, resampling).
        audio_config: The ``audio:`` YAML block (frontend kind, sample rate,
            channel count). A flat config dict is accepted too.
        task: ``"asr"`` or ``"audio_classification"``.
        split: Optional split override (otherwise ``dataset_config.split``).

    Raises:
        ValueError: If neither ``dataset_name`` nor ``dataset_dir`` is set, if
            the task is unknown, or if ``audio_classification`` has no
            ``label_column``.

    Attributes:
        task: Resolved task name.
        sample_rate: Target sample rate in Hz.
        feature: Frontend kind (``logmel``, ``spectrogram`` or ``raw``).
        in_channels: Channel count used by ``feature: raw``.
        tokenizer: :class:`CharacterTokenizer` (ASR only, otherwise ``None``).
        items: Loaded dataset rows.
    """

    def __init__(
        self,
        dataset_config: Dict[str, Any],
        audio_config: Dict[str, Any],
        task: str,
        split: Optional[str] = None,
    ) -> None:
        self.task = str(task or "asr").strip().lower()
        if self.task not in {"asr", "audio_classification"}:
            raise ValueError(
                "AudioDataset supports task 'asr' or 'audio_classification', "
                f"got {self.task!r}"
            )
        self.dataset_config = dict(dataset_config or {})
        self.audio_config = dict(audio_config or {})

        self.sample_rate = int(_audio_value(self.audio_config, "sample_rate", 16000))
        self.feature = str(
            _audio_value(self.audio_config, "feature", "logmel")
        ).lower()
        self.in_channels = int(_audio_value(self.audio_config, "in_channels", 1))

        self.dataset_name = str(self.dataset_config.get("dataset_name") or "").strip()
        self.dataset_dir = str(self.dataset_config.get("dataset_dir") or "").strip()
        self.split = str(
            split or self.dataset_config.get("split") or "train"
        ).strip()
        self.audio_column = str(self.dataset_config.get("audio_column") or "audio")
        self.text_column = str(self.dataset_config.get("text_column") or "text")
        self.label_column = self.dataset_config.get("label_column")
        self.max_duration_sec = float(
            self.dataset_config.get("max_duration_sec", 30.0) or 30.0
        )
        self.resample = bool(self.dataset_config.get("resample", True))

        if self.task == "audio_classification" and not self.label_column:
            raise ValueError(
                "task=audio_classification requires audio_dataset.label_column "
                "(name of the integer class-label column)"
            )

        if not self.dataset_name and not self.dataset_dir:
            raise ValueError(
                "AudioDataset requires audio_dataset.dataset_name (HuggingFace "
                "id) or audio_dataset.dataset_dir (saved dataset directory)"
            )

        self.items: List[Dict[str, Any]] = []
        self._load_items()

        self.tokenizer: Optional[CharacterTokenizer] = None
        if self.task == "asr":
            texts = [
                str(item.get(self.text_column, ""))
                for item in self.items
                if self.text_column in item
            ]
            if not texts:
                raise ValueError(
                    f"task=asr requires the audio_dataset.text_column "
                    f"({self.text_column!r}) to be present in the dataset"
                )
            self.tokenizer = CharacterTokenizer(texts)

    def _load_items(self) -> None:
        """Load the dataset rows via ``datasets`` (hub id or saved dir)."""
        from datasets import load_dataset, load_from_disk

        if self.dataset_name:
            loaded = load_dataset(self.dataset_name, split=self.split)
        else:
            loaded = load_from_disk(self.dataset_dir)
            if hasattr(loaded, "column_names") and isinstance(
                loaded.column_names, dict
            ):
                # A DatasetDict: pick the requested split (or the first one).
                key = self.split if self.split in loaded else next(iter(loaded))
                loaded = loaded[key]
        for row in loaded:
            self.items.append(dict(row))

    def __len__(self) -> int:
        """Number of examples."""
        return len(self.items)

    @property
    def vocab_size(self) -> int:
        """ASR vocabulary size (``None``-safe fallback of ``2``)."""
        return self.tokenizer.vocab_size if self.tokenizer is not None else 2

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one waveform + target example.

        Args:
            idx: Row index.

        Returns:
            Dict with ``waveform`` (float32 ``(T,)`` or ``(T, C)``),
            ``labels`` (long), ``input_lengths``, ``target_lengths`` and the
            ``task`` string.

        Raises:
            ValueError: If the waveform column is missing.
        """
        item = self.items[idx]
        if self.audio_column not in item:
            raise ValueError(
                f"audio_dataset.audio_column={self.audio_column!r} not found in "
                f"the dataset row (columns: {sorted(item)})"
            )
        waveform, source_rate = _as_waveform_tensor(item[self.audio_column])
        if self.resample and source_rate and source_rate != self.sample_rate:
            waveform = _resample_waveform(waveform, source_rate, self.sample_rate)

        if self.feature != "raw" and waveform.dim() == 2:
            # Mel/STFT frontends consume mono audio.
            waveform = waveform.mean(dim=-1)

        max_samples = int(self.max_duration_sec * self.sample_rate)
        if max_samples > 0 and waveform.size(0) > max_samples:
            waveform = waveform[:max_samples]

        result: Dict[str, Any] = {
            "waveform": waveform.float().contiguous(),
            "input_lengths": torch.tensor(waveform.size(0), dtype=torch.long),
            "task": self.task,
        }

        if self.task == "asr":
            assert self.tokenizer is not None  # nosec - built for ASR above
            target_ids = self.tokenizer.encode(item.get(self.text_column, ""))
            if not target_ids:
                target_ids = [self.tokenizer.unk_id]
            result["labels"] = torch.tensor(target_ids, dtype=torch.long)
            result["target_lengths"] = torch.tensor(
                len(target_ids), dtype=torch.long
            )
        else:
            label = int(item.get(str(self.label_column), 0))
            result["labels"] = torch.tensor(label, dtype=torch.long)
            result["target_lengths"] = torch.tensor(0, dtype=torch.long)

        return result


class DummyAudioDataset(Dataset):
    """Deterministic synthetic audio dataset for tests and smoke configs.

    Each example is a sum of sines plus low-amplitude noise, generated from a
    per-index seeded generator so repeated runs and workers are reproducible.

    Args:
        length: Number of examples.
        audio_config: The ``audio:`` block (or flat config dict).
        task: ``"asr"`` or ``"audio_classification"``.
        num_classes: Number of classification classes.
        vocab_size: ASR vocabulary size (targets stay in ``[1, vocab_size)``).
        max_frames: Waveform length in samples.

    Attributes:
        length: Number of examples.
        sample_rate: Sample rate in Hz.
        in_channels: Number of signal channels.
        vocab_size: ASR vocabulary size (``None`` tokenizer for dummy data).
        tokenizer: Always ``None`` (targets are already token ids).
    """

    def __init__(
        self,
        length: int = 16,
        audio_config: Optional[Dict[str, Any]] = None,
        task: str = "asr",
        num_classes: int = 2,
        vocab_size: int = 32,
        max_frames: int = 1600,
    ) -> None:
        self.length = max(1, int(length))
        self.audio_config = dict(audio_config or {})
        self.task = str(task or "asr").strip().lower()
        if self.task not in {"asr", "audio_classification"}:
            raise ValueError(
                "DummyAudioDataset supports task 'asr' or "
                f"'audio_classification', got {self.task!r}"
            )
        self.num_classes = max(2, int(num_classes))
        self.vocab_size = max(4, int(vocab_size))
        self.max_frames = max(64, int(max_frames))
        self.sample_rate = int(_audio_value(self.audio_config, "sample_rate", 16000))
        self.feature = str(
            _audio_value(self.audio_config, "feature", "logmel")
        ).lower()
        self.in_channels = int(_audio_value(self.audio_config, "in_channels", 1))
        self.tokenizer = None

    def __len__(self) -> int:
        """Number of synthetic examples."""
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate one synthetic example (deterministic in ``idx``).

        Args:
            idx: Example index.

        Returns:
            Same keys as :meth:`AudioDataset.__getitem__`.
        """
        generator = torch.Generator()
        generator.manual_seed(1234 + int(idx))
        num_frames = self.max_frames
        time = torch.arange(num_frames, dtype=torch.float32) / float(self.sample_rate)

        def _tone() -> torch.Tensor:
            freq = 80.0 + float(torch.randint(40, 400, (1,), generator=generator))
            phase = float(torch.rand(1, generator=generator)) * 6.283185
            wave = torch.sin(2.0 * 3.14159265 * freq * time + phase)
            wave = wave + 0.5 * torch.sin(
                2.0 * 3.14159265 * (2.0 * freq) * time + phase
            )
            noise = 0.01 * torch.randn(num_frames, generator=generator)
            return (wave + noise) * 0.25

        channels = self.in_channels if self.feature == "raw" else 1
        if channels > 1:
            waveform = torch.stack([_tone() for _ in range(channels)], dim=-1)
        else:
            waveform = _tone()

        result: Dict[str, Any] = {
            "waveform": waveform.float().contiguous(),
            "input_lengths": torch.tensor(num_frames, dtype=torch.long),
            "task": self.task,
        }
        if self.task == "asr":
            target_len = 1 + int(idx) % 3
            result["labels"] = torch.randint(
                1, self.vocab_size, (target_len,), generator=generator, dtype=torch.long
            )
            result["target_lengths"] = torch.tensor(target_len, dtype=torch.long)
        else:
            result["labels"] = torch.tensor(
                int(idx) % self.num_classes, dtype=torch.long
            )
            result["target_lengths"] = torch.tensor(0, dtype=torch.long)
        return result


def audio_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Pad waveforms (zeros) and targets (blank ``0``) into dict batches.

    Args:
        batch: List of items returned by :class:`AudioDataset` /
            :class:`DummyAudioDataset`.

    Returns:
        Dict with ``waveform`` ``(B, T[, C])``, ``labels``, ``input_lengths``
        and ``target_lengths``. The per-item ``task`` string is dropped so the
        trainer can move every value to the device.
    """
    waveforms = [item["waveform"] for item in batch]
    max_frames = max(int(wave.size(0)) for wave in waveforms)
    num_channels = waveforms[0].size(-1) if waveforms[0].dim() == 2 else None

    if num_channels is None:
        padded = waveforms[0].new_zeros((len(waveforms), max_frames))
    else:
        padded = waveforms[0].new_zeros((len(waveforms), max_frames, num_channels))
    for index, wave in enumerate(waveforms):
        padded[index, : wave.size(0)] = wave

    target_lengths = torch.stack(
        [item["target_lengths"].to(dtype=torch.long) for item in batch]
    )
    labels_list = [item["labels"].to(dtype=torch.long) for item in batch]
    if labels_list[0].dim() == 0:
        labels = torch.stack(labels_list)
    else:
        max_labels = max(int(labels.size(0)) for labels in labels_list)
        labels = labels_list[0].new_zeros((len(labels_list), max_labels))
        for index, target in enumerate(labels_list):
            labels[index, : target.size(0)] = target

    return {
        "waveform": padded,
        "labels": labels,
        "input_lengths": torch.stack(
            [item["input_lengths"].to(dtype=torch.long) for item in batch]
        ),
        "target_lengths": target_lengths,
    }


def build_audio_dataloader_from_config(
    loaded: Any,
    resolved_device: str = "cpu",
    cli_batch_size: Optional[int] = None,
    task: Optional[str] = None,
    *,
    shuffle: bool = True,
    split: Optional[str] = None,
) -> DataLoader:
    """Build an audio DataLoader from a ``LoadedTrainingConfig``.

    Falls back to :class:`DummyAudioDataset` when the ``audio_dataset`` block
    has no ``dataset_name`` / ``dataset_dir`` (smoke configs), mirroring the
    vision builder. The dataset's ASR vocabulary size is exposed through
    ``dataset.vocab_size`` for the caller to inject into the model config
    before construction.

    Args:
        loaded: Validated :class:`~src.training.config_loader.LoadedTrainingConfig`.
        resolved_device: Resolved torch device string (for ``pin_memory``).
        cli_batch_size: Optional batch-size override (CLI / engine argument).
        task: Optional task override (defaults to ``loaded.task``).
        shuffle: Whether to shuffle (train) or not (validation).
        split: Optional split override.

    Returns:
        A ``DataLoader`` yielding collated dict batches.

    Raises:
        ValueError: If the task is not an audio task or the batch size is not
            positive.
    """
    task = str(task or getattr(loaded, "task", "") or "").strip().lower()
    if task not in {"asr", "audio_classification"}:
        raise ValueError(
            "build_audio_dataloader_from_config supports asr and "
            f"audio_classification, got {task!r}"
        )

    dataset_config = dict(getattr(loaded, "audio_dataset_config", None) or {})
    audio_config = dict(getattr(loaded, "audio_config", None) or {})
    training_runtime = getattr(loaded, "training_runtime", None) or {}

    batch_size = training_runtime.get("batch_size", cli_batch_size or 4)
    if cli_batch_size is not None:
        if int(cli_batch_size) <= 0:
            raise ValueError("--batch-size must be > 0")
        batch_size = cli_batch_size
    batch_size = max(1, int(batch_size))

    dataset_name = str(dataset_config.get("dataset_name") or "").strip()
    dataset_dir = str(dataset_config.get("dataset_dir") or "").strip()
    if dataset_name or dataset_dir:
        dataset: Dataset = AudioDataset(
            dataset_config, audio_config, task, split=split
        )
    else:
        model_config = getattr(loaded, "model_config", None)
        dataset = DummyAudioDataset(
            length=16,
            audio_config=audio_config,
            task=task,
            num_classes=int(getattr(model_config, "num_classes", 2) or 2),
            vocab_size=int(getattr(model_config, "vocab_size", 32) or 32),
            max_frames=1600,
        )

    num_workers = int(training_runtime.get("dataloader_workers", 0) or 0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle),
        num_workers=num_workers,
        pin_memory=str(resolved_device).startswith("cuda"),
        collate_fn=audio_collate_fn,
    )
