#!/usr/bin/env python3
"""Tests for the RNN-Transducer loss, greedy decoders and CER.

The pure-torch RNN-T implementation is validated against brute-force alignment
enumeration on tiny lattices (no torchaudio needed); the optional torchaudio
comparison is skipped when the extra is not installed.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.training.rnnt import (  # noqa: E402
    character_error_rate,
    ctc_greedy_decode,
    greedy_rnnt_decode,
    rnnt_loss,
)


def _brute_force_rnnt(logits, targets, input_length, target_length, blank=0):
    """Exhaustive sum over one-symbol-per-frame alignments (tiny lattices)."""
    log_probs = F.log_softmax(logits, dim=-1)
    total = 0.0

    def recurse(frame, consumed, probability):
        nonlocal total
        if frame == input_length:
            if consumed == target_length:
                total += probability
            return
        # Emit blank at this frame and stay on the same label state.
        recurse(
            frame + 1,
            consumed,
            probability * float(log_probs[0, frame, consumed, blank].exp()),
        )
        # Emit the next target label at this frame.
        if consumed < target_length:
            token = int(targets[consumed])
            recurse(
                frame + 1,
                consumed + 1,
                probability * float(log_probs[0, frame, consumed, token].exp()),
            )

    recurse(0, 0, 1.0)
    return -math.log(total)


def _reference(logits, targets, target_length):
    input_length = int(logits.size(1))
    return rnnt_loss(
        logits,
        targets,
        torch.tensor([input_length]),
        torch.tensor([target_length]),
        blank=0,
        backend="torch",
    )


class TestRnntLoss:
    @pytest.mark.parametrize(
        "num_frames,num_labels",
        [(1, 0), (1, 1), (2, 1), (2, 2), (3, 2)],
    )
    def test_matches_brute_force_enumeration(self, num_frames, num_labels):
        generator = torch.Generator().manual_seed(7)
        vocab = 4
        logits = torch.randn(
            1, num_frames, num_labels + 1, vocab, generator=generator
        )
        targets = torch.randint(
            1, vocab, (1, max(1, num_labels)), generator=generator
        )
        expected = _brute_force_rnnt(
            logits, targets[0], num_frames, num_labels, blank=0
        )
        actual = float(_reference(logits, targets, num_labels))
        assert actual == pytest.approx(expected, abs=1e-4)

    def test_matches_torchaudio_when_available(self):
        torchaudio = pytest.importorskip("torchaudio")
        from torchaudio.functional import rnnt_loss as ta_rnnt_loss

        generator = torch.Generator().manual_seed(11)
        logits = torch.randn(2, 4, 3, 5, generator=generator)
        targets = torch.randint(1, 5, (2, 2), generator=generator)
        input_lengths = torch.tensor([4, 3])
        target_lengths = torch.tensor([2, 1])

        expected = ta_rnnt_loss(
            logits,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            clamp=-1.0,
            reduction="mean",
        )
        actual = rnnt_loss(
            logits,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            backend="torch",
        )
        assert torch.allclose(actual, expected, atol=1e-4)

    def test_gradients_flow(self):
        generator = torch.Generator().manual_seed(3)
        logits = torch.randn(2, 3, 2, 5, generator=generator, requires_grad=True)
        targets = torch.randint(1, 5, (2, 1), generator=generator)
        loss = rnnt_loss(
            logits,
            targets,
            torch.tensor([3, 2]),
            torch.tensor([1, 1]),
            backend="torch",
        )
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert float(logits.grad.abs().sum()) > 0.0

    def test_empty_targets_are_finite(self):
        generator = torch.Generator().manual_seed(5)
        logits = torch.randn(1, 3, 1, 4, generator=generator)
        loss = rnnt_loss(
            logits,
            torch.zeros(1, 1, dtype=torch.long),
            torch.tensor([3]),
            torch.tensor([0]),
            backend="torch",
        )
        assert torch.isfinite(loss)
        expected = _brute_force_rnnt(logits, torch.zeros(1), 3, 0)
        assert float(loss) == pytest.approx(expected, abs=1e-4)

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="backend"):
            rnnt_loss(
                torch.zeros(1, 2, 2, 4),
                torch.zeros(1, 1, dtype=torch.long),
                torch.tensor([2]),
                torch.tensor([1]),
                backend="warp",
            )


class TestCtcGreedyDecode:
    @staticmethod
    def _one_hot(sequences, vocab=4):
        logits = torch.full((len(sequences), len(sequences[0]), vocab), -10.0)
        logits[..., 0] = 10.0
        for b, sequence in enumerate(sequences):
            for t, token in enumerate(sequence):
                logits[b, t, token] = 10.0
                if token != 0:
                    logits[b, t, 0] = -10.0
        return logits

    def test_repeats_and_blanks_collapsed(self):
        logits = self._one_hot([[1, 1, 0, 2, 2], [1, 2, 2, 1, 3]])
        decoded = ctc_greedy_decode(logits, torch.tensor([5, 4]), blank=0)
        assert decoded == [[1, 2], [1, 2, 1]]

    def test_lengths_limit_frames(self):
        logits = self._one_hot([[1, 2, 3, 0, 0]])
        assert ctc_greedy_decode(logits, torch.tensor([2])) == [[1, 2]]

    def test_accepts_2d_and_defaults_to_full_length(self):
        logits = self._one_hot([[1, 0, 2]])[0]
        assert ctc_greedy_decode(logits) == [[1, 2]]


class _ScriptedTransducer(nn.Module):
    """Transducer stub whose joint logits follow a fixed token script."""

    def __init__(self, script, vocab=8, dim=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.rnn = nn.LSTM(dim, dim, batch_first=True)
        self.script = list(script)
        self.vocab = vocab
        self.calls = 0

    def joint_logits(self, encoder_out, decoder_out):
        token = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        logits = torch.full(
            (encoder_out.size(0), encoder_out.size(1), decoder_out.size(1), self.vocab),
            -10.0,
        )
        logits[..., token] = 10.0
        return logits


class TestGreedyRnntDecode:
    def test_emits_non_blanks_until_blank(self):
        transducer = _ScriptedTransducer([2, 3, 0, 0, 0])
        encoder_out = torch.zeros(1, 2, 4)
        assert greedy_rnnt_decode(encoder_out, transducer, blank=0) == [[2, 3]]

    def test_max_symbols_per_step_stops_emission(self):
        transducer = _ScriptedTransducer([1, 2, 3, 4, 0, 0, 0])
        encoder_out = torch.zeros(1, 2, 4)
        decoded = greedy_rnnt_decode(
            encoder_out, transducer, max_symbols_per_step=2, blank=0
        )
        # Frame 0 emits 1 and 2 (hits the cap), frame 1 emits 3 and 4.
        assert decoded == [[1, 2, 3, 4]]

    def test_batch_members_are_independent(self):
        transducer = _ScriptedTransducer([5, 0, 5, 0])
        encoder_out = torch.zeros(2, 1, 4)
        assert greedy_rnnt_decode(encoder_out, transducer, blank=0) == [[5], [5]]


class TestCharacterErrorRate:
    def test_identical_sequences(self):
        assert character_error_rate([1, 2, 3], [1, 2, 3]) == 0.0

    def test_substitution(self):
        assert character_error_rate([1, 2, 4], [1, 2, 3]) == pytest.approx(1 / 3)

    def test_deletion_and_insertion(self):
        assert character_error_rate([1, 2], [1, 2, 3]) == pytest.approx(1 / 3)
        assert character_error_rate([1, 2, 3, 4], [1, 2, 3]) == pytest.approx(1 / 3)

    def test_empty_reference(self):
        assert character_error_rate([], []) == 0.0
        assert character_error_rate([1], []) == pytest.approx(1.0)

    def test_string_inputs(self):
        assert character_error_rate("hola", "hola") == 0.0
        assert character_error_rate("hole", "hola") == pytest.approx(1 / 4)

    def test_batch_mean(self):
        predictions = [[1], [1, 2]]
        targets = [[1], [1, 2, 3]]
        assert character_error_rate(predictions, targets) == pytest.approx(1 / 6)

    def test_mismatched_batches_raise(self):
        with pytest.raises(ValueError, match="same number"):
            character_error_rate([[1]], [[1], [2]])


class TestCtcLossTraining:
    def test_ctc_loss_decreases(self):
        torch.manual_seed(0)
        model = nn.Linear(8, 5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        features = torch.randn(2, 6, 8)
        targets = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
        input_lengths = torch.tensor([6, 6])
        target_lengths = torch.tensor([2, 2])

        losses = []
        for _ in range(15):
            logits = model(features)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = F.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=0,
                zero_infinity=True,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
            assert math.isfinite(losses[-1])

        assert losses[-1] < losses[0]
