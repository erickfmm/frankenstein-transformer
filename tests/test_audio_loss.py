#!/usr/bin/env python3
"""Tests for the audio task losses in :class:`TitanTrainer`.

Builds tiny :class:`FrankensteinAudio` models and exercises ``compute_asr_loss``
(CTC and RNN-Transducer heads) plus ``compute_audio_classification_loss`` on
CPU, asserting finite scalar losses, gradient flow and auxiliary-loss handling.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.model.config import FrankensteinModelConfig  # noqa: E402
from src.model.frankenstein_audio import FrankensteinAudio  # noqa: E402
from src.training.trainer import TitanTrainer, TrainingConfig  # noqa: E402


def _tiny_audio_config(**overrides) -> FrankensteinModelConfig:
    params = dict(
        vocab_size=16,
        hidden_size=32,
        num_layers=1,
        num_loops=1,
        num_heads=4,
        num_experts=2,
        top_k_experts=1,
        dropout=0.0,
        norm_type="layer_norm",
        use_bitnet=False,
        layer_pattern=["standard_attn"],
        use_moe=False,
        ode_solver="rk4",
        ode_steps=1,
        ffn_hidden_size=64,
        ffn_activation="gelu",
        positional_encoding="sinusoidal_absolute",
        audio_n_mels=16,
        audio_conv_channels=4,
        audio_sample_rate=16000,
        audio_window_ms=25.0,
        audio_stride_ms=10.0,
        audio_n_fft=400,
        num_classes=3,
    )
    params.update(overrides)
    return FrankensteinModelConfig(**params)


def _make_trainer(model, task, tmp_path):
    training_config = TrainingConfig(
        csv_log_path=str(tmp_path / "metrics.csv"),
        gpu_temp_guard_enabled=False,
        enable_block_grad_norms=False,
        gradient_accumulation_steps=1,
    )
    return TitanTrainer(
        model,
        model.config,
        training_config=training_config,
        device="cpu",
        task=task,
    )


def _batch(batch_size=2, num_frames=1600, num_labels=2):
    waveform = torch.randn(batch_size, num_frames) * 0.1
    labels = torch.randint(1, 6, (batch_size, num_labels), dtype=torch.long)
    input_lengths = torch.full((batch_size,), num_frames, dtype=torch.long)
    target_lengths = torch.full((batch_size,), num_labels, dtype=torch.long)
    return waveform, labels, input_lengths, target_lengths


class TestAsrLoss:
    def test_ctc_loss_and_cer(self, tmp_path):
        model = FrankensteinAudio(_tiny_audio_config(asr_loss="ctc"))
        trainer = _make_trainer(model, "asr", tmp_path)
        try:
            loss, cer = trainer.compute_asr_loss(*_batch())
            assert loss.dim() == 0
            assert torch.isfinite(loss)
            assert loss.item() > 0.0
            assert 0.0 <= float(cer)
            loss.backward()
            assert any(
                param.grad is not None and torch.isfinite(param.grad).all()
                for param in model.parameters()
            )
        finally:
            trainer.close()

    def test_transducer_loss_and_cer(self, tmp_path):
        model = FrankensteinAudio(
            _tiny_audio_config(
                asr_loss="transducer", asr_decoder_dim=16, asr_decoder_layers=1
            )
        )
        trainer = _make_trainer(model, "asr", tmp_path)
        try:
            loss, cer = trainer.compute_asr_loss(*_batch())
            assert loss.dim() == 0
            assert torch.isfinite(loss)
            assert loss.item() > 0.0
            assert 0.0 <= float(cer)
            loss.backward()
        finally:
            trainer.close()

    def test_auxiliary_losses_are_added(self, tmp_path):
        model = FrankensteinAudio(_tiny_audio_config(asr_loss="ctc"))
        trainer = _make_trainer(model, "asr", tmp_path)
        try:
            loss, _ = trainer.compute_asr_loss(*_batch())
            aux = torch.tensor(0.25, requires_grad=True)
            model.last_auxiliary_losses = {"test_router": aux}
            combined = trainer._add_audio_auxiliary_losses(loss)
            assert float(combined - loss) == pytest.approx(0.25, abs=1e-6)
        finally:
            trainer.close()

    def test_non_finite_auxiliary_loss_raises(self, tmp_path):
        model = FrankensteinAudio(_tiny_audio_config(asr_loss="ctc"))
        trainer = _make_trainer(model, "asr", tmp_path)
        try:
            loss, _ = trainer.compute_asr_loss(*_batch())
            model.last_auxiliary_losses = {
                "bad": torch.tensor(float("nan"), requires_grad=True)
            }
            with pytest.raises(RuntimeError, match="Non-finite auxiliary"):
                trainer._add_audio_auxiliary_losses(loss)
        finally:
            trainer.close()


class TestAudioClassificationLoss:
    def test_loss_and_accuracy(self, tmp_path):
        model = FrankensteinAudio(_tiny_audio_config())
        trainer = _make_trainer(model, "audio_classification", tmp_path)
        try:
            waveform = torch.randn(3, 1600) * 0.1
            labels = torch.tensor([0, 1, 2], dtype=torch.long)
            loss, accuracy = trainer.compute_audio_classification_loss(
                waveform, labels
            )
            assert loss.dim() == 0
            assert torch.isfinite(loss)
            assert 0.0 <= accuracy.item() <= 1.0
            loss.backward()
        finally:
            trainer.close()

    def test_forward_audio_logits_shapes(self, tmp_path):
        model = FrankensteinAudio(_tiny_audio_config())
        trainer = _make_trainer(model, "audio_classification", tmp_path)
        try:
            logits = trainer._forward_audio_logits(
                torch.randn(2, 1600) * 0.1, task="audio_classification"
            )
            assert logits.shape == (2, model.config.num_classes)
        finally:
            trainer.close()


class TestSpecAugmentFrontend:
    """SpecAugment now lives in the model frontend (``audio.spec_augment``)."""

    def test_masks_apply_only_while_training(self):
        config = _tiny_audio_config(
            audio_spec_augment_enabled=True,
            audio_spec_augment_freq_mask_width=4,
            audio_spec_augment_time_masks=2,
            audio_spec_augment_time_mask_ratio=0.5,
        )
        model = FrankensteinAudio(config)
        waveform = torch.randn(2, 1600) * 0.1

        model.eval()
        eval_a = model.encode(waveform)
        eval_b = model.encode(waveform)
        assert torch.allclose(eval_a, eval_b)

        model.train()
        torch.manual_seed(0)
        train_out = model.encode(waveform)
        assert train_out.shape == eval_a.shape
        assert not torch.allclose(train_out, eval_a)

    def test_masking_zeroes_frequency_and_time_bands(self):
        config = _tiny_audio_config(
            audio_spec_augment_enabled=True,
            audio_spec_augment_freq_mask_width=8,
            audio_spec_augment_time_masks=3,
            audio_spec_augment_time_mask_ratio=0.5,
        )
        frontend = FrankensteinAudio(config).frontend
        features = torch.randn(2, 16, 12)
        torch.manual_seed(0)
        masked = frontend._spec_augment(features)
        assert masked.shape == features.shape
        assert int((masked == 0).sum()) > int((features == 0).sum())
        # Unmasked entries keep their original values.
        keep = masked != 0
        assert torch.allclose(masked[keep], features[keep])
