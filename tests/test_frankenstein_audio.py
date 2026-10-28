#!/usr/bin/env python3
"""Tests for :class:`FrankensteinAudio` (Conformer-ready audio encoder).

Covers the waveform frontends (log-mel, spectrogram, raw multichannel
signals), the exact ``output_lengths`` bookkeeping, CTC/RNN-Transducer and
audio-classification heads, pooling modes, forced ``mode: encoder``, gradient
flow, mixer combinations (``conformer_attn`` + ``standard_attn``), mHC /
AttnRes / Hyperloop smoke tests, positional-encoding length changes and
``configure_optimizers``.
"""

from __future__ import annotations

import pytest

try:
    import torch
    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_audio import (
        AudioFrontend,
        FrankensteinAudio,
        TransducerHead,
    )
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is a hard test dependency
    TORCH_AVAILABLE = False


def _tiny_audio_config(**overrides) -> "FrankensteinModelConfig":
    """Return a minimal valid audio config for fast CPU tests."""
    params = dict(
        vocab_size=64,
        hidden_size=64,
        num_layers=2,
        num_loops=1,
        num_heads=4,
        retention_heads=4,
        num_experts=2,
        top_k_experts=1,
        dropout=0.0,
        norm_type="layer_norm",
        use_bitnet=False,
        layer_pattern=["standard_attn"],
        use_moe=False,
        ode_solver="rk4",
        ode_steps=1,
        ffn_hidden_size=128,
        ffn_activation="gelu",
        positional_encoding="sinusoidal_absolute",
        audio_n_mels=32,
        audio_conv_channels=8,
        audio_sample_rate=16000,
        audio_window_ms=25.0,
        audio_stride_ms=10.0,
        audio_n_fft=400,
        num_classes=3,
    )
    params.update(overrides)
    return FrankensteinModelConfig(**params)


@pytest.fixture
def tiny_audio_config():
    return _tiny_audio_config()


@pytest.fixture
def dummy_waveform():
    return torch.randn(2, 1600)  # 0.1 s mono at 16 kHz


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestAudioFrontend:
    def test_logmel_shape(self, tiny_audio_config):
        frontend = AudioFrontend(tiny_audio_config)
        out = frontend(torch.randn(2, 1600))
        # 1 + 1600 // 160 = 11 frames → 6 → 3 after two stride-2 convs.
        assert out.shape == (2, 3, 64)

    def test_spectrogram_shape(self):
        cfg = _tiny_audio_config(audio_feature="spectrogram")
        frontend = AudioFrontend(cfg)
        out = frontend(torch.randn(2, 1600))
        assert out.shape == (2, 3, 64)

    def test_freq_after_accounts_for_both_strides(self, tiny_audio_config):
        frontend = AudioFrontend(tiny_audio_config)
        # n_mels=32 → 16 → 8 after the two stride-2 Conv2d layers.
        assert frontend.freq_after == 8
        spec = AudioFrontend(_tiny_audio_config(audio_feature="spectrogram"))
        # n_fft=400 → 201 one-sided bins → 101 → 51.
        assert spec.freq_after == 51

    def test_raw_multichannel_shape(self):
        cfg = _tiny_audio_config(audio_feature="raw", audio_in_channels=4)
        frontend = AudioFrontend(cfg)
        out = frontend(torch.randn(2, 512, 4))
        assert out.shape == (2, 128, 64)

    def test_raw_channel_mismatch_raises(self):
        cfg = _tiny_audio_config(audio_feature="raw", audio_in_channels=4)
        frontend = AudioFrontend(cfg)
        with pytest.raises(ValueError):
            frontend(torch.randn(2, 512, 3))

    def test_multichannel_waveform_is_mixed_to_mono(self, tiny_audio_config):
        frontend = AudioFrontend(tiny_audio_config)
        mono = frontend(torch.randn(2, 1600))
        stereo = frontend(torch.randn(2, 1600, 2))
        assert stereo.shape == mono.shape

    def test_output_lengths_logmel(self, tiny_audio_config):
        frontend = AudioFrontend(tiny_audio_config)
        lengths = torch.tensor([1600, 8000, 320])
        # frames = 1 + T // 160, then //2 twice (ceil halves).
        assert frontend.output_lengths(lengths).tolist() == [3, 13, 1]
        for t, expected in zip([1600, 8000, 320], [3, 13, 1]):
            out = frontend(torch.randn(1, t))
            assert out.shape[1] == expected

    def test_output_lengths_spectrogram(self):
        frontend = AudioFrontend(_tiny_audio_config(audio_feature="spectrogram"))
        assert frontend.output_lengths(torch.tensor([1600, 3200])).tolist() == [3, 6]

    def test_output_lengths_raw(self):
        frontend = AudioFrontend(_tiny_audio_config(audio_feature="raw"))
        # (T + 1) // 2 twice.
        assert frontend.output_lengths(torch.tensor([512, 400, 1])).tolist() == [
            128,
            100,
            1,
        ]
        assert frontend(torch.randn(1, 400)).shape[1] == 100


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestFrankensteinAudio:
    def test_mode_forced_to_encoder(self):
        cfg = _tiny_audio_config(mode="decoder")
        model = FrankensteinAudio(cfg)
        assert model.config.mode == "encoder"
        assert model.config.model_class == "frankenstein_audio"

    def test_ctc_asr_logits_logmel(self, tiny_audio_config, dummy_waveform):
        model = FrankensteinAudio(tiny_audio_config)
        out = model(dummy_waveform, task="asr")
        assert out.shape == (2, 3, 64)  # (B, S, vocab_size)
        assert model.output_lengths(torch.tensor([1600])).item() == out.shape[1]

    def test_ctc_asr_logits_spectrogram(self, dummy_waveform):
        model = FrankensteinAudio(_tiny_audio_config(audio_feature="spectrogram"))
        out = model(dummy_waveform, task="asr")
        assert out.shape == (2, 3, 64)

    def test_ctc_asr_logits_raw_multichannel(self):
        model = FrankensteinAudio(
            _tiny_audio_config(audio_feature="raw", audio_in_channels=4)
        )
        out = model(torch.randn(2, 512, 4), task="asr")
        assert out.shape == (2, 128, 64)

    def test_mlm_returns_vocab_logits(self, tiny_audio_config, dummy_waveform):
        model = FrankensteinAudio(tiny_audio_config)
        out = model(dummy_waveform, task="mlm")
        assert out.shape == (2, 3, 64)

    def test_raw_eeg_classification_and_gradients(self):
        """Multichannel raw signals (EEG-like, C=4) classify and backprop."""
        model = FrankensteinAudio(
            _tiny_audio_config(audio_feature="raw", audio_in_channels=4)
        )
        eeg = torch.randn(2, 512, 4)
        out = model(eeg, task="audio_classification")
        assert out.shape == (2, 3)
        out.sum().backward()
        grad = model.frontend.proj.weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()

    def test_transducer_forward_shapes(self, dummy_waveform):
        cfg = _tiny_audio_config(
            asr_loss="transducer",
            asr_decoder_dim=16,
            asr_decoder_layers=1,
            audio_n_mels=32,
        )
        model = FrankensteinAudio(cfg)
        assert isinstance(model.transducer, TransducerHead)
        targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
        out = model(dummy_waveform, task="asr", target_ids=targets)
        assert set(out) == {"encoder", "decoder", "joint"}
        assert out["encoder"].shape == (2, 3, 64)
        # Prepended blank ⇒ U + 1 prediction states.
        assert out["decoder"].shape == (2, 4, 16)
        assert out["joint"].shape == (2, 3, 4, 64)

    def test_transducer_requires_target_ids(self, dummy_waveform):
        model = FrankensteinAudio(
            _tiny_audio_config(asr_loss="transducer", asr_decoder_dim=16)
        )
        with pytest.raises(ValueError):
            model(dummy_waveform, task="asr")

    def test_pooling_gap_vs_cls(self, dummy_waveform):
        gap_cfg = _tiny_audio_config(pooling_mode="gap")
        gap_model = FrankensteinAudio(gap_cfg)
        assert not hasattr(gap_model, "cls_token")
        assert gap_model(dummy_waveform, task="audio_classification").shape == (2, 3)

        cls_cfg = _tiny_audio_config(pooling_mode="cls")
        cls_model = FrankensteinAudio(cls_cfg)
        assert cls_model.use_cls_token
        assert cls_model.cls_token.shape == (1, 1, 64)
        out = cls_model(dummy_waveform, task="audio_classification")
        assert out.shape == (2, 3)
        # Zero-initialized classification head ⇒ identical (zero) logits.
        assert torch.count_nonzero(out) == 0

    def test_cls_token_not_added_to_asr_frames(self, dummy_waveform):
        model = FrankensteinAudio(_tiny_audio_config(pooling_mode="cls"))
        out = model(dummy_waveform, task="asr")
        assert out.shape[1] == model.output_lengths(torch.tensor([1600])).item()

    def test_conformer_and_standard_mixers(self, dummy_waveform):
        cfg = _tiny_audio_config(
            layer_pattern=["conformer_attn", "standard_attn"],
            positional_encoding="sinusoidal_absolute",
        )
        model = FrankensteinAudio(cfg)
        assert model.layers[0].mixer.__class__.__name__ == "ConformerAttention"
        out = model(dummy_waveform, task="audio_classification")
        assert out.shape == (2, 3)

    def test_mhc_smoke(self, dummy_waveform):
        model = FrankensteinAudio(
            _tiny_audio_config(use_mhc=True, mhc_expansion_rate=2)
        )
        out = model(dummy_waveform, task="audio_classification")
        assert out.shape == (2, 3)
        out.sum().backward()

    def test_hyperloop_smoke(self, dummy_waveform):
        model = FrankensteinAudio(
            _tiny_audio_config(
                use_mhc=True,
                mhc_hyperloop=True,
                mhc_expansion_rate=2,
                num_loops=2,
            )
        )
        out = model(dummy_waveform, task="audio_classification")
        assert out.shape == (2, 3)

    def test_attnres_smoke(self, dummy_waveform):
        model = FrankensteinAudio(
            _tiny_audio_config(residual_type="full_attn", num_loops=2)
        )
        out = model(dummy_waveform, task="audio_classification")
        assert out.shape == (2, 3)
        out.sum().backward()

    def test_configure_optimizers(self, tiny_audio_config):
        model = FrankensteinAudio(tiny_audio_config)
        groups = model.configure_optimizers()
        names = [g["name"] for g in groups]
        for expected in ("embeddings", "attention", "head", "norms", "other"):
            assert expected in names
        by_name = {
            g["name"]: [p.shape for p in g["params"]] for g in groups
        }
        # Frontend parameters land in ``embeddings``; the classification head
        # in ``head``.
        assert by_name["embeddings"]
        assert any(len(shape) == 2 and shape[0] == 3 for shape in by_name["head"])

    def test_encode_returns_hidden_states(self, tiny_audio_config, dummy_waveform):
        model = FrankensteinAudio(tiny_audio_config)
        hidden = model.encode(dummy_waveform)
        assert hidden.shape == (2, 3, 64)

    def test_sinusoidal_pe_handles_new_lengths(self, dummy_waveform):
        model = FrankensteinAudio(_tiny_audio_config(sinusoidal_max_len=4))
        short = model(torch.randn(1, 1600), task="asr")  # 3 frames
        long = model(torch.randn(1, 8000), task="asr")  # 13 frames
        assert short.shape == (1, 3, 64)
        assert long.shape == (1, 13, 64)

    def test_dropout_zero_is_deterministic(self, dummy_waveform):
        model = FrankensteinAudio(_tiny_audio_config(dropout=0.0))
        model.eval()
        first = model.encode(dummy_waveform)
        second = model.encode(dummy_waveform)
        assert torch.allclose(first, second)

    def test_unknown_task_raises(self, tiny_audio_config, dummy_waveform):
        model = FrankensteinAudio(tiny_audio_config)
        with pytest.raises(ValueError):
            model(dummy_waveform, task="not_a_task")

    def test_active_task_default_is_audio_classification(
        self, tiny_audio_config, dummy_waveform
    ):
        model = FrankensteinAudio(tiny_audio_config)
        assert model._active_task == "audio_classification"
        assert model(dummy_waveform).shape == (2, 3)

    def test_build_audio_config_defaults(self):
        cfg = FrankensteinAudio.build_audio_config()
        assert cfg.hidden_size == 256
        assert cfg.num_layers == 17
        assert cfg.layer_pattern == ["conformer_attn"] * 17
        assert cfg.mode == "encoder"
        assert cfg.audio_feature == "logmel"
        assert cfg.ffn_activation == "silu"
        assert cfg.ffn_hidden_size == 256 * 4

    def test_config_validation(self):
        with pytest.raises(ValueError):
            _tiny_audio_config(audio_feature="mel")
        with pytest.raises(ValueError):
            _tiny_audio_config(asr_loss="attention")
        with pytest.raises(ValueError):
            _tiny_audio_config(audio_conv_channels=0)
        with pytest.raises(ValueError):
            _tiny_audio_config(audio_n_fft=100)  # < 25 ms @ 16 kHz = 400 samples
        with pytest.raises(ValueError):
            _tiny_audio_config(asr_decoder_layers=0)
        with pytest.raises(ValueError):
            _tiny_audio_config(asr_decoder_dropout=1.0)
