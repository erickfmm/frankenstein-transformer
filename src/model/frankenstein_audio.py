#!/usr/bin/env python3
"""Frankenstein Audio: Conformer-ready encoder for speech and 1-D signals.

This module hosts :class:`FrankensteinAudio`, the audio/signal model class.
It mirrors :class:`~src.model.frankenstein_vit.FrankensteinViT`: it replaces
the NLP token embedding with a waveform frontend (log-mel filterbanks,
log-power spectrogram or a raw multichannel 1-D signal) followed by the
paper's convolutional subsampling stack (arXiv:2005.08100, Sec. 2), stacks
:class:`HybridLayer` blocks (so every mixer — notably ``conformer_attn`` —
plus MoE, MoD, mHC, AttnRes, Hyperloop, BitNet and every norm/optimizer
keeps working), and exposes task heads:

* ``asr`` — CTC projection over the encoder frames (``asr_loss: ctc``,
  Graves et al. 2006) or an RNN-Transducer (``asr_loss: transducer``,
  arXiv:1211.3711) with a 1-layer LSTM prediction network and a joint
  network.
* ``audio_classification`` — linear head over a pooled representation
  (global average pooling or a learned ``[CLS]`` token).
* ``mlm`` — vocabulary logits over the frames, keeping the base-model path
  generic for masked-frame pretraining.

``audio.feature: raw`` bypasses the mel/STFT frontend and feeds generic
multichannel 1-D signals (EEG, EMG, accelerometer, ...) through the same
convolutional subsampling, so the class doubles as a biomedical signal
classifier.

The frontend prefers ``torchaudio.transforms.MelSpectrogram`` when
``torchaudio`` is importable and otherwise falls back to a pure-torch
``torch.stft`` + HTK mel filterbank, keeping CPU CI torchaudio-free (D7).

References:
    Gulati et al. (2020), "Conformer: Convolution-augmented Transformer for
    Speech Recognition", arXiv:2005.08100.
    Graves et al. (2006), "Connectionist Temporal Classification: Labelling
    Unsegmented Sequence Data with Recurrent Neural Networks", ICML.
    Graves (2012), "Sequence Transduction with Recurrent Neural Networks",
    arXiv:1211.3711.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# torchaudio is an optional extra (``audio``). Beyond a plain ImportError, a
# torchaudio/torch version or ABI mismatch surfaces as OSError/RuntimeError at
# import time; any of those means "fall back to the pure-torch frontend".
try:  # pragma: no cover - exercised only when the extra is installed
    import torchaudio as _torchaudio

    _HAS_TORCHAUDIO = True
except Exception:  # noqa: BLE001 - optional dependency, any failure falls back
    _torchaudio = None
    _HAS_TORCHAUDIO = False

from .config import FrankensteinModelConfig
from .hybrid_layer import HybridLayer
from .hyperloop import HyperloopConnections
from .attention.common import BitLinear
from .optimizer.base import categorize_param_groups
from .embeddings import build_pos_encoder
from .norm import get_norm
from .residuals import ResidualBase, build_residual

__all__ = [
    "AudioFrontend",
    "TransducerHead",
    "FrankensteinAudio",
    "HAS_TORCHAUDIO",
]

#: Whether ``torchaudio`` is importable (the frontend fast path).
HAS_TORCHAUDIO = _HAS_TORCHAUDIO


def _hz_to_mel(freq: float) -> float:
    """Convert Hertz to the HTK mel scale.

    Args:
        freq: Frequency in Hertz.

    Returns:
        The frequency in mels (``2595 * log10(1 + f / 700)``).
    """
    return 2595.0 * math.log10(1.0 + float(freq) / 700.0)


def _mel_to_hz(mel: float) -> float:
    """Convert HTK mels back to Hertz.

    Args:
        mel: Frequency in mels.

    Returns:
        The frequency in Hertz.
    """
    return 700.0 * (10.0 ** (float(mel) / 2595.0) - 1.0)


def build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> torch.Tensor:
    """Build an HTK triangular mel filterbank.

    Reproduces the classic ``torchaudio`` filterbank construction
    (``mel_scale="htz"``): ``n_mels + 2`` equally spaced points on the mel
    scale define triangular filters over the ``n_fft // 2 + 1`` one-sided STFT
    bins.

    Args:
        sample_rate: Audio sample rate in Hertz.
        n_fft: FFT size used by the STFT.
        n_mels: Number of mel filters.
        f_min: Lower filterbank edge in Hertz.
        f_max: Upper filterbank edge in Hertz (defaults to
            ``sample_rate / 2``).

    Returns:
        A ``(n_mels, n_fft // 2 + 1)`` float32 tensor.
    """
    n_freqs = n_fft // 2 + 1
    high = float(sample_rate) / 2.0 if f_max is None else float(f_max)
    all_freqs = torch.linspace(0.0, float(sample_rate) / 2.0, n_freqs)
    m_pts = torch.linspace(_hz_to_mel(f_min), _hz_to_mel(high), n_mels + 2)
    f_pts = torch.tensor([_mel_to_hz(float(m)) for m in m_pts], dtype=torch.float32)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down_slopes = -slopes[:, :-2] / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    fb = torch.clamp(torch.minimum(down_slopes, up_slopes), min=0.0)
    return fb.transpose(0, 1).contiguous()  # (n_mels, n_freqs)


class AudioFrontend(nn.Module):
    """Waveform → frame sequence frontend (arXiv:2005.08100, Sec. 2).

    For ``feature: logmel`` (80-channel log-mel filterbanks, 25 ms window,
    10 ms stride — the Conformer recipe) and ``feature: spectrogram``
    (log-power STFT) the frontend computes frame features and applies the
    paper's convolutional subsampling::

        (B, 1, F, T) → Conv2d(k=3, s=2, p=1) → SiLU
                     → Conv2d(k=3, s=2, p=1) → SiLU
                     → flatten(F, C) → Linear(·, hidden) → dropout

    giving a 4× time reduction (10 ms → 40 ms frames).

    For ``feature: raw`` the waveform (already a generic multichannel 1-D
    signal) skips the mel/STFT stage and is subsampled by two stride-2
    :class:`nn.Conv1d` layers::

        (B, C, T) → Conv1d(k=3, s=2, p=1) → SiLU
                  → Conv1d(k=3, s=2, p=1) → SiLU
                  → (B, T', C) → Linear(C, hidden) → dropout

    Args:
        config: Model configuration exposing the ``audio_*`` knobs and
            ``dropout``/``hidden_size``.

    Raises:
        ValueError: If ``audio_feature`` is unknown, or the numeric audio
            fields are not positive.

    Attributes:
        feature: Resolved frontend kind (``logmel``, ``spectrogram``, raw).
        subsampling_factor: Total time-reduction factor (``4``).
        hop_length: STFT hop in samples.
        win_length: STFT window in samples.
        freq_after: Frequency-axis size after the two Conv2d strides.
        window: Hann window buffer (mel/spectrogram paths).
        mel_fb: HTK mel filterbank buffer (mel path).
    """

    def __init__(self, config: FrankensteinModelConfig) -> None:
        super().__init__()
        self.feature = str(getattr(config, "audio_feature", "logmel")).lower()
        if self.feature not in {"logmel", "spectrogram", "raw"}:
            raise ValueError(
                "audio_feature must be one of {'logmel', 'spectrogram', 'raw'}, "
                f"got {self.feature!r}"
            )

        self.sample_rate = int(getattr(config, "audio_sample_rate", 16000))
        self.n_mels = int(getattr(config, "audio_n_mels", 80))
        self.window_ms = float(getattr(config, "audio_window_ms", 25.0))
        self.stride_ms = float(getattr(config, "audio_stride_ms", 10.0))
        self.n_fft = int(getattr(config, "audio_n_fft", 400))
        self.in_channels = int(getattr(config, "audio_in_channels", 1))
        self.conv_channels = int(getattr(config, "audio_conv_channels", 32))
        self.hidden_size = int(config.hidden_size)
        self.dropout_p = float(config.dropout)

        for name, value in (
            ("audio_sample_rate", self.sample_rate),
            ("audio_n_mels", self.n_mels),
            ("audio_n_fft", self.n_fft),
            ("audio_in_channels", self.in_channels),
            ("audio_conv_channels", self.conv_channels),
            ("audio_window_ms", self.window_ms),
            ("audio_stride_ms", self.stride_ms),
        ):
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")

        self.hop_length = max(1, int(round(self.stride_ms * self.sample_rate / 1000.0)))
        self.win_length = max(1, int(round(self.window_ms * self.sample_rate / 1000.0)))
        if self.n_fft < self.win_length:
            raise ValueError(
                f"audio_n_fft ({self.n_fft}) must be >= the window length "
                f"({self.win_length} samples)"
            )

        #: Two stride-2 convolutions ⇒ 4× time reduction.
        self.subsampling_factor = 4
        self.dropout = nn.Dropout(self.dropout_p)
        self._mel_transform = None

        if self.feature == "raw":
            self.subsample = nn.Sequential(
                nn.Conv1d(self.in_channels, self.conv_channels, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv1d(self.conv_channels, self.conv_channels, 3, stride=2, padding=1),
                nn.SiLU(),
            )
            self.proj = nn.Linear(self.conv_channels, self.hidden_size)
            self.freq_after = 0
            return

        # ---- Mel / spectrogram path ----
        self.n_freqs = self.n_fft // 2 + 1
        freq = self.n_mels if self.feature == "logmel" else self.n_freqs
        # Two stride-2 Conv2d with kernel 3 and padding 1: f' = (f + 1) // 2.
        self.freq_after = (freq + 1) // 2
        self.freq_after = (self.freq_after + 1) // 2

        self.subsample = nn.Sequential(
            nn.Conv2d(1, self.conv_channels, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.conv_channels, self.conv_channels, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.proj = nn.Linear(
            self.conv_channels * self.freq_after, self.hidden_size
        )

        # The Hann window and mel filterbank are pure-torch so the fallback
        # path never needs torchaudio. They are non-persistent buffers so old
        # checkpoints still load.
        self.register_buffer(
            "window", torch.hann_window(self.win_length), persistent=False
        )
        self.register_buffer(
            "mel_fb",
            build_mel_filterbank(self.sample_rate, self.n_fft, self.n_mels),
            persistent=False,
        )

        if self.feature == "logmel" and _HAS_TORCHAUDIO:  # pragma: no cover
            self._mel_transform = _torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                power=2.0,
                normalized=False,
            )

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def _to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """Collapse a ``(B, T)`` or ``(B, T, C)`` waveform to mono ``(B, T)``.

        Args:
            waveform: Waveform tensor.

        Returns:
            A ``(B, T)`` mono waveform (channels are averaged when C > 1).

        Raises:
            ValueError: If the tensor is not 2-D or 3-D.
        """
        if waveform.dim() == 2:
            return waveform
        if waveform.dim() == 3:
            if waveform.size(-1) == 1:
                return waveform.squeeze(-1)
            return waveform.mean(dim=-1)
        raise ValueError(
            "Audio waveform must have shape (B, T) or (B, T, C), got "
            f"{tuple(waveform.shape)}"
        )

    def _features(self, mono: torch.Tensor) -> torch.Tensor:
        """Compute log-mel or log-power spectrogram features.

        Args:
            mono: Mono waveform of shape ``(B, T)``.

        Returns:
            Log features of shape ``(B, n_mels | n_freqs, frames)``.
        """
        if self._mel_transform is not None:  # pragma: no cover - torchaudio path
            mel = self._mel_transform(mono)
            return torch.log(mel + 1e-6)

        power = torch.stft(
            mono,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        ).abs().pow(2.0)
        if self.feature == "logmel":
            power = self.mel_fb @ power
        return torch.log(power + 1e-6)

    def _forward_raw(self, waveform: torch.Tensor) -> torch.Tensor:
        """Subsample a raw multichannel 1-D signal.

        Args:
            waveform: ``(B, T)`` (single channel) or ``(B, T, C)`` signal.

        Returns:
            Hidden states of shape ``(B, T', hidden_size)``.

        Raises:
            ValueError: If the channel count is incompatible with
                ``audio_in_channels`` or the tensor rank is unsupported.
        """
        if waveform.dim() == 2:
            if self.in_channels != 1:
                raise ValueError(
                    "audio.feature=raw with audio.in_channels > 1 expects a "
                    f"(B, T, C) signal with C={self.in_channels}, got "
                    f"{tuple(waveform.shape)}"
                )
            signal = waveform.unsqueeze(1)  # (B, 1, T)
        elif waveform.dim() == 3:
            if waveform.size(-1) == self.in_channels:
                signal = waveform.transpose(1, 2)  # (B, C, T)
            elif self.in_channels == 1:
                signal = waveform.mean(dim=-1).unsqueeze(1)
            else:
                raise ValueError(
                    f"audio.in_channels={self.in_channels} does not match the "
                    f"{waveform.size(-1)} channels of the input signal"
                )
        else:
            raise ValueError(
                "Raw audio signals must have shape (B, T) or (B, T, C), got "
                f"{tuple(waveform.shape)}"
            )
        x = self.subsample(signal)  # (B, conv_channels, T')
        x = x.transpose(1, 2)  # (B, T', conv_channels)
        return self.dropout(self.proj(x))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert a waveform or raw signal into a frame sequence.

        Args:
            waveform: ``(B, T)`` mono waveform, ``(B, T, C)`` waveform (mono
                mix) for logmel/spectrogram, or ``(B, T, C)`` multichannel
                signal for ``feature: raw``.

        Returns:
            Frame embeddings of shape ``(B, S, hidden_size)``.
        """
        if self.feature == "raw":
            return self._forward_raw(waveform)

        mono = self._to_mono(waveform)
        feats = self._features(mono)  # (B, F, frames)
        feats = feats.unsqueeze(1)  # (B, 1, F, frames)
        x = self.subsample(feats)  # (B, C, F', S)
        batch, channels, freq, steps = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, steps, channels * freq)
        return self.dropout(self.proj(x))

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Map input sample counts to output frame counts.

        ``center=True`` STFT yields ``1 + T // hop`` frames; each stride-2
        convolution halves the time axis with ``f' = (f + 1) // 2``.

        Args:
            input_lengths: Integer tensor of input lengths (samples for
                logmel/spectrogram, time steps for raw signals).

        Returns:
            Integer tensor of the same shape with the output frame counts.
        """
        lengths = input_lengths.to(dtype=torch.long)
        if self.feature == "raw":
            lengths = (lengths + 1) // 2
            return (lengths + 1) // 2
        frames = 1 + lengths // self.hop_length
        frames = (frames + 1) // 2
        return (frames + 1) // 2


class TransducerHead(nn.Module):
    """RNN-Transducer prediction + joint networks (arXiv:1211.3711).

    The prediction network is a (default 1-layer) LSTM over the embedded
    target labels; the joint network fuses each encoder frame with each
    prediction state via ``Linear(hidden + decoder_dim, hidden) → SiLU →
    Linear(hidden, vocab_size)``.

    Args:
        config: Model configuration with ``vocab_size``, ``hidden_size`` and
            the ``asr_decoder_*`` knobs.

    Attributes:
        vocab_size: Output vocabulary size.
        encoder_dim: Encoder hidden size.
        decoder_dim: Prediction-network hidden size.
        embed: Target-label embedding.
        rnn: LSTM prediction network.
        joint_in: Joint-network input projection.
        joint_out: Joint-network output projection.
    """

    def __init__(self, config: FrankensteinModelConfig) -> None:
        super().__init__()
        self.vocab_size = int(config.vocab_size)
        self.encoder_dim = int(config.hidden_size)
        self.decoder_dim = int(getattr(config, "asr_decoder_dim", 320))
        num_layers = int(getattr(config, "asr_decoder_layers", 1))
        dropout = float(getattr(config, "asr_decoder_dropout", 0.1))
        if num_layers < 1:
            raise ValueError("asr_decoder_layers must be >= 1")
        if self.decoder_dim < 1:
            raise ValueError("asr_decoder_dim must be >= 1")

        self.embed = nn.Embedding(self.vocab_size, self.decoder_dim)
        self.rnn = nn.LSTM(
            self.decoder_dim,
            self.decoder_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        joint_hidden = self.encoder_dim
        self.joint_in = nn.Linear(self.encoder_dim + self.decoder_dim, joint_hidden)
        self.joint_out = nn.Linear(joint_hidden, self.vocab_size)

    def forward(
        self, encoder_out: torch.Tensor, target_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the prediction network with shifted teacher forcing.

        The blank id (``0``) is prepended to the target sequence, so the
        returned prediction states cover ``U + 1`` positions (the initial
        blank state plus one state per target token), matching the joint
        tensor's ``U + 1`` axis.

        Args:
            encoder_out: Encoder frames of shape ``(B, S, hidden_size)``.
            target_ids: Target token ids of shape ``(B, U)``.

        Returns:
            Tuple ``(encoder_out, decoder_out)`` with ``decoder_out`` of shape
            ``(B, U + 1, decoder_dim)``.
        """
        target_ids = target_ids.to(dtype=torch.long)
        blank = torch.zeros(
            (target_ids.size(0), 1), dtype=torch.long, device=target_ids.device
        )
        decoder_in = torch.cat([blank, target_ids], dim=1)  # (B, U + 1)
        decoder_out, _ = self.rnn(self.embed(decoder_in))
        return encoder_out, decoder_out

    def joint_logits(
        self, encoder_out: torch.Tensor, decoder_out: torch.Tensor
    ) -> torch.Tensor:
        """Compute joint logits without materialising the concatenation.

        The joint projection is split into its encoder and prediction halves
        and broadcast over the ``(S, U + 1)`` grid::

            silu(W_e·enc[:, :, None] + W_d·dec[:, None, :] + b)
                → Linear(hidden, vocab_size)

        Args:
            encoder_out: Encoder frames ``(B, S, hidden_size)``.
            decoder_out: Prediction states ``(B, U + 1, decoder_dim)``.

        Returns:
            Joint logits of shape ``(B, S, U + 1, vocab_size)``.
        """
        weight = self.joint_in.weight
        bias = self.joint_in.bias
        enc = F.linear(encoder_out, weight[:, : self.encoder_dim])
        dec = F.linear(decoder_out, weight[:, self.encoder_dim :])
        if bias is not None:
            fused = enc[:, :, None, :] + dec[:, None, :, :] + bias
        else:
            fused = enc[:, :, None, :] + dec[:, None, :, :]
        return self.joint_out(F.silu(fused))


class FrankensteinAudio(nn.Module):
    """Conformer-ready audio encoder (ASR + audio classification).

    Mirrors :class:`~src.model.frankenstein_vit.FrankensteinViT`: a config
    driven frontend (:class:`AudioFrontend`), an optional learned ``[CLS]``
    token, a :class:`HybridLayer` stack with looped depth / mHC / Hyperloop /
    AttnRes / MoD support, a final norm and the task heads. The model forces
    ``config.mode = "encoder"`` (audio encoders are bidirectional).

    The ``[CLS]`` token is only prepended for the pooled classification path,
    so the ASR frame axis stays exactly ``frontend.output_lengths(...)``.

    Attributes:
        config: The :class:`FrankensteinModelConfig` used to build the model.
        frontend: :class:`AudioFrontend` waveform frontend.
        subsampling_factor: Total time reduction applied by the frontend.
        pos_encoder: Shared positional encoding module.
        cls_token: Learnable ``[CLS]`` token when ``pooling_mode == "cls"``.
        layers: ModuleList of :class:`HybridLayer` blocks.
        final_norm: Final normalization before the heads.
        residual: :class:`ResidualBase` module.
        head: CTC / vocabulary projection.
        audio_classification_head: Linear audio classification head.
        transducer: :class:`TransducerHead` when ``asr_loss == "transducer"``.
        last_auxiliary_losses: Dict of auxiliary losses (MoE/MoD).
        last_mixture_of_depths_stats: Dict of MoD statistics.
    """

    def __init__(self, config: Optional[FrankensteinModelConfig] = None) -> None:
        super().__init__()
        self.config = config or self.build_audio_config()
        self.use_fixed_softmax = bool(getattr(self.config, "fixed_softmax", False))
        self.last_auxiliary_losses: Dict[str, Any] = {}
        self.last_mixture_of_depths_stats: Dict[str, Any] = {}

        # Audio encoders are bidirectional — force encoder mode.
        if self.config.mode != "encoder":
            self.config.mode = "encoder"
        # Record the concrete class for deployment dispatch (see
        # ``ModelDeployer._build_model``).
        self.config.model_class = "frankenstein_audio"

        cfg = self.config
        self.frontend = AudioFrontend(cfg)
        self.subsampling_factor = int(self.frontend.subsampling_factor)
        self.dropout = nn.Dropout(cfg.dropout)

        # ---- [CLS] token (only for pooled classification) ----
        self.use_cls_token = str(cfg.pooling_mode) == "cls"
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ---- Positional encoding ----
        self.pos_encoder = build_pos_encoder(cfg)

        # ---- HybridLayer stack (every layer_pattern mixer) ----
        self.layers = nn.ModuleList(
            [
                HybridLayer(
                    cfg,
                    layer_type=cfg.layer_pattern[i % len(cfg.layer_pattern)],
                    pos_encoder=self.pos_encoder,
                )
                for i in range(cfg.num_layers)
            ]
        )

        # ---- Final norm + residual (reused) ----
        self.final_norm = get_norm(cfg)
        self.residual: ResidualBase = build_residual(cfg)

        # ---- mHC / Hyperloop (reused) ----
        self.use_mhc = bool(getattr(cfg, "use_mhc", False))
        self.use_hyperloop = self.use_mhc and bool(getattr(cfg, "mhc_hyperloop", False))
        if self.use_hyperloop:
            self.hyperloop = HyperloopConnections(
                hidden_size=cfg.hidden_size,
                num_loops=int(cfg.num_loops),
                expansion_rate=cfg.mhc_expansion_rate,
                gating_init=cfg.mhc_gating_init,
                res_parameterization=cfg.mhc_hyperloop_res_parameterization,
                sinkhorn_iters=cfg.mhc_sinkhorn_iters,
                use_bitnet=cfg.use_bitnet,
                full_prec_under_bitnet=cfg.mhc_full_prec_under_bitnet,
            )
        elif self.use_mhc:
            n = int(cfg.mhc_expansion_rate)
            self.mhc_in_proj = nn.Linear(cfg.hidden_size, n * cfg.hidden_size)
            self.mhc_out_proj = nn.Linear(n * cfg.hidden_size, cfg.hidden_size)

        # ---- Task heads ----
        self.asr_loss = str(getattr(cfg, "asr_loss", "ctc")).lower()
        self._active_task: str = "audio_classification"
        proj_cls = BitLinear if cfg.use_bitnet else nn.Linear
        # CTC / vocabulary projection over the encoder frames.
        self.head = proj_cls(cfg.hidden_size, cfg.vocab_size)
        # Audio classification head: full precision, zero-initialized.
        self.audio_classification_head = nn.Linear(cfg.hidden_size, cfg.num_classes)
        nn.init.zeros_(self.audio_classification_head.weight)
        if self.audio_classification_head.bias is not None:
            nn.init.zeros_(self.audio_classification_head.bias)
        # RNN-Transducer (prediction + joint networks).
        self.transducer: Optional[TransducerHead] = None
        if self.asr_loss == "transducer":
            self.transducer = TransducerHead(cfg)

    @staticmethod
    def build_audio_config(
        hidden_size: int = 256,
        num_layers: int = 17,
        num_heads: int = 4,
        num_loops: int = 1,
        use_bitnet: bool = False,
        layer_pattern: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> FrankensteinModelConfig:
        """Build the default Conformer-M preset configuration.

        Reference sizes (arXiv:2005.08100, Table 1): S/M/L = 16/17/17 layers,
        ``d`` = 144/256/512, heads 4/4/8, decoder dim 320/640/640.

        Args:
            hidden_size: Hidden state dimensionality. Default: 256 (Conformer-M).
            num_layers: Number of HybridLayer blocks. Default: 17.
            num_heads: Number of attention heads. Default: 4.
            num_loops: Loop iterations. Default: 1.
            use_bitnet: Whether to use BitNet. Default: False.
            layer_pattern: Custom layer pattern. If None, defaults to
                ``["conformer_attn"] * num_layers``.
            **kwargs: Additional FrankensteinModelConfig fields.

        Returns:
            A pre-configured :class:`FrankensteinModelConfig` with audio
            defaults.
        """
        if layer_pattern is None:
            layer_pattern = ["conformer_attn"] * num_layers
        explicit_keys = {
            "hidden_size",
            "num_layers",
            "num_heads",
            "num_loops",
            "use_bitnet",
            "layer_pattern",
            "ffn_activation",
            "ffn_hidden_size",
            "use_moe",
            "norm_type",
            "positional_encoding",
            "mode",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in explicit_keys}
        # The audio frontend / ASR knobs keep their dataclass defaults
        # (`audio_feature="logmel"`, 16 kHz, 80 mels, 25 ms / 10 ms, n_fft=400,
        # 32 conv channels, `asr_loss="ctc"`, decoder dim 320) and can be
        # overridden through ``**kwargs``.
        return FrankensteinModelConfig(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_loops=num_loops,
            use_bitnet=use_bitnet,
            layer_pattern=layer_pattern,
            ffn_activation="silu",
            ffn_hidden_size=hidden_size * 4,  # Conformer FFN: 4×d
            use_moe=False,
            norm_type="layer_norm",
            positional_encoding="sinusoidal_absolute",
            mode="encoder",
            **filtered_kwargs,
        )

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Map input sample counts to encoder frame counts.

        Args:
            input_lengths: Integer tensor of input lengths.

        Returns:
            Integer tensor of encoder frame counts.
        """
        return self.frontend.output_lengths(input_lengths)

    def _run_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Run the HybridLayer stack with looped depth + residual + mHC.

        Mirrors :meth:`FrankensteinViT._run_encoder` so AttnRes, MoD, mHC,
        Hyperloop and looped depth behave identically for audio inputs.

        Args:
            x: Input hidden states of shape ``(B, S, D)``.

        Returns:
            Output hidden states of shape ``(B, S, D)``.
        """
        # ---- Middle-cycle partitioning (Hyperloop, arXiv:2604.21254) ----
        n_begin = (
            int(getattr(self.config, "mhc_hyperloop_begin_layers", 0))
            if self.use_hyperloop
            else 0
        )
        n_end = (
            int(getattr(self.config, "mhc_hyperloop_end_layers", 0))
            if self.use_hyperloop
            else 0
        )
        n_middle = self.config.num_layers - n_begin - n_end

        if self.use_mhc and not self.use_hyperloop:
            n = int(self.config.mhc_expansion_rate)
            bsz, seq_len, dim = x.shape
            x = self.mhc_in_proj(x).view(bsz, seq_len, n, dim)

        logical_depth = int(self.config.num_layers) * int(self.config.num_loops)
        self.residual.register_state(num_layers=logical_depth)
        self.residual.reset_state()
        if self.residual.is_attn_res:
            self.residual.set_embedding(x)

        logical_layer_idx = 0
        mixture_of_depths_aux_losses: List[torch.Tensor] = []
        mixture_of_depths_selected_fractions: List[float] = []

        def _run_layers(x_in, layers, idx0):
            """Run a layer slice, applying the AttnRes residual when active.

            Args:
                x_in: Input stream for the first layer in the slice.
                layers: The :class:`HybridLayer` objects to run in order.
                idx0: The logical layer index of the first layer in the slice.

            Returns:
                Tuple ``(x_out, next_idx)`` with the updated stream and the
                next logical layer index.
            """
            x = x_in
            idx = idx0
            for layer in layers:
                x = layer(x, logical_layer_idx=idx, input_ids=None)
                if self.residual.is_attn_res:
                    x = self.residual(idx, x)
                if (
                    layer.use_mixture_of_depths
                    and layer.last_mixture_of_depths_aux_loss is not None
                ):
                    mixture_of_depths_aux_losses.append(
                        layer.last_mixture_of_depths_aux_loss
                    )
                    mixture_of_depths_selected_fractions.append(
                        layer.last_mixture_of_depths_selected_fraction
                    )
                idx += 1
            return x, idx

        # ---- Hyperloop middle-cycle forward ----
        if self.use_hyperloop:
            hl = self.hyperloop
            x, logical_layer_idx = _run_layers(
                x, self.layers[:n_begin], logical_layer_idx
            )
            x = hl.expand(x)

            middle = self.layers[n_begin : n_begin + n_middle]
            for loop_idx in range(self.config.num_loops):
                h_pre, h_post, h_res = hl.mappings(x, loop_idx)
                block_in = hl.apply_fpre(x, h_pre)
                block_out, logical_layer_idx = _run_layers(
                    block_in, middle, logical_layer_idx
                )
                x = hl.apply_update(x, block_out, h_post, h_res, loop_idx)

            x = hl.collapse(x)
            x, logical_layer_idx = _run_layers(
                x, self.layers[n_begin + n_middle :], logical_layer_idx
            )
        else:
            for _ in range(self.config.num_loops):
                x, logical_layer_idx = _run_layers(x, self.layers, logical_layer_idx)

        x = self.residual.finalize(x)

        if self.use_mhc and not self.use_hyperloop:
            bsz, seq_len, n, dim = x.shape
            x = self.mhc_out_proj(x.reshape(bsz, seq_len, n * dim))

        x = self.final_norm(x)

        if mixture_of_depths_aux_losses:
            raw_aux_loss = torch.stack(mixture_of_depths_aux_losses).mean()
            weighted_aux_loss = raw_aux_loss * float(
                getattr(self.config, "mixture_of_depths_router_aux_loss_weight", 0.0)
            )
            self.last_auxiliary_losses = {
                "mixture_of_depths_router_loss": weighted_aux_loss,
            }
            self.last_mixture_of_depths_stats = {
                "average_selected_fraction": sum(mixture_of_depths_selected_fractions)
                / len(mixture_of_depths_selected_fractions),
                "raw_router_aux_loss": float(raw_aux_loss.detach().item()),
            }
        else:
            self.last_auxiliary_losses = {}
            self.last_mixture_of_depths_stats = {}
        return x

    def _encode_frames(
        self, waveform: torch.Tensor, prepend_cls: bool = False
    ) -> torch.Tensor:
        """Frontend → optional ``[CLS]`` → PE → dropout → layer stack.

        Args:
            waveform: Waveform / raw signal tensor.
            prepend_cls: Whether to prepend the learned ``[CLS]`` token
                (pooled classification only).

        Returns:
            Hidden states of shape ``(B, S (+1), D)``.
        """
        x = self.frontend(waveform)
        if prepend_cls and self.use_cls_token:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
        if self.pos_encoder is not None and hasattr(self.pos_encoder, "add"):
            x = self.pos_encoder.add(x)
        x = self.dropout(x)
        return self._run_encoder(x)

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return the encoder hidden states for a waveform.

        Args:
            waveform: Waveform / raw signal tensor.

        Returns:
            Hidden states of shape ``(B, S, hidden_size)`` (no ``[CLS]`` token,
            so the frame axis matches :meth:`output_lengths`).
        """
        return self._encode_frames(waveform, prepend_cls=False)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pool the encoder output for classification.

        Args:
            x: Hidden states of shape ``(B, S (+1), D)``.

        Returns:
            Pooled representation of shape ``(B, D)``.
        """
        if self.use_cls_token:
            return x[:, 0]
        return x.mean(dim=1)

    def forward(
        self,
        waveform: torch.Tensor,
        task: Optional[str] = None,
        target_ids: Optional[torch.Tensor] = None,
    ):
        """Forward pass through the audio encoder.

        Args:
            waveform: Waveform of shape ``(B, T)``, ``(B, T, C)`` (mel paths
                mix channels to mono) or a raw multichannel signal
                ``(B, T, C)`` for ``audio_feature: raw``.
            task: Task name (``asr``, ``audio_classification`` or ``mlm``).
                If None, uses ``self._active_task``.
            target_ids: Target token ids ``(B, U)`` required by
                ``task="asr"`` with ``asr_loss="transducer"``.

        Returns:
            Task-specific output:

            * ``asr`` with ``asr_loss="ctc"``: logits ``(B, S, vocab_size)``.
            * ``asr`` with ``asr_loss="transducer"``: dict with
              ``encoder`` ``(B, S, H)``, ``decoder`` ``(B, U + 1, Hd)`` and
              ``joint`` ``(B, S, U + 1, vocab_size)``.
            * ``audio_classification``: logits ``(B, num_classes)``.
            * ``mlm``: vocabulary logits ``(B, S, vocab_size)``.

        Raises:
            ValueError: If the task is unknown, or ``target_ids`` is missing
                for the transducer path.
        """
        task = task or self._active_task

        if task == "audio_classification":
            x = self._encode_frames(waveform, prepend_cls=True)
            return self.audio_classification_head(self._pool(x))

        if task in ("asr", "mlm"):
            x = self._encode_frames(waveform, prepend_cls=False)
            if task == "asr" and self.asr_loss == "transducer":
                if self.transducer is None:  # pragma: no cover - built in __init__
                    raise ValueError(
                        "asr_loss='transducer' was requested but the transducer "
                        "head is not built"
                    )
                if target_ids is None:
                    raise ValueError(
                        "task='asr' with asr_loss='transducer' requires "
                        "target_ids of shape (B, U)"
                    )
                enc, dec = self.transducer(x, target_ids)
                return {
                    "encoder": enc,
                    "decoder": dec,
                    "joint": self.transducer.joint_logits(enc, dec),
                }
            return self.head(x)

        raise ValueError(f"Unknown task: {task!r}")

    def configure_optimizers(self) -> List[Dict[str, Any]]:
        """Return named parameter groups for :func:`build_optimizer`.

        Groups follow the canonical block taxonomy shared with the parameter
        estimator and the per-block gradient-norm logs (``embeddings``,
        ``pos``, ``attention``, ``ffn``, ``norms``, ``head``, ``mhc``,
        ``router``, ``residual``, ``other``). Every group is returned even
        when empty, preserving the behavior of the text/ViT models.
        """
        groups = categorize_param_groups(
            self.named_parameters(), drop_empty=False
        )
        return [{"params": params, "name": name} for name, params in groups.items()]
