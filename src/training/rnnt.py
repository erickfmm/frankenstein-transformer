#!/usr/bin/env python3
"""RNN-Transducer loss, greedy decoders and error-rate helpers.

This module implements the training-side pieces of the transducer objective
introduced by Graves (2012), "Sequence Transduction with Recurrent Neural
Networks" (arXiv:1211.3711), plus the greedy CTC/RNN-T decoders and a
character error rate (CER) used as the ASR accuracy proxy.

The RNN-Transducer loss is implemented in pure PyTorch (log-domain dynamic
programming with :func:`torch.logaddexp`) so CPU CI never depends on
``torchaudio`` (see D7 in ``AGENTS.md``). When ``torchaudio`` is importable,
:func:`rnnt_loss` can dispatch to ``torchaudio.functional.rnnt_loss`` as a
fast path (``backend="auto"``), but the pure-torch reference stays the
default and is the fallback when the fast path is unavailable or its
signature differs.

References:
    Graves (2012), "Sequence Transduction with Recurrent Neural Networks",
    arXiv:1211.3711, Sec. 4.1-4.2 (loss and label sequence search).
    Graves et al. (2006), "Connectionist Temporal Classification", ICML
    (greedy CTC collapse).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

# torchaudio is an optional extra (``audio``). Any failure at import time
# (missing package, torch/torchaudio ABI mismatch) means "use pure torch".
try:  # pragma: no cover - exercised only when the extra is installed
    from torchaudio.functional import rnnt_loss as _ta_rnnt_loss

    _HAS_TORCHAUDIO_RNNT = True
except Exception:  # noqa: BLE001 - optional dependency
    _ta_rnnt_loss = None  # type: ignore[assignment]
    _HAS_TORCHAUDIO_RNNT = False

__all__ = [
    "HAS_TORCHAUDIO_RNNT",
    "rnnt_loss",
    "greedy_rnnt_decode",
    "ctc_greedy_decode",
    "character_error_rate",
]

#: Whether the optional ``torchaudio`` RNN-T fast path is importable.
HAS_TORCHAUDIO_RNNT = _HAS_TORCHAUDIO_RNNT

SequenceLike = Union[Sequence[int], str]


def _log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Log-softmax over the vocabulary axis (last dimension)."""
    return F.log_softmax(logits.float(), dim=-1)


def _pure_torch_rnnt(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int,
    clamp: float,
) -> torch.Tensor:
    """Reference RNN-Transducer loss (Graves 2012, log-domain DP).

    The dynamic program follows the original formulation: ``alpha[t, u]`` is
    the log-probability of emitting the first ``u`` labels (of the extended
    sequence ``[blank, y_1, ..., y_U]``) during the first ``t`` frames::

        alpha[0, 0] = 0
        alpha[t, u] = logaddexp(alpha[t-1, u] + logp_blank(t-1, u),
                                alpha[t, u-1] + logp_y(t-1, u-1))

    and the sequence probability closes over the two ways of emitting the
    last frame::

        p(y | x) = logaddexp(alpha[T-1, U] + logp_blank(T-1, U),
                             alpha[T-1, U-1] + logp_y(T-1, U-1))

    Args:
        logits: Raw joint-network logits of shape ``(B, T, U+1, V)``.
        targets: Target token ids of shape ``(B, U)`` (right padded).
        input_lengths: Encoder frame counts of shape ``(B,)``.
        target_lengths: Target lengths of shape ``(B,)``.
        blank: Blank token id (``0`` by convention).
        clamp: Minimum value for the log-probabilities (numerical floor).

    Returns:
        A scalar tensor with the mean loss over the batch.
    """
    log_probs = _log_softmax(logits)
    if clamp is not None:
        log_probs = log_probs.clamp(min=float(clamp))

    batch_size = log_probs.size(0)
    max_u = log_probs.size(2) - 1
    blank = int(blank)

    per_sample: List[torch.Tensor] = []
    for b in range(batch_size):
        num_frames = int(input_lengths[b].item())
        num_labels = int(target_lengths[b].item())
        num_labels = max(0, min(num_labels, max_u))
        device = log_probs.device

        if num_frames <= 0:
            # An empty encoder sequence can only transduce an empty target.
            per_sample.append(
                log_probs.new_zeros(())
                if num_labels == 0
                else log_probs.new_full((), float("inf"))
            )
            continue

        label_ids = targets[b, :num_labels].to(device=device, dtype=torch.long)
        # alpha[t, u] with t in [0, num_frames) covering the first t frames.
        alpha = log_probs.new_full((num_frames, num_labels + 1), float("-inf"))
        alpha[0, 0] = torch.zeros((), dtype=alpha.dtype, device=device)
        # Row t = 0: no frame has been processed yet, so only u = 0 is reachable.

        for t in range(1, num_frames):
            alpha[t, 0] = alpha[t - 1, 0] + log_probs[b, t - 1, 0, blank]
            # At most ``t`` labels can have been emitted after ``t`` frames.
            for u in range(1, min(num_labels, t) + 1):
                stay_blank = alpha[t - 1, u] + log_probs[b, t - 1, u, blank]
                emit_label = alpha[t - 1, u - 1] + log_probs[
                    b, t - 1, u - 1, int(label_ids[u - 1].item())
                ]
                alpha[t, u] = torch.logaddexp(stay_blank, emit_label)

        last = num_frames - 1
        blank_term = alpha[last, num_labels] + log_probs[b, last, num_labels, blank]
        if num_labels > 0:
            label_term = alpha[last, num_labels - 1] + log_probs[
                b, last, num_labels - 1, int(label_ids[num_labels - 1].item())
            ]
            log_prob = torch.logaddexp(blank_term, label_term)
        else:
            log_prob = blank_term

        per_sample.append(-log_prob)

    losses = torch.stack(per_sample)
    return losses.mean()


def rnnt_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    clamp: float = -1e4,
    backend: str = "auto",
    zero_infinity: bool = True,
) -> torch.Tensor:
    """Compute the RNN-Transducer loss (Graves 2012, arXiv:1211.3711).

    Args:
        logits: Raw joint-network logits of shape ``(B, T, U+1, V)``.
        targets: Target token ids of shape ``(B, U)`` (right padded with any
            value; only the first ``target_lengths[b]`` ids are read).
        input_lengths: Encoder frame counts of shape ``(B,)``.
        target_lengths: Target lengths of shape ``(B,)``.
        blank: Blank token id (default ``0``).
        clamp: Minimum log-probability applied before the DP. The default
            ``-1e4`` keeps unreachable paths finite without changing the
            loss for real alignments.
        backend: ``"auto"`` (torchaudio fast path when importable, pure-torch
            otherwise), ``"torch"`` (always the pure-torch reference) or
            ``"torchaudio"`` (require the fast path).
        zero_infinity: When true, non-finite per-sample losses (e.g. targets
            longer than the encoder output) are replaced by zero, mirroring
            ``torch.nn.CTCLoss(zero_infinity=True)``.

    Returns:
        A scalar tensor with the mean loss over the batch.

    Raises:
        ValueError: If ``backend`` is unknown, or ``backend="torchaudio"``
            while ``torchaudio`` is unavailable.
    """
    backend = str(backend or "auto").strip().lower()
    if backend not in {"auto", "torch", "torchaudio"}:
        raise ValueError(
            "backend must be one of {'auto', 'torch', 'torchaudio'}, "
            f"got {backend!r}"
        )

    if backend in {"auto", "torchaudio"} and _HAS_TORCHAUDIO_RNNT:
        try:
            try:
                fast_loss = _ta_rnnt_loss(  # type: ignore[misc]
                    logits,
                    targets,
                    input_lengths,
                    target_lengths,
                    blank=int(blank),
                    clamp=float(clamp),
                    reduction="mean",
                    fused_log_softmax=True,
                )
            except TypeError:
                # torchaudio < 2.1 has no ``fused_log_softmax`` keyword.
                fast_loss = _ta_rnnt_loss(  # type: ignore[misc]
                    logits,
                    targets,
                    input_lengths,
                    target_lengths,
                    blank=int(blank),
                    clamp=float(clamp),
                    reduction="mean",
                )
            if zero_infinity and not torch.isfinite(fast_loss):
                fast_loss = torch.nan_to_num(
                    fast_loss, nan=0.0, posinf=0.0, neginf=0.0
                )
            return fast_loss
        except (TypeError, RuntimeError):
            if backend == "torchaudio":
                raise
            # Fall through to the pure-torch reference.
    elif backend == "torchaudio":
        raise ValueError(
            "backend='torchaudio' requires the optional `audio` extra "
            "(pip install -e '.[audio]')"
        )

    loss = _pure_torch_rnnt(
        logits,
        targets,
        input_lengths,
        target_lengths,
        blank=int(blank),
        clamp=float(clamp),
    )
    if zero_infinity and not torch.isfinite(loss):
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    return loss


def ctc_greedy_decode(
    log_probs_or_logits: torch.Tensor,
    input_lengths: Optional[torch.Tensor] = None,
    blank: int = 0,
) -> List[List[int]]:
    """Greedy CTC decoding: per-frame argmax, collapse repeats, drop blanks.

    Args:
        log_probs_or_logits: Frame logits/log-probabilities of shape
            ``(B, T, V)`` (a 2-D ``(T, V)`` tensor is treated as one sample).
        input_lengths: Optional per-sample frame counts; frames beyond each
            length are ignored.
        blank: Blank token id (default ``0``).

    Returns:
        One list of token ids per sample.
    """
    if log_probs_or_logits.dim() == 2:
        log_probs_or_logits = log_probs_or_logits.unsqueeze(0)
    if log_probs_or_logits.dim() != 3:
        raise ValueError(
            "ctc_greedy_decode expects (B, T, V) or (T, V) logits, got "
            f"{tuple(log_probs_or_logits.shape)}"
        )

    with torch.no_grad():
        best = log_probs_or_logits.argmax(dim=-1)  # (B, T)
        batch_size, num_frames = best.shape
        lengths = (
            [num_frames] * batch_size
            if input_lengths is None
            else [int(v) for v in input_lengths.reshape(-1).tolist()]
        )

        decoded: List[List[int]] = []
        for b in range(batch_size):
            frames = best[b, : max(0, min(int(lengths[b]), num_frames))].tolist()
            collapsed: List[int] = []
            previous: Optional[int] = None
            for token in frames:
                token = int(token)
                if token != previous and token != int(blank):
                    collapsed.append(token)
                previous = token
            decoded.append(collapsed)
    return decoded


def greedy_rnnt_decode(
    encoder_out: torch.Tensor,
    transducer: Any,
    max_symbols_per_step: int = 5,
    blank: int = 0,
) -> List[List[int]]:
    """Greedy RNN-Transducer label-synchronous search (Graves 2012, §4.2).

    At every encoder frame the joint network scores the current prediction
    state; the best non-blank symbol is emitted (feeding the prediction
    network) until a blank is produced or ``max_symbols_per_step`` symbols
    have been emitted. Emitting a non-blank does not advance the frame.

    Args:
        encoder_out: Encoder frames of shape ``(B, S, hidden_size)``.
        transducer: A
            :class:`~src.model.frankenstein_audio.TransducerHead`-compatible
            module exposing ``embed``, ``rnn`` and ``joint_logits``.
        max_symbols_per_step: Maximum number of symbols emitted per frame.
        blank: Blank token id (default ``0``).

    Returns:
        One list of token ids per sample.
    """
    if encoder_out.dim() != 3:
        raise ValueError(
            "greedy_rnnt_decode expects encoder_out of shape (B, S, D), got "
            f"{tuple(encoder_out.shape)}"
        )

    batch_size, num_frames, _ = encoder_out.shape
    device = encoder_out.device
    max_symbols = max(1, int(max_symbols_per_step))
    results: List[List[int]] = [[] for _ in range(batch_size)]

    with torch.no_grad():
        for b in range(batch_size):
            encoder_b = encoder_out[b : b + 1]  # (1, S, D)
            # Prediction network state starts from the blank symbol.
            decoder_in = torch.full(
                (1, 1), int(blank), dtype=torch.long, device=device
            )
            decoder_out, state = transducer.rnn(transducer.embed(decoder_in))
            for t in range(num_frames):
                emitted = 0
                while emitted < max_symbols:
                    joint = transducer.joint_logits(
                        encoder_b[:, t : t + 1], decoder_out
                    )  # (1, 1, 1, V)
                    token = int(joint.argmax(dim=-1).item())
                    if token == int(blank):
                        break
                    results[b].append(token)
                    emitted += 1
                    decoder_in = torch.full(
                        (1, 1), token, dtype=torch.long, device=device
                    )
                    decoder_out, state = transducer.rnn(
                        transducer.embed(decoder_in), state
                    )
    return results


def _to_token_ids(sequence: SequenceLike) -> List[int]:
    """Normalise a token sequence (strings become per-character ordinals)."""
    if isinstance(sequence, str):
        return [ord(char) for char in sequence]
    return [int(token) for token in sequence]


def _levenshtein(reference: SequenceLike, hypothesis: SequenceLike) -> int:
    """Edit distance between two token sequences (insert/delete/substitute)."""
    ref = _to_token_ids(reference)
    hyp = _to_token_ids(hypothesis)
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)

    previous = list(range(len(hyp) + 1))
    for i, ref_token in enumerate(ref, start=1):
        current = [i]
        for j, hyp_token in enumerate(hyp, start=1):
            cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + cost,  # substitution
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(
    predictions: Sequence[SequenceLike],
    targets: Sequence[SequenceLike],
) -> float:
    """Character error rate (Levenshtein / reference length).

    Accepts either a single ``(prediction, target)`` pair or two equal-length
    batches of token sequences. Empty targets contribute ``0.0`` when the
    prediction is also empty (and the prediction length otherwise), which
    keeps the mean well defined.

    Args:
        predictions: Predicted token ids (or raw strings), one sequence per
            sample.
        targets: Reference token ids (or raw strings), one sequence per
            sample.

    Returns:
        The (batch-mean) character error rate.

    Raises:
        ValueError: If the two batches have different sizes.
    """
    def _is_single(sequence: Any) -> bool:
        if isinstance(sequence, str):
            return True
        if (
            isinstance(sequence, (list, tuple))
            and sequence
            and isinstance(sequence[0], (list, tuple, str))
        ):
            return False
        return True

    single_pair = _is_single(predictions) and _is_single(targets)
    if not single_pair and len(predictions) != len(targets):
        raise ValueError(
            "character_error_rate expects the same number of predictions and "
            f"targets, got {len(predictions)} and {len(targets)}"
        )

    pairs = [(predictions, targets)] if single_pair else list(zip(predictions, targets))

    if not pairs:
        return 0.0

    total = 0.0
    for prediction, target in pairs:
        ref_len = len(_to_token_ids(target)) if not isinstance(target, str) else len(target)
        distance = _levenshtein(target, prediction)
        if ref_len == 0:
            total += 0.0 if distance == 0 else float(distance)
        else:
            total += distance / float(ref_len)
    return total / float(len(pairs))
