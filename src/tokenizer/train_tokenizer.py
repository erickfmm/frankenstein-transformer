"""Train a HuggingFace tokenizer from a text iterator (``tokenizers`` library).

Implements the ``tokenizer.source: train_from_dataset`` mode: a NEW tokenizer
is trained from the dataset text using the HuggingFace ``tokenizers`` library
trainers (BPE, WordPiece, WordLevel, Unigram) and wrapped in a
``PreTrainedTokenizerFast`` so it integrates transparently with the rest of
the engine (``save_pretrained`` / ``AutoTokenizer.from_pretrained`` checkpoint
round-trips, ``len(tokenizer)`` vocab injection into ``model.dims.vocab_size``).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from src.utils.hf_compat import TOKENIZERS_AVAILABLE

if TOKENIZERS_AVAILABLE:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

DEFAULT_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

_ALGORITHMS = ("bpe", "wordpiece", "wordlevel", "unigram")


def _build_normalizer(lowercase: bool, strip_accents: bool) -> Any:
    """Compose a ``tokenizers`` normalizer from the config flags."""
    ops: List[Any] = []
    if lowercase:
        ops.append(normalizers.Lowercase())
    if strip_accents:
        # StripAccents operates on decomposed (NFD) unicode; NFD first.
        ops.append(normalizers.NFD())
        ops.append(normalizers.StripAccents())
    if not ops:
        return None
    return normalizers.Sequence(ops)


def _build_pre_tokenizer(name: str) -> Any:
    """Build the ``tokenizers`` pre-tokenizer from the config enum value."""
    name = str(name or "whitespace").strip().lower()
    if name == "whitespace":
        return pre_tokenizers.Whitespace()
    if name == "bytelevel":
        return pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=True)
    if name == "bert":
        return pre_tokenizers.BertPreTokenizer()
    raise ValueError(
        f"Unknown tokenizer.training.pre_tokenizer {name!r} "
        "(expected whitespace, bytelevel or bert)"
    )


def _build_model(algorithm: str, vocab_size: int, special_tokens: List[str]):
    """Instantiate the ``tokenizers`` model with special tokens pre-registered."""
    if algorithm == "bpe":
        return models.BPE(unk_token="[UNK]")
    if algorithm == "wordpiece":
        return models.WordPiece(unk_token="[UNK]")
    if algorithm == "wordlevel":
        return models.WordLevel(unk_token="[UNK]")
    if algorithm == "unigram":
        # Unigram accepts a seed vocabulary; BPE-style empty seed fails, so it
        # is initialized empty and the UnigramTrainer fills it in.
        return models.Unigram()
    raise ValueError(
        f"Unknown tokenizer.training.algorithm {algorithm!r} "
        "(expected bpe, wordpiece, wordlevel or unigram)"
    )


def _build_trainer(
    algorithm: str,
    vocab_size: int,
    special_tokens: List[str],
    min_frequency: int,
    max_token_length: int,
    pre_tokenizer_name: str = "whitespace",
):
    """Instantiate the matching ``tokenizers`` trainer."""
    if algorithm == "bpe":
        kwargs: Dict[str, Any] = {
            "vocab_size": vocab_size,
            "min_frequency": min_frequency,
            "special_tokens": special_tokens,
        }
        if pre_tokenizer_name == "bytelevel":
            # Guarantee the full 256-byte alphabet so any text round-trips
            # losslessly even when the corpus is ASCII-only.
            kwargs["initial_alphabet"] = pre_tokenizers.ByteLevel.alphabet()
        if max_token_length:
            kwargs["limit_alphabet"] = vocab_size
        return trainers.BpeTrainer(**kwargs)
    if algorithm == "wordpiece":
        return trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )
    if algorithm == "wordlevel":
        return trainers.WordLevelTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )
    if algorithm == "unigram":
        kwargs = {
            "special_tokens": special_tokens,
            "min_frequency": min_frequency,
        }
        if vocab_size:
            kwargs["vocab_size"] = vocab_size
        return trainers.UnigramTrainer(**kwargs)
    raise ValueError(f"Unknown tokenizer.training.algorithm {algorithm!r}")


def train_tokenizer_from_iterator(
    text_iterator: Iterable[str],
    *,
    algorithm: str = "bpe",
    vocab_size: int = 0,
    special_tokens: Optional[List[str]] = None,
    min_frequency: int = 2,
    lowercase: bool = False,
    strip_accents: bool = False,
    pre_tokenizer: str = "whitespace",
    max_token_length: int = 0,
    save_dir: Optional[str] = None,
) -> "PreTrainedTokenizerFast":
    """Train a HF tokenizer from a text iterator using the ``tokenizers`` library.

    Args:
        text_iterator: Iterable yielding raw text strings (the dataset text).
        algorithm: One of ``bpe``, ``wordpiece``, ``wordlevel``, ``unigram``.
        vocab_size: Target vocabulary size. Required (positive) for
            ``bpe``/``wordpiece``/``wordlevel``; optional (0 = auto-prune) for
            ``unigram``.
        special_tokens: Special tokens to register. Defaults to the standard
            BERT set (``[PAD]``/``[UNK]``/``[CLS]``/``[SEP]``/``[MASK]``).
        min_frequency: Minimum token frequency to include in the vocabulary.
        lowercase: Lowercase all text before training.
        strip_accents: Strip diacritical accents before training.
        pre_tokenizer: One of ``whitespace``, ``bytelevel``, ``bert``.
        max_token_length: Optional per-token character cap (0 = no limit).
        save_dir: Optional directory for ``save_pretrained()`` persistence.

    Returns:
        A ``PreTrainedTokenizerFast`` whose ``len()`` is the final vocabulary
        size (including special tokens). Callers inject ``len(tokenizer)``
        into ``model.dims.vocab_size`` so the embedding matrix matches.

    Raises:
        ImportError: If the ``tokenizers`` library is not installed.
        ValueError: If the arguments are inconsistent (unknown algorithm,
            missing vocab_size for algorithms that require it, empty corpus).
    """
    if not TOKENIZERS_AVAILABLE:
        raise ImportError(
            "The 'tokenizers' library is required for "
            "tokenizer.source=train_from_dataset. Install it with "
            "`pip install tokenizers` (it normally ships with transformers)."
        )

    algorithm = str(algorithm or "bpe").strip().lower()
    if algorithm not in _ALGORITHMS:
        raise ValueError(
            f"Unknown tokenizer algorithm {algorithm!r} "
            "(expected bpe, wordpiece, wordlevel or unigram)"
        )
    if algorithm in ("bpe", "wordpiece", "wordlevel"):
        if not isinstance(vocab_size, int) or vocab_size < 1:
            raise ValueError(
                f"tokenizer.training.vocab_size must be a positive integer for "
                f"algorithm={algorithm!r}"
            )

    tokens = list(special_tokens) if special_tokens else list(DEFAULT_SPECIAL_TOKENS)

    tok = Tokenizer(_build_model(algorithm, vocab_size, tokens))
    normalizer = _build_normalizer(lowercase, strip_accents)
    if normalizer is not None:
        tok.normalizer = normalizer
    tok.pre_tokenizer = _build_pre_tokenizer(pre_tokenizer)

    trainer = _build_trainer(
        algorithm, vocab_size, tokens, int(min_frequency or 2), int(max_token_length or 0),
        pre_tokenizer_name=str(pre_tokenizer or "whitespace").strip().lower(),
    )

    # Materialize the iterator: the tokenizers trainer requires a concrete
    # sequence of texts (or a batch iterator). Feeding list-of-strings keeps
    # this simple and lets us fail fast on an empty corpus.
    texts = [str(text) for text in text_iterator if text]
    if not texts:
        raise ValueError(
            "Cannot train a tokenizer from an empty dataset: the text iterator "
            "yielded no documents."
        )

    logger.info(
        "Training tokenizer: algorithm=%s vocab_size=%s texts=%d",
        algorithm, vocab_size or "auto", len(texts),
    )
    tok.train_from_iterator(texts, trainer=trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    logger.info("Tokenizer trained: vocab_size=%d", len(fast))

    if save_dir:
        fast.save_pretrained(save_dir)
        logger.info("Tokenizer saved to %s", save_dir)

    return fast