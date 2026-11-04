"""Tokenizer vocabulary helpers for EELBERT-style dynamic embeddings.

Dynamic embeddings (arXiv:2310.20144) hash the token **string**, not the token
id, so every code path that constructs a model with
``embedding.dynamic.enabled`` must attach an id-ordered vocabulary to the
model config before construction. These helpers extract that vocabulary from
either a HuggingFace tokenizer (``get_vocab()``) or the legacy SentencePiece
wrapper (``vocab`` token → id).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional


def token_strings(tokenizer: Any) -> Optional[List[str]]:
    """Return id-ordered token strings for a tokenizer (HuggingFace or SPM).

    Args:
        tokenizer: Tokenizer instance. HuggingFace tokenizers expose
            ``get_vocab()``; the legacy ``SpanishSPMTokenizer`` exposes a
            ``vocab`` mapping of token → id.

    Returns:
        A list of token strings indexed by token id, or ``None`` when the
        tokenizer does not expose a vocabulary.
    """
    if tokenizer is None:
        return None
    mapping: Optional[Dict[Any, Any]] = None
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            mapping = dict(get_vocab())
        except Exception:  # noqa: BLE001 - adapters may not implement it
            mapping = None
    if not mapping:
        legacy_vocab = getattr(tokenizer, "vocab", None)
        if isinstance(legacy_vocab, dict):
            mapping = legacy_vocab
    if not mapping:
        # Deployment adapters wrap the real tokenizer; delegate one level.
        for attr in ("_tok", "tokenizer", "hf_tokenizer"):
            inner = getattr(tokenizer, attr, None)
            if inner is not None and inner is not tokenizer:
                return token_strings(inner)
    if not mapping:
        return None
    size = max(int(index) for index in mapping.values()) + 1
    strings: List[str] = [""] * size
    for token, index in mapping.items():
        strings[int(index)] = str(token)
    return strings


def inject_token_vocab(config: Any, tokenizer: Any) -> None:
    """Attach the tokenizer vocabulary to the config when dynamic embeddings are on.

    No-op when the model does not use ``embedding.dynamic``: the vocabulary is
    only needed by the EELBERT-style hash embedding. When dynamic embeddings are
    enabled but the tokenizer exposes no vocabulary, a warning is logged and the
    model build fails later with an actionable error.

    Args:
        config: :class:`FrankensteinModelConfig` (or ``None``).
        tokenizer: Tokenizer instance used to encode the data.
    """
    if config is None or not bool(getattr(config, "use_dynamic_embedding", False)):
        return
    strings = token_strings(tokenizer)
    if strings is None:
        logging.warning(
            "embedding.dynamic.enabled is set but the tokenizer exposes no "
            "vocabulary; model construction will fail until token strings exist"
        )
        return
    config.token_vocab = strings
    # Keep the model vocabulary aligned with the ids the tokenizer can emit.
    # The HF paths already inject ``len(tokenizer)``, but the legacy SPM path
    # does not: its on-disk model may round the requested vocabulary size.
    config.vocab_size = len(strings)
