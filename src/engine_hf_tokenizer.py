"""HF tokenizer resolution for host-owned MLM corpora in the engine.

Small helper so :mod:`src.engine` can resolve a HuggingFace tokenizer from a
config's ``tokenizer`` block without touching the legacy Spanish-SPM path.
"""
from __future__ import annotations

import logging
from typing import Any

try:
    from .utils.hf_compat import load_auto_tokenizer
except ImportError:  # tests / CLI run with src/ itself on sys.path
    from utils.hf_compat import load_auto_tokenizer

logger = logging.getLogger(__name__)


def build_hf_tokenizer_from_config(tokenizer_config: dict) -> Any:
    """Resolve an HF tokenizer from a config's ``tokenizer`` block.

    Args:
        tokenizer_config: The ``tokenizer`` mapping (must carry
            ``name_or_path``).

    Returns:
        A ``PreTrainedTokenizer(Fast)`` with a pad token derived from the
        EOS token when missing.

    Raises:
        ValueError: If ``name_or_path`` is missing.
    """
    name = str((tokenizer_config or {}).get("name_or_path", "")).strip()
    if not name:
        raise ValueError("tokenizer.name_or_path is required for host-owned MLM corpora")
    trust_remote_code = bool(tokenizer_config.get("trust_remote_code", False))
    use_fast = bool(tokenizer_config.get("use_fast", True))
    logger.info("Loading HF tokenizer: %s (fast=%s)", name, use_fast)
    tokenizer = load_auto_tokenizer(
        name, use_fast=use_fast, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer