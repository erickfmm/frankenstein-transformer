"""HF checkpoint-loading compatibility helpers (transformers 4.x and 5.x).

transformers 5 removed the legacy fallback that resolved auto classes for old
checkpoints whose ``config.json`` lacks ``model_type`` and whose hub repo
predates ``tokenizer_config.json`` (e.g. ``prajjwal1/bert-tiny``). For those,
``AutoTokenizer.from_pretrained`` fails with an opaque backend-tokenizer
``ValueError`` (and ``AutoModelFor*.from_pretrained`` with ``Unrecognized
model``), while the explicit classes (``BertTokenizer``,
``BertForMaskedLM``) keep working on both majors.

These helpers restore the transformers-4 behaviour: on auto-class failure,
derive the architecture prefix from the ``architectures`` entry of the
checkpoint's ``config.json`` and retry with the matching explicit class.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Known task suffixes of ``transformers`` architecture class names, longest
# first so that e.g. ``ForConditionalGeneration`` wins over ``Model``.
_TASK_SUFFIXES = (
    "ForConditionalGeneration",
    "ForSequenceClassification",
    "ForTokenClassification",
    "ForQuestionAnswering",
    "ForImageClassification",
    "ForSemanticSegmentation",
    "ForObjectDetection",
    "ForMaskedLM",
    "ForCausalLM",
    "ForPreTraining",
    "LMHeadModel",
    "Model",
)


def _read_config_json(name_or_path: str) -> dict:
    """Read a checkpoint's ``config.json`` as a dict (best effort).

    Args:
        name_or_path: HF hub repo id or local directory.

    Returns:
        The parsed ``config.json`` mapping, or an empty dict when the file
        cannot be read (missing repo, offline, corrupted).
    """
    path: str | None = None
    if os.path.isdir(name_or_path):
        candidate = os.path.join(name_or_path, "config.json")
        if os.path.isfile(candidate):
            path = candidate
    else:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=name_or_path, filename="config.json")
        except Exception:  # noqa: BLE001 - best effort only
            return {}
    if path is None:
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:  # noqa: BLE001 - best effort only
        return {}


def _architecture_prefix(name_or_path: str) -> str | None:
    """Derive the model-family prefix (e.g. ``"Bert"``) for a checkpoint.

    Mirrors the transformers-4 resolution order that transformers 5 dropped:
    first the ``architectures`` entries, then a config-key signature probe for
    legacy checkpoints that carry neither ``model_type`` nor
    ``architectures`` (bare 2018-2020 BERT-family mirrors such as
    ``prajjwal1/bert-tiny``).

    Args:
        name_or_path: HF hub repo id or local directory.

    Returns:
        The model-family prefix, or ``None`` when nothing matches.
    """
    config = _read_config_json(name_or_path)
    for architecture in config.get("architectures") or []:
        if not isinstance(architecture, str):
            continue
        for suffix in _TASK_SUFFIXES:
            if architecture.endswith(suffix) and len(architecture) > len(suffix):
                return architecture[: -len(suffix)]
    if "type_vocab_size" in config and "num_hidden_layers" in config:
        return "Bert"
    return None


def load_auto_tokenizer(
    name_or_path: str,
    *,
    use_fast: bool = True,
    trust_remote_code: bool = False,
) -> Any:
    """Load a tokenizer via ``AutoTokenizer`` with a legacy-checkpoint fallback.

    Args:
        name_or_path: HF hub repo id or local directory.
        use_fast: Request a fast tokenizer (forwarded to the auto class
            only; the explicit fallback class decides its own backend).
        trust_remote_code: Allow remote code when loading.

    Returns:
        A ``PreTrainedTokenizer(Fast)`` instance.

    Raises:
        Exception: Whatever the original auto-class load raised, when the
            fallback cannot resolve an explicit class either.
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            name_or_path, use_fast=use_fast, trust_remote_code=trust_remote_code
        )
    except Exception as exc:  # noqa: BLE001 - retried below, re-raised on failure
        prefix = _architecture_prefix(name_or_path)
        if prefix:
            import transformers

            tokenizer_cls = getattr(transformers, f"{prefix}Tokenizer", None)
            if tokenizer_cls is not None:
                logger.warning(
                    "AutoTokenizer failed for %s (%s); retrying with %s",
                    name_or_path,
                    exc,
                    tokenizer_cls.__name__,
                )
                return tokenizer_cls.from_pretrained(
                    name_or_path, trust_remote_code=trust_remote_code
                )
        raise


def load_auto_model(
    name_or_path: str,
    *,
    task: str = "mlm",
    trust_remote_code: bool = False,
    **kwargs: Any,
) -> Any:
    """Load a causal/masked-LM model via auto classes with a legacy fallback.

    Args:
        name_or_path: HF hub repo id or local directory.
        task: ``"mlm"`` resolves ``AutoModelForMaskedLM``, ``"causal_lm"``
            resolves ``AutoModelForCausalLM``.
        trust_remote_code: Allow remote code when loading.
        **kwargs: Extra kwargs forwarded to ``from_pretrained``.

    Returns:
        A ``PreTrainedModel`` instance.

    Raises:
        Exception: Whatever the original auto-class load raised, when the
            fallback cannot resolve an explicit class either.
    """
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

    auto_cls = AutoModelForMaskedLM if task == "mlm" else AutoModelForCausalLM
    try:
        return auto_cls.from_pretrained(
            name_or_path, trust_remote_code=trust_remote_code, **kwargs
        )
    except Exception as exc:  # noqa: BLE001 - retried below, re-raised on failure
        prefix = _architecture_prefix(name_or_path)
        explicit_suffix = "ForMaskedLM" if task == "mlm" else "ForCausalLM"
        if prefix:
            import transformers

            model_cls = getattr(transformers, f"{prefix}{explicit_suffix}", None)
            if model_cls is not None:
                logger.warning(
                    "AutoModel load failed for %s (%s); retrying with %s",
                    name_or_path,
                    exc,
                    model_cls.__name__,
                )
                return model_cls.from_pretrained(
                    name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
        raise
