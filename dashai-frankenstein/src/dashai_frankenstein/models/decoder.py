"""FrankensteinDecoderModel — DashAI generative component (causal LM).

Wraps :class:`FrankensteinDecoder` (which exposes ``forward`` and
``generate``) as a DashAI ``BaseGenerativeModel`` bound to
``TextToTextGenerationTask``. Forces ``model_class: frankensteindecoder`` (which
forces ``mode: decoder``). Generation is autoregressive top-k sampling via the
decoder's own ``generate`` method; tokenization/detokenization uses an HF
tokenizer resolved from the YAML.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Union

from DashAI.back.core.utils import MultilingualString
from DashAI.back.models.base_generative_model import BaseGenerativeModel

from dashai_frankenstein.config import FrankensteinPassthroughSchema
from dashai_frankenstein.engine import (
    build_model_from_yaml,
    resolve_device,
    resolve_tokenizer,
    validate_training_yaml,
)
from dashai_frankenstein.models.base import resolve_yaml


class FrankensteinDecoderSchema(FrankensteinPassthroughSchema):
    """Schema for the causal decoder; adds generation parameters."""


class FrankensteinDecoderModel(BaseGenerativeModel):
    """Frankenstein causal decoder for text-to-text generation.

    Builds a :class:`FrankensteinDecoder` backbone (all mixers/norms/activations
    via the passthrough YAML) and exposes autoregressive generation. The decoder
    is not fine-tuned by DashAI's generative-task flow; it is instantiated from
    the YAML (or a checkpoint loaded via ``load``) and used for inference.

    Generation parameters (``max_new_tokens``, ``temperature``, ``top_k``) are
    surfaced on the schema as passthrough/convenience fields. Training
    parameters (``device``, ``batch_size``, ``num_epochs``, learning rate) are
    read from the Frankenstein YAML's ``training_runtime`` / optimizer
    parameters — they are NOT DashAI form fields.
    """

    COMPATIBLE_COMPONENTS = ["TextToTextGenerationTask"]
    DISPLAY_NAME: str = MultilingualString(
        en="Frankenstein Decoder",
        es="Decoder Frankenstein",
        pt="Decoder Frankenstein",
        de="Frankenstein Decoder",
        zh="Frankenstein 解码器",
    )
    DESCRIPTION: str = MultilingualString(
        en=(
            "Frankenstein causal decoder for autoregressive text generation. "
            "All attention mixers/norms/activations are configurable via the "
            "passthrough YAML."
        ),
        es=(
            "Decoder causal Frankenstein para generación de texto "
            "autorregresiva. Configurable vía YAML de paso."
        ),
        pt="Decoder causal Frankenstein para geração de texto.",
        de="Frankenstein Decoder für autoregressive Textgenerierung.",
        zh="用于自回归文本生成的 Frankenstein 因果解码器。",
    )
    COLOR: str = "#6A1B9A"
    ICON: str = "Forum"
    SCHEMA = FrankensteinDecoderSchema

    def __init__(self, **kwargs) -> None:
        kwargs = self.validate_and_transform(kwargs)
        self.frankenstein_yaml = kwargs.get("frankenstein_yaml", "")
        # Generation defaults (overridable via kwargs from the schema/runner).
        self.max_new_tokens = int(kwargs.get("max_new_tokens", 128))
        self.temperature = float(kwargs.get("temperature", 1.0))
        self.top_k = int(kwargs.get("top_k", 50))

        self.fitted = False
        self._frank_model = None
        self._loaded_config = None
        self._tokenizer = None
        self._device = "cpu"

    def _ensure_model(self) -> None:
        """Lazily build/resolve the decoder + tokenizer if not already loaded."""
        if self._frank_model is not None:
            return

        yaml_text = resolve_yaml(self)
        validate_training_yaml(yaml_text)

        model, loaded, _ = build_model_from_yaml(
            yaml_text, model_class_override="frankensteindecoder"
        )
        runtime = getattr(loaded, "training_runtime", {}) or {}
        device = resolve_device(runtime.get("device", "auto"))
        self._device = device

        tokenizer = resolve_tokenizer(loaded)
        if tokenizer is None:
            raise ValueError(
                "A tokenizer is required for generation. Set "
                "tokenizer.name_or_path (or base_model) in the Frankenstein YAML."
            )
        # Keep embedding vocab consistent with the tokenizer (constraint).
        tok_vocab = len(tokenizer)
        if hasattr(model, "backbone") and hasattr(model.backbone, "emb"):
            if tok_vocab != int(model.backbone.emb.num_embeddings):
                model, loaded, _ = build_model_from_yaml(
                    yaml_text,
                    model_class_override="frankensteindecoder",
                    vocab_size_override=tok_vocab,
                )

        self._frank_model = model.to(device)
        self._loaded_config = loaded
        self._tokenizer = tokenizer

    def generate(self, input: Union[Any, List[Any]]) -> List[str]:
        """Generate text from a prompt (autoregressive top-k sampling).

        Parameters
        ----------
        input : Any
            A prompt (string) or a list of chat-message dicts
            (``[{"role": ..., "content": ...}]``) as produced by
            ``TextToTextGenerationTask.prepare_for_task``.

        Returns
        -------
        list of str
            A single-element list containing the generated text.
        """
        import torch

        self._ensure_model()

        prompt = _extract_prompt(input)
        tok = self._tokenizer
        device = self._device
        enc = tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        self._frank_model.eval()
        out_ids = self._frank_model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
        )
        new_ids = out_ids[0, input_ids.shape[-1]:]
        text = tok.decode(new_ids, skip_special_tokens=True)
        return [text]

    def save(self, filename: Union[str, Path]) -> None:
        from dashai_frankenstein.models.base import persistence_save

        persistence_save(self, str(filename))

    @classmethod
    def load(cls, filename: Union[str, Path]) -> "FrankensteinDecoderModel":
        from dashai_frankenstein.models.base import persistence_load

        instance = persistence_load(cls, str(filename))
        instance._device = resolve_device("auto")
        if instance._frank_model is not None:
            instance._frank_model = instance._frank_model.to(instance._device)
        return instance


def _extract_prompt(input: Any) -> str:
    """Normalize a DashAI generative input into a plain prompt string."""
    if isinstance(input, str):
        return input
    if isinstance(input, list) and input:
        parts = []
        for msg in input:
            if isinstance(msg, dict) and "content" in msg:
                parts.append(str(msg["content"]))
            else:
                parts.append(str(msg))
        return "\n".join(parts)
    return str(input)
