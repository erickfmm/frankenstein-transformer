"""FrankensteinCausalLMModel — DashAI causal-LM pretraining component.

Trains a configurable Frankenstein decoder backbone with the real causal
language modeling objective (``task="causal_lm"`` in the Frankenstein engine)
on a DashAI text dataset (or on the corpus configured inside the
``frankenstein_json`` when the ``use_dashai_dataset`` checkbox is off).
Batches are tokenized at collate time with ``labels == input_ids`` and
consumed by :class:`TitanTrainer.compute_causal_lm_loss` (which shifts the
labels internally).

All step/epoch telemetry (loss, accuracy, learning rate, gradient norms,
GPU metrics) streams into DashAI's native metric store via
:class:`~dashai_frankenstein.adapters.telemetry.DashAITelemetryCallback`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from DashAI.back.core.utils import MultilingualString
from DashAI.back.models.base_model import BaseModel

from dashai_frankenstein.config import FrankensteinClassifierSchema
from dashai_frankenstein.engine import (
    build_model_from_json,
    resolve_device,
    resolve_tokenizer,
    validate_training_json,
)
from dashai_frankenstein.models.base import (
    persistence_load,
    persistence_save,
    resolve_json,
    _inject_engine_overrides,
)


class FrankensteinCausalLMModel(BaseModel):
    """Frankenstein decoder pre-trained with the causal-LM objective.

    Builds the decoder backbone from the passthrough config (all attention
    mixers / norms / activations available) and runs the engine's causal-LM
    loop on a DashAI text dataset (next-token prediction). The resulting
    model can be exported for generative inference through ``save``/``load``.

    ``predict`` returns the raw logits of the first batch preview only (causal
    LM has no supervised inference contract in DashAI; the artifact is the
    trained backbone saved by ``save``).
    """

    COMPATIBLE_COMPONENTS = ["CausalLMPretrainingTask"]
    DISPLAY_NAME: str = MultilingualString(
        en="Frankenstein Causal LM",
        es="Frankenstein LM Causal",
        pt="Frankenstein LM Causal",
        de="Frankenstein Causal LM",
        zh="Frankenstein 因果语言模型",
    )
    DESCRIPTION: str = MultilingualString(
        en=(
            "GPT-style causal language modeling pre-training on your text "
            "dataset (next-token prediction). Fully configurable backbone; "
            "every optimizer step streams metrics into the DashAI run."
        ),
        es=(
            "Pre-entrenamiento causal sobre tu dataset de texto (predicción "
            "del siguiente token). Backbone totalmente configurable; cada "
            "paso del optimizador se registra en el run de DashAI."
        ),
        pt=(
            "Pré-treino de modelagem causal sobre seu dataset de texto "
            "(previsão do próximo token). Backbone totalmente configurável."
        ),
        de=(
            "Causal-Language-Modeling-Pretraining auf Ihrem Textdatensatz "
            "(Next-Token-Prädiktion). Vollständig konfigurierbares Backbone."
        ),
        zh="在文本数据集上进行因果语言建模预训练（下一 token 预测）。骨干网络完全可配置。",
    )
    COLOR: str = "#5E35B1"
    ICON: str = "AutoStories"
    SCHEMA = FrankensteinClassifierSchema

    def __init__(self, **kwargs) -> None:
        kwargs = self.validate_and_transform(kwargs)
        self.frankenstein_json = kwargs.get("frankenstein_json", "")
        self.use_dashai_dataset = bool(kwargs.get("use_dashai_dataset", True))

        self.fitted = False
        self._frank_model = None
        self._loaded_config = None
        self._tokenizer = None
        self._device = "cpu"
        self._batch_size = 8
        self.x_data = None
        self.y_data = None

    def train(self, x_train, y_train, x_validation=None, y_validation=None):
        """Run causal-LM pre-training via the Frankenstein engine.

        When ``use_dashai_dataset`` is true (default) the DashAI text dataset
        is tokenized into causal-LM dict batches and passed to the engine.
        When false, ``dataset=None`` is passed and the engine builds the
        corpus from the config's ``text_dataset`` block (or the legacy
        RedPajama streaming corpus).
        """
        from DashAI.back.core.enums.metrics import SplitEnum

        from dashai_frankenstein.adapters.dataset import causal_lm_dataloader_dict
        from dashai_frankenstein.adapters.telemetry import DashAITelemetryCallback
        from dashai_frankenstein.engine import train_from_config

        json_text = resolve_json(self)
        validate_training_json(json_text)

        model, loaded, _ = build_model_from_json(
            json_text, model_class_override="frankensteindecoder"
        )

        runtime = getattr(loaded, "training_runtime", {}) or {}
        device = resolve_device(runtime.get("device", "auto"))
        batch_size = int(runtime.get("batch_size", 8) or 8)
        num_epochs = int(runtime.get("num_epochs", 3) or 3)
        max_length = int(runtime.get("max_length", 128) or 128)

        tokenizer = resolve_tokenizer(loaded)
        if tokenizer is None:
            raise ValueError(
                "A tokenizer is required for causal-LM pretraining. Set "
                "tokenizer.name_or_path (or base_model) in the Frankenstein "
                "config."
            )
        # Vocab must match the tokenizer (Frankenstein constraint).
        if hasattr(model, "emb") and hasattr(model.emb, "num_embeddings"):
            tok_vocab = len(tokenizer)
            if tok_vocab != int(model.emb.num_embeddings):
                model, loaded, _ = build_model_from_json(
                    json_text,
                    model_class_override="frankensteindecoder",
                    vocab_size_override=tok_vocab,
                )

        engine_cfg = _inject_engine_overrides(
            json_text,
            task="causal_lm",
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

        callback = DashAITelemetryCallback(
            self, x_train, y_train,
            split=SplitEnum.TRAIN, log_every_n_steps=1,
        )

        self._frank_model = model.to(device)
        self._loaded_config = loaded
        self._tokenizer = tokenizer
        self._device = device
        self._batch_size = batch_size

        train_loader = None
        if self.use_dashai_dataset:
            text_col = _first_text_column(x_train)
            train_loader = causal_lm_dataloader_dict(
                x_train, tokenizer, text_col,
                batch_size=batch_size, max_length=max_length,
                device=device, shuffle=True,
            )

        result = train_from_config(
            engine_cfg,
            dataset=train_loader,
            device=device,
            supervisor="off",
            metrics_callback=callback,
        )
        if getattr(result, "model", None) is not None:
            self._frank_model = result.model.to(device)

        self.fitted = True
        self.x_data = {"train": x_train}
        self.y_data = {"train": y_train}
        if x_validation is not None:
            self.x_data["validation"] = x_validation
            self.y_data["validation"] = y_validation
        return self

    def predict(self, x_pred):
        """Return raw next-token logits for the first batch of ``x_pred``.

        Causal LM has no supervised inference contract in DashAI; this
        exposes the decoder's raw logits ``(batch, seq, vocab)`` of the first
        batch for smoke-checking the artifact.
        """
        if not self.fitted:
            from sklearn.exceptions import NotFittedError

            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'train' before using it."
            )
        import torch

        from dashai_frankenstein.adapters.dataset import causal_lm_dataloader_dict

        loader = causal_lm_dataloader_dict(
            x_pred, self._tokenizer, _first_text_column(x_pred),
            batch_size=int(getattr(self, "_batch_size", 8) or 8),
            max_length=int(
                (getattr(self._loaded_config, "training_runtime", {}) or {})
                .get("max_length", 128) or 128
            ),
            device=self._device, shuffle=False,
        )
        model = self._frank_model
        model.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            logits = model(batch["input_ids"], batch["attention_mask"])
        return logits.detach().cpu().numpy()

    def save(self, filename: Union[str, Path]) -> None:
        persistence_save(self, str(filename))

    @classmethod
    def load(cls, filename: Union[str, Path]) -> "FrankensteinCausalLMModel":
        return persistence_load(cls, str(filename))


def _first_text_column(dataset: Any) -> str:
    """Return the single text (non-categorical) column of a dataset."""
    from DashAI.back.types.categorical import Categorical

    try:
        types = dataset.types
    except AttributeError:
        types = {}
    candidates = [
        col
        for col in dataset.column_names
        if not isinstance(types.get(col), Categorical)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Causal-LM pretraining expects exactly one text column, found "
            f"{candidates} in {list(dataset.column_names)}."
        )
    return candidates[0]