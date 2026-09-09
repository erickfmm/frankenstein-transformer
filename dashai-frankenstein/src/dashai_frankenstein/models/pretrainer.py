"""FrankensteinPretrainer — DashAI MLM pretraining component over the encoder.

Trains a configurable Frankenstein encoder backbone with the real masked
language modeling objective (``task="mlm"`` in the Frankenstein engine) on a
DashAI text dataset. Batches are tokenized + 15%-masked at collate time and
consumed by :class:`TitanTrainer.compute_mlm_loss`.

All step/epoch telemetry (loss, accuracy, learning rate, gradient norms,
GPU metrics) streams into DashAI's native metric store via
:class:`~dashai_frankenstein.adapters.telemetry.DashAITelemetryCallback`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from DashAI.back.core.utils import MultilingualString
from DashAI.back.models.base_model import BaseModel
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


class FrankensteinPretrainer(BaseModel):
    """Frankenstein encoder pre-trained with masked language modeling.

    Builds the encoder backbone from the passthrough config (all attention
    mixers / norms / activations available) and runs the engine's MLM loop
    on a DashAI text dataset. The resulting model can be exported for
    downstream fine-tuning (e.g. via ``FrankensteinMLMModel``) through
    ``save``/``load``.

    ``predict`` returns masked-token logits of the last batch preview only
    (MLM has no inference contract in DashAI; the artifact is the trained
    backbone saved by ``save``).
    """

    COMPATIBLE_COMPONENTS = ["MaskedLanguageModelingTask"]
    DISPLAY_NAME: str = MultilingualString(
        en="Frankenstein Pretrainer (MLM)",
        es="Pre-entrenador Frankenstein (MLM)",
        pt="Pré-treinador Frankenstein (MLM)",
        de="Frankenstein Pretrainer (MLM)",
        zh="Frankenstein 预训练器 (MLM)",
    )
    DESCRIPTION: str = MultilingualString(
        en=(
            "BERT-style masked language modeling pre-training on your text "
            "with a configurable Frankenstein encoder (33 mixers, BitNet, "
            "MoE via passthrough YAML). Use it to produce a pretrained "
            "backbone; metrics stream live into the DashAI run."
        ),
        es=(
            "Pre-entrenamiento MLM estilo BERT sobre tu texto con un encoder "
            "Frankenstein configurable (33 mixers, BitNet, MoE vía YAML). "
            "Genera un backbone pre-entrenado; las métricas se transmiten "
            "en vivo al run de DashAI."
        ),
        pt=(
            "Pré-treino MLM estilo BERT sobre seu texto com um encoder "
            "Frankenstein configurável (33 mixers, BitNet, MoE via YAML)."
        ),
        de=(
            "BERT-artiges MLM-Pretraining mit konfigurierbarem "
            "Frankenstein-Encoder (33 Mixer, BitNet, MoE via YAML)."
        ),
        zh="用可配置的 Frankenstein 编码器进行 BERT 风格的掩码语言建模预训练。",
    )
    COLOR: str = "#F57F17"
    ICON: str = "Psychology"
    SCHEMA = FrankensteinClassifierSchema

    def __init__(self, **kwargs) -> None:
        kwargs = self.validate_and_transform(kwargs)
        self.frankenstein_json = kwargs.get("frankenstein_json", "")

        self.fitted = False
        self._frank_model = None
        self._loaded_config = None
        self._tokenizer = None
        self._device = "cpu"
        self._batch_size = 8
        self.x_data = None
        self.y_data = None

    def train(self, x_train, y_train, x_validation=None, y_validation=None):
        """Run MLM pre-training on the DashAI text dataset via the engine.

        The same text column is used as input and target (self-supervised):
        masking happens at collate time, so ``y_train`` is only used to
        resolve the column when ``x_train`` carries no text (manual-input
        runs).
        """
        from DashAI.back.core.enums.metrics import SplitEnum

        from dashai_frankenstein.adapters.dataset import mlm_dataloader_dict
        from dashai_frankenstein.adapters.telemetry import DashAITelemetryCallback
        from dashai_frankenstein.engine import train_from_config

        json_text = resolve_json(self)
        validate_training_json(json_text)

        model, loaded, _ = build_model_from_json(json_text)

        runtime = getattr(loaded, "training_runtime", {}) or {}
        device = resolve_device(runtime.get("device", "auto"))
        batch_size = int(runtime.get("batch_size", 8) or 8)
        num_epochs = int(runtime.get("num_epochs", 3) or 3)
        max_length = int(runtime.get("max_length", 128) or 128)
        mlm_probability = float(runtime.get("mlm_probability", 0.15) or 0.15)

        tokenizer = resolve_tokenizer(loaded)
        if tokenizer is None:
            raise ValueError(
                "A tokenizer is required for MLM pretraining. Set "
                "tokenizer.name_or_path (or base_model) in the Frankenstein config."
            )
        # Vocab must match the tokenizer (Frankenstein constraint).
        if hasattr(model, "emb") and hasattr(model.emb, "num_embeddings"):
            tok_vocab = len(tokenizer)
            if tok_vocab != int(model.emb.num_embeddings):
                model, loaded, _ = build_model_from_json(
                    json_text, vocab_size_override=tok_vocab,
                )

        text_col = _first_text_column(x_train)
        train_loader = mlm_dataloader_dict(
            x_train, tokenizer, text_col,
            batch_size=batch_size, max_length=max_length,
            mlm_probability=mlm_probability, device=device, shuffle=True,
        )

        engine_cfg = _inject_engine_overrides(
            json_text,
            task="mlm",
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
        """Return masked-token logits for the first batch of ``x_pred``.

        MLM has no supervised inference contract in DashAI; this exposes the
        encoder's raw logits ``(num_masked, vocab_size)`` over masked tokens
        of the first batch for smoke-checking the artifact.
        """
        if not self.fitted:
            from sklearn.exceptions import NotFittedError

            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'train' before using it."
            )
        import torch

        from dashai_frankenstein.adapters.dataset import mlm_dataloader_dict

        loader = mlm_dataloader_dict(
            x_pred, self._tokenizer, _first_text_column(x_pred),
            batch_size=int(getattr(self, "_batch_size", 8) or 8),
            max_length=int(getattr(self._loaded_config.training_runtime, "get", lambda *_: 128)("max_length", 128) or 128),
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
    def load(cls, filename: Union[str, Path]) -> "FrankensteinPretrainer":
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
            f"MLM pretraining expects exactly one text column, found "
            f"{candidates} in {list(dataset.column_names)}."
        )
    return candidates[0]