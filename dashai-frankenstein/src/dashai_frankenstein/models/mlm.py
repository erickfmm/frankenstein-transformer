"""FrankensteinMLMModel — DashAI text-classification component over the encoder.

Wraps the Frankenstein encoder backbone with the Strategy-A sequence-level
classification head (see ``docs/dashai-plugin-audit.md`` §5.4). The encoder is
built in-process via the Frankenstein engine; DashAI drives train/predict/save/
load. The Frankenstein YAML (passed through the ``frankenstein_yaml`` field)
remains the source of truth for the backbone configuration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from sklearn.exceptions import NotFittedError

from DashAI.back.core.utils import MultilingualString
from DashAI.back.models.text_classification_model import TextClassificationModel

from dashai_frankenstein.config import FrankensteinClassifierSchema
from dashai_frankenstein.models.base import (
    classification_predict,
    classification_train,
    persistence_load,
    persistence_save,
    resolve_yaml,
)


class FrankensteinMLMModel(TextClassificationModel):
    """Frankenstein encoder (MLM backbone) for DashAI text classification.

    Builds a custom Frankenstein encoder with an optional classification head
    (DashAI Strategy A) and fine-tunes it on a labelled text dataset. Compatible
    with ``TextClassificationTask``.

    The backbone is fully configurable through the passthrough
    ``frankenstein_yaml`` field (all 33 attention mixers, 23 optimizers, 6
    norms, 43 activations are available). The classifier head is a full-precision
    ``nn.Linear`` over a pooled representation and is NOT BitNet-quantized.
    """

    DISPLAY_NAME: str = MultilingualString(
        en="Frankenstein Encoder",
        es="Encoder Frankenstein",
        pt="Encoder Frankenstein",
        de="Frankenstein Encoder",
        zh="Frankenstein 编码器",
    )
    DESCRIPTION: str = MultilingualString(
        en=(
            "Config-driven Frankenstein transformer encoder for text "
            "classification. Supports 33 attention mixers, BitNet, MoE, and "
            "custom norms/activations via the passthrough YAML."
        ),
        es=(
            "Encoder transformer Frankenstein configurable para clasificación "
            "de texto. 33 mixers de atención, BitNet, MoE y normas/activaciones "
            "personalizadas vía YAML de paso."
        ),
        pt=(
            "Encoder transformer Frankenstein configurável para classificação "
            "de texto via YAML."
        ),
        de=(
            "Konfigurierbarer Frankenstein-Encoder für Textklassifikation via "
            "Passthrough-YAML."
        ),
        zh="可配置的 Frankenstein 编码器，用于文本分类（通过 YAML）。",
    )
    COLOR: str = "#2E7D32"
    ICON: str = "Science"
    SCHEMA = FrankensteinClassifierSchema

    def __init__(self, **kwargs) -> None:
        """Initialize the component from DashAI form parameters.

        Parameters
        ----------
        **kwargs
            Validated against :class:`FrankensteinClassifierSchema`
            (``frankenstein_yaml``).
        """
        kwargs = self.validate_and_transform(kwargs)
        self.frankenstein_yaml = kwargs.get("frankenstein_yaml", "")

        self.num_labels = None
        self.fitted = False
        self.encodings: dict = {}
        self._frank_model = None
        self._loaded_config = None
        self._tokenizer = None
        self._device = "cpu"
        self._batch_size = 16
        self._label_column = None
        self._text_column_fn = lambda ds: _first_text_column(ds)
        self.x_data = None
        self.y_data = None

    # -- DashAI contract -----------------------------------------------------

    def train(
        self,
        x_train,
        y_train,
        x_validation=None,
        y_validation=None,
    ):
        """Fine-tune the encoder + classification head on the DashAI dataset.

        The label column is categorical on ``y_train``; it is integer-encoded
        and merged into ``x_train`` for tokenization.
        """
        from DashAI.back.dataloaders.classes.dashai_dataset_utils import (
            apply_categorical_label_encoder,
            categorical_label_encoder,
        )

        output_column_name = y_train.column_names[0]
        x_train_prepared, encodings = categorical_label_encoder(x_train, y_train)
        self.encodings = encodings

        num_labels = len(y_train.unique(output_column_name))
        x_train_merged = x_train_prepared

        x_val_merged = None
        if x_validation is not None and y_validation is not None:
            x_validation_prepared = apply_categorical_label_encoder(
                x_validation, encodings
            )
            x_val_merged = x_validation_prepared

        return classification_train(
            self,
            x_train_merged,
            y_train,
            x_val_merged,
            y_validation,
            num_labels=num_labels,
            label_column=output_column_name,
            text_column_fn=lambda ds: _first_text_column(ds),
            model_class_override=None,
        )

    def predict(self, x_pred):
        """Return an ``(N, num_labels)`` probability matrix (softmax)."""
        if not self.fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet."
            )
        x_pred_prepared = apply_encodings(x_pred, self.encodings)
        return classification_predict(self, x_pred_prepared)

    def save(self, filename: Union[str, Path]) -> None:
        """Persist the model bundle to the DashAI run directory."""
        persistence_save(self, str(filename))

    @classmethod
    def load(cls, filename: Union[str, Path]) -> "FrankensteinMLMModel":
        """Rebuild a model instance from a saved bundle."""
        return persistence_load(cls, str(filename))


# -- helpers ------------------------------------------------------------------

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
            f"Expected exactly one text column, found {candidates} in "
            f"{list(dataset.column_names)}."
        )
    return candidates[0]


def apply_encodings(dataset: Any, encodings: dict) -> Any:
    """Apply a previously-fit categorical label encoding to a predict dataset."""
    if not encodings:
        return dataset
    try:
        from DashAI.back.dataloaders.classes.dashai_dataset_utils import (
            apply_categorical_label_encoder,
        )

        return apply_categorical_label_encoder(dataset, encodings)
    except Exception:
        return dataset
