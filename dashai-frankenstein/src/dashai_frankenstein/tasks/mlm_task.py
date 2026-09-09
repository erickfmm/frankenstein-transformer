"""MaskedLanguageModelingTask — a DashAI BaseTask for MLM pretraining.

DashAI has no built-in language-modeling task, so this plugin provides one.
It declares a text input column and a text output column (the same corpus is
used for both: the model masks tokens itself via the Frankenstein MLM
objective) and binds to :class:`FrankensteinPretrainer` via its schema.

The task is *self-supervised*: the "output" column is the same text fed as
input — DashAI's run flow needs a labelled target column, so the user selects
the same text column twice. There is no ground truth to score against, so
the schema's metric list is empty (training telemetry flows through
:class:`~dashai_frankenstein.adapters.telemetry.DashAITelemetryCallback`
instead).
"""
from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING, Union

from DashAI.back.core.utils import MultilingualString
from DashAI.back.tasks.base_task import BaseTask
from DashAI.back.types.value_types import Text


class MaskedLanguageModelingTask(BaseTask):
    """Task for masked language modeling (BERT-style pre-training).

    Takes one text column as input; the same column is used as the MLM
    target (self-supervised). Compatible with :class:`FrankensteinPretrainer`.
    """

    COMPATIBLE_COMPONENTS: List[str] = []

    metadata: dict = {
        "inputs_types": [Text],
        "outputs_types": [Text],  # same text column reused as the MLM target
        "inputs_cardinality": 1,
        "outputs_cardinality": 1,
    }

    DESCRIPTION: str = MultilingualString(
        en=(
            "Masked language modeling pre-training on raw text. The model "
            "masks ~15% of tokens and learns to predict them (BERT-style). "
            "Select the same text column as input and output."
        ),
        es=(
            "Pre-entrenamiento MLM sobre texto crudo. El modelo enmascara "
            "~15% de los tokens y aprende a predecirlos (estilo BERT). "
            "Selecciona la misma columna de texto como entrada y salida."
        ),
        pt=(
            "Pré-treino de modelagem de linguagem mascarada sobre texto "
            "bruto. O modelo mascara ~15% dos tokens e aprende a prevê-los."
        ),
        de=(
            "Masked-Language-Modeling-Pretraining auf Rohtext. Das Modell "
            "maskiert ~15% der Token und lernt, sie vorherzusagen."
        ),
        zh="在原始文本上进行掩码语言建模预训练（BERT 风格），模型掩码约 15% 的 token 并学习预测。",
    )
    DISPLAY_NAME: str = MultilingualString(
        en="Masked Language Modeling",
        es="Modelado de Lenguaje Enmascarado",
        pt="Modelagem de Linguagem Mascarada",
        de="Masked Language Modeling",
        zh="掩码语言建模",
    )

    @property
    def schema(self) -> Dict[str, Any]:
        """Components compatible with this task.

        Returns
        -------
        dict
            Mapping with ``models`` and ``metrics`` lists. No supervised
            metrics apply to self-supervised MLM; training telemetry is
            streamed by the Frankenstein telemetry callback instead.
        """
        return {
            "models": ["FrankensteinPretrainer"],
            "metrics": [],
        }

    def prepare_for_task(
        self,
        dataset: Union["DashAIDataset", Any],
        input_columns: List[str],
        output_columns: List[str],
    ) -> "DashAIDataset":
        """Validate a text-in dataset for MLM pre-training.

        Both the input and the (reused) output column must be text; no type
        transformation is applied — the model tokenizes the raw text.
        """
        from DashAI.back.types.value_types import Text

        dashai_dataset = super().prepare_for_task(
            dataset, input_columns, output_columns
        )
        for col in input_columns:
            col_type = dashai_dataset.types.get(col)
            if col_type is not None and not isinstance(col_type, Text):
                raise TypeError(
                    f"MLM pretraining needs a text column; '{col}' is "
                    f"{type(col_type).__name__}."
                )
        return dashai_dataset

    def num_labels(self, dataset: "DashAIDataset", output_column: str) -> int | None:
        """MLM is self-supervised: no label count applies."""
        return None


if TYPE_CHECKING:  # pragma: no cover
    from DashAI.back.dataloaders.classes.dashai_dataset import DashAIDataset  # noqa: F401