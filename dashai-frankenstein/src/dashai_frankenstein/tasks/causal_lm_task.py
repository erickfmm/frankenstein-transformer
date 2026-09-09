"""CausalLMPretrainingTask — a DashAI BaseTask for causal-LM pretraining.

DashAI has no built-in language-modeling task, so this plugin provides one.
It declares a text input column and a text output column (the same corpus is
used for both: the model learns the next-token objective itself via the
Frankenstein causal-LM task) and binds to
:class:`FrankensteinCausalLMModel` via its schema.

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


class CausalLMPretrainingTask(BaseTask):
    """Task for causal language modeling (GPT-style pre-training).

    Takes one text column as input; the same column is used as the
    next-token target (self-supervised). Compatible with
    :class:`FrankensteinCausalLMModel`.
    """

    COMPATIBLE_COMPONENTS: List[str] = []

    metadata: dict = {
        "inputs_types": [Text],
        "outputs_types": [Text],  # same text column reused as the CLM target
        "inputs_cardinality": 1,
        "outputs_cardinality": 1,
    }

    DESCRIPTION: str = MultilingualString(
        en=(
            "Causal language modeling pre-training on raw text. The model "
            "learns to predict the next token (GPT-style). Select the same "
            "text column as input and output."
        ),
        es=(
            "Pre-entrenamiento causal sobre texto crudo. El modelo aprende "
            "a predecir el siguiente token (estilo GPT). Selecciona la misma "
            "columna de texto como entrada y salida."
        ),
        pt=(
            "Pré-treino de modelagem causal de linguagem sobre texto bruto. "
            "O modelo aprende a prever o próximo token."
        ),
        de=(
            "Causal-Language-Modeling-Pretraining auf Rohtext. Das Modell "
            "lernt, das nächste Token vorherzusagen."
        ),
        zh="在原始文本上进行因果语言建模预训练（GPT 风格），模型学习预测下一个 token。",
    )
    DISPLAY_NAME: str = MultilingualString(
        en="Causal Language Modeling",
        es="Modelado Causal de Lenguaje",
        pt="Modelagem Causal de Linguagem",
        de="Causal Language Modeling",
        zh="因果语言建模",
    )

    @property
    def schema(self) -> Dict[str, Any]:
        """Components compatible with this task.

        Returns
        -------
        dict
            Mapping with ``models`` and ``metrics`` lists. No supervised
            metrics apply to self-supervised CLM; training telemetry is
            streamed by the Frankenstein telemetry callback instead.
        """
        return {
            "models": ["FrankensteinCausalLMModel"],
            "metrics": [],
        }

    def prepare_for_task(
        self,
        dataset: Union["DashAIDataset", Any],
        input_columns: List[str],
        output_columns: List[str],
    ) -> "DashAIDataset":
        """Validate a text-in dataset for causal-LM pre-training.

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
                    f"Causal-LM pretraining needs a text column; '{col}' is "
                    f"{type(col_type).__name__}."
                )
        return dashai_dataset

    def num_labels(self, dataset: "DashAIDataset", output_column: str) -> int | None:
        """Causal LM is self-supervised: no label count applies."""
        return None


if TYPE_CHECKING:  # pragma: no cover
    from DashAI.back.dataloaders.classes.dashai_dataset import DashAIDataset  # noqa: F401