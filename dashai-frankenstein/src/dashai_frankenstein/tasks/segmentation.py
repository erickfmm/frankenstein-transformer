"""SegmentationTask — a DashAI BaseTask for image segmentation.

DashAI has no built-in segmentation task, so this plugin provides one (audit
§5.3 / §6 Phase 3). It declares an image input and an image-mask output and
binds to :class:`FrankensteinViTSegmenter` via ``COMPATIBLE_COMPONENTS``.

.. note::
   DashAI's frontend mask rendering is still an open question (audit §10 Q4).
   The backend contract (input image column in / per-pixel class map out) is
   wired here so the task registers and the model binds; output visualization
   may require a local frontend addition.
"""
from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING, Union

from DashAI.back.core.utils import MultilingualString
from DashAI.back.tasks.base_task import BaseTask
from DashAI.back.types.categorical import Categorical
from DashAI.back.types.dashai_image import DashAIImage


class SegmentationTask(BaseTask):
    """Task for per-pixel image segmentation (semantic masks).

    Takes one image column as input and produces a per-pixel class-index map.
    The output column carries the predicted segmentation mask. Compatible with
    :class:`FrankensteinViTSegmenter`.
    """

    COMPATIBLE_COMPONENTS: List[str] = []  # metrics auto-bound by name elsewhere

    metadata: dict = {
        "inputs_types": [DashAIImage],
        "outputs_types": [DashAIImage],  # predicted mask rendered as an image
        "inputs_cardinality": 1,
        "outputs_cardinality": 1,
    }

    DESCRIPTION: str = MultilingualString(
        en=(
            "Segment images into per-pixel class maps. E.g.: medical masks, "
            "scene parsing, defect detection."
        ),
        es=(
            "Segmenta imágenes en mapas de clases por píxel. Ej.: máscaras "
            "médicas, parsing de escena, detección de defectos."
        ),
        pt="Segmenta imagens em mapas de classes por pixel.",
        de="Segmentiert Bilder in Per-Pixel-Klassenkarten.",
        zh="将图像分割为逐像素类别图。",
    )
    DISPLAY_NAME: str = MultilingualString(
        en="Image Segmentation",
        es="Segmentación de Imágenes",
        pt="Segmentação de Imagens",
        de="Bildsegmentierung",
        zh="图像分割",
    )

    @property
    def schema(self) -> Dict[str, Any]:
        """Components compatible with this task.

        Returns
        -------
        dict
            Mapping with ``models`` and ``metrics`` lists (metric names are
            advisory; segmentation-specific metrics like mIoU are not yet in
            DashAI, so the list is intentionally minimal).
        """
        return {
            "models": ["FrankensteinViTSegmenter"],
            "metrics": [],
        }

    def prepare_for_task(
        self,
        dataset: Union["DashAIDataset", Any],
        input_columns: List[str],
        output_columns: List[str],
    ) -> "DashAIDataset":
        """Validate an image-in / mask-out dataset for segmentation."""
        dashai_dataset = super().prepare_for_task(
            dataset, input_columns, output_columns
        )
        return dashai_dataset

    def num_labels(self, dataset: "DashAIDataset", output_column: str) -> int | None:
        """Return the number of segmentation classes, if derivable.

        Segmentation masks are images, not ``Categorical`` columns, so this
        returns ``None`` unless the column happens to be categorical. The
        segmenter derives ``num_seg_classes`` from its YAML config instead.
        """
        output_type = dataset.types.get(output_column)
        if isinstance(output_type, Categorical):
            return output_type.num_categories()
        return None


if TYPE_CHECKING:  # pragma: no cover
    from DashAI.back.dataloaders.classes.dashai_dataset import DashAIDataset  # noqa: F401
