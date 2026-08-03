"""Training configuration loader with schema validation.

Loads YAML training configuration files, validates required fields,
and produces structured :class:`LoadedTrainingConfig` objects containing
model configuration, training parameters, tokenizer settings, and runtime
overrides.
"""

import os
from dataclasses import dataclass
from dataclasses import fields
from typing import Dict, Any, Optional

import yaml

try:
    from .trainer import TrainingConfig
    from ..model.tormented_bert_frankestein import UltraConfig
    from ..utils.config_flatten import flatten_model_dict
except ImportError:
    from training.trainer import TrainingConfig
    from model.tormented_bert_frankestein import UltraConfig
    from utils.config_flatten import flatten_model_dict


def _field_names(cls) -> set:
    return {field.name for field in fields(cls)}


_BITNET_BOOL_KEYS = ("use_bitnet", "bitnet_routers")


def _validate_ffn_activation(model_data: Dict[str, Any]) -> None:
    """Validate FFN activation fields when present in the model block.

    Lightweight pre-check that complements :class:`UltraConfig` validation
    and the JSON-Schema ``enum``. Ensures ``ffn_activation`` is a string and
    ``ffn_activation_config`` is a mapping when provided.

    Args:
        model_data: The ``model`` block from the YAML config.

    Raises:
        ValueError: If ``ffn_activation`` is not a string or
            ``ffn_activation_config`` is present but not a mapping.
    """
    act = model_data.get("ffn_activation")
    if act is not None and not isinstance(act, str):
        raise ValueError(
            f"model.ffn_activation must be a string, got "
            f"{type(act).__name__}: {act!r}"
        )
    cfg = model_data.get("ffn_activation_config")
    if cfg is not None and not isinstance(cfg, dict):
        raise ValueError(
            f"model.ffn_activation_config must be an object/mapping, got "
            f"{type(cfg).__name__}"
        )


def _validate_bitnet_flags(model_data: Dict[str, Any]) -> None:
    """Validate that BitNet-related keys, when present, are booleans.

    Args:
        model_data: The ``model`` block from the YAML config.

    Raises:
        ValueError: If ``bitnet_routers`` is true while ``use_bitnet`` is
            explicitly false, or if either key has a non-boolean type.
    """
    for key in _BITNET_BOOL_KEYS:
        if key in model_data and not isinstance(model_data[key], bool):
            raise ValueError(
                f"model.{key} must be a boolean, got "
                f"{type(model_data[key]).__name__}: {model_data[key]!r}"
            )
    use_bitnet = model_data.get("use_bitnet")
    bitnet_routers = model_data.get("bitnet_routers")
    if bitnet_routers and use_bitnet is False:
        raise ValueError(
            "model.bitnet_routers=true requires model.use_bitnet=true "
            "(routing quantization is only meaningful when BitNet is enabled)."
        )


@dataclass
class LoadedTrainingConfig:
    """Structured result of loading and validating a training YAML config.

    Attributes:
        task: Training task identifier (``"mlm"`` or ``"sbert"``).
        model_class: Model class name (``"frankenstein"``, ``"mini"``,
            ``"frankesteindecoder"``, or ``"base_model"``).
        model_config: :class:`UltraConfig` for custom models, or ``None``
            when ``base_model`` is used.
        base_model: HuggingFace model identifier/path when using a
            pre-trained base model, or ``None``.
        tokenizer_config: Tokenizer configuration dictionary.
        training_config: :class:`TrainingConfig` with optimizer, scheduler,
            and stability parameters.
        training_runtime: Dictionary of runtime-only keys from the YAML
            that are not part of :class:`TrainingConfig` fields.
    """

    task: str
    model_class: Optional[str]
    model_config: Optional[UltraConfig]
    base_model: Optional[str]
    tokenizer_config: Dict[str, Any]
    training_config: TrainingConfig
    training_runtime: Dict[str, Any]


def load_training_config(path: str) -> LoadedTrainingConfig:
    """Load and validate a training configuration from a YAML file.

    Parses the YAML, validates required sections (``model``, ``training``,
    ``tokenizer``), resolves the training task, and constructs
    :class:`UltraConfig` and :class:`TrainingConfig` objects.

    Args:
        path: Filesystem path to the YAML configuration file.

    Returns:
        A fully validated :class:`LoadedTrainingConfig`.

    Raises:
        ValueError: If required fields are missing, invalid, or
            inconsistent (e.g. missing ``training.task``, invalid
            ``model_class``, missing ``tokenizer.name_or_path`` for
            base-model MLM training).
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    base_model = data.get("base_model")
    if base_model is not None:
        base_model = str(base_model).strip()
        if not base_model:
            raise ValueError("base_model must be a non-empty string when provided")

    model_class = data.get("model_class")
    if model_class is not None:
        model_class = str(model_class).strip().lower()

    valid_model_classes = {"frankenstein", "mini", "frankesteindecoder"}
    if model_class is not None and model_class not in valid_model_classes:
        raise ValueError(
            "model_class must be one of: frankenstein, mini, frankesteindecoder"
        )

    model_data = data.get("model", {}) or {}
    training_data = data.get("training", {}) or {}
    tokenizer_config = data.get("tokenizer", {}) or {}
    if not isinstance(tokenizer_config, dict):
        raise ValueError("tokenizer must be an object when provided")

    task = str(training_data.get("task", "")).strip().lower()
    if task not in {"mlm", "sbert"}:
        raise ValueError("training.task is required and must be one of: mlm, sbert")

    model_config: Optional[UltraConfig] = None
    if not base_model:
        if not model_data:
            raise ValueError("model is required when base_model is not provided")
        model_data = flatten_model_dict(model_data)
        _validate_bitnet_flags(model_data)
        _validate_ffn_activation(model_data)
        model_config = UltraConfig(**model_data)
        if not model_class:
            model_class = "frankenstein"
    else:
        # model/model_class are intentionally ignored when base_model is set.
        model_class = model_class or "base_model"

    if base_model and task == "mlm":
        tokenizer_name_or_path = tokenizer_config.get("name_or_path")
        if not isinstance(tokenizer_name_or_path, str) or not tokenizer_name_or_path.strip():
            raise ValueError(
                "tokenizer.name_or_path must be provided for MLM training when base_model is set"
            )

    optimizer_data = training_data.get("optimizer")
    if task == "mlm":
        if not isinstance(optimizer_data, dict):
            raise ValueError(
                "Missing required 'training.optimizer' object in config. "
                "Legacy top-level optimizer fields were removed."
            )

        optimizer_class = optimizer_data.get("optimizer_class")
        if not isinstance(optimizer_class, str) or not optimizer_class.strip():
            raise ValueError("training.optimizer.optimizer_class must be a non-empty string")

        optimizer_parameters = optimizer_data.get("parameters", {})
        if not isinstance(optimizer_parameters, dict):
            raise ValueError("training.optimizer.parameters must be an object")
    else:
        optimizer_class = "adamw"
        optimizer_parameters = {}
        if isinstance(optimizer_data, dict):
            cfg_optimizer_class = optimizer_data.get("optimizer_class")
            if isinstance(cfg_optimizer_class, str) and cfg_optimizer_class.strip():
                optimizer_class = cfg_optimizer_class
            cfg_optimizer_params = optimizer_data.get("parameters", {})
            if isinstance(cfg_optimizer_params, dict):
                optimizer_parameters = cfg_optimizer_params

    training_fields = _field_names(TrainingConfig)
    training_kwargs = {k: v for k, v in training_data.items() if k in training_fields}
    training_kwargs["optimizer_class"] = optimizer_class.strip().lower()
    training_kwargs["optimizer_parameters"] = optimizer_parameters
    training_config = TrainingConfig(**training_kwargs)

    runtime_keys = set(training_data.keys()) - training_fields - {"optimizer"}
    training_runtime = {k: training_data[k] for k in runtime_keys}

    return LoadedTrainingConfig(
        task=task,
        model_class=model_class,
        model_config=model_config,
        base_model=base_model,
        tokenizer_config=tokenizer_config,
        training_config=training_config,
        training_runtime=training_runtime,
    )


def list_config_paths(config_dir: str) -> Dict[str, str]:
    """List available YAML configuration files in a directory.

    Args:
        config_dir: Path to the directory containing ``.yaml``/``.yml`` files.

    Returns:
        Dictionary mapping config names (filename without extension) to
        their absolute file paths. Returns an empty dict if the directory
        does not exist.
    """
    configs = {}
    if not os.path.isdir(config_dir):
        return configs

    for filename in sorted(os.listdir(config_dir)):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            name = os.path.splitext(filename)[0]
            configs[name] = os.path.join(config_dir, filename)
    return configs
