"""Export Frankenstein checkpoints to HuggingFace Transformers format.

Generates a self-contained directory with ``config.json``,
``pytorch_model.bin``, custom ``configuration_frankestein.py`` and
``modeling_frankestein.py`` wrappers, and the model source tree, enabling
loading via ``AutoModelForMaskedLM`` or ``AutoModelForCausalLM`` with
``trust_remote_code=True``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml

try:
    from .quantization import bake_bitnet_weights
    from ..model.attention.common import BitLinear
except ImportError:
    # transformers_export may be invoked from a flat sys.path context; the
    # helpers are optional for the compatibility-check path.
    bake_bitnet_weights = None  # type: ignore
    BitLinear = None  # type: ignore


_CONFIGURATION_FILE = """from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict

from transformers import PretrainedConfig

from .model.tormented_bert_frankestein import UltraConfig


_ULTRA_KEYS = {field.name for field in fields(UltraConfig)}


class FrankesteinConfig(PretrainedConfig):
    model_type = "frankestein"

    def __init__(
        self,
        model_class: str = "frankenstein",
        task: str = "mlm",
        ultra_config: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        ultra = dict(ultra_config or {})
        passthrough: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in _ULTRA_KEYS and key not in ultra:
                ultra[key] = value
            else:
                passthrough[key] = value

        self.model_class = model_class
        self.task = task
        self.ultra_config = ultra

        for key, value in ultra.items():
            setattr(self, key, value)

        super().__init__(**passthrough)

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload["model_class"] = self.model_class
        payload["task"] = self.task
        payload["ultra_config"] = dict(self.ultra_config)
        payload.update(self.ultra_config)
        return payload
"""


_MODELING_FILE = """from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput, MaskedLMOutput

from .configuration_frankestein import FrankesteinConfig
from .model.tormented_bert_frankestein import (
    FrankensteinDecoder,
    TormentedBertFrankenstein,
    TormentedBertMini,
    UltraConfig,
)


_ULTRA_KEYS = {field.name for field in fields(UltraConfig)}


def _to_ultra_config(config: FrankesteinConfig) -> UltraConfig:
    source = dict(getattr(config, "ultra_config", {}) or {})
    for key in _ULTRA_KEYS:
        if key not in source and hasattr(config, key):
            source[key] = getattr(config, key)
    return UltraConfig(**source)


def _build_core_model(config: FrankesteinConfig) -> nn.Module:
    ultra = _to_ultra_config(config)
    model_class = str(getattr(config, "model_class", "frankenstein")).lower()
    if model_class == "mini":
        return TormentedBertMini(ultra)
    if model_class == "frankesteindecoder":
        if ultra.mode != "decoder":
            ultra.mode = "decoder"
        return FrankensteinDecoder(ultra)
    return TormentedBertFrankenstein(ultra)


class FrankesteinPreTrainedModel(PreTrainedModel):
    config_class = FrankesteinConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False


class FrankesteinForMaskedLM(FrankesteinPreTrainedModel):
    def __init__(self, config: FrankesteinConfig) -> None:
        super().__init__(config)
        self.model = _build_core_model(config)
        self.post_init()

    def get_input_embeddings(self):
        return getattr(self.model, "emb", None) or getattr(getattr(self.model, "backbone", None), "emb", None)

    def set_input_embeddings(self, value):
        if hasattr(self.model, "emb"):
            self.model.emb = value
        elif hasattr(self.model, "backbone") and hasattr(self.model.backbone, "emb"):
            self.model.backbone.emb = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: Any,
    ) -> MaskedLMOutput:
        if input_ids is None:
            raise ValueError("input_ids is required")
        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return MaskedLMOutput(loss=loss, logits=logits)


class FrankesteinForCausalLM(FrankesteinPreTrainedModel):
    def __init__(self, config: FrankesteinConfig) -> None:
        super().__init__(config)
        self.model = _build_core_model(config)
        self.post_init()

    def get_input_embeddings(self):
        return getattr(self.model, "emb", None) or getattr(getattr(self.model, "backbone", None), "emb", None)

    def set_input_embeddings(self, value):
        if hasattr(self.model, "emb"):
            self.model.emb = value
        elif hasattr(self.model, "backbone") and hasattr(self.model.backbone, "emb"):
            self.model.backbone.emb = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: Any,
    ) -> CausalLMOutput:
        if input_ids is None:
            raise ValueError("input_ids is required")
        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(loss=loss, logits=logits)
"""


_INIT_FILE = """from .configuration_frankestein import FrankesteinConfig
from .modeling_frankestein import FrankesteinForCausalLM, FrankesteinForMaskedLM

__all__ = [
    "FrankesteinConfig",
    "FrankesteinForMaskedLM",
    "FrankesteinForCausalLM",
]
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _as_plain_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"Unsupported config payload type: {type(value)!r}")


def _extract_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]
    if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint  # already a raw state_dict
    raise ValueError("Unable to locate model weights in checkpoint")


def _bake_state_dict(
    model_config: Dict[str, Any],
    model_class: str,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Bake BitLinear weights in a state dict to faithful ternary values.

    Instantiates the model, loads ``state_dict``, applies
    :func:`bake_bitnet_weights`, and returns the resulting state dict along
    with the number of baked BitLinear layers.

    Args:
        model_config: Model configuration dictionary (the ``model`` block).
        model_class: Lower-cased model class string.
        state_dict: Source weight tensors.

    Returns:
        Tuple of ``(baked_state_dict, num_baked_layers)``.
    """
    from src.model.tormented_bert_frankestein import (
        FrankensteinDecoder,
        TormentedBertFrankenstein,
        TormentedBertMini,
        UltraConfig,
    )
    from src.utils.config_flatten import flatten_model_dict

    ultra = UltraConfig(**flatten_model_dict(model_config))
    mc = str(model_class).lower()
    if mc == "mini":
        model = TormentedBertMini(ultra)
    elif mc == "frankesteindecoder" or ultra.mode == "decoder":
        if ultra.mode != "decoder":
            ultra.mode = "decoder"
        model = FrankensteinDecoder(ultra)
    else:
        model = TormentedBertFrankenstein(ultra)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    baked = bake_bitnet_weights(model) if bake_bitnet_weights is not None else 0
    return model.state_dict(), baked


def _load_schema_layer_enum(schema_path: Path) -> List[str]:
    from src.utils.schema_loader import resolve_schema
    schema = resolve_schema(schema_path)
    return list(
        schema.get("properties", {})
        .get("model", {})
        .get("properties", {})
        .get("dims", {})
        .get("properties", {})
        .get("layer_pattern", {})
        .get("items", {})
        .get("enum", [])
    )


def _build_compatibility_report(
    yaml_data: Dict[str, Any],
    model_config: Dict[str, Any],
    schema_layer_types: List[str],
) -> Tuple[bool, Dict[str, Any]]:
    issues: List[str] = []
    warnings: List[str] = []

    base_model = yaml_data.get("base_model")
    if base_model:
        issues.append(
            "Configurations with `base_model` are not exportable by this command because the resulting "
            "checkpoint is not guaranteed to match the local Frankenstein architecture wrappers."
        )

    task = str((yaml_data.get("training", {}) or {}).get("task", "")).strip().lower()
    if task == "sbert":
        issues.append(
            "SBERT task is not currently exportable to AutoModelForMaskedLM/AutoModelForCausalLM wrappers. "
            "Use MLM/decoder checkpoints."
        )

    model_layers = list(model_config.get("layer_pattern") or [])
    unknown_layers = sorted(set(model_layers) - set(schema_layer_types))
    if unknown_layers:
        issues.append(
            "Unknown layer types (not in schema enum): " + ", ".join(unknown_layers)
        )

    eval_only_layers = {"fasa_attn", "sparge_attn"}
    used_eval_only = sorted(set(model_layers) & eval_only_layers)
    if used_eval_only:
        warnings.append(
            "Eval-only layer types present (inference export is valid, training with these is not): "
            + ", ".join(used_eval_only)
        )

    use_bitnet = bool(model_config.get("use_bitnet", False))
    bitnet_routers = bool(model_config.get("bitnet_routers", False))
    if use_bitnet and not bitnet_routers:
        warnings.append(
            "use_bitnet is enabled but bitnet_routers is false: routing/scoring "
            "projections (MoE router, Mixture-of-Depths router, sparse "
            "block-index/forecast, top-k score nets) remain full-precision in "
            "the exported checkpoint. This is the recommended default for "
            "routing stability."
        )

    return (len(issues) == 0), {
        "is_compatible": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "schema_layer_types": schema_layer_types,
        "used_layer_types": sorted(set(model_layers)),
        "use_bitnet": use_bitnet,
        "bitnet_routers": bitnet_routers,
    }


def _build_transformers_config(
    model_config: Dict[str, Any],
    model_class: str,
    task: str,
    modeling_class: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = dict(model_config)
    payload["model_type"] = "frankestein"
    payload["architectures"] = [modeling_class]
    payload["model_class"] = model_class
    payload["task"] = task
    payload["ultra_config"] = dict(model_config)
    payload["auto_map"] = {
        "AutoConfig": "configuration_frankestein.FrankesteinConfig",
        "AutoModelForMaskedLM": "modeling_frankestein.FrankesteinForMaskedLM",
        "AutoModelForCausalLM": "modeling_frankestein.FrankesteinForCausalLM",
    }
    if bool(model_config.get("use_bitnet", False)):
        payload["quantization_config"] = {
            "quant_method": "bitnet",
            "bits": 1.58,
            "ternary_weights": True,
            "activation_bits": 8,
            "bitnet_routers": bool(model_config.get("bitnet_routers", False)),
            "modules_to_not_convert": None,
        }
    return payload


def check_yaml_export_compatibility(yaml_path: str) -> Dict[str, Any]:
    """Check whether a training YAML is compatible with Transformers export.

    Validates that the YAML does not use ``base_model`` or ``sbert`` task,
    and that all layer types in ``layer_pattern`` are known schema values.

    Args:
        yaml_path: Path to the training YAML file.

    Returns:
        Dictionary with ``is_compatible`` (bool), ``issues`` (list of str),
        ``warnings`` (list of str), ``schema_layer_types``, and
        ``used_layer_types``.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    yaml_file = Path(yaml_path).expanduser().resolve()
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_file}")

    with yaml_file.open("r", encoding="utf-8") as handle:
        yaml_data = yaml.safe_load(handle) or {}

    model_config = _as_plain_dict(yaml_data.get("model"))
    if not model_config:
        compatibility = {
            "is_compatible": False,
            "issues": [
                "YAML must include a `model` block for compatibility pre-check."
            ],
            "warnings": [],
            "schema_layer_types": [],
            "used_layer_types": [],
        }
        return compatibility

    schema_layer_types = _load_schema_layer_enum(_repo_root() / "src" / "schema.yaml")
    _, compatibility = _build_compatibility_report(
        yaml_data=yaml_data,
        model_config=model_config,
        schema_layer_types=schema_layer_types,
    )
    return compatibility


def export_transformers_model(model_path: str, yaml_path: str, output_dir: str) -> Dict[str, Any]:
    """Export a checkpoint and training YAML to a HuggingFace Transformers folder.

    Produces a self-contained directory with ``config.json``,
    ``pytorch_model.bin``, custom configuration/modeling Python wrappers,
    the model source tree, and a compatibility report.

    Args:
        model_path: Path to the training checkpoint (``.pt``).
        yaml_path: Path to the YAML config used during training.
        output_dir: Directory to write the exported model into.

    Returns:
        Dictionary with ``status`` (``"ok"`` or ``"incompatible"``),
        ``output_dir``, ``modeling_class``, and ``compatibility_report``.

    Raises:
        FileNotFoundError: If model or YAML file does not exist.
        ValueError: If the checkpoint is not a dict or model config
            cannot be inferred.
    """
    model_file = Path(model_path).expanduser().resolve()
    yaml_file = Path(yaml_path).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_file}")

    with yaml_file.open("r", encoding="utf-8") as handle:
        yaml_data = yaml.safe_load(handle) or {}

    checkpoint = torch.load(str(model_file), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dict-like payload")

    state_dict = _extract_state_dict(checkpoint)
    model_config = _as_plain_dict(yaml_data.get("model"))
    if not model_config:
        model_config = _as_plain_dict(checkpoint.get("config"))
    if not model_config:
        raise ValueError("Unable to infer model config from YAML `model` section or checkpoint `config`")

    model_class = str(yaml_data.get("model_class") or "frankenstein").strip().lower()
    if model_class == "frankesteindecoder":
        model_config["mode"] = "decoder"

    training_block = yaml_data.get("training", {}) or {}
    task = str(training_block.get("task") or "mlm").strip().lower()

    compatibility = check_yaml_export_compatibility(str(yaml_file))
    is_compatible = bool(compatibility.get("is_compatible", False))

    output_path.mkdir(parents=True, exist_ok=True)
    if not is_compatible:
        report_path = output_path / "compatibility_report.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(compatibility, handle, indent=2)
        return {
            "status": "incompatible",
            "output_dir": str(output_path),
            "compatibility_report": str(report_path),
            "issues": compatibility["issues"],
        }

    modeling_class = (
        "FrankesteinForCausalLM"
        if model_config.get("mode") == "decoder" or model_class == "frankesteindecoder"
        else "FrankesteinForMaskedLM"
    )
    hf_config = _build_transformers_config(
        model_config=model_config,
        model_class=model_class,
        task=task,
        modeling_class=modeling_class,
    )

    (output_path / "config.json").write_text(
        json.dumps(hf_config, indent=2),
        encoding="utf-8",
    )

    export_state_dict = state_dict
    baked_layers = 0
    if bool(model_config.get("use_bitnet", False)) and bake_bitnet_weights is not None:
        # Instantiate the model so BitLinear layers can be baked to faithful
        # {-1, 0, 1} * scale weights before serialization. The exported
        # pytorch_model.bin then carries self-describing ternary values.
        try:
            export_state_dict, baked_layers = _bake_state_dict(
                model_config=model_config,
                model_class=model_class,
                state_dict=state_dict,
            )
        except Exception as exc:  # pragma: no cover - defensive
            warnings_log = compatibility.get("warnings", [])
            warnings_log.append(
                f"BitNet bake skipped (model instantiation failed): {exc}"
            )
            compatibility["warnings"] = warnings_log

    torch.save(export_state_dict, output_path / "pytorch_model.bin")

    model_src = _repo_root() / "src" / "model"
    shutil.copytree(model_src, output_path / "model", dirs_exist_ok=True)

    (output_path / "configuration_frankestein.py").write_text(_CONFIGURATION_FILE, encoding="utf-8")
    (output_path / "modeling_frankestein.py").write_text(_MODELING_FILE, encoding="utf-8")
    (output_path / "__init__.py").write_text(_INIT_FILE, encoding="utf-8")
    (output_path / "compatibility_report.json").write_text(
        json.dumps(compatibility, indent=2),
        encoding="utf-8",
    )

    usage = {
        "python_example": [
            "from transformers import AutoConfig, AutoModelForMaskedLM, AutoModelForCausalLM",
            "cfg = AutoConfig.from_pretrained('<model-dir>', trust_remote_code=True)",
            "model = AutoModelForMaskedLM.from_pretrained('<model-dir>', trust_remote_code=True)",
            "# or AutoModelForCausalLM.from_pretrained('<model-dir>', trust_remote_code=True)",
        ]
    }
    (output_path / "export_info.json").write_text(json.dumps(usage, indent=2), encoding="utf-8")

    return {
        "status": "ok",
        "output_dir": str(output_path),
        "modeling_class": modeling_class,
        "compatibility_report": str(output_path / "compatibility_report.json"),
        "bitnet": {
            "enabled": bool(model_config.get("use_bitnet", False)),
            "bitnet_routers": bool(model_config.get("bitnet_routers", False)),
            "baked_layers": baked_layers,
        },
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export Frankenstein checkpoint + training YAML to a Hugging Face Transformers-compatible "
            "folder (custom code with trust_remote_code=True)."
        )
    )
    parser.add_argument("--model", required=True, help="Path to training checkpoint (*.pt)")
    parser.add_argument("--yaml", required=True, help="Path to YAML used during training")
    parser.add_argument("--output", required=True, help="Output directory for exported model")
    args = parser.parse_args(argv)

    result = export_transformers_model(
        model_path=args.model,
        yaml_path=args.yaml,
        output_dir=args.output,
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
