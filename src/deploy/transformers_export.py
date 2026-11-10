"""Export Frankenstein checkpoints to HuggingFace Transformers format.

Generates a self-contained directory with ``config.json``,
``model.safetensors``, custom ``configuration_frankenstein.py`` and
``modeling_frankenstein.py`` wrappers, a ``quantization_frankenstein.py``
codec module, a ``load_frankenstein.py`` loader, and the model source tree.
Load with ``load_frankenstein.load_pretrained(<dir>)``: the nested ``model/``
package is not copied by HF's ``trust_remote_code`` dynamic-module loader.

Quantization targets only the safetensors artifact: ``q8_0``/``q4_0`` use
32-weight affine blocks and ``i2_s`` uses BitNet ternary packing.
Two load-time runtimes are supported:

* ``dequantize_on_load`` (v1): packed blocks are expanded to fp32 while the
  artifact loads; the runtime model is a normal Frankenstein model.
* ``quantized_linear`` (v2): every quantized ``nn.Linear`` is replaced by a
  :class:`PackedLinear` that keeps the packed buffers and dequantizes inside
  ``forward``.

No ``pytorch_model.bin``, bitsandbytes or other quantization runtime is
produced by this exporter.
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
    from ..utils.config_flatten import flatten_model_dict
except ImportError:
    # transformers_export may be invoked from a flat sys.path context; the
    # helpers are optional for the compatibility-check path.
    bake_bitnet_weights = None  # type: ignore
    BitLinear = None  # type: ignore
    from utils.config_flatten import flatten_model_dict  # type: ignore

try:
    from .safetensors_quant import (
        BLOCK_SIZE,
        I2_S,
        PACKED_SUFFIX,
        Q4_0,
        Q8_0,
        SCALE_SUFFIX,
        SUPPORTED_FORMATS,
        bits_per_weight,
        quantize_weight,
    )
except ImportError:  # flat sys.path context
    from safetensors_quant import (  # type: ignore
        BLOCK_SIZE,
        I2_S,
        PACKED_SUFFIX,
        Q4_0,
        Q8_0,
        SCALE_SUFFIX,
        SUPPORTED_FORMATS,
        bits_per_weight,
        quantize_weight,
    )

#: Quantization formats accepted by :func:`export_transformers_model`.
QUANT_FORMATS = ("none",) + tuple(SUPPORTED_FORMATS)
#: Load-time quantization runtimes.
QUANT_RUNTIMES = ("dequantize_on_load", "quantized_linear")
SAFETENSORS_NAME = "model.safetensors"
QUANT_MODULE_NAME = "quantization_frankenstein.py"


_CONFIGURATION_FILE = """from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict

from transformers import PretrainedConfig

from .model.config import FrankensteinModelConfig


_MODEL_KEYS = {field.name for field in fields(FrankensteinModelConfig)}


class FrankensteinConfig(PretrainedConfig):
    model_type = "frankenstein"

    def __init__(
        self,
        model_class: str = "frankenstein_text_encoder",
        task: str = "mlm",
        frankenstein_config: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        model_cfg = dict(frankenstein_config or {})
        passthrough: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if key == "num_labels" and value is None:
                # `num_labels` is a reserved HF property. transformers 5.x
                # keeps an explicit ``None`` (its ``__post_init__`` resolves
                # ``kwargs.get("num_labels", ...)`` to None and then calls
                # ``range(None)``), unlike 4.x which coerces None to 2. Treat
                # it as unset so artifacts exported before this guard still
                # load; the model-side value lives in ``frankenstein_config``.
                continue
            if key in _MODEL_KEYS:
                # Model keys never reach ``PretrainedConfig.__init__``: flat
                # names can collide with reserved HF config properties whose
                # kwargs are validated (``rope_scaling`` must be a dict in
                # transformers >= 5, while the Frankenstein config stores a
                # float factor). ``frankenstein_config`` stays authoritative.
                model_cfg.setdefault(key, value)
            else:
                passthrough[key] = value

        self.model_class = model_class
        self.task = task
        self.frankenstein_config = model_cfg

        super().__init__(**passthrough)

        for key, value in model_cfg.items():
            if key in {"model_class", "task", "model_type", "architectures"}:
                continue
            if key == "num_labels" and value is None:
                # `num_labels` is a reserved HF property; a None value breaks
                # its id2label/label2id map creation.
                continue
            try:
                setattr(self, key, value)
            except (AttributeError, TypeError, ValueError):
                # Some HF-reserved names reject the Frankenstein shape on set
                # (``rope_scaling`` in transformers >= 5); the value remains
                # available through ``frankenstein_config`` so the model still
                # builds from it.
                continue

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload["model_class"] = self.model_class
        payload["task"] = self.task
        payload["frankenstein_config"] = dict(self.frankenstein_config)
        payload.update(
            {
                key: value
                for key, value in self.frankenstein_config.items()
                if key not in {"model_class", "task", "model_type", "architectures"}
                and not (key == "num_labels" and value is None)
            }
        )
        return payload
"""


_MODELING_FILE = """from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput

from .configuration_frankenstein import FrankensteinConfig
from .model.config import FrankensteinModelConfig
from .model.frankenstein_decoder import FrankensteinDecoder
from .model.frankenstein_encoder import FrankensteinEncoder
from .quantization_frankenstein import (
    PACKED_SUFFIX,
    SCALE_SUFFIX,
    dequantize_weight,
    packed_numel,
)


_MODEL_KEYS = {field.name for field in fields(FrankensteinModelConfig)}


def _to_model_config(config: FrankensteinConfig) -> FrankensteinModelConfig:
    source = dict(getattr(config, "frankenstein_config", {}) or {})
    for key in _MODEL_KEYS:
        if key not in source and hasattr(config, key):
            source[key] = getattr(config, key)
    return FrankensteinModelConfig(**source)


def _build_core_model(config: FrankensteinConfig) -> nn.Module:
    model_cfg = _to_model_config(config)
    model_class = str(getattr(config, "model_class", "frankenstein_text_encoder")).lower()
    if model_class == "frankenstein_text_decoder":
        if model_cfg.mode != "decoder":
            model_cfg.mode = "decoder"
        return FrankensteinDecoder(model_cfg)
    if model_class == "frankenstein_vit":
        from .model.frankenstein_vit import FrankensteinViT
        return FrankensteinViT(model_cfg)
    return FrankensteinEncoder(model_cfg)


def _quantization_config(config: FrankensteinConfig) -> Dict[str, Any]:
    payload = getattr(config, "frankenstein_quantization", None)
    if isinstance(payload, dict) and payload.get("runtime"):
        return dict(payload)
    return {}


def _resolve_safetensors_file(source: str) -> str:
    path = Path(source)
    if path.is_dir():
        candidate = path / "model.safetensors"
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"No model.safetensors in {path}")
    if path.exists():
        return str(path)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(source, "model.safetensors")


def _dequantize_state_dict(
    state_dict: Dict[str, torch.Tensor],
    quantization_config: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    tensors = quantization_config.get("tensors") or {}
    result = {
        name: tensor
        for name, tensor in state_dict.items()
        if not name.endswith(PACKED_SUFFIX) and not name.endswith(SCALE_SUFFIX)
    }
    for name, metadata in tensors.items():
        module_name = name[: -len(".weight")] if name.endswith(".weight") else name
        packed = state_dict.get(module_name + PACKED_SUFFIX)
        if packed is None:
            continue
        result[name] = dequantize_weight(
            packed,
            state_dict.get(module_name + SCALE_SUFFIX),
            metadata["shape"],
            metadata["format"],
        )
    return result


class PackedLinear(nn.Module):
    # nn.Linear whose weights stay packed and dequantize in forward().

    def __init__(self, base: nn.Linear, fmt: str) -> None:
        super().__init__()
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.bias = base.bias
        self._format = str(fmt)
        self._shape = (self.out_features, self.in_features)
        # BitLinear also normalizes and int8-quantizes activations; preserve
        # that runtime behavior when the packed layer replaces one.
        self._bitlinear = getattr(base, "bake_ternary_weights", None) is not None
        self.register_buffer(
            "__packed__",
            torch.empty(packed_numel(self._shape, self._format), dtype=torch.uint8),
        )
        if self._format == "i2_s":
            self.register_buffer("__scale__", torch.zeros((), dtype=torch.float32))

    @property
    def weight(self) -> torch.Tensor:
        return dequantize_weight(
            self.__packed__,
            getattr(self, "__scale__", None),
            self._shape,
            self._format,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(dtype=x.dtype)
        if not self._bitlinear:
            return F.linear(x, weight, self.bias)
        x_norm = F.layer_norm(x, (x.shape[-1],))
        scale = 127.0 / x_norm.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_q = (x_norm * scale).round().clamp(-128, 127) / scale
        return F.linear(x_q, weight, self.bias)


def _apply_packed_linears(model: nn.Module, quantization_config: Dict[str, Any]) -> None:
    if str(quantization_config.get("runtime") or "") != "quantized_linear":
        return
    for name, metadata in (quantization_config.get("tensors") or {}).items():
        if not name.endswith(".weight"):
            continue
        core_name = name[6:] if name.startswith("model.") else name
        module_path = core_name[: -len(".weight")]
        parent_path, _, child = module_path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        base = getattr(parent, child, None)
        if not isinstance(base, nn.Linear):
            continue
        setattr(parent, child, PackedLinear(base, metadata["format"]))


class FrankensteinPreTrainedModel(PreTrainedModel):
    config_class = FrankensteinConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.get("config")
        if config is None:
            config = FrankensteinConfig.from_pretrained(pretrained_model_name_or_path)
            kwargs["config"] = config
        quantization_config = _quantization_config(config)
        runtime = str(quantization_config.get("runtime") or "")
        if runtime != "dequantize_on_load":
            return super().from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        state_dict = _dequantize_state_dict(
            load_file(_resolve_safetensors_file(str(pretrained_model_name_or_path))),
            quantization_config,
        )
        model = cls(config)
        model.load_state_dict(state_dict, strict=False)
        dtype = kwargs.get("torch_dtype") or kwargs.get("dtype")
        if dtype is not None:
            model.to(dtype=dtype)
        device = kwargs.get("device")
        if isinstance(device, str) and device not in ("auto", "disk"):
            model.to(device)
        return model.eval()


class FrankensteinForMaskedLM(FrankensteinPreTrainedModel):
    def __init__(self, config: FrankensteinConfig) -> None:
        super().__init__(config)
        self.model = _build_core_model(config)
        _apply_packed_linears(self.model, _quantization_config(config))
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


class FrankensteinForCausalLM(FrankensteinPreTrainedModel):
    def __init__(self, config: FrankensteinConfig) -> None:
        super().__init__(config)
        self.model = _build_core_model(config)
        _apply_packed_linears(self.model, _quantization_config(config))
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
        past_key_values: Any | None = None,
        use_cache: bool = False,
        cache_position: int | None = None,
        return_dict: bool = True,
        **_: Any,
    ) -> CausalLMOutputWithPast:
        if input_ids is None:
            raise ValueError("input_ids is required")
        # Real KV-cache support (plan blocker B7): the custom
        # FrankensteinCache is threaded through the decoder's prefill +
        # decode path instead of being silently discarded.
        if use_cache:
            logits, past_key_values = self.model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
                return_cache=True,
            )
        else:
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
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # HuggingFace ``generate`` compatibility: feed only the last token
        # once a cache exists (prefill -> decode).
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    def _reorder_cache(self, past_key_values: Any, beam_idx: torch.Tensor) -> Any:
        # Batch/beam reordering is a no-op for the v1 single-sequence cache.
        return past_key_values
"""


_INIT_FILE = """from .configuration_frankenstein import FrankensteinConfig
from .modeling_frankenstein import FrankensteinForCausalLM, FrankensteinForMaskedLM

__all__ = [
    "FrankensteinConfig",
    "FrankensteinForMaskedLM",
    "FrankensteinForCausalLM",
]
"""


_LOADER_FILE = '''"""Load this exported Frankenstein model without HF dynamic-module copying.

``transformers``' ``trust_remote_code`` loader only copies sibling ``*.py``
files, not the nested ``model/`` package this artifact ships. Import the
export directory directly as a package instead::

    from load_frankenstein import load_pretrained
    model = load_pretrained("<this-directory>")
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_pretrained(export_dir=None, **kwargs):
    """Load the exported wrapper class and its ``model.safetensors`` weights.

    Args:
        export_dir: Export directory; defaults to this file's directory.
        **kwargs: Forwarded to the wrapper's ``from_pretrained``.

    Returns:
        ``FrankensteinForMaskedLM`` or ``FrankensteinForCausalLM`` in eval mode.
    """
    if export_dir is None:
        export_dir = Path(__file__).resolve().parent
    else:
        export_dir = Path(export_dir).resolve()
    module_name = "frankenstein_export"
    for key in [
        k for k in sys.modules
        if k == module_name or k.startswith(module_name + ".")
    ]:
        del sys.modules[key]
    spec = importlib.util.spec_from_file_location(
        module_name,
        export_dir / "__init__.py",
        submodule_search_locations=[str(export_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import the export package from {export_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    config = module.FrankensteinConfig.from_pretrained(str(export_dir))
    wrapper = (
        module.FrankensteinForCausalLM
        if str(getattr(config, "model_class", "")).lower()
        == "frankenstein_text_decoder"
        else module.FrankensteinForMaskedLM
    )
    return wrapper.from_pretrained(str(export_dir), **kwargs)
'''


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
    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_decoder import FrankensteinDecoder
    from src.model.frankenstein_encoder import FrankensteinEncoder
    from src.utils.config_flatten import flatten_model_dict

    model_cfg = FrankensteinModelConfig(**flatten_model_dict(model_config))
    mc = str(model_class).lower()
    if mc == "frankenstein_text_decoder" or model_cfg.mode == "decoder":
        if model_cfg.mode != "decoder":
            model_cfg.mode = "decoder"
        model = FrankensteinDecoder(model_cfg)
    else:
        model = FrankensteinEncoder(model_cfg)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    baked = bake_bitnet_weights(model) if bake_bitnet_weights is not None else 0
    return model.state_dict(), baked


def _dynamic_embedding_enabled(model_config: Dict[str, Any]) -> bool:
    """Whether a (nested or flat) model config uses dynamic embeddings.

    Args:
        model_config: Nested ``model:`` block or flat checkpoint config.

    Returns:
        ``True`` when ``embedding.dynamic.enabled`` (nested) or
        ``use_dynamic_embedding`` (flat) is set.
    """
    if bool(model_config.get("use_dynamic_embedding", False)):
        return True
    embedding = model_config.get("embedding")
    if isinstance(embedding, dict):
        dynamic = embedding.get("dynamic")
        if isinstance(dynamic, dict) and bool(dynamic.get("enabled", False)):
            return True
    return False


def _resolve_dynamic_vocab(
    model_config: Dict[str, Any],
    checkpoint: Dict[str, Any],
    yaml_data: Dict[str, Any],
) -> List[str]:
    """Resolve the tokenizer vocabulary required by dynamic embeddings.

    Lookup order: flat config ``token_vocab`` → checkpoint ``config`` →
    ``tokenizer.name_or_path`` in the training YAML.

    Args:
        model_config: Model config resolved from the YAML or checkpoint.
        checkpoint: Loaded checkpoint payload.
        yaml_data: Parsed training YAML.

    Returns:
        Id-ordered token strings.

    Raises:
        ValueError: If no vocabulary can be resolved.
    """
    from src.utils.config_flatten import flatten_model_dict

    vocab = flatten_model_dict(model_config).get("token_vocab")
    if vocab:
        return list(vocab)

    checkpoint_config = _as_plain_dict(checkpoint.get("config"))
    vocab = checkpoint_config.get("token_vocab")
    if vocab:
        return list(vocab)

    tokenizer_cfg = yaml_data.get("tokenizer") or {}
    name_or_path = str(tokenizer_cfg.get("name_or_path") or "").strip()
    if name_or_path:
        try:
            from transformers import AutoTokenizer
            try:
                from ..utils.tokenizer_vocab import token_strings
            except ImportError:  # flat sys.path context
                from utils.tokenizer_vocab import token_strings

            tokenizer = AutoTokenizer.from_pretrained(
                name_or_path,
                use_fast=bool(tokenizer_cfg.get("use_fast", True)),
                trust_remote_code=bool(tokenizer_cfg.get("trust_remote_code", False)),
            )
            strings = token_strings(tokenizer)
            if strings:
                return strings
        except Exception as exc:  # noqa: BLE001 - fall through to the error below
            raise ValueError(
                "embedding.dynamic.enabled: could not load the tokenizer from "
                f"tokenizer.name_or_path={name_or_path!r} to materialize the "
                f"frozen embedding table: {exc}"
            ) from exc

    raise ValueError(
        "embedding.dynamic.enabled: the export needs the tokenizer vocabulary to "
        "materialize the frozen embedding table. Save the checkpoint with the "
        "training engine (which records token_vocab) or set tokenizer.name_or_path "
        "in the YAML."
    )


def _materialize_dynamic_embedding(
    model_config: Dict[str, Any],
    model_class: str,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], int]:
    """Swap an EELBERT hash embedding for a frozen lookup table for export.

    The HuggingFace artifact carries the computed ``vocab_size × hidden_size``
    table as a regular (frozen) ``nn.Embedding``, so the exported model runs
    without the custom hash code. The recorded config disables dynamic
    embeddings so the exported wrapper rebuilds a plain-table model.

    Args:
        model_config: Flat config including ``token_vocab``.
        model_class: Model class string.
        state_dict: Training checkpoint weights (no embedding table present).

    Returns:
        Tuple of ``(export_state_dict, export_config, baked_layers)``.
    """
    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_decoder import FrankensteinDecoder
    from src.model.frankenstein_encoder import FrankensteinEncoder
    from src.utils.config_flatten import flatten_model_dict

    flat = flatten_model_dict(model_config)
    model_cfg = FrankensteinModelConfig(**flat)
    decoder = str(model_class).lower() == "frankenstein_text_decoder" or model_cfg.mode == "decoder"
    if decoder:
        model_cfg.mode = "decoder"
        model = FrankensteinDecoder(model_cfg)
        embedding = model.backbone.emb
    else:
        model = FrankensteinEncoder(model_cfg)
        embedding = model.emb

    model.load_state_dict(state_dict, strict=False)
    table = embedding.embedding_table().detach().clone()
    frozen = torch.nn.Embedding.from_pretrained(table, freeze=True)
    if decoder:
        model.backbone.emb = frozen
    else:
        model.emb = frozen
    model.eval()

    baked = 0
    if bake_bitnet_weights is not None and bool(flat.get("use_bitnet", False)):
        baked = bake_bitnet_weights(model)

    exported = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    export_config = dict(flat)
    export_config["use_dynamic_embedding"] = False
    export_config["dynamic_embedding_precompute"] = False
    export_config["use_factorized_embedding"] = False
    export_config["token_vocab"] = None
    return exported, export_config, baked


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

    model_class = str(yaml_data.get("model_class") or "").strip().lower()
    if model_class == "frankenstein_audio":
        issues.append(
            "Audio models (`frankenstein_audio`) are not currently exportable to "
            "AutoModelForMaskedLM/AutoModelForCausalLM wrappers: the waveform frontend and the "
            "ASR/audio-classification heads have no Transformers equivalent. "
            "ASR/audio Transformers export is future work — use the native "
            "checkpoint or the deployment pipeline instead."
        )

    model_layers = list(model_config.get("layer_pattern") or [])
    unknown_layers = sorted(set(model_layers) - set(schema_layer_types))
    if unknown_layers:
        issues.append(
            "Unknown layer types (not in schema enum): " + ", ".join(unknown_layers)
        )

    inference_sparse_layers = {"fasa", "sparge"}
    used_sparse_layers = sorted(set(model_layers) & inference_sparse_layers)
    if used_sparse_layers:
        warnings.append(
            "Inference-sparse layer types present ("
            + ", ".join(used_sparse_layers)
            + "): these train as standard dense attention over their own "
            "projections and apply the sparse heuristic only in eval/inference; "
            "the exported checkpoint keeps the sparse eval path."
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
    quantization_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = dict(model_config)
    if payload.get("num_labels") is None:
        # `num_labels` is a reserved HF property. transformers 5.x takes an
        # explicit ``None`` literally and calls ``range(None)`` while building
        # the id2label map, so omit the key and let the default apply. The
        # model-side value stays in ``frankenstein_config`` below.
        payload.pop("num_labels", None)
    payload["model_type"] = "frankenstein"
    payload["architectures"] = [modeling_class]
    payload["model_class"] = model_class
    payload["task"] = task
    # `frankenstein_config` must be flat: the generated wrapper calls
    # FrankensteinModelConfig(**frankenstein_config) and the schema runtime
    # config is the flat dataclass.
    payload["frankenstein_config"] = flatten_model_dict(model_config)
    payload["auto_map"] = {
        "AutoConfig": "configuration_frankenstein.FrankensteinConfig",
        "AutoModelForMaskedLM": "modeling_frankenstein.FrankensteinForMaskedLM",
        "AutoModelForCausalLM": "modeling_frankenstein.FrankensteinForCausalLM",
    }
    # HF-compatible Mixture-of-Experts block (DeepSeek/Qwen-style names).
    if bool(model_config.get("use_moe", False)):
        top_k = int(model_config.get("top_k_experts", 2))
        bias_update = str(model_config.get("moe_bias_update", "none"))
        payload["num_experts"] = int(model_config.get("num_experts", 8))
        payload["num_experts_per_tok"] = top_k
        payload["moe"] = {
            "enabled": True,
            "num_experts": int(model_config.get("num_experts", 8)),
            "num_experts_per_tok": top_k,
            "moe_intermediate_size": model_config.get("moe_expert_hidden_size")
            or model_config.get("ffn_hidden_size"),
            "scoring_func": str(model_config.get("moe_router_scoring", "softmax")),
            "norm_topk_prob": bool(model_config.get("moe_router_renormalize", True)),
            "pre_softmax": bool(model_config.get("moe_router_pre_softmax", False)),
            "routed_scaling_factor": float(
                model_config.get("moe_routed_scaling_factor", 1.0)
            ),
            "router_dtype": str(model_config.get("moe_router_dtype", "float32")),
            "z_loss_coeff": float(model_config.get("moe_router_z_loss_coeff", 0.0)),
            "n_shared_experts": int(
                model_config.get("moe_shared_experts_count", 0)
            ),
            "shared_expert_intermediate_size": model_config.get(
                "moe_shared_expert_hidden_size"
            ),
            "shared_expert_gate": str(
                model_config.get("moe_shared_expert_gate", "none")
            ),
            "first_k_dense_replace": int(model_config.get("moe_dense_layers", 0)),
            "moe_layer_freq": int(model_config.get("moe_layer_frequency", 1)),
            "routing_type": str(
                model_config.get("moe_routing_type", "token_choice")
            ),
            "capacity_factor": float(
                model_config.get("moe_capacity_factor", 0.0)
            ),
            "hash_layers": int(model_config.get("moe_hash_layers", 0)),
            "latent": {
                "enabled": bool(model_config.get("moe_latent_enabled", False)),
                "dim": model_config.get("moe_latent_dim"),
                "norm_before_up_proj": bool(
                    model_config.get("moe_latent_norm_before_up_proj", False)
                ),
            },
            "balancing": {
                "aux_loss": str(model_config.get("moe_aux_loss_type", "batch")),
                "aux_loss_coeff": float(
                    model_config.get("moe_aux_loss_coeff", 0.01)
                ),
                "bias_update": bias_update,
                "bias_update_rate": float(
                    model_config.get("moe_bias_update_rate", 0.001)
                ),
            },
            "topk_method": "noaux_tc" if bias_update != "none" else "greedy",
            "activation_clamp": float(
                model_config.get("moe_activation_clamp", 0.0)
            ),
        }
    if quantization_config is not None:
        # Dedicated key: HF reserves `quantization_config` for its own quant
        # runtimes (bitsandbytes/GPTQ/...) and warns/skips unknown schemes.
        payload["frankenstein_quantization"] = dict(quantization_config)
    elif bool(model_config.get("use_bitnet", False)):
        payload["quantization_config"] = {
            "quant_method": "bitnet",
            "bits": 1.58,
            "ternary_weights": True,
            "activation_bits": 8,
            "bitnet_routers": bool(model_config.get("bitnet_routers", False)),
            "modules_to_not_convert": None,
            "storage": "float32_baked",
            "compressed": False,
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


def _build_export_model(
    model_config: Dict[str, Any],
    model_class: str,
    state_dict: Dict[str, torch.Tensor],
):
    """Instantiate the core model that the exported wrapper will rebuild.

    Args:
        model_config: Flat or nested model configuration.
        model_class: Lower-cased model class string.
        state_dict: Core-model weights to load before quantization.

    Returns:
        The instantiated core model in eval mode.
    """
    from src.model.config import FrankensteinModelConfig
    from src.model.frankenstein_decoder import FrankensteinDecoder
    from src.model.frankenstein_encoder import FrankensteinEncoder
    from src.utils.config_flatten import flatten_model_dict

    model_cfg = FrankensteinModelConfig(**flatten_model_dict(model_config))
    if str(model_class).lower() == "frankenstein_text_decoder" or model_cfg.mode == "decoder":
        if model_cfg.mode != "decoder":
            model_cfg.mode = "decoder"
        model = FrankensteinDecoder(model_cfg)
    elif str(model_class).lower() == "frankenstein_vit":
        from src.model.frankenstein_vit import FrankensteinViT

        model = FrankensteinViT(model_cfg)
    else:
        model = FrankensteinEncoder(model_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _quantizable_linear_names(model: torch.nn.Module) -> List[str]:
    """Return module paths whose 2-D weights can be block-quantized.

    Only ``nn.Linear`` modules (including ``BitLinear``) whose last dimension
    is a multiple of :data:`BLOCK_SIZE` are eligible; embeddings, norms,
    biases, convolutions and odd-sized projections stay full precision.

    Args:
        model: Core model to traverse.

    Returns:
        Sorted module paths.
    """
    names: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        if weight.shape[-1] % BLOCK_SIZE != 0:
            continue
        names.append(name)
    return sorted(names)


def _quantize_export_state(
    state_dict: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    fmt: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, Any]]]:
    """Replace eligible weight tensors with packed safetensors payloads.

    Args:
        state_dict: Core-model state dict (unprefixed keys).
        model: Core model used to discover quantizable modules.
        fmt: ``q8_0``, ``q4_0`` or ``i2_s``.

    Returns:
        Tuple of ``(packed_state_dict, tensor_metadata)``. Metadata keys use
        the unprefixed weight name and record ``format`` and original
        ``shape``.
    """
    packed_state = dict(state_dict)
    metadata: Dict[str, Dict[str, Any]] = {}
    for module_name in _quantizable_linear_names(model):
        key = f"{module_name}.weight" if module_name else "weight"
        weight = packed_state.pop(key, None)
        if weight is None:
            continue
        payload = quantize_weight(weight, fmt)
        packed_state[module_name + PACKED_SUFFIX] = payload[PACKED_SUFFIX]
        if SCALE_SUFFIX in payload:
            packed_state[module_name + SCALE_SUFFIX] = payload[SCALE_SUFFIX]
        metadata[key] = {
            "format": fmt,
            "shape": [int(dim) for dim in weight.shape],
        }
    return packed_state, metadata


def _build_quantization_config(
    fmt: str,
    runtime: str,
    tensors: Dict[str, Dict[str, Any]],
    use_bitnet: bool,
) -> Dict[str, Any]:
    """Build the ``quantization_config`` block stored in ``config.json``.

    Args:
        fmt: Storage format (``q8_0``, ``q4_0`` or ``i2_s``).
        runtime: ``dequantize_on_load`` (v1) or ``quantized_linear`` (v2).
        tensors: Per-weight metadata keyed by the prefixed weight name.
        use_bitnet: Whether the source model was BitNet-baked.

    Returns:
        JSON-serializable quantization config.
    """
    return {
        "quant_method": "frankenstein_block",
        "format": fmt,
        "runtime": runtime,
        "block_size": BLOCK_SIZE,
        "bits_per_weight": bits_per_weight(fmt),
        "compressed": True,
        "bitnet_baked": bool(use_bitnet),
        "tensors": dict(tensors),
    }


def _save_safetensors(state_dict: Dict[str, torch.Tensor], path: Path) -> None:
    """Write a state dict as a single safetensors file.

    Args:
        state_dict: Tensor mapping to serialize.
        path: Destination ``*.safetensors`` file.
    """
    from safetensors.torch import save_file

    # safetensors forbids shared storage and non-contiguous tensors.
    clean = {
        name: tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in state_dict.items()
    }
    save_file(clean, str(path), metadata={"format": "pt"})


def export_transformers_model(
    model_path: str,
    yaml_path: str,
    output_dir: str,
    quant: str = "none",
    quant_runtime: str = "dequantize_on_load",
) -> Dict[str, Any]:
    """Export a checkpoint and training YAML to a HuggingFace Transformers folder.

    Produces a self-contained directory with ``config.json``,
    ``model.safetensors``, custom configuration/modeling Python wrappers, the
    ``quantization_frankenstein.py`` codec module, the model source tree and a
    compatibility report.

    Args:
        model_path: Path to the training checkpoint (``.pt``).
        yaml_path: Path to the YAML config used during training.
        output_dir: Directory to write the exported model into.
        quant: Storage format: ``none``, ``q8_0``, ``q4_0`` or ``i2_s``.
        quant_runtime: Load-time runtime: ``dequantize_on_load`` (v1) or
            ``quantized_linear`` (v2).

    Returns:
        Dictionary with ``status`` (``"ok"`` or ``"incompatible"``),
        ``output_dir``, ``modeling_class``, ``quantization`` summary and
        ``compatibility_report``.

    Raises:
        FileNotFoundError: If model or YAML file does not exist.
        ValueError: If the checkpoint is not a dict, model config cannot be
            inferred, or the quantization format/runtime is unknown.
    """
    quant = str(quant or "none").strip().lower()
    quant_runtime = str(quant_runtime or "dequantize_on_load").strip().lower()
    if quant not in QUANT_FORMATS:
        raise ValueError(
            f"Unknown quantization format {quant!r}; expected one of {QUANT_FORMATS}"
        )
    if quant_runtime not in QUANT_RUNTIMES:
        raise ValueError(
            f"Unknown quantization runtime {quant_runtime!r}; expected one of {QUANT_RUNTIMES}"
        )
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

    # The trained checkpoint is authoritative for runtime fields that the
    # tokenizer/training pipeline may have changed (e.g. vocab_size). Merge it
    # over the YAML model block so the exported wrapper rebuilds a model whose
    # shapes match the saved weights.
    checkpoint_config = flatten_model_dict(_as_plain_dict(checkpoint.get("config")))
    model_config = {**flatten_model_dict(model_config), **checkpoint_config}

    model_class = str(yaml_data.get("model_class") or "frankenstein_text_encoder").strip().lower()
    if model_class == "frankenstein_audio":
        raise ValueError(
            "frankenstein_audio is not currently exportable to "
            "AutoModelForMaskedLM/AutoModelForCausalLM wrappers: the waveform "
            "frontend and the ASR/audio-classification heads have no "
            "Transformers equivalent. ASR/audio Transformers export is future "
            "work — use the native checkpoint or the deployment pipeline."
        )
    if model_class == "frankenstein_text_decoder":
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
        "FrankensteinForCausalLM"
        if model_config.get("mode") == "decoder" or model_class == "frankenstein_text_decoder"
        else "FrankensteinForMaskedLM"
    )

    export_state_dict = state_dict
    baked_layers = 0
    if _dynamic_embedding_enabled(model_config):
        # Dynamic (EELBERT) hash embeddings have no lookup table and are not
        # expressible in the HF wrapper. Materialize the computed table as a
        # frozen nn.Embedding so the export runs without the custom hash code.
        model_config = dict(model_config)
        model_config["token_vocab"] = _resolve_dynamic_vocab(
            model_config, checkpoint, yaml_data
        )
        export_state_dict, model_config, baked_layers = _materialize_dynamic_embedding(
            model_config, model_class, state_dict
        )
        compatibility.setdefault("warnings", []).append(
            "Dynamic (EELBERT) embeddings were materialized as a frozen "
            "vocab_size x hidden_size table for the Transformers export; the "
            "size advantage of the hash embedding is not preserved in this "
            "artifact."
        )

    # Build the core model once so BitNet baking and block quantization see
    # the exact module graph the exported wrapper will rebuild.
    export_model = _build_export_model(model_config, model_class, export_state_dict)
    if bool(model_config.get("use_bitnet", False)) and bake_bitnet_weights is not None:
        try:
            baked_layers = bake_bitnet_weights(export_model)
        except Exception as exc:  # pragma: no cover - defensive
            warnings_log = compatibility.get("warnings", [])
            warnings_log.append(
                f"BitNet bake skipped (model instantiation failed): {exc}"
            )
            compatibility["warnings"] = warnings_log

    export_state_dict = {
        name: tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in export_model.state_dict().items()
    }

    quantization_config: Dict[str, Any] | None = None
    if quant != "none":
        export_state_dict, tensor_metadata = _quantize_export_state(
            export_state_dict, export_model, quant
        )
        # The HF wrappers nest the core model under `model.`.
        prefixed_metadata = {
            f"model.{name}": meta for name, meta in tensor_metadata.items()
        }
        quantization_config = _build_quantization_config(
            quant,
            quant_runtime,
            prefixed_metadata,
            bool(model_config.get("use_bitnet", False)),
        )

    # The HF wrappers hold the core model as `self.model`, so every tensor is
    # stored under the `model.` prefix (this is also what makes both v1
    # dequantize-on-load and v2 PackedLinear state-dict keys line up).
    export_state_dict = {
        f"model.{name}": tensor for name, tensor in export_state_dict.items()
    }

    hf_config = _build_transformers_config(
        model_config=model_config,
        model_class=model_class,
        task=task,
        modeling_class=modeling_class,
        quantization_config=quantization_config,
    )

    (output_path / "config.json").write_text(
        json.dumps(hf_config, indent=2),
        encoding="utf-8",
    )

    _save_safetensors(export_state_dict, output_path / SAFETENSORS_NAME)
    # Copy the codec into the artifact so `trust_remote_code=True` loads do
    # not depend on the Frankenstein source tree.
    shutil.copy2(
        Path(__file__).with_name("safetensors_quant.py"),
        output_path / QUANT_MODULE_NAME,
    )

    model_src = _repo_root() / "src" / "model"
    shutil.copytree(model_src, output_path / "model", dirs_exist_ok=True)

    (output_path / "configuration_frankenstein.py").write_text(_CONFIGURATION_FILE, encoding="utf-8")
    (output_path / "modeling_frankenstein.py").write_text(_MODELING_FILE, encoding="utf-8")
    (output_path / "__init__.py").write_text(_INIT_FILE, encoding="utf-8")
    (output_path / "load_frankenstein.py").write_text(_LOADER_FILE, encoding="utf-8")
    (output_path / "compatibility_report.json").write_text(
        json.dumps(compatibility, indent=2),
        encoding="utf-8",
    )

    usage = {
        "python_example": [
            "import sys",
            "sys.path.insert(0, '<model-dir>')",
            "from load_frankenstein import load_pretrained",
            "model = load_pretrained('<model-dir>')",
            "# The artifact is a self-contained package; HF's trust_remote_code",
            "# dynamic loader does not copy the nested model/ source tree.",
        ]
    }
    (output_path / "export_info.json").write_text(
        json.dumps(
            {
                "quantization": {
                    "format": quant,
                    "runtime": quant_runtime,
                    "safetensors": SAFETENSORS_NAME,
                },
                **usage,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "status": "ok",
        "output_dir": str(output_path),
        "modeling_class": modeling_class,
        "compatibility_report": str(output_path / "compatibility_report.json"),
        "quantization": {
            "format": quant,
            "runtime": quant_runtime,
            "compressed": quant != "none",
            "bits_per_weight": (
                None if quant == "none" else bits_per_weight(quant)
            ),
            "tensors": (
                0 if quantization_config is None else len(quantization_config["tensors"])
            ),
        },
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
    parser.add_argument(
        "--quant",
        choices=list(QUANT_FORMATS),
        default="none",
        help="Storage format for model.safetensors (default: none).",
    )
    parser.add_argument(
        "--quant-runtime",
        choices=list(QUANT_RUNTIMES),
        default="dequantize_on_load",
        help=(
            "Load-time runtime: dequantize_on_load (v1) or quantized_linear "
            "(v2). Ignored when --quant none (default: dequantize_on_load)."
        ),
    )
    args = parser.parse_args(argv)

    result = export_transformers_model(
        model_path=args.model,
        yaml_path=args.yaml,
        output_dir=args.output,
        quant=args.quant,
        quant_runtime=args.quant_runtime,
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
