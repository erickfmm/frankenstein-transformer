"""Shared machinery for Frankenstein classification DashAI components.

Both the NLP encoder (:class:`FrankensteinMLMModel`) and the ViT classifier
(:class:`FrankensteinViTClassifier`) reduce to the same DashAI contract: train
a Frankenstein backbone with a classification head on a labelled
``DashAIDataset`` and return a per-class probability matrix from ``predict``.

This module factors that loop out so the concrete components stay thin. It
drives training through the Frankenstein engine
(:func:`src.engine.train_from_config`, ``task="text_classification"``), which
runs the full :class:`TitanTrainer` loop (AMP, grad clipping, NaN retry,
checkpoints, CSV telemetry) and streams every per-step/per-epoch metric into
DashAI's native metric store via :class:`~dashai_frankenstein.adapters.telemetry.DashAITelemetryCallback`.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from dashai_frankenstein.adapters import io as io_adapter
from dashai_frankenstein.adapters.telemetry import DashAITelemetryCallback
from dashai_frankenstein.engine import (
    build_model_from_json,
    resolve_device,
    resolve_tokenizer,
    validate_training_json,
)

log = logging.getLogger(__name__)


def resolve_json(self) -> str:
    """Return the effective Frankenstein JSON text for this component.

    Returns
    -------
    str
        A Frankenstein training config as a single-line JSON string.

    Raises
    ------
    ValueError
        If ``frankenstein_json`` is empty.
    """
    json_text = str(getattr(self, "frankenstein_json", "") or "").strip()
    if not json_text:
        raise ValueError(
            "frankenstein_json is required. Build a config with the "
            "Frankenstein YAML builder "
            "(https://erickfmm.github.io/frankenstein-transformer/index.html), "
            "convert it to a one-line JSON, and paste it into the field."
        )
    return json_text


def _extract_lr_from_optimizer(
    optimizer_class: str, optimizer_parameters: Dict[str, Any]
) -> float:
    """Extract a learning rate from the Frankenstein optimizer parameters.

    The Frankenstein optimizer schema uses prefixed per-group keys
    (``<opt_class>-lr_<group>``). The classifier head is a single ``nn.Linear``
    over a pooled representation (group ``other``). This helper tries, in order:
    ``<opt>-lr_other``, ``<opt>-lr_attention``, ``<opt>-lr_embeddings``,
    ``<opt>-lr_norms``, and falls back to ``1e-4`` if none are set.

    Parameters
    ----------
    optimizer_class : str
        Optimizer class name (e.g. ``"adamw"``).
    optimizer_parameters : dict
        The flat optimizer parameters dict from the loaded config.

    Returns
    -------
    float
        A learning rate for the classifier-head optimizer.
    """
    prefix = str(optimizer_class or "adamw").strip().lower()
    for group in ("other", "attention", "embeddings", "norms"):
        key = f"{prefix}-lr_{group}"
        val = optimizer_parameters.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return 1e-4


def _inject_engine_overrides(
    json_text: str,
    *,
    task: str = "text_classification",
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Parse a Frankenstein JSON config and inject DashAI runtime overrides.

    Ensures the config is engine-drivable: sets ``training.task``, maps the
    DashAI device label, and optionally forces ``num_epochs`` /
    ``batch_size`` (the engine's runtime keys). Arbitrary nested extras are
    deep-merged last.

    Parameters
    ----------
    json_text : str
        Frankenstein training config (one-line JSON or a dict).
    task : str
        Frankenstein ``training.task`` for the engine path.
    num_epochs, batch_size : int, optional
        Runtime overrides for the engine loop.
    extra : dict, optional
        Nested overrides deep-merged into the parsed config.

    Returns
    -------
    dict
        The merged config ready for ``engine.train_from_config``.
    """
    from dashai_frankenstein.engine import _deep_merge

    parsed: Dict[str, Any]
    if isinstance(json_text, dict):
        parsed = dict(json_text)
    else:
        import json

        parsed = json.loads(json_text) or {}

    training = parsed.setdefault("training", {})
    training["task"] = task
    if num_epochs is not None:
        training["num_epochs"] = int(num_epochs)
    if batch_size is not None:
        training["batch_size"] = int(batch_size)
    if extra:
        _deep_merge(parsed, extra)
    return parsed


def classification_train(
    self,
    x_train: Any,
    y_train: Any,
    x_validation: Any = None,
    y_validation: Any = None,
    *,
    num_labels: int,
    label_column: str,
    text_column_fn,
    model_class_override: Optional[str] = None,
) -> Any:
    """Run the Frankenstein classification training through the engine.

    Delegates to :func:`src.engine.train_from_config` with
    ``task="text_classification"`` and a pre-built dict-batch DataLoader.
    The engine instantiates :class:`TitanTrainer` (AMP, gradient clipping,
    NaN retry, rolling/best checkpoints, CSV telemetry) and streams every
    step/epoch metric into DashAI via :class:`DashAITelemetryCallback`.

    Parameters
    ----------
    self : BaseModel
        The DashAI component instance (carries ``run_id`` injected by
        ``ModelFactory``).
    x_train, y_train : DashAIDataset
        Training features and labels.
    x_validation, y_validation : DashAIDataset, optional
        Validation features and labels.
    num_labels : int
        Number of target classes.
    label_column : str
        Name of the integer-label column in ``x_train``.
    text_column_fn : callable
        ``f(dataset) -> text_column_name``; resolves the input column to tokenize.
    model_class_override : str, optional
        Force a Frankenstein ``model_class``.

    Returns
    -------
    BaseModel
        ``self``, fitted.
    """
    from DashAI.back.core.enums.metrics import SplitEnum

    from dashai_frankenstein.adapters.dataset import tokenized_dataloader_dict
    from dashai_frankenstein.engine import train_from_config

    self.num_labels = int(num_labels)
    json_text = resolve_json(self)
    validate_training_json(json_text)

    # Build model + config; tokenizer resolved to match the embedding vocab.
    model, loaded, _ = build_model_from_json(
        json_text,
        model_class_override=model_class_override,
        num_labels=self.num_labels,
    )

    # Runtime params from the loaded config's training_runtime.
    runtime = getattr(loaded, "training_runtime", {}) or {}
    device = resolve_device(runtime.get("device", "auto"))
    batch_size = int(runtime.get("batch_size", 16) or 16)
    num_epochs = int(runtime.get("num_epochs", 3) or 3)

    tokenizer = resolve_tokenizer(loaded)
    if tokenizer is None:
        raise ValueError(
            "A tokenizer is required for text classification. Set "
            "tokenizer.name_or_path (or base_model) in the Frankenstein config."
        )
    # Ensure the model embedding matches the tokenizer vocabulary.
    if hasattr(model, "emb") and hasattr(model.emb, "num_embeddings"):
        tok_vocab = len(tokenizer)
        if tok_vocab != int(model.emb.num_embeddings):
            model, loaded, _ = build_model_from_json(
                json_text,
                model_class_override=model_class_override,
                num_labels=self.num_labels,
                vocab_size_override=tok_vocab,
            )

    text_col = text_column_fn(x_train)
    train_loader = None
    val_loader = None
    if bool(getattr(self, "use_dashai_dataset", True)):
        train_loader = tokenized_dataloader_dict(
            x_train, tokenizer, text_col, label_column,
            batch_size=batch_size, device=device, shuffle=True,
        )

        if x_validation is not None and y_validation is not None:
            # Merge the label column into the validation features view.
            val_text_col = text_column_fn(x_validation)
            val_loader = tokenized_dataloader_dict(
                x_validation, tokenizer, val_text_col, label_column,
                batch_size=batch_size, device=device, shuffle=False,
            )
    # The engine trains on a single iterable; chain train(+val) so the
    # validation split is still visited each epoch by the trainer loop.
    engine_dataset = train_loader if val_loader is None else _ConcatLoader(train_loader, val_loader)

    # Engine config: inject the DashAI-driven task and runtime overrides.
    engine_cfg = _inject_engine_overrides(
        json_text,
        task="text_classification",
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    callback = DashAITelemetryCallback(
        self, x_train, y_train, x_validation, y_validation,
        split=SplitEnum.TRAIN, log_every_n_steps=1,
    )

    self._frank_model = model
    self._loaded_config = loaded
    self._tokenizer = tokenizer
    self._device = device
    self._batch_size = batch_size
    self._label_column = label_column
    self._text_column_fn = text_column_fn

    use_dashai_dataset = bool(getattr(self, "use_dashai_dataset", True))
    result = train_from_config(
        engine_cfg,
        dataset=engine_dataset if use_dashai_dataset else None,
        device=device,
        supervisor="off",
        metrics_callback=callback,
    )
    if getattr(result, "model", None) is not None:
        self._frank_model = result.model

    self.fitted = True
    self.x_data = {"train": x_train}
    self.y_data = {"train": y_train}
    if x_validation is not None:
        self.x_data["validation"] = x_validation
        self.y_data["validation"] = y_validation
    return self


class _ConcatLoader:
    """Chain two DataLoaders into one iterable (train + validation pass).

    The engine's ``text_classification`` path consumes a single iterable of
    dict batches; this wrapper yields the training loader fully followed by
    the validation loader each epoch so both splits receive telemetry.

    Parameters
    ----------
    first, second : Iterable
        The loaders to chain (``second`` may be ``None``).
    """

    def __init__(self, first: Any, second: Any = None) -> None:
        self.first = first
        self.second = second

    def __iter__(self):
        for batch in self.first:
            yield batch
        if self.second is not None:
            for batch in self.second:
                yield batch

    def __len__(self) -> int:
        try:
            length = len(self.first)
        except TypeError:
            length = 0
        if self.second is not None:
            try:
                length += len(self.second)
            except TypeError:
                pass
        return length


def classification_predict(self, x_pred: Any, *, max_length: int = 512) -> np.ndarray:
    """Return a per-class probability matrix for ``x_pred``.

    Parameters
    ----------
    x_pred : DashAIDataset
        Inference features.
    max_length : int
        Max token length.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, num_labels)`` of softmax probabilities.
    """
    from sklearn.exceptions import NotFittedError

    if not getattr(self, "fitted", False):
        raise NotFittedError(
            f"This {type(self).__name__} instance is not fitted yet. Call "
            "'train' before using it."
        )

    import torch

    from dashai_frankenstein.adapters.dataset import prediction_loader

    model = self._frank_model
    tokenizer = self._tokenizer
    device = self._device
    text_col = self._text_column_fn(x_pred)

    loader = prediction_loader(
        x_pred, tokenizer, text_col, batch_size=32, max_length=max_length, device=device
    )
    model.eval()
    probs = []
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids)
            probs.append(logits.softmax(dim=-1).detach().cpu().numpy())
    if not probs:
        return np.zeros((0, int(self.num_labels)))
    return np.vstack(probs)


def persistence_save(self, filename: str) -> None:
    """DashAI ``save`` -> Frankenstein checkpoint bundle."""
    import os

    os.makedirs(filename, exist_ok=True)
    io_adapter.save_run(
        filename,
        getattr(self, "_frank_model", None),
        getattr(self, "_loaded_config", None),
        getattr(self, "_tokenizer", None),
        extra={
            "num_labels": getattr(self, "num_labels", None),
            "fitted": getattr(self, "fitted", False),
        },
    )


def persistence_load(cls, filename: str):
    """DashAI ``load`` -> rebuild from a Frankenstein checkpoint bundle."""
    model, loaded, tokenizer, extra = io_adapter.load_run(filename)
    instance = cls(
        frankenstein_json="",
    )
    instance._frank_model = model
    instance._loaded_config = loaded
    instance._tokenizer = tokenizer
    instance._device = "cpu"
    instance._batch_size = 16
    instance.num_labels = extra.get("num_labels")
    instance.fitted = bool(extra.get("fitted", True))
    return instance