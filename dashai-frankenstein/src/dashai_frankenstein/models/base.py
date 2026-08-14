"""Shared machinery for Frankenstein classification DashAI components.

Both the NLP encoder (:class:`FrankensteinMLMModel`) and the ViT classifier
(:class:`FrankensteinViTClassifier`) reduce to the same DashAI contract: train
a Frankenstein backbone with a classification head on a labelled
``DashAIDataset`` and return a per-class probability matrix from ``predict``.

This module factors that loop out so the concrete components stay thin. It uses
the Frankenstein engine (Strategy A classification head) and writes per-epoch
metrics through DashAI's ``calculate_metrics``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from dashai_frankenstein.adapters import io as io_adapter
from dashai_frankenstein.adapters.metrics import EpochMetricsHook
from dashai_frankenstein.engine import (
    build_model_from_yaml,
    resolve_device,
    resolve_tokenizer,
    validate_training_yaml,
)

log = logging.getLogger(__name__)


def resolve_yaml(self) -> str:
    """Return the effective Frankenstein YAML text for this component.

    Returns
    -------
    str
        A Frankenstein training YAML document.

    Raises
    ------
    ValueError
        If ``frankenstein_yaml`` is empty.
    """
    yaml_text = str(getattr(self, "frankenstein_yaml", "") or "").strip()
    if not yaml_text:
        raise ValueError(
            "frankenstein_yaml is required. Build a YAML with the "
            "Frankenstein YAML builder "
            "(https://erickfmm.github.io/frankenstein-transformer/index.html) "
            "and paste it into the field."
        )
    return yaml_text


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
    """Run the in-process Frankenstein classification training loop.

    Parameters
    ----------
    self : BaseModel
        The DashAI component instance.
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
    import torch
    import torch.nn as nn

    from dashai_frankenstein.adapters.dataset import tokenized_dataloader

    self.num_labels = int(num_labels)
    yaml_text = resolve_yaml(self)
    validate_training_yaml(yaml_text)

    # Build model + config; tokenizer resolved to match the embedding vocab.
    model, loaded, _ = build_model_from_yaml(
        yaml_text,
        model_class_override=model_class_override,
        num_labels=self.num_labels,
    )

    # Runtime params from the loaded config's training_runtime.
    runtime = getattr(loaded, "training_runtime", {}) or {}
    device = resolve_device(runtime.get("device", "auto"))
    batch_size = int(runtime.get("batch_size", 16) or 16)
    num_epochs = int(runtime.get("num_epochs", 3) or 3)
    # lr from optimizer parameters (source of truth = Frankenstein schema).
    opt_cfg = getattr(loaded.training_config, "optimizer_parameters", {}) or {}
    opt_class = str(getattr(loaded.training_config, "optimizer_class", "adamw"))
    lr = _extract_lr_from_optimizer(opt_class, opt_cfg)

    tokenizer = resolve_tokenizer(loaded)
    if tokenizer is None:
        raise ValueError(
            "A tokenizer is required for text classification. Set "
            "tokenizer.name_or_path (or base_model) in the Frankenstein YAML."
        )
    # Ensure the model embedding matches the tokenizer vocabulary.
    if hasattr(model, "emb") and hasattr(model.emb, "num_embeddings"):
        tok_vocab = len(tokenizer)
        if tok_vocab != int(model.emb.num_embeddings):
            model, loaded, _ = build_model_from_yaml(
                yaml_text,
                model_class_override=model_class_override,
                num_labels=self.num_labels,
                vocab_size_override=tok_vocab,
            )

    model = model.to(device)
    self._frank_model = model
    self._loaded_config = loaded
    self._tokenizer = tokenizer
    self._device = device
    self._batch_size = batch_size
    self._label_column = label_column
    self._text_column_fn = text_column_fn

    text_col = text_column_fn(x_train)
    train_loader = tokenized_dataloader(
        x_train, tokenizer, text_col, label_column,
        batch_size=batch_size, device=device, shuffle=True,
    )

    val_loader = None
    if x_validation is not None and y_validation is not None:
        # Merge the label column into the validation features view.
        val_text_col = text_column_fn(x_validation)
        val_loader = tokenized_dataloader(
            x_validation, tokenizer, val_text_col, label_column,
            batch_size=batch_size, device=device, shuffle=False,
        )

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=float(lr)
    )
    criterion = nn.CrossEntropyLoss()

    hook = EpochMetricsHook(
        self, x_train, y_train, x_validation, y_validation, log_every_n_epochs=1
    )

    model.train()
    for epoch in range(1, num_epochs + 1):
        running, count = 0.0, 0
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids)  # (B, num_labels) via Strategy-A head
            loss = criterion(logits, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += float(loss.item())
            count += 1
        log.info("epoch %s/%s loss=%.4f", epoch, num_epochs, running / max(count, 1))
        hook(epoch=epoch)

    self.fitted = True
    self.x_data = {"train": x_train}
    self.y_data = {"train": y_train}
    if x_validation is not None:
        self.x_data["validation"] = x_validation
        self.y_data["validation"] = y_validation
    return self


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
        frankenstein_yaml="",
    )
    instance._frank_model = model
    instance._loaded_config = loaded
    instance._tokenizer = tokenizer
    instance._device = "cpu"
    instance._batch_size = 16
    instance.num_labels = extra.get("num_labels")
    instance.fitted = bool(extra.get("fitted", True))
    return instance