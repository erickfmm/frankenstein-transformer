"""Dataset adapter: DashAIDataset <-> Frankenstein training tensors.

A ``DashAIDataset`` is a HuggingFace ``datasets.Dataset`` wrapper. Frankenstein
models consume tensors directly. This module extracts the relevant columns,
tokenizes text (for NLP components), and yields PyTorch ``DataLoader`` batches
ready for the in-process training loop.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


def find_text_column(dataset: Any, exclude: Optional[List[str]] = None) -> str:
    """Return the single non-categorical text column of a DashAIDataset.

    Parameters
    ----------
    dataset : DashAIDataset
        A dataset whose ``column_names`` and ``types`` are available.
    exclude : list of str, optional
        Column names to ignore (e.g. the label column).

    Returns
    -------
    str
        The name of the text column.

    Raises
    ------
    ValueError
        If there is not exactly one text column.
    """
    from DashAI.back.types.categorical import Categorical

    exclude = set(exclude or [])
    try:
        types = dataset.types
    except AttributeError:
        types = {}
    text_cols = [
        col
        for col in dataset.column_names
        if col not in exclude and not isinstance(types.get(col), Categorical)
    ]
    if len(text_cols) != 1:
        raise ValueError(
            f"Expected exactly one text column, found {text_cols} "
            f"(columns={list(dataset.column_names)}, excluded={sorted(exclude)})."
        )
    return text_cols[0]


def extract_label_column(dataset: Any) -> str:
    """Return the (single) categorical/output column name."""
    from DashAI.back.types.categorical import Categorical

    try:
        types = dataset.types
    except AttributeError:
        types = {}
    cat_cols = [
        col for col in dataset.column_names if isinstance(types.get(col), Categorical)
    ]
    if len(cat_cols) != 1:
        raise ValueError(
            f"Expected exactly one categorical label column, found {cat_cols}."
        )
    return cat_cols[0]


def tokenized_dataloader(
    dataset: Any,
    tokenizer: Any,
    text_column: str,
    label_column: str,
    *,
    batch_size: int = 16,
    max_length: int = 512,
    device: str = "cpu",
    shuffle: bool = True,
) -> Any:
    """Build a torch DataLoader of tokenized (input_ids, attention_mask, labels).

    Parameters
    ----------
    dataset : DashAIDataset
        Source dataset.
    tokenizer : Any
        HF tokenizer (or Frankenstein SPM tokenizer exposing ``__call__``
        returning ``input_ids``).
    text_column : str
        Name of the text column.
    label_column : str
        Name of the integer-label column.
    batch_size : int
        Batch size.
    max_length : int
        Max token length.
    device : str
        Torch device (for pinning).
    shuffle : bool
        Whether to shuffle.

    Returns
    -------
    torch.utils.data.DataLoader
        Yields dicts with ``input_ids``, ``attention_mask``, ``labels`` tensors.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    texts = list(dataset[text_column])
    labels = list(dataset[label_column])

    enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
    label_tensor = torch.tensor(np.asarray(labels).astype("int64"), dtype=torch.long)

    ds = TensorDataset(input_ids, attention_mask, label_tensor)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=str(device).startswith("cuda"),
    )


def prediction_loader(
    dataset: Any,
    tokenizer: Any,
    text_column: str,
    *,
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cpu",
) -> Any:
    """Build a DataLoader for inference (no labels required)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    texts = list(dataset[text_column])
    enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
    ds = TensorDataset(input_ids, attention_mask)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=str(device).startswith("cuda"),
    )
