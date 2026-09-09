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


def mlm_dataloader_dict(
    dataset: Any,
    tokenizer: Any,
    text_column: str,
    *,
    batch_size: int = 8,
    max_length: int = 128,
    mlm_probability: float = 0.15,
    device: str = "cpu",
    shuffle: bool = True,
) -> Any:
    """Build a DataLoader yielding MLM dict batches for the engine trainer.

    Tokenizes the text column and applies BERT-style masking at collate time
    so each epoch re-samples the mask (like ``DataCollatorForLanguageModeling``).
    Each batch is a dict with ``input_ids`` (masked), ``attention_mask`` and
    ``labels`` (``-100`` on unmasked positions) — the format
    :class:`TitanTrainer.compute_mlm_loss` expects for ``task="mlm"``.

    Parameters
    ----------
    dataset : DashAIDataset
        Source dataset carrying the text column.
    tokenizer : Any
        HF tokenizer (or compatible, exposing ``__call__``).
    text_column : str
        Name of the text column to tokenize.
    batch_size, max_length, mlm_probability, device, shuffle
        Loader / masking options.

    Returns
    -------
    torch.utils.data.DataLoader
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    texts = list(dataset[text_column])
    if not texts:
        raise ValueError(f"Column '{text_column}' has no rows for MLM pretraining.")

    enc = tokenizer(texts, truncation=True, padding="max_length",
                    max_length=max_length)
    input_ids_all = torch.tensor(enc["input_ids"], dtype=torch.long)
    attention_mask_all = torch.tensor(enc["attention_mask"], dtype=torch.long)

    special_ids = {
        tok_id for key in ("cls_token_id", "sep_token_id", "pad_token_id", "bos_token_id", "eos_token_id")
        if (tok_id := getattr(tokenizer, key, None)) is not None
    }

    class _MLMDataset(Dataset):
        def __len__(self):
            return input_ids_all.shape[0]

        def __getitem__(self, idx):
            return input_ids_all[idx], attention_mask_all[idx]

    def _collate_mask(batch):
        ids = torch.stack([b[0] for b in batch])
        attn = torch.stack([b[1] for b in batch])
        labels = ids.clone()
        # Candidates: real tokens, attended, not special, not padding.
        prob_matrix = torch.full(labels.shape, mlm_probability)
        prob_matrix[attn == 0] = 0.0
        for tok_id in special_ids:
            prob_matrix[labels == tok_id] = 0.0
        masked = torch.bernoulli(prob_matrix).bool()
        labels[~masked] = -100  # unmasked positions are not supervised
        # 80% [MASK], 10% random token, 10% keep (BERT recipe).
        ids_masked = ids.clone()
        mask_token_id = getattr(tokenizer, "mask_token_id", None)
        vocab = getattr(tokenizer, "vocab_size", None) or int(ids.max().item()) + 1
        random_tokens = torch.randint(0, max(vocab - 1, 1), ids.shape)
        r = torch.rand(ids.shape)
        if mask_token_id is not None:
            ids_masked[masked & (r < 0.8)] = mask_token_id
        random_idx = masked & (r >= 0.8) & (r < 0.9)
        ids_masked[random_idx] = random_tokens[random_idx]
        # 10% keep original token (no change).
        return {
            "input_ids": ids_masked,
            "attention_mask": attn,
            "labels": labels,
        }

    return DataLoader(
        _MLMDataset(),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_mask,
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


def tokenized_dataloader_dict(
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
    """Build a DataLoader yielding **dict** batches for the engine trainer.

    Same tokenization contract as :func:`tokenized_dataloader` but each batch
    is a dict with ``input_ids``, ``attention_mask`` and ``labels`` keys — the
    format :class:`TitanTrainer` expects (``batch["input_ids"]`` etc. for
    ``task="text_classification"``).
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

    def _collate_dict(batch):
        ids, mask, lbl = zip(*batch)
        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(mask),
            "labels": torch.stack(lbl),
        }

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_dict,
        pin_memory=str(device).startswith("cuda"),
    )


def image_dataloader_dict(
    dataset: Any,
    y_dataset: Any = None,
    *,
    image_size: int = 224,
    batch_size: int = 32,
    device: str = "cpu",
    shuffle: bool = True,
    label_to_idx: Optional[dict] = None,
    num_seg_classes: Optional[int] = None,
) -> Any:
    """Build a DataLoader yielding **dict** batches for the engine trainer.

    Mirrors :func:`image_dataloader` but yields dicts: ``{"pixel_values",
    "labels"}`` for classification and ``{"pixel_values", "segmentation_map"}``
    for segmentation — the format :class:`TitanTrainer`'s vision loss
    methods expect.

    Parameters
    ----------
    label_to_idx : dict, optional
        Pre-computed label mapping (from a prior :func:`image_dataloader`
        call). When ``None`` and ``y_dataset`` is given, the mapping is
        derived here.
    num_seg_classes : int, optional
        Number of segmentation classes (required for segmentation tasks).

    Returns
    -------
    tuple
        ``(dataloader, label_to_idx, num_classes)`` like
        :func:`image_dataloader`.
    """
    import torch
    import torch.utils.data
    from torchvision import transforms

    image_col = _image_column(dataset)
    label_col = None
    if y_dataset is not None:
        label_col = y_dataset.column_names[0]

    if label_to_idx is None and y_dataset is not None and label_col is not None:
        cat = (getattr(y_dataset, "types", {}) or {}).get(label_col)
        if cat is not None and getattr(cat, "categories", None):
            unique_labels = sorted(cat.categories)
        else:
            unique_labels = sorted(set(y_dataset[label_col]))
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    label_to_idx = dict(label_to_idx or {})

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    class _ImageDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.x = dataset
            self.y = y_dataset

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            image = transform(self.x[idx][image_col].to_pil())
            if self.y is None:
                return image
            label_str = self.y[idx][label_col]
            return image, int(label_to_idx.get(label_str, -1))

    def _collate_cls(batch):
        images = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return {"pixel_values": images, "labels": labels}

    def _collate_images(batch):
        return {"pixel_values": torch.stack(batch)}

    ds_obj = _ImageDataset()
    loader = torch.utils.data.DataLoader(
        ds_obj,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_cls if y_dataset is not None else _collate_images,
        pin_memory=str(device).startswith("cuda"),
    )
    return loader, label_to_idx, len(label_to_idx)


def segmentation_dataloader_dict(
    dataset: Any,
    *,
    image_size: int = 224,
    batch_size: int = 8,
    device: str = "cpu",
    shuffle: bool = True,
    num_seg_classes: int = 2,
) -> Any:
    """Build a DataLoader yielding ``{"pixel_values", "segmentation_map"}`` dicts.

    Targets are pseudo-masks derived from the input image luminance quantized
    to ``num_seg_classes`` levels (same contract as the historical bespoke
    loop; explicit mask columns should be decoded here in the future).

    Returns
    -------
    torch.utils.data.DataLoader
    """
    import torch
    import torch.utils.data
    from torchvision import transforms

    image_col = _image_column(dataset)

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    class _SegDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.x = dataset

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            image = transform(self.x[idx][image_col].to_pil())
            with torch.no_grad():
                gray = image.mean(dim=0)  # (H, W) in [0, 1]
                target = (gray * num_seg_classes).long().clamp(0, num_seg_classes - 1)
            return image, target

    def _collate(batch):
        images = torch.stack([b[0] for b in batch])
        targets = torch.stack([b[1] for b in batch])
        return {"pixel_values": images, "segmentation_map": targets}

    return torch.utils.data.DataLoader(
        _SegDataset(),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate,
        pin_memory=str(device).startswith("cuda"),
    )


# ---------------------------------------------------------------------------
# Vision: DashAI image columns -> (pixel_values, labels) tensors
# ---------------------------------------------------------------------------

def _image_column(dataset: Any) -> str:
    """Return the single image column of a DashAIDataset (DashAIImage type)."""
    try:
        from DashAI.back.types.dashai_image import DashAIImage
    except ImportError:  # pragma: no cover
        DashAIImage = ()  # type: ignore
    try:
        types = dataset.types
    except AttributeError:
        types = {}
    img_cols = [
        col
        for col in dataset.column_names
        if (DashAIImage and isinstance(types.get(col), DashAIImage))
        or str(types.get(col)).lower() == "dashaiimage"
    ]
    if len(img_cols) != 1:
        raise ValueError(
            f"Expected exactly one image column, found {img_cols} "
            f"(columns={list(dataset.column_names)})."
        )
    return img_cols[0]


def image_dataloader(
    dataset: Any,
    y_dataset: Any = None,
    *,
    image_size: int = 224,
    batch_size: int = 32,
    device: str = "cpu",
    shuffle: bool = True,
    label_column: Optional[str] = None,
) -> Any:
    """Build a DataLoader of (pixel_values, labels) or pixel_values tensors.

    Images are resized to ``image_size`` x ``image_size`` and normalized with
    ImageNet statistics (matching the torchvision DashAI classifiers).

    Parameters
    ----------
    dataset : DashAIDataset
        Source dataset carrying an image column.
    y_dataset : DashAIDataset, optional
        Label dataset (train mode). When ``None``, the loader yields images only.
    image_size : int
        Target square image size.
    batch_size, device, shuffle
        Loader options.
    label_column : str, optional
        Override label column name in ``y_dataset``.

    Returns
    -------
    torch.utils.data.DataLoader
    """
    import torch
    import torch.utils.data
    from torchvision import transforms

    image_col = _image_column(dataset)
    label_col = label_column
    if y_dataset is not None and label_col is None:
        label_col = y_dataset.column_names[0]

    label_to_idx: dict = {}
    if y_dataset is not None and label_col is not None:
        cat = (getattr(y_dataset, "types", {}) or {}).get(label_col)
        if cat is not None and getattr(cat, "categories", None):
            unique_labels = sorted(cat.categories)
        else:
            unique_labels = sorted(set(y_dataset[label_col]))
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    class _ImageDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.x = dataset
            self.y = y_dataset

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            image = transform(self.x[idx][image_col].to_pil())
            if self.y is None:
                return image
            label_str = self.y[idx][label_col]
            return image, int(label_to_idx.get(label_str, -1))

    def _collate_with_labels(batch):
        images = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return images, labels

    def _collate_images(batch):
        return torch.stack(batch)

    ds_obj = _ImageDataset()
    return (
        torch.utils.data.DataLoader(
            ds_obj,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_with_labels if y_dataset is not None else _collate_images,
            pin_memory=str(device).startswith("cuda"),
        ),
        label_to_idx,
        len(label_to_idx),
    )
