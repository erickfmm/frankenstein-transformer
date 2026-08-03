"""Streaming MLM dataset with fault-tolerant parallel processing.

Provides :class:`StreamingMLMDataset`, a PyTorch ``Dataset`` that streams
text from HuggingFace datasets or local Parquet files, tokenizes and applies
MLM masking in parallel via process-pool workers, and caches processed
batches to disk for fault-tolerant resumption.
"""

from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
import torch
from torch.utils.data import Dataset as TorchDataset

try:
    from ..utils.storage_manager import StorageManager
    from ..tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer
except ImportError:
    from utils.storage_manager import StorageManager
    from tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer


def _load_tokenizer_from_spec(tokenizer_spec: Dict[str, Any]):
    """Reconstruct a tokenizer from a serialized specification dict.

    Args:
        tokenizer_spec: Dictionary with ``backend`` (``"spm"`` or ``"hf"``)
            and backend-specific fields.

    Returns:
        A tokenizer instance (:class:`SpanishSPMTokenizer` or HuggingFace
        ``AutoTokenizer``).

    Raises:
        RuntimeError: If ``transformers`` is not installed for HF backend.
        ValueError: If the backend is unsupported.
    """
    backend = tokenizer_spec["backend"]
    if backend == "spm":
        return SpanishSPMTokenizer(model_path=tokenizer_spec["model_path"])

    if backend == "hf":
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for HF tokenizer processing in StreamingMLMDataset"
            ) from exc

        return AutoTokenizer.from_pretrained(
            tokenizer_spec["name_or_path"],
            use_fast=bool(tokenizer_spec.get("use_fast", True)),
            trust_remote_code=bool(tokenizer_spec.get("trust_remote_code", False)),
        )

    raise ValueError(f"Unsupported tokenizer backend: {backend}")


def _encode_text_with_spec(
    tokenizer_spec: Dict[str, Any],
    tokenizer: Any,
    text: str,
    max_length: int,
) -> Dict[str, List[int]]:
    """Encode text to token IDs using the appropriate tokenizer backend.

    Args:
        tokenizer_spec: Tokenizer specification dict with ``backend`` key.
        tokenizer: Pre-loaded tokenizer instance.
        text: Raw input text string.
        max_length: Maximum sequence length for truncation/padding.

    Returns:
        Dictionary with ``"input_ids"`` and ``"attention_mask"`` lists.
    """
    backend = tokenizer_spec["backend"]

    if backend == "spm":
        return tokenizer.encode(text, max_length)

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        add_special_tokens=True,
        return_attention_mask=True,
    )
    return {
        "input_ids": [int(v) for v in encoded["input_ids"]],
        "attention_mask": [int(v) for v in encoded["attention_mask"]],
    }


def _apply_mlm_mask_standalone(
    input_ids: List[int],
    attention_mask: List[int],
    mlm_probability: float,
    vocab_size: int,
    mask_token_id: int,
    special_token_ids: List[int],
    pad_token_id: int,
) -> Tuple[List[int], List[int]]:
    """Apply standard BERT-style MLM masking to token IDs.

    Masks 15% of eligible tokens: 80% replaced with ``[MASK]``, 10% with
    random tokens, 10% left unchanged. Special tokens and padding are
    excluded from masking.

    Args:
        input_ids: List of token IDs.
        attention_mask: List of attention mask values (1 = attend).
        mlm_probability: Fraction of eligible tokens to mask.
        vocab_size: Vocabulary size for random token replacement.
        mask_token_id: Token ID for ``[MASK]``.
        special_token_ids: Token IDs excluded from masking.
        pad_token_id: Padding token ID excluded from masking.

    Returns:
        Tuple of ``(masked_input_ids, labels)`` where ``labels`` is a copy
        of the original ``input_ids``.
    """
    labels = input_ids.copy()
    special_tokens = set(int(token_id) for token_id in special_token_ids)

    maskable_positions = [
        idx
        for idx, token_id in enumerate(input_ids)
        if int(attention_mask[idx]) == 1
        and int(token_id) not in special_tokens
        and int(token_id) != int(pad_token_id)
    ]

    if not maskable_positions:
        return input_ids, labels

    num_to_mask = max(1, int(len(maskable_positions) * mlm_probability))
    masked_positions = random.sample(maskable_positions, min(num_to_mask, len(maskable_positions)))

    for pos in masked_positions:
        rand = random.random()
        if rand < 0.8:
            input_ids[pos] = int(mask_token_id)
        elif rand < 0.9:
            input_ids[pos] = random.randint(0, max(int(vocab_size) - 1, 0))

    return input_ids, labels


def _process_single_example(args):
    """Process a single text example: tokenize and apply MLM masking.

    Designed for use with :class:`ProcessPoolExecutor` for parallel
    processing. Each worker reconstructs its own tokenizer from the
    serialized spec to avoid pickling issues.

    Args:
        args: Tuple of ``(text, tokenizer_spec, max_length, mlm_probability)``.

    Returns:
        Dictionary with ``"input_ids"``, ``"attention_mask"``, and
        ``"labels"``, or ``None`` on error.
    """
    text, tokenizer_spec, max_length, mlm_probability = args

    try:
        tokenizer = _load_tokenizer_from_spec(tokenizer_spec)
        encoded = _encode_text_with_spec(tokenizer_spec, tokenizer, text, max_length)
        mlm_input, mlm_labels = _apply_mlm_mask_standalone(
            encoded["input_ids"].copy(),
            encoded["attention_mask"],
            mlm_probability,
            int(tokenizer_spec["vocab_size"]),
            int(tokenizer_spec["mask_token_id"]),
            list(tokenizer_spec["special_token_ids"]),
            int(tokenizer_spec["pad_token_id"]),
        )

        return {
            "input_ids": mlm_input,
            "attention_mask": encoded["attention_mask"],
            "labels": mlm_labels,
        }
    except Exception as exc:
        logging.warning("Error processing example: %s", exc)
        return None


class StreamingMLMDataset(TorchDataset):
    """Streaming MLM dataset with fault-tolerant parallel processing.

    Streams raw text from HuggingFace datasets or local Parquet files,
    tokenizes and applies MLM masking in parallel using process-pool workers,
    and caches processed batches to disk as pickle files. Supports resumption
    from partially completed caches and optional context-window joining for
    longer contiguous sequences.

    Attributes:
        tokenizer: The tokenizer instance.
        max_length: Maximum sequence length for tokenization.
        mlm_probability: Fraction of tokens to mask for MLM.
        max_samples: Target maximum number of processed examples.
        batch_size: Number of examples per cached batch file.
        num_workers: Number of parallel process-pool workers.
        total_examples: Total indexed examples available for ``__getitem__``.
        cache_dir: Active cache directory path.
        data_source: Label describing the data source used.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 512,
        mlm_probability: float = 0.15,
        max_samples: int = 1_000_000,
        batch_size: int = 5000,
        num_workers: Optional[int] = None,
        cache_dir: Optional[str] = None,
        max_batch_in_memory: Optional[int] = None,
        parallel_chunksize: int = 32,
        local_parquet_dir: Optional[str] = None,
        prefer_local_cache: bool = True,
        stream_local_parquet: bool = True,
        join_temp_data_context_window: int = 0,
        join_temp_data_min_remainder_tokens: int = 128,
    ):
        """Initialize the streaming MLM dataset.

        Args:
            tokenizer: Tokenizer instance (SPM or HuggingFace).
            max_length: Maximum sequence length after tokenization.
            mlm_probability: Fraction of eligible tokens to mask.
            max_samples: Target maximum number of examples to produce.
            batch_size: Number of examples per cached batch file.
            num_workers: Number of parallel workers. Defaults to
                ``min(8, cpu_count // 2)``.
            cache_dir: Directory for cached batch files. Defaults to
                ``./temp_data/dataset_cache``.
            max_batch_in_memory: Maximum in-memory batch size during
                processing. Defaults to ``min(batch_size, 2000)``.
            parallel_chunksize: Chunksize for ``ProcessPoolExecutor.map``.
            local_parquet_dir: Optional path to local Parquet files.
            prefer_local_cache: If ``True``, prefer local Parquet over
                streaming from HuggingFace.
            stream_local_parquet: If ``True``, stream local Parquet files
                instead of loading them entirely into memory.
            join_temp_data_context_window: If > 0, join cached examples
                into contiguous sequences of this length.
            join_temp_data_min_remainder_tokens: Minimum remaining tokens
                to form a final joined sequence.
        """
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.mlm_probability = float(mlm_probability)
        self.max_samples = int(max_samples)
        self.batch_size = int(batch_size)
        requested_workers = num_workers or min(8, max(1, mp.cpu_count() // 2))
        self.num_workers = max(1, min(int(requested_workers), mp.cpu_count()))
        self.parallel_chunksize = max(1, int(parallel_chunksize))
        default_max_batch = min(self.batch_size, 2000)
        self.max_batch_in_memory = max_batch_in_memory or default_max_batch
        self.processing_batch_size = min(self.batch_size, self.max_batch_in_memory)
        self.storage_manager = StorageManager()
        self.prefer_local_cache = bool(prefer_local_cache)
        self.stream_local_parquet = bool(stream_local_parquet)
        self.local_parquet_dir = Path(local_parquet_dir) if local_parquet_dir else None
        self.data_source = "unknown"

        self.join_temp_data_context_window = max(0, int(join_temp_data_context_window))
        self.join_temp_data_min_remainder_tokens = max(1, int(join_temp_data_min_remainder_tokens))

        self.tokenizer_spec = self._build_tokenizer_spec(tokenizer)
        self.special_token_ids = set(int(v) for v in self.tokenizer_spec["special_token_ids"])
        self.pad_token_id = int(self.tokenizer_spec["pad_token_id"])
        self.mask_token_id = int(self.tokenizer_spec["mask_token_id"])
        self.vocab_size = int(self.tokenizer_spec["vocab_size"])
        self.cls_token_id = self.tokenizer_spec.get("cls_token_id")
        self.sep_token_id = self.tokenizer_spec.get("sep_token_id")

        if cache_dir:
            self.base_cache_dir = Path(cache_dir)
        else:
            self.base_cache_dir = Path("./temp_data/dataset_cache")
        self.base_cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = self.base_cache_dir
        self.metadata_file = self.cache_dir / "metadata.json"
        self.progress_file = self.cache_dir / "progress.json"

        self.metadata = self._load_metadata()

        self.batch_meta: List[Dict[str, int]] = []
        self.cumulative_sizes: List[int] = []
        self.total_examples = 0
        self._batch_cache_id: Optional[int] = None
        self._batch_cache: Optional[List[Dict[str, Any]]] = None

        self._prepare_examples()
        if self.join_temp_data_context_window > 0:
            self._prepare_joined_cache()

    def _build_tokenizer_spec(self, tokenizer: Any) -> Dict[str, Any]:
        if isinstance(tokenizer, SpanishSPMTokenizer):
            if not tokenizer.model_path:
                raise ValueError("SpanishSPMTokenizer requires a model_path for multiprocessing")

            vocab = getattr(tokenizer, "vocab", {}) or {}
            pad_token_id = int(vocab.get("[PAD]", 0))
            mask_token_id = int(vocab.get("[MASK]", 3))
            cls_token_id = vocab.get("[CLS]", None)
            sep_token_id = vocab.get("[SEP]", None)
            unk_token_id = vocab.get("<unk>", 1)
            special_token_ids = {
                pad_token_id,
                mask_token_id,
                int(unk_token_id),
            }
            if cls_token_id is not None:
                special_token_ids.add(int(cls_token_id))
            if sep_token_id is not None:
                special_token_ids.add(int(sep_token_id))

            return {
                "backend": "spm",
                "model_path": tokenizer.model_path,
                "vocab_size": int(tokenizer.vocab_size),
                "special_token_ids": sorted(special_token_ids),
                "mask_token_id": mask_token_id,
                "pad_token_id": pad_token_id,
                "cls_token_id": int(cls_token_id) if cls_token_id is not None else None,
                "sep_token_id": int(sep_token_id) if sep_token_id is not None else None,
            }

        tokenizer_name_or_path = getattr(tokenizer, "name_or_path", None)
        if not tokenizer_name_or_path:
            raise ValueError(
                "HF tokenizer must expose name_or_path to support multiprocessing tokenization"
            )

        special_ids = set(int(v) for v in getattr(tokenizer, "all_special_ids", []) if v is not None)
        for attr_name in [
            "pad_token_id",
            "mask_token_id",
            "cls_token_id",
            "sep_token_id",
            "bos_token_id",
            "eos_token_id",
            "unk_token_id",
        ]:
            value = getattr(tokenizer, attr_name, None)
            if value is not None:
                special_ids.add(int(value))

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

        mask_token_id = getattr(tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("HF tokenizer must have mask_token_id for MLM training")

        cls_token_id = getattr(tokenizer, "cls_token_id", None)
        if cls_token_id is None:
            cls_token_id = getattr(tokenizer, "bos_token_id", None)

        sep_token_id = getattr(tokenizer, "sep_token_id", None)
        if sep_token_id is None:
            sep_token_id = getattr(tokenizer, "eos_token_id", None)

        tokenizer_length = len(tokenizer)
        init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
        return {
            "backend": "hf",
            "name_or_path": str(tokenizer_name_or_path),
            "use_fast": bool(getattr(tokenizer, "is_fast", True)),
            "trust_remote_code": bool(init_kwargs.get("trust_remote_code", False)),
            "vocab_size": int(tokenizer_length),
            "special_token_ids": sorted(special_ids),
            "mask_token_id": int(mask_token_id),
            "pad_token_id": int(pad_token_id),
            "cls_token_id": int(cls_token_id) if cls_token_id is not None else None,
            "sep_token_id": int(sep_token_id) if sep_token_id is not None else None,
        }

    def _load_metadata(self) -> Dict[str, Any]:
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                logging.info(
                    "Loaded metadata: %s batches completed",
                    len(metadata.get("completed_batches", [])),
                )
                return metadata
            except Exception as exc:
                logging.warning("Error loading metadata: %s, creating new", exc)

        return {
            "version": "2.0",
            "total_samples_target": self.max_samples,
            "total_samples_processed": 0,
            "completed_batches": [],
            "batch_size": self.batch_size,
            "last_batch_id": -1,
        }

    def _save_metadata(self):
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as handle:
                json.dump(self.metadata, handle, indent=2)
        except Exception as exc:
            logging.error("Error saving metadata: %s", exc)

    def _get_batch_file(self, batch_id: int) -> Path:
        return self.cache_dir / f"batch_{batch_id:05d}.pkl"

    def _get_batch_file_for_dir(self, directory: Path, batch_id: int) -> Path:
        return directory / f"batch_{batch_id:05d}.pkl"

    def _load_batch(self, batch_id: int) -> List[Dict[str, Any]]:
        batch_file = self._get_batch_file(batch_id)
        if not batch_file.exists():
            return []

        try:
            with open(batch_file, "rb") as handle:
                return pickle.load(handle)
        except Exception as exc:
            logging.error("Error loading batch %s: %s", batch_id, exc)
            return []

    def _load_batch_from_dir(self, directory: Path, batch_id: int) -> List[Dict[str, Any]]:
        batch_file = self._get_batch_file_for_dir(directory, batch_id)
        if not batch_file.exists():
            return []

        try:
            with open(batch_file, "rb") as handle:
                return pickle.load(handle)
        except Exception as exc:
            logging.error("Error loading batch %s from %s: %s", batch_id, directory, exc)
            return []

    def _save_batch(self, batch_id: int, examples: List[Dict[str, Any]]) -> bool:
        batch_file = self._get_batch_file(batch_id)
        try:
            with open(batch_file, "wb") as handle:
                pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as exc:
            logging.error("Error saving batch %s: %s", batch_id, exc)
            return False

    def _save_batch_to_dir(self, directory: Path, batch_id: int, examples: List[Dict[str, Any]]) -> bool:
        batch_file = self._get_batch_file_for_dir(directory, batch_id)
        try:
            with open(batch_file, "wb") as handle:
                pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as exc:
            logging.error("Error saving batch %s to %s: %s", batch_id, directory, exc)
            return False

    def _append_batch_meta(self, batch_id: int, batch_size: int):
        if batch_size <= 0:
            return
        self.batch_meta.append({"batch_id": batch_id, "size": batch_size})
        self.total_examples += batch_size
        self.cumulative_sizes.append(self.total_examples)

    def _reset_batch_index(self):
        self.batch_meta = []
        self.cumulative_sizes = []
        self.total_examples = 0
        self._batch_cache_id = None
        self._batch_cache = None

    def _load_existing_batches(self) -> None:
        completed_batches = sorted(self.metadata.get("completed_batches", []))
        logging.info("Indexing %s existing batches from %s...", len(completed_batches), self.cache_dir)

        for batch_id in completed_batches:
            batch_examples = self._load_batch(batch_id)
            self._append_batch_meta(batch_id, len(batch_examples))

            if self.total_examples % 50000 == 0 and self.total_examples > 0:
                logging.info("Indexed %s examples so far...", self.total_examples)

        logging.info(
            "Indexed %s examples from %s batches",
            self.total_examples,
            len(completed_batches),
        )

    def _prepare_examples(self) -> None:
        self._load_existing_batches()
        samples_processed = self.total_examples

        if samples_processed >= self.max_samples:
            logging.info("Already have %s samples, skipping data preparation", samples_processed)
            return

        logging.info("Starting data preparation from %s samples...", samples_processed)
        logging.info("Using %s parallel workers", self.num_workers)
        if self.processing_batch_size < self.batch_size:
            logging.info(
                "Reducing in-memory batch size to %s (configured batch_size=%s)",
                self.processing_batch_size,
                self.batch_size,
            )

        try:
            dataset = self._load_dataset_source()
            dataset_iter = iter(dataset)

            if samples_processed > 0:
                logging.info("Skipping first %s examples...", samples_processed)
                for _ in range(samples_processed):
                    try:
                        next(dataset_iter)
                    except StopIteration:
                        logging.warning("Dataset exhausted during skip")
                        return

            current_batch_id = self.metadata.get("last_batch_id", -1) + 1
            batch_texts: List[str] = []

            while samples_processed < self.max_samples:
                try:
                    example = next(dataset_iter)
                    if "text" not in example:
                        continue
                    text = str(example["text"]).strip()
                    if len(text) <= 10:
                        continue

                    batch_texts.append(text)
                    if len(batch_texts) < self.processing_batch_size:
                        continue

                    new_examples = self._process_batch_parallel(batch_texts, current_batch_id)
                    batch_texts = []

                    if not new_examples:
                        current_batch_id += 1
                        continue

                    batch_size = len(new_examples)
                    samples_processed += batch_size

                    if self._save_batch(current_batch_id, new_examples):
                        self.metadata["completed_batches"].append(current_batch_id)
                        self.metadata["last_batch_id"] = current_batch_id
                        self.metadata["total_samples_processed"] = samples_processed
                        self._save_metadata()
                        self._append_batch_meta(current_batch_id, batch_size)

                        logging.info(
                            "Batch %s completed: %s examples, total: %s/%s",
                            current_batch_id,
                            batch_size,
                            samples_processed,
                            self.max_samples,
                        )

                    current_batch_id += 1

                    if not self.storage_manager.register_file(str(self.cache_dir)):
                        logging.warning("Storage limit reached")
                        break

                except StopIteration:
                    logging.info("Dataset exhausted")
                    break
                except Exception as exc:
                    logging.error("Error reading from dataset: %s", exc)
                    break

            if batch_texts and samples_processed < self.max_samples:
                new_examples = self._process_batch_parallel(batch_texts, current_batch_id)
                if new_examples:
                    batch_size = len(new_examples)
                    samples_processed += batch_size

                    if self._save_batch(current_batch_id, new_examples):
                        self.metadata["completed_batches"].append(current_batch_id)
                        self.metadata["last_batch_id"] = current_batch_id
                        self.metadata["total_samples_processed"] = samples_processed
                        self._save_metadata()
                        self._append_batch_meta(current_batch_id, batch_size)

                        logging.info(
                            "Final batch %s completed: %s examples, total: %s",
                            current_batch_id,
                            batch_size,
                            samples_processed,
                        )

            logging.info("Data preparation completed: %s total examples", samples_processed)

        except Exception as exc:
            logging.error("=" * 60)
            logging.error("🚨 CRITICAL ERROR: Dataset preparation failed")
            logging.error("=" * 60)
            logging.error("Error type: %s", type(exc).__name__)
            logging.error("Error message: %s", exc)
            logging.error("Dataset: erickfmm/red_pajama_es_hq_35")
            logging.error("Samples processed before failure: %s", samples_processed)
            logging.error("Examples indexed: %s", self.total_examples)
            logging.error("\nAttempted configuration:")
            logging.error("  - max_samples: %s", f"{self.max_samples:,}")
            logging.error("  - batch_size: %s", f"{self.batch_size:,}")
            logging.error("  - num_workers: %s", self.num_workers)
            logging.error("  - cache_dir: %s", self.cache_dir)

            if self.total_examples:
                logging.error("\n⚠️  %s examples available from cache", self.total_examples)
                logging.error("Continuing with cached data only (no synthetic fallback)")
                logging.error("=" * 60)
                return

            logging.error("\n⛔ SYSTEM HALTED - No cached data and no synthetic fallback allowed")
            logging.error("=" * 60)
            import traceback

            logging.error("\nFull traceback:")
            logging.error(traceback.format_exc())
            raise RuntimeError(
                "Failed to prepare dataset and no cached data available. "
                f"Error: {exc}. Check logs for details."
            ) from exc

    def _resolve_local_parquet_files(self) -> List[str]:
        candidate_dirs: List[Path] = []
        if self.local_parquet_dir:
            candidate_dirs.append(self.local_parquet_dir)

        hf_cache = os.getenv("HF_DATASETS_CACHE")
        if hf_cache:
            candidate_dirs.append(Path(hf_cache) / "hub" / "datasets--erickfmm--red_pajama_es_hq_35")

        candidate_dirs.append(
            Path.home() / ".cache" / "huggingface" / "hub" / "datasets--erickfmm--red_pajama_es_hq_35"
        )

        expanded_dirs: List[Path] = []
        for base_dir in candidate_dirs:
            if not base_dir.exists():
                continue

            expanded_dirs.append(base_dir)
            snapshots_dir = base_dir / "snapshots"
            if snapshots_dir.exists():
                for train_dir in snapshots_dir.glob("*/train"):
                    expanded_dirs.append(train_dir)

        for base_dir in expanded_dirs:
            if not base_dir.exists():
                continue

            parquet_files = sorted(str(path.resolve()) for path in base_dir.rglob("*.parquet"))
            if parquet_files:
                return parquet_files

        return []

    def _load_dataset_source(self):
        parquet_files = self._resolve_local_parquet_files() if self.prefer_local_cache else []
        if parquet_files:
            mode = "stream" if self.stream_local_parquet else "in_memory"
            self.data_source = f"local_parquet:{len(parquet_files)}:{mode}"
            logging.info(
                "Using local parquet cache with %s files (no streaming download), mode=%s.",
                len(parquet_files),
                mode,
            )
            return load_dataset(
                "parquet",
                data_files=parquet_files,
                split="train",
                streaming=self.stream_local_parquet,
            )

        self.data_source = "streaming_remote"
        logging.info(
            "Local parquet cache not found (checked user path/HF cache); falling back to streaming."
        )
        return load_dataset("erickfmm/red_pajama_es_hq_35", split="train", streaming=True)

    def _process_batch_parallel(self, texts: List[str], batch_id: int) -> List[Dict[str, Any]]:
        if not texts:
            return []

        results: List[Dict[str, Any]] = []
        args_iter = (
            (
                text,
                self.tokenizer_spec,
                self.max_length,
                self.mlm_probability,
            )
            for text in texts
        )

        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                for result in executor.map(
                    _process_single_example,
                    args_iter,
                    chunksize=self.parallel_chunksize,
                ):
                    if result is not None:
                        results.append(result)

        except Exception as exc:
            logging.error("Error in parallel processing for batch %s: %s", batch_id, exc)
            logging.info("Falling back to sequential processing...")
            for text in texts:
                try:
                    result = _process_single_example(
                        (
                            text,
                            self.tokenizer_spec,
                            self.max_length,
                            self.mlm_probability,
                        )
                    )
                    if result is not None:
                        results.append(result)
                except Exception as fallback_exc:
                    logging.warning("Error in sequential fallback: %s", fallback_exc)

        return results

    def _wrapper_special_tokens_count(self) -> int:
        count = 0
        if self.cls_token_id is not None:
            count += 1
        if self.sep_token_id is not None:
            count += 1
        return count

    def _extract_content_tokens(self, example: Dict[str, Any]) -> List[int]:
        labels = [int(token_id) for token_id in example.get("labels", [])]
        attention_mask = [int(v) for v in example.get("attention_mask", [1] * len(labels))]

        result: List[int] = []
        for idx, token_id in enumerate(labels):
            if idx >= len(attention_mask) or attention_mask[idx] == 0:
                continue
            if token_id in self.special_token_ids:
                continue
            if token_id == self.pad_token_id:
                continue
            result.append(token_id)
        return result

    def _build_joined_sequence(self, content_tokens: List[int], context_window: int) -> Dict[str, List[int]]:
        sequence: List[int] = []
        if self.cls_token_id is not None:
            sequence.append(int(self.cls_token_id))

        sequence.extend(int(token_id) for token_id in content_tokens)

        if self.sep_token_id is not None:
            sequence.append(int(self.sep_token_id))

        if len(sequence) > context_window:
            sequence = sequence[:context_window]
            if self.sep_token_id is not None:
                sequence[-1] = int(self.sep_token_id)

        attention_mask = [1] * len(sequence)
        if len(sequence) < context_window:
            pad_len = context_window - len(sequence)
            sequence.extend([self.pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)

        masked_input_ids, labels = _apply_mlm_mask_standalone(
            sequence.copy(),
            attention_mask,
            self.mlm_probability,
            self.vocab_size,
            self.mask_token_id,
            list(self.special_token_ids),
            self.pad_token_id,
        )
        return {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _prepare_joined_cache(self):
        context_window = self.join_temp_data_context_window
        joined_cache_dir = self.base_cache_dir / f"joined_ctx_{context_window}"
        joined_cache_dir.mkdir(parents=True, exist_ok=True)

        joined_metadata_file = joined_cache_dir / "metadata.json"
        joined_metadata: Optional[Dict[str, Any]] = None

        if joined_metadata_file.exists():
            try:
                with open(joined_metadata_file, "r", encoding="utf-8") as handle:
                    candidate = json.load(handle)
                if (
                    candidate.get("join_temp_data_context_window") == context_window
                    and candidate.get("source_cache_dir") == str(self.base_cache_dir)
                    and candidate.get("total_samples_target") == self.max_samples
                    and candidate.get("source_completed_batches")
                    == sorted(self.metadata.get("completed_batches", []))
                ):
                    joined_metadata = candidate
            except Exception as exc:
                logging.warning("Failed to load joined cache metadata: %s", exc)

        if joined_metadata is None:
            logging.info(
                "Building joined context cache at %s (context_window=%s)",
                joined_cache_dir,
                context_window,
            )
            joined_metadata = self._build_joined_cache(joined_cache_dir, context_window)

        self._activate_cache(joined_cache_dir, joined_metadata)

    def _build_joined_cache(self, joined_cache_dir: Path, context_window: int) -> Dict[str, Any]:
        source_batches = sorted(self.metadata.get("completed_batches", []))
        wrapper_tokens = self._wrapper_special_tokens_count()
        target_content_len = max(1, context_window - wrapper_tokens)

        total_joined_examples = 0
        current_batch_id = 0
        joined_batch_examples: List[Dict[str, Any]] = []
        content_buffer: List[int] = []
        completed_batches: List[int] = []

        for source_batch_id in source_batches:
            source_examples = self._load_batch_from_dir(self.base_cache_dir, source_batch_id)
            if not source_examples:
                continue

            for source_example in source_examples:
                content_tokens = self._extract_content_tokens(source_example)
                if not content_tokens:
                    continue
                content_buffer.extend(content_tokens)

                while len(content_buffer) >= target_content_len and total_joined_examples < self.max_samples:
                    chunk = content_buffer[:target_content_len]
                    del content_buffer[:target_content_len]
                    joined_batch_examples.append(self._build_joined_sequence(chunk, context_window))
                    total_joined_examples += 1

                    if len(joined_batch_examples) >= self.processing_batch_size:
                        if self._save_batch_to_dir(joined_cache_dir, current_batch_id, joined_batch_examples):
                            completed_batches.append(current_batch_id)
                        current_batch_id += 1
                        joined_batch_examples = []

                if total_joined_examples >= self.max_samples:
                    break

            if total_joined_examples >= self.max_samples:
                break

        if (
            total_joined_examples < self.max_samples
            and len(content_buffer) >= self.join_temp_data_min_remainder_tokens
        ):
            chunk = content_buffer[:target_content_len]
            joined_batch_examples.append(self._build_joined_sequence(chunk, context_window))
            total_joined_examples += 1

        if joined_batch_examples:
            if self._save_batch_to_dir(joined_cache_dir, current_batch_id, joined_batch_examples):
                completed_batches.append(current_batch_id)

        metadata = {
            "version": "3.0-joined",
            "total_samples_target": self.max_samples,
            "total_samples_processed": total_joined_examples,
            "completed_batches": completed_batches,
            "batch_size": self.batch_size,
            "last_batch_id": completed_batches[-1] if completed_batches else -1,
            "source_cache_dir": str(self.base_cache_dir),
            "source_completed_batches": source_batches,
            "join_temp_data_context_window": context_window,
            "join_temp_data_min_remainder_tokens": self.join_temp_data_min_remainder_tokens,
        }

        with open(joined_cache_dir / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        logging.info(
            "Joined cache ready: %s examples across %s batches",
            total_joined_examples,
            len(completed_batches),
        )
        return metadata

    def _activate_cache(self, directory: Path, metadata: Dict[str, Any]):
        self.cache_dir = directory
        self.metadata_file = self.cache_dir / "metadata.json"
        self.progress_file = self.cache_dir / "progress.json"
        self.metadata = metadata

        self._reset_batch_index()
        self._load_existing_batches()

    def _apply_mlm_mask(self, input_ids: List[int], attention_mask: List[int]) -> Tuple[List[int], List[int]]:
        return _apply_mlm_mask_standalone(
            input_ids,
            attention_mask,
            self.mlm_probability,
            self.vocab_size,
            self.mask_token_id,
            list(self.special_token_ids),
            self.pad_token_id,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return dataset statistics and configuration summary.

        Returns:
            Dictionary with keys including ``total_examples``,
            ``completed_batches``, ``total_samples_processed``,
            ``batch_size``, ``num_workers``, ``cache_dir``,
            ``data_source``, and tokenizer backend.
        """
        return {
            "total_examples": self.total_examples,
            "completed_batches": len(self.metadata.get("completed_batches", [])),
            "total_samples_processed": self.metadata.get("total_samples_processed", 0),
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "cache_dir": str(self.cache_dir),
            "data_source": self.data_source,
            "prefer_local_cache": self.prefer_local_cache,
            "local_parquet_dir": str(self.local_parquet_dir) if self.local_parquet_dir else None,
            "stream_local_parquet": self.stream_local_parquet,
            "join_temp_data_context_window": self.join_temp_data_context_window,
            "join_temp_data_min_remainder_tokens": self.join_temp_data_min_remainder_tokens,
            "tokenizer_backend": self.tokenizer_spec["backend"],
        }

    def cleanup_cache(self):
        import shutil

        if self.cache_dir.exists():
            try:
                shutil.rmtree(self.cache_dir)
                logging.info("Cleaned up cache directory: %s", self.cache_dir)
            except Exception as exc:
                logging.error("Error cleaning up cache: %s", exc)

    def __len__(self):
        return self.total_examples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_examples:
            raise IndexError("Index out of range")

        batch_pos = bisect_right(self.cumulative_sizes, idx)
        batch_meta = self.batch_meta[batch_pos]
        batch_id = batch_meta["batch_id"]

        if self._batch_cache_id != batch_id:
            self._batch_cache = self._load_batch(batch_id)
            self._batch_cache_id = batch_id

        batch_start = 0 if batch_pos == 0 else self.cumulative_sizes[batch_pos - 1]
        local_idx = idx - batch_start
        example = self._batch_cache[local_idx]
        return {
            "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(example["labels"], dtype=torch.long),
        }
