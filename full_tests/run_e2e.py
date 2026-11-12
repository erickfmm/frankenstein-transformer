"""Exhaustive end-to-end smoke tests for frankenstein-transformer.

This script drives the real CLI with tiny models, exercising:

* every trainable attention mixer in unordered pairs,
* every optimizer,
* every normalization type,
* every model-level positional encoding (rope, hope, nope, alibi, pape, pape_efficient,
  pape_ri, sinusoidal_absolute, sinusoidal_rotary, learned_absolute, none),
* every ViT `pos_embedding_type` across all 3 vision tasks (classification,
  patch_prediction, segmentation),
* major transversal toggles (BitNet, MoE, MoD, mHC, embeddings, residuals/AttnRes, etc.),
* a curated Mixture-of-Experts subtype sweep (`--category moe`): one design axis
  per combo (scoring, scaling, z-loss, bias, aux scope, expert choice, capacity,
  hash layers, shared experts, latent MoE, layer layout, activation clamp,
  top-1 pre-softmax) without a combinatorial product,
* encoder/MLM and decoder/causal-LM tasks,
* vision tasks (frankenstein_vit: patch_prediction, classification, segmentation),
* audio tasks (frankenstein_audio: asr with CTC and RNN-Transducer,
  audio_classification, and `feature: raw` EEG-style signals),
* real-data combos (`--category realdata`): every task family trained against
  small HuggingFace Hub datasets — MLM/causal-LM on Spanish poetry, text
  classification, SBERT triplets, ViT segmentation/classification/patch
  prediction, and audio ASR/classification.

By default (`--realonall`, enabled) **every** category trains against real
HuggingFace Hub data instead of the offline toy corpus:

* text sweeps use ``adleme94/borges`` + the BETO tokenizer,
* vision sweeps use breast-ultrasound (classification/patch prediction) and
  teeth panoramic X-rays (segmentation),
* audio sweeps use the Chilean-jokes ASR corpus and EMOTION_VOICES (the raw
  EEG case keeps the synthetic ``DummyAudioDataset``).

Pass ``--no-realonall`` for the older fully-offline synthetic behaviour. These
combos download models/corpora on first use, so allow a generous per-combo
timeout (e.g. ``--timeout 1800``).

Every combo now runs its **whole pipeline in the same worker, sequentially**:
training → deploy (``standard``) → inference evaluation → deploy
(``quantized``) → inference evaluation → Transformers exports (plain, plus
``q8_0``/``q4_0``/``i2_s`` in both v1/v2 runtimes for text combos; audio export
must be rejected with the documented message) → HF inference evaluation →
result persistence, and (in ``--save-to delete|memory``) artifact cleanup. The
harness writes, under ``tmp/results/``:

* ``inference.csv`` — predicted vs real comparisons per deployed artifact,
* ``inference_accuracy.csv`` — accuracy, balanced accuracy and macro/weighted
  F1 aggregated per combo/artifact (plain, ``hf_q8_0_v1``, ``hf_q4_0_v2``,
  ``hf_i2_s_v2``, ...),
* ``param_metrics.json`` — ``{model: {block: nparams, ..., "total": n}}`` using
  the canonical 10 block categories,
* ``training_metrics/<model>.csv`` — one ready-to-process training-metrics CSV
  per model (plus the held-out ``test_accuracy``/``test_loss``/``test_macro_f1``
  reported by the trainer in ``test_metrics.json``),
* ``results.json`` / ``results.md`` — the run status table, including
  ``test_acc``/``test_f1`` columns with the held-out accuracy and macro-F1
  measured at the end of training,
* ``configs/<combo_id>.yaml`` — the exact full YAML used for each combo,
  copied before cleanup so it survives ``--save-to delete|memory`` (disable
  with ``--no-keep-configs``),
* ``system_telemetry.csv`` — a 1 Hz host time series for the whole sweep:
  machine-wide CPU/RAM percentages plus the summed RSS and count of every live
  child process. Use it to size ``--workers`` from the run's own output.

It is intentionally NOT collected by ``pytest tests/`` (it lives in ``full_tests/``
at the repository root) and is meant to be left running for several hours.

Run from the repo root with the uv project env:

    uv run --extra cu118 --extra train python full_tests/run_e2e.py

To run only a subset:

    uv run --extra cu118 --extra train python full_tests/run_e2e.py --category opt --limit 3

To run one specific combo (ids look like ``opt_muon2``; comma-separated
patterns are matched case-insensitively as substrings, exact ids included):

    uv run --extra cu118 --extra train python full_tests/run_e2e.py --only opt_muon2

To discover the combo ids without training anything:

    uv run --extra cu118 --extra train python full_tests/run_e2e.py --category opt --list-combos

To restore the fully-offline synthetic data path:

    uv run --extra cu118 --extra train python full_tests/run_e2e.py --no-realonall

To run only the real-data (HuggingFace Hub) combos:

    uv run --extra cu118 --extra train python full_tests/run_e2e.py --category realdata --timeout 1800

To run on a specific device (default is cpu):

    uv run --extra cu118 --extra train python full_tests/run_e2e.py --device cuda

To enable the GPU thermal guard during training (with optional thresholds):

    uv run --extra cu118 --extra train python full_tests/run_e2e.py --device cuda --gpu-temp-guard

To resume from existing tmp/results directory (skip cleanup):

    uv run --extra cu118 --extra train python full_tests/run_e2e.py --resume

To validate every generated config WITHOUT training anything (real config
loader, dims invariants, SSOG grid vs max_length, optimizer parameter keys, and
any ``model.*`` key the assembled schema does not declare — a flat legacy key
would be silently dropped at runtime; exits non-zero on any invalid combo — run
this before a multi-hour sweep):

    uv run --extra cu118 --extra train python full_tests/run_e2e.py --dry-run

To run the training and deploy combos with N parallel workers (default is 28,
i.e. 28 concurrent CLI subprocesses; pass ``--workers 1`` for a sequential run):

    uv run --extra cu118 --extra train python full_tests/run_e2e.py --workers 28

Each worker owns complete pipelines — it trains, deploys, evaluates and
exports one combo before moving to the next — and launches every CLI step as an
isolated subprocess in its own ``tmp/runs/<combo_id>`` (or
``tmp/deploy/<combo_id>_<fmt>``) directory. Only the main thread appends to the
shared result artifacts, so with ``--workers > 1`` rows are recorded in
completion order rather than combo order. Keep ``--workers 1`` when sharing a
single GPU (VRAM/thermal-guard contention) or on a cold HuggingFace cache with
``--realonall`` (do one sequential run first to populate it).

CPU threads per worker are controlled with ``--omp-threads`` (default 2), which
exports ``OMP_NUM_THREADS`` plus the rest of the OpenMP/BLAS family (``MKL``,
``OPENBLAS``, ``NUMEXPR``, ``VECLIB``) into every subprocess. Values are never
clamped: ``--workers 28 --omp-threads 4`` is allowed and only logs an
oversubscription warning (28 x 4 = 112 threads over 56 cores). ``auto`` divides
the core count by ``--workers`` instead. Arbitrary variables can be added with a
repeatable ``--env KEY=VALUE``, applied after ``--omp-threads`` so it wins;
``PYTHONHASHSEED`` and ``FRANKENSTEIN_TEST_SEED`` are reserved for the
determinism contract.

    uv run --extra cu118 --extra train python full_tests/run_e2e.py \
        --workers 28 --omp-threads 2 --env TOKENIZERS_PARALLELISM=false

Before touching any directory the harness prints its own command line and the
effective value of every parameter (defaults marked ``[default]``) to stdout, so
the sweep's log is self-describing.

Artifact retention is selected with ``--save-to`` (default ``disk``):

    uv run --extra cu118 --extra train python full_tests/run_e2e.py --save-to disk

* ``disk`` — every run directory, checkpoint, deployed model and corpus is kept
  under ``full_tests/tmp/``.
* ``delete`` — each combo's run directory, models and data are deleted right
  after its pipeline finishes; only the results (metrics CSVs,
  ``inference.csv``, ``param_metrics.json``, ``configs/``, ``results.json|md``)
  survive. The whole scratch tree is dropped when the run ends. Combos whose
  pipeline failed keep their artifacts unless ``--no-keep-failed`` is passed
  (default: ``--keep-failed``, so failures can be reproduced).
* ``memory`` — the run directory and deployed models live on a RAM-backed
  tmpfs (``/dev/shm/frankenstein-full-tests-<pid>``) and no harness artifact
  touches the disk; results (including ``configs/``) still go to
  ``tmp/results/``. The tmpfs tree is removed at the end. Requires a writable
  ``/dev/shm``; the HuggingFace ``datasets`` cache (outside the harness'
  control) is not redirected.

Held-out evaluation is enabled by default:

    uv run --extra cu118 --extra train python full_tests/run_e2e.py --test-percentage 20

The value is written to ``training.test_percentage``; the trainer reserves that
percentage of each combo's dataset before training and reports the held-out
loss/accuracy plus balanced accuracy and macro/weighted F1 in
``test_metrics.json``, which the harness copies into ``results.json`` and the
``results.md`` ``test_acc``/``test_f1`` columns. Classification combos are
stratified by default and the exact indices are written to ``test_split.json``;
the deploy/inference evaluation reuses them, so the reported inference accuracy
and F1 are measured on the same held-out examples. Use
``--test-percentage 0`` to disable the split and restore the all-data
behavior.

Per-sweep defaults (what every flag falls back to when it is not passed)
------------------------------------------------------------------------
Every combo trains a single epoch, so
``steps = floor(floor(rows / batch_size) / gradient_accumulation_steps)``.
Unless ``--test-percentage 0`` is passed, the split removes ~20% of the rows
before that formula is applied (so e.g. 28 toy windows become 22, 32 real rows
become 26, 64 synthetic images become 51 and 16 synthetic utterances become
13).

| sweep | batch | sample cap | accum | optimizer steps |
|---|---|---|---|---|
| text sweeps (attn, opt, norm, pe, transversal, task) | 1 | 28 toy windows / 32 real Borges | 1 | 28 (32 with `--realonall`) |
| `batch_size_1` / `batch_size_2` | 1 / 2 | same as above | 1 | 32 / 16 (offline 28 / 14) |
| `tokenizer_train_*` | 1 | `text_dataset.max_samples: 200` | 1 | 200 |
| vision + vision_pe (`DummyImageDataset`) | 2 | 64 (hardcoded in `main.py`) | 4 | 8 |
| audio (`DummyAudioDataset`) | 16 | 16 (hardcoded in `main.py`) | 1 | 1 |
| realdata mlm / causal_lm | 2 | `text_dataset.max_samples: 32` | 1 | 16 |
| realdata text_classification | 2 | `text_dataset.max_samples: 50` | 1 | 25 |
| realdata vit (3 combos) | 2 | full HF split (no cap key) | 4 | split / 8 |
| realdata audio (2 combos) | 4 | full HF split (no cap key) | 1 | split / 4 |
| realdata sbert | `sbert.batch_size: 16` | `sbert.max_train_samples: 64` | — | 4 |

Timeouts default to 600 s for training and 300 s for deploy, inference-eval,
transformers-export and bitnet-gguf.

Global knobs (``full_tests`` only — every value is written into a documented
schema key, never into an ad-hoc one):

* ``--timeout N`` sets the budget for *every* subprocess; ``0`` disables all
  timeouts (wait forever).
* ``--max-samples N`` caps the samples per combo; ``0`` means all samples (the
  cap key is dropped, so each loader uses its uncapped default). It is written
  to ``training.max_samples`` (streaming toy corpus),
  ``text_dataset.max_samples`` (real/data-dir corpora) or
  ``sbert.max_train_samples`` (SBERT). Vision and audio combos expose no
  sample-cap key in the schema — the synthetic datasets are hardcoded and the
  real splits are consumed whole — so the flag leaves them untouched.
* ``--max-epochs N`` (``>= 1``) overrides ``training.num_epochs``, the active
  task sub-block (``classification``/``patch_prediction``/``segmentation``/
  ``asr``/``sbert``) and ``sbert.num_epochs``.
* ``--batch-size N`` overrides ``training.batch_size`` (and ``sbert.batch_size``
  for the SBERT combo). Note that changing it also changes the step count.
* ``--test-percentage N`` (``0–100``, default ``20``) writes
  ``training.test_percentage``: the trainer holds out that share of each
  combo's dataset and reports the held-out loss/accuracy/F1 in
  ``test_metrics.json`` (stratified for classification tasks by default).
  ``0`` disables the split. SBERT reports ``n/a`` (its own evaluator uses
  ``sbert.max_eval_samples``).
* ``--save-to {disk,delete,memory}`` (default ``disk``) selects artifact
  retention; see the retention examples above.
* ``--keep-configs`` / ``--no-keep-configs`` (default: on) copies every combo's
  full YAML into ``tmp/results/configs/<combo_id>.yaml`` before cleanup, so the
  exact config survives ``--save-to delete|memory``.

When ``--test-percentage 0`` is passed and none of the other overrides is
used, the harness writes exactly the same YAML it has always written.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import shlex
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Make the helpers importable whether this file is run as a script or module.
_FULL_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FULL_TESTS_DIR.parent
sys.path.insert(0, str(_FULL_TESTS_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import _helpers as helpers  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Canonical lists from the schema
# ---------------------------------------------------------------------------
# fasa and sparge are training-free/eval-only; skip them here.
ATTENTIONS = [
    "retnet", "mamba", "ode", "titan", "standard",
    "sigmoid", "sparse_transformer", "longformer", "bigbird",
    "sparsek", "nsa", "gla", "deltanet", "gated_deltanet",
    "gated_deltanet2", "hgrn2", "fox", "gated_softmax",
    "engram", "gqa", "mla", "gqla", "mlra",
    "tucker", "iha", "gta", "mtla", "cca", "ccgqa",
    # Kronecker Product Attention (arXiv:2610.12657): Kronecker-factored K/V
    # with a per-token latent factor cache; trains densely in both modes.
    "kpa",
    "msa", "sparda", "kda", "gma", "ssog",
    "falcon1", "falcon2", "falcon3", "falcon1a",
    "falcon2a", "falcon3a",
    # Convolution-augmented attention (arXiv:2005.08100): macaron FFN₁ + MHSA
    # + convolution module. Trains densely in both encoder and decoder modes
    # (causal padding in the decoder), so it belongs in the pair sweep too.
    "conformer",
    # Memory Attention (arXiv:2609.28399): V = K + Norm(E[s]) — trains
    # densely in encoder and decoder modes; conventional KV cache.
    "memory",
    # Latent Sparse Attention / N-LSA (arXiv:2610.42147): semantic key
    # clustering + top-k selection; trains natively in encoder and decoder
    # modes; exact bucket-cache incremental decoding.
    "lsa",
    # Fast Conformer limited-context attention (arXiv:2305.05084): local
    # window + head-pinned global attention token; trains its windowed pattern
    # natively in encoder and decoder modes; exact ring-cache incremental
    # decoding.
    "fastconformer",
    # Continuum Attention (arXiv:2610.20586): exact integration of an
    # exponential relevance density over piecewise-linear key/value
    # trajectories; trains natively in encoder and decoder modes; exact
    # KV-cache incremental decoding.
    "ca",
    # First Passage Attention (arXiv:2610.22714): softmax + path-extremum
    # (range-min) forget bias; trains natively in encoder and decoder modes;
    # exact KV-cache incremental decoding (two per-key/head scalars).
    "fpa",
    # Exclusive Self Attention (arXiv:2603.09078): standard softmax attention
    # + post-SDPA orthogonalization against the self value vector; trains
    # natively in encoder and decoder modes; exact KV-cache incremental
    # decoding.
    "xsa",
    # Multi-Token Attention (arXiv:2504.00927): causal key-query and
    # head-group convolutions over the attention score map, pre and post
    # softmax, with a gated group norm; trains natively in encoder and
    # decoder modes; exact KV-cache incremental decoding with a cq-1 query
    # tail.
    "mta",
]

# Model-wide positional encodings (src/schema/_model/_positional_encoding.yaml).
# Each is exercised on a fixed baseline layer_pattern — no cross product with mixers.
POSITIONAL_ENCODINGS = [
    "rope", "hope", "nope", "alibi", "bam", "pape", "pape_efficient", "pape_ri",
    "sinusoidal_absolute", "sinusoidal_rotary", "learned_absolute", "none",
]

# ViT image positional embedding types (src/schema/_image.yaml pos_embedding_type enum).
# Exercised across all 3 vision tasks — no cross product with mixers.
VISION_PE_TYPES = [
    "learned_1d", "none", "learned_absolute", "sinusoidal_absolute",
    "sinusoidal_rotary", "pape", "pape_efficient", "pape_ri",
    "rope", "hope", "nope", "alibi", "bam",
]

OPTIMIZERS = [
    "sgd", "sgd_momentum", "rmsprop", "adamw", "adafactor", "galore_adamw", "prodigy", "lion",
    "sophia", "muon", "turbo_muon", "af_muon", "ada_muon", "radam", "adan", "adopt", "ademamix",
    "mars_adamw", "cautious_adamw", "lamb", "schedulefree_adamw", "shampoo",
    "soap", "anon", "apollo", "apollo_mini", "q_apollo", "dion3", "eve", "ea4llm",
    "root", "muon2", "adam_pm", "nor_muon", "conda", "tuon", "namo", "namo_d",
]

NORMS = ["layer_norm", "dynamic_tanh", "derf", "rms_norm", "prms_norm", "flash_norm"]


# ---------------------------------------------------------------------------
# Real-data combos (HuggingFace Hub datasets)
# ---------------------------------------------------------------------------
# NLP tasks driven by a ``text_dataset`` block require an explicit HF
# tokenizer source (the engine falls back to the legacy toy SPM otherwise),
# and the tokenizer's vocab size is injected into ``model.dims.vocab_size``
# automatically — the configured dims.vocab_size stays a harmless placeholder.
REALDATA_TEXT_TOKENIZER = "dccuchile/bert-base-spanish-wwm-cased"

# SBERT training hard-requires a top-level ``base_model`` (``_train_sbert`` in
# ``src/engine.py``); BETO matches the shipped ``es_sbert_*`` examples.
REALDATA_SBERT_BASE_MODEL = "dccuchile/bert-base-spanish-wwm-cased"

# --- ``--realonall`` dataset bindings -------------------------------------
# Every category reuses the same small, publicly available corpora already
# exercised by ``--category realdata`` so the whole sweep trains on real data.
REAL_TEXT_TOKENIZER_SPEC: Dict[str, Any] = {
    "source": "hf_repo",
    "name_or_path": REALDATA_TEXT_TOKENIZER,
}
REAL_TEXT_DATASET: Dict[str, Any] = {
    "dataset_name": "adleme94/borges",
    "split": "train",
    "text_column": "poem",
    "max_samples": 32,
}
REAL_TEXT_MAX_LENGTH = 64
REAL_TEXT_TOKENIZER_SAMPLES = 400

REAL_VISION_CLASSIFICATION_DATASET: Dict[str, Any] = {
    "dataset_name": "as-cle-bert/breastcanc-ultrasound-class",
    "split": "train",
    "image_column": "image",
    "label_column": "label",
}
REAL_VISION_SEGMENTATION_DATASET: Dict[str, Any] = {
    "dataset_name": "SerdarHelli/SegmentationOfTeethPanoramicXRayImages",
    "split": "train",
    "image_column": "image",
    "segmentation_column": "label",
}
REAL_VISION_IMAGE_SIZE = {"height": 64, "width": 64}

REAL_AUDIO_ASR_DATASET: Dict[str, Any] = {
    "dataset_name": "astroza/chilean-jokes-festival-de-vina",
    "split": "test",
    "audio_column": "audio",
    "text_column": "text",
    "max_duration_sec": 10.0,
    "resample": True,
}
REAL_AUDIO_CLASSIFICATION_DATASET: Dict[str, Any] = {
    "dataset_name": "danielquillanroxas/EMOTION_VOICES",
    "split": "train",
    "audio_column": "audio",
    "label_column": "label",
    "max_duration_sec": 10.0,
    "resample": True,
}
REAL_AUDIO_CLASSIFICATION_NUM_CLASSES = 7

TEXT_MODEL_CLASSES = {
    "frankenstein_text_encoder",
    "frankenstein_text_decoder",
}

#: Default number of parallel pipeline workers (each launches its own CLI
#: subprocess, so this is the concurrency the host actually sees).
DEFAULT_WORKERS = 28

#: Default CPU threads per worker subprocess. The harness used to pin
#: ``OMP_NUM_THREADS=1``; two threads keeps the tiny e2e GEMMs out of the
#: single-thread regime without oversubscribing 28 x 2 = 56 cores.
DEFAULT_OMP_THREADS = "2"

#: The OpenMP/BLAS family. ``torch`` sizes its intra-op pool from
#: ``OMP_NUM_THREADS`` at import time; MKL, OpenBLAS, NumExpr and VecLib each
#: keep their own pool, so they are set together to keep the worker honest.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

#: Environment variables ``--env`` may not touch: they are part of the
#: determinism contract written into every subprocess by ``_helpers``.
RESERVED_ENV_VARS = frozenset({"PYTHONHASHSEED", "FRANKENSTEIN_TEST_SEED"})


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------
def _deep_update(base: dict, updates: dict) -> dict:
    """Return a new dict merging ``updates`` into ``base`` recursively.

    Scalar/list values in ``updates`` win over ``base``; nested dicts merge.
    """
    out = dict(base)
    for key, value in updates.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _load_example_config(name: str, **overrides: Any) -> dict:
    """Load ``configs/examples/<name>.yaml`` and apply deep overrides.

    Some features (the Conformer ViT, the audio model class/tasks) are already
    described by schema-validated example presets. Reusing them here keeps the
    harness honest about the shipped examples; only the data source and the
    step budget are overridden so each case stays a fast offline CPU smoke
    test. Any overrides are written into the run's ``config.yaml``, which is
    the harness' standard temporary-config mechanism.

    Args:
        name: Example file stem (without ``.yaml``).
        **overrides: Recursive overrides merged into the loaded config.

    Returns:
        The example config as a dict, with ``overrides`` applied.
    """
    path = _PROJECT_ROOT / "configs" / "examples" / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if overrides:
        config = _deep_update(config, overrides)
    return config


def _base_config(vocab_size: int, num_layers: int, num_loops: int) -> Tuple[dict, dict]:
    model = helpers.default_model_cfg(
        vocab_size=vocab_size, num_layers=num_layers, num_loops=num_loops
    )
    training = helpers.default_training_cfg(
        helpers.ensure_toy_parquet(), batch_size=1, num_epochs=1
    )
    return model, training


def _ensure_ssog_grid(model: dict, training: dict, *, force: bool = False) -> dict:
    """Give ``ssog`` layers a 1D grid matching the training ``max_length``.

    SSOG requires ``seq_len == grid_h * grid_w``. The toy MLM corpus is padded
    and truncated to exactly ``max_length`` tokens, so a degenerate
    ``1 x max_length`` grid fits the NLP runs (vision runs derive the patch
    grid automatically and must NOT call this).

    With ``force=True`` an already-present auto-derived grid is *overwritten*
    with the current ``max_length``. :func:`apply_realonall` raises
    ``max_length`` from 32 to 64 after the config makers ran, so the plain
    ``setdefault`` pass would leave a stale 1x32 grid behind and every SSOG
    run would crash at the first forward.
    """
    pattern = model.get("dims", {}).get("layer_pattern", [])
    if "ssog" not in pattern:
        return model
    max_length = int(training.get("max_length", 32))
    attention = dict(model.get("attention") or {})
    ssog_block = dict(attention.get("ssog") or {})
    if force or "grid_h" not in ssog_block:
        ssog_block["grid_h"] = 1
    if force or "grid_w" not in ssog_block:
        ssog_block["grid_w"] = max_length
    attention["ssog"] = ssog_block
    model = dict(model)
    model["attention"] = attention
    return model


def apply_realonall(combo_id: str, config: dict, real_text_dir: Optional[Path]) -> dict:
    """Swap a combo's synthetic data source for the real HuggingFace corpus.

    Applied to every generated combo when ``--realonall`` is enabled (the
    default). The model architecture stays untouched — only the data-bearing
    blocks (``tokenizer``/``text_dataset``, ``vision_dataset``/``image``,
    ``audio_dataset``/``audio``) are rewritten, so each sweep still isolates
    the mixer/norm/optimizer/toggle under test.

    Args:
        combo_id: Combo identifier (only used for log messages).
        config: The combo's YAML config dict.
        real_text_dir: Local parquet directory with the real text corpus, used
            by the ``train_from_dataset`` tokenizer cases.

    Returns:
        A new config dict bound to real data.
    """
    model_class = str(config.get("model_class") or "").strip().lower()
    training = dict(config.get("training") or {})
    task = str(training.get("task") or "mlm").strip().lower()
    config = dict(config)

    if model_class in TEXT_MODEL_CLASSES:
        tokenizer = dict(config.get("tokenizer") or {})
        existing_text_dataset = dict(config.get("text_dataset") or {})
        already_real = bool(existing_text_dataset.get("dataset_name"))
        if str(tokenizer.get("source") or "").strip().lower() == "train_from_dataset":
            # The tokenizer category exercises tokenizer training itself; it
            # reads a local parquet directory, so point it at the real corpus.
            if real_text_dir is not None:
                config["text_dataset"] = {
                    "data_dir": str(real_text_dir),
                    "text_column": "text",
                    "max_samples": 200,
                }
        elif not already_real:
            config["tokenizer"] = dict(REAL_TEXT_TOKENIZER_SPEC)
            config["text_dataset"] = dict(REAL_TEXT_DATASET)
        if task in {"mlm", "causal_lm", "text_classification"}:
            training["max_length"] = max(
                int(training.get("max_length", 0) or 0), REAL_TEXT_MAX_LENGTH
            )
        config["training"] = training
        # max_length may have just grown (32 -> 64): re-sync the SSOG grid so
        # grid_h * grid_w still equals the padded sequence length.
        config["model"] = _ensure_ssog_grid(
            config.get("model") or {}, training, force=True
        )
        logging.debug(
            "realonall: %s -> text data (%s)",
            combo_id,
            config.get("text_dataset", {}).get("dataset_name") or "real corpus",
        )
        return config

    if model_class == "frankenstein_vit":
        image = dict(config.get("image") or {})
        existing_vision_dataset = dict(config.get("vision_dataset") or {})
        already_real = bool(existing_vision_dataset.get("dataset_name"))
        image["image_size"] = dict(REAL_VISION_IMAGE_SIZE)
        if task == "segmentation":
            if not already_real:
                config["vision_dataset"] = dict(REAL_VISION_SEGMENTATION_DATASET)
            image.setdefault("seg_head_type", "pixel")
            image["num_seg_classes"] = 2
        elif task == "classification":
            if not already_real:
                config["vision_dataset"] = dict(REAL_VISION_CLASSIFICATION_DATASET)
            image["num_classes"] = 2
        elif not already_real:
            config["vision_dataset"] = {
                "dataset_name": REAL_VISION_CLASSIFICATION_DATASET["dataset_name"],
                "split": REAL_VISION_CLASSIFICATION_DATASET["split"],
                "image_column": REAL_VISION_CLASSIFICATION_DATASET["image_column"],
            }
        config["image"] = image
        logging.debug("realonall: %s -> vision data (%s)", combo_id, task)
        return config

    if model_class == "frankenstein_audio":
        audio = dict(config.get("audio") or {})
        dataset = dict(config.get("audio_dataset") or {})
        feature = str(
            audio.get("feature") or dataset.get("feature") or "logmel"
        ).strip().lower()
        already_real = bool(dataset.get("dataset_name"))
        if task == "asr":
            if not already_real:
                config["audio_dataset"] = dict(REAL_AUDIO_ASR_DATASET)
        elif task == "audio_classification" and feature != "raw":
            if not already_real:
                config["audio_dataset"] = dict(REAL_AUDIO_CLASSIFICATION_DATASET)
            audio["num_classes"] = REAL_AUDIO_CLASSIFICATION_NUM_CLASSES
        # ``feature: raw`` (EEG-style multichannel) keeps DummyAudioDataset:
        # no real EEG corpus is bound to this harness.
        config["audio"] = audio
        logging.debug("realonall: %s -> audio data (%s)", combo_id, task)
        return config

    return config


def apply_global_overrides(
    combo_id: str,
    config: dict,
    *,
    max_samples: Optional[int] = None,
    max_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> dict:
    """Apply the harness-wide ``--max-samples``/``--max-epochs``/``--batch-size``.

    ``batch_size``, ``num_epochs`` and ``max_samples`` are training *runtime*
    keys (not :class:`TrainingConfig` fields), so every task runner reads them
    from the saved YAML. Writing them here therefore needs no schema or source
    change. ``None`` means "leave this knob alone", which is how the harness
    preserves the per-sweep defaults documented in the module docstring.

    ``max_samples=0`` means "all samples": the cap key is removed (or set to
    ``null`` for SBERT), so each loader falls back to its uncapped default.
    Vision and audio combos expose no sample-cap key — the synthetic datasets
    are hardcoded (64 / 16 images/utterances) and the real HuggingFace splits
    are consumed whole — so ``max_samples`` is ignored for them.

    Args:
        combo_id: Combo identifier (used for log messages only).
        config: The combo's YAML config dict.
        max_samples: Optional sample cap; ``0`` removes the cap.
        max_epochs: Optional epoch count (``>= 1``).
        batch_size: Optional batch size (``>= 1``).

    Returns:
        A new config dict with the requested overrides applied.
    """
    training = dict(config.get("training") or {})
    config = dict(config)
    task = str(training.get("task") or "mlm").strip().lower()
    model_class = str(config.get("model_class") or "").strip().lower()
    is_sbert = task == "sbert" or bool(config.get("base_model"))
    changed: List[str] = []

    # ``num_epochs``/``batch_size`` also live inside the task sub-blocks
    # (vision's classification/patch_prediction/segmentation, audio's asr,
    # and sbert). Mirror the override into every block that declares them.
    def mirror(key: str, value: Any) -> None:
        for block_name, block in list(training.items()):
            if not isinstance(block, dict):
                continue
            if key in block:
                updated = dict(block)
                updated[key] = value
                training[block_name] = updated

    if max_epochs is not None:
        training["num_epochs"] = int(max_epochs)
        mirror("num_epochs", int(max_epochs))
        changed.append(f"num_epochs={max_epochs}")

    if batch_size is not None:
        training["batch_size"] = int(batch_size)
        mirror("batch_size", int(batch_size))
        if is_sbert:
            sbert = dict(training.get("sbert") or {})
            sbert["batch_size"] = int(batch_size)
            training["sbert"] = sbert
        changed.append(f"batch_size={batch_size}")

    if max_samples is not None:
        samples = int(max_samples)
        text_dataset = config.get("text_dataset")
        if is_sbert:
            sbert = dict(training.get("sbert") or {})
            # ``null`` is the schema's "all samples" value for SBERT.
            sbert["max_train_samples"] = None if samples == 0 else samples
            training["sbert"] = sbert
            changed.append(
                "sbert.max_train_samples="
                + ("all" if samples == 0 else str(samples))
            )
        elif model_class in TEXT_MODEL_CLASSES and isinstance(text_dataset, dict) and text_dataset:
            text_dataset = dict(text_dataset)
            if samples == 0:
                # Schema minimum is 1: omitting the key means "all samples".
                text_dataset.pop("max_samples", None)
            else:
                text_dataset["max_samples"] = samples
            config["text_dataset"] = text_dataset
            changed.append(
                "text_dataset.max_samples=" + ("all" if samples == 0 else str(samples))
            )
        elif model_class in TEXT_MODEL_CLASSES:
            # Toy/streaming corpus: the cap is training.max_samples. Its schema
            # default (20M) effectively means "all".
            if samples == 0:
                training.pop("max_samples", None)
            else:
                training["max_samples"] = samples
            changed.append("max_samples=" + ("all" if samples == 0 else str(samples)))
        else:
            logging.info(
                "max-samples: %s (%s) has no sample-cap key; left untouched",
                combo_id,
                model_class or task,
            )

    config["training"] = training
    if changed:
        logging.debug("Global overrides for %s: %s", combo_id, ", ".join(changed))
    return config


def _with_optimizer(training: dict, optimizer_class: str) -> dict:
    """Build a minimal per-group parameter block for the given optimizer class.

    The generated keys are filtered against the factory's
    ``OPTIMIZER_ALLOWED_KEYS`` (the same source of truth the CLI enforces), so
    an unsupported key can never turn into a run failure — e.g. ``rank`` is
    valid for ``apollo``/``q_apollo`` but not for ``apollo_mini``, which is
    rank-1 by construction.
    """
    prefix = f"{optimizer_class}-"
    params: Dict[str, Any] = {
        f"{prefix}lr_embeddings": 1e-4,
        f"{prefix}lr_norms": 1e-4,
        f"{prefix}lr_attention": 1e-4,
        f"{prefix}lr_other": 1e-4,
        f"{prefix}wd_embeddings": 0.0,
        f"{prefix}wd_norms": 0.0,
        f"{prefix}wd_attention": 0.0,
        f"{prefix}wd_other": 0.0,
    }
    # Adam-style families also accept betas/eps per group.  Providing them is
    # harmless for non-Adam optimizers because the factory simply ignores keys
    # it does not understand, while it satisfies the schema's minProperties.
    if optimizer_class in {
        "adamw", "galore_adamw", "prodigy", "radam", "adan", "adopt", "ademamix",
        "mars_adamw", "cautious_adamw", "schedulefree_adamw", "soap", "lion",
        "conda", "tuon", "namo", "namo_d",
    }:
        for group in ["embeddings", "norms", "attention", "other"]:
            params[f"{prefix}betas_{group}"] = [0.9, 0.999]
            params[f"{prefix}eps_{group}"] = 1e-8
    # A few optimizers have documented specific globals that are safer to set.
    if optimizer_class == "prodigy":
        params[f"{prefix}d_coef"] = 0.8
    if optimizer_class in {"muon", "turbo_muon"}:
        params[f"{prefix}momentum"] = 0.9
        params[f"{prefix}ns_steps"] = 5
    if optimizer_class == "ada_muon":
        params[f"{prefix}momentum"] = 0.9
        params[f"{prefix}ns_steps"] = 5
    if optimizer_class == "sophia":
        params[f"{prefix}rho"] = 0.05
        params[f"{prefix}update_k"] = 10
    if optimizer_class == "cautious_adamw":
        params[f"{prefix}cautious_clip"] = 1.0
    if optimizer_class == "anon":
        params[f"{prefix}gamma"] = 0.5
    if optimizer_class == "galore_adamw":
        params[f"{prefix}rank"] = 8
        params[f"{prefix}update_proj_gap"] = 50
    if optimizer_class in {"apollo", "apollo_mini", "q_apollo"}:
        params[f"{prefix}rank"] = 8
        params[f"{prefix}update_proj_gap"] = 50
        params[f"{prefix}scale"] = 1.0
        params[f"{prefix}proj_type"] = "std"
    if optimizer_class == "dion3":
        params[f"{prefix}momentum"] = 0.9
        params[f"{prefix}fraction"] = 0.25
        params[f"{prefix}ns_steps"] = 5
    if optimizer_class == "eve":
        params[f"{prefix}adam_steps"] = 3
        params[f"{prefix}de_weight"] = 0.5
    if optimizer_class == "ea4llm":
        params[f"{prefix}population"] = 30
        params[f"{prefix}sigma"] = 1e-3
    if optimizer_class == "root":
        params[f"{prefix}momentum"] = 0.9
        params[f"{prefix}ns_steps"] = 5
        params[f"{prefix}quantile"] = 0.9
    if optimizer_class == "muon2":
        params[f"{prefix}ns_steps"] = 5
        params[f"{prefix}ns_eps"] = 1e-7
    if optimizer_class == "adam_pm":
        params[f"{prefix}group_size"] = 8
        params[f"{prefix}pool_dim"] = "row"
    if optimizer_class == "nor_muon":
        params[f"{prefix}momentum"] = 0.95
        params[f"{prefix}ns_steps"] = 5
        params[f"{prefix}ns_eps"] = 1e-7
    if optimizer_class == "conda":
        params[f"{prefix}update_proj_gap"] = 4
        params[f"{prefix}scale"] = 0.25
        params[f"{prefix}correct_bias"] = True
    if optimizer_class == "tuon":
        params[f"{prefix}momentum"] = 0.95
        params[f"{prefix}ns_steps"] = 5
        params[f"{prefix}refresh_interval"] = 4
    if optimizer_class == "namo":
        params[f"{prefix}betas"] = [0.95, 0.99]
        params[f"{prefix}ns_steps"] = 5
        params[f"{prefix}ns_eps"] = 1e-7
    if optimizer_class == "namo_d":
        params[f"{prefix}betas"] = [0.95, 0.99]
        params[f"{prefix}ns_steps"] = 5
        params[f"{prefix}ns_eps"] = 1e-7
        params[f"{prefix}clamp"] = 0.1

    # Drop keys this optimizer family does not accept (e.g. `rank` for
    # apollo_mini). The factory is the contract; an unknown key would fail
    # the whole training run with `ValueError: Unknown parameters`.
    try:
        from src.model.optimizer.factory import OPTIMIZER_ALLOWED_KEYS

        allowed = OPTIMIZER_ALLOWED_KEYS.get(optimizer_class)
        if allowed is not None:
            dropped = [
                key for key in params if key[len(prefix):] not in allowed
            ]
            for key in dropped:
                params.pop(key)
            if dropped:
                logging.debug(
                    "opt %s: dropped unsupported parameters %s",
                    optimizer_class,
                    [key[len(prefix):] for key in dropped],
                )
    except Exception as exc:  # noqa: BLE001
        logging.debug(
            "Could not validate optimizer keys for %s: %s", optimizer_class, exc
        )

    training = training.copy()
    training["optimizer"] = {
        "optimizer_class": optimizer_class,
        "parameters": params,
    }
    return training


def make_attention_pair_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []
    for a, b in itertools.combinations(ATTENTIONS, 2):
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {
            "dims": {
                "layer_pattern": [a, b],
            },
        })
        model = _ensure_ssog_grid(model, training)
        combo_id = f"attn_{a}__{b}"
        configs.append((combo_id, {"model_class": "frankenstein_text_encoder", "model": model, "training": training}))
    return configs


def make_single_attention_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []
    for a in ATTENTIONS:
        model, training = _base_config(vocab_size, num_layers=1, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": [a]}})
        model = _ensure_ssog_grid(model, training)
        configs.append((f"attn_single_{a}", {"model_class": "frankenstein_text_encoder", "model": model, "training": training}))
    return configs


def make_optimizer_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []
    for opt in OPTIMIZERS:
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": ["standard", "titan"]}})
        training = _with_optimizer(training, opt)
        configs.append((f"opt_{opt}", {"model_class": "frankenstein_text_encoder", "model": model, "training": training}))
    return configs


def make_norm_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []
    for norm in NORMS:
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": ["standard", "titan"]}})
        norm_block: Dict[str, Any] = {"type": norm}
        if norm == "prms_norm":
            norm_block["partial_ratio"] = 0.5
        if norm == "flash_norm":
            norm_block["flashnorm_partial_ratio"] = 0.0
        model = _deep_update(model, {"norm": norm_block})
        configs.append((f"norm_{norm}", {"model_class": "frankenstein_text_encoder", "model": model, "training": training}))
    return configs


def make_positional_encoding_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    """One config per model-wide ``positional_encoding`` value.

    Uses a fixed baseline ``[standard, titan]`` pattern — no cross
    product with mixers.  Only the schema-defined ``positional_encoding`` and
    ``positional_encoding_parameters`` keys are touched.
    """
    configs: List[Tuple[str, dict]] = []
    for pe in POSITIONAL_ENCODINGS:
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": ["standard", "titan"]}})
        overrides: Dict[str, Any] = {"positional_encoding": pe}
        # PaPE family shares the ``pape`` parameters sub-object (num_parabolas,
        # num_positions, rotation_invariant).  Provide sane toy values.
        if pe in {"pape", "pape_efficient", "pape_ri"}:
            overrides["positional_encoding_parameters"] = {
                "pape": {"num_parabolas": 4, "num_positions": 1, "rotation_invariant": pe == "pape_ri"},
            }
        if pe == "sinusoidal_absolute":
            overrides["positional_encoding_parameters"] = {
                "sinusoidal": {"max_len": 64, "base": 10000.0, "scale": 1.0},
            }
        if pe == "learned_absolute":
            overrides["positional_encoding_parameters"] = {
                "learned": {"max_len": 64, "init_std": 0.02},
            }
        if pe == "bam":
            overrides["positional_encoding_parameters"] = {
                "bam": {"learn_mu": False, "theta_init": 0.0, "mu_init": 0.0, "eps": 1e-5},
            }
        if pe in {"rope", "hope"}:
            overrides["positional_encoding_parameters"] = {
                pe: {"base": 10000.0, "scaling": 1.0} if pe == "rope" else {"base": 10000.0, "damping": 0.01},
            }
        model = _deep_update(model, overrides)
        configs.append((f"pe_{pe}", {"model_class": "frankenstein_text_encoder", "model": model, "training": training}))
    return configs


def make_moe_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    """Curated Mixture-of-Experts subtype sweep.

    Each combo exercises exactly one MoE design axis on the standard two-layer
    toy geometry. The variants are curated rather than a cartesian product, so
    the whole sweep stays small while every schema feature is covered once:
    router scoring, output scale, z-loss, both load-balancing families, aux
    scope, expert choice, capacity/dropping, hash layers, shared experts,
    latent MoE, dense/interleaved layout, activation clamping and the
    softmax top-1 pre-softmax rule.

    Args:
        vocab_size: Vocabulary size for the toy model.

    Returns:
        List of ``(combo_id, config)`` tuples.
    """
    configs: List[Tuple[str, dict]] = []

    def add(cid: str, moe: dict) -> None:
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": ["standard", "titan"]}})
        moe_block = {"enabled": True, "num_experts": 4, "top_k": 2}
        moe_block.update(moe)
        model = _deep_update(model, {"moe": moe_block})
        model = _ensure_ssog_grid(model, training)
        configs.append(
            (
                cid,
                {
                    "model_class": "frankenstein_text_encoder",
                    "model": model,
                    "training": training,
                },
            )
        )

    # 3.1 Router scoring and output scale.
    add("moe_scoring_softmax", {})
    add("moe_scoring_sigmoid", {"router": {"scoring": "sigmoid"}})
    add("moe_scoring_softplus_sqrt", {"router": {"scoring": "softplus_sqrt"}})
    add(
        "moe_scaling_renorm",
        {"router": {"renormalize": True, "routed_scaling_factor": 2.0}},
    )
    add("moe_z_loss", {"router": {"z_loss_coeff": 0.001}})
    # 3.3 Load balancing: both families and the sequence scope.
    add(
        "moe_bias_sign",
        {
            "router": {"scoring": "sigmoid"},
            "balancing": {
                "aux_loss": "none",
                "bias_update": "sign",
                "bias_update_rate": 0.01,
            },
        },
    )
    add(
        "moe_bias_centered_sign",
        {
            "router": {"scoring": "sigmoid"},
            "balancing": {
                "aux_loss": "none",
                "bias_update": "centered_sign",
                "bias_update_rate": 0.01,
            },
        },
    )
    add(
        "moe_aux_sequence",
        {"balancing": {"aux_loss": "sequence", "aux_loss_coeff": 1e-3}},
    )
    # 3.2 Routing strategy and capacity.
    add(
        "moe_expert_choice",
        {"routing": {"type": "expert_choice"}, "balancing": {"aux_loss": "none"}},
    )
    add("moe_capacity_factor", {"routing": {"capacity_factor": 1.5}})
    add("moe_hash_layers", {"routing": {"hash_layers": 1}})
    add(
        "moe_top1_pre_softmax",
        {"top_k": 1, "router": {"pre_softmax": True}},
    )
    # 3.4 Experts: shared, latent, clamp.
    add(
        "moe_shared_gated",
        {"shared": {"count": 1, "hidden_size": 128, "gate": "sigmoid"}},
    )
    add(
        "moe_latent",
        {
            "top_k": 4,
            "latent": {
                "enabled": True,
                "dim": 48,
                "norm_before_up_proj": True,
            },
        },
    )
    add("moe_activation_clamp", {"experts": {"activation_clamp": 10.0}})
    # 3.5 Layer layout.
    add("moe_dense_prefix", {"layout": {"dense_layers": 1}})
    add("moe_interleaved", {"layout": {"every_n_layers": 2}})
    return configs


def make_transversal_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []

    def add(cid: str, overrides: dict) -> None:
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": ["standard", "titan"]}})
        model = _deep_update(model, overrides)
        model = _ensure_ssog_grid(model, training)
        configs.append((cid, {"model_class": "frankenstein_text_encoder", "model": model, "training": training}))

    add("trans_bitnet_on", {"bitnet": {"enabled": True}})
    add("trans_bitnet_routers", {
        "bitnet": {"enabled": True, "routers": True},
        "moe": {"enabled": True, "num_experts": 2, "top_k": 1},
    })
    add("trans_bitnet_conv", {"bitnet": {"enabled": True, "conv": True}, "embedding": {"conv": {"enabled": True, "kernel": 3}}})
    add("trans_factorized_emb", {"embedding": {"factorized": {"enabled": True, "dim": 32}}})
    # EELBERT dynamic embeddings (arXiv:2310.20144): the input lookup table is
    # replaced by an on-the-fly n-gram hash of the token strings. Real runs get
    # the vocabulary from the engine; the meta-device parameter count injects
    # placeholder strings (see _count_param_categories).
    add("trans_dynamic_emb", {
        "embedding": {
            "dynamic": {
                "enabled": True,
                "ngram_max": 3,
                "bucket": 1000000007,
                "seed": 42,
                "precompute": True,
            },
            "conv": {"enabled": False},
        },
    })
    add("trans_embedding_conv", {"embedding": {"conv": {"enabled": True, "kernel": 3}}})
    add("trans_moe", {"moe": {"enabled": True, "num_experts": 2, "top_k": 1}})
    add("trans_mod", {"mixture_of_depths": {"enabled": True, "capacity_ratio": 0.5}})
    add("trans_residuals_none", {"residuals": {"type": "none"}})
    add("trans_residuals_full_attn", {"residuals": {"type": "full_attn", "full_attn": {"init_query_zero": True, "use_rmsnorm_keys": True}}})
    add("trans_residuals_block_attn", {"residuals": {"type": "block_attn", "block_attn": {"num_blocks": 2, "init_query_zero": True, "use_rmsnorm_keys": True}}})
    # ``titan`` reads the *model-wide* PE (``model.positional_encoding``); it has
    # no per-mixer parameters any more (the registry spec carries no
    # params_schema), so overriding ``attention.titan`` would fail ``bind_params``.
    add("trans_pos_hope", {"positional_encoding": "hope"})
    add("trans_ssmax", {"ssmax": {"enabled": True, "s_init": 1.0}})
    # Fixed numerics (arXiv:2610.31642): cancellation-free softmax and
    # zero-sum-gradient hard-target cross-entropy.
    add("trans_fixed_softmax", {"numerics": {"fixed_softmax": True}})
    add("trans_fixed_ce", {"numerics": {"fixed_cross_entropy": True}})
    add("trans_fixed_both", {"numerics": {"fixed_softmax": True, "fixed_cross_entropy": True}})
    # SSMax rescales logits (larger margins) and BitNet perturbs weights: the
    # interaction is exactly where the precision cliff matters.
    add("trans_fixed_softmax_ssmax_bitnet", {
        "numerics": {"fixed_softmax": True},
        "ssmax": {"enabled": True, "s_init": 1.0},
        "bitnet": {"enabled": True},
    })
    add("trans_num_loops_2", {"dims": {"num_layers": 1, "num_loops": 2, "layer_pattern": ["standard"]}})
    add("trans_ffn_relu", {"ffn": {"activation": "relu"}})
    add("trans_ffn_swiglu", {"ffn": {"activation": "swiglu"}})
    add("trans_mhc", {
        "mhc": {"enabled": True, "expansion_rate": 2, "sinkhorn_iters": 5, "gating_init": 0.01, "checkpoint": False, "full_prec_under_bitnet": True},
    })
    # SSOG transversals: fixed (content-blind) field, and BitNet-wired SSOG.
    add("trans_ssog_fixed_field", {
        "dims": {"layer_pattern": ["ssog", "ssog"]},
        "attention": {"ssog": {"lookat": False}},
    })
    add("trans_ssog_bitnet", {
        "dims": {"layer_pattern": ["ssog", "ssog"]},
        "bitnet": {"enabled": True},
    })
    return configs


def make_task_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []

    # Encoder MLM (default path)
    model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
    model = _deep_update(model, {"dims": {"layer_pattern": ["standard", "titan"]}})
    configs.append(("task_mlm_encoder", {"model_class": "frankenstein_text_encoder", "model": model, "training": training}))

    # Decoder causal LM
    model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
    model = _deep_update(model, {"dims": {"mode": "decoder", "layer_pattern": ["standard", "titan"]}})
    training = training.copy()
    training["task"] = "causal_lm"
    configs.append(("task_causal_lm_decoder", {"model_class": "frankenstein_text_decoder", "model": model, "training": training}))

    # Decoder causal LM with the Conformer mixer: the convolution module must
    # switch to causal (left) padding in ``mode: decoder`` so training is
    # streaming-safe.
    model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
    model = _deep_update(model, {
        "dims": {"mode": "decoder", "layer_pattern": ["conformer", "standard"]},
        "attention": {"conformer": {"kernel_size": 8, "macaron": True, "conv_norm": "batch"}},
    })
    training = training.copy()
    training["task"] = "causal_lm"
    configs.append(("task_causal_lm_conformer", {"model_class": "frankenstein_text_decoder", "model": model, "training": training}))

    # Decoder causal LM with fixed numerics: exercises the ignore_index=-100
    # path of the fixed cross-entropy and the decoder softmax/sampling plumbing.
    model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
    model = _deep_update(model, {"dims": {"mode": "decoder", "layer_pattern": ["standard", "titan"]}})
    model = _deep_update(model, {"numerics": {"fixed_softmax": True, "fixed_cross_entropy": True}})
    training = training.copy()
    training["task"] = "causal_lm"
    configs.append(("task_causal_lm_fixed_numerics", {"model_class": "frankenstein_text_decoder", "model": model, "training": training}))
    return configs


def make_batch_size_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    configs: List[Tuple[str, dict]] = []
    for bs in [1, 2]:
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": ["standard", "titan"]}})
        training = training.copy()
        training["batch_size"] = bs
        configs.append((f"batch_size_{bs}", {"model_class": "frankenstein_text_encoder", "model": model, "training": training}))
    return configs


def _vit_model(vocab_size: int) -> dict:
    """Minimal ``model:`` block for a frankenstein_vit (Vision Transformer)."""
    return {
        "dims": {
            "hidden_size": 64, "num_layers": 2, "num_heads": 4, "num_loops": 1,
            "layer_pattern": ["standard"], "mode": "encoder", "dropout": 0.0,
            "vocab_size": vocab_size,
        },
        "norm": {"type": "layer_norm"},
        "embedding": {},
        "attention": {},
        "bitnet": {"enabled": False},
        "moe": {"enabled": False},
        "ffn": {"hidden_size": 256, "activation": "gelu"},
    }


def _vit_training(task: str) -> dict:
    """Minimal ``training:`` block for a frankenstein_vit task."""
    return {
        "task": task,
        "batch_size": 2,
        "num_epochs": 1,
        "optimizer": {
            "optimizer_class": "adamw",
            "parameters": {
                "adamw-lr_embeddings": 1e-4,
                "adamw-lr_norms": 1e-4,
                "adamw-lr_attention": 1e-4,
                "adamw-lr_other": 1e-4,
                "adamw-wd_embeddings": 0.0,
                "adamw-wd_norms": 0.0,
                "adamw-wd_attention": 0.0,
                "adamw-wd_other": 0.0,
            },
        },
        task: {"batch_size": 2, "num_epochs": 1, "learning_rate": 1e-4},
    }


def make_vision_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    """Build frankenstein_vit (Vision Transformer) e2e configs for all 3 tasks."""
    configs: List[Tuple[str, dict]] = []

    base_image = {
        "image_size": {"height": 32, "width": 32},
        "patch_size": 16, "in_channels": 3, "pos_embedding_type": "learned_1d",
        "cls_token": True, "pooling_mode": "cls",
    }
    base_dataset = {"rescale": {"height": 32, "width": 32}}

    # Classification
    image = dict(base_image, num_classes=10)
    configs.append(("vit_classification", {
        "model_class": "frankenstein_vit", "model": _vit_model(vocab_size),
        "image": image, "vision_dataset": base_dataset,
        "training": _vit_training("classification"),
    }))

    # Patch prediction (autosupervised)
    image = dict(base_image, mask_ratio=0.5, mask_token_strategy="bert",
                 prediction_target="mean_color_3bit")
    configs.append(("vit_patch_prediction", {
        "model_class": "frankenstein_vit", "model": _vit_model(vocab_size),
        "image": image, "vision_dataset": base_dataset,
        "training": _vit_training("patch_prediction"),
    }))

    # Segmentation (pixel head)
    image = dict(base_image, seg_head_type="pixel", num_seg_classes=5)
    configs.append(("vit_segmentation_pixel", {
        "model_class": "frankenstein_vit", "model": _vit_model(vocab_size),
        "image": image, "vision_dataset": base_dataset,
        "training": _vit_training("segmentation"),
    }))

    # Fixed numerics (arXiv:2610.31642) on the two CE+softmax vision paths:
    # classification CE and segmentation CE + Dice softmax.
    fixed_vit = _deep_update(_vit_model(vocab_size), {
        "numerics": {"fixed_softmax": True, "fixed_cross_entropy": True},
    })
    image = dict(base_image, num_classes=10)
    configs.append(("vit_classification_fixed_numerics", {
        "model_class": "frankenstein_vit", "model": fixed_vit,
        "image": image, "vision_dataset": base_dataset,
        "training": _vit_training("classification"),
    }))
    image = dict(base_image, seg_head_type="pixel", num_seg_classes=5)
    configs.append(("vit_segmentation_fixed_numerics", {
        "model_class": "frankenstein_vit", "model": fixed_vit,
        "image": image, "vision_dataset": base_dataset,
        "training": _vit_training("segmentation"),
    }))

    # SSOG variants — the Gaussian field needs the raw patch raster, so no
    # [CLS] token (gap pooling instead). The grid auto-derives from the image:
    # (32/16)x(32/16) = 2x2.
    ssog_image = {
        "image_size": {"height": 32, "width": 32},
        "patch_size": 16, "in_channels": 3, "pos_embedding_type": "learned_1d",
        "cls_token": False, "pooling_mode": "gap",
    }
    ssog_model = _deep_update(_vit_model(vocab_size), {
        "dims": {"layer_pattern": ["ssog"]},
        "attention": {"ssog": {"num_atoms": 2, "max_offset": 1.0}},
    })

    image = dict(ssog_image, num_classes=10)
    configs.append(("vit_classification_ssog", {
        "model_class": "frankenstein_vit", "model": ssog_model,
        "image": image, "vision_dataset": base_dataset,
        "training": _vit_training("classification"),
    }))

    image = dict(ssog_image, mask_ratio=0.5, mask_token_strategy="bert",
                 prediction_target="mean_color_3bit")
    configs.append(("vit_patch_prediction_ssog", {
        "model_class": "frankenstein_vit", "model": ssog_model,
        "image": image, "vision_dataset": base_dataset,
        "training": _vit_training("patch_prediction"),
    }))

    image = dict(ssog_image, seg_head_type="pixel", num_seg_classes=5)
    configs.append(("vit_segmentation_pixel_ssog", {
        "model_class": "frankenstein_vit", "model": ssog_model,
        "image": image, "vision_dataset": base_dataset,
        "training": _vit_training("segmentation"),
    }))

    # Conformer mixer inside the ViT: the patch raster is treated as the
    # sequence axis for the convolution module. Uses the canonical example
    # preset (`configs/examples/vit_conformer.yaml`), shrunk to a tiny batch.
    configs.append(("vit_conformer", _load_example_config(
        "vit_conformer",
        training={"batch_size": 4, "num_epochs": 1, "dataloader_workers": 0},
    )))

    return configs


def make_vision_pe_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    """Sweep every ViT ``pos_embedding_type`` across all 3 vision tasks.

    No cross product with mixers — each combo is one (task, PE) pair using the
    default ``standard`` ViT layer pattern.  Mirrors the strategy of
    ``make_positional_encoding_configs`` but for the vision image block.
    """
    configs: List[Tuple[str, dict]] = []
    base_image = {
        "image_size": {"height": 32, "width": 32},
        "patch_size": 16, "in_channels": 3, "cls_token": True, "pooling_mode": "cls",
    }
    base_dataset = {"rescale": {"height": 32, "width": 32}}

    for task, image_extras in [
        ("classification", {"num_classes": 10}),
        ("patch_prediction", {"mask_ratio": 0.5, "mask_token_strategy": "bert",
                             "prediction_target": "mean_color_3bit"}),
        ("segmentation", {"seg_head_type": "pixel", "num_seg_classes": 5}),
    ]:
        for pe in VISION_PE_TYPES:
            image = dict(base_image, pos_embedding_type=pe, **image_extras)
            configs.append((f"vit_pe_{task}_{pe}", {
                "model_class": "frankenstein_vit", "model": _vit_model(vocab_size),
                "image": image, "vision_dataset": base_dataset,
                "training": _vit_training(task),
            }))
    return configs


def make_audio_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    """Build ``frankenstein_audio`` cases for ASR, classification and raw EEG.

    All four cases reuse the canonical example presets
    (``configs/examples/audio_*.yaml``) with two offline/tiny overrides:

    * ``audio_dataset.dataset_name``/``dataset_dir`` are cleared, so the
      engine falls back to :class:`DummyAudioDataset` (synthetic waves and
      transcripts) and never touches the network. When ``--realonall`` is
      active, :func:`apply_realonall` restores the real Hub datasets for the
      ASR and classification cases afterwards (the raw-EEG case stays
      synthetic);
    * the training block is reduced to a single optimizer step
      (``batch_size=16`` over 16 dummy examples, ``gradient_accumulation_steps=1``,
      ``num_epochs=1``). The accumulation override is required: with the
      examples' default of 4 no step completes in a one-batch epoch and no
      checkpoint is written for the deploy smoke.

    Args:
        vocab_size: Unused (the audio presets carry their own ``vocab_size``);
            accepted so every config maker has the same signature.

    Returns:
        List of ``(combo_id, config_dict)`` tuples.
    """
    del vocab_size  # audio presets define their own vocabulary size

    offline_dataset = {"dataset_name": "", "dataset_dir": ""}
    tiny_training = {
        "batch_size": 16,
        "num_epochs": 1,
        "dataloader_workers": 0,
        "gradient_accumulation_steps": 1,
    }

    def audio_case(combo_id: str, example: str, **overrides: Any) -> Tuple[str, dict]:
        config = _load_example_config(
            example,
            audio_dataset=offline_dataset,
            training=tiny_training,
            **overrides,
        )
        return combo_id, config

    return [
        audio_case("audio_asr_ctc", "audio_asr_conformer"),
        audio_case("audio_asr_transducer", "audio_asr_transducer"),
        audio_case("audio_classification", "audio_classification_conformer"),
        audio_case("audio_eeg_classification", "audio_eeg_classification"),
    ]


def make_tokenizer_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    """Tokenizer source modes: HF-repo load vs train-from-dataset (all algorithms).

    Exercises the top-level ``tokenizer`` block on the CUSTOM (non-base_model)
    path:

    * ``source: hf_repo`` — load a pretrained tokenizer from a repo id; the
      vocab size is injected into ``model.dims.vocab_size``.
    * ``source: train_from_dataset`` — train a NEW tokenizer from the local
      parquet corpus (``text_dataset`` block) with each of the 4 algorithms
      (bpe, wordpiece, wordlevel, unigram). The toy parquet is used unless
      ``--realonall`` is active, in which case :func:`apply_realonall` points
      ``data_dir`` at the downloaded real-text corpus.
    """
    configs: List[Tuple[str, dict]] = []

    for algo in ["bpe", "wordpiece", "wordlevel", "unigram"]:
        model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {"dims": {"layer_pattern": ["standard", "standard"]}})
        training = training.copy()
        training["task"] = "mlm"
        tok_training: Dict[str, Any] = {
            "algorithm": algo,
            "min_frequency": 1,
            "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            "save_dir": "trained_tokenizer",
        }
        if algo != "unigram":
            tok_training["vocab_size"] = vocab_size
        configs.append((f"tokenizer_train_{algo}", {
            "model_class": "frankenstein_text_encoder",
            "model": model,
            "tokenizer": {
                "source": "train_from_dataset",
                "training": tok_training,
            },
            "text_dataset": {
                "data_dir": str(helpers.DATA_DIR),
                "text_column": "text",
                "max_samples": 200,
            },
            "training": training,
        }))

    return configs


# ---------------------------------------------------------------------------
# Real-data combos (HuggingFace Hub datasets)
# ---------------------------------------------------------------------------
def make_real_dataset_configs(vocab_size: int) -> List[Tuple[str, dict]]:
    """Build one real-data combo per task family, against HF Hub datasets.

    Every combo exercises the *actual* dataset field of its task:

    * ``text_dataset.dataset_name`` — MLM, causal LM and text classification.
      The text column of ``adleme94/borges`` is ``poem`` (not ``text``), and
      ``C-L-V/mini_corpus_sentiment_es`` carries string labels
      (``negativo``/``neutral``/``positivo`` → 3 classes via
      ``model.classification_head``).
    * ``vision_dataset.dataset_name`` — the teeth X-ray repo stores the
      segmentation mask in the ``label`` column (an Image), so
      ``segmentation_column: label`` is required; the breast-cancer repo is a
      2-class ClassLabel and also feeds the patch-prediction combo.
    * ``audio_dataset.dataset_name`` — ``astroza/chilean-jokes...`` only has a
      ``test`` split; ``danielquillanroxas/EMOTION_VOICES`` has 7 emotion
      classes (``audio.num_classes: 7``).
    * ``training.sbert.dataset_name`` — Spanish QQP triplets; SBERT training
      hard-requires a top-level ``base_model``.

    All combos are tiny (1 epoch, small batches, ``max_samples`` caps) but
    they download from the HuggingFace Hub — use a generous ``--timeout``.

    Args:
        vocab_size: Toy vocab size used only for the harness' tokenizer copy;
            the engine injects the real tokenizer vocab into the NLP models.

    Returns:
        List of ``(combo_id, config_dict)`` tuples.
    """
    configs: List[Tuple[str, dict]] = []

    hf_tokenizer = {"source": "hf_repo", "name_or_path": REALDATA_TEXT_TOKENIZER}

    # --- MLM on Spanish poetry (Borges) -----------------------------------
    model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
    model = _deep_update(model, {"dims": {"layer_pattern": ["standard", "titan"]}})
    training = _deep_update(training, {"task": "mlm", "batch_size": 2, "max_length": 64})
    configs.append(("realdata_mlm_borges", {
        "model_class": "frankenstein_text_encoder",
        "model": model,
        "tokenizer": hf_tokenizer,
        "text_dataset": {
            "dataset_name": "adleme94/borges",
            "split": "train",
            "text_column": "poem",
            "max_samples": 32,
        },
        "training": training,
    }))

    # --- Autoregressive causal LM on the same corpus -----------------------
    model, training = _base_config(vocab_size, num_layers=2, num_loops=1)
    model = _deep_update(model, {
        "dims": {"mode": "decoder", "layer_pattern": ["standard", "titan"]},
    })
    training = _deep_update(training, {"task": "causal_lm", "batch_size": 2, "max_length": 64})
    configs.append(("realdata_causal_lm_borges", {
        "model_class": "frankenstein_text_decoder",
        "model": model,
        "tokenizer": hf_tokenizer,
        "text_dataset": {
            "dataset_name": "adleme94/borges",
            "split": "train",
            "text_column": "poem",
            "max_samples": 32,
        },
        "training": training,
    }))

    # --- Supervised text classification (3 string labels) ------------------
    model, training = _base_config(vocab_size, num_layers=1, num_loops=1)
    model = _deep_update(model, {
        "dims": {"layer_pattern": ["standard"]},
        "classification_head": {"enabled": True, "num_labels": 3, "pooling_mode": "cls"},
    })
    training = _deep_update(training, {"task": "text_classification", "batch_size": 2, "max_length": 64})
    configs.append(("realdata_text_classification_mini_corpus", {
        "model_class": "frankenstein_text_encoder",
        "model": model,
        "tokenizer": hf_tokenizer,
        "text_dataset": {
            "dataset_name": "C-L-V/mini_corpus_sentiment_es",
            "split": "train",
            "text_column": "text",
            "label_column": "label",
            "use_labels": True,
            "max_samples": 50,
        },
        "training": training,
    }))

    # --- ViT segmentation: teeth panoramic X-rays --------------------------
    # The mask lives in the ``label`` column (an Image), not the default
    # ``segmentation_map`` — override segmentation_column. Binary masks.
    configs.append(("realdata_vit_segmentation_teeth", {
        "model_class": "frankenstein_vit",
        "model": _vit_model(vocab_size),
        "image": {
            "image_size": {"height": 64, "width": 64},
            "patch_size": 16, "in_channels": 3, "pos_embedding_type": "learned_1d",
            "cls_token": True, "pooling_mode": "cls",
            "seg_head_type": "pixel", "num_seg_classes": 2,
        },
        "vision_dataset": {
            "dataset_name": "SerdarHelli/SegmentationOfTeethPanoramicXRayImages",
            "split": "train",
            "image_column": "image",
            "segmentation_column": "label",
        },
        "training": _vit_training("segmentation"),
    }))

    # --- ViT classification: breast cancer ultrasound (2 classes) ----------
    configs.append(("realdata_vit_classification_breastcancer", {
        "model_class": "frankenstein_vit",
        "model": _vit_model(vocab_size),
        "image": {
            "image_size": {"height": 64, "width": 64},
            "patch_size": 16, "in_channels": 3, "pos_embedding_type": "learned_1d",
            "cls_token": True, "pooling_mode": "cls",
            "num_classes": 2,
        },
        "vision_dataset": {
            "dataset_name": "as-cle-bert/breastcanc-ultrasound-class",
            "split": "train",
            "image_column": "image",
            "label_column": "label",
        },
        "training": _vit_training("classification"),
    }))

    # --- ViT patch prediction: autosupervised on the same corpus -----------
    configs.append(("realdata_vit_patch_prediction_breastcancer", {
        "model_class": "frankenstein_vit",
        "model": _vit_model(vocab_size),
        "image": {
            "image_size": {"height": 64, "width": 64},
            "patch_size": 16, "in_channels": 3, "pos_embedding_type": "learned_1d",
            "cls_token": True, "pooling_mode": "cls",
            "mask_ratio": 0.5, "mask_token_strategy": "bert",
            "prediction_target": "mean_color_3bit",
        },
        "vision_dataset": {
            "dataset_name": "as-cle-bert/breastcanc-ultrasound-class",
            "split": "train",
            "image_column": "image",
        },
        "training": _vit_training("patch_prediction"),
    }))

    # --- Audio ASR: Chilean jokes (only a `test` split exists) -------------
    tiny_training = {
        "batch_size": 4,
        "num_epochs": 1,
        "dataloader_workers": 0,
        "gradient_accumulation_steps": 1,
    }
    configs.append(("realdata_audio_asr_vina", _load_example_config(
        "audio_asr_conformer",
        audio_dataset={
            "dataset_name": "astroza/chilean-jokes-festival-de-vina",
            "split": "test",
            "audio_column": "audio",
            "text_column": "text",
            "max_duration_sec": 10.0,
            "resample": True,
        },
        training=tiny_training,
    )))

    # --- Audio classification: 7 emotion classes ---------------------------
    configs.append(("realdata_audio_classification_emotion_voices", _load_example_config(
        "audio_classification_conformer",
        audio={"num_classes": 7},
        audio_dataset={
            "dataset_name": "danielquillanroxas/EMOTION_VOICES",
            "split": "train",
            "audio_column": "audio",
            "label_column": "label",
            "max_duration_sec": 10.0,
            "resample": True,
        },
        training=tiny_training,
    )))

    # --- SBERT triplets (requires a top-level base_model) ------------------
    # The dataset's NEGs column is a stringified list; the trainer parses it
    # via ast.literal_eval (see tests/test_sbert_triplets_stringified_*).
    configs.append(("realdata_sbert_qqp_triplets_es", {
        "base_model": REALDATA_SBERT_BASE_MODEL,
        # Ignored by the loader when base_model is set; present only so the
        # harness' run_training helper finds dims.vocab_size for its toy-SPM
        # copy.
        "model": {"dims": {"vocab_size": vocab_size}},
        "training": {
            "task": "sbert",
            "sbert": {
                "dataset_name": "erickfmm/embedding-data__QQP_triplets__translated_to_spanish",
                "dataset_type": "triplets",
                "columns": {
                    "query": "Q_traducida",
                    "positive": "POS_traducida",
                    "negatives": "NEGs_traducidas",
                },
                "output_dir": "./output/sbert_qqp_triplets_es",
                "batch_size": 16,
                "num_epochs": 1,
                "learning_rate": 2.0e-5,
                "max_train_samples": 64,
                "max_eval_samples": 16,
                "max_length": 128,
                "pooling_mode": "mean",
                "use_amp": False,
                "trust_remote_code": False,
            },
        },
    }))

    return configs


# ---------------------------------------------------------------------------
# Metrics artifacts (parameter counts / training metrics / inference rows)
# ---------------------------------------------------------------------------
def _count_param_categories(combo_id: str) -> Optional[Dict[str, Any]]:
    """Count a run's parameters per canonical block category.

    Builds the model on the ``meta`` device from ``tmp/runs/<combo_id>/config.yaml``
    (same recipe as ``full_tests/param_count_check.py``) so no weights are
    allocated. The result feeds ``results/param_metrics.json``:
    ``{model: {block: nparams, ..., "total": n}}`` using the 10 canonical
    categories from ``src/utils/param_categories.py``.

    Args:
        combo_id: Run/model identifier.

    Returns:
        The per-category counts, ``None`` when the config has no buildable
        ``model`` block (base-model/SBERT presets), or ``{"error": ...}`` when
        the build failed.
    """
    yaml_path = helpers.RUNS_DIR / combo_id / "config.yaml"
    if not yaml_path.exists():
        return None
    try:
        import torch  # lazy import: keeps the harness importable without torch

        from src.engine import build_model
        from src.training.config_loader import load_training_config
        from src.utils.param_categories import BLOCK_CATEGORIES, categorize_param
    except Exception as exc:  # noqa: BLE001
        logging.warning("Parameter counting unavailable: %s", exc)
        return None
    try:
        loaded = load_training_config(str(yaml_path))
        if loaded.model_config is None:
            return None
        if bool(loaded.model_config.embedding.dynamic.enabled):
            # No tokenizer on the meta-device counting path: hash embeddings
            # only need one string per id (content does not affect shapes).
            loaded.model_config.token_vocab = helpers.dynamic_embedding_vocab(
                loaded.model_config.dims.vocab_size
            )
        with torch.device("meta"):
            model = build_model(loaded.model_class, loaded.model_config)
        counts: Dict[str, Any] = {category: 0 for category in BLOCK_CATEGORIES}
        total = 0
        for name, parameter in model.named_parameters():
            numel = int(parameter.numel())
            counts[categorize_param(name)] += numel
            total += numel
        counts["total"] = total
        del model
        return counts
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _combo_training_config(combo_id: str) -> Dict[str, Any]:
    """Load the YAML config saved for a run (``tmp/runs/<combo_id>/config.yaml``)."""
    yaml_path = helpers.RUNS_DIR / combo_id / "config.yaml"
    if not yaml_path.exists():
        return {}
    try:
        return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}


def _combo_meta(combo_id: str) -> Tuple[str, str]:
    """Return ``(model_class, task)`` for a run from its saved YAML."""
    config = _combo_training_config(combo_id)
    model_class = str(
        config.get("model_class") or "frankenstein_text_encoder"
    ).strip().lower()
    task = str((config.get("training") or {}).get("task") or "mlm").strip().lower()
    return model_class, task


def _combo_vocab_size(combo_id: str, fallback: int) -> int:
    """Return the configured ``model.dims.vocab_size`` for a run (or ``fallback``)."""
    config = _combo_training_config(combo_id)
    dims = (config.get("model") or {}).get("dims") or {}
    try:
        return int(dims.get("vocab_size") or fallback)
    except Exception:  # noqa: BLE001
        return fallback


def _unknown_schema_keys(document: dict) -> List[str]:
    """Return the ``model:`` paths the assembled schema does not declare.

    The runtime config is nested only: a ``model.*`` key the schema does not
    know is logged as ``WARNING [schema] ... unknown key(s) ignored`` and
    dropped, which is how the harness silently trained a different model for a
    while (the flat ``use_bitnet``/``ffn_hidden_size`` keys were ignored, leaving
    BitNet and MoE on and a 2x FFN). The preflight uses the same validator the
    loader uses, but escalates these ``additionalProperties`` violations to hard
    errors — scoped to ``model:`` so unrelated legacy keys elsewhere (the vision
    task sub-blocks, ``vision_dataset.split``) keep their documented warn-and-drop
    behaviour instead of blocking a sweep.

    Args:
        document: The full combo YAML document (``model:``, ``training:``, ...).

    Returns:
        Sorted paths such as ``model.attention.titan``; empty when clean.
    """
    try:
        import jsonschema

        from src.training.config_loader import _root_schema
        from src.utils.schema_warnings import classify_schema_errors
    except Exception:  # noqa: BLE001 - validation is best effort
        return []
    try:
        validator_cls = getattr(
            jsonschema, "Draft202012Validator", jsonschema.Draft7Validator
        )
        _, extra_errors = classify_schema_errors(
            validator_cls(_root_schema()).iter_errors(document)
        )
    except Exception:  # noqa: BLE001
        return []
    paths = set()
    for error in extra_errors:
        path = ".".join(str(part) for part in getattr(error, "absolute_path", ()))
        if path == "model" or path.startswith("model."):
            paths.add(path)
    return sorted(paths)


# ---------------------------------------------------------------------------
# Pre-flight validation (fail fast before hours of training)
# ---------------------------------------------------------------------------
def validate_combos(combos: List[Tuple[str, dict]], *, deep: bool = True) -> List[str]:
    """Statically validate every generated combo before launching training.

    Catches the config classes of bugs that cost experiment 2 ~14h of wasted
    runs (a stale SSOG grid after ``--realonall`` raised ``max_length``, and
    optimizer keys the factory rejects), without building any model. Checks
    per combo:

    * dict level: ``dims`` invariants (``hidden_size % num_heads == 0``,
      non-empty ``layer_pattern`` — it is tiled to ``num_layers`` when
      shorter —, ``num_kv_heads`` divides ``num_heads``), ``ssog`` grid
      equals ``training.max_length`` (text
      model classes only; ViT derives it), and every
      ``<optimizer>-<param>`` key is accepted by the factory's
      ``OPTIMIZER_ALLOWED_KEYS``;
    * loader level (``deep=True``): the config is written to
      ``tmp/preflight/<combo>.yaml`` and run through the real
      :func:`src.training.config_loader.load_training_config` — the exact
      validation the CLI performs before training (enums, task/model-class
      rules, tokenizer blocks, flattening of the harness' flat model keys).

    Args:
        combos: List of ``(combo_id, config)`` pairs as built by ``main``.
        deep: Also run the config-loader validation (imports torch).

    Returns:
        List of human-readable error strings (empty when everything passes).
    """
    errors: List[str] = []

    try:
        from src.model.optimizer.factory import (
            OPTIMIZER_ALLOWED_KEYS as _ALLOWED,
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("Optimizer key validation unavailable: %s", exc)
        _ALLOWED = {}

    load_training_config = None
    if deep:
        try:
            from src.training.config_loader import load_training_config
        except Exception as exc:  # noqa: BLE001
            logging.warning("Config-loader validation unavailable: %s", exc)

    for combo_id, config in combos:
        dims = (config.get("model") or {}).get("dims") or {}
        hidden = int(dims.get("hidden_size") or 0)
        heads = int(dims.get("num_heads") or 0)
        pattern = dims.get("layer_pattern") or []
        num_layers = int(dims.get("num_layers") or 0)
        if hidden and heads and hidden % heads != 0:
            errors.append(f"{combo_id}: hidden_size {hidden} % num_heads {heads} != 0")
        if num_layers and not pattern:
            # The pattern is tiled (i % len) when shorter than num_layers, but
            # an empty pattern has nothing to tile.
            errors.append(f"{combo_id}: layer_pattern is empty")
        kv_heads = dims.get("num_kv_heads")
        if kv_heads and heads and heads % int(kv_heads) != 0:
            errors.append(
                f"{combo_id}: num_kv_heads {kv_heads} does not divide num_heads {heads}"
            )

        model_class = str(config.get("model_class") or "").strip().lower()
        training = config.get("training") or {}
        if model_class in TEXT_MODEL_CLASSES and "ssog" in pattern:
            ssog = ((config.get("model") or {}).get("attention") or {}).get("ssog") or {}
            grid = int(ssog.get("grid_h") or 0) * int(ssog.get("grid_w") or 0)
            max_length = int(training.get("max_length") or 32)
            if grid != max_length:
                errors.append(
                    f"{combo_id}: ssog grid {ssog.get('grid_h')}x{ssog.get('grid_w')} "
                    f"({grid} tokens) != training.max_length ({max_length})"
                )

        optimizer = training.get("optimizer") or {}
        optimizer_class = str(optimizer.get("optimizer_class") or "")
        # ``OPTIMIZER_ALLOWED_KEYS`` is a registry-backed mapping that raises
        # (not KeyError) on an unknown name, so the empty class of combos without
        # a ``training.optimizer`` block (audio, sbert) must not be looked up.
        allowed = _ALLOWED.get(optimizer_class) if optimizer_class else None
        if optimizer_class and allowed is not None:
            prefix = f"{optimizer_class}-"
            unknown = [
                key
                for key in (optimizer.get("parameters") or {})
                if key.startswith(prefix) and key[len(prefix):] not in allowed
            ]
            if unknown:
                errors.append(f"{combo_id}: optimizer {optimizer_class} rejects {unknown}")

        unknown_config_keys = _unknown_schema_keys(config)
        if unknown_config_keys:
            errors.append(
                f"{combo_id}: unknown config key(s) — the runtime config is "
                f"nested only, a flat key would be silently dropped: "
                f"{unknown_config_keys}"
            )

        if load_training_config is not None:
            try:
                preflight_dir = helpers.TMP_DIR / "preflight"
                preflight_dir.mkdir(parents=True, exist_ok=True)
                yaml_path = helpers.write_yaml(
                    config, preflight_dir / f"{combo_id}.yaml"
                )
                load_training_config(str(yaml_path))
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"{combo_id}: load_training_config failed: "
                    f"{type(exc).__name__}: {exc}"
                )

    return errors


# ---------------------------------------------------------------------------
# CPU/env plumbing and the startup banner
# ---------------------------------------------------------------------------
def _effective_threads(omp_threads: Any, workers: int) -> Optional[int]:
    """Resolve ``--omp-threads`` to a thread count, or ``None`` when invalid.

    Args:
        omp_threads: Raw flag value (``"auto"`` or an integer-like string).
        workers: Number of parallel pipeline workers.

    Returns:
        The resolved thread count (``>= 1``), or ``None`` when the value cannot
        be interpreted. ``"auto"`` divides the available cores by the worker
        count, never going below one thread.
    """
    cpu_count = os.cpu_count() or 1
    workers = max(int(workers or 1), 1)
    raw = str(omp_threads).strip().lower()
    if raw == "auto":
        return max(1, cpu_count // workers)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 1 else None


def _resolve_worker_env(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> Tuple[Dict[str, str], int, List[str]]:
    """Build the environment handed to every harness subprocess.

    The thread variables must exist in the child's environment, not in the
    parent: ``torch`` reads ``OMP_NUM_THREADS`` when it is imported, so setting
    it here (and passing it through ``subprocess.run(env=...)``) is what
    actually bounds a worker's thread pool.

    Args:
        args: Parsed CLI arguments.
        parser: The argument parser, used for ``parser.error`` on bad values.

    Returns:
        ``(env, threads, notes)`` where ``env`` is the complete override map,
        ``threads`` the resolved per-worker thread count and ``notes`` the
        human-readable remarks to log once logging is configured.
    """
    cpu_count = os.cpu_count() or 1
    notes: List[str] = []
    threads = _effective_threads(args.omp_threads, args.workers)
    if threads is None:
        parser.error("--omp-threads must be an integer >= 1 or 'auto'")
    if str(args.omp_threads).strip().lower() == "auto":
        notes.append(
            f"--omp-threads auto -> {threads} thread(s) per worker "
            f"(cpu_count {cpu_count} // workers {args.workers})"
        )

    env: Dict[str, str] = {name: str(threads) for name in THREAD_ENV_VARS}
    for item in args.env or []:
        key, separator, value = str(item).partition("=")
        key = key.strip()
        if not separator or not key:
            parser.error(f"--env expects KEY=VALUE, got {item!r}")
        if key in RESERVED_ENV_VARS:
            parser.error(
                f"--env cannot override {key}: it is part of the determinism "
                "contract (PYTHONHASHSEED=0, FRANKENSTEIN_TEST_SEED=SEED)"
            )
        env[key] = value

    total = args.workers * threads
    if total > cpu_count:
        notes.append(
            f"oversubscribed: {args.workers} workers x {threads} threads = "
            f"{total} threads over {cpu_count} cores"
        )
    return env, threads, notes


def _format_invocation(argv: Optional[List[str]]) -> str:
    """Return the exact command line that launched this run.

    Args:
        argv: Argument list passed to :func:`main` (``None`` means ``sys.argv``).

    Returns:
        Shell-quoted command line, starting with the script name.
    """
    if argv is None:
        argv = sys.argv[1:]
    program = sys.argv[0] if sys.argv else "full_tests/run_e2e.py"
    try:
        program = str(Path(program).resolve().relative_to(Path.cwd()))
    except (ValueError, OSError):
        program = str(program)
    return " ".join(shlex.quote(part) for part in [program, *map(str, argv)])


def _emit_run_banner(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: Optional[List[str]] = None,
) -> None:
    """Print the invocation and every effective parameter value to stdout.

    This is the first thing ``main`` does after parsing: the sweep log then
    always starts with the exact command and the value of every flag, including
    the ones left at their default (marked ``default``). That makes a long run
    reproducible from its own stdout, which matters when several sweeps are
    compared and only the log survives.

    Args:
        parser: The argument parser (used to recover the declared defaults).
        args: Parsed CLI arguments.
        argv: Raw argument list, for the command line reproduction.
    """
    try:
        defaults = vars(build_parser().parse_args([]))
        try:
            import psutil

            total_ram_gib = psutil.virtual_memory().total / (1024.0 ** 3)
            swap_gib = psutil.swap_memory().total / (1024.0 ** 3)
        except Exception:  # noqa: BLE001
            total_ram_gib = float("nan")
            swap_gib = float("nan")

        cpu_count = os.cpu_count() or 1
        threads = _effective_threads(getattr(args, "omp_threads", "1"), args.workers)
        thread_note = (
            f"{threads} per worker = {args.workers * threads} threads / {cpu_count} cores"
            if threads is not None
            else f"unresolved ({getattr(args, 'omp_threads', '?')!r})"
        )

        lines: List[str] = ["=" * 86, "run_e2e: effective parameters", "=" * 86]
        lines.append(f"command  : {_format_invocation(argv)}")
        lines.append(f"cwd      : {Path.cwd()}")
        lines.append(f"python   : {sys.executable}")
        lines.append(
            f"host     : {cpu_count} cores | {total_ram_gib:.1f} GiB RAM | "
            f"{swap_gib:.1f} GiB swap | {sys.platform}"
        )
        lines.append(f"threads  : {thread_note}")
        env_pairs = [f"{name}={value}" for name, value in _resolve_env_display(args)]
        lines.append(f"env      : {', '.join(env_pairs) if env_pairs else '(none)'}")
        lines.append("-" * 86)
        for action in parser._actions:  # noqa: SLF001 - intentional introspection
            if isinstance(action, argparse._HelpAction) or not action.option_strings:
                continue
            flag = action.option_strings[0]
            value = getattr(args, action.dest, None)
            default = defaults.get(action.dest)
            source = "default" if value == default else "explicit"
            shown = "(none)" if value is None else (
                ", ".join(map(str, value)) if isinstance(value, list) and value else str(value)
            )
            lines.append(f"  {flag:<40} {shown:<28} [{source}]")
        lines.append("=" * 86)
        print("\n".join(lines), flush=True)
    except Exception as exc:  # noqa: BLE001 - a banner must never abort a sweep
        print(f"run_e2e: argv={argv!r} (banner failed: {exc})", flush=True)


def _resolve_env_display(args: argparse.Namespace) -> List[Tuple[str, str]]:
    """Return the ``(name, value)`` thread/env pairs for the banner.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Sorted-by-name pairs; the thread variables are derived the same way as
        :func:`_resolve_worker_env` so the banner shows what will be exported.
    """
    try:
        env, _, _ = _resolve_worker_env(args, argparse.ArgumentParser(add_help=False))
    except SystemExit:
        return []
    return sorted(env.items())


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exhaustive end-to-end smoke tests for frankenstein-transformer.",
    )
    parser.add_argument(
        "--category",
        choices=["all", "attn", "opt", "norm", "pe", "transversal", "moe", "task", "batch_size", "vision", "vision_pe", "audio", "tokenizer", "realdata", "deploy"],
        default="all",
        help="Which sweep to run (default: all).",
    )
    parser.add_argument(
        "--logging-level",
        choices=["none", "info", "debug", "only-errors"],
        default="info",
        help="How much progress logging to emit to stdout (default: info).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N training combos (useful for a quick smoke run).",
    )
    parser.add_argument(
        "--only",
        default=None,
        metavar="PATTERN[,PATTERN...]",
        help=(
            "Run only the combos whose id matches one of the comma-separated "
            "patterns (case-insensitive substring; an exact id also matches). "
            "Example: --only opt_muon2 or --only muon2,apollo. Applied before "
            "--limit, --realonall and the global overrides."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help=(
            "Per-subprocess timeout in seconds for training, deploy, "
            "inference-eval, transformers-export and bitnet-gguf "
            "(default: 600). Use 0 to disable the timeout (wait forever)."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap the number of samples per combo. 0 means all samples (remove "
            "the cap). Applied to the streaming toy corpus, real/data-dir text "
            "corpora and SBERT; vision/audio combos have no sample-cap key. "
            "Default: keep each sweep's current value."
        ),
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override training.num_epochs (>= 1) for every combo, including "
            "the task sub-block and sbert.num_epochs. "
            "Default: keep each sweep's current value (1)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override training.batch_size (and sbert.batch_size) for every "
            "combo. Default: keep each sweep's current value (1 for the text "
            "sweeps, 2 for vision/realdata text, 16/4 for audio, 16 for SBERT)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Number of parallel worker threads for the training and deploy "
            f"combos (default: {DEFAULT_WORKERS}). Each worker launches its own "
            "CLI subprocess, so the host sees --workers concurrent processes."
        ),
    )
    parser.add_argument(
        "--omp-threads",
        default=DEFAULT_OMP_THREADS,
        metavar="N|auto",
        help=(
            "CPU threads per worker subprocess: sets OMP_NUM_THREADS and the "
            "OpenMP/BLAS family (MKL, OpenBLAS, NumExpr, VecLib). Values are "
            "never clamped. "
            f"Default: {DEFAULT_OMP_THREADS}. 'auto' = max(1, cpu_count // --workers)."
        ),
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra environment variable for every subprocess (train, deploy, "
            "inference-eval, transformers-export, bitnet-gguf). Repeatable, and "
            "applied after --omp-threads, so it wins. PYTHONHASHSEED and "
            "FRANKENSTEIN_TEST_SEED are reserved for the determinism contract."
        ),
    )
    parser.add_argument(
        "--skip-attn-pairs",
        action="store_true",
        help="Skip the expensive attention-pair sweep.",
    )
    parser.add_argument(
        "--skip-singles",
        action="store_true",
        help="Skip the single-attention sweep.",
    )
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Skip deploy/infer/export smoke tests.",
    )
    parser.add_argument(
        "--hf-quant-formats",
        default="q8_0,q4_0,i2_s",
        metavar="FORMATS",
        help=(
            "Comma-separated safetensors quantization formats exported for "
            "every text combo (default: q8_0,q4_0,i2_s)."
        ),
    )
    parser.add_argument(
        "--hf-quant-runtimes",
        default="dequantize_on_load,quantized_linear",
        metavar="RUNTIMES",
        help=(
            "Comma-separated load runtimes tested per format "
            "(default: dequantize_on_load,quantized_linear = v1,v2)."
        ),
    )
    parser.add_argument(
        "--skip-hf-quant",
        action="store_true",
        help="Skip the q8_0/q4_0/i2_s transformers exports and their accuracy evals.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=helpers.TOY_VOCAB_SIZE,
        help="Toy tokenizer vocab size (default: 256).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="cpu",
        help="Device to run training/deploy/infer on (default: cpu).",
    )
    parser.add_argument(
        "--gpu-temp-guard",
        action="store_true",
        help="Enable the GPU thermal guard during training (only meaningful on cuda).",
    )
    parser.add_argument(
        "--gpu-temp-pause-threshold-c",
        type=float,
        default=None,
        help="GPU temp (C) at which training pauses (default: CLI default).",
    )
    parser.add_argument(
        "--gpu-temp-resume-threshold-c",
        type=float,
        default=None,
        help="GPU temp (C) at which training resumes (default: CLI default).",
    )
    parser.add_argument(
        "--gpu-temp-critical-threshold-c",
        type=float,
        default=None,
        help="GPU temp (C) considered critical (default: CLI default).",
    )
    parser.add_argument(
        "--gpu-temp-poll-interval-seconds",
        type=float,
        default=None,
        help="GPU temp polling interval in seconds (default: CLI default).",
    )
    parser.add_argument(
        "--gpu-temp-checkpoint-grace-seconds",
        type=float,
        default=None,
        help="Grace period in seconds before a thermal checkpoint (default: CLI default).",
    )
    parser.add_argument(
        "--save-to",
        choices=["disk", "delete", "memory"],
        default="disk",
        help=(
            "Where the pipeline keeps its artifacts (default: disk). "
            "'disk' keeps run directories, checkpoints, deployed models and "
            "data as regular files. 'delete' still uses the disk but removes "
            "each combo's run directory, models and data right after its "
            "train/deploy/infer/export pipeline finishes, keeping only the "
            "results (metrics CSVs, inference.csv, param_metrics.json, "
            "results.json|md); the scratch tree is dropped at the end. "
            "'memory' keeps the run directory and the deployed models on a "
            "RAM-backed tmpfs (/dev/shm) and never writes harness artifacts to "
            "the disk; results stay in tmp/results."
        ),
    )
    parser.add_argument(
        "--keep-configs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Copy every combo's full generated YAML into "
            "tmp/results/configs/<combo_id>.yaml before any cleanup, so the "
            "exact config survives --save-to delete|memory (default: on). "
            "Pass --no-keep-configs to keep only the live run directory "
            "config.yaml under --save-to disk."
        ),
    )
    parser.add_argument(
        "--test-percentage",
        type=int,
        default=20,
        metavar="N",
        help=(
            "Hold out N%% of each combo's dataset from training and measure "
            "accuracy on it at the end of training (default: 20). The value is "
            "written to training.test_percentage; the trainer reports the "
            "held-out loss/accuracy/F1 in test_metrics.json, which the harness "
            "copies into results.json and the results.md test_acc/test_f1 "
            "columns. Classification tasks are stratified by default. "
            "0 disables the split."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing tmp/results directory instead of cleaning up first.",
    )
    parser.add_argument(
        "--realonall",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Train every category on real HuggingFace Hub data (default: on). "
            "Use --no-realonall for the fully-offline synthetic corpus."
        ),
    )
    parser.add_argument(
        "--infer-samples",
        type=int,
        default=None,
        help=(
            "Real samples compared per deployed artifact when writing "
            "results/inference.csv. Default: auto (32 when the held-out split "
            "has at least 32 examples, else 16 or the split size)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build every combo, apply --realonall/global overrides and "
            "validate the configs (real config loader, dims invariants, "
            "SSOG grid vs max_length, optimizer parameter keys, and any "
            "model.* key the assembled schema does not declare — the runtime "
            "config is nested only, so a flat legacy key would be silently "
            "dropped) WITHOUT training, deploying or exporting anything. Exits "
            "non-zero if any combo is invalid. Run this before any multi-hour "
            "sweep."
        ),
    )
    parser.add_argument(
        "--list-combos",
        action="store_true",
        help=(
            "Print the ids of the combos that would run (after --category, "
            "--only and --limit) and exit without training or validating."
        ),
    )
    parser.add_argument(
        "--keep-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep a combo's scratch artifacts (run dir, checkpoints, models, "
            "data) when any of its stages fails, even under --save-to "
            "delete|memory, so failures can be reproduced (default: on). "
            "Pass --no-keep-failed to restore the strict cleanup contract."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # First thing after parsing: print the exact invocation and every effective
    # parameter (defaults included) so the sweep log is self-describing.
    _emit_run_banner(parser, args, argv)

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.timeout < 0:
        parser.error("--timeout must be >= 0 (0 disables the timeout)")
    if args.max_samples is not None and args.max_samples < 0:
        parser.error("--max-samples must be >= 0 (0 means all samples)")
    if args.max_epochs is not None and args.max_epochs < 1:
        parser.error("--max-epochs must be >= 1")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if not 0 <= args.test_percentage <= 100:
        parser.error("--test-percentage must be between 0 and 100")
    if args.infer_samples is not None and args.infer_samples < 1:
        parser.error("--infer-samples must be >= 1")
    if args.save_to == "memory" and args.resume:
        parser.error("--resume is incompatible with --save-to memory (nothing persists)")

    hf_quant_formats = [
        item.strip().lower()
        for item in str(args.hf_quant_formats).split(",")
        if item.strip()
    ]
    hf_quant_runtimes = [
        item.strip().lower()
        for item in str(args.hf_quant_runtimes).split(",")
        if item.strip()
    ]
    for fmt in hf_quant_formats:
        if fmt not in {"q8_0", "q4_0", "i2_s"}:
            parser.error(
                f"unknown --hf-quant-formats entry {fmt!r}; expected q8_0, q4_0 or i2_s"
            )
    for runtime in hf_quant_runtimes:
        if runtime not in {"dequantize_on_load", "quantized_linear"}:
            parser.error(
                f"unknown --hf-quant-runtimes entry {runtime!r}; expected "
                "dequantize_on_load or quantized_linear"
            )
    args.hf_quant_format_list = hf_quant_formats
    args.hf_quant_runtime_list = hf_quant_runtimes

    # 0 (or any non-positive value) disables the timeout for ``subprocess.run``,
    # which expects ``None`` to wait forever.
    timeout_secs: Optional[int] = args.timeout if args.timeout > 0 else None

    log_level = {
        "none": logging.CRITICAL + 10,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "only-errors": logging.ERROR,
    }[args.logging_level]
    helpers.setup_logging(log_level)

    # Thread/env overrides for every subprocess (validated here so a bad --env
    # stops the run before any directory is touched).
    worker_env, omp_threads, env_notes = _resolve_worker_env(args, parser)
    for note in env_notes:
        logging.info("Threads: %s", note)

    # ``--save-to memory`` redirects every transient artifact to a RAM-backed
    # tmpfs; the result directory stays on disk. ``disk``/``delete`` use the
    # repository's tmp/ tree as before.
    scratch_root = helpers.FULL_TESTS_DIR / "tmp"
    if args.save_to == "memory":
        shm_root = Path("/dev/shm")
        if not shm_root.is_dir() or not os.access(shm_root, os.W_OK):
            parser.error(
                "--save-to memory requires a writable /dev/shm (tmpfs); "
                "use --save-to disk or --save-to delete on this machine"
            )
        scratch_root = shm_root / f"frankenstein-full-tests-{os.getpid()}"
        if scratch_root.exists():
            shutil.rmtree(scratch_root, ignore_errors=True)
    helpers.set_scratch_root(scratch_root)

    # Clean up the scratch directory unless --resume is specified
    if not args.resume and helpers.TMP_DIR.exists():
        logging.info("Cleaning up tmp directory: %s", helpers.TMP_DIR)
        shutil.rmtree(helpers.TMP_DIR)

    helpers.TMP_DIR.mkdir(parents=True, exist_ok=True)
    helpers.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    helpers.write_sitecustomize(helpers.TMP_DIR)
    tokenizer_path = helpers.ensure_toy_tokenizer(args.vocab_size)
    parquet_path = helpers.ensure_toy_parquet()
    runner, env_extra = helpers.resolve_runner(worker_env)
    python_runner = helpers.resolve_python_runner() if not args.skip_deploy else []

    # ``--realonall`` binds every sweep to the real corpora. The tokenizer
    # category trains from a local parquet directory, so materialize the real
    # text corpus once (falls back to the toy corpus when unavailable).
    real_text_dir: Optional[Path] = None
    if args.realonall:
        try:
            real_text_dir = helpers.ensure_real_text_parquet(
                dataset_name=REAL_TEXT_DATASET["dataset_name"],
                split=REAL_TEXT_DATASET["split"],
                text_column=REAL_TEXT_DATASET["text_column"],
                max_samples=REAL_TEXT_TOKENIZER_SAMPLES,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("Could not materialize the real text corpus: %s", exc)
            real_text_dir = None

    logging.info("=" * 60)
    logging.info("Frankenstein exhaustive end-to-end smoke tests")
    logging.info("Tokenizer: %s", tokenizer_path)
    logging.info("Parquet:   %s", parquet_path)
    logging.info("Runner:    %s", " ".join(runner))
    if python_runner:
        logging.info("Python:    %s", " ".join(python_runner))
    logging.info("Real data: %s", "enabled (--realonall)" if args.realonall else "disabled")
    logging.info("Workers:   %d", args.workers)
    logging.info("Threads:   %d per worker (%s)", omp_threads, ", ".join(
        f"{name}={value}" for name, value in sorted(env_extra.items())
    ))
    logging.info("Save to:   %s (scratch: %s)", args.save_to, helpers.TMP_DIR)
    logging.info(
        "Configs:   %s",
        f"saved to {helpers.RESULTS_CONFIGS_DIR}" if args.keep_configs else "not saved",
    )
    logging.info(
        "Timeout:   %s",
        "no limit" if timeout_secs is None else f"{timeout_secs}s",
    )
    logging.info(
        "Test split: %s",
        "disabled" if args.test_percentage == 0 else f"{args.test_percentage}% held out",
    )
    logging.info(
        "Overrides: max_samples=%s max_epochs=%s batch_size=%s",
        "per-sweep"
        if args.max_samples is None
        else ("all" if args.max_samples == 0 else args.max_samples),
        "per-sweep" if args.max_epochs is None else args.max_epochs,
        "per-sweep" if args.batch_size is None else args.batch_size,
    )
    logging.info("=" * 60)

    combos: List[Tuple[str, dict]] = []

    if args.category in {"all", "attn"}:
        if not args.skip_attn_pairs:
            combos.extend(make_attention_pair_configs(args.vocab_size))
        if not args.skip_singles:
            combos.extend(make_single_attention_configs(args.vocab_size))
    if args.category in {"all", "opt"}:
        combos.extend(make_optimizer_configs(args.vocab_size))
    if args.category in {"all", "norm"}:
        combos.extend(make_norm_configs(args.vocab_size))
    if args.category in {"all", "pe"}:
        combos.extend(make_positional_encoding_configs(args.vocab_size))
    if args.category in {"all", "transversal"}:
        combos.extend(make_transversal_configs(args.vocab_size))
    if args.category in {"all", "moe"}:
        combos.extend(make_moe_configs(args.vocab_size))
    if args.category in {"all", "task"}:
        combos.extend(make_task_configs(args.vocab_size))
    if args.category in {"all", "batch_size"}:
        combos.extend(make_batch_size_configs(args.vocab_size))
    if args.category in {"all", "vision"}:
        combos.extend(make_vision_configs(args.vocab_size))
    if args.category in {"all", "vision_pe"}:
        combos.extend(make_vision_pe_configs(args.vocab_size))
    if args.category in {"all", "audio"}:
        combos.extend(make_audio_configs(args.vocab_size))
    if args.category in {"all", "tokenizer"}:
        combos.extend(make_tokenizer_configs(args.vocab_size))
    if args.category in {"all", "realdata"}:
        combos.extend(make_real_dataset_configs(args.vocab_size))
        if 0 < args.timeout <= 600:
            logging.warning(
                "realdata combos download models/corpora from the HuggingFace "
                "Hub; consider --timeout 1800 (current: %ds)", args.timeout,
            )

    if args.only:
        patterns = [p.strip().lower() for p in args.only.split(",") if p.strip()]
        if not patterns:
            parser.error("--only requires at least one non-empty pattern")
        selected = [
            (combo_id, config)
            for combo_id, config in combos
            if any(pattern in combo_id.lower() for pattern in patterns)
        ]
        if not selected:
            available = ", ".join(combo_id for combo_id, _ in combos[:20])
            if len(combos) > 20:
                available += f", ... (+{len(combos) - 20} more)"
            parser.error(
                f"--only matched none of the {len(combos)} combos built for "
                f"category '{args.category}'. Patterns: {patterns}. "
                f"Available ids: {available}"
            )
        logging.info(
            "--only: selected %d of %d combos (%s)",
            len(selected),
            len(combos),
            ", ".join(patterns),
        )
        combos = selected

    if args.realonall:
        logging.info(
            "Binding every combo to real data: text=%s, vision=%s, audio=%s",
            REAL_TEXT_DATASET["dataset_name"],
            REAL_VISION_CLASSIFICATION_DATASET["dataset_name"],
            REAL_AUDIO_CLASSIFICATION_DATASET["dataset_name"],
        )
        combos = [
            (combo_id, apply_realonall(combo_id, config, real_text_dir))
            for combo_id, config in combos
        ]
        if 0 < args.timeout <= 600:
            logging.warning(
                "--realonall downloads models/corpora from the HuggingFace Hub "
                "on first use; consider --timeout 1800 (current: %ds)",
                args.timeout,
            )

    # Global overrides run after ``apply_realonall`` so the CLI flags win over
    # the real-data bindings (e.g. --max-samples beats the Borges 32-row cap).
    if (
        args.max_samples is not None
        or args.max_epochs is not None
        or args.batch_size is not None
    ):
        combos = [
            (
                combo_id,
                apply_global_overrides(
                    combo_id,
                    config,
                    max_samples=args.max_samples,
                    max_epochs=args.max_epochs,
                    batch_size=args.batch_size,
                ),
            )
            for combo_id, config in combos
        ]

    if args.limit is not None:
        combos = combos[: args.limit]

    if args.list_combos:
        for combo_id, _ in combos:
            print(combo_id)
        return 0

    # Pre-flight: fail fast on invalid configs instead of burning hours of
    # training on combos that crash at build time (experiment 2 lost 164 runs
    # to exactly this class of error).
    validation_errors = validate_combos(combos)
    if validation_errors:
        for error in validation_errors[:20]:
            logging.error("preflight: %s", error)
        if len(validation_errors) > 20:
            logging.error("preflight: ... and %d more", len(validation_errors) - 20)
    if args.dry_run:
        logging.info(
            "Dry run complete: %d combos validated, %d error(s)",
            len(combos),
            len(validation_errors),
        )
        return 2 if validation_errors else 0
    if validation_errors:
        logging.error(
            "Aborting: %d combo config(s) failed pre-flight validation. "
            "Fix them (or run with --dry-run to inspect) before a long sweep.",
            len(validation_errors),
        )
        return 2

    logging.info("Planning to run %d training combos", len(combos))
    logging.debug("Category=%s limit=%s skip_attn_pairs=%s skip_singles=%s skip_deploy=%s",
                  args.category, args.limit, args.skip_attn_pairs, args.skip_singles, args.skip_deploy)

    # 1 Hz host CPU/RAM series for the whole sweep: machine-wide usage plus the
    # summed RSS of every live child process. This is the artifact that answers
    # "how many workers fit here" after the run.
    telemetry_recorder = helpers.SystemTelemetryRecorder(
        workers=args.workers
    ).start()
    logging.info(
        "Host telemetry: %s (1 Hz)", helpers.RESULTS_SYSTEM_TELEMETRY_CSV_PATH
    )

    if args.workers > 1:
        if args.device == "cuda":
            logging.warning(
                "%d parallel workers share one CUDA device; concurrent training "
                "subprocesses may OOM (each one also runs its own thermal guard).",
                args.workers,
            )
        if args.realonall:
            logging.warning(
                "%d parallel workers download HuggingFace Hub datasets "
                "concurrently; populate a cold cache with one sequential run "
                "(--workers 1) first to avoid `datasets` cache races.",
                args.workers,
            )

    # ``run_training`` calls ``ensure_toy_tokenizer`` per combo (audio presets
    # carry their own vocab size). Pre-train every distinct tokenizer here so
    # parallel workers never race while writing the shared SPM model file.
    vocab_sizes = set()
    for _, combo_config in combos:
        try:
            vocab_sizes.add(int(combo_config["model"]["dims"]["vocab_size"]))
        except (KeyError, TypeError, ValueError):
            continue
    for vocab_size in sorted(vocab_sizes):
        try:
            helpers.ensure_toy_tokenizer(vocab_size)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Could not pre-build the tokenizer for vocab_size=%d: %s", vocab_size, exc)

    results: List[helpers.RunResult] = []
    param_metrics: Dict[str, Dict[str, Any]] = {}
    if args.resume and helpers.RESULTS_PARAM_METRICS_PATH.exists():
        try:
            loaded_metrics = json.loads(
                helpers.RESULTS_PARAM_METRICS_PATH.read_text(encoding="utf-8")
            )
            if isinstance(loaded_metrics, dict):
                param_metrics.update(loaded_metrics)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Could not load existing param_metrics.json: %s", exc)
    inference_rows: List[dict] = []

    train_kwargs = dict(
        runner=runner,
        env_extra=env_extra,
        timeout=timeout_secs,
        device=args.device,
        gpu_temp_guard=args.gpu_temp_guard,
        gpu_temp_pause_threshold_c=args.gpu_temp_pause_threshold_c,
        gpu_temp_resume_threshold_c=args.gpu_temp_resume_threshold_c,
        gpu_temp_critical_threshold_c=args.gpu_temp_critical_threshold_c,
        gpu_temp_poll_interval_seconds=args.gpu_temp_poll_interval_seconds,
        gpu_temp_checkpoint_grace_seconds=args.gpu_temp_checkpoint_grace_seconds,
        test_percentage=args.test_percentage,
    )

    def record_inference(rows: List[dict]) -> None:
        """Append inference-vs-real rows and rewrite ``results/inference.csv``."""
        if not rows:
            return
        timestamp = datetime.now().isoformat(timespec="seconds")
        for row in rows:
            row.setdefault("timestamp", timestamp)
        inference_rows.extend(rows)
        helpers.write_inference_csv(inference_rows)

    def deploy_and_evaluate(
        training_result: helpers.RunResult,
    ) -> Tuple[List[helpers.RunResult], List[dict]]:
        """Deploy every format, evaluate inference and export each checkpoint.

        Runs, in order, ``deploy --format standard`` + inference evaluation, the
        same pair for ``quantized``, then the plain Transformers export and the
        ``q8_0``/``q4_0``/``i2_s`` variants in both v1/v2 runtimes (text models
        only), each followed by an HF accuracy evaluation. Audio models have no
        Transformers equivalent, so that export must be rejected with the
        documented message and is scored ``OK`` when it is.

        Safe to call from a worker thread: every artifact path is keyed by
        ``combo_id`` (``tmp/deploy/<combo>_<fmt>``,
        ``tmp/inference_eval/<combo>_<fmt>.json``,
        ``tmp/transformers-export/<combo>/<label>``), so independent
        checkpoints never touch the same files.

        Args:
            training_result: A successful training ``RunResult`` with a checkpoint.

        Returns:
            ``(produced_results, inference_rows)`` for this checkpoint.
        """
        produced: List[helpers.RunResult] = []
        rows_out: List[dict] = []

        combo_id = training_result.combo_id
        checkpoint_path = Path(training_result.checkpoint_path)
        model_class, task = _combo_meta(combo_id)
        yaml_path = helpers.RUNS_DIR / combo_id / "config.yaml"
        run_dir = helpers.RUNS_DIR / combo_id
        test_split_path = run_dir / "test_split.json"
        if not test_split_path.exists():
            test_split_path = None
        infer_samples = (
            args.infer_samples
            if args.infer_samples is not None
            else helpers.effective_inference_samples(test_split_path)
        )
        vocab_size = _combo_vocab_size(combo_id, args.vocab_size)
        is_audio = model_class == "frankenstein_audio"
        is_text = model_class in TEXT_MODEL_CLASSES

        for fmt in ("standard", "quantized"):
            deploy_dir = helpers.TMP_DIR / "deploy" / f"{combo_id}_{fmt}"
            logging.info("Running deploy (%s) %s -> %s", fmt, combo_id, deploy_dir)
            deploy_result = helpers.run_deploy(
                checkpoint_path,
                deploy_dir,
                f"{combo_id}_deploy_{fmt}",
                runner,
                env_extra,
                fmt=fmt,
                timeout=timeout_secs,
                device=args.device,
                task=task,
            )
            produced.append(deploy_result)
            logging.info(
                "%s -> %s (%.1fs) %s",
                deploy_result.combo_id,
                deploy_result.status,
                deploy_result.duration_sec,
                deploy_result.notes,
            )
            if deploy_result.status != "OK":
                continue

            if is_text:
                # The eval tokenizes real text with the deployed tokenizer.
                helpers.copy_tokenizer_to_deploy_dir(
                    deploy_dir,
                    vocab_size,
                    run_dir=helpers.RUNS_DIR / combo_id,
                )
            eval_result, rows = helpers.run_inference_eval(
                deploy_dir,
                yaml_path,
                combo_id,
                python_runner,
                env_extra,
                model_class=model_class,
                task=task,
                deploy_format=fmt,
                samples=infer_samples,
                timeout=timeout_secs,
                device=args.device,
                test_split=test_split_path,
            )
            produced.append(eval_result)
            rows_out.extend(rows)
            logging.info(
                "%s -> %s (%.1fs) %s",
                eval_result.combo_id,
                eval_result.status,
                eval_result.duration_sec,
                eval_result.notes,
            )

        # transformers-export (needs the original YAML).
        if yaml_path.exists():
            hf_variants: List[Tuple[str, str, str]] = [
                ("none", "dequantize_on_load", "hf_none")
            ]
            if is_text and not args.skip_hf_quant:
                for runtime in args.hf_quant_runtime_list:
                    runtime_label = (
                        "v1" if runtime == "dequantize_on_load" else "v2"
                    )
                    for quant in args.hf_quant_format_list:
                        hf_variants.append(
                            (quant, runtime, f"hf_{quant}_{runtime_label}")
                        )

            for quant, quant_runtime, label in hf_variants:
                export_dir = (
                    helpers.TMP_DIR / "transformers-export" / combo_id / label
                )
                logging.info(
                    "Running transformers-export (%s/%s) %s -> %s",
                    quant,
                    quant_runtime,
                    combo_id,
                    export_dir,
                )
                export_result = helpers.run_transformers_export(
                    checkpoint_path,
                    yaml_path,
                    export_dir,
                    f"{combo_id}_export_{label}",
                    runner,
                    env_extra,
                    timeout=timeout_secs,
                    quant=quant,
                    quant_runtime=quant_runtime,
                )
                if is_audio:
                    combined = f"{export_result.stdout}\n{export_result.stderr}".lower()
                    rejected_as_documented = (
                        export_result.status == "FAILED"
                        and "frankenstein_audio" in combined
                        and "not currently exportable" in combined
                    )
                    if rejected_as_documented:
                        export_result.status = "OK"
                        export_result.notes = (
                            "audio transformers export rejected with the documented message"
                        )
                    else:
                        export_result.status = "FAILED"
                        export_result.notes = (
                            "audio transformers export was not rejected as documented "
                            f"(returncode={export_result.returncode})"
                        )
                produced.append(export_result)
                logging.info(
                    "%s -> %s (%.1fs) %s",
                    export_result.combo_id,
                    export_result.status,
                    export_result.duration_sec,
                    export_result.notes,
                )

                if (
                    export_result.status != "OK"
                    or not is_text
                    or (args.skip_hf_quant and label != "hf_none")
                ):
                    continue

                # The HF accuracy evaluator tokenizes real text with the
                # deployed tokenizer.
                helpers.copy_tokenizer_to_deploy_dir(
                    export_dir,
                    vocab_size,
                    run_dir=helpers.RUNS_DIR / combo_id,
                )
                eval_result, rows = helpers.run_inference_eval(
                    export_dir,
                    yaml_path,
                    combo_id,
                    python_runner,
                    env_extra,
                    model_class=model_class,
                    task=task,
                    deploy_format=label,
                    samples=infer_samples,
                    timeout=timeout_secs,
                    device=args.device,
                    engine="hf",
                    test_split=test_split_path,
                )
                produced.append(eval_result)
                rows_out.extend(rows)
                logging.info(
                    "%s -> %s (%.1fs) %s",
                    eval_result.combo_id,
                    eval_result.status,
                    eval_result.duration_sec,
                    eval_result.notes,
                )

        return produced, rows_out

    def run_combo_pipeline(
        combo_id: str, config: dict
    ) -> Tuple[List[helpers.RunResult], List[dict], Optional[Dict[str, Any]]]:
        """Train, deploy, infer and export one combo, sequentially, in one worker.

        The training metrics CSV and the per-block parameter counts are
        collected **before** any cleanup, so ``--save-to delete``/``memory`` can
        drop the whole run directory without losing results. With
        ``--save-to delete|memory`` the combo's transient artifacts are removed
        right after its pipeline finishes.

        Args:
            combo_id: Combo identifier.
            config: The combo's YAML config dict.

        Returns:
            ``(produced_results, inference_rows, param_counts)``. Nothing is
            written to the shared result artifacts here: the caller (main
            thread) persists them, so parallel workers never race.
        """
        produced: List[helpers.RunResult] = []
        rows: List[dict] = []
        param_counts: Optional[Dict[str, Any]] = None

        training_result = helpers.run_training(
            config=config,
            combo_id=combo_id,
            batch_size=config["training"].get("batch_size", 1),
            **train_kwargs,
        )
        produced.append(training_result)

        if training_result.status in ("OK", "GRAD_EXPLODED"):
            # Copy the metrics CSV and count parameters while the run dir exists.
            metrics_copy = helpers.copy_training_metrics(combo_id)
            if metrics_copy is not None:
                training_result.metrics_path = str(metrics_copy)
            param_counts = _count_param_categories(combo_id)

        if (
            not args.skip_deploy
            and training_result.status == "OK"
            and training_result.checkpoint_path
            and Path(training_result.checkpoint_path).exists()
        ):
            deploy_results, eval_rows = deploy_and_evaluate(training_result)
            produced.extend(deploy_results)
            rows.extend(eval_rows)

        if args.save_to in {"delete", "memory"}:
            combo_failed = any(result.status != "OK" for result in produced)
            if combo_failed and args.keep_failed:
                logging.warning(
                    "%s: keeping scratch artifacts for post-mortem "
                    "(--save-to %s, --keep-failed on)",
                    combo_id,
                    args.save_to,
                )
            else:
                helpers.cleanup_combo_artifacts(combo_id)

        return produced, rows, param_counts

    def persist_pipeline(
        combo_id: str,
        produced: List[helpers.RunResult],
        rows: List[dict],
        param_counts: Optional[Dict[str, Any]],
        config: dict,
    ) -> None:
        """Persist one finished pipeline and flush the shared artifacts.

        Main thread only: with parallel workers the shared result files are
        written once per completed combo, in completion order. The combo YAML
        is copied into ``results/configs/`` here (rather than in the worker)
        so all ``tmp/results`` writes stay on the main thread; the config dict
        already carries the overrides applied by the worker's training run.

        Args:
            combo_id: Combo identifier.
            produced: Results emitted by the pipeline (training + deploy/eval/export).
            rows: Inference-vs-real rows produced by the evaluation steps.
            param_counts: Per-block parameter counts, or ``None``.
            config: The combo's full config dict, as launched.
        """
        results.extend(produced)
        if param_counts is not None:
            param_metrics[combo_id] = param_counts
            helpers.write_param_metrics(param_metrics)
        record_inference(rows)
        if args.keep_configs:
            try:
                config_path = helpers.save_results_config(combo_id, config)
                logging.debug("Saved combo config to %s", config_path)
            except Exception as exc:  # noqa: BLE001
                logging.warning(
                    "Could not save the config YAML for %s: %s", combo_id, exc
                )
        helpers.write_results(results)

    def pipeline_worker(
        combo_id: str, config: dict
    ) -> Tuple[str, List[helpers.RunResult], List[dict], Optional[Dict[str, Any]]]:
        """Run one full pipeline in a worker thread; never writes shared files."""
        try:
            produced, rows, counts = run_combo_pipeline(combo_id, config)
        except Exception as exc:  # noqa: BLE001
            logging.exception("Pipeline worker for %s crashed", combo_id)
            produced = [
                helpers.RunResult(
                    category="training",
                    combo_id=combo_id,
                    status="FAILED",
                    duration_sec=0.0,
                    stdout="",
                    stderr=str(exc),
                    returncode=-2,
                    notes=f"worker exception: {exc}",
                )
            ]
            rows, counts = [], None
        return combo_id, produced, rows, counts

    def log_pipeline_result(
        combo_id: str, produced: List[helpers.RunResult], prefix: str = ""
    ) -> None:
        """Log one finished pipeline as a compact train/deploy/eval summary."""
        summary = ", ".join(f"{result.category}={result.status}" for result in produced)
        logging.info("%s%s -> %s", prefix, combo_id, summary or "no results")
        for result in produced:
            if result.status != "OK":
                logging.warning(
                    "  %s %s: %s", result.category, result.status, result.notes
                )

    if args.workers == 1:
        for idx, (combo_id, config) in enumerate(combos, start=1):
            logging.info(
                "[%d/%d] %s (train -> deploy -> infer -> export)",
                idx,
                len(combos),
                combo_id,
            )
            combo_id_out, produced, rows, counts = pipeline_worker(combo_id, config)
            persist_pipeline(combo_id_out, produced, rows, counts, config)
            log_pipeline_result(
                combo_id_out, produced, prefix=f"[{idx}/{len(combos)}] "
            )
    elif combos:
        logging.info(
            "Running %d combos with %d parallel pipeline workers",
            len(combos),
            args.workers,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(pipeline_worker, combo_id, config): (idx, combo_id, config)
                for idx, (combo_id, config) in enumerate(combos, start=1)
            }
            completed = 0
            for future in as_completed(futures):
                idx, combo_id, config = futures[future]
                combo_id_out, produced, rows, counts = future.result()
                persist_pipeline(combo_id_out, produced, rows, counts, config)
                completed += 1
                log_pipeline_result(
                    combo_id_out,
                    produced,
                    prefix=f"[{completed}/{len(combos)}] (combo {idx}) ",
                )

    # Dedicated BitNet+standard training + bitnet-gguf export, if not already covered.
    if not args.skip_deploy and (args.category in {"all", "transversal"}):
        model, training = _base_config(args.vocab_size, num_layers=2, num_loops=1)
        model = _deep_update(model, {
            "dims": {"layer_pattern": ["standard", "standard"]},
            "bitnet": {"enabled": True},
        })
        bitnet_id = "bitnet_std_attn_for_gguf"
        bitnet_config: Dict[str, Any] = {
            "model_class": "frankenstein_text_encoder",
            "model": model,
            "training": training,
        }
        if args.realonall:
            bitnet_config = apply_realonall(bitnet_id, bitnet_config, real_text_dir)
        logging.info("Running dedicated %s training", bitnet_id)
        bitnet_result = helpers.run_training(
            config=bitnet_config,
            combo_id=bitnet_id,
            batch_size=1,
            **train_kwargs,
        )
        produced: List[helpers.RunResult] = [bitnet_result]
        bitnet_counts: Optional[Dict[str, Any]] = None
        if bitnet_result.status in ("OK", "GRAD_EXPLODED"):
            # Copy metrics and record parameter counts for the dedicated run too,
            # before any cleanup removes the run directory.
            metrics_copy = helpers.copy_training_metrics(bitnet_id)
            if metrics_copy is not None:
                bitnet_result.metrics_path = str(metrics_copy)
            bitnet_counts = _count_param_categories(bitnet_id)
        if bitnet_result.status == "OK" and bitnet_result.checkpoint_path:
            yaml_path = helpers.RUNS_DIR / bitnet_id / "config.yaml"
            gguf_path = helpers.TMP_DIR / "bitnet-gguf" / f"{bitnet_id}.gguf"
            logging.info("Running bitnet-gguf %s -> %s", bitnet_id, gguf_path)
            produced.append(helpers.run_bitnet_gguf(
                Path(bitnet_result.checkpoint_path),
                yaml_path,
                gguf_path,
                f"{bitnet_id}_gguf",
                runner,
                env_extra,
                timeout=timeout_secs,
            ))
        if args.save_to in {"delete", "memory"}:
            if any(result.status != "OK" for result in produced) and args.keep_failed:
                logging.warning(
                    "%s: keeping scratch artifacts for post-mortem "
                    "(--save-to %s, --keep-failed on)",
                    bitnet_id,
                    args.save_to,
                )
            else:
                helpers.cleanup_combo_artifacts(bitnet_id)
        persist_pipeline(bitnet_id, produced, [], bitnet_counts, bitnet_config)
        log_pipeline_result(bitnet_id, produced)

    # Final write (ensures everything is flushed), then drop the scratch tree in
    # the modes whose contract is "only the results survive".
    telemetry_recorder.stop()
    helpers.write_results(results)
    if args.save_to in {"delete", "memory"}:
        helpers.cleanup_scratch()
    return helpers.print_summary(results)


if __name__ == "__main__":
    raise SystemExit(main())
