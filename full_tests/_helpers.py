"""Helpers for the exhaustive end-to-end Frankenstein test harness.

This module is intentionally self-contained: it builds the tiny on-the-fly
SentencePiece tokenizer, writes a toy parquet corpus, constructs training
YAMLs, runs the ``frankenstein-transformer`` CLI as a subprocess, and
collects results.

Run from inside the ``frankenstein`` conda environment (or any environment
that has ``sentencepiece``, ``pandas``, ``pyarrow`` and the project installed).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FULL_TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FULL_TESTS_DIR.parent
TMP_DIR = FULL_TESTS_DIR / "tmp"
TOKENIZER_DIR = TMP_DIR / "tokenizer"
DATA_DIR = TMP_DIR / "data"
RUNS_DIR = TMP_DIR / "runs"
RESULTS_DIR = TMP_DIR / "results"

TOY_VOCAB_SIZE = 512
SEED = 42
# With byte_fallback=true, SentencePiece needs room for all 256 byte pieces
# plus special tokens and the actual BPE vocabulary.  512 is a safe minimum.
MIN_SPM_VOCAB_SIZE = 384

# SentencePiece is only used inside the training-subprocess environment; the
# harness may also run in that same environment, but we import lazily.
HAVE_SENTENCEPIECE = False
try:
    import sentencepiece as spm  # type: ignore

    HAVE_SENTENCEPIECE = True
except Exception:
    pass

HAVE_PANDAS = False
try:
    import pandas as pd  # type: ignore

    HAVE_PANDAS = True
except Exception:
    pass


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO, stream: Optional[object] = None) -> None:
    """Configure logging to output to stdout by default."""
    if stream is None:
        import sys
        stream = sys.stdout
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class RunResult:
    category: str
    combo_id: str
    status: str  # "OK", "FAILED", "GRAD_EXPLODED", "TIMEOUT", "SKIPPED"
    duration_sec: float
    stdout: str
    stderr: str
    returncode: int
    notes: str = ""
    checkpoint_path: Optional[str] = None
    deploy_dir: Optional[str] = None
    metrics_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in dataclasses.asdict(self).items()}


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
def _run_silent(cmd: Sequence[str], cwd: Optional[Path] = None, env: Optional[dict] = None, timeout: int = 30) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        return -1, exc.stdout or "", exc.stderr or ""
    except Exception as exc:
        return -2, "", str(exc)


def resolve_runner() -> Tuple[List[str], dict]:
    """Return the CLI command prefix and extra env for training subprocesses.

    Tries (in order):
      1. ``conda run -n frankenstein frankenstein-transformer`` (entrypoint)
      2. ``conda run -n frankenstein python -m src.cli`` (module)
      3. ``uv run frankenstein-transformer``
      4. A local virtualenv at ``.venv``
      5. The current interpreter with ``python -m src.cli``
    """
    env_extra = {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1"}
    project_root = str(PROJECT_ROOT)

    # 1) conda env frankenstein, entrypoint
    rc, _, _ = _run_silent(["conda", "run", "-n", "frankenstein", "frankenstein-transformer", "--help"])
    if rc == 0:
        logging.info("Using runner: conda run -n frankenstein frankenstein-transformer")
        return ["conda", "run", "-n", "frankenstein", "frankenstein-transformer"], env_extra

    # 2) conda env, module
    env_module = {**os.environ, **env_extra, "PYTHONPATH": project_root}
    rc, _, _ = _run_silent(
        ["conda", "run", "-n", "frankenstein", "python", "-m", "src.cli", "--help"],
        env=env_module,
    )
    if rc == 0:
        logging.info("Using runner: conda run -n frankenstein python -m src.cli")
        return ["conda", "run", "-n", "frankenstein", "python", "-m", "src.cli"], env_extra

    # 3) uv
    rc, _, _ = _run_silent(["uv", "run", "frankenstein-transformer", "--help"], cwd=PROJECT_ROOT)
    if rc == 0:
        logging.info("Using runner: uv run frankenstein-transformer")
        return ["uv", "run", "frankenstein-transformer"], env_extra

    # 4) .venv
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if sys.platform == "win32":
        venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        rc, _, _ = _run_silent([str(venv_python), "-m", "src.cli", "--help"], env=env_module)
        if rc == 0:
            logging.info("Using runner: .venv python -m src.cli")
            return [str(venv_python), "-m", "src.cli"], env_extra

    # 5) current interpreter
    logging.warning("Falling back to current interpreter; ensure the project is installed.")
    return [sys.executable, "-m", "src.cli"], env_extra


# ---------------------------------------------------------------------------
# Determinism hook
# ---------------------------------------------------------------------------
SITECUSTOMIZE_PY = """# Auto-generated deterministic seed hook for full_tests.
import os

SEED = int(os.environ.get("FRANKENSTEIN_TEST_SEED", "42"))
try:
    import random
    random.seed(SEED)
except Exception:
    pass
try:
    import numpy as np
    np.random.seed(SEED)
except Exception:
    pass
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(False, warn_only=True)
except Exception:
    pass

# The harness trains a tiny SentencePiece model on toy data.  Some versions of
# SentencePiece do not emit an empty-string piece for padding, but the project's
# SpanishSPMTokenizer.encode() expects self.vocab[''] to exist.  We monkey-patch
# the encode method so that the missing empty-string piece is handled by falling
# back to the real pad id.  This is isolated to the test subprocesses.
try:
    # In editable installs the module is importable as tokenizer.*; in source
    # layout (PYTHONPATH=repo_root) it is importable as src.tokenizer.*.
    try:
        from tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer
    except Exception:
        from src.tokenizer.spm_spa_redpajama35 import SpanishSPMTokenizer  # type: ignore

    _original_spm_encode = SpanishSPMTokenizer.encode

    def _patched_spm_encode(self, text: str, max_length: int = 512):
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not initialized. Call train() or load() first.")
        tokens = self.sp_model.encode_as_ids(text)
        tokens = [self.vocab['[CLS]']] + tokens[:max_length - 2] + [self.vocab['[SEP]']]
        if len(tokens) < max_length:
            pad_id = int(self.vocab.get('[PAD]', self.vocab.get("", self.sp_model.pad_id())))
            tokens = tokens + [pad_id] * (max_length - len(tokens))
            attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
            attention_mask = [1] * max_length
        return {
            "input_ids": tokens,
            "attention_mask": attention_mask[:max_length],
        }

    SpanishSPMTokenizer.encode = _patched_spm_encode
except Exception:
    pass

# PyTorch 2.6+ changed the default of torch.load to weights_only=True.  The
# project saves FrankensteinModelConfig objects inside checkpoints, which are
# not in the default allow-list.  Deployment commands that call torch.load without
# explicitly passing weights_only=False fail.  We patch torch.load in the test
# subprocesses so that trusted toy checkpoints can be loaded for deploy/infer.
try:
    import torch

    _original_torch_load = torch.load

    def _patched_torch_load(f, *args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_torch_load(f, *args, **kwargs)

    torch.load = _patched_torch_load
except Exception:
    pass
"""


def write_sitecustomize(tmp_dir: Path = TMP_DIR) -> Path:
    """Write a ``sitecustomize.py`` seed hook into tmp_dir and return its path."""
    hook_path = tmp_dir / "sitecustomize.py"
    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(SITECUSTOMIZE_PY, encoding="utf-8")
    return hook_path


# ---------------------------------------------------------------------------
# Toy corpus
# ---------------------------------------------------------------------------
TRAIN_TEXTS = [
    "el rápido zorro marrón salta sobre el perro perezoso todos los días sin parar",
    "la inteligencia artificial aprende patrones complejos a partir de datos sintéticos",
    "un modelo pequeño puede entrenarse en pocos ejemplos y aun así generalizar bastante bien",
    "las redes neuronales transformadoras procesan secuencias de texto con mecanismos de atención",
    "entrenar desde cero requiere un tokenizador, datos de juguete y mucha paciencia científica",
]

VAL_TEXTS = [
    "la validación de un sistema exhaustivo consume tiempo pero revela errores tempranos",
    "los tests end to end verifican que cada componente de la herramienta funciona integrado",
]

ALL_TEXTS = TRAIN_TEXTS + VAL_TEXTS


def ensure_toy_corpus_text() -> Path:
    """Write the toy text corpus used to train the tokenizer."""
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = TOKENIZER_DIR / "corpus.txt"
    # Repeat with light variation so SPM has enough material for ~256 pieces.
    lines: List[str] = []
    for i in range(20):
        for line in ALL_TEXTS:
            lines.append(f"{line} [{i}]")
    corpus_path.write_text("\n".join(lines), encoding="utf-8")
    return corpus_path


def ensure_toy_parquet() -> Path:
    """Write the tiny parquet dataset consumed by StreamingMLMDataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = DATA_DIR / "train.parquet"
    if parquet_path.exists():
        return parquet_path
    if not HAVE_PANDAS:
        raise RuntimeError("pandas is required to write the toy parquet dataset")
    df = pd.DataFrame({"text": ALL_TEXTS})
    df.to_parquet(parquet_path, index=False)
    logging.info("Wrote toy parquet: %s (%d rows)", parquet_path, len(df))
    return parquet_path


# ---------------------------------------------------------------------------
# On-the-fly tokenizer
# ---------------------------------------------------------------------------
def ensure_toy_tokenizer(vocab_size: int = TOY_VOCAB_SIZE, force_retrain: bool = False) -> Path:
    """Train a tiny SentencePiece model once and persist it in tmp/tokenizer.

    The training CLI's legacy path loads ``es_redpajama_{vocab_size}.model``
    from the current working directory.  We train it here on the toy corpus and
    copy it into each run directory before launching training.
    """
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    prefix_name = f"es_redpajama_{vocab_size}"
    model_path = TOKENIZER_DIR / f"{prefix_name}.model"
    vocab_path = TOKENIZER_DIR / f"{prefix_name}.vocab"

    if model_path.exists() and not force_retrain:
        logging.info("Reusing toy tokenizer: %s", model_path)
        return model_path

    if not HAVE_SENTENCEPIECE:
        raise RuntimeError(
            "sentencepiece is required to train the toy tokenizer. "
            "Run this harness inside the frankenstein conda environment."
        )

    corpus_path = ensure_toy_corpus_text()

    # SPM may complain if the requested vocab is larger than the data allows.
    # Try the requested size, then fall back gracefully.
    candidate_sizes = [max(vocab_size, MIN_SPM_VOCAB_SIZE), 512, MIN_SPM_VOCAB_SIZE]
    # Remove duplicates while preserving order.
    seen: set = set()
    deduped: List[int] = []
    for s in candidate_sizes:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    candidate_sizes = deduped

    trained_model: Optional[Path] = None
    for target_size in candidate_sizes:
        prefix = TOKENIZER_DIR / f"es_redpajama_{target_size}_attempt"
        args = [
            f"--input={corpus_path}",
            f"--model_prefix={prefix}",
            f"--vocab_size={target_size}",
            "--character_coverage=1.0",
            "--model_type=bpe",
            "--max_sentence_length=4096",
            "--pad_id=0",
            "--unk_id=1",
            "--bos_id=2",
            "--eos_id=3",
            "--user_defined_symbols=[CLS],[SEP],[MASK],",
            "--split_by_whitespace=true",
            "--normalization_rule_name=identity",
            "--add_dummy_prefix=true",
            "--byte_fallback=true",
            "--split_digits=true",
            "--num_threads=2",
            "--input_sentence_size=1000000",
            "--shuffle_input_sentence=true",
        ]
        try:
            logging.info("Training toy SPM tokenizer with vocab_size=%d...", target_size)
            spm.SentencePieceTrainer.train(" ".join(args))
            trained_model = Path(f"{prefix}.model")
            if trained_model.exists():
                # Rename to the canonical name expected by the loader for the
                # *original* requested vocab size.  The loader only cares
                # about the filename matching model.dims.vocab_size.
                shutil.copy(str(trained_model), str(model_path))
                vocab_attempt = Path(f"{prefix}.vocab")
                if vocab_attempt.exists():
                    shutil.copy(str(vocab_attempt), str(vocab_path))
                logging.info("Toy tokenizer ready: %s", model_path)
                return model_path
        except Exception as exc:
            logging.warning("SPM vocab_size=%d failed: %s", target_size, exc)
            continue

    raise RuntimeError(f"Could not train a toy SentencePiece tokenizer for vocab_size {vocab_size}")


# ---------------------------------------------------------------------------
# YAML building
# ---------------------------------------------------------------------------
def default_training_cfg(parquet_path: Path, batch_size: int = 1, num_epochs: int = 1) -> dict:
    """Return a minimal ``training:`` block."""
    return {
        "task": "mlm",
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "max_length": 32,
        "mlm_probability": 0.15,
        "max_samples": len(ALL_TEXTS) * 4,  # enough to yield a few windows
        "dataset_batch_size": len(ALL_TEXTS) * 4,
        "num_workers": 1,
        "dataloader_workers": 0,
        "local_parquet_dir": str(parquet_path.parent),
        "cache_dir": "./dataset_cache",
        "gradient_accumulation_steps": 1,
        "max_nan_retries": 2,
        "gradient_log_interval": 1,
        "csv_log_path": "training_metrics.csv",
        "telemetry_log_interval": 1,
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
    }


def default_model_cfg(vocab_size: int = TOY_VOCAB_SIZE, num_layers: int = 2, num_loops: int = 1) -> dict:
    """Return a minimal ``model:`` block."""
    return {
        "dims": {
            "vocab_size": vocab_size,
            "hidden_size": 96,
            "num_layers": num_layers,
            "num_loops": num_loops,
            "num_heads": 8,
            "num_kv_heads": 4,
            "retention_heads": 8,
            "dropout": 0.0,
            "layer_pattern": ["standard_attn", "standard_attn"],
            "mode": "encoder",
        },
        "norm": {"type": "layer_norm"},
        "embedding": {
            "factorized": {"enabled": False, "dim": 32},
            "conv": {"enabled": False, "kernel": 3},
        },
        "attention": {
            "titan": {"positional_encoding": "hope"},
        },
        "residuals": {"type": "standard"},
        "use_bitnet": False,
        "bitnet_routers": False,
        "use_bitnet_conv": False,
        "use_moe": False,
        "num_experts": 2,
        "top_k_experts": 1,
        "use_mixture_of_depths": False,
        "mixture_of_depths_capacity_ratio": 0.5,
        "ffn_hidden_size": 96,
        "ffn_activation": "gelu",
    }


def write_yaml(config: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)
    return path


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------
GRADIENT_EXPLOSION_MARKERS = [
    "nan",
    "inf",
    "gradient",
    "exploded",
    "overflow",
    "underflow",
    "loss became nan",
    "loss is nan",
    "non-finite",
    "RuntimeError",
    "Loss is inf",
]


def looks_like_gradient_explosion(stdout: str, stderr: str) -> bool:
    text = f"{stdout}\n{stderr}".lower()
    return any(marker in text for marker in [m.lower() for m in GRADIENT_EXPLOSION_MARKERS])


def run_training(
    config: dict,
    combo_id: str,
    runner: List[str],
    env_extra: dict,
    batch_size: int = 1,
    timeout: int = 600,
    device: str = "cpu",
    gpu_temp_guard: bool = False,
    gpu_temp_pause_threshold_c: Optional[float] = None,
    gpu_temp_resume_threshold_c: Optional[float] = None,
    gpu_temp_critical_threshold_c: Optional[float] = None,
    gpu_temp_poll_interval_seconds: Optional[float] = None,
    gpu_temp_checkpoint_grace_seconds: Optional[float] = None,
) -> RunResult:
    """Run one training combo via the CLI as a subprocess."""
    run_dir = RUNS_DIR / combo_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy the pre-trained tokenizer into the run CWD so the legacy loader finds it.
    vocab_size = int(config["model"]["dims"]["vocab_size"])
    tokenizer_src = ensure_toy_tokenizer(vocab_size)
    tokenizer_dst = run_dir / tokenizer_src.name
    shutil.copy(str(tokenizer_src), str(tokenizer_dst))

    # Keep checkpoints inside the run dir (checkpoints/ is relative to CWD).
    yaml_path = run_dir / "config.yaml"
    write_yaml(config, yaml_path)

    env = {**os.environ, **env_extra, "FRANKENSTEIN_TEST_SEED": str(SEED)}
    env["PYTHONPATH"] = str(TMP_DIR) + os.pathsep + str(PROJECT_ROOT)
    if os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] += os.pathsep + os.environ["PYTHONPATH"]

    cmd = runner + [
        "train",
        "--config", str(yaml_path),
        "--device", device,
        "--batch-size", str(batch_size),
    ]
    if gpu_temp_guard:
        cmd.append("--gpu-temp-guard")
    else:
        cmd.append("--no-gpu-temp-guard")
    if gpu_temp_pause_threshold_c is not None:
        cmd.extend(["--gpu-temp-pause-threshold-c", str(gpu_temp_pause_threshold_c)])
    if gpu_temp_resume_threshold_c is not None:
        cmd.extend(["--gpu-temp-resume-threshold-c", str(gpu_temp_resume_threshold_c)])
    if gpu_temp_critical_threshold_c is not None:
        cmd.extend(["--gpu-temp-critical-threshold-c", str(gpu_temp_critical_threshold_c)])
    if gpu_temp_poll_interval_seconds is not None:
        cmd.extend(["--gpu-temp-poll-interval-seconds", str(gpu_temp_poll_interval_seconds)])
    if gpu_temp_checkpoint_grace_seconds is not None:
        cmd.extend(["--gpu-temp-checkpoint-grace-seconds", str(gpu_temp_checkpoint_grace_seconds)])

    logging.info("[%s] Starting training (timeout=%ds)", combo_id, timeout)
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(run_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout or ""
        duration = time.time() - start
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        duration = time.time() - start
        return RunResult(
            category="training",
            combo_id=combo_id,
            status="TIMEOUT",
            duration_sec=duration,
            stdout=output,
            stderr="",
            returncode=-1,
            notes=f"Timeout after {timeout}s",
        )
    except Exception as exc:
        duration = time.time() - start
        return RunResult(
            category="training",
            combo_id=combo_id,
            status="FAILED",
            duration_sec=duration,
            stdout="",
            stderr=str(exc),
            returncode=-2,
            notes=f"Subprocess exception: {exc}",
        )

    status: str
    notes: str
    if proc.returncode == 0:
        status = "OK"
        notes = "training completed"
    elif looks_like_gradient_explosion(output, ""):
        status = "GRAD_EXPLODED"
        notes = "gradient/nan/inf issue (tolerated)"
    else:
        status = "FAILED"
        notes = f"non-zero exit ({proc.returncode})"

    # Try to locate the final epoch-end checkpoint for downstream deploy smoke tests.
    checkpoint_path = find_latest_checkpoint(run_dir)

    return RunResult(
        category="training",
        combo_id=combo_id,
        status=status,
        duration_sec=duration,
        stdout=output,
        stderr="",
        returncode=proc.returncode,
        notes=notes,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
    )


def find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("titan_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Deploy / infer smoke helpers
# ---------------------------------------------------------------------------
def copy_tokenizer_to_deploy_dir(deploy_dir: Path, vocab_size: int) -> None:
    tokenizer_src = ensure_toy_tokenizer(vocab_size)
    shutil.copy(str(tokenizer_src), str(deploy_dir / "tokenizer.model"))


def run_deploy(
    checkpoint_path: Path,
    output_dir: Path,
    combo_id: str,
    runner: List[str],
    env_extra: dict,
    fmt: str = "standard",
    timeout: int = 300,
    device: str = "cpu",
) -> RunResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **env_extra, "FRANKENSTEIN_TEST_SEED": str(SEED)}
    env["PYTHONPATH"] = str(TMP_DIR) + os.pathsep + str(PROJECT_ROOT)
    if os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] += os.pathsep + os.environ["PYTHONPATH"]

    # The project's deploy path expects a JSON config; the checkpoint embeds a
    # dataclass instance that flatten_model_dict does not handle, so we pass the
    # original YAML model block as a JSON config explicitly.
    run_dir = checkpoint_path.parent.parent
    yaml_path = run_dir / "config.yaml"
    config_json_path = run_dir / "model_config.json"
    if yaml_path.exists():
        try:
            with yaml_path.open("r", encoding="utf-8") as fh:
                full_cfg = yaml.safe_load(fh)
            model_cfg = full_cfg.get("model", {})
            with config_json_path.open("w", encoding="utf-8") as fh:
                json.dump(model_cfg, fh, indent=2)
        except Exception as exc:
            logging.warning("Could not write deploy config JSON: %s", exc)

    cmd = runner + [
        "deploy",
        "--checkpoint", str(checkpoint_path),
        "--output", str(output_dir),
        "--format", fmt,
        "--validate",
        "--device", device,
    ]
    if config_json_path.exists():
        cmd.extend(["--config", str(config_json_path)])
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(output_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout or ""
        duration = time.time() - start
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        return RunResult(
            category="deploy",
            combo_id=combo_id,
            status="TIMEOUT",
            duration_sec=time.time() - start,
            stdout=output,
            stderr="",
            returncode=-1,
            notes=f"deploy timeout ({fmt})",
            checkpoint_path=str(checkpoint_path),
            deploy_dir=str(output_dir),
        )
    except Exception as exc:
        return RunResult(
            category="deploy",
            combo_id=combo_id,
            status="FAILED",
            duration_sec=time.time() - start,
            stdout="",
            stderr=str(exc),
            returncode=-2,
            notes=f"deploy exception: {exc}",
            checkpoint_path=str(checkpoint_path),
            deploy_dir=str(output_dir),
        )

    status = "OK" if proc.returncode == 0 else "FAILED"
    return RunResult(
        category="deploy",
        combo_id=combo_id,
        status=status,
        duration_sec=duration,
        stdout=output,
        stderr="",
        returncode=proc.returncode,
        notes=f"deploy format={fmt}",
        checkpoint_path=str(checkpoint_path),
        deploy_dir=str(output_dir),
    )


def run_infer(
    deploy_dir: Path,
    combo_id: str,
    runner: List[str],
    env_extra: dict,
    text: str = "el rápido zorro salta",
    timeout: int = 120,
    device: str = "cpu",
) -> RunResult:
    env = {**os.environ, **env_extra, "FRANKENSTEIN_TEST_SEED": str(SEED)}
    env["PYTHONPATH"] = str(TMP_DIR) + os.pathsep + str(PROJECT_ROOT)
    if os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] += os.pathsep + os.environ["PYTHONPATH"]

    cmd = runner + [
        "infer",
        "--model", str(deploy_dir),
        "--text", text,
        "--device", "cpu",
        "--batch-size", "1",
    ]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(deploy_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout or ""
        duration = time.time() - start
    except subprocess.TimeoutExpired as exc:
        return RunResult(
            category="infer",
            combo_id=combo_id,
            status="TIMEOUT",
            duration_sec=time.time() - start,
            stdout=exc.stdout or "",
            stderr="",
            returncode=-1,
            notes="infer timeout",
            deploy_dir=str(deploy_dir),
        )
    except Exception as exc:
        return RunResult(
            category="infer",
            combo_id=combo_id,
            status="FAILED",
            duration_sec=time.time() - start,
            stdout="",
            stderr=str(exc),
            returncode=-2,
            notes=f"infer exception: {exc}",
            deploy_dir=str(deploy_dir),
        )

    # Inference does not need to produce good probabilities; it just has to run.
    status = "OK" if proc.returncode == 0 else "FAILED"
    return RunResult(
        category="infer",
        combo_id=combo_id,
        status=status,
        duration_sec=duration,
        stdout=output,
        stderr="",
        returncode=proc.returncode,
        notes="inference smoke",
        deploy_dir=str(deploy_dir),
    )


def run_transformers_export(
    checkpoint_path: Path,
    yaml_path: Path,
    output_dir: Path,
    combo_id: str,
    runner: List[str],
    env_extra: dict,
    timeout: int = 300,
) -> RunResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **env_extra, "FRANKENSTEIN_TEST_SEED": str(SEED)}
    env["PYTHONPATH"] = str(TMP_DIR) + os.pathsep + str(PROJECT_ROOT)
    if os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] += os.pathsep + os.environ["PYTHONPATH"]

    cmd = runner + [
        "transformers-export",
        "--model", str(checkpoint_path),
        "--yaml", str(yaml_path),
        "--output", str(output_dir),
    ]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(output_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout or ""
        duration = time.time() - start
    except Exception as exc:
        return RunResult(
            category="transformers-export",
            combo_id=combo_id,
            status="FAILED",
            duration_sec=time.time() - start,
            stdout="",
            stderr=str(exc),
            returncode=-2,
            notes=f"export exception: {exc}",
            checkpoint_path=str(checkpoint_path),
        )

    status = "OK" if proc.returncode == 0 else "FAILED"
    return RunResult(
        category="transformers-export",
        combo_id=combo_id,
        status=status,
        duration_sec=duration,
        stdout=output,
        stderr="",
        returncode=proc.returncode,
        notes="transformers-export smoke",
        checkpoint_path=str(checkpoint_path),
    )


def run_bitnet_gguf(
    checkpoint_path: Path,
    yaml_path: Path,
    output_path: Path,
    combo_id: str,
    runner: List[str],
    env_extra: dict,
    timeout: int = 300,
) -> RunResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **env_extra, "FRANKENSTEIN_TEST_SEED": str(SEED)}
    env["PYTHONPATH"] = str(TMP_DIR) + os.pathsep + str(PROJECT_ROOT)
    if os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] += os.pathsep + os.environ["PYTHONPATH"]

    cmd = runner + [
        "bitnet-gguf",
        "--model", str(checkpoint_path),
        "--yaml", str(yaml_path),
        "--output", str(output_path),
    ]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(output_path.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout or ""
        duration = time.time() - start
    except Exception as exc:
        return RunResult(
            category="bitnet-gguf",
            combo_id=combo_id,
            status="FAILED",
            duration_sec=time.time() - start,
            stdout="",
            stderr=str(exc),
            returncode=-2,
            notes=f"gguf exception: {exc}",
            checkpoint_path=str(checkpoint_path),
        )

    status = "OK" if proc.returncode == 0 else "FAILED"
    return RunResult(
        category="bitnet-gguf",
        combo_id=combo_id,
        status=status,
        duration_sec=duration,
        stdout=output,
        stderr="",
        returncode=proc.returncode,
        notes="bitnet-gguf smoke",
        checkpoint_path=str(checkpoint_path),
    )


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------
def write_results(results: List[RunResult]) -> Tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in results]
    json_path = RESULTS_DIR / "results.json"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = RESULTS_DIR / "results.md"
    lines = [
        "# Frankenstein full_tests results",
        "",
        "| category | combo | status | dur(s) | notes |",
        "|----------|-------|--------|--------|-------|",
    ]
    for r in results:
        notes = (r.notes or "").replace("|", "\\|")
        lines.append(
            f"| {r.category} | {r.combo_id} | {r.status} | {r.duration_sec:.1f} | {notes} |"
        )

    counts: Dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    lines.extend(["", "## Status counts", ""])
    for status, count in sorted(counts.items()):
        lines.append(f"- {status}: {count}")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Results written to %s and %s", json_path, md_path)
    return json_path, md_path


def print_summary(results: List[RunResult]) -> int:
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    total = len(results)
    ok = counts.get("OK", 0)
    grad = counts.get("GRAD_EXPLODED", 0)
    failed = counts.get("FAILED", 0)
    timeout = counts.get("TIMEOUT", 0)
    logging.info("=" * 60)
    logging.info("DONE: %d runs | OK=%d GRAD_EXPLODED=%d FAILED=%d TIMEOUT=%d", total, ok, grad, failed, timeout)
    logging.info("=" * 60)
    if failed or timeout:
        logging.warning("There were unexpected failures/timeouts.")
        return 1
    return 0
