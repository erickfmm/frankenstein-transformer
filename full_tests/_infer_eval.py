"""Inference-vs-real evaluation for deployed Frankenstein artifacts.

The end-to-end harness deploys every successful training run and then runs this
helper as a subprocess to compare the deployed artifact's predictions against
ground-truth targets from the *real* dataset named in the run's YAML config.

Task coverage:

* ``mlm`` — mask one real token per sample and compare the model's top-1/top-5
  token against the masked (real) token.
* ``causal_lm`` — teacher-forced next-token prediction on real text: first-token
  match and token-level accuracy against the real continuation.
* ``text_classification`` — argmax class vs the real label.
* ``classification`` (ViT) — argmax class vs the real image label.
* ``audio_classification`` — argmax class vs the real audio label.
* ``segmentation`` / ``patch_prediction`` / ``asr`` — no label-match metric;
  a single ``SKIPPED_NO_INFER`` row is emitted.

Rows are written as a JSON list to ``--out`` and are appended by the harness to
``tmp/results/inference.csv``. The helper never raises on a per-sample failure:
it records ``FAILED``/``SKIPPED_*`` rows instead so the harness keeps running.

Usage (normally invoked by ``_helpers.run_inference_eval``):

    python full_tests/_infer_eval.py \
        --deploy-dir full_tests/tmp/deploy/<combo>_standard \
        --run-config full_tests/tmp/runs/<combo>/config.yaml \
        --combo-id <combo> --model-class frankenstein_text_encoder \
        --task mlm --deploy-format standard --samples 4 --device cpu \
        --out full_tests/tmp/inference_eval/<combo>_standard.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deploy.inference import FrankensteinInference  # noqa: E402
from src.engine import build_model  # noqa: E402
from src.model.config import FrankensteinModelConfig  # noqa: E402
from src.training.config_loader import load_training_config  # noqa: E402
from src.utils.config_flatten import flatten_model_dict  # noqa: E402

TEXT_MODEL_CLASSES = {
    "frankenstein_text_encoder",
    "frankenstein_text_decoder",
}
LOGGER = logging.getLogger("infer_eval")


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------
def _truncate(value: Any, limit: int = 160) -> str:
    """Return a single-line, length-bounded string representation."""
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    if len(text) > limit:
        text = f"{text[: limit - 3]}..."
    return text


def _row(
    *,
    combo_id: str,
    model_class: str,
    task: str,
    deploy_format: str,
    sample_id: int,
    status: str,
    input_excerpt: str = "",
    predicted: Any = "",
    real: Any = "",
    metric_name: str = "",
    metric_value: Any = "",
    notes: str = "",
) -> Dict[str, Any]:
    """Build one inference-comparison row with the harness' CSV schema."""
    return {
        "combo_id": combo_id,
        "model_class": model_class,
        "task": task,
        "deploy_format": deploy_format,
        "sample_id": sample_id,
        "input_excerpt": _truncate(input_excerpt),
        "predicted": predicted,
        "real": real,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "status": status,
        "notes": notes,
    }


def _skip_row(args: argparse.Namespace, status: str, notes: str) -> Dict[str, Any]:
    return _row(
        combo_id=args.combo_id,
        model_class=args.model_class,
        task=args.task,
        deploy_format=args.deploy_format,
        sample_id=0,
        status=status,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_raw_model(deploy_dir: Path, model_class: str, device: str) -> torch.nn.Module:
    """Build a ViT/audio model from the deploy ``config.json`` and load weights.

    ``FrankensteinInference`` only auto-detects encoder/decoder, so vision and
    audio artifacts are rebuilt explicitly from the recorded ``model_class``.
    """
    config_dict = json.loads((deploy_dir / "config.json").read_text(encoding="utf-8"))
    config_dict.pop("inference", None)
    config = FrankensteinModelConfig(**flatten_model_dict(config_dict))
    model = build_model(model_class, config)

    model_files = sorted(deploy_dir.glob("model*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model*.pt file found in {deploy_dir}")
    model_path = model_files[0]
    if "quantized" in model_path.name:
        from src.deploy.quantization import load_quantized_checkpoint

        load_quantized_checkpoint(str(model_path), model)
    else:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


def _decode(tokenizer: Any, ids: List[int]) -> str:
    if tokenizer is None or not hasattr(tokenizer, "decode"):
        return " ".join(str(int(i)) for i in ids)
    try:
        decoded = tokenizer.decode(ids)
        if isinstance(decoded, list):
            return " ".join(str(item) for item in decoded)
        return str(decoded)
    except Exception:  # noqa: BLE001
        return " ".join(str(int(i)) for i in ids)


def _text_loader(
    loaded: Any,
    tokenizer: Any,
    device: str,
    batch_size: int,
    task: str,
):
    from src.training.text_dataset_builder import build_text_dataloader_from_config

    return build_text_dataloader_from_config(
        loaded, tokenizer, device, cli_batch_size=batch_size, task=task
    )


def _load_text_tokenizer(
    run_config: Dict[str, Any],
    run_config_path: Path,
    deploy_dir: Path,
) -> Any:
    """Load the HuggingFace tokenizer used to train the run's text_dataset.

    ``build_text_dataloader_from_config`` calls the tokenizer directly, so it
    needs a real ``AutoTokenizer`` (the deployed SPM adapter is not callable).
    Sources, in order: the configured ``hf_repo`` id, the artifacts saved next
    to the run's checkpoints, and finally the deploy directory itself.

    Returns:
        An ``AutoTokenizer`` or ``None`` when the run never used HF text data.
    """
    try:
        from transformers import AutoTokenizer
    except Exception:  # noqa: BLE001
        return None

    tokenizer_cfg = dict(run_config.get("tokenizer") or {})
    candidates: List[str] = []
    if str(tokenizer_cfg.get("source") or "").strip().lower() == "hf_repo":
        name = str(tokenizer_cfg.get("name_or_path") or "").strip()
        if name:
            candidates.append(name)

    run_dir = run_config_path.parent
    for pattern in ("trained_tokenizer", "checkpoints/*/tokenizer"):
        for tokenizer_dir in sorted(run_dir.glob(pattern)):
            if (tokenizer_dir / "tokenizer.json").exists() or (
                tokenizer_dir / "tokenizer_config.json"
            ).exists():
                candidates.append(str(tokenizer_dir))
    if (deploy_dir / "tokenizer.json").exists():
        candidates.append(str(deploy_dir))

    for candidate in candidates:
        try:
            return AutoTokenizer.from_pretrained(candidate)
        except Exception:  # noqa: BLE001
            continue
    return None


# ---------------------------------------------------------------------------
# Task evaluators
# ---------------------------------------------------------------------------
def eval_text_mlm(
    args: argparse.Namespace,
    engine: FrankensteinInference,
    loaded: Any,
    device: str,
    tokenizer: Any,
) -> List[Dict[str, Any]]:
    """Mask real tokens and compare top-1/top-5 predictions with the targets."""
    if tokenizer is None:
        return [_skip_row(args, "SKIPPED_NO_DATASET", "no HuggingFace tokenizer for text data")]
    loader = _text_loader(loaded, tokenizer, device, max(1, args.samples), "mlm")
    if loader is None:
        return [_skip_row(args, "SKIPPED_NO_DATASET", "no text_dataset configured")]

    rows: List[Dict[str, Any]] = []
    sample = 0
    for batch in loader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")
        for row_idx in range(input_ids.shape[0]):
            if len(rows) >= args.samples:
                break
            valid = int(attention_mask[row_idx].sum()) if attention_mask is not None else int(input_ids.shape[1])
            seq = input_ids[row_idx, :valid].unsqueeze(0).to(device)
            target_labels = labels[row_idx, :valid]
            masked = (target_labels != -100).nonzero(as_tuple=True)[0]
            if masked.numel() == 0:
                continue
            position = int(masked[0])
            target = int(target_labels[position])
            with torch.no_grad():
                logits = engine.model(seq)
            if logits.dim() != 3:
                return [_skip_row(args, "SKIPPED_NO_INFER", "model did not return token logits")]
            top_k = min(5, int(logits.shape[-1]))
            top_ids = torch.topk(logits[0, position], k=top_k).indices.tolist()
            rows.append(
                _row(
                    combo_id=args.combo_id,
                    model_class=args.model_class,
                    task=args.task,
                    deploy_format=args.deploy_format,
                    sample_id=sample,
                    status="OK",
                    input_excerpt=_decode(tokenizer, seq[0].tolist()),
                    predicted=f"{_decode(tokenizer, [top_ids[0]])} (id={top_ids[0]})",
                    real=f"{_decode(tokenizer, [target])} (id={target})",
                    metric_name="mlm_top1",
                    metric_value=1.0 if top_ids[0] == target else 0.0,
                    notes=f"mask_position={position} top5_hit={target in top_ids}",
                )
            )
            sample += 1
        if len(rows) >= args.samples:
            break
    if not rows:
        return [_skip_row(args, "SKIPPED_NO_DATASET", "no masked positions produced by text_dataset")]
    return rows


def eval_text_causal_lm(
    args: argparse.Namespace,
    engine: FrankensteinInference,
    loaded: Any,
    device: str,
    tokenizer: Any,
) -> List[Dict[str, Any]]:
    """Teacher-forced next-token prediction on real text."""
    if tokenizer is None:
        return [_skip_row(args, "SKIPPED_NO_DATASET", "no HuggingFace tokenizer for text data")]
    loader = _text_loader(loaded, tokenizer, device, max(1, args.samples), "causal_lm")
    if loader is None:
        return [_skip_row(args, "SKIPPED_NO_DATASET", "no text_dataset configured")]

    rows: List[Dict[str, Any]] = []
    sample = 0
    for batch in loader:
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        for row_idx in range(input_ids.shape[0]):
            if len(rows) >= args.samples:
                break
            valid = int(attention_mask[row_idx].sum()) if attention_mask is not None else int(input_ids.shape[1])
            if valid < 3:
                continue
            seq = input_ids[row_idx, :valid].unsqueeze(0).to(device)
            with torch.no_grad():
                logits = engine.model(seq)
            if logits.dim() != 3:
                return [_skip_row(args, "SKIPPED_NO_INFER", "model did not return token logits")]
            predicted = torch.argmax(logits[0, :-1], dim=-1)
            target = seq[0, 1:]
            first_pred = int(predicted[0])
            first_real = int(target[0])
            accuracy = float((predicted == target).float().mean().item())
            rows.append(
                _row(
                    combo_id=args.combo_id,
                    model_class=args.model_class,
                    task=args.task,
                    deploy_format=args.deploy_format,
                    sample_id=sample,
                    status="OK",
                    input_excerpt=_decode(tokenizer, seq[0].tolist()),
                    predicted=_decode(tokenizer, [first_pred]),
                    real=_decode(tokenizer, [first_real]),
                    metric_name="causal_next_token",
                    metric_value=1.0 if first_pred == first_real else 0.0,
                    notes=f"first_token_match={first_pred == first_real} token_accuracy={accuracy:.4f}",
                )
            )
            sample += 1
        if len(rows) >= args.samples:
            break
    if not rows:
        return [_skip_row(args, "SKIPPED_NO_DATASET", "no usable causal-LM windows")]
    return rows


def eval_text_classification(
    args: argparse.Namespace,
    engine: FrankensteinInference,
    loaded: Any,
    device: str,
    tokenizer: Any,
) -> List[Dict[str, Any]]:
    """Compare argmax class logits with the real label of real text rows."""
    if tokenizer is None:
        return [_skip_row(args, "SKIPPED_NO_DATASET", "no HuggingFace tokenizer for text data")]
    loader = _text_loader(
        loaded, tokenizer, device, max(1, args.samples), "text_classification"
    )
    if loader is None:
        return [_skip_row(args, "SKIPPED_NO_DATASET", "no text_dataset configured")]

    rows: List[Dict[str, Any]] = []
    sample = 0
    for batch in loader:
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        if labels is None:
            return [_skip_row(args, "SKIPPED_NO_INFER", "text_dataset did not expose labels")]
        attention_mask = batch.get("attention_mask")
        for row_idx in range(input_ids.shape[0]):
            if len(rows) >= args.samples:
                break
            valid = int(attention_mask[row_idx].sum()) if attention_mask is not None else int(input_ids.shape[1])
            seq = input_ids[row_idx, :valid].unsqueeze(0).to(device)
            with torch.no_grad():
                logits = engine.model(seq)
            if logits.dim() != 2:
                return [_skip_row(args, "SKIPPED_NO_INFER", "model has no classification head")]
            predicted = int(logits.argmax(dim=-1)[0])
            real = int(labels[row_idx])
            rows.append(
                _row(
                    combo_id=args.combo_id,
                    model_class=args.model_class,
                    task=args.task,
                    deploy_format=args.deploy_format,
                    sample_id=sample,
                    status="OK",
                    input_excerpt=_decode(tokenizer, seq[0].tolist()),
                    predicted=f"label={predicted}",
                    real=f"label={real}",
                    metric_name="text_classification_accuracy",
                    metric_value=1.0 if predicted == real else 0.0,
                )
            )
            sample += 1
        if len(rows) >= args.samples:
            break
    if not rows:
        return [_skip_row(args, "SKIPPED_NO_DATASET", "no labelled text rows")]
    return rows


def eval_vit_classification(
    args: argparse.Namespace,
    model: torch.nn.Module,
    run_config: Dict[str, Any],
    device: str,
) -> List[Dict[str, Any]]:
    """Compare ViT classification logits with the real image labels."""
    from src.training.vision_dataset import ImageDataset

    dataset_config = dict(run_config.get("vision_dataset") or {})
    image_config = dict(run_config.get("image") or {})
    if not dataset_config.get("dataset_name") and not dataset_config.get("dataset_dir"):
        return [_skip_row(args, "SKIPPED_NO_DATASET", "no vision_dataset configured")]

    split = str(dataset_config.get("split") or "train")
    dataset = ImageDataset(dataset_config, image_config, "classification", split=split)
    if len(dataset) == 0:
        return [_skip_row(args, "SKIPPED_NO_DATASET", "vision_dataset returned no rows")]

    rows: List[Dict[str, Any]] = []
    for index in range(min(args.samples, len(dataset))):
        item = dataset[index]
        pixel_values = item["pixel_values"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(pixel_values, task="classification")
        predicted = int(logits.argmax(dim=-1)[0])
        real = int(item["labels"])
        rows.append(
            _row(
                combo_id=args.combo_id,
                model_class=args.model_class,
                task=args.task,
                deploy_format=args.deploy_format,
                sample_id=index,
                status="OK",
                input_excerpt=f"{dataset_config.get('dataset_name')} [{split}] #{index}",
                predicted=f"label={predicted}",
                real=f"label={real}",
                metric_name="vit_classification_accuracy",
                metric_value=1.0 if predicted == real else 0.0,
            )
        )
    return rows


def eval_audio_classification(
    args: argparse.Namespace,
    model: torch.nn.Module,
    loaded: Any,
    run_config: Dict[str, Any],
    device: str,
) -> List[Dict[str, Any]]:
    """Compare audio classification logits with the real audio labels."""
    from itertools import islice

    from src.training.audio_dataset import build_audio_dataloader_from_config

    dataset_config = dict(run_config.get("audio_dataset") or {})
    if not dataset_config.get("dataset_name") and not dataset_config.get("dataset_dir"):
        return [_skip_row(args, "SKIPPED_NO_DATASET", "no audio_dataset configured")]

    loader = build_audio_dataloader_from_config(
        loaded,
        resolved_device=device,
        cli_batch_size=1,
        task="audio_classification",
        shuffle=False,
        split=str(dataset_config.get("split") or "train"),
    )
    rows: List[Dict[str, Any]] = []
    for index, batch in enumerate(islice(loader, args.samples)):
        waveform = batch["waveform"].to(device)
        with torch.no_grad():
            logits = model(waveform, task="audio_classification")
        if logits.dim() != 2:
            return [_skip_row(args, "SKIPPED_NO_INFER", "model has no audio classification head")]
        predicted = int(logits.argmax(dim=-1)[0])
        real = int(batch["labels"].reshape(-1)[0])
        rows.append(
            _row(
                combo_id=args.combo_id,
                model_class=args.model_class,
                task=args.task,
                deploy_format=args.deploy_format,
                sample_id=index,
                status="OK",
                input_excerpt=f"{dataset_config.get('dataset_name')} #{index}",
                predicted=f"label={predicted}",
                real=f"label={real}",
                metric_name="audio_classification_accuracy",
                metric_value=1.0 if predicted == real else 0.0,
                notes=f"waveform_samples={int(waveform.reshape(waveform.shape[0], -1).shape[-1])}",
            )
        )
    if not rows:
        return [_skip_row(args, "SKIPPED_NO_DATASET", "audio_dataset returned no rows")]
    return rows


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference-vs-real evaluation.")
    parser.add_argument("--deploy-dir", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--combo-id", required=True)
    parser.add_argument("--model-class", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--deploy-format", default="standard")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    args = build_parser().parse_args(argv)
    args.samples = max(1, int(args.samples))
    device = _resolve_device(args.device)

    deploy_dir = Path(args.deploy_dir)
    run_config_path = Path(args.run_config)
    run_config: Dict[str, Any] = {}
    if run_config_path.exists():
        run_config = yaml.safe_load(run_config_path.read_text(encoding="utf-8")) or {}

    model_class = str(args.model_class or "").strip().lower()
    task = str(args.task or "").strip().lower()

    rows: List[Dict[str, Any]] = []
    try:
        if task in {"segmentation", "patch_prediction", "asr"}:
            rows = [
                _skip_row(
                    args,
                    "SKIPPED_NO_INFER",
                    f"task '{task}' has no label-match inference metric",
                )
            ]
        elif model_class in TEXT_MODEL_CLASSES:
            engine = FrankensteinInference(str(deploy_dir), device=device)
            loaded = load_training_config(str(run_config_path))
            tokenizer = _load_text_tokenizer(run_config, run_config_path, deploy_dir)
            if task == "causal_lm":
                rows = eval_text_causal_lm(args, engine, loaded, device, tokenizer)
            elif task == "text_classification":
                rows = eval_text_classification(args, engine, loaded, device, tokenizer)
            else:
                rows = eval_text_mlm(args, engine, loaded, device, tokenizer)
        elif model_class == "frankenstein_vit":
            model = _load_raw_model(deploy_dir, model_class, device)
            if task == "classification":
                rows = eval_vit_classification(args, model, run_config, device)
            else:
                rows = [
                    _skip_row(
                        args,
                        "SKIPPED_NO_INFER",
                        f"vision task '{task}' has no label-match inference metric",
                    )
                ]
        elif model_class == "frankenstein_audio":
            model = _load_raw_model(deploy_dir, model_class, device)
            if task == "audio_classification":
                loaded = load_training_config(str(run_config_path))
                rows = eval_audio_classification(args, model, loaded, run_config, device)
            else:
                rows = [
                    _skip_row(
                        args,
                        "SKIPPED_NO_INFER",
                        f"audio task '{task}' has no label-match inference metric",
                    )
                ]
        else:
            rows = [_skip_row(args, "SKIPPED_NO_INFER", f"unsupported model_class '{model_class}'")]
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Inference eval failed: %s", exc)
        traceback.print_exc()
        rows = [
            _row(
                combo_id=args.combo_id,
                model_class=args.model_class,
                task=args.task,
                deploy_format=args.deploy_format,
                sample_id=0,
                status="FAILED",
                notes=f"{type(exc).__name__}: {exc}",
            )
        ]

    if not rows:
        rows = [_skip_row(args, "SKIPPED_NO_DATASET", "evaluation produced no rows")]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Wrote %d inference rows to %s", len(rows), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
