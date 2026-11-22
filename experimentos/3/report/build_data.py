"""Build the joined Experiment 3 dataset used by the charts and the annex.

Experiment 3 is the full mixer atlas: 1,653 training runs (1,431 hybrid pairs,
54 single mixers and 68 ablations) plus 9,800 pipeline stage records. This
script reads the flat CSVs under ``export/`` (produced by
``experimentos/to_csv.py``), parses the 1,653 per-run YAML configs, streams the
6.6 GB per-step log exactly once, and writes compact artefacts for the report
generator:

* ``master.csv``      one row per training run (results, params, config
                      features, final/mean optimisation statistics);
* ``epochs.csv``      per-(run, epoch) means of loss, accuracy, gradient norms
                      per parameter block, step time and throughput;
* ``blocks.csv``      long-format gradient statistics per parameter block;
* ``inference.csv``   cleaned per-sample inference rows (deploy stages removed);
* ``inference_acc.csv`` per-(run, loader, codec, metric) accuracy aggregates;
* ``inference_pairs.csv`` paired codec-vs-baseline deltas for the v2 runtime.

Usage:
    uv run --no-sync --with pandas --with pyyaml python \
        experimentos/3/report/build_data.py [--quick]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

TRAIN_COLS = [
    "run", "timestamp", "epoch", "global_step", "loss", "accuracy",
    "grad_norm",
    "grad_norm_embeddings", "grad_norm_pos", "grad_norm_attention",
    "grad_norm_ffn", "grad_norm_norm", "grad_norm_head", "grad_norm_mhc",
    "grad_norm_router", "grad_norm_residual", "grad_norm_other",
    "step_time_ms", "tokens_per_sec", "clip_ratio",
    "has_nan", "has_inf", "has_zero", "repair_action",
    "cpu_rss_peak_mb",
]

#: Columns averaged per epoch (in addition to ``loss``/``accuracy``).
EPOCH_NUM = [
    "loss", "accuracy", "grad_norm", "global_step",
    *[f"grad_norm_{group}" for group in C.BLOCK_GROUPS],
    "step_time_ms", "tokens_per_sec", "clip_ratio",
]


def _min_ts(left, right):
    """Element-wise minimum that ignores missing timestamps."""
    if pd.isna(left):
        return right
    if pd.isna(right):
        return left
    return min(left, right)


def _max_ts(left, right):
    """Element-wise maximum that ignores missing timestamps."""
    if pd.isna(left):
        return right
    if pd.isna(right):
        return left
    return max(left, right)


# ---------------------------------------------------------------------------
# Config features
# ---------------------------------------------------------------------------
def load_config_features() -> pd.DataFrame:
    """Parse the 1,653 per-run YAML configs into one feature row per run.

    Returns:
        DataFrame indexed by ``run`` with the structural switches (normalization,
        residual type, positional encoding, BitNet/MoE/mHC), the model shape and
        the optimizer settings that vary across ablations.
    """
    rows = []
    for path in sorted(C.CONFIGS.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        model = cfg.get("model") or {}
        dims = model.get("dims") or {}
        train = cfg.get("training") or {}
        optimizer = train.get("optimizer") or {}
        parameters = optimizer.get("parameters") or {}
        image = cfg.get("image") or {}

        def first_value(*suffixes):
            for name, value in parameters.items():
                if any(str(name).endswith(suffix) for suffix in suffixes):
                    return value
            return None

        learning_rates = [
            float(value)
            for key, value in parameters.items()
            if ("-lr_" in str(key) or str(key).endswith("-lr"))
            and isinstance(value, (int, float))
        ]
        weight_decays = [
            float(value)
            for key, value in parameters.items()
            if ("-wd_" in str(key) or str(key).endswith("-wd"))
            and isinstance(value, (int, float))
        ]
        pattern = dims.get("layer_pattern") or []
        rows.append(
            {
                "run": path.stem,
                "model_class": cfg.get("model_class") or "frankenstein_text_encoder",
                "task": train.get("task") or "mlm",
                "hidden_size": dims.get("hidden_size"),
                "num_layers": dims.get("num_layers"),
                "num_heads": dims.get("num_heads"),
                "num_kv_heads": dims.get("num_kv_heads"),
                "vocab_size": dims.get("vocab_size"),
                "dropout": dims.get("dropout"),
                "num_loops": dims.get("num_loops"),
                "layer_pattern": "+".join(str(x) for x in pattern),
                "norm_type": (model.get("norm") or {}).get("type"),
                "residual_type": (model.get("residuals") or {}).get("type", "standard"),
                "activation": (model.get("ffn") or {}).get("activation"),
                "pos_encoding": model.get("positional_encoding")
                or image.get("pos_embedding_type")
                or "default",
                "bitnet": bool((model.get("bitnet") or {}).get("enabled", False)),
                "bitnet_routers": bool((model.get("bitnet") or {}).get("routers", False)),
                "bitnet_conv": bool((model.get("bitnet") or {}).get("conv", False)),
                "moe": bool((model.get("moe") or {}).get("enabled", False)),
                "mixture_of_depths": bool(
                    (model.get("mixture_of_depths") or {}).get("enabled", False)
                ),
                "factorized_embedding": bool(
                    ((model.get("embedding") or {}).get("factorized") or {}).get("enabled", False)
                ),
                "dynamic_embedding": bool(
                    ((model.get("embedding") or {}).get("dynamic") or {}).get("enabled", False)
                ),
                "conv_embedding": bool(
                    ((model.get("embedding") or {}).get("conv") or {}).get("enabled", False)
                ),
                "optimizer": optimizer.get("optimizer_class") or "adamw",
                "lr": first_value("-lr_other", "-lr") or (max(learning_rates) if learning_rates else None),
                "weight_decay": first_value("-wd_other", "-wd")
                or (max(weight_decays) if weight_decays else None),
                "num_epochs_cfg": train.get("num_epochs"),
                "batch_size_cfg": train.get("batch_size"),
                "max_length_cfg": train.get("max_length"),
                "max_samples_cfg": train.get("max_samples"),
                "grad_accum_cfg": train.get("gradient_accumulation_steps"),
                "grad_clip_cfg": train.get("grad_clip_max_norm"),
            }
        )
    frame = pd.DataFrame(rows).set_index("run")
    frame["family"] = [C.classify(run) for run in frame.index]
    frame["pair_a"] = [C.tidy_mixer(C.split_pair(run)[0]) for run in frame.index]
    frame["pair_b"] = [C.tidy_mixer(C.split_pair(run)[1]) for run in frame.index]
    return frame


# ---------------------------------------------------------------------------
# Per-step log streaming
# ---------------------------------------------------------------------------
def stream_training_log(quick: bool = False):
    """Aggregate the 20.5M-row per-step log into per-epoch and per-run tables.

    Args:
        quick: Only read the first two chunks (smoke test).

    Returns:
        ``(epochs, extras)`` where ``epochs`` is per-(run, epoch) means and
        ``extras`` holds per-run counters (NaN/Inf steps, repairs, peak RSS,
        first/last timestamp).
    """
    path = C.CSVS / "training_metrics.csv"
    runs = sorted(p.stem for p in C.TRAINING_METRICS.glob("*.csv"))
    dtype = {"run": pd.CategoricalDtype(categories=runs, ordered=True)}
    for column in EPOCH_NUM:
        dtype[column] = "float32"
    counts = ["has_nan", "has_inf", "has_zero"]
    reader = pd.read_csv(
        path, usecols=TRAIN_COLS, dtype=dtype, chunksize=400_000, low_memory=False
    )
    epoch_acc: pd.DataFrame | None = None
    counts_acc: pd.DataFrame | None = None
    times_acc: pd.DataFrame | None = None
    for index, chunk in enumerate(reader):
        if quick and index >= 2:
            break
        chunk["has_nan"] = chunk["has_nan"].fillna(0).astype("int8")
        chunk["has_inf"] = chunk["has_inf"].fillna(0).astype("int8")
        chunk["has_zero"] = chunk["has_zero"].fillna(0).astype("int8")
        chunk["repair"] = (
            chunk["repair_action"].fillna("none").astype(str).ne("none").astype("int8")
        )
        # The trainer emits a sentinel row (``loss=0``, ``has_zero=1``) when the
        # last batch cannot be formed; it must not enter the epoch means, or the
        # final loss of every such run reads as zero.
        valid = chunk[(chunk["has_zero"] == 0) & (chunk["has_inf"] == 0)]
        grouped = valid.groupby(["run", "epoch"], observed=True)
        frame = grouped[EPOCH_NUM].sum()
        frame["n"] = grouped.size()
        epoch_acc = frame if epoch_acc is None else epoch_acc.add(frame, fill_value=0)

        per_run = chunk.groupby("run", observed=True).agg(
            steps=("loss", "size"),
            nan_steps=("has_nan", "sum"),
            inf_steps=("has_inf", "sum"),
            zero_steps=("has_zero", "sum"),
            repair_steps=("repair", "sum"),
            rss_peak_mb=("cpu_rss_peak_mb", "max"),
        )
        if counts_acc is None:
            counts_acc = per_run.astype("float64")
        else:
            counts_acc = counts_acc.add(per_run, fill_value=0)
            counts_acc["rss_peak_mb"] = np.maximum(
                counts_acc["rss_peak_mb"], per_run["rss_peak_mb"]
            )
        times = chunk.groupby("run", observed=True)["timestamp"].agg(["min", "max"])
        times.columns = ["first_ts", "last_ts"]
        if times_acc is None:
            times_acc = times
        else:
            aligned = times.reindex(times_acc.index.union(times.index))
            base = times_acc.reindex(aligned.index)
            times_acc = pd.DataFrame(
                {
                    "first_ts": base["first_ts"].combine(aligned["first_ts"], _min_ts),
                    "last_ts": base["last_ts"].combine(aligned["last_ts"], _max_ts),
                }
            )
        print(f"  chunk {index + 1}: {len(chunk):>7,} rows", flush=True)

    assert epoch_acc is not None and counts_acc is not None and times_acc is not None
    epochs = epoch_acc[EPOCH_NUM].div(epoch_acc["n"], axis=0).reset_index()
    epochs["n_steps"] = epoch_acc["n"].values
    extras = counts_acc.join(times_acc, how="left").reset_index()
    for column in ("nan_steps", "inf_steps", "zero_steps", "repair_steps", "steps"):
        extras[column] = extras[column].astype("int64")
    return epochs, extras


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def build_inference() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean the inference rows and derive aggregate and paired tables.

    Deploy-derived formats (``standard``/``quantized``) are dropped: the deploy
    stage had a configuration-payload bug in this round, so only the
    Transformers-export path (``hf_*``) is analysed.

    Returns:
        ``(rows, accuracy, paired)``.
    """
    frame = pd.read_csv(C.CSVS / "inference.csv", low_memory=False)
    # ``to_csv.py`` appends the note-parsed columns after the canonical ones,
    # so ``top5_hit``/``token_accuracy`` appear twice; coalesce them.
    for column in ("top5_hit", "token_accuracy", "first_token_match"):
        alt = f"{column}.1"
        if alt in frame.columns:
            frame[column] = frame[column].fillna(frame[alt])
            frame = frame.drop(columns=[alt])
    frame = frame[frame["deploy_format"].astype(str).str.startswith("hf_")].copy()
    frame["codec"] = frame["deploy_format"].map(C.codec_of)
    frame["runtime"] = frame["deploy_format"].map(C.runtime_of)
    for column in ("metric_value", "top5_hit", "token_accuracy", "first_token_match"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["ok"] = frame["status"].eq("OK")
    frame["family"] = frame["combo_id"].map(C.classify)

    grouped = frame.groupby(
        ["combo_id", "deploy_format", "codec", "runtime", "task", "metric_name", "family"],
        observed=True,
    )
    accuracy = grouped.agg(
        n=("metric_value", "size"),
        n_ok=("ok", "sum"),
        accuracy=("metric_value", "mean"),
        top5=("top5_hit", "mean"),
        token_accuracy=("token_accuracy", "mean"),
        agreement_rows=("predicted", "nunique"),
    ).reset_index()

    # Paired comparison inside the v2 runtime, whose mask positions coincide
    # with ``hf_none`` (v1 sampled a different mask seed, so it cannot be
    # paired sample by sample).
    key = ["combo_id", "sample_id", "mask_position"]
    subset = frame[frame["runtime"].isin(("none", "v2"))].copy()
    subset["sample_key"] = (
        subset[key].astype("string").fillna("").agg("|".join, axis=1)
    )
    pivot = subset.pivot_table(
        index=["combo_id", "sample_key"],
        columns="deploy_format",
        values=["predicted", "top5_hit", "metric_value"],
        aggfunc="first",
    )
    pivot.columns = [f"{a}::{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    paired_rows = []
    for codec, name in (("q8_0", "hf_q8_0_v2"), ("q4_0", "hf_q4_0_v2"), ("i2_s", "hf_i2_s_v2")):
        base_pred, base_top5 = "predicted::hf_none", "top5_hit::hf_none"
        alt_pred, alt_top5 = f"predicted::{name}", f"top5_hit::{name}"
        if alt_pred not in pivot.columns:
            continue
        both = pivot.dropna(subset=[base_pred, alt_pred])
        agree = (both[base_pred] == both[alt_pred]).mean()
        both5 = pivot.dropna(subset=[base_top5, alt_top5])
        paired_rows.append(
            {
                "codec": codec,
                "format": name,
                "n_pairs": int(len(both)),
                "n_combos": int(both["combo_id"].nunique()),
                "agreement": float(agree),
                "top5_base": float(both5[base_top5].mean()) if len(both5) else np.nan,
                "top5_quant": float(both5[alt_top5].mean()) if len(both5) else np.nan,
                "n_pairs_top5": int(len(both5)),
            }
        )
    paired = pd.DataFrame(paired_rows)
    return frame, accuracy, paired


# ---------------------------------------------------------------------------
# Master table
# ---------------------------------------------------------------------------
def build_master(
    epochs: pd.DataFrame, extras: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """Join results, parameters, config features and optimisation statistics."""
    results = pd.read_csv(C.CSVS / "results.json.csv", low_memory=False)
    results = results[results["category"] == "training"].set_index("combo_id")
    results = results.rename_axis("run")
    params = pd.read_csv(C.CSVS / "param_metrics.csv").set_index("combo_id")
    params = params.rename_axis("run")
    avg = pd.read_csv(C.CSVS / "training_metrics_avg.csv").set_index("run")
    std = pd.read_csv(C.CSVS / "training_metrics_std.csv").set_index("run")

    # ``training_metrics_final.csv`` is the literal last logged row, which is a
    # zero sentinel for runs whose final batch was skipped. Use the last epoch
    # that actually aggregated a step instead.
    last = (
        epochs.sort_values(["run", "epoch"])
        .groupby("run", sort=False)
        .tail(1)
        .set_index("run")
    )
    final = pd.DataFrame(
        {
            "final_loss": last["loss"],
            "final_accuracy": last["accuracy"],
            "final_grad_norm": last["grad_norm"],
            "final_grad_embeddings": last["grad_norm_embeddings"],
            "final_grad_pos": last["grad_norm_pos"],
            "final_grad_attention": last["grad_norm_attention"],
            "final_grad_ffn": last["grad_norm_ffn"],
            "final_grad_norms": last["grad_norm_norm"],
            "final_grad_head": last["grad_norm_head"],
            "final_grad_mhc": last["grad_norm_mhc"],
            "final_grad_router": last["grad_norm_router"],
            "final_grad_residual": last["grad_norm_residual"],
            "final_grad_other": last["grad_norm_other"],
            "final_step_time_ms": last["step_time_ms"],
            "final_tokens_per_sec": last["tokens_per_sec"],
            "final_clip_ratio": last["clip_ratio"],
            "final_epoch": last["epoch"],
            "final_global_step": last["global_step"].round(),
        }
    )
    keep_avg = {
        "loss": "mean_loss",
        "accuracy": "mean_accuracy",
        "grad_norm": "mean_grad_norm",
        "step_time_ms": "mean_step_time_ms",
        "tokens_per_sec": "mean_tokens_per_sec",
        "clip_ratio": "mean_clip_ratio",
        "effective_batch_size": "mean_effective_batch",
        "cpu_rss_mb": "mean_rss_mb",
    }
    keep_std = {
        "loss": "std_loss",
        "accuracy": "std_accuracy",
        "grad_norm": "std_grad_norm",
        "clip_ratio": "std_clip_ratio",
    }
    master = (
        results[["status", "duration_sec", "notes"]]
        .join(params, how="left")
        .join(final, how="left")
        .join(avg[list(keep_avg)].rename(columns=keep_avg), how="left")
        .join(std[list(keep_std)].rename(columns=keep_std), how="left")
        .join(extras.set_index("run"), how="left")
        .join(features, how="left")
    )
    master["family"] = [C.classify(run) for run in master.index]
    master["duration_h"] = master["duration_sec"] / 3600.0
    master["params_m"] = master["total"] / 1e6
    master["minutes_per_mparam"] = master["duration_sec"] / 60.0 / master["params_m"]
    master["tokens_per_sec"] = master["mean_tokens_per_sec"].where(
        master["mean_tokens_per_sec"] < 1e5
    )
    master["steps"] = master["final_global_step"]
    return master.reset_index()


def main() -> None:
    """Build every artefact under ``out/data``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Smoke test on two chunks.")
    args = parser.parse_args()

    C.DATA.mkdir(parents=True, exist_ok=True)
    print("configs...", flush=True)
    features = load_config_features()
    print(f"  {len(features)} configs", flush=True)

    print("streaming training log...", flush=True)
    epochs, extras = stream_training_log(quick=args.quick)
    print(f"  {len(epochs):,} run-epochs / {len(extras):,} runs", flush=True)

    print("inference...", flush=True)
    inference, accuracy, paired = build_inference()
    print(f"  {len(inference):,} rows / {len(accuracy):,} aggregates", flush=True)

    print("master...", flush=True)
    master = build_master(epochs, extras, features)

    targets = {
        "master.csv": master,
        "epochs.csv": epochs,
        "inference.csv": inference,
        "inference_acc.csv": accuracy,
        "inference_pairs.csv": paired,
        "param_metrics.csv": pd.read_csv(C.CSVS / "param_metrics.csv"),
    }
    # blocks.csv is derived directly from the avg/std/final aggregates.
    avg = pd.read_csv(C.CSVS / "training_metrics_avg.csv").set_index("run")
    std = pd.read_csv(C.CSVS / "training_metrics_std.csv").set_index("run")
    final_master = master.set_index("run")
    long_rows = []
    for group in C.BLOCK_GROUPS:
        column = f"grad_norm_{group}"
        if column not in avg.columns:
            continue
        final_column = "final_grad_norms" if group == "norm" else f"final_grad_{group}"
        long_rows.append(
            pd.DataFrame(
                {
                    "run": avg.index,
                    "group": group,
                    "mean": avg[column].values,
                    "std": std[column].reindex(avg.index).values,
                    "final": final_master[final_column].reindex(avg.index).values,
                }
            )
        )
    blocks = pd.concat(long_rows, ignore_index=True)
    blocks["family"] = blocks["run"].map(C.classify)
    targets["blocks.csv"] = blocks

    for name, frame in targets.items():
        path = C.DATA / name
        frame.to_csv(path, index=False)
        print(f"  wrote {path.relative_to(C.ROOT.parent)} ({len(frame):,} rows)")


if __name__ == "__main__":
    main()
