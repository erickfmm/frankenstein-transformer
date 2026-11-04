"""Build the joined master dataset for the Experiment 2 report.

Reads the raw CSVs under ``experimentos/2`` (per-run metrics directory and the
flat ``csvs/`` exports), streams the 702 MB per-step log once, and writes
compact artefacts used by the report generator and the web chart maker.

Usage:
    uv run --with pandas --with numpy python experimentos/2/report/build_data.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # experimentos/2
CSVS = ROOT / "csvs"
OUT = Path(__file__).resolve().parent / "out"
DATA = OUT / "data"

PAIR_RE = re.compile(r"^attn_(?P<a>.+?)__(?P<b>.+)$")

# Columns streamed from the 702 MB per-step log.
STEP_COLS = [
    "run", "epoch", "loss", "accuracy", "learning_rate", "grad_norm",
    "step_time_ms", "tokens_per_sec", "clip_ratio",
    "grad_norm_embeddings", "grad_norm_pos", "grad_norm_attention",
    "grad_norm_ffn", "grad_norm_norm", "grad_norm_head",
]
# source column suffix -> output column name (``norm`` would clash with the
# global ``grad_norm`` already present in the per-run summary).
GRAD_GROUPS = {"embeddings": "grad_embeddings", "pos": "grad_pos",
               "attention": "grad_attention", "ffn": "grad_ffn",
               "norm": "grad_norms", "head": "grad_head"}


def classify(combo: str) -> str:
    """Family of a combo id (pair, single, opt, pe, vit, ...)."""
    if "__" in combo:
        return "pair"
    if combo.startswith("attn_single_"):
        return "single"
    return combo.split("_")[0]


def split_pair(combo: str):
    """Return (mixer_a, mixer_b) for hybrid pairs, else (None, None)."""
    m = PAIR_RE.match(combo)
    if not m:
        return None, None
    return m.group("a"), m.group("b")


def base_combo(combo: str) -> str:
    """Strip the pipeline stage suffix from a results.json combo id."""
    for suffix in ("_deploy_standard", "_deploy_quantized", "_eval_standard",
                   "_eval_quantized", "_export", "_gguf"):
        if combo.endswith(suffix):
            return combo[: -len(suffix)]
    return combo


def tidy_mixer(name: str | None) -> str | None:
    """Human label for a mixer id: ``bigbird_attn`` -> ``bigbird``."""
    if name is None:
        return None
    return name[:-5] if name.endswith("_attn") else name


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ params
    params = pd.read_csv(CSVS / "param_metrics.csv")
    params["family"] = params["combo_id"].map(classify)
    pairs = params["combo_id"].map(split_pair)
    params["mixer_a"] = [tidy_mixer(a) for a, _ in pairs]
    params["mixer_b"] = [tidy_mixer(b) for _, b in pairs]
    params["is_pair"] = params["family"] == "pair"

    # ------------------------------------------------------- training summary
    final = pd.read_csv(CSVS / "training_metrics_final.csv").rename(
        columns={"run": "combo_id", "loss": "loss_final", "accuracy": "acc_final"}
    )
    avg = pd.read_csv(CSVS / "training_metrics_avg.csv").rename(
        columns={"run": "combo_id", "loss": "loss_avg", "accuracy": "acc_avg"}
    )
    std = pd.read_csv(CSVS / "training_metrics_std.csv").rename(
        columns={"run": "combo_id", "loss": "loss_std", "accuracy": "acc_std"}
    )
    keep_final = ["combo_id", "epoch", "step", "global_step", "loss_final", "acc_final",
                  "learning_rate", "grad_norm", "step_time_ms", "tokens_per_sec",
                  "clip_ratio", "scaler_scale"]
    keep_avg = ["combo_id", "loss_avg", "acc_avg", "step_time_ms", "tokens_per_sec",
                "grad_norm", "clip_ratio"]
    keep_std = ["combo_id", "loss_std", "acc_std", "grad_norm", "step_time_ms",
                "tokens_per_sec"]
    final = final[keep_final].rename(columns={
        "step_time_ms": "step_ms_final", "tokens_per_sec": "tok_s_final",
        "grad_norm": "grad_final", "clip_ratio": "clip_final",
        "learning_rate": "lr_final"})
    avg = avg[keep_avg].rename(columns={
        "step_time_ms": "step_ms_avg", "tokens_per_sec": "tok_s_avg",
        "grad_norm": "grad_avg", "clip_ratio": "clip_avg"})
    std = std[keep_std].rename(columns={
        "grad_norm": "grad_std", "step_time_ms": "step_ms_std",
        "tokens_per_sec": "tok_s_std"})

    # ------------------------------------------------------ pipeline stages
    res = pd.read_csv(CSVS / "results.json.csv")
    res["base"] = res["combo_id"].map(base_combo)
    res["family"] = res["base"].map(classify)

    def duration(category: str, fmt: str | None = None) -> pd.Series:
        sub = res[res["category"] == category]
        if fmt is not None:
            sub = sub[sub["combo_id"].str.endswith("_" + fmt)]
        return sub.set_index("base")["duration_sec"]

    def status(category: str, fmt: str | None = None) -> pd.Series:
        sub = res[res["category"] == category]
        if fmt is not None:
            sub = sub[sub["combo_id"].str.endswith("_" + fmt)]
        return sub.set_index("base")["status"]

    train_res = res[res["category"] == "training"].set_index("base")
    stages = pd.DataFrame({
        "train_dur": train_res["duration_sec"],
        "train_status": train_res["status"],
        "test_acc": train_res["test_accuracy"],
        "test_loss": train_res["test_loss"],
        "deploy_std_dur": duration("deploy", "standard"),
        "deploy_q_dur": duration("deploy", "quantized"),
        "deploy_std_status": status("deploy", "standard"),
        "deploy_q_status": status("deploy", "quantized"),
        "eval_std_dur": duration("inference-eval", "standard"),
        "eval_q_dur": duration("inference-eval", "quantized"),
        "export_dur": duration("transformers-export"),
        "export_status": status("transformers-export"),
    })

    # -------------------------------------------------------------- inference
    inf = pd.read_csv(CSVS / "inference.csv")
    inf["base"] = inf["combo_id"]
    mlm = inf[inf["metric_name"] == "mlm_top1"]
    inf_agg = (
        mlm.groupby(["base", "deploy_format"])
        .agg(top1=("metric_value", "mean"), top5=("top5_hit", "mean"),
             n=("metric_value", "size"))
        .reset_index()
    )
    wide = inf_agg.pivot(index="base", columns="deploy_format")
    inf_wide = pd.DataFrame({
        "inf_n_std": wide[("n", "standard")],
        "inf_top1_std": wide[("top1", "standard")],
        "inf_top5_std": wide[("top5", "standard")],
        "inf_top1_q": wide[("top1", "quantized")],
        "inf_top5_q": wide[("top5", "quantized")],
    })

    # --------------------------------------------------- per-step aggregation
    print("streaming training_metrics.csv ...", flush=True)
    chunk_iter = pd.read_csv(
        CSVS / "training_metrics.csv", usecols=STEP_COLS, chunksize=600_000,
        low_memory=False,
    )
    per_epoch_parts = []
    per_run_parts = []
    for k, chunk in enumerate(chunk_iter):
        g = chunk.groupby(["run", "epoch"], sort=False)
        per_epoch_parts.append(g.mean().reset_index())
        run_means = chunk.groupby("run", sort=False).mean(numeric_only=True)
        run_means["steps"] = chunk.groupby("run", sort=False).size()
        per_run_parts.append(run_means.reset_index())
        print(f"  chunk {k + 1}", flush=True)

    per_epoch = pd.concat(per_epoch_parts).groupby(["run", "epoch"]).mean().reset_index()
    per_epoch = per_epoch.rename(columns={"run": "combo_id"})
    per_epoch["combo_id"] = per_epoch["combo_id"].astype(str)
    per_run = pd.concat(per_run_parts).groupby("run").mean(numeric_only=True).reset_index()
    per_run = per_run.rename(columns={"run": "combo_id"})
    per_run["combo_id"] = per_run["combo_id"].astype(str)
    grad_means = per_run[["combo_id"] + [f"grad_norm_{g}" for g in GRAD_GROUPS]].rename(
        columns={f"grad_norm_{g}": out for g, out in GRAD_GROUPS.items()}
    )
    per_run = per_run.drop(columns=[f"grad_norm_{g}" for g in GRAD_GROUPS] + ["grad_norm"])

    # ------------------------------------------------------------ master join
    master = (
        params.merge(final, on="combo_id", how="left")
        .merge(avg, on="combo_id", how="left")
        .merge(std, on="combo_id", how="left")
        .merge(stages, left_on="combo_id", right_index=True, how="left")
        .merge(inf_wide, left_on="combo_id", right_index=True, how="left")
        .merge(per_run, on="combo_id", how="left")
        .merge(grad_means, on="combo_id", how="left")
    )
    # Clean up names from the streaming pass and drop duplicated epoch columns.
    master = master.drop(columns=[c for c in ("epoch_x", "epoch_y") if c in master.columns])
    master = master.rename(columns={
        "loss": "loss_step_mean", "accuracy": "acc_step_mean",
        "learning_rate": "lr_mean", "step_time_ms": "step_ms_mean",
        "tokens_per_sec": "tok_s_mean", "clip_ratio": "clip_mean",
        "steps": "n_steps",
    })
    master["params_per_mixer"] = master["total"] / 2
    master["tok_s_median"] = master[["tok_s_avg", "tok_s_final"]].median(axis=1)
    master["speed_score"] = master["total"] / master["tok_s_median"].replace(0, np.nan)
    master["quant_top5_delta"] = master["inf_top5_q"] - master["inf_top5_std"]
    master["quant_top1_delta"] = master["inf_top1_q"] - master["inf_top1_std"]
    master["deploy_speedup"] = master["deploy_std_dur"] / master["deploy_q_dur"]
    master["eval_speedup"] = master["eval_std_dur"] / master["eval_q_dur"]
    master["acc_per_mparam"] = master["acc_final"] / (master["total"] / 1e6)

    # ------------------------------------------------------------ write files
    master.to_csv(DATA / "master.csv", index=False)
    per_epoch.to_csv(DATA / "training_per_epoch.csv", index=False)
    inf.to_csv(DATA / "inference_samples.csv", index=False)
    res.to_csv(DATA / "pipeline_stages.csv", index=False)
    for name in ("param_metrics.csv", "training_metrics_avg.csv",
                 "training_metrics_final.csv", "training_metrics_std.csv",
                 "results.md.csv"):
        (DATA / name).write_bytes((CSVS / name).read_bytes())

    # ------------------------------------------------------------ headline stats
    pair = master[master["family"] == "pair"]
    ok_train = master[master["train_status"] == "OK"]
    stats = {
        "generated": pd.Timestamp.utcnow().isoformat(),
        "n_configs": int(len(master)),
        "n_pairs": int(len(pair)),
        "n_singles": int((master["family"] == "single").sum()),
        "n_families": int(master["family"].nunique()),
        "families": master["family"].value_counts().to_dict(),
        "n_pipeline_rows": int(len(res)),
        "n_steps": int(per_run["steps"].sum()),
        "n_inference_rows": int(len(inf)),
        "n_inference_configs": int(mlm["combo_id"].nunique()),
        "n_deploy_ok": int((res["category"] == "deploy").sum()),
        "n_deploy_failed": int(((res["category"] == "deploy") & (res["status"] == "FAILED")).sum()),
        "n_train_failed": int(((res["category"] == "training") & (res["status"] == "FAILED")).sum()),
        "n_train_records": int((res["category"] == "training").sum()),
        "param_min": int(master["total"].min()),
        "param_max": int(master["total"].max()),
        "param_median": float(master["total"].median()),
        "loss_min": float(master["loss_final"].min()),
        "loss_median": float(master["loss_final"].median()),
        "acc_max": float(master["acc_final"].max()),
        "acc_median": float(master["acc_final"].median()),
        "top5_global": float(mlm["top5_hit"].mean()),
        "top1_global": float(mlm["metric_value"].mean()),
        "tok_s_median": float(master["tok_s_median"].median()),
        "tok_s_max": float(master["tok_s_median"].max()),
        "train_hours_median": float(master["train_dur"].median() / 3600),
        "train_hours_total": float(master["train_dur"].sum() / 3600),
        "quant_top5_std": float(master["inf_top5_std"].mean()),
        "quant_top5_q": float(master["inf_top5_q"].mean()),
        "quant_top1_std": float(master["inf_top1_std"].mean()),
        "quant_top1_q": float(master["inf_top1_q"].mean()),
        "n_epoch_curves": int(len(per_epoch)),
        "mixers_a": sorted(pair["mixer_a"].dropna().unique().tolist()),
        "mixers_b": sorted(pair["mixer_b"].dropna().unique().tolist()),
        "n_mixers_a": int(pair["mixer_a"].nunique()),
        "n_mixers_b": int(pair["mixer_b"].nunique()),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, list)}, indent=2))
    print("per-epoch rows:", len(per_epoch))
    print("master columns:", len(master.columns))


if __name__ == "__main__":
    main()
