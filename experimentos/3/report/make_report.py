"""Compute the Experiment 3 statistics and render the two final annexes.

The script reads the compact artefacts produced by ``build_data.py`` (never the
6.6 GB log), computes every number quoted in the annex, writes ``out/stats.json``
and then renders the appendix in English (``docs/paper``) and Spanish
(``docs/paper-es``). Figures must already exist (``make_paper_figures.py``).

Usage:
    uv run --no-sync --with pandas --with numpy python \
        experimentos/3/report/make_report.py [--stats-only]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import make_charts as mc  # noqa: E402

EN_ANNEX = C.PROJECT / "docs" / "paper" / "appendices" / "annex-26-experiment-3.tex"
ES_ANNEX = C.PROJECT / "docs" / "paper-es" / "appendices" / "annex-26-experimento-3.tex"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _q(series, q):
    """Quantile that is NaN-safe."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if len(values) else float("nan")


def _table_counts(series):
    """Ordered value counts as a plain dict."""
    return {str(k): int(v) for k, v in series.value_counts().items()}


def compute_stats() -> dict:
    """Compute every number quoted in the annex."""
    data = mc.load()
    master = data["master"]
    epochs = data["epochs"]
    blocks = data["blocks"]
    acc = data["inference_acc"]
    paired = data["inference_pairs"]
    results = pd.read_csv(C.CSVS / "results.json.csv", low_memory=False)

    stats: dict = {}
    training = master[master["status"].notna()].copy()
    # Smoke runs (5-epoch MoE variants, the BitNet conversion) and truncated
    # pairs are excluded from every cross-run comparison.
    analysis_runs = training[training["final_epoch"].fillna(0) >= 100].copy()
    stats["population"] = {
        "n_runs": int(len(training)),
        "n_records": int(len(results)),
        "n_stages": int((~results["category"].eq("training")).sum()),
        "stage_counts": _table_counts(results["category"]),
        "stage_status": {
            str(category): _table_counts(group["status"])
            for category, group in results.groupby("category")
        },
        "n_pairs": int((training["family"] == "pair").sum()),
        "n_singles": int((training["family"] == "single").sum()),
        "n_ablations": int(
            (~training["family"].isin(["pair", "single"])).sum()
        ),
        "n_mixers": int(
            len(
                {
                    mixer
                    for run in training.loc[training.family == "pair", "run"]
                    for mixer in C.split_pair(run)
                    if mixer
                }
            )
        ),
        "family_counts": _table_counts(training["family"]),
        "compute_h": float(training["duration_h"].sum()),
        "duration_median_h": float(training["duration_h"].median()),
        "duration_p90_h": _q(training["duration_h"], 0.9),
        "duration_max_h": float(training["duration_h"].max()),
        "steps_total": int(training["final_global_step"].fillna(0).sum()),
        "steps_median": float(training["final_global_step"].median()),
        "campaign_start": str(training["first_ts"].dropna().min()),
        "campaign_end": str(training["last_ts"].dropna().max()),
        "campaign_days": float(
            (
                pd.to_datetime(training["last_ts"].dropna().max())
                - pd.to_datetime(training["first_ts"].dropna().min())
            ).total_seconds()
            / 86400
        ),
        "training_ok": int((training["status"] == "OK").sum()),
        "training_failed": int((training["status"] != "OK").sum()),
        "early_stop": int((training["final_epoch"] < 490).sum()),
        "smoke_runs": int((training["final_epoch"].fillna(0) < 100).sum()),
        "n_analysis": int(len(analysis_runs)),
        "short_runs": int((training["final_global_step"] < 5000).sum()),
    }

    # ---------------------------------------------------------------- params
    stats["params"] = {
        "min": float(training["total"].min()),
        "median": float(training["total"].median()),
        "p90": _q(training["total"], 0.9),
        "max": float(training["total"].max()),
        "largest_run": str(training.loc[training["total"].idxmax(), "run"]),
        "smallest_run": str(training.loc[training["total"].idxmin(), "run"]),
    }
    params = data["params"].merge(
        training[["run", "family"]], left_on="combo_id", right_on="run"
    )
    parts = ["embeddings", "attention", "ffn", "head", "pos", "norm", "other"]
    share = params[parts].div(params["total"], axis=0) * 100
    stats["composition"] = {
        "median_share": {
            part: float(share[part].median()) for part in parts
        },
        "pair_share": {
            part: float(share[params.family == "pair"][part].median())
            for part in parts
        },
    }
    pair_runs = analysis_runs[analysis_runs.family == "pair"]
    mixer_cost = []
    for mixer in sorted({m for run in pair_runs["run"] for m in C.split_pair(run) if m}):
        rows = pair_runs[pair_runs["run"].str.contains(f"__{mixer}$|attn_{mixer}__", regex=True)]
        if len(rows) >= 10:
            mixer_cost.append((C.tidy_mixer(mixer), float(rows["total"].median())))
    mixer_cost.sort(key=lambda item: item[1])
    stats["mixer_cost"] = {
        "cheapest": mixer_cost[:5],
        "most_expensive": mixer_cost[-5:][::-1],
    }
    stats["params"]["pair_median"] = float(pair_runs["total"].median())
    stats["params"]["loss_rho"] = mc._spearman(
        np.log10(pair_runs["params_m"]), pair_runs["final_loss"]
    )

    # ------------------------------------------------------------- outcomes
    stats["loss"] = {
        "pair_median": float(pair_runs["final_loss"].median()),
        "pair_min": float(pair_runs["final_loss"].min()),
        "pair_max": float(pair_runs["final_loss"].max()),
        "by_family": {
            family: {
                "median": float(group["final_loss"].median()),
                "p10": _q(group["final_loss"], 0.1),
                "p90": _q(group["final_loss"], 0.9),
                "n": int(len(group)),
            }
            for family, group in analysis_runs.groupby("family")
            if group["final_loss"].notna().sum() >= 3
        },
    }
    ranked = pair_runs.dropna(subset=["final_loss"]).nsmallest(10, "final_loss")
    stats["top_loss"] = [
        {
            "run": str(row["run"]),
            "params": float(row["total"]),
            "loss": float(row["final_loss"]),
            "duration_h": float(row["duration_h"]),
        }
        for _, row in ranked.iterrows()
    ]
    worst = pair_runs.dropna(subset=["final_loss"]).nlargest(5, "final_loss")
    stats["worst_loss"] = [
        {"run": str(row["run"]), "loss": float(row["final_loss"])} for _, row in worst.iterrows()
    ]

    base = acc[(acc["deploy_format"] == "hf_none")].drop(columns=["family"]).merge(
        analysis_runs[["run", "family", "final_loss", "params_m"]],
        left_on="combo_id",
        right_on="run",
        how="left",
    )
    mlm_base = base[base["task"] == "mlm"]
    stats["inference"] = {
        "n_combos": int(acc["combo_id"].nunique()),
        "n_rows": int(len(data["inference"])),
        "n_pairs_combos": int(mlm_base["combo_id"].nunique()),
        "top1_median": float(mlm_base["accuracy"].median()),
        "top5_median": float(mlm_base["top5"].median()),
        "top1_mean": float(mlm_base["accuracy"].mean()),
        "top5_mean": float(mlm_base["top5"].mean()),
        "tasks": _table_counts(base["task"]),
        "loss_top5_rho": mc._spearman(
            mlm_base.loc[mlm_base.family == "pair", "final_loss"],
            mlm_base.loc[mlm_base.family == "pair", "top5"],
        ),
    }
    best_acc = mlm_base.dropna(subset=["top5"]).nlargest(10, "top5")
    stats["top_inference"] = [
        {
            "run": str(row["combo_id"]),
            "top1": float(row["accuracy"]),
            "top5": float(row["top5"]),
            "loss": float(row["final_loss"]) if pd.notna(row["final_loss"]) else None,
        }
        for _, row in best_acc.iterrows()
    ]

    # ------------------------------------------------------------- atlas
    pairs_loss = pair_runs.dropna(subset=["final_loss"]).copy()
    by_a = pairs_loss.groupby("pair_a")["final_loss"].agg(["median", "size"])
    by_b = pairs_loss.groupby("pair_b")["final_loss"].agg(["median", "size"])
    common_roles = sorted(set(by_a.index) & set(by_b.index))
    med_a = by_a.loc[common_roles, "median"]
    med_b = by_b.loc[common_roles, "median"]
    singles = analysis_runs[analysis_runs["family"] == "single"].dropna(
        subset=["final_loss"]
    )
    best_single = singles.nsmallest(1, "final_loss").iloc[0]

    # Synergy: how far a pair is from the mean of its two marginals.
    members: dict[str, list[float]] = {}
    for _, row in pairs_loss.iterrows():
        members.setdefault(row["pair_a"], []).append(row["final_loss"])
        members.setdefault(row["pair_b"], []).append(row["final_loss"])
    marginals = {name: float(np.mean(v)) for name, v in members.items()}
    pairs_loss["synergy"] = pairs_loss.apply(
        lambda row: (marginals[row["pair_a"]] + marginals[row["pair_b"]]) / 2
        - row["final_loss"],
        axis=1,
    )

    # Bit-identical pairs: window mixers that collapse at a 64-token context.
    dup_groups = pairs_loss.groupby(pairs_loss["final_loss"].round(6))["run"]
    dup_sizes = dup_groups.size()

    top5_b = pairs_loss.merge(
        mlm_base[["combo_id", "top5"]], left_on="run", right_on="combo_id"
    ).groupby("pair_b")["top5"].agg(["mean", "size"])
    top5_b = top5_b[top5_b["size"] >= 20].sort_values("mean", ascending=False)

    pareto_rows = pairs_loss[pairs_loss["task"] == "mlm"].sort_values("duration_h")
    front: list[dict] = []
    frontier = np.inf
    for _, row in pareto_rows.iterrows():
        if row["final_loss"] < frontier:
            frontier = row["final_loss"]
            front.append(
                {
                    "run": str(row["run"]),
                    "loss": float(row["final_loss"]),
                    "duration_h": float(row["duration_h"]),
                }
            )

    def _role_rows(frame, role):
        rows = []
        for mixer, group in frame.iterrows():
            top5_row = top5_b.loc[mixer] if mixer in top5_b.index else None
            rows.append(
                {
                    "mixer": str(mixer),
                    "n": int(group["size"]),
                    "median": float(group["median"]),
                    "median_other": float(
                        (med_b if role == "a" else med_a).get(mixer, np.nan)
                    ),
                    "top5": float(top5_row["mean"]) if top5_row is not None else None,
                }
            )
        return rows

    stats["atlas"] = {
        "n_cells": int((training["family"] == "pair").sum()),
        "n_measured": int(len(pairs_loss)),
        "n_holes": int(
            training.loc[training["family"] == "pair", "final_loss"].isna().sum()
        ),
        "roles": {
            "n_common": len(common_roles),
            "range_a": float(med_a.max() - med_a.min()),
            "range_b": float(med_b.max() - med_b.min()),
            "best_a": [str(med_a.idxmin()), float(med_a.min())],
            "worst_a": [str(med_a.idxmax()), float(med_a.max())],
            "best_b": [str(med_b.idxmin()), float(med_b.min())],
            "worst_b": [str(med_b.idxmax()), float(med_b.max())],
            "rho": mc._spearman(med_a.to_numpy(), med_b.to_numpy()),
            "only_a": sorted(set(by_a.index) - set(by_b.index)),
            "only_b": sorted(set(by_b.index) - set(by_a.index)),
        },
        "singles": {
            "n": int(len(singles)),
            "median": float(singles["final_loss"].median()),
            "best": [str(best_single["run"]), float(best_single["final_loss"])],
            "pairs_better_than_best_single": int(
                (pairs_loss["final_loss"] < best_single["final_loss"]).sum()
            ),
        },
        "synergy": {
            "abs_median": float(pairs_loss["synergy"].abs().median()),
            "abs_p90": float(pairs_loss["synergy"].abs().quantile(0.9)),
            "share_above_015": float((pairs_loss["synergy"] > 0.15).mean() * 100),
            "top": [
                {"run": str(r["run"]), "loss": float(r["final_loss"]),
                 "expected": float(r["final_loss"] + r["synergy"]),
                 "synergy": float(r["synergy"])}
                for _, r in pairs_loss.nlargest(5, "synergy").iterrows()
            ],
            "bottom": [
                {"run": str(r["run"]), "loss": float(r["final_loss"]),
                 "expected": float(r["final_loss"] + r["synergy"]),
                 "synergy": float(r["synergy"])}
                for _, r in pairs_loss.nsmallest(5, "synergy").iterrows()
            ],
        },
        "duplicates": {
            "n_groups": int((dup_sizes > 1).sum()),
            "n_pairs": int(dup_sizes[dup_sizes > 1].sum()),
        },
        "top5_by_b": [
            {"mixer": str(mixer), "top5": float(group["mean"]),
             "n": int(group["size"])}
            for mixer, group in top5_b.iterrows()
        ],
        "best_b": _role_rows(by_b.sort_values("median").head(8), "b"),
        "worst_b": _role_rows(by_b.sort_values("median").tail(8).iloc[::-1], "b"),
        "pareto": front,
    }

    # ----------------------------------------------------------- optimizers
    opt = analysis_runs[analysis_runs["family"] == "optimizer"].dropna(
        subset=["final_loss"]
    ).copy()
    adamw_loss = float(opt.loc[opt["optimizer"] == "adamw", "final_loss"].iloc[0])
    cluster = sorted(opt.loc[(opt["final_loss"] - adamw_loss).abs() < 1e-4, "optimizer"])
    opt_sorted = opt.sort_values("final_loss")
    stats["optimizers"] = {
        "n": int(len(opt)),
        "rows": [
            {
                "name": str(row["optimizer"]),
                "epochs": float(row["final_epoch"]),
                "loss": float(row["final_loss"]),
                "grad_mean": float(row["mean_grad_norm"]),
                "grad_final": float(row["final_grad_norm"]),
                "clip": float(row["mean_clip_ratio"]),
                "step_ms": float(row["mean_step_time_ms"]),
                "tokens_per_sec": float(row["tokens_per_sec"])
                if pd.notna(row["tokens_per_sec"]) else None,
                "rss_mb": float(row["mean_rss_mb"]),
                "hours": float(row["duration_h"]),
            }
            for _, row in opt_sorted.iterrows()
        ],
        "cluster": cluster,
        "rho_loss_grad": mc._spearman(opt["final_loss"], opt["mean_grad_norm"]),
        "rho_loss_grad_final": mc._spearman(opt["final_loss"], opt["final_grad_norm"]),
        "rho_loss_clip": mc._spearman(opt["final_loss"], opt["mean_clip_ratio"]),
        "rho_loss_step": mc._spearman(opt["final_loss"], opt["mean_step_time_ms"]),
        "rho_loss_epochs": mc._spearman(opt["final_loss"], opt["final_epoch"]),
        "fast_timer": sorted(opt.loc[opt["mean_step_time_ms"] < 150, "optimizer"]),
        "rss_median": float(opt["mean_rss_mb"].median()),
        "rss_max": [
            str(opt.loc[opt["mean_rss_mb"].idxmax(), "optimizer"]),
            float(opt["mean_rss_mb"].max()),
        ],
        "step_min": [
            str(opt.loc[opt["mean_step_time_ms"].idxmin(), "optimizer"]),
            float(opt["mean_step_time_ms"].min()),
        ],
        "step_max": [
            str(opt.loc[opt["mean_step_time_ms"].idxmax(), "optimizer"]),
            float(opt["mean_step_time_ms"].max()),
        ],
    }

    # ------------------------------------------------------------- ablations
    def _rows(family, dropna=True, min_epochs=None):
        frame = analysis_runs[analysis_runs["family"] == family]
        if min_epochs:
            frame = frame[frame["final_epoch"].fillna(0) >= min_epochs]
        if dropna:
            frame = frame.dropna(subset=["final_loss"])
        return [
            {
                "run": str(row["run"]),
                "loss": float(row["final_loss"]) if pd.notna(row["final_loss"]) else None,
                "epochs": float(row["final_epoch"]) if pd.notna(row["final_epoch"]) else None,
                "params": float(row["total"]),
                "task": str(row["task"]),
            }
            for _, row in frame.sort_values("final_loss").iterrows()
        ]

    vit_pe = analysis_runs[analysis_runs["family"] == "vit_pe"].dropna(
        subset=["final_loss"]
    )
    vit_pe_spread = {
        task: float(group["final_loss"].max() - group["final_loss"].min())
        for task, group in vit_pe.groupby("task")
    }
    stats["ablations"] = {
        "norm": _rows("norm"),
        "pe": _rows("pe"),
        "switch": _rows("switch"),
        "batch": _rows("batch"),
        "task": _rows("task"),
        "tokenizer": _rows("tokenizer"),
        "moe_full": _rows("moe", min_epochs=100),
        "moe_smoke": int(
            (
                training.loc[training["family"] == "moe", "final_epoch"].fillna(0) < 100
            ).sum()
        ),
        "vit_pe_spread": vit_pe_spread,
        "vit_pe_n": int(len(vit_pe)),
        "vision": _rows("vision"),
        "audio": _rows("audio"),
        "realdata": _rows("realdata"),
    }


    # ------------------------------------------------------------- dynamics
    runs = pair_runs[pair_runs["task"] == "mlm"]["run"].tolist()
    curves = mc._curves(data["epochs"], runs)["loss"]
    if len(curves):
        p50 = np.median(curves, axis=0)
        stats["dynamics"] = {
            "n_curves": int(len(curves)),
            "p50": {
                "start": float(p50[0]),
                "q25": float(p50[12]),
                "q50": float(p50[25]),
                "q75": float(p50[38]),
                "end": float(p50[-1]),
            },
            "band": {
                "p10_end": float(np.percentile(curves[:, -1], 10)),
                "p90_end": float(np.percentile(curves[:, -1], 90)),
                "p10_start": float(np.percentile(curves[:, 0], 10)),
                "p90_start": float(np.percentile(curves[:, 0], 90)),
            },
            "rank_stability_10": mc._spearman(curves[:, 5], curves[:, -1]),
            "rank_stability_25": mc._spearman(curves[:, 12], curves[:, -1]),
        }
    speed = _convergence(epochs, analysis_runs)
    stats["convergence"] = {
        "by_family": {
            family: {
                "median": float(group["fraction"].median()) * 100,
                "n": int(len(group)),
            }
            for family, group in speed.groupby("family")
            if len(group) >= 5
        },
        "overall_median": float(speed["fraction"].median()) * 100,
    }

    # ------------------------------------------------------------ gradients
    block_stats = {}
    for group in C.BLOCK_GROUPS:
        rows = blocks[(blocks["group"] == group) & blocks["final"].notna()]
        if not len(rows):
            continue
        mean = rows["mean"].to_numpy()
        std = rows["std"].to_numpy()
        cv = np.divide(std, mean, out=np.full_like(std, np.nan), where=mean > 0)
        block_stats[group] = {
            "mean_median": float(np.nanmedian(mean)),
            "final_median": float(np.nanmedian(rows["final"])),
            "cv_median": float(np.nanmedian(cv)),
            "share_active": float((rows["final"] > 0).mean() * 100),
        }
    stats["gradients"] = {
        "blocks": block_stats,
        "clip_median": float(training["mean_clip_ratio"].median()),
        "clip_share_above_1": float((training["mean_clip_ratio"] > 1).mean() * 100),
        "clip_p90": _q(training["mean_clip_ratio"], 0.9),
        "nan_steps": int(training["nan_steps"].fillna(0).sum()),
        "inf_steps": int(training["inf_steps"].fillna(0).sum()),
        "repair_steps": int(training["repair_steps"].fillna(0).sum()),
        "runs_with_nan": int((training["nan_steps"].fillna(0) > 0).sum()),
        "runs_with_repair": int((training["repair_steps"].fillna(0) > 0).sum()),
        "grad_cv_loss_rho": mc._spearman(
            (training["std_grad_norm"] / training["mean_grad_norm"]).replace([np.inf, -np.inf], np.nan),
            training["final_loss"],
        ),
        # ``cpu_rss_peak_mb`` was only logged for part of the campaign (the
        # vision families), so the headline numbers use the mean resident
        # memory, which every run reports.
        "rss_median_mb": float(training["rss_peak_mb"].median()),
        "rss_max_mb": float(training["rss_peak_mb"].max()),
        "rss_runs_with_peak": int(training["rss_peak_mb"].notna().sum()),
        "rss_mean_median_mb": float(analysis_runs["mean_rss_mb"].median()),
        "rss_mean_p90_mb": _q(analysis_runs["mean_rss_mb"], 0.9),
        "rss_mean_max_mb": float(analysis_runs["mean_rss_mb"].max()),
        "rss_mean_max_run": str(analysis_runs.loc[analysis_runs["mean_rss_mb"].idxmax(), "run"]),
        "rss_vision_median_mb": float(
            analysis_runs.loc[analysis_runs["family"].isin(["vision", "vit_pe"]), "mean_rss_mb"].median()
        ),
    }

    # -------------------------------------------------------------- compute
    fit_rows = pair_runs[(pair_runs["task"] == "mlm")].dropna(subset=["duration_h", "params_m"])
    fit_rows = fit_rows[(fit_rows["duration_h"] > 0) & (fit_rows["params_m"] > 0)]
    x = np.log10(fit_rows["params_m"].to_numpy())
    y = np.log10(fit_rows["duration_h"].to_numpy())
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    r2 = 1 - float(((y - predicted) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
    family_cost = []
    for family, group in analysis_runs.groupby("family"):
        all_rows = training[training["family"] == family]
        family_cost.append(
            {
                "family": family,
                "n": int(len(all_rows)),
                "n_analysis": int(len(group)),
                "hours": float(all_rows["duration_h"].sum()),
                "share": float(100 * all_rows["duration_h"].sum() / training["duration_h"].sum()),
                "minutes_per_mparam": float(group["minutes_per_mparam"].median()),
                "tokens_per_sec": float(group["tokens_per_sec"].median()),
                "params_median": float(group["total"].median()),
                "loss_median": float(group["final_loss"].median()),
                "med_epoch": float(group["final_epoch"].median()),
                "smoke": int((all_rows["final_epoch"].fillna(0) < 100).sum()),
            }
        )
    family_cost.sort(key=lambda row: row["hours"], reverse=True)
    stats["compute"] = {
        "time_slope": float(slope),
        "time_r2": float(r2),
        "minutes_per_mparam_median": float(training["minutes_per_mparam"].median()),
        "tokens_per_sec_median": float(training["tokens_per_sec"].median()),
        "by_family": family_cost,
        "params_loss_rho": stats["params"]["loss_rho"],
    }

    # ------------------------------------------------------------ quantized
    mlm = acc[(acc["task"] == "mlm") & acc["deploy_format"].str.startswith("hf_")].copy()
    pivot = mlm.pivot_table(index="combo_id", columns="deploy_format", values="accuracy")
    quant = {}
    for fmt in [c for c in pivot.columns if c != "hf_none"]:
        delta = (pivot[fmt] - pivot["hf_none"]).dropna() * 100
        if not len(delta):
            continue
        low, high = mc._bootstrap_ci(delta)
        quant[fmt] = {
            "n": int(len(delta)),
            "mean_delta_pp": float(delta.mean()),
            "median_delta_pp": float(delta.median()),
            "ci_low": float(low),
            "ci_high": float(high),
            "pct_worse": float((delta < -1e-9).mean() * 100),
            "pct_better": float((delta > 1e-9).mean() * 100),
            "pct_equal": float((delta.abs() <= 1e-9).mean() * 100),
        }
    # Per-combo agreement, used for the cluster bootstrap (a single global
    # agreement would hide the between-combo variance).
    inference = data["inference"]
    both_runtimes = inference[inference["runtime"].isin(("none", "v2"))].copy()
    both_runtimes["sample_key"] = (
        both_runtimes["combo_id"]
        + "|"
        + both_runtimes["sample_id"].astype(str)
        + "|"
        + both_runtimes["mask_position"].astype(str)
    )
    wide = both_runtimes.pivot_table(
        index=["combo_id", "sample_key"],
        columns="deploy_format",
        values="predicted",
        aggfunc="first",
    )
    agreement = {}
    for codec, fmt in (("q8_0", "hf_q8_0_v2"), ("q4_0", "hf_q4_0_v2"), ("i2_s", "hf_i2_s_v2")):
        if fmt not in wide.columns or "hf_none" not in wide.columns:
            continue
        pairs = wide[["hf_none", fmt]].dropna()
        per_combo = pairs.groupby(level="combo_id").apply(
            lambda frame: float((frame["hf_none"] == frame[fmt]).mean())
        )
        low, high = mc._bootstrap_ci(per_combo.to_numpy())
        agreement[codec] = {
            "mean": float(per_combo.mean() * 100),
            "ci_low": float(low * 100),
            "ci_high": float(high * 100),
            "n_combos": int(len(per_combo)),
        }

    if "hf_i2_s_v2" in pivot.columns:
        delta_i2 = (pivot["hf_i2_s_v2"] - pivot["hf_none"]).dropna() * 100
        merged_i2 = analysis_runs.set_index("run")[["params_m"]].join(
            delta_i2.rename("delta"), how="inner"
        )
        i2_rho = mc._spearman(np.log10(merged_i2["params_m"]), merged_i2["delta"])
    else:
        i2_rho = float("nan")

    stats["quantization"] = {
        "formats": quant,
        "paired": paired.to_dict(orient="records"),
        "agreement": agreement,
        "top5_by_format": {
            fmt: float(group["top5"].mean() * 100)
            for fmt, group in mlm.groupby("deploy_format")
        },
        "top1_by_format": {
            fmt: float(group["accuracy"].mean() * 100)
            for fmt, group in mlm.groupby("deploy_format")
        },
        "combos_per_format": _table_counts(mlm["deploy_format"]),
        "i2_delta_params_rho": i2_rho,
    }

    # ----------------------------------------------------------- reliability
    family_reliability = []
    for family, group in training.groupby("family"):
        family_reliability.append(
            {
                "family": family,
                "n": int(len(group)),
                "failed": int((group["status"] != "OK").sum()),
                "failure_rate": float((group["status"] != "OK").mean() * 100),
            }
        )
    stats["reliability"] = {
        "by_family": sorted(family_reliability, key=lambda row: -row["failure_rate"]),
        "failed_notes": _table_counts(
            training.loc[training["status"] != "OK", "notes"].fillna("unknown")
        ),
        "stage_status": stats["population"]["stage_status"],
    }

    # ---------------------------------------------------------- correlations
    features = pd.DataFrame(
        {
            "params_m": pair_runs["params_m"],
            "final_loss": pair_runs["final_loss"],
            "mean_loss": pair_runs["mean_loss"],
            "final_accuracy": pair_runs["final_accuracy"],
            "grad_norm": pair_runs["mean_grad_norm"],
            "step_time_ms": pair_runs["mean_step_time_ms"],
            "duration_h": pair_runs["duration_h"],
            "tokens_per_sec": pair_runs["tokens_per_sec"],
            "epochs": pair_runs["final_epoch"],
        }
    ).dropna()
    corr = features.corr(method="spearman")
    all_loss = analysis_runs.dropna(subset=["final_loss", "params_m"])
    stats["correlations"] = {
        "labels": list(corr.columns),
        "matrix": corr.to_numpy().round(2).tolist(),
        "n": int(len(features)),
        "named": {
            "duration_epochs": float(corr.loc["duration_h", "epochs"]),
            "loss_epochs": float(corr.loc["mean_loss", "epochs"]),
            "grad_meanloss": float(corr.loc["grad_norm", "mean_loss"]),
            "tokens_step": float(corr.loc["tokens_per_sec", "step_time_ms"]),
            "duration_params": float(corr.loc["duration_h", "params_m"]),
            "params_grad": float(corr.loc["params_m", "grad_norm"]),
        },
        "params_loss_all": mc._spearman(
            np.log10(all_loss["params_m"]), all_loss["final_loss"]
        ),
        "params_loss_all_n": int(len(all_loss)),
    }
    return stats


def _convergence(epochs: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    """Fraction of a run's schedule needed to reach 90% of its loss drop."""
    records = []
    for run, group in epochs.groupby("run", sort=False):
        group = group.sort_values("epoch")
        loss = group["loss"].to_numpy(dtype=float)
        if len(loss) < 5 or not np.isfinite(loss).all():
            continue
        drop = loss[0] - loss[-1]
        if drop <= 0:
            continue
        index = int(np.argmax(loss <= loss[-1] + 0.1 * drop))
        records.append({"run": run, "fraction": index / max(len(loss) - 1, 1)})
    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame(columns=["run", "fraction", "family"])
        return frame
    return frame.merge(training[["run", "family"]], on="run", how="left")


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------
def tex_table(caption: str, label: str, header: str, rows: list[list[str]], align: str) -> str:
    """Return a booktabs table with the given caption, label and rows."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{" + align + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def figure(path: str, caption: str, label: str) -> str:
    """Return a figure float that references a generated PDF."""
    return "\n".join(
        [
            r"\begin{figure}[tbp]",
            r"\centering",
            r"\fitfigure{\includegraphics{" + path + "}}",
            r"\caption{" + caption + "}",
            r"\label{" + label + "}",
            r"\end{figure}",
        ]
    )


def _pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}\\%"


def build_tables(st: dict, lang: str) -> dict[str, str]:
    """Render every table of the annex in the requested language."""
    en = lang == "en"
    reliability = {row["family"]: row for row in st["reliability"]["by_family"]}
    compute = {row["family"]: row for row in st["compute"]["by_family"]}

    rows = []
    for family in C.FAMILIES:
        if family not in compute:
            continue
        row = compute[family]
        ok = 100 - reliability.get(family, {"failure_rate": 0})["failure_rate"]
        rows.append(
            [
                C.family_label(family, lang),
                f"{row['n']:,}".replace(",", "{,}"),
                str(row["smoke"]),
                f"{row['med_epoch']:.0f}",
                f"{row['params_median'] / 1e6:.2f}",
                f"{row['loss_median']:.3f}" if np.isfinite(row["loss_median"]) else "n/a",
                f"{row['minutes_per_mparam']:.1f}",
                _pct(ok),
            ]
        )
    tables = {
        "family_table": tex_table(
            "Población de la ronda~3 por familia: corridas, corridas excluidas como pruebas de humo (menos de 100 épocas), mediana de épocas y parámetros de las corridas analizadas, mediana de la pérdida final, minutos por Mparámetro y tasa de éxito del entrenamiento."
            if not en
            else "Round~3 population by family: runs, median parameters, median final loss, wall hours per run, minutes per Mparam and training success rate.",
            "tab:r3-families",
            (r"Family & $n$ & Smoke & Epochs & Params (M) & Final loss & min/Mparam & OK") if en
            else (r"Familia & $n$ & Humo & Épocas & Parám. (M) & Pérdida & min/Mparám & OK"),
            rows,
            "lrrrrrrr",
        )
    }

    rows = []
    for rank, row in enumerate(st["top_loss"], start=1):
        rows.append(
            [
                str(rank),
                C.tex_tt(row["run"]),
                f"{row['params'] / 1e6:.2f}",
                f"{row['loss']:.3f}",
                f"{row['duration_h']:.2f}",
            ]
        )
    tables["top_loss_table"] = tex_table(
        "Las diez mejores parejas híbridas por pérdida final de entrenamiento."
        if not en
        else "The ten best hybrid pairs by final training loss.",
        "tab:r3-top-loss",
        r"\# & Pair & Params (M) & Final loss & Hours" if en
        else r"\# & Pareja & Parám. (M) & Pérdida & Horas",
        rows,
        "rlrrr",
    )

    rows = []
    for rank, row in enumerate(st["top_inference"], start=1):
        loss = f"{row['loss']:.3f}" if row["loss"] is not None else "n/a"
        rows.append(
            [
                str(rank),
                C.tex_tt(row["run"]),
                _pct(100 * row["top1"], 2),
                _pct(100 * row["top5"], 1),
                loss,
            ]
        )
    tables["top_inference_table"] = tex_table(
        "Los diez mejores pares por acierto top-5 retenido (exportación sin cuantizar)."
        if not en
        else "The ten best pairs by held-out top-5 hit rate (unquantized export).",
        "tab:r3-top-inference",
        r"\# & Pair & Top-1 & Top-5 & Final loss" if en
        else r"\# & Pareja & Top-1 & Top-5 & Pérdida final",
        rows,
        "rlrrr",
    )

    rows = []
    for group in C.BLOCK_GROUPS:
        if group not in st["gradients"]["blocks"]:
            continue
        row = st["gradients"]["blocks"][group]
        rows.append(
            [
                C.block_label(group, lang),
                f"{row['mean_median']:.3f}",
                f"{row['final_median']:.3f}",
                f"{row['cv_median']:.2f}" if np.isfinite(row["cv_median"]) else "n/a",
                _pct(row["share_active"]),
            ]
        )
    tables["grad_table"] = tex_table(
        "Norma del gradiente por bloque de parámetros: mediana de la media por corrida, mediana del último paso, coeficiente de variación y fracción de corridas con bloque activo."
        if not en
        else "Gradient norm per parameter block: median of the per-run mean, median of the last step, coefficient of variation and share of runs where the block is active.",
        "tab:r3-gradients",
        r"Block & Mean & Final & CV & Active" if en else r"Bloque & Media & Final & CV & Activo",
        rows,
        "lrrrr",
    )

    rows = []
    for row in st["compute"]["by_family"]:
        rows.append(
            [
                C.family_label(row["family"], lang),
                f"{row['hours']:,.1f}".replace(",", "{,}"),
                _pct(row["share"]),
                f"{row['minutes_per_mparam']:.1f}",
                f"{row['tokens_per_sec']:,.0f}".replace(",", "{,}")
                if np.isfinite(row["tokens_per_sec"])
                else "n/a",
            ]
        )
    tables["cost_table"] = tex_table(
        "Coste de entrenamiento por familia: horas de pared acumuladas, proporción del cómputo total, mediana de minutos por Mparámetro y mediana de tokens por segundo."
        if not en
        else "Training cost by family: accumulated wall hours, share of the total compute, median minutes per Mparam and median tokens per second.",
        "tab:r3-cost",
        r"Family & Hours & Share & min/Mparam & tok/s" if en
        else r"Familia & Horas & Proporción & min/Mparám & tok/s",
        rows,
        "lrrrr",
    )

    quant = st["quantization"]["formats"]
    order = [
        "hf_none",
        "hf_q8_0_v1", "hf_q8_0_v2",
        "hf_q4_0_v1", "hf_q4_0_v2",
        "hf_i2_s_v1", "hf_i2_s_v2",
    ]
    rows = []
    for fmt in order:
        if fmt == "hf_none":
            combos = st["quantization"]["combos_per_format"].get(fmt, 0)
            rows.append(
                [
                    "none",
                    str(combos),
                    f"{st['quantization']['top1_by_format'].get(fmt, float('nan')):.2f}",
                    f"{st['quantization']['top5_by_format'].get(fmt, float('nan')):.1f}",
                    "0.000",
                    "--",
                    "0.0\\%",
                ]
            )
            continue
        row = quant.get(fmt)
        if not row:
            continue
        rows.append(
            [
                C.tex_tt(fmt.replace("hf_", "")),
                str(row["n"]),
                f"{st['quantization']['top1_by_format'].get(fmt, float('nan')):.2f}",
                f"{st['quantization']['top5_by_format'].get(fmt, float('nan')):.1f}",
                f"{row['mean_delta_pp']:+.3f}",
                f"{row['ci_low']:+.3f} / {row['ci_high']:+.3f}",
                _pct(row["pct_worse"]),
            ]
        )
    tables["format_table"] = tex_table(
        "Exactitud retenida por formato de exportación (MLM): pares evaluados, top-1 y top-5 medios, delta medio frente a la exportación sin cuantizar, intervalo bootstrap del 95\\% y fracción de pares que empeoran."
        if not en
        else "Held-out accuracy per export format (MLM): evaluated pairs, mean top-1 and top-5, mean delta against the unquantized export, 95\\% bootstrap interval and share of pairs that degrade.",
        "tab:r3-formats",
        r"Format & $n$ & Top-1 & Top-5 & $\Delta$ (pp) & 95\% CI & Worse" if en
        else r"Formato & $n$ & Top-1 & Top-5 & $\Delta$ (pp) & IC 95\% & Peor",
        rows,
        "lrrrcrr",
    )

    rows = []
    for row in st["reliability"]["by_family"]:
        if row["n"] < 3:
            continue
        rows.append(
            [
                C.family_label(row["family"], lang),
                str(row["n"]),
                str(row["failed"]),
                _pct(row["failure_rate"]),
            ]
        )
    tables["reliability_table"] = tex_table(
        "Fallos de entrenamiento por familia."
        if not en
        else "Training failures by family.",
        "tab:r3-reliability",
        r"Family & $n$ & Failed & Rate" if en else r"Familia & $n$ & Fallos & Tasa",
        rows,
        "lrrr",
    )

    rows = []
    for row in st["optimizers"]["rows"]:
        rows.append(
            [
                C.tex_tt(row["name"]),
                f"{row['epochs']:.0f}",
                f"{row['loss']:.3f}",
                f"{row['grad_mean']:.2f} / {row['grad_final']:.2f}",
                f"{row['clip']:.2f}",
                f"{row['step_ms']:.0f}",
                f"{row['rss_mb']:.0f}",
            ]
        )
    tables["optimizer_table"] = tex_table(
        "Ablación de optimizadores sobre el mismo esqueleto: épocas efectivas, pérdida final, norma del gradiente (media y final), cociente de recorte, tiempo por paso y memoria residente media. Orden por pérdida."
        if not en
        else "Optimizer ablation on the shared backbone: effective epochs, final loss, gradient norm (mean and final), clip ratio, step time and mean resident memory. Sorted by loss.",
        "tab:r3-optimizers",
        r"Optimizer & Ep. & Loss & Grad (m/f) & Clip & Step (ms) & RSS (MB)" if en
        else r"Optimizador & Ép. & Pérdida & Grad (m/f) & Recorte & Paso (ms) & RSS (MB)",
        rows,
        "lrrrrrr",
    )

    def _ablation_table(family_key, label, caption_en, caption_es, header_en, header_es):
        rows = []
        for row in st["ablations"][family_key]:
            if row["loss"] is None:
                continue
            name = str(row["run"])
            for prefix in (f"{family_key}_", "trans_"):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            rows.append(
                [
                    C.tex_tt(name),
                    f"{row['loss']:.3f}",
                    f"{row['epochs']:.0f}",
                    f"{row['params'] / 1e3:.0f}k",
                ]
            )
        return tex_table(
            caption_es if not en else caption_en, label, header_en if en else header_es,
            rows, "lrrr",
        )

    tables["norm_table"] = _ablation_table(
        "norm", "tab:r3-norms",
        "Normalization ablation: final loss, effective epochs and parameters.",
        "Ablación de normalización: pérdida final, épocas efectivas y parámetros.",
        r"Variant & Final loss & Epochs & Params",
        r"Variante & Pérdida & Épocas & Parám.",
    )
    tables["pe_table"] = _ablation_table(
        "pe", "tab:r3-pe",
        "Text positional-encoding ablation (the six-epoch \\texttt{hope} run is a smoke test and is excluded).",
        "Ablación de codificación posicional de texto (la corrida de seis épocas de \\texttt{hope} es una prueba de humo y queda excluida).",
        r"Encoding & Final loss & Epochs & Params",
        r"Codificación & Pérdida & Épocas & Parám.",
    )
    tables["switch_table"] = _ablation_table(
        "switch", "tab:r3-switches",
        "Structural switches with a full schedule, ordered by final loss. The names are shortened (\\texttt{trans\\_} dropped); smoke runs are excluded.",
        "Interruptores estructurales con calendario completo, ordenados por pérdida final. Los nombres están abreviados (sin \\texttt{trans\\_}); las pruebas de humo quedan excluidas.",
        r"Switch & Final loss & Epochs & Params",
        r"Interruptor & Pérdida & Épocas & Parám.",
    )

    rows = []
    for row in st["atlas"]["best_b"] + st["atlas"]["worst_b"]:
        top5 = f"{row['top5'] * 100:.1f}\\%" if row["top5"] is not None else "n/a"
        other = f"{row['median_other']:.3f}" if np.isfinite(row["median_other"]) else "n/a"
        rows.append(
            [
                C.tex_tt(row["mixer"]),
                str(row["n"]),
                f"{row['median']:.3f}",
                other,
                top5,
            ]
        )
    tables["best_b_table"] = tex_table(
        "Segundos mixers: ocho mejores y ocho peores por pérdida mediana en el rol B, con el número de pares, la mediana del mismo mixer en el rol A y el top-5 retenido medio de sus pares."
        if not en
        else "Second mixers: the eight best and eight worst by median loss in the B role, with the pair count, the same mixer's median in the A role and the mean held-out top-5 of its pairs.",
        "tab:r3-best-b",
        r"Mixer & Pairs & B median & A median & Top-5" if en
        else r"Mixer & Pares & Mediana B & Mediana A & Top-5",
        rows,
        "lrrrr",
    )

    rows = []
    for row in st["atlas"]["synergy"]["top"] + st["atlas"]["synergy"]["bottom"]:
        rows.append(
            [
                C.tex_tt(row["run"]),
                f"{row['loss']:.3f}",
                f"{row['expected']:.3f}",
                f"{row['synergy']:+.3f}",
            ]
        )
    tables["synergy_table"] = tex_table(
        "Las mejores y peores interacciones por sinergia ($S_{A,B}=(\\mu_A+\\mu_B)/2-L_{A,B}$). Valores positivos: el par supera lo que predicen sus marginales."
        if not en
        else "The best and worst mixer interactions by synergy ($S_{A,B}=(\\mu_A+\\mu_B)/2-L_{A,B}$). Positive values mean the pair beats what its parts predict.",
        "tab:r3-synergy",
        r"Pair & Pair loss & Expected & Synergy" if en
        else r"Par & Pérdida & Esperada & Sinergia",
        rows,
        "lrrr",
    )
    return tables


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Compute the statistics, dump them and render the annexes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    C.OUT.mkdir(parents=True, exist_ok=True)
    stats = compute_stats()
    (C.OUT / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {C.OUT / 'stats.json'}")
    if args.stats_only:
        return
    import annex_text

    for lang, target in (("en", EN_ANNEX), ("es", ES_ANNEX)):
        tables = build_tables(stats, lang)
        text = annex_text.render(stats, tables, lang)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target.relative_to(C.PROJECT)} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
