"""Bilingual prose for the Experiment 3 appendix.

``make_report.py`` computes the statistics, renders the tables and calls
:func:`render` with the table/figure snippets; this module only owns the text.
Placeholders use the ``«name»`` form so LaTeX braces and math stay untouched.
"""
from __future__ import annotations

import re

import math

from common import PROJECT, fmt_pct, tex_tt  # noqa: F401


def fmt_num(value, digits: int = 2) -> str:
    """Fixed-point number with thousands separators, ``n/a`` for missing data."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:,.{digits}f}".replace(",", "{,}")


def fmt_int(value) -> str:
    """Thousands-separated integer, ``n/a`` for missing data."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{int(round(number)):,}".replace(",", "{,}")

# ---------------------------------------------------------------------------
# Value assembly
# ---------------------------------------------------------------------------


def values(stats: dict) -> dict[str, str]:
    """Flatten the nested statistics into the strings used by the templates."""
    pop = stats["population"]
    par = stats["params"]
    comp = stats["composition"]["median_share"]
    loss = stats["loss"]
    dyn = stats.get("dynamics", {})
    conv = stats["convergence"]
    grad = stats["gradients"]
    comp_cost = stats["compute"]
    quant = stats["quantization"]
    rel = stats["reliability"]
    corr = stats["correlations"]
    inf = stats["inference"]
    atlas = stats["atlas"]
    opt = stats["optimizers"]
    abl = stats["ablations"]

    def block(name):
        return grad["blocks"].get(name, {})

    roles = atlas["roles"]
    syn = atlas["synergy"]
    opt_rows = opt["rows"]

    def opt_row(name):
        return next(row for row in opt_rows if row["name"] == name)

    def abl_loss(family, run):
        row = next(r for r in abl[family] if r["run"] == run)
        return row["loss"]

    def tex_list(names):
        return ", ".join(tex_tt(name) for name in names)

    opt_cluster_names = tex_list(opt["cluster"])
    opt_best, opt_second, opt_third = opt_rows[0], opt_rows[1], opt_rows[2]
    opt_worst = opt_rows[-1]
    timer_rows = [opt_row(name) for name in ("af_muon", "ada_muon", "ea4llm")]
    comparable = [row for row in opt_rows if row["name"] not in opt["fast_timer"]]
    step_lo = min(comparable, key=lambda row: row["step_ms"])
    step_hi = max(comparable, key=lambda row: row["step_ms"])

    v = {
        "n_runs": fmt_int(pop["n_runs"]),
        "n_records": fmt_int(pop["n_records"]),
        "n_stages": fmt_int(pop["n_stages"]),
        "n_pairs": fmt_int(pop["n_pairs"]),
        "n_singles": fmt_int(pop["n_singles"]),
        "n_ablations": fmt_int(pop["n_ablations"]),
        "n_mixers": str(pop["n_mixers"]),
        "compute_h": fmt_num(pop["compute_h"], 0),
        "compute_days": fmt_num(pop["compute_h"] / 24, 0),
        "campaign_days": fmt_num(pop["campaign_days"], 1),
        "campaign_start": pop["campaign_start"][:10],
        "campaign_end": pop["campaign_end"][:10],
        "steps_total": f"{pop['steps_total'] / 1e6:.1f}",
        "steps_median": fmt_int(pop["steps_median"]),
        "duration_median": fmt_num(pop["duration_median_h"], 2),
        "duration_p90": fmt_num(pop["duration_p90_h"], 2),
        "duration_max": fmt_num(pop["duration_max_h"], 2),
        "training_ok": fmt_int(pop["training_ok"]),
        "training_failed": str(pop["training_failed"]),
        "early_stop": fmt_int(pop["early_stop"]),
        "ran_full": fmt_int(pop["n_runs"] - pop["early_stop"]),
        "short_runs": str(pop["short_runs"]),
        "smoke_runs": str(pop["smoke_runs"]),
        "n_analysis": fmt_int(pop["n_analysis"]),
        "deploy_total": fmt_int(pop["stage_counts"].get("deploy", 0)),
        "deploy_failed": fmt_int(pop["stage_status"].get("deploy", {}).get("FAILED", 0)),
        "export_total": fmt_int(pop["stage_counts"].get("transformers-export", 0)),
        "inference_total": fmt_int(pop["stage_counts"].get("inference-eval", 0)),
        "inference_failed": str(pop["stage_status"].get("inference-eval", {}).get("FAILED", 0)),
        "params_min": fmt_int(par["min"]),
        "params_median": fmt_int(par["median"]),
        "pair_params_median": fmt_int(par["pair_median"]),
        "params_p90": fmt_int(par["p90"]),
        "params_max": fmt_int(par["max"]),
        "params_max_m": fmt_num(par["max"] / 1e6, 2),
        "largest_run": tex_tt(par["largest_run"]),
        "smallest_run": tex_tt(par["smallest_run"]),
        "share_emb": fmt_num(comp["embeddings"], 1),
        "share_attn": fmt_num(comp["attention"], 1),
        "share_ffn": fmt_num(comp["ffn"], 1),
        "share_head": fmt_num(comp["head"], 1),
        "share_norm": fmt_num(comp["norm"], 1),
        "mixer_cheap": ", ".join(
            f"{tex_tt(name)} ({value / 1e6:.2f}M)" for name, value in stats["mixer_cost"]["cheapest"][:3]
        ),
        "mixer_expensive": ", ".join(
            f"{tex_tt(name)} ({value / 1e6:.2f}M)" for name, value in stats["mixer_cost"]["most_expensive"][:3]
        ),
        "rho_params_loss": f"{par['loss_rho']:+.2f}",
        "loss_median": fmt_num(loss["pair_median"], 2),
        "loss_min": fmt_num(loss["pair_min"], 2),
        "loss_max": fmt_num(loss["pair_max"], 2),
        "moe_loss": fmt_num(loss["by_family"].get("moe", {}).get("median", float("nan")), 2),
        "best_pair": tex_tt(stats["top_loss"][0]["run"]) if stats["top_loss"] else "--",
        "best_pair_loss": fmt_num(stats["top_loss"][0]["loss"], 3) if stats["top_loss"] else "--",
        "worst_pair": tex_tt(stats["worst_loss"][0]["run"]) if stats["worst_loss"] else "--",
        "worst_pair_loss": fmt_num(stats["worst_loss"][0]["loss"], 2) if stats["worst_loss"] else "--",
        "rank_10": fmt_num(dyn.get("rank_stability_10", float("nan")), 2),
        "rank_25": fmt_num(dyn.get("rank_stability_25", float("nan")), 2),
        "curve_start": fmt_num(dyn.get("p50", {}).get("start", float("nan")), 2),
        "curve_q25": fmt_num(dyn.get("p50", {}).get("q25", float("nan")), 2),
        "curve_q50": fmt_num(dyn.get("p50", {}).get("q50", float("nan")), 2),
        "curve_q75": fmt_num(dyn.get("p50", {}).get("q75", float("nan")), 2),
        "curve_end": fmt_num(dyn.get("p50", {}).get("end", float("nan")), 2),
        "band_p10": fmt_num(dyn.get("band", {}).get("p10_end", float("nan")), 2),
        "band_p90": fmt_num(dyn.get("band", {}).get("p90_end", float("nan")), 2),
        "conv_median": fmt_num(conv["overall_median"], 0),
        "conv_pair": fmt_num(conv["by_family"].get("pair", {}).get("median", float("nan")), 0),
        "conv_vision": fmt_num(conv["by_family"].get("vision", {}).get("median", float("nan")), 0),
        "conv_norm": fmt_num(conv["by_family"].get("norm", {}).get("median", float("nan")), 0),
        "g_head": fmt_num(block("head").get("mean_median", float("nan")), 2),
        "g_head_final": fmt_num(block("head").get("final_median", float("nan")), 2),
        "g_attn": fmt_num(block("attention").get("mean_median", float("nan")), 2),
        "g_attn_final": fmt_num(block("attention").get("final_median", float("nan")), 2),
        "g_ffn": fmt_num(block("ffn").get("mean_median", float("nan")), 2),
        "g_ffn_final": fmt_num(block("ffn").get("final_median", float("nan")), 2),
        "g_norm": fmt_num(block("norm").get("mean_median", float("nan")), 2),
        "g_emb": fmt_num(block("embeddings").get("mean_median", float("nan")), 2),
        "cv_head": fmt_num(block("head").get("cv_median", float("nan")), 2),
        "cv_attn": fmt_num(block("attention").get("cv_median", float("nan")), 2),
        "cv_ffn": fmt_num(block("ffn").get("cv_median", float("nan")), 2),
        "clip_median": fmt_num(grad["clip_median"], 2),
        "clip_above": fmt_num(grad["clip_share_above_1"], 1),
        "clip_p90": fmt_num(grad["clip_p90"], 2),
        "inf_steps": str(grad["inf_steps"]),
        "repair_runs": str(grad["runs_with_repair"]),
        "rho_grad_cv_loss": f"{grad['grad_cv_loss_rho']:+.2f}",
        "rss_mean_median": fmt_int(grad["rss_mean_median_mb"]),
        "rss_mean_p90": fmt_int(grad["rss_mean_p90_mb"]),
        "rss_mean_max": fmt_int(grad["rss_mean_max_mb"]),
        "rss_max_run": tex_tt(grad["rss_mean_max_run"]),
        "rss_vision": fmt_int(grad["rss_vision_median_mb"]),
        "rss_peak_runs": str(grad["rss_runs_with_peak"]),
        "rss_peak_max": fmt_int(grad["rss_max_mb"]),
        "time_slope": fmt_num(comp_cost["time_slope"], 2),
        "time_r2": fmt_num(comp_cost["time_r2"], 2),
        "min_per_mparam": fmt_num(comp_cost["minutes_per_mparam_median"], 0),
        "min_per_mparam_pair": fmt_num(
            next((row["minutes_per_mparam"] for row in comp_cost["by_family"] if row["family"] == "pair"), float("nan")), 0
        ),
        "min_per_mparam_opt": fmt_num(
            next((row["minutes_per_mparam"] for row in comp_cost["by_family"] if row["family"] == "optimizer"), float("nan")), 0
        ),
        "min_per_mparam_single": fmt_num(
            next((row["minutes_per_mparam"] for row in comp_cost["by_family"] if row["family"] == "single"), float("nan")), 0
        ),
        "tokens_audio": fmt_num(
            next((row["tokens_per_sec"] for row in comp_cost["by_family"] if row["family"] == "audio"), float("nan")), 1
        ),
        "tokens_vision": fmt_num(
            next((row["tokens_per_sec"] for row in comp_cost["by_family"] if row["family"] == "vision"), float("nan")), 1
        ),
        "tokens_median": fmt_num(comp_cost["tokens_per_sec_median"], 0),
        "pair_share": fmt_num(comp_cost["by_family"][0]["share"], 1),
        "pair_hours": fmt_int(comp_cost["by_family"][0]["hours"]),
        "second_family": comp_cost["by_family"][1]["family"] if len(comp_cost["by_family"]) > 1 else "single",
        "second_share": fmt_num(comp_cost["by_family"][1]["share"], 1) if len(comp_cost["by_family"]) > 1 else "0",
        "n_combos_inf": fmt_int(inf["n_combos"]),
        "n_rows_inf": fmt_int(inf["n_rows"]),
        "n_mlm_combos": fmt_int(inf["n_pairs_combos"]),
        "top1_mean": fmt_num(inf["top1_mean"] * 100, 2),
        "top5_mean": fmt_num(inf["top5_mean"] * 100, 1),
        "rho_loss_top5": f"{inf['loss_top5_rho']:+.2f}",
        "q8_delta": f"{quant['formats']['hf_q8_0_v2']['mean_delta_pp']:+.2f}",
        "q8_equal": fmt_num(quant["formats"]["hf_q8_0_v2"]["pct_equal"], 1),
        "q8_worse": fmt_num(quant["formats"]["hf_q8_0_v2"]["pct_worse"], 1),
        "q4_delta": f"{quant['formats']['hf_q4_0_v2']['mean_delta_pp']:+.2f}",
        "q4_equal": fmt_num(quant["formats"]["hf_q4_0_v2"]["pct_equal"], 1),
        "q4_worse": fmt_num(quant["formats"]["hf_q4_0_v2"]["pct_worse"], 1),
        "i2_delta": f"{quant['formats']['hf_i2_s_v2']['mean_delta_pp']:+.2f}",
        "i2_equal": fmt_num(quant["formats"]["hf_i2_s_v2"]["pct_equal"], 1),
        "i2_worse": fmt_num(quant["formats"]["hf_i2_s_v2"]["pct_worse"], 1),
        "agree_q8": fmt_num(quant["agreement"]["q8_0"]["mean"], 1),
        "agree_q4": fmt_num(quant["agreement"]["q4_0"]["mean"], 1),
        "agree_i2": fmt_num(quant["agreement"]["i2_s"]["mean"], 1),
        "top5_none": fmt_num(quant["top5_by_format"]["hf_none"], 1),
        "top5_q8_v1": fmt_num(quant["top5_by_format"]["hf_q8_0_v1"], 1),
        "top5_i2_v2": fmt_num(quant["top5_by_format"]["hf_i2_s_v2"], 1),
        "i2_params_rho": f"{quant['i2_delta_params_rho']:+.2f}",
        "failed_total": str(pop["training_failed"]),
        "failure_rate": fmt_num(100 * pop["training_failed"] / pop["n_runs"], 1),
        "rho_duration_epochs": f"{corr['named']['duration_epochs']:+.2f}",
        "rho_duration_params": f"{corr['named']['duration_params']:+.2f}",
        "rho_loss_epochs": f"{corr['named']['loss_epochs']:+.2f}",
        "rho_grad_meanloss": f"{corr['named']['grad_meanloss']:+.2f}",
        "rho_params_grad": f"{corr['named']['params_grad']:+.2f}",
        "rho_tokens_step": f"{corr['named']['tokens_step']:+.2f}",
        "rho_params_loss_all": f"{corr['params_loss_all']:+.2f}",
        "n_all_rows": fmt_int(corr["params_loss_all_n"]),
        # ------------------------------------------------------------- atlas
        "n_cells": fmt_int(atlas["n_cells"]),
        "n_measured": fmt_int(atlas["n_measured"]),
        "n_holes": str(atlas["n_holes"]),
        "n_common_roles": str(roles["n_common"]),
        "range_a": fmt_num(roles["range_a"], 2),
        "range_b": fmt_num(roles["range_b"], 2),
        "best_a_name": tex_tt(roles["best_a"][0]),
        "best_a_med": fmt_num(roles["best_a"][1], 3),
        "worst_a_name": tex_tt(roles["worst_a"][0]),
        "worst_a_med": fmt_num(roles["worst_a"][1], 3),
        "best_b_name": tex_tt(roles["best_b"][0]),
        "best_b_med": fmt_num(roles["best_b"][1], 3),
        "worst_b_name": tex_tt(roles["worst_b"][0]),
        "worst_b_med": fmt_num(roles["worst_b"][1], 3),
        "rho_roles": f"{roles['rho']:+.2f}",
        "only_a": tex_list(roles["only_a"]),
        "only_b": tex_list(roles["only_b"]),
        "single_median": fmt_num(atlas["singles"]["median"], 3),
        "best_single": tex_tt(atlas["singles"]["best"][0]),
        "best_single_loss": fmt_num(atlas["singles"]["best"][1], 3),
        "pairs_better_single": fmt_int(atlas["singles"]["pairs_better_than_best_single"]),
        "syn_median": fmt_num(syn["abs_median"], 2),
        "syn_p90": fmt_num(syn["abs_p90"], 2),
        "syn_share": fmt_num(syn["share_above_015"], 1),
        "syn_top_name": tex_tt(syn["top"][0]["run"]),
        "syn_top_val": f"{syn['top'][0]['synergy']:+.2f}",
        "syn_bottom_name": tex_tt(syn["bottom"][0]["run"]),
        "syn_bottom_val": f"{syn['bottom'][0]['synergy']:+.2f}",
        "dup_groups": fmt_int(atlas["duplicates"]["n_groups"]),
        "dup_pairs": fmt_int(atlas["duplicates"]["n_pairs"]),
        "top5_b_name": tex_tt(atlas["top5_by_b"][0]["mixer"]),
        "top5_b_val": fmt_num(atlas["top5_by_b"][0]["top5"] * 100, 1),
        "top5_b_conformer": fmt_num(
            next(row["top5"] for row in atlas["top5_by_b"] if row["mixer"] == "conformer") * 100, 1
        ),
        "pareto_n": str(len(atlas["pareto"])),
        "pareto_fast_name": tex_tt(atlas["pareto"][0]["run"]),
        "pareto_fast_loss": fmt_num(atlas["pareto"][0]["loss"], 2),
        "pareto_fast_hours": fmt_num(atlas["pareto"][0]["duration_h"], 2),
        "pareto_mid_name": tex_tt(atlas["pareto"][7]["run"]),
        "pareto_mid_loss": fmt_num(atlas["pareto"][7]["loss"], 2),
        "pareto_mid_hours": fmt_num(atlas["pareto"][7]["duration_h"], 2),
        # -------------------------------------------------------- optimizers
        "opt_n": str(opt["n"]),
        "opt_cluster_names": opt_cluster_names,
        "opt_best": tex_tt(opt_best["name"]),
        "opt_best_loss": fmt_num(opt_best["loss"], 3),
        "opt_best_grad": fmt_num(opt_best["grad_final"], 2),
        "opt_best_epochs": fmt_num(opt_best["epochs"], 0),
        "opt_second": tex_tt(opt_second["name"]),
        "opt_second_loss": fmt_num(opt_second["loss"], 3),
        "opt_second_grad": fmt_num(opt_second["grad_final"], 2),
        "opt_third": tex_tt(opt_third["name"]),
        "opt_third_loss": fmt_num(opt_third["loss"], 3),
        "opt_worst": tex_tt(opt_worst["name"]),
        "opt_worst_loss": fmt_num(opt_worst["loss"], 3),
        "opt_sgd": tex_tt("sgd_momentum"),
        "opt_sgd_loss": fmt_num(opt_row("sgd_momentum")["loss"], 3),
        "opt_sgd_grad": fmt_num(opt_row("sgd_momentum")["grad_final"], 2),
        "rho_opt_grad_final": f"{opt['rho_loss_grad_final']:+.2f}",
        "rho_opt_grad_mean": f"{opt['rho_loss_grad']:+.2f}",
        "rho_opt_clip": f"{opt['rho_loss_clip']:+.2f}",
        "rho_opt_step": f"{opt['rho_loss_step']:+.2f}",
        "rho_opt_epochs": f"{opt['rho_loss_epochs']:+.2f}",
        "opt_step_lo_name": tex_tt(step_lo["name"]),
        "opt_step_lo": fmt_num(step_lo["step_ms"], 0),
        "opt_step_hi_name": tex_tt(step_hi["name"]),
        "opt_step_hi": fmt_num(step_hi["step_ms"], 0),
        "opt_timer_names": tex_list(["af_muon", "ada_muon", "ea4llm"]),
        "opt_timer_lo": fmt_num(min(row["step_ms"] for row in timer_rows), 0),
        "opt_timer_hi": fmt_num(max(row["step_ms"] for row in timer_rows), 0),
        "opt_rss_median": fmt_int(opt["rss_median"]),
        "opt_rss_max_name": tex_tt(opt["rss_max"][0]),
        "opt_rss_max": fmt_int(opt["rss_max"][1]),
        # --------------------------------------------------------- ablations
        "norm_best_name": tex_tt("prms_norm"),
        "norm_best": fmt_num(abl_loss("norm", "norm_prms_norm"), 3),
        "norm_layer": fmt_num(abl_loss("norm", "norm_layer_norm"), 3),
        "norm_flash": fmt_num(abl_loss("norm", "norm_flash_norm"), 3),
        "norm_worst_name": tex_tt("dynamic_tanh"),
        "norm_worst": fmt_num(abl_loss("norm", "norm_dynamic_tanh"), 3),
        "pe_best_name": tex_tt("pape_ri"),
        "pe_best": fmt_num(abl_loss("pe", "pe_pape_ri"), 3),
        "pe_pape": fmt_num(abl_loss("pe", "pe_pape"), 3),
        "pe_rope": fmt_num(abl_loss("pe", "pe_rope"), 3),
        "pe_none": fmt_num(abl_loss("pe", "pe_none"), 3),
        "pe_alibi": fmt_num(abl_loss("pe", "pe_alibi"), 3),
        "sw_factorized": fmt_num(abl_loss("switch", "trans_factorized_emb"), 3),
        "sw_dynamic": fmt_num(abl_loss("switch", "trans_dynamic_emb"), 3),
        "sw_ssmax": fmt_num(abl_loss("switch", "trans_ssmax"), 3),
        "sw_base": fmt_num(abl_loss("switch", "trans_fixed_softmax"), 3),
        "sw_swiglu": fmt_num(abl_loss("switch", "trans_ffn_swiglu"), 3),
        "sw_mhc": fmt_num(abl_loss("switch", "trans_mhc"), 3),
        "sw_bitnet": fmt_num(abl_loss("switch", "trans_bitnet_on"), 3),
        "sw_resid_none": fmt_num(abl_loss("switch", "trans_residuals_none"), 3),
        "sw_loops": fmt_num(abl_loss("switch", "trans_num_loops_2"), 3),
        "batch_two": fmt_num(abl_loss("batch", "batch_size_2"), 3),
        "batch_one": fmt_num(abl_loss("batch", "batch_size_1"), 3),
        "moe_latent": fmt_num(abl_loss("moe_full", "moe_latent") if abl["moe_full"] else float("nan"), 3),
        "moe_interleaved": fmt_num(abl_loss("moe_full", "moe_interleaved") if abl["moe_full"] else float("nan"), 3),
        "task_lm_conformer": fmt_num(abl_loss("task", "task_causal_lm_conformer"), 3),
        "task_lm_decoder": fmt_num(abl_loss("task", "task_causal_lm_decoder"), 3),
        "task_mlm": fmt_num(abl_loss("task", "task_mlm_encoder"), 3),
        "tok_lo": fmt_num(min(row["loss"] for row in abl["tokenizer"]), 2),
        "tok_hi": fmt_num(max(row["loss"] for row in abl["tokenizer"]), 2),
        "vit_pe_runs": str(abl["vit_pe_n"]),
        "vit_pe_spread": fmt_num(max(abl["vit_pe_spread"].values()), 3),
        "vit_class": fmt_num(abl_loss("vision", "vit_classification"), 3),
        "vit_patch": fmt_num(abl_loss("vision", "vit_patch_prediction"), 3),
        "vit_seg": fmt_num(abl_loss("vision", "vit_segmentation_pixel"), 3),
        "vit_class_ssog": fmt_num(abl_loss("vision", "vit_classification_ssog"), 3),
        "vit_conformer": fmt_num(abl_loss("vision", "vit_conformer"), 3),
        "audio_ctc": fmt_num(abl_loss("audio", "audio_asr_ctc"), 3),
        "audio_cls": fmt_num(abl_loss("audio", "audio_classification"), 3),
        "rd_borges_lm": fmt_num(abl_loss("realdata", "realdata_causal_lm_borges"), 2),
        "rd_borges_mlm": fmt_num(abl_loss("realdata", "realdata_mlm_borges"), 2),
    }
    return v


# ---------------------------------------------------------------------------
# English
# ---------------------------------------------------------------------------
EN = r"""
\section{Round 3: The Complete Mixer Atlas}
\label{sec:annex-experiment-3}

This annex reports Round~3 in full. Round~2 measured 868 configurations and left 194 of the 861 hybrid cells empty; Round~3 closes the matrix. The campaign trained every unordered pair of the $\nMixers$ registered mixers---$\binom{\nMixers}{2}=\nPairs$ cells, none missing---plus $\nSingles$ single-mixer baselines and $\nAblations$ ablation runs that move one switch at a time (optimizer family, normalization, positional encoding, Mixture-of-Experts routing, residual path, task and model class). It is the largest measurement in the project: $\nRuns$ training runs, $\nSteps$ million logged optimisation steps, and $\nComputeH$ hours of training wall time spread over $\nCampaignDays$ calendar days.

The annex follows the order of the analysis: the population and its caveats, the parameter landscape, the training outcomes, the mixer atlas with its roles, synergies and Pareto front, the learning dynamics, the gradient pressure per parameter block, the optimizer and component ablations, the cost of training, the inference and quantization measurements, the reliability of the pipeline, and the correlations between the measured features. Every number is regenerated by \texttt{make\_report.py} from \texttt{master.csv}; every figure is regenerated by \texttt{make\_paper\_figures.py} from the same builders as the rest of the report, so this text and the artefacts cannot disagree.

\subsection{Setup, population and exclusions}

The harness produced $\nRecords$ stage records: $\nRuns$ training attempts, $\nExportTotal$ Transformers exports, $\nInferenceTotal$ inference evaluations, $\nDeployTotal$ deploy records and one BitNet/GGUF conversion. Training succeeded $\nTrainingOk$ times and failed $\nTrainingFailed$ times; the failures are ten \texttt{engram} pairs that exit with a non-zero code and are discussed in Section~\ref{sec:r3-reliability}. The population is summarised in Table~\ref{tab:r3-families}.

\tblFamily

Two exclusions shape the rest of the annex. First, the \emph{deployment} stage is not analysed: during the campaign its configuration-payload handling was affected by a bug that has since been patched in the harness, so this round's deploy records are set aside and the analysis of quantized weights uses the Transformers-export path instead, which exported $\nExportTotal$ models with no failures. Second, only $\nCombosInf$ of the training runs produced Hugging Face inference rows ($\nRowsInf$ sample rows over $\nMlmCombos$ MLM pairs plus a handful of causal-LM and text-classification runs); the inference section is a study of that subset, not of the whole atlas.

The runs are heterogeneous in a way that matters for the curves. The configurations ask for 500 epochs, but the harness stops a run earlier (by schedule, plateau criterion or an ablation-specific cap): $\nRanFull$ runs reached the last epoch and $\nEarlyStop$ stopped earlier. The learning curves are therefore stretched over each run's own progress instead of being plotted against absolute steps, so a 300-epoch run and a 500-epoch run can be compared at the same fraction of their schedule. The throughput and cost analysis keeps absolute times, because that is what the campaign paid. The $\ShortRuns$ runs with fewer than 5{,}000 logged steps are dominated by the audio, tokenizer and real-data families, whose datasets are small enough to finish in seconds. A stronger distinction matters for every comparison that follows: $\nSmoke$ runs stopped before epoch~100---fifteen of the seventeen MoE ablations (five-epoch routing smoke tests), the single BitNet/GGUF conversion, five switch and PE smokes, one audio transducer run that produced no valid epoch, and twenty-two hybrid pairs (twelve truncated, ten the \texttt{engram} crashes discussed in Section~\ref{sec:r3-reliability}). They stay in the population and in the compute ledger, but every loss, dynamics, gradient, cost-per-run and correlation result is computed on the $\nAnalysis$ runs with at least 100 epochs. The budget column of Table~\ref{tab:r3-families} makes the distinction visible. One data-quality note completes the setup: the trainer writes a zero-loss sentinel row when the final batch cannot be formed, and those rows are dropped before aggregation, so every ``final loss'' quoted in this annex is the last \emph{valid} epoch of the run.

\figParamsFamily
\figParamsComposition

\subsection{The parameter landscape}

The $\nRuns$ runs span $\nParamsMin$ to $\nParamsMax$ parameters (median $\nParamsMedian$). The median run sits at $\nParamsMedian$ parameters and the median hybrid pair at $\nPairParamsMedian$, with a composition that is almost independent of the family: the token embedding and the output head are fixed costs of roughly $\ShareEmb\%$ and $\ShareHead\%$ of the budget, attention takes $\ShareAttn\%$, the FFN $\ShareFfn\%$ and the normalization layers $\ShareNorm\%$. The architectural choice therefore moves a quarter of the parameters; the other three quarters are the price of having a vocabulary and a classifier. Vision transformers are the exception: patch embeddings replace the token embedding and the classification head is small, so their budget is dominated by attention and FFN.

Parameter cost has a clear per-mixer reading, and it is not the reading one would guess from the literature. The cheapest second mixers are \MixerCheap; the most expensive are \MixerExpensive. \texttt{engram} pays for its key--value memory bank, and \texttt{gmla} pays for the latent projections of the gated multi-head latent attention. Cheapness does not predict quality: \texttt{ccgqa} and \texttt{gqa} are among the cheapest and end slightly below the pair median, while \texttt{mamba}, \texttt{cca} and \texttt{mlra} are equally cheap and end above it; \texttt{engram} buys 0.28M extra parameters for a loss indistinguishable from the median.

The scatter of final loss against parameter count (Figure~\ref{fig:r3-params-loss}) has a rank correlation of $\RhoParamsLoss$ over the $\nPairs$ pairs: larger models are somewhat better, but size explains only a sixth of the ranking, and the atlas spans more than a factor of ten in size against two nats of loss. The budget helps; it does not decide. This is the same conclusion as Round~2, now measured on the complete grid rather than on two thirds of it.

\figParamsVsLoss

\subsection{Training outcomes}

The hybrid pairs end at a median final loss of $\LossMedian$ nats (from $\LossMin$ to $\LossMax$), the best being $\BestPair$ at $\BestPairLoss$ and the worst $\WorstPair$ at $\WorstPairLoss$. Table~\ref{tab:r3-top-loss} lists the ten best pairs. The spread is wide: the difference between the best and the worst pair is larger than the difference between the best and the median optimizer, normalization or positional-encoding ablation. The mixer stack is the dominant design decision of the round, exactly as in Round~2.

\tblTopLoss
\figLossFamily

Loss is not comparable across families because the task changes: the MLM families sit at 8.8--9.3 nats, the vision families at about 1.0, the audio and real-data families at about 1.0 (cross-entropy over a different label space), and the tokenizer runs at 6.0. Figure~\ref{fig:r3-loss-family} keeps them on a log axis for that reason. The MoE ablations need the budget caveat: the fifteen five-epoch routing variants never leave the first epoch and are excluded, so only the two full-schedule variants---\texttt{moe\_interleaved} and \texttt{moe\_latent}---are comparable with the pair population, where they land at the pair median. The routing study is therefore a smoke test in this round and not evidence about expert capacity.

Held-out quality is measured by the inference rows. Across the $\nMlmCombos$ MLM pairs the mean top-1 accuracy is $\TopOneMean\%$ and the mean top-5 hit rate is $\TopFiveMean\%$; the medians are 0 and 28.6\%, so the distribution is a mixture of a few pairs that learned something and a majority that did not. Table~\ref{tab:r3-top-inference} lists the best pairs. The rank correlation between final training loss and held-out top-5 is only $\RhoLossTopFive$, so a lower training loss does not imply a better model at this scale---the same warning as in Round~2, now on 653 pairs instead of 667.

\tblTopInference

A second mixer is worth a fifth of a nat. The $\nSingles$ single-mixer baselines---each layer running the same mixer twice---end at a median of $\SingleMedian$ nats against $\LossMedian$ for the pairs, and $\PairsBetterSingle$ of the $\nMeasured$ analysed pairs end below the best single, the doubled \texttt{conformer} at $\BestSingleLoss$ nats. Doubling a good mixer is a reasonable stack; mixing two different ones is, at the median, the better deal, and the top of the pair table is well beyond any homogeneous stack.

\subsection{The mixer atlas: roles, synergies and the Pareto front}

Figure~\ref{fig:r3-matrix} is the centrepiece of the round: the complete $\nMixers\times\nMixers$ matrix in which each cell is one hybrid pair, rows index the first mixer (A), columns the second (B) and colour encodes the final loss. The ten \texttt{engram} pairs that crash are the only holes---$\nHoles$ empty cells of $\nCells$. The matrix is read in two directions. Column by column---fix the second mixer, vary the first---the second mixer dominates: a good B (\texttt{conformer}, \texttt{falcon3a}, \texttt{falcon2a}) keeps almost every A in a narrow band, while a bad B (\texttt{mamba}, \texttt{cca}, \texttt{iha}) drags even the best A to the bottom of the range. Row by row---no A rescues a bad B. Round~2 measured this asymmetry on two thirds of the grid; Round~3 measures it on all of it.

\figMatrix

Across the $\nCommonRoles$ mixers that appear in both roles with a full schedule, the median loss spans $\RangeB$ nats in the B role (best \BestBName at $\BestBMed$, worst \WorstBName at $\WorstBMed$) against $\RangeA$ nats in the A role (best \BestAName at $\BestAMed$, worst \WorstAName at $\WorstAMed$): the second mixer is the lever, exactly as in Round~2. The two readings are not independent---the role medians correlate at $\RhoRoles$ (Figure~\ref{fig:r3-roles})---so a mixer that is good in one role tends to be good in the other, but the B axis is the wider one and carries the decision. \OnlyA appears only in the A role and \OnlyB only in the B role after the smoke filter; both are excluded from the comparison.

\figRoles
\tblBestB

The best-loss second mixer is not the best-quality one. Table~\ref{tab:r3-best-b} pairs the loss medians with the held-out top-5 of the same pairs, and the two orderings disagree: \texttt{conformer} is the best B by loss but almost the worst by held-out top-5 ($\TopBConformer\%$), while \TopBName tops the quality ranking at $\TopBVal\%$ with a mid-table loss. Figure~\ref{fig:r3-top5-b} shows the full ordering---the role-level form of the loss/quality warning of the previous section.

\figTopB

\paragraph{Interaction beats marginals.}
Ranking mixers individually cannot reconstruct the atlas, because pairs interact. Following Round~2, the synergy of a pair measures how far it is from the average of its two marginals,
\[
S_{A,B} = \frac{\mu_A + \mu_B}{2} - L_{A,B},
\]
where $\mu_A$ and $\mu_B$ are the mean losses of all pairs containing A and B respectively. The median absolute synergy is $\SynMedian$ nats and the P90 is $\SynPninety$: for one pair in ten the interaction moves the result by more than a quarter of a nat, which is more than most optimizer, norm or PE ablations move in total. $\SynShare\%$ of the pairs deviate by more than 0.15 nats from what their parts predict. The positive tail (Table~\ref{tab:r3-synergy}) is led by \SynTopName at $\SynTopVal$ nats: the pairs that beat their marginals stack a cheap, stable first mixer under a strong second one. The negative tail is dominated by \texttt{mtla} as B---\SynBottomName loses $\SynBottomVal$ nats against what its parts predict---mixers that look average in isolation but clash when stacked. For synthesis purposes the message is the same as in Round~2: a ranked list of mixers is a necessary summary but an insufficient design tool; the atlas, or at least the synergy table, must be consulted before composing a stack.

\tblSynergyTable
\figSynergy

\paragraph{The speed--quality front.}
$\ParetoN$ of the $\nMeasured$ analysed pairs sit on the speed--quality Pareto front (Figure~\ref{fig:r3-pareto}): no other pair is both faster and lower-loss. The front runs from \ParetoFastName---$\ParetoFastLoss$ nats in $\ParetoFastHours$ hours, the cheapest cell of the atlas---to the global best \BestPair at $\BestPairLoss$ nats in 2.75 hours, through \ParetoMidName at $\ParetoMidLoss$ nats in $\ParetoMidHours$ hours. The front prices the trade: the first 1.4 hours buy 1.4 nats, the next 1.4 buy 0.5, and the optimiser of a real deployment picks the point on that curve, not a leaderboard rank. Four of the $\ParetoN$ front members use a gated-delta first mixer and every one of them pairs a cheap A with a strong B---the front is the quantitative form of the same asymmetry the matrix shows.

\figPareto

\paragraph{Duplicates and collapses.}
The atlas measures fewer distinct models than it names. $\DupPairs$ of the $\nMeasured$ analysed pairs share a bit-identical final loss with at least one other pair, in $\DupGroups$ duplicate groups. Two structural facts explain most of them. First, the four window mixers \texttt{standard}, \texttt{bigbird}, \texttt{longformer} and \texttt{titan} are the same model at a 64-token context: their attention windows cover the whole sequence, so the four agree on all 49 shared second mixers in the A role and on all seven in the B role. Second, \texttt{fastconformer} and \texttt{swa} are bit-identical wherever both were paired (46 shared second-mixer partners). The same collapse appears inside the ablations: \texttt{none} and \texttt{nope} reproduce each other exactly, \texttt{pape} and \texttt{pape\_efficient} are the same run, and \texttt{bitnet\_conv} without the conv embedding is indistinguishable from \texttt{bitnet\_on}. Counting equivalence classes, the $\nMixers$ registered mixers reduce to roughly 50 distinct models at this context length. Each of these facts is a caveat carried forward rather than a defect hidden, exactly as in Round~2.

\subsection{Learning dynamics}

Figure~\ref{fig:r3-curves-quantiles} shows the population curve. The median pair starts at $\CurveStart$ nats and ends at $\CurveEnd$; the P10--P90 band at the end is $\BandPten$--$\BandPninety$, narrower than the spread between families. Most of the reduction happens in the first quarter: the median reaches $\CurveQtwentyfive$ at 25\% of the schedule, $\CurveQfifty$ at half and $\CurveQseventyfive$ at three quarters. Round~3 therefore measures how fast a configuration learns, not where it converges: the curves are still descending when the schedule ends.

\figCurvesQuantiles
\figCurvesTopTen

The ordering of the best pairs is established early. By a tenth of the schedule the per-pair loss already ranks the pairs with a Spearman correlation of $\RankTen$ against their final ordering, and by a quarter the correlation is $\RankTwentyfive$; in Figure~\ref{fig:r3-curves-top10} the ten best separate from the population before the halfway mark and rarely cross afterwards. The number of epochs needed to reach 90\% of a run's total loss drop confirms this: the median is $\ConvMedian\%$ of the schedule, and the family medians range from $\ConvVision\%$ for the vision tasks (where the loss collapses almost immediately) to $\ConvNorm\%$ for the normalization ablations (where the differences between norms only appear late). Hybrid pairs sit at $\ConvPair\%$.

\figConvergence

\subsection{Gradient pressure per parameter block}

The trainer logs the L2 norm of the gradient of every parameter block after clipping: embeddings, positional encoding, attention, FFN, normalization, head, mHC, router, residual and other. Table~\ref{tab:r3-gradients} reproduces the medians and Figure~\ref{fig:r3-grad-groups} the trajectories over training.

\tblGradients
\figGradGroups

Three facts stand out. First, the head carries the largest gradient ($\GHead$ on average, $\GHeadFinal$ at the end), followed by attention ($\GAttn$, $\GAttnFinal$) and the FFN ($\GFfn$, $\GFfnFinal$); embeddings and normalization are an order of magnitude smaller ($\GEmb$ and $\GNorm$). The ordering is stable across the whole campaign, which is why the per-block normalization of the optimizer groups is not a cosmetic choice. Second, the coefficients of variation are small---$\CvHead$ for the head, $\CvAttn$ for attention, $\CvFfn$ for the FFN---so the gradient scale of a block is a property of the architecture, not of the step. Third, the blocks that only exist in some configurations (\texttt{mhc}, \texttt{router}, \texttt{residual}) are exactly zero in the runs that do not enable them, which is what makes the zero-median rows of the table meaningful rather than missing data.

\figGradHeatmap

The heatmap (Figure~\ref{fig:r3-grad-heatmap}) reads the same statistics as a function of progress. The head is flat, attention grows slightly, and the FFN decays by a factor of about two over the first half of training before stabilizing; the embeddings decay as the vocabulary statistics are learned. Nothing explodes: there are no NaN steps in the whole campaign, $\InfSteps$ Inf steps (all repaired in place, in $\RepairRuns$ runs), and the gradient-norm coefficient of variation is inversely related to the final loss ($\RhoGradCvLoss$), i.e.\ the runs that end lower have slightly more variable gradients, not less.

\figGradHealth

Clipping, however, is routine rather than exceptional. The median mean clip ratio is $\ClipMedian$---just above the threshold of 1---and $\ClipAbove\%$ of the runs spend the campaign clipping on average, with a P90 of $\ClipPninety$. Round~2 reported a median of 0.90 and clipping as a rare event; the difference is the longer schedule and the higher learning rates of Round~3. The consequence for the report is that gradient norms should be read as post-clip norms, which is what the trainer logs.

The memory footprint is modest and flat. The median analysis run holds $\RssMeanMedian$ MB of resident memory (P90 $\RssMeanPninety$ MB); the vision families are the exception at about $\RssVision$ MB, and the largest run-to-run mean is $\RssMeanMax$ MB (\RssMaxRun). Peak-resident telemetry exists for $\RssPeakRuns$ runs of the campaign and tops at $\RssPeakMax$ MB. The whole atlas fits on a single CPU host with a dozen workers.

\subsection{Optimizer and component ablations}

\paragraph{Optimizers: loss, gradient pressure, CPU and memory.}
The optimizer sweep runs $\OptN$ optimizers on the same 309k-parameter backbone, the same data and the same learning rate; Table~\ref{tab:r3-optimizers} reports the effective epochs, the final loss, the gradient norm, the clip ratio, the step time and the mean resident memory of each. Three results stand out.

First, the sweep has an identity cluster. Eight optimizers (\OptClusterNames) produce trajectories that agree to within $2\times10^{-6}$ nats---they are the same update rule at these hyperparameters---so the ablation really measures the remaining 31. Round~2 saw the same cluster and read it as interchangeable optimizers; the bit-level identity of the trajectories makes the reading precise.

Second, the loss ranking is a gradient-pressure ranking. Across the $\OptN$ runs, the rank correlation between the final loss and the final gradient norm is $\RhoOptGradFinal$ (against the per-run mean gradient, $\RhoOptGradMean$), and against the clip ratio $\RhoOptClip$. The optimizers that end lower are the ones whose gradients stay large and keep clipping: \OptBest reaches $\OptBestLoss$ nats with the largest final gradient of the sweep ($\OptBestGrad$, after $\OptBestEpochs$ epochs), \OptSecond reaches $\OptSecondLoss$ with $\OptSecondGrad$, while \OptSgd ends at $\OptSgdLoss$ with $\OptSgdGrad$. Learning speed at this scale is bought with gradient magnitude, not with a fancier preconditioner---the optimizer-level form of the block-level finding that the head's gradient dominates.

Third, quality and CPU cost are orthogonal: the correlation between final loss and step time is $\RhoOptStep$, and the effective schedule only adds $\RhoOptEpochs$. Among the 36 runs with comparable telemetry the step time spans $\OptStepLo$--$\OptStepHi$ ms per step (\OptStepLoName to \OptStepHiName), about 40\%; memory is flatter still: the median optimizer holds $\OptRssMedian$ MB of resident memory, and only \OptRssMaxName stands out at $\OptRssMax$ MB, a third above the median, paying for its extra state. The three late additions to the sweep (\OptTimerNames) log step times of $\OptTimerLo$--$\OptTimerHi$ ms against 305--438 for the rest while running longer wall-clock schedules, so their step timer covers only part of the work and is excluded from the spread above; their losses and gradient norms are unaffected.

\tblOptimizers
\figOptimizerPanel

\paragraph{Normalization.}
The ranking flips relative to Round~2 (Table~\ref{tab:r3-norms}). \NormBestName wins at $\NormBest$ nats---half a nat ahead of \texttt{layer\_norm} at $\NormLayer$---while \texttt{flash\_norm}, the clear winner of Round~2 at 100 epochs, closes the table at $\NormFlash$ (284 effective epochs) and \NormWorstName is worst at $\NormWorst$. The schedules differ (284--370 effective epochs), so the ordering carries the usual budget caveat; but the Round~2 winner does not survive the longer schedule, and ``modern'' is not automatically better at this scale either.

\figAblationLoss
\tblNorms

\paragraph{Positional encoding.}
The PAPE family leads, as in Round~2 (Table~\ref{tab:r3-pe}): \PeBestName at $\PeBest$ nats, then \texttt{pape} and \texttt{pape\_efficient} at $\PePape$---bit-identical twins. Classical \texttt{rope} and \texttt{sinusoidal\_rotary} reproduce the baseline exactly ($\PeRope$, 312 epochs), \texttt{alibi} sits at $\PeAlibi$, and the position-free variants \texttt{none}, \texttt{nope} and \texttt{bam} cluster at $\PeNone$, with \texttt{none} and \texttt{nope} identical to the sixth decimal. The six-epoch \texttt{hope} run is a smoke test and is excluded from the table.

\tblPE

\paragraph{Structural switches.}
The three wins over the shared baseline are factorised embeddings ($\SwFactorized$ nats, 499 effective epochs), dynamic embeddings ($\SwDynamic$, 499) and the fixed-softmax/ssmax swap (\texttt{ssmax} at $\SwSsmax$). The \texttt{fixed\_softmax}, \texttt{fixed\_ce}, \texttt{fixed\_both} toggles and the conv embedding reproduce the baseline exactly ($\SwBase$), which identifies them as no-ops at this configuration. The residual variants all cost loss (\texttt{full\_attn} $+0.14$, \texttt{block\_attn} $+0.16$, \texttt{none} $+0.22$ over the standard path at $\SwBase$), SwiGLU costs $+0.19$ at a shorter schedule (281 epochs), two Hyperloop loops cost $+0.16$, mHC $+0.13$ at 443 epochs, and the BitNet pair (\texttt{bitnet\_on}/\texttt{bitnet\_conv}, identical) lands at $\SwBitnet$ after the full 499 epochs---the longest schedule of the family, consistent with BitNet's known slow start. The switches without a full schedule (\texttt{moe}, \texttt{mod}, \texttt{pos\_hope}, \texttt{bitnet\_routers}) are smoke runs and are excluded from the table.

\tblSwitches

\paragraph{Batch size, tasks, tokenizers and MoE.}
Doubling the batch size reaches $\BatchTwo$ nats (499 effective epochs) against $\BatchOne$ for the batch-1 reference (312); the comparison carries a schedule confound, and the MoE routing smoke tests (15 of 17 variants, five epochs) say nothing about expert capacity---only \texttt{moe\_latent} ($\MoeLatent$) and \texttt{moe\_interleaved} ($\MoeInterleaved$) ran the full schedule, both at the pair median. The task switches anchor the loss scales: causal LM runs at $\TaskLmConformer$--$\TaskLmDecoder$ nats (the conformer decoder at $\TaskLmConformer$ is the strongest of the four), MLM at $\TaskMlm$, and the tokenizer family at $\TokLo$--$\TokHi$. None of these numbers is comparable with the MLM pair population; they are scale anchors, as in Round~2.

\paragraph{Vision, audio and real data.}
The vision families behave as in Round~2, with one addition: the ViT positional-encoding ablation is a clean null result. All $\VitPeRuns$ \texttt{vit\_pe} runs---13 encodings $\times$ 3 tasks---reproduce, within each task, one and the same loss to the sixth decimal (spread $\VitPeSpread$: classification $\VitClass$, patch prediction $\VitPatch$, segmentation $\VitSeg$): at a $4\times4$ patch grid the positional encoding is invisible. The task ordering matches Round~2 (segmentation easiest, patch prediction hardest); the SSOG variant costs 0.18 nats on classification ($\VitClassSsog$ against $\VitClass$) and the conformer ViT lands at $\VitConformer$. The audio family spans CTC ASR at $\AudioCtc$ to audio classification at $\AudioCls$, and the real-data runs bracket the synthetic families: Borges causal LM lands at $\RdBorgesLm$ and Borges MLM at $\RdBorgesMlm$, indistinguishable from the synthetic MLM population---the sanity check that the synthetic atlas was not measuring an artefact.

\subsection{Throughput and cost}

Training time grows only weakly with parameters: a log-log fit over the MLM pairs gives an exponent of $\TimeSlope$ with $R^2=\TimeRtwo$. The reason is that the parameter count is not what the clock measures---the number of steps dominates, and the step count varies by family and by early stopping. The normalised cost (Figure~\ref{fig:r3-cost-mparam}) is the quantity that compares architectures fairly: the median run spends $\MinPerMparam$ minutes per million parameters, and the family medians span a factor of about three around that value. A hybrid pair costs $\MinPerMparamPair$ minutes per Mparam, slightly below a single-mixer baseline ($\MinPerMparamSingle$) and above an optimizer ablation ($\MinPerMparamOpt$); the differences come from the effective schedule and from the update cost, not from the parameter count.

\figTimeParams
\figCostMparam
\figThroughput
\figComputeFamily

Figure~\ref{fig:r3-compute} shows where the $\nComputeH$ hours went. The hybrid pairs consume $\PairShare\%$ of the campaign ($\PairHours$ hours), the single-mixer baselines $\SecondShare\%$ and every other family less than 3\%. The vision and audio families are cheap in wall time despite their long schedules because their steps are short; the throughput panel (Figure~\ref{fig:r3-throughput}) makes the same point in tokens per second: the MLM pairs run at a median of $\TokensMedian$ tokens/s, while the vision family is an order of magnitude slower ($\TokensVision$ tokens/s) because its inputs are images rather than 64-token sequences.

The campaign spent $\nComputeH$ hours of training wall time spread over $\nCampaignDays$ calendar days with 13--26 workers, which is the parallel efficiency of the harness rather than a property of the models.

\subsection{Inference and quantization}

The inference section studies the $\nCombosInf$ runs that produced Hugging Face rows: $\nMlmCombos$ MLM pairs, four causal-LM runs and one text-classification run. Each run was exported unquantized (\texttt{hf\_none}) and re-exported with three codecs at two loader runtimes (\texttt{q8\_0}, \texttt{q4\_0}, \texttt{i2\_s}, each in \texttt{v1} and \texttt{v2}). The two runtimes sampled different mask positions---only 3.5\% of the \texttt{v1} sample keys coincide with the baseline---so \texttt{v1} cannot be paired with \texttt{hf\_none}; the clean comparison is the \texttt{v2} family, which reuses the baseline masks for 99.4\% of the keys. This is a property of the harness, not of the models, and it is the main caveat of the section.

\figInferenceFormat

The aggregate view (Figure~\ref{fig:r3-inference-format} and Table~\ref{tab:r3-formats}) says that 8-bit and 4-bit weights are essentially free at this scale. Relative to the unquantized export, \texttt{q8\_0\_v2} changes the top-1 accuracy by $\QEightDelta$ percentage points and leaves $\QEightEqual\%$ of the pairs unchanged, with only $\QEightWorse\%$ degrading; \texttt{q4\_0\_v2} changes it by $\QFourDelta$ points with $\QFourEqual\%$ unchanged and $\QFourWorse\%$ degrading. The 2-bit codec is the exception: \texttt{i2\_s\_v2} changes the top-1 accuracy by $\ITwoDelta$ points on average, leaves $\ITwoEqual\%$ of the pairs intact and degrades $\ITwoWorse\%$ of them. The \texttt{v1} rows are shown in the table for completeness but are not comparable with the baseline, because their masks are harder: the same pairs score $\TopFiveQEightVone\%$ top-5 under \texttt{v1} against $\TopFiveNone\%$ under \texttt{hf\_none}.

\tblFormats

The sample-level view is sharper than the aggregate. Matching each quantized prediction to the baseline prediction of the same mask, \texttt{q8\_0\_v2} agrees on $\AgreeQEight\%$ of the samples (95\% CI from the per-combo bootstrap), \texttt{q4\_0\_v2} on $\AgreeQFour\%$ and \texttt{i2\_s\_v2} on only $\AgreeITwo\%$. The 2-bit model is not a slightly worse copy of the teacher: it is a different model that happens to preserve the aggregate top-1 on the pairs whose baseline accuracy was already near zero.

\figQuantDelta
\figQuantAgreement

Two negative results complete the section. Parameter count does not predict robustness to 2-bit quantization (Spearman $\RhoITwoParams$ between the \texttt{i2\_s} delta and the parameter count), and training loss does not predict held-out quality (Spearman $\RhoLossTopFive$). At this scale the only reliable signal is the comparison itself: measure the quantized export against the unquantized one, on the same samples, and report the agreement rather than the aggregate.

\figQuantParams
\figTrainInference

\subsection{Reliability and failure taxonomy}
\label{sec:r3-reliability}

Training is reliable: $\TrainingOk$ of $\nRuns$ runs succeed ($\FailureRate\%$ failure rate). All ten failures are hybrid pairs that contain \texttt{engram}; eight exit with code 1 and two with code 134 (SIGABRT), and no other family fails. The failure is structural rather than random---\texttt{engram} is the only mixer with an external memory bank, and ten of the fifty-three pairs that use it crash while the other forty-three train---which is why the atlas still contains every cell: the ten failed pairs are present in the population table but absent from the loss and inference analyses.

The rest of the pipeline is healthier than in Round~2. The Transformers export never fails ($\nExportTotal$ records, all OK) and inference evaluation fails once (a \texttt{pape\_efficient} run). The stage ledger (Figure~\ref{fig:r3-stage-status}) makes the picture visible: training and export are almost entirely green, and inference is green with one red row.

\figStageStatus

\subsection{Correlations}

Figure~\ref{fig:r3-correlation} is the Spearman matrix of the measured features over the $\nPairs$ pairs. The strongest non-trivial correlations are the ones that describe the schedule rather than the architecture: training time grows with the number of epochs ($\RhoDurationEpochs$), throughput is almost the inverse of step time ($\RhoTokensStep$) and mean loss falls as the schedule lengthens ($\RhoLossEpochs$). The parameter count correlates moderately with loss ($\RhoParamsLoss$) and gradient magnitude ($\RhoParamsGrad$) and weakly with wall time ($\RhoDurationParams$), the quantitative form of the weak cost law of the previous section.

One contrast with Round~2 is worth making explicit. There, the all-rows correlation matrix was dominated by the family contrast and read $-0.43$ against $-0.20$ within the pairs; here, recomputing the parameter--loss correlation over all $\nAllRows$ analysis runs instead of the pairs only leaves it at $\RhoParamsLossAll$, because the ablation families share the backbone and the population is homogeneous. Round~3 has no confound to defuse: the within-pairs number is the population number.

\figCorrelation

\subsection{Summary of findings}

\begin{enumerate}[leftmargin=1.4em,itemsep=2pt]
\item The complete atlas is measurable in one week of a single CPU host: $\nRuns$ runs, $\nSteps$ million steps, $\nComputeH$ hours, a $\FailureRate\%$ training failure rate and no NaN step.
\item The mixer stack dominates the design space. The best pair reaches $\BestPairLoss$ nats and the worst $\WorstPairLoss$, while the optimizer, normalization and positional-encoding ablations move the loss by less than the pair spread.
\item The second mixer is the lever and the interaction is real: the B-role medians span $\RangeB$ nats against $\RangeA$ for A, the P90 absolute synergy is $\SynPninety$ nats, and mixing two different mixers beats doubling one by 0.22 nats at the median.
\item Size helps but does not decide: the rank correlation between parameters and loss is $\RhoParamsLoss$, and the cheapest mixers sit on both sides of the median.
\item Learning is fast and the ranking is decided early: the median pair reaches 90\% of its loss drop at $\ConvMedian\%$ of the schedule, the quarter-schedule loss already ranks the pairs at $\RankTwentyfive$, and the top ten separate from the population in the first 15\%.
\item Gradient pressure is stable and block-specific: the head ($\GHead$), attention ($\GAttn$) and FFN ($\GFfn$) dominate, the coefficients of variation of the active blocks stay below 0.35, and the block ordering is constant across the campaign.
\item The optimizer ablation turns the loss ranking into a gradient-pressure ranking ($\RhoOptGradFinal$ between final loss and final gradient norm across $\OptN$ optimizers), an eight-optimizer cluster is bit-identical, and quality is orthogonal to CPU cost ($\RhoOptStep$) and memory (median $\OptRssMedian$ MB).
\item The component ablations separate: \NormBestName leads the norms at $\NormBest$ (Round~2's \texttt{flash\_norm} win does not survive the longer schedule), the PAPE family leads the positional encodings, factorised embeddings win the switches, and the ViT positional-encoding sweep is a null result ($\VitPeRuns$ runs, zero spread within each task).
\item Clipping is routine, not exceptional: the median clip ratio is $\ClipMedian$ and $\ClipAbove\%$ of the runs clip on average, which changes the reading of the gradient norms to post-clip quantities.
\item Training time is almost independent of the parameter count at a fixed schedule (log-log exponent $\TimeSlope$, $R^2=\TimeRtwo$): the number of steps dominates, and the hybrid pairs absorb $\PairShare\%$ of the $\nComputeH$ hours.
\item Quantization at 8 and 4 bits is essentially lossless in the \texttt{v2} runtime ($\AgreeQEight\%$ and $\AgreeQFour\%$ sample agreement), while 2-bit weights change the model qualitatively ($\AgreeITwo\%$ agreement, $\ITwoDelta$ points of top-1).
\item The pipeline is reliable end to end: training fails $\FailureRate\%$ of the time, all of it in one mixer's memory bank, the Transformers export never fails and evaluation fails once.
\item The round inherits the central warning of Round~2: training loss and held-out quality are almost uncorrelated ($\RhoLossTopFive$), so the atlas should be read as a map of learning speed and stability, not as a leaderboard.
\end{enumerate}

\subsection{Reproducing this annex}

The annex is generated by four scripts under \texttt{experimentos/3/report/}, in order:

\begin{verbatim}
python experimentos/to_csv.py 3                       # flat CSV export
python experimentos/3/report/build_data.py            # master/epochs/inference
python experimentos/3/report/make_paper_figures.py    # PDF/SVG, both languages
python experimentos/3/report/make_report.py           # stats + this annex
\end{verbatim}

\texttt{build\_data.py} streams the 6.6\,GB per-step log once (about ten minutes) and caches every aggregate; the other three scripts run in seconds. \texttt{make\_report.py} writes \texttt{out/stats.json} with every number quoted above, so any claim can be traced back to the CSV that produced it.
"""


# ---------------------------------------------------------------------------
# Spanish
# ---------------------------------------------------------------------------
ES = r"""
\section{Ronda 3: el atlas completo de mixers}
\label{sec:annex-experiment-3}

Este apéndice reporta la ronda~3 en su totalidad. La ronda~2 midió 868 configuraciones y dejó vacías 194 de las 861 celdas híbridas; la ronda~3 cierra la matriz. La campaña entrenó todos los pares no ordenados de los $\nMixers$ mixers registrados---$\binom{\nMixers}{2}=\nPairs$ celdas, sin ninguna ausente---más $\nSingles$ líneas base de un solo mixer y $\nAblations$ corridas de ablación que mueven un interruptor a la vez (familia de optimizador, normalización, codificación posicional, enrutamiento MoE, camino residual, tarea y clase de modelo). Es la medición más grande del proyecto: $\nRuns$ corridas de entrenamiento, $\nSteps$ millones de pasos de optimización registrados y $\nComputeH$ horas de tiempo de pared de entrenamiento repartidas en $\nCampaignDays$ días calendario.

El apéndice sigue el orden del análisis: la población y sus advertencias, el paisaje de parámetros, los resultados de entrenamiento, el atlas de mixers con sus roles, sinergias y frontera de Pareto, la dinámica de aprendizaje, la presión del gradiente por bloque, las ablaciones de optimizador y componentes, el coste del entrenamiento, las mediciones de inferencia y cuantización, la fiabilidad del pipeline y las correlaciones entre las variables medidas. Todos los números los regenera \texttt{make\_report.py} a partir de \texttt{master.csv}; todas las figuras las regenera \texttt{make\_paper\_figures.py} con los mismos constructores que el resto del informe, así que este texto y los artefactos no pueden contradecirse.

\subsection{Configuración, población y exclusiones}

El harness produjo $\nRecords$ registros de etapa: $\nRuns$ intentos de entrenamiento, $\nExportTotal$ exportaciones a Transformers, $\nInferenceTotal$ evaluaciones de inferencia, $\nDeployTotal$ registros de despliegue y una conversión BitNet/GGUF. El entrenamiento tuvo éxito $\nTrainingOk$ veces y falló $\nTrainingFailed$ veces; los fallos son diez pares con \texttt{engram} que terminan con un código distinto de cero y se discuten en la Sección~\ref{sec:r3-reliability}. La población se resume en la Tabla~\ref{tab:r3-families}.

\tblFamily

Dos exclusiones condicionan el resto del apéndice. Primero, la etapa de \emph{despliegue} no se analiza: durante la campaña su manejo del payload de configuración se vio afectado por un error que ya está parchado en el harness, así que los registros de despliegue de esta ronda se dejan de lado y el análisis de pesos cuantizados usa la ruta de exportación a Transformers, que exportó $\nExportTotal$ modelos sin fallos. Segundo, solo $\nCombosInf$ de las corridas produjeron filas de inferencia de Hugging Face ($\nRowsInf$ filas de muestra sobre $\nMlmCombos$ pares MLM más unas pocas corridas de LM causal y clasificación de texto); la sección de inferencia estudia ese subconjunto, no el atlas completo.

Las corridas son heterogéneas de una forma que importa para las curvas. Las configuraciones piden 500 épocas, pero el harness detiene una corrida antes (por calendario, criterio de meseta o un tope específico de la ablación): $\nRanFull$ corridas llegaron a la última época y $\nEarlyStop$ se detuvieron antes. Por eso las curvas de aprendizaje se estiran sobre el progreso propio de cada corrida en lugar de dibujarse contra pasos absolutos, de modo que una corrida de 300 épocas y una de 500 se comparan en la misma fracción de su calendario. El análisis de rendimiento y coste conserva los tiempos absolutos, porque eso es lo que pagó la campaña. Las $\ShortRuns$ corridas con menos de 5{,}000 pasos registrados están dominadas por las familias de audio, tokenizadores y datos reales, cuyos conjuntos son lo bastante pequeños como para terminar en segundos. Una distinción más fuerte importa para todas las comparaciones que siguen: $\nSmoke$ corridas se detuvieron antes de la época~100---quince de las diecisiete ablaciones MoE (pruebas de humo de enrutamiento de cinco épocas), la única conversión BitNet/GGUF, cinco pruebas de humo de interruptores y PE, una corrida de transductor de audio sin épocas válidas y veintidós pares híbridos (doce truncados, diez los fallos de \texttt{engram} que se discuten en la Sección~\ref{sec:r3-reliability}). Permanecen en la población y en el registro de cómputo, pero todos los resultados de pérdida, dinámica, gradiente, coste por corrida y correlación se calculan sobre las $\nAnalysis$ corridas con al menos 100 épocas. La columna de presupuesto de la Tabla~\ref{tab:r3-families} hace visible la distinción. Una nota de calidad de datos completa la configuración: el entrenador escribe una fila centinela con pérdida cero cuando no puede formar el último lote, y esas filas se descartan antes de agregar, así que cada «pérdida final» citada en este apéndice es la última época \emph{válida} de la corrida.

\figParamsFamily
\figParamsComposition

\subsection{El paisaje de parámetros}

Las $\nRuns$ corridas abarcan de $\nParamsMin$ a $\nParamsMax$ parámetros (mediana $\nParamsMedian$). La corrida mediana se sitúa en $\nParamsMedian$ parámetros y el par híbrido mediano en $\nPairParamsMedian$, con una composición casi independiente de la familia: el embedding de tokens y la cabeza de salida son costes fijos de aproximadamente $\ShareEmb\%$ y $\ShareHead\%$ del presupuesto, la atención toma $\ShareAttn\%$, la FFN $\ShareFfn\%$ y las capas de normalización $\ShareNorm\%$. La decisión arquitectónica mueve, por tanto, un cuarto de los parámetros; los otros tres cuartos son el precio de tener vocabulario y clasificador. Los transformers de visión son la excepción: los embeddings de parches reemplazan al embedding de tokens y la cabeza de clasificación es pequeña, así que su presupuesto lo dominan la atención y la FFN.

El coste por mixer tiene una lectura clara, y no es la que se adivinaría desde la literatura. Los segundos mixers más baratos son \MixerCheap; los más caros son \MixerExpensive. \texttt{engram} paga su banco de memoria clave--valor y \texttt{gmla} paga las proyecciones latentes de la atención latente multimando con compuerta. Barato no predice bueno: \texttt{ccgqa} y \texttt{gqa} están entre los más baratos y terminan algo por debajo de la mediana de los pares, mientras \texttt{mamba}, \texttt{cca} y \texttt{mlra} son igual de baratos y terminan por encima; \texttt{engram} paga 0,28M parámetros extra por una pérdida indistinguible de la mediana.

La dispersión de la pérdida final contra el número de parámetros (Figura~\ref{fig:r3-params-loss}) tiene una correlación de rango de $\RhoParamsLoss$ sobre los $\nPairs$ pares: los modelos más grandes son algo mejores, pero el tamaño explica solo una sexta parte del orden, y el atlas abarca más de un factor diez de tamaño frente a dos nats de pérdida. El presupuesto ayuda; no decide. Es la misma conclusión de la ronda~2, ahora medida sobre la rejilla completa en lugar de sobre dos tercios de ella.

\figParamsVsLoss

\subsection{Resultados de entrenamiento}

Los pares híbridos terminan con una pérdida final mediana de $\LossMedian$ nats (de $\LossMin$ a $\LossMax$), el mejor es $\BestPair$ con $\BestPairLoss$ y el peor $\WorstPair$ con $\WorstPairLoss$. La Tabla~\ref{tab:r3-top-loss} lista los diez mejores pares. La dispersión es amplia: la diferencia entre el mejor y el peor par es mayor que la diferencia entre la mejor ablación de optimizador, normalización o codificación posicional y la mediana de su familia. La pila de mixers es la decisión de diseño dominante de la ronda, exactamente como en la ronda~2.

\tblTopLoss
\figLossFamily

La pérdida no es comparable entre familias porque cambia la tarea: las familias MLM se sitúan en 8,8--9,3 nats, las de visión en torno a 1,0, las de audio y datos reales en torno a 1,0 (entropía cruzada sobre otro espacio de etiquetas) y los tokenizadores en 6,0. La Figura~\ref{fig:r3-loss-family} las mantiene en eje logarítmico por esa razón. Las ablaciones MoE necesitan la advertencia de presupuesto: las quince variantes de enrutamiento de cinco épocas nunca salen de la primera época y quedan excluidas, así que solo las dos variantes de calendario completo---\texttt{moe\_interleaved} y \texttt{moe\_latent}---son comparables con la población de pares, donde aterrizan en la mediana. El estudio de enrutamiento es, por tanto, una prueba de humo en esta ronda y no evidencia sobre la capacidad de los expertos.

La calidad retenida se mide con las filas de inferencia. Sobre los $\nMlmCombos$ pares MLM, la exactitud top-1 media es $\TopOneMean\%$ y el acierto top-5 medio es $\TopFiveMean\%$; las medianas son 0 y 28,6\%, así que la distribución mezcla unos pocos pares que aprendieron algo con una mayoría que no. La Tabla~\ref{tab:r3-top-inference} lista los mejores pares. La correlación de rango entre la pérdida final de entrenamiento y el top-5 retenido es solo $\RhoLossTopFive$: una pérdida menor no implica un modelo mejor a esta escala, la misma advertencia de la ronda~2, ahora sobre 653 pares en lugar de 667.

\tblTopInference

Un segundo mixer vale un quinto de nat. Las $\nSingles$ líneas base de un solo mixer---cada capa con el mismo mixer dos veces---terminan con una mediana de $\SingleMedian$ nats frente a $\LossMedian$ de los pares, y $\PairsBetterSingle$ de los $\nMeasured$ pares analizados terminan por debajo de la mejor línea base, el \texttt{conformer} duplicado con $\BestSingleLoss$ nats. Duplicar un buen mixer es una pila razonable; mezclar dos distintos es, en la mediana, el mejor negocio, y la cima de la tabla de pares está muy por encima de cualquier pila homogénea.

\subsection{El atlas de mixers: roles, sinergias y frontera de Pareto}

La Figura~\ref{fig:r3-matrix} es la pieza central de la ronda: la matriz completa de $\nMixers\times\nMixers$ en la que cada celda es un par híbrido, las filas indexan el primer mixer (A), las columnas el segundo (B) y el color codifica la pérdida final. Los diez pares con \texttt{engram} que fallan son los únicos huecos---$\nHoles$ celdas vacías de $\nCells$. La matriz se lee en dos direcciones. Columna a columna---fijar el segundo mixer, variar el primero---el segundo mixer domina: un buen B (\texttt{conformer}, \texttt{falcon3a}, \texttt{falcon2a}) mantiene casi cualquier A en una banda estrecha, mientras un mal B (\texttt{mamba}, \texttt{cca}, \texttt{iha}) arrastra incluso al mejor A al fondo del rango. Fila a fila---ningún A rescata a un mal B. La ronda~2 midió esta asimetría sobre dos tercios de la rejilla; la ronda~3 la mide completa.

\figMatrix

Sobre los $\nCommonRoles$ mixers que aparecen en ambos roles con calendario completo, la pérdida mediana abarca $\RangeB$ nats en el rol B (mejor \BestBName con $\BestBMed$, peor \WorstBName con $\WorstBMed$) frente a $\RangeA$ nats en el rol A (mejor \BestAName con $\BestAMed$, peor \WorstAName con $\WorstAMed$): el segundo mixer es la palanca, exactamente como en la ronda~2. Las dos lecturas no son independientes---las medianas por rol correlacionan en $\RhoRoles$ (Figura~\ref{fig:r3-roles})---así que un mixer bueno en un rol tiende a serlo en el otro, pero el eje B es el más ancho y carga con la decisión. \OnlyA aparece solo en el rol A y \OnlyB solo en el rol B tras el filtro de humo; ambos quedan fuera de la comparación.

\figRoles
\tblBestB

El mejor segundo mixer en pérdida no es el mejor en calidad. La Tabla~\ref{tab:r3-best-b} empareja las medianas de pérdida con el top-5 retenido de los mismos pares, y los dos órdenes discrepan: \texttt{conformer} es el mejor B en pérdida pero casi el peor en top-5 retenido ($\TopBConformer\%$), mientras \TopBName encabeza el ranking de calidad con $\TopBVal\%$ y una pérdida de mitad de tabla. La Figura~\ref{fig:r3-top5-b} muestra el orden completo---la forma a nivel de rol de la advertencia pérdida/calidad de la sección anterior.

\figTopB

\paragraph{La interacción supera a los marginales.}
Ordenar los mixers individualmente no reconstruye el atlas, porque los pares interactúan. Siguiendo la ronda~2, la sinergia de un par mide cuánto se aleja del promedio de sus dos marginales,
\[
S_{A,B} = \frac{\mu_A + \mu_B}{2} - L_{A,B},
\]
donde $\mu_A$ y $\mu_B$ son las pérdidas medias de todos los pares que contienen A y B. La sinergia absoluta mediana es $\SynMedian$ nats y el P90 es $\SynPninety$: en uno de cada diez pares la interacción mueve el resultado en más de un cuarto de nat, más de lo que mueven en total la mayoría de las ablaciones de optimizador, normalización o PE. El $\SynShare\%$ de los pares se desvía más de 0,15 nats de lo que predicen sus partes. La cola positiva (Tabla~\ref{tab:r3-synergy}) la lidera \SynTopName con $\SynTopVal$ nats: los pares que superan a sus marginales apilan un primer mixer barato y estable bajo un segundo fuerte. La cola negativa la domina \texttt{mtla} como B---\SynBottomName pierde $\SynBottomVal$ nats frente a lo que predicen sus partes---mixers que parecen medianos aislados pero chocan al apilarlos. Para síntesis el mensaje es el mismo de la ronda~2: una lista ordenada de mixers es un resumen necesario pero una herramienta de diseño insuficiente; hay que consultar el atlas, o al menos la tabla de sinergias, antes de componer una pila.

\tblSynergyTable
\figSynergy

\paragraph{La frontera velocidad--calidad.}
$\ParetoN$ de los $\nMeasured$ pares analizados están sobre la frontera de Pareto velocidad--calidad (Figura~\ref{fig:r3-pareto}): ningún otro par es a la vez más rápido y con menor pérdida. La frontera va de \ParetoFastName---$\ParetoFastLoss$ nats en $\ParetoFastHours$ horas, la celda más barata del atlas---al mejor global \BestPair con $\BestPairLoss$ nats en 2,75 horas, pasando por \ParetoMidName con $\ParetoMidLoss$ nats en $\ParetoMidHours$ horas. La frontera pone precio al intercambio: la primera hora y media compra 1,4 nats, la siguiente hora y media compra 0,5, y quien optimiza un despliegue real elige un punto de esa curva, no un puesto de tabla de líderes. Cuatro de los $\ParetoN$ miembros de la frontera usan un primer mixer gated-delta y todos emparejan un A barato con un B fuerte---la forma cuantitativa de la misma asimetría que muestra la matriz.

\figPareto

\paragraph{Duplicados y colapsos.}
El atlas mide menos modelos distintos de los que nombra. $\DupPairs$ de los $\nMeasured$ pares analizados comparten una pérdida final idéntica bit a bit con al menos otro par, en $\DupGroups$ grupos duplicados. Dos hechos estructurales explican la mayoría. Primero, los cuatro mixers de ventana \texttt{standard}, \texttt{bigbird}, \texttt{longformer} y \texttt{titan} son el mismo modelo en un contexto de 64 tokens: sus ventanas de atención cubren la secuencia completa, así que los cuatro coinciden en los 49 segundos mixers compartidos en el rol A y en los siete en el rol B. Segundo, \texttt{fastconformer} y \texttt{swa} son idénticos bit a bit allí donde ambos fueron emparejados (46 segundos mixers compartidos). El mismo colapso aparece dentro de las ablaciones: \texttt{none} y \texttt{nope} se reproducen exactamente, \texttt{pape} y \texttt{pape\_efficient} son la misma corrida, y \texttt{bitnet\_conv} sin el embedding convolutivo es indistinguible de \texttt{bitnet\_on}. Contando clases de equivalencia, los $\nMixers$ mixers registrados se reducen a unos 50 modelos distintos a esta longitud de contexto. Cada uno de estos hechos es una advertencia que se arrastra y no un defecto escondido, exactamente como en la ronda~2.

\subsection{Dinámica de aprendizaje}

La Figura~\ref{fig:r3-curves-quantiles} muestra la curva de la población. El par mediano empieza en $\CurveStart$ nats y termina en $\CurveEnd$; la banda P10--P90 al final es $\BandPten$--$\BandPninety$, más estrecha que la dispersión entre familias. La mayor parte de la reducción ocurre en el primer cuarto: la mediana alcanza $\CurveQtwentyfive$ al 25\% del calendario, $\CurveQfifty$ a la mitad y $\CurveQseventyfive$ a las tres cuartas partes. La ronda~3 mide, por tanto, cuán rápido aprende una configuración, no dónde converge: las curvas siguen bajando cuando termina el calendario.

\figCurvesQuantiles
\figCurvesTopTen

El orden de los mejores pares se establece temprano. A la décima parte del calendario la pérdida por par ya ordena los pares con una correlación de Spearman de $\RankTen$ frente a su orden final, y al cuarto la correlación es $\RankTwentyfive$; en la Figura~\ref{fig:r3-curves-top10} los diez mejores se separan de la población antes de la mitad y rara vez se cruzan después. El número de épocas necesarias para alcanzar el 90\% de la caída total de pérdida de una corrida lo confirma: la mediana es $\ConvMedian\%$ del calendario, y las medianas por familia van desde $\ConvVision\%$ en las tareas de visión (donde la pérdida colapsa casi de inmediato) hasta $\ConvNorm\%$ en las ablaciones de normalización (donde las diferencias entre normas solo aparecen tarde). Los pares híbridos se sitúan en $\ConvPair\%$.

\figConvergence

\subsection{Presión del gradiente por bloque de parámetros}

El entrenador registra la norma L2 del gradiente de cada bloque de parámetros después del recorte: embeddings, codificación posicional, atención, FFN, normalización, cabeza, mHC, router, residual y otros. La Tabla~\ref{tab:r3-gradients} reproduce las medianas y la Figura~\ref{fig:r3-grad-groups} las trayectorias durante el entrenamiento.

\tblGradients
\figGradGroups

Tres hechos destacan. Primero, la cabeza soporta el gradiente mayor ($\GHead$ de media, $\GHeadFinal$ al final), seguida de la atención ($\GAttn$, $\GAttnFinal$) y la FFN ($\GFfn$, $\GFfnFinal$); los embeddings y la normalización son un orden de magnitud menores ($\GEmb$ y $\GNorm$). El orden es estable en toda la campaña, y por eso la normalización por bloque del optimizador no es una elección cosmética. Segundo, los coeficientes de variación son pequeños---$\CvHead$ para la cabeza, $\CvAttn$ para la atención, $\CvFfn$ para la FFN---así que la escala del gradiente de un bloque es una propiedad de la arquitectura, no del paso. Tercero, los bloques que solo existen en algunas configuraciones (\texttt{mhc}, \texttt{router}, \texttt{residual}) valen exactamente cero en las corridas que no los activan, lo que hace que las filas de mediana cero de la tabla sean informativas y no datos faltantes.

\figGradHeatmap

El mapa de calor (Figura~\ref{fig:r3-grad-heatmap}) lee las mismas estadísticas en función del progreso. La cabeza es plana, la atención crece ligeramente y la FFN decae por un factor de aproximadamente dos durante la primera mitad del entrenamiento antes de estabilizarse; los embeddings decaen a medida que se aprenden las estadísticas del vocabulario. Nada explota: no hay pasos con NaN en toda la campaña, hay $\InfSteps$ pasos con Inf (todos reparados en el sitio, en $\RepairRuns$ corridas) y el coeficiente de variación de la norma del gradiente se relaciona inversamente con la pérdida final ($\RhoGradCvLoss$), es decir, las corridas que terminan más abajo tienen gradientes algo más variables, no menos.

\figGradHealth

El recorte, sin embargo, es rutina y no excepción. La mediana del cociente de recorte medio es $\ClipMedian$---justo por encima del umbral 1---y el $\ClipAbove\%$ de las corridas pasan la campaña recortando de media, con un P90 de $\ClipPninety$. La ronda~2 reportó una mediana de 0,90 y un recorte excepcional; la diferencia es el calendario más largo y las tasas de aprendizaje más altas de la ronda~3. La consecuencia para el informe es que las normas de gradiente deben leerse como normas después del recorte, que es lo que registra el entrenador.

La huella de memoria es modesta y plana. La corrida mediana del análisis mantiene $\RssMeanMedian$ MB de memoria residente (P90 $\RssMeanPninety$ MB); las familias de visión son la excepción con unos $\RssVision$ MB, y la mayor media corrida a corrida es $\RssMeanMax$ MB (\RssMaxRun). La telemetría de pico existe para $\RssPeakRuns$ corridas de la campaña y alcanza los $\RssPeakMax$ MB. El atlas completo cabe en un solo host de CPU con una docena de workers.

\subsection{Ablaciones de optimizador y componentes}

\paragraph{Optimizadores: pérdida, presión del gradiente, CPU y memoria.}
El barrido de optimizadores corre $\OptN$ optimizadores sobre el mismo esqueleto de 309k parámetros, los mismos datos y la misma tasa de aprendizaje; la Tabla~\ref{tab:r3-optimizers} reporta las épocas efectivas, la pérdida final, la norma del gradiente, el cociente de recorte, el tiempo por paso y la memoria residente media de cada uno. Tres resultados destacan.

Primero, el barrido tiene un clúster de identidad. Ocho optimizadores (\OptClusterNames) producen trayectorias que coinciden dentro de $2\times10^{-6}$ nats---son la misma regla de actualización con estos hiperparámetros---así que la ablación mide realmente los 31 restantes. La ronda~2 vio el mismo clúster y lo leyó como optimizadores intercambiables; la identidad bit a bit de las trayectorias hace la lectura precisa.

Segundo, el ranking de pérdida es un ranking de presión de gradiente. Sobre las $\OptN$ corridas, la correlación de rango entre la pérdida final y la norma final del gradiente es $\RhoOptGradFinal$ (contra la media por corrida, $\RhoOptGradMean$) y contra el cociente de recorte $\RhoOptClip$. Los optimizadores que terminan más abajo son los que mantienen gradientes grandes y siguen recortando: \OptBest llega a $\OptBestLoss$ nats con el mayor gradiente final del barrido ($\OptBestGrad$, tras $\OptBestEpochs$ épocas), \OptSecond llega a $\OptSecondLoss$ con $\OptSecondGrad$, mientras \OptSgd termina en $\OptSgdLoss$ con $\OptSgdGrad$. La velocidad de aprendizaje a esta escala se compra con magnitud de gradiente, no con un precondicionador más sofisticado---la forma a nivel de optimizador del hallazgo por bloques de que el gradiente de la cabeza domina.

Tercero, la calidad y el coste de CPU son ortogonales: la correlación entre pérdida final y tiempo por paso es $\RhoOptStep$, y el calendario efectivo solo añade $\RhoOptEpochs$. Entre las 36 corridas con telemetría comparable, el tiempo por paso abarca $\OptStepLo$--$\OptStepHi$ ms (\OptStepLoName a \OptStepHiName), un 40\%; la memoria es aún más plana: el optimizador mediano mantiene $\OptRssMedian$ MB de memoria residente y solo \OptRssMaxName destaca con $\OptRssMax$ MB, un tercio por encima de la mediana, pagando su estado extra. Las tres incorporaciones tardías al barrido (\OptTimerNames) registran tiempos por paso de $\OptTimerLo$--$\OptTimerHi$ ms frente a 305--438 del resto con calendarios de pared más largos, así que su cronómetro cubre solo parte del trabajo y queda excluido del rango anterior; sus pérdidas y normas de gradiente no se ven afectadas.

\tblOptimizers
\figOptimizerPanel

\paragraph{Normalización.}
El ranking se invierte respecto de la ronda~2 (Tabla~\ref{tab:r3-norms}). \NormBestName gana con $\NormBest$ nats---medio nat por delante de \texttt{layer\_norm} con $\NormLayer$---mientras que \texttt{flash\_norm}, el ganador claro de la ronda~2 a 100 épocas, cierra la tabla con $\NormFlash$ (284 épocas efectivas) y \NormWorstName es el peor con $\NormWorst$. Los calendarios difieren (284--370 épocas efectivas), así que el orden carga con la advertencia de presupuesto habitual; pero el ganador de la ronda~2 no sobrevive al calendario más largo, y «moderno» tampoco es automáticamente mejor a esta escala.

\figAblationLoss
\tblNorms

\paragraph{Codificación posicional.}
La familia PAPE lidera, como en la ronda~2 (Tabla~\ref{tab:r3-pe}): \PeBestName con $\PeBest$ nats, luego \texttt{pape} y \texttt{pape\_efficient} con $\PePape$---gemelos idénticos bit a bit. La \texttt{rope} clásica y \texttt{sinusoidal\_rotary} reproducen la línea base exactamente ($\PeRope$, 312 épocas), \texttt{alibi} se sitúa en $\PeAlibi$ y las variantes sin posición \texttt{none}, \texttt{nope} y \texttt{bam} se agrupan en $\PeNone$, con \texttt{none} y \texttt{nope} idénticos hasta el sexto decimal. La corrida de seis épocas de \texttt{hope} es una prueba de humo y queda excluida de la tabla.

\tblPE

\paragraph{Interruptores estructurales.}
Las tres victorias sobre la línea base compartida son los embeddings factorizados ($\SwFactorized$ nats, 499 épocas efectivas), los embeddings dinámicos ($\SwDynamic$, 499) y el intercambio softmax-fijada/ssmax (\texttt{ssmax} con $\SwSsmax$). Los interruptores \texttt{fixed\_softmax}, \texttt{fixed\_ce}, \texttt{fixed\_both} y el embedding convolutivo reproducen la línea base exactamente ($\SwBase$), lo que los identifica como no-ops en esta configuración. Las variantes residuales cuestan pérdida (\texttt{full\_attn} $+0,14$, \texttt{block\_attn} $+0,16$, \texttt{none} $+0,22$ sobre el camino estándar en $\SwBase$), SwiGLU cuesta $+0,19$ con un calendario más corto (281 épocas), dos bucles Hyperloop cuestan $+0,16$, mHC $+0,13$ a 443 épocas, y el par BitNet (\texttt{bitnet\_on}/\texttt{bitnet\_conv}, idénticos) aterriza en $\SwBitnet$ tras las 499 épocas completas---el calendario más largo de la familia, consistente con el arranque lento conocido de BitNet. Los interruptores sin calendario completo (\texttt{moe}, \texttt{mod}, \texttt{pos\_hope}, \texttt{bitnet\_routers}) son pruebas de humo y quedan excluidos de la tabla.

\tblSwitches

\paragraph{Tamaño de lote, tareas, tokenizadores y MoE.}
Duplicar el tamaño de lote alcanza $\BatchTwo$ nats (499 épocas efectivas) frente a $\BatchOne$ de la referencia con lote 1 (312); la comparación carga con un confuso de calendario, y las pruebas de humo de enrutamiento MoE (15 de 17 variantes, cinco épocas) no dicen nada sobre la capacidad de los expertos---solo \texttt{moe\_latent} ($\MoeLatent$) y \texttt{moe\_interleaved} ($\MoeInterleaved$) corrieron el calendario completo, ambos en la mediana de los pares. Los interruptores de tarea anclan las escalas de pérdida: la LM causal corre entre $\TaskLmConformer$ y $\TaskLmDecoder$ nats (el decodificador conformer con $\TaskLmConformer$ es el más fuerte de los cuatro), la MLM en $\TaskMlm$ y la familia de tokenizadores entre $\TokLo$ y $\TokHi$. Ninguno de estos números es comparable con la población de pares MLM; son anclas de escala, como en la ronda~2.

\paragraph{Visión, audio y datos reales.}
Las familias de visión se comportan como en la ronda~2, con una novedad: la ablación de codificación posicional del ViT es un nulo limpio. Las $\VitPeRuns$ corridas \texttt{vit\_pe}---13 codificaciones $\times$ 3 tareas---reproducen, dentro de cada tarea, una misma pérdida hasta el sexto decimal (dispersión $\VitPeSpread$: clasificación $\VitClass$, predicción de parches $\VitPatch$, segmentación $\VitSeg$): en una rejilla de parches de $4\times4$ la codificación posicional es invisible. El orden de tareas coincide con la ronda~2 (segmentación la más fácil, predicción de parches la más difícil); la variante SSOG cuesta 0,18 nats en clasificación ($\VitClassSsog$ frente a $\VitClass$) y el ViT conformer aterriza en $\VitConformer$. La familia de audio va del ASR CTC con $\AudioCtc$ a la clasificación de audio con $\AudioCls$, y las corridas con datos reales enmarcan a las sintéticas: la LM causal de Borges aterriza en $\RdBorgesLm$ y la MLM de Borges en $\RdBorgesMlm$, indistinguible de la población MLM sintética---la comprobación de que el atlas sintético no medía un artefacto.

\subsection{Rendimiento y coste}

El tiempo de entrenamiento crece solo débilmente con los parámetros: un ajuste log-log sobre los pares MLM da un exponente de $\TimeSlope$ con $R^2=\TimeRtwo$. La razón es que lo que mide el reloj no es el número de parámetros---dominan el número de pasos, y el número de pasos varía por familia y por parada temprana. El coste normalizado (Figura~\ref{fig:r3-cost-mparam}) es la cantidad que compara arquitecturas con justicia: la corrida mediana gasta $\MinPerMparam$ minutos por millón de parámetros, y las medianas por familia se mueven en un factor de aproximadamente tres alrededor de ese valor. Un par híbrido cuesta $\MinPerMparamPair$ minutos por Mparámetro, algo por debajo de una línea base de un solo mixer ($\MinPerMparamSingle$) y por encima de una ablación de optimizador ($\MinPerMparamOpt$); las diferencias vienen del calendario efectivo y del coste de la actualización, no del número de parámetros.

\figTimeParams
\figCostMparam
\figThroughput
\figComputeFamily

La Figura~\ref{fig:r3-compute} muestra a dónde fueron las $\nComputeH$ horas. Los pares híbridos consumen el $\PairShare\%$ de la campaña ($\PairHours$ horas), las líneas base de un solo mixer el $\SecondShare\%$ y todas las demás familias menos del 3\%. Las familias de visión y audio son baratas en tiempo de pared pese a sus calendarios largos porque sus pasos son cortos; el panel de rendimiento (Figura~\ref{fig:r3-throughput}) dice lo mismo en tokens por segundo: los pares MLM corren a una mediana de $\TokensMedian$ tokens/s, mientras que la familia de visión es un orden de magnitud más lenta ($\TokensVision$ tokens/s) porque sus entradas son imágenes en lugar de secuencias de 64 tokens.

La campaña gastó $\nComputeH$ horas de tiempo de pared de entrenamiento repartidas en $\nCampaignDays$ días calendario con 13--26 workers, lo que es la eficiencia paralela del harness y no una propiedad de los modelos.

\subsection{Inferencia y cuantización}

La sección de inferencia estudia las $\nCombosInf$ corridas que produjeron filas de Hugging Face: $\nMlmCombos$ pares MLM, cuatro corridas de LM causal y una de clasificación de texto. Cada corrida se exportó sin cuantizar (\texttt{hf\_none}) y se reexportó con tres códecs en dos runtimes de carga (\texttt{q8\_0}, \texttt{q4\_0}, \texttt{i2\_s}, cada uno en \texttt{v1} y \texttt{v2}). Los dos runtimes muestrearon posiciones de máscara distintas---solo el 3,5\% de las claves de muestra de \texttt{v1} coinciden con la base---así que \texttt{v1} no puede emparejarse con \texttt{hf\_none}; la comparación limpia es la familia \texttt{v2}, que reutiliza las máscaras de la base en el 99,4\% de las claves. Esto es una propiedad del harness, no de los modelos, y es la principal advertencia de la sección.

\figInferenceFormat

La vista agregada (Figura~\ref{fig:r3-inference-format} y Tabla~\ref{tab:r3-formats}) dice que los pesos de 8 y 4 bits son esencialmente gratis a esta escala. Respecto de la exportación sin cuantizar, \texttt{q8\_0\_v2} cambia la exactitud top-1 en $\QEightDelta$ puntos porcentuales y deja sin cambios el $\QEightEqual\%$ de los pares, con solo el $\QEightWorse\%$ degradado; \texttt{q4\_0\_v2} la cambia en $\QFourDelta$ puntos con el $\QFourEqual\%$ sin cambios y el $\QFourWorse\%$ degradado. El códec de 2 bits es la excepción: \texttt{i2\_s\_v2} cambia la exactitud top-1 en $\ITwoDelta$ puntos de media, deja intacto el $\ITwoEqual\%$ de los pares y degrada el $\ITwoWorse\%$. Las filas \texttt{v1} se muestran en la tabla por completitud pero no son comparables con la base, porque sus máscaras son más difíciles: los mismos pares logran $\TopFiveQEightVone\%$ de top-5 con \texttt{v1} frente a $\TopFiveNone\%$ con \texttt{hf\_none}.

\tblFormats

La vista por muestra es más nítida que el agregado. Emparejando cada predicción cuantizada con la predicción base de la misma máscara, \texttt{q8\_0\_v2} coincide en el $\AgreeQEight\%$ de las muestras (IC del 95\% del bootstrap por par), \texttt{q4\_0\_v2} en el $\AgreeQFour\%$ y \texttt{i2\_s\_v2} en solo el $\AgreeITwo\%$. El modelo de 2 bits no es una copia algo peor del profesor: es un modelo distinto que casualmente conserva el top-1 agregado en los pares cuya exactitud base ya era casi cero.

\figQuantDelta
\figQuantAgreement

Dos resultados negativos completan la sección. El número de parámetros no predice la robustez a la cuantización de 2 bits (Spearman $\RhoITwoParams$ entre el delta de \texttt{i2\_s} y el recuento de parámetros), y la pérdida de entrenamiento no predice la calidad retenida (Spearman $\RhoLossTopFive$). A esta escala la única señal fiable es la comparación misma: medir la exportación cuantizada contra la no cuantizada, sobre las mismas muestras, y reportar el acuerdo en lugar del agregado.

\figQuantParams
\figTrainInference

\subsection{Fiabilidad y taxonomía de fallos}
\label{sec:r3-reliability}

El entrenamiento es fiable: $\TrainingOk$ de $\nRuns$ corridas terminan bien (tasa de fallo $\FailureRate\%$). Los diez fallos son pares híbridos que contienen \texttt{engram}; ocho salen con código 1 y dos con código 134 (SIGABRT), y ninguna otra familia falla. El fallo es estructural y no aleatorio---\texttt{engram} es el único mixer con un banco de memoria externo, y diez de los cincuenta y tres pares que lo usan fallan mientras los otros cuarenta y tres entrenan---y por eso el atlas sigue conteniendo todas las celdas: los diez pares fallidos están en la tabla de población pero ausentes de los análisis de pérdida e inferencia.

El resto del pipeline está más sano que en la ronda~2. La exportación a Transformers no falla nunca ($\nExportTotal$ registros, todos OK) y la evaluación de inferencia falla una vez (una corrida \texttt{pape\_efficient}). El registro de etapas (Figura~\ref{fig:r3-stage-status}) hace visible el cuadro: entrenamiento y exportación están casi enteramente en verde, y la inferencia está en verde con una fila roja.

\figStageStatus

\subsection{Correlaciones}

La Figura~\ref{fig:r3-correlation} es la matriz de Spearman de las variables medidas sobre los $\nPairs$ pares. Las correlaciones fuertes no triviales son las que describen el calendario y no la arquitectura: el tiempo de entrenamiento crece con el número de épocas ($\RhoDurationEpochs$), el rendimiento es casi el inverso del tiempo por paso ($\RhoTokensStep$) y la pérdida media baja a medida que el calendario se alarga ($\RhoLossEpochs$). El número de parámetros se correlaciona moderadamente con la pérdida ($\RhoParamsLoss$) y con la magnitud del gradiente ($\RhoParamsGrad$) y débilmente con el tiempo de pared ($\RhoDurationParams$), la forma cuantitativa de la ley de coste débil de la sección anterior.

Merece la pena hacer explícito un contraste con la ronda~2. Allí la matriz sobre todas las filas estaba dominada por el contraste entre familias y leía $-0,43$ contra $-0,20$ dentro de los pares; aquí, recalcular la correlación parámetros--pérdida sobre las $\nAllRows$ corridas de análisis en lugar de solo los pares la deja en $\RhoParamsLossAll$, porque las familias de ablación comparten esqueleto y la población es homogénea. La ronda~3 no tiene ese confuso que desactivar: el número dentro de los pares es el número de la población.

\figCorrelation

\subsection{Resumen de hallazgos}

\begin{enumerate}[leftmargin=1.4em,itemsep=2pt]
\item El atlas completo es medible en una semana de un solo host de CPU: $\nRuns$ corridas, $\nSteps$ millones de pasos, $\nComputeH$ horas, una tasa de fallo de entrenamiento del $\FailureRate\%$ y ningún paso con NaN.
\item La pila de mixers domina el espacio de diseño. El mejor par alcanza $\BestPairLoss$ nats y el peor $\WorstPairLoss$, mientras que las ablaciones de optimizador, normalización y codificación posicional mueven la pérdida menos que la dispersión entre pares.
\item El segundo mixer es la palanca y la interacción es real: las medianas del rol B abarcan $\RangeB$ nats frente a $\RangeA$ del rol A, el P90 de la sinergia absoluta es $\SynPninety$ nats, y mezclar dos mixers distintos supera a duplicar uno por 0,22 nats en la mediana.
\item El tamaño ayuda pero no decide: la correlación de rango entre parámetros y pérdida es $\RhoParamsLoss$, y los mixers más baratos se reparten a ambos lados de la mediana.
\item El aprendizaje es rápido y el ranking se decide temprano: el par mediano alcanza el 90\% de su caída de pérdida al $\ConvMedian\%$ del calendario, la pérdida al cuarto del calendario ya ordena los pares con $\RankTwentyfive$, y los diez mejores se separan de la población en el primer 15\%.
\item La presión del gradiente es estable y específica por bloque: dominan la cabeza ($\GHead$), la atención ($\GAttn$) y la FFN ($\GFfn$), los coeficientes de variación de los bloques activos se quedan por debajo de 0,35 y el orden de los bloques es constante en la campaña.
\item La ablación de optimizadores convierte el ranking de pérdida en un ranking de presión de gradiente ($\RhoOptGradFinal$ entre pérdida final y norma final del gradiente sobre $\OptN$ optimizadores), un clúster de ocho optimizadores es idéntico bit a bit, y la calidad es ortogonal al coste de CPU ($\RhoOptStep$) y a la memoria (mediana $\OptRssMedian$ MB).
\item Las ablaciones de componentes se separan: \NormBestName lidera las normalizaciones con $\NormBest$ (la victoria de \texttt{flash\_norm} de la ronda~2 no sobrevive al calendario más largo), la familia PAPE lidera las codificaciones posicionales, los embeddings factorizados ganan los interruptores, y el barrido de codificaciones posicionales del ViT es un resultado nulo ($\VitPeRuns$ corridas, dispersión cero dentro de cada tarea).
\item El recorte es rutina, no excepción: la mediana del cociente de recorte es $\ClipMedian$ y el $\ClipAbove\%$ de las corridas recortan de media, lo que cambia la lectura de las normas de gradiente a cantidades posteriores al recorte.
\item El tiempo de entrenamiento es casi independiente del número de parámetros con un calendario fijo (exponente log-log $\TimeSlope$, $R^2=\TimeRtwo$): lo domina el número de pasos, y los pares híbridos absorben el $\PairShare\%$ de las $\nComputeH$ horas.
\item La cuantización a 8 y 4 bits es esencialmente sin pérdida en el runtime \texttt{v2} ($\AgreeQEight\%$ y $\AgreeQFour\%$ de acuerdo por muestra), mientras que los pesos de 2 bits cambian el modelo cualitativamente ($\AgreeITwo\%$ de acuerdo, $\ITwoDelta$ puntos de top-1).
\item El pipeline es fiable de extremo a extremo: el entrenamiento falla el $\FailureRate\%$ de las veces, todo en el banco de memoria de un solo mixer, la exportación a Transformers no falla nunca y la evaluación falla una vez.
\item La ronda hereda la advertencia central de la ronda~2: la pérdida de entrenamiento y la calidad retenida son casi incorrelacionadas ($\RhoLossTopFive$), así que el atlas debe leerse como un mapa de velocidad y estabilidad de aprendizaje, no como una tabla de líderes.
\end{enumerate}

\subsection{Reproducir este apéndice}

El apéndice lo generan cuatro scripts en \texttt{experimentos/3/report/}, en orden:

\begin{verbatim}
python experimentos/to_csv.py 3                       # exportación plana a CSV
python experimentos/3/report/build_data.py            # master/epochs/inference
python experimentos/3/report/make_paper_figures.py    # PDF/SVG, ambos idiomas
python experimentos/3/report/make_report.py           # estadísticas + este apéndice
\end{verbatim}

\texttt{build\_data.py} recorre una sola vez el log por paso de 6,6\,GB (unos diez minutos) y cachea todos los agregados; los otros tres scripts tardan segundos. \texttt{make\_report.py} escribe \texttt{out/stats.json} con todos los números citados arriba, así que cualquier afirmación se puede rastrear hasta el CSV que la produjo.
"""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
FIG_ALIASES = {
    "ParamsFamily": "params_family_box",
    "ParamsComposition": "params_composition",
    "ParamsVsLoss": "params_vs_loss",
    "LossFamily": "loss_family_box",
    "CurvesQuantiles": "curves_quantiles",
    "CurvesTopTen": "curves_top10",
    "Convergence": "convergence_speed",
    "Matrix": "matrix_loss",
    "Roles": "roles_scatter",
    "TopB": "top5_by_b",
    "Synergy": "synergy",
    "Pareto": "pareto",
    "GradGroups": "grad_groups_curves",
    "GradHeatmap": "grad_heatmap",
    "GradHealth": "grad_health",
    "OptimizerPanel": "optimizer_panel",
    "AblationLoss": "ablation_loss",
    "TimeParams": "time_vs_params",
    "CostMparam": "cost_per_mparam",
    "Throughput": "throughput_family",
    "ComputeFamily": "compute_by_family",
    "InferenceFormat": "inference_format_accuracy",
    "QuantDelta": "quant_delta_dist",
    "QuantAgreement": "quant_agreement",
    "QuantParams": "quant_delta_vs_params",
    "TrainInference": "train_vs_inference",
    "StageStatus": "stage_status",
    "Correlation": "correlation",
}

TBL_ALIASES = {
    "Family": "family_table",
    "TopLoss": "top_loss_table",
    "TopInference": "top_inference_table",
    "Gradients": "grad_table",
    "Cost": "cost_table",
    "Formats": "format_table",
    "Reliability": "reliability_table",
    "Optimizers": "optimizer_table",
    "Norms": "norm_table",
    "PE": "pe_table",
    "Switches": "switch_table",
    "BestB": "best_b_table",
    "SynergyTable": "synergy_table",
}

#: Prose token -> statistics key.
NUM_ALIASES = {
    "nMixers": "n_mixers", "nPairs": "n_pairs", "nSingles": "n_singles",
    "nAblations": "n_ablations", "nRuns": "n_runs", "nRecords": "n_records",
    "nSteps": "steps_total", "nComputeH": "compute_h",
    "nCampaignDays": "campaign_days", "CampaignDays": "campaign_days",
    "ComputeDays": "compute_days", "nExportTotal": "export_total",
    "nInferenceTotal": "inference_total", "nTrainingOk": "training_ok",
    "TrainingOk": "training_ok", "nTrainingFailed": "training_failed", "TrainingFailed": "training_failed",
    "nDeployFailed": "deploy_failed", "nDeployTotal": "deploy_total",
    "nCombosInf": "n_combos_inf", "nRowsInf": "n_rows_inf",
    "nMlmCombos": "n_mlm_combos", "nRanFull": "ran_full",
    "nEarlyStop": "early_stop", "nSmoke": "smoke_runs", "nAnalysis": "n_analysis", "nParamsMin": "params_min",
    "nParamsMedian": "params_median", "nParamsMax": "params_max",
    "ShareEmb": "share_emb", "ShareHead": "share_head",
    "ShareAttn": "share_attn", "ShareFfn": "share_ffn",
    "ShareNorm": "share_norm", "MixerCheap": "mixer_cheap",
    "MixerExpensive": "mixer_expensive", "RhoParamsLoss": "rho_params_loss",
    "LossMedian": "loss_median", "LossMax": "loss_max", "LossMin": "loss_min",
    "MoeLoss": "moe_loss", "BestPair": "best_pair",
    "BestPairLoss": "best_pair_loss", "WorstPair": "worst_pair",
    "WorstPairLoss": "worst_pair_loss", "TopOneMean": "top1_mean",
    "TopFiveMean": "top5_mean", "RhoLossTopFive": "rho_loss_top5",
    "CurveStart": "curve_start", "CurveEnd": "curve_end",
    "CurveQtwentyfive": "curve_q25", "CurveQfifty": "curve_q50",
    "CurveQseventyfive": "curve_q75", "BandPten": "band_p10",
    "BandPninety": "band_p90", "ConvMedian": "conv_median",
    "ConvPair": "conv_pair", "ConvVision": "conv_vision",
    "ConvNorm": "conv_norm", "GHead": "g_head",
    "GHeadFinal": "g_head_final", "GAttn": "g_attn",
    "GAttnFinal": "g_attn_final", "GFfn": "g_ffn",
    "GFfnFinal": "g_ffn_final", "GNorm": "g_norm", "GEmb": "g_emb",
    "CvHead": "cv_head", "CvAttn": "cv_attn", "CvFfn": "cv_ffn",
    "ClipMedian": "clip_median", "ClipAbove": "clip_above",
    "ClipPninety": "clip_p90", "InfSteps": "inf_steps",
    "RepairRuns": "repair_runs", "RhoGradCvLoss": "rho_grad_cv_loss",
    "RssMedian": "rss_median", "RssMax": "rss_max",
    "TimeSlope": "time_slope", "TimeRtwo": "time_r2",
    "MinPerMparam": "min_per_mparam", "TokensMedian": "tokens_median",
    "PairShare": "pair_share", "PairHours": "pair_hours",
    "SecondShare": "second_share", "QEightDelta": "q8_delta",
    "QEightEqual": "q8_equal", "QEightWorse": "q8_worse",
    "QFourDelta": "q4_delta", "QFourEqual": "q4_equal",
    "QFourWorse": "q4_worse", "ITwoDelta": "i2_delta",
    "ITwoEqual": "i2_equal", "ITwoWorse": "i2_worse",
    "AgreeQEight": "agree_q8", "AgreeQFour": "agree_q4",
    "AgreeITwo": "agree_i2", "TopFiveNone": "top5_none",
    "TopFiveQEightVone": "top5_q8_v1", "RhoITwoParams": "i2_params_rho",
    "FailureRate": "failure_rate", "RhoDurationEpochs": "rho_duration_epochs",
    "RhoLossEpochs": "rho_loss_epochs", "RhoGradMeanloss": "rho_grad_meanloss",
    "RhoParamsGrad": "rho_params_grad",
    "RhoTokensStep": "rho_tokens_step",
    "RhoDurationParams": "rho_duration_params",
    "RankTwentyfive": "rank_25",
    "PairParamsMedian": "pair_params_median",
    "nPairParamsMedian": "pair_params_median",
    "MinPerMparamOpt": "min_per_mparam_opt",
    "MinPerMparamSingle": "min_per_mparam_single",
    "MinPerMparamPair": "min_per_mparam_pair",
    "TokensVision": "tokens_vision",
    "TokensAudio": "tokens_audio",
    "ShortRuns": "short_runs",
    "nCells": "n_cells", "nMeasured": "n_measured", "nHoles": "n_holes",
    "nCommonRoles": "n_common_roles", "RangeA": "range_a", "RangeB": "range_b",
    "BestAName": "best_a_name", "BestAMed": "best_a_med",
    "WorstAName": "worst_a_name", "WorstAMed": "worst_a_med",
    "BestBName": "best_b_name", "BestBMed": "best_b_med",
    "WorstBName": "worst_b_name", "WorstBMed": "worst_b_med",
    "RhoRoles": "rho_roles", "OnlyA": "only_a", "OnlyB": "only_b",
    "SingleMedian": "single_median", "BestSingleLoss": "best_single_loss",
    "PairsBetterSingle": "pairs_better_single",
    "SynMedian": "syn_median", "SynPninety": "syn_p90", "SynShare": "syn_share",
    "SynTopName": "syn_top_name", "SynTopVal": "syn_top_val",
    "SynBottomName": "syn_bottom_name", "SynBottomVal": "syn_bottom_val",
    "DupGroups": "dup_groups", "DupPairs": "dup_pairs",
    "TopBName": "top5_b_name", "TopBVal": "top5_b_val",
    "TopBConformer": "top5_b_conformer",
    "ParetoN": "pareto_n", "ParetoFastName": "pareto_fast_name",
    "ParetoFastLoss": "pareto_fast_loss", "ParetoFastHours": "pareto_fast_hours",
    "ParetoMidName": "pareto_mid_name", "ParetoMidLoss": "pareto_mid_loss",
    "ParetoMidHours": "pareto_mid_hours",
    "RankTen": "rank_10",
    "RssMeanMedian": "rss_mean_median", "RssMeanPninety": "rss_mean_p90",
    "RssMeanMax": "rss_mean_max", "RssMaxRun": "rss_max_run",
    "RssVision": "rss_vision", "RssPeakRuns": "rss_peak_runs",
    "RssPeakMax": "rss_peak_max",
    "RhoParamsLossAll": "rho_params_loss_all", "nAllRows": "n_all_rows",
    "OptN": "opt_n", "OptClusterNames": "opt_cluster_names",
    "OptBest": "opt_best", "OptBestLoss": "opt_best_loss",
    "OptBestGrad": "opt_best_grad", "OptBestEpochs": "opt_best_epochs",
    "OptSecond": "opt_second", "OptSecondLoss": "opt_second_loss",
    "OptSecondGrad": "opt_second_grad",
    "OptThird": "opt_third", "OptThirdLoss": "opt_third_loss",
    "OptWorst": "opt_worst", "OptWorstLoss": "opt_worst_loss",
    "OptSgd": "opt_sgd", "OptSgdLoss": "opt_sgd_loss", "OptSgdGrad": "opt_sgd_grad",
    "RhoOptGradFinal": "rho_opt_grad_final", "RhoOptGradMean": "rho_opt_grad_mean",
    "RhoOptClip": "rho_opt_clip", "RhoOptStep": "rho_opt_step",
    "RhoOptEpochs": "rho_opt_epochs",
    "OptStepLoName": "opt_step_lo_name", "OptStepLo": "opt_step_lo",
    "OptStepHiName": "opt_step_hi_name", "OptStepHi": "opt_step_hi",
    "OptTimerNames": "opt_timer_names", "OptTimerLo": "opt_timer_lo",
    "OptTimerHi": "opt_timer_hi",
    "OptRssMedian": "opt_rss_median", "OptRssMaxName": "opt_rss_max_name",
    "OptRssMax": "opt_rss_max",
    "NormBestName": "norm_best_name", "NormBest": "norm_best",
    "NormLayer": "norm_layer", "NormFlash": "norm_flash",
    "NormWorstName": "norm_worst_name", "NormWorst": "norm_worst",
    "PeBestName": "pe_best_name", "PeBest": "pe_best", "PePape": "pe_pape",
    "PeRope": "pe_rope", "PeNone": "pe_none", "PeAlibi": "pe_alibi",
    "SwFactorized": "sw_factorized", "SwDynamic": "sw_dynamic",
    "SwSsmax": "sw_ssmax", "SwBase": "sw_base", "SwSwiglu": "sw_swiglu",
    "SwMhc": "sw_mhc", "SwBitnet": "sw_bitnet",
    "SwResidNone": "sw_resid_none", "SwLoops": "sw_loops",
    "BatchTwo": "batch_two", "BatchOne": "batch_one",
    "MoeLatent": "moe_latent", "MoeInterleaved": "moe_interleaved",
    "TaskLmConformer": "task_lm_conformer", "TaskLmDecoder": "task_lm_decoder",
    "TaskMlm": "task_mlm", "TokLo": "tok_lo", "TokHi": "tok_hi",
    "VitPeRuns": "vit_pe_runs", "VitPeSpread": "vit_pe_spread",
    "VitClass": "vit_class", "VitPatch": "vit_patch", "VitSeg": "vit_seg",
    "VitClassSsog": "vit_class_ssog", "VitConformer": "vit_conformer",
    "AudioCtc": "audio_ctc", "AudioCls": "audio_cls",
    "RdBorgesLm": "rd_borges_lm", "RdBorgesMlm": "rd_borges_mlm",
}


def _figure_snippet(name: str, prefix: str, suffix: str, captions: dict) -> str:
    """Wrap a generated PDF in a figure float."""
    caption, label = captions[name]
    return (
        "\\begin{figure}[tbp]\n\\centering\n"
        f"\\fitfigure{{\\includegraphics{{{prefix}/{name}{suffix}.pdf}}}}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{figure}}"
    )


def render(stats: dict, tables: dict[str, str], lang: str) -> str:
    """Render the annex for ``lang`` (``en`` or ``es``).

    Args:
        stats: Output of ``make_report.compute_stats``.
        tables: LaTeX snippets from ``make_report.build_tables``.
        lang: ``en`` or ``es``.

    Returns:
        The complete LaTeX document body.
    """
    en = lang == "en"
    suffix = "" if en else "-es"
    prefix = "figures/experiments/round3" if en else "../paper/figures/experiments/round3"
    captions = dict(_FIGURES_EN if en else _FIGURES_ES)

    numbers = values(stats)
    resolved = {
        token: numbers.get(key, "n/a")
        for token, key in NUM_ALIASES.items()
    }

    text = EN if en else ES
    text = re.sub(
        r"\\fig([A-Z]\w*)",
        lambda match: _figure_snippet(FIG_ALIASES[match.group(1)], prefix, suffix, captions)
        if match.group(1) in FIG_ALIASES
        else match.group(0),
        text,
    )
    text = re.sub(
        r"\\tbl([A-Z]\w*)",
        lambda match: tables.get(TBL_ALIASES.get(match.group(1), ""), match.group(0)),
        text,
    )
    text = re.sub(
        r"\\(n?[A-Z]\w*)",
        lambda match: resolved.get(match.group(1), match.group(0)),
        text,
    )
    keep = {"Delta", "Theta", "Omega", "Sigma", "Gamma", "Lambda", "Phi", "Psi"}
    if not en:
        # Spanish typography uses a decimal comma; the generated numbers are
        # formatted with dots, so translate the decimal separator.
        text = re.sub(r"(?<=\d)\.(?=\d)", ",", text)
        text = text.replace("leftmargin=1,4em", "leftmargin=1.4em")
    leftovers = sorted(set(re.findall(r"\\(n?[A-Z]\w*)", text)) - keep)
    if leftovers:
        print(f"  warning: unresolved annex tokens: {leftovers}")
    return text


#: Figure captions and labels, keyed by the figure stem.
_FIGURES_EN = {
    "params_family_box": ("Parameter budget by experiment family (log scale).", "fig:r3-params-family"),
    "params_composition": ("Where the parameters live, by family. Embeddings and the head are fixed costs; attention and the FFN carry the architectural differences.", "fig:r3-params-composition"),
    "params_vs_loss": ("Final training loss against parameter count for the hybrid pairs.", "fig:r3-params-loss"),
    "loss_family_box": ("Final training loss by family. Tasks are not comparable across families; the log axis keeps them visible.", "fig:r3-loss-family"),
    "curves_quantiles": ("Population learning curves with a P10--P90 band, normalised to each run's own schedule.", "fig:r3-curves-quantiles"),
    "curves_top10": ("The ten best hybrid pairs over their own training schedule.", "fig:r3-curves-top10"),
    "convergence_speed": ("Fraction of the schedule needed to reach 90\\% of a run's total loss drop, by family.", "fig:r3-convergence"),
    "matrix_loss": ("The mixer atlas: final training loss of every hybrid pair in the $54\\times54$ grid. Rows are the first mixer (A), columns the second (B); the ten empty cells are the \\texttt{engram} pairs that crash.", "fig:r3-matrix"),
    "roles_scatter": ("Median loss of every mixer in the A role against the B role (52 mixers with a full schedule in both roles). The diagonal is the identity; the B axis is the wider one.", "fig:r3-roles"),
    "top5_by_b": ("Mean held-out top-5 hit rate of the pairs grouped by second mixer (mixers with at least 20 evaluated pairs). \\texttt{conformer} (orange) is the best-loss B.", "fig:r3-top5-b"),
    "synergy": ("The eight best and eight worst mixer interactions by synergy $S_{A,B}=(\\mu_A+\\mu_B)/2-L_{A,B}$. Positive values beat what the marginals predict.", "fig:r3-synergy"),
    "pareto": ("The speed--quality Pareto front of the hybrid pairs. No pair off the front is both faster and lower-loss.", "fig:r3-pareto"),
    "grad_groups_curves": ("Median gradient norm per parameter block over training (post-clip).", "fig:r3-grad-groups"),
    "grad_heatmap": ("Gradient norm per block relative to its first epoch, in log10 units.", "fig:r3-grad-heatmap"),
    "grad_health": ("Clip ratio distribution and gradient variability against final loss.", "fig:r3-grad-health"),
    "optimizer_panel": ("Optimizer ablation, sorted by final loss: loss, mean gradient norm, step time and mean resident memory per optimizer.", "fig:r3-optimizers"),
    "ablation_loss": ("Component ablations on the shared backbone: normalization, text positional encodings and structural switches (smoke runs excluded).", "fig:r3-ablations"),
    "time_vs_params": ("Training wall time against parameter count, with the log-log fit.", "fig:r3-time-params"),
    "cost_per_mparam": ("Training minutes per million parameters, by family.", "fig:r3-cost-mparam"),
    "throughput_family": ("Logging throughput by family (tokens per second, log scale).", "fig:r3-throughput"),
    "compute_by_family": ("Share of the training wall time by family.", "fig:r3-compute"),
    "inference_format_accuracy": ("Held-out top-1 and top-5 accuracy of the unquantized export and the three v2 codecs.", "fig:r3-inference-format"),
    "quant_delta_dist": ("Mean top-1 delta against the unquantized export (95\\% bootstrap CI) and the share of pairs that degrade, stay equal or improve.", "fig:r3-quant-delta"),
    "quant_agreement": ("Sample-level agreement with the unquantized predictions for the v2 runtime.", "fig:r3-quant-agreement"),
    "quant_delta_vs_params": ("2-bit accuracy delta against parameter count.", "fig:r3-quant-params"),
    "train_vs_inference": ("Final training loss against held-out top-5 hit rate for the MLM pairs.", "fig:r3-train-inference"),
    "stage_status": ("Pipeline status per stage.", "fig:r3-stage-status"),
    "correlation": ("Spearman correlations of the measured features over the hybrid pairs.", "fig:r3-correlation"),
}

_FIGURES_ES = {
    "params_family_box": ("Presupuesto de parámetros por familia (escala logarítmica).", "fig:r3-params-family"),
    "params_composition": ("Dónde viven los parámetros, por familia. Los embeddings y la cabeza son costes fijos; la atención y la FFN concentran las diferencias arquitectónicas.", "fig:r3-params-composition"),
    "params_vs_loss": ("Pérdida final de entrenamiento contra número de parámetros para los pares híbridos.", "fig:r3-params-loss"),
    "loss_family_box": ("Pérdida final de entrenamiento por familia. Las tareas no son comparables entre familias; el eje logarítmico las mantiene visibles.", "fig:r3-loss-family"),
    "curves_quantiles": ("Curvas de aprendizaje de la población con banda P10--P90, normalizadas al calendario propio de cada corrida.", "fig:r3-curves-quantiles"),
    "curves_top10": ("Los diez mejores pares híbridos sobre su propio calendario de entrenamiento.", "fig:r3-curves-top10"),
    "convergence_speed": ("Fracción del calendario necesaria para alcanzar el 90\\% de la caída total de pérdida, por familia.", "fig:r3-convergence"),
    "matrix_loss": ("El atlas de mixers: pérdida final de entrenamiento de cada par híbrido en la rejilla $54\\times54$. Las filas son el primer mixer (A), las columnas el segundo (B); las diez celdas vacías son los pares con \\texttt{engram} que fallan.", "fig:r3-matrix"),
    "roles_scatter": ("Pérdida mediana de cada mixer en el rol A frente al rol B (52 mixers con calendario completo en ambos roles). La diagonal es la identidad; el eje B es el más ancho.", "fig:r3-roles"),
    "top5_by_b": ("Acierto top-5 retenido medio de los pares agrupados por segundo mixer (mixers con al menos 20 pares evaluados). \\texttt{conformer} (naranja) es el mejor B en pérdida.", "fig:r3-top5-b"),
    "synergy": ("Las ocho mejores y ocho peores interacciones por sinergia $S_{A,B}=(\\mu_A+\\mu_B)/2-L_{A,B}$. Los valores positivos superan lo que predicen los marginales.", "fig:r3-synergy"),
    "pareto": ("La frontera de Pareto velocidad--calidad de los pares híbridos. Ningún par fuera de la frontera es a la vez más rápido y con menor pérdida.", "fig:r3-pareto"),
    "grad_groups_curves": ("Norma del gradiente mediana por bloque de parámetros durante el entrenamiento (post-recorte).", "fig:r3-grad-groups"),
    "grad_heatmap": ("Norma del gradiente por bloque relativa a su primera época, en unidades log10.", "fig:r3-grad-heatmap"),
    "grad_health": ("Distribución del cociente de recorte y variabilidad del gradiente contra la pérdida final.", "fig:r3-grad-health"),
    "optimizer_panel": ("Ablación de optimizadores, ordenada por pérdida final: pérdida, norma media del gradiente, tiempo por paso y memoria residente media por optimizador.", "fig:r3-optimizers"),
    "ablation_loss": ("Ablaciones de componentes sobre el esqueleto compartido: normalización, codificaciones posicionales de texto e interruptores estructurales (sin pruebas de humo).", "fig:r3-ablations"),
    "time_vs_params": ("Tiempo de pared de entrenamiento contra número de parámetros, con el ajuste log-log.", "fig:r3-time-params"),
    "cost_per_mparam": ("Minutos de entrenamiento por millón de parámetros, por familia.", "fig:r3-cost-mparam"),
    "throughput_family": ("Rendimiento de registro por familia (tokens por segundo, escala logarítmica).", "fig:r3-throughput"),
    "compute_by_family": ("Proporción del tiempo de pared de entrenamiento por familia.", "fig:r3-compute"),
    "inference_format_accuracy": ("Exactitud top-1 y top-5 retenida de la exportación sin cuantizar y de los tres códecs v2.", "fig:r3-inference-format"),
    "quant_delta_dist": ("Delta medio de top-1 contra la exportación sin cuantizar (IC bootstrap del 95\\%) y proporción de pares que empeoran, quedan igual o mejoran.", "fig:r3-quant-delta"),
    "quant_agreement": ("Acuerdo por muestra con las predicciones sin cuantizar para el runtime v2.", "fig:r3-quant-agreement"),
    "quant_delta_vs_params": ("Delta de exactitud de 2 bits contra número de parámetros.", "fig:r3-quant-params"),
    "train_vs_inference": ("Pérdida final de entrenamiento contra acierto top-5 retenido para los pares MLM.", "fig:r3-train-inference"),
    "stage_status": ("Estado del pipeline por etapa.", "fig:r3-stage-status"),
    "correlation": ("Correlaciones de Spearman de las variables medidas sobre los pares híbridos.", "fig:r3-correlation"),
}
