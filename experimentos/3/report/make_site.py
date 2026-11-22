"""Build the Experiment 3 atlas page for erickfmm.github.io.

The page mirrors the experiment-3 appendix and adds the interactive A x B
atlas that Experiment 2 introduced: a 54 x 54 matrix with one coloured cell
per hybrid pair, a metric selector, the mixer ranking by median loss and the
A-role vs B-role effect.

Usage:
    uv run --no-sync --with pandas --with numpy --with matplotlib python \
        experimentos/3/report/make_site.py

Outputs (in the sibling website repository):

    erickfmm.github.io/frankenstein-transformer/experiments/3/index.html
    .../3/data/atlas.json, atlas.csv, mixers.csv
    .../3/img/<figure>-{dark,light}.png   (21 figures x 2 themes)
    .../3/vendor/echarts.min.js           (copied from Experiment 2)
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import make_charts as mc  # noqa: E402

SITE = C.PROJECT / "erickfmm.github.io" / "frankenstein-transformer" / "experiments" / "3"
EXP2_PAGE = C.PROJECT / "erickfmm.github.io" / "frankenstein-transformer" / "experiments" / "2" / "index.html"
EXP2_VENDOR = EXP2_PAGE.parent / "vendor" / "echarts.min.js"
EXP3_PAGE = SITE / "index.html"
EXP3_DATA = SITE / "data"
EXP3_IMG = SITE / "img"

#: Analysis threshold shared with the report (smoke runs are excluded).
MIN_EPOCHS = 100


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_atlas(master: pd.DataFrame, accuracy: pd.DataFrame) -> dict:
    """Assemble the matrix, the ranking and the A/B role effect.

    Args:
        master: Per-run table from ``build_data.py``.
        accuracy: Per-(run, format, metric) inference aggregates.

    Returns:
        JSON-serialisable atlas payload.
    """
    pairs = master[master["family"] == "pair"].copy()
    pairs["a"] = pairs["run"].str.extract(r"^attn_(.+?)__")[0].map(mc.C.tidy_mixer)
    pairs["b"] = pairs["run"].str.extract(r"__(.+)$")[0].map(mc.C.tidy_mixer)
    # The axes follow the registry order used to generate the pairs (A before
    # B), reconstructed by topologically sorting the a -> b edges. Alphabetical
    # order would push half of the pairs below the diagonal and make the
    # upper-triangle atlas look perforated.
    edges = [(a, b) for a, b in zip(pairs["a"], pairs["b"]) if a and b]
    nodes = sorted({mixer for edge in edges for mixer in edge})
    successors = {mixer: [] for mixer in nodes}
    indegree = {mixer: 0 for mixer in nodes}
    for a, b in edges:
        successors[a].append(b)
        indegree[b] += 1
    queue = sorted(mixer for mixer in nodes if indegree[mixer] == 0)
    order = []
    while queue:
        mixer = queue.pop(0)
        order.append(mixer)
        for successor in successors[mixer]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)
        queue.sort()
    mixers = order if len(order) == len(nodes) else sorted(nodes)
    index = {mixer: position for position, mixer in enumerate(mixers)}

    base = accuracy[(accuracy["deploy_format"] == "hf_none")].set_index("combo_id")
    pairs = pairs.join(base[["accuracy", "top5"]], on="run")

    cells = []
    for _, row in pairs.iterrows():
        a, b = row["a"], row["b"]
        if a not in index or b not in index:
            continue
        cells.append(
            [
                index[a],
                index[b],
                {
                    "loss": None if pd.isna(row["final_loss"]) else round(float(row["final_loss"]), 4),
                    "params": None if pd.isna(row["total"]) else round(float(row["total"]) / 1e6, 4),
                    "duration": None if pd.isna(row["duration_h"]) else round(float(row["duration_h"]), 4),
                    "top1": None if pd.isna(row.get("accuracy")) else round(float(row["accuracy"]), 5),
                    "top5": None if pd.isna(row.get("top5")) else round(float(row["top5"]), 5),
                    "status": str(row["status"]),
                },
            ]
        )

    analysis = pairs[pairs["final_epoch"].fillna(0) >= MIN_EPOCHS]
    by_a = analysis.groupby("a")["final_loss"].median()
    by_b = analysis.groupby("b")["final_loss"].median()
    common = sorted(set(by_a.index) & set(by_b.index))
    rank = (
        analysis.groupby("a")
        .agg(loss=("final_loss", "median"), params=("total", "median"), n=("final_loss", "size"))
        .reset_index()
        .rename(columns={"a": "mixer"})
    )
    rank = rank.sort_values("loss")

    top = pairs.dropna(subset=["final_loss"]).nsmallest(10, "final_loss")
    bottom = pairs.dropna(subset=["final_loss"]).nlargest(10, "final_loss")

    def pair_rows(frame):
        out = []
        for _, row in frame.iterrows():
            out.append(
                {
                    "pair": f"{row['a']} + {row['b']}",
                    "loss": round(float(row["final_loss"]), 3),
                    "params": round(float(row["total"]) / 1e6, 2),
                    "duration": round(float(row["duration_h"]), 2),
                    "top5": None if pd.isna(row.get("top5")) else round(float(row["top5"]) * 100, 1),
                }
            )
        return out

    return {
        "mixers": mixers,
        "cells": cells,
        "ranking": rank.round(4).to_dict(orient="records"),
        "top": pair_rows(top),
        "bottom": pair_rows(bottom),
        "roles": {
            "a": {mixer: round(float(by_a[mixer]), 4) for mixer in common},
            "b": {mixer: round(float(by_b[mixer]), 4) for mixer in common},
            "range_a": round(float(by_a[common].max() - by_a[common].min()), 3),
            "range_b": round(float(by_b[common].max() - by_b[common].min()), 3),
            "best_a": [str(by_a[common].idxmin()), round(float(by_a[common].min()), 3)],
            "best_b": [str(by_b[common].idxmin()), round(float(by_b[common].min()), 3)],
            "worst_a": [str(by_a[common].idxmax()), round(float(by_a[common].max()), 3)],
            "worst_b": [str(by_b[common].idxmax()), round(float(by_b[common].max()), 3)],
        },
        "meta": {
            "pairs": int(len(pairs)),
            "mixers": len(mixers),
            "with_inference": int(pairs["top5"].notna().sum()),
            "analysis_pairs": int(len(analysis)),
            "failed_pairs": int((pairs["status"] != "OK").sum()),
        },
    }


def write_data(atlas: dict) -> None:
    """Write ``atlas.json`` and the two downloadable CSVs."""
    EXP3_DATA.mkdir(parents=True, exist_ok=True)
    (EXP3_DATA / "atlas.json").write_text(
        json.dumps(atlas, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    mixers = atlas["mixers"]
    rows = []
    for i, j, cell in atlas["cells"]:
        rows.append(
            {
                "mixer_a": mixers[i],
                "mixer_b": mixers[j],
                "loss": cell["loss"],
                "params_m": cell["params"],
                "duration_h": cell["duration"],
                "top1": cell["top1"],
                "top5": cell["top5"],
                "status": cell["status"],
            }
        )
    pd.DataFrame(rows).to_csv(EXP3_DATA / "atlas.csv", index=False)
    pd.DataFrame(atlas["ranking"]).to_csv(EXP3_DATA / "mixers.csv", index=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def render_figures() -> None:
    """Render the 21 report figures in both site themes."""
    EXP3_IMG.mkdir(parents=True, exist_ok=True)
    mc.set_lang("en")
    for theme_name in ("dark", "light"):
        theme = mc.apply_theme(theme_name)
        for name, builder, _ in mc.FIGURES:
            figure = builder(theme)
            figure.savefig(
                EXP3_IMG / f"{name}-{theme_name}.png",
                dpi=150,
                bbox_inches="tight",
                facecolor=theme["face"],
            )
            mc.plt.close(figure)
        print(f"  {theme_name}: {len(mc.FIGURES)} figures")


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def site_style() -> str:
    """Return the ``<style>`` block of the Experiment 2 page."""
    text = EXP2_PAGE.read_text(encoding="utf-8")
    match = re.search(r"<style>.*?</style>", text, re.S)
    if not match:
        raise SystemExit(f"no <style> block in {EXP2_PAGE}")
    return match.group(0)


def site_script() -> str:
    """Return the theme/language toggle script of the Experiment 2 page."""
    text = EXP2_PAGE.read_text(encoding="utf-8")
    blocks = re.findall(r"<script>(.*?)</script>", text, re.S)
    for block in reversed(blocks):
        if "ftSetTheme" in block:
            return "<script>" + block + "</script>"
    raise SystemExit(f"no toggle script in {EXP2_PAGE}")


def T(english: str, spanish: str) -> str:
    """Bilingual inline span pair."""
    return f'<span class="t-en">{english}</span><span class="t-es">{spanish}</span>'


def p(english: str, spanish: str) -> str:
    """Bilingual paragraph."""
    return f"<p>{T(english, spanish)}</p>"


def figure(name: str, english: str, spanish: str) -> str:
    """Bilingual figure that follows the site theme."""
    return (
        f'<figure class="fig"><img data-dark="img/{name}-dark.png" '
        f'data-light="img/{name}-light.png" src="img/{name}-dark.png" '
        f'loading="lazy" alt="{name}"><figcaption>{T(english, spanish)}</figcaption></figure>'
    )


def figure_grid(figures: list[str]) -> str:
    """Two-column grid of figures on wide screens."""
    return '<div class="fig-grid">' + "".join(figures) + "</div>"


def table(headers: list[str], rows: list[list[str]], cls: str = "dt") -> str:
    """Simple table with bilingual headers."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def kpi(value: str, english: str, spanish: str, color: str) -> str:
    """One KPI card."""
    return f'<div class="kpi {color}"><div class="kpi-v">{value}</div><div class="kpi-l">{T(english, spanish)}</div></div>'


def render_page(stats: dict, atlas: dict) -> str:
    """Render the complete page."""
    pop = stats["population"]
    loss = stats["loss"]
    grad = stats["gradients"]
    quant = stats["quantization"]
    compute = stats["compute"]
    opt = stats["optimizers"]
    abl = stats["ablations"]
    atlas_st = stats["atlas"]
    syn = atlas_st["synergy"]
    roles = atlas["roles"]
    meta = atlas["meta"]

    def n(value):
        return f"{int(round(value)):,}"

    def d(value, digits=2):
        return f"{value:.{digits}f}"

    mixers = n(pop["n_mixers"])
    pairs = n(pop["n_pairs"])
    runs = n(pop["n_runs"])
    ablations = n(pop["n_ablations"])
    singles = n(pop["n_singles"])
    hours = n(pop["compute_h"])
    steps = f"{pop['steps_total'] / 1e6:.1f}"
    failure = d(100 * pop["training_failed"] / pop["n_runs"], 1)
    loss_median = d(loss["pair_median"])
    loss_min = d(loss["pair_min"])
    loss_max = d(loss["pair_max"])
    best = atlas["top"][0] if atlas["top"] else {"pair": "-", "loss": 0}
    worst = atlas["bottom"][0] if atlas["bottom"] else {"pair": "-", "loss": 0}
    top5_mean = d(stats["inference"]["top5_mean"] * 100, 1)
    top1_mean = d(stats["inference"]["top1_mean"] * 100, 2)
    q8 = quant["formats"]["hf_q8_0_v2"]
    q4 = quant["formats"]["hf_q4_0_v2"]
    i2 = quant["formats"]["hf_i2_s_v2"]
    agree = quant["agreement"]
    g = grad["blocks"]

    # ------------------------------------------------------------- figures
    figs = {
        "params_family": figure("params_family_box", "Parameter budget by family", "Presupuesto de parámetros por familia"),
        "params_composition": figure("params_composition", "Where the parameters live", "Dónde viven los parámetros"),
        "params_loss": figure("params_vs_loss", "Loss against size", "Pérdida frente a tamaño"),
        "loss_family": figure("loss_family_box", "Final loss by family", "Pérdida final por familia"),
        "curves": figure("curves_quantiles", "Population learning curves", "Curvas de aprendizaje de la población"),
        "top10": figure("curves_top10", "The ten best pairs over time", "Los diez mejores pares en el tiempo"),
        "convergence": figure("convergence_speed", "How early the ranking is decided", "Cuán temprano se decide el ranking"),
        "matrix": figure("matrix_loss", "The atlas in print: final loss of every pair", "El atlas impreso: pérdida final de cada par"),
        "roles": figure("roles_scatter", "The role effect: A vs B", "El efecto del rol: A vs B"),
        "topb": figure("top5_by_b", "Held-out quality by second mixer", "Calidad retenida por segundo mixer"),
        "synergy": figure("synergy", "Extreme synergies", "Sinergias extremas"),
        "pareto": figure("pareto", "The speed–quality Pareto front", "La frontera de Pareto velocidad–calidad"),
        "grad_groups": figure("grad_groups_curves", "Gradient norm per block", "Norma del gradiente por bloque"),
        "grad_heatmap": figure("grad_heatmap", "Gradient norm relative to the first epoch", "Gradiente relativo a la primera época"),
        "grad_health": figure("grad_health", "Clip ratio and gradient variability", "Recorte y variabilidad del gradiente"),
        "optimizer_panel": figure("optimizer_panel", "Optimizers: loss, gradients, CPU and memory", "Optimizadores: pérdida, gradientes, CPU y memoria"),
        "ablation_loss": figure("ablation_loss", "Component ablations on the shared backbone", "Ablaciones de componentes sobre el esqueleto compartido"),
        "time": figure("time_vs_params", "Time against parameter count", "Tiempo frente a parámetros"),
        "cost": figure("cost_per_mparam", "Minutes per million parameters", "Minutos por millón de parámetros"),
        "throughput": figure("throughput_family", "Throughput by family", "Rendimiento por familia"),
        "compute": figure("compute_by_family", "Where the compute went", "A dónde fue el cómputo"),
        "formats": figure("inference_format_accuracy", "Held-out accuracy by export format", "Exactitud por formato de exportación"),
        "quant_delta": figure("quant_delta_dist", "Quantization delta and outcome", "Delta de cuantización y resultado"),
        "quant_agree": figure("quant_agreement", "Sample-level agreement", "Acuerdo por muestra"),
        "quant_params": figure("quant_delta_vs_params", "2-bit delta against size", "Delta de 2 bits frente a tamaño"),
        "train_inf": figure("train_vs_inference", "Training loss against held-out quality", "Pérdida de entrenamiento frente a calidad"),
        "stage": figure("stage_status", "Pipeline reliability by stage", "Fiabilidad del pipeline por etapa"),
        "correlation": figure("correlation", "Spearman correlations", "Correlaciones de Spearman"),
    }

    opt_rows = [
        [
            f"<code>{row['name']}</code>",
            f"{row['epochs']:.0f}",
            f"{row['loss']:.3f}",
            f"{row['grad_mean']:.2f} / {row['grad_final']:.2f}",
            f"{row['clip']:.2f}",
            f"{row['step_ms']:.0f}",
            f"{row['rss_mb']:.0f}",
        ]
        for row in opt["rows"]
    ]
    optimizer_table = table(
        [
            T("Optimizer", "Optimizador"), "Ep.",
            T("Loss", "Pérdida"), T("Grad (m/f)", "Grad (m/f)"),
            T("Clip", "Recorte"), T("Step (ms)", "Paso (ms)"), T("RSS (MB)", "RSS (MB)"),
        ],
        opt_rows,
    )

    family_rows = []
    for row in compute["by_family"]:
        family_rows.append(
            [
                T(C.family_label(row["family"], "en"), C.family_label(row["family"], "es")),
                n(row["n"]),
                str(int(row["smoke"])),
                n(row["params_median"]),
                d(row["loss_median"], 2) if np.isfinite(row["loss_median"]) else "n/a",
                d(row["minutes_per_mparam"], 0),
            ]
        )
    family_table = table(
        [
            T("Family", "Familia"),
            T("Runs", "Corridas"),
            T("Smoke", "Humo"),
            T("Params", "Parámetros"),
            T("Final loss", "Pérdida final"),
            T("min/Mparam", "min/Mparám"),
        ],
        family_rows,
    )

    grad_rows = [
        [
            T(C.block_label(group, "en"), C.block_label(group, "es")),
            d(g[group]["mean_median"], 3),
            d(g[group]["final_median"], 3),
            d(g[group]["cv_median"], 2) if np.isfinite(g[group]["cv_median"]) else "n/a",
            d(g[group]["share_active"], 1) + "%",
        ]
        for group in C.BLOCK_GROUPS
        if group in g
    ]
    grad_table = table(
        [
            T("Block", "Bloque"),
            T("Mean", "Media"),
            T("Final", "Final"),
            T("CV", "CV"),
            T("Active", "Activo"),
        ],
        grad_rows,
    )

    format_rows = []
    for fmt, label in (
        ("hf_none", "none"),
        ("hf_q8_0_v1", "q8_0 v1"),
        ("hf_q8_0_v2", "q8_0 v2"),
        ("hf_q4_0_v1", "q4_0 v1"),
        ("hf_q4_0_v2", "q4_0 v2"),
        ("hf_i2_s_v1", "i2_s v1"),
        ("hf_i2_s_v2", "i2_s v2"),
    ):
        if fmt == "hf_none":
            format_rows.append(
                [
                    f"<code>{label}</code>",
                    n(quant["combos_per_format"].get(fmt, 0)),
                    d(quant["top1_by_format"][fmt], 2),
                    d(quant["top5_by_format"][fmt], 1),
                    "0.000",
                    "0.0%",
                ]
            )
            continue
        row = quant["formats"].get(fmt)
        if not row:
            continue
        format_rows.append(
            [
                f"<code>{label}</code>",
                n(row["n"]),
                d(quant["top1_by_format"][fmt], 2),
                d(quant["top5_by_format"][fmt], 1),
                f"{row['mean_delta_pp']:+.3f}",
                f"{row['pct_worse']:.1f}%",
            ]
        )
    format_table = table(
        [
            "Format", "n", "Top-1", "Top-5", T("Δ (pp)", "Δ (pp)"), T("Worse", "Peor"),
        ],
        format_rows,
    )

    reliability_rows = [
        [
            T(C.family_label(row["family"], "en"), C.family_label(row["family"], "es")),
            n(row["n"]),
            str(row["failed"]),
            f"{row['failure_rate']:.1f}%",
        ]
        for row in stats["reliability"]["by_family"]
        if row["n"] >= 3
    ]
    reliability_table = table(
        [T("Family", "Familia"), "n", T("Failed", "Fallos"), T("Rate", "Tasa")],
        reliability_rows,
    )

    # ------------------------------------------------------------- sections
    hero = f"""
<header class="hero">
  <div class="hero-inner">
    <div class="eyebrow">{T(f"Experiment 3 · {runs} runs · {pairs} hybrid pairs · {steps} M training steps",
                            f"Experimento 3 · {runs} corridas · {pairs} pares híbridos · {steps} M de pasos")}</div>
    <h1>{T("The complete mixer atlas", "El atlas completo de mixers")}</h1>
    <p class="lede">{T(
        "Every unordered pair of the 54 registered mixers, plus 168 ablations and 54 single-mixer baselines — trained on one CPU host, measured end to end and charted. The matrix is complete: 1,431 of 1,431 cells.",
        "Todos los pares no ordenados de los 54 mixers registrados, más 168 ablaciones y 54 líneas base de un solo mixer — entrenados en un solo host de CPU, medidos de extremo a extremo y graficados. La matriz está completa: 1.431 de 1.431 celdas.")}</p>
    <div class="hero-actions">
      <a class="btn btn-primary" href="#atlas">{T("Open the interactive atlas", "Abrir el atlas interactivo")} →</a>
      <a class="btn" href="#summary">{T("Read the findings", "Leer los hallazgos")}</a>
      <a class="btn" href="../2/index.html">{T("Experiment 2 · the first atlas", "Experimento 2 · el primer atlas")}</a>
      <a class="btn" href="data/atlas.csv" download>{T("Download the atlas (CSV)", "Descargar el atlas (CSV)")}</a>
    </div>
    <div class="hero-meta">
      <span>{T("Generated", "Generado")} 2026-11-21</span><span>·</span>
      <span>{T("Compute", "Cómputo")}: {hours} h</span><span>·</span>
      <span>{T("Calendar days", "Días calendario")}: {d(pop['campaign_days'], 1)}</span><span>·</span>
      <span>{T("Training failure rate", "Tasa de fallo")}: {failure}%</span>
    </div>
  </div>
</header>

<section class="kpis">
  {kpi(runs, "training runs", "corridas de entrenamiento", "blue")}
  {kpi(pairs, "hybrid pairs (54 × 53 / 2)", "pares híbridos (54 × 53 / 2)", "violet")}
  {kpi(ablations, "ablations and task switches", "ablaciones e interruptores de tarea", "green")}
  {kpi(steps + " M", "logged optimisation steps", "pasos de optimización registrados", "cyan")}
  {kpi(hours + " h", "of training wall time", "de tiempo de pared de entrenamiento", "orange")}
  {kpi(top5_mean + "%", "mean held-out top-5 (MLM)", "top-5 retenido medio (MLM)", "pink")}
</section>

<section id="summary" class="section">
<h2>{T("Executive summary", "Resumen ejecutivo")}</h2>
{p(
 f"Round 3 closes the matrix that Round 2 left open. It trains every unordered pair of the {mixers} registered mixers — C(54,2) = {pairs} cells, none missing — plus {singles} single-mixer baselines and {ablations} ablation runs that move one switch at a time: optimizer, normalization, positional encoding, Mixture-of-Experts routing, residual path, task and model class. All of it on one CPU host, in {d(pop['campaign_days'],0)} calendar days, for {hours} hours of training wall time.",
 f"La ronda 3 cierra la matriz que la ronda 2 dejó abierta. Entrena todos los pares no ordenados de los {mixers} mixers registrados — C(54,2) = {pairs} celdas, sin ninguna ausente — más {singles} líneas base de un solo mixer y {ablations} corridas de ablación que mueven un interruptor a la vez: optimizador, normalización, codificación posicional, enrutamiento MoE, camino residual, tarea y clase de modelo. Todo en un solo host de CPU, en {d(pop['campaign_days'],0)} días calendario, por {hours} horas de tiempo de pared de entrenamiento.")}
<div class="callout insight"><span class="callout-icon">◆</span><div><div class="t-en"><b>The headline:</b> the best pair is <code>{best['pair']}</code> at <b>{d(best['loss'],3)} nats</b>, and the worst is <code>{worst['pair']}</code> at <b>{d(worst['loss'],2)}</b> — a spread wider than any ablation moves the loss. The second mixer is the lever: the median loss spans <b>{roles['range_b']} nats</b> across B roles against <b>{roles['range_a']}</b> across A roles. At 8 and 4 bits quantization is free (sample agreement <b>{d(agree['q8_0']['mean'],1)}%</b> and <b>{d(agree['q4_0']['mean'],1)}%</b>), while 2-bit weights change the model (<b>{d(agree['i2_s']['mean'],1)}%</b>).</div><div class="t-es"><b>Lo principal:</b> el mejor par es <code>{best['pair']}</code> con <b>{d(best['loss'],3)} nats</b> y el peor es <code>{worst['pair']}</code> con <b>{d(worst['loss'],2)}</b> — una dispersión mayor que la de cualquier ablación. El segundo mixer es la palanca: la pérdida mediana abarca <b>{roles['range_b']} nats</b> entre roles B frente a <b>{roles['range_a']}</b> entre roles A. Con 8 y 4 bits la cuantización es gratis (acuerdo por muestra <b>{d(agree['q8_0']['mean'],1)} %</b> y <b>{d(agree['q4_0']['mean'],1)} %</b>), mientras que los pesos de 2 bits cambian el modelo (<b>{d(agree['i2_s']['mean'],1)} %</b>).</div></div></div>
<ul class="findings">
<li><b>{T("The grid is complete", "La rejilla está completa")}</b><br>{T(
 f"{pairs} of {pairs} cells trained. The ten <code>engram</code> pairs that crash are still cells of the atlas; they simply have no loss, which is why the matrix shows ten holes.",
 f"{pairs} de {pairs} celdas entrenadas. Los diez pares con <code>engram</code> que fallan siguen siendo celdas del atlas; simplemente no tienen pérdida, y por eso la matriz muestra diez huecos.")}</li>
<li><b>{T("The second mixer is the lever", "El segundo mixer es la palanca")}</b><br>{T(
 f"Across roles, the median loss spans {roles['range_b']} nats for B (best <code>{roles['best_b'][0]}</code> at {roles['best_b'][1]}, worst <code>{roles['worst_b'][0]}</code> at {roles['worst_b'][1]}) and {roles['range_a']} for A. The same asymmetry that Experiment 2 measured at 100 epochs survives at 500.",
 f"Entre roles, la pérdida mediana abarca {roles['range_b']} nats para B (mejor <code>{roles['best_b'][0]}</code> con {roles['best_b'][1]}, peor <code>{roles['worst_b'][0]}</code> con {roles['worst_b'][1]}) y {roles['range_a']} para A. La misma asimetría que el Experimento 2 midió a 100 épocas sobrevive a 500.")}</li>
<li><b>{T("Size helps but does not decide", "El tamaño ayuda pero no decide")}</b><br>{T(
 f"Parameter count correlates with final loss at ρ = {stats['params']['loss_rho']:+.2f} and with held-out quality at ρ = {stats['inference']['loss_top5_rho']:+.2f}. The largest run ({d(stats['params']['max'] / 1e6,2)} M parameters) is not the best, and the cheapest mixers sit on both sides of the median.",
 f"El número de parámetros correlaciona con la pérdida final en ρ = {stats['params']['loss_rho']:+.2f} y con la calidad retenida en ρ = {stats['inference']['loss_top5_rho']:+.2f}. La corrida más grande ({d(stats['params']['max'] / 1e6,2)} M de parámetros) no es la mejor, y los mixers más baratos se reparten a ambos lados de la mediana.")}</li>
<li><b>{T("Learning is fast; the ranking is decided early", "El aprendizaje es rápido; el ranking se decide temprano")}</b><br>{T(
 f"The median pair reaches 90% of its loss drop at {d(stats['convergence']['overall_median'],0)}% of the schedule, and the per-pair loss at a quarter of the schedule already ranks the pairs with ρ = {d(stats['dynamics']['rank_stability_25'],2)} against the final order. Every curve is still descending when the schedule ends.",
 f"El par mediano alcanza el 90 % de su caída de pérdida al {d(stats['convergence']['overall_median'],0)} % del calendario, y la pérdida por par a un cuarto del calendario ya ordena los pares con ρ = {d(stats['dynamics']['rank_stability_25'],2)} frente al orden final. Todas las curvas siguen bajando cuando termina el calendario.")}</li>
<li><b>{T("Gradient pressure is block-specific", "La presión del gradiente es específica por bloque")}</b><br>{T(
 f"The head carries the largest gradient ({d(g['head']['mean_median'],2)}), then attention ({d(g['attention']['mean_median'],2)}) and the FFN ({d(g['ffn']['mean_median'],2)}); embeddings and normalization are an order of magnitude smaller. The active blocks keep a coefficient of variation below 0.35, and clipping is routine: the median clip ratio is {d(grad['clip_median'],2)}.",
 f"La cabeza soporta el gradiente mayor ({d(g['head']['mean_median'],2)}), luego la atención ({d(g['attention']['mean_median'],2)}) y la FFN ({d(g['ffn']['mean_median'],2)}); los embeddings y la normalización son un orden de magnitud menores. Los bloques activos mantienen un coeficiente de variación por debajo de 0,35, y el recorte es rutina: la mediana del cociente es {d(grad['clip_median'],2)}.")}</li>
<li><b>{T("Optimizers buy speed with gradient, not preconditioning", "Los optimizadores compran velocidad con gradiente, no con precondicionamiento")}</b><br>{T(
 f"Across the 39 optimizers, final loss and final gradient norm correlate at {opt['rho_loss_grad_final']:+.2f}: <code>af_muon</code> ends at 5.685 nats with the largest final gradient, while eight AdamW-family optimizers are bit-identical. Quality is orthogonal to step time ({opt['rho_loss_step']:+.2f}) and to memory (median {n(opt['rss_median'])} MB; only <code>eve</code> pays extra state).",
 f"Entre los 39 optimizadores, la pérdida final y la norma final del gradiente correlacionan en {opt['rho_loss_grad_final']:+.2f}: <code>af_muon</code> termina en 5.685 nats con el mayor gradiente final, mientras que ocho optimizadores de la familia AdamW son idénticos bit a bit. La calidad es ortogonal al tiempo por paso ({opt['rho_loss_step']:+.2f}) y a la memoria (mediana {n(opt['rss_median'])} MB; solo <code>eve</code> paga estado extra).")}</li>
<li><b>{T("Quantization: 8 and 4 bits are free, 2 bits is not", "Cuantización: 8 y 4 bits son gratis, 2 bits no")}</b><br>{T(
 f"Against the unquantized export, <code>q8_0_v2</code> changes the top-1 by {q8['mean_delta_pp']:+.2f} pp and leaves {d(q8['pct_equal'],1)}% of the pairs unchanged; <code>q4_0_v2</code> changes it by {q4['mean_delta_pp']:+.2f} pp with {d(q4['pct_equal'],1)}% unchanged; <code>i2_s_v2</code> loses {d(abs(i2['mean_delta_pp']),2)} pp and degrades {d(i2['pct_worse'],1)}% of the pairs.",
 f"Frente a la exportación sin cuantizar, <code>q8_0_v2</code> cambia el top-1 en {q8['mean_delta_pp']:+.2f} pp y deja sin cambios el {d(q8['pct_equal'],1)} % de los pares; <code>q4_0_v2</code> lo cambia en {q4['mean_delta_pp']:+.2f} pp con el {d(q4['pct_equal'],1)} % sin cambios; <code>i2_s_v2</code> pierde {d(abs(i2['mean_delta_pp']),2)} pp y degrada el {d(i2['pct_worse'],1)} % de los pares.")}</li>
</ul>
</section>

<section id="method" class="section">
<h2>{T("What was run, and how to read it", "Qué se corrió y cómo leerlo")}</h2>
{p(
 f"Experiment 3 is the same harness as Experiment 2 with the grid completed and the schedule extended to 500 epochs. Every configuration asks for 500 epochs, but the harness stops runs earlier — by schedule, plateau criterion or an ablation-specific cap: {n(pop['n_runs'] - pop['early_stop'])} runs reached the end and {n(pop['early_stop'])} stopped sooner. The 44 runs that stopped before epoch 100 (fifteen five-epoch MoE routing tests, the BitNet/GGUF conversion, five switch/PE smokes, one audio transducer run and twenty-two hybrid pairs — twelve truncated, ten the <code>engram</code> crashes) are excluded from every loss, dynamics, gradient and correlation comparison; they stay in the population and the compute ledger.",
 f"El Experimento 3 es el mismo arnés que el Experimento 2 con la rejilla completa y el calendario extendido a 500 épocas. Todas las configuraciones piden 500 épocas, pero el arnés detiene corridas antes — por calendario, criterio de meseta o un tope específico de la ablación: {n(pop['n_runs'] - pop['early_stop'])} corridas llegaron al final y {n(pop['early_stop'])} se detuvieron antes. Las 44 corridas que se detuvieron antes de la época 100 (quince pruebas de enrutamiento MoE de cinco épocas, la conversión BitNet/GGUF, cinco pruebas de humo de interruptores y PE, una corrida de transductor de audio y veintidós pares híbridos —doce truncados, diez los fallos de <code>engram</code>—) quedan excluidas de toda comparación de pérdida, dinámica, gradiente y correlación; permanecen en la población y en el registro de cómputo.")}
{p(
 "<b>Deployment is excluded.</b> During the campaign the deploy stage's configuration-payload handling was affected by a bug that has since been patched in the harness, so this round's deploy records are set aside; the quantized analysis uses the Transformers-export path, which exported 4,686 models with no failures.",
 "<b>El despliegue queda excluido.</b> Durante la campaña, el manejo del payload de configuración de la etapa de despliegue se vio afectado por un error que ya está parchado en el harness, así que los registros de despliegue de esta ronda se dejan de lado; el análisis cuantizado usa la ruta de exportación a Transformers, que exportó 4.686 modelos sin fallos.")}
{p(
 f"The population by family is below. <b>Smoke</b> counts the runs shorter than 100 epochs; the loss and cost columns are computed over the remaining {n(pop['n_analysis'])} runs. The tokenizer, task and real-data families run different tasks, so their losses are not comparable with MLM.",
 f"La población por familia está abajo. <b>Humo</b> cuenta las corridas de menos de 100 épocas; las columnas de pérdida y coste se calculan sobre las {n(pop['n_analysis'])} corridas restantes. Las familias de tokenizadores, tareas y datos reales corren tareas distintas, así que sus pérdidas no son comparables con MLM.")}
{family_table}
<div class="callout warn"><span class="callout-icon">!</span><div><div class="t-en"><b>Data-quality notes.</b> The trainer writes a zero-loss sentinel row when the final batch cannot be formed; those rows are dropped before aggregation, so the final loss is the last <em>valid</em> epoch. Fifteen of the seventeen MoE ablations are five-epoch routing smoke tests and are not evidence about expert capacity. Nine mixers appear in only one role after the smoke filter and are excluded from the A/B comparison. And the atlas measures fewer distinct models than it names: 297 of the 1,409 analysed pairs share a bit-identical loss with another pair, because <code>standard</code>, <code>bigbird</code>, <code>longformer</code> and <code>titan</code> collapse into one model at a 64-token context and <code>fastconformer</code> equals <code>swa</code> wherever both were paired.</div><div class="t-es"><b>Notas de calidad de datos.</b> El entrenador escribe una fila centinela con pérdida cero cuando no puede formar el último lote; esas filas se descartan antes de agregar, así que la pérdida final es la última época <em>válida</em>. Quince de las diecisiete ablaciones MoE son pruebas de humo de enrutamiento de cinco épocas y no son evidencia sobre la capacidad de los expertos. Nueve mixers aparecen en un solo rol tras el filtro de humo y quedan fuera de la comparación A/B. Y el atlas mide menos modelos distintos de los que nombra: 297 de los 1.409 pares analizados comparten una pérdida idéntica bit a bit con otro par, porque <code>standard</code>, <code>bigbird</code>, <code>longformer</code> y <code>titan</code> colapsan en un solo modelo en un contexto de 64 tokens y <code>fastconformer</code> equivale a <code>swa</code> allí donde ambos fueron emparejados.</div></div></div>
</section>

<section id="params" class="section">
<h2>{T("1 · The parameter landscape", "1 · El paisaje de parámetros")}</h2>
{p(
 f"The {runs} runs span {n(stats['params']['min'])} to {n(stats['params']['max'])} parameters, with a median of {n(stats['params']['median'])}. Embeddings and the output head are fixed costs of roughly 31% each; attention takes 25% and the FFN 12%, so the architectural choice moves a quarter of the budget. The cheapest second mixers are <code>mamba</code>, <code>ccgqa</code> and <code>cca</code> (around 0.29 M); the most expensive are <code>gmla</code> (1.67 M), <code>engram</code> (0.60 M) and <code>mfa</code> (0.54 M).",
 f"Las {runs} corridas abarcan de {n(stats['params']['min'])} a {n(stats['params']['max'])} parámetros, con una mediana de {n(stats['params']['median'])}. Los embeddings y la cabeza de salida son costes fijos de aproximadamente 31 % cada uno; la atención toma 25 % y la FFN 12 %, así que la decisión arquitectónica mueve un cuarto del presupuesto. Los segundos mixers más baratos son <code>mamba</code>, <code>ccgqa</code> y <code>cca</code> (unos 0,29 M); los más caros son <code>gmla</code> (1,67 M), <code>engram</code> (0,60 M) y <code>mfa</code> (0,54 M).")}
{figure_grid([figs['params_family'], figs['params_composition']])}
{figure_grid([figs['params_loss']])}
{p(
 f"Size helps, but it explains only a sixth of the ranking (ρ = {stats['params']['loss_rho']:+.2f} over {pairs} pairs). Within the same budget, architecture decides more than scale — the same conclusion as Round 2, now measured on the complete grid.",
 f"El tamaño ayuda, pero explica solo una sexta parte del orden (ρ = {stats['params']['loss_rho']:+.2f} sobre {pairs} pares). Con el mismo presupuesto, la arquitectura decide más que la escala — la misma conclusión de la ronda 2, ahora medida sobre la rejilla completa.")}
</section>

<section id="outcomes" class="section">
<h2>{T("2 · Training outcomes", "2 · Resultados de entrenamiento")}</h2>
{p(
 f"The hybrid pairs end at a median final loss of {loss_median} nats, from {loss_min} to {loss_max}. The ten best pairs are listed below; the difference between the best and the worst pair is larger than the difference between the best and the median optimizer, normalization or positional-encoding ablation. The mixer stack is the dominant design decision of the round.",
 f"Los pares híbridos terminan con una pérdida final mediana de {loss_median} nats, de {loss_min} a {loss_max}. Los diez mejores pares están abajo; la diferencia entre el mejor y el peor par es mayor que la diferencia entre la mejor ablación de optimizador, normalización o codificación posicional y la mediana de su familia. La pila de mixers es la decisión de diseño dominante de la ronda.")}
{figure_grid([figs['loss_family']])}
{p(
 f"Held-out quality comes from the export path: {n(stats['inference']['n_combos'])} configurations produced Hugging Face rows, {n(stats['inference']['n_pairs_combos'])} of them MLM pairs. The mean top-1 is {top1_mean}% and the mean top-5 is {top5_mean}%, with a median top-5 of 28.6%: a few pairs learned something, most did not. The rank correlation between final training loss and held-out top-5 is only {stats['inference']['loss_top5_rho']:+.2f}, so training loss is not a leaderboard.",
 f"La calidad retenida viene de la ruta de exportación: {n(stats['inference']['n_combos'])} configuraciones produjeron filas de Hugging Face, {n(stats['inference']['n_pairs_combos'])} de ellas pares MLM. El top-1 medio es {top1_mean} % y el top-5 medio es {top5_mean} %, con una mediana de top-5 de 28,6 %: unos pocos pares aprendieron algo, la mayoría no. La correlación de rango entre pérdida final de entrenamiento y top-5 retenido es solo {stats['inference']['loss_top5_rho']:+.2f}, así que la pérdida de entrenamiento no es una tabla de líderes.")}
<p>{T(
 f"A second mixer is worth a fifth of a nat: the 54 single-mixer baselines (each layer running the same mixer twice) end at a median of {d(atlas_st['singles']['median'], 3)} nats against {loss_median} for the pairs, and {n(atlas_st['singles']['pairs_better_than_best_single'])} of the 1,409 analysed pairs end below the best single, the doubled <code>conformer</code> at {d(atlas_st['singles']['best'][1], 3)}. Mixing two different mixers beats doubling one at the median.",
 f"Un segundo mixer vale un quinto de nat: las 54 líneas base de un solo mixer (cada capa con el mismo mixer dos veces) terminan con una mediana de {d(atlas_st['singles']['median'], 3)} nats frente a {loss_median} de los pares, y {n(atlas_st['singles']['pairs_better_than_best_single'])} de los 1.409 pares analizados terminan por debajo de la mejor línea base, el <code>conformer</code> duplicado con {d(atlas_st['singles']['best'][1], 3)}. Mezclar dos mixers distintos supera a duplicar uno en la mediana.")}</p>
</section>

<section id="dynamics" class="section">
<h2>{T("3 · Learning dynamics", "3 · Dinámica de aprendizaje")}</h2>
{p(
 f"The median pair starts at {d(stats['dynamics']['p50']['start'],2)} nats and ends at {d(stats['dynamics']['p50']['end'],2)}; the P10–P90 band at the end is {d(stats['dynamics']['band']['p10_end'],2)}–{d(stats['dynamics']['band']['p90_end'],2)}. Most of the reduction happens in the first quarter: {d(stats['dynamics']['p50']['q25'],2)} at 25% of the schedule, {d(stats['dynamics']['p50']['q50'],2)} at half, {d(stats['dynamics']['p50']['q75'],2)} at three quarters. The curves are still descending when the schedule ends, so the round measures learning speed, not converged quality.",
 f"El par mediano empieza en {d(stats['dynamics']['p50']['start'],2)} nats y termina en {d(stats['dynamics']['p50']['end'],2)}; la banda P10–P90 al final es {d(stats['dynamics']['band']['p10_end'],2)}–{d(stats['dynamics']['band']['p90_end'],2)}. La mayor parte de la reducción ocurre en el primer cuarto: {d(stats['dynamics']['p50']['q25'],2)} al 25 % del calendario, {d(stats['dynamics']['p50']['q50'],2)} a la mitad, {d(stats['dynamics']['p50']['q75'],2)} a las tres cuartas partes. Las curvas siguen bajando al terminar el calendario, así que la ronda mide velocidad de aprendizaje, no calidad convergida.")}
{figure_grid([figs['curves'], figs['top10']])}
{p(
 f"By a tenth of the schedule the per-pair loss already ranks the pairs with ρ = {d(stats['dynamics']['rank_stability_10'],2)}, and by a quarter the correlation reaches {d(stats['dynamics']['rank_stability_25'],2)} against the final order — the ten best separate from the population before the halfway mark. The number of epochs needed to reach 90% of a run's loss drop confirms it: the median is {d(stats['convergence']['overall_median'],0)}% of the schedule, from {d(stats['convergence']['by_family']['pair']['median'],0)}% for hybrid pairs to {d(stats['convergence']['by_family']['vision']['median'],0)}% for vision tasks.",
 f"A la décima parte del calendario la pérdida por par ya ordena los pares con ρ = {d(stats['dynamics']['rank_stability_10'],2)}, y al cuarto la correlación alcanza {d(stats['dynamics']['rank_stability_25'],2)} frente al orden final — los diez mejores se separan de la población antes de la mitad. El número de épocas necesarias para alcanzar el 90 % de la caída de pérdida lo confirma: la mediana es el {d(stats['convergence']['overall_median'],0)} % del calendario, desde el {d(stats['convergence']['by_family']['pair']['median'],0)} % de los pares híbridos hasta el {d(stats['convergence']['by_family']['vision']['median'],0)} % de las tareas de visión.")}
{figure_grid([figs['convergence']])}
</section>

<section id="gradients" class="section">
<h2>{T("4 · Gradient pressure per block", "4 · Presión del gradiente por bloque")}</h2>
{p(
 "The trainer logs the post-clip L2 norm of every parameter block: embeddings, positional encoding, attention, FFN, normalization, head, mHC, router, residual and other. The head carries the largest gradient, then attention and the FFN; embeddings and normalization are an order of magnitude smaller. The ordering is stable across the whole campaign, which is why the per-block optimizer groups are not cosmetic.",
 "El entrenador registra la norma L2 post-recorte de cada bloque de parámetros: embeddings, codificación posicional, atención, FFN, normalización, cabeza, mHC, router, residual y otros. La cabeza soporta el gradiente mayor, luego la atención y la FFN; los embeddings y la normalización son un orden de magnitud menores. El orden es estable en toda la campaña, y por eso los grupos por bloque del optimizador no son cosméticos.")}
{figure_grid([figs['grad_groups'], figs['grad_heatmap']])}
{grad_table}
{p(
 f"The heatmap tells the same story over time: the head is flat, attention grows slightly, and the FFN decays by a factor of about two over the first half before stabilizing; embeddings decay as the vocabulary statistics are learned. Nothing explodes: no NaN steps in the whole campaign, {grad['inf_steps']} Inf steps (all repaired in place, in {grad['runs_with_repair']} runs), and a gradient coefficient of variation that is inversely related to the final loss ({grad['grad_cv_loss_rho']:+.2f}).",
 f"El mapa de calor cuenta la misma historia en el tiempo: la cabeza es plana, la atención crece ligeramente y la FFN decae por un factor de aproximadamente dos durante la primera mitad antes de estabilizarse; los embeddings decaen a medida que se aprenden las estadísticas del vocabulario. Nada explota: ningún paso con NaN en toda la campaña, {grad['inf_steps']} pasos con Inf (todos reparados en el sitio, en {grad['runs_with_repair']} corridas) y un coeficiente de variación del gradiente inversamente relacionado con la pérdida final ({grad['grad_cv_loss_rho']:+.2f}).")}
{figure_grid([figs['grad_health']])}
<p>{T(
 f"Clipping is routine, not exceptional: the median mean clip ratio is {d(grad['clip_median'],2)} — just above the threshold of 1 — and {d(grad['clip_share_above_1'],1)}% of the runs clip on average, with a P90 of {d(grad['clip_p90'],2)}. Round 2 reported a median of 0.90 and clipping as a rare event; the difference is the longer schedule and the higher learning rates of Round 3.",
 f"El recorte es rutina, no excepción: la mediana del cociente de recorte medio es {d(grad['clip_median'],2)} — justo por encima del umbral 1 — y el {d(grad['clip_share_above_1'],1)} % de las corridas recortan de media, con un P90 de {d(grad['clip_p90'],2)}. La ronda 2 reportó una mediana de 0,90 y un recorte excepcional; la diferencia es el calendario más largo y las tasas de aprendizaje más altas de la ronda 3.")}</p>
</section>

<section id="optimizers" class="section">
<h2>{T("5 · Optimizers and component ablations", "5 · Optimizadores y ablaciones de componentes")}</h2>
{p(
 f"The optimizer sweep runs {opt['n']} optimizers on the same 309k-parameter backbone, the same data and the same learning rate. Eight of them (<code>adamw</code>, <code>adan</code>, <code>adopt</code>, <code>soap</code>, <code>ademamix</code>, <code>mars_adamw</code>, <code>schedulefree_adamw</code>, <code>cautious_adamw</code>) produce trajectories that agree to within 2×10⁻⁶ nats — they are the same update rule at these hyperparameters — so the ablation really measures the remaining 31.",
 f"El barrido de optimizadores corre {opt['n']} optimizadores sobre el mismo esqueleto de 309k parámetros, los mismos datos y la misma tasa de aprendizaje. Ocho de ellos (<code>adamw</code>, <code>adan</code>, <code>adopt</code>, <code>soap</code>, <code>ademamix</code>, <code>mars_adamw</code>, <code>schedulefree_adamw</code>, <code>cautious_adamw</code>) producen trayectorias que coinciden dentro de 2×10⁻⁶ nats — son la misma regla de actualización con estos hiperparámetros — así que la ablación mide realmente los 31 restantes.")}
<p>{T(
 f"The loss ranking is a gradient-pressure ranking: the rank correlation between final loss and final gradient norm across the 39 runs is {opt['rho_loss_grad_final']:+.2f} (against the clip ratio, {opt['rho_loss_clip']:+.2f}). The optimizers that end lower are the ones whose gradients stay large and keep clipping: <code>{opt['rows'][0]['name']}</code> reaches {opt['rows'][0]['loss']:.3f} nats with the largest final gradient of the sweep ({opt['rows'][0]['grad_final']:.2f}), <code>{opt['rows'][1]['name']}</code> reaches {opt['rows'][1]['loss']:.3f} with {opt['rows'][1]['grad_final']:.2f}, while <code>sgd_momentum</code> ends at 10.206 with 4.08. Learning speed at this scale is bought with gradient magnitude, not with a fancier preconditioner.",
 f"El ranking de pérdida es un ranking de presión de gradiente: la correlación de rango entre pérdida final y norma final del gradiente sobre las 39 corridas es {opt['rho_loss_grad_final']:+.2f} (contra el cociente de recorte, {opt['rho_loss_clip']:+.2f}). Los optimizadores que terminan más abajo son los que mantienen gradientes grandes y siguen recortando: <code>{opt['rows'][0]['name']}</code> llega a {opt['rows'][0]['loss']:.3f} nats con el mayor gradiente final del barrido ({opt['rows'][0]['grad_final']:.2f}), <code>{opt['rows'][1]['name']}</code> llega a {opt['rows'][1]['loss']:.3f} con {opt['rows'][1]['grad_final']:.2f}, mientras que <code>sgd_momentum</code> termina en 10.206 con 4.08. La velocidad de aprendizaje a esta escala se compra con magnitud de gradiente, no con un precondicionador más sofisticado.")}</p>
<p>{T(
 f"Quality and CPU cost are orthogonal (ρ = {opt['rho_loss_step']:+.2f} between loss and step time). Among the 36 runs with comparable telemetry the step time spans 305–438 ms per step, about 40%, and memory is flatter still: the median optimizer holds {n(opt['rss_median'])} MB of resident memory and only <code>{opt['rss_max'][0]}</code> stands out at {n(opt['rss_max'][1])} MB. The three late additions (<code>af_muon</code>, <code>ada_muon</code>, <code>ea4llm</code>) log step timers that cover only part of the work (111–121 ms against 305–438 for the rest, with longer wall-clock schedules), so their step-time column is not comparable; their losses and gradient norms are unaffected.",
 f"La calidad y el coste de CPU son ortogonales (ρ = {opt['rho_loss_step']:+.2f} entre pérdida y tiempo por paso). Entre las 36 corridas con telemetría comparable, el tiempo por paso abarca 305–438 ms, un 40 %, y la memoria es aún más plana: el optimizador mediano mantiene {n(opt['rss_median'])} MB de memoria residente y solo <code>{opt['rss_max'][0]}</code> destaca con {n(opt['rss_max'][1])} MB. Las tres incorporaciones tardías (<code>af_muon</code>, <code>ada_muon</code>, <code>ea4llm</code>) registran cronómetros que cubren solo parte del trabajo (111–121 ms frente a 305–438 del resto, con calendarios de pared más largos), así que su columna de tiempo por paso no es comparable; sus pérdidas y normas de gradiente no se ven afectadas.")}</p>
{optimizer_table}
{figure_grid([figs['optimizer_panel']])}
<p>{T(
 f"The component ablations separate more than the optimizers. The normalization ranking flips relative to Experiment 2: <code>prms_norm</code> wins at 8.340 nats — half a nat ahead of <code>layer_norm</code> — while <code>flash_norm</code>, the clear winner at 100 epochs, closes the table at 9.135 (284 effective epochs). The PAPE family leads the positional encodings as before (<code>pape_ri</code> at 8.696, <code>pape</code>/<code>pape_efficient</code> at 8.764 as bit-identical twins, classical <code>rope</code> exactly at the baseline). Factorised embeddings win the structural switches at 8.655, the fixed-softmax toggles are no-ops on this backbone, the residual variants all cost loss, and the BitNet pair needs the full 499 epochs to land at 9.563. The ViT positional-encoding sweep is a clean null result: all 39 <code>vit_pe</code> runs (13 encodings × 3 tasks) reproduce one loss per task to the sixth decimal — at a 4×4 patch grid the positional encoding is invisible.",
 f"Las ablaciones de componentes se separan más que los optimizadores. El ranking de normalización se invierte respecto del Experimento 2: <code>prms_norm</code> gana con 8.340 nats —medio nat por delante de <code>layer_norm</code>— mientras que <code>flash_norm</code>, el ganador claro a 100 épocas, cierra la tabla con 9.135 (284 épocas efectivas). La familia PAPE lidera las codificaciones posicionales como antes (<code>pape_ri</code> con 8.696, <code>pape</code>/<code>pape_efficient</code> con 8.764 como gemelos idénticos bit a bit, la <code>rope</code> clásica exactamente en la línea base). Los embeddings factorizados ganan los interruptores estructurales con 8.655, los interruptores de softmax fijada son no-ops sobre este esqueleto, las variantes residuales cuestan pérdida y el par BitNet necesita las 499 épocas completas para aterrizar en 9.563. El barrido de codificaciones posicionales del ViT es un nulo limpio: las 39 corridas <code>vit_pe</code> (13 codificaciones × 3 tareas) reproducen una pérdida por tarea hasta el sexto decimal — en una rejilla de parches de 4×4 la codificación posicional es invisible.")}</p>
{figure_grid([figs['ablation_loss']])}
</section>

<section id="cost" class="section">
<h2>{T("6 · Throughput and cost", "6 · Rendimiento y coste")}</h2>
{p(
 f"Training time grows only weakly with parameters: a log-log fit over the MLM pairs gives an exponent of {d(compute['time_slope'],2)} with R² = {d(compute['time_r2'],2)}. The clock is dominated by the number of steps, not by the parameter count. The normalised cost is the fair comparison: the median run spends {n(compute['minutes_per_mparam_median'])} minutes per million parameters, with family medians spanning a factor of about three.",
 f"El tiempo de entrenamiento crece solo débilmente con los parámetros: un ajuste log-log sobre los pares MLM da un exponente de {d(compute['time_slope'],2)} con R² = {d(compute['time_r2'],2)}. El reloj lo domina el número de pasos, no el de parámetros. El coste normalizado es la comparación justa: la corrida mediana gasta {n(compute['minutes_per_mparam_median'])} minutos por millón de parámetros, con medianas por familia que abarcan un factor de aproximadamente tres.")}
{figure_grid([figs['time'], figs['cost']])}
{figure_grid([figs['throughput'], figs['compute']])}
{p(
 f"The hybrid pairs absorb {d(compute['by_family'][0]['share'],1)}% of the {hours} hours ({n(compute['by_family'][0]['hours'])} h); the single-mixer baselines take {d(compute['by_family'][1]['share'],1)}% and every other family less than 3%. Vision and audio are cheap in wall time despite their long schedules because their steps are short, and the throughput panel says the same in tokens per second: the MLM pairs run at a median of {n(compute['tokens_per_sec_median'])} tokens/s.",
 f"Los pares híbridos absorben el {d(compute['by_family'][0]['share'],1)} % de las {hours} horas ({n(compute['by_family'][0]['hours'])} h); las líneas base de un solo mixer toman el {d(compute['by_family'][1]['share'],1)} % y todas las demás familias menos del 3 %. Visión y audio son baratas en tiempo de pared pese a sus calendarios largos porque sus pasos son cortos, y el panel de rendimiento dice lo mismo en tokens por segundo: los pares MLM corren a una mediana de {n(compute['tokens_per_sec_median'])} tokens/s.")}
</section>

<section id="inference" class="section">
<h2>{T("7 · Inference and quantization", "7 · Inferencia y cuantización")}</h2>
{p(
 f"{n(stats['inference']['n_combos'])} configurations produced Hugging Face rows: {n(stats['inference']['n_pairs_combos'])} MLM pairs, four causal-LM runs and one text-classification run. Each was exported unquantized (<code>hf_none</code>) and re-exported with three codecs at two loader runtimes (<code>q8_0</code>, <code>q4_0</code>, <code>i2_s</code>, each in <code>v1</code> and <code>v2</code>). The two runtimes sampled different mask positions — only 3.5% of the <code>v1</code> keys coincide with the baseline — so <code>v1</code> cannot be paired; the clean comparison is the <code>v2</code> family, which reuses the baseline masks for 99.4% of the keys. This is a harness property, not a model property.",
 f"{n(stats['inference']['n_combos'])} configuraciones produjeron filas de Hugging Face: {n(stats['inference']['n_pairs_combos'])} pares MLM, cuatro corridas de LM causal y una de clasificación de texto. Cada una se exportó sin cuantizar (<code>hf_none</code>) y se reexportó con tres códecs en dos runtimes de carga (<code>q8_0</code>, <code>q4_0</code>, <code>i2_s</code>, cada uno en <code>v1</code> y <code>v2</code>). Los dos runtimes muestrearon posiciones de máscara distintas — solo el 3,5 % de las claves de <code>v1</code> coinciden con la base — así que <code>v1</code> no puede emparejarse; la comparación limpia es la familia <code>v2</code>, que reutiliza las máscaras de la base en el 99,4 % de las claves. Esto es una propiedad del arnés, no de los modelos.")}
{figure_grid([figs['formats']])}
{format_table}
{p(
 f"8 and 4 bits are essentially free: <code>q8_0_v2</code> changes the top-1 by {q8['mean_delta_pp']:+.2f} pp with {d(q8['pct_equal'],1)}% of the pairs unchanged and only {d(q8['pct_worse'],1)}% worse; <code>q4_0_v2</code> changes it by {q4['mean_delta_pp']:+.2f} pp with {d(q4['pct_equal'],1)}% unchanged and {d(q4['pct_worse'],1)}% worse. The 2-bit codec is the exception: {i2['mean_delta_pp']:+.2f} pp on average, {d(i2['pct_equal'],1)}% of the pairs intact and {d(i2['pct_worse'],1)}% degraded.",
 f"8 y 4 bits son esencialmente gratis: <code>q8_0_v2</code> cambia el top-1 en {q8['mean_delta_pp']:+.2f} pp con el {d(q8['pct_equal'],1)} % de los pares sin cambios y solo el {d(q8['pct_worse'],1)} % peor; <code>q4_0_v2</code> lo cambia en {q4['mean_delta_pp']:+.2f} pp con el {d(q4['pct_equal'],1)} % sin cambios y el {d(q4['pct_worse'],1)} % peor. El códec de 2 bits es la excepción: {i2['mean_delta_pp']:+.2f} pp de media, {d(i2['pct_equal'],1)} % de los pares intactos y {d(i2['pct_worse'],1)} % degradados.")}
{figure_grid([figs['quant_delta'], figs['quant_agree']])}
{p(
 f"The sample-level view is sharper. Matching each quantized prediction to the baseline prediction of the same mask, <code>q8_0_v2</code> agrees on {d(agree['q8_0']['mean'],1)}% of the samples (95% CI {d(agree['q8_0']['ci_low'],1)}–{d(agree['q8_0']['ci_high'],1)}), <code>q4_0_v2</code> on {d(agree['q4_0']['mean'],1)}% and <code>i2_s_v2</code> on only {d(agree['i2_s']['mean'],1)}%. The 2-bit model is not a slightly worse copy: it is a different model that happens to preserve the aggregate top-1 on pairs whose baseline accuracy was already near zero. Parameter count does not predict robustness (ρ = {quant['i2_delta_params_rho']:+.2f}).",
 f"La vista por muestra es más nítida. Emparejando cada predicción cuantizada con la predicción base de la misma máscara, <code>q8_0_v2</code> coincide en el {d(agree['q8_0']['mean'],1)} % de las muestras (IC 95 % {d(agree['q8_0']['ci_low'],1)}–{d(agree['q8_0']['ci_high'],1)}), <code>q4_0_v2</code> en el {d(agree['q4_0']['mean'],1)} % y <code>i2_s_v2</code> en solo el {d(agree['i2_s']['mean'],1)} %. El modelo de 2 bits no es una copia algo peor: es un modelo distinto que casualmente conserva el top-1 agregado en pares cuya exactitud base ya era casi cero. El número de parámetros no predice la robustez (ρ = {quant['i2_delta_params_rho']:+.2f}).")}
{figure_grid([figs['quant_params'], figs['train_inf']])}
<p>{T(
 f"The warning of Round 2 survives the complete grid: training loss and held-out quality are almost uncorrelated (ρ = {stats['inference']['loss_top5_rho']:+.2f}). Read the atlas as a map of learning speed and stability, not as a leaderboard.",
 f"La advertencia de la ronda 2 sobrevive a la rejilla completa: pérdida de entrenamiento y calidad retenida están casi incorrelacionadas (ρ = {stats['inference']['loss_top5_rho']:+.2f}). Lee el atlas como un mapa de velocidad y estabilidad de aprendizaje, no como una tabla de líderes.")}</p>
</section>

<section id="reliability" class="section">
<h2>{T("8 · Reliability and failure taxonomy", "8 · Fiabilidad y taxonomía de fallos")}</h2>
{p(
 f"Training is reliable: {n(pop['training_ok'])} of {runs} runs succeed ({failure}% failure rate). All ten failures are hybrid pairs that contain <code>engram</code>; eight exit with code 1 and two with code 134 (SIGABRT), and no other family fails. The Transformers export never fails (4,686 records, all OK) and inference evaluation fails once (a <code>pape_efficient</code> run).",
 f"El entrenamiento es fiable: {n(pop['training_ok'])} de {runs} corridas terminan bien (tasa de fallo {failure} %). Los diez fallos son pares híbridos que contienen <code>engram</code>; ocho salen con código 1 y dos con código 134 (SIGABRT), y ninguna otra familia falla. La exportación a Transformers no falla nunca (4.686 registros, todos OK) y la evaluación de inferencia falla una vez (una corrida <code>pape_efficient</code>).")}
{figure_grid([figs['stage']])}
{reliability_table}
</section>

<section id="correlations" class="section">
<h2>{T("9 · Correlations", "9 · Correlaciones")}</h2>
<p>{T(
 "The Spearman matrix over the hybrid pairs shows that the strongest non-trivial correlations describe the schedule rather than the architecture: training time grows with the number of epochs (+0.74), throughput is almost the inverse of step time (-0.95) and mean loss falls as the schedule lengthens (-0.76). The parameter count correlates moderately with loss (-0.42) and gradient magnitude (+0.47) and weakly with wall time (+0.23).",
 "La matriz de Spearman sobre los pares híbridos muestra que las correlaciones fuertes no triviales describen el calendario y no la arquitectura: el tiempo de entrenamiento crece con el número de épocas (+0,74), el rendimiento es casi el inverso del tiempo por paso (-0,95) y la pérdida media baja a medida que el calendario se alarga (-0,76). El número de parámetros correlaciona moderadamente con la pérdida (-0,42) y con la magnitud del gradiente (+0,47) y débilmente con el tiempo de pared (+0,23).")}</p>
{figure_grid([figs['correlation']])}
</section>

<section id="atlas" class="section">
<h2>{T("10 · The interactive atlas", "10 · El atlas interactivo")}</h2>
<p>{T(
 f"The print version of the same matrix: every cell is one hybrid pair, coloured by final loss, with the ten <code>engram</code> holes visible. Column by column the second mixer dominates — a good B keeps almost every A in a narrow band, a bad B drags even the best A down — and no A rescues a bad B.",
 f"La versión impresa de la misma matriz: cada celda es un par híbrido, coloreado por pérdida final, con los diez huecos de <code>engram</code> visibles. Columna a columna el segundo mixer domina — un buen B mantiene casi cualquier A en una banda estrecha, un mal B arrastra incluso al mejor A — y ningún A rescata a un mal B.")}</p>
{figure_grid([figs['matrix']])}
{p(
 f"One cell per hybrid pair: rows are the first mixer (A), columns the second (B), and the upper triangle holds the {pairs} measured pairs. Pick a metric and hover a cell; the ten <code>engram</code> pairs that crash are shown as empty cells. The colour always runs from best (green) to worst (red) for the selected metric.",
 f"Una celda por par híbrido: las filas son el primer mixer (A), las columnas el segundo (B), y el triángulo superior contiene los {pairs} pares medidos. Elige una métrica y pasa el cursor por una celda; los diez pares con <code>engram</code> que fallan aparecen como celdas vacías. El color siempre va del mejor (verde) al peor (rojo) para la métrica elegida.")}
<div class="atlas-controls">
  <label for="atlas-metric" id="atlas-label">{T("Metric", "Métrica")}</label>
  <select id="atlas-metric">
    <option value="loss" data-en="Final loss (nats)" data-es="Pérdida final (nats)">Final loss (nats)</option>
    <option value="params" data-en="Parameters (M)" data-es="Parámetros (M)">Parameters (M)</option>
    <option value="duration" data-en="Training time (h)" data-es="Tiempo de entrenamiento (h)">Training time (h)</option>
    <option value="top5" data-en="Held-out top-5 (%)" data-es="Top-5 retenido (%)">Held-out top-5 (%)</option>
  </select>
  <span class="atlas-hint" id="atlas-hint"></span>
</div>
<div id="atlas-chart" class="chart-xl" role="img" aria-label="A by B matrix of mixers"></div>
<div class="split-tables">
  <div><h3>{T("Ten best pairs", "Diez mejores pares")}</h3><div id="top-pairs"></div></div>
  <div><h3>{T("Ten worst pairs", "Diez peores pares")}</h3><div id="bottom-pairs"></div></div>
</div>
<h3>{T("The role effect: A vs B", "El efecto del rol: A vs B")}</h3>
<p>{T(
 f"Each point is one mixer: its median loss when it sits in layer 0 (A) against its median loss when it sits in layer 1 (B). Points off the diagonal are role-sensitive; the B axis is the wider one ({roles['range_b']} nats against {roles['range_a']}), which is the quantitative form of “the second mixer is the lever”.",
 f"Cada punto es un mixer: su pérdida mediana cuando ocupa la capa 0 (A) frente a su pérdida mediana cuando ocupa la capa 1 (B). Los puntos fuera de la diagonal son sensibles al rol; el eje B es el más ancho ({roles['range_b']} nats frente a {roles['range_a']}), que es la forma cuantitativa de “el segundo mixer es la palanca”.")}</p>
<div id="roles-chart" class="chart-lg" role="img" aria-label="A-role versus B-role median loss"></div>
<h3>{T("Synergy, quality and the Pareto front", "Sinergia, calidad y frontera de Pareto")}</h3>
<p>{T(
 f"Ranking mixers individually cannot reconstruct the atlas, because pairs interact. The synergy of a pair is how far it sits from the average of its two marginals, S = (μ<sub>A</sub> + μ<sub>B</sub>)/2 − L<sub>A,B</sub>: the median absolute synergy is {d(syn['abs_median'], 2)} nats and the P90 is {d(syn['abs_p90'], 2)}, so one pair in ten moves by more than a quarter of a nat on interaction alone — more than most ablations move in total. The positive tail stacks a cheap, stable first mixer under a strong second one (<code>falcon2a + conformer</code> at +{d(syn['top'][0]['synergy'], 2)}); the negative tail is dominated by <code>mtla</code> as B (<code>mtla + tpa</code> at {d(syn['bottom'][0]['synergy'], 2)}).",
 f"Ordenar los mixers individualmente no reconstruye el atlas, porque los pares interactúan. La sinergia de un par mide cuánto se aleja del promedio de sus dos marginales, S = (μ<sub>A</sub> + μ<sub>B</sub>)/2 − L<sub>A,B</sub>: la sinergia absoluta mediana es {d(syn['abs_median'], 2)} nats y el P90 es {d(syn['abs_p90'], 2)}, así que uno de cada diez pares se mueve más de un cuarto de nat solo por interacción — más de lo que mueven la mayoría de las ablaciones en total. La cola positiva apila un primer mixer barato y estable bajo un segundo fuerte (<code>falcon2a + conformer</code> con +{d(syn['top'][0]['synergy'], 2)}); la cola negativa la domina <code>mtla</code> como B (<code>mtla + tpa</code> con {d(syn['bottom'][0]['synergy'], 2)}).")}</p>
{figure_grid([figs['synergy'], figs['topb']])}
<p>{T(
 f"The two panels above close the loop with the loss/quality warning at role level: <code>conformer</code> is the best second mixer by loss and nearly the worst by held-out top-5 (12.8% against <code>ccgqa</code>'s 30.4%). And the speed–quality trade has a front: {len(atlas_st['pareto'])} of the 1,409 analysed pairs are non-dominated, from <code>iha + mtla</code> (9.57 nats in 0.57 h) to <code>falcon2a + conformer</code> (7.74 in 2.75). A deployment picks a point on that curve, not a leaderboard rank.",
 f"Los dos paneles de arriba cierran el circuito con la advertencia pérdida/calidad a nivel de rol: <code>conformer</code> es el mejor segundo mixer en pérdida y casi el peor en top-5 retenido (12.8 % frente al 30.4 % de <code>ccgqa</code>). Y el intercambio velocidad–calidad tiene una frontera: {len(atlas_st['pareto'])} de los 1.409 pares analizados no son dominados, de <code>iha + mtla</code> (9.57 nats en 0.57 h) a <code>falcon2a + conformer</code> (7.74 en 2.75). Un despliegue elige un punto de esa curva, no un puesto de tabla de líderes.")}</p>
{figure_grid([figs['pareto']])}
<h3>{T("Mixer ranking by median loss", "Ranking de mixers por pérdida mediana")}</h3>
<div id="ranking-chart" class="chart-xl" role="img" aria-label="Mixer ranking by median final loss"></div>
</section>

<section id="data" class="section">
<h2>{T("11 · Data and reproducibility", "11 · Datos y reproducibilidad")}</h2>
<p>{T(
 f"Two files are published next to this page: <a href=\"data/atlas.csv\" download>atlas.csv</a> (one row per hybrid pair: loss, parameters, duration, top-1 and top-5) and <a href=\"data/mixers.csv\" download>mixers.csv</a> (one row per mixer: median loss and parameters). The full per-run tables stay in the experiment tree and are summarised in the appendix of the paper.",
 f"Junto a esta página se publican dos archivos: <a href=\"data/atlas.csv\" download>atlas.csv</a> (una fila por par híbrido: pérdida, parámetros, duración, top-1 y top-5) y <a href=\"data/mixers.csv\" download>mixers.csv</a> (una fila por mixer: pérdida y parámetros medianos). Las tablas completas por corrida quedan en el árbol del experimento y se resumen en el apéndice del paper.")}</p>
<p>{T(
 "Rebuild: <code>python experimentos/to_csv.py 3 &amp;&amp; python experimentos/3/report/build_data.py &amp;&amp; python experimentos/3/report/make_site.py</code>. The only slow step is the single streaming pass over the 6.6 GB per-step log (about ten minutes); everything else runs in seconds on CPU.",
 "Reconstruir: <code>python experimentos/to_csv.py 3 &amp;&amp; python experimentos/3/report/build_data.py &amp;&amp; python experimentos/3/report/make_site.py</code>. El único paso lento es la pasada única sobre el log por paso de 6,6 GB (unos diez minutos); el resto corre en segundos en CPU.")}</p>
</section>

<section id="conclusions" class="section">
<h2>{T("12 · What the complete grid changes", "12 · Qué cambia la rejilla completa")}</h2>
<div class="conclusions">
  <div class="conclusion"><span class="num">01</span><div><b>{T("Completeness is a result", "La completitud es un resultado")}</b><p>{T(
    f"Round 2 measured two thirds of the grid; Round 3 measures all {pairs} cells. The rankings that survive — the second mixer as the lever, the weakness of scale, the loss/quality gap — survive because the missing third of the grid could not have hidden a different answer.",
    f"La ronda 2 midió dos tercios de la rejilla; la ronda 3 mide las {pairs} celdas. Los rankings que sobreviven —el segundo mixer como palanca, la debilidad de la escala, la brecha pérdida/calidad— sobreviven porque el tercio faltante no podía esconder otra respuesta.")}</p></div></div>
  <div class="conclusion"><span class="num">02</span><div><b>{T("The schedule, not the model, sets the loss", "El calendario, no el modelo, fija la pérdida")}</b><p>{T(
    "Training time grows with epochs and barely with parameters; mean loss falls as the schedule lengthens. Any cross-family reading of the loss must control for the number of epochs, which is why this page reports the budget next to every family.",
    "El tiempo de entrenamiento crece con las épocas y apenas con los parámetros; la pérdida media baja a medida que el calendario se alarga. Cualquier lectura transversal de la pérdida debe controlar el número de épocas, y por eso esta página reporta el presupuesto junto a cada familia.")}</p></div></div>
  <div class="conclusion"><span class="num">03</span><div><b>{T("Optimizers buy speed with gradient, not preconditioning", "Los optimizadores compran velocidad con gradiente, no con precondicionamiento")}</b><p>{T(
    f"Across the 39 optimizers, final loss and final gradient norm correlate at {opt['rho_loss_grad_final']:+.2f} and quality is orthogonal to CPU cost ({opt['rho_loss_step']:+.2f}) and to memory (median {n(opt['rss_median'])} MB). Pick the optimizer that keeps gradient pressure high; pick the fastest one you can deploy.",
    f"Entre los 39 optimizadores, la pérdida final y la norma final del gradiente correlacionan en {opt['rho_loss_grad_final']:+.2f} y la calidad es ortogonal al coste de CPU ({opt['rho_loss_step']:+.2f}) y a la memoria (mediana {n(opt['rss_median'])} MB). Elige el optimizador que mantiene alta la presión de gradiente; elige el más rápido que puedas desplegar.")}</p></div></div>
  <div class="conclusion"><span class="num">04</span><div><b>{T("Quantize at 8 or 4 bits, not at 2", "Cuantiza a 8 o 4 bits, no a 2")}</b><p>{T(
    f"The v2 runtime keeps {d(agree['q8_0']['mean'],1)}% and {d(agree['q4_0']['mean'],1)}% of the predictions at 8 and 4 bits; 2 bits keeps {d(agree['i2_s']['mean'],1)}% and costs {d(abs(i2['mean_delta_pp']),2)} pp of top-1. Two bits is a different model, not a smaller file.",
    f"El runtime v2 conserva el {d(agree['q8_0']['mean'],1)} % y el {d(agree['q4_0']['mean'],1)} % de las predicciones a 8 y 4 bits; 2 bits conserva el {d(agree['i2_s']['mean'],1)} % y cuesta {d(abs(i2['mean_delta_pp']),2)} pp de top-1. Dos bits es otro modelo, no un archivo más pequeño.")}</p></div></div>
  <div class="conclusion"><span class="num">05</span><div><b>{T("The atlas measures fewer models than it names", "El atlas mide menos modelos de los que nombra")}</b><p>{T(
    "297 of the 1,409 analysed pairs share a bit-identical loss with another pair: the four window mixers collapse into one at a 64-token context, fastconformer equals swa, none equals nope, and pape equals pape_efficient. Counting equivalence classes is part of reading the grid, not a defect to hide.",
    "297 de los 1.409 pares analizados comparten una pérdida idéntica bit a bit con otro par: los cuatro mixers de ventana colapsan en uno en un contexto de 64 tokens, fastconformer equivale a swa, none equivale a nope y pape equivale a pape_efficient. Contar clases de equivalencia es parte de leer la rejilla, no un defecto que esconder.")}</p></div></div>
  <div class="conclusion"><span class="num">06</span><div><b>{T("Measure quality, not only loss", "Mide la calidad, no solo la pérdida")}</b><p>{T(
    f"The loss/quality rank correlation is {stats['inference']['loss_top5_rho']:+.2f}: the atlas ranks learning speed, not model quality. A larger held-out set with per-token log-likelihood is the next measurement that would turn the atlas into a ranking.",
    f"La correlación de rango pérdida/calidad es {stats['inference']['loss_top5_rho']:+.2f}: el atlas ordena velocidad de aprendizaje, no calidad de modelo. Un conjunto retenido más grande con log-verosimilitud por token es la próxima medición que convertiría el atlas en un ranking.")}</p></div></div>
</div>
</section>
"""

    # ------------------------------------------------------------- document
    style = site_style()
    script = site_script()
    extra_style = """
<style>
.fig-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:8px 18px}
.fig-grid .fig{margin:14px 0}
.chart-xl{height:860px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);margin:18px 0}
.chart-lg{height:560px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);margin:18px 0}
#ranking-chart{height:780px}
.atlas-controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:12px 0}
.atlas-controls label{font-size:14px;color:var(--muted)}
.atlas-controls select{background:var(--surface-2);color:var(--text);border:1px solid var(--border);
  border-radius:8px;padding:6px 10px;font:inherit}
.atlas-hint{color:var(--muted);font-size:13px}
@media (max-width:720px){.chart-xl{height:620px}.chart-lg{height:420px}#ranking-chart{height:620px}
  .fig-grid{grid-template-columns:1fr}}
</style>
"""

    topbar = f"""
<div class="topbar">
  <a class="brand" href="../../index.html"><span class="dot"></span>Frankenstein<span class="brand-path" style="color:var(--muted);font-weight:500">/ experiments / 3</span></a>
  <nav>
    <a href="../index.html">{T("All experiments", "Todos los experimentos")}</a>
    <a href="../2/index.html">{T("Experiment 2", "Experimento 2")}</a>
    <a href="#summary">{T("Summary", "Resumen")}</a>
    <a href="#atlas">{T("Atlas", "Atlas")}</a>
    <a href="#optimizers">{T("Optimizers", "Optimizadores")}</a>
    <a href="#gradients">{T("Gradients", "Gradientes")}</a>
    <a href="#inference">{T("Quantization", "Cuantización")}</a>
    <a href="#data">{T("Data", "Datos")}</a>
    <a href="#conclusions">{T("Conclusions", "Conclusiones")}</a>
  </nav>
  <div class="tools">
    <button class="tbtn" data-lang-btn="en" onclick="ftSetLang('en')">EN</button>
    <button class="tbtn" data-lang-btn="es" onclick="ftSetLang('es')">ES</button>
    <button class="tbtn" data-theme-btn="dark" onclick="ftSetTheme('dark')">◐</button>
    <button class="tbtn" data-theme-btn="light" onclick="ftSetTheme('light')">☀</button>
  </div>
</div>
"""

    atlas_js = """
<script src="vendor/echarts.min.js"></script>
<script>
(function(){
  var root=document.documentElement;
  var data=null, metric='loss', charts={}, cells={};
  var TXT={
    en:{metric:'Metric',loss:'Final loss (nats)',params:'Parameters (M)',duration:'Training time (h)',
        top5:'Held-out top-5 (%)',pair:'Pair',nopair:'no data',hint:'rows = first mixer (A), columns = second mixer (B)',
        best:'Ten best pairs',worst:'Ten worst pairs',paramsShort:'Params (M)',timeShort:'Hours',top5Short:'Top-5 (%)'},
    es:{metric:'Métrica',loss:'Pérdida final (nats)',params:'Parámetros (M)',duration:'Tiempo (h)',
        top5:'Top-5 retenido (%)',pair:'Par',nopair:'sin datos',hint:'filas = primer mixer (A), columnas = segundo mixer (B)',
        best:'Diez mejores pares',worst:'Diez peores pares',paramsShort:'Parám. (M)',timeShort:'Horas',top5Short:'Top-5 (%)'}
  };
  var METRICS={
    loss:{key:'loss',good:'low',fmt:function(v){return v.toFixed(2);}},
    params:{key:'params',good:'low',fmt:function(v){return v.toFixed(2);}},
    duration:{key:'duration',good:'low',fmt:function(v){return v.toFixed(2);}},
    top5:{key:'top5',good:'high',fmt:function(v){return (v*100).toFixed(1);}}
  };
  function lang(){return root.getAttribute('data-lang')||'en';}
  function css(name){return getComputedStyle(root).getPropertyValue(name).trim();}
  function txt(){return TXT[lang()];}
  function themeColors(){
    return {text:css('--text'),muted:css('--muted'),grid:css('--border-soft'),
            surface:css('--surface'),accent:css('--accent')};
  }
  function renderAtlas(){
    var t=txt(), m=METRICS[metric], c=themeColors();
    var vals=data.cells.map(function(x){return x[2][m.key];}).filter(function(v){return v!==null&&isFinite(v);});
    var min=Math.min.apply(null,vals), max=Math.max.apply(null,vals);
    // ECharts heatmap points are [x, y, value]: x is the column (B) and y the
    // row (A), so the stored (A, B) pair is swapped here.
    var series=data.cells.filter(function(x){return x[2][m.key]!==null;}).map(function(x){return [x[1],x[0],x[2][m.key]];});
    var colors=m.good==='high'?['#cf222e','#d2991d','#1a7f37']:['#1a7f37','#d2991d','#cf222e'];
    charts.atlas.setOption({
      backgroundColor:'transparent',
      animation:false,
      tooltip:{trigger:'item',backgroundColor:c.surface,borderColor:c.grid,textStyle:{color:c.text,fontSize:12},
        formatter:function(p){
          if(!p.data||p.data[2]==null)return '';
          var meta=cells[p.data[0]+','+p.data[1]]; if(!meta)return '';
          var out='<b>'+data.mixers[p.data[1]]+' + '+data.mixers[p.data[0]]+'</b><br>';
          out+=t.loss+': '+(meta.loss==null?t.nopair:meta.loss.toFixed(3))+'<br>';
          out+=t.params+': '+(meta.params==null?t.nopair:meta.params.toFixed(2))+'<br>';
          out+=t.duration+': '+(meta.duration==null?t.nopair:meta.duration.toFixed(2))+'<br>';
          out+=t.top5+': '+(meta.top5==null?t.nopair:(meta.top5*100).toFixed(1)+'%');
          return out;
        }},
      grid:{left:110,right:96,top:18,bottom:150},
      xAxis:{type:'category',data:data.mixers,splitArea:{show:false},
        axisLabel:{rotate:90,fontSize:9,color:c.muted},axisLine:{lineStyle:{color:c.grid}},
        axisTick:{show:false}},
      yAxis:{type:'category',data:data.mixers,inverse:true,
        axisLabel:{fontSize:9,color:c.muted},axisLine:{lineStyle:{color:c.grid}},
        axisTick:{show:false}},
      visualMap:{min:min,max:max,calculable:true,orient:'vertical',right:10,top:'middle',
        itemWidth:16,itemHeight:200,textStyle:{color:c.muted,fontSize:11},
        text:[m.fmt(max),m.fmt(min)],inRange:{color:colors}},
      series:[{name:t[metric],type:'heatmap',data:series,
        emphasis:{itemStyle:{borderColor:c.text,borderWidth:1}},
        progressive:0}]
    },true);
    document.getElementById('atlas-hint').textContent=t.hint;
    document.getElementById('atlas-label').textContent=t.metric;
    Array.prototype.forEach.call(document.querySelectorAll('#atlas-metric option'),function(o){
      o.textContent=o.getAttribute('data-'+lang());
    });
    document.getElementById('atlas-metric').value=metric;
  }
  function renderRoles(){
    var t=txt(), c=themeColors(), R=data.roles;
    var names=Object.keys(R.a).filter(function(k){return R.b[k]!=null;});
    var pts=names.map(function(k){return [R.a[k],R.b[k],k];});
    var lo=Math.min.apply(null,pts.map(function(p){return Math.min(p[0],p[1]);}))-0.05;
    var hi=Math.max.apply(null,pts.map(function(p){return Math.max(p[0],p[1]);}))+0.05;
    charts.roles.setOption({
      backgroundColor:'transparent',
      tooltip:{formatter:function(p){return '<b>'+p.data[2]+'</b><br>A: '+p.data[0].toFixed(3)+'<br>B: '+p.data[1].toFixed(3);},
        backgroundColor:c.surface,borderColor:c.grid,textStyle:{color:c.text,fontSize:12}},
      grid:{left:70,right:30,top:20,bottom:60},
      xAxis:{name:'A',nameTextStyle:{color:c.muted},min:lo,max:hi,
        axisLabel:{color:c.muted,fontSize:10,formatter:function(v){return v.toFixed(2);}},
        splitLine:{lineStyle:{color:c.grid}}},
      yAxis:{name:'B',nameTextStyle:{color:c.muted},min:lo,max:hi,
        axisLabel:{fontSize:10,color:c.muted,formatter:function(v){return v.toFixed(2);}},
        splitLine:{lineStyle:{color:c.grid}}},
      series:[
        {type:'scatter',data:pts,symbolSize:8,itemStyle:{color:c.accent,opacity:0.8}},
        {type:'line',data:[[lo,lo],[hi,hi]],showSymbol:false,lineStyle:{color:c.muted,type:'dashed',width:1},silent:true}
      ]
    },true);
  }
  function renderRank(){
    var c=themeColors();
    var rows=data.ranking.slice().sort(function(a,b){return b.loss-a.loss;});
    var lo=Math.min.apply(null,rows.map(function(r){return r.loss;}))-0.1;
    charts.rank.setOption({
      backgroundColor:'transparent',
      tooltip:{trigger:'axis',axisPointer:{type:'shadow'},
        formatter:function(ps){var p=ps[0];var r=rows[p.dataIndex];
          return '<b>'+r.mixer+'</b><br>'+r.loss.toFixed(3)+' nats<br>'+r.params.toFixed(2)+' M params<br>n = '+r.n;},
        backgroundColor:c.surface,borderColor:c.grid,textStyle:{color:c.text,fontSize:12}},
      grid:{left:130,right:40,top:16,bottom:40},
      xAxis:{type:'value',min:lo,axisLabel:{color:c.muted,fontSize:10,formatter:function(v){return v.toFixed(1);}},
        splitLine:{lineStyle:{color:c.grid}}},
      yAxis:{type:'category',data:rows.map(function(r){return r.mixer;}),
        axisLabel:{color:c.muted,fontSize:9},axisTick:{show:false}},
      series:[{type:'bar',data:rows.map(function(r){return r.loss;}),itemStyle:{color:c.accent,opacity:0.85}}]
    },true);
  }
  function fmtDelta(v){return (v>=0?'+':'')+v.toFixed(1);}
  function buildTables(){
    var t=txt();
    function rows(list){
      var html='<div class="table-wrap"><table class="dt"><thead><tr><th>'+t.pair+'</th><th>'+
        t.loss+'</th><th>'+t.paramsShort+'</th><th>'+t.timeShort+'</th><th>'+t.top5Short+'</th></tr></thead><tbody>';
      list.forEach(function(r){
        html+='<tr><td><code>'+r.pair.replace(/ \\+ /g,' + ')+'</code></td><td>'+r.loss.toFixed(3)+
          '</td><td>'+r.params.toFixed(2)+'</td><td>'+r.duration.toFixed(2)+
          '</td><td>'+(r.top5==null?'—':r.top5.toFixed(1))+'</td></tr>';
      });
      return html+'</tbody></table></div>';
    }
    document.getElementById('top-pairs').innerHTML=rows(data.top);
    document.getElementById('bottom-pairs').innerHTML=rows(data.bottom);
  }
  function renderAll(){ if(!data)return; renderAtlas(); renderRoles(); renderRank(); }
  function init(){
    fetch('data/atlas.json').then(function(r){return r.json();}).then(function(j){
      data=j;
      j.cells.forEach(function(x){cells[x[1]+','+x[0]]=x[2];});
      charts.atlas=echarts.init(document.getElementById('atlas-chart'));
      charts.roles=echarts.init(document.getElementById('roles-chart'));
      charts.rank=echarts.init(document.getElementById('ranking-chart'));
      document.getElementById('atlas-metric').addEventListener('change',function(e){
        metric=e.target.value; renderAtlas();
      });
      buildTables(); renderAll();
      window.addEventListener('resize',function(){Object.keys(charts).forEach(function(k){charts[k].resize();});});
      var baseLang=window.ftSetLang, baseTheme=window.ftSetTheme;
      window.ftSetLang=function(l){baseLang(l); buildTables(); renderAll();};
      window.ftSetTheme=function(th){baseTheme(th); renderAll();};
    });
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
})();
</script>
"""

    document = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark" data-lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment 3 · The complete mixer atlas · Frankenstein Transformer</title>
<meta name="description" content="Experiment 3: 1,653 runs, 1,431 hybrid pairs and 20.5 M training steps — the complete mixer atlas, with per-block gradients, per-optimizer loss/CPU/memory, quantization and an interactive A×B matrix.">
<meta property="og:type" content="article">
<meta property="og:title" content="Experiment 3 · The complete mixer atlas">
<meta property="og:description" content="1,653 runs, 1,431 hybrid pairs, 20.5 M steps: the complete mixer atlas with gradients, per-optimizer cost, quantization and the mixer × mixer matrix.">
<meta property="og:image" content="https://erickfmm.github.io/frankenstein-transformer/experiments/3/img/matrix_loss-dark.png">
<meta property="og:url" content="https://erickfmm.github.io/frankenstein-transformer/experiments/3/">
<meta name="twitter:card" content="summary_large_image">
<meta name="theme-color" content="#0d1117">
<link rel="canonical" href="https://erickfmm.github.io/frankenstein-transformer/experiments/3/">
<link rel="icon" href="../../../favicon.ico">
{style}
{extra_style}
</head>
<body>
{topbar}
<main id="main">
{hero}
</main>
<footer>
  <span class="t-en">Generated from experimentos/3 on</span><span class="t-es">Generado desde experimentos/3 el</span> 2026-11-21 ·
  <a href="../../index.html"><span class="t-en">Config Builder</span><span class="t-es">Constructor de configs</span></a> ·
  <a href="../2/index.html"><span class="t-en">Experiment 2</span><span class="t-es">Experimento 2</span></a> ·
  <a href="data/atlas.csv" download>atlas.csv</a>
</footer>
{script}
{atlas_js}
</body>
</html>
"""
    return document


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Build the data, the figures and the page."""
    stats = json.loads((C.OUT / "stats.json").read_text(encoding="utf-8"))
    master = pd.read_csv(C.DATA / "master.csv")
    accuracy = pd.read_csv(C.DATA / "inference_acc.csv")

    print("atlas data...")
    atlas = build_atlas(master, accuracy)
    write_data(atlas)
    print(f"  {atlas['meta']['pairs']} pairs, {len(atlas['mixers'])} mixers")

    print("figures...")
    render_figures()

    print("vendor...")
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "vendor").mkdir(exist_ok=True)
    if EXP2_VENDOR.exists():
        shutil.copy2(EXP2_VENDOR, SITE / "vendor" / "echarts.min.js")

    print("page...")
    EXP3_PAGE.write_text(render_page(stats, atlas), encoding="utf-8")
    print(f"  wrote {EXP3_PAGE}")


if __name__ == "__main__":
    main()
