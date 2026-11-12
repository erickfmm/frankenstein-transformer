"""Generate every figure of the Experiment 2 report.

Produces light and dark variants of each figure under ``out/img`` so the page
can swap them with its theme toggle, plus ``out/registry.json`` describing the
figures for the HTML generator.

Usage:
    uv run --with pandas --with numpy --with matplotlib --with seaborn \
        python experimentos/2/report/make_charts.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------- i18n
#: Language used by :func:`tr`; ``make_paper_figures.py`` flips it to render the
#: Spanish variants of the paper charts. Every user-visible string goes through
#: ``tr("english", "español")`` so the two languages cannot drift apart.
_LANG = {"code": "en"}


def set_lang(lang: str) -> None:
    """Select the language for every string emitted after this call."""
    _LANG["code"] = lang


def tr(en: str, es: str) -> str:
    """Return ``es`` when the render language is Spanish, else ``en``."""
    return en if _LANG["code"] == "en" else es

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
IMG = OUT / "img"
IMG.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- style
THEMES = {
    "dark": dict(
        face="#0d1117", axes="#11161d", text="#d6deea", muted="#8b949e",
        grid="#30363d", spine="#30363d", accent="#58a6ff",
    ),
    "light": dict(
        face="#ffffff", axes="#ffffff", text="#1f2328", muted="#59636e",
        grid="#d8dee4", spine="#d0d7de", accent="#0969da",
    ),
}

FAMILY_COLORS = {
    "pair": "#58a6ff", "single": "#79c0ff", "vit": "#d2a8ff", "opt": "#3fb950",
    "trans": "#d2991d", "pe": "#ff7b72", "norm": "#ffa657", "realdata": "#f778ba",
    "task": "#a5d6ff", "tokenizer": "#7ee787", "batch": "#e3b341",
    "audio": "#ff9bce", "bitnet": "#8b949e",
}
FAMILY_LABELS = {
    "pair": "Hybrid pairs", "single": "Singles", "vit": "ViT", "opt": "Optimizers",
    "trans": "Structural", "pe": "Pos. encoding", "norm": "Normalization",
    "realdata": "Real data", "task": "Tasks", "tokenizer": "Tokenizer",
    "batch": "Batch size", "audio": "Audio", "bitnet": "BitNet/GGUF",
}

REGISTRY: list[dict] = []


def apply_theme(mode: str) -> dict:
    t = THEMES[mode]
    sns.set_theme(style="darkgrid" if mode == "dark" else "whitegrid")
    plt.rcParams.update({
        "figure.facecolor": t["face"],
        "savefig.facecolor": t["face"],
        "axes.facecolor": t["axes"],
        "axes.edgecolor": t["spine"],
        "axes.labelcolor": t["text"],
        "axes.titlecolor": t["text"],
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.grid": True,
        "grid.color": t["grid"],
        "grid.alpha": 0.55,
        "grid.linewidth": 0.7,
        "text.color": t["text"],
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.facecolor": t["axes"],
        "legend.edgecolor": t["spine"],
        "legend.fontsize": 9,
        "legend.framealpha": 0.85,
        "font.family": "DejaVu Sans",
        "figure.dpi": 120,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
    })
    return t


def fig(name: str, section: str, en: str, es: str, builder) -> None:
    """Build ``name`` in both themes and append it to the registry."""
    for mode in ("dark", "light"):
        t = apply_theme(mode)
        out = builder(t)
        out.savefig(IMG / f"{name}-{mode}.png", facecolor=t["face"],
                    dpi=130, bbox_inches="tight", pad_inches=0.25)
        plt.close(out)
    REGISTRY.append({"name": name, "section": section, "en": en, "es": es})
    print(f"  ok {name}")


def pct(x, _):
    return f"{x * 100:.0f}%"


def kfmt(x, _):
    return f"{x / 1000:.0f}k" if x >= 1000 else f"{x:.0f}"


# --------------------------------------------------------------------------- data
M = pd.read_csv(OUT / "data" / "master.csv")
PE = pd.read_csv(OUT / "data" / "training_per_epoch.csv")
RES = pd.read_csv(OUT / "data" / "pipeline_stages.csv")
INF = pd.read_csv(OUT / "data" / "inference_samples.csv")
PAIRS = M[M.family == "pair"].copy()

# De-duplicate the retnet / retnet_attn name-level twins (bit-identical rows).
PAIRS = PAIRS.drop_duplicates(subset=["mixer_a", "mixer_b"], keep="first").copy()
PAIRS["label"] = PAIRS["mixer_a"] + " + " + PAIRS["mixer_b"]

FAM_ORDER = [f for f in FAMILY_COLORS if f in set(M.family)]
#: Spanish names for the same families; ``FAM_LABEL`` resolves at call time so
#: ``set_lang`` changes the labels of every figure rendered afterwards.
FAMILY_LABELS_ES = {
    "pair": "Pares híbridos",
    "single": "Individuales",
    "vit": "ViT",
    "opt": "Optimizadores",
    "trans": "Estructural",
    "pe": "Codificación pos.",
    "norm": "Normalización",
    "realdata": "Datos reales",
    "task": "Tareas",
    "tokenizer": "Tokenizador",
    "batch": "Tamaño de lote",
    "audio": "Audio",
    "bitnet": "BitNet/GGUF",
}
#: Metric column -> (English, Spanish) axis label.
METRIC_LABELS = {
    "loss_final": ("Final loss", "Pérdida final"),
    "test_acc": ("Held-out accuracy", "Exactitud en validación"),
    "loss": ("Loss", "Pérdida"),
    "step_ms_avg": ("Mean step time (ms)", "Tiempo medio por paso (ms)"),
    "tok_s_avg": ("Throughput (tokens/s)", "Rendimiento (tokens/s)"),
    "total": ("Total parameters", "Parámetros totales"),
}


def metric_label(col: str) -> str:
    """Human label for a metric column, in the active language."""
    en, es = METRIC_LABELS.get(col, (col.replace("_", " "), col.replace("_", " ")))
    return tr(en, es)


FAM_LABEL = lambda f: tr(FAMILY_LABELS.get(f, f), FAMILY_LABELS_ES.get(f, f))


def ordered_families(frame: pd.DataFrame) -> list[str]:
    return [f for f in FAM_ORDER if f in set(frame.family)]


# =========================================================================== params
def params_hist(t):
    fig_, ax = plt.subplots(figsize=(10, 5.2))
    data = M["total"] / 1000
    sns.histplot(data, bins=34, color=t["accent"], edgecolor=t["face"], ax=ax, alpha=0.85)
    med = data.median()
    ax.axvline(med, color="#f85149", lw=1.6, ls="--")
    ax.text(med, ax.get_ylim()[1] * 0.92, f" median {med:.0f}k", color="#f85149", fontsize=9)
    ax.set_xlabel(tr("Total parameters (thousands)", "Parámetros totales (miles)"))
    ax.set_ylabel(tr("Configurations", "Configuraciones"))
    ax.set_title(tr("Parameter budget across all 868 configurations", "Presupuesto de parámetros en las 868 configuraciones"))
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    return fig_


def params_family_box(t):
    order = ordered_families(M)
    data = M.assign(fam=M.family.map(FAM_LABEL))
    order_l = [FAM_LABEL(f) for f in order]
    fig_, ax = plt.subplots(figsize=(11, 5.4))
    sns.stripplot(data=data, x="fam", y="total", order=order_l, color=t["text"],
                  size=1.9, alpha=0.25, jitter=0.22, ax=ax, zorder=1)
    sns.boxplot(data=data, x="fam", y="total", order=order_l, hue="fam",
                palette=[FAMILY_COLORS[f] for f in order], legend=False,
                width=0.52, linewidth=1.1, fliersize=0, ax=ax,
                saturation=0.9, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel(tr("Total parameters (log)", "Parámetros totales (log)"))
    ax.set_title(tr("Parameter budget by experiment family", "Presupuesto de parámetros por familia de experimento"))
    ax.tick_params(axis="x", rotation=30)
    return fig_


def params_composition(t):
    comp = ["attention", "embeddings", "ffn", "head", "norm", "pos", "mhc", "router", "residual"]
    grp = M.groupby("family")[comp].mean()
    grp = grp.loc[[f for f in ordered_families(M)]]
    share = grp.div(grp.sum(axis=1), axis=0) * 100
    fig_, ax = plt.subplots(figsize=(12, 5.4))
    palette = sns.color_palette("crest", len(comp))
    bottom = np.zeros(len(share))
    for c, col in zip(comp, palette):
        ax.barh([FAM_LABEL(f) for f in share.index], share[c], left=bottom,
                color=col, label=c, edgecolor=t["face"], linewidth=0.6)
        bottom += share[c].values
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(tr("Share of total parameters (%)", "Proporción del total de parámetros (%)"))
    ax.set_title(tr("Where the parameters live, by family", "Dónde viven los parámetros, por familia"))
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return fig_


def params_extremes(t, largest=True):
    sub = M.nlargest(20, "total") if largest else M.nsmallest(20, "total")
    sub = sub.sort_values("total")
    fig_, ax = plt.subplots(figsize=(10, 6.4))
    colors = [FAMILY_COLORS.get(f, "#8b949e") for f in sub.family]
    ax.barh(sub.combo_id.str.replace("attn_", "", regex=False), sub.total / 1000, color=colors)
    ax.set_xlabel(tr("Total parameters (thousands)", "Parámetros totales (miles)"))
    ax.set_title(("The 20 largest" if largest else "The 20 smallest") + " configurations")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    return fig_


# ========================================================================= training
def loss_family_box(t):
    order = ordered_families(M)
    data = M.assign(fam=M.family.map(FAM_LABEL))
    ol = [FAM_LABEL(f) for f in order]
    fig_, ax = plt.subplots(figsize=(11, 5.4))
    sns.stripplot(data=data, x="fam", y="loss_final", order=ol, color=t["text"],
                  size=1.8, alpha=0.22, jitter=0.22, ax=ax, zorder=1)
    sns.boxplot(data=data, x="fam", y="loss_final", order=ol, hue="fam",
                palette=[FAMILY_COLORS[f] for f in order], legend=False,
                width=0.52, linewidth=1.1, fliersize=0, ax=ax,
                saturation=0.9, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.18, 13)
    ax.set_xlabel("")
    ax.set_ylabel(tr("Final training loss (log)", "Pérdida final de entrenamiento (log)"))
    ax.set_title(tr("Final loss by family — MLM vs vision vs ablations", "Pérdida final por familia — MLM vs visión vs ablaciones"))
    ax.tick_params(axis="x", rotation=30)
    return fig_


def acc_family_box(t):
    order = ordered_families(M)
    data = M.assign(fam=M.family.map(FAM_LABEL))
    ol = [FAM_LABEL(f) for f in order]
    fig_, ax = plt.subplots(figsize=(11, 5.4))
    sns.stripplot(data=data, x="fam", y="test_acc", order=ol, color=t["text"],
                  size=1.8, alpha=0.22, jitter=0.22, ax=ax, zorder=1)
    sns.boxplot(data=data, x="fam", y="test_acc", order=ol, hue="fam",
                palette=[FAMILY_COLORS[f] for f in order], legend=False,
                width=0.52, linewidth=1.1, fliersize=0, ax=ax,
                saturation=0.9, zorder=3)
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xlabel("")
    ax.set_ylabel(tr("Held-out metric", "Métrica de validación"))
    ax.set_title(tr("Held-out score by family (MLM top-1, vision accuracy)", "Puntaje de validación por familia (top-1 MLM, exactitud de visión)"))
    ax.tick_params(axis="x", rotation=30)
    return fig_


def loss_acc_scatter(t):
    fig_, ax = plt.subplots(figsize=(9.4, 6.4))
    for fam in ordered_families(M):
        s = M[M.family == fam]
        ax.scatter(s.loss_final, s.test_acc, s=28, alpha=0.75,
                   color=FAMILY_COLORS[fam], label=FAM_LABEL(fam), edgecolor="none")
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xlabel(tr("Final training loss (log)", "Pérdida final de entrenamiento (log)"))
    ax.set_ylabel(tr("Held-out metric", "Métrica de validación"))
    ax.set_title(tr("Loss vs held-out score — families separate cleanly", "Pérdida vs puntaje de validación — las familias se separan limpiamente"))
    ax.legend(ncol=2, fontsize=8)
    return fig_


def loss_hist_pairs(t):
    fig_, ax = plt.subplots(figsize=(10, 5.2))
    sns.histplot(PAIRS.loss_final, bins=40, color=t["accent"], edgecolor=t["face"], ax=ax)
    for q, c in ((0.05, "#3fb950"), (0.5, "#d2991d"), (0.95, "#f85149")):
        v = PAIRS.loss_final.quantile(q)
        ax.axvline(v, color=c, lw=1.3, ls="--")
    ax.set_xlabel(tr("Final training loss", "Pérdida final de entrenamiento"))
    ax.set_ylabel(tr("Hybrid pairs", "Pares híbridos"))
    ax.set_title(tr("Loss distribution of the 667 unique hybrid pairs", "Distribución de pérdida de los 667 pares híbridos únicos"))
    return fig_


def acc_hist_pairs(t):
    fig_, ax = plt.subplots(figsize=(10, 5.2))
    sns.histplot(PAIRS.test_acc, bins=40, color="#3fb950", edgecolor=t["face"], ax=ax)
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xlabel(tr("Held-out MLM top-1", "Top-1 MLM en validación"))
    ax.set_ylabel(tr("Hybrid pairs", "Pares híbridos"))
    ax.set_title(tr("Held-out accuracy distribution — a heavy tail at zero", "Distribución de exactitud en validación — una cola pesada en cero"))
    return fig_


def params_loss_pairs(t):
    fig_, ax = plt.subplots(figsize=(9.4, 6.2))
    x = PAIRS.total / 1000
    y = PAIRS.loss_final
    ax.scatter(x, y, s=26, alpha=0.6, color=t["accent"], edgecolor="none")
    z = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(z, xs), color="#f85149", lw=1.6,
            label=f"slope {z[0]:+.4f} / k-param")
    ax.set_xlabel(tr("Total parameters (thousands)", "Parámetros totales (miles)"))
    ax.set_ylabel(tr("Final training loss", "Pérdida final de entrenamiento"))
    ax.set_title(f"Bigger is (slightly) better: r = {np.corrcoef(x, y)[0,1]:.2f}")
    ax.legend()
    return fig_


def params_acc_pairs(t):
    fig_, ax = plt.subplots(figsize=(9.4, 6.2))
    x = PAIRS.total / 1000
    y = PAIRS.test_acc
    ax.scatter(x, y, s=26, alpha=0.6, color="#3fb950", edgecolor="none")
    z = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(z, xs), color="#f85149", lw=1.6,
            label=f"r = {np.corrcoef(x, y)[0,1]:+.2f}")
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xlabel(tr("Total parameters (thousands)", "Parámetros totales (miles)"))
    ax.set_ylabel(tr("Held-out MLM top-1", "Top-1 MLM en validación"))
    ax.set_title(tr("Parameter count barely predicts held-out accuracy", "El número de parámetros apenas predice la exactitud en validación"))
    ax.legend()
    return fig_


def ranking_bar(t, col, n, best, title, fmt=None):
    sub = PAIRS.nlargest(n, col) if best else PAIRS.nsmallest(n, col)
    sub = sub.sort_values(col)
    fig_, ax = plt.subplots(figsize=(10, 6.6))
    vals = sub[col]
    ax.barh(sub.label, vals, color=t["accent"] if best else "#f85149")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}" if fmt is None else fmt(v), va="center", fontsize=7.5)
    ax.set_xlabel(metric_label(col))
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=7.5)
    return fig_


def marginal_bar(t, key, col, best, title, n=10):
    g = PAIRS.groupby(key)[col].mean().sort_values(ascending=not best).head(n)
    fig_, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.barh(g.index[::-1], g.values[::-1], color=t["accent"] if best else "#f85149")
    ax.set_xlabel(tr(f"Mean {metric_label(col).lower()}",
                     f"Media de {metric_label(col).lower()}"))
    ax.set_title(title)
    for i, v in enumerate(g.values[::-1]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    return fig_


# ========================================================================= dynamics
def curves_family(t):
    pe = PE.merge(M[["combo_id", "family"]], on="combo_id", how="left")
    fig_, ax = plt.subplots(figsize=(10.5, 5.6))
    for fam in ordered_families(M):
        s = pe[pe.family == fam]
        if s.empty:
            continue
        g = s.groupby("epoch").loss.agg(["mean", "std"]).reset_index()
        ax.plot(g.epoch, g["mean"], lw=1.8, color=FAMILY_COLORS[fam], label=FAM_LABEL(fam))
        ax.fill_between(g.epoch, g["mean"] - g["std"] * 0.25, g["mean"] + g["std"] * 0.25,
                        color=FAMILY_COLORS[fam], alpha=0.12, linewidth=0)
    ax.set_xlabel(tr("Epoch", "Época"))
    ax.set_ylabel(tr("Training loss (epoch mean)", "Pérdida de entrenamiento (media por época)"))
    ax.set_title(tr("Learning curves by family — the MLM pairs are still descending", "Curvas de aprendizaje por familia — los pares MLM aún descienden"))
    ax.legend(ncol=3, fontsize=8)
    return fig_


def curves_top10(t):
    top = PAIRS.nsmallest(10, "loss_final")[["combo_id", "label"]]
    pe = PE[PE.combo_id.isin(top.combo_id)]
    fig_, ax = plt.subplots(figsize=(10.5, 5.8))
    palette = sns.color_palette("crest", 10)
    for (cid, label), col in zip(top.itertuples(index=False), palette):
        s = pe[pe.combo_id == cid].sort_values("epoch")
        ax.plot(s.epoch, s.loss, lw=1.7, color=col, label=label.replace(" + ", "+"))
    ax.set_xlabel(tr("Epoch", "Época"))
    ax.set_ylabel(tr("Training loss (epoch mean)", "Pérdida de entrenamiento (media por época)"))
    ax.set_title(tr("The ten best hybrid pairs, epoch by epoch", "Los diez mejores pares híbridos, época por época"))
    ax.legend(fontsize=7.5, ncol=2)
    return fig_


def curves_quantiles(t):
    pe = PE[PE.combo_id.isin(PAIRS.combo_id)]
    g = pe.groupby("epoch").loss.agg(
        p10=lambda s: s.quantile(0.10), p50="median",
        p90=lambda s: s.quantile(0.90), mean="mean")
    fig_, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.fill_between(g.index, g.p10, g.p90, color=t["accent"], alpha=0.18,
                    label=tr("P10–P90", "P10–P90"))
    ax.plot(g.index, g.p50, color=t["accent"], lw=2, label=tr("Median", "Mediana"))
    ax.plot(g.index, g["mean"], color="#f85149", lw=1.4, ls="--", label=tr("Mean", "Media"))
    ax.set_xlabel(tr("Epoch", "Época"))
    ax.set_ylabel(tr("Training loss", "Pérdida de entrenamiento"))
    ax.set_title(tr("Where the pair population sits at each epoch", "Dónde está la población de pares en cada época"))
    ax.legend()
    return fig_


GRAD_COLS = {"embeddings": "grad_embeddings", "pos": "grad_pos",
             "attention": "grad_attention", "ffn": "grad_ffn",
             "norm": "grad_norms", "head": "grad_head"}


def grad_groups(t):
    groups = list(GRAD_COLS)
    means = [M[GRAD_COLS[g]].mean() for g in groups]
    errs = [M[GRAD_COLS[g]].std() for g in groups]
    fig_, ax = plt.subplots(figsize=(9.6, 5.2))
    bars = ax.bar(groups, means, yerr=errs, capsize=4, color=sns.color_palette("crest", len(groups)),
                  edgecolor=t["face"], linewidth=1)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel(tr("Mean gradient norm across runs", "Norma del gradiente media entre corridas"))
    ax.set_title(tr("Gradient pressure by parameter group", "Presión del gradiente por grupo de parámetros"))
    return fig_


def grad_heatmap(t):
    d = M.groupby("family")[list(GRAD_COLS.values())].mean()
    d.columns = list(GRAD_COLS)
    d.index = [FAM_LABEL(f) for f in d.index]
    fig_, ax = plt.subplots(figsize=(9.2, 6.2))
    sns.heatmap(d, annot=True, fmt=".1f", cmap="rocket_r", ax=ax,
                cbar_kws={"label": "grad norm"}, linewidths=0.6, linecolor=t["face"])
    ax.set_title(tr("Gradient norm by family and parameter group", "Norma del gradiente por familia y grupo de parámetros"))
    ax.set_xlabel("")
    ax.set_ylabel("")
    return fig_


def clip_hist(t):
    fig_, ax = plt.subplots(figsize=(9.6, 5.0))
    sns.histplot(M.clip_avg.clip(0, 3), bins=45, color="#d2991d", edgecolor=t["face"], ax=ax)
    ax.axvline(1.0, color="#f85149", lw=1.4, ls="--")
    ax.text(1.02, ax.get_ylim()[1] * 0.9, "clip threshold", color="#f85149", fontsize=9)
    ax.set_xlabel(tr("Mean clip ratio (clipped / unclipped)", "Ratio de recorte medio (recortado / sin recortar)"))
    ax.set_ylabel(tr("Configurations", "Configuraciones"))
    ax.set_title(tr("Gradient clipping: most runs sit right at the threshold", "Recorte de gradiente: casi todas las corridas quedan en el umbral"))
    return fig_


def lr_curve(t):
    pe = PE.merge(M[["combo_id", "family"]], on="combo_id", how="left")
    fig_, ax = plt.subplots(figsize=(10, 5))
    for fam in ("pair", "opt", "vit"):
        s = pe[pe.family == fam]
        if s.empty:
            continue
        g = s.groupby("epoch").learning_rate.mean()
        ax.plot(g.index, g.values, lw=1.8, color=FAMILY_COLORS[fam], label=FAM_LABEL(fam))
    ax.set_yscale("log")
    ax.set_xlabel(tr("Epoch", "Época"))
    ax.set_ylabel(tr("Learning rate (log)", "Tasa de aprendizaje (log)"))
    ax.set_title(tr("Warm-up and cosine decay used in every run", "Warm-up y decaimiento coseno en todas las corridas"))
    ax.legend()
    return fig_


def speed_family_box(t):
    order = ordered_families(M)
    data = M.assign(fam=M.family.map(FAM_LABEL))
    ol = [FAM_LABEL(f) for f in order]
    fig_, ax = plt.subplots(figsize=(11, 5.4))
    sns.stripplot(data=data, x="fam", y="tok_s_avg", order=ol, color=t["text"],
                  size=1.8, alpha=0.22, jitter=0.22, ax=ax, zorder=1)
    sns.boxplot(data=data, x="fam", y="tok_s_avg", order=ol, hue="fam",
                palette=[FAMILY_COLORS[f] for f in order], legend=False,
                width=0.52, linewidth=1.1, fliersize=0, ax=ax,
                saturation=0.9, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel(tr("Training throughput (tokens/s, log)", "Rendimiento de entrenamiento (tokens/s, log)"))
    ax.set_title(tr("Throughput by family — the cheapest models are not the fastest", "Rendimiento por familia — los modelos más baratos no son los más rápidos"))
    ax.tick_params(axis="x", rotation=30)
    return fig_


def step_vs_params(t):
    fig_, ax = plt.subplots(figsize=(9.4, 6.0))
    x = M.total / 1000
    y = M.step_ms_avg
    ax.scatter(x, y, s=24, alpha=0.6, color="#d2991d", edgecolor="none")
    z = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(z, xs), color="#f85149", lw=1.5)
    ax.set_xlabel(tr("Total parameters (thousands)", "Parámetros totales (miles)"))
    ax.set_ylabel(tr("Mean step time (ms)", "Tiempo medio por paso (ms)"))
    ax.set_title(f"Step time tracks parameter count (r = {np.corrcoef(x, y)[0,1]:.2f})")
    return fig_


def duration_hist(t):
    fig_, ax = plt.subplots(figsize=(9.6, 5.0))
    sns.histplot(M.train_dur / 60, bins=34, color="#d2a8ff", edgecolor=t["face"], ax=ax)
    ax.set_xlabel(tr("Training wall-clock (minutes)", "Tiempo real de entrenamiento (minutos)"))
    ax.set_ylabel(tr("Configurations", "Configuraciones"))
    ax.set_title(tr("How long a 100-epoch run took on the shared GPU", "Cuánto tardó una corrida de 100 épocas en la GPU compartida"))
    return fig_


# ========================================================================= pairs
def pair_matrix(t, value, cmap, title, cbar, fmt=".2f", annot=True, log=False, center=None):
    piv = PAIRS.pivot_table(index="mixer_a", columns="mixer_b", values=value, aggfunc="mean")
    order_a = PAIRS.groupby("mixer_a")[value].mean().sort_values().index
    order_b = PAIRS.groupby("mixer_b")[value].mean().sort_values().index
    piv = piv.reindex(index=order_a, columns=order_b)
    fig_, ax = plt.subplots(figsize=(18, 13))
    sns.heatmap(piv, cmap=cmap, ax=ax, annot=annot, fmt=fmt,
                annot_kws={"fontsize": 5.2}, linewidths=0.4, linecolor=t["face"],
                cbar_kws={"label": cbar, "shrink": 0.55},
                center=center, norm=None)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(tr("Second mixer (B)", "Segundo mixer (B)"))
    ax.set_ylabel(tr("First mixer (A)", "Primer mixer (A)"))
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    return fig_


def synergy(t):
    ma = PAIRS.groupby("mixer_a").loss_final.mean()
    mb = PAIRS.groupby("mixer_b").loss_final.mean()
    p = PAIRS.copy()
    p["expected"] = (p.mixer_a.map(ma) + p.mixer_b.map(mb)) / 2
    p["synergy"] = p.expected - p.loss_final
    s = pd.concat([p.nlargest(12, "synergy"), p.nsmallest(12, "synergy")]).drop_duplicates()
    s = s.sort_values("synergy")
    colors = ["#f85149" if v < 0 else "#3fb950" for v in s.synergy]
    fig_, ax = plt.subplots(figsize=(10, 7.4))
    ax.barh(s.label, s.synergy, color=colors)
    ax.axvline(0, color=t["muted"], lw=0.9)
    ax.set_xlabel(tr("Synergy (nats better than the two marginal means)", "Sinergia (nats mejor que las dos medias marginales)"))
    ax.set_title(tr("Best and worst interactions between mixers", "Mejores y peores interacciones entre mixers"))
    ax.tick_params(axis="y", labelsize=7.5)
    return fig_


def coverage(t):
    piv = PAIRS.pivot_table(index="mixer_a", columns="mixer_b", values="loss_final", aggfunc="size")
    fig_, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(piv.notna().astype(int), cmap=sns.color_palette(["#21262d", "#58a6ff"], as_cmap=True),
                ax=ax, cbar=False, linewidths=0.4, linecolor=t["face"])
    ax.set_title(tr("Pair coverage: which A×B cells were actually trained", "Cobertura de pares: qué celdas A×B se entrenaron realmente"))
    ax.set_xlabel(tr("Second mixer (B)", "Segundo mixer (B)"))
    ax.set_ylabel(tr("First mixer (A)", "Primer mixer (A)"))
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    return fig_


def pareto_speed_loss(t):
    p = PAIRS.dropna(subset=["tok_s_median", "loss_final"]).copy()
    p = p[p.tok_s_median < 1500]
    fig_, ax = plt.subplots(figsize=(9.6, 6.2))
    ax.scatter(p.tok_s_median, p.loss_final, s=30, alpha=0.65, color=t["accent"], edgecolor="none")
    s = p.sort_values("tok_s_median")
    best = np.inf
    xs, ys = [], []
    for _, r in s.iterrows():
        if r.loss_final < best:
            best = r.loss_final
            xs.append(r.tok_s_median)
            ys.append(r.loss_final)
    ax.step(xs, ys, where="post", color="#3fb950", lw=1.6, label=tr("Pareto front", "Frente de Pareto"))
    for x, y, lab in zip(xs[:5], ys[:5], s[s.loss_final.isin(ys)].label.head(5)):
        ax.annotate(lab.replace(" + ", "+"), (x, y), fontsize=7, color=t["muted"],
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(tr("Throughput (tokens/s)", "Rendimiento (tokens/s)"))
    ax.set_ylabel(tr("Final training loss", "Pérdida final de entrenamiento"))
    ax.set_title(tr("Speed–quality Pareto front of the hybrid pairs", "Frente de Pareto velocidad–calidad de los pares híbridos"))
    ax.legend()
    return fig_


def top5_by_b(t):
    g = PAIRS.groupby("mixer_b").inf_top5_std.agg(["mean", "count"]).dropna()
    g = g[g["count"] >= 5].sort_values("mean", ascending=False).head(15)
    fig_, ax = plt.subplots(figsize=(9.8, 5.6))
    ax.barh(g.index[::-1], g["mean"].values[::-1], color="#3fb950")
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xlabel(tr("Top-5 hit rate (MLM, 4 samples)", "Tasa de acierto top-5 (MLM, 4 muestras)"))
    ax.set_title(tr("Which second mixer puts the answer in the top 5", "Qué segundo mixer pone la respuesta en el top 5"))
    return fig_


def top5_vs_loss(t):
    fig_, ax = plt.subplots(figsize=(9.2, 6.0))
    p = PAIRS.dropna(subset=["inf_top5_std"])
    ax.scatter(p.loss_final, p.inf_top5_std, s=26, alpha=0.7, color="#d2a8ff", edgecolor="none")
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xlabel(tr("Final training loss", "Pérdida final de entrenamiento"))
    ax.set_ylabel(tr("Top-5 hit rate", "Tasa de acierto top-5"))
    ax.set_title(f"Training loss barely predicts inference quality (r = {p[['loss_final','inf_top5_std']].corr().iloc[0,1]:+.2f})")
    return fig_


# ========================================================================= ablations
def group_bar(t, family, col, title, fmt=".3f", sort_asc=True, color="#58a6ff", rotate=0):
    sub = M[M.family == family].dropna(subset=[col]).sort_values(col, ascending=sort_asc)
    fig_, ax = plt.subplots(figsize=(10.5, 5.6))
    bars = ax.bar(range(len(sub)), sub[col], color=color)
    ax.set_xticks(range(len(sub)))
    labels = [c.replace(family + "_", "") for c in sub.combo_id]
    ax.set_xticklabels(labels, rotation=rotate or (55 if len(sub) > 12 else 0),
                       ha="right" if len(sub) > 12 else "center", fontsize=8)
    for i, v in enumerate(sub[col]):
        ax.text(i, v, f"{v:{fmt}}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel(metric_label(col))
    ax.set_title(title)
    ax.margins(y=0.12)
    return fig_


def opt_speed(t):
    sub = M[M.family == "opt"].sort_values("step_ms_avg")
    fig_, ax = plt.subplots(figsize=(11, 5.6))
    ax.barh([c.replace("opt_", "") for c in sub.combo_id], sub.step_ms_avg, color="#3fb950")
    ax.set_xlabel(tr("Mean step time (ms)", "Tiempo medio por paso (ms)"))
    ax.set_title(tr("Optimizer overhead — under 30% between fastest and slowest", "Sobrecoste del optimizador — menos del 30% entre el más rápido y el más lento"))
    ax.tick_params(axis="y", labelsize=8)
    return fig_


def vit_split(t):
    sub = M[M.family == "vit"].copy()
    sub["grp"] = sub.combo_id.str.extract(r"vit_(?:pe_)?(classification|patch_prediction|segmentation)")[0]
    fig_, ax = plt.subplots(figsize=(10.4, 5.6))
    sns.boxplot(data=sub, x="grp", y="test_acc", color="#d2a8ff", width=0.5, ax=ax)
    sns.stripplot(data=sub, x="grp", y="test_acc", color=t["text"], size=3.4, alpha=0.55, ax=ax)
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xlabel("")
    ax.set_ylabel(tr("Held-out accuracy", "Exactitud en validación"))
    ax.set_title(tr("Vision tasks: segmentation is the easiest, classification the noisiest", "Tareas de visión: la segmentación es la más fácil, la clasificación la más ruidosa"))
    return fig_


def tokenizer_task(t):
    fig_, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, fam, title in zip(axes, ("tokenizer", "task"), ("Tokenizer", "Task / model class")):
        sub = M[M.family == fam].sort_values("loss_final")
        ax.barh([c.replace(fam + "_", "") for c in sub.combo_id], sub.loss_final, color="#7ee787")
        ax.set_title(title)
        ax.set_xlabel(tr("Final loss", "Pérdida final"))
        ax.tick_params(axis="y", labelsize=8)
    fig_.tight_layout()
    return fig_


# ========================================================================= deploy
def inference_format(t):
    fig_, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
    for ax, col, title in zip(axes, ("inf_top1", "inf_top5"), ("Top-1 accuracy", tr("Top-5 hit rate", "Tasa de acierto top-5"))):
        means = [M[f"{col}_std"].mean(), M[f"{col}_q"].mean()]
        ax.bar(["standard", "quantized"], means, color=["#58a6ff", "#d2991d"])
        for i, v in enumerate(means):
            ax.text(i, v, f"{v*100:.2f}%", ha="center", va="bottom", fontsize=9)
        ax.yaxis.set_major_formatter(FuncFormatter(pct))
        ax.set_title(title)
        ax.margins(y=0.18)
    fig_.suptitle("Post-training quantization costs almost nothing", fontsize=12, fontweight="bold")
    fig_.tight_layout()
    return fig_


def quant_delta_dist(t):
    d = M.quant_top5_delta.dropna()
    fig_, ax = plt.subplots(figsize=(9.4, 4.8))
    vals = d.value_counts().sort_index()
    ax.bar(vals.index * 100, vals.values, width=8, color="#d2991d")
    ax.axvline(0, color=t["muted"], lw=1)
    ax.set_xlabel(tr("Top-5 delta (quantized − standard, percentage points)", "Delta top-5 (cuantizado − estándar, puntos porcentuales)"))
    ax.set_ylabel(tr("Configurations", "Configuraciones"))
    ax.set_title(f"Quantization delta: mean {d.mean()*100:+.2f} pp, {int((d==0).sum())} of {len(d)} exactly zero")
    return fig_


def quant_scatter(t):
    p = M.dropna(subset=["inf_top5_std", "inf_top5_q"])
    fig_, ax = plt.subplots(figsize=(7.6, 6.6))
    jitter = lambda s: s + np.random.default_rng(0).normal(0, 0.008, len(s))
    ax.scatter(jitter(p.inf_top5_std), jitter(p.inf_top5_q), s=22, alpha=0.6,
               color="#58a6ff", edgecolor="none")
    ax.plot([0, 1], [0, 1], color="#f85149", lw=1.2, ls="--", label=tr("identity", "identidad"))
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xlabel(tr("Standard top-5 hit rate", "Tasa de acierto top-5 estándar"))
    ax.set_ylabel(tr("Quantized top-5 hit rate", "Tasa de acierto top-5 cuantizada"))
    ax.set_title(tr("Standard vs quantized, configuration by configuration", "Estándar vs cuantizado, configuración por configuración"))
    ax.legend()
    return fig_


def deploy_duration(t):
    rows = []
    for col, lab in (("deploy_std_dur", "deploy standard"), ("deploy_q_dur", "deploy quantized"),
                     ("export_dur", "HF export"), ("eval_std_dur", "eval standard"),
                     ("eval_q_dur", "eval quantized")):
        s = M[col].dropna()
        rows += [{"stage": lab, "sec": v} for v in s]
    d = pd.DataFrame(rows)
    fig_, ax = plt.subplots(figsize=(9.8, 5.0))
    sns.boxplot(data=d, x="stage", y="sec", hue="stage", palette="crest", legend=False, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel(tr("Wall-clock (s, log)", "Tiempo real (s, log)"))
    ax.set_title(tr("Pipeline stage cost per configuration", "Coste por etapa del pipeline y por configuración"))
    ax.tick_params(axis="x", rotation=15)
    return fig_


def top5_by_family(t):
    g = M.groupby("family").inf_top5_std.mean().dropna()
    g = g.sort_values()
    fig_, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.barh([FAM_LABEL(f) for f in g.index], g.values, color=[FAMILY_COLORS[f] for f in g.index])
    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xlabel(tr("Mean top-5 hit rate", "Tasa de acierto top-5 media"))
    ax.set_title(tr("Inference quality by family", "Calidad de inferencia por familia"))
    return fig_


# ========================================================================= reliability
def stage_status(t):
    r = RES.copy()
    r["stage"] = r.category.replace({"transformers-export": "HF export", "inference-eval": "inference eval",
                                     "bitnet-gguf": "GGUF"})
    tab = r.groupby(["stage", "status"]).size().unstack(fill_value=0)
    tab = tab.reindex(columns=["OK", "FAILED"], fill_value=0)
    fig_, ax = plt.subplots(figsize=(9.6, 4.8))
    bottom = np.zeros(len(tab))
    for status, col in (("OK", "#3fb950"), ("FAILED", "#f85149")):
        ax.barh(tab.index, tab[status], left=bottom, color=col, label=status)
        bottom += tab[status].values
    for i, (ok, bad) in enumerate(zip(tab["OK"], tab["FAILED"])):
        ax.text(ok + bad + 8, i, f"{ok} ok / {bad} failed", va="center", fontsize=8.5)
    ax.set_xlabel(tr("Pipeline records", "Registros del pipeline"))
    ax.set_title(tr("Pipeline reliability — 95% of stage records succeeded", "Fiabilidad del pipeline — el 95% de los registros de etapa tuvo éxito"))
    ax.legend()
    return fig_


def stage_duration_box(t):
    r = RES[RES.status == "OK"].copy()
    r["stage"] = r.category.replace({"transformers-export": "HF export", "inference-eval": "inference eval",
                                     "bitnet-gguf": "GGUF"})
    fig_, ax = plt.subplots(figsize=(9.6, 5.0))
    sns.boxplot(data=r, x="stage", y="duration_sec", hue="stage", palette="flare", legend=False, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel(tr("Duration (s, log)", "Duración (s, log)"))
    ax.set_title(tr("Stage durations — training dwarfs everything else", "Duraciones por etapa — el entrenamiento eclipsa todo lo demás"))
    ax.tick_params(axis="x", rotation=15)
    return fig_


def failures_family(t):
    f = RES[RES.status == "FAILED"].groupby("family").size().sort_values()
    fig_, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.barh([FAM_LABEL(x) if x in FAMILY_LABELS else x for x in f.index], f.values, color="#f85149")
    for i, v in enumerate(f.values):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    ax.set_xlabel(tr("Failed stage records", "Registros de etapa fallidos"))
    ax.set_title(tr("Where the 275 failures landed", "Dónde cayeron los 275 fallos"))
    return fig_


def duplicate_fingerprints(t):
    names = ["standard", "bigbird", "longformer", "titan"]
    sub = PAIRS[PAIRS.mixer_a.isin(names)].groupby("mixer_a").loss_final.agg(["mean", "std"])
    fig_, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(sub.index, sub["mean"], color=["#f85149", "#d2991d", "#58a6ff", "#3fb950"])
    ax.set_ylim(sub["mean"].min() - 0.2, sub["mean"].max() + 0.2)
    ax.set_ylabel(tr("Mean final loss", "Pérdida final media"))
    ax.set_title(tr("Four 'different' first mixers produce identical models", "Cuatro primeros mixers 'distintos' producen modelos idénticos"))
    for i, v in enumerate(sub["mean"]):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    return fig_


# ========================================================================= correlation
def correlation(t):
    cols = ["total", "loss_final", "test_acc", "tok_s_median", "step_ms_avg",
            "grad_avg", "clip_avg", "train_dur", "inf_top5_std", "quant_top5_delta"]
    d = M[cols].corr()
    fig_, ax = plt.subplots(figsize=(9.4, 7.6))
    sns.heatmap(d, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
                ax=ax, square=True, linewidths=0.6, linecolor=t["face"],
                cbar_kws={"shrink": 0.7})
    ax.set_title(tr("Correlation matrix of the experiment features", "Matriz de correlación de las variables del experimento"))
    return fig_


# ========================================================================= main
def main() -> None:
    apply_theme("dark")
    print("generating figures ...")

    fig("params_hist", "params", "Parameter budget across all configurations",
        "Presupuesto de parámetros de todas las configuraciones", params_hist)
    fig("params_family_box", "params", tr("Parameter budget by experiment family", "Presupuesto de parámetros por familia de experimento"),
        "Parámetros por familia de experimentos", params_family_box)
    fig("params_composition", "params", tr("Where the parameters live, by family", "Dónde viven los parámetros, por familia"),
        "Dónde viven los parámetros, por familia", params_composition)
    fig("params_top20", "params", "The 20 largest configurations",
        "Las 20 configuraciones más grandes", lambda t: params_extremes(t, True))
    fig("params_smallest20", "params", "The 20 smallest configurations",
        "Las 20 configuraciones más pequeñas", lambda t: params_extremes(t, False))

    fig("loss_family_box", "training", "Final loss by family", "Pérdida final por familia", loss_family_box)
    fig("acc_family_box", "training", "Held-out score by family", "Métrica de validación por familia", acc_family_box)
    fig("loss_acc_scatter", "training", "Loss vs held-out score", "Pérdida vs métrica de validación", loss_acc_scatter)
    fig("loss_hist_pairs", "training", "Loss distribution of the hybrid pairs",
        "Distribución de pérdida de los pares híbridos", loss_hist_pairs)
    fig("acc_hist_pairs", "training", "Held-out accuracy distribution of the pairs",
        "Distribución de precisión de los pares", acc_hist_pairs)
    fig("params_loss_pairs", "training", "Parameters vs loss (pairs)",
        "Parámetros vs pérdida (pares)", params_loss_pairs)
    fig("params_acc_pairs", "training", "Parameters vs accuracy (pairs)",
        "Parámetros vs precisión (pares)", params_acc_pairs)
    fig("top20_loss", "training", "The 20 best hybrid pairs by loss",
        "Los 20 mejores pares por pérdida",
        lambda t: ranking_bar(t, "loss_final", 20, False, "The 20 best hybrid pairs by loss"))
    fig("bottom20_loss", "training", "The 20 worst hybrid pairs by loss",
        "Los 20 peores pares por pérdida",
        lambda t: ranking_bar(t, "loss_final", 20, True, "The 20 worst hybrid pairs by loss"))
    fig("top20_acc", "training", "The 20 best hybrid pairs by held-out accuracy",
        "Los 20 mejores pares por precisión",
        lambda t: ranking_bar(t, "test_acc", 20, True, "The 20 best hybrid pairs by held-out accuracy"))
    fig("marginal_a_loss", "training", "Marginal loss by first mixer",
        "Pérdida marginal del primer mixer",
        lambda t: marginal_bar(t, "mixer_a", "loss_final", False, "First mixer (A): mean loss across its pairs"))
    fig("marginal_b_loss", "training", "Marginal loss by second mixer",
        "Pérdida marginal del segundo mixer",
        lambda t: marginal_bar(t, "mixer_b", "loss_final", False, "Second mixer (B): mean loss across its pairs"))
    fig("marginal_a_acc", "training", "Marginal accuracy by first mixer",
        "Precisión marginal del primer mixer",
        lambda t: marginal_bar(t, "mixer_a", "test_acc", True, "First mixer (A): mean held-out accuracy"))
    fig("marginal_b_acc", "training", "Marginal accuracy by second mixer",
        "Precisión marginal del segundo mixer",
        lambda t: marginal_bar(t, "mixer_b", "test_acc", True, "Second mixer (B): mean held-out accuracy"))

    fig("curves_family", "dynamics", "Learning curves by family",
        "Curvas de aprendizaje por familia", curves_family)
    fig("curves_top10", "dynamics", "The ten best pairs over time",
        "Los diez mejores pares a lo largo del tiempo", curves_top10)
    fig("curves_quantiles", "dynamics", "Population curves with P10–P90 band",
        "Curvas de la población con banda P10–P90", curves_quantiles)
    fig("grad_groups", "dynamics", tr("Gradient pressure by parameter group", "Presión del gradiente por grupo de parámetros"),
        "Presión de gradiente por grupo", grad_groups)
    fig("grad_heatmap", "dynamics", "Gradient norms by family and group",
        "Normas de gradiente por familia y grupo", grad_heatmap)
    fig("clip_hist", "dynamics", "Gradient clipping behaviour",
        "Comportamiento del recorte de gradiente", clip_hist)
    fig("lr_curve", "dynamics", "Learning-rate schedule", "Calendario de learning rate", lr_curve)
    fig("speed_family_box", "dynamics", "Throughput by family",
        "Rendimiento por familia", speed_family_box)
    fig("step_vs_params", "dynamics", "Step time vs parameters",
        "Tiempo por paso vs parámetros", step_vs_params)
    fig("duration_hist", "dynamics", "Training wall-clock distribution",
        "Distribución del tiempo de entrenamiento", duration_hist)

    fig("matrix_params", "pairs", "A×B matrix of total parameters",
        "Matriz A×B de parámetros totales",
        lambda t: pair_matrix(t, "total", "viridis", "Parameter count for every A×B hybrid pair",
                              "parameters", fmt=".0f"))
    fig("matrix_loss", "pairs", "A×B matrix of final loss",
        "Matriz A×B de pérdida final",
        lambda t: pair_matrix(t, "loss_final", "crest_r", "Final loss for every A×B hybrid pair",
                              "final loss", fmt=".2f"))
    fig("matrix_acc", "pairs", "A×B matrix of held-out accuracy",
        "Matriz A×B de precisión",
        lambda t: pair_matrix(t, "test_acc", "crest", "Held-out accuracy for every A×B hybrid pair",
                              "accuracy", fmt=".3f"))
    fig("matrix_speed", "pairs", "A×B matrix of throughput",
        "Matriz A×B de rendimiento",
        lambda t: pair_matrix(t, "tok_s_avg", "rocket", "Training throughput for every A×B hybrid pair",
                              "tokens/s", fmt=".0f"))
    fig("matrix_duration", "pairs", "A×B matrix of training time",
        "Matriz A×B de tiempo de entrenamiento",
        lambda t: pair_matrix(t, "train_dur", "flare", "Training wall-clock for every A×B hybrid pair",
                              "seconds", fmt=".0f"))
    fig("synergy", "pairs", "Best and worst mixer interactions",
        "Mejores y peores interacciones entre mixers", synergy)
    fig("coverage", "pairs", "Coverage of the A×B grid",
        "Cobertura de la rejilla A×B", coverage)
    fig("pareto_speed_loss", "pairs", "Speed–quality Pareto front",
        "Frontera de Pareto velocidad–calidad", pareto_speed_loss)
    fig("top5_by_b", "pairs", "Top-5 hit rate by second mixer",
        "Top-5 por segundo mixer", top5_by_b)
    fig("top5_vs_loss", "pairs", "Inference quality vs training loss",
        "Calidad de inferencia vs pérdida", top5_vs_loss)

    fig("opt_loss", "ablations", "Optimizer ablation: final loss",
        "Ablación de optimizadores: pérdida final",
        lambda t: group_bar(t, "opt", "loss_final", "22 optimizer families on the same architecture",
                            color="#3fb950", sort_asc=True))
    fig("opt_speed", "ablations", "Optimizer ablation: step time",
        "Ablación de optimizadores: tiempo por paso", opt_speed)
    fig("norm_loss", "ablations", "Normalization ablation",
        "Ablación de normalización",
        lambda t: group_bar(t, "norm", "loss_final", "Six normalization strategies", color="#ffa657"))
    fig("pe_loss", "ablations", "Positional-encoding ablation",
        "Ablación de codificación posicional",
        lambda t: group_bar(t, "pe", "loss_final", "Twelve positional encodings", color="#ff7b72"))
    fig("trans_loss", "ablations", "Structural toggles",
        "Conmutadores estructurales",
        lambda t: group_bar(t, "trans", "loss_final", "Twenty architecture switches (BitNet, MoE, mHC, …)",
                            color="#d2991d", rotate=55))
    fig("vit_split", "ablations", "Vision tasks by objective",
        "Tareas de visión por objetivo", vit_split)
    fig("tokenizer_task", "ablations", "Tokenizer and task ablations",
        "Ablaciones de tokenizer y tarea", tokenizer_task)

    fig("inference_format", "deploy", "Standard vs quantized inference",
        "Inferencia estándar vs cuantizada", inference_format)
    fig("quant_delta_dist", "deploy", "Quantization delta distribution",
        "Distribución del delta de cuantización", quant_delta_dist)
    fig("quant_scatter", "deploy", "Standard vs quantized top-5",
        "Top-5 estándar vs cuantizado", quant_scatter)
    fig("deploy_duration", "deploy", "Pipeline stage cost",
        "Coste por etapa del pipeline", deploy_duration)
    fig("top5_by_family", "deploy", tr("Inference quality by family", "Calidad de inferencia por familia"),
        "Calidad de inferencia por familia", top5_by_family)

    fig("stage_status", "reliability", "Pipeline reliability",
        "Fiabilidad del pipeline", stage_status)
    fig("stage_duration_box", "reliability", "Stage durations",
        "Duraciones por etapa", stage_duration_box)
    fig("failures_family", "reliability", "Failures by family",
        "Fallos por familia", failures_family)
    fig("duplicate_fingerprints", "reliability", "Identical fingerprints",
        "Huellas idénticas", duplicate_fingerprints)

    fig("correlation", "correlation", "Feature correlation matrix",
        "Matriz de correlación de variables", correlation)

    (OUT / "registry.json").write_text(json.dumps(REGISTRY, indent=2, ensure_ascii=False))
    print(f"\n{len(REGISTRY)} figures x 2 themes -> {IMG}")


if __name__ == "__main__":
    main()
