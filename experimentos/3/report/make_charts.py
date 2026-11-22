"""Matplotlib chart builders for the Experiment 3 report.

Every builder takes the resolved theme dictionary (see :func:`apply_theme`) and
returns a ``matplotlib`` figure, so the same code renders the web PNGs and the
print PDFs. Labels are bilingual through :func:`tr`/:func:`set_lang`; the two
papers only differ in the language selected before the figure is built.

Usage:
    uv run --no-sync --with pandas --with numpy --with matplotlib python \
        experimentos/3/report/make_charts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

LANG = "en"
_CACHE: dict[str, pd.DataFrame] = {}

#: Grid used for the population curves (fraction of the run's own length).
GRID = np.linspace(0.0, 1.0, 51)


def tr(english: str, spanish: str) -> str:
    """Return ``english`` or ``spanish`` depending on the active language."""
    return english if LANG == "en" else spanish


def set_lang(lang: str) -> None:
    """Select the label language (``en`` or ``es``)."""
    global LANG
    LANG = lang


def apply_theme(name: str = "light") -> dict[str, str]:
    """Return the colour palette used by every builder."""
    if name == "dark":
        return {
            "face": "#0d1117",
            "fg": "#e6edf3",
            "muted": "#8b949e",
            "grid": "#30363d",
            "accent": "#58a6ff",
            "warm": "#f0883e",
            "bad": "#f85149",
            "good": "#3fb950",
            "purple": "#bc8cff",
        }
    return {
        "face": "#ffffff",
        "fg": "#1f2328",
        "muted": "#57606a",
        "grid": "#d8dee4",
        "accent": "#2456c4",
        "warm": "#d97706",
        "bad": "#cf222e",
        "good": "#1a7f37",
        "purple": "#8250df",
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load() -> dict[str, pd.DataFrame]:
    """Load (once) every artefact written by ``build_data.py``."""
    if _CACHE:
        return _CACHE
    master = pd.read_csv(C.DATA / "master.csv")
    master["family"] = master["family"].fillna("pair")
    epochs = pd.read_csv(C.DATA / "epochs.csv")
    epochs = epochs.merge(
        master[["run", "family", "task", "params_m"]], on="run", how="left"
    )
    _CACHE.update(
        master=master,
        epochs=epochs,
        blocks=pd.read_csv(C.DATA / "blocks.csv"),
        inference=pd.read_csv(C.DATA / "inference.csv", low_memory=False),
        inference_acc=pd.read_csv(C.DATA / "inference_acc.csv"),
        inference_pairs=pd.read_csv(C.DATA / "inference_pairs.csv"),
        params=pd.read_csv(C.DATA / "param_metrics.csv"),
    )
    return _CACHE


#: Runs shorter than this many epochs are smoke tests (5-epoch MoE variants,
#: one BitNet conversion and a dozen truncated pairs) and are excluded from
#: the outcome, dynamics, gradient, cost-per-run and correlation analyses.
MIN_EPOCHS = 100


def analysis(master: pd.DataFrame) -> pd.DataFrame:
    """Return the runs with a full-enough schedule for cross-run comparisons."""
    return master[master["final_epoch"].fillna(0) >= MIN_EPOCHS]


def families(master: pd.DataFrame, min_n: int = 3) -> list[str]:
    """Families present with at least ``min_n`` runs, in report order."""
    counts = master["family"].value_counts()
    return [f for f in C.FAMILIES if counts.get(f, 0) >= min_n]


def _style(ax, theme, grid_axis: str = "y"):
    """Apply the shared axis styling."""
    ax.set_facecolor(theme["face"])
    ax.figure.set_facecolor(theme["face"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], labelsize=8)
    ax.xaxis.label.set_color(theme["fg"])
    ax.yaxis.label.set_color(theme["fg"])
    ax.title.set_color(theme["fg"])
    ax.grid(axis=grid_axis, color=theme["grid"], linewidth=0.6, alpha=0.8) if grid_axis in ("x", "y", "both") else None
    ax.set_axisbelow(True)


def _family_ticks(ax, labels: list[str], rotate: int = 35):
    """Set the family labels on the x axis."""
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right")


def _bootstrap_ci(values, statistic=np.mean, n_boot: int = 400, seed: int = 7):
    """Return a 95% bootstrap confidence interval for ``statistic``."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = [statistic(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def _spearman(x, y) -> float:
    """Spearman rank correlation ignoring missing pairs."""
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 5:
        return float("nan")
    return float(frame["x"].corr(frame["y"], method="spearman"))


def _curves(epochs: pd.DataFrame, runs: list[str]) -> dict[str, np.ndarray]:
    """Interpolate every run's loss/accuracy curve onto ``GRID``.

    Args:
        epochs: Per-epoch table.
        runs: Run ids to interpolate.

    Returns:
        Mapping ``loss``/``accuracy`` -> ``(n_runs, len(GRID))`` array. Runs that
        stopped early are stretched over the common grid, so the grid measures
        progress through each run's own schedule rather than wall-clock steps.
    """
    subset = epochs[epochs["run"].isin(runs)]
    out = {"loss": [], "accuracy": []}
    for _, group in subset.groupby("run", sort=False):
        group = group.sort_values("epoch")
        if len(group) < 3:
            continue
        x = group["epoch"].to_numpy(dtype=float)
        x = (x - x.min()) / max(x.max() - x.min(), 1e-9)
        for key, column in (("loss", "loss"), ("accuracy", "accuracy")):
            y = group[column].to_numpy(dtype=float)
            if not np.isfinite(y).any():
                continue
            out[key].append(np.interp(GRID, x, y))
    return {
        key: np.vstack(values) if values else np.empty((0, len(GRID)))
        for key, values in out.items()
    }


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
def params_family_box(theme):
    """Total parameters by experiment family (log scale)."""
    master = load()["master"]
    fams = families(master)
    data = [master.loc[master["family"] == f, "params_m"].dropna() for f in fams]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.6,
        showfliers=False,
        medianprops={"color": theme["fg"], "linewidth": 1.2},
    )
    for patch, family in zip(box["boxes"], fams):
        patch.set_facecolor(C.FAMILY_COLOR.get(family, theme["accent"]))
        patch.set_alpha(0.75)
        patch.set_edgecolor(theme["muted"])
    ax.set_yscale("log")
    _family_ticks(ax, [f"{C.family_label(f, LANG)}\n(n={int((master.family == f).sum())})" for f in fams])
    ax.set_ylabel(tr("parameters (millions, log)", "parámetros (millones, log)"))
    ax.set_title(tr("Parameter budget by experiment family", "Presupuesto de parámetros por familia"))
    _style(ax, theme)
    fig.tight_layout()
    return fig


def params_composition(theme):
    """Where the parameters live, as median shares by family."""
    master = load()["master"]
    params = load()["params"]
    merged = params.merge(master[["run", "family"]], left_on="combo_id", right_on="run")
    parts = ["embeddings", "attention", "ffn", "head", "pos", "norm", "other"]
    colors = ["#2456c4", "#d97706", "#1a7f37", "#8250df", "#bf3989", "#8b949e", "#c2a83c"]
    fams = [f for f in families(master, min_n=4) if f != "bitnet"]
    shares = []
    totals = []
    for family in fams:
        rows = merged[merged["family"] == family]
        median = rows[parts].median()
        total = float(rows["total"].median())
        shares.append((median / total).to_numpy() * 100.0)
        totals.append(total / 1e6)
    shares = np.vstack(shares)
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    left = np.zeros(len(fams))
    part_label = {
        "embeddings": tr("embeddings", "embeddings"),
        "attention": tr("attention", "atención"),
        "ffn": "FFN",
        "head": tr("head", "cabeza"),
        "pos": tr("pos", "pos"),
        "norm": tr("norms", "norm."),
        "other": tr("other", "otros"),
    }
    for index, (part, color) in enumerate(zip(parts, colors)):
        ax.barh(
            range(len(fams)),
            shares[:, index],
            left=left,
            color=color,
            label=part_label[part],
            height=0.68,
        )
        left += shares[:, index]
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels(
        [f"{C.family_label(f, LANG)} ({t:.2f}M)" for f, t in zip(fams, totals)]
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel(tr("share of the parameter budget (%)", "proporción del presupuesto de parámetros (%)"))
    ax.set_title(tr("Parameter composition by family (median run)", "Composición de parámetros por familia (mediana)"))
    ax.legend(ncol=4, fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    _style(ax, theme, grid_axis="x")
    fig.tight_layout()
    return fig


def params_vs_loss(theme):
    """Final training loss against parameter count for the hybrid pairs."""
    master = load()["master"]
    sub = analysis(master)
    sub = sub[(sub["family"] == "pair") & sub["final_loss"].notna()].copy()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.scatter(sub["params_m"], sub["final_loss"], s=9, alpha=0.5, color=theme["accent"], linewidths=0)
    ax.set_xscale("log")
    rho = _spearman(np.log10(sub["params_m"]), sub["final_loss"])
    ax.text(
        0.02,
        0.97,
        tr(f"Spearman ρ = {rho:+.2f} (n={len(sub):,})", f"Spearman ρ = {rho:+.2f} (n={len(sub):,})"),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color=theme["muted"],
    )
    ax.set_xlabel(tr("parameters (millions, log)", "parámetros (millones, log)"))
    ax.set_ylabel(tr("final training loss (nats)", "pérdida final de entrenamiento (nats)"))
    ax.set_title(tr("Size helps, but explains little of the loss", "El tamaño ayuda, pero explica poco de la pérdida"))
    _style(ax, theme)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Training outcomes
# ---------------------------------------------------------------------------
def loss_family_box(theme):
    """Final training loss by family (tasks are not comparable across families)."""
    master = analysis(load()["master"])
    fams = families(master)
    data = [master.loc[master["family"] == f, "final_loss"].dropna() for f in fams]
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    box = ax.boxplot(
        data, patch_artist=True, widths=0.6, showfliers=False,
        medianprops={"color": theme["fg"], "linewidth": 1.2},
    )
    for patch, family in zip(box["boxes"], fams):
        patch.set_facecolor(C.FAMILY_COLOR.get(family, theme["accent"]))
        patch.set_alpha(0.75)
        patch.set_edgecolor(theme["muted"])
    ax.set_yscale("log")
    ax.set_ylim(0.2, 15)
    _family_ticks(ax, [C.family_label(f, LANG) for f in fams])
    ax.set_ylabel(tr("final loss (log)", "pérdida final (log)"))
    ax.set_title(tr("Final training loss by family", "Pérdida final de entrenamiento por familia"))
    _style(ax, theme)
    fig.tight_layout()
    return fig


def curves_quantiles(theme):
    """Population learning curves (median with a P10-P90 band)."""
    data = load()
    master, epochs = analysis(data["master"]), data["epochs"]
    runs = master.loc[(master.family == "pair") & (master.task == "mlm"), "run"].tolist()
    curves = _curves(epochs, runs)["loss"]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    if len(curves):
        p10, p50, p90 = np.percentile(curves, [10, 50, 90], axis=0)
        ax.fill_between(GRID * 100, p10, p90, color=theme["accent"], alpha=0.18, linewidth=0)
        ax.plot(GRID * 100, p50, color=theme["accent"], linewidth=2, label=tr("median", "mediana"))
        ax.plot(GRID * 100, curves.mean(axis=0), color=theme["warm"], linewidth=1.2, linestyle="--", label=tr("mean", "media"))
    ax.set_xlabel(tr("training progress (% of the run)", "progreso del entrenamiento (% de la ejecución)"))
    ax.set_ylabel(tr("training loss (nats)", "pérdida de entrenamiento (nats)"))
    ax.set_title(tr(f"Learning curves of {len(curves):,} hybrid pairs", f"Curvas de aprendizaje de {len(curves):,} pares híbridos"))
    ax.legend(frameon=False, fontsize=8)
    _style(ax, theme)
    fig.tight_layout()
    return fig


def curves_top10(theme):
    """The ten best hybrid pairs over their own training schedule."""
    data = load()
    master, epochs = analysis(data["master"]), data["epochs"]
    sub = master[(master.family == "pair") & (master.task == "mlm")].dropna(subset=["final_loss"])
    best = sub.nsmallest(10, "final_loss")["run"].tolist()
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    cmap = plt.get_cmap("viridis")
    for index, run in enumerate(best):
        curve = _curves(epochs, [run])["loss"]
        if not len(curve):
            continue
        ax.plot(GRID * 100, curve[0], linewidth=1.3, color=cmap(index / max(len(best) - 1, 1)))
        ax.annotate(
            C.tex_id(run).replace(r"\_", "_").replace("attn_", ""),
            (100, curve[0][-1]),
            fontsize=6,
            color=theme["muted"],
            va="center",
        )
    ax.set_xlabel(tr("training progress (% of the run)", "progreso del entrenamiento (% de la ejecución)"))
    ax.set_ylabel(tr("training loss (nats)", "pérdida de entrenamiento (nats)"))
    ax.set_title(tr("The ten best hybrid pairs over time", "Los diez mejores pares híbridos en el tiempo"))
    _style(ax, theme)
    fig.tight_layout()
    return fig


def convergence_speed(theme):
    """Fraction of the schedule needed to reach 90% of the loss drop, by family."""
    data = load()
    master, epochs = analysis(data["master"]), data["epochs"]
    records = []
    for run, group in epochs.groupby("run", sort=False):
        group = group.sort_values("epoch")
        loss = group["loss"].to_numpy(dtype=float)
        if len(loss) < 5 or not np.isfinite(loss).all():
            continue
        drop = loss[0] - loss[-1]
        if drop <= 0:
            continue
        target = loss[-1] + 0.1 * drop
        index = int(np.argmax(loss <= target))
        records.append(
            {
                "run": run,
                "fraction": index / max(len(loss) - 1, 1),
                "epochs_to_90": index,
                "total_epochs": len(loss),
            }
        )
    speed = pd.DataFrame(records).merge(master[["run", "family", "task"]], on="run")
    speed = speed[speed["task"] == "mlm"]
    fams = [f for f in families(master) if (speed.family == f).sum() >= 5]
    data_box = [speed.loc[speed.family == f, "fraction"].to_numpy() * 100 for f in fams]
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    if not data_box:
        ax.text(0.5, 0.5, tr("no data", "sin datos"), ha="center", va="center")
        _style(ax, theme)
        return fig
    box = ax.boxplot(
        data_box, patch_artist=True, widths=0.6, showfliers=False,
        medianprops={"color": theme["fg"], "linewidth": 1.2},
    )
    for patch, family in zip(box["boxes"], fams):
        patch.set_facecolor(C.FAMILY_COLOR.get(family, theme["accent"]))
        patch.set_alpha(0.75)
        patch.set_edgecolor(theme["muted"])
    _family_ticks(ax, [C.family_label(f, LANG) for f in fams])
    ax.set_ylabel(tr("schedule used to reach 90% of the drop (%)", "horario usado para el 90% de la caída (%)"))
    ax.set_title(tr("How early the loss is decided", "Cuán temprano se decide la pérdida"))
    _style(ax, theme)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gradient health per parameter block
# ---------------------------------------------------------------------------
def grad_groups_curves(theme):
    """Gradient norm per parameter block over training (median and IQR)."""
    data = load()
    master, epochs = analysis(data["master"]), data["epochs"]
    runs = master.loc[(master.family == "pair") & (master.task == "mlm"), "run"].tolist()
    subset = epochs[epochs["run"].isin(runs)]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    colors = plt.get_cmap("tab10")
    plotted = 0
    for index, group in enumerate(C.BLOCK_GROUPS):
        column = f"grad_norm_{group}"
        if column not in subset:
            continue
        curves = []
        for _, part in subset.groupby("run", sort=False):
            part = part.sort_values("epoch")
            if len(part) < 3:
                continue
            x = part["epoch"].to_numpy(dtype=float)
            x = (x - x.min()) / max(x.max() - x.min(), 1e-9)
            y = part[column].to_numpy(dtype=float)
            if np.isfinite(y).any():
                curves.append(np.interp(GRID, x, y))
        if not curves:
            continue
        curves = np.vstack(curves)
        median = np.median(curves, axis=0)
        if not np.isfinite(median).any() or np.nanmax(median) <= 1e-8:
            continue
        plotted += 1
        ax.plot(GRID * 100, median, linewidth=1.6, color=colors(index % 10), label=C.block_label(group, LANG))
    ax.set_yscale("log")
    ax.set_xlabel(tr("training progress (% of the run)", "progreso del entrenamiento (% de la ejecución)"))
    ax.set_ylabel(tr("gradient norm (log)", "norma del gradiente (log)"))
    ax.set_title(tr("Gradient pressure by parameter block", "Presión del gradiente por bloque de parámetros"))
    if plotted:
        ax.legend(ncol=1, fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _style(ax, theme)
    fig.tight_layout()
    return fig


def grad_heatmap(theme):
    """Median gradient norm per block relative to its initial value."""
    data = load()
    master, epochs = analysis(data["master"]), data["epochs"]
    runs = master.loc[(master.family == "pair") & (master.task == "mlm"), "run"].tolist()
    subset = epochs[epochs["run"].isin(runs)]
    bins = np.linspace(0, 1, 26)
    centers = (bins[:-1] + bins[1:]) / 2
    matrix = np.full((len(C.BLOCK_GROUPS), len(centers)), np.nan)
    for row, group in enumerate(C.BLOCK_GROUPS):
        column = f"grad_norm_{group}"
        if column not in subset:
            continue
        curves = []
        for _, part in subset.groupby("run", sort=False):
            part = part.sort_values("epoch")
            if len(part) < 3:
                continue
            x = part["epoch"].to_numpy(dtype=float)
            x = (x - x.min()) / max(x.max() - x.min(), 1e-9)
            y = part[column].to_numpy(dtype=float)
            if np.isfinite(y).all() and np.all(y > 0):
                curves.append(np.interp(centers, x, y))
        if not curves:
            continue
        curves = np.vstack(curves)
        ratio = curves / curves[:, [0]]
        matrix[row] = np.median(np.log10(np.clip(ratio, 1e-3, 1e3)), axis=0)
    cmap = LinearSegmentedColormap.from_list(
        "grad", [theme["bad"], theme["face"], theme["accent"]]
    )
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    image = ax.imshow(
        matrix, aspect="auto", cmap=cmap, vmin=-1.0, vmax=1.0,
        extent=(0, 100, len(C.BLOCK_GROUPS), 0),
        interpolation="nearest",
    )
    ax.set_yticks(np.arange(len(C.BLOCK_GROUPS)) + 0.5)
    ax.set_yticklabels([C.block_label(g, LANG) for g in C.BLOCK_GROUPS], fontsize=8)
    ax.set_xlabel(tr("training progress (%)", "progreso del entrenamiento (%)"))
    ax.set_title(tr("Gradient norm relative to the first epoch (log10)", "Norma del gradiente relativa a la primera época (log10)"))
    bar = fig.colorbar(image, ax=ax, pad=0.02)
    bar.set_label(tr("log10(ratio)", "log10(cociente)"), fontsize=8)
    bar.ax.tick_params(labelsize=7, colors=theme["muted"])
    _style(ax, theme, grid_axis="")
    fig.tight_layout()
    return fig


def grad_health(theme):
    """Clip ratio distribution and the gradient coefficient of variation."""
    data = load()
    master = analysis(data["master"])
    sub = master[(master["family"] == "pair") & master["task"].eq("mlm")]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    left, right = axes
    clip = sub["mean_clip_ratio"].dropna()
    left.hist(clip, bins=40, color=theme["accent"], alpha=0.8)
    left.axvline(1.0, color=theme["bad"], linestyle="--", linewidth=1.2)
    left.text(
        1.01, 0.95, tr("clip threshold", "umbral de recorte"),
        transform=left.get_xaxis_transform(), fontsize=7, color=theme["bad"], va="top",
    )
    left.set_xlabel(tr("mean clip ratio", "cociente de recorte medio"))
    left.set_ylabel(tr("runs", "ejecuciones"))
    left.set_title(tr("Clipping is routine, not exceptional", "El recorte es rutina, no excepción"), fontsize=10)
    _style(left, theme)

    cv = (sub["std_grad_norm"] / sub["mean_grad_norm"]).replace([np.inf, -np.inf], np.nan)
    right.scatter(cv, sub["final_loss"], s=8, alpha=0.4, color=theme["warm"], linewidths=0)
    rho = _spearman(cv, sub["final_loss"])
    right.text(0.02, 0.97, f"ρ = {rho:+.2f}", transform=right.transAxes, va="top", fontsize=8, color=theme["muted"])
    right.set_xlabel(tr("gradient-norm coefficient of variation", "coeficiente de variación de la norma"))
    right.set_ylabel(tr("final loss (nats)", "pérdida final (nats)"))
    right.set_title(tr("Stable gradients, lower loss", "Gradientes estables, menor pérdida"), fontsize=10)
    _style(right, theme)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Throughput, cost and resources
# ---------------------------------------------------------------------------
def time_vs_params(theme):
    """Training wall time against parameter count, with a log-log fit."""
    master = analysis(load()["master"])
    sub = master[(master["family"] == "pair") & master["task"].eq("mlm")]
    sub = sub.dropna(subset=["duration_h", "params_m"])
    sub = sub[(sub["duration_h"] > 0) & (sub["params_m"] > 0)]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.scatter(sub["params_m"], sub["duration_h"], s=9, alpha=0.45, color=theme["accent"], linewidths=0)
    if len(sub) > 5:
        x = np.log10(sub["params_m"].to_numpy())
        y = np.log10(sub["duration_h"].to_numpy())
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        grid = np.linspace(x.min(), x.max(), 50)
        ax.plot(10 ** grid, 10 ** (slope * grid + intercept), color=theme["bad"], linewidth=1.4)
        ax.text(
            0.02, 0.97,
            tr(f"log-log slope {slope:.2f}, R² = {r2:.2f}", f"pendiente log-log {slope:.2f}, R² = {r2:.2f}"),
            transform=ax.transAxes, va="top", fontsize=8, color=theme["muted"],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(tr("parameters (millions, log)", "parámetros (millones, log)"))
    ax.set_ylabel(tr("training wall time (hours, log)", "tiempo de pared (horas, log)"))
    ax.set_title(tr("Training time is nearly independent of size", "El tiempo de entrenamiento es casi independiente del tamaño"))
    _style(ax, theme)
    fig.tight_layout()
    return fig


def cost_per_mparam(theme):
    """Training minutes per million parameters, by family."""
    master = analysis(load()["master"])
    fams = families(master)
    data = [master.loc[master.family == f, "minutes_per_mparam"].dropna() for f in fams]
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    box = ax.boxplot(
        data, patch_artist=True, widths=0.6, showfliers=False,
        medianprops={"color": theme["fg"], "linewidth": 1.2},
    )
    for patch, family in zip(box["boxes"], fams):
        patch.set_facecolor(C.FAMILY_COLOR.get(family, theme["accent"]))
        patch.set_alpha(0.75)
        patch.set_edgecolor(theme["muted"])
    ax.set_yscale("log")
    _family_ticks(ax, [C.family_label(f, LANG) for f in fams])
    ax.set_ylabel(tr("training minutes per Mparam (log)", "minutos por Mparámetro (log)"))
    ax.set_title(tr("Normalised training cost", "Coste de entrenamiento normalizado"))
    _style(ax, theme)
    fig.tight_layout()
    return fig


def throughput_family(theme):
    """Logging throughput (tokens/s) by family."""
    master = analysis(load()["master"])
    sub = master[(master["task"].eq("mlm")) & master["tokens_per_sec"].notna()]
    fams = [f for f in families(master) if (sub.family == f).sum() >= 5]
    data = [sub.loc[sub.family == f, "tokens_per_sec"].dropna() for f in fams]
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    box = ax.boxplot(
        data, patch_artist=True, widths=0.6, showfliers=False,
        medianprops={"color": theme["fg"], "linewidth": 1.2},
    )
    for patch, family in zip(box["boxes"], fams):
        patch.set_facecolor(C.FAMILY_COLOR.get(family, theme["accent"]))
        patch.set_alpha(0.75)
        patch.set_edgecolor(theme["muted"])
    ax.set_yscale("log")
    _family_ticks(ax, [C.family_label(f, LANG) for f in fams])
    ax.set_ylabel(tr("tokens per second (log)", "tokens por segundo (log)"))
    ax.set_title(tr("Throughput by family", "Rendimiento por familia"))
    _style(ax, theme)
    fig.tight_layout()
    return fig


def compute_by_family(theme):
    """Total training wall time per family."""
    master = load()["master"]
    hours = master.groupby("family")["duration_h"].sum().sort_values(ascending=False)
    hours = hours[hours > 0]
    total = hours.sum()
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    bars = ax.bar(
        range(len(hours)),
        hours.values,
        color=[C.FAMILY_COLOR.get(f, theme["accent"]) for f in hours.index],
        width=0.68,
    )
    for bar, value in zip(bars, hours.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.02,
            f"{100 * value / total:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            color=theme["muted"],
        )
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels([C.family_label(f, LANG) for f in hours.index], rotation=35, ha="right")
    ax.set_ylabel(tr("training wall time (hours)", "tiempo de pared (horas)"))
    ax.set_title(tr("Where the compute went", "A dónde fue el cómputo"))
    _style(ax, theme)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Inference and quantization
# ---------------------------------------------------------------------------
def inference_format_accuracy(theme):
    """Held-out accuracy per loader/codec format (top-1 and top-5)."""
    acc = load()["inference_acc"]
    sub = acc[acc["task"].eq("mlm")]
    order = ["hf_none", "hf_q8_0_v2", "hf_q4_0_v2", "hf_i2_s_v2"]
    order = [f for f in order if (sub["deploy_format"] == f).any()]
    top1, top5, errors = [], [], []
    for fmt in order:
        values = sub.loc[sub["deploy_format"] == fmt, "accuracy"].dropna()
        top1.append(values.mean() * 100 if len(values) else np.nan)
        top5.append(sub.loc[sub["deploy_format"] == fmt, "top5"].mean() * 100)
        low, high = _bootstrap_ci(values * 100) if len(values) else (np.nan, np.nan)
        errors.append((top1[-1] - low, high - top1[-1]) if np.isfinite(low) else (0, 0))
    fig, ax = plt.subplots(figsize=(8.0, 3.9))
    x = np.arange(len(order))
    width = 0.38
    lower = [e[0] for e in errors]
    upper = [e[1] for e in errors]
    ax.bar(x - width / 2, top1, width, color=theme["accent"], label=tr("top-1", "top-1"), yerr=[lower, upper], error_kw={"elinewidth": 0.8, "ecolor": theme["muted"], "capsize": 2})
    ax.bar(x + width / 2, top5, width, color=theme["warm"], alpha=0.85, label=tr("top-5 hit", "acierto top-5"))
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("hf_", "") for f in order], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(tr("mean accuracy (%)", "exactitud media (%)"))
    ax.set_title(tr("Held-out accuracy by export format (MLM)", "Exactitud retenida por formato de exportación (MLM)"))
    ax.legend(frameon=False, fontsize=8)
    _style(ax, theme)
    fig.tight_layout()
    return fig


def quant_delta_dist(theme):
    """Per-combo accuracy delta against the unquantized export.

    Two panels: the mean delta with a 95\\% bootstrap interval, and the share of
    pairs that get worse, stay equal or improve. The median delta is zero for
    every codec (most predictions do not change), so a box plot would collapse;
    the stacked panel is what shows the asymmetry.
    """
    acc = load()["inference_acc"]
    sub = acc[acc["task"].eq("mlm")].copy()
    pivot = sub.pivot_table(index="combo_id", columns="deploy_format", values="accuracy")
    base = "hf_none"
    preferred = [
        "hf_q8_0_v2",
        "hf_q4_0_v2",
        "hf_i2_s_v2",
    ]
    formats = [f for f in preferred if f in pivot.columns]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1.05, 1]}
    )
    labels, means, errors, worse, equal, better = [], [], [], [], [], []
    for fmt in formats:
        delta = (pivot[fmt] - pivot[base]).dropna() * 100
        if len(delta) < 5:
            continue
        labels.append(fmt.replace("hf_", ""))
        means.append(delta.mean())
        low, high = _bootstrap_ci(delta)
        errors.append((means[-1] - low, high - means[-1]))
        worse.append(100 * (delta < -1e-9).mean())
        equal.append(100 * (delta.abs() <= 1e-9).mean())
        better.append(100 * (delta > 1e-9).mean())
    x = np.arange(len(labels))
    colors = [theme["good"] for _ in labels]
    left.bar(
        x, means, width=0.6, color=colors, alpha=0.75,
        yerr=[[e[0] for e in errors], [e[1] for e in errors]],
        error_kw={"elinewidth": 0.8, "ecolor": theme["muted"], "capsize": 2},
    )
    left.axhline(0, color=theme["fg"], linewidth=0.9)
    left.set_xticks(x)
    left.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    left.set_ylabel(tr("mean Δ top-1 (pp)", "Δ medio top-1 (pp)"))
    left.set_title(tr("Mean degradation (95% CI)", "Degradación media (IC 95%)"), fontsize=10)
    _style(left, theme)

    right.bar(x, worse, color=theme["bad"], label=tr("worse", "peor"), width=0.6)
    right.bar(x, equal, bottom=worse, color=theme["muted"], label=tr("equal", "igual"), width=0.6, alpha=0.6)
    right.bar(
        x, better, bottom=np.array(worse) + np.array(equal),
        color=theme["good"], label=tr("better", "mejor"), width=0.6,
    )
    right.set_xticks(x)
    right.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    right.set_ylabel(tr("share of pairs (%)", "proporción de pares (%)"))
    right.set_title(tr("Outcome of the comparison", "Resultado de la comparación"), fontsize=10)
    right.set_ylim(0, 118)
    right.legend(frameon=False, fontsize=7, ncol=3, loc="upper center")
    _style(right, theme)
    fig.tight_layout()
    return fig


def quant_agreement(theme):
    """Sample-level agreement with the unquantized predictions (v2 runtime)."""
    paired = load()["inference_pairs"]
    if paired.empty:
        paired = pd.DataFrame({"codec": [], "agreement": []})
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    order = [c for c in ("q8_0", "q4_0", "i2_s") if (paired["codec"] == c).any()]
    values = [paired.loc[paired.codec == c, "agreement"].mean() * 100 for c in order]
    x = np.arange(len(order))
    ax.bar(x, values, color=[theme["good"], theme["warm"], theme["bad"]][: len(order)], width=0.55)
    for index, value in enumerate(values):
        ax.text(index, value + 0.6, f"{value:.2f}%", ha="center", fontsize=8, color=theme["fg"])
    ax.set_xticks(x)
    ax.set_xticklabels([C.codec_label(c, LANG) for c in order])
    ax.set_ylim(0, 105)
    ax.set_ylabel(tr("predictions identical to baseline (%)", "predicciones idénticas a la base (%)"))
    ax.set_title(tr("Sample-level agreement after quantization", "Acuerdo por muestra tras cuantizar"))
    _style(ax, theme)
    fig.tight_layout()
    return fig


def quant_delta_vs_params(theme):
    """i2_s accuracy delta against the parameter count."""
    acc = load()["inference_acc"]
    master = load()["master"]
    sub = acc[acc["task"].eq("mlm")].copy()
    pivot = sub.pivot_table(index="combo_id", columns="deploy_format", values="accuracy")
    if "hf_i2_s_v2" not in pivot or "hf_none" not in pivot:
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        ax.text(0.5, 0.5, "no data", ha="center")
        return fig
    delta = (pivot["hf_i2_s_v2"] - pivot["hf_none"]).dropna() * 100
    merged = master.set_index("run")[["params_m", "family"]].join(delta.rename("delta"))
    merged = merged.dropna()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.scatter(merged["params_m"], merged["delta"], s=8, alpha=0.45, color=theme["bad"], linewidths=0)
    rho = _spearman(merged["params_m"], merged["delta"])
    ax.axhline(0, color=theme["fg"], linewidth=0.9)
    ax.text(0.02, 0.03, tr(f"Spearman ρ = {rho:+.2f} (n={len(merged):,})", f"Spearman ρ = {rho:+.2f} (n={len(merged):,})"), transform=ax.transAxes, fontsize=8, color=theme["muted"])
    ax.set_xscale("log")
    ax.set_xlabel(tr("parameters (millions, log)", "parámetros (millones, log)"))
    ax.set_ylabel(tr("Δ top-1 (pp), i2_s vs none", "Δ top-1 (pp), i2_s vs none"))
    ax.set_title(tr("Who pays for 2-bit weights", "Quién paga los pesos de 2 bits"))
    _style(ax, theme)
    fig.tight_layout()
    return fig


def train_vs_inference(theme):
    """Final training loss against held-out top-5 hit rate (MLM pairs)."""
    acc = load()["inference_acc"]
    master = load()["master"]
    base = acc[(acc["deploy_format"] == "hf_none") & acc["task"].eq("mlm")].set_index("combo_id")
    merged = master.set_index("run").join(base[["accuracy", "top5"]], how="inner")
    merged = merged[merged["family"].eq("pair")].dropna(subset=["final_loss", "top5"])
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.scatter(merged["final_loss"], merged["top5"] * 100, s=9, alpha=0.45, color=theme["purple"], linewidths=0)
    rho = _spearman(merged["final_loss"], merged["top5"])
    ax.text(0.02, 0.97, tr(f"Spearman ρ = {rho:+.2f} (n={len(merged):,})", f"Spearman ρ = {rho:+.2f} (n={len(merged):,})"), transform=ax.transAxes, va="top", fontsize=8, color=theme["muted"])
    ax.set_xlabel(tr("final training loss (nats)", "pérdida final (nats)"))
    ax.set_ylabel(tr("held-out top-5 hit rate (%)", "acierto top-5 retenido (%)"))
    ax.set_title(tr("Training loss barely ranks held-out quality", "La pérdida apenas ordena la calidad retenida"))
    _style(ax, theme)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Reliability and correlation
# ---------------------------------------------------------------------------
def stage_status(theme):
    """Pipeline status counts per stage."""
    results = pd.read_csv(C.CSVS / "results.json.csv", low_memory=False)
    results = results[~results["category"].eq("deploy")]
    counts = results.groupby(["category", "status"]).size().unstack(fill_value=0)
    order = ["training", "transformers-export", "inference-eval", "bitnet-gguf"]
    counts = counts.reindex([c for c in order if c in counts.index])
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    left = np.zeros(len(counts))
    palette = {"OK": theme["good"], "FAILED": theme["bad"], "SKIPPED_NO_INFER": theme["muted"]}
    for status in counts.columns:
        ax.barh(range(len(counts)), counts[status], left=left, color=palette.get(status, theme["muted"]), label=status)
        left += counts[status].to_numpy()
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index)
    ax.set_xscale("log")
    ax.set_xlabel(tr("stage records (log)", "registros de etapa (log)"))
    ax.set_title(tr("Pipeline reliability by stage", "Fiabilidad del pipeline por etapa"))
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower right")
    _style(ax, theme, grid_axis="x")
    fig.tight_layout()
    return fig


def correlation(theme):
    """Spearman correlation matrix of the measured features (hybrid pairs)."""
    master = analysis(load()["master"])
    sub = master[(master["family"] == "pair") & master["task"].eq("mlm")].copy()
    features = {
        "params_m": sub["params_m"],
        "final_loss": sub["final_loss"],
        "mean_loss": sub["mean_loss"],
        "final_accuracy": sub["final_accuracy"],
        "grad_norm": sub["mean_grad_norm"],
        "step_time_ms": sub["mean_step_time_ms"],
        "duration_h": sub["duration_h"],
        "tokens_per_sec": sub["tokens_per_sec"],
        "epochs": sub["final_epoch"],
    }
    frame = pd.DataFrame(features).dropna()
    corr = frame.corr(method="spearman")
    labels = list(corr.columns)
    cmap = LinearSegmentedColormap.from_list("corr", [theme["bad"], theme["face"], theme["accent"]])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.to_numpy()[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=6, color=theme["fg"])
    ax.set_title(tr("Spearman correlations (hybrid pairs, MLM)", "Correlaciones de Spearman (pares híbridos, MLM)"))
    bar = fig.colorbar(image, ax=ax, pad=0.02)
    bar.ax.tick_params(labelsize=7, colors=theme["muted"])
    _style(ax, theme, grid_axis="")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# The mixer atlas: super matrix, roles, synergy, Pareto, per-B quality
# ---------------------------------------------------------------------------
def _mixer_order(pairs: pd.DataFrame) -> list[str]:
    """Registry (generation) order of the mixers, reconstructed from the pairs.

    The axes follow the order used to generate the pairs (A before B); an
    alphabetical order would push half of the pairs below the diagonal and
    make the upper-triangle atlas look perforated.
    """
    edges = [(a, b) for a, b in zip(pairs["pair_a"], pairs["pair_b"]) if a and b]
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
    return order if len(order) == len(nodes) else sorted(nodes)


def matrix_loss(theme):
    """The 54x54 super matrix: final loss of every hybrid pair (A x B)."""
    master = load()["master"]
    pairs = master[master["family"] == "pair"].copy()
    mixers = _mixer_order(pairs)
    index = {mixer: position for position, mixer in enumerate(mixers)}
    grid = np.full((len(mixers), len(mixers)), np.nan)
    for _, row in pairs.iterrows():
        a, b = row.get("pair_a"), row.get("pair_b")
        if a in index and b in index and pd.notna(row.get("final_loss")):
            grid[index[a], index[b]] = row["final_loss"]
    cmap = LinearSegmentedColormap.from_list(
        "atlas", [theme["good"], "#d2991d", theme["bad"]]
    )
    cmap.set_bad(color=theme["grid"])
    fig, ax = plt.subplots(figsize=(10.6, 9.0))
    masked = np.ma.masked_invalid(grid)
    image = ax.imshow(masked, aspect="equal", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(len(mixers)))
    ax.set_xticklabels(mixers, rotation=90, fontsize=6.2, color=theme["fg"])
    ax.set_yticks(range(len(mixers)))
    ax.set_yticklabels(mixers, fontsize=6.2, color=theme["fg"])
    ax.set_xlabel(tr("second mixer (B)", "segundo mixer (B)"))
    ax.set_ylabel(tr("first mixer (A)", "primer mixer (A)"))
    ax.set_title(
        tr(
            "The mixer atlas: final loss of the 1,431 pairs",
            "El atlas de mixers: pérdida final de los 1,431 pares",
        )
    )
    bar = fig.colorbar(image, ax=ax, pad=0.015, fraction=0.046)
    bar.set_label(tr("final loss (nats)", "pérdida final (nats)"), fontsize=8)
    bar.ax.tick_params(labelsize=7, colors=theme["muted"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def roles_scatter(theme):
    """Median loss of every mixer in the A role against the B role."""
    master = analysis(load()["master"])
    pairs = master[master["family"] == "pair"].dropna(subset=["final_loss"])
    by_a = pairs.groupby("pair_a")["final_loss"].median()
    by_b = pairs.groupby("pair_b")["final_loss"].median()
    common = sorted(set(by_a.index) & set(by_b.index))
    x = by_a[common].to_numpy()
    y = by_b[common].to_numpy()
    lo = float(min(x.min(), y.min())) - 0.05
    hi = float(max(x.max(), y.max())) + 0.05
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color=theme["muted"])
    ax.scatter(x, y, s=26, alpha=0.8, color=theme["accent"], linewidths=0)
    for mixer, xv, yv in zip(common, x, y):
        if mixer in ("conformer", "mamba", "iha", "mtla", "ccgqa", "engram"):
            ax.annotate(
                mixer, (xv, yv), fontsize=7, color=theme["fg"],
                xytext=(4, 4), textcoords="offset points",
            )
    rho = _spearman(x, y)
    ax.text(0.02, 0.97, tr(f"Spearman ρ = {rho:+.2f}", f"Spearman ρ = {rho:+.2f}"),
            transform=ax.transAxes, va="top", fontsize=8, color=theme["muted"])
    ax.set_xlabel(tr("median loss as first mixer (A)", "pérdida mediana como primer mixer (A)"))
    ax.set_ylabel(tr("median loss as second mixer (B)", "pérdida mediana como segundo mixer (B)"))
    ax.set_title(tr("The role effect: A vs B", "El efecto del rol: A vs B"))
    _style(ax, theme)
    fig.tight_layout()
    return fig


def top5_by_b(theme):
    """Mean held-out top-5 hit rate of the pairs grouped by second mixer."""
    data = load()
    master = analysis(data["master"])
    acc = data["inference_acc"]
    base = acc[(acc["deploy_format"] == "hf_none") & acc["task"].eq("mlm")]
    pairs = master[(master["family"] == "pair")].merge(
        base[["combo_id", "top5"]], left_on="run", right_on="combo_id"
    ).dropna(subset=["top5", "final_loss"])
    grouped = pairs.groupby("pair_b")["top5"].agg(["mean", "size"])
    grouped = grouped[grouped["size"] >= 20].sort_values("mean")
    labels = list(grouped.index)
    values = grouped["mean"].to_numpy() * 100
    counts = grouped["size"].to_numpy()
    colors = [theme["warm"] if name == "conformer" else theme["accent"] for name in labels]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.barh(range(len(labels)), values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"{name} (n={n})" for name, n in zip(labels, counts)], fontsize=8)
    ax.set_xlabel(tr("mean held-out top-5 (%)", "top-5 retenido medio (%)"))
    ax.set_title(
        tr(
            "Held-out quality by second mixer — the best-loss B is not the best-quality B",
            "Calidad retenida por segundo mixer — el mejor B en pérdida no es el mejor en calidad",
        ), fontsize=10)
    _style(ax, theme, grid_axis="x")
    fig.tight_layout()
    return fig


def synergy(theme):
    """Best and worst mixer interactions by synergy (marginals minus pair)."""
    master = analysis(load()["master"])
    pairs = master[master["family"] == "pair"].dropna(subset=["final_loss"]).copy()
    means = {}
    for _, row in pairs.iterrows():
        means.setdefault(row["pair_a"], []).append(row["final_loss"])
        means.setdefault(row["pair_b"], []).append(row["final_loss"])
    means = {name: float(np.mean(values)) for name, values in means.items()}
    pairs["synergy"] = pairs.apply(
        lambda row: (means[row["pair_a"]] + means[row["pair_b"]]) / 2 - row["final_loss"],
        axis=1,
    )
    top = pairs.nlargest(8, "synergy")
    bottom = pairs.nsmallest(8, "synergy").iloc[::-1]
    frame = pd.concat([bottom, top])
    labels = [run.replace("attn_", "").replace("__", " + ") for run in frame["run"]]
    values = frame["synergy"].to_numpy()
    colors = [theme["good"] if v > 0 else theme["bad"] for v in values]
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.barh(range(len(frame)), values, color=colors, alpha=0.82)
    ax.axvline(0, color=theme["fg"], linewidth=0.9)
    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(tr("synergy: expected minus observed loss (nats)", "sinergia: pérdida esperada menos observada (nats)"))
    ax.set_title(tr("Interaction beats marginals: extreme synergies", "La interacción supera a los marginales: sinergias extremas"))
    _style(ax, theme, grid_axis="x")
    fig.tight_layout()
    return fig


def pareto(theme):
    """Speed--quality Pareto front of the hybrid pairs (MLM)."""
    master = analysis(load()["master"])
    pairs = master[(master["family"] == "pair") & master["task"].eq("mlm")]
    pairs = pairs.dropna(subset=["final_loss", "duration_h"])
    ordered = pairs.sort_values("duration_h")
    front, best = [], np.inf
    for _, row in ordered.iterrows():
        if row["final_loss"] < best:
            front.append(row)
            best = row["final_loss"]
    front = pd.DataFrame(front)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.scatter(pairs["duration_h"], pairs["final_loss"], s=8, alpha=0.35,
               color=theme["muted"], linewidths=0, label=tr("pairs", "pares"))
    ax.plot(front["duration_h"], front["final_loss"], color=theme["bad"],
            linewidth=1.2, marker="o", markersize=3.5, label=tr("Pareto front", "frontera de Pareto"))
    for _, row in front.iterrows():
        name = row["run"].replace("attn_", "").replace("__", " + ")
        if row["final_loss"] < 8.7 or row["duration_h"] < 0.7:
            ax.annotate(name, (row["duration_h"], row["final_loss"]),
                        fontsize=6.5, color=theme["fg"],
                        xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel(tr("training wall time (hours)", "tiempo de pared (horas)"))
    ax.set_ylabel(tr("final loss (nats)", "pérdida final (nats)"))
    ax.set_title(tr(f"The speed--quality Pareto front ({len(front)} of {len(pairs):,} pairs)",
                    f"La frontera de Pareto velocidad-calidad ({len(front)} de {len(pairs):,} pares)"))
    ax.legend(frameon=False, fontsize=8)
    _style(ax, theme)
    fig.tight_layout()
    return fig


def optimizer_panel(theme):
    """Per-optimizer loss, gradient pressure, step time and memory (4 panels)."""
    master = analysis(load()["master"])
    opt = master[master["family"] == "optimizer"].dropna(subset=["final_loss"]).copy()
    opt = opt.sort_values("final_loss")
    names = list(opt["optimizer"])
    y = range(len(names))
    panels = [
        ("final_loss", tr("final loss (nats)", "pérdida final (nats)"), theme["accent"]),
        ("mean_grad_norm", tr("mean grad norm (post-clip)", "norma media del gradiente (post-recorte)"), theme["purple"]),
        ("mean_step_time_ms", tr("mean step time (ms)", "tiempo medio por paso (ms)"), theme["warm"]),
        ("mean_rss_mb", tr("mean resident memory (MB)", "memoria residente media (MB)"), theme["good"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 10.4))
    for ax, (column, label, color) in zip(axes.flat, panels):
        ax.barh(y, opt[column].to_numpy(), color=color, alpha=0.82)
        ax.set_yticks(list(y))
        ax.set_yticklabels(names, fontsize=6.4)
        ax.set_xlabel(label, fontsize=8)
        ax.invert_yaxis()
        _style(ax, theme, grid_axis="x")
    fig.suptitle(
        tr(
            "Optimizer ablation: loss, gradient pressure, CPU and memory (sorted by loss)",
            "Ablación de optimizadores: pérdida, gradiente, CPU y memoria (orden por pérdida)",
        ), fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig


def ablation_loss(theme):
    """Normalization, text-PE and structural-switch ablations (3 panels)."""
    master = analysis(load()["master"])
    groups = [
        ("norm", tr("normalization", "normalización")),
        ("pe", tr("positional encoding (text)", "codificación posicional (texto)")),
        ("switch", tr("structural switches", "interruptores estructurales")),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 5.6))
    for ax, (family, title) in zip(axes, groups):
        rows = master[(master["family"] == family)].dropna(subset=["final_loss"])
        rows = rows.sort_values("final_loss", ascending=False)
        labels = [run.replace(f"{family}_", "").replace("trans_", "").replace("_", " ")
                  for run in rows["run"]]
        ax.barh(range(len(rows)), rows["final_loss"], color=C.FAMILY_COLOR[family],
                alpha=0.82)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels, fontsize=7)
        lo = float(rows["final_loss"].min())
        ax.set_xlim(lo - 0.25, float(rows["final_loss"].max()) + 0.25)
        ax.axvline(lo, color=theme["muted"], linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(tr("final loss (nats)", "pérdida final (nats)"), fontsize=8)
        _style(ax, theme, grid_axis="x")
    fig.suptitle(tr("Component ablations on the shared backbone (smoke runs excluded)",
                    "Ablaciones de componentes sobre el esqueleto compartido (sin pruebas de humo)"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


#: Builder registry used by the PNG and the paper-figure scripts.
FIGURES = [
    ("params_family_box", params_family_box, 0.75),
    ("params_composition", params_composition, 0.75),
    ("params_vs_loss", params_vs_loss, 0.85),
    ("loss_family_box", loss_family_box, 0.75),
    ("curves_quantiles", curves_quantiles, 0.85),
    ("curves_top10", curves_top10, 0.85),
    ("convergence_speed", convergence_speed, 0.78),
    ("matrix_loss", matrix_loss, 0.95),
    ("roles_scatter", roles_scatter, 0.9),
    ("top5_by_b", top5_by_b, 0.82),
    ("synergy", synergy, 0.82),
    ("pareto", pareto, 0.85),
    ("grad_groups_curves", grad_groups_curves, 0.82),
    ("grad_heatmap", grad_heatmap, 0.82),
    ("grad_health", grad_health, 0.72),
    ("optimizer_panel", optimizer_panel, 0.72),
    ("ablation_loss", ablation_loss, 0.8),
    ("time_vs_params", time_vs_params, 0.85),
    ("cost_per_mparam", cost_per_mparam, 0.75),
    ("throughput_family", throughput_family, 0.78),
    ("compute_by_family", compute_by_family, 0.8),
    ("inference_format_accuracy", inference_format_accuracy, 0.78),
    ("quant_delta_dist", quant_delta_dist, 0.8),
    ("quant_agreement", quant_agreement, 0.9),
    ("quant_delta_vs_params", quant_delta_vs_params, 0.85),
    ("train_vs_inference", train_vs_inference, 0.85),
    ("stage_status", stage_status, 0.85),
    ("correlation", correlation, 0.85),
]


def main() -> None:
    """Render every figure as a light-theme PNG for the web report."""
    C.IMG.mkdir(parents=True, exist_ok=True)
    set_lang("en")
    theme = apply_theme("light")
    for name, builder, _ in FIGURES:
        figure = builder(theme)
        figure.savefig(C.IMG / f"{name}.png", dpi=160, bbox_inches="tight", facecolor=theme["face"])
        plt.close(figure)
        print(f"  ok out/img/{name}.png")


if __name__ == "__main__":
    main()
