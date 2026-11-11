#!/usr/bin/env python3
"""Render the pedagogical charts of the beginner tutorial (Annex 1).

Every chart is self-contained: no data files, fixed numbers, and the light
print theme of ``optimizers/tutorial/make_tutorial_figures.py``. Each figure
is written twice -- a vector PDF for LaTeX and an SVG for the website render,
which indexes figures by stem (see ``paper/flatten_tex.py``).

Each chart exists in an English and a Spanish flavour: ``<name>.pdf`` is the
English one used by ``docs/paper``, and ``<name>-es.pdf`` is the Spanish one
used by ``docs/paper-es`` (the sentences in the annex differ by language, so
the beginner figures are the only bilingual set in the paper).

Usage:
    uv run --with matplotlib --with seaborn --with numpy \\
        python docs/paper/figures/tutorial/make_tutorial_figures.py [NAME ...]

Passing one or more figure names renders only those (handy while iterating).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- style
FACE = "#ffffff"
AXES = "#ffffff"
TEXT = "#1f2328"
MUTED = "#59636e"
GRID = "#d8dee4"
SPINE = "#d0d7de"

BLUE = "#0969da"
GREEN = "#1a7f37"
ORANGE = "#bc4c00"
RED = "#cf222e"
PURPLE = "#8250df"
GREY = "#6e7781"

#: White-to-blue ramp for the attention-weight matrices.
HEAT = LinearSegmentedColormap.from_list(
    "heat", ["#ffffff", "#eaf2fb", "#c9ddf5", "#9ec7ec", "#6ea6dd"]
)

#: One stable color per word, shared by the softmax chart, so a word keeps
#: its identity across panels and languages (position-wise).
WORD_COLORS = [GREY, BLUE, PURPLE, GREEN, ORANGE, RED, "#953800"]


def theme() -> None:
    """Apply the shared print theme (called once per figure)."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": FACE,
            "savefig.facecolor": FACE,
            "axes.facecolor": AXES,
            "axes.edgecolor": SPINE,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.7,
            "grid.linewidth": 0.6,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.facecolor": AXES,
            "legend.edgecolor": SPINE,
            "legend.fontsize": 7.5,
            "legend.framealpha": 0.92,
            "font.family": "DejaVu Sans",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


# --------------------------------------------------------------------------- data
#: Illustrative attention weights for the two tutorial sentences. Rows are the
#: asking word (query), columns the word being attended to (key); every row
#: sums to 1.0. These are the same numbers quoted in the annex prose and in
#: ``tutorial-attention-steps.mmd``.
EN_CTX1 = {
    "words": ["The", "cat", "eats", "fish"],
    "M": np.array(
        [
            [0.25, 0.55, 0.10, 0.10],
            [0.05, 0.10, 0.60, 0.25],
            [0.05, 0.55, 0.05, 0.35],
            [0.05, 0.50, 0.35, 0.10],
        ]
    ),
    "focus": "eats",
    "title": "(a) “The cat eats fish”",
}
EN_CTX2 = {
    "words": ["The", "restaurant", "eats", "into", "profit", "margins"],
    "M": np.array(
        [
            [0.25, 0.50, 0.10, 0.05, 0.05, 0.05],
            [0.05, 0.10, 0.55, 0.05, 0.15, 0.10],
            [0.03, 0.30, 0.03, 0.04, 0.25, 0.35],
            [0.05, 0.10, 0.30, 0.10, 0.25, 0.20],
            [0.03, 0.10, 0.20, 0.05, 0.15, 0.47],
            [0.03, 0.15, 0.25, 0.05, 0.40, 0.12],
        ]
    ),
    "focus": "eats",
    "title": "(b) “The restaurant eats into profit margins”",
}
ES_CTX1 = {
    "words": ["El", "gato", "come", "pescado"],
    "M": EN_CTX1["M"],
    "focus": "come",
    "title": "(a) “El gato come pescado”",
}
ES_CTX2 = {
    "words": ["El", "restaurante", "come", "los", "márgenes", "de", "beneficio"],
    "M": np.array(
        [
            [0.25, 0.50, 0.10, 0.05, 0.04, 0.03, 0.03],
            [0.05, 0.10, 0.55, 0.05, 0.10, 0.05, 0.10],
            [0.03, 0.30, 0.03, 0.04, 0.30, 0.03, 0.27],
            [0.10, 0.10, 0.30, 0.10, 0.20, 0.10, 0.10],
            [0.03, 0.15, 0.25, 0.05, 0.12, 0.10, 0.30],
            [0.05, 0.10, 0.20, 0.10, 0.30, 0.05, 0.20],
            [0.03, 0.20, 0.25, 0.03, 0.35, 0.04, 0.10],
        ]
    ),
    "focus": "come",
    "title": "(b) “El restaurante come los márgenes de beneficio”",
}

#: Compatibility scores for the query ``eats``/``come`` and their softmax
#: weights (5% / 55% / 5% / 35%) -- the same row highlighted in the heatmaps.
EN_SOFTMAX = {
    "words": ["The", "cat", "eats", "fish"],
    "scores": [1.0, 3.4, 1.0, 2.95],
}
ES_SOFTMAX = {
    "words": ["El", "gato", "come", "pescado"],
    "scores": [1.0, 3.4, 1.0, 2.95],
}


def _softmax(x: list[float]) -> np.ndarray:
    e = np.exp(np.asarray(x, dtype=float))
    return e / e.sum()


# ------------------------------------------------------------------ 1. heatmap
def _heat_panel(ax: plt.Axes, spec: dict) -> None:
    """One annotated attention matrix with the focus row highlighted."""
    words, M = spec["words"], spec["M"]
    n = len(words)
    ax.imshow(M, cmap=HEAT, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(words, rotation=30, ha="right", fontsize=7.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(words, fontsize=7.6)
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=FACE, linewidth=1.8)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="major", visible=False)
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{M[i, j]:.0%}", ha="center", va="center",
                fontsize=7.0, color=TEXT if M[i, j] < 0.75 else "#ffffff",
            )
    row = words.index(spec["focus"])
    ax.add_patch(
        Rectangle((-.5, row - .5), n, 1.0, fill=False, edgecolor=ORANGE,
                  lw=1.8, zorder=5)
    )
    ax.set_title(spec["title"], fontsize=8.4, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def tut_attention_heatmap(lang: str = "en") -> plt.Figure:
    """The same verb, two contexts: full attention matrices side by side."""
    ctx1 = EN_CTX1 if lang == "en" else ES_CTX1
    ctx2 = EN_CTX2 if lang == "en" else ES_CTX2
    fig, axes = plt.subplots(
        1, 2, figsize=(6.6, 3.0),
        gridspec_kw={"width_ratios": [len(ctx1["words"]), len(ctx2["words"])]},
    )
    _heat_panel(axes[0], ctx1)
    _heat_panel(axes[1], ctx2)
    axes[0].set_ylabel("asking word (query)")
    axes[0].set_xlabel("attends to (key)")
    axes[1].set_xlabel("attends to (key)")
    fig.tight_layout(pad=0.3)
    return fig


def tut_attention_heatmap_es() -> plt.Figure:
    return tut_attention_heatmap("es")


# ------------------------------------------------------------------ 2. softmax
def tut_softmax_focus(lang: str = "en") -> plt.Figure:
    """Raw compatibility scores become an attention budget that sums to 100%."""
    spec = EN_SOFTMAX if lang == "en" else ES_SOFTMAX
    words, scores = spec["words"], spec["scores"]
    weights = _softmax(scores)
    colors = WORD_COLORS[: len(words)]

    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(6.4, 3.1), gridspec_kw={"height_ratios": [2.5, 1.0]},
    )

    top.bar(words, scores, color=colors, width=0.62, zorder=3)
    for i, (w, s) in enumerate(zip(words, scores)):
        top.text(i, s + 0.10, f"{s:.2f}", ha="center", fontsize=7.4, color=TEXT)
    top.set_ylim(0, max(scores) * 1.22)
    label = "compatibility score" if lang == "en" else "puntaje de compatibilidad"
    top.set_ylabel(label)
    title = (
        "(a) dot products q·k for the query “eats”"
        if lang == "en"
        else "(a) productos punto q·k para la consulta “come”"
    )
    top.set_title(title, fontsize=8.4, pad=4)

    left = 0.0
    for w, p, c in zip(words, weights, colors):
        bot.barh([0], [p], left=left, color=c, height=0.55, zorder=3)
        bot.text(
            left + p / 2, 0, f"{p:.0%}", ha="center", va="center",
            fontsize=7.4, color="#ffffff" if p > 0.12 else TEXT,
        )
        bot.text(left + p / 2, 0.52, w, ha="center", va="bottom",
                 fontsize=7.4, color=c if p > 0.06 else MUTED)
        left += p
    bot.set_xlim(0, 1)
    bot.set_ylim(-0.6, 1.0)
    bot.set_yticks([])
    bot.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    bot.grid(axis="x", color=GRID, lw=0.6)
    bot.grid(axis="y", visible=False)
    title2 = (
        "(b) after softmax: a 100% attention budget"
        if lang == "en"
        else "(b) tras softmax: un presupuesto de atención del 100%"
    )
    bot.set_title(title2, fontsize=8.4, pad=4)
    fig.tight_layout(pad=0.3)
    return fig


def tut_softmax_focus_es() -> plt.Figure:
    return tut_softmax_focus("es")


# ---------------------------------------------------------------- 3. complexity
def tut_complexity_growth(lang: str = "en") -> plt.Figure:
    """Word-to-word comparisons: dense n² vs sparse n·log n vs linear n."""
    n = np.logspace(1, np.log10(2.6e5), 400)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(n, n**2, color=BLUE, lw=1.8,
            label="dense attention:  n² comparisons")
    ax.plot(n, n * np.log2(n), color=GREEN, lw=1.8,
            label="sparse variants:  n·log₂ n")
    ax.plot(n, n, color=ORANGE, lw=1.8, label="linear / recurrent:  n")

    ax.plot([1000], [1e6], "o", ms=5, color=BLUE, zorder=6)
    if lang == "en":
        note = "1,000 words → 1,000,000 comparisons"
        note2 = "sparse: only ~10,000"
    else:
        note = "1.000 palabras → 1.000.000 de comparaciones"
        note2 = "dispersa: solo ~10.000"
    ax.annotate(
        note, xy=(1000, 1e6), xytext=(1800, 2.2e5), fontsize=7.6, color=BLUE,
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8),
    )
    ax.plot([1000], [1000 * np.log2(1000)], "o", ms=4, color=GREEN, zorder=6)
    ax.annotate(
        note2, xy=(1000, 1000 * np.log2(1000)), xytext=(1800, 3200),
        fontsize=7.6, color=GREEN,
        arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.8),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 2.6e5)
    ax.set_ylim(10, 3e11)
    ax.set_xticks([10, 100, 1000, 10_000, 100_000])
    ax.set_xticklabels(["10", "100", "1,000", "10,000", "100,000"])
    if lang == "en":
        ax.set_xlabel("sentence length n (words)")
        ax.set_ylabel("word-to-word comparisons")
        ax.set_title("The price of comparing every word with every other")
    else:
        ax.set_xlabel("longitud de la oración n (palabras)")
        ax.set_ylabel("comparaciones palabra a palabra")
        ax.set_title("El precio de comparar cada palabra con todas las demás")
    ax.legend(loc="upper left", handlelength=1.4)
    fig.tight_layout(pad=0.3)
    return fig


def tut_complexity_growth_es() -> plt.Figure:
    return tut_complexity_growth("es")


# --------------------------------------------------------------------------- main
FIGURES = [
    ("tut-attention-heatmap", tut_attention_heatmap),
    ("tut-attention-heatmap-es", tut_attention_heatmap_es),
    ("tut-softmax-focus", tut_softmax_focus),
    ("tut-softmax-focus-es", tut_softmax_focus_es),
    ("tut-complexity-growth", tut_complexity_growth),
    ("tut-complexity-growth-es", tut_complexity_growth_es),
]


def main(argv: list[str]) -> int:
    """Render every figure (or only the ones named on the command line)."""
    wanted = set(argv[1:])
    selected = [(n, f) for n, f in FIGURES if not wanted or n in wanted]
    if not selected:
        print(f"unknown figure(s): {' '.join(sorted(wanted))}", file=sys.stderr)
        return 2
    for name, builder in selected:
        theme()
        fig = builder()
        for ext in ("pdf", "svg"):
            fig.savefig(HERE / f"{name}.{ext}", facecolor=FACE)
        plt.close(fig)
        print(f"  ok {name}.pdf + .svg")
    print(f"\n{len(selected)} figures -> {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
