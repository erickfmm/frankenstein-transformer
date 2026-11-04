"""Export the curated Experiment 2 figures as vector PDFs for the papers.

The web report (``make_charts.py``) renders every figure twice, in a dark and a
light theme, as PNG at web sizes. The LaTeX papers need the light theme, a
print-sized canvas and a vector format, so this script imports the chart
builders from ``make_charts.py``, retunes the canvas and writes PDFs into
``docs/paper/figures/experiments/round2/``. The same figures are also written
as SVG so the website render of the paper (which indexes figures by stem and
embeds the SVG) keeps every chart.

Usage:
    uv run --with pandas --with numpy --with matplotlib --with seaborn \
        python experimentos/2/report/make_paper_figures.py

Prerequisite: ``report/out/data/master.csv`` and friends, produced by
``build_data.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DST = ROOT / "docs" / "paper" / "figures" / "experiments" / "round2"

sys.path.insert(0, str(HERE))
import matplotlib.pyplot as plt  # noqa: E402
import make_charts as mc  # noqa: E402

#: Default canvas scale: the web figures are ~10-11 in wide, the paper canvas
#: is ~6.5 in, so 0.62 keeps the point sizes readable after the final crop.
DEFAULT_SCALE = 0.62

#: Dense heatmaps need a larger canvas and no in-cell annotations.
MATRIX_SCALE = 0.42

FIGURES = [
    # ---------------------------------------------------------- parameters
    ("params_composition", mc.params_composition, 0.55),
    ("params_family_box", mc.params_family_box, DEFAULT_SCALE),
    # ------------------------------------------------------------ training
    ("loss_family_box", mc.loss_family_box, DEFAULT_SCALE),
    ("acc_family_box", mc.acc_family_box, DEFAULT_SCALE),
    ("loss_acc_scatter", mc.loss_acc_scatter, DEFAULT_SCALE),
    ("curves_quantiles", mc.curves_quantiles, DEFAULT_SCALE),
    ("curves_top10", mc.curves_top10, DEFAULT_SCALE),
    ("grad_groups", mc.grad_groups, DEFAULT_SCALE),
    # ---------------------------------------------------------- A x B atlas
    (
        "matrix_loss",
        lambda t: mc.pair_matrix(
            t,
            "loss_final",
            "crest_r",
            "Final loss for every A\\u00d7B hybrid pair",
            "final loss (nats)",
            fmt=".2f",
            annot=False,
        ),
        MATRIX_SCALE,
    ),
    ("coverage", mc.coverage, MATRIX_SCALE),
    ("synergy", mc.synergy, DEFAULT_SCALE),
    ("pareto_speed_loss", mc.pareto_speed_loss, DEFAULT_SCALE),
    ("top5_by_b", mc.top5_by_b, DEFAULT_SCALE),
    ("top5_vs_loss", mc.top5_vs_loss, DEFAULT_SCALE),
    # ---------------------------------------------------------- ablations
    (
        "opt_loss",
        lambda t: mc.group_bar(
            t,
            "opt",
            "loss_final",
            "Twenty-three optimizer families on the same architecture",
            color="#3fb950",
            sort_asc=True,
        ),
        DEFAULT_SCALE,
    ),
    ("opt_speed", mc.opt_speed, DEFAULT_SCALE),
    (
        "norm_loss",
        lambda t: mc.group_bar(
            t, "norm", "loss_final", "Six normalization strategies", color="#ffa657"
        ),
        DEFAULT_SCALE,
    ),
    (
        "pe_loss",
        lambda t: mc.group_bar(
            t, "pe", "loss_final", "Twelve positional encodings", color="#ff7b72"
        ),
        DEFAULT_SCALE,
    ),
    (
        "trans_loss",
        lambda t: mc.group_bar(
            t,
            "trans",
            "loss_final",
            "Twenty architecture switches (BitNet, MoE, mHC, ...)",
            color="#d2991d",
            rotate=55,
        ),
        DEFAULT_SCALE,
    ),
    ("vit_split", mc.vit_split, DEFAULT_SCALE),
    ("tokenizer_task", mc.tokenizer_task, DEFAULT_SCALE),
    # ---------------------------------------------------------- deployment
    ("inference_format", mc.inference_format, DEFAULT_SCALE),
    ("quant_delta_dist", mc.quant_delta_dist, DEFAULT_SCALE),
    ("quant_scatter", mc.quant_scatter, DEFAULT_SCALE),
    ("deploy_duration", mc.deploy_duration, DEFAULT_SCALE),
    # --------------------------------------------------------- reliability
    ("stage_status", mc.stage_status, DEFAULT_SCALE),
    ("failures_family", mc.failures_family, DEFAULT_SCALE),
    ("correlation", mc.correlation, DEFAULT_SCALE),
]


def main() -> None:
    """Render every selected figure once, in the light theme, as PDF."""
    DST.mkdir(parents=True, exist_ok=True)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    for name, builder, scale in FIGURES:
        theme = mc.apply_theme("light")
        figure = builder(theme)
        width, height = figure.get_size_inches()
        figure.set_size_inches(width * scale, height * scale)
        target = DST / f"{name}.pdf"
        options = dict(
            facecolor=theme["face"],
            bbox_inches="tight",
            pad_inches=0.05,
            dpi=300,
        )
        figure.savefig(target, **options)
        figure.savefig(DST / f"{name}.svg", **options)
        plt.close(figure)
        print(f"  ok {target.relative_to(ROOT)} (+ .svg)")
    print(f"\n{len(FIGURES)} paper figures -> {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
