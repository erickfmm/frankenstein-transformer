"""Export the Experiment 3 figures as vector PDFs/SVGs for the two papers.

The report builders live in ``make_charts.py``; this script retunes them for a
print canvas and writes ``<name>.pdf/.svg`` (English, used by ``docs/paper``)
and ``<name>-es.pdf/.svg`` (Spanish, used by ``docs/paper-es``).

Usage:
    uv run --no-sync --with pandas --with numpy --with matplotlib python \
        experimentos/3/report/make_paper_figures.py

Prerequisite: ``out/data/*.csv`` from ``build_data.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402
import make_charts as mc  # noqa: E402


def main() -> None:
    """Render every figure in both languages, PDF and SVG."""
    mc.C.PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    for lang in ("en", "es"):
        mc.set_lang(lang)
        suffix = "" if lang == "en" else "-es"
        for name, builder, scale in mc.FIGURES:
            theme = mc.apply_theme("light")
            figure = builder(theme)
            width, height = figure.get_size_inches()
            figure.set_size_inches(width * scale, height * scale)
            options = dict(
                facecolor=theme["face"],
                bbox_inches="tight",
                pad_inches=0.05,
                dpi=300,
            )
            figure.savefig(mc.C.PAPER_FIGURES / f"{name}{suffix}.pdf", **options)
            figure.savefig(mc.C.PAPER_FIGURES / f"{name}{suffix}.svg", **options)
            plt.close(figure)
            print(f"  ok {name}{suffix}.pdf (+ .svg)")
    print(
        f"\n{len(mc.FIGURES)} figures x 2 languages -> "
        f"{mc.C.PAPER_FIGURES.relative_to(mc.C.PROJECT)}"
    )


if __name__ == "__main__":
    main()
