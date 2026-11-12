#!/usr/bin/env python3
"""Render the pedagogical charts of the optimizer tutorial (Annex 9.2).

Every chart is self-contained: no data files, fixed random seeds, and a light
print theme that matches ``experimentos/2/report/make_charts.py``. Each figure
is written twice -- a vector PDF for LaTeX and an SVG for the website render,
which indexes figures by stem (see ``paper/flatten_tex.py``) -- and in two
languages: ``<name>`` for ``docs/paper`` and ``<name>-es`` for
``docs/paper-es`` (only the strings change; canvases and theme are shared).

Usage:
    uv run --with matplotlib --with seaborn --with numpy \\
        python docs/paper/figures/optimizers/tutorial/make_tutorial_figures.py [NAME ...]

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

ACCENT = BLUE


def tr(lang: str, en: str, es: str) -> str:
    """Pick the string for ``lang`` (``en`` or ``es``)."""
    return en if lang == "en" else es

#: Light-to-blue ramp used for every loss bowl, so the figures read as a set.
BOWL = LinearSegmentedColormap.from_list(
    "bowl", ["#ffffff", "#e8f1fc", "#c9ddf5", "#a6c8ec"]
)


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


# --------------------------------------------------------------------------- helpers
def bowl(ax: plt.Axes, a: float, b: float, lim: float, levels: int = 12) -> None:
    """Draw the elliptical loss bowl ``0.5*(a*x^2 + b*y^2)`` behind a path."""
    g = np.linspace(-lim, lim, 420)
    xx, yy = np.meshgrid(g, g)
    zz = 0.5 * (a * xx**2 + b * yy**2)
    ax.contourf(xx, yy, zz, levels=levels, cmap=BOWL)
    ax.contour(xx, yy, zz, levels=levels, colors="#b8c6d4", linewidths=0.45)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(False)


def path(
    ax: plt.Axes,
    pts: np.ndarray,
    color: str,
    label: str,
    ls: str = "-",
    lw: float = 1.5,
    every: int = 0,
    zorder: float = 3,
) -> None:
    """Draw an optimisation path, optionally marking every ``every``-th step."""
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls, label=label,
            zorder=zorder, solid_capstyle="round")
    if every:
        ax.plot(pts[::every, 0], pts[::every, 1], "o", ms=2.6, color=color,
                zorder=zorder, markeredgewidth=0)


def gd_path(a_mat: np.ndarray, p0, eta: float, steps: int) -> np.ndarray:
    """Full-batch gradient descent on the quadratic with Hessian ``a_mat``."""
    p = np.array(p0, dtype=float)
    out = [p.copy()]
    for _ in range(steps):
        p = p - eta * (a_mat @ p)
        out.append(p.copy())
    return np.array(out)


def momentum_path(
    a_mat: np.ndarray, p0, eta: float, beta: float, steps: int, nesterov: bool = False
) -> np.ndarray:
    """Heavy-ball momentum, with the optional Nesterov look-ahead."""
    p = np.array(p0, dtype=float)
    v = np.zeros(2)
    out = [p.copy()]
    for _ in range(steps):
        probe = p + beta * v if nesterov else p
        v = beta * v - eta * (a_mat @ probe)
        p = p + v
        out.append(p.copy())
    return np.array(out)


def rms_path(a_mat: np.ndarray, p0, eta: float, beta: float, steps: int) -> np.ndarray:
    """Per-coordinate normalised descent: the mechanism behind RMS-type methods."""
    p = np.array(p0, dtype=float)
    v = None
    out = [p.copy()]
    for _ in range(steps):
        g = a_mat @ p
        # Warm start: without it the very first step divides by an almost
        # empty accumulator and overshoots far past the minimum.
        v = g**2 if v is None else beta * v + (1 - beta) * g**2
        p = p - eta * g / (np.sqrt(v) + 1e-8)
        out.append(p.copy())
    return np.array(out)


def marker(
    ax: plt.Axes, xy, text: str, color: str = TEXT, offset=(5, 4), ha: str = "left"
) -> None:
    """Label a point on a path."""
    ax.plot(*xy, "o", ms=4.6, mfc="white", mec=color, mew=1.3, zorder=6)
    ax.annotate(text, xy, textcoords="offset points", xytext=offset,
                fontsize=7.2, color=color, zorder=7, ha=ha)


# --------------------------------------------------------------------------- 1. GD
def tut_gd_landscape(lang: str = "en") -> plt.Figure:
    """Three bowls: round, elongated, and unstable step size."""
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35))

    cases = [
        (np.diag([1.0, 1.0]), 0.45,
         tr(lang, "(a) round bowl: straight down",
            "(a) cuenco redondo: baja recto"), 2.0),
        (np.diag([0.05, 1.0]), 0.90,
         tr(lang, "(b) long valley: crawl the floor",
            "(b) valle alargado: se arrastra por el fondo"), 2.4),
        (np.diag([0.05, 1.0]), 1.90,
         tr(lang, r"(c) $\eta\to 2/L$: sloshing",
            r"(c) $\eta\to 2/L$: oscilación"), 2.4),
    ]
    for ax, (a_mat, eta, title, lim) in zip(axes, cases):
        bowl(ax, a_mat[0, 0], a_mat[1, 1], lim)
        pts = gd_path(a_mat, (-1.7, 1.4), eta, 26)
        path(ax, pts, ACCENT, f"$\\eta={eta:.2f}$", every=1)
        marker(ax, (0, 0), tr(lang, "minimum", "mínimo"), MUTED, offset=(6, -4))
        marker(ax, tuple(pts[0]), tr(lang, "start", "inicio"), TEXT, offset=(-2, 6))
        ax.set_title(title, fontsize=8.6, pad=3)
        ax.legend(loc="lower left", handlelength=1.0, borderpad=0.3,
                  labelspacing=0.25).set_zorder(9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0.3)
    return fig


# ------------------------------------------------------------------- 2. step size
def tut_learning_rate(lang: str = "en") -> plt.Figure:
    """The same valley, three step sizes, loss against iteration."""
    a_mat = np.diag([0.05, 1.0])
    fig, ax = plt.subplots(figsize=(4.3, 2.6))

    for eta, color, label in [
        (0.02, GREY, r"$\eta=0.02$: " + tr(lang, "too small", "demasiado pequeño")),
        (0.90, ACCENT, r"$\eta=0.90$: " + tr(lang, "good", "bueno")),
        (2.10, RED, r"$\eta=2.10$: " + tr(lang, "diverges", "diverge")),
    ]:
        pts = gd_path(a_mat, (-2.0, 1.6), eta, 120)
        loss = 0.5 * np.einsum("ij,jk,ik->i", pts, a_mat, pts)
        ax.plot(np.arange(len(loss)), loss, color=color, lw=1.5, label=label)

    ax.annotate(tr(lang, r"the hard edge is $\eta=2/L$",
                   r"el límite duro es $\eta=2/L$"), xy=(38, 6e2), xytext=(10, 2.0),
                 fontsize=7, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 6e3)
    ax.set_xlim(0, 120)
    ax.set_xlabel(tr(lang, "iteration", "iteración"))
    ax.set_ylabel(r"loss $f(\theta_t)$")
    ax.set_title(tr(lang, "One knob decides everything: the step size",
                    "Una sola perilla lo decide todo: el tamaño del paso"))
    ax.legend(loc="lower left", borderpad=0.3, labelspacing=0.25,
              framealpha=0.95, fontsize=7.2)
    fig.tight_layout(pad=0.3)
    return fig


# ---------------------------------------------------------------- 3. diagnostics
def tut_loss_diagnostics(lang: str = "en") -> plt.Figure:
    """Four loss-curve shapes and the sentence each one is telling you."""
    rng = np.random.default_rng(3)
    t = np.arange(160)
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 3.5), sharex=True)

    healthy = 0.15 + 1.25 * np.exp(-t / 16)
    low = 0.55 + 1.0 * np.exp(-t / 70)
    high = 1.0 + 0.10 * 1.035**t + 0.12 * np.sin(t / 1.3)
    noisy = 0.18 + 1.1 * np.exp(-t / 22) + 0.07 * rng.standard_normal(t.size)

    data = [
        (axes[0, 0], healthy, GREEN, tr(lang, "(a) healthy", "(a) sana"),
         tr(lang, "smooth, keeps falling.", "suave, sigue bajando.")),
        (axes[0, 1], high, RED, tr(lang, r"(b) $\eta$ too high", r"(b) $\eta$ demasiado alto"),
         tr(lang, "spikes, then climbs: cut the rate.",
            "picos y luego sube: baja la tasa.")),
        (axes[1, 0], low, ORANGE, tr(lang, r"(c) $\eta$ too small", r"(c) $\eta$ demasiado pequeño"),
         tr(lang, "still high: buy speed.", "sigue alta: compra velocidad.")),
        (axes[1, 1], noisy, PURPLE, tr(lang, "(d) batch too small", "(d) lote demasiado pequeño"),
         tr(lang, "noisy band, no progress.", "banda ruidosa, sin avance.")),
    ]
    for ax, curve, color, title, note in data:
        ax.plot(t, curve, color=color, lw=1.4)
        ax.set_title(title, fontsize=8.4, pad=3)
        ax.set_ylim(0, 2.5)
        ax.text(0.97, 0.86, note, transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color=MUTED, style="italic")
    for ax in axes[1]:
        ax.set_xlabel(tr(lang, "iteration", "iteración"))
    for ax in axes[:, 0]:
        ax.set_ylabel(tr(lang, "loss", "pérdida"))
    fig.tight_layout(pad=0.35, h_pad=0.6, w_pad=0.6)
    return fig


# ----------------------------------------------------------- 4. stochastic noise
def _regression_data(seed: int = 0, n: int = 4096):
    """Correlated 2-D regression data: an elongated but real MSE surface."""
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(np.array([[1.0, 0.85], [0.85, 1.0]]))
    x = rng.standard_normal((n, 2)) @ chol.T
    w_true = np.array([1.5, -1.0])
    y = x @ w_true + 0.05 * rng.standard_normal(n)
    return x, y, w_true


def _mse_grad(x, y, w):
    r = x @ w - y
    return x.T @ r / len(x)


def tut_batch_noise(lang: str = "en") -> plt.Figure:
    """Full batch versus mini-batch versus one sample, on real data."""
    x, y, w_true = _regression_data()
    hess = x.T @ x / len(x)
    b_vec = x.T @ y / len(x)
    w_star = np.linalg.solve(hess, b_vec)
    c_min = 0.5 * np.mean(y**2) - 0.5 * w_star @ hess @ w_star
    eta = 0.5 / np.linalg.eigvalsh(hess).max()
    rng = np.random.default_rng(11)

    fig = plt.figure(figsize=(6.5, 2.7))
    ax = fig.add_subplot(1, 2, 1)
    xlim, ylim = (-0.55, 2.05), (-1.55, 0.65)
    gx = np.linspace(*xlim, 320)
    gy = np.linspace(*ylim, 320)
    xx, yy = np.meshgrid(gx, gy)
    zz = (0.5 * (hess[0, 0] * xx**2 + 2 * hess[0, 1] * xx * yy
                 + hess[1, 1] * yy**2)
          - b_vec[0] * xx - b_vec[1] * yy + 0.5 * np.mean(y**2))
    levels = np.geomspace(c_min + 2e-3, zz.max(), 15)
    ax.contourf(xx, yy, zz, levels=levels, cmap=BOWL, extend="both")
    ax.contour(xx, yy, zz, levels=levels, colors="#b8c6d4", linewidths=0.45)

    def keep(pts: np.ndarray) -> np.ndarray:
        """Break the line instead of drawing it outside the zoom window."""
        out = pts.copy()
        inside = ((out[:, 0] > xlim[0] - 0.3) & (out[:, 0] < xlim[1] + 0.3)
                  & (out[:, 1] > ylim[0] - 0.3) & (out[:, 1] < ylim[1] + 0.3))
        out[~inside] = np.nan
        return out

    w_start = np.zeros(2)
    for batch, color in [(1, RED), (32, GREEN)]:
        w = w_start.copy()
        gen = np.random.default_rng(7 if batch == 32 else 5)
        pts = [w.copy()]
        for _ in range(45):
            idx = gen.choice(len(x), size=batch, replace=False)
            w = w - eta * _mse_grad(x[idx], y[idx], w)
            pts.append(w.copy())
        path(ax, keep(np.array(pts)), color,
             tr(lang, "single sample", "una muestra") if batch == 1
             else tr(lang, f"mini-batch $b={batch}$", f"minilote $b={batch}$"),
             lw=1.1, every=3)

    w = w_start.copy()
    full = [w.copy()]
    for _ in range(45):
        w = w - eta * _mse_grad(x, y, w)
        full.append(w.copy())
    path(ax, keep(np.array(full)), ACCENT,
         tr(lang, "full batch", "lote completo"), lw=1.9, zorder=5)
    ax.plot(*w_star, "x", ms=6, color=TEXT, mew=1.5, zorder=8)
    marker(ax, tuple(w_start), tr(lang, "start", "inicio"), TEXT, offset=(6, -10))
    ax.annotate(r"$w^\star$", w_star, textcoords="offset points", xytext=(7, -3),
                fontsize=8, color=TEXT, zorder=9)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(tr(lang, "(a) 45 steps each, same step size",
                    "(a) 45 pasos cada uno, mismo tamaño de paso"),
                 fontsize=8.6, pad=3)
    ax.legend(loc="upper left", borderpad=0.3, labelspacing=0.25,
              handlelength=1.2, fontsize=7).set_zorder(9)

    # noise against batch size
    ax2 = fig.add_subplot(1, 2, 2)
    w0 = np.zeros(2)
    g_full = _mse_grad(x, y, w0)
    batches = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    rel = []
    for b in batches:
        errs = []
        for _ in range(60):
            idx = rng.choice(len(x), size=b, replace=False)
            errs.append(np.linalg.norm(_mse_grad(x[idx], y[idx], w0) - g_full)
                        / np.linalg.norm(g_full))
        rel.append(np.mean(errs))
    rel = np.array(rel)
    ax2.loglog(batches, rel, "o-", color=ACCENT, ms=3.2, lw=1.3,
               label=tr(lang, "measured noise", "ruido medido"))
    ref = rel[0] / np.sqrt(batches)
    ax2.loglog(batches, ref, "--", color=ORANGE, lw=1.1,
               label=r"reference $\propto 1/\sqrt{b}$")
    ax2.set_xlabel(tr(lang, "batch size $b$", "tamaño de lote $b$"))
    ax2.set_ylabel(r"$\|\hat g_b-g\|/\|g\|$")
    ax2.set_title(tr(lang, "(b) more data per step, less noise",
                    "(b) más datos por paso, menos ruido"), fontsize=8.6, pad=3)
    ax2.legend(loc="lower left", borderpad=0.3, labelspacing=0.25)
    fig.tight_layout(pad=0.35, w_pad=0.9)
    return fig


# --------------------------------------------------------------- 5. convergence
def tut_convergence(lang: str = "en") -> plt.Figure:
    """Rates from the tutorial: GD 1/t, SGD 1/sqrt(t), fixed step stalls."""
    t = np.arange(1, 4001)
    gd = 1.0 / t
    sgd = 1.0 / np.sqrt(t)
    fixed = 1.0 / t + 0.06

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.55))

    ax = axes[0]
    ax.loglog(t, gd, color=ACCENT, lw=1.5, label=r"GD: $O(1/t)$")
    ax.loglog(t, sgd, color=GREEN, lw=1.5, label=r"SGD: $O(1/\sqrt{t})$")
    ax.loglog(t, fixed, color=RED, lw=1.5, ls="--",
              label=tr(lang, r"fixed $\eta$: stalls at $O(\eta)$",
                       r"$\eta$ fijo: se estanca en $O(\eta)$"))
    ax.axhline(0.06, color=RED, lw=0.8, ls=":", alpha=0.8)
    ax.annotate(tr(lang, r"plateau $\approx\eta$", r"meseta $\approx\eta$"),
                xy=(1500, 0.06), xytext=(300, 0.22),
                fontsize=7, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax.set_xlabel(tr(lang, "iteration $t$", "iteración $t$"))
    ax.set_ylabel(r"$f(\theta_t)-f^\star$")
    ax.set_title(tr(lang, "(a) per iteration", "(a) por iteración"),
                 fontsize=8.6, pad=3)
    ax.legend(loc="lower left", borderpad=0.3, labelspacing=0.25,
              framealpha=0.95)
    ax.set_ylim(1e-4, 2)

    # Same wall-clock axis for every curve: one GD step costs 100 SGD steps.
    ax = axes[1]
    cost = np.logspace(0, 5.5, 400)
    ax.loglog(cost, 100.0 / cost, color=ACCENT, lw=1.5, label="GD")
    ax.loglog(cost, 1.0 / np.sqrt(cost), color=GREEN, lw=1.5, label="SGD")
    ax.loglog(cost, 100.0 / cost + 0.06, color=RED, lw=1.5, ls="--",
              label=tr(lang, "SGD, fixed step", "SGD, paso fijo"))
    ax.axvline(1e4, color=MUTED, lw=0.7, ls=":")
    ax.annotate(tr(lang, "SGD ahead", "SGD va delante"), xy=(60, 0.02),
                fontsize=7, color=GREEN)
    ax.annotate(tr(lang, "GD ahead", "GD va delante"), xy=(2.2e4, 1.1e-4),
                fontsize=7, color=ACCENT)
    ax.set_xlabel(tr(lang, "wall-clock (one GD step = 100 units)",
                     "tiempo real (un paso GD = 100 unidades)"))
    ax.set_ylabel(r"$f(\theta_t)-f^\star$")
    ax.set_title(tr(lang, "(b) per unit of time", "(b) por unidad de tiempo"),
                 fontsize=8.6, pad=3)
    ax.legend(loc="lower left", borderpad=0.3, labelspacing=0.25,
              framealpha=0.95)
    ax.set_ylim(1e-4, 2)
    fig.tight_layout(pad=0.35, w_pad=0.8)
    return fig


# ------------------------------------------------------------------ 6. backprop
def tut_backprop(lang: str = "en") -> plt.Figure:
    """A tiny network: what travels forward, what travels back, and the cost."""
    x = np.array([1.0, -0.5])
    w1 = np.array([[0.6, -0.4], [-0.5, 0.8], [0.3, 0.5]])
    b1 = np.array([0.1, -0.2, 0.0])
    w2 = np.array([[0.7, -0.6, 0.2], [-0.3, 0.4, 0.9]])
    b2 = np.array([0.0, 0.1])
    target = np.array([0.5, -1.0])

    a1 = w1 @ x + b1
    z1 = np.tanh(a1)
    a2 = w2 @ z1 + b2
    delta2 = a2 - target
    delta1 = (w2.T @ delta2) * (1 - z1**2)

    fig = plt.figure(figsize=(6.5, 2.55))
    ax = fig.add_axes([0.0, 0.02, 0.76, 0.86])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-0.55, 2.55)
    ax.axis("off")

    layers = [[(0.0, 1.75), (0.0, 0.75)],
              [(1.0, 2.1), (1.0, 1.25), (1.0, 0.4)],
              [(2.0, 1.75), (2.0, 0.75)]]
    for li in range(len(layers) - 1):
        for (x0, y0) in layers[li]:
            for (x1, y1) in layers[li + 1]:
                ax.plot([x0, x1], [y0, y1], color="#cdd6df", lw=0.8, zorder=1)
    for li, nodes in enumerate(layers):
        for (cx, cy) in nodes:
            ax.add_patch(plt.Circle((cx, cy), 0.22, fc="white", ec=SPINE,
                                    lw=1.0, zorder=3))
        names = (tr(lang, ["input", "hidden", "output"],
                    ["entrada", "oculta", "salida"]))
        ax.text(nodes[0][0], 2.42, names[li], ha="center", fontsize=7.4, color=MUTED)

    for i, (cx, cy) in enumerate(layers[0]):
        ax.text(cx, cy, f"{x[i]:.2f}", ha="center", va="center", fontsize=7,
                zorder=4)
    for i, (cx, cy) in enumerate(layers[1]):
        ax.text(cx, cy, f"{z1[i]:.2f}", ha="center", va="center", fontsize=7,
                zorder=4)
    for i, (cx, cy) in enumerate(layers[2]):
        ax.text(cx, cy, f"{a2[i]:.2f}", ha="center", va="center", fontsize=7,
                zorder=4)
    reds = [(1.0, 2.1), (1.0, 1.25), (1.0, 0.4), (2.0, 1.75), (2.0, 0.75)]
    deltas = list(delta1) + list(delta2)
    for (cx, cy), d in zip(reds, deltas):
        ax.text(cx, cy - 0.42, f"\u03b4={d:+.2f}", ha="center", fontsize=6.6,
                color=RED)
    ax.text(1.0, -0.42,
            tr(lang, "black: forward value  \u2022  red: backward gradient",
               "negro: valor hacia delante  \u2022  rojo: gradiente hacia atrás"),
            ha="center", fontsize=7, color=MUTED)
    ax.set_title(tr(lang, "(a) one graph, two passes", "(a) un grafo, dos pasadas"),
                 fontsize=8.6, pad=2)

    ax2 = fig.add_axes([0.79, 0.3, 0.19, 0.5])
    bars = ax2.bar(tr(lang, ["fwd", "bwd"], ["ida", "vuelta"]), [1.0, 2.0],
                   color=[ACCENT, ORANGE], width=0.6)
    for rect, val in zip(bars, [1.0, 2.0]):
        ax2.text(rect.get_x() + rect.get_width() / 2, val + 0.06, f"{val:.0f}\u00d7",
                 ha="center", fontsize=7.6, color=TEXT)
    ax2.set_ylim(0, 2.5)
    ax2.set_ylabel(tr(lang, "step cost", "coste del paso"), fontsize=7.2)
    ax2.set_title(tr(lang, "(b) two passes", "(b) dos pasadas"), fontsize=8, pad=2)
    ax2.tick_params(axis="x", labelsize=7)
    return fig


# ------------------------------------------------------------------ 7. momentum
def tut_momentum(lang: str = "en") -> plt.Figure:
    """A running average of past steps, and what it buys."""
    a_mat = np.diag([0.05, 1.0])
    eta, beta, steps = 0.08, 0.9, 120
    p0 = (-2.2, 1.4)
    mo = momentum_path(a_mat, p0, eta, beta, steps)
    na = momentum_path(a_mat, p0, eta, beta, steps, nesterov=True)
    gd = gd_path(a_mat, p0, eta, steps)

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.8),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 0.72]})

    ax = axes[0]
    bowl(ax, 0.05, 1.0, 2.4)
    path(ax, gd, GREY, tr(lang, "plain GD, same $\\eta$",
                          "GD simple, mismo $\\eta$"), ls="--", lw=1.2, every=12)
    path(ax, mo, ACCENT, r"momentum $\beta=0.9$", lw=1.7, every=12)
    path(ax, na, ORANGE, tr(lang, "Nesterov look-ahead", "adelanto de Nesterov"),
         lw=1.2)
    ax.set_title(tr(lang, "(a) the zig-zag is filtered out",
                    "(a) se filtra el zigzag"), fontsize=8.4, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", borderpad=0.3, labelspacing=0.25,
              handlelength=1.2, fontsize=6.8).set_zorder(9)

    ax = axes[1]
    for pts, color, label, ls in [
        (gd, GREY, tr(lang, "plain GD", "GD simple"), "--"),
        (mo, ACCENT, r"momentum $\beta=0.9$", "-"),
        (na, ORANGE, "Nesterov", "-"),
    ]:
        loss = 0.5 * np.einsum("ij,jk,ik->i", pts, a_mat, pts)
        ax.semilogy(np.arange(len(loss)), np.maximum(loss, 1e-6), color=color,
                    lw=1.5, ls=ls, label=label)
    ax.set_xlabel(tr(lang, "iteration", "iteración"))
    ax.set_ylabel(r"loss $f(\theta_t)$")
    ax.set_ylim(1e-6, 3)
    ax.set_title(tr(lang, "(b) same step size: speed along the floor",
                    "(b) mismo paso: velocidad por el fondo"), fontsize=8.4, pad=3)
    ax.legend(loc="upper right", borderpad=0.3, labelspacing=0.25, fontsize=6.8)

    # the rule itself, drawn once, in a clean frame
    ax = axes[2]
    mom = np.array([-0.55, 0.38])
    grad = np.array([0.78, -0.80])
    total = mom + grad
    for start, vec, color in [((0, 0), mom, ACCENT), (mom, grad, RED),
                              ((0, 0), total, GREEN)]:
        ax.annotate("", xy=(start[0] + vec[0], start[1] + vec[1]), xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.7,
                                    shrinkA=0, shrinkB=0))
    ax.plot([mom[0], total[0]], [mom[1], total[1]], ls=":", color=RED, lw=0.8)
    ax.plot([0, mom[0]], [0, mom[1]], ls=":", color=ACCENT, lw=0.8)
    ax.text(-0.34, 0.30, r"$\beta v_{t-1}$", fontsize=7.2, color=ACCENT)
    ax.text(0.04, 0.30, r"$-\eta g_t$", fontsize=7.2, color=RED)
    ax.text(0.28, -0.52, r"$v_t$", fontsize=7.4, color=GREEN)
    ax.plot(0, 0, "o", ms=3.5, color=TEXT, zorder=6)
    ax.set_xlim(-0.85, 0.55)
    ax.set_ylim(-0.72, 0.72)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(r"$v_t=\beta v_{t-1}-\eta g_t$", fontsize=8.4, pad=3)
    fig.tight_layout(pad=0.35, w_pad=0.55)
    return fig


# ------------------------------------------------------------------ 8. adaptive
def tut_adaptive(lang: str = "en") -> plt.Figure:
    """One shared step size versus one step size per coordinate."""
    a_mat = np.diag([0.01, 1.0])
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))

    ax = axes[0]
    bowl(ax, 0.01, 1.0, 2.6)
    p0 = (-2.2, 1.4)
    plain = gd_path(a_mat, p0, 0.9, 90)
    rms = rms_path(a_mat, p0, 0.3, 0.9, 90)
    path(ax, plain, GREY, tr(lang, r"one $\eta$ for all", r"un $\eta$ para todo"),
         ls="--", lw=1.3, every=6)
    path(ax, rms, ACCENT, tr(lang, "one $\\eta$ per coordinate",
                             "un $\\eta$ por coordenada"), lw=1.6, every=4)
    ax.plot(0, 0, "o", ms=4.6, mfc="white", mec=MUTED, mew=1.3, zorder=6)
    ax.set_title(tr(lang, "(a) 100:1 curvature, 90 steps each",
                    "(a) curvatura 100:1, 90 pasos cada uno"), fontsize=8.6, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", borderpad=0.3, labelspacing=0.25,
              handlelength=1.2).set_zorder(9)

    ax = axes[1]
    g0 = a_mat @ np.array(p0)
    v = g0**2
    step_plain = 0.9 * np.abs(g0)
    step_rms = 0.3 * np.abs(g0) / (np.sqrt(v) + 1e-8)
    labels = tr(lang, [r"flat direction ($x$)", r"steep direction ($y$)"],
                [r"dirección plana ($x$)", r"dirección empinada ($y$)"])
    yy = np.arange(2)
    ax.barh(yy - 0.19, step_plain, height=0.34, color=GREY,
            label=tr(lang, "one shared $\\eta$", "un $\\eta$ compartido"))
    ax.barh(yy + 0.19, step_rms, height=0.34, color=ACCENT,
            label=tr(lang, "per-coordinate", "por coordenada"))
    ax.set_xscale("log")
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=7.6)
    ax.set_xlim(1e-2, 5)
    ax.set_xlabel(tr(lang, "first-step displacement (log)",
                     "desplazamiento del primer paso (log)"))
    ax.set_title(tr(lang, "(b) the 100\u00d7 imbalance disappears",
                    "(b) desaparece el desbalance 100\u00d7"), fontsize=8.6, pad=3)
    for yi, (a_val, b_val) in enumerate(zip(step_plain, step_rms)):
        ax.text(a_val * 1.15, yi - 0.19, f"{a_val:.2g}", va="center", fontsize=6.8,
                color=MUTED)
        ax.text(b_val * 1.15, yi + 0.19, f"{b_val:.2g}", va="center", fontsize=6.8,
                color=ACCENT)
    ax.legend(loc="lower right", borderpad=0.3, labelspacing=0.25, fontsize=7,
              framealpha=0.95)
    fig.tight_layout(pad=0.35, w_pad=0.6)
    return fig


# ------------------------------------------------------------------- 9. families
def tut_families(lang: str = "en") -> plt.Figure:
    """A fingerprint of the eight families: which ingredients each one uses."""
    rows = tr(lang, [
        "first-order baseline",
        "momentum + variance reduction",
        "memory-efficient / low-rank",
        "schedule-free / parameter-free",
        "curvature / second-order",
        "geometric / orthogonality",
        "adaptivity-tunable",
        "evolutionary / population",
    ], [
        "base de primer orden",
        "momentum + reducción de varianza",
        "eficiente en memoria / bajo rango",
        "schedule-free / sin parámetros",
        "curvatura / segundo orden",
        "geométrico / ortogonalidad",
        "adaptatividad ajustable",
        "evolutivo / población",
    ])
    cols = tr(lang, ["gradient", "1st moment", "2nd moment", "grad. history",
                     "curvature", "matrix shape", "population", "state < Adam"],
              ["gradiente", "1.er momento", "2.º momento", "historial de grad.",
               "curvatura", "forma matricial", "población", "estado < Adam"])
    grid = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 0, 0],
        [1, .5, 0, 0, 0, 1, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0],
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 3.15))
    ax.imshow(grid, cmap=LinearSegmentedColormap.from_list(
        "ing", ["#ffffff", "#c9ddf5"]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=28, ha="right", fontsize=7.4)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=7.6)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color=FACE, linewidth=2.0)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="major", visible=False)
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = grid[i, j]
            glyph = "\u2713" if val == 1 else ("~" if val == .5 else "")
            ax.text(j, i, glyph, ha="center", va="center", fontsize=9,
                    color=ACCENT if val == 1 else MUTED)
    ax.set_title(tr(lang,
        "Which ingredient each family turns on  (\u2713 used, ~ only some members)",
        "Qué ingrediente activa cada familia  (\u2713 usado, ~ sólo algunos miembros)"),
        fontsize=8.4, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.25)
    return fig


# --------------------------------------------------------------------------- main
FIGURES = [
    ("tut-gd-landscape", tut_gd_landscape),
    ("tut-learning-rate", tut_learning_rate),
    ("tut-loss-diagnostics", tut_loss_diagnostics),
    ("tut-batch-noise", tut_batch_noise),
    ("tut-convergence", tut_convergence),
    ("tut-backprop", tut_backprop),
    ("tut-momentum", tut_momentum),
    ("tut-adaptive", tut_adaptive),
    ("tut-families", tut_families),
    # Spanish render used by docs/paper-es (same canvases and theme).
    ("tut-gd-landscape-es", lambda: tut_gd_landscape("es")),
    ("tut-learning-rate-es", lambda: tut_learning_rate("es")),
    ("tut-loss-diagnostics-es", lambda: tut_loss_diagnostics("es")),
    ("tut-batch-noise-es", lambda: tut_batch_noise("es")),
    ("tut-convergence-es", lambda: tut_convergence("es")),
    ("tut-backprop-es", lambda: tut_backprop("es")),
    ("tut-momentum-es", lambda: tut_momentum("es")),
    ("tut-adaptive-es", lambda: tut_adaptive("es")),
    ("tut-families-es", lambda: tut_families("es")),
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
