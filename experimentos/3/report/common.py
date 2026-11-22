"""Shared taxonomy, labels and LaTeX helpers for the Experiment 3 report.

The report pipeline is four scripts that run in order::

    build_data.py         -> out/data/*.csv          (joined master dataset)
    make_charts.py        -> out/img/*.png           (web-sized figures)
    make_paper_figures.py -> docs/paper/figures/...  (print PDF/SVG, bilingual)
    make_report.py        -> out/report-{en,es}.md and the two annexes

Every script imports the taxonomy from here so a combo id is classified the
same way everywhere.
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                    # experimentos/3
PROJECT = ROOT.parents[1]             # repository root
CSVS = ROOT / "csvs"
CONFIGS = ROOT / "configs"
TRAINING_METRICS = ROOT / "training_metrics"
OUT = HERE / "out"
DATA = OUT / "data"
IMG = OUT / "img"
PAPER_FIGURES = PROJECT / "docs" / "paper" / "figures" / "experiments" / "round3"

#: Experiment 3 family buckets, in report order.
FAMILIES = (
    "pair",
    "single",
    "optimizer",
    "pe",
    "vit_pe",
    "vision",
    "audio",
    "realdata",
    "tokenizer",
    "switch",
    "moe",
    "norm",
    "batch",
    "task",
    "bitnet",
)

#: Short bilingual display names (``en``, ``es``).
FAMILY_LABEL = {
    "pair": ("hybrid pairs", "pares híbridos"),
    "single": ("single mixers", "mixers únicos"),
    "optimizer": ("optimizers", "optimizadores"),
    "pe": ("text PE ablations", "ablaciones de PE de texto"),
    "vit_pe": ("ViT PE ablations", "ablaciones de PE del ViT"),
    "vision": ("vision tasks", "tareas de visión"),
    "audio": ("audio tasks", "tareas de audio"),
    "realdata": ("real-data runs", "ejecuciones con datos reales"),
    "tokenizer": ("tokenizers", "tokenizadores"),
    "switch": ("architecture switches", "interruptores de arquitectura"),
    "moe": ("MoE variants", "variantes MoE"),
    "norm": ("normalization", "normalización"),
    "batch": ("batch size", "tamaño de lote"),
    "task": ("task switches", "interruptores de tarea"),
    "bitnet": ("BitNet/GGUF", "BitNet/GGUF"),
}

FAMILY_COLOR = {
    "pair": "#3060d0",
    "single": "#8a63d2",
    "optimizer": "#1b9e77",
    "pe": "#e6a020",
    "vit_pe": "#d99a2b",
    "vision": "#d95f02",
    "audio": "#a6761d",
    "realdata": "#7570b3",
    "tokenizer": "#666666",
    "switch": "#e7298a",
    "moe": "#66a61e",
    "norm": "#e6ab02",
    "batch": "#b15928",
    "task": "#33a02c",
    "bitnet": "#9c9c9c",
}

#: Parameter-block categories logged by the trainer (order of the heatmap).
BLOCK_GROUPS = (
    "embeddings",
    "pos",
    "attention",
    "ffn",
    "norm",
    "head",
    "mhc",
    "router",
    "residual",
    "other",
)

_BLOCK_LABEL = {
    "embeddings": ("embeddings", "embeddings"),
    "pos": ("positional", "posicional"),
    "attention": ("attention", "atención"),
    "ffn": ("FFN", "FFN"),
    "norm": ("norms", "normalización"),
    "head": ("head", "cabeza"),
    "mhc": ("mHC", "mHC"),
    "router": ("router", "router"),
    "residual": ("residual", "residual"),
    "other": ("other", "otros"),
}

#: Two loader runtimes produced the HF inference rows; ``v1`` and ``v2``
#: sampled different masks, so only aggregate comparisons are valid across
#: them (see the annex).
CODEC_LABEL = {
    "none": ("unquantized", "sin cuantizar"),
    "q8_0": ("q8_0", "q8_0"),
    "q4_0": ("q4_0", "q4_0"),
    "i2_s": ("i2_s", "i2_s"),
}

_PAIR_RE = re.compile(r"^attn_(?P<a>.+?)__(?P<b>.+)$")


def classify(combo_id: str) -> str:
    """Return the experiment family of ``combo_id``.

    Args:
        combo_id: Combo identifier as written by the harness.

    Returns:
        One of :data:`FAMILIES`.
    """
    if combo_id.startswith("attn_single_"):
        return "single"
    if combo_id.startswith("attn_") and "__" in combo_id:
        return "pair"
    if combo_id.startswith("opt_"):
        return "optimizer"
    if combo_id.startswith("vit_pe_"):
        return "vit_pe"
    if combo_id.startswith("pe_"):
        return "pe"
    if combo_id.startswith("vit_"):
        return "vision"
    if combo_id.startswith("audio_"):
        return "audio"
    if combo_id.startswith("realdata_"):
        return "realdata"
    if combo_id.startswith("tokenizer_"):
        return "tokenizer"
    if combo_id.startswith("trans_"):
        return "switch"
    if combo_id.startswith("moe_"):
        return "moe"
    if combo_id.startswith("norm_"):
        return "norm"
    if combo_id.startswith("batch_size_"):
        return "batch"
    if combo_id.startswith("task_"):
        return "task"
    if combo_id.startswith("bitnet_"):
        return "bitnet"
    return "other"


def split_pair(combo_id: str):
    """Return ``(mixer_a, mixer_b)`` for hybrid pairs, else ``(None, None)``."""
    match = _PAIR_RE.match(combo_id)
    if not match:
        return None, None
    return match.group("a"), match.group("b")


def tidy_mixer(name: str | None) -> str | None:
    """Strip the registry ``_attn`` suffix from a mixer id."""
    if name is None:
        return None
    return name[:-5] if name.endswith("_attn") else name


def family_label(family: str, lang: str = "en") -> str:
    """Bilingual display label for a family bucket."""
    en, es = FAMILY_LABEL.get(family, (family, family))
    return en if lang == "en" else es


def block_label(group: str, lang: str = "en") -> str:
    """Bilingual display label for a parameter-block category."""
    en, es = _BLOCK_LABEL.get(group, (group, group))
    return en if lang == "en" else es


def codec_label(codec: str, lang: str = "en") -> str:
    """Bilingual display label for a quantization codec."""
    en, es = CODEC_LABEL.get(codec, (codec, codec))
    return en if lang == "en" else es


def codec_of(deploy_format: str) -> str:
    """Map an ``hf_<codec>_v<n>`` format to its codec (``none`` for hf_none)."""
    if deploy_format == "hf_none":
        return "none"
    match = re.match(r"hf_(q8_0|q4_0|i2_s)_v\d", deploy_format)
    return match.group(1) if match else "unknown"


def runtime_of(deploy_format: str) -> str:
    """Return ``v1``/``v2`` for a HF format, ``none`` for the baseline."""
    match = re.search(r"_v(\d)$", deploy_format)
    return f"v{match.group(1)}" if match else "none"


def tex_id(text: str) -> str:
    """Escape a combo id (or any identifier) for LaTeX text mode."""
    escaped = str(text).replace("\\", r"\textbackslash{}")
    for char in ("&", "%", "$", "#", "{", "}"):
        escaped = escaped.replace(char, "\\" + char)
    escaped = escaped.replace("_", r"\_")
    escaped = escaped.replace("~", r"\textasciitilde{}")
    escaped = escaped.replace("^", r"\textasciicircum{}")
    return escaped


def tex_tt(text: str) -> str:
    """Return ``\\texttt{...}`` with the identifier escaped."""
    return r"\texttt{" + tex_id(text) + "}"


def fmt_int(value: float) -> str:
    """Thousands separator for LaTeX (``12{,}820``)."""
    return f"{int(round(value)):,}".replace(",", "{,}")


def fmt_pct(value: float, digits: int = 1) -> str:
    """Percent with a fixed number of decimals and a LaTeX-safe sign."""
    return f"{value:.{digits}f}\\%"


def fmt_num(value: float, digits: int = 2) -> str:
    """Fixed-point number with thousands separators."""
    return f"{value:,.{digits}f}".replace(",", "{,}")
