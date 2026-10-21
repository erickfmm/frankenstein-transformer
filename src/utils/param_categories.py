"""Canonical parameter-block taxonomy shared by tooling and training logs.

The block set mirrors the breakdown exposed by the website parameter
estimator (``ft-param-estimator.js``) and by the parameter ground-truth
script ``full_tests/param_count_check.py``:

    embeddings, pos, attention, ffn, norm, head, mhc, router, residual, other

Keeping a single categorizer guarantees that per-block gradient norms logged
during training line up one-to-one with the parameter-count breakdown shown
on the website, and that both stay aligned with the real modules in
``src/model/**``.

The categorizer is path-aware (it inspects every dotted segment), so nested
parameters such as ``layers.0.mixer.q_norm.weight`` are attributed to the
mixer (``attention``) rather than to the standalone block norm (``norm``),
matching the estimator's per-module formulas.
"""

from __future__ import annotations

from typing import Final, Tuple

#: Ordered block categories; the order is the canonical logging/display order.
BLOCK_CATEGORIES: Final[Tuple[str, ...]] = (
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

#: Top-level module names that hold task/output heads.
_HEAD_MODULES: Final[frozenset[str]] = frozenset(
    {
        "head",
        "cls_head",
        "classification_head",
        "patch_pred_head",
        "seg_head",
        "seg_upsampler",
    }
)


def categorize_param(name: str) -> str:
    """Map a parameter name to a canonical block category.

    Args:
        name: Dotted parameter name as produced by
            :meth:`torch.nn.Module.named_parameters`.

    Returns:
        One of :data:`BLOCK_CATEGORIES`.
    """
    parts = [part for part in name.split(".") if part]
    if not parts:
        return "other"

    # Decoder wraps the encoder as ``backbone``; ViT keeps its components at
    # the top level.
    top = parts[1] if parts[0] == "backbone" and len(parts) > 1 else parts[0]

    # Embedding / patch embedding.
    if top in ("emb", "patch_embed"):
        return "embeddings"

    # Positional encodings, plus the ViT cls/mask tokens that the estimator
    # folds into its ``pos`` bucket.
    if "pos_encoder" in parts or "positional_encoding" in parts:
        return "pos"
    if top in ("cls_token", "mask_token"):
        return "pos"

    # mHC stream expansion/collapse and Hyperloop per-loop parameters.
    if any(
        part == "mhc" or part.startswith("mhc_")
        or part == "hyperloop" or part.startswith("hyperloop_")
        for part in parts
    ):
        return "mhc"

    # AttnRes residual module (full/block attention over residuals).
    if "residual" in parts:
        return "residual"

    # MoE router and Mixture-of-Depths depth router.
    if any(part in ("router", "depth_router") for part in parts):
        return "router"

    # FFN bodies and MoE experts.
    if "ffn" in parts or "experts" in parts:
        return "ffn"

    # Attention mixers, including their internal norms (QK-norm, GroupNorm,
    # etc.), which the estimator counts inside the mixer's parameter budget.
    if "mixer" in parts:
        return "attention"

    # Standalone block norms: norm1, norm2, final_norm and *_norm modules.
    if any(part in ("norm1", "norm2", "final_norm") for part in parts):
        return "norm"
    if any(part == "norm" or part.endswith("_norm") for part in parts):
        return "norm"

    # Task/output heads.
    if top in _HEAD_MODULES:
        return "head"

    return "other"
