#!/usr/bin/env python3
"""Ground-truth parameter counter for the web param estimator.

Builds every example config through the real engine (load_training_config →
build_model) and dumps per-category parameter counts to JSON. The companion
Node script (param_estimate_check.mjs) runs the website's
``ft-param-estimator.js`` on the same configs and compares totals.

Categories mirror the estimator's breakdown:
  embeddings, pos, attention, ffn, norm, head, mhc, router, residual, other

Usage:
    .venv/bin/python full_tests/param_count_check.py [out.json] [--configs GLOB ...]

Default output: full_tests/param_truth.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only; we never run data

import torch  # noqa: E402

from engine import build_model  # noqa: E402
from training.config_loader import load_training_config  # noqa: E402


def _categorize(name: str) -> str:
    """Map a top-level-ish module path to a breakdown category."""
    # Decoder wraps everything in `backbone.`; ViT has its own extras.
    n = name.split("layers.")[0].rstrip(".")
    leaf = name.rsplit(".", 1)[-1]
    if n in ("emb", "backbone.emb", "patch_embed"):
        return "embeddings"
    if "pos_encoder" in name:
        return "pos"
    if leaf in ("mixer",) or ".mixer." in name or n.endswith(".mixer"):
        return "attention"
    if leaf in ("ffn",) or ".ffn." in name or n.startswith("ffn") or ".experts" in name or ".router" in name or leaf == "experts":
        # experts/router handled below; keep ffn separate
        pass
    if leaf in ("norm1", "norm2", "final_norm") or n.endswith("final_norm"):
        return "norm"
    if "mhc" in name:
        return "mhc"
    if leaf in ("router", "depth_router"):
        return "router"
    if leaf in ("residual",):
        return "residual"
    if leaf in ("head", "cls_head", "classification_head", "patch_pred_head", "seg_head", "seg_upsampler"):
        return "head"
    if leaf in ("ffn",) or ".ffn." in name or ".experts" in name:
        return "ffn"
    if leaf in ("cls_token", "mask_token"):
        return "other"
    return "other"


def count_model(model: torch.nn.Module) -> Dict[str, int]:
    cats: Dict[str, int] = {}
    total = 0
    for pname, p in model.named_parameters():
        c = _categorize(pname)
        cats[c] = cats.get(c, 0) + p.numel()
        total += p.numel()
    cats["total"] = total
    return cats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=os.path.join(REPO, "full_tests", "param_truth.json"))
    ap.add_argument("--configs", nargs="*", default=None)
    args = ap.parse_args()

    patterns = args.configs or [
        os.path.join(REPO, "configs", "*.yaml"),
        os.path.join(REPO, "configs", "examples", "*.yaml"),
    ]
    paths: list[str] = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)))

    results: Dict[str, Any] = {}
    failures: Dict[str, str] = {}
    for path in paths:
        name = os.path.relpath(path, REPO)
        try:
            with open(path) as fh:
                import yaml

                raw = yaml.safe_load(fh)
            loaded = load_training_config(path)
            if loaded.model_config is None:
                continue  # base-model configs: nothing to build
            # Meta device: builds the full module tree with shape-only
            # tensors (no RAM/VRAM), so 70B-param presets count instantly.
            with torch.device("meta"):
                model = build_model(loaded.model_class, loaded.model_config)
            cats = count_model(model)
            results[name] = {
                "model_class": loaded.model_class or "frankenstein",
                "layer_pattern": list(getattr(loaded.model_config, "layer_pattern", []) or []),
                "cats": cats,
                "config": raw,
            }
        except Exception as e:  # noqa: BLE001
            failures[name] = f"{type(e).__name__}: {e}"

    payload = {"results": results, "failures": failures}
    with open(args.out, "w") as fh:
        json.dump(payload, fh)
    ok = len(results)
    print(f"Wrote {args.out}: {ok} configs counted, {len(failures)} failures")
    for k, v in failures.items():
        print(f"  FAIL {k}: {v[:120]}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
