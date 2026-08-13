#!/usr/bin/env python3
"""Standalone demo: drive Frankenstein in-process via ``src/engine.py``.

This is the Phase 0 DashAI-integration exit criterion (see
``docs/dashai-plugin-audit.md`` §6): build a model from an in-memory config,
run a few training steps in-process, toggle the classification head, and
round-trip a checkpoint through ``save_checkpoint``/``load_checkpoint`` —
without invoking ``src/cli.py`` or spawning the GPU-thermal supervisor.

Run from the repo root:

    uv run --extra cpu python examples/engine_demo.py

For full YAML-driven training in-process (streaming dataset, optimizer,
metrics callback), use the engine's ``train_from_config``:

    from src.engine import train_from_config
    result = train_from_config(
        "configs/mini.yaml",            # path to a Frankenstein YAML
        device="cpu",
        supervisor="off",               # run in this process (no subprocess)
        metrics_callback=lambda m: print(m),
    )
"""
from __future__ import annotations

import os
import sys
import tempfile

import torch
import torch.nn.functional as F

# Allow running directly from the repo root without an installed package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.engine import build_model, load_checkpoint, save_checkpoint  # noqa: E402
from src.model.config import FrankensteinModelConfig  # noqa: E402

VOCAB, HIDDEN, HEADS, SEQ, BSZ = 64, 48, 6, 8, 2


def tiny_config() -> FrankensteinModelConfig:
    """Minimal Frankenstein encoder config that trains in seconds on CPU."""
    return FrankensteinModelConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_layers=1,
        num_loops=1,
        num_heads=HEADS,
        retention_heads=HEADS,
        num_experts=2,
        top_k_experts=1,
        dropout=0.0,
        layer_pattern=["standard_attn"],
        ode_solver="rk4",
        ode_steps=1,
        use_bitnet=False,
        norm_type="layer_norm",
        use_factorized_embedding=False,
        use_moe=False,
        ffn_activation="gelu",
    )


def fake_batch():
    ids = torch.randint(0, VOCAB, (BSZ, SEQ))
    labels = torch.randint(0, VOCAB, (BSZ, SEQ))
    return ids, labels


def train_a_few_steps(model: torch.nn.Module, steps: int = 3) -> list:
    """Run ``steps`` AdamW steps on fake data and return the loss history."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(steps):
        ids, labels = fake_batch()
        logits = model(ids)  # (B, S, vocab)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), labels.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return losses


def main() -> int:
    torch.manual_seed(0)

    print("=" * 60)
    print("1. build_model — encoder (MLM backbone)")
    print("=" * 60)
    model = build_model(None, tiny_config())
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  built {type(model).__name__}: {n_params/1e3:.1f}K params")

    print("\n" + "=" * 60)
    print("2. train a few steps in-process (no CLI, no supervisor)")
    print("=" * 60)
    losses = train_a_few_steps(model, steps=3)
    for i, loss in enumerate(losses, 1):
        print(f"  step {i}: loss={loss:.4f}")
    if losses[-1] < losses[0]:
        print("  -> loss decreased ✓")

    print("\n" + "=" * 60)
    print("3. Strategy A — sequence-level classification head")
    print("=" * 60)
    clf_model = build_model(None, tiny_config(), num_labels=5)
    clf_model.eval()
    with torch.no_grad():
        out = clf_model(fake_batch()[0])
    head_type = type(clf_model.cls_head).__name__
    print(f"  num_labels=5 -> output shape {tuple(out.shape)} (B, num_labels)")
    print(f"  cls_head is {head_type} (full-precision, NOT BitNet)")

    print("\n" + "=" * 60)
    print("4. save_checkpoint -> load_checkpoint round-trip")
    print("=" * 60)
    from src.training.config_loader import load_training_config

    loaded = load_training_config(os.path.join(_REPO_ROOT, "configs", "mini.yaml"))
    ckpt_model = build_model(loaded.model_class, loaded.model_config)
    probe = next(p for p in ckpt_model.parameters() if p.requires_grad).detach().clone()

    with tempfile.TemporaryDirectory() as tmp:
        path = save_checkpoint(tmp, ckpt_model, loaded)
        print(f"  saved bundle -> {tmp}")
        print(f"    files: {sorted(os.listdir(tmp))}")
        model2, loaded2, _tok, extra = load_checkpoint(tmp)

    probe2 = next(p for p in model2.parameters() if p.requires_grad).detach().clone()
    ok = torch.equal(probe, probe2)
    print(f"  reloaded task={loaded2.task!r}, model_class={loaded2.model_class!r}")
    print(f"  weights match after reload: {ok}")

    print("\nDone. All engine primitives work in-process without src/cli.py.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
