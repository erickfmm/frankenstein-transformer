# Plan: Implement FlashNorm (arXiv:2407.09577)

## Paper
**Title**: "FlashNorm: Fast Normalization for Transformers"
**Authors**: Nils Graef, Filip Makraduli, Andrew Wasielewski, Matthew Clapp (OpenMachine, 2026). arXiv:2407.09577v5.
**Bib key**: `graef2026flashnorm`

FlashNorm is an *exact algebraic rewrite* of `RMSNorm → Linear`, not a new mathematical normalization. Three propositions:
- **Prop. 1 (weight folding)**: `W* = diag(g)·W` — eliminates the per-dim scale `g`.
- **Prop. 2 (deferred RMS)**: `(a/RMS(a))·W* = (a·W*)·(1/RMS(a))` — matmul ‖ RMS run in parallel.
- **Prop. 3 (RMS cancellation)**: `RMSNorm → Linear → RMSNorm` lets the first RMSNorm be dropped entirely (QKV-norm, MLA latent-norm).

## Design decisions (confirmed by user)
- **Semantics**: Full fused integration — standalone module + `FlashNormLinear` pair wired into `HybridLayer`'s centralized FFN input projection; per-mixer QKV fusion deferred to kernel-level work.
- **Partial ratio**: New optional `flashnorm_partial_ratio` knob in `_norm.yaml` (composes Prop. 1 with pRMSNorm).
- **Annex scope**: Full FlashNorm subsection in `annex-7-*` + summary table update, English first then Spanish.

## Architectural scope (pragmatic "full fused")
- **Standalone `FlashNorm` module** returned by `get_norm(config)` — weightless RMSNorm (no learnable `g`) with optional `partial_ratio`. Drop-in replacement.
- **`FlashNormLinear`** fuses `FlashNorm + nn.Linear`: applies Prop. 2 (deferred scalar after matmul).
- **`FlashNormBitLinear`** wraps BitLinear: folding fp `g` into ternary weights breaks quantization → falls back to sequential `FlashNorm → BitLinear` when BitNet is on (graceful, documented).
- **`fold_norm_weights(linear, g)` utility** for post-hoc weight folding of pretrained checkpoints (Prop. 1).
- **FFN input projection in `HybridLayer`**: fully fused when `norm_type == "flash_norm"`.
- **Pre-attention QKV path**: standalone `FlashNorm`; post-hoc recipe documented in annex. Per-mixer auto-fusion is left to a future Triton kernel (matches paper §5 recommendation).

## Files to Create/Modify

### Phase 1 — Core module + schema (foundation)
1. **NEW** `src/model/norm/flash.py` — `FlashNorm`, `FlashNormLinear`, `FlashNormBitLinear`, `fold_norm_weights`.
2. `src/model/norm/__init__.py` — exports.
3. `src/model/norm/factory.py` — dispatch `flash_norm`.
4. `src/schema/_model/_norm.yaml` — enum + `flashnorm_partial_ratio` field (EN + ES descriptions).
5. `src/model/tormented_bert_frankestein.py` (`UltraConfig`) — new field, validation, docstring.
6. `src/utils/config_flatten.py` — map `norm.flashnorm_partial_ratio → flashnorm_partial_ratio`.

### Phase 2 — Fused integration in HybridLayer
7. `src/model/tormented_bert_frankestein.py` (`HybridLayer`) — wrap FFN input projection with `FlashNormLinear`/`FlashNormBitLinear` when `norm_type == "flash_norm"`.

### Phase 3 — Tests
8. `tests/test_common_modules.py` — `FlashNormTests`, `FlashNormLinearTests`, factory test.
9. `tests/test_model_variants.py` — forward, default ratio, ratio validation.
10. `tests/test_config_loader.py` — YAML fixture with `flashnorm_partial_ratio`.

### Phase 4 — Example config (required by AGENTS.md §9)
11. **NEW** `configs/examples/norm_flashnorm.yaml` — smoke-test config.

### Phase 5 — Docs
12. `docs/bibliography/other.bib` — new entry `graef2026flashnorm`.
13. `docs/paper/appendices/annex-7-norm-bitnet-mod.tex` (EN) — subsubsection + Props + pseudocode + tables + comparison row.
14. `docs/paper/appendices/annex-summary-tables.tex` (EN) — 5 → 6 norms; new row.
15. `docs/paper-es/appendices/annex-7-norm-bitnet-mod.tex` (ES) — translation of #13.
16. `docs/paper-es/appendices/annex-summary-tables.tex` (ES) — translation of #14.
17. `AGENTS.md` — hard constraint #4 enum update.

### Phase 6 — Verification
18. `conda run -n frankestein python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings`

## LaTeX content to add (FlashNorm subsection)

### Mathematical formulation
- Standard RMSNorm: `y_i = g_i · a_i / RMS(a)`, `RMS(a) = sqrt((1/n)Σ a_i²)`
- Followed by linear `W`: `z = RMSNorm(a) · W`

**Prop. 1 (Weight folding)**: `W*_{i,j} = g_i · W_{i,j}` ⇒ `z = (a/RMS(a)) · W*`. The weight `g` becomes redundant.

**Prop. 2 (Deferred normalization)**: for bias-free `W*`: `z = (a · W*) · (1/RMS(a))`. Matmul and RMS are independent → parallelize on matrix/vector units.

**Prop. 3 (RMS cancellation)**: `RMSNorm((a/RMS(a))·W*) = RMSNorm(a·W*)` by RMS scale invariance. First RMSNorm redundant.

### Pseudocode (PyTorch)
```python
# Prop. 1: weight folding at load time
W_star = g.unsqueeze(1) * W          # diag(g) @ W; g then discarded

# Prop. 2: forward pass with deferred RMS (bias-free)
b = a @ W_star                       # matrix unit / tensor cores
rms_inv = a.pow(2).mean(dim=-1).rsqrt()  # vector unit, independent of b
z = b * rms_inv.unsqueeze(-1)        # single vector-scalar multiply
# With bias: z = b * rms_inv.unsqueeze(-1) + c  (Prop. 2 fails but matmul ‖ RMS still parallel)
```

### Latency table (NVIDIA T4 GPU, norm-then-project op)
| Scale | Tokens | Sequential | FlashNorm | Speedup |
|---|---|---|---|---|
| SmolLM2-135M | 4096 | 0.704 ms | 0.468 ms | +33.6% |
| SmolLM2-135M | 8192 | 0.929 ms | 0.599 ms | +35.5% |
| Llama-7B | 1024 | 1.882 ms | 1.654 ms | +12.1% |
| Llama-7B | 4096 | 7.628 ms | 6.570 ms | +13.9% |

### Zero-loss folding validation (lm-evaluation-harness, fp16/fp32, A100)
| Model | Precision | Result |
|---|---|---|
| SmolLM2-135M | fp16 | max diff = 0.0, cosine sim = 1.0 |
| Llama-3.2-1B | fp32 | bit-identical greedy generation |
| Llama-3.1-8B | fp32 | bit-identical greedy generation |

### Extensions (brief)
- LayerNorm mean-centering folds into preceding linear: `V*_{i,j} = V_{i,j} - s_i/n` where `s_i = Σ_j V_{i,j}`.
- GLU FFNs: scale both Gate and Up projections; defer one to FFN output.
- RoPE attention: fuse `1/RMS(a)` into precomputed cos/sin tables.

## Verification criteria
- `pytest tests/` passes (all existing tests + new FlashNorm tests).
- `FlashNorm` with `partial_ratio=0` is bit-identical to `RMSNorm(weight=1)` for same `eps`.
- `FlashNormLinear` (bias-free) is bit-identical to `RMSNorm + nn.Linear` after weight folding.
- `FlashNormLinear` (with bias) applies RMS before bias (Prop. 2 doesn't hold).
- Schema validation accepts `norm: { type: flash_norm, flashnorm_partial_ratio: 0.25 }`.
- `configs/examples/norm_flashnorm.yaml` loads via `test_yaml_examples.py`.
