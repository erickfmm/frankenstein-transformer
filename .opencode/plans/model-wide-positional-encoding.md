# Model-Wide Positional Encoding Implementation Plan

## Status: IN PROGRESS
## Created: 2026-08-17
## Scope: Code-only (Phase 1). Docs/paper deferred to Phase 2.

## Overview

Add a **model-level `positional_encoding` enum** + **`positional_encoding_parameters` block** as the single source of truth for positional encoding across all attention mixers. Implement three new PEs from the literature (PaPE, NoPE, ALiBi) plus sinusoidal and learned-absolute variants. Wire a single shared PE module into every mixer via `HybridLayer`, with per-mixer `use_pe` flags to opt out for recurrent/decay mixers.

## Enum (11 values, model-level)

`rope, hope, nope, alibi, pape, pape_efficient, pape_ri, sinusoidal_absolute, sinusoidal_rotary, learned_absolute, none`

## References

- **PaPE** — arXiv:2602.01418 (Øhrstrøm et al., 2026). Code: `pape/nn/positions/{pape_naive,pape_efficient,pape_ri}.py` at https://github.com/DTU-PAS/parabolic-position-encoding
- **ALiBi** — arXiv:2108.12409 (Press & Smith, 2021)
- **NoPE** — arXiv:2305.19466 (Kazemnejad et al., 2023)
- **Sinusoidal** — arXiv:1706.03762 (Vaswani et al., 2017)
- **RoPE** — arXiv:2104.09864 (Su et al., 2024)
- **HoPE** — arXiv:2503.0 (Dai et al., 2025)

## Architecture Decisions (locked in)

1. **Shared injected module**: One PE module built in `FrankensteinEncoder`/`FrankensteinViT`, injected into every `HybridLayer` → every mixer via `__init__(config, pos_encoder=None)` + `forward(..., pos_encoder=None)`.
2. **Per-mixer `use_pe` flag**: defaults `True` for attention mixers, `False` for recurrent/decay mixers (`retnet`, `retnet_attn`, `mamba`, `ode`, `gla_attn`, `deltanet_attn`, `gated_deltanet_attn`, `gated_deltanet2_attn`, `hgrn2_attn`, `fox_attn`, `kda_attn`, `mtla_attn`). User-overridable.
3. **Sinusoidal: both variants** — `sinusoidal_absolute` (additive, embedding-level) + `sinusoidal_rotary` (q/k-level, absolute-position sin/cos rotation).
4. **ViT unification**: `pos_embedding_type` becomes an alias for `positional_encoding`. `learned_1d` → `learned_absolute`, `none` → `none`. New 2D options (`sinusoidal_absolute`, `pape`) added.
5. **Execution**: All code in one pass (Phase 1). Docs/paper in Phase 2 (deferred).

## Three PE injection styles

1. **Rotation** (rope, hope, sinusoidal_rotary): `q = pos_encoder(q, logical_layer_idx)`, `k = pos_encoder(k, logical_layer_idx)` after projection, before scores.
2. **Score bias** (alibi): `attn_scores += pos_encoder.bias(seq_len, device, dtype)` before softmax.
3. **q/k augmentation** (pape, pape_efficient, pape_ri): `(q, k) = pos_encoder.encode_qk(x, q, k, positions)` — replaces q/k, pads to mult of 8.
4. **Additive** (sinusoidal_absolute, learned_absolute, none): no-op in attention (handled at embedding level).

## Files to create (9 new)

| File | Contents |
|---|---|
| `src/model/embeddings/nope.py` | `NoPE` (identity, arXiv:2305.19466) |
| `src/model/embeddings/alibi.py` | `ALiBi` (Press slopes + `bias(seq_len)`, arXiv:2108.12409) |
| `src/model/embeddings/pape.py` | `PaPE` (port `pape_naive.py`, arXiv:2602.01418) |
| `src/model/embeddings/pape_efficient.py` | `PaPEEfficient` (port `pape_efficient.py`) |
| `src/model/embeddings/pape_ri.py` | `PaPERI` (port `pape_ri.py`, rotation-invariant) |
| `src/model/embeddings/sinusoidal.py` | `SinusoidalAbsolute` (additive, Vaswani 2017) + `SinusoidalRotary` (absolute sin/cos rotation) |
| `src/model/embeddings/learned_absolute.py` | `LearnedAbsolutePE` (additive `nn.Parameter`, factored from ViT's `pos_embed`) |
| `src/model/embeddings/factory.py` | `build_pos_encoder(config)` dispatch on `config.positional_encoding` enum |
| `src/schema/_model/_positional_encoding.yaml` | New schema: `positional_encoding` enum + `positional_encoding_parameters` block (bilingual) |

## Files to edit (existing)

### Embeddings
- `src/model/embeddings/__init__.py` — export all 8 new classes + `build_pos_encoder`

### Config
- `src/model/config.py` — new fields (lines 267-273 region): `positional_encoding: str = "rope"`, `positional_encoding_parameters: dict`, per-mixer `<mixer>_use_pe: bool` flags, ALiBi/PaPE/sinusoidal/learned knobs; validation at lines 542-549 (extend enum) and 613-616 (ViT alias)

### Schema
- `src/schema/_model.yaml` — add `positional_encoding` + `positional_encoding_parameters` `$ref`s (alongside `dims`, `norm`, `embedding`, `attention` at lines 30-83)
- `src/schema/_model/_model_flat.yaml` — add per-mixer `use_pe` boolean flags (pattern: lines 190-214)
- `src/schema/_model/_attention_titan.yaml` — mark titan-specific `positional_encoding`/`use_hope`/`hope`/`rope` as deprecated/legacy (lines 4-128)
- `src/schema/_image.yaml` — extend `pos_embedding_type` enum (lines 73-92) with `sinusoidal_absolute`, `learned_absolute`, `pape`, `sinusoidal_rotary`

### Flattener
- `src/utils/config_flatten.py` — flatten `model.positional_encoding` + `model.positional_encoding_parameters.*` + per-mixer `use_pe` flags (update `_flatten_titan` at lines 214-231 + add new flattening in `flatten_model_dict` at lines 234+)

### Encoder/Decoder/ViT
- `src/model/frankenstein_encoder.py` — build `self.pos_encoder = build_pos_encoder(config)` (after line 100), pass to `HybridLayer(config, layer_type, pos_encoder=self.pos_encoder)` (lines 102-107), apply additive PE in forward after `self.emb` (line 153)
- `src/model/frankenstein_decoder.py` — no direct change (delegates to encoder, line 111)
- `src/model/frankenstein_vit.py` — build `self.pos_encoder` (replace `pos_embed` at lines 147-152), apply in forward (line 417), pass to HybridLayer (lines 158-164)
- `src/model/hybrid_layer.py` — `__init__` accepts `pos_encoder` (line 108), stores `self.pos_encoder`; `_forward_dense` passes `pos_encoder=self.pos_encoder` to `self.mixer(...)` at lines 339-346 (all four branches); same for `_forward_dense_mhc` at lines 426-437

### Attention mixers (35 files) — uniform interface

Each mixer gains `__init__(self, config, pos_encoder=None)` and `forward(..., pos_encoder=None)`. A shared helper `src/model/attention/common.py::apply_pe(...)` dispatches on PE type (rotation / score-bias / q/k-aug / no-op).

- `src/model/attention/common.py` — add `apply_pe` helper
- `src/model/attention/standard.py`, `sigmoid.py`, `grouped_query_attention.py` — add PE call after q/k projection
- `src/model/attention/titan.py` — replace titan-specific `pos_encoder` (lines 81-97) with injected module; keep `logical_layer_idx` (lines 121-122)
- `src/model/attention/latent/mla_attn.py` — delete `_apply_rope` (lines 41-58), use injected module (lines 131-132)
- `src/model/attention/latent/cca_attn.py` — delete `_apply_rope` (lines 61-78), use injected module (lines 307-308, 554-555)
- `src/model/attention/latent/{gqla,mlra,tucker,iha,gta,mtla,gma}_attn.py` — add PE call
- `src/model/attention/gated/{gla,deltanet,gated_deltanet,gated_deltanet2,hgrn2,fox,kda,gated_softmax}_attn.py` — default `use_pe=False`; honor flag
- `src/model/attention/sparse/{bigbird,longformer,sparse_transformer,sparsek,nsa,sparda,msa,sparge,fasa}_attn.py` — add PE call
- `src/model/attention/retnet.py`, `ode.py`, `engram.py` — default `use_pe=False`; honor flag

### Examples (CI rule #9 — auto-tested by `tests/test_yaml_examples.py` lines 80-83)

- `configs/examples/pe_{rope,hope,nope,alibi,pape,pape_efficient,pape_ri,sinusoidal_absolute,sinusoidal_rotary,learned_absolute}.yaml` + `pe_mixed_use_pe.yaml`

### Tests
- `tests/test_positional_encodings.py` — `NoPETests`, `ALiBiTests`, `PaPETests`, `PaPEEfficientTests`, `PaPERITests`, `SinusoidalAbsoluteTests`, `SinusoidalRotaryTests`, `LearnedAbsolutePETests`, `BuildPosEncoderFactoryTests`
- `tests/test_attention_modules.py` — update `_cfg`/`_make` (lines 18-41, 128-130) to pass `pos_encoder`; add per-mixer PE-on/PE-off tests
- `tests/test_attention_refactor.py` — update import block (lines 18-23) + `positional_encoding_override` test (lines 109-114) + `invalid_positional_encoding_raises` (lines 116-118)
- `tests/test_model_config.py` — new enum values + `positional_encoding_parameters` validation + per-mixer `use_pe` defaults + legacy compat
- `tests/test_yaml_examples.py` — auto-injects for 11 new YAMLs (lines 80-83); add content assertions in `YamlExamplesContentTests` (line 92+)

## Execution order

1. Create 8 new embedding modules + factory + `__init__`
2. Update `config.py` (fields + validation)
3. Update schema (`_positional_encoding.yaml`, `_model.yaml`, `_model_flat.yaml`, `_attention_titan.yaml`, `_image.yaml`)
4. Update `config_flatten.py`
5. Update `hybrid_layer.py` (accept + forward `pos_encoder`)
6. Update all 35 mixer files (uniform interface + `apply_pe` helper in `common.py`)
7. Update `frankenstein_encoder.py` + `frankenstein_vit.py` (build + inject + additive PE)
8. Add 11 example YAMLs
9. Update/create tests
10. Verify: `conda run -n frankenstein python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings`

## Phase 2 (follow-up, after green CI) — DEFERRED

- Bibliography: 4 new `.bib` entries (ohrstrom_pape_2026, press_alibi_2021, kazemnejad_nope_2023, vaswani_attention_2017)
- Paper/paper-es: new "Positional Encoding" annex, extended embedding table, schema annex, vision annex
- Specs: `attention-mixers.md`, `schema-reference.md`, `vision.md`

## Per-mixer use_pe defaults

**Attention mixers (use_pe=True by default):**
standard_attn, titan_attn, sigmoid_attn, gated_softmax_attn, gqa_attn, mla_attn, gqla_attn, mlra_attn, tucker_attn, iha_attn, gta_attn, cca_attn, ccgqa_attn, msa_attn, sparda_attn, gma_attn, longformer_attn, bigbird_attn, sparse_transformer_attn, sparsek_attn, nsa_attn, fasa_attn, sparge_attn, engram_attn

**Recurrent/decay mixers (use_pe=False by default):**
retnet, retnet_attn, mamba, ode, gla_attn, deltanet_attn, gated_deltanet_attn, gated_deltanet2_attn, hgrn2_attn, fox_attn, kda_attn, mtla_attn

## Schema layout (new `model.positional_encoding_parameters`)

```yaml
positional_encoding:
  type: string
  enum: [rope, hope, nope, alibi, pape, pape_efficient, pape_ri,
         sinusoidal_absolute, sinusoidal_rotary, learned_absolute, none]
  default: rope

positional_encoding_parameters:
  type: object
  additionalProperties: false
  properties:
    rope:        { base: number, scaling: number }
    hope:        { base: number, damping: number }
    alibi:       { num_heads: integer }
    pape:        { num_parabolas: integer, num_positions: integer, rotation_invariant: boolean }
    sinusoidal:  { max_len: integer, base: number, scale: number }
    learned:     { max_len: integer, init_std: number }
    use_pe:      # per-mixer booleans
      standard_attn: bool
      titan_attn: bool
      # ... one per mixer
```

## Flat config keys (after flattening)

- `positional_encoding` (str)
- `positional_encoding_parameters` is NOT flattened as a dict; its leaves are flattened:
  - `rope_base`, `rope_scaling` (existing, kept)
  - `hope_base`, `hope_damping` (existing, kept)
  - `alibi_num_heads` (new)
  - `pape_num_parabolas`, `pape_num_positions`, `pape_rotation_invariant` (new)
  - `sinusoidal_max_len`, `sinusoidal_base`, `sinusoidal_scale` (new)
  - `learned_max_len`, `learned_init_std` (new)
- Per-mixer: `<mixer>_use_pe` (e.g. `standard_attn_use_pe`, `retnet_use_pe`, ...) — new booleans