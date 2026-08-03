# Schema Hierarchical Refactor Plan

**Status:** DRAFT — awaiting approval before execution.
**Scope:** Breaking refactor (no backward compatibility) of the YAML config schema and its runtime consumers. Two coupled changes:

1. **Hierarchical model schema** — move flat `model.*` keys into nested sub-objects (`model.dims.*`, `model.norm.*`, `model.embedding.*`, `model.attention.<mixer>.*`).
2. **Optimizer param-group cleanup** — remove the `ode` / `retnet` / `mamba` dedicated optimizer parameter groups; route their parameters into the `attention` group. Affects the shared per-group suffix families (`lr_*`, `wd_*`, `betas_*`, `eps_*`).

---

## 1. Target Hierarchy

### 1.1 Model schema (`src/schema/_model.yaml` → restructure into modular files)

Current: a single 1717-line `_model.yaml` with ~60 flat properties under `model:`.

New top-level `model:` block structure:

```yaml
model:
  # --- Core dimensions / architecture (was flat) ---
  dims:
    vocab_size, hidden_size, num_layers, num_loops, num_heads,
    num_kv_heads, retention_heads, dropout, layer_pattern, mode   # 10 keys, unchanged leaf names

  # --- Normalization (was norm_type, prms_partial_ratio) ---
  norm:
    type, partial_ratio    # leaf renames: norm_type→type, prms_partial_ratio→partial_ratio

  # --- Embeddings (was use_factorized_embedding, factorized_embedding_dim, use_embedding_conv, embedding_conv_kernel) ---
  embedding:
    factorized:
      enabled, dim         # renames: use_factorized_embedding→enabled, factorized_embedding_dim→dim
    conv:
      enabled, kernel      # renames: use_embedding_conv→enabled, embedding_conv_kernel→kernel

  # --- Attention (per-mixer subkeys) ---
  attention:
    titan:                  # was positional_encoding, use_hope, hope_base, hope_damping, rope_base, rope_scaling
      positional_encoding
      use_hope
      hope:
        base, damping        # renames: hope_base→base, hope_damping→damping
      rope:
        base, scaling        # renames: rope_base→base, rope_scaling→scaling
    mla:
      latent_rank           # was mla_latent_rank
    gqla:
      latent_rank, num_groups, decode_path   # was gqla_*
    mlra:
      latent_rank, num_latent_heads          # was mlra_*
    tucker:
      query_rank, key_rank, value_rank       # was tucker_*
    iha:
      num_pseudo_heads                        # was iha_num_pseudo_heads
    gta:
      num_shared_groups, value_latent_rank   # was gta_*
    mtla:
      latent_rank, merge_factor, stride      # was mtla_*
    cca:
      latent_rank, num_conv_layers, conv_kernel_seq, conv_kernel_ch,
      qk_mean, value_shift                    # was cca_*
    ccgqa:
      query_latent_rank, kv_latent_rank, num_kv_heads, num_conv_layers,
      conv_kernel_seq, conv_kernel_ch, qk_mean, value_shift   # was ccgqa_*
    msa:
      block_size, topk_blocks, index_dim, kl_loss_weight     # was msa_*
    sparda:
      block_size, topk_blocks, forecast_dim                   # was sparda_*
    engram:
      max_ngram_size, n_heads_per_ngram, embed_dim_per_head,
      kernel_size, seed                                       # was engram_*

  # --- STAY flat at model.* (not moved) ---
  # use_moe, num_experts, top_k_experts
  # use_bitnet, bitnet_routers, use_bitnet_conv
  # use_mixture_of_depths, mixture_of_depths_capacity_ratio, mixture_of_depths_router_aux_loss_weight
  # ffn_hidden_size, ffn_activation, ffn_activation_config
  # ode_solver, ode_steps
```

### 1.2 Schema file restructure

To keep files manageable, split `_model.yaml` into a directory mirroring `src/model/`:

```
src/schema/
  _model.yaml                 # becomes a thin $ref aggregator: properties: dims / norm / embedding / attention / (flat keys)
  _model/
    _dims.yaml                # 10 core dimension fields
    _norm.yaml                # type, partial_ratio
    _embedding.yaml           # factorized + conv
    _attention_titan.yaml     # titan + hope + rope
    _attention_latent.yaml    # mla, gqla, mlra, tucker, iha, gta, mtla  (mirrors src/model/attention/latent/)
    _attention_cca.yaml       # cca, ccgqa
    _attention_sparse.yaml    # msa, sparda (mirrors src/model/attention/sparse/)
    _attention_engram.yaml    # engram (mirrors src/model/attention/engram.py)
    _model_flat.yaml          # the staying-flat keys (moe, bitnet, mod, ffn, ode)
```

`src/schema.yaml` already refs `_model.yaml`; no change to the root file's `$ref` structure, only to what `_model.yaml` contains internally.

`src/schema/_conditional_rules.yaml` — update any `allOf` rules that reference moved field paths (e.g. `model.norm_type` → `model.norm.type`, `model.layer_pattern` → `model.dims.layer_pattern`, `model.num_kv_heads` → `model.dims.num_kv_heads`). Need to read this file before editing to enumerate the rules.

### 1.3 Optimizer param-group cleanup (separate concern, schema side)

In `src/schema/_optimizer.yaml`:
- Lines 186-188, 193-195, 200-202, 207-209: remove the 12 `*_ode` / `*_retnet` / `*_mamba` entries from `shared_per_group_suffixes`.
- Lines 131, 158, 163-165: remove `ode`, `retnet`, `mamba` from the prose lists of groups.

In `src/schema/_examples.yaml` lines 49-51, 56-58, 63-65, 70-72: remove the 12 example keys.

---

## 2. Runtime Code Changes

### 2.1 The central adapter: `src/training/config_loader.py`

**The single construction site is `config_loader.py:166` (`UltraConfig(**model_data)`).** Decision: keep `UltraConfig` a **flat** dataclass (all existing fields, existing leaf names) to avoid touching every attention module, deploy, export, and 20+ tests that read `config.<field>` attributes. Instead, add a **flattening adapter** in `config_loader` that converts the new nested `model:` dict into the flat kwargs `UltraConfig` expects.

New helper `_flatten_model_dict(model_data: dict) -> dict` in `config_loader.py` that:
- Reads `model_data["dims"]` and spreads its 10 keys to the top level.
- Reads `model_data["norm"]` and emits `norm_type = norm["type"]`, `prms_partial_ratio = norm["partial_ratio"]`.
- Reads `model_data["embedding"]["factorized"]` → `use_factorized_embedding`, `factorized_embedding_dim`; `model_data["embedding"]["conv"]` → `use_embedding_conv`, `embedding_conv_kernel`.
- Reads `model_data["attention"]["titan"]` → `positional_encoding`, `use_hope`, `hope_base` (from `titan["hope"]["base"]`), `hope_damping`, `rope_base` (from `titan["rope"]["base"]`), `rope_scaling`.
- Reads each mixer subkey (`mla`, `gqla`, `mlra`, `tucker`, `iha`, `gta`, `mtla`, `cca`, `ccgqa`, `msa`, `sparda`, `engram`) and emits the flat prefixed keys (e.g. `cca_latent_rank = attention["cca"]["latent_rank"]`).
- Passes through the staying-flat keys verbatim from `model_data`.

`_validate_bitnet_flags` (config_loader.py:69-76) and `_validate_ffn_activation` (config_loader.py:45-51) currently read the **flat** `model_data`. After flattening, they continue to work unchanged. (Alternatively, run them BEFORE flattening on the nested dict — but they read `ffn_activation` / `use_bitnet` which are staying flat, so either order works. Simplest: flatten first, then validate.)

### 2.2 `UltraConfig` dataclass (`src/model/tormented_bert_frankestein.py:333-525`)

**No structural change.** Keep all fields flat with current names. `__post_init__` cross-references (`mla_latent_rank ← hidden_size`, etc.) keep working because the dataclass is still flat.

The ONLY optional improvement: add a docstring note that the dataclass is the flat internal representation and the YAML schema is nested (loader flattens). No code change strictly required.

### 2.3 Attention modules, norm factory, embeddings, activation factory

**No changes.** They all read `config.<flat_field>` and the `UltraConfig` instance still exposes flat attributes. Confirmed by the explore report:
- `cca_attn.py:188` reads `getattr(config, "cca_latent_rank", ...)` — unchanged.
- `titan.py:87` reads `config.hope_base` — unchanged.
- `norm/factory.py:39` reads `config.norm_type` — unchanged.
- `factorized_embedding.py:44-50` reads `config.factorized_embedding_dim` etc. — unchanged.

### 2.4 `src/training/main.py`, `src/training/trainer.py`, `src/sbert/train_sbert.py`

**No changes.** They read the `UltraConfig` instance which remains flat. The only access sites (`main.py:140-197`, `trainer.py:932-1183` logging, `train_sbert.py:200-1907`) all touch flat attributes that survive the refactor.

### 2.5 `src/deploy/` (deploy.py, inference.py, transformers_export.py, bitnet_gguf_export.py)

These read **raw dicts** (YAML `model:` block or JSON `config.json`), not `UltraConfig`. They need the same flattening treatment.

- `deploy.py:268, 358, 361` and `inference.py:106` construct `UltraConfig(**config_dict)` from JSON files that previously stored **flat** keys. Since old checkpoints have flat keys and new configs will be nested, add a `_flatten_model_dict` call (reuse the one from config_loader — extract to `src/utils/schema_loader.py` or a new `src/utils/config_flatten.py` so both loader and deploy share it). Old checkpoints with flat keys: the flattener must be tolerant — if `model_data` already has `hidden_size` at top level (old shape), pass through; if it has `dims:` (new shape), flatten. This gives graceful upgrade of old checkpoints without explicit migration.
- `deploy.py:108-110` iterates `saved_config.items()` and `setattr(self.config, key, value)` — wrap with the flattener so nested keys are flattened before setattr. Also `deploy.py:177` `self.config.__dict__.copy()` serialization: since `UltraConfig` stays flat, `__dict__` stays flat — old checkpoints remain loadable, new saves are flat JSON. Acceptable (the on-disk checkpoint format is an internal detail, not the YAML schema).
- `transformers_export.py:42` `_ULTRA_KEYS = {field.name for field in fields(UltraConfig)}` — stays correct (UltraConfig is flat). `transformers_export.py:265` `UltraConfig(**model_config)` — flatten first. `transformers_export.py:289-291` schema introspection for `layer_pattern` enum: update path from `model.properties.layer_pattern` to `model.properties.dims.properties.layer_pattern`.
- `bitnet_gguf_export.py` reads raw `model:` dict via `model_config.get("hidden_size")` etc. (lines 291-442). Replace each `model_config.get("X")` with a flattened lookup, OR flatten the dict once at load (lines 291, 348) and keep the downstream `.get()` calls. Flattening once is cleaner.

### 2.6 `src/streamlit_gui/app.py`

**No changes.** It is fully schema-driven (`app.py:434-446` walks `schema["properties"]["model"]["properties"]` recursively). The new nested schema will auto-render nested expanders and produce nested YAML, which the loader then flattens.

### 2.7 Optimizer param-group cleanup (code side)

- `src/model/optimizer/base.py:14-22` — remove `"ode"`, `"retnet"`, `"mamba"` from `GROUP_NAMES` (keep `embeddings`, `norms`, `attention`, `other`). Update docstring (lines 23-38).
- `src/training/trainer.py:1026-1110` — remove the `ode_params`, `retnet_params`, `mamba_params` accumulators (lines 1026, 1028, 1029), the classification branches (lines 1038-1042), and the three group dicts (lines 1070-1093). Route ODE/RetNet/Mamba parameters into `titan_attn_params`/`attention` group by extending the classification at line 1044 to also match `'ode'`, `'retention'`, `'retnet'`, `'mamba'` substrings. Concretely: merge lines 1038-1042 into line 1044's condition.
- `src/sbert/train_sbert.py:656-735` — telemetry-only `ode`/`retnet`/`mamba` CSV columns. Optional cleanup: remove the columns, or keep them emitting 0.0. Recommend removing for consistency (they never had real values). This is a telemetry-column change, not a schema change — low risk.

**Important runtime behavior note (from research):** `retnet` and `mamba` groups are *already dead* — no parameter name contains "retnet"/"retention"/"mamba" as a substring (the modules store params under generic names like `q_proj`, `mixer`), so those groups were always empty and filtered out at `trainer.py:1113`. Only `ode` is live (`ODEFunc` → `layers.{i}.mixer.ode_func.qkv.weight`). So merging ode→attention is the only behavioral change; retnet/mamba removal is a no-op at runtime.

**Failure mode if old configs still set `adamw-lr_ode`:** `build_optimizer` raises `ValueError` from `ensure_no_unknown_parameters` (factory.py:202-206) because `lr_ode` is no longer in `_COMMON_PER_GROUP_KEYS`. Therefore all 34 YAML configs that set these keys MUST be updated in the same change (§4).

---

## 3. Tests Changes

### 3.1 Tests that build inline `UltraConfig(...)` or inline YAML `model:` blocks

These pass **flat kwargs** to `UltraConfig(...)` (which stays flat) — **no change needed** for kwargs-style construction. Files: `test_model_config.py`, `test_model_variants.py`, `test_attention_modules.py`, `test_gated_attention.py`, `test_latent_attention.py`, `test_sparse_attention.py`, `test_bitnet_coverage.py`, `test_bitnet_quantization.py`, `test_bitnet_export.py`, `test_bitnet_gguf.py`, `test_hybrid_layer.py`, `test_activation_functions.py`, `test_model_training_fake_data.py`, `test_transformers_export.py`, `test_factorized_embedding.py`, `test_attention_refactor.py`, `test_common_modules.py`.

These tests construct `UltraConfig(vocab_size=..., norm_type=..., cca_latent_rank=..., positional_encoding=...)` directly. Since `UltraConfig` keeps flat fields, these all keep working.

### 3.2 Tests that write inline YAML and call `load_training_config`

These exercise the flattener and MUST be updated to the new nested YAML shape:
- `tests/test_config_loader.py` — ~8 inline `model:` blocks (lines 35-283). Rewrite each to nested shape. Assertions on `cfg.model_config.vocab_size` etc. stay valid (UltraConfig still has `vocab_size`).
- `tests/test_bitnet_export.py` `_yaml_text` (lines 26-42) — rewrite to nested.
- `tests/test_bitnet_gguf.py` `_write_yaml` (lines 32-47) — rewrite to nested.
- `tests/test_transformers_export.py` inline YAML `model:` dict (lines 62-97) — rewrite to nested.

### 3.3 Tests that introspect the schema

- `tests/test_schema_attention_layers.py`:
  - Lines 9-14: `schema["properties"]["model"]["properties"]["layer_pattern"]...` → update to `...["model"]["properties"]["dims"]["properties"]["layer_pattern"]...`.
  - Lines 69-113: asserts each of the 36 moved field names is a key in `model_properties`. Update to look them up at their new nested paths (e.g. `cca_latent_rank` → `model.properties.attention.properties.cca.properties.latent_rank`). Rewrite the assertion list to walk nested paths.
- `tests/test_yaml_examples.py` lines 92-171 (`YamlExamplesContentTests`) — assertions on `cfg.model_config.layer_pattern`, `.mode`, `.norm_type`, `.positional_encoding` stay valid (UltraConfig flat attributes unchanged). **No change.** The smoke-test loader path (lines 67-88) also stays valid as long as the flattener works.

### 3.4 Optimizer tests

- `tests/test_optimizer_base.py` lines 162-169 — only asserts `embeddings`/`norms`/`attention`/`other` are in `GROUP_NAMES`. **No change** (already doesn't assert ode/retnet/mamba).
- `tests/test_optimizer_factory.py` — uses only `other`/`embeddings` group names. **No change.**

---

## 4. Config YAML Updates

### 4.1 Top-level configs (8 files with `model:` blocks)

`embbert.yaml`, `frankenstein.yaml`, `frankesteindecoder.yaml`, `mini.yaml`, `standard_hope.yaml`, `standard.yaml`, `tinybert.yaml` — rewrite `model:` block to nested shape. `modernbert_continual.yaml` uses `base_model` (no `model:` block) — unaffected.

### 4.2 Example configs (`configs/examples/*.yaml`)

~30 files with `model:` blocks need the nested rewrite (the 18 `base_model`-only files are unaffected). The per-mixer examples (`es_arch_cca_adamw.yaml`, `es_arch_ccgqa_adamw.yaml`, `es_arch_mla_adamw.yaml`, `es_arch_gqla_adamw.yaml`, `es_arch_mlra_adamw.yaml`, `es_arch_tucker_adamw.yaml`, `es_arch_iha_adamw.yaml`, `es_arch_gta_adamw.yaml`, `es_arch_mtla_adamw.yaml`, `es_arch_msa_adamw.yaml`, `es_arch_sparda_adamw.yaml`) move their mixer-specific keys under `model.attention.<mixer>.*`.

### 4.3 Optimizer key removal in configs (34 files)

The 8 root configs + 26 example configs listed in §4.2 of the optimizer research each set 12 keys (`lr_ode`/`lr_retnet`/`lr_mamba` × lr/wd/betas/eps). Remove all 12 lines from each file's optimizer `parameters:` block. This is a mechanical deletion; do it in the same pass as the model-block nesting rewrite to avoid touching files twice.

### 4.4 `configs/README.md`

- Lines 51-81 (example optimizer block): remove the 12 ode/retnet/mamba lines.
- Lines 166-173 (reference list): remove ode/retnet/mamba from each of the 4 suffix family lines.
- Lines 218-298 ("### model (UltraConfig)" + "### Latent / KV-compression family"): rewrite to document the new nested paths (`model.dims.*`, `model.norm.*`, `model.embedding.*`, `model.attention.<mixer>.*`).

---

## 5. Docs Updates

### 5.1 Paper annexes (both languages)

`docs/paper/appendices/annex-schema.tex` and `docs/paper-es/appendices/annex-schema.tex`:

- **Model fields table** (lines 41-117 / 41-117): restructure into nested subsections. Replace the single flat `longtable` with:
  - `dims.*` rows (10)
  - `norm.*` rows (2)
  - `embedding.factorized.*` + `embedding.conv.*` rows (4)
  - `attention.titan.*` rows (6, with nested hope/rope)
  - `attention.<mixer>.*` rows (grouped per mixer)
  - Staying-flat rows (moe, bitnet, mod, ffn, ode) in their own subsection
- **Layer Pattern Enumeration** (lines 120-174): the enum *values* (mixer names) are unchanged; only the prose intro referencing `layer_pattern` path needs updating to `model.dims.layer_pattern`.
- **Optimizer Shared Per-Group Suffixes** (lines 258-267): remove `lr_ode`/`lr_retnet`/`lr_mamba` and the wd/betas/eps variants from the 4 bullet lists. Update the surrounding prose.
- **Validation Rules** (lines 350-361): update rule #2 (`hidden_size`/`num_heads` → `model.dims.*`), rule #3 (`num_kv_heads` → `model.dims.num_kv_heads`), rule #8 (`layer_pattern` → `model.dims.layer_pattern`).

### 5.2 Specs

`docs/specs/optimizers.md` lines 31-33: remove the ODE/RetNet/Mamba rows from the "Shared Per-Group Suffix Families" table.

`docs/specs/schema-reference.md` (if it enumerates model fields — confirm during execution): rewrite to nested paths.

### 5.3 AGENTS.md

Update the "Where to Edit by Feature Type" table and the "Hard Constraints" section to reflect:
- New nested schema paths.
- The flattener in `config_loader.py` / `src/utils/config_flatten.py`.
- Removed optimizer groups.
- The `src/schema/_model/` directory replacing the single `_model.yaml`.

---

## 6. Execution Order

1. **Read** `src/schema/_conditional_rules.yaml` and `docs/specs/schema-reference.md` to complete the plan's unknowns.
2. **Schema files first** (`src/schema/_model/` directory + `_optimizer.yaml` + `_examples.yaml` + `_conditional_rules.yaml`). Validate they parse via `resolve_schema`.
3. **Shared flattener** in `src/utils/config_flatten.py` (new) + unit-test it in isolation.
4. **`config_loader.py`** — call flattener before `UltraConfig(**...)`.
5. **Deploy/export** — wire flattener into `deploy.py`, `inference.py`, `transformers_export.py`, `bitnet_gguf_export.py`; fix the schema-introspection path in `transformers_export.py:289-291`.
6. **Optimizer code** — `base.py` `GROUP_NAMES`, `trainer.py` classification, SBERT telemetry.
7. **Config YAMLs** — rewrite all 8 + ~30 example files (nesting + optimizer-key removal in one pass).
8. **Tests** — update the 4 inline-YAML tests + `test_schema_attention_layers.py` schema-path assertions. Run full suite.
9. **Docs** — `configs/README.md`, `docs/specs/optimizers.md`, both `annex-schema.tex`, AGENTS.md.
10. **Verify** — `conda run -n frankestein python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings`.

---

## 7. Risks & Notes

- **Checkpoint compatibility:** `trainer.py` pickles `UltraConfig` into `.pt` files. Since `UltraConfig` stays flat, old checkpoints still load. New checkpoints save flat `__dict__`. The JSON `config.json` saved by `deploy.py:177` is flat; the flattener tolerates both flat (old) and nested (new) on read.
- **HF export (`transformers_export.py`):** exported `config.json` stores a flat `ultra_config` dict. Stays flat (UltraConfig is flat). Old exported models remain importable. Only the YAML *input* to the exporter changes shape (flattened by the exporter before `UltraConfig`).
- **`build_optimizer` strictness:** removing `ode`/`retnet`/`mamba` from `GROUP_NAMES` makes `build_optimizer` reject configs still setting those keys. All 34 YAMLs must be updated in step 7 before running any training test. The smoke tests in `test_yaml_examples.py` only call `load_training_config` (schema-permissive, `additionalProperties: true` on `parameters`) so they won't catch a stale `lr_ode` key — only a real `build_optimizer` call would. Consider adding one `build_optimizer` smoke test in step 8 to guard against regressions.
- **`retnet`/`mamba` groups are already dead** at runtime (never populated). Their removal is purely cosmetic for the user; only `ode`→`attention` is a real behavioral merge.
- **Streamlit auto-adapts** (schema-driven). No GUI code change.
- **No backward compatibility** per user instruction: no migration shim for old YAML files (they must be rewritten), no dual-shape support in the schema. The flattener's tolerance for flat-on-disk checkpoints/JSON is a robustness nicety, not a compatibility promise.