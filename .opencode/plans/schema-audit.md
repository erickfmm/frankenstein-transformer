# Full Schema Audit — `frankenstein-transformer`

**Date:** 2026-10-19
**Scope:** `src/schema.yaml` + all 15 files in `src/schema/` + all 15 files in `src/schema/_model/` (≈7,000 schema lines), cross-referenced against the runtime consumers: `src/model/config.py` (`FrankensteinModelConfig`), `src/utils/config_flatten.py`, `src/training/config_loader.py`, `src/training/trainer.py` (`TrainingConfig`), `src/training/main.py`, `src/engine.py`, `src/utils/schema_loader.py`, and `src/streamlit_gui/app.py`.

**Requested audit axes:**
1. Legacy fields no longer necessary
2. Bad fields / badly placed fields
3. Fields no longer necessary
4. Misalignment with YAML-schema best practices and good hierarchical organization

---

## Executive summary

The modular `$ref` architecture (root → `schema/_*.yaml` → `schema/_model/_*.yaml`) is sound and consistently mirrors `src/model/`. Strictness (`additionalProperties: false` everywhere) is excellent. The dominant problems are:

- **An unfinished migration**: the hierarchical refactor moved `dims/norm/embedding/attention` into nested blocks but left MoE, BitNet, MoD, ODE, SSMax, and the NLP classification head as flat `use_*`-boolean + scattered-parameter clusters that contradict the `subblock.enabled` idiom used everywhere else (`embedding.factorized.enabled`, `mhc.enabled`, …).
- **A legacy titan positional-encoding subtree** that duplicates the model-wide PE system, is self-marked deprecated, has a divergent enum (no `bam`), and silently loses precedence conflicts in the flattener.
- **Three competing sources of defaults** (dataclass, schema `default:`, schema `examples[0]`) with the Streamlit GUI resolving `examples[0]` *before* `default` — this already flips effective defaults (e.g. `use_bitnet` is `True` in the dataclass, `false` in schema examples).
- **Copy-paste validation machinery**: 24 identical `if/then` optimizer blocks in `_training.yaml`, a triplicated optimizer-class enum (23 values ×3 counting prose), and a quad-documented `layer_pattern` enum (43 values ×4 including bilingual tables).

---

## A. Legacy fields not necessary (explicitly superseded/deprecated)

### A1. `model.attention.titan.positional_encoding` — LEGACY, self-documented
- **File:** `src/schema/_model/_attention_titan.yaml:4-33`
- **Evidence:** Its own description begins "**Legacy.** This titan-specific `positional_encoding` is superseded by the model-level `model.positional_encoding` enum… Kept for backward compatibility."
- **Additional defect:** its enum (11 values) is missing `bam`, which the model-level enum supports (`_positional_encoding.yaml:40-52` has 12). The legacy path can no longer express the full feature set — divergence is untracked by any test.
- **Recommendation:** schedule removal (deprecation window → delete). While it exists, sync or restrict the enum and have `tests/test_schema_attention_layers.py` assert enum-subset parity with the model-level enum.

### A2. `model.attention.titan.use_hope` — DEPRECATED, self-documented
- **File:** `_attention_titan.yaml:34-50`
- **Evidence:** "**Deprecated.** Legacy fallback… Will be removed in future versions." Runtime counterpart: `config.py:455,914,923` keeps realigning `use_hope` from `positional_encoding`, and `titan.py:87` still reads it as a fallback.
- **Recommendation:** remove field + the fallback in `titan.py:87` + the `use_hope` line in `config_flatten.py:249` in the same change.

### A3. `model.attention.titan.hope.{base,damping}` / `model.attention.titan.rope.{base,scaling}` — duplicated knobs
- **Files:** `_attention_titan.yaml:51-132` vs `_positional_encoding.yaml` (`positional_encoding_parameters.rope/hope`)
- **Evidence:** both paths flatten to the *same* dataclass fields (`hope_base`, `hope_damping`, `rope_base`, `rope_scaling` — see `config_flatten.py:250-261` vs `264-278`). Ordering in `flatten_model_dict` flattens `attention` (line 404) **before** `positional_encoding_parameters` (line 411), so the model-level block silently wins if both are set — no warning, no schema rule forbidding the combination.
- **Impact:** two YAML spellings for one knob; silent last-writer-wins precedence; users reading titan docs get stale guidance.
- **Recommendation:** drop the titan `hope:`/`rope:` sub-objects together with A1/A2; at minimum add a `_conditional_rules.yaml` `if/then` that rejects configs setting both `attention.titan.{hope,rope}` and `positional_encoding_parameters.{hope,rope}`.

### A4. `image.pos_embedding_type: learned_1d` — legacy alias
- **File:** `_image.yaml:73-109`
- **Evidence:** description says `learned_1d` "is a legacy alias for `learned_absolute`". The enum accepts 12 values, 11 of which duplicate the model-level `model.positional_encoding` enum — a third PE selector (see B9).
- **Recommendation:** deprecate `learned_1d` (keep as alias one release), and long-term unify with `model.positional_encoding` + `positional_encoding_parameters`.

### A5. Loader-only legacy alias `dataset:` → `vision_dataset:`
- **File:** `src/training/config_loader.py:250` — `data.get("vision_dataset", data.get("dataset", {}) or {})`
- **Evidence:** the schema root (`schema.yaml`, `additionalProperties: false`) does **not** define `dataset:`, so any YAML using it passes the loader but fails schema validation. Docs still teach it: `_model_class.yaml:8` says the ViT "Requires … the `dataset:` block for data loading" — the real key is `vision_dataset:`. The Spanish description (line 21) repeats the error.
- **Recommendation:** fix the `_model_class.yaml` wording now; decide whether to keep the loader fallback (if yes, document it as deprecated; if the DashAI host no longer emits `dataset:`, remove it).

### A6. Dual GaLore paths: `training.use_galore` + 4 `galore_*` flags vs `optimizer_class: galore_adamw`
- **Files:** `_training.yaml:906-987`; `_optimizer.yaml` (galore_adamw entry)
- **Evidence:** both are live (`trainer.py:887-894` implements manual GaLore; `galore_adamw` is a registered optimizer with its own `rank`/`update_proj_gap` params). Two config vocabularies for the same technique with different parameter names (`galore_update_interval` vs `update_proj_gap`).
- **Recommendation:** pick one canonical path (the optimizer route, which composes with the param-group system) and deprecate the trainer-side flags.

### A7. `alibi_num_heads` — code-only knob invisible to the schema
- **Files:** `src/model/config.py:458,925-926`; `_positional_encoding.yaml:111-118`
- **Evidence:** the schema's `alibi` sub-object is an empty placeholder ("parameter-free; slopes derived from num_heads"), yet the dataclass carries `alibi_num_heads` and `embeddings/factory.py:71` reads it. It can never be set from YAML (strict schema), so it is effectively a derived field wearing a config-field costume.
- **Recommendation:** either expose it (if there is a use case) or demote it to a non-dataclass derived attribute set in `__post_init__` only.

---

## B. Badly placed fields (hierarchy violations)

### B1. Flat feature clusters that contradict the established nesting idiom ⚠ highest-impact structural finding
The refactor established the idiom *"`<feature>.enabled` + nested params"* (`embedding.factorized.enabled`, `embedding.conv.enabled`, `mhc.enabled`, `residuals.type`, `norm.type`). These six clusters stayed flat at `model.*` with `use_*` booleans and scattered siblings (`_model.yaml:92-130`, `_model_flat.yaml`):

| Cluster | Current flat keys | Idiomatic target |
|---|---|---|
| MoE | `use_moe`, `num_experts`, `top_k_experts` | `moe: {enabled, num_experts, top_k}` |
| BitNet | `use_bitnet`, `bitnet_routers`, `use_bitnet_conv` | `bitnet: {enabled, routers, conv}` |
| MoD | `use_mixture_of_depths`, `mixture_of_depths_capacity_ratio`, `mixture_of_depths_router_aux_loss_weight` | `mixture_of_depths: {enabled, capacity_ratio, router_aux_loss_weight}` |
| ODE | `ode_solver`, `ode_steps` (no enable flag — implicit via `layer_pattern`) | `attention.ode: {solver, steps}` (see B2) |
| SSMax | `use_ssmax`, `ssmax_s_init` | `positional_encoding_parameters.ssmax: {enabled, s_init}` |
| NLP head | `classification_head`, `num_labels`, `encoder_pooling_mode` | `classification_head: {enabled, num_labels, pooling_mode}` |

The `_model.yaml:92` comment ("Flat keys … stay at model.* top level") documents the decision, but the codebase has since added nested blocks for *newer* features (mhc, residuals, PE parameters), making the flat remnants the inconsistency rather than the rule.
- **Recommendation:** one breaking pass to complete the migration (flattener already centralizes renaming; the website-mirror param estimator and Streamlit GUI are schema-driven and will follow).

### B2. `ode_solver` / `ode_steps` are mixer knobs living outside `model.attention`
- **Files:** `_model.yaml:97-100`; `_model_flat.yaml:44-82`
- **Evidence:** `ode` is a `layer_pattern` mixer exactly like `msa`, `sparda`, `engram`, `gma` — all of which have `attention.<mixer>.*` sub-objects. ODE is the *only* mixer whose knobs are flat. The `_model.yaml:43-53` attention description even lists `ode` among "mixers without config knobs", which is false (it has two).
- **Recommendation:** move to `attention.ode.{solver, steps}`; add `ode` to the mixer map in `config_flatten.py`.

### B3. Mixer-specific head counts stranded in `dims`
- **Files:** `_dims.yaml:123-158`
- **Evidence:** `num_kv_heads` ("for `gqa_attn`") and `retention_heads` ("for RetNet layers") are per-mixer knobs, while the analogous `gta.num_shared_groups`, `iha.num_pseudo_heads`, `ccgqa.num_kv_heads` correctly live under `attention.<mixer>`. Two spellings for the same concept (`dims.num_kv_heads` vs `attention.ccgqa.num_kv_heads`) confuse users and the GUI.
- **Recommendation:** long-term move to `attention.gqa.num_kv_heads` / `attention.retnet.heads`; short-term add cross-references in the descriptions.

### B4. NLP classification-head triplet vs. vision-head block asymmetry
- **Files:** `_model_flat.yaml:518-545` vs `_image.yaml`
- **Evidence:** the ViT gets a whole sibling top-level block (`image:` with `num_classes`, `pooling_mode`, `cls_token`, seg heads), while the NLP encoder head is three flat keys on `model.*`. `encoder_pooling_mode` even says "Mirrors the ViT `pooling_mode`" — mirroring done by naming, not by structure.
- **Recommendation:** fold into `model.classification_head.*` (B1) or a symmetric top-level `head:` block if more heads are coming.

### B5. `dropout` in `dims`
- **File:** `_dims.yaml:159-185`
- **Evidence:** regularization rate inside the "core dimensions" object. Not a dimension; the only non-structural key there.
- **Recommendation:** acceptable to keep (model-wide, checkpoint-relevant), but if B1 is executed, a top-level `model.dropout` would be more honest. Low priority.

### B6. `residuals.mhc_stream_mode` — cross-cutting knob on one side of the crossing
- **File:** `_residuals.yaml:271-312`
- **Evidence:** governs the AttnRes × mHC interaction, is only meaningful when *both* `residuals.type ∈ {full_attn, block_attn}` *and* `mhc.enabled` — the name says mHC, the location says residuals. The conditional rule enforcing the mHC side lives in `_conditional_rules.yaml:188-204`.
- **Recommendation:** defensible where it is (AttnRes owns the attention math); add a "see also" line in `_mhc.yaml` so mHC readers discover it. Cosmetic.

### B7. Two parallel gradient-checkpoint flags for the residual machinery
- **Files:** `_mhc.yaml:121-134` (`checkpoint`) and `_residuals.yaml:313-335` (`gradient_checkpoint`)
- **Evidence:** both wrap related residual computations in `torch.utils.checkpoint`; a user enabling one and not the other gets half the memory savings. No schema rule ties them together.
- **Recommendation:** document the pairing, or add a shared umbrella if more checkpoint targets appear.

### B8. `image.image_size` vs `vision_dataset.rescale` — two knobs for the same pixels
- **Files:** `_image.yaml:19-39` vs `_vision_dataset.yaml:57-77`
- **Evidence:** both define target (height, width) before patch embedding. If they diverge (`rescale: 224×224`, `image_size: 256×256`), behavior depends on transform order, not on anything the schema expresses. No conditional rule reconciles them.
- **Recommendation:** make `rescale` optional-with-default-null and derive from `image.image_size` when absent (single source of truth in `image:`, dataset only normalizes).

### B9. Three positional-encoding selectors
- **Files:** `model.positional_encoding` + `positional_encoding_parameters` (canonical), `model.attention.titan.positional_encoding` (legacy, A1), `image.pos_embedding_type` (ViT, 12-value enum incl. legacy alias)
- **Evidence:** the ViT block accepts nearly the same enum as the model-level selector; interplay/precedence between `image.pos_embedding_type` and `model.positional_encoding` for `frankenstein_vit` is documented nowhere in the schema.
- **Recommendation:** after removing A1, document (or rule-enforce) the ViT precedence, then converge `pos_embedding_type` onto the model-level enum.

### B10. Inconsistent parallel naming between `training.*` and `training.sbert.*`
- **Files:** `_training.yaml` vs `_sbert.yaml`

  | Concept | MLM key | SBERT key |
  |---|---|---|
  | Sequence length | `max_length` | `max_seq_length` |
  | Epochs | `num_epochs` | `epochs` |
  | Grad clipping | `grad_clip_max_norm` | `max_grad_norm` |
  | Resume | `resume_from_checkpoint` (string `auto`/path) | `resume_from_checkpoint` (**boolean**) |

- **Evidence:** same property name `resume_from_checkpoint` with *different types* in sibling blocks is the worst offender — tooling and users cannot generalize.
- **Recommendation:** align names; for resume, adopt the string form in SBERT (`auto` semantics ≈ `true`).

### B11. Two incompatible `optimizer.parameters` conventions
- **Files:** `_optimizer.yaml:122-179` (prefixed keys `adamw-lr_embeddings`, enforced by `_training.yaml:1016-1338`) vs `_sbert.yaml:370-382` (unprefixed keys `momentum`, `ns_steps`)
- **Evidence:** identical property name (`parameters`), opposite key grammar, different validation strictness (SBERT's stays `additionalProperties: true`, untyped). `train_sbert.py:1273-1282` parses the unprefixed form.
- **Recommendation:** either accept prefixed keys in SBERT too (strip prefix at parse), or rename the SBERT field (`optimizer_kwargs`) to kill the false symmetry. Document whichever convention survives.

### B12. Per-mixer PE toggle matrix incomplete
- **File:** `_positional_encoding.yaml:265-321`
- **Evidence:** `positional_encoding_parameters.use_pe` covers 36 mixers; `layer_pattern` accepts 43. Missing: `ssog_attn` (also absent as `ssog_attn_use_pe` in `config.py` — schema and dataclass agree, so SSOG simply cannot opt into model-wide PE, by design or by omission — nothing records which).
- **Recommendation:** add `ssog_attn: {type: boolean, default: …}` to both schema and dataclass, or a comment stating SSOG is positional-field-only by design.

---

## C. Fields no longer necessary (redundant / dead weight)

### C1. `training.patch_prediction` — empty placeholder block
- **File:** `_vision_patch.yaml` — `properties: {}`, "exists for forward compatibility"
- **Evidence:** the conditional rule for `task: patch_prediction` (`_conditional_rules.yaml:126-147`) does *not* require the block; patch knobs actually live in `image.*` (`mask_ratio`, `mask_token_strategy`, `prediction_target`). An empty object in `training:` that users may dutifully type for nothing.
- **Recommendation:** remove the block + the `_training.yaml:1010-1011` ref, or actually require it for symmetry with `classification`/`segmentation`. Current state is the worst of both.

### C2. `positional_encoding_parameters.alibi` — empty placeholder
- **File:** `_positional_encoding.yaml:111-118` — `properties: {}`, "reserved for future extensions"
- **Evidence:** YAML `alibi: {}` or `alibi:` with any key fails `additionalProperties: false` anyway; the placeholder adds a GUI expander with nothing in it.
- **Recommendation:** drop the key; re-add when a real knob exists (A7).

### C3. Redundant conditional rule (double enforcement of the same invariant)
- **File:** `_conditional_rules.yaml:1-5` (`anyOf: [required base_model] | [required model_class+model]`) **and** lines 118-125 (`if not base_model → required model_class+model`)
- **Evidence:** the second rule is exactly the anyOf's second branch restated; every config is validated twice for the same thing.
- **Recommendation:** keep the `anyOf` (it also permits base-model-only), delete the `if/not` rule — not both.

### C4. `x_parameter_reference` in `_optimizer.yaml` — third copy of the prefix table
- **File:** `_optimizer.yaml:181-291`
- **Evidence:** the prefix list exists (1) here as a custom extension, (2) as 24 `if/then` blocks in `_training.yaml`, (3) implicitly in `src/model/optimizer/factory.py`. AGENTS.md even calls `_optimizer.yaml`'s `by_optimizer` "the prefix list" — but the enforcing artifact is `_training.yaml`.
- **Recommendation:** make `by_optimizer` the single source and *generate* the `_training.yaml` allOf (build script or loader-side validation); see D2.

### C5. Trailing orphan comments in split schema files
- **Files:** `_attention_latent.yaml:257-259` (CCA banner after mtla), `_attention_cca.yaml:224-226` (MSA/SparDA banner after ccgqa), `_attention_engram.yaml:100-103` (latent-family banner)
- **Evidence:** section banners left behind when mixers were moved to their own files; they now misdescribe file contents.
- **Recommendation:** delete or reposition the banners.

### C6. `_examples.yaml` flagship example teaches the legacy path
- **File:** `_examples.yaml:21-30`
- **Evidence:** example #1 sets `attention.titan.positional_encoding: hope` **and** `use_hope: false` (contradictory — `__post_init__` resolves PE to `hope` then forces `use_hope=True`), and never shows the canonical `model.positional_encoding`. It also sets `num_experts/top_k_experts/ode_solver/ode_steps` alongside `use_moe: false` (inert keys presented as normal).
- **Recommendation:** rewrite example #1 around `model.positional_encoding` + `positional_encoding_parameters`, drop the titan legacy keys and the inert MoE/ODE keys.

---

## D. Misalignment with JSON-Schema / YAML best practices

### D1. Defaults: three competing sources, GUI precedence inverted ⚠ highest-risk finding
- **Evidence chain:**
  - `FrankensteinModelConfig` (`src/model/config.py`): `use_bitnet=True`, `use_embedding_conv=True`, `norm_type="dynamic_tanh"`, `ffn_activation="silu"`, `use_moe=True`, `use_hope=True`…
  - Schema `_model_flat.yaml`: `use_bitnet` has **no `default:`**, `examples: [false]`; `use_moe` no default, `examples: [true]`; `_norm.yaml` examples `[layer_norm]`; `_model_flat.yaml` `ffn_activation` examples `[gelu]`.
  - Streamlit `app.py:125-146` `_resolve_default`: precedence `examples[0]` → `default` → `minimum` → fallback. **Examples outrank defaults.**
- **Impact:** a GUI-authored config that leaves BitNet untouched emits `use_bitnet: false`, silently diverging from the dataclass default `True` (AGENTS.md Hard Constraint #3 says BitNet defaults to True — the GUI violates it). Same class of flip for `use_embedding_conv`, `norm_type`, `ffn_activation`.
- **Recommendation:**
  1. Change `_resolve_default` precedence to `default` → `examples[0]` → `minimum`.
  2. Backfill `default:` in the schema for every field whose dataclass default exists (schema becomes self-describing; needed anyway for the website mirror estimator).
  3. Audit `examples:` arrays so their first element never contradicts the intended default.

### D2. 24-fold copy-pasted `if/then` optimizer validation with untyped values
- **File:** `_training.yaml:1016-1338`
- **Evidence:** one `if/then` per optimizer class, each identical except the prefix regex (`^sgd_momentum-`, `^adamw-`, …). `patternProperties: {"^x-": {}}` uses the **empty schema** — any JSON value passes (`adamw-lr_embeddings: "fast"` is schema-valid). Adding optimizer #24 touches `_optimizer.yaml` enum + `by_optimizer` + `_training.yaml` allOf + `_sbert.yaml` enum (see D3) — four places, three of them hand-synced.
- **Recommendation:** generate the allOf list from `by_optimizer` (a tiny codegen step or loader-side check), and give the pattern values a real schema: `{type: [number, boolean]}` at minimum; ideally typed per suffix family (`lr_*`: `number, minimum: 0`).

### D3. Duplicated optimizer-class enum (23 values ×2)
- **Files:** `_optimizer.yaml:97-120` and `_sbert.yaml:345-368`
- **Evidence:** byte-identical enums maintained by hand; adding `q_apollo` had to be done twice. A `$ref` to a shared fragment (the pointer idiom already used for `_attention_latent.yaml#/properties/mla`) fixes it.
- **Recommendation:** `$ref` the SBERT enum (and its prose description) to the canonical one.

### D4. `layer_pattern` enum documented 4×, enforced 1×
- **File:** `_dims.yaml:186-341`
- **Evidence:** the 43-value enum appears once as `items.enum`, then twice more inside the EN/ES description tables (each a full 43-row markdown table), and the tables are the only place marking `sparge_attn`/`fasa_attn` as eval-only (⚠). Any mixer addition edits ~180 lines of schema prose.
- **Recommendation:** keep the tables (they are the product doc), but add a test that parses the description tables and asserts set-equality with `items.enum` (catches drift mechanically). Consider a `_model/_mixers_meta.yaml` map that both the table and tests consume.

### D5. Non-idiomatic nullability: `type: integer` for fields whose runtime default is `None`
- **Files:** `_attention_latent.yaml` (`mla.latent_rank`, `gqla.*`, `mlra.*`, `tucker.*`, `iha.*`, `gta.*`, `mtla.latent_rank/stride`), `_attention_cca.yaml` (`cca.latent_rank`, `ccgqa.query/kv_latent_rank`, `ccgqa.num_kv_heads`), `_attention_gma.yaml` (`routing_dim`)
- **Evidence:** the dataclass treats `null` as meaningful ("defaults to hidden_size // 2") and `_attention_ssog.yaml`/`_attention_falcon.yaml` already do it right (`type: [integer, "null"]`). But writing `mla: {latent_rank: null}` in YAML **fails** the strict schema even though it is the documented runtime semantic.
- **Recommendation:** add `"null"` to the type union of every optional-rank field (mechanical, low risk, matches ssog/falcon precedent).

### D6. `required` vs `default` contradiction
- **Files:** `_conditional_rules.yaml:148-170` (classification requires `image.num_classes`) + `_image.yaml:203-210` (`num_classes` declares `default: 1000`)
- **Evidence:** JSON Schema validators do not apply defaults; the field is simultaneously mandatory and defaulted. Users reading the schema reasonably conclude they may omit it.
- **Recommendation:** either drop `default:` from required fields or drop them from `required` — not both.

### D7. Inconsistent bounds on numerics
- **Examples:** `hidden_size`, `num_layers`, `num_heads`, `ffn_hidden_size`, `ode_steps`, `num_experts`, `top_k_experts`, `vocab_size` have no `minimum`; meanwhile `patch_size`, `seg_*`, optimizer ranks do. `_text_dataset.yaml:90-96` `max_samples` has `minimum: 1` but `_sbert.yaml:229-237` `max_train_samples` doesn't.
- **Recommendation:** sweep numerics for `minimum: 1` / `exclusiveMinimum: 0` parity. Cheap and prevents `num_layers: 0` reaching `__post_init__`.

### D8. Fragment-pointer `$ref` layout instead of `$defs`
- **Files:** `_model.yaml:59-87` (`_attention_latent.yaml#/properties/mla`, …)
- **Evidence:** works with the custom resolver (`schema_loader.py`), but each multi-mixer file is a pseudo-`$defs` container whose top-level shape is an implicit contract; nothing validates that `_attention_latent.yaml` actually contains `properties.mla`. A wrong fragment fails only at resolve time with a KeyError-style error. The resolver also **drops sibling keywords next to `$ref`** (`schema_loader.py:38-48` returns the target, discarding siblings) — currently harmless because no `$ref` carries siblings, but it is a latent draft-2020-12 incompatibility worth a code comment.
- **Recommendation:** acceptable as-is (custom resolver, no JSON-Schema tooling in the loop); if the website mirror ever adopts a real validator, migrate to `$defs:` per file.

### D9. Non-standard keywords are fine, but bilingual coverage drifts
- **Evidence:** `title_es`/`description_es` and `x_parameter_reference` are ignored by validators — legitimate project extensions. However several `description_es` fields switch back to English mid-sentence (copy-paste artifacts): `_optimizer.yaml` description_es line 17 ("…afecta significativamente la dinámica de entrenamiento **and final quality is…**"), `_dims.yaml:101,121`, `_embedding.yaml:50,72,89`, multiple `_training.yaml` fields, `_sbert.yaml:19`. `_mhc.yaml` EN/ES carry different caveats (ES missing the hyperloop/MoD-compat notes).
- **Recommendation:** a lint pass for "English tail in `_es` fields" (regex for common English words in `description_es`).

### D10. Root `examples:` keyword overloading (minor)
- **File:** `schema.yaml:31-32`
- **Evidence:** under draft 2020-12, root `examples` should be an array of instances of this schema — which is coincidentally what `_examples.yaml` is, so it is semantically legal. But its content doubles as project documentation (C6). Standard tooling may render these as validation examples.
- **Recommendation:** informational only.

### D11. `use_pe` map uses terse inline maps without titles/descriptions
- **File:** `_positional_encoding.yaml:280-321`
- **Evidence:** 36 keys of `standard_attn: { type: boolean, default: true }` — the only place in the schema with completely undocumented leaves (no `title`/`description`); the GUI renders them as bare checkboxes with no help text.
- **Recommendation:** add a one-line description per mixer (or at least `title`).

### D12. Optimizer `parameters` blocks: `minProperties` inconsistency
- **Evidence:** `_optimizer.yaml:179` `minProperties: 1` vs `_sbert.yaml:377` `minProperties: 0` on the same-named property. Harmless but symptomatic of B11.
- **Recommendation:** resolve together with B11.

---

## E. Schema ↔ code alignment gaps (summary matrix)

| # | Item | Schema | Code | Verdict |
|---|---|---|---|---|
| E1 | `alibi_num_heads` | absent | `config.py:458` | code-only knob (A7) |
| E2 | `ssog_attn` per-mixer PE | absent from `use_pe` | no `ssog_attn_use_pe` | consistent but coverage gap (B12) |
| E3 | titan PE enum | 11 values (no `bam`) | model-level 12 values | divergence (A1) |
| E4 | `dataset:` top-level key | rejected | accepted alias (`config_loader.py:250`) | misalignment (A5) |
| E5 | `mla/gqla/... latent_rank: null` | rejected (`type: integer`) | `None` is the documented default | schema stricter than runtime (D5) |
| E6 | effective defaults (`use_bitnet` etc.) | examples-only | dataclass defaults differ | GUI flips defaults (D1) |
| E7 | `training.*` runtime-only keys (`num_epochs`, `batch_size`, `max_length`, `mlm_probability`, `max_samples`, `dataset_batch_size`, `num_workers`, `cache_dir`, `local_parquet_dir`, `prefer_local_cache`, `stream_local_parquet`, `join_temp_data_*`, `hf_output_dir`) | declared under `training:` | NOT `TrainingConfig` fields → flow to `training_runtime` dict | by-design two-path split; document it in `_training.yaml`'s block description |
| E8 | vision task blocks (`classification`, `segmentation`) | declared | consumed via runtime dict (`main.py:639-656`, `engine.py:912-930`) | fine |
| E9 | `patch_prediction` block | declared, empty | nothing to consume | dead placeholder (C1) |
| E10 | optimizer param prefixes | `by_optimizer` + 24 allOf + sbert enum | `factory.py` registry | 3–4 hand-synced copies (D2/D3/C4) |

---

## F. Documentation drift inside schema descriptions (legacy flat names still taught)

| # | File:line | Says | Should say |
|---|---|---|---|
| F1 | `_embedding.yaml:15-22,40,79,83` | `use_factorized_embedding=true`, `factorized_embedding_dim`, `use_embedding_conv=true`, `embedding_conv_kernel` | `embedding.factorized.enabled`, `embedding.factorized.dim`, `embedding.conv.enabled`, `embedding.conv.kernel` |
| F2 | `_norm.yaml:49,52` | `prms_partial_ratio` | `norm.partial_ratio` |
| F3 | `_model_flat.yaml:161-162,166-167` | `use_embedding_conv`, `factorized_embedding_dim` | `embedding.conv.enabled`, `embedding.factorized.dim` |
| F4 | `_model_class.yaml:8,21` | "the `dataset:` block" | "`vision_dataset:` block" |
| F5 | `_model.yaml:44-53` | mixer list "without sub-objects" omits `gqa_attn`, `gated_deltanet_attn`, `gated_deltanet2_attn`, `ssog_attn`, `falcon1/2/3/1a/2a/3a_attn` | refresh list (gqa→dims knobs per B3; falcons→shared `falcon` block; ssog→has a block!) |
| F6 | `_dims.yaml:128,148` | describes `num_kv_heads`/`retention_heads` as mixer knobs without noting the flat-vs-nested inconsistency | add cross-refs (B3) |
| F7 | `config_flatten.py:9-52` docstring | mixer inventory omits `falcon` (handled at lines 140-152), `norm.flashnorm_partial_ratio`, PE `bam`/`use_pe` | refresh docstring |
| F8 | `docs/specs/schema-reference.md:39,41,277` | flat names (`prms_partial_ratio`, `use_factorized_embedding`, `norm_type` as top-level) | hierarchical paths |
| F9 | `_mhc.yaml` EN vs ES | EN description documents the Hyperloop mode lifting the MoD incompatibility; ES description lacks it | sync bilingual content |

---

## G. What the schema gets right (keep and protect)

1. **`additionalProperties: false` at every object** — the single best guard against typo'd keys; only the two optimizer `parameters` blocks are open (by necessity).
2. **Modular `$ref` file layout mirroring `src/model/`** — predictable discovery (`_attention_<family>.yaml` ↔ `src/model/attention/<family>/`).
3. **Conditional rules for task→class binding** (`causal_lm`⇒`frankensteindecoder`, vision⇒`frankenstein_vit`) with loader-side runtime double-checks (`config_loader.py:306-326`) — defense in depth.
4. **Eval-only mixers flagged in the enum table** (`sparge_attn`/`fasa_attn` ⚠) matching `TRAINING_FREE_LAYERS`.
5. **SSOG encoder-only constraint enforced twice** (`_conditional_rules.yaml:171-187` + `config.py:902-907`).
6. **Bilingual `title_es`/`description_es` discipline** (modulo F9/D9 drift) — unusual and valuable for the Streamlit GUI.
7. **`type: [integer, "null"]` precedent** in ssog/falcon — the pattern D5 asks to generalize.

---

## H. Prioritized remediation plan

| Priority | Finding | Action | Effort |
|---|---|---|---|
| P0 | D1 | Fix GUI default precedence (`default` before `examples[0]`) + backfill schema `default:` | S |
| P0 | F1-F5 | Rewrite stale descriptions referencing flat/legacy key names | S |
| P1 | A1-A3 | Remove titan legacy PE subtree (enum + `use_hope` + hope/rope dupes) after one deprecation release | M |
| P1 | D2/D3/C4 | Single-source the optimizer prefix table; generate the 24 allOf blocks; `$ref` the SBERT enum | M |
| P1 | C1/C2/C3 | Delete `training.patch_prediction`, `…alibi` placeholders, redundant conditional rule | S |
| P2 | D5 | Add `"null"` to optional-rank type unions | S |
| P2 | B1/B2 | Complete nesting migration (moe/bitnet/mod/ssmax/ode/classification_head → sub-objects) — breaking; coordinate with `configs/`, tests, website mirror | L |
| P2 | A4/B9 | Deprecate `learned_1d`; document ViT vs model-level PE precedence | S |
| P2 | A5 | Fix `_model_class.yaml` `dataset:` → `vision_dataset:`; decide loader alias fate | S |
| P3 | B3/B4/B5/B6/B7/B8/B10/B11/B12 | Structural consistency items (dims knobs, head blocks, sbert naming, PE matrix) | M |
| P3 | D4/D6/D7/D9/D11 | Enum-table parity test, required/default sweep, bounds sweep, `_es` lint, use_pe docs | S-M |
| P3 | A6/A7 | GaLore path consolidation; `alibi_num_heads` demotion | S |
| P3 | C5/C6/F6-F8 | Comment/banner cleanup, example #1 modernization, docstring refresh | S |

**Test additions to lock in any remediation:**
- Enum-parity test: `layer_pattern` description tables ⇔ `items.enum`; titan PE enum ⊂ model PE enum; sbert optimizer enum == optimizer enum.
- Default-parity test: for each schema leaf with `default:`, assert `FrankensteinModelConfig`/`TrainingConfig` default equality (catches D1 regressions).
- Null-tolerance test: YAML round-trip of every optional-rank field with explicit `null`.

---

## Appendix: inventory of reviewed files

**Root:** `src/schema.yaml` (32 lines; required: training; allOf → conditional rules; examples ref)

**`src/schema/` (15):** `_base_model.yaml` (string type), `_tokenizer.yaml` (source/training/name_or_path/use_fast/trust_remote_code), `_model_class.yaml` (3-value enum), `_model.yaml` (aggregator), `_training.yaml` (1,338 lines incl. 24 optimizer allOf), `_optimizer.yaml` (23 classes + prefix table), `_sbert.yaml` (dataset/columns/optimizer/wandb), `_conditional_rules.yaml` (14 rules), `_examples.yaml`, `_image.yaml` (ViT), `_vision_dataset.yaml`, `_text_dataset.yaml`, `_vision_patch.yaml` (empty), `_vision_classification.yaml`, `_vision_segmentation.yaml`

**`src/schema/_model/` (15):** `_dims.yaml` (10 fields + 43-value layer_pattern), `_norm.yaml` (type/partial_ratio/flashnorm_partial_ratio), `_embedding.yaml` (factorized/conv), `_positional_encoding.yaml` (enum + parameters + use_pe 36), `_attention_titan.yaml` (legacy PE), `_attention_latent.yaml` (mla/gqla/mlra/tucker/iha/gta/mtla), `_attention_cca.yaml` (cca/ccgqa), `_attention_sparse.yaml` (msa/sparda), `_attention_engram.yaml`, `_attention_gma.yaml`, `_attention_ssog.yaml`, `_attention_falcon.yaml`, `_mhc.yaml` (9 fields incl. hyperloop×4), `_residuals.yaml` (type/full_attn/block_attn/mhc_stream_mode/gradient_checkpoint), `_model_flat.yaml` (17 flat keys)

**Code cross-referenced:** `src/model/config.py` (FrankensteinModelConfig, ~200 fields), `src/utils/config_flatten.py`, `src/utils/schema_loader.py`, `src/training/config_loader.py`, `src/training/trainer.py` (TrainingConfig), `src/training/main.py`, `src/engine.py`, `src/sbert/train_sbert.py`, `src/streamlit_gui/app.py`, `src/model/attention/retnet.py`, `src/model/embeddings/factory.py`, `docs/specs/schema-reference.md`.
