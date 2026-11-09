# Plan: drop `_attn` suffix from all mixers, delete `retnet_attn` alias

**Date:** 2026-11-09
**Type:** Breaking change (config surface + module layout). Version bump to `1.2.0`.
**Status:** Implemented (2026-11-09)

## Decisions (confirmed by user)

| # | Decision |
|---|---|
| D1 | **Hard break** — no legacy-name shim; old `*_attn` configs fail validation cleanly. |
| D2 | **Rename Python modules too** — 30 `*_attn.py` files + sphinx rst, so file names match mixer names. |
| D3 | **Paper PDF rebuilds + website mirror sync in scope** (latexmk available; mirror is separate git repo). |
| D4 | **History untouched** — `experimentos/`, `.opencode/plan/` (other plans), `dist/`, caches keep old names. |

## Rename map

Exact-token regex per mixer: `\b<name>_attn\b` → `<name>`. **Never** do a blanket `_attn` strip.

45 mixers drop the suffix:

```
standard_attn→standard          sigmoid_attn→sigmoid
gated_softmax_attn→gated_softmax titan_attn→titan
gla_attn→gla                    deltanet_attn→deltanet
gated_deltanet_attn→gated_deltanet
gated_deltanet2_attn→gated_deltanet2
hgrn2_attn→hgrn2                fox_attn→fox
nsa_attn→nsa                    engram_attn→engram
gqa_attn→gqa                    longformer_attn→longformer
swa_attn→swa                    bigbird_attn→bigbird
sparse_transformer_attn→sparse_transformer
sparsek_attn→sparsek            sparge_attn→sparge
fasa_attn→fasa                  msa_attn→msa
sparda_attn→sparda              kda_attn→kda
falcon1_attn→falcon1  falcon2_attn→falcon2  falcon3_attn→falcon3
falcon1a_attn→falcon1a falcon2a_attn→falcon2a falcon3a_attn→falcon3a
mla_attn→mla                    gqla_attn→gqla
mlra_attn→mlra                  tucker_attn→tucker
iha_attn→iha                    gta_attn→gta
mtla_attn→mtla                  cca_attn→cca
ccgqa_attn→ccgqa                gma_attn→gma
ssog_attn→ssog                  conformer_attn→conformer
fastconformer_attn→fastconformer
memory_attn→memory              lsa_attn→lsa
```

- `<mixer>_attn_use_pe` config keys/fields → `<mixer>_use_pe` (schema `_positional_encoding.yaml` + `model/config.py`).
- **`retnet_attn` DELETED** — it is a pure alias: `src/model/attention/gated/retnet_attn.py` (`RetNetAttention`) wraps `MultiScaleRetention` (`retnet`) and delegates 100%. Remove from: enum, bilingual tables, `hybrid_layer.py` registry, `cache/capabilities.py` (`IMPLEMENTED_MIXERS` + `CACHE_SPECS["retnet_attn"]`), `retnet_attn_use_pe` field, all docs. Users use `retnet`.
- `retnet`, `mamba`, `ode`: unchanged. Enum goes 49 → 48 values.
- Side benefit: mixer names now equal the already-suffix-free `attention.<name>` sub-block keys.

### Do-not-touch traps (contain `_attn` but are NOT mixers)

- Preset names: `full_attn_res`, `block_attn_res`, `mhc_full_attn_res` (attention-residual schemes).
- Attribute `self.mhc_attn` in `hybrid_layer.py` (mHC).
- Combo ids like `attn_x__ssog_attn` → only the trailing mixer part changes (`attn_x__ssog`).
- `.venv`, `.pytest_cache`, `__pycache__`, `dist/`, `experimentos/`, `.opencode/plan/`.

## Work packages (execute in order)

### 1. Schema — `src/schema/_model/`
- `_dims.yaml`: enum list (49→48), EN + ES mixer tables (drop the `retnet` / `retnet_attn` double row → single `retnet` row), all description/examples text mentioning mixer names.
- `_norm.yaml`: supported-mixers lists (qk-norm etc., EN+ES).
- `_positional_encoding.yaml`: per-mixer PE boolean keys `standard_attn:` → `standard:` etc.; delete `retnet_attn` entry.
- All `_attention_*.yaml` (`_swa`, `_conformer`, `_falcon`, `_ssog`, `_lsa`, `_retnet`, `_latent`, `_sparse`, `_engram`, `_gma`, `_cca`, `_fastconformer`, `_ode`): descriptions EN/ES.
- `_model.yaml`, `_conditional_rules.yaml`: any mixer-name references (e.g. fasa/sparse rules).

### 2. Code — `src/`
- `git mv` module files (drop `_attn`), update family `__init__.py` exports and all imports:
  - `gated/`: `deltanet_attn, falcon_attn, fox_attn, gated_deltanet2_attn, gated_deltanet_attn, gated_softmax_attn, gla_attn, hgrn2_attn, kda_attn` (keep `falcon_common.py`)
  - `latent/`: `cca_attn, gma_attn, gqla_attn, gta_attn, iha_attn, mla_attn, mlra_attn, mtla_attn, tucker_attn`
  - `sparse/`: `bigbird_attn, fasa_attn, fastconformer_attn, longformer_attn, lsa_attn, msa_attn, nsa_attn, sparda_attn, sparge_attn, sparsek_attn, sparse_transformer_attn, swa_attn`
- **Delete** `src/model/attention/gated/retnet_attn.py` + its export in `gated/__init__.py`.
- `hybrid_layer.py`: registry keys, `TRAINING_FREE_LAYERS` → `{"fasa", "sparge"}`, import of `RetNetAttention` removed; docstrings.
- `model/config.py`: all `<mixer>_use_pe` fields (drop `retnet_attn_use_pe`), default `layer_pattern` → `["retnet", "ode", "mamba", "titan"] * 3`, docstrings.
- `model/cache/capabilities.py`: `IMPLEMENTED_MIXERS` (frozenset), `CACHE_SPECS` — remove `"retnet_attn"` spec; comments.
- `utils/config_flatten.py`: mixer-name references; `_ATTENTION_MIXER_RENAMES` keys already clean (mla, gqla, …) — verify only.
- `training/main.py`, `training/config_loader.py`, `deploy/deploy.py`, `deploy/transformers_export.py`, `deploy/bitnet_gguf_export.py`, `deploy/README.md`.
- `src/streamlit_gui/` has 0 occurrences — verify only.

### 3. Configs
- All `configs/*.yaml` + `configs/examples/*.yaml` (156 files): `layer_pattern` values, `_use_pe` keys, descriptions.
- `git mv configs/memory_attn.yaml configs/memory.yaml`.
- `configs/README.md`, `configs/examples/examples.md`.

### 4. Tests (217 files)
- Update all mixer strings across `tests/`.
- Gates that must stay green: `test_schema_consistency.py`, `test_yaml_examples.py`, `test_schema_attention_layers.py`, `test_optimizer_factory.py`, `test_common_modules.py`, `test_kv_cache_*.py` (incl. `test_kv_cache_mixers.py`, auto-parametrized over `IMPLEMENTED_MIXERS`), `test_model_variants.py`.
- `test_naming_conventions.py`: **add rule** — no enum mixer ends with `_attn`; `retnet_attn` must not exist anywhere in `src/`.
- e2e: `tests/test_run_e2e_configs.py`, `test_e2e_helpers.py`, `test_kv_cache_quant_e2e.py`, `test_deploy_validation.py`, `test_cli_parser.py`.

### 5. Docs (repo)
- `docs/architecture.md` (83 occ. — canonical spec, incl. extension recipes), root `README.md` (16), `AGENTS.md` if it cites examples.
- Sphinx: rename `docs/source/api/model/attention/{gated,latent,sparse}/*_attn.rst` → suffix-free, fix `automodule` paths + toctree `index.rst` entries; regenerate with `bash docs/build_docs.sh`.

### 6. Papers — `docs/paper/` + `docs/paper-es/`
- All `.tex` (sections + appendices; 457 occ. total, EN+ES): mixer names in prose, tables, pseudocode listings.
- `figures/mixer-taxonomy.mmd`: node labels; regenerate SVGs that contain `_attn` (`mixer-taxonomy.svg`, `latent-map.svg`, `experiments/round2/trans_loss.svg`) via `npx -y @mermaid-js/mermaid-cli` (check for `.mmd` sources; if an SVG has no mmd source, hand-edit labels).
- Rebuild PDFs: `latexmk -pdf` in `docs/paper/` and `docs/paper-es/` (refresh `.bbl` if citations changed — they should not).
- Add **one line** in experiments section: mixer names pre-1.2 carried an `_attn` suffix (mapping note); do NOT rewrite `experimentos/` data.

### 7. Website mirror — `erickfmm.github.io/frankenstein-transformer/`
- Run `.opencode/command/sync-website.md` flow (driven by `.opencode/command/sync-website-state.sh`): sync schema, presets, examples, paper (tex/pdf/figures), pandoc HTML rebuild.
- Commit in the mirror repo (`git -C erickfmm.github.io ...`), separate from main-repo commit.

### 8. dashai-frankenstein plugin
- 3 test files (4 occ.): `tests/test_e2e_engine_telemetry.py`, `tests/test_dashai_dataset_toggle.py`, `tests/test_mlm_pretraining.py`.
- Check `pyproject.toml` version pin against bumped `frankenstein-transformer`.
- (`../dashAI` upstream repo: 0 occurrences outside `.venv` — nothing to do.)

### 9. Version
- `pyproject.toml`: bump to `1.2.0` (breaking config change).

## Validation gates (must all pass)

1. `uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings` — fully green.
2. Audit grep returns ONLY allowed leftovers:
   ```bash
   grep -rnE '\b[a-z0-9]+_attn\b' src tests configs docs/paper docs/paper-es \
     docs/source docs/architecture.md README.md configs/README.md dashai-frankenstein \
     | grep -vE 'full_attn_res|block_attn_res|mhc_full_attn_res|mhc_attn'
   # expected: empty
   ```
3. Preset/example schema strict validation (covered by `test_yaml_examples` auto-glob).
4. `bash docs/build_docs.sh` succeeds; PDFs rebuilt.
5. `bash .opencode/command/sync-website-state.sh` reports zero drift after mirror sync.

## Explicit non-goals

- No legacy-name compatibility shim (D1).
- No changes in `experimentos/`, `.opencode/plan/` (history), `dist/`, `.venv`, caches.
- No renaming of Python *class* names (`SWAAttention`, `LongformerAttention`, …) — only mixer strings and module file names. Exception: `RetNetAttention` class is deleted together with its alias module.

## Execution log

- **Schema/code**: 47-value enum; `retnet_attn` alias deleted (`gated/retnet_attn.py`, `RetNetAttention`, its cache spec, PE flag and config field). 30 mixer modules renamed via `git mv`; all imports/exports updated; `<mixer>_use_pe` fields (47) kept in sync with schema and `getattr` lookups (`falcon` uses `f"{variant}_use_pe"`).
- **Configs/tests**: 156 YAMLs + 53 test files updated; `configs/memory_attn.yaml` → `configs/memory.yaml`; new guards in `tests/test_naming_conventions.py` (no `_attn` enum value, no `retnet_attn` in `src/`, no `*_attn.py` module) and hardcoded enum counts updated (48 → 47) in `test_schema_attention_layers.py` and `test_kv_cache_fallback.py`.
- **Docs**: `docs/architecture.md` (counts 47, duplicate `retnet` row removed), `README.md` (alias language removed), sphinx `docs/source/**` rst renamed and rebuilt, `.opencode/command/add-mixer.md` recipe now says "no `_attn` suffix".
- **Papers**: EN/ES `.tex` (names + counts forty-seven; `retnet\_attn` alias noted once as removed history), citation keys protected from underscore escaping, `mixer-taxonomy.mmd` (+ regenerated `.svg`/`.pdf` via mermaid-cli), PDFs rebuilt clean with latexmk.
- **Website mirror**: schema/examples/paper/figures synced, paper HTML rebuilt (pandoc + flatten + postprocess), site-only `ft-diagram.js` / `ft-param-estimator.js` / `index.html` de-aliased, `examples/memory_attn.yaml` removed in favour of `memory.yaml`; `sync-website-state.sh` reports `diff=0 missing_site=0 only_site=0`, 0 probe failures.
- **Untouched per D4**: `experimentos/`, historical `.opencode/plan/*`, `checkpoints/` (untracked), `dist/`, caches.
