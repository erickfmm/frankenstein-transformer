# Plan: post-registry consistency cleanup

> Scope: fix everything left inconsistent or redundant by the registry rollout
> (`d1f6ad0`) and the flattener removal (`70082e4`): stale command files, dead
> schema shadow blocks, a live `use_pe` key-shape bug, the duplicated
> `CacheSpec` declaration, small e2e/test bugs, and the GitHub Pages mirror
> drift. All decisions below were approved by the user in the planning session
> (2026-11-11).

## 1. Context

No mixer or optimizer has been added since the registry rollout. The two
addition commands (`.opencode/command/add-mixer.md`, `add-optimizer.md`) still
describe the pre-registry recipe (hand-edited enums, `mixer_registry`,
`OPTIMIZER_REGISTRY`, flattener renames), and `sync-website.md` predates the
mirror's `registry/` copy and today's tool availability. Beyond the commands,
the codebase carries four genuine leftovers of the migration:

1. **Raw schema shadow blocks** — hand-maintained copies that
   `src/registry/schema_assembly.py` overwrites at assembly time. Verified in
   sync today (dims enum 49 == registry, opt enum 36 == registry,
   `by_optimizer` 36 == registry), but they are a second list that can drift.
2. **`use_pe` key-shape bug (live)** — the raw
   `positional_encoding_parameters.use_pe.properties` block declares 49
   **unsuffixed** keys (`standard:`, `retnet:` …), but the runtime
   (`use_pe_for`, `src/model/attention/params.py:60-66`) only reads
   **suffixed** keys (`standard_use_pe`), which the assembler injects
   (`schema_assembly.py:106-131`). The assembled schema therefore carries 98
   keys of which 49 are schema-valid but silently ignored.
   `configs/examples/pe_mixed_use_pe.yaml` already uses the dead shape (it
   works only because its values coincide with the spec defaults), and
   `docs/architecture.md:168` + `configs/README.md:359,426,443,543` document
   the dead shape.
3. **Duplicated `CacheSpec`** — the registry `MixerSpec.cache`
   (`src/registry/spec.py:54`, whose docstring already says "the registry is
   now the source of truth and the cache package re-exports derived views")
   is byte-identical (fields and notes) to the hand dict
   `CACHE_SPECS` in `src/model/cache/capabilities.py:111-472`. Verified: zero
   field/note drift across all 49 mixers, keys equal.
4. **Small bugs** — `"retnet"` duplicated in `full_tests/run_e2e.py:238`
   `ATTENTIONS` (makes `itertools.combinations` generate `attn_retnet__retnet`
   plus every retnet pair twice ≈ 50 extra training combos) and in
   `tests/test_schema_attention_layers.py:19-20`; broken default estimator
   path in `full_tests/param_estimate_check.mjs:24` (`../../erickfmm.github.io`
   resolves outside the repo); stale census "80 files"
   (`docs/architecture.md:98`, actual: 84).

Mirror (`erickfmm.github.io`, a separate git repo, `.gitignore:47`): the
state report (`bash .opencode/command/sync-website-state.sh`) shows
`diff=10 missing_site=16 only_site=0 ok=458`, all probes PASS, estimator at
1.0% tolerance (168 configs). The 10 DIFFs are the tutorial/MoE paper rewrites
(`063752b`, `44a479c`) not yet synced; the 16 MISSING are the new
`docs/paper/figures/tutorial/*.svg`. Neither the file map nor the state
script pairs `src/registry/** → registry/**` (the mirror already has a manual
copy: `binding.py` DIFF, `snapshot.py` missing — and
`src/utils/schema_loader.py:36-37` imports `registry.schema_assembly` in the
mirror layout, so the copy is functional, not accidental). The mirror's
`paper/figures/` also carries stale `.mmd`/`.pdf`/`build-figures.sh`/etc.
leftovers that the state script never pairs (only `*.svg` is tracked).

Consumers audit (why deleting the raw shadow blocks is safe): every schema
consumer resolves through `src/utils/schema_loader.resolve_schema`, which
**always** runs `assemble_schema` — `config_loader.py:284`,
`streamlit_gui/app.py:142`, `deploy/transformers_export.py:717`,
`registry/snapshot.py:40`. No raw-YAML validation path exists except the two
consistency tests being rewritten in this plan.

## 2. Workstream A — the three command files

### A1 `.opencode/command/add-mixer.md`

1. **Frontmatter description** — keep; it already says "registry-first".
2. **Header quote** — keep; it matches reality. Fix only the phrase
   "The schema enums, `by_optimizer` reference, `if/then` prefix rules,
   attention property fragments …" (still true) — no change needed except the
   trailing claim about `model.attention` being generated: correct as-is.
3. **Canonical recipe** — replace "§11.1 ('Add an attention mixer')" with
   "`docs/architecture.md` §11.1 'Add or change a component (any family)'
   (:1058), §11.3 'Capability declarations' (:1086) and the **Mixer** row of
   `AGENTS.md`".
4. **Phase 1** — knobs: "become the `model.attention.<subkey>` block (schema +
   `_ATTENTION_MIXER_RENAMES` + dataclass fields)" → "become the spec's
   `params_schema` fragment (`schema/_model/_attention_<name>.yaml#/properties/<name>`)
   + `defaults`, read through `mixer_params`". "`per_mixer_use_pe` default" →
   "the spec's `needs_pos_encoder` flag (consumed by `use_pe_for` and by the
   assembler's `use_pe.<name>_use_pe` injection)".
5. **Phase 1 duplicate gate** — "already in the `mixer_registry` of
   `src/model/hybrid_layer.py`" → "already in the registry
   (`src/registry/builtin/attention.py` / `registry.names(ComponentFamily.MIXER)`)".
6. **Phase 2 anchors** — replace:
   - `src/model/hybrid_layer.py`: `mixer_registry` gone. New anchors:
     `TRAINING_FREE_LAYERS` derived from
     `registry.property_set(ComponentFamily.MIXER, "training_free")` (:80),
     dense surrogate `_training_free_dense_forward` (:306), `use_fixed_softmax`
     attach (:176, via `config.numerics.fixed_softmax`), registry dispatch in
     `__init__`.
   - Add: `src/registry/spec.py` (`MixerSpec`/`CacheSpec` fields, :54, :174),
     `src/registry/builtin/attention.py` (`ATTENTION_SPECS`, spec pattern),
     `src/model/attention/params.py` (`mixer_params`, `use_pe_for`),
     `src/registry/schema_assembly.py` (`_inject_mixers` :68-131).
   - `src/schema/_model/_dims.yaml`: the enum is registry-injected; only the
     two bilingual tables are hand-maintained (4-column rows; the assembler's
     `_append_missing_rows` fallback emits 2-column rows, so hand-write the
     row).
   - Drop `_model.yaml` and `_positional_encoding.yaml` from the anchor list
     (both are fully generated surfaces now).
   - `src/model/cache/capabilities.py`: `CACHE_SPECS` is **derived from the
     registry** after Workstream C; the plan's cache rows must say "declare
     `cache=`/`cache_implemented` on the `MixerSpec`" instead of editing the
     dict. `IMPLEMENTED_MIXERS` stays a hand frozenset.
   - Tests: `test_schema_attention_layers.py` (hand expected set;
     `test_cache_specs_match_layer_pattern_enum` locks the census — bump the
     count when adding a mixer), plus the registry gates
     (`test_registry_core.py`, `test_registry_binding.py`,
     `test_schema_assembly.py`).
   - e2e: `ATTENTIONS` (:237-272, one comment block per recent mixer),
     `OPTIMIZERS` (:289), `make_attention_pair_configs` (:790),
     `make_single_attention_configs` (:805), flags `--category/--only/--limit/
     --list-combos (:2032)/--realonall (:2002, default ON)/--no-realonall/
     --max-samples/--max-epochs/--batch-size/--timeout/--skip-attn-pairs/
     --skip-singles`.
   - Paper annexes: numbered family annexes `annex-2-recurrent-attention.tex`
     … `annex-8-conformer.tex` (+ ES `annex-2`…`annex-8`); dense peers live in
     `annex-2` today (see `ca`, added by `6dc1170`) — keep the "grep the peer"
     rule and name the real files. KV-cache annex: `annex-15-kv-cache.tex` (EN)
     / `annex-23-cache-kv.tex` (ES). Census lines: `sections/01-introduction.tex:14`
     ("forty-nine sequence mixers … thirty-six optimizer families"),
     `sections/04-architecture-taxonomy.tex:7,23`, sections 10/11 EN+ES,
     `annex-summary-tables.tex` intro, `docs/architecture.md` §1 (:25) and
     §7.4 (:390), `README.md:164-180`.
   - Sphinx: `docs/source/api/model/attention/` — top-level `.rst` + `gated/`,
     `latent/`, `sparse/` subdirs (not every family has a dir).
   - Figures: family maps are `dense-map.mmd`, `latent-map.mmd`,
     `memory-map.mmd`, `recurrent-map.mmd`, `sparse-map.mmd` (+ taxonomy);
     gated/fast-weight/ssog/conformer have no map.
   - Inventory: "77 diagrams" claims → 99 `.mmd` / 142 `.svg`.
7. **Phase 3 decisions** — KV-cache decision now maps to the **spec** flags:
   decision 1 → `cache=CacheSpec(supported=True, slot_type=…, notes=…)` +
   `cache_implemented=True` on the `MixerSpec` **and** the incremental branch
   **and** `IMPLEMENTED_MIXERS` membership (capabilities re-exports, no dict
   edit); decision 2 → `cache=CacheSpec(supported=True, notes=…D5 fallback…)`,
   `cache_implemented=False`; decision 3 → `cache=CacheSpec(supported=False,
   notes=…)`. `per_mixer_use_pe` decision → `needs_pos_encoder` flag.
8. **File-by-file table** — rewrite rows:
   - Row 3 (`hybrid_layer.py`) → **removed** (registry dispatch automatic;
     `TRAINING_FREE_LAYERS` derives from `training_free=True`).
   - Row 4 → `src/registry/builtin/attention.py`: full `MixerSpec` (name,
     `builder="src.model.attention.<family>.<name>:<Class>"`, EN/ES
     descriptions, `references`, `params_schema`, `defaults`, `cache=`,
     `cache_implemented`, `qk_norm_supported`, `qk_clip_supported`,
     `training_free`, `needs_pos_encoder`).
   - Row 6 (`_dims.yaml`) → bilingual table rows only (EN + ES, 4 columns,
     with `(paper: arxiv:<id>)`; the raw enum is no longer hand-maintained).
   - Row 8 (`_model.yaml` `$ref` + description lists) → **removed**
     (assembler rebuilds `model.attention.properties` from spec fragments).
   - Row 9 (`_positional_encoding.yaml` `per_mixer_use_pe`) → **removed**
     (assembler injects `use_pe.<name>_use_pe` from `needs_pos_encoder`).
   - Row 10 (`capabilities.py`) → **removed as a dict edit**; replaced by the
     spec `cache=`/`cache_implemented` flags (Workstream C makes
     `CACHE_SPECS` derived). `IMPLEMENTED_MIXERS` membership remains a real
     edit under KV-cache decision 1.
   - Row 14 (`test_schema_attention_layers.py`) → add the name to the expected
     set **and** bump `test_cache_specs_match_layer_pattern_enum`
     (`len(enum_values)` 49 → 50) — note: after Workstream B the count can be
     derived from the registry; keep the explicit number only if the census
     test keeps it.
   - Row 17 (`test_kv_cache_mixers.py`) → **no hand edit**: the suite
     parametrizes itself from `IMPLEMENTED_MIXERS` (:23-27).
   - New row: `docs/schema.assembled.yaml` regeneration
     (`uv run --extra cpu --extra train python -m src.registry.snapshot
     --out docs/schema.assembled.yaml`).
   - New row: `annex-15-kv-cache.tex` (+ ES `annex-23`) incremental-decode
     list when the mixer is cacheable (FPA precedent, `2990b69`).
   - New row: sections 01/10/11 census (EN + ES) when the global mixer count
     changes (FPA precedent).
9. **Execution checklist** — targeted gates += `tests/test_registry_core.py
   tests/test_registry_binding.py tests/test_schema_assembly.py`; §6 website
   sync: figures flow to the mirror as **`*.svg` only**; §7 commit +=
   `docs/schema.assembled.yaml`; hard-rules note += D13/D14/D15.
10. **Website-sync section (§6)** — "figures incl. the new `.mmd`/`.pdf`/
    `.svg`" → "the new `.svg`"; site-only files list stays
    (`index.html`, `ft-diagram.js`, `ft-param-estimator.js`, paper build
    pipeline); add `CATS`/`SUBSTRUCT`/`STYLES` + `mixerParams` rows (keep).

### A2 `.opencode/command/add-optimizer.md`

1. **Canonical recipe** → `docs/architecture.md` §11.1 (any family) + the
   **Optimizer** row of `AGENTS.md`.
2. **Phase 1 duplicate gate** — "already in `factory.OPTIMIZER_REGISTRY`" →
   "already in the registry (`src/registry/builtin/optimizer.py` /
   `registry.names(ComponentFamily.OPTIMIZER)`)".
3. **Phase 2 anchors**:
   - `factory.py`: `OPTIMIZER_REGISTRY`/`OPTIMIZER_ALLOWED_KEYS` are derived
     read-only views (:129-132); `build_optimizer` has no per-optimizer
     branches (:186+).
   - `base.py` anchors: `GROUP_NAMES` (:19), `to_float` (:174), `to_int`
     (:194), `to_betas` (:213), `extract_prefixed_parameters` (:232),
     `ensure_no_unknown_parameters` (:257),
     `categorize_param_groups` (§8.1 note).
   - Schema: after Workstream B there is **no** `_optimizer.yaml`/`
     `_training.yaml`/`_sbert.yaml` edit at all; the only optional edit is the
     bilingual use-case table row (3-column style, for the raw-YAML docs).
   - Tests: `test_optimizer_factory.py` (`test_registry_contains_all_required`
     :31 hand set; test-class pattern), registry gates.
   - Doc targets: `annex-9-optimizers_<bucket>.tex` (adaptive/evolutionary/
     geometry/memory/momentum + `tutorialintro` + `summary`; ES:
     `annex-9-optimizadores_<bucket>.tex`), figures under
     `docs/paper/figures/optimizers/`, `tab:all-optimizers`
     (`annex-summary-tables.tex:130`, intro "all 36" :113),
     `sections/05-optimizer-families.tex:5` ("thirty-six optimizer families in
     eight categories"), census in `sections/01:14`.
   - e2e: `OPTIMIZERS` (:289) + `_with_optimizer` (:661) — specific globals go
     there; keys are auto-filtered by `OPTIMIZER_ALLOWED_KEYS` (:762-764).
   - Census: read the current number from the repo (**36** today; do not
     hardcode).
4. **File-by-file table** — remove rows 3 (`factory.py`), 4 (`_optimizer.yaml`
   a/c/d), 5 (`_training.yaml`), 6 (`_sbert.yaml`); replace with one row:
   `src/registry/builtin/optimizer.py` — `OptimizerSpec(name, builder,
   description, description_es, references, param_bindings={suffix:
   ParamBinding(coercion, default)} with coercions from
   `src/registry/coercions.py`, group_policy, custom_build only for special
   constructors (pattern `_build_adafactor`), capability flags). Keep row 4(b)
   as an *optional* row: hand-write the EN + ES use-case table row (3-column
   style) so the assembler's 2-column fallback does not fire. Keep rows
   example-YAML / `configs/README.md` / e2e / tests / bibliography / annexes /
   summary / section 05 / census (paths updated to `annex-9-*`). Add the
   `docs/schema.assembled.yaml` regeneration row. Targeted gates += registry
   gates.

### A3 `.opencode/command/sync-website.md`

1. **File map**:
   - Figures row → **`*.svg` only** (state script :129; "sources/PDFs stay in
     the main repo").
   - New row: `src/registry/**` → `registry/**` (all `*.py`, `__pycache__`
     excluded; the mirror's `utils/schema_loader.py` imports
     `registry.schema_assembly` in standalone layout, so this copy is
     functional).
   - `docs/bibliography/**` note → "all files are paired (script uses
     `-type f`); legacy review `*.md` are pre-authorized deletions".
   - Site-only infrastructure list += `experiments/` (experiment atlas:
     `atlas.css`, `index.html`) and the mirror's `schema.yaml` regeneration
     note stays.
2. **Workflow §1** — add the decision rule: `ONLY_SITE <file>` under
   `paper/figures/` that is **not** `*.svg` → pre-authorized deletion
   (`git -C erickfmm.github.io rm`), leftover of pre-registry syncs
   (`.mmd`, `.pdf`, `build-figures.sh`, `puppeteer-config.json`,
   `mermaid-paper.json`, `README.md`, `web-theme.css`).
3. **Workflow §3.1** — fix "(current machine: `pdflatex=no`)": TeX is now
   available; compiling in the mirror is the primary path, copying prebuilt
   PDFs remains the fallback.
4. **Workflow §3.4** — "all 77 diagrams" → drop the hardcoded count (99
   `.mmd` / 142 `.svg` today; say "all diagrams (~100)").
5. **Workflow §3 apply step** — note that `cp` per file is deliberate
   (reviewability); drop the stale "(rsync is not installed on this machine)"
   justification (rsync exists; keep `cp` anyway).
6. **Workflow §4/§5** — complete the probe catalogue from the state script:
   `cmd-select`, `paper-figures-en/es`, `estimator-drift/truth/run`, `uv`, and
   the `[python-probes]` (`schema-refs`, `example-yaml`, `cli-flags`) /
   `[estimator]` sections; `github-url` checks the "frankestein" typo.
7. **§5.2** — after Workstream D3 the estimator default path is fixed; change
   "the .mjs default is broken" to "always pass explicit paths (the state
   script does)".
8. **§5.4** — CLI parse block: verified current against `src/cli.py`
   (subcommands train/deploy/infer/web-server; deploy `--format` choices
   `quantized|standard|transformers|gguf`, `--yaml`, `--safetensors-format`,
   `--quant-runtime`; infer `--task mlm|sbert|causal_lm` + kv flags). No edit
   beyond a "re-verify at run time" note.

## 3. Workstream B — remove schema shadow blocks + fix `use_pe`

**B1 `src/schema/_model/_dims.yaml`** — delete the hand-written
`layer_pattern.items.enum` list (:334-389, the 49 `- name` entries under
`items.enum`). Keep `type: string`, both descriptions and `examples`. The
assembler sets the enum (`schema_assembly.py:84`).

**B2 `src/schema/_optimizer.yaml`** — delete `optimizer_class.enum`
(:159-195) and the hand `x_parameter_reference.by_optimizer` block (:332-476).
Keep `x_parameter_reference.shared_per_group_suffixes` (:291-331, hand docs
of the shared surface) and the bilingual use-case tables.

**B3 `src/schema/_training.yaml`** — delete the hand `allOf:` if/then list
(:1240+).

**B4 `src/registry/schema_assembly.py::_inject_optimizers`** (:219-224) —
replace the `if "allOf" in training / else root` branch with the unconditional
`training["allOf"] = rules` (the root fallback is wrong-shaped for the
generated rules and becomes unreachable once the raw block is gone).

**B5 `src/schema/_model/_positional_encoding.yaml`** — delete the 49
**unsuffixed** entries under `use_pe.properties` (:394+). Keep the node
(`type: object`, `additionalProperties: false`, titles/descriptions); adjust
the description to name the key shape explicitly
(`use_pe.<mixer>_use_pe`). The assembler injects all 49 suffixed keys with
spec-derived defaults (`schema_assembly.py:106-131`).

**B6 configs** — update `configs/examples/pe_mixed_use_pe.yaml` to the
suffixed shape (`standard_use_pe: true`, `retnet_use_pe: false`,
`titan_use_pe: true`, `mamba_use_pe: false`). It is the only config using the
block (verified by grep), and the smoke gates enforce the change.

**B7 docs** — fix the key-shape wording in `docs/architecture.md:168`
(`use_pe.<mixer>` → `use_pe.<mixer>_use_pe`) and `:529`, plus
`configs/README.md:359,426,443,543`.

**B8 gates rewrite** (`tests/test_schema_consistency.py`):
- `test_layer_pattern_enum_matches_documentation` (:74-81) → resolve the
  assembled schema and assert every enum name appears in **both** description
  tables (hand rows + auto-appended fallback rows).
- `test_optimizer_prefixes_match_conditional_rules` (:84-98) → drop the raw
  regex scan; assert against the assembled schema that
  `by_optimizer` keys == registry optimizer names and that each `if/then`
  rule's `const` matches its `^<name>-` pattern (generator self-consistency).
- Extend `tests/test_model_config.py::test_per_mixer_use_pe_defaults` (or add
  a new test): assembled `use_pe` properties == `{f"{n}_use_pe"}` for every
  registry mixer and nothing else; `use_pe_for` honours an explicit suffixed
  override from the config mapping.

**B9 `docs/architecture.md` §11/§12 wording** — §11.1 step 4 already says "No
schema enum, `by_optimizer`, `if/then` or `config.py` field edit is needed";
add one sentence noting the raw YAML no longer carries component enums at all
(single source: the registry). §12 gate table rows for the two rewritten
tests stay accurate — update their "Locks" text if it still says "raw
`_training.yaml`".

**B10 regenerate `docs/schema.assembled.yaml`** —
`uv run --extra cpu --extra train python -m src.registry.snapshot --out
docs/schema.assembled.yaml`. Expected diff: the 49 unsuffixed `use_pe` keys
disappear; nothing else changes (the removed raw enums/`by_optimizer`/`if/then`
were already overwritten by the assembler).

## 4. Workstream C — derive `CACHE_SPECS` from the registry

**C1 `src/model/cache/capabilities.py`** — delete the local `CacheSpec`
dataclass and the hand `CACHE_SPECS` dict (:40-60, :111-472). Replace with:

```python
from src.registry.spec import CacheSpec  # re-export; __all__ unchanged

def _derive_cache_specs() -> Dict[str, CacheSpec]:
    from src.registry import ComponentFamily, ensure_builtins, registry
    ensure_builtins()
    return {
        spec.name: spec.cache
        for spec in registry.specs(ComponentFamily.MIXER)
        if spec.cache is not None
    }

CACHE_SPECS: Dict[str, "CacheSpec"] = _derive_cache_specs()
```

`IMPLEMENTED_MIXERS`, `spec_for`, `is_cacheable` (with the `engram` special
case) stay byte-identical. Zero behavioural change: verified today that all
49 registry `spec.cache` entries equal the hand dict (fields and notes).

**C2 guard against drift** — keep
`tests/test_schema_attention_layers.py::test_cache_specs_match_layer_pattern_enum`
(`set(CACHE_SPECS) == enum`, census count) and add one assertion to
`tests/test_registry_core.py::test_training_free_and_cache_sets` (or a new
test): every `MixerSpec` has a non-`None` `cache` declaration (the derivation
silently skips `None`, so require completeness).

**C3 docs** — `docs/architecture.md` §7.13 (:758-762 "holds one `CacheSpec`
per `layer_pattern` enum value") → "re-exports the per-mixer `CacheSpec`
declared on each registry `MixerSpec`". AGENTS.md "Where things live" row for
`src/model/cache/` stays; the "Mixer registry + `HybridLayer`" row becomes
"`HybridLayer` (registry-driven mixer dispatch)".

## 5. Workstream D — small fixes

- **D1** `full_tests/run_e2e.py:238` — remove the duplicate `"retnet"` from
  `ATTENTIONS`. Effect: `attn_single_retnet` appears once; the pair sweep
  drops `attn_retnet__retnet` and the duplicated retnet pairs (~50 fewer
  combos). Verify with `--category attn --list-combos | grep -c retnet`
  before/after (read-only at implementation time).
- **D2** `tests/test_schema_attention_layers.py:19-20` — remove the duplicate
  `"retnet"` in the expected set.
- **D3** `full_tests/param_estimate_check.mjs:24` — fix the default estimator
  path: `path.resolve(here, '../../erickfmm.github.io/...')` →
  `path.resolve(here, '../erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js')`
  (one `..` too many today). Follow-up wording in sync-website.md §5.2 (A3.7).
- **D4** `docs/architecture.md:98` — "80 files" → "84 files" (counted; keep
  the sentence shape).
- **D5** `AGENTS.md` — "Mixer registry + `HybridLayer`" table row →
  "`HybridLayer` (registry-driven mixer dispatch)" (see C3).

## 6. Workstream E — mirror sync + state script

**E1 `.opencode/command/sync-website-state.sh`** (4th file, companion of A3) —
add next to the schema pairing (:107):

```bash
pair_tree "$REPO_ROOT/src/registry" "$SITE/registry" "src/registry" "registry" -type f -name "*.py"
```

(`*.py` only → `__pycache__` never pairs; includes `snapshot.py` which the
mirror currently lacks). This makes the existing manual `registry/` mirror a
tracked surface.

**E2 apply the sync** (per the *updated* `sync-website.md` workflow):
1. Re-run the state script → copy the reported drift: the 10 DIFF files
   (paper sections 01/09 EN+ES, annex-1/annex-18 EN+ES, `paper.pdf`,
   `paper-es.pdf`), the 16 MISSING `docs/paper/figures/tutorial/*.svg`, and
   the newly-paired `registry/` files (`binding.py`, `snapshot.py`).
2. Copy prebuilt PDFs (`docs/paper/paper.pdf` → `paper/en/paper.pdf`;
   ES likewise) — they are fresher than the mirror's — then the top-level
   copies `cp paper/en/paper.pdf paper/en.pdf` / `cp paper/es/paper-es.pdf
   paper/es.pdf` (probes `paper-en-copy`/`paper-es-copy`).
3. Rebuild the Paper-tab HTML (sections/appendices changed):
   `flatten_tex.py --web` + `pandoc` + `postprocess_html.py` for en+es
   (exact commands in sync-website.md §3.3).
4. Pre-authorized cleanup (user-approved): `git -C erickfmm.github.io rm`
   every non-`*.svg` file under `paper/figures/` (leftover `.mmd`, `.pdf`,
   `build-figures.sh`, `puppeteer-config.json`, `mermaid-paper.json`,
   `README.md`, `web-theme.css`, …). Command shape:
   `find erickfmm.github.io/frankenstein-transformer/paper/figures -type f ! -name '*.svg' -print0 | xargs -0 git -C erickfmm.github.io rm`.
5. Validation Gate: `param_count_check.py` → `0 failures` (truth regenerated;
   expected unchanged — no config parameter counts changed),
   `param_estimate_check.mjs … --tol 0.01` → `ALL PASS`, JSON round-trip
   script, CLI parse script, state script → `drift: diff=0 missing_site=0
   only_site=0` and `probes_failed=0`.
6. Leave **both repos uncommitted** (per sync-website.md §6); commits go
   through `/make-a-commit` only if the user asks.

## 7. Execution order

1. **A** (command files) — docs-only, no code impact.
2. **D** (small fixes) — independent, cheap; do early so the later snapshot
   regen and mirror sync carry them.
3. **B** (schema shadows + `use_pe`) — schema + code + tests + snapshot.
4. **C** (CacheSpec derivation) — code + tests.
5. **E** (state script patch + mirror sync) — last, once the main repo is
   clean, so one mirror sync carries everything.

Per-phase commands (implementation time):

```bash
# after A+D:
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings

# after B:
uv run --extra cpu --extra train python -m pytest \
  tests/test_schema_consistency.py tests/test_schema_attention_layers.py \
  tests/test_model_config.py tests/test_registry_core.py \
  tests/test_schema_assembly.py tests/test_yaml_examples.py -v --tb=short -p no:warnings
uv run --extra cpu --extra train python -m src.registry.snapshot --out docs/schema.assembled.yaml
git diff --stat docs/schema.assembled.yaml   # expect: only the 49 unsuffixed use_pe keys removed

# after C:
uv run --extra cpu --extra train python -m pytest \
  tests/test_kv_cache_slots.py tests/test_kv_cache_mixers.py \
  tests/test_kv_cache_fallback.py tests/test_kv_cache_mechanisms.py \
  tests/test_inference_config.py tests/test_registry_core.py \
  tests/test_schema_attention_layers.py -v --tb=short -p no:warnings

# e2e sanity for D1 (no training):
uv run --extra cpu --extra train python full_tests/run_e2e.py --category attn --list-combos | grep -c retnet

# E: mirror sync per the updated sync-website.md, then:
bash .opencode/command/sync-website-state.sh   # expect drift=0, probes_failed=0
```

## 8. Verification matrix

| Gate | Command | Expected |
|---|---|---|
| Registry gates | pytest `test_registry_core.py test_registry_binding.py test_schema_assembly.py test_plugin_loader.py` | PASS |
| Schema gates | pytest `test_schema_consistency.py test_schema_attention_layers.py test_model_config.py` | PASS (rewritten tests) |
| Cache suites | pytest `test_kv_cache_*.py test_inference_config.py` | PASS (zero behavioural change) |
| Presets | pytest `test_yaml_examples.py` + `test_all_presets_validate` | PASS (incl. updated `pe_mixed_use_pe.yaml`) |
| Full suite | pytest `tests/ --continue-on-collection-errors` | PASS |
| Snapshot | `python -m src.registry.snapshot --out docs/schema.assembled.yaml` | diff = removed unsuffixed `use_pe` keys only |
| e2e combos | `run_e2e.py --category attn --list-combos \| grep -c retnet` | each retnet combo id once |
| Param truth | `full_tests/param_count_check.py` | `0 failures`, truth unchanged |
| Estimator | `node full_tests/param_estimate_check.mjs <mirror>/ft-param-estimator.js full_tests/param_truth.json --tol 0.01` | `ALL PASS` |
| State script | `bash .opencode/command/sync-website-state.sh` | `drift: diff=0 missing_site=0 only_site=0`, `probes_failed=0` |
| Mirror git | `git -C erickfmm.github.io status --short` | only intended copies + pre-authorized figure deletions |

## 9. Risks and assumptions

- **Hidden raw-schema consumers**: audited — every consumer resolves through
  `resolve_schema` (config_loader, GUI, transformers export, snapshot). The
  only raw readers are the two consistency tests rewritten in B8. Re-grep at
  implementation time before deleting the raw blocks.
- **`use_pe` breaking change**: configs using the unsuffixed shape now fail
  validation loudly instead of being ignored silently. Only
  `configs/examples/pe_mixed_use_pe.yaml` uses it (fixed in B6). This is the
  intended bugfix direction (schema-valid keys must never be silently
  ignored).
- **CacheSpec derivation**: import cycle risk is nil (`src/registry` does not
  import `src.model.cache`; verified by grep). Content equality verified
  (fields + notes, all 49). `spec_for`'s unknown-key fallback unchanged.
- **`training.allOf` relocation** (B4): the unconditional assignment is the
  pre-assembly path that already runs today (the raw file has `allOf`), so
  behaviour is identical.
- **Mirror deletions are irreversible locally but git-tracked** in the mirror
  repo (recoverable until the mirror is pushed).
- Line numbers drift: all anchors quoted against the current `main`
  (2026-11-11); re-locate by content if they moved.

## 10. Out of scope

- No new components, no census changes (49 mixers / 36 optimizers stay).
- No paper (LaTeX) edits beyond none needed — this plan does not touch
  `docs/paper*/` (census sentences are all correct today; the mirror paper
  sync is a copy, not an edit).
- No commits/pushes; both repos stay uncommitted unless the user asks.
- The `.opencode/command/*.md` backtick-command examples inside the two
  add-* commands keep their existing structure; only stale content is fixed.
