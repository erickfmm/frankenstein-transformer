---
description: Plan only (no implementation) the addition of a new sequence mixer from an arXiv/alphaXiv paper — read the paper, explore the repo and emit a complete file-by-file plan covering implementation, registry, schema, KV-cache decision, tests, e2e real-data smoke, mermaid figures, bilingual paper annexes, PDF rebuild, website sync (ft-diagram.js, ft-param-estimator.js) and git
agent: plan
subtask: false
---

> **Registry-first recipe (current architecture).** Component additions no
> longer edit a second list by hand. Every pluggable component is a frozen
> `ComponentSpec` in `src/registry/builtin/<family>.py` (or a plugin calling
> `src/plugins/api.py`): name, lazy builder, parameter-schema fragment,
> runtime defaults, capability flags and bilingual description. The schema
> enums, `by_optimizer` reference, `if/then` prefix rules, attention property
> fragments, factory dispatch, Streamlit pickers and e2e listing are all
> **generated from the spec** by `src/registry/schema_assembly.py` and the
> family factories. `src/utils/config_flatten.py` no longer exists and the flat
> constructor (`from_flat`) has been removed: the runtime config is the nested
> YAML shape (`FrankensteinModelConfig.from_dict`) end-to-end.
>
> The file lists below are retained for the *paper/annex/census/website*
> phases of a core research component; the implementation/schema/dispatch
> rows are superseded by: **implement the module → add a spec → add an
> example → run the gates.**

Produce a **complete, implementation-ready plan** for adding one new sequence
mixer to frankenstein-transformer — and nothing else. This command runs in
**plan mode**: it reads the paper, explores the repository and writes the plan
to `.opencode/plan/add-mixer-<arxiv-id>.md`. It does **not** create, edit or
delete code, schema, presets, tests, figures, docs, PDFs, website-mirror files
or git history.

The canonical recipe is `docs/architecture.md` §11.1 ("Add or change a
component (any family)", ~line 1058), §11.3 ("Capability declarations",
~line 1086) and the **Mixer** row of `AGENTS.md`. The registry is the single
source of truth (D13/D14): the schema enums, `model.attention` fragments,
`use_pe.<mixer>_use_pe` entries, factories and pickers derive from the
`MixerSpec` — no second list is edited by hand. The plan must cover every step
of that recipe plus paper reading, KV-cache decision, mmd figures, bilingual
annexes, summary tables, census bumps, Sphinx, e2e smoke (including a targeted
**real-data training** run), param gates, PDF compilation, website sync and
git — in enough detail that the implementer can execute it top to bottom
without re-deriving anything from the paper.

## Non-negotiable guardrails

This is a planning command. The prohibitions below win over any later sentence
in this document that describes an action:

| Forbidden now | Required instead |
|---|---|
| Creating/editing/deleting anything outside `.opencode/plan/` | Describe the exact change in the plan |
| `git add`, `commit`, `push`, `checkout`, `stash`, `reset` | `git status --short`, `git log`, `git diff` (read-only) |
| Running `pytest`, `full_tests/run_e2e.py`, `full_tests/param_count_check.py`, `full_tests/param_estimate_check.mjs` | List them as *planned verification* with expected results |
| Running `pdflatex`, `bibtex`, `pandoc`, `flatten_tex.py`, `docs/build_docs.sh`, `figures/build-figures.sh` | Write the exact command sequence for implementation time |
| Copying/syncing into `erickfmm.github.io/`, editing `ft-diagram.js` / `ft-param-estimator.js` / `index.html` | Enumerate the exact site-only edits to make later |
| `uv sync`, training, inference, `web-server` | Quote the CPU-first commands for the implementer |
| Touching `src/`, `configs/`, `tests/`, `docs/`, `full_tests/`, `docs/source/` | Read only |

If the runtime nevertheless exposes edit tools for paths outside `.opencode/plan/`,
ignore them: the deliverable is a plan, not a patch. Never "do a small fix while
you are there". If the paper implies a destructive or out-of-scope step (renaming
an existing mixer, deleting a family, touching unrelated drift), do not perform
it — flag it in the plan as requiring explicit user approval.

## Argument

`$ARGUMENTS` must contain an arXiv ID or URL of the mixer paper. Accepted shapes:

- `2510.26692`, `arXiv:2510.26692`
- `https://arxiv.org/abs/2510.26692`
- `https://www.alphaxiv.org/abs/2510.26692` (or `/overview`, `/html`, …)

Normalize to the bare arXiv ID and use it in the plan filename
(`.opencode/plan/add-mixer-<arxiv-id>.md`). If `$ARGUMENTS` is empty or no ID can
be extracted, STOP and ask the user for the paper link — do not invent a topic.

## Phase 1 — Read the paper (read-only)

Fetch the paper, in order of preference:

1. alphaXiv/arXiv MCP tools through `execute` (`tools.arxiv.download_paper` +
   `read_paper`, or the `tools.alphaxiv` paper tools) — they render the full text.
2. Fallback: `webfetch` on `https://arxiv.org/abs/<id>` (metadata + abstract) and,
   if needed, `https://arxiv.org/pdf/<id>`.

Extract and record verbatim (everything downstream depends on it):

- **Forward equations**: every equation of the mixer in LaTeX, exactly as in the
  paper — projections (Q/K/V/O or packed), gates/decays, the mixing core
  (recurrence, kernel, mask, latent path…), output normalization/gating. These
  feed the module, the annexes (EN + ES) and the `.mmd` diagram.
- **Algorithm/pseudocode**: parallel/training form and recurrent/decode form if
  both exist.
- **Parameters**: every learned tensor (name, shape, bias?) — enough to write the
  **exact parameter-count formula** in terms of `H, nH, nKv, hd, fH` that
  `ft-param-estimator.js` and `full_tests/param_count_check.py` must agree on.
- **Hyperparameters/knobs**: name, type, default, range, constraints. These
  become the spec's `params_schema` fragment
  (`schema/_model/_attention_<name>.yaml#/properties/<name>`) plus its
  `defaults`, read at runtime through `mixer_params`; there is no flat
  dataclass field and no `_ATTENTION_MIXER_RENAMES` entry. "No knobs" is a
  valid outcome (e.g. `kda`).
- **State & cache**: recurrent state / KV / latent / ring buffers (shape, count),
  per-step decode cost, and which cache compatibility class (i)–(vii) of
  `src/model/cache/capabilities.py` it matches, if any.
- **Complexity**: training and inference Big-O in `n, d, w, r, k` for
  `tab:all-mixers`.
- **Trainability**: does the paper define a trainable from-scratch form, or is it
  an inference-only/training-free method (→ `TRAINING_FREE_LAYERS` dense surrogate
  path like `fasa`/`sparge`, hard rule 5)?
- **Positional encoding**: does the mixer consume positional information?
  Determines the spec's `needs_pos_encoder` flag (`true` softmax-attention-like,
  `false` recurrent/decay-like). `use_pe_for` reads it, and the assembler
  injects the matching `use_pe.<name>_use_pe` schema entry — no hand edit.
- **Mode constraints**: encoder, decoder, or both; causal formulation present?
  Grid/vision-only (like `ssog`)? Audio-compatible?
- **BibTeX data**: authors, year, title, arXiv ID.

**Decision gate.** If the paper is not a sequence-mixing method, duplicates a
mixer already in the registry (`src/registry/builtin/attention.py` /
`registry.names(ComponentFamily.MIXER)`), or is
underspecified (missing equations/shapes that cannot be recovered), stop after
Phase 1 and return a short rejection report: verdict, evidence (quoted paper
text + the existing registry entry for duplicates), and what a usable paper would
need to define. Do not write a plan. If the paper proposes several mixer
variants, plan the main one and record the others under "Open questions"; ask the
user only if the choice is genuinely ambiguous.

## Phase 2 — Explore the codebase (read-only)

Read before planning anything, and cite concrete anchors (path + line) in the
plan:

- `AGENTS.md` — **Mixer** extension row, hard rules 2, 5, 6 and 9, gates list.
- `docs/architecture.md` §7.4 (mixer table, census, `TRAINING_FREE_LAYERS`
  note), §7.13 (cache lists), §11.1 (recipe), §12 (gates).
- Peers in `src/model/attention/`: pick the **closest peer by family and shape**
  (e.g. `gated/kda.py` for a gated delta-rule mixer, `latent/cca.py`
  for a latent+conv mixer, `sparse/swa.py` for a masked-window mixer) and
  note exactly which parts will be mirrored (constructor signature, `use_pe`
  handling, `use_fixed_softmax` dispatch via `numerics.module_softmax`, decoder
  masking, `forward`/cached call shapes S1/S2 per §7.13).
- `src/model/hybrid_layer.py` — registry-driven mixer dispatch (no local
  `mixer_registry` dict anymore); `TRAINING_FREE_LAYERS` is derived as
  `registry.property_set(ComponentFamily.MIXER, "training_free")` (~line 80);
  the dense surrogate `_training_free_dense_forward` (~line 306);
  `use_fixed_softmax` is attached from `config.numerics.fixed_softmax`
  (~line 176). A `training_free=True` spec needs **no** edit here.
- `src/registry/spec.py` — `MixerSpec` fields and flags (`cache`, `cache_implemented`,
  `training_free`, `needs_pos_encoder`, `qk_norm_supported`, `qk_clip_supported`,
  `aliases`, ~line 174) and the `CacheSpec` dataclass (~line 54).
- `src/registry/builtin/attention.py` — the `ATTENTION_SPECS` tuple: name,
  `builder="src.model.attention.<family>.<name>:<Class>"`, EN/ES descriptions,
  `references`, `params_schema` fragment pointer, `defaults`, cache flags.
- `src/model/attention/params.py` — `mixer_params(config, "<name>", params)` and
  `use_pe_for(config, "<name>", default)`; knobs/spec defaults flow through here.
- `src/model/attention/__init__.py` and the family `__init__.py` export pattern.
- `src/model/cache/capabilities.py` — `CacheSpec`, `CACHE_SPECS` (derived from
  the registry `MixerSpec.cache` after the consistency cleanup), `IMPLEMENTED_MIXERS`
  (hand-maintained frozenset), compatibility classes; `tests/test_kv_cache_mixers.py`
  parametrizes itself from `IMPLEMENTED_MIXERS` (no hand edit).
- Schema: `src/schema/_model/_dims.yaml` — **both** bilingual tables only (the
  enum is registry-injected at assembly; the assembler's fallback appends a
  2-column row, so hand-write the 4-column row in EN and ES). One
  `_attention_<mixer>.yaml` knob file as template (`_attention_swa.yaml`),
  referenced from the spec via `params_schema`. `_model.yaml` and
  `_positional_encoding.yaml` are fully generated surfaces — do not edit them.
- `src/registry/schema_assembly.py` — `_inject_mixers` (~line 68): enum
  replacement, missing bilingual rows, `model.attention.properties` rebuilt from
  spec fragments, `use_pe.<name>_use_pe` auto-injection from `needs_pos_encoder`.
- Tests: `tests/test_schema_attention_layers.py` (hand expected set;
  `test_cache_specs_match_layer_pattern_enum` locks the enum census — bump its
  count when adding a mixer), `tests/test_schema_consistency.py`
  (`test_layer_pattern_enum_matches_documentation` against the assembled schema,
  `test_schema_defaults_match_dataclass`, preset validation),
  `tests/test_yaml_examples.py`, the family test files, `tests/test_bitnet_coverage.py`,
  and the registry gates (`test_registry_core.py`, `test_registry_binding.py`,
  `test_schema_assembly.py`).
- e2e: `full_tests/run_e2e.py` — the `ATTENTIONS` list (~lines 237–272, one
  comment block per recent mixer, with its training-free skip note), the combo
  builders `make_attention_pair_configs` (~line 790, `attn_<a>__<b>`, all pairs)
  and `make_single_attention_configs` (~line 805, `attn_single_<mixer>`, 1 layer),
  and the smoke flags: `--category attn`, `--only PATTERN[,PATTERN…]` (substring
  filter over combo ids, applied before `--limit`), `--realonall` (**default ON**:
  every combo trains on the real Borges corpus instead of the synthetic toy one;
  `--no-realonall` is the offline fallback), `--list-combos` (~line 2032, print
  ids without training), `--skip-attn-pairs`/`--skip-singles`, and the sizing
  knobs `--max-samples/--max-epochs/--batch-size/--timeout`.
- Param gates: `full_tests/param_count_check.py`, `full_tests/param_truth.json`,
  `full_tests/param_estimate_check.mjs`.
- Figures: `docs/paper/figures/mixers/<peer>.mmd` (house style),
  `mixer-taxonomy.mmd`, and the family map **when the family has one**
  (`dense-map.mmd`, `latent-map.mmd`, `memory-map.mmd`, `recurrent-map.mmd`,
  `sparse-map.mmd`; gated/fast-weight/geometric/conformer have no map),
  `build-figures.sh`.
- Paper EN/ES: the **numbered family annex** for the chosen family (grep the
  peer's escaped name, e.g. `kda`, across `docs/paper{,-es}/appendices/*.tex`;
  dense peers live in `annex-2-recurrent-attention.tex` today, e.g. `ca`),
  `sections/01-introduction.tex` (global census), `sections/04-architecture-taxonomy.tex`
  (+ ES), `annex-summary-tables.tex` (+ ES), `annex-schema.tex` (+ ES),
  `annex-15-kv-cache.tex` (EN) / `annex-23-cache-kv.tex` (ES) for cacheable
  mixers, `docs/bibliography/attention_types.bib`.
- Website: `.opencode/command/sync-website.md` (file map, site-only files,
  Validation Gate) and the mirror's `ft-diagram.js` (`CATS`, `SUBSTRUCT`,
  `STYLES`) + `ft-param-estimator.js` (`mixerParams`) — fetch from
  `https://raw.githubusercontent.com/erickfmm/erickfmm.github.io/main/frankenstein-transformer/`
  if the mirror is not cloned. Only `*.svg` figures flow to the mirror.
- Sphinx: `docs/source/api/model/attention/` — top-level `.rst` plus the
  `gated/`, `latent/`, `sparse/` subdirectories (not every family has a dir)
  and the matching `index.rst` toctree; `docs/build_docs.sh`.
- Reference blueprint: `git log --oneline -3 -- <peer module>` and skim the
  commit that added the most recent mixer — the plan should mirror that scope.
  Note: no mixer has been added since the registry rollout, so the
  pre-registry commits (e.g. `2990b69`, FPA) show the paper/annex scope but
  their schema rows are superseded by the spec-first recipe.
- Read-only inventory to quote in the plan: current census numbers (enum count,
  per-family counts in `README.md` "Mixer Categories", `docs/architecture.md`
  §7.4 ~line 390, `annex-summary-tables.tex` intro + family headers,
  `mixer-taxonomy.mmd` labels, `configs/README.md` lists); available
  `pdflatex`/`bibtex` (`command -v pdflatex bibtex`); Chrome/Chromium for
  mermaid (`command -v google-chrome chromium chromium-browser`);
  `git status --short` (must be clean).

Do not run mutating commands to "check" anything. If a fact can only be
confirmed by running code, record it as an **assumption to verify at
implementation time**.

## Phase 3 — Decide

Record these decisions explicitly, with the evidence that supports them. Use the
question tool to ask the user whenever a decision is genuinely ambiguous
(family assignment, registry name, cache scope, preset — see below):

- registry name `<name>` (snake_case, from the paper's short name; no
  `_attn` suffix — it was removed in 1.2) and class name `<Name>Attention`;
- family directory `gated/ | latent/ | sparse/ |` top level, and the **paper
  family label** used by every census table (Dense/GQA/Recurrent/Sparse/Gated/
  Fast-weight/Latent/Geometric/Convolution-augmented — or a justified new one,
  which additionally costs a new `mixer-taxonomy.mmd` branch, a new family map
  figure, and a new `ft-diagram.js` STYLES class);
- closest peer module and the parts to mirror;
- knob table: schema key (`model.attention.<subkey>.<leaf>`), the spec
  `params_schema` fragment leaf, type, default, constraints, coercion, EN + ES
  description text — or "no knobs";
- `needs_pos_encoder` flag (the `use_pe` default; no dataclass field);
- **KV-cache decision (ask the user when non-obvious)**, expressed as `MixerSpec` flags:
  1. *simple decode form* (classic KV / latent KV / fixed state / ring / token
     tail) → `cache=CacheSpec(supported=True, slot_type=…)` +
     `cache_implemented=True` **and** the incremental branch **and**
     `IMPLEMENTED_MIXERS` membership (the parity suite auto-covers it);
  2. *cacheable in principle but heavy* (chunk-level statistics, bespoke kernels)
     → `cache=CacheSpec(supported=True, notes=…D5 fallback…)`,
     `cache_implemented=False`, and list the incremental branch as an optional
     follow-up phase requiring user approval;
  3. *not sequence-attention over past tokens / no causal form* →
     `cache=CacheSpec(supported=False, notes=…)` (conformer/ssog precedent).
- training-free / eval-only? If yes: plan the `dense_forward` surrogate,
  `training_free=True` on the spec (`TRAINING_FREE_LAYERS` then derives it — no
  `hybrid_layer.py` edit), the ⚠️ rows in `_dims.yaml` and `README.md`,
  hard-rule-5 wording everywhere, and **exclusion from `ATTENTIONS`** (its
  header comment reserves it for trainable mixers).
- example preset: `configs/examples/<family>_<mixer>_adamw.yaml` (encoder MLM
  baseline, `es_arch_*` shape such as `es_arch_fpa_adamw.yaml`,
  `load_training_config`-clean);
- **optional full preset**: if the paper describes a reference configuration,
  ask the user whether to also draft `configs/<name>.yaml` (like
  `configs/kimi_k3.yaml`) — it joins the smoke-tested preset set either way;
- census bumps: the *actual* old numbers read from the repo (do not hardcode 49
  or 9) — global mixer count, per-family count, taxonomy labels. Today: 50
  mixers in 9 families (49 concrete classes + the `mamba` placeholder), 36
  optimizers, 84 test files, 99 `.mmd` diagrams / 142 `.svg` renders.

Print the **file-by-file plan table** (file → change) in the chat response
before writing the plan document, so the user can veto a decision before the
plan is finalised.

## Phase 4 — Write the plan document

Create `.opencode/plan/add-mixer-<arxiv-id>.md` with this skeleton. Every section
is mandatory; "n/a" is allowed only with a one-line reason.

### 1. Header
`# Plan: add <Name> mixer (<arXiv id>)` + one-paragraph paper summary (what
problem, what mechanism, why it belongs in this repo).

### 2. Paper extract
Forward equations in LaTeX, parallel + recurrent pseudocode, parameter inventory
and the exact parameter-count formula, knob table, state buffers and cache
compatibility class, complexity (train/inference), trainability, PE interaction,
mode constraints, bibtex data.

### 3. Decisions
From Phase 3: names, family, peer, knobs, `use_pe` default, KV-cache decision
(with the gate's evidence), example + preset, census bumps.

### 4. File-by-file plan table
One row per file, in execution order, with the *intended content*, not just the
intent:

| # | File | Planned change (be concrete) |
|---|---|---|
| 1 | `src/model/attention/<family>/<name>.py` | Module citing the paper in the docstring; `from __future__ import annotations`; Google-style docstrings; CPU-safe; `(config, pos_encoder=None, params=None)` constructor per §11.1; equations mapped to code lines; knobs via `mixer_params(config, "<name>", params)`; `use_pe` via `use_pe_for(config, "<name>", <default>)` (default = the spec's `needs_pos_encoder`); honor `self.use_fixed_softmax`; decoder causal path; `forward` call shape S1/S2 per §7.13; `dense_forward` only if training-free. |
| 2 | `src/model/attention/<family>/__init__.py` + `attention/__init__.py` | Import + `__all__` entry (quote surrounding lines). |
| 3 | *(removed)* | `src/model/hybrid_layer.py` needs **no** edit: dispatch reads the registry, and `TRAINING_FREE_LAYERS` derives from the spec's `training_free` flag. |
| 4 | `src/registry/builtin/attention.py` | Full `MixerSpec`: `name`, `builder="src.model.attention.<family>.<name>:<Class>"`, `description`/`description_es`, `references`, `params_schema="schema/_model/_attention_<name>.yaml#/properties/<name>"`, `defaults`, `cache=CacheSpec(...)`, `cache_implemented`, `qk_norm_supported`, `qk_clip_supported`, `training_free`, `needs_pos_encoder`. This is the **single** declaration: the schema enum, `model.attention` fragment and `use_pe` key all derive from it. |
| 5 | *(removed)* | The config flattener is gone; no flat rename entry exists. |
| 6 | `src/schema/_model/_dims.yaml` | One row in **both** the EN and ES bilingual tables with `(paper: arxiv:<id>)` (4 columns; ⚠️ wording if training-free). The enum list is registry-injected — do not hand-edit it. |
| 7 | `src/schema/_model/_attention_<mixer>.yaml` | Knob block (only if knobs) — referenced from the spec's `params_schema`; draft the full bilingual YAML (title/title_es, descriptions with examples + references, defaults matching the spec `defaults`). |
| 8 | *(removed)* | `src/schema/_model.yaml` needs no edit: the assembler rebuilds `model.attention.properties` from the spec fragments. |
| 9 | *(removed)* | `src/schema/_model/_positional_encoding.yaml` needs no edit: the assembler injects `use_pe.<name>_use_pe` from `needs_pos_encoder`. |
| 10 | *(spec flags only)* | Cache capability lives on the `MixerSpec` (`cache=`, `cache_implemented=`); `src/model/cache/capabilities.py` re-exports it. `IMPLEMENTED_MIXERS` membership + the incremental branch are the only real edits (KV-cache decision 1). |
| 11 | `configs/examples/<family>_<mixer>_adamw.yaml` | Tiny example; must pass `load_training_config`. Draft the full YAML. |
| 12 | *(optional)* `configs/<name>.yaml` | Full preset if the user approved it (D-C3-style question). Draft it. |
| 13 | `full_tests/run_e2e.py` | Append the name to `ATTENTIONS` (~lines 237–272; add a one-line comment with the arXiv id like the recent entries). This **auto-generates the e2e combos**: `attn_single_<name>` (1-layer solo) and every `attn_<a>__<name>` / `attn_<name>__<b>` pair — no builder edits needed. Skip when training-free (see the list's header comment; `fasa`/`sparge` precedent). |
| 14 | `tests/test_schema_attention_layers.py` | Add the name to the expected enum set **and** bump the count in `test_cache_specs_match_layer_pattern_enum` (50 → 51). |
| 15 | Family test (`tests/test_<family>_attention.py` or `test_attention_modules.py`) | Test class: builds on a tiny config, forward in encoder + decoder modes, output shape, knobs honored, no NaN/Inf, BitNet on; name the methods. |
| 16 | `tests/test_bitnet_coverage.py` | Add the name to the mixer list. |
| 17 | `tests/test_kv_cache_mixers.py` | **No hand edit**: the parity suite parametrizes itself from `IMPLEMENTED_MIXERS`; only the incremental branch + membership are real edits (KV-cache decision 1). |
| 18 | `docs/paper/figures/mixers/<short>.mmd` | Draft the full mermaid diagram in house style (init header, flowchart TD, tensor-shaped labels, labelled edges, dotted recurrence). |
| 19 | `docs/paper/figures/mixer-taxonomy.mmd` | Extend the family chain + bump its count label. |
| 20 | *(family-dependent)* `docs/paper/figures/<family>-map.mmd` | Add the node when a family map exists. |
| 21 | `docs/bibliography/attention_types.bib` | Entry keyed `<firstauthor>_<shortname>_<year>`. Draft it. |
| 22 | `docs/paper/appendices/annex-<n>-<family>.tex` | `\subsection{N. <Name>}` numbered after the annex's last; figure `\fitfigure{figures/mixers/<short>.pdf}` + label; Mathematical Foundation with the paper's equations; algorithm float; Empirical Performance and Trade-offs + strength/limitation table; `\citep{<key>}`; extend the annex's closing unified-principle list if present. Locate the annex by grepping the peer. Draft the subsection. |
| 23 | `docs/paper/sections/04-architecture-taxonomy.tex` | Family paragraph sentence + counts. Quote the lines holding the old count. |
| 24 | `docs/paper/appendices/annex-summary-tables.tex` | `tab:all-mixers` row + family count header + intro census. Draft the row. |
| 25 | `docs/paper/appendices/annex-schema.tex` | Mixer row. Draft it. |
| 26 | `docs/paper-es/appendices/annex-<n>-<family>.tex` | Faithful Spanish translation of #22 (figures from `../paper/figures/`). |
| 27 | `docs/paper-es/sections/01-*.tex`, `04-*.tex`, `annex-summary-tables.tex`, `annex-schema.tex` | Spanish counterparts of #23–#25 (+ the global census in section 01). |
| 28 | `README.md`, `docs/architecture.md`, `configs/README.md` | Census pass: README Mixer Categories row + counts (+ ⚠️ note when eval-only); architecture §7.4 census + row (+ §7.13 cache list, + TRAINING_FREE note); configs/README family list. List every occurrence of the old counts found by grepping them. |
| 29 | `docs/source/api/model/attention/<family>/<name>.rst` + `<family>/index.rst` | `automodule` page + toctree entry (top-level `attention/` for families without a subdir). |
| 30 | `docs/schema.assembled.yaml` | Regenerate with `python -m src.registry.snapshot --out docs/schema.assembled.yaml` (the snapshot the mirror consumes). |
| 31 | `docs/paper/appendices/annex-15-kv-cache.tex` + `docs/paper-es/appendices/annex-23-cache-kv.tex` | Add the mixer to the incremental-decode list when it is cacheable (KV-cache decision 1). |
| 32 | `docs/paper/sections/01-introduction.tex` (+ ES), `10-*.tex`, `11-*.tex` | Global census bump when the total mixer count changes (FPA precedent; the abstract sentence lives in section 01). |

Hard rules to restate in the plan: the schema is the contract
(`additionalProperties: false`); every pluggable component is a declarative
`ComponentSpec` and the registry is the single source of truth (D13), so the
validated schema is core YAML + registry fragments (D14) and the runtime config
is nested only (D15); `hidden_size % num_heads == 0`, `num_kv_heads`
divides `num_heads`, `len(layer_pattern) == num_layers`; inference never fails
because of the cache (D5); `fasa`/`sparge`-style mixers train dense and
go sparse only in eval (rule 5); the papers are the theory source of truth (D11),
so annex formulations must match the arXiv paper and `paper-es` must translate
them faithfully.

### 5. Ordered execution checklist
The implementer's happy path, as checkboxes: implementation (rows 1–13) → tests
(14–17) → **e2e smoke, incl. targeted real-data training** (below) → figures
(18–20) → docs EN/ES (21–28) → Sphinx (29) → PDFs → website sync → commit/push.
Include the exact commands to run **at implementation time** (do not run them
now):

```bash
# targeted gates
uv run --extra cpu --extra train python -m pytest \
  tests/test_schema_attention_layers.py tests/test_schema_consistency.py \
  tests/test_yaml_examples.py tests/test_attention_refactor.py \
  tests/test_<family>_attention.py tests/test_bitnet_coverage.py \
  tests/test_registry_core.py tests/test_registry_binding.py \
  tests/test_schema_assembly.py \
  -v --tb=short -p no:warnings
# (plus tests/test_kv_cache_mixers.py under KV-cache decision 1 — it
# parametrizes itself from IMPLEMENTED_MIXERS)

# full suite
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings
```

```bash
# e2e — enumerate the new mixer's combos (build-only, no training)
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --list-combos | grep -F <name>

# e2e TARGETED SMOKE — ONLY the new mixer: actual training on REAL data.
# --realonall is the DEFAULT: every selected combo trains on the real Borges
# corpus (not the synthetic toy one). First invocation downloads the
# tokenizer/corpus from the HF Hub. '--only <name>' substring-matches the
# auto-generated combo ids: attn_single_<name> plus every pair
# attn_<a>__<name> / attn_<name>__<b>.
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only <name>

# minimal real-data run if the pair sweep is too heavy: the single solo combo
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only attn_single_<name>

# offline synthetic fallback (no HF download), same selection
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only <name> --no-realonall

# sizing knobs when needed: --max-samples N --max-epochs 1 --batch-size 2 --timeout 1800
# GPU smoke (real training box): swap the extra for --extra cu118 --extra train.
```

Expected evidence of the real-data smoke: each selected combo reports PASS with
a finite training loss over real corpus windows, and the run directory keeps the
generated `config.yaml` + metrics for inspection. A NaN/Inf loss, a schema
rejection or a timeout is a blocker, not a warning.

```bash
# figures (needs Chrome/Chromium or PUPPETEER_EXECUTABLE)
( cd docs/paper/figures && ./build-figures.sh both )

# param ground truth + JS estimator (after the example lands)
uv run --extra cpu --extra train python full_tests/param_count_check.py
node full_tests/param_estimate_check.mjs \
  erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js \
  full_tests/param_truth.json --tol 0.01

# Sphinx API docs
bash docs/build_docs.sh
```

```bash
# PDF rebuild, from the repository root — only at implementation time
cd docs/paper
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
rm -f *.aux *.log *.out *.toc *.blg

cd ../paper-es
pdflatex -interaction=nonstopmode paper-es.tex
bibtex paper-es
pdflatex -interaction=nonstopmode paper-es.tex
pdflatex -interaction=nonstopmode paper-es.tex
rm -f *.aux *.log *.out *.toc *.blg
```

Record preconditions and fallbacks: if `pdflatex`/`bibtex` are unavailable,
leave sources updated and note the rebuild must happen on a TeX machine before
the next release (the website sync copies prebuilt PDFs only); if no Chrome is
available for mermaid, commit the `.mmd` source and note the `.pdf`/`.svg`
render must be produced before the paper/website sync; if the HF Hub is
unreachable for the real-data smoke, record the offline `--no-realonall` PASS
and re-run the real-data pass later. Verification to require after PDF
compilation: no `Fatal error`, no `Citation .* undefined`, `paper.pdf` /
`paper-es.pdf` newer than the edited `.tex`/`.bbl`, the new `\citep{<key>}`
resolves, and `git status --short docs/paper docs/paper-es` shows only intended
`.tex`/`.bbl`/`.pdf`/figure files.

### 6. Website sync (planned, not executed)
Spell out the `/sync-website` steps for this change:

1. Bootstrap/refresh the mirror and copy per its file map: `docs/schema.assembled.yaml`
   (regenerated) + `src/schema/**`, `src/registry/**`, `configs/*.yaml` +
   `configs/README.md`, `configs/examples/**`, `docs/paper/**` +
   `docs/paper-es/**` (tex, bbl, pdf, and the **new `.svg`** figures —
   sources/PDFs stay in the main repo), `docs/bibliography/**`,
   `docs/architecture.md`.
2. **Site-only edits** (mirror is the source of truth for these):
   - `ft-diagram.js`: add `CATS["<name>"]` and a `SUBSTRUCT["<name>"]`
     tensor-flow template (or string alias to the closest peer) matching the
     paper's dataflow; add a `STYLES`/`STYLES_PRINT` class only for a new family
     category.
   - `ft-param-estimator.js`: add the `mixerParams()` case with the exact
     parameter formula from the plan's paper extract (bias- and knob-aware).
   - `index.html`: grep a peer mixer name; extend every hardcoded list found.
3. Paper tab HTML rebuild (`flatten_tex.py --web` + `pandoc` +
   `postprocess_html.py`, en + es) and top-level `paper/en.pdf` / `es.pdf` copies.
4. Validation Gate: `param_count_check.py` → `0 failures` (regenerates
   `full_tests/param_truth.json` — intentional diff), estimator `--tol 0.01` →
   `ALL PASS`, JSON round-trip script, CLI parse script, and the state-script
   probes (`schema-refs`, `example-yaml`, `cli-flags`, …) all PASS.
5. The mirror stays uncommitted until the commit phase.

### 7. Commit plan (planned, not executed)
Follow `/make-a-commit` conventions (English, conventional tag, structured body;
never stage `*.aux *.log *.out *.toc *.blg`, caches or mermaid temp files;
`param_truth.json` changes are intentional):

1. Main repo: `feat(attention): add <Name> mixer (<arXiv id>)` — body covering
   the module, registry spec, schema (bilingual table rows + knob fragment),
   cache decision, example (+ preset), tests, e2e `ATTENTIONS` + real-data
   smoke result, figures (mmd + rendered pdf/svg), docs (annexes EN/ES, summary
   tables, schema annex, kv-cache annex, bibliography, sections 01/04/10/11,
   README/architecture/configs README, Sphinx rst), regenerated
   `docs/schema.assembled.yaml`, rebuilt PDFs; then `git push origin HEAD:main`.
2. Mirror: `git -C erickfmm.github.io …` commit
   `docs(site): sync <Name> mixer (<arXiv id>)`; then
   `git -C erickfmm.github.io push origin HEAD:main`.

List the exact paths to stage for each repo.

### 8. Verification matrix
Gate → command → expected result, one row per gate: targeted pytest families,
full pytest, e2e combo enumeration (`--list-combos` shows `attn_single_<name>` +
pairs), e2e **targeted real-data training** (`--category attn --only <name>` →
all selected combos PASS with finite loss on the real corpus; solo-only variant
`--only attn_single_<name>`), offline fallback (`--no-realonall`), figure
render, PDF checks, Sphinx build, param count (`0 failures`), estimator
(`ALL PASS` ≤1%), mirror probes, KV parity (when decision 1).

### 9. Risks, assumptions and open questions
Assumptions to re-check at implementation time (line drift, tool availability,
HF Hub reachability for the real-data smoke), naming collisions, family
ambiguity, task/mode restrictions (rule 6), BitNet interactions,
chunkwise-kernel simplifications vs the paper, and anything the paper leaves
ambiguous.

### 10. Out of scope
Explicitly: no code, schema, preset, test, figure, doc, PDF, mirror or git change
was made by this command.

## Phase 5 — Deliver

1. Write the plan to `.opencode/plan/add-mixer-<arxiv-id>.md`. If the plan-mode
   runtime denies that write, keep the full plan in the chat response and say so.
2. Reply with: the paper in one paragraph, the decisions (name, family, peer,
   knobs, use_pe, KV-cache decision, example + preset, census bumps), the
   file-by-file table, and the path of the plan file.
3. End with the confirmation line: `Plan only — no files outside .opencode/plan/
   were changed`, plus `git status --short` output. It must list only the new
   plan file (untracked) and no modification to `src/`, `configs/`, `tests/`,
   `docs/`, `full_tests/` or the mirror; if other paths appear, report them and
   leave them alone. Do not `git add` the plan file — committing it is the
   user's call.
4. Hand the plan to the user for approval. Do not offer to "go ahead and
   implement" in this session; implementation is a separate, explicitly
   requested step.

## Definition of done (for the plan)

- Every step of the §11.1 recipe is represented: module + exports, registry
  spec (single declaration), bilingual table rows, knob fragment (when knobs
  exist), example YAML, cache decision (spec flags), family tests.
- Every mixer-specific surface is represented: mmd diagram + taxonomy/family-map
  bump, EN + ES family annex subsection, summary-table row + census, schema-annex
  row, section 01/04 mentions EN + ES, bibliography entry, README/architecture/
  configs-README census, Sphinx rst, e2e `ATTENTIONS` **+ targeted real-data
  training smoke**, bitnet coverage, param-truth/estimator formula,
  `ft-diagram.js` CATS/SUBSTRUCT (+STYLES when a new family), `index.html`
  lists, snapshot regeneration, PDF rebuild, website sync, commit/push.
- Every row of the file-by-file table names the file and the concrete content to
  add, with anchors into the current repo (path + line or quoted snippet).
- The paper extract carries the exact equations, the parameter-count formula,
  the knob table and the cache-class analysis.
- No command that mutates the repository was executed; no file outside
  `.opencode/plan/` was written.
