# Plan: new command `/add-mixer` (`.opencode/command/add-mixer.md`)

> Originally drafted in `/home/erickfmm/.opencode/plan/` (plan-mode write
> restriction); now moved into the repo. Updated after user review to make the
> e2e smoke **specific to the new mixer and run against real data with actual
> training** (`run_e2e.py --category attn --only <name>`, `--realonall`
> default-on).

**Objective.** Create one new repo file — the plan-mode command
`.opencode/command/add-mixer.md` — modelled on `.opencode/command/add-optimizer.md`
but for **sequence mixers**. Given an arXiv/alphaXiv URL, the command reads the
paper, explores the repo and emits an implementation-ready plan
(`.opencode/plan/add-mixer-<arxiv-id>.md`) covering code, registry, schema,
KV-cache decision, tests, mmd diagrams, bilingual paper annexes by family,
README/architecture, Sphinx, PDF rebuilds, website sync (including
`ft-diagram.js` and `ft-param-estimator.js`) and git. The command plans; it never
implements.

## 1. Confirmed decisions (user-approved)

| # | Decision |
|---|---|
| D-C1 | Command file: `.opencode/command/add-mixer.md`, front matter `agent: plan`, `subtask: false`. Output plan: `.opencode/plan/add-mixer-<arxiv-id>.md` in the repo. |
| D-C2 | **KV-cache decision gate per paper**: the command always plans a `CacheSpec`; it plans the full incremental branch + parity tests only when the paper's decode form is simple (classic KV / latent KV / fixed state / ring), otherwise `supported=True` with the D5 fallback (like `nsa_attn`/`msa_attn` v1) and the incremental work is listed as an optional follow-up phase requiring explicit user approval. |
| D-C3 | **Example + optional preset**: the command always plans `configs/examples/<family>_<mixer>_adamw.yaml`; when the paper describes a reference configuration, the command *asks the user* (question tool) whether to also draft a full `configs/<name>.yaml` preset (like `configs/kimi_k3.yaml`). |
| D-C4 | Sphinx API docs are in scope (per-mixer `.rst` files exist, e.g. `docs/source/api/model/attention/gated/kda_attn.rst`). |

## 2. Mixer extension surface — verified census (anchors)

Everything below was verified in the repo on 2026-11-06. Line numbers drift; the
command must re-locate by grep when it runs. Current census: **45 enum values,
9 families** (`docs/architecture.md` line 350; `annex-summary-tables.tex` line 4).

### 2.1 Code

| Surface | Anchor | What changes |
|---|---|---|
| Module | `src/model/attention/<family>/<name>.py` (families `gated/ latent/ sparse/` or top level, e.g. `sigmoid.py`, `engram.py`) | Implementation. Constructor `(config, pos_encoder=...)` (recipe §11.1 step 1, `docs/architecture.md:881`). Module must tolerate `mixer.use_fixed_softmax` being set externally (`hybrid_layer.py:243`) and read its `use_pe` flag via `getattr(config, "<name>_use_pe", …)` (`gated/kda_attn.py:142`). |
| Exports | `attention/<family>/__init__.py` + `attention/__init__.py` (`__all__`) | Import + name (pattern: `gated/__init__.py`). |
| Registry | `src/model/hybrid_layer.py` `mixer_registry` dict (lines 183–228) | `"<name>_attn": <Class>`. Eval-only sparse mixers also join `TRAINING_FREE_LAYERS` (line 121) and implement `dense_forward` (surrogate at line 340). |
| Component spec | `src/registry/builtin/attention.py` | `MixerSpec` with name, builder, `params_schema`, defaults, capabilities and bilingual docs. |
| Flatten | *(removed)* | The config flattener and `from_flat` no longer exist; knobs live in `model.attention.<mixer>`. |
| Cache | `src/model/cache/capabilities.py` | `CACHE_SPECS["<name>_attn"]` — **one entry per enum value, mandatory** (doc says 45 total). `IMPLEMENTED_MIXERS` (lines 68–103) only when the incremental branch is implemented + parity-tested. Compatibility classes (i)–(vii) documented lines 22–29. |

### 2.2 Schema (the contract — `additionalProperties: false`)

| Surface | Anchor | What changes |
|---|---|---|
| Enum + tables | `src/schema/_model/_dims.yaml` `layer_pattern.items.enum` (lines 330–375) **and both** `description` (EN, 219–261) / `description_es` (ES, 275–317) tables. `tests/test_schema_consistency.py::test_layer_pattern_enum_matches_documentation` (line 66) enforces enum ↔ both tables. | One enum entry + one row per table, with `(paper: arxiv:<id>)` reference. |
| Knob file | `src/schema/_model/_attention_<mixer>.yaml` (template `_attention_swa.yaml`: `title`/`title_es`, bilingual `description` with **Examples explained** + **References**, `default`, `examples`) | New file when the mixer has knobs. |
| `$ref` wiring | `src/schema/_model.yaml` `model.attention.properties` (property-level refs, lines 71–107) + the mixer name list inside the attention block `description`/`description_es` (lines ~53/63) | `$ref: "_model/_attention_<mixer>.yaml#/properties/<subkey>"` + name appended to both prose lists. |
| Per-mixer PE | `src/schema/_model/_positional_encoding.yaml` `per_mixer_use_pe` map (lines ~395–430) | `<name>: {type: boolean, default: …}` — `true` for softmax-attention-like, `false` for recurrent/decay mixers. |
| Defaults parity | `tests/test_schema_consistency.py::test_schema_defaults_match_dataclass` (line 229) | Schema defaults must equal `FrankensteinModelConfig` defaults. |

### 2.3 Tests, e2e, param gates

| Surface | Anchor |
|---|---|
| Enum expectation set | `tests/test_schema_attention_layers.py` lines 18–64 |
| Family tests | `tests/test_gated_attention.py` / `test_sparse_attention.py` / `test_latent_attention.py` / `test_attention_modules.py` / `test_attention_refactor.py` (recipe §11.1 step 8) |
| BitNet coverage | `tests/test_bitnet_coverage.py` mixer list (~line 45) — BitNet is on by default |
| KV parity | `tests/test_kv_cache_mixers.py` — parametrizes every mixer in `IMPLEMENTED_MIXERS` (full forward vs prefill+decode) |
| e2e | `full_tests/run_e2e.py` `ATTENTIONS` list (lines 224–238; training-free mixers excluded by the header comment). Joining the list **auto-generates** the combos `attn_single_<mixer>` (`make_single_attention_configs`, 1-layer solo) and `attn_<a>__<b>` (`make_attention_pair_configs`, all pairs). Smoke flags: `--category attn` (choices at line 1636), `--only PATTERN` (substring combo filter, applied before `--limit`), `--realonall` **default ON** — every combo trains on the real Borges corpus, not the synthetic toy one (`--no-realonall` = offline fallback), `--list-combos` (ids only), `--max-samples/--max-epochs/--batch-size/--timeout` |
| Param truth | `full_tests/param_count_check.py` → `full_tests/param_truth.json` (regenerates; presets counted via real engine) |

### 2.4 Figures (mermaid `.mmd` → pdf + svg)

- Per-mixer diagram: `docs/paper/figures/mixers/<short-name>.mmd` — house style per
  `mixers/kda.mmd`: `%%{init: …}%%` neutral theme, `flowchart TD`, tensor shapes in
  labels (`[B,n,H]`, `[dk,dv]`), labelled edges (`-- "delta rule" -->`), recurrence
  as dotted `-. "next" .->`, equations inside node labels.
- Build: `docs/paper/figures/build-figures.sh [pdf|svg|both]` — auto-globs all
  `.mmd` (no script edit needed); width 1400px for `mixers/*`; needs Chrome via
  auto-detect or `PUPPETEER_EXECUTABLE`; `pnpm dlx @mermaid-js/mermaid-cli`.
  SVGs use `web-theme.css`. Outputs (`.pdf` + `.svg`) are versioned in the main repo.
- Taxonomy: `docs/paper/figures/mixer-taxonomy.mmd` — one chain per family with
  counts (e.g. `g["Gated (8)"] --> …`), must be bumped.
- Family maps: `dense-map.mmd` / `latent-map.mmd` / `sparse-map.mmd` /
  `recurrent-map.mmd` — add a node when the mixer belongs to that family.

### 2.5 Paper EN (`docs/paper/`) and ES (`docs/paper-es/`)

- Family annexes (grep the closest peer's escaped name, e.g. `kda\_attn`, to
  confirm): `annex-2-recurrent-attention` (dense+recurrent), `annex-3-latent-attention`,
  `annex-4-sparse-attention`, `annex-5-gated-attention`, `annex-13-ssog`,
  `annex-14-fast-weight`, `annex-17-conformer`. ES counterparts:
  `annex-2-recurrent-attention`, `annex-3-latent-attention`, `annex-4-sparse-attention`,
  `annex-5-gated-attention`, `annex-13-ssog`, `annex-14-atencion-peso-rapido`,
  `annex-17-conformer`.
- Mixer subsection template (from `annex-5` KDA, lines 494–558):
  `\subsection{N. <Name>}` (numbered after the annex's last) →
  `\begin{figure}[H] \fitfigure{\includegraphics{figures/mixers/<name>.pdf}}` +
  `\label{fig:mixer-<name>}` → `\subsubsection{Mathematical Foundation}` (paper
  equations) → `algorithm` float (recurrent/parallel reference form) →
  `\subsubsection{Empirical Performance and Trade-offs}` + `tabularx`
  strength/limitation table + `\citep{<key>}`. Some annexes close with a
  "Unified … Principle" enumerate that lists every member (annex-5 lines 560–568) —
  extend it. ES figures live in `../paper/figures/` (shared).
- `sections/04-architecture-taxonomy.tex` — family subsections at lines 4 / 27 / 44 /
  54 / 64; add the mixer to its family paragraph + counts. ES counterpart
  `sections/04-*.tex`.
- `appendices/annex-summary-tables.tex` — `tab:all-mixers` family blocks with count
  headers (lines 28–94) + row (Name/Family/Training/Inference/Key
  Characteristic/`\citep`); intro census sentence line 4. ES counterpart.
- `appendices/annex-schema.tex` — mixer rows (~lines 238–263). ES counterpart.
- Bibliography: `docs/bibliography/attention_types.bib`, key style
  `<firstauthor>_<shortname>_<year>` (e.g. `kimi_kda_2025`, line 254).
- Rebuild: `pdflatex → bibtex → pdflatex ×2` per paper; clean
  `*.aux *.log *.out *.toc *.blg`. (This machine currently reports `pdflatex=no`;
  fallback documented in the command's checklist.)

### 2.6 Docs & README

- `README.md` "Mixer Categories" table (lines 126–141) + the ⚠️ eval-only note
  (line 139) when applicable.
- `docs/architecture.md` §7.4 census (line 350) + table rows (353–399) +
  `TRAINING_FREE_LAYERS` note (401–405) + §7.13 cacheable list (~line 629).
- `configs/README.md` family block lists (~line 270).
- Sphinx: `docs/source/api/model/attention/<family>/<name>.rst` +
  `<family>/index.rst` toctree; rebuild `bash docs/build_docs.sh`
  (also refreshes `docs/source/_static/papers/paper-{en,es}.html` copies).

### 2.7 Website mirror (site-only files + `/sync-website`)

- `/sync-website` file map copies: `src/schema.yaml` + `src/schema/**`,
  `configs/**` + READMEs, `docs/paper{,-es}/**` (tex/bbl/pdf/figures),
  `docs/bibliography/**`, `docs/architecture.md`, streamlit GUI files.
- **Site-only edits in `erickfmm.github.io/frankenstein-transformer/`**:
  - `ft-diagram.js`: `CATS["<name>_attn"] = { c: <cat>, l: <label> }` (categories:
    attn/recur/ssm/ode/sparse/eval/field/fastw), a `SUBSTRUCT["<name>_attn"]`
    per-mixer tensor-flow template (or a string alias to a peer's template, e.g.
    `gqla_attn: 'mla_attn'`), and a new `STYLES`/`STYLES_PRINT` class only for a
    brand-new category.
  - `ft-param-estimator.js`: `mixerParams()` switch case with the **exact**
    per-module parameter formula (bias-aware, knob-aware; see the `kda_attn` case
    `5*H*H + (H*nH+nH) + 2*H + 2*H`), verified against regenerated
    `full_tests/param_truth.json` via `full_tests/param_estimate_check.mjs … --tol 0.01`.
  - `index.html`: grep a peer mixer name; add entries wherever lists are hardcoded.
- Figures: `./build-figures.sh svg` then copy all `*.svg` to the mirror.
- Paper tab HTML: `flatten_tex.py --web` + `pandoc` + `postprocess_html.py` (en/es).
- Validation Gate: `param_count_check.py` → `0 failures`; estimator `ALL PASS`;
  JSON round-trip; CLI parse probes; state-script probes (`schema-refs`,
  `example-yaml`, `cli-flags`, …) all PASS. Mirror stays uncommitted until the
  commit phase.

## 3. Command file draft — APPLIED as `.opencode/command/add-mixer.md`

> **Status:** this draft was applied verbatim to `.opencode/command/add-mixer.md`.
> That file is now the canonical text — it additionally carries the post-review
> e2e deltas (targeted `--only <name>` smoke, default-on `--realonall`
> real-data training, `--list-combos` enumeration, `--no-realonall` fallback,
> sizing knobs, and the e2e rows in the checklist/verification matrix). The
> fenced draft below is kept as the reviewed baseline; where it differs, the
> command file wins.

Baseline draft (pre-e2e-deltas, for reference):

````markdown
---
description: Plan only (no implementation) the addition of a new sequence mixer from an arXiv/alphaXiv paper — read the paper, explore the repo and emit a complete file-by-file plan covering implementation, registry, schema, KV-cache decision, tests, mermaid figures, bilingual paper annexes, PDF rebuild, website sync (ft-diagram.js, ft-param-estimator.js) and git
agent: plan
subtask: false
---
Produce a **complete, implementation-ready plan** for adding one new sequence
mixer to frankenstein-transformer — and nothing else. This command runs in
**plan mode**: it reads the paper, explores the repository and writes the plan
to `.opencode/plan/add-mixer-<arxiv-id>.md`. It does **not** create, edit or
delete code, schema, presets, tests, figures, docs, PDFs, website-mirror files
or git history.

The canonical recipe is `docs/architecture.md` §11.1 ("Add an attention mixer")
and the **Mixer** row of `AGENTS.md`. The plan must cover every step of that
recipe plus paper reading, KV-cache decision, mmd figures, bilingual annexes,
summary tables, census bumps, Sphinx, e2e smoke, param gates, PDF compilation,
website sync and git — in enough detail that the implementer can execute it top
to bottom without re-deriving anything from the paper.

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
- **Hyperparameters/knobs**: name, type, default, range, constraints — these
  become the `model.attention.<subkey>` block (schema + `_ATTENTION_MIXER_RENAMES`
  + dataclass fields). "No knobs" is a valid outcome (e.g. `kda_attn`).
- **State & cache**: recurrent state / KV / latent / ring buffers (shape, count),
  per-step decode cost, and which cache compatibility class (i)–(vii) of
  `src/model/cache/capabilities.py` it matches, if any.
- **Complexity**: training and inference Big-O in `n, d, w, r, k` for
  `tab:all-mixers`.
- **Trainability**: does the paper define a trainable from-scratch form, or is it
  an inference-only/training-free method (→ `TRAINING_FREE_LAYERS` dense surrogate
  path like `fasa_attn`/`sparge_attn`, hard rule 5)?
- **Positional encoding**: does the mixer consume positional information?
  Determines the `per_mixer_use_pe` default (`true` softmax-attention-like,
  `false` recurrent/decay-like).
- **Mode constraints**: encoder, decoder, or both; causal formulation present?
  Grid/vision-only (like `ssog_attn`)? Audio-compatible?
- **BibTeX data**: authors, year, title, arXiv ID.

**Decision gate.** If the paper is not a sequence-mixing method, duplicates a
mixer already in the `mixer_registry` of `src/model/hybrid_layer.py`, or is
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
  (e.g. `gated/kda_attn.py` for a gated delta-rule mixer, `latent/cca_attn.py`
  for a latent+conv mixer, `sparse/swa_attn.py` for a masked-window mixer) and
  note exactly which parts will be mirrored (constructor signature, `use_pe`
  handling, `use_fixed_softmax` dispatch via `numerics.module_softmax`, decoder
  masking, `forward`/cached call shapes S1/S2 per §7.13).
- `src/model/hybrid_layer.py` — `mixer_registry` (lines ~183–228),
  `TRAINING_FREE_LAYERS` (~121), the dense surrogate `_training_free_dense_forward`
  (~340), `use_fixed_softmax` attach (~243).
- `src/model/attention/__init__.py` and the family `__init__.py` export pattern.
- `src/model/cache/capabilities.py` — `CacheSpec`, `CACHE_SPECS`, `IMPLEMENTED_MIXERS`,
  compatibility classes; `tests/test_kv_cache_mixers.py` parametrization.
- Schema: `src/schema/_model/_dims.yaml` (enum + **both** bilingual tables),
  `src/schema/_model.yaml` (`model.attention` `$ref` block + the mixer name lists
  in its bilingual descriptions), one `_attention_<mixer>.yaml` knob file as
  template (`_attention_swa.yaml`), `src/schema/_model/_positional_encoding.yaml`
  `use_pe.<mixer>_use_pe` map, and the mixer's `ComponentSpec` in
  `src/registry/builtin/attention.py`.
- Tests: `tests/test_schema_attention_layers.py` (expected enum set),
  `tests/test_schema_consistency.py` (`test_layer_pattern_enum_matches_documentation`,
  `test_schema_defaults_match_dataclass`, preset validation),
  `tests/test_yaml_examples.py`, the family test files, `tests/test_bitnet_coverage.py`.
- e2e + params: `full_tests/run_e2e.py` (`ATTENTIONS`, `--category attn`),
  `full_tests/param_count_check.py`, `full_tests/param_truth.json`,
  `full_tests/param_estimate_check.mjs`.
- Figures: `docs/paper/figures/mixers/<peer>.mmd` (house style),
  `mixer-taxonomy.mmd`, the family `*-map.mmd`, `build-figures.sh`.
- Paper EN/ES: the family annex for the chosen family (grep the peer's escaped
  name, e.g. `kda\_attn`, across `docs/paper{,-es}/appendices/*.tex`),
  `sections/04-architecture-taxonomy.tex` (+ ES), `annex-summary-tables.tex`
  (+ ES), `annex-schema.tex` (+ ES), `docs/bibliography/attention_types.bib`.
- Website: `.opencode/command/sync-website.md` (file map, site-only files,
  Validation Gate) and the mirror's `ft-diagram.js` (`CATS`, `SUBSTRUCT`,
  `STYLES`) + `ft-param-estimator.js` (`mixerParams`) — fetch from
  `https://raw.githubusercontent.com/erickfmm/erickfmm.github.io/main/frankenstein-transformer/`
  if the mirror is not cloned.
- Sphinx: `docs/source/api/model/attention/<family>/` peer `.rst` + `index.rst`,
  `docs/build_docs.sh`.
- Reference blueprint: `git log --oneline -3 -- <peer module>` and skim the
  commit that added the most recent mixer — the plan should mirror that scope.
- Read-only inventory to quote in the plan: current census numbers (enum count,
  per-family counts in `README.md` "Mixer Categories", `docs/architecture.md`
  §7.4 line ~350, `annex-summary-tables.tex` line ~4 + family headers,
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

- registry name `<name>_attn` (snake_case, from the paper's short name; bare
  `<name>` only for legacy top-level forms) and class name `<Name>Attention`;
- family directory `gated/ | latent/ | sparse/ |` top level, and the **paper
  family label** used by every census table (Dense/GQA/Recurrent/Sparse/Gated/
  Fast-weight/Latent/Geometric/Convolution-augmented — or a justified new one,
  which additionally costs a new `mixer-taxonomy.mmd` branch, a new family map
  figure, and a new `ft-diagram.js` STYLES class);
- closest peer module and the parts to mirror;
- knob table: schema key (`model.attention.<subkey>.<leaf>`), flat dataclass
  field, type, default, constraints, coercion, EN + ES description text — or
  "no knobs";
- `per_mixer_use_pe` default and the `<name>_use_pe` dataclass default;
- **KV-cache decision (ask the user when non-obvious)**:
  1. *simple decode form* (classic KV / latent KV / fixed state / ring / token
     tail) → plan `CacheSpec(supported=True, slot_type=…)` **and** the incremental
     branch **and** `IMPLEMENTED_MIXERS` membership **and**
     `test_kv_cache_mixers` parity;
  2. *cacheable in principle but heavy* (chunk-level statistics, bespoke kernels)
     → plan `CacheSpec(supported=True, notes=…D5 fallback…)` only, and list the
     incremental branch as an optional follow-up phase requiring user approval;
  3. *not sequence-attention over past tokens / no causal form* →
     `CacheSpec(supported=False)` (conformer/ssog precedent).
- training-free / eval-only? If yes: plan the `dense_forward` surrogate,
  `TRAINING_FREE_LAYERS` membership, the ⚠️ rows in `_dims.yaml` and `README.md`,
  and hard-rule-5 wording everywhere.
- example preset: `configs/examples/<family>_<mixer>_adamw.yaml` (encoder MLM
  baseline, `es_arch_kda_adamw.yaml` shape, `load_training_config`-clean);
- **optional full preset**: if the paper describes a reference configuration,
  ask the user whether to also draft `configs/<name>.yaml` (like
  `configs/kimi_k3.yaml`) — it joins the smoke-tested preset set either way;
- census bumps: the *actual* old numbers read from the repo (do not hardcode 45
  or 9) — global mixer count, per-family count, taxonomy labels.

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
| 1 | `src/model/attention/<family>/<name>.py` | Module citing the paper in the docstring; `from __future__ import annotations`; Google-style docstrings; CPU-safe; `(config, pos_encoder=…)` constructor; equations mapped to code lines; `use_pe` via `use_pe_for(config, "<name>", <default>)`; honor `self.use_fixed_softmax`; decoder causal path; `forward` call shape S1/S2 per §7.13; `dense_forward` only if training-free. |
| 2 | `src/model/attention/<family>/__init__.py` + `attention/__init__.py` | Import + `__all__` entry (quote surrounding lines). |
| 3 | `src/model/hybrid_layer.py` | `mixer_registry["<name>_attn"] = <Class>` (+ `TRAINING_FREE_LAYERS` when eval-only). |
| 4 | `src/registry/builtin/attention.py` | Register the `MixerSpec`; knobs are read through `mixer_params`. |
| 5 | *(removed)* | No flattener entry exists. |
| 6 | `src/schema/_model/_dims.yaml` | Enum entry (position) + one row in **both** the EN and ES tables with `(paper: arxiv:<id>)`; ⚠️ wording if training-free. |
| 7 | `src/schema/_model/_attention_<mixer>.yaml` | Knob block (only if knobs) — draft the full bilingual YAML (title/title_es, descriptions with examples + references, defaults matching the dataclass). |
| 8 | `src/schema/_model.yaml` | Property-level `$ref` under `model.attention` + the new name appended to both attention description lists. |
| 9 | `src/schema/_model/_positional_encoding.yaml` | `per_mixer_use_pe.<name>` entry with default + bilingual one-liners. |
| 10 | `src/model/cache/capabilities.py` | `CACHE_SPECS["<name>_attn"] = CacheSpec(…)` with paper-based `notes`; `IMPLEMENTED_MIXERS` + incremental branch + parity tests only under KV-cache decision 1. |
| 11 | `configs/examples/<family>_<mixer>_adamw.yaml` | Tiny example; must pass `load_training_config`. Draft the full YAML. |
| 12 | *(optional)* `configs/<name>.yaml` | Full preset if the user approved it (D-C3-style question). Draft it. |
| 13 | `full_tests/run_e2e.py` | Append the name to `ATTENTIONS` — this **auto-generates the e2e combos** `attn_single_<name>` (solo, 1 layer) and every `attn_<a>__<name>` / `attn_<name>__<b>` pair; no builder edits. Skip when training-free (list header comment; `fasa_attn`/`sparge_attn` precedent). |
| 14 | `tests/test_schema_attention_layers.py` | Add the name to the expected enum set. |
| 15 | Family test (`tests/test_<family>_attention.py` or `test_attention_modules.py`) | Test class: builds on a tiny config, forward in encoder + decoder modes, output shape, knobs honored, no NaN/Inf, BitNet on; name the methods. |
| 16 | `tests/test_bitnet_coverage.py` | Add the name to the mixer list. |
| 17 | `tests/test_kv_cache_mixers.py` | Only under KV-cache decision 1 — parity assertions and any slot fixtures. |
| 18 | `docs/paper/figures/mixers/<short>.mmd` | Draft the full mermaid diagram in house style (init header, flowchart TD, tensor-shaped labels, labelled edges, dotted recurrence). |
| 19 | `docs/paper/figures/mixer-taxonomy.mmd` | Extend the family chain + bump its count label. |
| 20 | *(family-dependent)* `docs/paper/figures/<family>-map.mmd` | Add the node when a family map exists. |
| 21 | `docs/bibliography/attention_types.bib` | Entry keyed `<firstauthor>_<shortname>_<year>`. Draft it. |
| 22 | `docs/paper/appendices/annex-<family>.tex` | `\subsection{N. <Name>}` numbered after the annex's last; figure `\fitfigure{figures/mixers/<short>.pdf}` + label; Mathematical Foundation with the paper's equations; algorithm float; Empirical Performance and Trade-offs + strength/limitation table; `\citep{<key>}`; extend the annex's closing unified-principle list if present. Draft the subsection. |
| 23 | `docs/paper/sections/04-architecture-taxonomy.tex` | Family paragraph sentence + counts. Quote the lines holding the old count. |
| 24 | `docs/paper/appendices/annex-summary-tables.tex` | `tab:all-mixers` row + family count header + intro census. Draft the row. |
| 25 | `docs/paper/appendices/annex-schema.tex` | Mixer row. Draft it. |
| 26 | `docs/paper-es/appendices/annex-<family>.tex` | Faithful Spanish translation of #22 (figures from `../paper/figures/`). |
| 27 | `docs/paper-es/sections/04-*.tex`, `annex-summary-tables.tex`, `annex-schema.tex` | Spanish counterparts of #23–#25. |
| 28 | `README.md`, `docs/architecture.md`, `configs/README.md` | Census pass: README Mixer Categories row + counts (+ ⚠️ note when eval-only); architecture §7.4 census + row (+ §7.13 cache list, + TRAINING_FREE note); configs/README family list. List every occurrence of the old counts found by grepping them. |
| 29 | `docs/source/api/model/attention/<family>/<name>.rst` + `<family>/index.rst` | `automodule` page + toctree entry. |

Hard rules to restate in the plan: the schema is the contract
(`additionalProperties: false`); `hidden_size % num_heads == 0`, `num_kv_heads`
divides `num_heads`, `len(layer_pattern) == num_layers`; inference never fails
because of the cache (D5); `fasa_attn`/`sparge_attn`-style mixers train dense and
go sparse only in eval (rule 5); the papers are the theory source of truth (D11),
so annex formulations must match the arXiv paper and `paper-es` must translate
them faithfully.

### 5. Ordered execution checklist
The implementer's happy path, as checkboxes: implementation (rows 1–13) → tests
(14–17) → figures (18–20) → docs EN/ES (21–28) → Sphinx (29) → PDFs → website
sync → commit/push. Include the exact commands to run **at implementation time**
(do not run them now):

```bash
# targeted gates
uv run --extra cpu --extra train python -m pytest \
  tests/test_schema_attention_layers.py tests/test_schema_consistency.py \
  tests/test_yaml_examples.py tests/test_attention_refactor.py \
  tests/test_<family>_attention.py tests/test_bitnet_coverage.py \
  -v --tb=short -p no:warnings
# (plus tests/test_kv_cache_mixers.py under KV-cache decision 1)

# full suite
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings

# e2e — enumerate the new mixer's combos (build-only, no training)
uv run --extra cpu --extra train python full_tests/run_e2e.py --category attn --list-combos | grep -F <name>

# e2e TARGETED SMOKE — ONLY the new mixer, ACTUAL TRAINING on REAL data:
# --realonall is the default (real Borges corpus; first run downloads it from
# the HF Hub). '--only <name>' substring-matches attn_single_<name> plus every
# pair attn_<a>__<name> / attn_<name>__<b>.
uv run --extra cpu --extra train python full_tests/run_e2e.py --category attn --only <name>

# minimal real-data variant: the solo single-layer combo only
uv run --extra cpu --extra train python full_tests/run_e2e.py --category attn --only attn_single_<name>

# offline synthetic fallback (no HF download), same selection
uv run --extra cpu --extra train python full_tests/run_e2e.py --category attn --only <name> --no-realonall
# sizing knobs when needed: --max-samples N --max-epochs 1 --batch-size 2 --timeout 1800
# GPU smoke: swap the extra for --extra cu118 --extra train.

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
render must be produced before the paper/website sync. Verification to require
after compilation: no `Fatal error`, no `Citation .* undefined`, `paper.pdf` /
`paper-es.pdf` newer than the edited `.tex`/`.bbl`, the new `\citep{<key>}`
resolves, and `git status --short docs/paper docs/paper-es` shows only intended
`.tex`/`.bbl`/`.pdf`/figure files.

### 6. Website sync (planned, not executed)
Spell out the `/sync-website` steps for this change:

1. Bootstrap/refresh the mirror and copy per its file map: `src/schema.yaml` +
   `src/schema/**`, `configs/*.yaml` + `configs/README.md`, `configs/examples/**`,
   `docs/paper/**` + `docs/paper-es/**` (tex, bbl, pdf, figures incl. the new
   `.mmd`/`.pdf`/`.svg`), `docs/bibliography/**`, `docs/architecture.md`.
2. **Site-only edits** (mirror is the source of truth for these):
   - `ft-diagram.js`: add `CATS["<name>_attn"]` and a `SUBSTRUCT["<name>_attn"]`
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
   the module, registry, schema (enum + tables + knob block + `$ref` + use_pe),
   cache decision, example (+ preset), tests, figures (mmd + rendered pdf/svg),
   docs (annexes EN/ES, summary tables, schema annex, bibliography, section 04,
   README/architecture/configs README, Sphinx rst), rebuilt PDFs; then
   `git push origin HEAD:main`.
2. Mirror: `git -C erickfmm.github.io …` commit
   `docs(site): sync <Name> mixer (<arXiv id>)`; then
   `git -C erickfmm.github.io push origin HEAD:main`.

List the exact paths to stage for each repo.

### 8. Verification matrix
Gate → command → expected result, one row per gate: targeted pytest families,
full pytest, e2e combo enumeration (`--list-combos` shows `attn_single_<name>` +
pairs), e2e **targeted real-data training** (`--category attn --only <name>` →
all selected combos PASS with finite loss on the real corpus; solo variant
`--only attn_single_<name>`), offline fallback (`--no-realonall`), figure
render, PDF checks, Sphinx build, param count (`0 failures`), estimator
(`ALL PASS` ≤1%), mirror probes, KV parity (when decision 1).

### 9. Risks, assumptions and open questions
Assumptions to re-check at implementation time (line drift, tool availability),
naming collisions, family ambiguity, task/mode restrictions (rule 6), BitNet
interactions, chunkwise-kernel simplifications vs the paper, and anything the
paper leaves ambiguous.

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

- Every step of the §11.1 recipe is represented: module + exports, registry,
  enum + both bilingual tables, knob file + `$ref` + renames + dataclass fields
  (when knobs exist), example YAML, cache decision, family tests.
- Every mixer-specific surface is represented: mmd diagram + taxonomy/family-map
  bump, EN + ES family annex subsection, summary-table row + census, schema-annex
  row, section 04 mention EN + ES, bibliography entry, README/architecture/
  configs-README census, Sphinx rst, e2e `ATTENTIONS`, bitnet coverage,
  param-truth/estimator formula, `ft-diagram.js` CATS/SUBSTRUCT (+STYLES when a
  new family), `index.html` lists, PDF rebuild, website sync, commit/push.
- Every row of the file-by-file table names the file and the concrete content to
  add, with anchors into the current repo (path + line or quoted snippet).
- The paper extract carries the exact equations, the parameter-count formula,
  the knob table and the cache-class analysis.
- No command that mutates the repository was executed; no file outside
  `.opencode/plan/` was written.
````

## 4. Implementation steps — APPLIED

1. ✅ Created `.opencode/command/add-mixer.md` (the §3 draft **plus** the
   user-requested e2e deltas: mixer-specific smoke with actual training on real
   data via `run_e2e.py --category attn --only <name>` under default
   `--realonall`, the solo-only minimal variant, the `--no-realonall` offline
   fallback, `--list-combos` enumeration, sizing knobs, and matching
   Phase-2/row-13/verification-matrix text).
2. ✅ Moved this plan document into the repo (from
   `/home/erickfmm/.opencode/plan/`, which the plan-mode runtime forced).
3. No other file changes. In particular: no registry/schema/doc edits now — those
   belong to the future plans this command will produce.

## 5. Validating the command after implementation

- Syntax/paths: re-check that every path quoted in the command exists
  (`hybrid_layer.py` registry lines, `_dims.yaml`, `capabilities.py`, annex
  names, `attention_types.bib`, `run_e2e.py` `--category attn`).
- Control run (read-only): invoke `/add-mixer <arXiv id of a recent mixer>` (e.g.
  KDA `2510.26692` or SWA `2608.28444`) and verify the emitted plan's
  file-by-file table reproduces the real touch points that mixer landed in
  (git history is the oracle: `git log --oneline --stat` for its commit).
- Guardrails: confirm the run changed nothing outside `.opencode/plan/`.
- Rejection path: run once with a non-mixer paper URL; expect the Phase-1
  rejection report and no plan file.

## 6. Out of scope (this plan)

No code, schema, preset, test, figure, paper, PDF, mirror or git change is made
now. The only artifact of this plan is this document. Implementation of the
command file itself is a separate, explicitly requested step (switch agents to
apply).
