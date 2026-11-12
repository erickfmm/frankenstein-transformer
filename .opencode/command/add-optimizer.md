---
description: Plan only (no implementation) the addition of a new optimizer from an arXiv/alphaXiv paper — read the paper, explore the repo and emit a complete file-by-file plan covering implementation, schema, tests, bilingual docs, PDF rebuild, website sync and git
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

Produce a **complete, implementation-ready plan** for adding one new optimizer family
to frankenstein-transformer — and nothing else. This command runs in **plan mode**: it
reads the paper, explores the repository and writes the plan to
`.opencode/plan/add-optimizer-<arxiv-id>.md`. It does **not** create, edit or delete code,
schema, presets, tests, docs, PDFs, website-mirror files or git history.

The canonical recipe is `docs/architecture.md` §11.1 ("Add or change a component
(any family)", ~line 1058) and the **Optimizer** row of `AGENTS.md`. The registry is
the single source of truth (D13/D14): the optimizer enum, the `by_optimizer` reference
and the `^<name>-` `if/then` rules are generated from the `OptimizerSpec` by
`src/registry/schema_assembly.py` — never hand-edited. The plan must cover every phase
of that recipe plus paper reading, documentation, PDF compilation, e2e smoke, website
sync and git, in enough detail that the implementer can execute it top to bottom
without re-deriving anything from the paper.

## Non-negotiable guardrails

This is a planning command. The prohibitions below win over any later sentence in this
document that describes an action:

| Forbidden now | Required instead |
|---|---|
| Creating/editing/deleting anything outside `.opencode/plan/` | Describe the exact change in the plan |
| `git add`, `commit`, `push`, `checkout`, `stash`, `reset` | `git status --short`, `git log`, `git diff` (read-only) |
| Running `pytest`, `full_tests/run_e2e.py`, `param_count_check.py`, `param_estimate_check.mjs` | List them as *planned verification* with expected results |
| Running `pdflatex`, `bibtex`, `pandoc`, `flatten_tex.py` | Write the exact command sequence for implementation time |
| Copying/syncing into `erickfmm.github.io/` or rebuilding the site | Enumerate the files and steps to sync later |
| `uv sync`, training, inference, `web-server` | Quote the CPU-first commands for the implementer |
| Touching `src/`, `configs/`, `tests/`, `docs/`, `full_tests/` | Read only |

If the runtime nevertheless exposes edit tools for paths outside `.opencode/plan/`,
ignore them: the deliverable is a plan, not a patch. Never "do a small fix while you are
there". If the paper implies a destructive or out-of-scope step (deleting a family,
renaming an existing optimizer, touching unrelated drift), do not perform it — flag it in
the plan as requiring explicit user approval before implementation.

## Argument

`$ARGUMENTS` must contain an arXiv ID or URL of the optimizer paper. Accepted shapes:

- `2610.01395`, `arXiv:2610.01395`
- `https://arxiv.org/abs/2610.01395`
- `https://www.alphaxiv.org/abs/2610.01395` (or `/overview`, `/html`, …)

Normalize to the bare arXiv ID and use it in the plan filename
(`.opencode/plan/add-optimizer-<arxiv-id>.md`). If `$ARGUMENTS` is empty or no ID can be
extracted, STOP and ask the user for the paper link — do not invent a topic.

## Phase 1 — Read the paper (read-only)

Fetch the paper, in order of preference:

1. alphaXiv/arXiv MCP tools through `execute` (`tools.arxiv.download_paper` +
   `read_paper`, or the `tools.alphaxiv` paper tools) — they render the full text.
2. Fallback: `webfetch` on `https://arxiv.org/abs/<id>` (metadata + abstract) and,
   if needed, `https://arxiv.org/pdf/<id>`.

Extract and record verbatim (everything downstream depends on it):

- **Update rule**: every equation — moment updates, bias correction, preconditioner,
  parameter update, weight-decay handling — in LaTeX, exactly as in the paper.
- **Algorithm/pseudocode** and any warmup/schedule interactions.
- **Hyperparameters**: name, type, default, range, global vs per-parameter-group.
  These become the `<name>-<suffix>` keys.
- **State buffers** (count/shape) and **per-step cost** — needed for summary tables.
- **Category**: which existing family bucket it belongs to (`docs/architecture.md`
  §8.1) or whether it needs a new one.
- **BibTeX data**: authors, year, title, arXiv ID/DOI.

**Decision gate.** If the paper is not an optimizer method, duplicates a family already
in the registry (`src/registry/builtin/optimizer.py` /
`registry.names(ComponentFamily.OPTIMIZER)`), or is underspecified (missing
equations/defaults that cannot be recovered), stop after Phase 1 and return a short
rejection report: verdict, evidence (quoted paper text + the existing registry entry for
duplicates), and what a usable paper would need to define. Do not write a plan. If the paper proposes several
optimizers, plan the main one and record the others under "Open questions"; ask the user
only if the choice is genuinely ambiguous.

## Phase 2 — Explore the codebase (read-only)

Read before planning anything, and cite concrete anchors (path + line) in the plan:

- `AGENTS.md` — **Optimizer** extension row, hard rules 7 and 9, gates list.
- `docs/architecture.md` §8.1 (family table, prefix contract) and §11.1 (component
  recipe).
- Peers in `src/model/optimizer/`: `anon.py` (single-knob adaptive),
  `apollo.py` / `q_apollo.py` (projection + extra knobs), `muon.py` (matrix
  family), and `base.py` (`GROUP_NAMES` ~line 19, `to_float` ~174, `to_int` ~194,
  `to_betas` ~213, `extract_prefixed_parameters` ~232,
  `ensure_no_unknown_parameters` ~257, `categorize_param_groups`).
  Pick the closest peer and note exactly which parts will be mirrored.
- `src/registry/builtin/optimizer.py` — the `OPTIMIZER_SPECS` tuple:
  `OptimizerSpec(name, builder, description/description_es, references,
  param_bindings={suffix: ParamBinding(coercion, default)}, group_policy,
  custom_build, capability flags)`; coercions come from
  `src/registry/coercions.py` (`as_float`, `as_int`, `as_betas`, `as_bool`,
  `as_lower_str`, `as_optional_int`). `_build_adafactor` is the pattern for a
  genuinely special constructor.
- `src/model/optimizer/factory.py` — `OPTIMIZER_REGISTRY` and
  `OPTIMIZER_ALLOWED_KEYS` are **derived read-only views** over the registry
  (~lines 129-132); `build_optimizer` has no per-optimizer branches (~line 186).
  There is nothing to hand-edit here.
- Schema: after the registry rollout the raw YAML no longer carries optimizer
  enums, `by_optimizer` or `if/then` blocks (the assembler injects them:
  `_inject_optimizers` ~line 139). The only optional edit is the bilingual
  use-case table row in `src/schema/_optimizer.yaml` (3-column style) and a
  bullet in the EN + ES `parameters.description*` suffix lists — pure
  documentation; validation derives from the spec.
- Tests: `tests/test_optimizer_factory.py` (`test_registry_contains_all_required`
  ~line 31, sophia / Q-APOLLO test classes as the pattern),
  `tests/test_optimizer_base.py`, `tests/test_schema_consistency.py`,
  `tests/test_yaml_examples.py`, and the registry gates
  (`tests/test_registry_core.py`, `tests/test_registry_binding.py`,
  `tests/test_schema_assembly.py`).
- Doc targets: `docs/paper/appendices/annex-9-optimizers_<bucket>.tex`
  (`adaptive`, `evolutionary`, `geometry`, `memory`, `momentum`, plus
  `tutorialintro` and `summary`), figures under
  `docs/paper/figures/optimizers/<name>.{mmd,pdf,svg}` plus the `<name>-es.*`
  twins (same header/ids/formulas, translated labels),
  `docs/paper/appendices/annex-summary-tables.tex` (`tab:all-optimizers` ~line 130,
  intro census "all 36" ~line 113), `docs/paper/sections/05-optimizer-families.tex`
  and the `paper-es` counterparts (`annex-9-optimizadores_<bucket>.tex`,
  `05-familias-optimizadores.tex`), `docs/bibliography/optimizers.bib`,
  `configs/README.md`, `configs/examples/` naming conventions,
  `full_tests/run_e2e.py` (`OPTIMIZERS` list ~line 289 and the
  `_with_optimizer` helper ~line 661, whose keys are auto-filtered by
  `OPTIMIZER_ALLOWED_KEYS`), `full_tests/param_count_check.py`,
  `.opencode/command/sync-website-state.sh`.
- Read-only inventory to quote in the plan: current optimizer count in `README.md`,
  `docs/architecture.md` §1/§8.1 (~line 875) and the two family sections; current
  example YAMLs for the chosen task; available `pdflatex`/`bibtex`
  (`command -v pdflatex bibtex`); `git status --short` (must stay clean). Today:
  36 optimizers / 8 families, 84 test files.

Do not run mutating commands to "check" anything. If a fact can only be confirmed by
running code, record it as an **assumption to verify at implementation time**.

## Phase 3 — Decide

Record these decisions explicitly, with the evidence that supports them:

- registry name `<name>` (snake_case, from the paper's short name) and class name,
- closest peer module and the parts to mirror,
- family category for the docs tables (existing bucket from `docs/architecture.md`
  §8.1, or a justified new one),
- suffix table: key (`<name>-<suffix>` / `<name>-<group>_<suffix>`), type, default,
  `ParamBinding` coercion (`as_float`/`as_int`/`as_betas`/`as_bool`/`as_lower_str`/
  `as_optional_int` or a custom callable), scope (global vs per-group),
  EN and ES description text,
- `supports_shared_per_group_suffixes` (the norm is `true`; it maps to
  `OptimizerSpec.supports_shared_per_group_suffixes`),
- example preset: task, file name, and why it is the smallest valid config,
- census bump: the *actual* old number read from the repo (do not hardcode `23`;
  today it is 36 optimizers in 8 families — verify before editing).

Print the **file-by-file plan table** (file → change) in the chat response before writing
the plan document, so the user can veto a decision before the plan is finalised.

## Phase 4 — Write the plan document

Create `.opencode/plan/add-optimizer-<arxiv-id>.md` with this skeleton. Every section is
mandatory; "n/a" is allowed only with a one-line reason.

### 1. Header
`# Plan: add <Name> optimizer (<arXiv id>)` + one-paragraph paper summary (what problem,
what mechanism, why it belongs in this repo).

### 2. Paper extract
Update rule in LaTeX, pseudocode, hyperparameter table, state buffers, per-step cost,
complexity, category, bibtex data.

### 3. Decisions
From Phase 3: names, peer, category, suffix table, `supports_shared_per_group_suffixes`,
example preset, census bump.

### 4. File-by-file plan table
One row per file, in execution order, with the *intended content*, not just the intent:

| # | File | Planned change (be concrete) |
|---|---|---|
| 1 | `src/model/optimizer/<name>.py` | `torch.optim.Optimizer` subclass citing the paper in the module docstring; `from __future__ import annotations`; Google-style docstrings; CPU-safe (no `.cuda()`); defaults dict; param-group validation; `step()` sketch with the paper's equations mapped to lines. |
| 2 | `src/model/optimizer/__init__.py` | Import + `__all__` entry (quote the surrounding lines). |
| 3 | `src/registry/builtin/optimizer.py` | One `OptimizerSpec`: `name`, `builder="src.model.optimizer.<name>:<Class>"`, `description`/`description_es`, `references`, `param_bindings` (one `ParamBinding(coercion, default)` per suffix), `group_policy`, capability flags (`closure_based`, `sbert_allowed`, `requires_grad_accum_one`, …) and `custom_build` **only** for genuinely special constructors (pattern `_build_adafactor`). This single declaration feeds the enum, `by_optimizer` and the `if/then` rule — no `factory.py` edit exists. |
| 4 | *(optional)* `src/schema/_optimizer.yaml` | (a) one row in **both** the EN and ES use-case tables (3-column style; else the assembler appends a 2-column fallback row); (b) one bullet in the EN + ES "Optimizer-specific suffixes" lists of `parameters.description*`. Pure documentation — the enum, `by_optimizer` and `if/then` are generated and must not be hand-edited. |
| 5 | *(removed)* | `src/schema/_training.yaml` needs no edit: the assembler writes `training.allOf` from the specs. |
| 6 | *(removed)* | `src/schema/_sbert.yaml` needs no edit: the SBERT enum is synced by the assembler and the suffix prose is shared. |
| 7 | `configs/examples/…_<name>.yaml` | Tiny example (`es_arch_<topic>_<name>.yaml` like `es_arch_hybrid_turbo_muon.yaml`, or `causal_lm_<name>.yaml` for decoder-only families); must pass `load_training_config`. Draft the full YAML. |
| 8 | `configs/README.md` | Optimizer list, per-optimizer suffix list, example keys — quote the surrounding format. |
| 9 | `full_tests/run_e2e.py` | Append to the `OPTIMIZERS` list (~line 289) and, only when the optimizer needs safe non-default globals, add a block in `_with_optimizer` (~line 661). Keys are auto-filtered by `OPTIMIZER_ALLOWED_KEYS`, so an unsupported key can never break the run. |
| 10 | `tests/test_optimizer_factory.py` | Add `<name>` to `test_registry_contains_all_required` plus a test class: builds without error, `step()` runs on a tiny tensor, custom suffix parsed. Name the test methods. |
| 11 | `docs/bibliography/optimizers.bib` | Entry keyed `<firstauthor>_<shortname>_<year>`, matching the existing Zotero arXiv style. Draft the entry. |
| 12 | `docs/paper/appendices/annex-9-optimizers_<bucket>.tex` | `\paragraph{<Name>.}` in the right category file (`adaptive`/`evolutionary`/`geometry`/`memory`/`momentum`): figure (only if `docs/paper/figures/optimizers/<name>.pdf` in the same vector style can be produced — otherwise omit) from `<name>.mmd` + `<name>-es.mmd`, equations from the paper, `algorithm` float, Pros/Cons, `\citep{<key>}`. Draft the paragraph. |
| 13 | `docs/paper/appendices/annex-summary-tables.tex` | Row in `tab:all-optimizers` under the correct category header: family, state buffers, per-step cost, key hyperparams, `\citep{<key>}`. Draft the row. |
| 14 | `docs/paper/sections/05-optimizer-families.tex` | Family sentence + counts. Quote the exact lines currently holding the old count. |
| 15 | `docs/paper-es/appendices/annex-9-optimizadores_<bucket>.tex` | Faithful Spanish translation of #12 (figures live in `../paper/figures/optimizers/`, using the `-es` variants). |
| 16 | `docs/paper-es/appendices/annex-summary-tables.tex` | Spanish summary row + counts. |
| 17 | `docs/paper-es/sections/05-familias-optimizadores.tex` | Family sentence + counts. |
| 18 | `docs/architecture.md`, `README.md`, `configs/README.md` | Census pass across all three (§1 table, §8.1 heading + family table, mermaid diagram, suffix docs). List every occurrence of the census number/sentence to edit, as found by grepping the old count. |
| 19 | `docs/schema.assembled.yaml` | Regenerate with `python -m src.registry.snapshot --out docs/schema.assembled.yaml` (the snapshot the mirror consumes). |

Hard rules to restate in the plan: the schema is the contract
(`additionalProperties: false`); optimizer keys are prefixed
`<name>-<group>_<param>` / `<name>-<param>`; never read an undocumented key; every
component is a declarative `ComponentSpec` and the registry is the single source of
truth (D13/D14); the papers are the theory source of truth (`AGENTS.md` D11), so annex
formulations must match the arXiv paper and `paper-es` must translate them faithfully.

### 5. Ordered execution checklist
The implementer's happy path, as checkboxes: implementation (files 1–9) → tests (10) →
docs (11–18) → PDFs → website sync → commit/push. Include the exact commands to run
**at implementation time** (do not run them now):

```bash
# targeted gates
uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py tests/test_registry_core.py tests/test_registry_binding.py tests/test_schema_assembly.py -v --tb=short -p no:warnings

# full suite
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings

# enumerate the optimizer combos (no training) and then a fast smoke
uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --list-combos | grep -F <name>
uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --only <name>
# (--realonall is the default: the sweep trains on the real Borges corpus;
#  --no-realonall is the offline fallback; --limit N caps the sweep)
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

Record the PDF preconditions and the fallback: if `pdflatex`/`bibtex` are unavailable,
leave sources updated and note that the rebuild must happen on a TeX machine before the
next release (the website sync copies prebuilt PDFs only). Verification to require after
compilation: no `Fatal error`, no `Citation .* undefined`, `paper.pdf` / `paper-es.pdf`
newer than the edited `.tex`/`.bbl`, the new `\citep{<key>}` resolves, and
`git status --short docs/paper docs/paper-es` shows only intended `.tex`/`.bbl`/`.pdf`.

### 6. Website sync (planned, not executed)
Spell out the `/sync-website` steps for this change: regenerate `docs/schema.assembled.yaml`
first, then copy it plus `src/schema/**`, `src/registry/**`, `configs/examples/**` +
`configs/README.md`, `docs/paper/**` (tex, bbl, pdf, and the **`*.svg`** figures only),
`docs/paper-es/**`, `docs/bibliography/**`, `docs/architecture.md` into
`erickfmm.github.io/frankenstein-transformer/`; rebuild the Paper-tab HTML
(`flatten_tex.py --web` + `pandoc` + `postprocess_html.py`) and refresh
`paper/en.pdf` / `paper/es.pdf`; then the Validation Gate — `param_count_check.py` →
`0 failures` (regenerates `full_tests/param_truth.json`), `param_estimate_check.mjs …
--tol 0.01`, and the state-script probes (`schema-refs`, `example-yaml`, `cli-flags`,
`paper-figures-en/es`, estimator) all PASS. Note that the mirror stays uncommitted until
the commit phase.

### 7. Commit plan (planned, not executed)
Follow `/make-a-commit` conventions (English, conventional tag, structured body; never
stage `*.aux *.log *.out *.toc *.blg` or caches; `param_truth.json` changes are
intentional):

1. Main repo: `feat(optimizer): add <Name> (<arXiv id>)` — body covering implementation
   notes, new suffixes, the registry spec, docs (annexes EN/ES, summary tables,
   bibliography, counts, rebuilt PDFs), example, tests, regenerated
   `docs/schema.assembled.yaml`; then `git push origin HEAD:main`.
2. Mirror: `git -C erickfmm.github.io …` commit
   `docs(site): sync <Name> optimizer (<arXiv id>)`; then
   `git -C erickfmm.github.io push origin HEAD:main`.

List the exact paths to stage for each repo.

### 8. Verification matrix
Gate → command → expected result, one row per gate in the original recipe (targeted
pytest, full pytest, e2e smoke, PDF checks, param count, estimator, state probes).

### 9. Risks, assumptions and open questions
Assumptions that must be re-checked at implementation time, naming collisions, mixer /
task incompatibilities (`training.task` rules), and anything the paper leaves ambiguous.

### 10. Out of scope
Explicitly: no code, schema, preset, test, doc, PDF, mirror or git change was made by
this command.

## Phase 5 — Deliver

1. Write the plan to `.opencode/plan/add-optimizer-<arxiv-id>.md`. If the plan-mode
   runtime denies that write, keep the full plan in the chat response and say so.
2. Reply with: the paper in one paragraph, the decisions (name, category, suffixes,
   example preset, census bump), the file-by-file table, and the path of the plan file.
3. End with the confirmation line: `Plan only — no files outside .opencode/plan/ were
   changed`, plus `git status --short` output. It must list only the new plan file
   (untracked) and no modification to `src/`, `configs/`, `tests/`, `docs/`,
   `full_tests/` or the mirror; if other paths appear, report them and leave them alone.
   Do not `git add` the plan file — committing it is the user's call.
4. Hand the plan to the user for approval. Do not offer to "go ahead and implement" in
   this session; implementation is a separate, explicitly requested step.

## Definition of done (for the plan)

- Every phase of the original end-to-end recipe is represented: paper extract,
  registry/implementation (single `OptimizerSpec`), generated enum + suffix reference +
  `if/then` rule (no hand edit), optional use-case table prose, example YAML, e2e list,
  tests, bibliography, EN + ES annexes, summary tables, family sections, census bump,
  snapshot regeneration, PDF rebuild, website sync, commit/push.
- Every row of the file-by-file table names the file and the concrete content to add,
  with anchors into the current repo (path + line or quoted snippet).
- No command that mutates the repository was executed; no file outside
  `.opencode/plan/` was written.
