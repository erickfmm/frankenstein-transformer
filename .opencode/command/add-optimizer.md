---
description: Add a new optimizer from an arXiv/alphaXiv paper — read the paper, explore + plan, implement + register, schema enum and custom fields, example YAML, tests and fast e2e smoke, bilingual docs (annexes, summary tables, bibliography, READMEs) with PDF compilation, website sync, commit and push both repos
agent: build
subtask: true
---
Add one new optimizer family to frankenstein-transformer, end to end: paper →
implementation → schema → tests → bilingual docs → website mirror → commit and push.

The canonical recipe is `docs/architecture.md` §11.2 ("Add an optimizer") and the
**Optimizer** row of `AGENTS.md`. This command wraps that recipe with paper reading,
documentation, PDF compilation, e2e smoke, website sync and git. The order below is
a sensible default and may vary per paper — but every phase must be completed.

## Argument

`$ARGUMENTS` must contain an arXiv ID or URL of the optimizer paper. Accepted shapes:

- `2610.01395`, `arXiv:2610.01395`
- `https://arxiv.org/abs/2610.01395`
- `https://www.alphaxiv.org/abs/2610.01395` (or `/overview`, `/html`, …)

Normalize to the bare arXiv ID. If `$ARGUMENTS` is empty or no ID can be extracted,
STOP and ask the user for the paper link.

## Workflow

### 1. Read the paper

Fetch the paper, in order of preference:

1. alphaXiv/arXiv MCP tools through `execute` (`tools.arxiv.download_paper` +
   `read_paper`, or the `tools.alphaxiv` paper tools) — they render the full text.
2. Fallback: `webfetch` on `https://arxiv.org/abs/<id>` (metadata + abstract) and,
   if needed, `https://arxiv.org/pdf/<id>`.

Extract and record (everything downstream depends on it):

- **Update rule**: every equation — moment updates, bias correction, preconditioner,
  parameter update, weight-decay handling — in LaTeX, exactly as in the paper.
- **Algorithm/pseudocode** and any warmup/schedule interactions.
- **Hyperparameters**: name, type, default, range, global vs per-parameter-group.
  These become the `<name>-<suffix>` keys.
- **State buffers** (count/shape) and **per-step cost** — needed for summary tables.
- **Category**: which existing family bucket it belongs to (`docs/architecture.md`
  §8.1) or whether it needs a new one.
- **BibTeX data**: authors, year, title, arXiv ID/DOI.

**Decision gate.** Stop and report instead of implementing if: the paper is not an
optimizer method; it duplicates a family already in `factory.OPTIMIZER_REGISTRY`;
or the algorithm is underspecified (missing equations/defaults that cannot be
recovered). If it proposes several optimizers, implement the main one and note the
rest; ask the user only if the choice is genuinely ambiguous.

### 2. Explore and plan

Read before writing anything:

- `AGENTS.md` — **Optimizer** extension row, hard rules 7 and 9, gates list.
- `docs/architecture.md` §8.1 (family table, prefix contract) and §11.2 (recipe).
- Peers in `src/model/optimizer/`: `anon.py` (single-knob adaptive),
  `apollo.py` / `q_apollo.py` (projection + extra knobs), `muon.py` (matrix
  family), and `base.py` (`GROUP_NAMES`, `extract_prefixed_parameters`,
  `to_float`/`to_int`/`to_betas` coercion, `ensure_no_unknown_parameters`).
- `src/model/optimizer/factory.py`, `src/model/optimizer/__init__.py`.
- `src/schema/_optimizer.yaml`, `src/schema/_training.yaml` (`allOf` if/then),
  `src/schema/_sbert.yaml` (`optimizer_kwargs` description).
- Tests: `tests/test_optimizer_factory.py`, `tests/test_optimizer_base.py`,
  `tests/test_schema_consistency.py`, `tests/test_yaml_examples.py`.
- Doc targets: `docs/paper/appendices/annex-1-optimizers.tex`,
  `docs/paper/appendices/annex-summary-tables.tex`,
  `docs/paper/sections/05-optimizer-families.tex` and the three `paper-es`
  counterparts, `docs/bibliography/optimizers.bib`, `configs/README.md`,
  `full_tests/run_e2e.py` (`OPTIMIZERS` list).

Decide, then print a **file-by-file plan table** (file → change) before writing:

- registry name `<name>` (snake_case, from the paper's short name) and class name,
- family category for the docs tables,
- specific global suffixes with defaults and coercion helpers,
- `supports_shared_per_group_suffixes` (the norm is `true`).

### 3. Implement (in order)

| # | File | Change |
|---|---|---|
| 1 | `src/model/optimizer/<name>.py` | New `torch.optim.Optimizer` subclass citing the paper in the module docstring; mirror a peer's structure (defaults, `step()`, param-group handling). `from __future__ import annotations`, Google-style docstrings, CPU-safe (no `.cuda()`). |
| 2 | `src/model/optimizer/__init__.py` | Import + `__all__`. |
| 3 | `src/model/optimizer/factory.py` | Import; `OPTIMIZER_REGISTRY["<name>"]`; `OPTIMIZER_ALLOWED_KEYS["<name>"] = _COMMON_PER_GROUP_KEYS \| {…}` (empty suffix set if none). |
| 4 | `src/schema/_optimizer.yaml` | (a) `optimizer_class.enum`; (b) one row in **both** the EN and ES use-case tables; (c) one bullet in the EN + ES "Optimizer-specific suffixes" lists of `parameters.description*`; (d) `x_parameter_reference.by_optimizer.<name>` with `prefix`, `supports_shared_per_group_suffixes`, `specific_global_suffixes`. |
| 5 | `src/schema/_training.yaml` | Append one `if/then` block: `const: <name>` → `patternProperties: {^<name>-: …}`. Exactly one per optimizer (parity test). |
| 6 | `src/schema/_sbert.yaml` | Extend the `optimizer_kwargs` description (EN + ES) with the new suffixes when it introduces new ones. |
| 7 | `configs/examples/…_<name>.yaml` | Tiny example (`es_arch_<topic>_<name>.yaml` like `es_arch_hybrid_turbo_muon.yaml`, or `causal_lm_<name>.yaml` for decoder-only families); must pass `load_training_config`. |
| 8 | `configs/README.md` | Optimizer list, per-optimizer suffix list, example keys. |
| 9 | `full_tests/run_e2e.py` | Append to the `OPTIMIZERS` list (+ documented specific globals in the param map if any). |

Hard rules: the schema is the contract (`additionalProperties: false`); optimizer
keys are prefixed `<name>-<group>_<param>` / `<name>-<param>`; never read an
undocumented key.

### 4. Tests and fast e2e smoke (CPU-first)

1. Extend `tests/test_optimizer_factory.py`: add `<name>` to
   `test_registry_contains_all_required` plus a small test class (builds without
   error, `step()` runs on a tiny tensor, custom suffix parsed — see the sophia /
   Q-APOLLO classes for the pattern).
2. Targeted gates:

   ```bash
   uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py -v --tb=short -p no:warnings
   ```

3. Full suite:

   ```bash
   uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings
   ```

4. Fast e2e smoke (real CLI, tiny synthetic data, CPU):

   ```bash
   uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --limit 3
   ```

   Raise `--limit` if the new optimizer is not reached.

### 5. Documentation

Update **every** place that mentions the optimizer census — grep the old number
(e.g. `grep -rniE "23 optimi|twenty-(three|four) optimi" README.md docs/`) and bump
consistently in: `README.md` ("23 optimizer families" in the mermaid diagram),
`docs/architecture.md` (§1 table, §8.1 heading + family table), both
`05-…-families…tex` sections (written-out counts + family sentence), and both
`annex-summary-tables.tex` intro lines.

| File | Change |
|---|---|
| `docs/bibliography/optimizers.bib` | New entry, key `<firstauthor>_<shortname>_<year>`, matching the existing Zotero arXiv style. |
| `docs/paper/appendices/annex-1-optimizers.tex` | `\paragraph{<Name>.}` in the right subsection: figure (only if you can produce `docs/paper/figures/optimizers/<name>.pdf` in the same vector style — otherwise omit), equations from the paper, `algorithm` float, Pros/Cons, `\citep{<key>}`. |
| `docs/paper/appendices/annex-summary-tables.tex` | Row in `tab:all-optimizers` under the correct category header: family, state buffers, per-step cost, key hyperparams, `\citep{<key>}`. |
| `docs/paper/sections/05-optimizer-families.tex` | Family sentence + counts. |
| `docs/paper-es/appendices/annex-1-optimizadores.tex` | Faithful Spanish translation (figures live in `../paper/figures/optimizers/`). |
| `docs/paper-es/appendices/annex-summary-tables.tex` | Spanish summary row + counts. |
| `docs/paper-es/sections/05-familias-optimizadores.tex` | Family sentence + counts. |
| `docs/architecture.md`, `README.md`, `configs/README.md` | Counts + family table + suffix docs. |

The papers are the theory source of truth (`AGENTS.md` D11): annex formulations
must match the arXiv paper, translated faithfully for `paper-es`.

### 5b. Compile the PDFs (both languages)

The committed `paper.bbl` / `paper-es.bbl` must be regenerated so the new citation
resolves, and the committed `paper.pdf` / `paper-es.pdf` must reflect the new annex
and tables. Check tools first (`command -v pdflatex bibtex`), then, **from the
repository root**:

```bash
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

Verify before moving on:

- Both runs end without fatal errors (`grep -l "Fatal error" *.log` finds nothing;
  undefined-citation warnings (`Citation .* undefined`) mean bibtex was skipped or
  failed — rerun the sequence).
- `paper.pdf` / `paper-es.pdf` mtimes are newer than the edited `.tex`/`.bbl`.
- The new `\citep{<key>}` resolves (no `Warning--I didn't find a database entry`
  in the bibtex output) and the new annex/table rows appear in the PDFs.
- No aux artifacts remain (`git status --short docs/paper docs/paper-es` shows only
  the intended `.tex`, `.bbl`, `.pdf` changes).

If `pdflatex`/`bibtex` are unavailable on this machine, skip the compilation, leave
the sources updated, and note it in the final report — the website sync will copy
the prebuilt PDFs only, so the rebuild must happen later on a TeX machine before
the next release.

### 6. Website sync

Apply the `/sync-website` workflow to the current state
(`bash .opencode/command/sync-website-state.sh`), which for this change means:

1. Copy the drift into `erickfmm.github.io/frankenstein-transformer/`:
   `src/schema.yaml` + `src/schema/**` (enum + parameter reference),
   `configs/examples/**` (new example) + `configs/README.md`,
   `docs/paper/**` (tex, bbl, pdf, figures), `docs/paper-es/**`,
   `docs/bibliography/**`, `docs/architecture.md`.
2. Rebuild the paper-tab HTML (`flatten_tex.py` + `pandoc`) and refresh the
   top-level PDF copies (`paper/en.pdf`, `paper/es.pdf`) per that command.
3. Validation Gate (all must pass; fix root causes, never the probes):
   - `uv run --extra cpu --extra train python full_tests/param_count_check.py` →
     `0 failures` — regenerates `full_tests/param_truth.json`; a new example YAML
     changes the preset census even though the optimizer itself does not change
     parameter counts,
   - `node full_tests/param_estimate_check.mjs erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js full_tests/param_truth.json --tol 0.01`,
   - state-script probes `schema-refs`, `example-yaml`, `cli-flags`, estimator —
     all PASS.
4. Leave the mirror uncommitted until step 7.

### 7. Commit and push (both repos, separately, on `main`)

Follow `/make-a-commit` conventions (English, conventional tag, structured body;
never stage `*.aux *.log *.out *.toc *.blg` or caches; `param_truth.json` changes
are intentional):

1. Main repo: `feat(optimizer): add <Name> (<arXiv id>)` — body covering
   implementation notes, new suffixes, schema/registry changes, docs (annexes
   EN/ES, summary tables, bibliography, counts, rebuilt PDFs), example, tests.
   Then `git push origin HEAD:main`.
2. Mirror: `git -C erickfmm.github.io add …` the synced schema/examples/paper/
   bibliography/spec + rebuilt HTML/PDFs; commit
   `docs(site): sync <Name> optimizer (<arXiv id>)`; push
   `git -C erickfmm.github.io push origin HEAD:main`.

Finish with a report: one-paragraph paper summary, files touched per phase, PDF
compilation result, gate results (pytest summary, e2e smoke, param/estimator
numbers), sync drift applied, commit hashes.

## Guardrails

- CPU-first for everything except real training: always
  `uv run --extra cpu --extra train …`; tests must never require CUDA.
- Keep enum, `by_optimizer`, the `if/then` rule, the registry entry and
  `OPTIMIZER_ALLOWED_KEYS` in sync — `tests/test_schema_consistency.py`
  cross-checks them and CI fails on mismatch.
- Update EN and ES docs in lockstep; the description tables and annexes are
  scanned or cross-referenced by tests and the site.
- Compile both PDFs before the website sync; never commit LaTeX aux artifacts.
- If any gate fails, fix the root cause before continuing; never weaken tests,
  probes or tolerances.
- Ask the user before any destructive or out-of-scope change (deleting files,
  renaming existing families, touching unrelated drift).
