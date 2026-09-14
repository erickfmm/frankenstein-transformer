---
description: Sync main repo → erickfmm.github.io/frankenstein-transformer (schema, presets, paper) and validate estimator ≤1%, YAML/JSON, Command-tab CLI
agent: build
subtask: true
---
Sync the frankenstein-transformer project into its GitHub Pages mirror and validate the result. This workflow is **generalized**: it works from any state of (de)sync, driven by the live state report — do not assume any particular set of drifts.

Live state at invocation time (produced by the READ-ONLY script `.opencode/command/sync-website-state.sh`):

!`bash .opencode/command/sync-website-state.sh`

If the block above is empty or you need fresh state after making changes, re-run it yourself:

```bash
bash .opencode/command/sync-website-state.sh
```

## Arguments

`$ARGUMENTS` may contain these modes (any other text is free-form extra instruction from the user):

- `state-only` — analyze the report, explain what would be done, **modify nothing**.
- `skip-validate` — apply the sync but skip the Validation Gate (avoid unless the user insists).
- `no-html` — skip the pandoc HTML rebuild of the Paper tab.

## Locations

- Source of truth: this repo (the directory containing `src/`, `configs/`, `docs/`).
- Mirror: `erickfmm.github.io/frankenstein-transformer/` — a **separate git repo** (GitHub Pages, `origin=https://github.com/erickfmm/erickfmm.github.io.git`), gitignored inside this repo. All git operations there use `git -C erickfmm.github.io ...`.

## File map (main repo → mirror)

| Main repo | Mirror | Notes |
|---|---|---|
| `src/schema.yaml` | `schema.yaml` | strict JSON-Schema entry |
| `src/schema/**` | `schema/**` | `$ref` modules |
| `configs/*.yaml` + `configs/README.md` | `examples/*.yaml` + `examples/README.md` | presets; **top level only**, do not recurse here |
| `configs/examples/**` | `examples/examples/**` | full recursion, including `examples.md` |
| `docs/paper/{paper.tex,paper.bbl,paper.pdf}` | `paper/en/…` | + `sections/*.tex`, `appendices/*.tex` |
| `docs/paper-es/{paper-es.tex,paper-es.bbl,paper-es.pdf}` | `paper/es/…` | + `sections/*.tex`, `appendices/*.tex` |
| `docs/bibliography/**` | `paper/bibliography/**` | `.bib` + `.md` |
| `docs/specs/**` | `paper/specs/**` | |
| `src/streamlit_gui/{app.py,__init__.py}` | `streamlit_gui/` | |
| `src/utils/schema_loader.py` | `utils/schema_loader.py` | |

Site-only infrastructure (**never delete or overwrite**): `index.html`, `ft-param-estimator.js`, `ft-diagram.js`, `paper/flatten_tex.py`, `paper/README.md`, generated `paper/paper-en.html`, `paper/paper-es.html`, `paper/en.pdf`, `paper/es.pdf`.

## Workflow

### 0. Bootstrap the mirror repo

If the report says `WEBSITE_REPO=MISSING`:

```bash
git clone https://github.com/erickfmm/erickfmm.github.io.git erickfmm.github.io
```

(the path is already gitignored in this repo), then re-run the state script and continue.

### 1. Analyze the `[drift]` section and decide

Decision rules per status — apply them to whatever the report shows:

- `MISSING_SITE <file>` → copy main → mirror (`cp` or `mkdir -p` + `cp`; **rsync is not installed on this machine**).
- `DIFF <file>` → copy main → mirror (main repo wins: schema, presets, paper sources, specs, bibliography, `app.py`, `schema_loader.py`).
- `ONLY_SITE <file>` ending in `.yaml` under `examples/` → a preset that exists only on the website:
  1. Validate it: `conda run -n frankenstein python -c "import sys; sys.path.insert(0,'.'); from src.training.config_loader import load_training_config; load_training_config('<mirror file>')"` must succeed.
  2. If valid → **reverse-sync**: copy it into `configs/` (or `configs/examples/` when under `examples/examples/`) so CI smoke-tests it, and keep the mirror copy. This is the only case where files flow mirror → main.
  3. If invalid → **STOP and ask the user** before touching anything.
- `ONLY_SITE <file>` of any other kind → do not delete; list it in the final report and ask.
- Never delete mirror files unless the user explicitly confirms; "delete strays" is not part of the default sync.

### 2. Apply the sync

Use plain `cp` with `mkdir -p` (no rsync). Work file-by-file from the drift list so the operation is reviewable. After copying, re-run the state script: the `[drift]` section must shrink accordingly (`only_site` may stay > 0 while presets await reverse-sync decisions).

### 3. Paper artifacts

After syncing paper sources:

1. **PDFs** — if `[tools]` shows `pdflatex=yes bibtex=yes`, compile in the mirror (from `erickfmm.github.io/frankenstein-transformer/paper/`, see its README): `pdflatex → bibtex → pdflatex → pdflatex` for `en/paper.tex` and `es/paper-es.tex`, then **delete all LaTeX aux artifacts** (`*.aux *.log *.out *.toc *.blg`) from the mirror so they never get committed. If TeX is unavailable (current machine: `pdflatex=no`), copy the prebuilt PDFs from the main repo instead.
2. **Top-level copies**: `cp paper/en/paper.pdf paper/en.pdf` and `cp paper/es/paper-es.pdf paper/es.pdf` (the probes `paper-en-copy` / `paper-es-copy` check this).
3. **Paper tab HTML** (unless `no-html`) — always rebuild, pandoc is installed:

```bash
cd erickfmm.github.io/frankenstein-transformer/paper
python3 flatten_tex.py en/paper.tex    en/paper.bbl    /tmp/paper-en-flat.tex
python3 flatten_tex.py es/paper-es.tex es/paper-es.bbl /tmp/paper-es-flat.tex
pandoc /tmp/paper-en-flat.tex -f latex -t html5 --standalone --mathjax --toc --toc-depth=3 -V lang=en -o paper-en.html
pandoc /tmp/paper-es-flat.tex -f latex -t html5 --standalone --mathjax --toc --toc-depth=3 -V lang=es -o paper-es.html
```

### 4. Fix `[probes]` failures in `index.html`

The probes compare the page's Command tab and asset URLs against `pyproject.toml` and `src/cli.py`. For every `FAIL`, make a **minimal targeted edit** to `index.html`:

- `cmd-binary-name` → the generated command prefix (`let cmd = '…'`) must equal the `[project.scripts]` name from `pyproject.toml` (currently `frankenstein-transformer`).
- `github-url` → sidebar repo link must point to the real repo (`github.com/erickfmm/frankenstein-transformer`).
- `deploy-format` → `--format` `<option>` values must be a subset of the `choices` in `src/cli.py` (`quantized`, `standard`); remove invalid options and add missing ones.
- `cmd-config-flag` → the `--config` append condition (`['train', …].includes(selectedCmd)`) must list only subcommands whose parser in `src/cli.py` defines `--config` (currently `train`, `deploy`, `quantize` — **not** `infer`, `sbert-train`, `sbert-infer`).
- `sbert-mode-default` → the `--mode` select must not offer an empty `(default)` option (`sbert-infer --mode` is required; default the select to `--mode similarity`).
- `schema-refs` / `example-yaml` / `asset-base-url` failures indicate broken mirrored content or wrong `/frankenstein-transformer/…` URLs — fix the content or URL, never the probe's expectations.

Re-run the state script until all probes PASS.

### 5. Validation Gate (skip only with `skip-validate`)

Run in order; each step must pass before continuing. On failure, fix the root cause (estimator formula, mirrored file, or index.html) and repeat.

1. **Regenerate parameter ground truth** (CPU, meta device — builds every preset through the real engine; also covers presets reverse-synced in step 1):

   ```bash
   conda run -n frankenstein python full_tests/param_count_check.py
   ```

   Must print `… configs counted, 0 failures`.

2. **JS estimator ≤ 1% drift** (always pass explicit paths; the .mjs default is broken):

   ```bash
   node full_tests/param_estimate_check.mjs \
     erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js \
     full_tests/param_truth.json --tol 0.01
   ```

   Exit 0 required (`ALL PASS`). If a config exceeds tolerance, open the worst offenders printed by the script, compare against the real modules in `src/model/**`, and correct the formula in `ft-param-estimator.js`.

3. **YAML/JSON validity of the mirror**: the state script's `schema-refs` and `example-yaml` probes must PASS (schema `$ref` resolution + every `examples/**/*.yaml` through `load_training_config`). Additionally verify the JSON serialization the page produces round-trips:

   ```bash
   conda run --no-capture-output -n frankenstein python - <<'PY'
   import glob, json, sys, yaml
   bad = 0
   for p in glob.glob('erickfmm.github.io/frankenstein-transformer/examples/**/*.yaml', recursive=True):
       d = yaml.safe_load(open(p))
       if json.loads(json.dumps(d)) != d:
           print('JSON round-trip FAILED:', p); bad += 1
   print('json round-trip bad =', bad)
   sys.exit(1 if bad else 0)
   PY
   ```

4. **Command-tab CLI validity** — every command the page can generate must parse with the real parser (no torch needed):

   ```bash
   conda run --no-capture-output -n frankenstein python - <<'PY'
   import sys; sys.path.insert(0, '.')
   from src.cli import build_parser
   cmds = [
       ['train', '--config', './config_generated.yaml', '--device', 'cpu'],
       ['train', '--config-name', 'frankenstein', '--device', 'cuda'],
       ['deploy', '--config', './config_generated.yaml', '--checkpoint', './model.pt',
        '--output', './out', '--format', 'quantized', '--validate'],
       ['quantize', '--checkpoint', './model.pt', '--output', './out', '--validate'],
       ['infer', '--model', './deploy/model.pt', '--text', 'hola', '--batch-size', '8', '--benchmark'],
       ['sbert-train', '--output_dir', './out', '--batch_size', '16', '--epochs', '4', '--pooling_mode', 'mean'],
       ['sbert-infer', '--model_path', './sbert', '--mode', 'similarity',
        '--sentence1', 'a', '--sentence2', 'b'],
   ]
   for c in cmds:
       build_parser().parse_args(c)   # raises SystemExit on invalid
       print('OK', ' '.join(c))
   PY
   ```

   The state-script probes (flags, choices, subcommands, binary name) complement this: all must PASS.

### 6. Finish

- **Leave both repos uncommitted.** Never commit, never push, never stage. If the user wants to commit the mirror, point them to `/make-a-commit` (note: it commits **and pushes to main**) and let them run it themselves.
- Print a final report: decisions taken per drift class (files copied, presets reverse-synced, index.html fixes), Validation Gate results (estimator max drift %, probes), remaining `git -C erickfmm.github.io status --short` output, and any open questions.
- If anything remains unresolved (invalid site-only preset, unreachable validation), end with a clear failure summary instead of a partial success claim.

## Guardrails

- Read state first, write second; re-run the state script after each major step to verify progress.
- Minimal diffs in `index.html` — fix exactly what a probe flags; do not reformat or refactor the page.
- Never delete files without explicit user confirmation; never touch git history in either repo.
- The main repo is the source of truth for content; the mirror is the source of truth only for `index.html`, `ft-param-estimator.js`, `ft-diagram.js` and the paper build pipeline (`flatten_tex.py`, HTML outputs).
