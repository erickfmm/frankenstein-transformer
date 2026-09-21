# CLI Surface Reduction — Audit & Implementation Plan

**Decision (locked with author):** keep only **4 subcommands** — `train`, `deploy`, `infer`, `web-server`. Everything else is merged into them. The YAML schema is the single source of truth for all model/training settings; the **only** YAML-overriding CLI argument is `--device`.

> Intended final home: `<repo>/.opencode/plans/cli-surface-reduction.md` (repo convention — move it there when implementing).

---

## 1. Audit Findings

### 1.1 Current surface: 9 subcommands, ~70 flags

`src/cli.py` exposes `train`, `deploy`, `quantize`, `infer`, `sbert-train`, `sbert-infer`, `transformers-export`, `bitnet-gguf`, `web-server`.

### 1.2 Redundant subcommands

| Subcommand | Problem | Evidence |
|---|---|---|
| `quantize` | **100% redundant** — it is `deploy` with `--format quantized` hardcoded… which is already deploy's *default* format | `_run_quantize` (cli.py) just calls `deploy_main` with `--format quantized` |
| `transformers-export` | An export format, not a workflow — same shape as deploy: model + yaml + output | `transformers_export.py:570-572` (model/yaml/output only) |
| `bitnet-gguf` | Same: export format + a `--check` compat flag | `bitnet_gguf_export.py:489-492` |
| `sbert-train` | Duplicates the schema. `src/training/main.py:589` **already dispatches `task == "sbert"`** and `_build_sbert_supervisor_child_argv` (main.py:467-559) builds the child argv **from the `training.sbert` YAML block**. Presets exist (`configs/modernbert_sbert.yaml`, `modernbert_sbert_continual.yaml`). The subcommand also **bypasses schema validation** entirely | main.py:589; `_sbert.yaml` |
| `sbert-infer` | Four inference modes (similarity/search/cluster/encode) — a natural `infer --task sbert` | `inference_sbert.py:474-503` |

### 1.3 CLI flags that duplicate YAML schema keys (violations of single-source-of-truth)

All exist in `src/schema/_training.yaml`:

| CLI flag on `train` | YAML key | Schema location |
|---|---|---|
| `--batch-size` | `training.batch_size` | `_training.yaml` |
| `--model-mode` | top-level `model_class` | `_model_class.yaml` |
| `--gpu-temp-guard` / `--no-gpu-temp-guard` | `training.gpu_temp_guard_enabled` (default `true`) | `_training.yaml:781` |
| `--switch-on-thermal` / `--no-switch-on-thermal` | `training.switch_on_thermal` | `_training.yaml:805` |
| `--gpu-temp-pause-threshold-c` | `training.gpu_temp_pause_threshold_c` | `_training.yaml:833` |
| `--gpu-temp-resume-threshold-c` | `training.gpu_temp_resume_threshold_c` | `_training.yaml:845` |
| `--gpu-temp-critical-threshold-c` | `training.gpu_temp_critical_threshold_c` | `_training.yaml:857` |
| `--gpu-temp-poll-interval-seconds` | `training.gpu_temp_poll_interval_seconds` | `_training.yaml:870` |
| `--gpu-temp-checkpoint-grace-seconds` | `training.gpu_temp_checkpoint_grace_seconds` | `_training.yaml:882` |
| `--resume-from-checkpoint` | `training.resume_from_checkpoint` (also in `_sbert.yaml:211`) | `_training.yaml:894` |

`sbert-train`'s ~20 flags (`--batch_size`, `--epochs`, `--learning_rate`, `--hidden_size`, `--num_layers`, `--pooling_mode`, …) duplicate `training.sbert.*` **and** `model.dims.*`.

### 1.4 Dead / misleading flags

- `--transformers-export` is defined on `infer`, `sbert-train`, `sbert-infer` but each `_run_*` **rejects it with rc=2** — pure dead weight (and tests assert the dead behavior).
- `deploy --config` expects a **JSON** file (deploy.py:357-360) — confusing vs. YAML everywhere else, and redundant because checkpoints embed their config (`load_training_checkpoint` reads `checkpoint['config']`).
- `train --transformers-export` relies on `_latest_checkpoint_path("checkpoints")` — mtime-guessing magic for "which checkpoint to export".

### 1.5 Double argparse layer

`src/cli.py` parses args, then **rebuilds argv strings** forwarded to inner `main(argv)` parsers that re-parse them: `src/training/main.py:710`, `src/deploy/deploy.py:311`, `src/deploy/inference.py:466`, `src/sbert/train_sbert.py`, `src/sbert/inference_sbert.py:474`. Every flag is defined twice.

**Constraint:** `src/sbert/train_sbert.py`'s parser must **stay** — `src/training/main.py:_build_sbert_supervisor_child_argv` (and `src/engine.py`) spawn it as a subprocess (`-m src.sbert.train_sbert`) with argv built from YAML. Inner parsers of the other modules remain as library-level entrypoints (`python -m src.deploy.deploy …` still works); the top-level CLI simply stops duplicating their knobs. Flag pruning inside inner parsers is an optional follow-up, **not** part of this change.

---

## 2. Target CLI Surface (4 commands)

```
frankenstein-transformer train
    --config PATH | --config-name NAME [--list-configs] [--device auto|cpu|cuda|mps]

frankenstein-transformer deploy
    --checkpoint CKPT --output DIR
    [--format standard|quantized|transformers|gguf]   # default: quantized
    [--yaml YAML]        # required for --format transformers|gguf
    [--validate]         # post-export validation (standard/quantized)
    [--check]            # compatibility check only, no export (gguf)
    [--device auto|cpu|cuda|mps]

frankenstein-transformer infer
    --model PATH
    [--task mlm|sbert]                       # default: mlm
    # mlm task:
    [--text T] [--input FILE] [--output FILE] [--fp16] [--benchmark]
    # sbert task (--mode required):
    [--mode similarity|search|cluster|encode]
    [--sentence1 S] [--sentence2 S] [--query Q] [--corpus-file F] [--top-k N]
    [--sentences-file F] [--n-clusters N] [--input-file F] [--output-file F]
    [--batch-size N] [--device auto|cpu|cuda|mps]

frankenstein-transformer web-server
    [--server-port 8501] [--server-address localhost] [--server-headless] [--development-mode]
```

Notes:
- Unified hyphenated flag names on `infer` (`--corpus-file`, not `--corpus_file`); `_run_infer` translates when forwarding to `inference_sbert.main`.
- `deploy` dispatch: `standard|quantized` → `deploy_main`; `transformers` → `transformers_export_main`; `gguf` → `bitnet_gguf_main` (passes `--check` through). Missing `--yaml` with those formats → rc 2 with a clear message.
- `--device` stays everywhere: it selects the runtime environment, not a training hyperparameter.

---

## 3. Migration Mapping (for docs)

| Old | New |
|---|---|
| `train --batch-size 8` | set `training.batch_size` in YAML |
| `train --model-mode X` | set top-level `model_class` in YAML |
| `train --gpu-temp-*`, `--switch-on-thermal` | set `training.gpu_temp_*` / `training.switch_on_thermal` in YAML |
| `train --resume-from-checkpoint auto` | set `training.resume_from_checkpoint` in YAML |
| `train --transformers-export` | run `deploy --format transformers` afterwards |
| `quantize --checkpoint C --output O [--validate]` | `deploy --checkpoint C --output O [--validate]` (quantized is default) |
| `transformers-export --model M --yaml Y --output O` | `deploy --checkpoint M --yaml Y --output O --format transformers` |
| `bitnet-gguf --model M --yaml Y --output O [--check]` | `deploy --checkpoint M --yaml Y --output O --format gguf [--check]` |
| `deploy --config config.json` | not needed — checkpoint embeds its config |
| `sbert-train <20 flags>` | `train --config configs/modernbert_sbert.yaml` (all knobs live in `training.sbert`; `base_model` is top-level YAML) |
| `sbert-infer --model_path M --mode similarity …` | `infer --model M --task sbert --mode similarity …` |

---

## 4. File-by-File Changes

### 4.1 Code

- **`src/cli.py`** — the refactor:
  - Delete subparsers + `_run_*` for `quantize`, `sbert-train`, `sbert-infer`, `transformers-export`, `bitnet-gguf`.
  - Delete from `train`: `--batch-size`, `--model-mode`, all 7 gpu-temp/thermal flags, `--resume-from-checkpoint`, `--transformers-export`.
  - Delete `--transformers-export` everywhere; delete `deploy --config` (JSON).
  - Extend `deploy` with formats `transformers`/`gguf`, `--yaml`, `--check`; extend `infer` with `--task`, `--mode` + sbert args.
  - Remove now-dead helpers `_latest_checkpoint_path`, `_resolve_train_yaml_path`; `_validate_transformers_export_compatibility` / `_run_transformers_export_to_subfolder` shrink into the deploy-format dispatch.
  - Update module + `build_parser` docstrings (subcommand list).
- **`src/streamlit_gui/app.py`** — Command tab: `COMMANDS` list (lines 26-27 remove sbert entries; also quantize/transformers-export/bitnet-gguf entries), builders at 1055-1078 / 1170 / 1191-1249; deploy gains `--format` select (4 choices) + `--yaml` + `--check`; infer gains `--task` + sbert mode fields.
- **`pyproject.toml`** — unchanged (same console script).
- Inner mains (`training/main.py`, `deploy/*`, `sbert/*`) — **untouched** (subprocess contract + library entrypoints).

### 4.2 Tests

- **`tests/test_cli_parser.py`** — rewrite for the 4-command surface: defaults, device choices, deploy format choices + `--yaml`-required-for-transformers/gguf (rc 2), infer task/mode choices, web-server defaults, dispatch mocks (incl. `--format transformers` → `transformers_export_main`, `--format gguf [--check]` → `bitnet_gguf_main`), and `parse_args(["quantize", …])` now exits.
- **`tests/test_cli_gpu_temp_flags.py`** — **delete** (flags removed; YAML path is covered by config-loader/trainer tests).
- Others (`test_yaml_examples.py`, etc.) unaffected.

### 4.3 Docs (main repo)

- **`README.md`** — CLI table → 4 rows; "At a Glance" `CLI Commands: 9` → `4`; Feature Matrix `CLI subcommands | 9` → `4`; mermaid node "Deploy / Quantize" → "Deploy (quantize · export)"; add migration mapping pointer.
- **`docs/README.md`** — subcommand list (57-60), examples (153-190, 214-225), workflow section (245-247).
- **`docs/specs/cli-reference.md`** — rewrite: overview table, per-command sections, remove GPU-thermal-flag section (point to `training.gpu_temp_*` YAML keys), update troubleshooting rows.
- **`docs/specs/deployment.md`** — lines 85, 181, 237-240: `transformers-export`/`bitnet-gguf` → `deploy --format …`.
- **`docs/specs/sbert-workflows.md`** — lines 167-196: `sbert-train` → `train --config modernbert_sbert*.yaml`; `sbert-infer` → `infer --task sbert …`.
- **`docs/specs/training-safety.md`** — replace CLI thermal-flag documentation with YAML keys (verify by grep).
- **`docs/dashai-plugin-audit.md`** — lines 86, 224 (subcommand list; `transformers-export` usage).
- **`docs/transformers_compatibility.md`** — update any `transformers-export` invocations (verify by grep).
- **`AGENTS.md`** — CLI subcommand list line: `(subcommands: train, deploy, infer, web-server)`.
- **`configs/README.md`** — grep for stale command mentions.
- **Paper sources** `docs/paper*/**.tex` — subcommand enumerations (e.g. `paper-es.tex:47`, EN intro/figure 1 listing; `.opencode/plans/paper-cleanup-plan.md:54` tracks fig. 1) → `train / deploy / infer / web-server`.

### 4.4 Website mirror (gitignored clone — mirror-only edits, never synced from main)

- **`erickfmm.github.io/frankenstein-transformer/index.html`** — Command tab: remove `sbert-train`/`sbert-infer` (and quantize/transformers-export/bitnet-gguf) options (834-835), opt groups (889-898), JS branches (2503-2551); add deploy format/yaml/check and infer task/mode fields.
- **`.opencode/command/sync-website.md`** — update `cmd-config-flag` condition (line 100: `--config` now only on `train`), `sbert-mode-default` probe (101, 156-157) → "`infer --task sbert` requires `--mode`", command-validity list.
- **`.opencode/command/sync-website-state.sh`** — sbert-infer mode probe (229-233) → target the new `infer` sbert fields.
- Mirror paper copies regenerate via the sync command after the main-repo `.tex` edits.

---

## 5. Rollout Order

1. Refactor `src/cli.py` (pure argparse/dispatch change; no inner-module edits).
2. Update `src/streamlit_gui/app.py` Command tab.
3. Rewrite `tests/test_cli_parser.py`; delete `tests/test_cli_gpu_temp_flags.py`.
4. Run: `conda run -n frankenstein python -m pytest tests/test_cli_parser.py tests/test_yaml_examples.py -v`.
5. Update docs (README, docs/README, specs, AGENTS.md, paper `.tex`).
6. Mirror: edit `index.html` Command tab + sync-command probes, then run `/sync-website` validations.

## 6. Risks / Notes

- **Breaking change** for any script using the removed subcommands — intentional; the mapping table in §3 is the migration guide. No hidden aliases (clean cut, per author decision).
- `sbert-train --base-model <hf-id>` is fully covered: `base_model` is a top-level YAML key (schema `_base_model.yaml`) used by `configs/modernbert_sbert.yaml`.
- Optional follow-up (out of scope): prune duplicated flags from inner `main()` parsers (except `src/sbert/train_sbert.py`, which must keep its argv contract for the supervisor subprocess and `src/engine.py`).
- `_run_web_server`'s `str(__file__).replace("cli.py", …)` is fragile — opportunistic micro-fix to `Path(__file__).with_name(...)` while touching the file.
