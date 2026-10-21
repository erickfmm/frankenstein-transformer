# AGENTS.md

- **Project**: `frankenstein-transformer` — schema-first, config-driven toolkit for transformer experiments (mixers, optimizers, norms, positional encodings, model classes).
- **Canonical implementation spec**: [`docs/architecture.md`](docs/architecture.md). Read it before any non-trivial change.
- **Theory source of truth**: the LaTeX papers under `docs/paper/` and `docs/paper-es/`. The markdown docs never override them.
- **Removed**: `docs/specs/` no longer exists. Do not recreate it; put implementation/spec content in `docs/architecture.md`.

## Environment & commands (CPU-first)

- **Develop and test on CPU.** Use `--extra cpu` for tests, schema/config work, docs, and CI reproduction. Tests must never require CUDA; GPU code is guarded by `TORCH_AVAILABLE`.
  - Install: `uv sync --extra cpu --extra train`
  - Run: `uv run --extra cpu --extra train <cmd>`
  - All tests: `uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings`
  - Single test: `uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py -v`
- **GPU only for real training/inference** (Tesla P40): `uv sync --extra cu118 --extra train`, then `uv run --extra cu118 --extra train frankenstein-transformer train --config …`. The torch extras (`cpu`/`cu118`/`cu126`/`cu128`) are mutually exclusive; a bare `uv run` resyncs without extras and swaps torch — always pass the same extras.
- **CLI**: `uv run --extra cpu --extra train frankenstein-transformer {train,deploy,infer,web-server}`. `web-server` is the Streamlit YAML builder.
- **Style**: no linter/formatter configured. Match surrounding code; `from __future__ import annotations`; absolute `src.` imports; Google-style docstrings.
- **CI**: Python 3.10–3.12, CPU torch 2.6.0 first, then `requirements.txt` with torch/extra-index lines stripped, then `pip install -e . --no-deps`.

## Where things live

| Topic | Path |
|---|---|
| Schema entrypoint / modules | `src/schema.yaml`, `src/schema/**` |
| Flat model config | `src/model/config.py` (`FrankensteinModelConfig`) |
| Mixer registry + `HybridLayer` | `src/model/hybrid_layer.py` |
| Mixer implementations | `src/model/attention/{gated,latent,sparse}/`, top-level modules |
| Nested → flat config | `src/utils/config_flatten.py` |
| Optimizers + factory | `src/model/optimizer/{factory.py,<family>.py}` |
| Norms + factory | `src/model/norm/{factory.py,<impl>.py}` |
| Positional encodings | `src/model/embeddings/` |
| Activations | `src/model/activation_function/` |
| Residuals / mHC / Hyperloop | `src/model/residuals/`, `src/model/mhc.py`, `hyperloop.py` |
| Model classes | `src/model/frankenstein_{encoder,decoder,vit}.py`, `src/model/__init__.py` |
| Validation / task dispatch | `src/training/config_loader.py`, `src/training/main.py` |
| Training loop | `src/training/trainer.py` |
| Framework-neutral builder | `src/engine.py` (`build_model`) |
| Deployment / export / inference | `src/deploy/` |
| SBERT | `src/sbert/` |
| Streamlit UI | `src/streamlit_gui/app.py` |
| Presets | `configs/*.yaml`, `configs/examples/*.yaml` |
| Tests | `tests/` (46 files; see gates below) |
| Theory / citations | `docs/paper*/`, `docs/bibliography/` |
| Website mirror sync | `.opencode/command/sync-website.md` |

> `src/model/frankenstein_model.py` **does not exist** (split into the files above; a regression test fails if it is recreated).

## Hard rules

1. **Schema is the contract.** Every YAML key must exist in `src/schema/**` (`additionalProperties: false`). Never read an undocumented key.
2. **Invariants**: `hidden_size % num_heads == 0`; `num_kv_heads` divides `num_heads`; `len(layer_pattern) == num_layers`; `vocab_size` matches the tokenizer.
3. **BitNet is on by default** (`bitnet.enabled: true`). `bitnet.routers: true` requires `bitnet.enabled: true`; `bitnet.conv` requires BitNet + `embedding.conv.enabled`.
4. **Normalization**: enum `layer_norm`, `dynamic_tanh`, `derf`, `rms_norm`, `prms_norm`, `flash_norm`. `prms_norm` uses `norm.partial_ratio`; `flash_norm` composes with `norm.flashnorm_partial_ratio`.
5. **`fasa_attn` / `sparge_attn` are eval-only** (`HybridLayer.TRAINING_FREE_LAYERS`). Never put them in a training preset.
6. **Forced modes**: `frankenstein_text_decoder` → `mode: decoder`; `frankenstein_vit` → `mode: encoder`. `training.task: causal_lm` requires the decoder; vision tasks require the ViT.
7. **Optimizer keys are prefixed**: `<optimizer>-<group>_<param>` or `<optimizer>-<param>`. Groups: `embeddings`, `norms`, `attention`, `other`. A new family needs an enum value, a `by_optimizer` prefix, and a matching `if/then` rule.
8. **`training.task` is required**, one of `mlm`, `sbert`, `causal_lm`, `patch_prediction`, `classification`, `segmentation`, `text_classification`. No legacy top-level optimizer keys.
9. **All presets are smoke-tested** (`configs/*.yaml` + `configs/examples/*.yaml`). A new mixer/optimizer needs an example YAML.
10. **The activation enum is the source of truth** (43 values). GLU variants (`swiglu`/`geglu`/`reglu`) are handled by `GatedFFN`, not `get_activation()`.
11. **mHC and Mixture-of-Depths are mutually exclusive.** Do not enable both.

## Extending: files in order

| Feature | Touch (in order) |
|---|---|
| **Mixer** | `src/model/attention/<family>/<name>.py` → `attention/<family>/__init__.py` + `attention/__init__.py` → `hybrid_layer.py` registry → `_dims.yaml` enum + both description tables → optional `_attention_<mixer>.yaml` + `_model.yaml` `$ref` → `config_flatten._ATTENTION_MIXER_RENAMES` → `model/config.py` fields → `configs/examples/` → tests |
| **Optimizer** | `src/model/optimizer/<name>.py` → `optimizer/__init__.py` → `factory.py` (`OPTIMIZER_REGISTRY`, `OPTIMIZER_ALLOWED_KEYS`) → `_optimizer.yaml` enum + `by_optimizer` → `_training.yaml` prefix rule → example → `test_optimizer_factory.py` → `configs/README.md` |
| **Norm** | `src/model/norm/<name>.py` → `norm/factory.py` → `_norm.yaml` enum/params → `config_flatten.py` + `model/config.py` → `_DEFAULT_PAIRS` → `configs/examples/norm_*.yaml` → `test_common_modules.py` |
| **Positional encoding** | `src/model/embeddings/<name>.py` → `embeddings/factory.py` → `_positional_encoding.yaml` → `config_flatten._flatten_pe_parameters` → `model/config.py` → `configs/examples/pe_*.yaml` → `test_positional_encodings.py` |
| **Activation** | `src/model/activation_function/<family>.py` → `factory.py` (`ALL_ACTIVATIONS`) → `_ffn.yaml` enum/`activation_config` → `model/config.py` validation → `configs/examples/activation_*.yaml` → `test_activation_functions.py` |
| **Residual** | `src/model/residuals/<name>.py` → `residuals/factory.py` → `_residuals.yaml` → flatten/config → example → `test_residuals.py` |
| **Model class** | `src/model/frankenstein_<class>.py` → `model/__init__.py` → `_model_class.yaml` → `config_loader.py` rules → `engine.py` + `training/main.py` → `deploy/*` → `cli.py` choices → preset + example → `test_model_variants.py` |
| **Config field** | schema module → `config_flatten.py` (if nested) → `model/config.py` → `_DEFAULT_PAIRS` (if default) → example YAML → `docs/architecture.md` §7 |
| **Dataset/task** | `src/schema/_<name>.yaml` + `schema.yaml` `$ref` → `src/training/<builder>.py` → `engine.py` → `config_loader.py` + `trainer.py` loss → example → dataset tests |

## Gates (must stay green)

- `tests/test_schema_consistency.py` — presets validate, mixer enum ↔ bilingual table, optimizer prefixes ↔ `if/then`, schema defaults ↔ dataclass.
- `tests/test_yaml_examples.py` — auto-globs both config directories.
- `tests/test_schema_attention_layers.py` — mixer enum + `attention.*` sub-keys.
- `tests/test_optimizer_factory.py`, `test_optimizer_base.py` — registry/allowed keys/step.
- `tests/test_common_modules.py`, `test_positional_encodings.py`, `test_activation_functions.py`, `test_residuals.py`.
- `tests/test_naming_conventions.py` — no legacy names, no `frankenstein_model.py` shim.
- `tests/test_frankenstein_vit.py`, `test_vision_dataset.py`, `test_vision_loss.py`.

## Decisions

| # | Decision |
|---|---|
| D1 | Schema is the executable contract; `additionalProperties: false` everywhere |
| D2 | One flat `FrankensteinModelConfig` + explicit flattener (no nested runtime config) |
| D3 | Model core split into `config.py` / `hybrid_layer.py` / encoder / decoder / ViT; no `frankenstein_model.py` |
| D4 | BitNet enabled by default |
| D5 | `fasa_attn` / `sparge_attn` are eval-only |
| D6 | mHC and Mixture-of-Depths are mutually exclusive |
| D7 | CPU-first development and testing; CUDA only for real runs |
| D8 | Every preset/example is schema-smoke-tested |
| D9 | Optimizer prefixes are enumerated and cross-checked with conditional rules |
| D10 | Model classes force their modes (`decoder`, `vit`) |
| D11 | Paper = theory source of truth; `docs/architecture.md` = canonical implementation spec |
| D12 | `AGENTS.md` points to code/tests; it does not restate the spec |

## Skills

Repo-local skills live in `.agents/skills/<name>/SKILL.md`; load the matching one before specialist work.

| Skill | Use when |
|---|---|
| `pytorch-patterns` | Model, trainer, optimizer or attention code |
| `python-testing-patterns` | Writing/fixing tests or CI |
| `bash-defensive-patterns` | Shell scripts and automation |
| `developing-with-streamlit` | `src/streamlit_gui/` |
| `docstring` | Adding/updating Google-style docstrings |
| `python-executor` | Sandboxed data/script execution |
| `pandas-data-analysis`, `pandas-pro` | Dataset/analysis work |
| `research-paper-writer` | `docs/paper*` and annexes |
| `frontend-design`, `seo`, `accessibility` | Website mirror (`erickfmm.github.io/`) |

Environment-provided skills (`hf-cli`, `opencode`, `report`, `understand*`, `officecli`, `find-skills`) are available globally.

## Docs

- Implementation spec + extension recipes: [`docs/architecture.md`](docs/architecture.md).
- Presets/optimizer parameters: [`configs/README.md`](configs/README.md).
- Theory, formulations, citations: `docs/paper/`, `docs/paper-es/`, `docs/bibliography/`.
- Build API docs: `bash docs/build_docs.sh` (Sphinx → `docs/_build/html`).
