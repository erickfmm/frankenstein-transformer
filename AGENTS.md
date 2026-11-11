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
- **GPU only for real training/inference** (Tesla P40): `uv sync --extra cu118 --extra train`, then `uv run --extra cu118 --extra train frankenstein-transformer train --config …`. The torch extras (`cpu`/`cu118`/`cu126`/`cu128`) are mutually exclusive; a bare `uv run` resyncs without extras and swaps torch — always pass the same extras. The non-torch extras (`sbert`, `web`, `wandb`) do not pin torch and are all pulled in by `--extra train`.
- **CLI**: `uv run --extra cpu --extra train frankenstein-transformer {train,deploy,infer,web-server}`. `web-server` is the Streamlit YAML builder.
- **Style**: no linter/formatter configured. Match surrounding code; `from __future__ import annotations`; absolute `src.` imports; Google-style docstrings.
- **CI**: Python 3.10–3.12, CPU torch 2.7.0 first, then `requirements.txt` with torch/extra-index lines stripped, then `pip install -e . --no-deps`.

## Where things live

| Topic | Path |
|---|---|
| Schema entrypoint / modules | `src/schema.yaml`, `src/schema/**` |
| Component registry + built-in specs | `src/registry/**` (`spec.py`, `registry.py`, `binding.py`, `schema_assembly.py`, `builtin/**`, `snapshot.py`) |
| Plugin API + layered loader | `src/plugins/**` (`api.py`, `loader.py`), `examples/plugins/` |
| Runtime model config (nested) | `src/model/config.py` (`FrankensteinModelConfig.from_dict`) |
| Mixer dispatch + `HybridLayer` | `src/model/hybrid_layer.py` |
| Mixer implementations | `src/model/attention/{gated,latent,sparse}/`, top-level modules |
| Assembled schema snapshot | `docs/schema.assembled.yaml` (`python -m src.registry.snapshot`) |
| Optimizers + factory | `src/model/optimizer/{factory.py,<family>.py}` |
| Norms + factory | `src/model/norm/{factory.py,<impl>.py}` |
| Positional encodings | `src/model/embeddings/` |
| KV cache / incremental decoding | `src/model/cache/` (`manager.py`, `slots.py`, `capabilities.py`, `hooks.py`, `quant/`, `compress/`) |
| Inference runtime config | `src/deploy/inference_config.py`, `src/schema/_inference.yaml` |
| Activations | `src/model/activation_function/` |
| Residuals / mHC / Hyperloop | `src/model/residuals/`, `src/model/mhc.py`, `hyperloop.py` |
| MoE | `src/model/moe/{router,balancing,layer}.py` |
| Fixed numerics | `src/model/numerics.py` (fixed softmax/cross-entropy, arXiv:2610.31642) |
| Model classes | `src/model/frankenstein_{encoder,decoder,vit,audio}.py`, `src/model/__init__.py` |
| Audio (Conformer / ASR) | `src/model/frankenstein_audio.py`, `src/training/audio_dataset.py`, `src/schema/_audio*.yaml` |
| ASR losses / metrics | `src/training/rnnt.py` (RNN-T, CTC, CER), `src/training/metrics.py` |
| Validation / task dispatch | `src/training/config_loader.py`, `src/training/main.py` |
| Training loop | `src/training/trainer.py` |
| Framework-neutral builder | `src/engine.py` (`build_model`) |
| Deployment / export / inference | `src/deploy/` |
| HF safetensors quantization codecs | `src/deploy/safetensors_quant.py` (`q8_0`, `q4_0`, `i2_s`) |
| SBERT | `src/sbert/` |
| Streamlit UI | `src/streamlit_gui/app.py` |
| Presets | `configs/*.yaml`, `configs/examples/*.yaml` |
| Installation guide | `INSTALL.md` |
| Plugin authoring guide | `HOWTO_MAKE_PLUGIN.md`, `examples/plugins/` |
| Tests | `tests/` (80 files; see gates below) |
| End-to-end harness | `full_tests/run_e2e.py` |
| Theory / citations | `docs/paper*/`, `docs/bibliography/` |
| Website mirror sync | `.opencode/command/sync-website.md` |

> `src/model/frankenstein_model.py` **does not exist** (split into the files above; a regression test fails if it is recreated).

## Hard rules

1. **Schema is the contract.** The validated document is the hand-authored `src/schema/**` plus registry-injected component enumerations/fragments (`src/registry/schema_assembly.py`); `additionalProperties: false` holds everywhere. Presets are validated strictly. At runtime, a key the assembled schema does not know is logged as `WARNING [schema] … unknown key(s) ignored` (logger `frankenstein.config`) and dropped. Never *read* or implement behavior for an undocumented key.
2. **Invariants**: `hidden_size % num_heads == 0`; `num_kv_heads` divides `num_heads`; `len(layer_pattern) == num_layers`; `vocab_size` matches the tokenizer.
3. **BitNet is on by default** (`bitnet.enabled: true`). `bitnet.routers: true` requires `bitnet.enabled: true`; `bitnet.conv` requires BitNet + `embedding.conv.enabled`.
4. **Normalization**: enum `layer_norm`, `dynamic_tanh`, `derf`, `rms_norm`, `prms_norm`, `flash_norm`. `prms_norm` uses `norm.partial_ratio`; `flash_norm` composes with `norm.flashnorm_partial_ratio`.
5. **`fasa` / `sparge` are inference-sparse** (`HybridLayer.TRAINING_FREE_LAYERS`). In `train()` mode `HybridLayer` runs standard dense attention over the mixer's own projections (`mixer.dense_forward`); the sparse heuristic activates in `eval()` only. They are valid in training presets.
6. **Forced modes**: `frankenstein_text_decoder` → `mode: decoder`; `frankenstein_vit` and `frankenstein_audio` → `mode: encoder`. `training.task: causal_lm` requires the decoder; vision tasks require the ViT; audio tasks (`asr`, `audio_classification`) require `frankenstein_audio`.
7. **Optimizer keys are prefixed**: `<optimizer>-<group>_<param>` or `<optimizer>-<param>`. Groups: `embeddings`, `pos`, `attention`, `ffn`, `norms`, `head`, `mhc`, `router`, `residual`, `other`. The enum, `by_optimizer` reference and `if/then` prefix rule are **generated from the optimizer's `OptimizerSpec`** by `src/registry/schema_assembly.py` — never hand-edited.
8. **`training.task` is required**, one of `mlm`, `sbert`, `causal_lm`, `patch_prediction`, `classification`, `segmentation`, `text_classification`, `asr`, `audio_classification`. No legacy top-level optimizer keys.
9. **All presets are smoke-tested** (`configs/*.yaml` + `configs/examples/*.yaml`). A new mixer/optimizer needs an example YAML; its schema entry comes from its component spec.
10. **The component registry is the source of truth** (43 activations, 49 mixers, 36 optimizers, 6 norms, 4 residuals, 11 PEs + `none`, 3 embeddings). The activation enum is registry-derived; GLU variants (`swiglu`/`geglu`/`reglu`) carry `is_glu=True` and are handled by `GatedFFN`, not `get_activation()`.
11. **mHC and Mixture-of-Depths are mutually exclusive.** Do not enable both.
12. **Docs stay in sync with the install/dispatch surface.** Any change to extras, CLI subcommands/flags, the plugin API, the registry contract or the assembled schema must update `INSTALL.md`, `HOWTO_MAKE_PLUGIN.md`, `README.md` and `docs/architecture.md` in the same change; regenerate `docs/schema.assembled.yaml`.

## Extending: one declarative recipe

Every pluggable component (mixer, optimizer, norm, activation, residual,
positional encoding, token embedding) is a frozen `ComponentSpec` registered
through `src/registry/register_spec` (built-ins: `src/registry/builtin/**`;
plugins: `src/plugins/api.py`). The spec is the **single source of truth** for
the name, lazy builder, parameter schema fragment, runtime defaults,
capability flags (`training_free`, `cache`, `qk_norm_supported`, `is_glu`, …)
and bilingual documentation rows. The schema enum/table/prefix/`if/then`
entries, the factories, the Streamlit pickers and the e2e harness all derive
from it.

| Feature | Touch (in order) |
|---|---|
| **Any component** | implement the module → add/register a spec (built-in: `src/registry/builtin/<family>.py`; plugin: `register_*` from `src/plugins/api.py`) → example YAML if user-facing → family tests |
| **Mixer** | `src/model/attention/<family>/<name>.py` (`builder(config, pos_encoder=None, params=None)` reading bound `params`) → `src/registry/builtin/attention.py` → optional `_attention_<mixer>.yaml` fragment referenced by the spec → example. Capabilities: `CacheSpec`, `cache_implemented`, `qk_norm_supported`, `qk_clip_supported`, `training_free`, `needs_pos_encoder`. |
| **Optimizer** | `src/model/optimizer/<name>.py` → `src/registry/builtin/optimizer.py` (`param_bindings` + `group_policy`; `custom_build` for genuinely special constructors) → example → `test_optimizer_factory.py`. `build_optimizer` has no per-optimizer branches. |
| **Norm / activation / residual / PE / embedding** | implementation module → builder function → spec in `src/registry/builtin/<family>.py` → example → family test. |
| **Model class** | `src/model/frankenstein_<class>.py` → `model/__init__.py` → `_model_class.yaml` → `config_loader.py` rules → `engine.py` + `training/main.py` → `deploy/*` → `cli.py` choices → preset + example → `test_model_variants.py` (model classes stay structural; they are not plugin components). |
| **Config field** | schema module (or the spec's `params_schema` for a component knob) → `model/config.py` block (only structural blocks are typed dataclasses) → example YAML → `docs/architecture.md` §7. |
| **Dataset/task** | `src/schema/_<name>.yaml` + `schema.yaml` `$ref` → `src/training/<builder>.py` → `engine.py` → `config_loader.py` + `trainer.py` loss → example → dataset tests. |
| **Plugin (external)** | `register_*` from `src/plugins/api.py` in the plugin module → declare `plugins:` in the config (`./path.py`, `pkg.module`) or advertise the `frankenstein.plugins` entry-point group. |

The runtime configuration is the **nested** YAML shape (`FrankensteinModelConfig`):
there is no nested-to-flat flattener, no `from_flat()` constructor and no
attribute aliasing. Read structural blocks as `config.dims.hidden_size`,
`config.norm.type`, `config.bitnet.enabled`, …; per-component knobs live in
`model.attention.<mixer>` / `model.positional_encoding_parameters.<pe>` and are
read through the bound `ComponentParams` (or the small read-only PE properties
such as `config.rope_base`).

## Gates (must stay green)

- `tests/test_schema_consistency.py` — presets validate against the assembled schema, bilingual tables, optimizer prefixes ↔ `if/then`, schema defaults ↔ dataclass.
- `tests/test_registry_core.py`, `test_registry_binding.py`, `test_schema_assembly.py`, `test_plugin_loader.py` — registry contract, param binding, registry-derived schema, layered plugin loading.
- `tests/test_yaml_examples.py` — auto-globs both config directories.
- `tests/test_schema_attention_layers.py` — mixer enum + `attention.*` sub-keys.
- `tests/test_optimizer_factory.py`, `test_optimizer_base.py` — registry/allowed keys/step.
- `tests/test_common_modules.py`, `test_positional_encodings.py`, `test_activation_functions.py`, `test_residuals.py`.
- `tests/test_naming_conventions.py` — no legacy names, no `frankenstein_model.py` shim.
- `tests/test_frankenstein_vit.py`, `test_vision_dataset.py`, `test_vision_loss.py`.
- `tests/test_frankenstein_audio.py`, `test_audio_dataset.py`, `test_audio_loss.py`,
  `test_asr_losses.py`, `test_audio_training.py`, `test_audio_param_categories.py`.
- `tests/test_moe.py` — MoE router/balancing/expert contract.
- `tests/test_qk_norm.py` — QK-norm methods, QK-Clip, QuacK controller.
- `tests/test_fixed_numerics.py` — fixed softmax/cross-entropy swaps.
- `tests/test_block_grad_norms.py`, `test_test_split.py`, `test_metrics.py` —
  per-block grad norms, stratified hold-out split, imbalance-aware metrics.
- `tests/test_kv_cache_*.py` — slot/codec/mixer-parity/fallback/mechanism/eval-only/quant suites;
  `test_inference_config.py`, `test_inference_generate.py`, `test_transformers_export.py`,
  `test_transformers_safetensors_quant.py`, `test_safetensors_quant.py`.
  `test_kv_cache_mixers.py` parametrizes every mixer in `IMPLEMENTED_MIXERS`
  (`src/model/cache/capabilities.py`) except the stateless `mamba` passthrough:
  full-forward vs prefill+decode parity.

## Decisions

| # | Decision |
|---|---|
| D1 | Schema is the executable contract; `additionalProperties: false` everywhere |
| D2 | One nested `FrankensteinModelConfig` tree built by `from_dict` (no flat constructor aliases; superseded by D15) |
| D3 | Model core split into `config.py` / `hybrid_layer.py` / encoder / decoder / ViT / audio; no `frankenstein_model.py` |
| D4 | BitNet enabled by default |
| D5 | `fasa` / `sparge` train as dense attention, sparse eval |
| D6 | mHC and Mixture-of-Depths are mutually exclusive |
| D7 | CPU-first development and testing; CUDA only for real runs |
| D8 | Every preset/example is schema-smoke-tested |
| D9 | Optimizer prefixes are enumerated and cross-checked with conditional rules |
| D10 | Model classes force their modes (`decoder` for the text decoder, `encoder` for ViT and audio) |
| D11 | Paper = theory source of truth; `docs/architecture.md` = canonical implementation spec |
| D12 | `AGENTS.md` points to code/tests; it does not restate the spec |
| D13 | Every pluggable component is a declarative `ComponentSpec`; the registry is the single source of truth |
| D14 | The validated schema is assembled from core YAML + registry fragments; `additionalProperties: false` holds for plugins too |
| D15 | The runtime config is nested only: the flattener, `from_flat()`, the attribute aliases and the export flat view are removed |
| D16 | Plugin loading is layered: entry-points (skip-with-warning), `plugins:` config paths (fail-hard), built-ins (same API) |
| D17 | Test gates are registry-derived: adding a component automatically covers docs/defaults/prefix/cache coverage |

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

- Installation (uv/pip, CPU/CUDA extras, plugins, troubleshooting): [`INSTALL.md`](INSTALL.md).
- Plugin authoring (specs, all seven families, packaging, testing): [`HOWTO_MAKE_PLUGIN.md`](HOWTO_MAKE_PLUGIN.md).
- Implementation spec + extension recipes: [`docs/architecture.md`](docs/architecture.md).
- Assembled schema snapshot: [`docs/schema.assembled.yaml`](docs/schema.assembled.yaml) (regenerate with `python -m src.registry.snapshot --out docs/schema.assembled.yaml`).
- Presets/optimizer parameters: [`configs/README.md`](configs/README.md).
- Theory, formulations, citations: `docs/paper/`, `docs/paper-es/`, `docs/bibliography/`.
- Implementation plans (all of them, single directory): `.opencode/plan/`.
- Build API docs: `bash docs/build_docs.sh` (Sphinx → `docs/_build/html`).
