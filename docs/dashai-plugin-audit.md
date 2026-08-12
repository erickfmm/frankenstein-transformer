# DashAI Plugin Integration Audit — Frankenstein Transformer

**Status:** Planning / architecture audit (no code executed yet)
**Scope:** Turn Frankenstein Transformer into a `dashai-frankenstein` plugin that registers Frankenstein's model classes as DashAI components, with **minimal changes to DashAI** and **contained, well-defined refactors to Frankenstein**.
**Decisions locked with the author:**

1. **Schema (v1):** *Passthrough YAML* — a single Frankenstein YAML payload validated by Frankenstein's own JSON Schema. Curated native form fields are deferred to v2.
2. **Component scope:** *All four* — MLM text classifier, causal decoder (generative), ViT image classifier, and ViT segmentation (+ a plugin-provided segmentation task).
3. **Delivery:** Frankenstein is **published to PyPI** and consumed by the plugin as a normal dependency.

---

## 1. Executive summary

Frankenstein is a standalone, config-driven transformer toolkit (124 Python files, ~29 300 LOC, 101 presets, 27 JSON-Schema files, 33 attention mixers, 23 optimizer families). DashAI is a component-based ML platform whose extensibility model is **PyPI plugins discovered via the `dashai.plugins` entry-points group**, where each registered class is a *component* inheriting a DashAI base class.

**The two systems are not directly compatible** — different config formats (JSON Schema vs pydantic), different model contracts (`nn.Module.forward()` vs `train/save/load/predict` on a `DashAIDataset`), and different runtime models (CLI + GPU-thermal supervisor vs FastAPI + Huey job queue). They cannot be merged "as is".

The good news: **Frankenstein's core (model, trainer, schema validator) can be reused verbatim as a dependency**, wrapped by a thin adapter layer in a new standalone package, `dashai-frankenstein`. The burden splits as:

- **Frankenstein (this repo):** contained refactors — relax the torch pin, split optional deps into extras, extract a non-CLI engine API, make the supervisor optional, formalize a classification head, and publish to PyPI. No change to the CLI behavior or the schema (source of truth).
- **DashAI:** **zero core changes** in the best case. New Tasks (e.g. segmentation) and Models all live inside the plugin and are auto-registered.
- **New code:** the `dashai-frankenstein` adapter package (entry points, pydantic passthrough schema, dataset/IO/metrics adapters, four model components).

The recommended path is phased: prepare Frankenstein (Phase 0), ship a one-model MVP that proves the full contract (Phase 1), then add the decoder and vision components (Phases 2–3).

---

## 2. Goals and hard constraints

### Goals
- Register Frankenstein's model classes inside DashAI so end users can train, evaluate, predict, save, and load them from the DashAI UI.
- Expose all four Frankenstein capabilities: encoder/MLM, causal decoder, ViT classification, ViT segmentation.
- Keep the Frankenstein JSON Schema as the single source of truth for model/training configuration.
- Minimize DashAI edits (target: none in the backend core).

### Hard constraints (from `AGENTS.md`)
These still hold and the integration must respect them:

1. **Schema is the source of truth.** Every config field must exist in `src/schema/_*.yaml`. The plugin must never invent undocumented keys.
2. **Cross-component compatibility** (divisibility of `hidden_size` by `num_heads`, `num_kv_heads` | `num_heads`, vocab/tokenizer match, `task` ↔ optimizer/sbert rules, etc.).
3. **BitNet defaults to `True`**; `bitnet_routers` requires `use_bitnet`.
4. **`fasa_attn`/`sparge_attn` are eval-only** (raise in training).
5. **`frankensteindecoder` forces `mode: decoder`.**
6. **Optimizer prefixed keys** `<class>-<group>_<param>`.
7. **`training.task` is required** (`mlm`/`sbert`/vision/causal).
8. **Adding a mixer/optimizer requires an example YAML** (smoke-tested in CI).

### Non-goals (for v1)
- Native form-UI generation for all ~150 `FrankensteinModelConfig` fields (deferred to v2).
- Replacing Frankenstein's trainer with DashAI's `transformers.Trainer`.
- Running Frankenstein's Streamlit web-server inside DashAI.

---

## 3. Ground truth: the two systems

### 3.1 DashAI plugin & component model

Authoritative sources: `DashAI/back/plugins/utils.py`, `DashAI/back/dependencies/registry/component_registry.py`, `DashAI/back/models/base_model.py`, `DashAI/back/converters/base_converter.py`, `DashAI/back/config_object.py`, and the official docs at <https://docs.dash-ai.com/deep-dive/components>.

**Discovery.** `get_available_plugins()` calls `importlib.metadata.entry_points(group="dashai.plugins")` and `load()`s each entry (`plugins/utils.py:199`). `register_plugin_components()` then calls `registry.register_component(plugin_class)` for each (`plugins/utils.py:287`). Install/uninstall is a `pip` subprocess plus a DB-tracked `Plugin` row; plugins are expected to be PyPI packages whose name starts with `dashai-` (`plugins/utils.py:168`, `:221`).

**The component contract.** When a class is registered, `_get_base_type()` walks the MRO and requires **exactly one** base class that carries a `TYPE` attribute (`component_registry.py:138`). More than one → `TypeError`. A class is a *configurable object* iff `ConfigObject` is in its MRO, in which case it **must** implement `get_schema()` (`component_registry.py:214`), which `ConfigObject` provides by serializing a pydantic `SCHEMA` to JSON Schema (`config_object.py:14`).

**Component `TYPE`s** relevant here: `Model` (`BaseModel`), `GenerativeModel` (`BaseGenerativeModel`), `Task` (`BaseTask`), `GenerativeTask`, `Converter`, `DataLoader`, `Optimizer`, `Metric`, `Explorer`, `Job`.

**`BaseModel` contract** (`base_model.py`): abstract `save(filename)`, `load(filename)`, `train(x_train, y_train, x_validation=None, y_validation=None)`; hooks `predict`, `prepare_dataset`, `prepare_output`, `calculate_metrics`. Metrics are persisted to SQLite via `_save_metrics` (`base_model.py:107`), keyed by `run_id`. Data unit is `DashAIDataset` (a HuggingFace `datasets` wrapper). Training executes inside a **Huey** background job.

**Models ↔ Tasks.** Binding is by name via the class attribute `COMPATIBLE_COMPONENTS` (e.g. `TextClassificationModel.COMPATIBLE_COMPONENTS = ["TextClassificationTask"]`). The registry collects these along the whole MRO (`component_registry.py:174`).

**Schema system.** pydantic `BaseSchema` + `schema_field(type, placeholder, description, alias)` (`schema_field.py`). Available field types: `int_field`, `float_field`, `enum_field`, `string_field`, `bool_field`, `list_field`, `none_type`, `union_type`, `component_field`, `optimizer_float_field`, `optimizer_int_field`. The frontend renders forms directly from the generated JSON Schema; `replace_defs_in_schema` (`base_schema.py:6`) flattens `$defs` into inline properties.

**Pipeline.** A DAG: `DataSelector → Converter → Train → Prediction → Exploration`. Each node is a `BaseJob`; results flow through a shared `context` dict.

**Reference implementation to mimic:** `HuggingFaceTextClassificationTransformer` (`models/hugging_face/base_text_classification_transformer.py`) — it wraps a HF `AutoModelForSequenceClassification`, trains with `transformers.Trainer`, streams metrics back via a `MetricsCallback`, saves/loads through `save_pretrained`/`from_pretrained` with custom hyperparameters embedded in the HF config. The Frankenstein adapters will mirror this skeleton.

### 3.2 Frankenstein architecture

Authoritative sources: `AGENTS.md`, `pyproject.toml`, `src/cli.py`, `src/schema.yaml`, `src/model/config.py`, `src/training/main.py`, `src/training/trainer.py`.

- **Entrypoint:** `src/cli.py:main` → console script `frankenstein-transformer` (`pyproject.toml:30`). Subcommands: `train`, `deploy`, `quantize`, `infer`, `sbert-train`, `sbert-infer`, `transformers-export`, `bitnet-gguf`, `web-server`.
- **Source of truth:** JSON Schema. `src/schema.yaml` `$ref`s into modular `src/schema/_*.yaml` and `src/schema/_model/_*.yaml`; `additionalProperties: false` everywhere. **Not pydantic.**
- **Config objects (dataclasses):** `FrankensteinModelConfig` — **151 typed fields** (`src/model/config.py`); `TrainingConfig` — ~100 fields (`src/training/trainer.py`); plus `image`/`dataset`/`sbert` blocks. The YAML `model:` block is nested and **flattened** before constructing `FrankensteinModelConfig` (`src/utils/config_flatten.py:flatten_model_dict`, `flatten_image_dict`).
- **Model classes:** `FrankensteinEncoder` (`src/model/frankenstein_encoder.py`, MLM/encoder), `FrankensteinDecoder` (causal), `FrankensteinViT` (`src/model/frankenstein_vit.py`, vision). All are **plain `nn.Module` exposing only `forward()`** — no DashAI-style lifecycle methods.
- **Training orchestration** is fused into `src/training/main.py`: `main()` at line 910 dispatches on `model_class`/`task`; `_run_vision_task` at 837; `_run_sbert_task` at 635; `_run_under_supervisor` at 565 spawns a **GPU-thermal supervisor subprocess**. The loop itself is `TitanTrainer` (`src/training/trainer.py`).
- **Persistence / inference:** `src/deploy/` — `deploy.py` (checkpoint→artifacts), `inference.py` (batch/interactive), `quantization.py` (BitNet), `transformers_export.py` (HF export). No `save`/`load` on the model classes themselves.
- **Tokenizer:** custom `SpanishSPMTokenizer` (`src/tokenizer/`) or HF `AutoTokenizer` (the `base_model` path in `main.py:_load_base_model_and_tokenizer`).
- **Dependencies (`pyproject.toml`):** `torch==2.6.0+cu118` (hard pin with a default uv index), `transformers`, `sentence-transformers`, `sentencepiece`, `datasets`, `streamlit`, plus `numpy`, `tqdm`, `psutil`, `PyYAML`. Python `>=3.9, <3.13`.

### 3.3 Compatibility matrix

| Aspect | Frankenstein | DashAI | Compatible? |
|---|---|---|---|
| Config format | JSON Schema (`src/schema.yaml`) | pydantic `BaseSchema` → JSON Schema | **No** (different metadata; see §5) |
| Config object | dataclasses (`FrankensteinModelConfig`) | pydantic models / raw dict via `validate_and_transform` | Bridged by adapter |
| Model contract | `nn.Module.forward()` | `train/save/load/predict` on `DashAIDataset` | **No** (needs adapter) |
| Data unit | HF dataset name / streaming corpus / tensors | `DashAIDataset` (HF `datasets` wrapper) | Bridged (both use HF `datasets`) |
| Training runtime | CLI + GPU-thermal supervisor subprocess | Huey job inside FastAPI worker | Needs in-process mode |
| Persistence | torch checkpoint + YAML + deploy pipeline | `save(filename)`/`load(filename)` per run | Bridged by adapter |
| Metrics | CSV log (`training_metrics.csv`) | SQLite via `_save_metrics` + `MetricsCallback` | Bridged via callback |
| Torch | pinned `2.6.0+cu118` | unpinned, cpu/cuda extras | **Conflict** — relax pin |
| Python | `>=3.9, <3.13` | `>=3.10` | OK (3.10–3.12 overlap) |

---

## 4. Answers to the guiding questions

**Q1 — How do you make a plugin on DashAI?**
Ship a PyPI package named `dashai-<name>`. In `pyproject.toml`, register one or more classes under `[project.entry-points."dashai.plugins"]`. Each class inherits a DashAI base (`BaseModel`/`BaseGenerativeModel`/`BaseTask`/…) that carries a `TYPE`, defines a pydantic `SCHEMA`, and implements the required abstract methods. Install via pip; DashAI discovers it through `entry_points(group="dashai.plugins")` and registers each class in `ComponentRegistry`. **No DashAI source edits are required.**

**Q2 — Can Frankenstein be integrated "as is", just adding code?**
**No.** Its models are bare `nn.Module`s, configured by JSON Schema/dataclasses and driven by a CLI/supervisor — none of which matches the DashAI component contract. An **adapter layer is mandatory**. However, the Frankenstein **core is reused verbatim as a dependency**; only adapters + entry points are new code.

**Q3 — Can the plugin reuse the Frankenstein schema?**
**Not directly** — Frankenstein is JSON Schema, DashAI is pydantic `BaseSchema`. Three options were evaluated (see §5.1). **Chosen for v1: passthrough YAML** — a single `frankenstein_yaml` field on the pydantic schema; the payload is validated by Frankenstein's own `load_training_config` (its JSON Schema stays the source of truth). Curated native fields are a v2 follow-up.

**Q4 — How to manage the multi-model architecture?**
Map Frankenstein's 3 model classes × tasks to **multiple DashAI components in one plugin**, each with the right `COMPATIBLE_COMPONENTS`:

| Frankenstein | DashAI component | Base | Binds to DashAI Task |
|---|---|---|---|
| `frankenstein` (encoder/MLM) | `FrankensteinMLMModel` | `BaseModel` | `TextClassificationTask` |
| `frankensteindecoder` (causal) | `FrankensteinDecoderModel` | `BaseGenerativeModel` | `TextToTextGenerationTask` |
| `frankenstein_vit` classification | `FrankensteinViTClassifier` | `BaseModel` | `ImageClassificationTask` |
| `frankenstein_vit` segmentation | `FrankensteinViTSegmenter` | `BaseModel` | `SegmentationTask` (new; see below) |

All four import the **same Frankenstein engine**, so there is exactly one heavy dependency.

**Q5 — One plugin or many?**
**One package, `dashai-frankenstein`**, registering multiple components. Splitting (e.g. text vs vision) would duplicate the heavy shared stack (torch/transformers/Frankenstein core) and add packaging overhead for no real benefit. A single entry-points group can register Models, Tasks, and (optionally) Converters freely.

**Q6 — The plan to turn Frankenstein into a DashAI plugin.** → §6 (phased) and §7/§8 (per-project changes).

---

## 5. Integration design

### 5.1 Schema strategy (v1: passthrough YAML)

The plugin exposes a **minimal pydantic schema** per component. The main field is `frankenstein_yaml` (a string containing a full Frankenstein training YAML, or a dict). On `train()`, the adapter:

1. Parses the YAML into a Python dict.
2. Injects DashAI-derived values (tokenizer, dataset reference, run outputs, `vocab_size`/`num_classes` from the `DashAIDataset`).
3. Writes it to a temp file and calls Frankenstein's `load_training_config(path)` — **Frankenstein's JSON Schema is the validator.** A validation error surfaces to the DashAI user verbatim.

This preserves all 33 mixers / 23 optimizers / 6 norms / 43 activations instantly, keeps `AGENTS.md` constraint #1 intact, and requires zero schema translation. To make the UX bearable, the schema also offers:

- `preset: enum_field([...])` — a dropdown of **bundled Frankenstein presets** (loaded from Frankenstein's `configs/*.yaml` via `list_config_paths`). Selecting one populates `frankenstein_yaml` with the preset content.
- `device`, `batch_size`, `num_epochs` — a few convenience overrides that the adapter merges into the YAML before validation (these are also native Frankenstein keys, so they are schema-legal).

**v2 (deferred):** translate ~15 high-leverage fields (`model_class`, `hidden_size`, `num_layers`, `num_heads`, `layer_pattern`, `task`, `optimizer_class`, `ffn_activation`, `norm_type`, `use_moe`, `use_bitnet`, …) into native `schema_field`s for first-class form UX, while still serializing them into the same Frankenstein YAML under the hood. Full translation of all 151 model + ~100 training fields is **not recommended** (brittle against an actively-evolving schema).

### 5.2 Plugin package layout

```
dashai-frankenstein/                      # standalone PyPI package
  pyproject.toml                          # name="dashai-frankenstein"
                                          #   [project.entry-points."dashai.plugins"]
                                          #     frankenstein_mlm      = "dashai_frankenstein:FrankensteinMLMModel"
                                          #     frankenstein_decoder  = "dashai_frankenstein:FrankensteinDecoderModel"
                                          #     frankenstein_vit_cls  = "dashai_frankenstein:FrankensteinViTClassifier"
                                          #     frankenstein_vit_seg  = "dashai_frankenstein:FrankensteinViTSegmenter"
                                          #     segmentation_task     = "dashai_frankenstein:SegmentationTask"
                                          # dependencies = ["transformer-encoder-frankenstein", ...]
  README.md
  src/dashai_frankenstein/
    __init__.py                           # re-exports the 5 component classes (entry-point targets)
    engine.py                             # thin facade over Frankenstein: build_model/train_from_config/
                                          #   save_checkpoint/load_checkpoint  (calls the new src/engine.py)
    config.py                             # pydantic schemas (FrankensteinPassthroughSchema + per-component)
    presets.py                            # loads Frankenstein configs/*.yaml into the preset enum
    adapters/
      dataset.py                          # DashAIDataset <-> Frankenstein dataset expectation
      io.py                               # save/load <-> torch checkpoint + YAML + tokenizer
      metrics.py                          # Frankenstein step metrics -> DashAI _save_metrics / MetricsCallback
    models/
      mlm.py                              # FrankensteinMLMModel(BaseModel)
      decoder.py                          # FrankensteinDecoderModel(BaseGenerativeModel)
      vit_classifier.py                   # FrankensteinViTClassifier(BaseModel)
      vit_segmenter.py                    # FrankensteinViTSegmenter(BaseModel)
    tasks/
      segmentation.py                     # SegmentationTask(BaseTask)  [DashAI has none]
```

### 5.3 Component designs

All components share the same engine facade and differ mainly in: base class, `COMPATIBLE_COMPONENTS`, the task-specific head, and how a `DashAIDataset` maps to Frankenstein inputs.

**`FrankensteinMLMModel(BaseModel)`** — text classification via the encoder.
- `COMPATIBLE_COMPONENTS = ["TextClassificationTask"]`.
- `train(x_train, y_train, x_val, y_val)`:
  1. Resolve a text column + label column from the `DashAIDataset`; derive `num_labels` and a tokenizer `name_or_path`.
  2. Build the Frankenstein YAML (from `frankenstein_yaml`/preset + injected `vocab_size`, `num_classes`, dataset ref, task=`mlm` or a classification fine-tune path).
  3. Call `engine.train_from_config(...)` (in-process, supervisor off). Either (a) pretrain MLM then attach a linear classification head and fine-tune, or (b) use Frankenstein's `base_model` bridge with a classification head — see §5.4.
  4. Stream metrics via the adapter callback into `_save_metrics`.
- `predict(x_pred)`: tokenize, forward through the encoder + head, return a class-probability matrix (shape `(N, num_labels)`), mirroring `HuggingFaceTextClassificationTransformer.predict`.
- `save(filename)` / `load(filename)`: serialize `{state_dict, frankenstein_yaml, tokenizer, num_labels, head_state}` into the DashAI run dir; reload by rebuilding the model from the YAML and `load_state_dict`.

**`FrankensteinDecoderModel(BaseGenerativeModel)`** — causal generation.
- `COMPATIBLE_COMPONENTS = ["TextToTextGenerationTask"]`; forces `model_class: frankensteindecoder` (which already forces `mode: decoder`).
- `generate(prompt)` / `predict(...)`: autoregressive decode using the decoder; aligns with DashAI's generative-model interface.
- Otherwise mirrors the MLM adapter with a generation head instead of a classification head.

**`FrankensteinViTClassifier(BaseModel)`** — image classification.
- `COMPATIBLE_COMPONENTS = ["ImageClassificationTask"]`.
- Maps the DashAI image column to Frankenstein's `image:` + `dataset:` blocks (`task: classification`). Reuses `FrankensteinViT`'s classification head (`src/model/frankenstein_vit.py`).

**`FrankensteinViTSegmenter(BaseModel)`** — segmentation.
- Binds to **`SegmentationTask`**, a new `BaseTask` also provided by this plugin (DashAI has no segmentation task). The task declares `inputs_types`/`outputs_types` for image-column in / mask out and the appropriate cardinality.
- Uses the ViT segmentation head (`seg_head_type: pixel` or `eomt`).

### 5.4 The classification-head question

Frankenstein's encoder is an MLM backbone (token-level logits over the vocab). DashAI's `TextClassificationTask`/`ImageClassificationTask` expect sequence/image-level class probabilities. Two viable strategies:

- **Strategy A (recommended): fine-tune a classification head on top of the encoder pooler.** Add a small pooling + linear head (weight-initialized, not BitNet-quantized) on the encoder output; train it jointly or as a second stage. This is exactly what HF's `AutoModelForSequenceClassification` does and what `HuggingFaceTextClassificationTransformer` wraps.
- **Strategy B: HF-export bridge.** Use Frankenstein's existing `transformers_export.py` to convert a trained checkpoint into a HF model, then load it via `AutoModelForSequenceClassification`. Reuses DashAI's existing HF infra but adds an export step and is limited to export-compatible configs (`check_yaml_export_compatibility`).

The audit recommends **Strategy A** for v1 (in-process, no export dependency), with Strategy B documented as an alternative for users who want to ship a Frankenstein model into the rest of the HF ecosystem.

### 5.5 Dataset, IO, and metrics adapters

- **Dataset adapter** (`adapters/dataset.py`): `DashAIDataset` is an HF `datasets` wrapper, and Frankenstein already consumes HF datasets (`streaming_mlm_dataset.py`, the `base_model` path uses `datasets` names). The adapter extracts the relevant column(s), optionally writes a temporary HF dataset shard or passes an in-memory `Dataset`, and supplies the dataset reference that Frankenstein's loader expects. For vision, image bytes from DashAI's `DashAIImage` are materialized into the format used by `src/training/vision_dataset.py`.
- **IO adapter** (`adapters/io.py`): one artifact bundle per DashAI run directory — `model.pt` (state dict), `config.yaml` (the validated Frankenstein YAML), `tokenizer/` (SPM model file or HF tokenizer files), and a small `dashai_meta.json` (`num_labels`, `task`, `model_class`, `head_state`). `load` rebuilds the `nn.Module` from `config.yaml` via `engine.build_model` and restores weights.
- **Metrics adapter** (`adapters/metrics.py`): wraps a DashAI-style callback (mirror of `models/hugging_face/metrics_callback.py`) that Frankenstein's `TitanTrainer` invokes per step/epoch, translating Frankenstein's metric dict into `BaseModel._save_metrics(split, level, results, log_index)` writes. This is what makes training visible in the DashAI UI.

---

## 6. Phased plan

### Phase 0 — Prepare Frankenstein (this repo; no plugin yet)
Goal: make Frankenstein importable and drivable from a non-CLI host.

1. Relax torch pin and split extras (§7.1, §7.2).
2. Extract `src/engine.py` — a non-CLI, non-supervisor façade (§7.3).
3. Add an in-process / supervisor-off mode to training (§7.4).
4. Formalize classification heads on encoder + reuse ViT heads (§7.5).
5. Publish to PyPI (§7.6).
6. Verify the existing CLI, presets, and `tests/test_yaml_examples.py` still pass unchanged.

**Exit criterion:** a small Python script can `import` Frankenstein, build a model from a YAML dict, train one step in-process, and save/load — without touching `src/cli.py`.

### Phase 1 — MVP plugin (one model)
Goal: prove the full DashAI contract end-to-end.

- Create the `dashai-frankenstein` package skeleton, entry points, and the passthrough schema (§5.1).
- Implement `FrankensteinMLMModel` + the dataset/IO/metrics adapters.
- Validate the loop: install plugin in DashAI → it appears in the registry → user picks `FrankensteinMLMModel` for `TextClassificationTask` → trains via Huey → metrics stream to the UI → save/load/predict work.

**Exit criterion:** a Frankenstein encoder trains on a DashAI text-classification dataset and predicts, fully from the DashAI UI.

### Phase 2 — Decoder + ViT classifier
- `FrankensteinDecoderModel` (`BaseGenerativeModel` → `TextToTextGenerationTask`).
- `FrankensteinViTClassifier` (`ImageClassificationTask`).
- Reuse the Phase-1 engine and adapters; add only the generation head and the vision dataset path.

### Phase 3 — UX + segmentation + polish
- `FrankensteinViTSegmenter` + `SegmentationTask` (new `BaseTask`).
- Curated native pydantic fields (v2 schema) for the highest-leverage knobs.
- Bundled-preset dropdown populated from `configs/*.yaml`.
- Hardening: error surfacing, device handling, large-model memory guards.

---

## 7. Changes needed in Frankenstein (detailed)

These are the **required** modifications in this repository. Each is scoped to be backward-compatible with the existing CLI and tests.

### 7.1 Relax the torch pin *(packaging blocker)*
- **File:** `pyproject.toml:17`.
- **Today:** `torch==2.6.0+cu118` as a hard dependency with a default uv index (`pyproject.toml:11-14`).
- **Change:** make the core requirement `torch>=2.0` (no index, no `+cu118` suffix). Keep `cu118`/`cu126`/`cu128`/`cpu` as **optional extras** that pin specific wheels for users who want them (including the dev box P40/CUDA-11.8 path).
- **Why:** DashAI brings its own torch via `cpu`/`cuda` extras (`DashAI/pyproject.toml:84-92`); a hard `torch==2.6.0+cu118` makes the two uninstallable together. The CI matrix already tests CPU torch (`AGENTS.md` CI quirks), so `torch>=2.0` is safe.

### 7.2 Move optional deps into extras
- **File:** `pyproject.toml:16-27`.
- **Change:**
  - Core (required): `transformers`, `datasets`, `PyYAML`, `numpy`, `tqdm`, `psutil` (and `torch>=2.0`).
  - Extra `train` (or keep as-is for local installs): keeps the current full set.
  - Extra `sbert`: `sentence-transformers`, `sentencepiece`.
  - Extra `web`: `streamlit`.
- **Why:** the DashAI plugin should not force `streamlit` or `sentence-transformers` on every DashAI user; only the components that need them pull the extra.

### 7.3 Extract a reusable engine API *(main refactor)*
- **Today:** model construction + tokenizer setup + dataset wiring + supervisor spawn are all fused in `src/training/main.py` (`_load_legacy_frankenstein_model:115`, `_load_base_model_and_tokenizer:46`, `_build_dataloader:198`, `_run_vision_task:837`, `main:910`). None of this is callable without the CLI/supervisor.
- **Change:** add `src/engine.py` exposing pure functions (no argparse, no subprocess):
  - `build_model(model_class: str, model_config: FrankensteinModelConfig) -> nn.Module`
  - `build_tokenizer(loaded: LoadedTrainingConfig) -> Any`
  - `train_from_config(config: str | dict, *, dataset=None, device="auto", supervisor="auto", metrics_callback=None) -> TrainResult` — where `TrainResult` holds the trained `nn.Module`, the validated `LoadedTrainingConfig`, and the tokenizer.
  - `save_checkpoint(path, model, loaded, tokenizer, *, extra=None)` and `load_checkpoint(path) -> (model, loaded, tokenizer, extra)`.
- **Refactor discipline:** `src/training/main.py` is rewritten to **delegate** to `engine.py` (the `train`/`deploy`/etc. subcommands call the same functions). CLI behavior, exit codes, and the GPU-thermal supervisor path are preserved — the supervisor simply wraps `engine.train_from_config(..., supervisor="auto")`.
- **Why:** the DashAI plugin must drive training in-process from a Huey worker; it cannot shell out to `frankenstein-transformer train` nor spawn a supervisor. This refactor is also a pure quality win for the standalone project (testability, notebooks, HF export reuse).

### 7.4 Optional in-process / supervisor-off mode
- **Files:** `src/training/main.py:565 _run_under_supervisor`, `src/training/trainer.py` (`TrainingConfig`).
- **Change:** add `supervisor` control to `TrainingConfig` (values: `auto` | `off`) and to `engine.train_from_config`. When `off`, `TitanTrainer` runs directly in the current process; the GPU-thermal guard still operates as an in-process polling loop (it already supports this), but no child process is spawned.
- **Why:** DashAI hosts training in its own worker process; a nested supervisor subprocess would conflict with Huey's lifecycle and signals.

### 7.5 Formalize classification heads
- **Files:** `src/model/frankenstein_encoder.py` (text head), `src/model/frankenstein_vit.py` (already has classification + segmentation heads).
- **Change:** add an optional, non-BitNet pooling + linear classification head on the encoder (disabled by default, so the MLM CLI is unaffected), constructible from a `num_labels` argument. Expose it through `engine.build_model(..., num_labels=...)`.
- **Why:** DashAI's classification tasks require sequence/image-level probabilities (see §5.4, Strategy A).

### 7.6 Publish to PyPI
- **Today:** `name = "transformer_encoder_frankenstein"`, `version = "0.1.0"`, no PyPI release.
- **Change:** publish (under a public name — e.g. `transformer-encoder-frankenstein`) so the plugin can list it in `dependencies`. Tag a release; keep the version compatible with DashAI's Python `>=3.10`.
- **Why:** locked decision (PyPI dependency).

### 7.7 (Optional, v2) Field-metadata export
- **Change:** emit `FrankensteinModelConfig`/`TrainingConfig` field metadata (name, type, default, range, enum) as JSON (e.g. `src/schema/_field_metadata.json` generated at build time).
- **Why:** lets the plugin auto-generate curated pydantic fields instead of hand-translating 151+100 fields, and keeps the curated UI in sync with the schema automatically.

### Summary table — Frankenstein changes

| # | File(s) | Change | Risk | Backward-compatible? |
|---|---|---|---|---|
| 7.1 | `pyproject.toml` | Relax torch pin to `>=2.0`, CUDA via extras | Low | Yes (extras preserve local CUDA path) |
| 7.2 | `pyproject.toml` | `streamlit`/`sbert` → extras | Low | Yes (extras preserve full install) |
| 7.3 | new `src/engine.py`, rewrite `src/training/main.py` | Extract non-CLI engine façade | Medium | Yes (CLI delegates, behavior preserved) |
| 7.4 | `src/training/main.py`, `trainer.py` | `supervisor: off` mode | Low-Medium | Yes (default `auto` unchanged) |
| 7.5 | `frankenstein_encoder.py`, `engine.py` | Optional classification head | Low | Yes (opt-in) |
| 7.6 | release | PyPI publish | Low | n/a |
| 7.7 | build-time generator | Field-metadata JSON (v2) | Low | Yes (additive) |

---

## 8. Changes needed in DashAI

**Target: 0 core changes.** The plugin system is purpose-built for this and requires no edits to `ComponentRegistry`, `initial_components.py`, `base_model.py`, the API, or the DI container.

The only realistic DashAI-side work, all of it **local and optional**:

- **0 changes** if we bind only to existing tasks (`TextClassificationTask`, `TextToTextGenerationTask`, `ImageClassificationTask`) — these already exist (`DashAI/back/tasks/`).
- **New `SegmentationTask`** for ViT segmentation: lives **inside the plugin** as a `BaseTask` component (tasks are components). No DashAI edit needed in the backend. The *only* place a DashAI change could become necessary is the **frontend**, if it cannot render a new output semantic type (e.g. a segmentation mask column). That would be a local frontend addition, not a backend core change — and only if segmentation is in scope (Phase 3).
- **Documentation:** a note in DashAI's plugin guide listing `dashai-frankenstein` (their repo, not ours).

Net: the "minimize DashAI changes" objective is **achievable as stated** — the backend core stays untouched; the entire burden is on Frankenstein (§7) plus the new plugin package.

---

## 9. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Torch version conflict between Frankenstein and DashAI | High (today) | Install fails | §7.1 (relax pin) |
| Engine refactor breaks CLI/tests | Medium | Regression | Phase 0 exit criterion: CLI + `tests/test_yaml_examples.py` green; rewrite `main.py` as a thin delegate |
| Supervisor subprocess collides with Huey worker | Medium | Hung/dead jobs | §7.4 (`supervisor: off`) |
| MLM-only backbone ≠ classification output | Medium | Wrong `predict` shape | §5.4 Strategy A (classification head); unit-test shape `(N, num_labels)` |
| Dataset-shape mismatch (streaming MLM corpus vs DashAI tabular dataset) | Medium | Train failure | §5.5 dataset adapter; start with a text-classification dataset that has a clean text+label schema |
| Schema drift (Frankenstein evolves) breaks passthrough | Low | Stale presets | Passthrough defers to Frankenstein's own validator, so it is drift-tolerant by construction; v2 curated fields gated behind §7.7 metadata export |
| Large-model memory under DashAI's CPU-default install | Medium | OOM | Surface clear errors; document CUDA install; consider `--device` handling |
| SBERT/Streamlit pulled into DashAI unnecessarily | Low | Bloat | §7.2 extras |

---

## 10. Open questions (to resolve before Phase 1)

1. **Classification training recipe** for `FrankensteinMLMModel`: joint head training vs. MLM-pretrain-then-finetune. Decide based on a small quality benchmark on a DashAI text-classification dataset.
2. **Tokenizer default**: ship a bundled SPM tokenizer for the encoder path, or always require an HF `name_or_path` (the `base_model` path)? The latter is simpler and matches DashAI's HF-centric UX.
3. **PyPI package name** for Frankenstein (`transformer-encoder-frankenstein` vs. a shorter alias).
4. **Segmentation output semantic type**: confirm whether DashAI's frontend can render a mask-typed output column, or whether Phase 3 needs a local frontend addition.
5. **Metrics granularity**: map Frankenstein's per-step CSV metrics to DashAI's `STEP`/`BATCH`/`EPOCH` levels precisely (which Frankenstein metric → which DashAI `LevelEnum`).

---

## 11. References (file:line)

Frankenstein:
- CLI & subcommands: `src/cli.py:412 build_parser`, `src/cli.py:565 main`
- JSON Schema root: `src/schema.yaml`; modular: `src/schema/_*.yaml`, `src/schema/_model/_*.yaml`
- Model config (151 fields): `src/model/config.py:91 FrankensteinModelConfig`
- Training config (~100 fields): `src/training/trainer.py:52 TrainingConfig`
- Training orchestration: `src/training/main.py:910 main`, `:837 _run_vision_task`, `:635 _run_sbert_task`, `:565 _run_under_supervisor`
- Model classes: `src/model/frankenstein_encoder.py:29`, `src/model/frankenstein_vit.py`
- Config flatten (nested YAML → flat dataclass): `src/utils/config_flatten.py`
- Deploy/inference: `src/deploy/{deploy,inference,quantization,transformers_export}.py`

DashAI:
- Plugin discovery/registration: `DashAI/back/plugins/utils.py:199`, `:287`, `:168`, `:221`
- Component registry: `DashAI/back/dependencies/registry/component_registry.py:138`, `:174`, `:202`
- Configurable objects / schema: `DashAI/back/config_object.py:14`; `DashAI/back/core/schema_fields/{base_schema,schema_field,enum_field,list_field,union_type}.py`
- Model contract: `DashAI/back/models/base_model.py:20` (abstract `save/load/train`), `:107 _save_metrics`, `:226 calculate_metrics`
- Reference HF wrapper: `DashAI/back/models/hugging_face/base_text_classification_transformer.py` (train/predict/save/load + `MetricsCallback`)
- Base converter: `DashAI/back/converters/base_converter.py:24`
- Base task: `DashAI/back/tasks/base_task.py:14`
- Existing tasks: `DashAI/back/tasks/{text_classification,image_classification,text_to_text_generation,...}_task.py`
- Plugin docs: <https://docs.dash-ai.com/deep-dive/components>, <https://docs.dash-ai.com/deep-dive/architecture>
