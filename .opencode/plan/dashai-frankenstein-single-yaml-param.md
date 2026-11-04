# Plan: dashai-frankenstein — single YAML parameter + pre-launch validation

## Goal
Reduce every Frankenstein DashAI model component to a **single user-facing parameter**: a
`frankenstein_yaml` string holding a full Frankenstein training YAML. Remove the convenience
overrides (`preset`, `device`, `batch_size`, `num_epochs`, `learning_rate`) — all of these are
already expressible in the Frankenstein YAML and are "guessed from schema" there. Validate the
YAML against the Frankenstein JSON Schema **before** launching train/inference, surfacing a
readable error to the DashAI user.

The field description links to the hosted Frankenstein YAML builder at
<https://erickfmm.github.io/frankenstein-transformer/index.html>.

## Decision: Option A (chosen) — `string_field()` single-line, no DashAI edit

An investigation of the DashAI rendering pipeline (see "DashAI field capabilities" appendix
below) confirmed that a DashAI plugin, using **only** backend schema declarations and **without**
modifying DashAI source, cannot obtain a multiline textarea or a file-upload field:

- `type: "string"` renders a single-line MUI `TextField` (no `multiline`).
- `type: "text"` is a dead branch the backend never emits; even if forced, it renders single-line.
- `json_schema_extra` is locked to `{placeholder, display_name}`; the frontend reads no
  `widget`/`ui:widget`/`format`/`multiline`/`x-*` key.
- `array` is a comma-delimited single-line input (`join(",")`→`split(",")`) — commas inside YAML
  break it; not a multiline editor.
- There is no `file_field`/`FileInput`/`Dropzone`/`path_field`/`uri` case in the renderer.
  `multipart`/`FormData` in the frontend is for dataset/job upload, not model parameters.
- `component_field`/`ClassInput` only yields a dropdown + schema-driven subform; plugins register
  Python classes only (no frontend assets, no custom React injection).
- `extraOptions`/`getValues` are legacy, never passed a non-null value, and `FormSchema.jsx`
  (the current path) does not forward them.

**Therefore the plugin keeps `frankenstein_yaml` as `string_field()` (single-line paste).**
The textarea improvement is deferred to a future upstream Issue/PR against DashAI (see
"Future work" below) and does not block this plan. This keeps the plugin 100% self-contained.

## Scope

Executable in this repo: `dashai-frankenstein/` only. No edits to the DashAI distribution.

### 1. `dashai-frankenstein/src/dashai_frankenstein/config.py`

- Collapse `FrankensteinPassthroughSchema` to a **single field**: `frankenstein_yaml`.
- Remove: `preset`, `device`, `batch_size`, `num_epochs`, `learning_rate`, `_preset_enum()`,
  `_DEVICE_CHOICES`, `_DEVICE_PLACEHOLDER`.
- Keep `frankenstein_yaml` as `schema_field(string_field(), ...)` (single-line string — Option A).
- Update the `description` (MultilingualString en + es) to include the link:
  > "A full Frankenstein training YAML. Build it with the
  > [Frankenstein YAML builder](https://erickfmm.github.io/frankenstein-transformer/index.html)
  > and paste it here. Validated against the Frankenstein JSON Schema before launch."
- Update `alias` (en/es) to "Frankenstein YAML".
- `FrankensteinClassifierSchema(FrankensteinPassthroughSchema)` stays as an empty subclass so
  ViT components need no schema edits.

### 2. NEW `dashai-frankenstein/src/dashai_frankenstein/validate.py`

`validate_yaml(yaml_text: str) -> None`:

1. `yaml.safe_load(yaml_text)` — raises on malformed YAML (syntax error).
2. Resolve the Frankenstein JSON Schema via `src.utils.schema_loader.resolve_schema(<frankenstein_pkg>/src/schema.yaml)`.
3. `jsonschema.validate(instance=parsed, schema=resolved)` — catches
   `additionalProperties: false` violations and enum errors that the loader misses.
4. Write the parsed YAML to a temp file and call `load_training_config(tmp_path)` — catches
   cross-component constraints (`hidden_size % num_heads == 0`, `num_kv_heads` divides
   `num_heads`, `vocab_size` match, bitnet flags, ffn activation, optimizer presence,
   task/model_class compatibility, `frankensteindecoder` forces `mode: decoder`, vision tasks
   require `frankenstein_vit`, etc.).
5. On any failure raise a single `ValueError` with a concatenated, user-readable message
   (prefix each stage's error so the user knows which check failed).

`jsonschema` is already a Frankenstein dependency (present in `uv.lock`); add it to the plugin's
`pyproject.toml` `dependencies` to be explicit.

### 3. `dashai-frankenstein/src/dashai_frankenstein/engine.py`

- Re-export `resolve_schema` from `src.utils.schema_loader` and `load_training_config`
  (already imported) so `validate.py` and the models go through the facade.
- Optionally add a `validate_training_yaml(yaml_text)` convenience wrapper that calls
  `dashai_frankenstein.validate.validate_yaml` (keeps the facade as the single entry point).

### 4. `dashai-frankenstein/src/dashai_frankenstein/models/base.py`

- `resolve_yaml(self)`: return `str(self.frankenstein_yaml or "").strip()`; remove the `preset`
  fallback. Raise `ValueError` if empty.
- `classification_train(self, ...)`:
  - Call `validate_yaml(yaml_text)` **before** `build_model_from_yaml`.
  - Remove reads of `self.device`, `self.batch_size`, `self.num_epochs`, `self.learning_rate`.
  - Read runtime params from `loaded.training_runtime` after `build_model_from_yaml` returns:
    - `device = resolve_device(loaded.training_runtime.get("device", "auto"))`
      (fallback: `"auto"` → `resolve_device` maps to cuda/cpu).
    - `batch_size = int(loaded.training_runtime.get("batch_size", 16))`.
    - `num_epochs = int(loaded.training_runtime.get("num_epochs", 3))`.
    - `lr`: read from the optimizer parameters in the loaded config
      (`loaded.training_config.optimizer_parameters`) — e.g. `adamw-lr`. If absent, fall back to
      `1e-4`. **Do not** read `learning_rate` from `training_runtime` (the Frankenstein schema
      does not define it there; the optimizer parameters are the source of truth).
  - Keep the rest of the loop (tokenizer resolve, vocab match, AdamW head optimizer, epoch loop,
    `EpochMetricsHook`).
- `persistence_load(cls, filename)`: instantiate with `frankenstein_yaml=""` only; drop the
  `preset=""`, `device="CPU"`, `batch_size=16`, `num_epochs=1`, `learning_rate=None` kwargs.

### 5. `dashai-frankenstein/src/dashai_frankenstein/models/decoder.py`

- `FrankensteinDecoderSchema(FrankensteinPassthroughSchema)`: keep as empty subclass.
- `__init__`: keep only `self.frankenstein_yaml`. Remove `self.preset`, `self.device`,
  `self.batch_size`, `self.num_epochs`, `self.learning_rate`.
- **Keep** `self.max_new_tokens`, `self.temperature`, `self.top_k` — these are **inference-time
  generation parameters**, not training parameters, and the Frankenstein schema has no home for
  them. (Confirmed decision: preserve gen params.)
- `_ensure_model()`:
  - Call `validate_yaml(resolve_yaml(self))` before `build_model_from_yaml`.
  - `device = resolve_device(loaded.training_runtime.get("device", "auto"))` (read from loaded
    config, not `self.device`).
- `load(cls, ...)`: after `persistence_load`, set `instance._device` from the loaded config's
  `training_runtime.device` (or `"auto"`); remove `instance.device` reads.

### 6. `dashai-frankenstein/src/dashai_frankenstein/models/vit_classifier.py`

- `__init__`: keep only `self.frankenstein_yaml`. Remove `self.preset`, `self.device`,
  `self.batch_size`, `self.num_epochs`, `self.learning_rate`.
- `train(...)`:
  - `validate_yaml(resolve_yaml(self))` before `build_model_from_yaml`.
  - `device = resolve_device(loaded.training_runtime.get("device", "auto"))`.
  - `batch_size = int(loaded.training_runtime.get("batch_size", 32))`.
  - `num_epochs = int(loaded.training_runtime.get("num_epochs", 3))`.
  - `lr`: from `loaded.training_config.optimizer_parameters` (e.g. `adamw-lr`), fallback `1e-4`.
- `predict(...)`: read `batch_size` from `loaded.training_runtime` cached on the instance
  (`self._batch_size`) at train time; for unfitted-from-checkpoint predict, default `32`.

### 7. `dashai-frankenstein/src/dashai_frankenstein/models/vit_segmenter.py`

Same pattern as `vit_classifier.py`:
- `__init__`: only `self.frankenstein_yaml`.
- `train(...)`: `validate_yaml` first; read `device`/`batch_size`/`num_epochs`/`lr` from
  `loaded.training_runtime` / optimizer parameters.
- `predict(...)`: use `self._batch_size` cached at train time.

### 8. `dashai-frankenstein/src/dashai_frankenstein/presets.py`

- **Delete** the file. The preset dropdown is removed; `config.py` no longer imports `preset_names`.

### 9. `dashai-frankenstein/pyproject.toml`

- Add `"jsonschema>=4.0"` to `dependencies` (explicit, even though Frankenstein pulls it in).
- Entry-points (`[project.entry-points."dashai.plugins"]`) unchanged — class names and TYPEs
  stay the same.

### 10. `dashai-frankenstein/README.md`

- Update the "Schema (v1: passthrough YAML)" section:
  - Single field `frankenstein_yaml` (single-line string paste).
  - Link to the YAML builder: <https://erickfmm.github.io/frankenstein-transformer/index.html>.
  - State that the YAML is validated against the Frankenstein JSON Schema
    (`additionalProperties: false` + cross-component rules) before launch; errors surface to the
    DashAI user.
  - Remove mentions of `preset`, `device`, `batch_size`, `num_epochs`, `learning_rate`.
- Add a note that a true multiline textarea is tracked as a future upstream improvement to
  DashAI (see "Future work").

## Files to create/modify (summary)

| # | File | Action |
|---|---|---|
| 1 | `src/dashai_frankenstein/config.py` | Modify — single field, remove convenience fields |
| 2 | `src/dashai_frankenstein/validate.py` | **NEW** — `validate_yaml` |
| 3 | `src/dashai_frankenstein/engine.py` | Modify — re-export `resolve_schema`; optional wrapper |
| 4 | `src/dashai_frankenstein/models/base.py` | Modify — validate first; read runtime from `loaded` |
| 5 | `src/dashai_frankenstein/models/decoder.py` | Modify — validate first; keep gen params |
| 6 | `src/dashai_frankenstein/models/vit_classifier.py` | Modify — validate first; runtime from `loaded` |
| 7 | `src/dashai_frankenstein/models/vit_segmenter.py` | Modify — validate first; runtime from `loaded` |
| 8 | `src/dashai_frankenstein/models/mlm.py` | Modify — drop deleted attrs in `__init__` |
| 9 | `src/dashai_frankenstein/presets.py` | **DELETE** |
| 10 | `pyproject.toml` | Modify — add `jsonschema` dep |
| 11 | `README.md` | Modify — single-field docs + link + validation note |

## Verification

1. `conda run -n frankenstein python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings` (Frankenstein suite; the plugin has no tests yet — adding tests is out of scope for this plan but noted as follow-up).
2. Plugin import smoke test (in the DashAI env where `DashAI` is importable):
   `python -c "import dashai_frankenstein; print(dashai_frankenstein.__all__)"`
3. Schema sanity: instantiate `FrankensteinPassthroughSchema` and confirm only
   `frankenstein_yaml` is present.
4. `validate_yaml` against a known-good preset (`configs/mini.yaml`) → passes; against a YAML
   with an unknown top-level key → raises `ValueError` mentioning `additionalProperties`.
5. `validate_yaml` against a YAML violating `hidden_size % num_heads` → raises `ValueError`
   from the loader stage.

## Future work (deferred — upstream DashAI Issue/PR)

A true multiline textarea for `frankenstein_yaml` requires changes to DashAI itself. This is
**not** executed in this plan; it will be evaluated as a future Issue (and possibly a PR via fork)
against the official DashAI repository. The proposed change (for the Issue body):

- **Title**: "Support multiline textarea (`text_field`) for plugin schema fields"
- **Proposal**:
  1. Add `DashAI/back/core/schema_fields/text_field.py` — helper producing a schema field whose
     resolved JSON-schema `type` is `"text"`.
  2. Patch `front/src/components/shared/FormSchemaField.jsx:55` and
     `front/src/components/configurableObject/FormRenderer.jsx:77` so `case "text"` renders
     `<TextInput {...commonProps} multiline minRows={10} />` (MUI `TextField` already accepts
     `multiline`/`minRows` via `InputWithDebounce`→`Input`).
  3. Yup validation in `front/src/utils/paramFormValidation.js:92` already covers `"text"`
     (`Yup.string()`), no change needed.
- Once upstream lands `text_field`, the plugin switches `frankenstein_yaml` from
  `string_field()` to `text_field()` — a one-line change in `config.py`.

## Appendix: DashAI field capabilities (investigation findings)

A read-only investigation of `/home/erick-merino/src/proyecto-plugin/dashAI/DashAI/` confirmed
the following for a plugin using only backend schema declarations (no DashAI source edit):

- **File upload field**: NOT SUPPORTED. No `file_field`/`path_field`/`upload_field` in
  `back/core/schema_fields/`; no `FileInput`/`Dropzone` in `front/.../Inputs/`; the renderer has
  no `file`/`upload`/`path`/`uri` case. `multipart`/`FormData` in the frontend is for
  dataset/job upload, not model parameters. No JSON-schema `format` keyword is honored by the
  renderer.
- **`type: "text"`**: NOT SUPPORTED as multiline. `FormSchemaField.jsx:55` and
  `FormRenderer.jsx:77` have a `case "text"` returning `<TextInput .../>`, but the backend never
  emits `type: "text"` (pydantic emits `"string"`), and `TextInput.jsx` passes no `multiline`
  prop. `InputWithDebounce.jsx:38` → `Input` (`InputStyles.jsx:4`, a `styled(TextField)`)
  receives no `multiline`/`minRows`.
- **`json_schema_extra` passthrough**: NOT SUPPORTED. `schema_field.py:39` hardcodes
  `json_schema_extra = {"placeholder", "display_name"}`. The frontend consumes only: `type`,
  `title`, `description`, `placeholder` (+ `placeholder.optimize`), `enum`, `enumNames`,
  `anyOf`, `parent`, `items`, `minItems`/`maxItems`, `properties`, `minimum`/`maximum`,
  `exclusiveMinimum`/`exclusiveMaximum`, `required`. No `widget`/`ui:widget`/`format`/
  `multiline`/`minRows`/`json_schema_extra`/`x-*` key is read.
- **`array` of strings**: PARTIALLY SUPPORTED, impractical. `ArrayInput.jsx:17,35-42` keeps a
  single `inputValue = value.join(",")` and splits on `","`. Newlines are not split on; commas
  inside YAML corrupt the split; single-line. Unsuitable for large YAML payloads.
- **Precedent for large-text config field**: NONE. All shipped DashAI models (HuggingFace,
  sklearn, PyMC, GGUF/llama) use only scalar/enum/array/component/optimizer field types.
  `pretrained_dir` is an internal `__init__` kwarg, not a user-facing `SCHEMA` field.
- **Dataset/artifact as config carrier**: NOT SUPPORTED. `artifacts.py` defines outputs only;
  there is no `path`/`column`/`dataset` schema-field type. Model `train`/`predict` receive a
  `DashAIDataset` from the run machinery, not from the parameter form.
- **`component_field`/`ClassInput` custom UI**: NOT SUPPORTED. `component_field` yields a
  dropdown + schema-driven subform. Plugin discovery is Python-only
  (`back/plugins/utils.py:209` `entry_points(group="dashai.plugins")`); there is no frontend
  plugin loader or dynamic component registry. A plugin cannot ship React assets.
- **`extraOptions`/`getValues` hooks**: NOT SUPPORTED for plugins. Legacy props in
  `ParameterForm.jsx`/`MainForm.jsx`; `FormSchema.jsx` (current path) does not forward them;
  no caller ever passes a non-null `extraOptions`. Unreachable from a backend schema.

**Bottom line**: the only backend-only mechanism that works today is `type: "string"`
(single-line `TextField`). Hence Option A.