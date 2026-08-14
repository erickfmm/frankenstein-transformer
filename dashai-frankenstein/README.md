# dashai-frankenstein

A [DashAI](https://docs.dash-ai.com/) plugin that registers
[Frankenstein Transformer](https://github.com/erickfmm/frankenstein-transformer)
model classes as DashAI components, so end users can train, evaluate, predict,
save, and load them from the DashAI UI.

## Components registered

| Entry point | Class | DashAI base | Binds to task |
|---|---|---|---|
| `frankenstein_mlm` | `FrankensteinMLMModel` | `BaseModel` | `TextClassificationTask` |
| `frankenstein_decoder` | `FrankensteinDecoderModel` | `BaseGenerativeModel` | `TextToTextGenerationTask` |
| `frankenstein_vit_cls` | `FrankensteinViTClassifier` | `BaseModel` | `ImageClassificationTask` |
| `frankenstein_vit_seg` | `FrankensteinViTSegmenter` | `BaseModel` | `SegmentationTask` |
| `segmentation_task` | `SegmentationTask` | `BaseTask` | (new task provided by this plugin) |

## Schema (v1: passthrough JSON)

Each model exposes a minimal pydantic schema with a **single user-facing field**:
`frankenstein_json`, a string containing a full Frankenstein training config as a
**single-line JSON**. The Frankenstein JSON Schema is the source of truth — the
JSON is validated against it (`additionalProperties: false` + enums) and
Frankenstein's config loader (cross-component constraints) **before** any
train/inference launches. Errors surface to the DashAI user as a readable
`ValueError`.

Build your YAML with the
[Frankenstein YAML builder](https://erickfmm.github.io/frankenstein-transformer/index.html),
**convert it to a one-line JSON string**, and paste it into the field:

```bash
python -c "import yaml,json,sys; print(json.dumps(yaml.safe_load(open(sys.argv[1]))))" my_config.yaml
```

Training parameters (`device`, `batch_size`, `num_epochs`, learning rate) are read
from the config's `training_runtime` block and optimizer parameters — they are NOT
separate DashAI form fields. Generation parameters (`max_new_tokens`,
`temperature`, `top_k`) on the decoder component are kept as DashAI fields (they
are inference-time, not training-time, and the Frankenstein schema has no home
for them).

> **Note:** The field is a single-line text input (DashAI does not yet support
> a multiline textarea for plugin schema fields), which is why the config is
> passed as a one-line JSON string rather than a multiline YAML document. A
> true multiline textarea is tracked as a future upstream improvement to DashAI.

## Install

```bash
pip install dashai-frankenstein            # from PyPI once published
# or, from this repo:
pip install -e ./dashai-frankenstein
```

DashAI discovers the plugin via the `dashai.plugins` entry-points group on
startup — no DashAI source edits required.

## Architecture

See `docs/dashai-plugin-audit.md` in the Frankenstein repo for the full
integration design (§5 component designs, §6 phased plan, §7 Frankenstein
changes). This package is the Phase 1–3 adapter layer; it consumes the
Frankenstein engine API (`src.engine`) added in Phase 0.
