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

## Schema (v1: passthrough YAML)

Each model exposes a minimal pydantic schema. The primary field,
`frankenstein_yaml`, is a string containing a full Frankenstein training YAML,
validated by Frankenstein's own config loader + JSON Schema (the Frankenstein
schema remains the single source of truth). A `preset` dropdown is populated
from Frankenstein's bundled `configs/*.yaml`; `device`, `batch_size`, and
`num_epochs` are convenience overrides merged into the YAML before validation.

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
