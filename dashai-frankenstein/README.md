# dashai-frankenstein

A [DashAI](https://docs.dash-ai.com/) plugin that registers
[Frankenstein Transformer](https://github.com/erickfmm/frankenstein-transformer)
model classes as DashAI components, so end users can train, evaluate, predict,
save, and load them from the DashAI UI.

## Components registered

| Entry point | Class | DashAI base | Binds to task |
|---|---|---|---|
| `frankenstein_mlm` | `FrankensteinMLMModel` | `BaseModel` | `TextClassificationTask` |
| `frankenstein_decoder` | `FrankensteinDecoderModel` | `BaseGenerativeModel` | `TextToTextGenerationTask` (inference only) |
| `frankenstein_pretrainer` | `FrankensteinPretrainer` | `BaseModel` | `MaskedLanguageModelingTask` (provided by this plugin) |
| `frankenstein_causal_lm` | `FrankensteinCausalLMModel` | `BaseModel` | `CausalLMPretrainingTask` (provided by this plugin) |
| `frankenstein_vit_cls` | `FrankensteinViTClassifier` | `BaseModel` | `ImageClassificationTask` |
| `frankenstein_vit_seg` | `FrankensteinViTSegmenter` | `BaseModel` | `SegmentationTask` (provided by this plugin) |
| `segmentation_task` | `SegmentationTask` | `BaseTask` | (new task provided by this plugin) |
| `masked_language_modeling_task` | `MaskedLanguageModelingTask` | `BaseTask` | (new task provided by this plugin) |
| `causal_lm_pretraining_task` | `CausalLMPretrainingTask` | `BaseTask` | (new task provided by this plugin) |

## The `use_dashai_dataset` checkbox

Every Frankenstein model component exposes a **Use DashAI dataset** checkbox
(default: checked) in its configuration form:

- **Checked (default)**: training consumes the DashAI run dataset
  (`x_train`/`y_train`) — the dataset section of the Frankenstein JSON, if
  any, is overridden. Text is tokenized by the plugin's dataset adapters
  (masking for MLM, plain ids for causal LM, tokenized+labels for
  classification) and passed to the engine as a pre-built DataLoader.
- **Unchecked**: the Frankenstein engine itself loads the corpus from the
  config pasted in `frankenstein_json`:
  - NLP tasks (`mlm`, `causal_lm`, `text_classification`) use the
    **`text_dataset`** block (HF hub `dataset_name`, `split`,
    `text_column`, optional `label_column` + `use_labels`, `data_dir` with
    parquet/json, `streaming`, `max_samples`).
  - Vision tasks use the **`vision_dataset`** block (HF `dataset_name` or
    local `dataset_dir`); without either, a dummy smoke dataset is used.

## MLM pretraining (Masked Language Modeling)

The `FrankensteinPretrainer` component runs **real BERT-style MLM
pre-training** (`task: mlm` in the Frankenstein engine) on a DashAI text
dataset:

1. Create a dataset with a single `Text` column.
2. Create a `MaskedLanguageModelingTask` session and select the same text
   column as input **and** output (self-supervised target).
3. Paste a Frankenstein config (one-line JSON) with `tokenizer.name_or_path`
   set — the text is tokenized + 15%-masked at collate time (80% [MASK] /
   10% random / 10% keep) and consumed by the engine's
   `TitanTrainer.compute_mlm_loss`.

Every optimizer update streams the full telemetry (loss, masked-token
accuracy, learning rate, gradient norms per block, GPU temp/power/mem,
throughput) into the DashAI run's metric store; the trained backbone is
saved via the run's artifact and can be loaded later for classification
fine-tuning with `FrankensteinMLMModel`.

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

### Co-install compatibility with DashAI

`dashai-frankenstein >= 0.4.0` (with `frankenstein-transformer >= 1.3.0`)
allows **transformers 4.x and 5.x** (`>=4.45,<6`), so installing the plugin
into an existing DashAI environment keeps DashAI's modern stack intact
(transformers 5 / huggingface-hub 1.x / diffusers 0.40) with no downgrades
and no `pip check` conflicts.

If your environment must stay on transformers 4.x, pin
`diffusers<0.40` alongside it — diffusers 0.40.0 requires
`huggingface-hub>=1.23`, which transformers 4.x (hub `<1.0`) cannot provide;
diffusers 0.39 works with either hub series.

## Architecture

See `docs/dashai-plugin-audit.md` in the Frankenstein repo for the full
integration design (§5 component designs, §6 phased plan, §7 Frankenstein
changes). This package is the Phase 1–3 adapter layer; it consumes the
Frankenstein engine API (`src.engine`) added in Phase 0.
