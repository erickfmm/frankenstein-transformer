# AGENTS.md

## Environment & Commands

- **Conda env**: `frankestein` (Python 3.9). Prefix commands: `conda run -n frankestein <cmd>`.
- **Install (local, CUDA 11.8 torch)**: `conda run -n frankestein pip install -e ".[train]"`
- **Run all tests**: `conda run -n frankestein python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings`
- **Run a single test**: `conda run -n frankestein python -m pytest tests/test_optimizer_factory.py -v` or `tests/test_foo.py::TestClass::test_method`
- **CLI**: `conda run -n frankestein frankestein-transformer <subcommand>` (subcommands: `train`, `deploy`, `quantize`, `infer`, `sbert-train`, `sbert-infer`, `transformers-export`, `web-server`)
- **Web UI**: `frankestein-transformer web-server` (Streamlit schema-driven YAML builder)
- **No linter/formatter configured** — match surrounding style; all modules use `from __future__ import annotations` and absolute imports from `src/`.

### CI quirks (`.github/workflows/tests.yml`)

- Matrix is **Python 3.10/3.11/3.12** (conda env is 3.9 — keep code compatible across all four).
- CI installs **CPU-only torch first** (`torch==2.6.0+cpu`), then `requirements.txt` with torch/extra-index lines stripped, then `pip install -e . --no-deps`. Never assume a CUDA torch in tests.
- GPU code paths are guarded by `TORCH_AVAILABLE = find_spec("torch") is not None`; tests skip cleanly without CUDA.

## Architecture

- **Entrypoint**: `src/cli.py` → `frankestein-transformer` (registered in `pyproject.toml` `[project.scripts]`).
- **Package root**: `src/` — `model/`, `training/`, `deploy/`, `sbert/`, `tokenizer/`, `streamlit_gui/`, `utils/`.
- **Model core**: `src/model/tormented_bert_frankestein.py` — `UltraConfig` dataclass + `TormentedBertFrankenstein` backbone. This is the single largest coupling point.
- **Schema (source of truth)**: `src/schema.yaml` is the JSON-Schema entry; it `$ref`s into the **modular** `src/schema/` directory (`_model.yaml`, `_training.yaml`, `_optimizer.yaml`, `_model_class.yaml`, `_tokenizer.yaml`, `_sbert.yaml`, `_base_model.yaml`, `_conditional_rules.yaml`, `_examples.yaml`). The `_model.yaml` is itself an aggregator that `$ref`s into the `src/schema/_model/` subdirectory (`_dims.yaml`, `_norm.yaml`, `_embedding.yaml`, `_attention_titan.yaml`, `_attention_latent.yaml`, `_attention_cca.yaml`, `_attention_sparse.yaml`, `_attention_engram.yaml`, `_model_flat.yaml`) mirroring `src/model/`. ⚠️ There is **no** `configs/schema.yaml` — older docs reference it; the real file is `src/schema.yaml`.
- **Hierarchical model schema**: The YAML `model:` block is nested: `model.dims.*` (core dims + `layer_pattern` + `mode`), `model.norm.*` (`type`, `partial_ratio`), `model.embedding.{factorized,conv}.*`, `model.attention.<mixer>.*` (per-mixer subkeys: `titan`, `mla`, `gqla`, `mlra`, `tucker`, `iha`, `gta`, `mtla`, `cca`, `ccgqa`, `msa`, `sparda`, `engram`), plus flat top-level keys for cross-cutting features (`use_moe`, `num_experts`, `use_bitnet`, `ffn_*`, `ode_*`, etc.). `UltraConfig` remains a **flat** dataclass internally; the loader flattens via `src/utils/config_flatten.py` (`flatten_model_dict`) before `UltraConfig(**kwargs)`.
- **Strict schema**: `additionalProperties: false` at top level and at every nested object. Adding a config key without extending the schema → validation failure.
- **Training presets**: `configs/*.yaml` (named: `mini`, `frankenstein`, `frankesteindecoder`, `standard`, `tinybert`, `embbert`, …) and `configs/examples/*.yaml` (optimizer × architecture combos).
- **Norm layers**: `src/model/norm/` — `factory.py` dispatches on `norm_type`; implementations in `derf.py`, `dynamic_tanh.py`, `rms.py` (RMSNorm + pRMSNorm).
- **Optimizers**: `src/model/optimizer/` — `factory.py` builds the optimizer; one file per family.
- **Attention mixers**: `src/model/attention/` — families live in subpackages `gated/`, `latent/`, `sparse/` plus top-level modules (`standard.py`, `sigmoid.py`, `titan.py`, `retnet.py`, `ode.py`, `engram.py`, `grouped_query_attention.py`, `common.py`). `common.py` hosts `BitLinear`/`BitConv1d`.

## Hard Constraints (respect these always)

1. **Schema is the source of truth.** Every config field must exist in `src/schema/_*.yaml`. Never accept a YAML key the schema doesn't define; never add a code path that reads an undocumented key.
2. **Cross-component compatibility must hold.** Enforced partly by `src/schema/_conditional_rules.yaml` and partly at runtime:
   - `model.dims.hidden_size` must be divisible by `model.dims.num_heads`.
   - `model.dims.num_kv_heads` (for `gqa_attn`) must divide `model.dims.num_heads` exactly.
   - `model.dims.vocab_size` must match the tokenizer's vocab (mismatch corrupts token IDs).
   - `base_model` path requires `training.task` and, for MLM, `tokenizer.name_or_path`.
   - `task: mlm` requires `training.optimizer`; `task: sbert` requires `training.sbert`.
3. **Respect BitNet.** `use_bitnet` **defaults to `True`** — primary/gate `nn.Linear` are replaced by `BitLinear` (ternary weights) unless explicitly disabled. `bitnet_routers=true` **requires** `use_bitnet=true` (enforced in `config_loader._validate_bitnet_flags`). `use_bitnet_conv` only applies when both `use_bitnet` and `model.embedding.conv.enabled` are true. Changing BitNet wiring touches `src/model/attention/common.py`, `tormented_bert_frankestein.py`, and `src/deploy/quantization.py` + `bitnet_gguf_export.py`.
4. **`model.norm.type` enum is `[layer_norm, dynamic_tanh, derf, rms_norm, prms_norm, flash_norm]`.** `rms_norm` (RMSNorm) and `prms_norm` (partial RMSNorm, arXiv:1910.07467) are implemented in `src/model/norm/rms.py` and dispatched by `norm/factory.py`. `prms_norm` uses the `model.norm.partial_ratio` config field (default `0.0625`, range `(0, 1]`) — validated in `UltraConfig.__post_init__` (flattened from `norm.partial_ratio`). `flash_norm` (FlashNorm, arXiv:2407.09577) is implemented in `src/model/norm/flash.py` as a *weightless* RMSNorm (no learnable per-dim scale `g`); it composes with partial-RMS via `model.norm.flashnorm_partial_ratio` (default `0.0`, range `[0, 1]`) — validated in `UltraConfig.__post_init__` (flattened from `norm.flashnorm_partial_ratio`). When `norm_type == "flash_norm"` and FFN is non-GLU + non-MoE, `HybridLayer` fuses the pre-FFN norm into the FFN's first projection via `FlashNormLinear` (or `FlashNormBitLinear` under BitNet) — Prop. 2 of the paper (deferred RMS). GLU/MoE paths keep a standalone `FlashNorm` for `norm2`.
5. **`fasa_attn` and `sparge_attn` are eval-only.** They raise `RuntimeError` during training (`TRAINING_FREE_LAYERS` set in `tormented_bert_frankestein.py`). Never put them in a training `model.dims.layer_pattern`.
6. **`model_class: frankesteindecoder` forces `model.dims.mode: decoder`** at runtime (preset in `tormented_bert_frankestein.py`). Don't set `mode: encoder` with this class.
7. **Optimizer parameters use prefixed keys**: `<optimizer_class>-<group>_<param>` (e.g. `adamw-lr_embeddings`, `muon-ns_steps`, `sgd_momentum-momentum`). The prefix list lives in `src/schema/_optimizer.yaml` under `by_optimizer`. A new optimizer must add a `prefix` entry there. Parameter groups are: `embeddings`, `norms`, `attention`, `other` (the `ode`/`retnet`/`mamba` groups were removed; their params route into `attention`).
8. **`training.task` is required** (`mlm` or `sbert`). Legacy top-level optimizer keys are gone — use `training.optimizer` (MLM) or `training.sbert` (SBERT).
9. **Both `configs/*.yaml` and `configs/examples/*.yaml` are smoke-tested** by `tests/test_yaml_examples.py` (one injected test per file). Adding a new optimizer family or mixer **requires** an example YAML or CI breaks. The `examples.md` file is skipped (only `*.yaml`/`*.yml` are globbed).
10. **`ffn_activation` enum is the source of truth** (43 values in `src/schema/_model.yaml`, mirrored in `src/model/activation_function/factory.py` `ALL_ACTIVATIONS`). Adding an activation requires extending the enum **and** the factory dispatch **and** `UltraConfig` validation. GLU variants (`swiglu`/`geglu`/`reglu`) are gated FFN units — `get_activation` raises on them; `HybridLayer` builds a `GatedFFN` instead. The `ffn_activation_config` nested object (`additionalProperties: false`) groups learnable-activation params (RAF degrees/version/init, PReLU init, etc.). See `docs/specs/activations.md`.

## Where to Edit by Feature Type

Read recent commits on these paths first (`git log --oneline -20 -- <path>`).

| Feature / Fix | Primary files to read & edit |
|---|---|
| Add a new **attention mixer** | `src/model/attention/<family>/new_attn.py`, `src/model/attention/__init__.py`, `src/model/tormented_bert_frankestein.py` (registry dict), `src/schema/_model/_dims.yaml` (`layer_pattern` enum) + a new `src/schema/_model/_attention_<mixer>.yaml` if the mixer has config knobs (and `$ref` it from `_model.yaml`), `src/utils/config_flatten.py` (add the mixer to `MIXER_MAP`), `configs/examples/` (add example), `tests/test_schema_attention_layers.py`, `tests/test_attention_refactor.py` |
| Add a new **optimizer** | `src/model/optimizer/<new_opt>.py`, `src/model/optimizer/factory.py`, `src/model/optimizer/__init__.py`, `src/schema/_optimizer.yaml` (enum + `by_optimizer` prefix entry), `configs/examples/` (add example), `tests/test_optimizer_factory.py`, `tests/test_yaml_examples.py` |
| Modify **training config / schema** | `src/schema/_*.yaml` / `src/schema/_model/_*.yaml` (the relevant submodule), `src/training/config_loader.py`, `src/utils/config_flatten.py` (if nesting changes), `configs/README.md`, `src/streamlit_gui/app.py`, `tests/test_config_loader.py`, `tests/test_yaml_examples.py` |
| Modify **model architecture / UltraConfig** | `src/model/tormented_bert_frankestein.py`, `src/schema/_model/_*.yaml` + `src/utils/config_flatten.py` (if nesting changes), `tests/test_model_config.py`, `tests/test_model_variants.py` |
| Modify **training loop / trainer** | `src/training/trainer.py`, `src/training/main.py`, `tests/test_mlm_masking.py`, `tests/test_model_training_fake_data.py` |
| Modify **SBERT training/inference** | `src/sbert/train_sbert.py`, `src/sbert/inference_sbert.py`, `src/schema/_sbert.yaml`, `configs/modernbert_sbert*.yaml`, `tests/test_sbert_thermal_offload.py` |
| Modify **BitNet / deployment / quantization** | `src/deploy/deploy.py`, `src/deploy/inference.py`, `src/deploy/quantization.py`, `src/deploy/bitnet_gguf_export.py`, `src/model/attention/common.py` (`BitLinear`/`BitConv1d`), `src/model/tormented_bert_frankestein.py` (BitNet wiring), `tests/test_bitnet_*.py` |
| Modify **CLI / entrypoint** | `src/cli.py`, `pyproject.toml` (scripts entry), `tests/test_cli_parser.py`, `tests/test_cli_gpu_temp_flags.py` |
| Add/modify **GPU thermal guard** | `src/utils/gpu_temp_guard.py`, `src/training/trainer.py`, `src/cli.py`, `tests/test_gpu_temp_guard*.py` |
| Modify **positional encodings** | `src/model/attention/rope.py`, `src/model/attention/hope.py`, `src/model/attention/common.py`, `src/schema/_model/_attention_titan.yaml` (`positional_encoding` enum), `tests/test_positional_encodings.py` |
| Modify **normalization layers** | `src/model/norm/factory.py`, `src/model/norm/<impl>.py` (`rms.py`, `derf.py`, `dynamic_tanh.py`), `src/schema/_model/_norm.yaml` (`type` enum), `tests/test_common_modules.py` |
| Modify **activation functions** | `src/model/activation_function/factory.py`, `src/model/activation_function/<family>.py` (`common.py`, `rectified.py`, `exponential.py`, `learnable.py`, `glu.py`), `src/model/tormented_bert_frankestein.py` (`HybridLayer` FFN wiring, `_validate_ffn_activation_config`), `src/schema/_model/_model_flat.yaml` (`ffn_activation` enum + `ffn_activation_config` object), `configs/examples/activation_*.yaml`, `tests/test_activation_functions.py`, `docs/specs/activations.md`, paper `docs/paper*/appendices/annex-8-*.tex` + `annex-summary-tables.tex` |
| Modify **streaming dataset** | `src/training/streaming_mlm_dataset.py`, `src/utils/storage_manager.py` |
| Modify **Streamlit web interface** | `src/streamlit_gui/app.py` |
| Update **paper / docs** | `docs/paper.tex`, `docs/paper-es.tex`, `docs/bibliography/*.bib`, `docs/bibliography/*.md`, `docs/specs/*.md` |

## Documentation

- **Docstrings**: Google-style (Args, Returns, Raises, Attributes). Sphinx autodoc renders them to Read the Docs (`docs/source/api/`).
- **Generate API docs locally**: `bash docs/build_docs.sh` (Sphinx → `docs/_build/html`); Read the Docs hosts the same at https://frankestein-transformer.readthedocs.io
- **Specs**: `docs/specs/*.md` — architecture, attention-mixers, optimizers, schema-reference, cli-reference, deployment, sbert-workflows, training-safety. Update the relevant spec when adding a mixer/optimizer.
- **Academic refs**: `docs/paper.tex`, `docs/bibliography/*.md` contain formulations and citations for all mixers/optimizers.
- **Config reference**: `configs/README.md` has the full optimizer parameter reference and preset walkthrough.
