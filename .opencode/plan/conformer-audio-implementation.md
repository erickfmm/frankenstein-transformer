# Plan: `conformer_attn` mixer + `frankenstein_audio` class (ASR + audio classification)

## Papers

1. **arXiv:2005.08100** — "Conformer: Convolution-augmented Transformer for Speech Recognition" (Gulati et al., Interspeech 2020). Source of the mixer and the audio frontend.
2. **arXiv:1211.3711** — "Sequence Transduction with Recurrent Neural Networks" (Graves, 2012). RNN-Transducer loss.
3. **arXiv:1904.08779** — "SpecAugment" (Park et al., 2019). Audio augmentation.
4. **arXiv:1906.02762** — "Understanding and Improving Transformer …" (Macaron-Net, Lu et al., 2019). Half-step FFN pair.
5. **arXiv:1901.09465 / arXiv:1910.05895** — Transformer-XL / "Transformers without tears" relative sinusoidal PE and pre-norm residuals (already partly in repo).

## Paper 1 (Conformer) — key technical details

### Block (Sec. 2.4, Fig. 1)
For block input `x`:
```
x̃ = x + ½·FFN₁(x)            # macaron half-step FFN, pre-norm
x' = x̃ + MHSA(LN(x̃))         # multi-head self-attention, pre-norm, relative sinusoidal PE
x'' = x' + Conv(LN(x'))       # convolution module, pre-norm
y  = LN(x'' + ½·FFN₂(x''))    # second macaron FFN, final LayerNorm
```

### MHSA (Sec. 2.1, Fig. 3)
- Standard scaled dot-product MHSA, pre-norm residual unit, dropout.
- **Relative sinusoidal positional encoding** from Transformer-XL (generalizes across utterance lengths).
- The repo already owns the PE families; `conformer_attn_use_pe` flag selects.

### Convolution module (Sec. 2.2, Fig. 2)
```
LN → pointwise Conv1d (d → 2d) → GLU (→ d) → depthwise Conv1d (kernel k) → BatchNorm → Swish → pointwise Conv1d (d → d) → dropout
```
- kernel size 32 best (sweep {3,7,17,32,65}, Table 7).
- **Causal padding when `mode == "decoder"`** (streaming-safe), symmetric padding otherwise.

### FFN module (Sec. 2.3, Fig. 4)
```
LN → Linear(d → 4d) → Swish → dropout → Linear(4d → d) → dropout, ½-step residual
```

### Frontend (Sec. 2, 3.1, Fig. 1)
- 80-channel log-mel filterbanks, 25 ms window, 10 ms stride.
- **SpecAugment**: `F = 27`, 10 time masks, max time-mask ratio `pS = 0.05`.
- **Convolution subsampling**: two `Conv2d(k=3, stride=2)` → 4× time reduction (10 ms → 40 ms), then linear to `d`, dropout.
- Reference sizes (Table 1): S/M/L = 16/17/17 layers, d = 144/256/512, heads 4/4/8, kernel 32, decoder dim 320/640/640.

### Ablations used as design justification
- Table 3: removing the conv block is the largest degradation; removing the Macaron FFN pair costs ≈0.2/0.5 WER.
- Table 5: half-step vs **full-step** residuals ≈ 0.0–0.1 WER (1.9/4.4 vs 1.9/4.5 dev). ⇒ repo-integrated decomposition (below) is justified.

## Design decisions (user-approved)

1. **Repo-integrated Conformer block.** `conformer_attn` owns `½FFN₁ + MHSA + ConvModule` with internal pre-norm residuals; the shared `HybridLayer` norm₂+FFN plays FFN₂ at full step. This keeps MoE, mHC, MoD, flash-norm fusion, BitNet, every norm/optimizer/residual working with the mixer. Deviation documented in the annex, justified by Table 5.
2. **ASR: CTC *and* RNN-T.** `training.asr.loss: ctc | transducer` (default `ctc`). Transducer = 1-layer LSTM prediction network (`decoder_dim`, `decoder_layers`, `decoder_dropout`) + joint network. RNN-T loss implemented in pure torch (log-domain DP, autograd-friendly) with `torchaudio.functional.rnnt_loss` fast path when available. Greedy CTC-collapse and greedy RNN-T decoding + CER.
3. **torchaudio is an optional `audio` extra** (`torchaudio>=2.1`, import-guarded, loose pin so it never fights the torch extras). Frontend uses `torchaudio.transforms` when importable, else a pure-torch `torch.stft` + HTK/Slaney mel fallback. CPU CI stays torchaudio-free (D7).
4. **Streamlit: full audio UI.** Task guard (`asr`/`audio_classification` ⇒ `frankenstein_audio`) plus generic `render_object` rendering of `audio:` and `audio_dataset:` blocks with bilingual titles.
5. **Signal generality**: `audio.feature: logmel | spectrogram | raw`. `raw` skips mel and feeds multichannel 1-D signals (EEG, EMG, accelerometer, …) through the same conv subsampling → documented, at least for classification.
6. **Data**: `astroza/chilean-jokes-festival-de-vina` (HF, `audio`@16 kHz + `text`, 50 rows, ~1.1 MB) for the ASR example/CI-smoke config. `DummyAudioDataset` (synthetic waveforms) for tests/e2e/smoke, mirroring `DummyImageDataset`.
7. **KV cache**: `conformer_attn` gets a `CACHE_SPECS` entry with `supported=False` (notes: depthwise-conv state caching is future work); D5 fallback keeps `generate()` working on the legacy path.

## File-by-file implementation map

### T1 — `conformer_attn` mixer
| File | Change |
|---|---|
| `src/model/attention/conformer.py` (new) | `ConformerAttention`: ½FFN₁ (own norms, `ffn_hidden_size`, swish, BitLinear-aware) → MHSA (own `q/k/v/out` projections, repo PE via `common.apply_pe_*`, `numerics.module_softmax`, decoder triu mask) → `ConformerConvModule` (LN → pointwise → GLU → depthwise 1-D conv (causal/ symmetric padding) → BatchNorm1d → Swish → pointwise → dropout), internal pre-norm residuals; returns the delta so the outer residual machinery is preserved. `dense_forward` alias for uniformity if cheap. |
| `src/model/attention/__init__.py` | export |
| `src/model/hybrid_layer.py` | registry `"conformer_attn": ConformerAttention` |
| `src/schema/_model/_attention_conformer.yaml` (new) | `kernel_size` (default 32), `macaron` (default true), `conv_norm` (enum `batch|group|layer`, default `batch`) |
| `src/schema/_model/_model.yaml` | `$ref` the `conformer` sub-object |
| `src/schema/_model/_dims.yaml` | enum value + **both** bilingual description tables |
| `src/utils/config_flatten.py` | `_ATTENTION_MIXER_RENAMES["conformer"]` |
| `src/model/config.py` | `conformer_kernel_size`, `conformer_macaron`, `conformer_conv_norm`, `conformer_attn_use_pe` + `__post_init__` validation |
| `src/model/cache/capabilities.py` | `CACHE_SPECS["conformer_attn"] = CacheSpec(supported=False, …)` |
| `configs/examples/mixer_conformer.yaml` (new) | text encoder smoke |
| `tests/test_schema_attention_layers.py` | expected set + sub-key assertions |
| `tests/test_conformer_attention.py` (new) | shapes, causality, manual-formula parity, PE on/off, macaron on/off, BitNet path, conv-norm variants, fixed-softmax dispatch |
| `src/utils/param_categories.py` | (if needed) keep `mixer.*` → attention |

### T2 — `frankenstein_audio` model class
| File | Change |
|---|---|
| `src/model/frankenstein_audio.py` (new) | `AudioFrontend` (torchaudio-optional log-mel, pure-torch fallback, spectrogram, raw multichannel; Conv2d×2 subsampling → linear → dropout), `[CLS]`-free (classification uses GAP or learned CLS per `pooling_mode`), duplicated `_run_encoder`-style layer loop so AttnRes/mHC/Hyperloop/loops inherit, forced `mode: encoder`, `head` = vocab/CTC projection, `audio_classification_head`, `TransducerHead` (`PredictionNetwork` LSTM + joint projection), runtime task dispatch like ViT. |
| `src/model/__init__.py` | export |
| `src/schema/_model_class.yaml` | enum + bilingual descriptions |
| `src/schema/_audio.yaml` (new) | `feature` (`logmel|spectrogram|raw`), `sample_rate` (16000), `n_mels` (80), `window_ms` (25), `stride_ms` (10), `in_channels` (1), `subsampling_factor` (4), `pooling_mode` (`cls|gap`), `num_classes`, `conv_kernel`, `conv_channels` → flattened as `audio_*`, `pooling_mode`, `num_classes` |
| `src/schema/_audio_dataset.yaml` (new) | `dataset_name`/`dataset_dir`, `split`, `audio_column` (`audio`), `text_column` (`text`), `label_column`, `max_duration_sec`, `resample`, `spec_augment.{enabled,freq_mask_width,time_masks,time_mask_ratio}` |
| `src/schema.yaml` | `audio:` + `audio_dataset:` refs |
| `src/utils/config_flatten.py` | `flatten_audio_dict` |
| `src/model/config.py` | audio fields, `__post_init__` validation |
| `src/training/config_loader.py` | `valid_model_classes`, `audio_tasks`, audio block flattening, `LoadedTrainingConfig.audio_config/audio_dataset_config` |
| `src/engine.py` | `build_model` branch |
| `src/deploy/deploy.py` | model-class dispatch |
| `src/deploy/transformers_export.py` | explicit rejection for audio (like SBERT), documented |
| `tests/test_frankenstein_audio.py` (new) | forwarding per task, forced mode, gradients, pooling, frontends incl. **EEG-style raw multichannel**, mHC/AttnRes/multi-mixer combos, `configure_optimizers` |
| `tests/test_model_variants.py` | audio mode-forcing case |

### T3 — tasks, datasets, training
| File | Change |
|---|---|
| `src/training/audio_dataset.py` (new) | `AudioDataset` (HF `datasets`, audio decode to arrays, char-tokenizer built from `text_column`), `DummyAudioDataset` (synthetic waves/transcripts/labels), `SpecAugment`, length helpers |
| `src/training/trainer.py` | `compute_asr_loss` (CTC via `F.ctc_loss`; RNN-T via `rnnt_loss`), `compute_audio_classification_loss` (via `_task_cross_entropy`), `_forward_audio_*`, greedy decoders + CER, dispatch + telemetry |
| `src/training/rnnt.py` (new) or `src/model/asr/` | pure-torch log-domain RNN-T loss + greedy decode, torchaudio fast path |
| `src/engine.py` | `_train_audio`, dispatch |
| `src/training/main.py` | `_run_audio_task` |
| `src/schema/_training.yaml` | task enum (`asr`, `audio_classification`) + bilingual descriptions + `asr` sub-block ref |
| `src/schema/_audio_asr.yaml` (new) | `loss` (`ctc|transducer`), `decoder_dim`, `decoder_layers`, `decoder_dropout` |
| `src/schema/_conditional_rules.yaml` | audio tasks ⇒ `frankenstein_audio` + required `audio`/`audio_dataset`; `asr` requires `text_column`-style dataset |
| `src/streamlit_gui/app.py` | guard + audio/audio_dataset rendering |
| `configs/examples/audio_asr_conformer.yaml` (new) | jokes dataset, CTC |
| `configs/examples/audio_asr_transducer.yaml` (new) | jokes dataset, RNN-T |
| `configs/examples/audio_classification_conformer.yaml` (new) | dummy fallback |
| `tests/test_audio_dataset.py`, `tests/test_asr_losses.py`, `tests/test_audio_classification.py` (new) | shapes, mel equivalence (skipif torchaudio), SpecAugment, CTC + RNN-T (vs torchaudio skipif + hand-computed tiny case), greedy decode/CER |
| `tests/test_yaml_examples.py` | task set incl. `text_classification`, `asr`, `audio_classification` |
| `configs/README.md` | new presets |

### T4 — estimator + e2e
| File | Change |
|---|---|
| `src/utils/param_categories.py` | `audio_frontend`/`subsampler` → embeddings, `transducer`/audio heads → head |
| `full_tests/param_count_check.py` | auto-covers new example configs |
| `full_tests/run_e2e.py` | conformer across encoder/decoder/ViT, audio ctc + transducer + classification |
| `full_tests/param_truth.json` | regenerate |
| website `ft-param-estimator.js` | in T6 |

### T5 — papers
| File | Change |
|---|---|
| `docs/paper/figures/mixers/conformer.mmd` (new) | **mixer diagram** (Macaron → MHSA → Conv → FFN₂) |
| `docs/paper/figures/audio_frontend.mmd` (new) | waveform → log-mel → conv subsampling → blocks |
| `docs/paper/sections/annex-17-conformer.tex` (new) | formulations, pseudocode, diagram, ablation summary, deviation table |
| `docs/paper/sections/annex-18-audio.tex` (new) | class, frontend, tasks (CTC/RNN-T), **generic signals / EEG**, configs |
| `docs/paper/paper.tex` | input list |
| `docs/paper/sections/annex-summary-tables.tex` | mixer row + audio rows |
| `docs/bibliography/{attention_types,other}.bib` | Gulati2020, Dai2019, Park2019, Lu2019 |
| `docs/paper-es/...` | full ES mirror |
| `docs/architecture.md` | §7 fields, model classes, mixer tables, extension recipes |

### T6 — sync-website
Run `.opencode/command/sync-website.md` as a subtask; hand-edit the site-only `ft-param-estimator.js` for conformer + audio formulas; pass all validation gates.

### T7 — validation
`uv run --extra cpu --extra train python -m pytest tests/ …`, time-boxed `full_tests/run_e2e.py`, count-reference sweep ("44 mixers", "3 model classes", "7 tasks").

## Gates
- `test_schema_consistency`, `test_yaml_examples`, `test_schema_attention_layers`, `test_kv_cache_*` (CACHE_SPECS parity), `test_model_variants`, new conformer/audio/asr suites.
- CPU-only; torchaudio not required.

## Commits (conventional, after each green phase)
1. `feat(mixer): add conformer_attn convolution-augmented attention`
2. `feat(model): add frankenstein_audio with ASR and audio classification`
3. `feat(training): wire asr/audio_classification tasks, datasets and losses`
4. `test(e2e): cover conformer mixer and audio tasks`
5. `docs(paper): document conformer mixer and audio support (EN/ES)`
6. `chore(website): sync schema, examples, paper and estimator`
