# Plan: Implement `frankenstein_vit` — Vision Transformer with Patch Prediction, Classification, and Segmentation

## Papers

1. **arXiv:2010.11929** — "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., ICLR 2021). The original Vision Transformer (ViT).
2. **arXiv:2503.19108** — "Your ViT is Secretly an Image Segmentation Model" (Kerssies et al., 2025). Encoder-only Mask Transformer (EoMT) for segmentation.

## Paper 1: ViT (arXiv:2010.11929) — Key Technical Details

### Patch embedding (Eq. 1)
- Input image `x ∈ R^(H×W×C)` reshaped into `N = (H·W)/P²` flattened patches `x_p ∈ R^(N×(P²·C))`.
- Trainable linear projection `E ∈ R^(P²C × D)` maps patches to `D` dimensions → "patch embeddings" `(N, D)`.
- Implementation: `Conv2d(C, D, kernel_size=P, stride=P)` is mathematically equivalent to flatten+linear. Output `(B, D, H/P, W/P)` → flatten to `(B, N, D)`.

### Positional embedding
- **Learnable** `nn.Parameter`, **1D** (raster order), **not** sinusoidal or 2D.
- Shape `(1, N+1, D)` — the `+1` is for the `[class]` token.
- Added once at the stem (after patch embedding, before the encoder).
- At fine-tuning with higher resolution: bicubic interpolation of the positional embedding grid.
- Paper tested 1D, 2D, and relative pos-emb; found "little to no difference" (Table 8). Default = 1D learnable.

### [CLS] token (Eq. 4)
- Learnable embedding `x_class` of shape `(1, 1, D)`, prepended to patch embeddings → `(B, N+1, D)`.
- Output at position 0 (`z_0^L`) serves as the image representation `y` → fed to classification head.
- Appendix D.3: Global Average Pooling (GAP) also works but needs different LR. Both are valid.

### Transformer encoder (Eqs. 2, 3)
- **Pre-norm** (LN before every block, Wang et al. 2019 / Baevski & Auli 2019 style).
- Each block:
  - `z'_l = MSA(LN(z_{l-1})) + z_{l-1}`  (Eq. 2)
  - `z_l  = MLP(LN(z'_l)) + z'_l`        (Eq. 3)
- MLP = **two linear layers** with **GELU** non-linearity, hidden size **4·D**.
- Residual connections after every block.
- Standard scaled-dot-product MSA: `D_h = D/num_heads` (Eq. 5), `A = softmax(QK^T/√D_h)` (Eq. 6), `MSA(z) = [SA_1(z);...;SA_k(z)]U_msa` (Eq. 7).

### Model variants (Table 1)
| Model    | Layers | D    | MLP size | Heads | Params |
|----------|--------|------|----------|-------|--------|
| ViT-Base | 12     | 768  | 3072     | 12    | 86M    |
| ViT-Large| 24     | 1024 | 4096     | 16    | 307M   |
| ViT-Huge | 32     | 1280 | 5120     | 16    | 632M   |

MLP size = exactly 4·D in all variants.

### Classification head
- **Pre-training**: MLP with one hidden layer + tanh: `Linear(D, D_h) → tanh → Linear(D_h, K_pretrain)`.
- **Fine-tuning**: remove entire pre-training head, attach **zero-initialized** `Linear(D, K_downstream)`.
- Input is always `z_0^L` (the `[class]` token output at position 0).

### Masked patch prediction (self-supervised, Appendix B.1.2)
- Corrupt **50%** of patch embeddings (not 15% — they tried 15%, it was slightly worse).
- Per corrupted patch (BERT recipe):
  - 80% → replace with learnable `[mask]` embedding.
  - 10% → replace with a random other patch's embedding.
  - 10% → keep as-is.
- **Prediction targets** (three options tested):
  1. **3-bit mean color** (best): 512-way classification per corrupted patch. Cross-entropy.
  2. **4×4 downsized 3-bit patch**: 16 predictions of 512 colors. Cross-entropy.
  3. **Full-patch L2 regression**: 256 regressions on 3 RGB channels. MSE. Slightly worse.
- Optimizer: Adam, base LR 2e-4, 10k warmup, cosine decay, batch 4096, 1M steps.
- ViT-B/16 self-supervised reaches 79.9% on ImageNet (2% above scratch, 4% below supervised).

### Training (classification pre-training)
- Adam (β1=0.9, β2=0.999), batch 4096, weight decay 0.1, linear LR warmup (10k) + linear decay.
- Resolution 224. Cross-entropy loss (implied).
- Dropout after every dense layer except qkv and after pos-emb addition.

### Fine-tuning (classification)
- SGD + momentum 0.9, batch 512, no weight decay, cosine LR decay, grad clip at norm 1.
- Resolution 384 (default) or 512/518 (headline).
- Remove pre-training head, attach zero-init `Linear(D, K)`.
- Optional Polyak averaging factor 0.9999.

### Hybrid model
- Replace patch-extraction stem with BiT ResNet (GroupNorm, standardized convs).
- Feed feature map as tokens with patch size 1×1.
- Rest of ViT unchanged. Hybrids help only for small models/compute.

### Image preprocessing
- Training resolution 224. Fine-tune 384 default. Specific mean/std not given (inherited from BiT).
- Patch sizes: 16×16 (ViT-*/16), 32×32 (ViT-*/32), 14×14 (ViT-H/14).

## Paper 2: EoMT (arXiv:2503.19108) — Key Technical Details

### Core insight
Task-specific components (conv adapter, pixel decoder, transformer decoder) become redundant with large ViT + strong pre-training. EoMT strips them all away.

### Architecture (Fig. 2, Sec. 3.3)
- Plain ViT backbone **split into two groups**:
  - First **L₁** blocks: process **only patch tokens** (unchanged ViT).
  - Remaining **L₂** blocks: process **patch tokens + K learnable queries** jointly via standard MHSA.
- **No separate decoder**. L₂ ViT blocks *are* the decoder.
- MHSA inside L₂ blocks performs all four attention quadrants in one op:
  - patch-to-patch, query-to-query, query-to-patch, patch-to-query.
- L₂ values: ViT-S/B=3, ViT-L=4, ViT-g=5 (Sec. A.1).
- Query counts K: 200 (panoptic/instance), 100 (semantic).

### Mask module (Sec. 3.3)
- Linear layer on each query → class logits `c_i ∈ R^C`.
- 3-layer MLP on each query → mask embedding `q̃_i`.
- Dot product of `q̃_i` with upscaled features `F̃_4` → mask logits `m_i ∈ R^(H/4 × W/4)`.
- Feature upsampling: ViTDet-style transposed/depthwise conv stack (16→4).

### Masked self-attention (training only)
- Before each L₂ block, mask module predicts intermediate masks.
- These masks constrain query-to-patch attention (one quadrant of the attention matrix).
- Other three quadrants unmasked.

### Mask annealing (Sec. 3.3, Fig. 4)
- `P_mask` per block starts at 1.0, decays to 0.0 polynomially (factor 0.9).
- Staggered: earlier L₂ blocks anneal first.
- At inference: `P_mask = 0` everywhere → plain ViT, no masking overhead.
- Without annealing: train-with-mask/infer-without-mask **collapses** (PQ 27.4 vs 56.2).
- With annealing: PQ 56.0 @ 128 FPS vs 56.2 @ 61 FPS (masked).

### Loss (Eq. 2, Sec. A.2)
`L_total = λ_bce·L_bce + λ_dice·L_dice + λ_ce·L_ce` with `λ_bce=5.0, λ_dice=5.0, λ_ce=2.0`.
Hungarian matching of queries to ground-truth segments (Mask2Former recipe).

### Key empirical finding (Table 2)
MIM pretraining (DINOv2, EVA-02) closes the gap to complex baselines to ~1.1 PQ; supervised ImageNet leaves a 3.9–6.1 PQ gap. **MIM pretraining is the right pretraining for EoMT-style segmentation.**

### Inference (Sec. A.3)
- Panoptic/instance: padded square inference.
- Semantic: windowed sliding-crop.

## Codebase Findings (from exploration)

### Model class registry (NO central registry — ad-hoc branches)
- Dispatch via `if model_class == "frankensteindecoder"` in 8+ files: `main.py:171`, `deploy.py:85,112,269`, `inference.py:111`, `transformers_export.py:110,259`, `sbert/train_sbert.py:179,191`, `streamlit_gui/app.py:640`, `config_loader.py:147`, `cli.py:433`, `main.py:847`.
- Validation set: `{"frankenstein", "frankensteindecoder"}` in `config_loader.py:147`.

### FrankensteinModelConfig (src/model/config.py:91-508)
- Flat dataclass, ~70 fields. `vocab_size=50000`, `hidden_size=2048`, `num_layers=12`, `num_loops=2`, `layer_pattern` (35 mixer enum), `mode` (encoder/decoder), `positional_encoding` (hope/rope), `norm_type` (6 types), `use_moe`, `use_bitnet=True`, `ffn_activation`, `residual_type` (4 types), mHC fields, AttnRes fields, per-mixer latent-rank fields.
- `__post_init__` validates mode, positional_encoding, prms/flashnorm ratios, residual_type, ffn_activation, block_attn_num_blocks.
- `task` is NOT in the config — it lives in `training.task` and `LoadedTrainingConfig.task`.

### HybridLayer (src/model/hybrid_layer.py) — MODALITY-AGNOSTIC
- `mixer_registry` dict (line 156-191): 34 entries mapping layer_pattern names → attention classes.
- Operates on `(B, S, hidden_size)`. No causal mask. No token embedding.
- Only `engram_attn` needs `input_ids` (33/34 mixers work for ViT with `input_ids=None`).
- `TRAINING_FREE_LAYERS = {"fasa_attn", "sparge_attn"}` (eval-only).
- Forward signature: `forward(x, logical_layer_idx, input_ids)`.
- Special cases: `mamba` (Linear-as-residual), `ode`/`retnet` (no logical_layer_idx), `engram_attn` (needs input_ids).

### FrankensteinEncoder (src/model/frankenstein_encoder.py:84-208)
- `__init__`: `FactorizedEmbedding` or `nn.Embedding(vocab_size, hidden_size)` → dropout → `HybridLayer` stack → `final_norm` → head (`BitLinear` or `nn.Linear`, `hidden_size → vocab_size`) → `build_residual(config)` → optional mHC in/out proj.
- `forward(input_ids)`: emb → dropout → [mHC expand] → residual.register_state → looped layer loop → residual.finalize → [mHC collapse] → final_norm → head → logits.
- Exposes `last_auxiliary_losses` and `last_mixture_of_depths_stats`.

### FrankensteinDecoder (src/model/frankenstein_decoder.py:45-166)
- Wrapper pattern: `build_decoder_config()` static factory, `__init__` forces `mode="decoder"`, builds `FrankensteinEncoder` backbone, `forward` delegates, `generate()` for autoregressive.
- **This is the pattern FrankensteinViT mirrors.**

### Factories (all config-driven, model-agnostic)
- `get_norm(config)` — 6 norm types. Reads `config.norm_type` + `config.hidden_size`.
- `build_optimizer(optimizer_class, param_groups, parameters)` — 23 optimizers. Model needs `configure_optimizers` returning 4 named groups (embeddings, norms, attention, other).
- `build_residual(config)` — 4 residual types + lifecycle (register_state/reset_state/forward/finalize).
- `get_activation(config, dim)` — 42 activations. `make_gated_ffn` for GLU variants.
- `RoPE`/`HoPE` — modality-agnostic (operate on `(B, heads, S, head_dim)`), applied inside attention mixers.

### Schema structure
- `src/schema.yaml`: top-level, `additionalProperties: false`, 5 properties (base_model, tokenizer, model_class, model, training), `allOf → _conditional_rules.yaml`.
- `src/schema/_model_class.yaml`: enum `[frankenstein, frankensteindecoder]`.
- `src/schema/_training.yaml`: `task` enum `[mlm, sbert, causal_lm]`, `additionalProperties: false`, 22 optimizer if/then rules at bottom.
- `src/schema/_conditional_rules.yaml`: 6 rules (base_model OR model_class+model; task→optimizer/sbert/model_class; base_model→task+tokenizer).
- `src/schema/_sbert.yaml`: canonical task-specific sub-block pattern (389 lines).
- `src/schema/_model/_dims.yaml`: `layer_pattern` enum (35 mixers), `mode` enum `[encoder, decoder]`.
- New top-level blocks MUST be registered in `schema.yaml` properties (else rejected by `additionalProperties: false`).

### Training pipeline
- `config_loader.py:147`: `valid_model_classes = {"frankenstein", "frankensteindecoder"}`.
- `config_loader.py:160`: task validation `{"mlm", "sbert", "causal_lm"}`.
- `main.py:973`: `if loaded.task == "sbert": return _run_sbert_task(...)` — early return, bypasses TitanTrainer.
- `main.py:171`: `if model_class == "frankensteindecoder": ... else: FrankensteinEncoder(config)`.
- `trainer.py:1302`: `if self.task == "causal_lm": ... else: compute_mlm_loss(...)`.
- `trainer.py:1049`: `_forward_mlm_logits` calls `self.model(input_ids=..., attention_mask=...)` with TypeError fallback.
- `trainer.py:1293`: batch move to device `{k: v.to(...)}` — generic, works for vision keys.

### Config flattening (src/utils/config_flatten.py)
- `flatten_model_dict`: detects nested shape via `_is_nested_shape` (checks for dims/norm/embedding/attention/mhc/residuals keys).
- `_ATTENTION_MIXER_RENAMES`: maps per-mixer nested keys to flat config field names.
- `_flatten_titan`: handles titan sub-block (positional_encoding, hope/rope params).
- Grouping keys: `{"dims", "norm", "embedding", "attention", "mhc", "residuals"}`.

## Design Decisions (confirmed by user)

1. **Task names**: `patch_prediction`, `classification`, `segmentation` (in `training.task` enum).
2. **Schema placement**: New top-level `image:` block (vision model config) + task sub-blocks under `training:` (mirroring `training.sbert:`). New top-level `dataset:` block.
3. **Positional encoding**: Support BOTH — `pos_embedding_type` enum `[learned_1d, none]`. `learned_1d` = `nn.Parameter(1, N+1, D)` (faithful to ViT). `none` = rely on RoPE/HoPE inside attention mixers (treats patches as sequence).
4. **Masked patch prediction targets**: All three configurable via `prediction_target` enum `[mean_color_3bit, downsampled_3bit, full_patch_l2]`. Default = `mean_color_3bit` (paper's best).
5. **Segmentation head**: Both, selectable via `seg_head_type` enum `[pixel, eomt]`. `pixel` = per-pixel linear head (supports 1D grayscale + multicolor directly). `eomt` = EoMT query head (L₁/L₂ split, Hungarian matching, mask annealing).
6. **[CLS] vs GAP**: Both, via `pooling_mode` enum `[cls, gap]`. `cls` = prepend learnable token, head reads position 0. `gap` = mean-pool patches, head reads mean.
7. **Task ↔ model_class**: Vision tasks REQUIRE `model_class=frankenstein_vit` (enforced via conditional rules, like causal_lm requires frankensteindecoder).
8. **Dataset scope**: Full pipeline — new `ImageDataset` class + `dataset:` schema block with image fields (rescale, grayscale, columns, augmentations).
9. **Implementation scope**: Full implementation in a single plan (all files at once).

## Architecture Design

### FrankensteinViT class hierarchy
```
nn.Module
  └─ FrankensteinViT (wrapper, mirrors FrankensteinDecoder)
       ├─ PatchEmbed (Conv2d → flatten)        # replaces FactorizedEmbedding
       ├─ cls_token (nn.Parameter, optional)    # learnable [CLS]
       ├─ pos_embed (nn.Parameter, optional)    # learnable 1D absolute
       ├─ mask_token (nn.Parameter, optional)    # for patch_prediction
       ├─ dropout
       ├─ layers = [HybridLayer(...)] * num_layers  # REUSED — all 34 mixers
       ├─ final_norm = get_norm(config)           # REUSED — all 6 norms
       ├─ residual = build_residual(config)      # REUSED — all 4 residuals
       ├─ [mHC in/out proj]                       # REUSED
       └─ head (task-specific):
            ├─ patch_prediction: Linear(D, target_dim)
            ├─ classification: Linear(D, num_classes)  [zero-init for finetune]
            └─ segmentation:
                 ├─ pixel: Linear(D, num_seg_classes) + ViTDetUpsampler
                 └─ eomt: seg_queries + SegMaskModule + ViTDetUpsampler + HungarianMatcher
```

### Forward flow (per task)

**patch_prediction:**
```
pixel_values (B,C,H,W) → PatchEmbed → (B,N,D)
  [generate mask: mask_ratio*N positions, BERT strategy]
  [replace masked patch embeddings with mask_token/random/keep]
  [+ pos_embed] → dropout
  → [mHC expand] → HybridLayer loop (num_loops) → residual.finalize → [mHC collapse]
  → final_norm
  → head: Linear(D, P*P*C) or Linear(D, 512) or Linear(D, 16*512)
  → loss only on masked positions
  → residual reconstruction: full_image = input.clone(); full_image[mask] = pred[mask]
```

**classification:**
```
pixel_values (B,C,H,W) → PatchEmbed → (B,N,D)
  [prepend cls_token if pooling_mode=cls] → (B,N+1,D)
  [+ pos_embed] → dropout
  → HybridLayer loop → residual.finalize → final_norm
  → pool: x[:,0] (cls) or x.mean(dim=1) (gap)
  → head: Linear(D, num_classes) → CE loss
```

**segmentation (pixel):**
```
pixel_values (B,C,H,W) → PatchEmbed → (B,N,D)
  [+ pos_embed] → dropout
  → HybridLayer loop → residual.finalize → final_norm
  → head: Linear(D, num_seg_classes) → (B, N, num_seg_classes)
  → reshape to (B, num_seg_classes, H/P, W/P)
  → ViTDetUpsampler → (B, num_seg_classes, H, W)
  → CE + Dice loss vs segmentation_map (B, H, W)
```

**segmentation (eomt):**
```
pixel_values (B,C,H,W) → PatchEmbed → (B,N,D)
  [+ pos_embed] → dropout
  → L₁ blocks (patches only): first (num_layers - seg_l2_blocks) layers
  → concat seg_queries (1,K,D) → (B, N+K, D)
  → L₂ blocks (patches + queries): last seg_l2_blocks layers
       [if training & mask_annealing: predict intermediate masks, apply to query-to-patch attention]
  → final_norm → split patches and queries
  → SegMaskModule:
       queries → Linear → class_logits (B, K, C)
       queries → 3-layer MLP → mask_embeddings (B, K, D)
       patches → ViTDetUpsampler → F̃4 (B, D, H/4, W/4)
       mask_logits = mask_embeddings @ F̃4 → (B, K, H/4, W/4)
  → Hungarian match → BCE + Dice + CE loss
```

### Reuse summary (confirmed modality-agnostic)
| Component | Reuse mechanism | NLP coupling? |
|---|---|---|
| Attention mixers (34) | `mixer_registry` in HybridLayer | Only `engram_attn` (needs input_ids) |
| Norms (6) | `get_norm(config)` | None |
| Optimizers (23) | `build_optimizer` + `configure_optimizers` | None |
| Residuals (4) | `build_residual(config)` + lifecycle | None |
| Activations (42) | `get_activation` / `make_gated_ffn` | None |
| RoPE/HoPE | Applied inside attention mixers | None (patch index = sequence position) |
| HybridLayer | Operates on `(B, S, D)` | None (pass `input_ids=None`) |
| FactorizedEmbedding | NLP-only (token IDs) | **Replaced by PatchEmbed** |

## Files to Create/Modify

### Phase 1 — Config + schema (foundation)

1. **`src/model/config.py`** — Add ~16 vision fields to `FrankensteinModelConfig` + `__post_init__` validation.
   - `image_height`, `image_width`, `patch_size`, `in_channels`, `to_grayscale`, `pos_embedding_type`, `cls_token`, `pooling_mode`, `mask_ratio`, `mask_token_strategy`, `prediction_target`, `seg_head_type`, `num_classes`, `num_seg_classes`, `seg_num_queries`, `seg_l2_blocks`, `seg_mask_annealing`.
   - Validation: `pos_embedding_type in {learned_1d, none}`, `pooling_mode in {cls, gap}`, `mask_ratio in (0,1)`, `image_height % patch_size == 0`, `image_width % patch_size == 0`, `seg_head_type in {pixel, eomt}`, `prediction_target in {mean_color_3bit, downsampled_3bit, full_patch_l2}`, `mask_token_strategy in {bert, mask_only, random_only}`.

2. **`src/schema/_model_class.yaml`** — Add `frankenstein_vit` to enum + description.

3. **NEW `src/schema/_image.yaml`** — Vision model config schema (image_size, patch_size, in_channels, to_grayscale, pos_embedding_type, cls_token, pooling_mode, mask_ratio, mask_token_strategy, prediction_target, seg_head_type, num_classes, num_seg_classes, seg_num_queries, seg_l2_blocks, seg_mask_annealing).

4. **NEW `src/schema/_dataset.yaml`** — Dataset config schema (dataset_name, dataset_dir, image_column, label_column, segmentation_column, rescale, normalize, augmentations).

5. **`src/schema.yaml`** — Register `image:` and `dataset:` top-level blocks.

6. **`src/schema/_training.yaml`** — Extend `task` enum with `patch_prediction`, `classification`, `segmentation`; add `$ref`s to task sub-blocks.

7. **NEW `src/schema/_vision_patch.yaml`** — Task sub-block for patch_prediction (batch_size, epochs, learning_rate, optimizer, output_dir, checkpoint settings).

8. **NEW `src/schema/_vision_classification.yaml`** — Task sub-block for classification.

9. **NEW `src/schema/_vision_segmentation.yaml`** — Task sub-block for segmentation (includes seg_loss_weights for BCE/Dice/CE).

10. **`src/schema/_conditional_rules.yaml`** — Add vision task rules (require model_class=frankenstein_vit + image block + dataset block; relax tokenizer requirement).

11. **`src/utils/config_flatten.py`** — Add `image`/`dataset` to `_is_nested_shape`; add `_flatten_image` and `_flatten_dataset` functions; extend `flatten_model_dict`.

12. **`src/training/config_loader.py`** — Add `frankenstein_vit` to valid_model_classes; add vision tasks to task enum; relax tokenizer requirement for vision; load image/dataset blocks; add `image_config`/`dataset_config` to `LoadedTrainingConfig`.

### Phase 2 — Model

13. **NEW `src/model/frankenstein_vit.py`** — `PatchEmbed`, `FrankensteinViT` class, task-specific heads (patch_prediction, classification, pixel-seg), `configure_optimizers`, `build_vit_config` static factory, forward for all tasks.

14. **NEW `src/model/vit_seg_head.py`** — EoMT segmentation head: `SegMaskModule`, `ViTDetUpsampler`, `HungarianMatcher`, mask annealing logic. (If `seg_head_type == "eomt"`.)

15. **`src/model/__init__.py`** — Export `FrankensteinViT`.

### Phase 3 — Training pipeline

16. **NEW `src/training/vision_dataset.py`** — `ImageDataset` class: loads from HF `datasets` or local dir, rescaling, grayscale, normalization, augmentations, masking for patch_prediction, collate_fn.

17. **`src/training/trainer.py`** — Add `compute_patch_prediction_loss`, `compute_classification_loss`, `compute_segmentation_loss`; add `_forward_vision_logits`; extend loss dispatch in `train_epoch`.

18. **`src/training/main.py`** — Add `FrankensteinViT` import + model instantiation branch; add `_run_vision_task` + `_build_vision_dataloader`; add vision task early-return dispatch.

### Phase 4 — Deploy / inference

19. **`src/deploy/deploy.py`** — Add ViT branches in `_build_model`, `load_training_checkpoint`, `validate_deployment` (3 places); image-shaped dummy for validation.

20. **`src/deploy/inference.py`** — Add ViT branch in `_load_model`; add `predict_image` method.

21. **`src/deploy/transformers_export.py`** — Add ViT branch in `_build_core_model` and `_bake_state_dict`.

### Phase 5 — CLI

22. **`src/cli.py`** — Add `frankenstein_vit` to `--model-mode` choices; optionally add `vit-train`/`vit-infer` subcommands.

23. **`src/training/main.py`** — Add `frankenstein_vit` to `--model-mode` choices (line 847).

### Phase 6 — Example configs (required by AGENTS.md §9 — CI smoke-tests every YAML)

24. **NEW `configs/examples/vit_patch_prediction_adamw.yaml`**
25. **NEW `configs/examples/vit_classification_adamw.yaml`**
26. **NEW `configs/examples/vit_segmentation_pixel_adamw.yaml`**
27. **NEW `configs/examples/vit_segmentation_eomt_adamw.yaml`**
28. **NEW `configs/frankenstein_vit_base.yaml`** — Preset (ViT-Base: D=768, L=12, heads=12, patch=16, gelu, no MoE).

### Phase 7 — Tests

29. **NEW `tests/test_frankenstein_vit.py`** — Model tests (instantiation per task, forward shapes, patch embed, cls vs gap, pos_embed types, masking, gradient flow, BitNet on/off, multiple mixers, EoMT L₁/L₂ split).

30. **NEW `tests/test_vision_dataset.py`** — Dataset tests (loading, rescaling, grayscale, masking logic, collate).

31. **NEW `tests/test_vision_loss.py`** — Loss tests (patch_prediction 3bit CE + L2 MSE, classification CE, segmentation pixel CE+Dice, EoMT BCE+Dice+CE).

32. **`tests/test_config_loader.py`** — Add vision task validation, model_class=frankenstein_vit tests.

33. **`tests/test_naming_conventions.py`** — Update enum assertion (line ~82).

34. **`tests/test_cli_parser.py`** — Add `frankenstein_vit` choice test (line ~69).

35. **`tests/test_model_variants.py`** — Add ViT variant.

### Phase 8 — Docs

36. **NEW `docs/specs/vision.md`** — Vision spec (model class, 3 tasks, schema fields, config reference).

37. **NEW `docs/source/specs/vision.rst`** — Sphinx entry.

38. **`configs/README.md`** — Vision config reference section.

39. **`AGENTS.md`** — Add "Vision / frankenstein_vit" row to "Where to Edit" table; add vision hard constraints.

### Phase 9 — Verification

40. Run: `conda run -n frankenstein python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings`

## Key implementation details

### PatchEmbed
```python
class PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv2d(config.in_channels, config.hidden_size,
                              kernel_size=config.patch_size, stride=config.patch_size)
        self.num_patches = (config.image_height // config.patch_size) * (config.image_width // config.patch_size)
    def forward(self, pixel_values):  # (B, C, H, W) → (B, N, D)
        x = self.proj(pixel_values)       # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x
```

### Masking (patch_prediction, BERT recipe)
```python
def apply_masking(patch_embeddings, mask_token, mask_ratio, strategy="bert"):
    B, N, D = patch_embeddings.shape
    num_mask = int(N * mask_ratio)
    mask_bool = torch.zeros(B, N, dtype=torch.bool, device=patch_embeddings.device)
    for b in range(B):
        perm = torch.randperm(N, device=patch_embeddings.device)[:num_mask]
        mask_bool[b, perm] = True
    masked = patch_embeddings.clone()
    if strategy == "bert":
        rand = torch.rand_like(masked[mask_bool])
        mask_replace = rand < 0.8
        rand_replace = (rand >= 0.8) & (rand < 0.9)
        # 80% mask token, 10% random patch, 10% keep
        masked_indices = mask_bool.nonzero(as_tuple=True)
        masked[mask_bool] = torch.where(
            mask_replace.unsqueeze(-1), mask_token.expand_as(masked[mask_bool]),
            torch.where(rand_replace.unsqueeze(-1),
                        patch_embeddings[torch.randint_like(masked_indices[0], B), torch.randint_like(masked_indices[1], N)],
                        masked[mask_bool]))
    elif strategy == "mask_only":
        masked[mask_bool] = mask_token
    elif strategy == "random_only":
        rand_idx = torch.randint(0, N, (mask_bool.sum(),), device=patch_embeddings.device)
        masked[mask_bool] = patch_embeddings[torch.arange(B).repeat_interleave(num_mask), rand_idx]
    return masked, mask_bool
```

### Prediction targets
```python
def compute_mask_target(original_patches, patch_size, in_channels, target_type):
    # original_patches: (B, N, P*P*C) — raw patch pixels
    if target_type == "mean_color_3bit":
        # Mean over patch spatial dims → (B, N, C) → quantize to 3 bits (0-7) per channel
        mean = original_patches.view(B, N, patch_size*patch_size, C).mean(dim=2)
        quantized = (mean * 7).round().clamp(0, 7).long()
        # Encode as single 512-way class: R*64 + G*8 + B
        target = quantized[..., 0] * 64 + quantized[..., 1] * 8 + quantized[..., 2]  # (B, N)
        return target  # CE loss, 512 classes
    elif target_type == "downsampled_3bit":
        # 4x4 downsampled patch, 3-bit per pixel → 16 predictions of 512 classes
        ...
    elif target_type == "full_patch_l2":
        return original_patches  # (B, N, P*P*C) — MSE loss
```

### ViTDetUpsampler (for segmentation)
```python
class ViTDetUpsampler(nn.Module):
    """Upscale ViT patch features (H/P, W/P) to (H/4, W/4) via transposed/depthwise convs."""
    def __init__(self, hidden_size, patch_size, target_scale=4):
        super().__init__()
        layers = []
        current_scale = patch_size
        while current_scale > target_scale:
            layers.append(nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2))
            layers.append(nn.GELU())
            layers.append(nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size))  # depthwise
            layers.append(get_norm_by_type(...))  # or LayerNorm
            current_scale //= 2
        self.upsampler = nn.Sequential(*layers)
    def forward(self, x):  # (B, N, D) → (B, D, H/4, W/4)
        B, N, D = x.shape
        x = x.transpose(1, 2).view(B, D, H//P, W//P)
        return self.upsampler(x)
```

### HungarianMatcher (for EoMT segmentation)
```python
class HungarianMatcher(nn.Module):
    """Bipartite matching between predicted queries and ground-truth segments."""
    def __init__(self, cost_class=1.0, cost_mask=1.0, cost_dice=1.0):
        ...
    def forward(self, class_logits, mask_logits, class_targets, mask_targets):
        # Uses scipy.optimize.linear_sum_assignment per sample
        # Returns matched indices
        ...
```

### Mask annealing schedule
```python
def mask_annealing_prob(epoch, block_idx, total_l2_blocks, max_epochs, decay_factor=0.9):
    """Polynomial decay of P_mask per L₂ block, staggered."""
    # Earlier blocks anneal first
    offset = block_idx / total_l2_blocks  # stagger
    progress = max(0.0, (epoch / max_epochs - offset) / (1.0 - offset))
    return max(0.0, 1.0 - progress) ** decay_factor
```

### configure_optimizers
```python
def configure_optimizers(self):
    """Return 4 named param groups for build_optimizer."""
    groups = {"embeddings": [], "norms": [], "attention": [], "other": []}
    for name, param in self.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in ["patch_embed", "pos_embed", "cls_token", "mask_token", "seg_queries"]):
            groups["embeddings"].append(param)
        elif "norm" in name or "final_norm" in name:
            groups["norms"].append(param)
        elif "layers" in name and "mixer" in name:
            groups["attention"].append(param)
        else:
            groups["other"].append(param)  # FFN, head, upsampler, seg module
    return [{"params": g, "name": k} for k, g in groups.items()]
```

## Verification

After all files are implemented:
```bash
conda run -n frankenstein python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings
```

This exercises `test_yaml_examples.py` (auto-globs `configs/examples/*.yaml`) so the 4 new example YAMLs are smoke-tested automatically.