# Vision Transformer (frankenstein_vit) Specification

## Overview

The `frankenstein_vit` model class implements the Vision Transformer (ViT)
architecture from arXiv:2010.11929 ("An Image is Worth 16x16 Words",
Dosovitskiy et al., ICLR 2021), with segmentation support inspired by
arXiv:2503.19108 ("Your ViT is Secretly an Image Segmentation Model",
Kerssies et al., 2025).

It supports three tasks:

- **`patch_prediction`** — Masked patch prediction / autosupervised
  pre-training (ViT App. B.1.2). Patches are corrupted and the model
  predicts a reconstruction target.
- **`classification`** — Image classification via a linear head on the
  pooled image representation.
- **`segmentation`** — Image segmentation via a per-pixel linear head
  (`seg_head_type: pixel`) or an EoMT query-based head
  (`seg_head_type: eomt`).

## Architecture

### Patch Embedding (Eq. 1)
Images are split into non-overlapping `patch_size`×`patch_size` patches and
linearly projected to `hidden_size` dimensions via a `Conv2d` (equivalent to
flatten + linear). Output: `(B, N, D)` where `N = (H/P) * (W/P)`.

### Positional Encoding
- `learned_1d` (default): learnable `nn.Parameter(1, N+cls, D)` added after
  patch embedding (faithful to ViT paper).
- `none`: relies on RoPE/HoPE inside attention mixers (treats patches as a
  sequence).

### [CLS] Token
- `cls_token: true` (default): prepends a learnable token; classification
  head reads position 0.
- `pooling_mode: gap`: no [CLS] token; head reads mean over patches.

### Transformer Encoder
The `HybridLayer` stack from the Frankenstein codebase is reused **as-is**.
All 34 attention mixers, 6 norms, 4 residual types, 42 activations, and 23
optimizers are available. Only `engram_attn` requires `input_ids` (token
IDs) and is not usable for images.

### Task-Specific Heads
- **patch_prediction**: `Linear(D, pred_dim)` where pred_dim depends on
  `prediction_target` (512 for 3-bit color, 16×512 for downsampled, P*P*C
  for L2).
- **classification**: `Linear(D, num_classes)` (zero-init for finetune).
- **segmentation (pixel)**: `Linear(D, num_seg_classes)` applied per-pixel
  after ViTDet-style transposed-conv upsampling.
- **segmentation (eomt)**: K learnable queries concatenated for the last L₂
  blocks, mask module (linear class + 3-layer MLP mask embedding), dot
  product with upscaled F̃₄.

## Schema Configuration

### Top-level blocks
- `image:` — Vision model config (image_size, patch_size, channels,
  positional encoding, masking, segmentation head).
- `dataset:` — Dataset config (source, columns, rescaling, augmentations).
- `training.task` — `patch_prediction`, `classification`, or `segmentation`.
- `training.<task>:` — Task-specific sub-block (batch_size, epochs, LR,
  optimizer).

### Required fields
- `model_class: frankenstein_vit` (enforced by conditional rules).
- `image.image_size`, `image.patch_size` (required for all vision tasks).
- `image.num_classes` (required for classification).
- `dataset:` (required for all vision tasks).

### Key config fields
| Field | Type | Default | Description |
|---|---|---|---|
| `image.image_size.height/width` | int | 224 | Image dimensions |
| `image.patch_size` | int | 16 | Patch size P |
| `image.in_channels` | int | 3 | Input channels (1 for grayscale) |
| `image.to_grayscale` | bool | false | Convert to grayscale |
| `image.pos_embedding_type` | enum | learned_1d | `learned_1d` or `none` |
| `image.cls_token` | bool | true | Prepend [CLS] token |
| `image.pooling_mode` | enum | cls | `cls` or `gap` |
| `image.mask_ratio` | float | 0.5 | Fraction of patches to mask |
| `image.mask_token_strategy` | enum | bert | `bert`, `mask_only`, `random_only` |
| `image.prediction_target` | enum | mean_color_3bit | `mean_color_3bit`, `downsampled_3bit`, `full_patch_l2` |
| `image.seg_head_type` | enum | pixel | `pixel` or `eomt` |
| `image.num_classes` | int | 1000 | Classification classes |
| `image.num_seg_classes` | int | 21 | Segmentation classes |
| `image.seg_num_queries` | int | 100 | EoMT query count |
| `image.seg_l2_blocks` | int | 3 | EoMT L₂ blocks |
| `image.seg_mask_annealing` | bool | true | EoMT mask annealing |

## Training

The vision tasks use the same `TitanTrainer` as MLM/Causal-LM, with
task-specific loss methods:
- `compute_patch_prediction_loss` — CE (3-bit color) or MSE (full patch).
- `compute_classification_loss` — Cross-entropy.
- `compute_segmentation_loss` — CE + Dice (pixel head).

## Masked Patch Prediction Details

Following ViT App. B.1.2:
- Corrupt `mask_ratio` (default 50%) of patches.
- BERT recipe: 80% mask-token, 10% random patch, 10% keep.
- Targets: 3-bit mean color (512-way CE, best), 4×4 downsampled 3-bit
  (16×512 CE), or full-patch L2 (MSE).
- Residual reconstruction: `full_image = input.clone(); full_image[mask] = pred[mask]`.

## Segmentation Details

### Pixel head (simple)
- `Linear(D, num_seg_classes)` per patch → reshape to `(H/P, W/P)` →
  ViTDet upsampler → `(H, W)` → CE + Dice loss.
- Supports 1D grayscale (2 classes) and multicolor non-overlapping (N+1
  classes).

### EoMT head (advanced, arXiv:2503.19108)
- Split L₁ (patch-only) + L₂ (patches + K queries).
- Mask module: linear class logits + 3-layer MLP mask embedding.
- Dot product with ViTDet-upscaled F̃₄.
- Loss: BCE(×5) + Dice(×5) + CE(×2) with Hungarian matching.
- Mask annealing: P_mask decays polynomially (factor 0.9) to 0 at inference.

## References
- arXiv:2010.11929 — ViT (Dosovitskiy et al., ICLR 2021)
- arXiv:2503.19108 — EoMT (Kerssies et al., 2025)
- arXiv:2111.06377 — MAE (He et al., 2022, for full_patch_l2 target)