#!/usr/bin/env python3
"""Frankenstein Vision Transformer (arXiv:2010.11929).

This module hosts :class:`FrankensteinViT`, the Vision Transformer model
class for image understanding. It mirrors the :class:`FrankensteinDecoder`
wrapper pattern: it owns a patch embedding (Conv2d → flatten) that replaces
the NLP token embedding, a stack of :class:`HybridLayer` blocks (reused from
the encoder), and a task-specific head for one of three tasks:

* ``patch_prediction`` — masked patch prediction / autosupervised
  pre-training (arXiv:2010.11929, App. B.1.2). Patches are corrupted and
  the model predicts a reconstruction target (3-bit mean color, downsampled
  3-bit, or full-patch L2).
* ``classification`` — image classification via a linear head applied to the
  pooled image representation ([CLS] token or global average pooling).
* ``segmentation`` — image segmentation via a per-pixel linear head
  (``seg_head_type: pixel``) or an EoMT query-based head
  (``seg_head_type: eomt``, arXiv:2503.19108).

The model reuses **all** attention mixers, norms, residuals, activations,
and the optimizer infrastructure from the Frankenstein codebase.
:class:`HybridLayer` is modality-agnostic — it operates on ``(B, S, D)``
tensors with no causal mask or token-embedding coupling — so the only
NLP-specific component replaced is the token embedding (→ PatchEmbed) and
the vocabulary head (→ task-specific vision head).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FrankensteinModelConfig
from .hybrid_layer import HybridLayer
from .attention.common import BitLinear
from .embeddings import build_pos_encoder, LearnedAbsolutePE, SinusoidalAbsolute
from .norm import get_norm
from .residuals import ResidualBase, build_residual


class PatchEmbed(nn.Module):
    """Image patch embedding (arXiv:2010.11929, Eq. 1).

    Splits an image into non-overlapping ``patch_size``×``patch_size`` patches
    via a :class:`nn.Conv2d` with ``kernel_size=stride=patch_size``
    (mathematically equivalent to flatten + linear), then flattens the
    spatial dimensions into a sequence of ``N`` patch tokens.

    Attributes:
        proj: The Conv2d patch projection.
        num_patches: Number of patches ``N = (H/P) * (W/P)``.
        patch_size: Patch size ``P``.
    """

    def __init__(self, config: FrankensteinModelConfig) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.patch_size = config.patch_size
        self.num_patches = (
            (config.image_height // config.patch_size)
            * (config.image_width // config.patch_size)
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embed image patches.

        Args:
            pixel_values: Image tensor of shape ``(B, C, H, W)``.

        Returns:
            Patch embeddings of shape ``(B, N, D)``.
        """
        x = self.proj(pixel_values)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


def _trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0,
                   a: float = -2.0, b: float = 2.0) -> None:
    """Truncated normal initialization (in-place)."""
    with torch.no_grad():
        tensor.copy_(torch.randn_like(tensor) * std + mean)
        tensor.clamp_(a * std, b * std)


class FrankensteinViT(nn.Module):
    """Vision Transformer (arXiv:2010.11929) with segmentation (arXiv:2503.19108).

    Mirrors :class:`FrankensteinDecoder`'s wrapper pattern: builds a
    :class:`PatchEmbed` + :class:`HybridLayer` stack + task-specific head.
    Reuses all attention mixers, norms, residuals, and the optimizer
    infrastructure from the Frankenstein codebase.

    The model forces ``config.mode = "encoder"`` (ViT is bidirectional).

    Attributes:
        config: The :class:`FrankensteinModelConfig` used to build the model.
        patch_embed: :class:`PatchEmbed` module.
        cls_token: Learnable [CLS] token (if ``config.cls_token`` and
            ``config.pooling_mode == "cls"``).
        pos_encoder: Shared positional encoding module (if
            ``config.pos_embedding_type`` resolves to a learned/sinusoidal
            absolute PE), built via :func:`build_pos_encoder`.
        mask_token: Learnable mask token (for ``patch_prediction`` task).
        dropout: Embedding dropout.
        layers: ModuleList of :class:`HybridLayer` blocks.
        final_norm: Final normalization before the head.
        residual: :class:`ResidualBase` module.
        head: Task-specific output head.
        last_auxiliary_losses: Dict of auxiliary losses (MoE/MoD).
        last_mixture_of_depths_stats: Dict of MoD statistics.
    """

    def __init__(self, config: Optional[FrankensteinModelConfig] = None) -> None:
        super().__init__()
        self.config = config or self.build_vit_config()
        self.last_auxiliary_losses: Dict[str, Any] = {}
        self.last_mixture_of_depths_stats: Dict[str, Any] = {}

        # ViT is bidirectional — force encoder mode.
        if self.config.mode != "encoder":
            self.config.mode = "encoder"

        cfg = self.config
        self.num_patches = (
            (cfg.image_height // cfg.patch_size)
            * (cfg.image_width // cfg.patch_size)
        )

        # ---- Patch embedding ----
        self.patch_embed = PatchEmbed(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

        # ---- [CLS] token (for classification with cls pooling) ----
        self.use_cls_token = bool(cfg.cls_token and cfg.pooling_mode == "cls")
        if self.use_cls_token and "ssog_attn" in (cfg.layer_pattern or []):
            raise ValueError(
                "ssog_attn requires the raw patch grid: a prepended [CLS] "
                "token breaks the grid_h x grid_w raster the Gaussian field "
                "is defined on. Set image.cls_token=false (and "
                "image.pooling_mode='gap' for global average pooling)."
            )
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
            _trunc_normal_(self.cls_token, std=0.02)

        # ---- Learnable 1D positional embedding ----
        pe_type = str(getattr(cfg, "pos_embedding_type", "learned_1d"))
        if pe_type == "learned_1d":
            pe_type = "learned_absolute"
        original_pe = getattr(cfg, "positional_encoding", None)
        cfg.positional_encoding = pe_type
        self.pos_encoder = build_pos_encoder(cfg)
        cfg.positional_encoding = original_pe
        self.use_pos_embed = pe_type in ("learned_absolute", "sinusoidal_absolute")

        # ---- Mask token (for patch_prediction) ----
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        _trunc_normal_(self.mask_token, std=0.02)

        # ---- HybridLayer stack (reused — every layer_pattern mixer) ----
        self.layers = nn.ModuleList(
            [
                HybridLayer(cfg, layer_type=cfg.layer_pattern[i % len(cfg.layer_pattern)], pos_encoder=self.pos_encoder)
                for i in range(cfg.num_layers)
            ]
        )

        # ---- Final norm + residual (reused) ----
        self.final_norm = get_norm(cfg)
        self.residual: ResidualBase = build_residual(cfg)

        # ---- mHC (reused) ----
        self.use_mhc = bool(getattr(cfg, "use_mhc", False))
        if self.use_mhc:
            n = int(cfg.mhc_expansion_rate)
            self.mhc_in_proj = nn.Linear(cfg.hidden_size, n * cfg.hidden_size)
            self.mhc_out_proj = nn.Linear(n * cfg.hidden_size, cfg.hidden_size)

        # ---- Task-specific head ----
        proj_cls = BitLinear if cfg.use_bitnet else nn.Linear
        self._build_head(proj_cls)

    def _build_head(self, proj_cls) -> None:
        """Build the task-specific output head."""
        cfg = self.config
        # Determine head based on config fields. The active task is set by
        # the trainer via ``self._active_task`` (default: classification).
        self._active_task: str = "classification"
        # Classification head: Linear(D, num_classes), zero-init for finetune.
        self.classification_head = proj_cls(cfg.hidden_size, cfg.num_classes)
        nn.init.zeros_(self.classification_head.weight)
        if self.classification_head.bias is not None:
            nn.init.zeros_(self.classification_head.bias)
        # Patch prediction head: Linear(D, target_dim).
        if cfg.prediction_target == "mean_color_3bit":
            pred_dim = 512  # 3-bit per channel (8^3 = 512)
        elif cfg.prediction_target == "downsampled_3bit":
            pred_dim = 16 * 512  # 4x4 downsampled, 3-bit per pixel
        else:  # full_patch_l2
            pred_dim = cfg.patch_size * cfg.patch_size * cfg.in_channels
        self.patch_pred_head = proj_cls(cfg.hidden_size, pred_dim)
        # Segmentation (pixel head): Linear(D, num_seg_classes) + upsampler.
        self.seg_head = proj_cls(cfg.hidden_size, cfg.num_seg_classes)
        nn.init.zeros_(self.seg_head.weight)
        if self.seg_head.bias is not None:
            nn.init.zeros_(self.seg_head.bias)
        # ViTDet-style upsampler: (H/P, W/P) -> (H, W) via transposed convs.
        self.seg_upsampler = self._build_upsampler()

    def _build_upsampler(self) -> nn.Module:
        """Build the ViTDet-style transposed-conv upsampler for segmentation."""
        cfg = self.config
        layers: List[nn.Module] = []
        current_scale = cfg.patch_size
        while current_scale > 1:
            layers.append(nn.ConvTranspose2d(
                cfg.hidden_size, cfg.hidden_size, kernel_size=2, stride=2,
            ))
            layers.append(nn.GELU())
            layers.append(nn.Conv2d(
                cfg.hidden_size, cfg.hidden_size, kernel_size=3, padding=1,
                groups=cfg.hidden_size,  # depthwise
            ))
            current_scale //= 2
        return nn.Sequential(*layers)

    @staticmethod
    def build_vit_config(
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_loops: int = 1,
        patch_size: int = 16,
        image_height: int = 224,
        image_width: int = 224,
        in_channels: int = 3,
        use_bitnet: bool = False,
        layer_pattern: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> FrankensteinModelConfig:
        """Build the default ViT preset configuration.

        Args:
            hidden_size: Hidden state dimensionality. Default: 768 (ViT-Base).
            num_layers: Number of HybridLayer blocks. Default: 12.
            num_heads: Number of attention heads. Default: 12.
            num_loops: Loop iterations. Default: 1.
            patch_size: Patch size. Default: 16.
            image_height: Image height. Default: 224.
            image_width: Image width. Default: 224.
            in_channels: Input channels. Default: 3.
            use_bitnet: Whether to use BitNet. Default: False.
            layer_pattern: Custom layer pattern. If None, defaults to
                ``["standard_attn"] * 12`` (pure ViT).
            **kwargs: Additional FrankensteinModelConfig fields.

        Returns:
            A pre-configured :class:`FrankensteinModelConfig` with ViT defaults.
        """
        if layer_pattern is None:
            layer_pattern = ["standard_attn"] * num_layers
        # Remove any kwargs that duplicate explicit params to avoid conflicts.
        explicit_keys = {
            "hidden_size", "num_layers", "num_heads", "num_loops", "patch_size",
            "image_height", "image_width", "in_channels", "use_bitnet",
            "layer_pattern", "ffn_activation", "ffn_hidden_size", "use_moe",
            "norm_type", "pos_embedding_type", "mode",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in explicit_keys}
        return FrankensteinModelConfig(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_loops=num_loops,
            patch_size=patch_size,
            image_height=image_height,
            image_width=image_width,
            in_channels=in_channels,
            use_bitnet=use_bitnet,
            layer_pattern=layer_pattern,
            ffn_activation="gelu",
            ffn_hidden_size=hidden_size * 4,  # 4*D like ViT
            use_moe=False,
            norm_type="layer_norm",
            pos_embedding_type="learned_1d",
            mode="encoder",
            **filtered_kwargs,
        )

    def _apply_masking(
        self, patch_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply BERT-style masking to patch embeddings for patch_prediction.

        Args:
            patch_emb: Patch embeddings of shape ``(B, N, D)``.

        Returns:
            Tuple of (masked_embeddings, mask_bool) where ``mask_bool`` is
            ``(B, N)`` boolean tensor indicating corrupted positions.
        """
        B, N, D = patch_emb.shape
        num_mask = max(1, int(N * self.config.mask_ratio))
        mask_bool = torch.zeros(B, N, dtype=torch.bool, device=patch_emb.device)
        for b in range(B):
            perm = torch.randperm(N, device=patch_emb.device)[:num_mask]
            mask_bool[b, perm] = True

        masked = patch_emb.clone()
        strategy = self.config.mask_token_strategy
        if strategy == "bert":
            # 80% mask token, 10% random, 10% keep
            rand = torch.rand(B, N, device=patch_emb.device)
            mask_replace = mask_bool & (rand < 0.8)
            rand_replace = mask_bool & (rand >= 0.8) & (rand < 0.9)
            masked[mask_replace] = self.mask_token.expand(B, N, D)[mask_replace]
            # Random other patch embedding
            rand_idx = torch.randint(0, N, (B, N), device=patch_emb.device)
            rand_emb = patch_emb.gather(
                1, rand_idx.unsqueeze(-1).expand(B, N, D)
            )
            masked[rand_replace] = rand_emb[rand_replace]
        elif strategy == "mask_only":
            masked[mask_bool] = self.mask_token.expand(B, N, D)[mask_bool]
        elif strategy == "random_only":
            rand_idx = torch.randint(0, N, (B, N), device=patch_emb.device)
            rand_emb = patch_emb.gather(
                1, rand_idx.unsqueeze(-1).expand(B, N, D)
            )
            masked[mask_bool] = rand_emb[mask_bool]
        return masked, mask_bool

    def _run_encoder(
        self, x: torch.Tensor,
    ) -> torch.Tensor:
        """Run the HybridLayer stack with looped depth + residual + mHC.

        Args:
            x: Input hidden states of shape ``(B, S, D)``.

        Returns:
            Output hidden states of shape ``(B, S, D)``.
        """
        if self.use_mhc:
            n = int(self.config.mhc_expansion_rate)
            bsz, seq_len, dim = x.shape
            x = self.mhc_in_proj(x).view(bsz, seq_len, n, dim)

        logical_depth = int(self.config.num_layers) * int(self.config.num_loops)
        self.residual.register_state(num_layers=logical_depth)
        self.residual.reset_state()
        if self.residual.is_attn_res:
            self.residual.set_embedding(x)

        logical_layer_idx = 0
        for _ in range(self.config.num_loops):
            for layer in self.layers:
                x = layer(x, logical_layer_idx=logical_layer_idx, input_ids=None)
                if self.residual.is_attn_res:
                    x = self.residual(logical_layer_idx, x)
                logical_layer_idx += 1

        x = self.residual.finalize(x)

        if self.use_mhc:
            bsz, seq_len, n, dim = x.shape
            x = self.mhc_out_proj(x.reshape(bsz, seq_len, n * dim))

        x = self.final_norm(x)
        return x

    def forward(
        self,
        pixel_values: torch.Tensor,
        task: Optional[str] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the Vision Transformer.

        Args:
            pixel_values: Image tensor of shape ``(B, C, H, W)``.
            task: Task name (``patch_prediction``, ``classification``, or
                ``segmentation``). If None, uses ``self._active_task``.
            mask: Optional pre-computed mask boolean of shape ``(B, N)`` for
                ``patch_prediction``. If None, masking is generated internally.

        Returns:
            Task-specific output:
            - ``patch_prediction``: logits of shape ``(B, N, pred_dim)``.
            - ``classification``: logits of shape ``(B, num_classes)``.
            - ``segmentation``: logits of shape ``(B, num_seg_classes, H, W)``.
        """
        task = task or self._active_task
        B = pixel_values.shape[0]

        # ---- Patch embedding ----
        patch_emb = self.patch_embed(pixel_values)  # (B, N, D)

        if task == "patch_prediction":
            # Apply masking and store mask for loss computation.
            if mask is not None:
                mask_bool = mask
                masked = patch_emb.clone()
                masked[mask_bool] = self.mask_token.expand(B, self.num_patches, -1)[mask_bool]
            else:
                masked, mask_bool = self._apply_masking(patch_emb)
            self._last_mask_bool = mask_bool
            x = masked
        else:
            x = patch_emb

        # ---- [CLS] token ----
        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)

        # ---- Positional embedding ----
        if self.pos_encoder is not None and hasattr(self.pos_encoder, "add"):
            x = self.pos_encoder.add(x)

        x = self.dropout(x)

        # ---- Encoder ----
        x = self._run_encoder(x)

        # ---- Task-specific head ----
        if task == "patch_prediction":
            # Predict reconstruction for all patches; loss applied only on
            # masked positions by the trainer.
            logits = self.patch_pred_head(x)  # (B, N, pred_dim)
            return logits

        elif task == "classification":
            if self.use_cls_token:
                pooled = x[:, 0]  # [CLS] token
            else:
                pooled = x.mean(dim=1)  # GAP
            logits = self.classification_head(pooled)  # (B, num_classes)
            return logits

        elif task == "segmentation":
            # Strip [CLS] token if present (segmentation uses patch tokens only).
            if self.use_cls_token:
                x = x[:, 1:]
            # Reshape patch tokens to spatial feature map (B, D, H/P, W/P).
            h = self.config.image_height // self.config.patch_size
            w = self.config.image_width // self.config.patch_size
            feat = x.transpose(1, 2).reshape(B, self.config.hidden_size, h, w)
            # Upsample D-dim features to full resolution (B, D, H, W).
            feat = self.seg_upsampler(feat)
            # Apply seg head per-pixel: (B, D, H, W) -> (B, num_seg_classes, H, W).
            seg_logits = self.seg_head(feat.permute(0, 2, 3, 1))  # (B, H, W, C)
            seg_logits = seg_logits.permute(0, 3, 1, 2)  # (B, C, H, W)
            return seg_logits

        else:
            raise ValueError(f"Unknown task: {task!r}")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if (prefix + "pos_embed") in state_dict and (prefix + "pos_encoder.pos_embed") not in state_dict:
            state_dict[prefix + "pos_encoder.pos_embed"] = state_dict.pop(prefix + "pos_embed")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def configure_optimizers(self) -> List[Dict[str, Any]]:
        """Return named parameter groups for :func:`build_optimizer`.

        Groups: ``embeddings`` (patch_embed, pos_embed, cls_token, mask_token),
        ``norms``, ``attention``, ``other`` (FFN, head, upsampler).
        """
        groups: Dict[str, List[torch.nn.Parameter]] = {
            "embeddings": [], "norms": [], "attention": [], "other": [],
        }
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in
                   ["patch_embed", "pos_encoder", "pos_embed", "cls_token", "mask_token"]):
                groups["embeddings"].append(param)
            elif "norm" in name or "final_norm" in name:
                groups["norms"].append(param)
            elif "layers" in name and "mixer" in name:
                groups["attention"].append(param)
            else:
                groups["other"].append(param)
        return [{"params": g, "name": k} for k, g in groups.items()]