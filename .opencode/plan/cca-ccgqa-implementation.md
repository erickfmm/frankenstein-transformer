# Plan: Implement CCA & CCGQA (arXiv:2510.04476)

## Paper
**Title**: "Compressed Convolutional Attention: Efficient Attention in a Compressed Latent Space"
**Authors**: Figliola, Alonso, Iyer, Anthony & Millidge (Zyphra, 2025). arXiv:2510.04476.

**CCA** = Compressed Convolutional Attention. Down-projects q/k/v into a shared latent (ẽ = E/C), runs attention *entirely in the latent* (no up-projections for q/k/v, only W̃_O). Three toggleable tricks make this viable: (1) two causal convs (depthwise seq + grouped channel), (2) q-k-mean, (3) value-shift. QK L2-norm + learnable temp β, RoPE in-latent.

**CCGQA** = CCA + GQA head-sharing in the latent, with **decoupled** query (C₁) and KV (C₂) compression. Constraint: C₂/C₁ = num_heads/num_kv_heads (ensures matching per-head latent dim d_h).

## Files to Create/Modify (17 total)

### 1. NEW: `src/model/attention/latent/cca_attn.py`
Both `CCAAttention` and `CCGQAAttention` classes (user requested same file).

**CCAAttention config knobs** (all toggleable per user's choice):
- `cca_latent_rank` (ẽ, default hidden//4 → C=4). Must divide num_heads.
- `cca_num_conv_layers` (0/1/2, default 2)
- `cca_conv_kernel_seq` (k_seq, default 4), `cca_conv_kernel_ch` (k_ch, default 3)
- `cca_qk_mean` (bool, default True)
- `cca_value_shift` (bool, default True; requires num_heads even, latent_dim even)
- Reuses general `rope_base` (default 10000.0)

Pipeline: `linear_qk` (E→2ẽ packed) → causal convs → qk-mean → value-shift (2 projections: val_proj1 from x_t, val_proj2 from x_{t-1}) → QK L2-norm + temp β → in-latent RoPE → softmax attention → `out_proj` (ẽ→E).

**CCGQAAttention config knobs:**
- `ccgqa_query_latent_rank` (E/C₁, default hidden//2 → C₁=2). Must divide num_heads.
- `ccgqa_kv_latent_rank` (E/C₂, default hidden//8 → C₂=8). Must divide num_kv_heads, ≤ query_latent.
- `ccgqa_num_kv_heads` (default num_heads//4, must divide num_heads)
- `ccgqa_num_conv_layers` (default 2), `ccgqa_conv_kernel_seq` (4), `ccgqa_conv_kernel_ch` (3)
- `ccgqa_qk_mean` (bool, default True), `ccgqa_value_shift` (bool, default True)
- Constraint: query_latent/num_heads == kv_latent/num_kv_heads (same d_h)

Same pipeline, but q and k have different latent widths; kv heads are group-shared (repeat_interleave); qk-mean uses B_group (replicate kv→query) / E_group (average query→kv).

Key implementation details:
- Convs: conv_qk0 = depthwise causal (groups=packed_dim), conv_qk1 = grouped (groups=2*num_heads for CCA, num_heads+num_kv_heads for CCGQA)
- RoPE helper: local `_apply_rope` (same as mla_attn.py)
- Causal conv helper: `_apply_causal_convs` (left-pad + conv, both convs causal)
- Value-shift: `F.pad(x[:, :-1], (0, 0, 1, 0))` shifts right by 1
- QK-norm: q = q * sqrt(d_h) / ||q||, k = k * sqrt(d_h) / ||k|| * exp(β); β init 0
- Attention scale: 1/sqrt(d_h) (same as existing family)
- BitLinear support via `proj_cls = BitLinear if config.use_bitnet else nn.Linear`

### 2. `src/model/attention/latent/__init__.py`
Add imports + `__all__` entries for `CCAAttention`, `CCGQAAttention`. Update module docstring (7→9 variants).

### 3. `src/model/attention/__init__.py`
Add to latent import block + `__all__`.

### 4. `src/model/tormented_bert_frankestein.py`
- Import `CCAAttention`, `CCGQAAttention` from `.attention.latent`
- Add to `mixer_registry`: `"cca_attn": CCAAttention`, `"ccgqa_attn": CCGQAAttention`
- Add UltraConfig dataclass fields (after MTLA block, ~line 319):
  ```python
  # ---- CCA / CCGQA (arXiv:2510.04476) ----
  cca_latent_rank: Optional[int] = None
  cca_num_conv_layers: int = 2
  cca_conv_kernel_seq: int = 4
  cca_conv_kernel_ch: int = 3
  cca_qk_mean: bool = True
  cca_value_shift: bool = True
  ccgqa_query_latent_rank: Optional[int] = None
  ccgqa_kv_latent_rank: Optional[int] = None
  ccgqa_num_kv_heads: Optional[int] = None
  ccgqa_num_conv_layers: int = 2
  ccgqa_conv_kernel_seq: int = 4
  ccgqa_conv_kernel_ch: int = 3
  ccgqa_qk_mean: bool = True
  ccgqa_value_shift: bool = True
  ```
- Add `__post_init__` defaults (after MTLA block, ~line 370):
  ```python
  if self.cca_latent_rank is None:
      self.cca_latent_rank = max(1, self.hidden_size // 4)
  if self.ccgqa_query_latent_rank is None:
      self.ccgqa_query_latent_rank = half
  if self.ccgqa_kv_latent_rank is None:
      self.ccgqa_kv_latent_rank = max(1, self.hidden_size // 8)
  if self.ccgqa_num_kv_heads is None:
      self.ccgqa_num_kv_heads = max(1, self.num_heads // 4)
  ```
- Update docstrings: module docstring (latent family 7→9 variants), UltraConfig layer_pattern docstring.

### 5. `src/schema/_model.yaml`
- Add `cca_attn` and `ccgqa_attn` to `layer_pattern` items enum
- Add layer_pattern table rows (EN + ES descriptions)
- Add config field definitions (after mtla_stride, ~line 1260) with title/title_es/description/description_es/examples for all 14 new fields (6 CCA + 8 CCGQA, minus shared rope_base)
- Update layer_pattern description count (20→22 mixer types, 19→21 in ES)

### 6. `tests/test_latent_attention.py`
Add `CCAAttentionTests` and `CCGQAAttentionTests` classes with:
- test_output_shape (encoder + decoder)
- test_gradient_flows
- test_invalid_latent_rank_raises (not divisible by num_heads)
- test_invalid_num_conv_layers_raises
- test_value_shift_odd_heads_raises
- CCGQA: test_invalid_dh_mismatch_raises, test_kv_larger_than_query_raises

### 7. `tests/test_schema_attention_layers.py`
- Add `cca_attn`, `ccgqa_attn` to expected layer set
- Add all 14 new config fields to expected schema fields list

### 8. NEW: `configs/examples/es_arch_cca_adamw.yaml`
Based on es_arch_mla_adamw.yaml, with `layer_pattern: [cca_attn]`, `cca_latent_rank: 128`, conv knobs.

### 9. NEW: `configs/examples/es_arch_ccgqa_adamw.yaml`
With `layer_pattern: [ccgqa_attn]`, `ccgqa_query_latent_rank: 256`, `ccgqa_kv_latent_rank: 64`, `ccgqa_num_kv_heads: 2`.

### 10. `configs/README.md`
- Add cca_attn, ccgqa_attn to latent blocks list
- Add config knob reference entries

### 11. `docs/paper/appendices/annex-3-latent-attention.tex`
Add CCA and CCGQA subsections (mathematical formulation, algorithmic pseudocode, key characteristics) + comparison table rows.

### 12. `docs/paper-es/appendices/annex-3-latent-attention.tex`
Spanish translation of the above.

### 13. `docs/paper/sections/08-summary-tables.tex`
Add CCA and CCGQA rows to the attention summary table.

### 14. `docs/paper-es/sections/08-tablas-resumen.tex`
Spanish translation.

### 15. `docs/specs/attention-mixers.md`
Add CCA/CCGQA descriptions.

### 16. `docs/bibliography/TRANSFORMER_TYPES.md`
Add CCA/CCGQA sections with architectural profiles.

### 17. `docs/bibliography/attention_types.bib`
Add bibtex entry for arXiv:2510.04476.

## Verification
```bash
conda run -n frankestein python -m pytest tests/test_latent_attention.py tests/test_schema_attention_layers.py tests/test_yaml_examples.py tests/test_attention_refactor.py --continue-on-collection-errors -v --tb=short -p no:warnings
```

## Design Decisions
1. **Toggleable sub-features** (user confirmed): cca_qk_mean, cca_value_shift, cca_num_conv_layers are config knobs, not hardcoded.
2. **Latent rank style** (not compression factor): cca_latent_rank (ẽ directly), consistent with mla_latent_rank, gqla_latent_rank, etc.
3. **Separate ccgqa_* knobs**: CCGQA has its own prefixed knobs (not sharing CCA's), consistent with existing pattern (each mechanism has own prefix).
4. **Both convs causal**: Safe for decoder mode; matches paper's causal design.
5. **Attention scale = 1/sqrt(d_h)**: Same as existing family; QK-norm + sqrt(d_h) scaling + β provides additional stabilization. With β=0, reduces to standard attention scaling.
6. **Value-shift requires even num_heads/num_kv_heads**: First half of heads get current-token values, second half get previous-token values.
7. **Standard nn.Conv1d** (not BitConv1d): Convs aren't BitNet-quantized (consistent with existing latent modules).
