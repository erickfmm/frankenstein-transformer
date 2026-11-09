# MoE Completion Plan — router scoring, load balancing, shared/latent experts, layout, appendix & real-architecture presets

Date: 2026-11-09
Sources: HF "Ultimate MoE Training Handbook" §3 (https://huggingface-moe-blogpost.hf.space/#s-3), alphaXiv (LatentMoE 2601.18089 read in full + reference sweep), current code/schema audit.

---

## 1. Current state (gap analysis)

| Aspect | Today | Blog §3 / frontier practice |
|---|---|---|
| Schema `src/schema/_model/_moe.yaml` | `enabled`, `num_experts`, `top_k` only | router scoring, renorm, scaling factor, fp32, z-loss, balancing, shared, latent, layout, capacity… |
| `FrankensteinModelConfig` | `use_moe`, `num_experts=8`, `top_k_experts=2` | — |
| `HybridLayer` MoE forward (`hybrid_layer.py:488-511`) | `nn.Linear` router, `topk(softmax(logits))`, python-loop experts, **no renorm control, no aux loss, no z-loss, no bias balancing** | sigmoid/√softplus gating, renorm + routed scaling factor, fp32 router, z-loss |
| Expert width | reuses `ffn.hidden_size` (DeepSeek-V3 preset makes every expert 18432 wide — 9× too big; real = 2048) | separate `moe_intermediate_size` |
| Shared experts | none | 1–2 shared experts (DeepSeek/GLM/Kimi/Qwen3.8) |
| Latent MoE | none | ℓ-MoE (NVIDIA 2601.18089; Kimi K3, Nemotron 3) |
| Layer layout | MoE in **every** layer | leading dense layers (DeepSeek-V3: 3), interleave (Llama 4 Maverick: every 2nd) |
| Load balancing | none at all | aux loss (batch/seq scope), aux-loss-free bias (DeepSeek-V3 noaux_tc) |
| Capacity / dropping | dropless only | capacity factor + token dropping |
| Activation clamp | none | SwiGLU clamp 10 (DeepSeek-V4.1, Step 5) |
| Monitoring | none MoE-specific | z-loss, load-balancing loss, tokens-per-expert, MaxVio, expert-norm ratio |
| Named presets | expert count/top-k only; wrong widths; no shared/latent/layout | must mirror real `config.json` |
| Paper | no MoE appendix (annexes 1–21 exist) | annex-22 both languages |

Plumbing that already exists and will be reused:
- `last_auxiliary_losses` dict on models → `trainer.py:1488` sums tensors into the loss (MoD path already works this way).
- mHC path also has an MoE branch (`hybrid_layer.py:608`) — parity required.
- Param categories: any param name containing `router` → `router` group; `experts` → `ffn` group (`param_categories.py:97-102`) — keep those substrings in module names.
- BitNet `bitnet_routers` already routes the MoE router through `BitLinear`.

---

## 2. Schema contract first (source of truth)

Rewrite `src/schema/_model/_moe.yaml` (bilingual, `**References:** (paper: arXiv:<id>)` per key, `additionalProperties: false`). Flat names produced by `config_flatten.py`:

| Nested key | Flat field (dataclass) | Type / enum / default | Reference |
|---|---|---|---|
| `enabled` | `use_moe` | bool, `true` | 2407.06204 |
| `num_experts` | `num_experts` | int ≥1, `8` | 2101.03961 |
| `top_k` | `top_k_experts` | int ≥1, `2` | 2101.03961 |
| `expert_hidden_size` | `moe_expert_hidden_size` | int, `0` = fall back to `ffn.hidden_size` | 2412.19437 |
| `router.scoring` | `moe_router_scoring` | enum `softmax` \| `sigmoid` \| `softplus_sqrt`, default `softmax` | 2412.19437 (sigmoid), 2606.19348 (√softplus) |
| `router.pre_softmax` | `moe_router_pre_softmax` | bool, `false` (softmax over all E before top-k ⇒ no renorm; required for softmax+top-1) | Megatron `--moe-router-pre-softmax` |
| `router.renormalize` | `moe_router_renormalize` | bool, `true` (top-k weights sum→1) | 2506.02890 (renorm lowers loss) |
| `router.routed_scaling_factor` | `moe_routed_scaling_factor` | float, `1.0` (√k restores RMS; DS-V3 2.5) | 2412.19437 |
| `router.dtype` | `moe_router_dtype` | enum `float32` \| `bfloat16`, default `float32` | 2101.03961 (bf16 router diverged) |
| `router.jitter_eps` | `moe_router_jitter_eps` | float, `0.0` (Switch multiplicative input jitter) | 2101.03961 |
| `router.z_loss_coeff` | `moe_router_z_loss_coeff` | float, `0.0` (ST-MoE uses 1e-3) | 2202.08906 |
| `routing.type` | `moe_routing_type` | enum `token_choice` \| `expert_choice`, default `token_choice` | 2202.09368 |
| `routing.capacity_factor` | `moe_capacity_factor` | float, `0.0` (0 = dropless) | 2202.08906, 2505.04519 |
| `routing.hash_layers` | `moe_hash_layers` | int, `0` (first N MoE layers route by token-id table; DeepSeek-V4 = 3) | 2106.04426, 2606.19348 |
| `routing.hash_table` | `moe_hash_table` | enum `random` \| `round_robin`, default `random` (seeded) | 2106.04426 (random ≈ frequency-greedy; clustering worse) |
| `routing.hash_seed` | `moe_hash_seed` | int, `42` | 2106.04426 |
| `balancing.aux_loss` | `moe_aux_loss_type` | enum `none` \| `batch` \| `sequence`, default `batch` | 2401.06066; scope 2501.11873, 2412.19437 |
| `balancing.aux_loss_coeff` | `moe_aux_loss_coeff` | float, `0.01` (DS-V3 seq-wise 1e-4; Megatron 1e-3) | 2412.19437 |
| `balancing.bias_update` | `moe_bias_update` | enum `none` \| `sign` \| `centered_sign`, default `none` | 2408.15664 (noaux_tc), 2502.16982 (centered) |
| `balancing.bias_update_rate` | `moe_bias_update_rate` | float, `0.001` | 2408.15664, 2605.11689 |
| `shared.count` | `moe_shared_experts_count` | int, `0` | 2401.06066 |
| `shared.hidden_size` | `moe_shared_expert_hidden_size` | int, `0` = expert width | 2401.06066 |
| `shared.gate` | `moe_shared_expert_gate` | enum `none` \| `sigmoid`, default `none` | Qwen3-Next gated shared |
| `latent.enabled` | `moe_latent_enabled` | bool, `false` | 2601.18089 |
| `latent.dim` | `moe_latent_dim` | int, `0` ⇒ error when latent enabled | 2601.18089 (ℓ = d/α, α ≤ 4) |
| `latent.norm_before_up_proj` | `moe_latent_norm_before_up_proj` | bool, `false` (Kimi K3 stability) | 2607.24653 |
| `layout.dense_layers` | `moe_dense_layers` | int, `0` (leading dense layers; DS-V3 = 3) | 2412.19437, 2507.17702 |
| `layout.every_n_layers` | `moe_layer_frequency` | int ≥1, `1` (2 = Maverick interleave) | Llama-4 Maverick config |
| `experts.activation_clamp` | `moe_activation_clamp` | float, `0.0` = off (10 = DS-V4.1/Step-5; 7 = Step-5 Preview) | 2602.10604, 2609.19969 |

Validation (schema `if/then` + `config.py` runtime, bilingual error messages):
1. `top_k <= num_experts`.
2. `router.scoring == softmax and top_k == 1` requires `router.pre_softmax: true` (router would get zero grad) — error, mirrors Megatron.
3. `balancing.bias_update != none` requires `router.scoring != softmax` (bias only defined for sigmoid/√softplus) — error (Wang et al. 2408.15664).
4. `routing.type == expert_choice` forbids `balancing.*` (self-balancing) and is flagged training-only (future-token leakage at inference) — warning + doc.
5. `routing.hash_layers > 0` requires `token_choice`, forbids `bias_update` (a table cannot be rebalanced — warning) and keeps the learned router for the **weights only** (DeepSeek-V4 keeps its gate for weighting; selection gets no gradient). `hash_layers` ≤ number of MoE layers. Frequency-balanced greedy tables (Roller et al.) need corpus statistics — deferred utility; `random`/`round_robin` match their reported quality.
6. `latent.enabled` requires `0 < latent.dim < hidden_size`.
7. `layout.dense_layers < num_layers`; `every_n_layers >= 1`.
8. MoE + MoD unchanged; MoE + mHC allowed (parity with `hybrid_layer.py:608`).
9. `capacity_factor > 0` keeps highest-probability tokens per expert up to `cf · (T·top_k/E)`; rest dropped (megatron semantics).

---

## 3. Runtime implementation — new package `src/model/moe/`

Follows the `attention/`, `norm/`, `optimizer/` package pattern (D3).

### 3.1 `src/model/moe/router.py` — `MoERouter(nn.Module)`
- Linear `hidden → E` (BitLinear when `bitnet_routers`), built by HybridLayer as today but moved here.
- Forward: optional uniform jitter `eps·(2u−1)` (Switch), logits in `router.dtype` (fp32 default), scoring fn:
  - `softmax`: full softmax (pre_softmax) or post-top-k softmax over selected logits;
  - `sigmoid`: `σ(x)` scores; weights renormalized over top-k when `renormalize` (or raw when not — North-Mini style);
  - `softplus_sqrt`: `√softplus(x)` (DeepSeek-V4).
- Expert bias `b_e` added to scores **before** selection only (never into weights) when bias balancing on.
- Returns `(weights, indices, probs_for_aux)` + z-loss `(logsumexp(logits))²·mean` when coeff > 0.

### 3.2 `src/model/moe/balancing.py`
- `batch_aux_loss`: `E · Σ_e f_e · P_e` over the micro-batch (f = token share, P = mean router prob) — Switch/DeepSeekMoE form.
- `sequence_aux_loss`: same sum computed per sequence then averaged (DS-V3, coeff 1e-4 style).
- Expert-bias update rules: `sign` (DS-V3 noaux_tc: overloaded ⇒ −γ, underloaded ⇒ +γ) and `centered_sign` (subtract mean step — Moonlight/Trinity; avoids drift).
- `max_vio(counts)`: `(max_e c_e − mean) / mean` for logging.

### 3.3 `src/model/moe/layer.py` — `MoELayer(nn.Module)`
Owns experts (`nn.ModuleList`, width `expert_hidden_size` or fallback `ffn_hidden_size`), optional clamp on the expert activation (soft clamp at `activation_clamp`, applied to the gated intermediate), shared experts (count/width/gate), latent down/up projections `d→ℓ→d` (+ optional RMSNorm before up-proj), capacity factor + deterministic dropping (keep highest-probability assignments), `token_choice`/`expert_choice` routing, and the combined forward:

```
y = routed_scaling_factor · Up( Σ_k w_k · E_{i_k}( Clamp( Down(x_norm) ) ) )  [+ gated shared(x_norm)] [+ latent RMSNorm before Up]
```

- Module names keep `router` / `experts` substrings (param groups, `param_categories.py`).
- Registers buffers: `expert_bias` (fp32, non-persistent for inference), `last_moe_stats` (python dict): tokens per expert, MaxVio, load-balancing loss, z-loss, expert output-norm max/median ratio (rogue/dead expert signal, 2602.10604), routing entropy + top-1 weight.
- `expert_choice`: per-expert top-`T·k/E` tokens over the batch (training-only caveat).
- **Hash layers** (first `hash_layers` MoE layers): registered `tid2eid` buffer `(vocab_size, top_k)` built at init (`random` seeded / `round_robin`); selection = table lookup by `input_ids` (deterministic, context-free), weights = configured scoring gathered at the selected experts (renormalized per config). Requires plumbing `input_ids` into the MoE call in both `HybridLayer` paths (input_ids is already a forward parameter — Engram uses it; the MoE branch must now receive it, including the mHC path).
- Vectorized gather/scatter dispatch (replace the k×E python loop) — same math, faster CPU tests.

### 3.4 Integration
- `hybrid_layer.py`: build `MoELayer` instead of inline router/experts (keep `use_moe` flag, FlashNorm exclusion, BitNet router class, mHC parity at line 608). Collect `layer.last_auxiliary_losses` / stats like the MoD lists.
- Layer layout: `FrankensteinEncoder/Decoder/ViT/Audio` decide per layer index `i`: dense if `i < moe_dense_layers` or `(i − dense_layers) % every_n_layers != 0` (config helper `moe.is_moe_layer(i)` in `model/config.py`); dense layers use the plain FFN with `ffn.hidden_size` (fixes DeepSeek-V3 18432 vs 2048).
- Aux-loss plumbing: encoder/decoder/ViT/audio merge MoE losses into `last_auxiliary_losses` (`moe_load_balancing_loss`, `moe_router_z_loss`) exactly like `mixture_of_depths_router_loss`; trainer already sums tensors (`trainer.py:1488`); audio path `trainer.py:1626`.
- Bias update: `model.update_moe_expert_bias()` (walks layers, applies sign/centered update from accumulated fp32 counts); called by `trainer` after `optimizer.step()` in train mode only; eval() never updates.
- Logging: extend trainer metrics/CSV + W&B (`grad_norm/router` already exists): `moe/max_vio`, `moe/aux_loss`, `moe/z_loss`, `moe/expert_norm_ratio` (means over layers).
- `engine.py` / `training/main.py` startup log line: router scoring, balancing, shared/latent, layout.

### 3.5 Deploy / export
- `transformers_export.py`: export HF-compatible MoE block: `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `scoring_func`, `norm_topk_prob`, `routed_scaling_factor`, `n_shared_experts`, `first_k_dense_replace`, `topk_method` (map bias balancing → `noaux_tc`).
- `deploy/quantization.py`: expert weights already routed via `experts` → verify BitNet/quant paths still see them.
- `src/schema/_inference.yaml`: no change expected (MoE internal).

---

## 4. Named presets → real architectures (verify each against the HF `config.json` during implementation)

| Preset | Fix |
|---|---|
| `deepseek_v3.yaml` | expert_hidden 2048, dense ffn 18432 + `dense_layers: 3`, shared 1×2048, `scoring: sigmoid`, renorm, `routed_scaling_factor: 2.5`, `bias_update: sign` (γ=1e-3) + `aux_loss: sequence` (1e-4) [2412.19437] |
| `deepseek_v2.yaml` | expert_hidden 1536, shared 2, dense_layers 3, sigmoid, aux batch (device-level loss not applicable single-device — document deviation) [2405.04434] |
| `glm_5.yaml` | shared 1×2048, sigmoid, factor 2.5, dense_layers 3 |
| `glm_4_7.yaml` | verify vs GLM-4.5/4.7 config (factor 1.0 → shared gets half) |
| `kimi_k2.yaml`, `kimi_k2_7.yaml` | 384E top-8 + 1 shared; verify widths vs config.json |
| `kimi_k3.yaml` | `latent: {enabled, dim: 3584, norm_before_up_proj: true}`, shared 2×3072, `dense_layers: 1`, sigmoid, factor 1.0, `activation_clamp` ≈ SiTU-GLU caps (approximation documented; exact SiTU-GLU would need a new activation enum value — out of scope here) [2607.24653] |
| `gpt_oss_120b.yaml` | activation `swiglu`, verify scoring/renorm vs config.json |
| `llama4_scout.yaml` | top-1 + pre_softmax semantics, expert width 8192 (verify); Llama-4's input-side score scaling not implemented — document deviation |
| `llama4_maverick.yaml` | 128E top-1, `every_n_layers: 2` |
| `qwen3_next_80b.yaml` | shared 1×512 `gate: sigmoid` (gated shared expert) |
| `qwen3_8_2t4_a95b.yaml`, `qwen3_8_flash_next.yaml` | shared 1×2048, softmax renorm (blog §3.1 table) |
| `minimax_text01.yaml`, `minimax_m2_7.yaml`, `minimax_m3.yaml` | verify vs config.json (widths/shared/scoring) |
| `frankenstein_text_encoder.yaml` | keep 4/2, add `aux_loss: batch` as sane default showcase |

New examples (auto-globbed by `test_yaml_examples.py`): `configs/examples/moe_aux_loss_adamw.yaml`, `moe_noaux_bias_sigmoid.yaml`, `moe_shared_expert_gated.yaml`, `moe_latent_nemotron_style.yaml`, `moe_dense_layers_interleave.yaml`, `moe_expert_choice.yaml`, `moe_capacity_factor_drop.yaml`, `moe_activation_clamp.yaml`, `moe_hash_layers.yaml` (DeepSeek-V4-style: leading hash layers + learned rest).

---

## 5. Paper appendix (EN + ES) and bibliography

- `docs/paper/appendices/annex-22-moe.tex` and `docs/paper-es/appendices/annex-22-moe.tex` (Spanish mirror), `\input` in both `paper.tex` after annex-21. Structure:
  1. Overview & taxonomy (token vs expert choice; dropless vs capacity).
  2. Router scoring: softmax / sigmoid / √softplus formulas; renormalization; routed scaling factor (√k RMS argument); sharpness (jitter, fp32, z-loss `ℓ_z = (1/E)Σ(logsumexp)²`).
  3. Routing strategies: token choice; expert choice + future-token leakage caveat; hash layers (fixed token-id tables, router kept for weights only, no rebalancing possible — DeepSeek-V4 first layers).
  4. Load balancing: aux loss `L = α·E·Σ f_e·P_e` (batch vs sequence scope); aux-loss-free bias `b_e ∓ γ` update rule + centered variant; MaxVio definition; failure modes (flat-gate shortcut).
  5. Expert design: granularity/width; **LatentMoE** (α = d/ℓ, both ℓ-MoE_eff / ℓ-MoE_acc variants, RMSNorm before up-proj per Kimi K3, results from 2601.18089: MMLU-Pro +4.6 @73B, α ≤ 4); shared experts & output mixing (fixed weight vs gate vs joint normalization); activation clamping.
  6. Layer layout: leading dense layers, interleaving.
  7. Capacity factor & token dropping.
  8. Monitoring: z-loss, MaxVio, expert-norm max/median ratio, routing entropy.
  9. **Schema contract table**: every `moe.*` key ↔ formula ↔ default ↔ citation (schema = source of truth).
  10. Preset table: named configs ↔ real architectures.
  - Single-device scope note: expert parallelism / all-to-all / group-limited routing / EP replicas / FP8 experts are documented as related work, not implemented (CPU-first D7).
- Bibliography: new `docs/bibliography/moe.bib` (or extend `other.bib` where BASE 2103.16716 + survey 2407.06204 already live) with alphaXiv-verified entries: 1701.06538 (Shazeer sparsely-gated), 2006.16668 (GShard), 2101.03961 (Switch), 2103.16716 (BASE, exists), 2106.04426 (Hash Layers), 2202.08906 (ST-MoE), 2202.09368 (Expert Choice), 2211.15841 (MegaBlocks), 2212.05055 (Sparse Upcycling), 2401.04088 (Mixtral), 2401.06066 (DeepSeekMoE), 2404.05019 (ScMoE), 2405.04434 (DeepSeek-V2), 2406.06563 (Skywork-MoE gating logit norm), 2407.06204 (survey, exists), 2408.15664 (aux-loss-free bias), 2409.02060 (OLMoE), 2409.12210 (MoDSE), 2411.03884 (PolyNorm), 2412.19437 (DeepSeek-V3), 2501.11873 (global-batch scope), 2505.04519 (Pangu Ultra MoE), 2505.06839 (fine-grained expressivity), 2505.09388 (Qwen3), 2506.02890 (granularity/renorm), 2509.01322 (LongCat-Flash), 2601.08215 (granularity ladder), 2601.18089 (LatentMoE), 2602.10604 (Step-3.5-Flash rogue experts), 2607.24653 (Kimi K3), 2609.39137 (quantile balancing).
- Rebuild both PDFs (`latexmk`/`pdflatex + bibtex`) and verify annex numbering + `annex-summary-tables.tex` cross-links.

---

## 6. Docs

- `docs/architecture.md`: §7 rows for every new `moe.*` key; module map row `src/model/moe/`; invariant list additions (rules 2,3,4,5 above); param-category note.
- `configs/README.md`: new MoE parameters + preset table update.
- `src/deploy/README.md`: exported MoE config keys.

---

## 7. Tests & gates

New `tests/test_moe.py`:
- scoring ranking invariance (softmax vs sigmoid vs √softplus order), renorm sums to 1, scaling factor applied;
- softmax+top-1 without pre_softmax raises;
- bias + softmax raises; bias update moves overloaded expert down (sign + centered mean-zero);
- aux-loss exact value on hand-crafted probs (batch and sequence scopes);
- z-loss formula check;
- shared expert gate ∈ (0,1) bounds; latent shapes d→ℓ→d; norm-before-up-proj module present;
- `dense_layers` / `every_n_layers` construct routers on exactly the right layers;
- capacity factor drops exactly `ceil` overflow, keeps highest-prob;
- expert_choice equalizes token counts; eval() freezes bias;
- hash layers: same token id → same expert set across contexts; selection detached from router gradient; weights still renormalized; bias + hash warns;
- stats dict keys present; `param_categories` classification of new modules.
- KV-cache parity suite untouched (mixers unaffected), but decoder generation smoke test with MoE+latent.
Update: `test_schema_consistency.py` (defaults ↔ dataclass), `test_model_variants.py` (aux propagation encoder/decoder/ViT/audio), `test_transformers_export.py` (new keys), `test_yaml_examples.py` (auto), naming conventions.
Full gate run: `uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings`.

---

## 8. Order of work (per AGENTS.md "Config field" recipe)

1. `src/schema/_model/_moe.yaml` (full contract above) — schema is the contract.
2. `src/utils/config_flatten.py` (`_flatten` moe section, nested → flat names).
3. `src/model/config.py` (fields + validation rules + `is_moe_layer()` helper).
4. `src/model/moe/{router,balancing,layer}.py` + `moe/__init__.py`.
5. `src/model/hybrid_layer.py` integration (dense + mHC paths).
6. Encoder/decoder/ViT/audio layer-layout + aux-loss collection.
7. `src/training/trainer.py` (bias-update hook after step; metrics logging) + `engine.py`/`main.py` log line.
8. `src/deploy/transformers_export.py` + READMEs.
9. Presets (`configs/*.yaml`) + examples (`configs/examples/moe_*.yaml`).
10. Tests (new + updated).
11. Paper: bib entries → `annex-22-moe.tex` (EN) → Spanish mirror → `\input` both `paper.tex` → rebuild PDFs.
12. `docs/architecture.md` + `configs/README.md`.

## 9. Deferred (documented in appendix "future work"; not in this pass)

- Expert parallelism, token dispatchers (alltoall/flex), group-limited routing, EP replicas/EPLB — distributed-only.
- Global-batch aux loss (needs cross-rank reduction).
- Quantile balancing (histogram solver) — enum placeholder not added; revisit with `bias_update: quantile`.
- Dense→MoE upcycling (needs checkpoint-loading utility) and expert-count growth.
- Frequency-balanced hash-table construction from corpus statistics (random/round_robin shipped).
- Exact SiTU-GLU activation (would extend the 43-value activation enum — separate change).
- mHC shortcut-connected MoE block wiring (ScMoE Pos-1/Pos-2) — systems-level overlap hiding.
