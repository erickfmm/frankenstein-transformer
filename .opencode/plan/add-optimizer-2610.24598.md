# Plan: add AdamPM optimizer (arXiv:2610.24598)

> Status: **plan only** — nothing outside the plan directory was created,
> edited or deleted by the planning command. All commands below are to be run
> at implementation time.
>
> ⚠️ Note: the invoking command requested
> `.opencode/plan/add-optimizer-2610.24598.md` (repo-relative), but the
> plan-mode runtime only permits writes under `/home/erickfmm/.opencode/plan`.
> The file therefore lives here; move it into the repo's `.opencode/plan/`
> when implementation starts (repo convention per AGENTS.md).

## 1. Header

**Paper**: *AdamPM: Memory-Efficient LLM Optimization via Moment Pooling* —
Xiaowei Ji, Lishan Xu, Jianing An, Jie Luo, Wenjun Wu, Lei Huang (SKLCCSE,
School of Artificial Intelligence, Beihang University), arXiv:2610.24598 (2026).
Code: <https://github.com/jixiaowei95/AdamPM>.

Adam's two per-parameter moment buffers double optimizer-state memory and are
the main blocker for very large LLM training. AdamPM ("Adam with Pooled
Moments") is a **minimal modification of Adam**: it partitions every `s`
adjacent **second-moment** entries within each row (or column) into a group and
represents each group by its shared mean, storing the second moment at `1/s`
resolution (`mn(1+1/s)` state elements instead of Adam's `2mn`). The first
moment stays element-wise; hyperparameters (lr, β₁, β₂, ε) transfer directly
from Adam; the paper proves the same `O(log T/√T)` stationarity bound as Adam,
independent of `s`. It belongs in this repo as a new entry in the existing
**Memory-efficient** family bucket (§8.1), alongside Adafactor/GaLore/APOLLO,
with the distinctive property that it is model-agnostic, requires no projection
or SVD compute, and pools *all* parameter groups (no AdamW fallback routing).

Decision gate (passed): it is an optimizer method (Algorithm 1 + Theorem 3.1);
it is not a duplicate of any `factory.OPTIMIZER_REGISTRY` entry (30 entries
read at `src/model/optimizer/factory.py:56-87` — no moment-pooling family
exists); it is fully specified (update equations, algorithm, defaults,
padding rule, state shapes all in §3.3/Algorithm 1).

## 2. Paper extract

### Update rule (Algorithm 1, verbatim)

Given weight matrix `W ∈ ℝ^{m×n}`, group size `s`, `q = n/s` groups
(assuming `s | n`), learning rate `η_t`, decay rates `β₁, β₂ ∈ [0,1)`,
stability constant `ε > 0`:

```latex
G_t &= \nabla_W \mathcal{L}(W_{t-1})                       && \text{(gradient)} \\
M_t &= \beta_1 M_{t-1} + (1-\beta_1)\, G_t                 && \text{(first moment, element-wise)} \\
S_t &= G_t \odot G_t \in \mathbb{R}^{m \times n}           && \text{(squared gradient)} \\
R_t &= S_t.\mathrm{reshape}(m, q, s)                       && \text{(groups of } s \text{ adjacent entries)} \\
\bar{S}_t &= \frac{1}{s} \sum_{k=1}^{s} R_t[:,:,k] \in \mathbb{R}^{m \times q}  && \text{(group means)} \\
\bar{V}_t &= \beta_2 \bar{V}_{t-1} + (1-\beta_2)\, \bar{S}_t \in \mathbb{R}^{m \times q}  && \text{(compressed second-moment EMA)} \\
\hat{M}_t &= \frac{M_t}{1-\beta_1^t}                       && \text{(first-moment bias correction)} \\
\hat{\bar{V}}_t &= \frac{\bar{V}_t}{1-\beta_2^t}           && \text{(second-moment bias correction)} \\
\bar{V}_t^{\mathrm{bcast}} &= \hat{\bar{V}}_t \otimes \mathbf{1}_{1 \times s} \in \mathbb{R}^{m \times n}  && \text{(broadcast each group statistic)} \\
W_t &= W_{t-1} - \eta_t \cdot \frac{\hat{M}_t}{\sqrt{\bar{V}_t^{\mathrm{bcast}}} + \epsilon}  && \text{(update)}
```

Key design points (all quoted/paraphrased from the paper):

- **Order matters**: average squared gradients *within each group before* the
  EMA, so only the compressed `V̄_t ∈ ℝ^{m×q}` is persistent state —
  "reducing second-moment memory usage to 1/s of that required by Adam".
- **Padding rule**: "When `n` is not divisible by `s`, the final group of
  squared gradients is padded with its mean" ⇒ the last partial group's
  statistic is the **mean over its existing entries** (sum ÷ count), not
  sum ÷ `s`.
- **Pooling direction**: default is within each row (rows = output features);
  the column variant is `AdamPM-col` ("pooling is effective in either
  direction", Table 2).
- **Scope**: "AdamPM pools second moments for **all** weight matrices,
  including embed, lm_head, attention, and MLP weights. One-dimensional
  parameters, such as RMSNorm weights, are pooled as single-row or
  single-column matrices." → no AdamW fallback routing, unlike the Muon family.
- **First moment is NOT pooled**: "pooling the first moment naively leads to
  substantial performance degradation" (AdamPM-M, §4.2) — do not implement.
- **AdamPM-inf**: "all second moment entries within each row share a single
  scalar" — this is exactly `s ≥ n` under the padding rule (q = 1 group per
  row); no separate flag needed.
- **Weight decay**: Algorithm 1 defines none. This repo's convention for the
  shared `wd_*` suffixes is **decoupled** AdamW-style decay
  (`p ← p (1 − lr·wd)` before the adaptive update — see
  `adamw_fallback_update`, `src/model/optimizer/base.py:121-171`). Implement
  decoupled decay as a documented repo extension; note it in the annex.

### Pseudocode (Algorithm 1 structure)

```
Input: W_0 ∈ ℝ^{m×n}, total steps T, group size s, η_t, β₁, β₂, ε
t ← 0;  q ← n/s            # q groups, assuming s | n
M_0 ← 0_{m×n};  V̄_0 ← 0_{m×q}
repeat
  t ← t + 1
  G_t ← ∇_W L(W_{t-1})
  M_t ← β₁M_{t-1} + (1-β₁)G_t
  S_t ← G_t ⊙ G_t;  R_t ← reshape(S_t, m, q, s);  S̄_t ← mean_k R_t[:,:,k]
  V̄_t ← β₂V̄_{t-1} + (1-β₂)S̄_t
  M̂_t ← M_t/(1-β₁^t);  V̂̄_t ← V̄_t/(1-β₂^t)
  broadcast V̂̄_t to ℝ^{m×n};  W_t ← W_{t-1} − η_t·M̂_t/(√V̂̄_t + ε)
until t = T
```

### Hyperparameters

| Name | Type | Default / tested | Range | Scope | Notes |
|---|---|---|---|---|---|
| `s` (group size) | int | tested {2,3,4,5,8,16,∞}; "group sizes between 2 and 8 generally yield robust performance" | ≥ 1; s=1 → vanilla Adam; s≥n → AdamPM-inf | global | becomes `adam_pm-group_size` |
| pooling direction | str | row (default); col (`AdamPM-col`) | `row` \| `col` | global | becomes `adam_pm-pool_dim` |
| `η` (lr) | float | per-scale sweeps (4e-3 … 3e-4 in pre-training); "can directly reuse Adam's hyperparameters" | > 0 | per-group | shared `lr_*` |
| `β₁, β₂` | float pair | β₁=0.9, β₂=0.95 in all experiments | [0,1) | per-group | shared `betas_*` (repo default `(0.9, 0.95)` matches the paper) |
| `ε` | float | 1e-8 | > 0 | per-group | shared `eps_*` |
| weight decay | float | not in paper | — | per-group | repo extension: decoupled, shared `wd_*` |

Chosen default `adam_pm-group_size: 8` (87.5 % second-moment reduction; best
average GLUE score at s=8; inside the paper's robust 2–8 band).

### State buffers and per-step cost

- Buffers per parameter: `step` (scalar), `exp_avg` (`m×n`, full),
  `exp_avg_pooled` (`m×⌈n/s⌉` row-pooling, `⌈m/s⌉×n` column-pooling).
  Persistent optimizer-state elements: `mn(1 + 1/s)` vs Adam's `2mn`
  (paper §3.3). With default s=8: 56 % of Adam's optimizer state.
- Per-step cost: `O(n)` element-wise + one reshape + group mean (avg-pool) +
  broadcast — "negligible training overhead" (no SVD, no projection, no
  extra hyperparameter tuning).
- Convergence (Theorem 3.1 / Corollary 3.2): stationarity bound independent
  of `s`; under a relative-noise condition the same `O(log T/√T)` rate as
  Adam. Cosine similarity of preconditioned updates vs Adam ≥ 0.99 for most
  steps, **increasing with model scale** (Fig. 4d).

### Category

Existing bucket **Memory-efficient** (`docs/architecture.md` §8.1 table row:
`adafactor`, `lion`, `galore_adamw`, `apollo`, `apollo_mini`, `q_apollo`;
README "Memory-Efficient" row; annex subsection "Large-Batch, Memory-Efficient
& Parameter-Free"). No new category.

### BibTeX data

Authors: Xiaowei Ji, Lishan Xu, Jianing An, Jie Luo, Wenjun Wu, Lei Huang.
Year: 2026. Title: *AdamPM: Memory-Efficient LLM Optimization via Moment
Pooling*. arXiv:2610.24598 [cs.LG]. Entry drafted in §4 row 11.

## 3. Decisions

| Decision | Value | Evidence |
|---|---|---|
| Registry name | `adam_pm` | snake_case of the paper's short name; underscore split matches `ada_muon`, `galore_adamw` precedent; no existing prefix starts with `adam_pm` (checked `factory.py:56-87`), so `extract_prefixed_parameters` cannot mis-scope; `adamw` prefix unaffected |
| Class name | `AdamPM` | CamelCase precedent `AdaMuon`, `GaLoreAdamW`, `TurboMuon` |
| Closest peer | `src/model/optimizer/muon.py` (module skeleton: paper-citing docstring, Google docstrings, `defaults` dict, `@torch.no_grad() step`) + `anon.py` (single-knob global suffix pattern) + `cautious_adamw.py` (small factory branch). **No** `ADAMW_FALLBACK_GROUPS` routing — AdamPM pools every parameter |
| Family category | Memory-efficient (existing bucket, §8.1 row; README row; annex "Large-Batch, Memory-Efficient & Parameter-Free") |
| Suffix table | see below |
| `supports_shared_per_group_suffixes` | `true` (norm; per-group lr/wd/betas/eps all meaningful — β₁/β₂/ε enter the pooled update, wd is the decoupled extension) |
| Example preset | `configs/examples/es_arch_gqa_adam_pm.yaml` — smallest valid pattern: clone of `es_arch_gqa_adamw.yaml` (10-layer GQA encoder, MLM) with prefix keys renamed; MLM avoids the `causal_lm`-requires-decoder rule and any closure/grad-accum constraints |
| Census bump | **30 → 31** everywhere the live census reads 30: `docs/architecture.md:727` §8.1 heading; `README.md:27` (mermaid) and `README.md:93`; both `annex-summary-tables.tex:100`. The stale counts below are flagged, not silently bumped: `docs/architecture.md:26` §1 says **27**; README categories table (lines 145–153) omits `root`, `muon2`, `ea4llm` and sums to 27; both abstracts say "twenty-four / veinticuatro familias" (describe the 868-config measurement campaign — must NOT be bumped) |

### Suffix table (new keys)

| Key | Type | Default | Coercion | Scope | EN description text | ES description text |
|---|---|---|---|---|---|---|
| `adam_pm-group_size` | int | 8 | `to_int` | global | `group_size`: pooling group size `s` for the second moment; `s=1` reduces to vanilla Adam, `s` ≥ row/column length gives one statistic per row/column (AdamPM-inf); second-moment memory becomes 1/s | `group_size`: tamaño de grupo `s` para el pooling del segundo momento; `s=1` equivale a Adam vainilla, `s` ≥ longitud de fila/columna deja un estadístico por fila/columna (AdamPM-inf); la memoria del segundo momento pasa a 1/s |
| `adam_pm-pool_dim` | str | `row` | `str(...).strip().lower()` + class-side validation `{"row","col"}` | global | `pool_dim`: pooling direction for adjacent second-moment entries — `row` (within each row, default) or `col` (within each column, AdamPM-col) | `pool_dim`: dirección del pooling de entradas adyacentes del segundo momento — `row` (dentro de cada fila, por defecto) o `col` (dentro de cada columna, AdamPM-col) |

`OPTIMIZER_ALLOWED_KEYS["adam_pm"] = _COMMON_PER_GROUP_KEYS | {"group_size", "pool_dim"}`
(mirrors `"anon": ... | {"gamma"}`, `factory.py:143`).

## 4. File-by-file plan table

Execution order. Anchors are path:line as read on 2026-11-04.

| # | File | Planned change (concrete) |
|---|---|---|
| 1 | `src/model/optimizer/adam_pm.py` | **New file.** Module docstring citing "Ji, X., Xu, L., An, J., Luo, J., Wu, W., Huang, L. (2026). AdamPM: Memory-Efficient LLM Optimization via Moment Pooling. arXiv:2610.24598" (mirror `muon.py:1-17` style). `from __future__ import annotations`; `import torch`; `from torch.optim import Optimizer`. Google-style docstrings on class + every method. CPU-safe (no `.cuda()`, no device assumptions). Class `AdamPM(Optimizer)` with `__init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, group_size=8, pool_dim="row")`: validate `lr ≥ 0`, `0 ≤ betas < 1`, `eps ≥ 0`, `group_size ≥ 1` (int), `pool_dim ∈ {"row","col"}` (lowercased, `ValueError` otherwise). `defaults = dict(lr=..., betas=..., eps=..., weight_decay=..., group_size=..., pool_dim=...)`. `@torch.no_grad() step(closure=None)`: per group read (`lr`, `weight_decay`, `betas`, `eps` injected by the factory; `group_size`, `pool_dim` from `defaults` or group overrides) → per param `p` with `p.grad is not None`, reject sparse grads (`RuntimeError`, `anon.py:157` pattern). Lazy state init: `state["step"]=0`, `state["exp_avg"]=torch.zeros_like(p)`, `state["exp_avg_pooled"]=torch.zeros(compressed_shape)`. Shape handling: row pooling views any param as `(r, c)` with `c = p.shape[-1]`, `r = numel // c`; col pooling as `r = p.shape[0]`, `c = numel // r`; `q = ceil(c/s)` (row) or `ceil(r/s)` (col). Then map Algorithm 1 line-by-line: (a) `g2 = grad*grad`; (b) group mean with the paper's padding semantics — zero-pad the pooling axis to `q*s`, `sum` over groups, divide by per-group counts (`s` everywhere, `axis_len - (q-1)*s` for the last) ⇒ `S̄`; (c) `exp_avg_pooled.mul_(β₂).add_(S̄, alpha=1-β₂)`; (d) `exp_avg.mul_(β₁).add_(grad, alpha=1-β₁)`; (e) `t += 1`; `bc1 = 1-β₁ᵗ`, `bc2 = 1-β₂ᵗ`; (f) decoupled decay `p.mul_(1 - lr*wd)` when `wd > 0`; (g) broadcast: `v_hat = (exp_avg_pooled / bc2).repeat_interleave(s, dim=pool_axis)` truncated to the axis length; `denom = v_hat.sqrt() + eps`; (h) `p.addcdiv_(exp_avg, denom, value=-lr/bc1)`. 1-D params and arbitrary-N-D tensors follow the same flatten rule (paper: 1-D params are "single-row or single-column matrices"). `s = 1` must reduce exactly to Adam (test #10 asserts). |
| 2 | `src/model/optimizer/__init__.py` | Add `from .adam_pm import AdamPM` (before `from .ada_muon import AdaMuon`, line 2, keeping the rough A-group ordering) and `"AdamPM",` in `__all__` (inside the A block, lines 33-41). |
| 3 | `src/model/optimizer/factory.py` | (a) Import: `from .adam_pm import AdamPM` before `from .adan import Adan` (line 18). (b) Registry: `"adam_pm": AdamPM,` after `"adamw": AdamWOptimizer,` (line 58). (c) `OPTIMIZER_ALLOWED_KEYS`: `"adam_pm": _COMMON_PER_GROUP_KEYS \| {"group_size", "pool_dim"},` after the `"adamw"` entry (line 107). (d) Constructor branch after the `anon` branch (lines 276-278): `if optimizer_name == "adam_pm": group_size = to_int(scoped.get("group_size"), 8); pool_dim = str(scoped.get("pool_dim", "row")).strip().lower(); for g in converted_groups: g["group_size"] = group_size; g["pool_dim"] = pool_dim; return optimizer_cls(converted_groups, group_size=group_size, pool_dim=pool_dim)` — mirror the per-group injection style of the `adafactor` branch (`factory.py:263-269`). No `_build_param_groups_for_optimizer` change needed (keeps the default betas/eps injection). |
| 4 | `src/schema/_optimizer.yaml` | (a) `optimizer_class.enum` (lines 143-173): add `- adam_pm` after `- adamw` (line 145). (b) EN use-case table (lines 33-64): new row after `adamw` — `` \| `adam_pm` \| Memory-constrained training \| Pooled second moment: group size s shrinks the V-state to 1/s, Adam hyperparameters transfer (paper: arxiv:2610.24598) \``. (c) ES use-case table (lines 89-120): `` \| `adam_pm` \| Entrenamiento con memoria limitada \| Segundo momento agrupado: el tamaño de grupo s reduce el estado V a 1/s, los hiperparámetros de Adam se transfieren (paper: arxiv:2610.24598) \``. (d) EN `parameters.description` suffix list (lines 189-208): `- `adam_pm`: `group_size`, `pool_dim`. Row/column group-mean pooling of the second moment only; `s=1` → vanilla Adam, `s` ≥ axis length → one statistic per row/column (AdamPM-inf); first moment stays element-wise; the shared `lr_*`, `wd_*` (decoupled), `betas_*`, `eps_*` apply per group`. (e) ES `parameters.description_es` list (lines 232-251): `- `adam_pm`: `group_size`, `pool_dim`. Pooling por media de grupo del segundo momento únicamente (filas o columnas); `s=1` → Adam vainilla, `s` ≥ longitud del eje → un estadístico por fila/columna (AdamPM-inf); el primer momento queda por elemento; los sufijos compartidos `lr_*`, `wd_*` (desacoplado), `betas_*`, `eps_*` aplican por grupo`. (f) `x_parameter_reference.by_optimizer` (lines 298-418): after the `adamw:` block — `adam_pm:\n          prefix: "adam_pm-"\n          supports_shared_per_group_suffixes: true\n          specific_global_suffixes: [group_size, pool_dim]`. (g) Both **References** paragraphs (EN lines 73-85, ES lines 130-142): append ``; `adam_pm` (paper: arxiv:2610.24598)``. |
| 5 | `src/schema/_training.yaml` | One new `if/then` block appended after the `muon2` block (ends line 1609), same shape as lines 1596-1609: `if: properties.optimizer.properties.optimizer_class.const: adam_pm` → `then: properties.optimizer.properties.parameters: {additionalProperties: false, patternProperties: {^adam_pm-: {type: [number, boolean, array, string]}}}`. Exactly one block per optimizer is enforced by `test_optimizer_prefixes_match_conditional_rules` (`tests/test_schema_consistency.py:76`). |
| 6 | `src/schema/_sbert.yaml` | `optimizer_kwargs` EN description (line 400): insert after the `beta2_decay/clip_threshold/eps1/eps2 (for adafactor)` clause → `group_size/pool_dim (for adam_pm: second-moment pooling group size and row/col direction)`. Same insertion in `description_es` (line 401) → `group_size/pool_dim (para `adam_pm`: tamaño de grupo y dirección fila/columna del pooling del segundo momento)`. |
| 7 | `configs/examples/es_arch_gqa_adam_pm.yaml` | **New file** — clone of `configs/examples/es_arch_gqa_adamw.yaml` (read in full during planning) with: `optimizer_class: adam_pm`; every `adamw-` key renamed to `adam_pm-`; add `adam_pm-group_size: 8` and `adam_pm-pool_dim: row`; `csv_log_path: training_metrics_es_arch_gqa_adam_pm.csv`. Full draft: `model_class: frankenstein_text_encoder; model.dims: {vocab_size: 50000, hidden_size: 512, num_layers: 10, num_loops: 1, num_heads: 8, num_kv_heads: 4, layer_pattern: [gqa_attn, gqa_attn]}; model.norm.type: layer_norm; model.embedding.conv.enabled: false; model.attention.retnet.heads: 8; model.moe.enabled: false; model.bitnet.enabled: false; model.ffn: {hidden_size: 2048, activation: gelu}; training: {task: mlm, seed: 42, num_epochs: 4, batch_size: 4, dataloader_workers: 2, max_length: 512, mlm_probability: 0.15, max_samples: 6000000, dataset_batch_size: 15000, num_workers: 6, cache_dir: ./temp_data/v2_dataset_cache, local_parquet_dir: /home/erickfmm/.cache/huggingface/hub/datasets--erickfmm--red_pajama_es_hq_35/snapshots/bd7286c289a95dc3803c375bc36aaaeb138b1eab/train/, prefer_local_cache: true, stream_local_parquet: true, join_temp_data_context_window: 0, join_temp_data_min_remainder_tokens: 128, gradient_accumulation_steps: 8, optimizer: {optimizer_class: adam_pm, parameters: {adam_pm-lr_embeddings: 1e-6, adam_pm-lr_norms: 5e-6, adam_pm-lr_attention: 3e-6, adam_pm-lr_other: 2e-6, adam_pm-wd_embeddings: 0.01, adam_pm-wd_norms: 0.001, adam_pm-wd_attention: 0.01, adam_pm-wd_other: 0.01, adam_pm-betas_{embeddings,norms,attention,other}: [0.9, 0.95], adam_pm-eps_{embeddings,norms,attention,other}: 1e-8, adam_pm-group_size: 8, adam_pm-pool_dim: row}}, scheduler_total_steps: 9000, max_nan_retries: 4, gradient_log_interval: 10, csv_log_path: training_metrics_es_arch_gqa_adam_pm.csv, telemetry_log_interval: 5}`. Must pass `load_training_config` (gate `tests/test_yaml_examples.py` auto-globs it). |
| 8 | `configs/README.md` | (a) Optimizer Chooser list (lines 134-163): add `- adam_pm` after `- adamw` (line 135). (b) Suffix reference list (lines 184-215): add after `adamw: none` (line 187) → `- `adam_pm`: `group_size`, `pool_dim` (group-mean pooling of adjacent second-moment entries along rows or columns; second-moment state 1/s; `s=1` → vanilla Adam; first moment untouched; shared `lr_*`/`wd_*`/`betas_*`/`eps_*` per group)`. (c) Example keys (lines 217-229): add `- `adam_pm-group_size: 8`` and `- `adam_pm-pool_dim: col``. |
| 9 | `full_tests/run_e2e.py` | (a) `OPTIMIZERS` list (lines 255-261): append `"adam_pm"` → line 260 becomes `"root", "muon2", "adam_pm",`. (b) Param map after the `muon2` branch (lines 695-697): `if optimizer_class == "adam_pm": params[f"{prefix}group_size"] = 8; params[f"{prefix}pool_dim"] = "row"`. The allowed-key filter (lines 699-714) then drops/keeps correctly. |
| 10 | `tests/test_optimizer_factory.py` | (a) `test_registry_contains_all_required` (line 30): add `"adam_pm"` to the `required` set. (b) New `class BuildOptimizerAdamPMTests(unittest.TestCase)` after `BuildOptimizerMuon2Tests` (uses `_param_groups()` helper), methods: `test_builds_without_error`; `test_step_runs` (tiny `(8,16)` + `(16,)` tensors, 3 steps, finite); `test_second_moment_state_is_pooled` (`exp_avg_pooled.numel() == ⌈c/8⌉·r` row; dim-0 compressed for col); `test_s1_matches_vanilla_adam` (`group_size=1, wd=0` ≡ reference Adam step, `atol 1e-7`); `test_partial_group_mean_pads_with_mean` (`c=13, s=8` → last group stat = mean of last 5 entries); `test_specific_suffixes_parsed` (`{"adam_pm-group_size": "4", "adam_pm-pool_dim": "COL"}` → 4 / "col"; `"bad"` falls back to default 8); `test_unknown_group_suffix_raises` (`{"adam_pm-lr_bogus": 1.0}` → `ValueError`, pattern of lines 397-399). |
| 11 | `docs/bibliography/optimizers.bib` | Append after `liu_muon2_2026` (lines 383-391), same Zotero-arXiv shape: `@misc{ji_adampm_2026, title={AdamPM: Memory-Efficient LLM Optimization via Moment Pooling}, author={Xiaowei Ji and Lishan Xu and Jianing An and Jie Luo and Wenjun Wu and Lei Huang}, year={2026}, eprint={2610.24598}, archivePrefix={arXiv}, primaryClass={cs.LG}, url={https://arxiv.org/abs/2610.24598},}`. |
| 12 | `docs/paper/appendices/annex-1-optimizers.tex` | New `\paragraph{AdamPM (Pooled-Moment Adam).}` inside `\subsection{Large-Batch, Memory-Efficient, and Parameter-Free Optimizers}` (line 405), inserted **after the Q-APOLLO Pros block (ends line 693) and before `\paragraph{Anon ...}` (line 695)**. Content (matching the Muon² style, lines 1349-1431): (i) optional `\fitfigure{\includegraphics{figures/optimizers/adam-pm.pdf}}` — only if row 12b's figure is produced; (ii) intro: `AdamPM~\citep{ji_adampm_2026} compresses Adam's second moment by \textbf{group-mean pooling of adjacent entries}, leaving the first moment element-wise.`; (iii) `align` block with the equations of §2 above (moment EMA, reshape `(m,q,s)`, group mean, compressed EMA, bias corrections, Kronecker broadcast, update) + memory formula `$mn(1+1/s)$ vs $2mn$`; (iv) `\begin{algorithm}[H] \caption{AdamPM} \label{alg:adampm}` float mirroring Algorithm 1 (Require `W, T, s, η, β₁, β₂, ε`; steps: gradient, first-moment EMA, squared gradient, reshape, group mean with the last partial group padded by its mean, pooled EMA, bias corrections, broadcast, update); (v) scope paragraph: pools all matrices incl. embed and lm\_head; 1-D params as single-row/column matrices; `s=1` → Adam; `s ≥ n` → AdamPM-inf; the repo's decoupled `wd` extension; `O(log T/√T)` bound independent of `s` (Theorem 3.1); hyperparameter-transfer result; (vi) `\paragraph{Cons.}` — preconditioner coarsens with `s` (AdamPM-inf degrades at 350M/1.3B); row/col pooling only (no 2-D groups — paper Limitations); savings are optimizer-state only (first moment still full-size); (vii) `\paragraph{Pros.}` — ≈2× optimizer-state saving at s=8 (87.5 % V reduction) with Adam-level perplexity through 1.3B; direct hyperparameter transfer from Adam; model-agnostic, no SVD/projection compute; similarity to Adam grows with model scale (Fig. 4d). All formulations must match the arXiv paper verbatim (D11). |
| 12b | (optional) `docs/paper/figures/optimizers/adam-pm.{mmd,pdf,svg}` | Mermaid flowchart in the house style (see `anon.mmd` header `%%{init: {"theme":"neutral",...}}%%`; nodes: `g_t`, `m ← β₁m+(1−β₁)g`, `g² → reshape(m,q,s)`, `S̄ = mean_k R[:,:,k]`, `V̄ ← β₂V̄+(1−β₂)S̄`, `bcast V̂̄ ⊗ 1_s`, `θ ← θ − η·M̂/(√V̄+ε)`). Build: `( cd docs/paper/figures && ./build-figures.sh both )` — pnpm-based, auto-detects Chrome/Chromium (PUPPETEER_EXECUTABLE override). **Precedent allows omission**: `muon2`, `dion3`, `ada_muon` ship without figures. If Chromium/pnpm is unavailable, omit the `\fitfigure` line and the figure files. |
| 13 | `docs/paper/appendices/annex-summary-tables.tex` | (a) Row in `tab:all-optimizers` under `\textbf{Large-Batch, Memory-Efficient \& Parameter-Free}` (header line 136), after the Adafactor row (line 140): `AdamPM & Pooled second moment & 2 ($m$, $\bar{v}$ at $1/s$) & $\mathcal{O}(n)$ & lr, $\beta_1$, $\beta_2$, eps, wd, group size $s$, row/col & \citep{ji_adampm_2026} \\`. (b) Census line 100: "catalogs all 30 supported optimizers" → "all 31". |
| 14 | `docs/paper/sections/05-optimizer-families.tex` | Line 4 (exact current text read): "The system supports thirty optimizer families in eight categories: … memory-efficient and low-rank methods (Adafactor, GaLore, Lion, APOLLO, APOLLO-Mini, Q-APOLLO); …" → "supports thirty-one optimizer families"; memory-efficient list becomes "(Adafactor, GaLore, Lion, APOLLO, APOLLO-Mini, Q-APOLLO, AdamPM)". Category count "eight" unchanged. |
| 15 | `docs/paper-es/appendices/annex-1-optimizadores.tex` | Faithful Spanish translation of #12: `\paragraph{AdamPM (Adam con Momentos Agrupados).}` inside `\subsection{Optimizadores de Lote Grande, Eficientes en Memoria y Sin Tasa de Aprendizaje}` (line 405), after the Q-APOLLO block (lines 680-694), before `\paragraph{Anon ...}` (line 695). Equations untranslated; algorithm float `label{alg:adampm-es}` (match the neighbouring ES label convention — inspect lines 660-694 first); figure path if used: `\fitfigure{\includegraphics{../paper/figures/optimizers/adam-pm.pdf}}`; Pros/Cons headings copied from the ES neighbours. Formulations must match the EN annex and the paper (D11). |
| 16 | `docs/paper-es/appendices/annex-summary-tables.tex` | (a) Row mirroring #13 under the ES memory-efficient header, after the Adafactor row: `AdamPM & Segundo momento agrupado & 2 ($m$, $\bar{v}$ a $1/s$) & $\mathcal{O}(n)$ & lr, $\beta_1$, $\beta_2$, eps, wd, tamaño de grupo $s$, fila/columna & \citep{ji_adampm_2026} \\`. (b) Line 100: "cataloga los 30 optimizadores soportados" → "los 31 optimizadores soportados". |
| 17 | `docs/paper-es/sections/05-familias-optimizadores.tex` | Line 4 (exact current text read): "El sistema soporta treinta familias de optimizadores en ocho categorías: … métodos eficientes en memoria y de bajo rango (Adafactor, GaLore, Lion, APOLLO, APOLLO-Mini, Q-APOLLO); …" → "soporta treinta y una familias"; add "AdamPM": "(Adafactor, GaLore, Lion, APOLLO, APOLLO-Mini, Q-APOLLO, AdamPM)". |
| 18 | `docs/architecture.md` + `README.md` (census pass) | Every occurrence found by grepping the old counts. **`docs/architecture.md`**: line 26 §1 row `| Optimizer families | **27** | src/schema/_optimizer.yaml |` → `**31**` (⚠ pre-existing stale drift vs the 30-entry registry; see §9 A1); line 727 §8.1 heading `### 8.1 Optimizers (30)` → `(31)`; §8.1 table row (line 736) Memory-efficient cell: append `adam_pm` → `` `adafactor`, `lion`, `galore_adamw`, `apollo`, `apollo_mini`, `q_apollo`, `adam_pm` `` and extend the character cell with "pooled second moment (1/s state)"; §8.1 prefix-contract examples (lines 754-757) optionally add `adam_pm-group_size`. **`README.md`**: line 27 mermaid node `Train<br/>30 optimizer families` → `31 optimizer families`; line 93 `| Optimizer families | 30 across 7 categories |` → `31 across 7 categories`; line 149 Memory-Efficient row → append `` `adam_pm` ``, count `6` → `7` (⚠ row list pre-existing stale — missing `root`, `muon2`, `ea4llm`; see §9 A2). No other census numbers exist in `configs/README.md` beyond row 8's edits. |

### Hard rules restated

1. **Schema is the contract** (`additionalProperties: false`): the `^adam_pm-`
   `patternProperties` block in `_training.yaml` must exist and the runtime
   `OPTIMIZER_ALLOWED_KEYS` whitelist must match `x_parameter_reference`.
   Unknown keys raise at build time (`ensure_no_unknown_parameters`); the
   runtime schema loader logs `WARNING [schema] … unknown key(s) ignored` for
   keys the schema does not know *yet* — never read or implement an
   undocumented key.
2. Optimizer keys are prefixed `<name>-<group>_<param>` / `<name>-<param>`
   (AGENTS.md rule 7); groups are the 10 canonical `GROUP_NAMES`
   (`src/model/optimizer/base.py:19-30`).
3. Every preset/example is smoke-tested (rule 9): the new example YAML must
   validate + pass `load_training_config`.
4. **Papers are the theory source of truth** (D11): annex formulations must
   match arXiv:2610.24598; `paper-es` must translate them faithfully.
5. No legacy names (rule from `test_naming_conventions.py`, runs in full suite).

## 5. Ordered execution checklist

```markdown
Implementation (files 1-9)
[ ] 1.  Write src/model/optimizer/adam_pm.py (AdamPM class, equations mapped line-by-line)
[ ] 2.  Export from src/model/optimizer/__init__.py
[ ] 3.  Register in src/model/optimizer/factory.py (import, OPTIMIZER_REGISTRY, OPTIMIZER_ALLOWED_KEYS, constructor branch)
[ ] 4.  Extend src/schema/_optimizer.yaml (enum + EN/ES tables + EN/ES suffix bullets + by_optimizer + References ×2)
[ ] 5.  Append ^adam_pm- if/then block to src/schema/_training.yaml
[ ] 6.  Extend optimizer_kwargs EN/ES descriptions in src/schema/_sbert.yaml
[ ] 7.  Add configs/examples/es_arch_gqa_adam_pm.yaml
[ ] 8.  Update configs/README.md (chooser list, suffix reference, example keys)
[ ] 9.  Append adam_pm to full_tests/run_e2e.py OPTIMIZERS + param map
Tests (file 10)
[ ] 10. Extend tests/test_optimizer_factory.py (registry set + BuildOptimizerAdamPMTests)
[ ] 11. Run targeted gates (commands below) → all green
[ ] 12. Run full suite → all green
Docs (files 11-18)
[ ] 13. Add ji_adampm_2026 to docs/bibliography/optimizers.bib
[ ] 14. (optional) figures/optimizers/adam-pm.mmd + build-figures.sh both
[ ] 15. EN annex paragraph + summary row + 05-families sentence
[ ] 16. ES annex paragraph + summary row + 05-familias sentence
[ ] 17. Census pass: docs/architecture.md + README.md (row 18; confirm A1/A2 decisions first)
PDFs
[ ] 18. Rebuild both PDFs (commands below) + verify (no Fatal error, no undefined citations, PDFs newer than .tex/.bbl)
Website sync
[ ] 19. Run /sync-website steps (§6) — mirror copy, HTML rebuild, Validation Gate
Git
[ ] 20. /make-a-commit for the main repo, then the mirror (§7)
```

### Commands to run at implementation time (not now)

```bash
# targeted gates
uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py -v --tb=short -p no:warnings

# full suite
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings

# fast e2e smoke — adam_pm is appended LAST in OPTIMIZERS, so --limit must be
# raised until opt_adam_pm is reached; the precise form:
uv run --extra cpu --extra train python full_tests/run_e2e.py --only opt_adam_pm
# or for the tail of the category:
uv run --extra cpu --extra train python full_tests/run_e2e.py --category opt --limit 5
```

```bash
# PDF rebuild, from the repository root — only at implementation time
cd docs/paper
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
rm -f *.aux *.log *.out *.toc *.blg

cd ../paper-es
pdflatex -interaction=nonstopmode paper-es.tex
bibtex paper-es
pdflatex -interaction=nonstopmode paper-es.tex
pdflatex -interaction=nonstopmode paper-es.tex
rm -f *.aux *.log *.out *.toc *.blg
```

**PDF preconditions** (verified during planning: `/usr/bin/pdflatex`,
`/usr/bin/bibtex`, `/usr/bin/pandoc` all present):

- ⚠️ `docs/paper/paper.tex`, `docs/paper/paper.pdf`, `docs/paper-es/paper-es.tex`,
  `docs/paper-es/paper-es.pdf`, `annex-20-*-experimental.tex`,
  `annex-20-experiment-harness.tex`, `09-*.tex` are **already modified
  (uncommitted)** in the worktree — the rebuild will compile the user's
  in-flight changes into the rebuilt PDFs. Confirm with the user whether to
  (a) commit their work first, or (b) rebuild on top of it. See §9 R1.
- Fallback if TeX is unavailable: leave the `.tex`/`.bib` sources updated and
  note that the rebuild must happen on a TeX machine before the next release
  (the website sync copies prebuilt PDFs only).

**Verification required after compilation**: no `Fatal error` and no
`Citation .* undefined` in the logs; `docs/paper/paper.pdf` and
`docs/paper-es/paper-es.pdf` newer than every edited `.tex`/`.bbl`; the new
`\citep{ji_adampm_2026}` resolves (appears in `paper.bbl` / `paper-es.bbl`);
`git status --short docs/paper docs/paper-es` shows only intended
`.tex`/`.bbl`/`.pdf` changes (plus the pre-existing user modifications).

## 6. Website sync (planned, not executed)

Follow `.opencode/command/sync-website.md` end-to-end. Steps specific to this
change:

1. **State**: `bash .opencode/command/sync-website-state.sh` → read `[drift]`.
2. **Copy** (main → `erickfmm.github.io/frankenstein-transformer/`, plain
   `cp` + `mkdir -p`, no rsync): `src/schema.yaml` → `schema.yaml`;
   `src/schema/**` → `schema/**` (covers `_optimizer.yaml`, `_training.yaml`,
   `_sbert.yaml`); `configs/examples/es_arch_gqa_adam_pm.yaml` →
   `examples/examples/…`; `configs/README.md` → `examples/README.md`;
   `docs/paper/{paper.tex,paper.bbl,paper.pdf,sections/05-optimizer-families.tex,appendices/annex-1-optimizers.tex,appendices/annex-summary-tables.tex}` →
   `paper/en/…`; the four `docs/paper-es/…` counterparts → `paper/es/…`;
   `docs/bibliography/optimizers.bib` → `paper/bibliography/`;
   `docs/architecture.md` → `paper/specs/architecture.md`. `README.md` has no
   mirror counterpart (site copy is `index.html`, untouched unless a probe
   fails). If the figure was produced:
   `docs/paper/figures/optimizers/adam-pm.{mmd,pdf,svg}` →
   `paper/figures/optimizers/`.
3. **Top-level PDF copies**: `cp paper/en/paper.pdf paper/en.pdf` and
   `cp paper/es/paper-es.pdf paper/es.pdf` (probes `paper-en-copy`/`paper-es-copy`).
4. **Paper-tab HTML rebuild** (mirror-only pipeline, pandoc installed):

   ```bash
   cd erickfmm.github.io/frankenstein-transformer/paper
   python3 flatten_tex.py en/paper.tex    en/paper.bbl    /tmp/paper-en-flat.tex --web --manifest /tmp/paper-en-figures.json
   pandoc /tmp/paper-en-flat.tex -f latex -t html5 --standalone --mathjax --toc --toc-depth=3 -V lang=en -o paper-en.html
   python3 postprocess_html.py --manifest /tmp/paper-en-figures.json paper-en.html
   python3 flatten_tex.py es/paper-es.tex es/paper-es.bbl /tmp/paper-es-flat.tex --web --manifest /tmp/paper-es-figures.json
   pandoc /tmp/paper-es-flat.tex -f latex -t html5 --standalone --mathjax --toc --toc-depth=3 -V lang=es -o paper-es.html
   python3 postprocess_html.py --manifest /tmp/paper-es-figures.json paper-es.html
   ```
5. **Validation Gate** (in order, each must pass):
   - `uv run --extra cpu --extra train python full_tests/param_count_check.py` →
     prints `… configs counted, 0 failures` (regenerates
     `full_tests/param_truth.json`; the new example config is one more counted
     preset — `param_truth.json` changes are intentional);
   - `node full_tests/param_estimate_check.mjs erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js full_tests/param_truth.json --tol 0.01` →
     exit 0 `ALL PASS` (architecture unchanged ⇒ estimator untouched; it
     counts model params, not optimizer state);
   - state-script probes: `schema-refs`, `example-yaml`, `cli-flags` all PASS
     (plus `estimator` and `paper-figures-*` probes).
6. **Leave both repos uncommitted** until the commit phase (§7).

## 7. Commit plan (planned, not executed)

Follow `.opencode/command/make-a-commit.md` (English, conventional tag,
structured body, never stage `*.aux *.log *.out *.toc *.blg` or caches).

**Commit 1 — main repo** (`git push origin HEAD:main` after):

```
feat(optimizer): add AdamPM (arXiv:2610.24598)

Implement AdamPM ("Adam with Pooled Moments"), a memory-efficient Adam
variant that pools every s adjacent second-moment entries within rows or
columns into one shared group mean, shrinking the V-state to 1/s
(mn(1+1/s) optimizer-state elements vs Adam's 2mn). The first moment
stays element-wise; s=1 reduces to vanilla Adam and s >= axis length to
the AdamPM-inf row/column scalar. Hyperparameters transfer directly
from Adam (paper Theorem 3.1: same O(log T/sqrt T) bound, independent
of s; robust for group sizes 2-8).

Implementation:
- src/model/optimizer/adam_pm.py: torch.optim.Optimizer subclass;
  Algorithm 1 mapped line-by-line; last partial group averaged over its
  existing entries (paper padding rule); decoupled weight decay via the
  shared wd_* suffixes (repo extension; the paper defines none)
- factory: registry entry adam_pm + OPTIMIZER_ALLOWED_KEYS
  (group_size, pool_dim) + constructor branch
- schema: optimizer_class enum, EN/ES use-case rows, EN/ES suffix
  bullets, by_optimizer.adam_pm, ^adam_pm- if/then block in
  _training.yaml, SBERT optimizer_kwargs descriptions

Docs and examples:
- bibliography entry ji_adampm_2026; EN+ES annex-1 paragraph with
  algorithm float; EN+ES summary-table rows; EN+ES family-section
  sentences; census 30 -> 31 (architecture.md 8.1, README, both annex
  summary tables); rebuilt paper.pdf / paper-es.pdf
- configs/examples/es_arch_gqa_adam_pm.yaml (MLM, schema-smoke-tested)
- configs/README.md chooser/suffix/example-key lists; run_e2e.py
  OPTIMIZERS + param map

Tests:
- tests/test_optimizer_factory.py: registry membership +
  BuildOptimizerAdamPMTests (build, step, pooled-state shapes, s=1
  Adam parity, partial-group padding, suffix coercion, unknown-key
  rejection)
```

Stage exactly:
`src/model/optimizer/adam_pm.py src/model/optimizer/__init__.py src/model/optimizer/factory.py src/schema/_optimizer.yaml src/schema/_training.yaml src/schema/_sbert.yaml configs/examples/es_arch_gqa_adam_pm.yaml configs/README.md full_tests/run_e2e.py tests/test_optimizer_factory.py docs/bibliography/optimizers.bib docs/paper/appendices/annex-1-optimizers.tex docs/paper/appendices/annex-summary-tables.tex docs/paper/sections/05-optimizer-families.tex docs/paper/paper.bbl docs/paper/paper.pdf docs/paper-es/appendices/annex-1-optimizadores.tex docs/paper-es/appendices/annex-summary-tables.tex docs/paper-es/sections/05-familias-optimizadores.tex docs/paper-es/paper-es.bbl docs/paper-es/paper-es.pdf README.md docs/architecture.md`
(+ `docs/paper/figures/optimizers/adam-pm.{mmd,pdf,svg}` if the figure was
produced; + `full_tests/param_truth.json` if the Validation Gate regenerated
it). ⚠️ Do **not** stage the pre-existing user modifications
(`docs/paper/paper.tex`, `docs/paper-es/paper-es.tex`, `annex-20-*`, `09-*.tex`)
unless the user folded them into their own commit first — but note the rebuilt
PDFs will contain them either way (see R1).

**Commit 2 — mirror** (`git -C erickfmm.github.io …` then
`git -C erickfmm.github.io push origin HEAD:main`):

```
docs(site): sync AdamPM optimizer (arXiv:2610.24598)

Mirror the AdamPM addition: schema modules (+ adam_pm enum/suffix
reference/if-then rule), example YAML es_arch_gqa_adam_pm.yaml,
configs README, EN/ES paper sources + rebuilt PDFs and summary tables,
bibliography, architecture spec, Paper-tab HTML (flatten_tex + pandoc).
Validation gate: param_count_check 0 failures, estimator <= 1% drift,
schema-refs / example-yaml / cli-flags probes PASS.
```

Stage exactly the files listed in §6 steps 2-4 under
`erickfmm.github.io/frankenstein-transformer/` (schema, example, READMEs,
paper en/es tex+bbl+pdf, figures if any, bibliography, specs/architecture.md,
paper-en.html, paper-es.html, en.pdf, es.pdf) — never `index.html` or
`ft-param-estimator.js` unless a probe fix was needed.

## 8. Verification matrix

| Gate | Command (implementation time) | Expected result |
|---|---|---|
| Targeted optimizer/schema gates | `uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py -v --tb=short -p no:warnings` | All pass; `test_registry_contains_all_required` includes `adam_pm`; prefix↔if/then parity holds (schema-driven test, `tests/test_schema_consistency.py:76`); the new example YAML loads |
| Full suite | `uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings` | Whole suite green, incl. `test_naming_conventions.py` |
| s=1 Adam parity | `BuildOptimizerAdamPMTests.test_s1_matches_vanilla_adam` | `adam_pm(group_size=1)` step ≡ reference Adam step (`atol 1e-7`) |
| Pooled-state shape | `test_second_moment_state_is_pooled` | `exp_avg_pooled.numel() == ⌈c/s⌉·r` (row) / `⌈r/s⌉·c` (col) |
| Partial-group padding | `test_partial_group_mean_pads_with_mean` | Last group stat = mean of existing entries (sum ÷ count) |
| e2e smoke | `uv run --extra cpu --extra train python full_tests/run_e2e.py --only opt_adam_pm` (or `--category opt --limit 5` raised until reached) | `opt_adam_pm` combo trains/evaluates; no `ValueError: Unknown parameters` |
| PDF EN | `pdflatex→bibtex→pdflatex×2` in `docs/paper` | No `Fatal error`; no `Citation 'ji_adampm_2026' undefined`; `paper.pdf` newer than edited `.tex`/`.bbl` |
| PDF ES | same in `docs/paper-es` | Same checks for `paper-es.pdf` |
| Post-compile git hygiene | `git status --short docs/paper docs/paper-es` | Only intended `.tex`/`.bbl`/`.pdf` (plus pre-existing user modifications) |
| Param ground truth | `uv run --extra cpu --extra train python full_tests/param_count_check.py` | `… configs counted, 0 failures`; regenerates `param_truth.json` (+1 example config) |
| JS estimator | `node full_tests/param_estimate_check.mjs erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js full_tests/param_truth.json --tol 0.01` | Exit 0, `ALL PASS` (no estimator edit expected: model architecture unchanged) |
| State probes | `bash .opencode/command/sync-website-state.sh` | `schema-refs`, `example-yaml`, `cli-flags`, `estimator-drift`, `paper-figures-en/es`, `paper-en-copy`, `paper-es-copy` all PASS; drift list empty for the synced pairs |

## 9. Risks, assumptions and open questions

**Risks**

- **R1 — dirty worktree (highest)**: `git status --short` already shows 8
  modified files (`docs/paper{,-es}/{paper*.tex,paper*.pdf,annex-20-*,09-*}`).
  The implementation-time PDF rebuild will bake those in-flight edits into the
  rebuilt PDFs, and staging must avoid them. Ask the user to commit or stash
  their experiment-harness work before starting.
- **R2 — census drift is pre-existing**: `architecture.md:26` says 27 (true
  registry count is 30 → 31); README categories table (lines 145-153) omits
  `root`, `muon2`, `ea4llm` and sums to 27 while claiming "30 across 7
  categories"; §8.1 has 9 category rows vs README's 7 vs 05-families' "eight".
  Minimal-touch policy: bump only the live 30s and add `adam_pm` to its rows;
  repairing the missing rows / recounting categories needs explicit user
  approval (A2).
- **R3 — abstract counts**: both abstracts say "twenty-four / veinticuatro
  familias de optimizadores", tied to the 868-config measurement campaign.
  Do **not** bump — the campaign did not include AdamPM (nor root/muon2/ea4llm).
- **R4 — e2e reach**: `adam_pm` is appended last in `OPTIMIZERS`; the default
  `--limit` may not reach it. Use `--only opt_adam_pm`.
- **R5 — trainer integration**: AdamPM needs no closure and no
  `gradient_accumulation_steps: 1` constraint (unlike `eve`/`ea4llm`); the
  plain `optimizer.step()` path suffices. No SBERT restriction (uses the same
  enum via `$ref`, `_sbert.yaml:395`).

**Assumptions to verify at implementation time**

- A1 (needs user decision): fix `architecture.md:26` §1 count `27 → 31`
  (recommended — stale even before this change) or bump mechanically
  `27 → 28`. Row 18 assumes the true count 31.
- A2 (needs user decision): repair the README categories table by adding the
  missing `root`, `muon2`, `ea4llm` rows so it sums to 31, or leave the
  pre-existing gap and only append `adam_pm` to Memory-Efficient (6→7).
  Plan assumes minimal touch.
- A3: >2-D parameters (e.g. conv kernels) are flattened to a 2-D view
  (`rows = numel/last_dim` for `row`, `first_dim` for `col`) — the paper only
  defines matrices and 1-D vectors; the annex should state this
  generalization.
- A4: `pool_dim` coercion shape `str(...).strip().lower()` mirrors
  `scale_type`/`proj_type` (`factory.py:346-347`); strict validation happens
  in `AdamPM.__init__`.
- A5: `β₂` bias correction is applied to the pooled statistic before the
  broadcast (Algorithm 1: "Correct bias in the compressed second moment" then
  broadcast); ε is added **after** bias correction, inside `√V̂̄ + ε`,
  matching both the paper and `adamw_fallback_update`.
- A6: per-group `betas`/`eps` enter the pooled update; `group_size`/
  `pool_dim` are global-only (single constructor argument), consistent with
  `specific_global_suffixes`.
- A7: the factory passes `group_size`/`pool_dim` to the constructor and also
  writes them onto every converted group so `step()` can read them per group;
  mirror the per-group injection style of the `adafactor` branch
  (`factory.py:263-269`) — implementation detail covered by the test class.

**Naming collisions**: none — no registry key, module, or prefix starting with
`adam_pm`/`AdamPM`; the figure stem `adam-pm` is free in
`docs/paper/figures/optimizers/` (existing stems checked).

**Task/mixer compatibility**: none restricted — valid for every task
(`mlm`, `sbert`, `causal_lm`, vision, audio) and every mixer; the paper itself
validates pre-training, GLUE fine-tuning and multimodal SFT.

**Open questions (paper)**

- The paper also proposes `AdamPM-M` / `AdamPM-M†` (first-moment pooling) and
  reports they degrade — intentionally not planned. `AdamPM-col` **is** covered
  by `pool_dim: col`; `AdamPM-inf` falls out of large `group_size`.
- Publication venue beyond arXiv unknown at plan time; if the paper is
  accepted somewhere before implementation, update the bib entry accordingly.
- Default `group_size`: the paper prescribes no single default; 8 is chosen
  from "group sizes between 2 and 8 generally yield robust performance" +
  best GLUE average at s=8. If maximal fidelity-to-Adam is preferred, 2 or 4
  are equally defensible.

## 10. Out of scope

This planning command made **no** changes to code, schema, presets, tests,
docs, PDFs, website-mirror files or git history. Nothing outside the plan
file was written. No `pytest`, no `run_e2e.py`, no `param_count_check.py`, no
`param_estimate_check.mjs`, no `pdflatex`/`bibtex`/`pandoc`/`flatten_tex.py`,
no `uv sync`, no training or inference, no mirror copy/sync, and no
`git add/commit/push/checkout/stash/reset` were executed. Implementation is a
separate, explicitly requested step.
