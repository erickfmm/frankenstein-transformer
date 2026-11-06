# Plan: add LSA mixer (arXiv 2610.42147)

> **Status**: plan only — produced by the `/add-mixer` plan-mode command on 2026-11-06.
> Nothing outside the plan directory has been created, edited or deleted by this command.
> **Note**: the `/add-mixer` command names `.opencode/plan/add-mixer-2610.42147.md` (repo) as the
> destination, but the plan-mode runtime only permits writes under `/home/erickfmm/.opencode/plan/`.
> The plan was therefore written here; copy/move it to the repo's `.opencode/plan/` (where
> `add-mixer-2609.28399.md` lives) when leaving plan mode.

## 1. Header

**Paper**: *Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention*, Ales
Kubicek*, Andrea Caciolai*, Liangzu Peng*, Sukrit Sumant, Benjamin Hoover, Fraser Greenlee,
Ruairidh M. Battleday, Radim Urban, Torsten Hoefler, James C. R. Whittington† (Prefrontal AI,
London + ETH Zurich), arXiv:2610.42147v2 [cs.AI], 30 Oct 2026 (v2: 3 Nov 2026).

**One-paragraph summary.** At ultra-long context (>1M tokens) the indexer of a sparse-attention
layer — not the attention — dominates the cost (98% of prefill at 10M tokens, §2/Fig. 2). LSA
replaces the per-key/per-block scan with **semantic clustering**: keys are L2-normalized, pooled
every `n_pool` tokens, and every complete `n_cluster`-key *cluster block* is bottom-up
Ward-clustered into `m = n_cluster/n̄_b` immutable *buckets* whose unit-norm centroids are cached
in a small bucket cache. Each query then scores only the cached centroids, keeps the top-`p`
buckets, expands them best-first into `k` long-range keys, and attends to `n_sink` sinks +
selection + an `n_w = n_cluster` window (~2,560 keys/query at 128k instead of 128k). Two trainable
designs are proposed: **D-LSA**, distilled into pre-trained models via extra index projections
(`W_Q^(idx), W_K^(idx)`, eq. 1; forward-KL distillation, eq. 2) — on GLM-5.3-FLASH at 10M tokens
it speeds one attention layer by 22.7× (72.4×) in prefill and 4.6× (42.5×) in decode over its
original sparse (full) attention; and **N-LSA**, trained **natively on the next-token-prediction
loss alone** with the attention's own keys (single KV head, bucket score `s_{i,b} = max_h q̂_{i,h}·c_b`,
eq. 3, gradient stopped through the selection, **no positional encoding**, **zero extra
parameters**) — a 0.5B 24-layer hybrid trained at 128k reaches full-attention validation loss
(1.085 vs 1.085; NSA 1.117, PISA 1.100, SWA 1.154) with up to 1.6× per-layer training speedup.
**This plan implements N-LSA** — the variant this repo can train from scratch (user-confirmed);
D-LSA needs a frozen-teacher distillation pipeline this repo does not have (recorded in §9).
It belongs in this repo as the `sparse/` family's first *semantic* (content-clustered) selector,
complementing the positional/block selectors (`nsa_attn`, `msa_attn`, `sparda_attn`).

---

## 2. Paper extract (verbatim, everything downstream depends on this)

### 2.1 Forward equations

N-LSA reuses the standard attention projections and defines the selection process over the
attention's own keys/queries. Paper notation (Table 5): `T` context size; `i, j, h` query/key
position and head; `W_Q,K,V^(attn)` attention projections; `n_h, d_h` query heads, head dim;
`n_pool` keys summed per pool key; `n_cluster, m` keys per cluster block, buckets per block;
`n̄_b, n_max_b` average/max keys per bucket; `k_sum_b, |b|` bucket key sum and key count;
`a_b, Δ(b,b')` alignment and merge cost; `c_b, B_cache^(idx)` unit-norm centroid, bucket cache;
`s_{i,b}, p` bucket score, buckets selected per query; `k, n_sink, n_w` long-range, sink and
window key counts; `J_i` key positions query `i` attends to.

**Clustering (paper §3.1, numbered as in Fig. 3):**

1. **Key normalisation.** Each index key is first L2 normalised:
   `k_j^(idx) ← k_j^(idx) / ‖k_j^(idx)‖₂`
2. **Pooling.** "every `n_pool` consecutive index keys are summed into one pool key
   `k_pool^(idx)`, so that clustering runs over `T/n_pool` pool keys rather than `T` keys."
3. **Clustering.** "Keys are clustered in cluster blocks of `n_cluster` consecutive keys. Once a
   cluster block contains all its keys, its pool keys are clustered bottom up with Ward's method
   (Ward, 1963) into `m` buckets. Each pool key starts as a bucket of its own, and each bucket
   keeps the sum of its keys, whose normalisation is its centroid `c_b`. In every round, all
   pairs of buckets that are each other's cheapest merge are merged in parallel. The cost of a
   merge is how much it increases the keys' total squared distance from their centroid (Ward's
   criterion). Merging stops once there are `m = n_cluster/n̄_b` buckets … A merge is skipped if
   the merged bucket would hold more than `n_max_b` keys." **"Once a cluster block is clustered,
   its buckets never change."**
4. **Bucket cache.** `B_cache^(idx)` stores the centroid `c_b` of every bucket and an index table
   of the positions of its keys.

**Search (paper §3.2):**

A. **Query normalisation.** `q_i^(idx) ← q_i^(idx) / ‖q_i^(idx)‖₂`
B. **Scoring.** "The index query scores the unit-norm centroid of every bucket in the cluster
   blocks before its own via `s_{i,b} = c_b · q_i^(idx)`, which compares buckets of any size on
   one scale, and selects the top-`p` buckets. Its own cluster block is not clustered yet, and
   its keys are covered by the window instead." For **N-LSA** (eq. 3, §3.4):

   ```
   s_{i,b} = max_h  q̂_{i,h} · c_b                                      (3)
   ```

   "where `q̂_{i,h}` is the normalised query of head `h`" — i.e. with the single shared KV head,
   each bucket is scored by the **best similarity any query head gives its centroid**; the
   resulting index list `J_i` is shared across all query heads.
C. **Expansion.** "The selected `p` buckets are expanded into their keys, best first, until `k`
   'long-range' keys are selected. As every bucket holds at least `n_pool` keys, `p` is chosen
   such that `p · n_pool ≥ k`, and so expansion always fills `k` keys. Besides the `k`
   long-range keys, every query attends to `n_sink` sink keys (Xiao et al., 2024b) and the `n_w`
   most recent keys (window). A key of a selected bucket that falls in the sink or the window is
   already attended to, and the expansion moves on to the next key instead. We set the window to
   one cluster block, `n_w = n_cluster`, so that the window covers every key not yet clustered.
   This gives each query `i` a set `J_i` of at most `n_sink + k + n_w` key indices."
D. **Attention.** "Attention is then computed only over the keys indexed by `J_i`. A single
   `J_i` is shared across all query heads." Per head (Appendix F.3, N-LSA training kernels):
   `z_{i,j} = q_i · k_j / √d_h`; the per-head softmax is renormalised over the shared selection
   (cf. the Top-k Oracle protocol of Appendix A: "sum over all heads, pick top-k, renormalise
   per head over the shared selection").

**N-LSA specifics (paper §3.4, quoted):** "N-LSA trains LSA natively with the model, using only
the NTP loss. As the model learns with the selection from the start, we cluster the keys of the
attention itself, without dedicated index projections. We use a single key-value head, such that
one key per token serves every query head, and score each bucket by the best similarity any
query head gives its centroid … **We stop the gradient through the selection**, and the NTP loss
trains the attention projections only through the keys each query attends to, the selected keys
plus the sink and the window. N-LSA therefore **adds no parameters** to the model and needs no
teacher, no index projections and no auxiliary loss. The attention layers of N-LSA use **no
positional encoding**, as is becoming more common in hybrid models, and RoPE could be supported
by clustering only the RoPE-free part of the keys, as D-LSA does on QWEN-3.8-27B."

**D-LSA (out of scope, recorded for the annex):** extra per-layer projections
`k_j^(idx) = k_j^(attn) W_K^(idx)`, `q_i^(idx) = Σ_h q_{i,h}^(attn) W_Q^(idx)` (eq. 1) with
`d_idx = 64`; distillation minimises the forward KL `L_i = KL(A_i ‖ Â_i)` between the frozen
layer's head-averaged attention `A_{i,j} = (1/n_h) Σ_h softmax_j(q_{i,h}·k_j/√d_h)` and the
indexer softmax `Â_{i,j} = softmax_j(q_i^(idx)·k_j^(idx)/τ)` (eq. 2); a "D-LSA (Training-Free)"
variant replaces the projections with the leading principal directions of the keys.

**Tree variant (out of scope, future work):** levels ℓ ≥ 1 hold "node buckets" of `n_{b,ℓ}`
buckets each; beam search keeps the best `p_ℓ` per level (Algorithm 2); "With a constant beam per
level, a query scores O(log T) centroids, which makes the prefill of the indexer subquadratic,
O(T log T), where the flat search is quadratic." N-LSA was trained with flat search only
("We train N-LSA with flat search only, and leave native training with tree search to future
work", §4.4).

### 2.2 Algorithms (verbatim, Appendix B)

**Algorithm 1 — Ward clustering of one cluster block.**
Require: index keys `k_j^(idx)` of a complete cluster block; sizes `n_cluster, n_pool, m,
n_max_b`. Ensure: `B_cache^(idx)` with `m` centroids `c_b` and their key positions.
*Pooling:* 1: for all pool keys `k_pool^(idx)`, in parallel do 2: `k_pool^(idx) ← the sum of its
n_pool consecutive k_j^(idx)`; 3: end for; 4: start one bucket `b` per pool key,
`k_sum_b ← k_pool^(idx)`, `|b| ← n_pool`. *Merging:* 5: while more than `m` buckets remain do —
every pair of buckets, in parallel: 6: `G ← [k_sum_b · k_sum_{b'}]` (Gram matrix);
7: `a_b ← √G_{bb}`, `a_{b∪b'} ← √(G_{bb} + G_{b'b'} + 2G_{bb'})`; 8: `Δ(b,b') ← a_b + a_{b'} −
a_{b∪b'}` (merge cost); 9: `π(b) ← argmin_{b'} Δ(b,b')` s.t. `|b|+|b'| ≤ n_max_b`; then "the
cheapest mutual pairs `π(b)=b', π(b')=b`, at most as many as buckets above `m`, in parallel";
10: `k_sum_b ← k_sum_b + k_sum_{b'}`, `|b| ← |b| + |b'|`, remove `b'`; 11: end while.
*Bucket cache:* 12: for all buckets `b`, in parallel: 13: `c_b ← k_sum_b / ‖k_sum_b‖`;
14: store `c_b` and the positions of the keys of `b` in `B_cache^(idx)`; 15: end for.
*(D-LSA Tree only: 16: merge blocks of `n_{cluster,ℓ}` buckets into node buckets.)*

**Algorithm 2 — Beam search of one query in D-LSA (Tree).** Require: `q_i^(idx)`, levels
`ℓ = 0..ℓ_max` of `B_cache^(idx)`; sizes `p, p_ℓ, n_start`. 1: `C ← all node buckets of the
lowest level ℓ with ≤ n_start`; 2: while `ℓ > 0` do — every candidate in `C`, in parallel:
3: `s_{i,b} ← q_i^(idx) · c_b`; 4: `P ← the best p_ℓ candidates of C`; 5: `C ← the children of
P ∪ the open nodes of level ℓ−1`; 6: `ℓ ← ℓ−1`; 7: end while; 8–9: score every bucket in `C` and
return the top-`p` buckets of `C`.

### 2.3 Design parameters (paper Table 2, verbatim values)

| Unit | block | block | **bucket** | **bucket** | block | block | **bucket** | bucket |
|---|---|---|---|---|---|---|---|---|
| Model | QWEN-3.8-27B | | | | GLM-5.3-FLASH | | | |
| Search | flat | tree | flat | tree | flat | tree | flat | tree |
| `k` | 2,048 | 4,608 | 4,096 | 4,096 | 2,048 | 4,608 | 2,048/4,096 | 4,096 |
| `p` | 512 | 576 | 1,024 | 1,024 | 512 | 576 | 512/1,024 | 1,024 |
| `n_sink, n_w` | -, - | 8, 16 | **4, 512** | 4, 512 | -, - | 8, 16 | 4, 512 | 4, 512 |
| `d_idx` | 128 | 256 | 64 | 64 | 128 | 512 | 64 | 64 |
| `n_pool` | 4 | 8 | **4** | 4 | 4 | 8 | 4 | 4 |
| `n_cluster` | - | - | **512** | 512 | - | - | 512 | 512 |
| `n̄_b, n_max_b` | - | - | **8, 16** | 8, 16 | - | - | 8, 16 | 8, 16 |
| `n_{b,ℓ}` | - | 2 | - | 4 | - | 2 | - | 4 |
| `n_{cluster,ℓ}` | - | - | - | 4,096 | - | - | - | 4,096 |
| `p_ℓ` | - | 576 | - | 2,048 | - | 576 | - | 2,048 |

**N-LSA reference configuration (§4.4 + Appendix F.3, quoted):** hybrid model of about 0.5B
parameters, 24 layers, "every fourth layer is an attention layer and the others are Gated
DeltaNet layers"; "Each attention layer has 8 query heads over a single key-value head of 128
dimensions, without positional encoding." WSD schedule: 100B tokens FineWeb-Edu @ 4k
(warmup+stable) then 12B tokens LongDocs @ 128k (decay). "N-LSA selects `k = 2048` long-range
keys with flat search, next to a window of 512 keys and 4 sink keys" ("Every sparse method
attends to about 2,560 keys per query"). Appendix F.3: at 128k tokens "a row holds at most
16,384 buckets, from which a query selects `p = 512` for `k = 2048` keys, next to a window of
512 keys and 4 sink keys"; "the selection clusters and scores the attention's own normalised
keys and queries, of dimension 128, and takes the best score of the 8 query heads (Equation 3)";
"The keys change at every step as the attention projections learn, so every step clusters all
keys again" (training); "As N-LSA stops the gradient through the selection, the clustering and
the scoring of the centroids need no backward pass, and the only new backward kernel is that of
the sparse attention."

### 2.4 Parameter inventory and exact parameter-count formula

N-LSA **adds no parameters** (§3.4). Learned tensors are exactly the standard attention
projections (this repo: bias-free, `BitLinear` when BitNet is on):

| Tensor | Shape | Notes |
|---|---|---|
| `W_Q` (`q_proj`) | `(H, H)` where `H = nH·hd` | full query projection |
| `W_K` (`k_proj`) | `(H, nKv·hd)` | GQA/MQA width; paper: `nKv = 1` |
| `W_V` (`v_proj`) | `(H, nKv·hd)` | idem |
| `W_O` (`out_proj`) | `(H, H)` | full output projection |

No index projections (D-LSA only), no gates, no learned centroids — bucket centroids/tables are
**computed buffers, not parameters** (and not quantized). Therefore, with `H = hidden_size`,
`nH = num_heads`, `nKv = num_kv_heads`, `hd = H/nH`:

```
P_lsa = 2·H² + 2·H·(nKv·hd)          (bias-free; identical for BitNet — same shapes)
      = 2·H²·(1 + nKv/nH)
```

This is exactly `msa_attn`'s estimator formula minus the index projections
(`ft-param-estimator.js` case `msa_attn`, line 190: `2*H*H + 2*(H*kv) + H*(nKv*idx) + H*idx`),
so the JS case for `lsa_attn` is `return 2 * H * H + 2 * (H * kv);`.
`full_tests/param_count_check.py` (which builds the model on the meta device and counts) must
agree to ≤ 1% (`full_tests/param_estimate_check.mjs --tol 0.01`).

### 2.5 Hyperparameters / knobs → `model.attention.lsa.*`

| Schema key | Flat dataclass field | Type | Default | Constraints / coercion |
|---|---|---|---|---|
| `model.attention.lsa.num_long_range` | `lsa_num_long_range` | int | **2048** | `k ≥ 1`; clipped to available long-range keys at runtime |
| `model.attention.lsa.num_buckets` | `lsa_num_buckets` | int | **512** | `p ≥ 1`; runtime auto-clamps `p ← max(p, ceil(k/n_pool))` (paper: "p is chosen such that p·n_pool ≥ k"); clipped to available buckets |
| `model.attention.lsa.num_sinks` | `lsa_num_sinks` | int | **4** | `≥ 1` and `< cluster_block` (causal-clipped in decoder) |
| `model.attention.lsa.pool_size` | `lsa_pool_size` | int | **4** | `n_pool ≥ 1` |
| `model.attention.lsa.cluster_block` | `lsa_cluster_block` | int | **512** | `n_cluster ≥ 2`; equals the window `n_w` (paper: "we set the window to one cluster block") |
| `model.attention.lsa.keys_per_bucket` | `lsa_keys_per_bucket` | int | **8** | `n̄_b ≥ 1`, must divide `cluster_block` (⇒ `m = n_cluster/n̄_b` buckets per block; default 512/8 = 64) |
| `model.attention.lsa.max_keys_per_bucket` | `lsa_max_keys_per_bucket` | int | **16** | `n_max_b ≥ keys_per_bucket` (merge cap) |

All seven values come from Table 2 (bucket columns) + §4.4's N-LSA settings
(2048/512/4/4/512/8/16). The knob file follows `_attention_swa.yaml`'s **declared-defaults**
style (defaults in schema **and** `_DEFAULT_PAIRS` entries), not the `examples:`-only style of
`_attention_sparse.yaml`.

**Edge-case decisions to encode in the module** (documented as implementation notes):
- `S < n_cluster` ⇒ no complete cluster block ⇒ `J_i = sinks ∪ window` (paper-exact: "the window
  covers every key not yet clustered"); bucket scoring/expansion is skipped.
- `S % n_pool ≠ 0` within a complete block ⇒ the block's last pool key sums the remaining
  `< n_pool` keys (buckets count *real* keys, so `m·n̄_b = n_cluster` still holds).
- `T % n_cluster ≠ 0` ⇒ the trailing partial block is unclustered and belongs to the window.
- Decoder: a query's candidates are the complete blocks **before its own block**; its own block
  is the window, clipped causally (`keys ≤ i`). Sinks are the first `n_sink` absolute positions,
  causal-clipped for early queries.
- Encoder mode (documented generalization — the paper only defines the causal form): all
  complete blocks are candidates **except the query's own block** (covered by its window);
  bidirectional window = the query's own block.

### 2.6 State & cache analysis

- **Per-layer state**: the full KV cache (kept — "methods that keep the whole KV cache and
  select keys per query", §2.2) **plus** the bucket cache `B_cache^(idx)`: per KV head, the
  concatenated unit-norm centroids `(B, nKv, m·(#complete blocks), hd)` and a padded int index
  table of each bucket's key positions `(C_total, n_max_b)` + bucket sizes. Per decode step
  (flat search): score `N_i ≈ T/n̄_b` centroids, then `O(k + n_w + n_sink)` attention (§F.1:
  "The row holds `N_i = |B_i|` scores, which grows with the position of the query, to about
  `T/n̄_b`").
- **Compatibility class**: class **(i)** classic KV for the attention data + a **bespoke
  block-level bucket cache**. Unlike `nsa_attn`/`msa_attn`/`sparda_attn` (whose block statistics
  are chunk-level means that later writes can change), LSA's buckets are **immutable once a
  block completes** ("Once a cluster block is clustered, its buckets never change"), so an
  incremental branch is **exact** — the user chose to implement it (KV-cache decision 1):
  new `BucketKVSlot` (KVSlot subclass) + incremental branch + `IMPLEMENTED_MIXERS` membership +
  `test_kv_cache_mixers.py` parity (§4 rows 1, 12, 21).
- Bucket centroids/tables are block-level state: **never quantized** (analog of D12 for
  recurrent states); the slot keeps them full-precision regardless of `kv-quant`.

### 2.7 Complexity

- **Training / prefill (flat)**: indexer `O(T²/n̄_b)` centroid scores (quadratic, as the paper
  states) + clustering `O((T/n_cluster)·(n_cluster/n_pool)²)` Gram work + attention
  `O(T·(k+n_w+n_sink)·d_h)`. Tree variant: `O(T log T)` indexer (future work).
- **Decode (flat)**: `O(T/n̄_b)` centroid scores + `O(k+n_w+n_sink)` attention per step.
- **`tab:all-mixers` row values** (this repo's column semantics, cf. nsa/msa rows):
  Training `O(N²/n̄_b + N·(k+n_w+n_sink)·d_h)`; Inference `O(N/n̄_b + k+n_w+n_sink)`.

### 2.8 Trainability

**Natively trainable from scratch on the NTP loss alone** — *not* training-free: no
`TRAINING_FREE_LAYERS` membership, no `dense_forward` surrogate, valid in training presets, and
it joins `full_tests/run_e2e.py::ATTENTIONS`. The selection (clustering + centroid scoring +
expansion) runs under `torch.no_grad()` inside `forward` — "We stop the gradient through the
selection" — so gradients flow only through the attended keys/values (and the queries that
scored them), exactly as in the paper's training kernels ("the clustering and the scoring of the
centroids need no backward pass").

### 2.9 Positional encoding

**None.** "The attention layers of N-LSA use no positional encoding" (§3.4). ⇒
`per_mixer_use_pe.lsa_attn` default **false**, dataclass `lsa_attn_use_pe: bool = False`
(among the `*_use_pe = False` block, `src/model/config.py` lines 596–637). The flag remains
overridable (per-mixer opt-in); with PE enabled the module clusters the L2-normalized
**post-PE** keys as-is — a documented deviation (the paper suggests clustering only the
RoPE-free part for D-LSA on QWEN); default behaviour is paper-exact.

### 2.10 Mode constraints

Decoder/causal is the paper's formulation (autoregressive NTP). Encoder mode is a documented
bidirectional generalization (§2.5). Both modes valid ⇒ no `config_loader` rule, no mode
restriction in `model/config.py`; `causal_lm` (decoder) and `mlm`/`sbert`/`text_classification`
(encoder) presets are all legal; audio-compatible (encoder). No grid/vision-only constraint.

### 2.11 BibTeX

```bibtex
@misc{kubicek_lsa_2026,
      title={Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention},
      author={Ales Kubicek and Andrea Caciolai and Liangzu Peng and Sukrit Sumant and Benjamin Hoover and Fraser Greenlee and Ruairidh M. Battleday and Radim Urban and Torsten Hoefler and James C. R. Whittington},
      year={2026},
      eprint={2610.42147},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.42147},
      publisher={arXiv}
}
```

Key follows the `<firstauthor>_<shortname>_<year>` convention (`yuan_nsa_2025`,
`lai_minimax_2026`, `fu_sparda_2026`); style mirrors the newest entries (lines 232–252 of
`docs/bibliography/attention_types.bib`).

### 2.12 Key empirical numbers for the annex "Key Characteristics"

- N-LSA 0.5B hybrid, 24 layers (6 attention + 18 Gated DeltaNet), 8 query heads over one
  128-dim KV head, no PE; WSD (100B FineWeb-Edu @4k → 12B LongDocs @128k decay).
- Decay-phase validation loss @128k: **N-LSA 1.085 = Full Attention 1.085**; PISA 1.100;
  NSA 1.117; SWA 1.154. At 4k pre-training all methods within 0.004 ("their keys cover most of
  the context").
- Training speedup: **up to 1.6× per attention layer** (fwd+bwd) at 128k vs full attention.
- D-LSA (for context): at 10M tokens on GLM-5.3-FLASH, 22.7× (72.4×) prefill and 4.6× (42.5×)
  decode vs its original sparse (full) attention; RULER 63.9 vs 59.9 (full) at 10M on
  QWEN-3.8-27B; bucket cache at 10M = 246 MiB vs 10 GiB KV per layer.
- Appendix A motivation: at 128k, top-k *oracle* attention keeps 98% of RULER accuracy down to
  `k = 512` (0.8% of keys); single-needle holds to `k = 16`.

---

## 3. Decisions (with evidence)

| Decision | Choice | Evidence |
|---|---|---|
| Variant | **N-LSA only** | §2.8: natively trainable, no teacher; D-LSA needs a frozen-model distillation pipeline absent from `src/training/`. User-confirmed. D-LSA + Tree recorded in §9 Open questions. |
| Registry name | **`lsa_attn`** | Paper short name "LSA"; snake_case + `_attn` suffix like peers (`nsa_attn`, `msa_attn`); no collision in `mixer_registry` (`src/model/hybrid_layer.py` lines 184–230) or anywhere in the repo (grep: zero `lsa_attn` matches today). |
| Class name | **`LSAAttention`** | `<Name>Attention` convention (`NSAAttention`, `MSAAttention`, `SparDAAttention`). |
| Family | **`src/model/attention/sparse/`**, census label **Sparse** | The paper's own Table 1 groups LSA under *Semantic Sparse Attention*; mechanism = per-query key selection (the sparse family's defining trait). No new taxonomy branch, no new `ft-diagram.js` STYLES class (`sparse` exists). User-confirmed. |
| Peer module | **`sparse/nsa_attn.py`** (primary) + `sparse/fasa_attn.py` (cached-selection template) + `sparse/msa_attn.py` (GQA/kv-head handling) | nsa: constructor `(config, pos_encoder=None)`, knobs via `getattr`, `attach_qk_norm`, `apply_pe_to_qk`, `module_softmax`, `record_qk_max_logits`, decoder masking, `forward` S1 shape. fasa: `make_cache_slot` hook (lines 150–156), cached forward with `positions_for`/`kv_decode_append`/`causal_mask_from_positions` (lines 225–262). msa: `num_kv_heads` validation + `H×(nKv·hd)` KV projections (lines 89–108). |
| Knobs | **7 knobs** (§2.5), sub-key `model.attention.lsa.*` | Table 2 + §4.4; swa-style declared defaults. |
| `use_pe` default | **false** (`lsa_attn_use_pe: bool = False`) | §3.4 quote, §2.9. |
| KV-cache | **Decision 1 — full incremental branch** (user-confirmed): `CacheSpec(supported=True, slot_type="kv", …)` + new `BucketKVSlot` + `make_cache_slot` hook + `IMPLEMENTED_MIXERS` membership + `test_kv_cache_mixers.py` parity (incl. a dedicated tiny-knob bucket-formation test) | Exactness argument: §3.1 "Once a cluster block is clustered, its buckets never change" — unlike nsa/msa/sparda whose chunk-level statistics are not row-local. Implementation via the `make_cache_slot` mixer hook (`manager.py` lines 204–216); `slot_type="kv"` + `extras=("bucket_centroids","bucket_index_table")` keeps the declarative registry truthful (primary storage = KV; bucket tables are block-level extras documented in `notes`). |
| Training-free? | **No** | Natively trainable (§2.8). Not in `TRAINING_FREE_LAYERS` (`hybrid_layer.py` line 122 stays `{"fasa_attn", "sparge_attn"}`); joins `ATTENTIONS`. |
| Example preset | **`configs/examples/es_arch_lsa_adamw.yaml`** (encoder MLM, `es_arch_msa_adamw.yaml` shape, small knobs so the example actually forms buckets at `max_length: 512`) | Peer naming convention (`es_arch_<mixer>_adamw.yaml`). |
| Full preset | **`configs/lsa.yaml`** (user-approved) | Paper reference config §2.3 (N-LSA column); modeled on `configs/kimi_k3.yaml`. |
| Census bumps | **46 → 47 enum values** (46 mixer classes + `mamba`); **Sparse 10 → 11**; families stay 10 | Verified actuals: enum `src/schema/_model/_dims.yaml` lines 332–378 (46 values); `docs/architecture.md` §7.4 line 351 "46 enum values in 10 families"; `README.md` line 133 "Sparse (10)"; `annex-summary-tables.tex` line 4 "46 sequence mixers across ten families" + line 45 "Sparse (10)"; `annex-15-kv-cache.tex` line 20 "forty-five dispatchable layer types"; `mixer-taxonomy.mmd` line 3 "(46 variants)"; `capabilities.py` docstring line 6 "(46 total: 45 mixer classes…)". |

---

## 4. File-by-file plan table (execution order)

| # | File | Planned change (concrete) |
|---|---|---|
| 1 | `src/model/cache/slots.py` | New **`class BucketKVSlot(KVSlot)`** after `RingSlot` (lines 568–598 region), keeping inherited `kind="kv"` (so `manager.total_tokens` line 285 and fasa-style `kind in ("kv","ring")` checks keep working). Adds: `bucket_centroids` (growing `(B, nKv, C_total, hd)`), `bucket_positions` (growing padded `(C_total, n_max_b)` int64 + `bucket_sizes` `(C_total,)`), `bucket_block_ids` `(C_total,)`; methods `append_block(centroids, positions_padded, sizes)` (torch.cat growth, idempotent per block id), `read_bucket_cache()` accessor, `truncate(num_tokens)` override → `super().truncate(num_tokens)` then drop every bucket whose stored positions intersect `[num_tokens, ∞)` (rollback exactness), `to(device)` override → also move bucket tensors. Bucket tables are plain full-precision tensors (never codec-quantized; D12 analog). Google-style docstrings citing arXiv:2610.42147. |
| 2 | `src/model/attention/sparse/lsa_attn.py` | **New module** (peer: `nsa_attn.py`; cached template: `fasa_attn.py`). Module docstring cites Kubicek et al. 2026, arXiv:2610.42147 (N-LSA, §3.4). `from __future__ import annotations`; peer-style imports (`from ...numerics import module_softmax`, `from ...norm import attach_qk_norm, record_qk_max_logits`, `from ..common import BitLinear, apply_pe_to_qk`). `class LSAAttention(nn.Module)`, `__init__(self, config, pos_encoder=None)`: read knobs (§2.5 defaults) via `getattr`; validate `hidden_size % num_heads == 0` and `num_heads % num_kv_heads == 0` (msa lines 89–94 pattern); projections `q_proj: H→H`, `k_proj/v_proj: H→(nKv·hd)`, `out_proj: H→H`, `bias=False`, `proj_cls = BitLinear if config.use_bitnet else nn.Linear` (nsa lines 122–127); `self.scale = head_dim ** -0.5`; `self.use_pe = bool(getattr(config, "lsa_attn_use_pe", False))`; `attach_qk_norm(self, config, head_dim)`; `self.mode = getattr(config, "mode", "encoder")`. Methods: `_pool_keys(k_hat)` (§2.5 edge cases), `_ward_cluster_block(pool_keys)` → Algorithm 1 (Gram-matrix mutual-cheapest-merge rounds under `torch.no_grad`, vectorized over batch, `n_max_b` cap, stop at `m = cluster_block // keys_per_bucket`, stable index tie-breaking), `_select(q_hat, centroids, tables)` → eq. (3) `max` over the query heads sharing each KV head, top-`p` (`p` auto-clamped to `≥ ceil(k/n_pool)`), best-first expansion skipping sink/window positions. `forward(self, x, logical_layer_idx=None, pos_encoder=None, cache=None, cache_position=None)` (S1 shape, fasa line 184): project q/k/v → `qk_norm` → `apply_pe_to_qk` (with `position_offset` on the cached path, fasa line 231) → GQA expand k/v to query heads → under `no_grad`: L2-normalize keys per KV head, pool, cluster every complete block before each query's own block (decoder) / all blocks except own (encoder) → score centroids, top-p, expand → `J_i` = sinks ∪ long-range ∪ own-block window (causal-clipped in decoder) → per-head `z = q·k/√hd` over `J_i`, `record_qk_max_logits`, `module_softmax` over the shared selection, gather V → `out_proj`, `dropout`. **Cached branch** (mirrors fasa lines 225–262): `slot = cache if getattr(cache, "kind", "") in ("kv", "ring") else None`; with `hooks.positions_for` / `hooks.kv_decode_append` append k/v; cluster the **newly completed** blocks (from `cache_position`/`seq_len`) once and `slot.append_block(...)`; read centroids/tables from the slot; **D5 defense**: if the slot lacks bucket tables (e.g. a plain `KVSlot` from `_default_slot`), recompute the bucket cache on the fly from the slot's cached keys (exact, slower) — never raise. `make_cache_slot(self, meta, cache)` → `BucketKVSlot(meta, codec=kwargs["codec"], policy=None, num_sinks=kwargs["num_sinks"])` with `kwargs = cache.codec_slot_kwargs()` (fasa lines 150–156 pattern). No `dense_forward` (not training-free). CPU-safe pure PyTorch (no kernels). |
| 3 | `src/model/attention/sparse/__init__.py` | Import + `__all__`: add `from .lsa_attn import LSAAttention` (imports lines 1–10, after `longformer`), `"LSAAttention"` in `__all__` (lines 12–23, after `"LongformerAttention"`). |
| 4 | `src/model/attention/__init__.py` | Add `LSAAttention` to the `.sparse` import block (lines 5–16, after `LongformerAttention`) and to `__all__` (lines 54–102, sparse cluster after `"LongformerAttention"`). |
| 5 | `src/model/hybrid_layer.py` | Add `"lsa_attn": LSAAttention,` to `mixer_registry` after `"memory_attn": MemoryAttention,` (line 229; dict closes line 230). `TRAINING_FREE_LAYERS` (line 122) **unchanged**; generic dispatch (lines 234–235) and `use_fixed_softmax` attach (line 245) cover it automatically. |
| 6 | `src/model/config.py` | (a) Knob fields after the SparDA block (lines 861–864): `# ---- LSA / Latent Sparse Attention (arXiv:2610.42147) ----` + `lsa_num_long_range: int = 2048`, `lsa_num_buckets: int = 512`, `lsa_num_sinks: int = 4`, `lsa_pool_size: int = 4`, `lsa_cluster_block: int = 512`, `lsa_keys_per_bucket: int = 8`, `lsa_max_keys_per_bucket: int = 16`. (b) `lsa_attn_use_pe: bool = False` in the `*_use_pe` block after `sparda_attn_use_pe` (line 612; block spans 596–637). (c) Validation after the SWA block (lines 1106–1120), swa style: `lsa_num_sinks >= 1`, `lsa_pool_size >= 1`, `lsa_num_long_range >= 1`, `lsa_num_buckets >= 1`, `lsa_cluster_block >= 2`, `lsa_keys_per_bucket >= 1` + `lsa_cluster_block % lsa_keys_per_bucket == 0`, `lsa_max_keys_per_bucket >= lsa_keys_per_bucket`, `lsa_num_sinks < lsa_cluster_block`. (d) Class docstring mixer lists (lines 113/125 mention `msa_attn`, `sparda_attn`) — append `lsa_attn`. |
| 7 | `src/utils/config_flatten.py` | Add to `_ATTENTION_MIXER_RENAMES` (dict line 76, after `"sparda"` lines 126–130): `"lsa": {"num_long_range": "lsa_num_long_range", "num_buckets": "lsa_num_buckets", "num_sinks": "lsa_num_sinks", "pool_size": "lsa_pool_size", "cluster_block": "lsa_cluster_block", "keys_per_bucket": "lsa_keys_per_bucket", "max_keys_per_bucket": "lsa_max_keys_per_bucket"}`. |
| 8 | `src/schema/_model/_attention_lsa.yaml` | **New knob file** (template `_attention_swa.yaml`): `type: object` / `additionalProperties: false` / `properties: lsa:` with `title`/`title_es` ("Latent Sparse Attention" / "Atención Dispersa Latente"), bilingual block description citing `(paper: arxiv:2610.42147)` and N-LSA, and the seven leaves of §2.5 — each `type: integer`, `minimum`, `title`/`title_es`, EN+ES `description` (Table 2 values + constraints, `**References:** (paper: arxiv:2610.42147)`), `default:` (swa style) + `examples:`. E.g. `num_long_range`: "Long-range key budget `k` selected per query … Paper N-LSA default: 2048 (D-LSA bucket variants use 4096)." / `cluster_block`: "Cluster block size `n_cluster`; also the window `n_w` (the paper sets the window to one cluster block). Default: 512." / `keys_per_bucket`: "Average keys per bucket `n̄_b`; each complete block yields `m = cluster_block / keys_per_bucket` buckets (default 64). Must divide `cluster_block`." / `max_keys_per_bucket`: "Merge cap `n_max_b` (Ward merges that would exceed it are skipped). Default: 16." / `num_buckets`: "Buckets scored per query `p`; auto-clamped so `p · pool_size ≥ k`. Default: 512." |
| 9 | `src/schema/_model.yaml` | Under `model.attention.properties` add after the `sparda` `$ref` (lines 90–91): `lsa:` / `$ref: "_model/_attention_lsa.yaml#/properties/lsa"`. The bilingual attention description (lines 49–67) lists knob-*less* mixers only — LSA has knobs, so no list change. |
| 10 | `src/schema/_model/_dims.yaml` | (a) Append `- lsa_attn` to `layer_pattern.items.enum` after `- memory_attn` (line 378). (b) Append one row to the **EN** table after the `memory_attn` row (line 262): ``| `lsa_attn` | Latent Sparse Attention — semantic Ward clustering of keys + top-k long-range selection; natively trainable, no extra parameters, no PE (paper: arxiv:2610.42147) | ✅ | ✅ |``. (c) Append the faithful **ES** row after line 319: ``| `lsa_attn` | Atención Dispersa Latente — clustering Ward semántica de claves + selección top-k de largo alcance; entrenable de forma nativa, sin parámetros extra, sin PE (paper: arxiv:2610.42147) | ✅ | ✅ |``. (Gate: `test_layer_pattern_enum_matches_documentation` scans both tables for backticked enum values — `tests/test_schema_consistency.py` lines 66–73.) |
| 11 | `src/schema/_model/_positional_encoding.yaml` | (a) Add to `use_pe.properties` after `sparge_attn` (line 419): `lsa_attn: { type: boolean, title: "lsa_attn", description: "Model-wide PE for `lsa_attn` layers (N-LSA trains without positional encoding)", default: false, description_es: "PE a nivel de modelo para capas `lsa_attn` (N-LSA entrena sin codificación posicional)" }`. (b) Update the prose default lists enumerating PE-less mixers: `positional_encoding` description (~line 24) and the `use_pe` block description (lines 381–391 EN + ES) — add `lsa_attn` to the "False for …" list (e.g. "…, kda, mtla, falcon*, lsa_attn"). |
| 12 | `src/model/cache/capabilities.py` | (a) `IMPLEMENTED_MIXERS` (frozenset lines 68–104): add `"lsa_attn",` in the sparse-family group after `"fasa_attn",` (line 92) with comment `# F2 (bucket cache over immutable complete cluster blocks)`. (b) `CACHE_SPECS` sparse section (lines 227–301), after the `fasa_attn` entry (lines 296–301): `"lsa_attn": CacheSpec(supported=True, slot_type="kv", extras=("bucket_centroids", "bucket_index_table"), notes="LSA/N-LSA (arXiv:2610.42147); keeps the full KV cache and selects keys per query: keys are L2-normalized, pooled (n_pool) and Ward-clustered per complete n_cluster block into m = n_cluster/n̄_b immutable buckets — centroids + index table live in a BucketKVSlot built by the mixer's make_cache_slot hook. Buckets never change once a block completes, so incremental decode is exact (unlike nsa/msa/sparda); flat search scores ~T/n̄_b centroids per decode step. Bucket tables are never quantized.")`. (c) Module docstring line 6: "(46 total: 45 mixer classes + the `mamba` linear placeholder)" → "(47 total: 46 mixer classes + …)". |
| 13 | `src/model/norm/qk.py` | Add `"lsa_attn",` to `QK_NORM_MIXERS` (frozenset line 46) after `"sparda_attn",` (line 63). `QK_CLIP_MIXERS` (line 78) derives automatically — lsa forms per-head logits over `J_i`, so clip support is safe (gma-style exclusion not needed). Module calls `attach_qk_norm` + `record_qk_max_logits` like nsa/msa/sparda. |
| 14 | `configs/examples/es_arch_lsa_adamw.yaml` | **New example** (mirror `es_arch_msa_adamw.yaml` lines 1–80). Full draft: `model_class: frankenstein_text_encoder`; `model.dims`: `vocab_size: 50000`, `hidden_size: 512`, `num_layers: 10`, `num_loops: 1`, `num_heads: 8`, `num_kv_heads: 1` (paper MQA), `layer_pattern: [lsa_attn]`; `norm.type: layer_norm`; `embedding.conv.enabled: false`; `attention.lsa:` `cluster_block: 128`, `num_long_range: 64`, `num_buckets: 16`, `num_sinks: 4`, `pool_size: 4`, `keys_per_bucket: 8`, `max_keys_per_bucket: 16` (small enough that 512-token MLM windows form complete blocks); `attention.retnet.heads: 8` (peer shape); `moe.enabled: false`; `bitnet.enabled: false`; `ffn: {hidden_size: 2048, activation: gelu}`; `training`: copy the msa example block (task `mlm`, seed 42, `num_epochs: 4`, `batch_size: 4`, `max_length: 512`, `mlm_probability: 0.15`, dataset keys, prefixed `adamw-*` parameters, `scheduler_total_steps: 9000`) with `csv_log_path: training_metrics_es_arch_lsa_adamw.csv`. Must pass `load_training_config` (gate: `tests/test_yaml_examples.py`, auto-globbed). |
| 15 | `configs/lsa.yaml` | **New full preset** (user-approved; template `configs/kimi_k3.yaml`): `model_class: frankenstein_text_decoder`; `model.dims`: `vocab_size: 128256`, `hidden_size: 1024` (8 × 128-dim heads), `num_layers: 24`, `num_loops: 1`, `num_heads: 8`, `num_kv_heads: 1`, `dropout: 0.0`, `layer_pattern: [lsa_attn, gated_deltanet_attn, gated_deltanet_attn, gated_deltanet_attn] × 6` (attention every 4th layer, paper §4.4), `mode: decoder`; `norm.type: rms_norm`; `embedding.factorized.enabled: false` + `embedding.conv.enabled: false`; `attention.lsa:` paper reference values explicitly (`num_long_range: 2048`, `num_buckets: 512`, `num_sinks: 4`, `pool_size: 4`, `cluster_block: 512`, `keys_per_bucket: 8`, `max_keys_per_bucket: 16`); `moe.enabled: false`; `bitnet.enabled: false`; `ffn: {hidden_size: 4096, activation: silu}`; `positional_encoding: none` (N-LSA uses no PE); `training`: mirror `configs/examples/causal_lm_decoder.yaml` (task `causal_lm`, seed 42, `batch_size: 4`, `max_length: 4096` with a comment that the paper's decay phase runs at 131072, prefixed `adamw-*` parameters, `scheduler_total_steps: 10000`, `scheduler_warmup_ratio: 0.1`, `scheduler_type: cosine`). Header comment: "Native-LSA reference configuration (arXiv:2610.42147 §4.4): 0.5B-class 24-layer hybrid, every 4th layer N-LSA attention (8 query heads over one 128-dim KV head, no PE), remaining layers Gated DeltaNet." Joins the smoke-tested set (`test_all_presets_validate`). |
| 16 | `full_tests/run_e2e.py` | Append to `ATTENTIONS` (lines 224–241) after `"memory_attn",` (line 240) with a peer-style comment: `# Latent Sparse Attention / N-LSA (arXiv:2610.42147): semantic key` / `# clustering + top-k selection; trains natively in encoder and decoder modes.` + `"lsa_attn",`. Auto-generates `attn_single_lsa_attn` and every pair via `make_attention_pair_configs` (line 737) / `make_single_attention_configs` (line 752) — no builder edits. Trainable ⇒ belongs in the list (line-223 comment reserves it for trainable mixers; fasa/sparge are the exclusion precedent). |
| 17 | `tests/test_schema_attention_layers.py` | (a) Add `"lsa_attn"` to the `expected` set (lines 18–65). (b) Add to `expected_per_mixer` (lines 89–123): `"lsa": {"num_long_range", "num_buckets", "num_sinks", "pool_size", "cluster_block", "keys_per_bucket", "max_keys_per_bucket"}`. (c) `test_cache_specs_match_layer_pattern_enum` (lines 139–151): `self.assertEqual(len(enum_values), 46)` → **47**. |
| 18 | `tests/test_schema_consistency.py` | Add seven entries to `_DEFAULT_PAIRS` (dict lines 153–226, after the swa pair lines 199–200): `"properties.model.properties.attention.properties.lsa.properties.num_long_range": "lsa_num_long_range"` (likewise `num_buckets`→`lsa_num_buckets`, `num_sinks`→`lsa_num_sinks`, `pool_size`→`lsa_pool_size`, `cluster_block`→`lsa_cluster_block`, `keys_per_bucket`→`lsa_keys_per_bucket`, `max_keys_per_bucket`→`lsa_max_keys_per_bucket`). (The schema declares `default:` per row 8, so `test_schema_defaults_match_dataclass` covers them; the msa/sparda no-schema-defaults gap is intentionally not copied.) |
| 19 | `tests/test_sparse_attention.py` | Add `LSAAttentionTests` after `SparDAAttentionTests` (lines 293–310), using the `_cfg(**overrides)` fixture (lines 24–47) — knobs are plain dataclass fields, overrides work directly. Methods: `test_output_shape` (`LSAAttention(_cfg(num_kv_heads=1, lsa_cluster_block=8, lsa_pool_size=2, lsa_keys_per_bucket=2, lsa_num_buckets=2, lsa_num_long_range=4, lsa_num_sinks=1))`; assert `(BSZ, SEQ, DIM)`); `test_decoder_mode` (same + `mode="decoder"`); `test_gradient_flows` (`x.requires_grad_(True)` → backward; selection is `no_grad` but q/k/v/out gradients must flow); `test_invalid_kv_heads_raises` (`_cfg(num_kv_heads=4)` — 4 ∤ 6); `test_short_sequence_uses_window_only` (defaults `cluster_block=512` ≫ `SEQ=8` ⇒ no complete block ⇒ output finite, shape OK — paper-exact degenerate case); `test_buckets_form_and_select` (`cluster_block=4` with `SEQ=8` ⇒ one complete block ⇒ assert e.g. `attn.last_num_buckets > 0` and no NaN/Inf); `test_bitnet_on` (`use_bitnet=True` forward). Import `LSAAttention` in the module import block (lines 10–21). |
| 20 | `tests/test_bitnet_coverage.py` | Add `"lsa_attn",` to the `_make` `layer_pattern` list (lines 37–65, after `"memory_attn",` line 57). Existing gates (`test_gates_are_bitlinear_when_use_bitnet_true` etc., lines 118–183) then cover it. |
| 21 | `tests/test_kv_cache_mixers.py` | (a) **No edit needed** for the default parametrization: `MIXERS = sorted(m for m in IMPLEMENTED_MIXERS if m != "mamba")` (line 27) picks `lsa_attn` up automatically; `_parity` (lines 54–87) then asserts prefill+decode parity across `rope/nope/alibi` with default knobs (10-token sequence ⇒ window+sinks exactness of the append path). (b) **Add** `class LSABucketParityTests` after `MixerParityTests` (lines 91–127) with a local `_build_lsa(pe)` helper (mirror `_build` lines 31–51 plus `num_kv_heads=1, lsa_cluster_block=4, lsa_pool_size=2, lsa_keys_per_bucket=2, lsa_num_buckets=2, lsa_num_long_range=4, lsa_num_sinks=1`) and tests: `test_prefill_decode_parity_with_buckets` (6-token prompt + 4 decode steps ⇒ blocks 0–1 complete ⇒ real bucket formation + centroid scoring + expansion parity; greedy tokens equal, tolerance fallback `rtol=1e-4, atol=1e-5` as in `_parity`), `test_bucket_cache_populated` (after a cached prefill the slot exposes `bucket_centroids` with the expected `m × n_blocks` count), `test_truncate_drops_partial_buckets` (slot.truncate rollback drops buckets intersecting the cut). |
| 22 | `docs/paper/figures/mixers/lsa.mmd` | **New diagram**, house style (cf. `msa.mmd`, 16 lines; width 1400 px per `build-figures.sh` line 57). Full draft: init header `%%{init: {"theme":"neutral","flowchart":{"curve":"basis","nodeSpacing":18,"rankSpacing":26,"useMaxWidth":false}}}%%`; `flowchart TD`; nodes `x["x<br/>[B,n,H]"]`, `qkv["Q, K, V<br/>[B,n,H]"]`, `pool["pool keys<br/>k_pool (4:1)"]`, `ward["Ward buckets<br/>c_b (Alg. 1)"]`, `sel["top-p buckets →<br/>k long-range keys"]`, `j["J = sinks ∪ k<br/>∪ window"]`, `s["sparse scores<br/>= QKᵀ over J"]`, `sm["softmax"]`, `o["O = Wo·(softmax·V)<br/>[B,n,H]"]`; edges `x -- "Wqkv" --> qkv`, `qkv -- "K̂, L2" --> pool`, `pool -- "cluster" --> ward`, `ward -- "max_h q̂·c_b" --> sel`, `qkv -- "K,V" --> j`, `sel -- "expand" --> j`, `qkv -- "QKᵀ" --> s`, `j -- "QKᵀ" --> s`, `s -- "softmax" --> sm`, `sm -- "Wo" --> o`. Render with `( cd docs/paper/figures && ./build-figures.sh both )`. |
| 23 | `docs/paper/figures/mixer-taxonomy.mmd` | (a) Root label line 3: `"Sequence Mixer Registry (46 variants)"` → `(47 variants)`. (b) Sparse chain line 9: append `--> s10["lsa_attn"]` and set the family label `"Sparse (9)"` → **`"Sparse (11)"`** — and add the **pre-existing missing** `swa_attn` node at the same time (the chain lists 9 of the 10 sparse mixers; see §9 flag A1). If the user prefers not to touch pre-existing drift, add only `s10["lsa_attn"]` and label `(10)` — but then the diagram disagrees with every census surface (11). Recommended: add both nodes, label `(11)`. |
| 24 | `docs/paper/figures/sparse-map.mmd` | Extend the `select` node (line 8): `select["Selective pruning<br/>SparseK / NSA / FASA<br/>SpargeAttn / LSA"]` (LSA is the learned *semantic* selector of the family). |
| 25 | `docs/bibliography/attention_types.bib` | Append the `kubicek_lsa_2026` entry (§2.11) after the last entry (`kang_memory_attention_2026`, line 484), in the newest entries' indented style with `eprint`/`archivePrefix`/`primaryClass` (cf. `lai_minimax_2026` lines 232–241). |
| 26 | `docs/paper/appendices/annex-4-sparse-attention.tex` | Insert `\subsection{LSA (Latent Sparse Attention): Semantic Key Clustering with Top-k Selection}` **between SparDA (ends ~line 687) and `\subsection{Design Space and Selection Criteria}` (line 689)** — becomes 7.12; Design Space shifts to 7.13 (auto-numbering). House structure (per SparDA 631–687 / MSA 570–629): figure block `\begin{figure}[H] \centering \fitfigure{\includegraphics{figures/mixers/lsa.pdf}} \caption{Dataflow of N-LSA: pooled keys are Ward-clustered into buckets; each query scores the centroids, expands the best buckets into a top-k long-range set, and attends to sinks + selection + window.} \label{fig:mixer-lsa} \end{figure}`; `\subsubsection{Mathematical Formulation}` (this annex uses "Mathematical Formulation", **not** "Mathematical Foundation") with §2.1's equations (L2 norms, pooling, Ward cost `Δ(b,b') = a_b + a_{b'} − a_{b∪b'}` with `a_b = √G_bb`, centroid `c_b = k_sum_b/‖k_sum_b‖`, eq. (3) `s_{i,b} = max_h q̂_{i,h}·c_b`, top-p, best-first expansion, `J_i ≤ n_sink + k + n_w`, `z_{i,j} = q_i·k_j/√d_h` with per-head renormalisation over the shared selection, no PE, gradient stopped through the selection, zero extra parameters; `\citep{kubicek_lsa_2026}` on first mention); `\subsubsection{Algorithmic Pseudocode}` with an `algorithm`/`algorithmic` float titled "N-LSA: Ward clustering of one cluster block and flat semantic search" (Algorithm 1 verbatim + selection/expansion lines of §2.2); `\subsubsection{Key Characteristics}` as `itemize[leftmargin=*]` with `\textbf{Complexity}` (§2.7), `\textbf{Trainable}` (natively from scratch on the NTP loss; selection under no gradient; no teacher — unlike D-LSA), `\textbf{Strength}` (§2.12: 1.085 = full-attention val loss at 128k vs NSA 1.117 / PISA 1.100 / SWA 1.154; 1.6× per-layer training speedup; D-LSA 22.7×/4.6× at 10M; exact incremental decode — buckets immutable per block), `\textbf{Limitation}` (flat search keeps the indexer quadratic in T with a 1/n̄_b constant; tree search future work; single shared selection across heads assumes the MQA-style single KV head; no positional encoding by design). Then update the closing `\subsection{Design Space and Selection Criteria}`: line 691 "The ten sparse attention methods" → "The eleven sparse attention methods"; add an LSA item to the design-space `itemize` (lines 693–699) — semantic (content-clustered) selection rather than positional blocks, immutable per-block statistics ⇒ exact caching; extend the "Selection guidance" enumerate (lines 701–707) with an LSA clause (prefer it when keys cluster semantically and exact cached decoding matters). |
| 27 | `docs/paper/sections/04-architecture-taxonomy.tex` | (a) Line 7: "The system implements forty-six sequence mixers organized into ten functional categories" → **forty-seven**. (b) Line 11 sparse item: "(10): Sparse Transformer, Longformer, SWA with attention sinks, BigBird, SparseK, NSA, SpargeAttn, FASA, MSA, and SparDA" → "(11): …, MSA, SparDA, and **LSA**" + append ", or semantic key clustering" to the closing sentence. (c) Line 24 figure caption: "forty-six sequence mixers" → "forty-seven". (d) Sparse paragraph (line 71, under `\subsection{Sparse Attention Extensions}` line 70): "The registry includes ten sparse blocks \citep{…}" → "eleven sparse blocks" + append `kubicek_lsa_2026` to the `\citep` list + one sentence: "LSA is the family's semantic selector: keys are pooled and Ward-clustered per 512-token block, and each query scores the cached bucket centroids instead of every key, selecting top-p buckets that expand into its k long-range keys on top of sinks and a one-block window." |
| 28 | `docs/paper/sections/01-introduction.tex` (line 14), `10-discussion.tex` (line 16), `11-conclusion.tex` (line 8) | "forty-six sequence mixers" → "forty-seven" in each. |
| 29 | `docs/paper/appendices/annex-summary-tables.tex` | (a) Census line 4: "46 sequence mixers across ten families" → **47**. (b) Sparse family header line 45: `\multicolumn{6}{c}{\textbf{Sparse (10)}} \\` → **(11)**. (c) New row after the `sparda\_attn` row (line 56; sparse block lines 47–56): `lsa\_attn & Sparse & $\mathcal{O}(N^2/\bar{n}_b + N\,(k+n_w+n_{sink})\,d_h)$ & $\mathcal{O}(N/\bar{n}_b + k+n_w+n_{sink})$ & Semantic Ward clustering of keys; top-k long-range selection; natively trainable, no extra parameters & \citep{kubicek_lsa_2026} \\`. (Columns per header lines 13/17: Name & Family & Training & Inference & Key Characteristic & Reference.) |
| 30 | `docs/paper/appendices/annex-schema.tex` | (a) Row in `tab:layer-pattern-enum` after line 250: `lsa\_attn & Latent Sparse Attention --- semantic key clustering with top-k selection & \checkmark & \checkmark \\`. (b) `model.attention.lsa.*` sub-block after the sparda block (lines 140–144), mirroring the msa block (133–138): seven rows (`model.attention.lsa.num_long_range & 2048 & …`, bilingual descriptions of §2.5 with defaults). (c) Intro line 209 says "43 mixer codes" — **pre-existing stale count** (46 actual): set it to **47** while touching the file (flag A3 in §9). |
| 31 | `docs/paper/appendices/annex-15-kv-cache.tex` | Line 20 "forty-five dispatchable layer types" → **forty-six** (census surface surfaced by this pass; grep at implementation time for sibling sentences). |
| 32 | `docs/paper-es/appendices/annex-4-sparse-attention.tex` | Faithful Spanish translation of row 26: `\subsection{LSA (Atención Dispersa Latente): Clustering Semántico de Claves con Selección Top-k}` inserted before line 689 (`\subsection{Espacio de Diseño y Criterios de Selección}`); figure `\fitfigure{\includegraphics{../paper/figures/mixers/lsa.pdf}}` (ES papers reuse EN figures via `../paper/figures/`), `\label{fig:mixer-lsa}`; `\subsubsection{Formulación Matemática}` / `\subsubsection{Pseudocódigo Algorítmico}` / `\subsubsection{Características Clave}` with `\textbf{Complejidad}/\textbf{Entrenable}/\textbf{Fortaleza}/\textbf{Limitación}` bullets (structure per SparDA ES 631–687); equations **must match** the EN annex verbatim (D11: the arXiv paper is the source of truth; paper-es translates faithfully). Update the closing: line 691 "Los diez métodos de atención dispersa…" → "Los once métodos…", itemize 693–699, "Orientación de selección" 701–707. |
| 33 | `docs/paper-es/sections/04-taxonomia-arquitectura.tex` | Spanish counterparts of row 27: line 7 "cuarenta y seis mezcladores" → "cuarenta y siete"; line 11 sparse item "(10): … MSA y SparDA" → "(11): … MSA, SparDA y **LSA**"; line 24 caption; sparse paragraph line 66 "El registro incluye diez bloques dispersos \citep{…}" → "once bloques dispersos" + `kubicek_lsa_2026` + the translated LSA sentence. |
| 34 | `docs/paper-es/sections/01-introduccion.tex` (line 14), `10-discusion.tex` (line 16), `11-conclusion.tex` (line 8) | "cuarenta y seis mezcladores de secuencia" → "cuarenta y siete" each. |
| 35 | `docs/paper-es/appendices/annex-summary-tables.tex` | (a) Line 4: "46 mezcladores de secuencia en diez familias" → **47**. (b) Line 45: `\textbf{Disperso (10)}` → **(11)**. (c) Row after line 56: `lsa\_attn & Disperso & $\mathcal{O}(N^2/\bar{n}_b + N\,(k+n_w+n_{sink})\,d_h)$ & $\mathcal{O}(N/\bar{n}_b + k+n_w+n_{sink})$ & Clustering Ward semántico de claves; selección top-k de largo alcance; entrenable de forma nativa, sin parámetros extra & \citep{kubicek_lsa_2026} \\`. |
| 36 | `docs/paper-es/appendices/annex-schema.tex` | (a) Row after line 250: `lsa\_attn & Latent Sparse Attention --- clustering semántico de claves con selección top-k & \checkmark & \checkmark \\`. (b) `model.attention.lsa.*` block after line 144 (Spanish descriptions, same defaults). (c) Line 209 "43 códigos de mezclador" → **47** (flag A3). |
| 37 | `docs/paper-es/appendices/annex-15-*.tex` | ES counterpart of row 31: "cuarenta y cinco tipos de capa despachables" → "cuarenta y seis" (grep `cuarenta y cinco` at implementation time). |
| 38 | `README.md` | Mixer Categories (lines 126–143): sparse row line 133 — `**Sparse** (10)` → `**Sparse** (11)` and append `, `lsa_attn`` to the code list; extend the row description with "semantic key clustering". The ⚠️ note (line 141) is unchanged (LSA is not eval-only). (Stale "(43 variants / 8 families)" at line 180 is pre-existing — flag A4, not fixed without approval.) |
| 39 | `docs/architecture.md` | (a) §7.4 line 351: "46 enum values in 10 families" → "47 enum values in 10 families". (b) §7.4 table: add row after `memory_attn` (line 401): ``| `lsa_attn` | Sparse (semantic) | `lsa.*` | yes |``. (c) §7.4 closing paragraphs (swa/memory precedent, lines 409–421): add an LSA paragraph — mechanism summary, the seven knobs with defaults, flat-search complexity, bucket immutability ⇒ exact caching, `use_pe` default false, D-LSA/tree out of scope. (d) §7.13: add `lsa_attn` to the "Implemented cached branches" list (lines 631–638) and add a sentence + slot-table note (lines 615–622) that `lsa_attn` uses a `BucketKVSlot` (KV + block-level bucket centroids/index table, never quantized); **do not** add it to the "Documented fallbacks" list (lines 639–645). (e) §14 Known limitations: no lsa entry (it is cacheable); leave the fallback list untouched. |
| 40 | `configs/README.md` | (a) Line 264: "`num_kv_heads`: key-value heads for GQA (also used by `msa_attn`/`sparda_attn`)" → add `/lsa_attn` (paper default 1). (b) Line 269 sparse list: append `, `lsa_attn` (Latent Sparse Attention, arXiv:2610.42147)`. (c) QK-stabilization mixer list (lines 301–309): add `lsa_attn`. (d) Sub-object list (lines 337–340): add `lsa`. (e) New section after the MSA/SparDA section (lines 408–416): `### Latent Sparse Attention (model.attention.lsa)` with the seven knobs + defaults + the `p·pool_size ≥ k` auto-clamp note. |
| 41 | `docs/source/api/model/attention/sparse/lsa_attn.rst` + `docs/source/api/model/attention/sparse/index.rst` | New `lsa_attn.rst` (7 lines, peer template): title `LSA Attention` / `===============` + `.. automodule:: src.model.attention.sparse.lsa_attn` / `   :members:` / `   :undoc-members:` / `   :show-inheritance:`. Append `lsa_attn` to `index.rst` toctree (lines 4–15). Build gate: `bash docs/build_docs.sh` runs `sphinx-build -W --keep-going` — a wrong module path fails the build. |
| 42 | **Mirror (site-only files — edited in `erickfmm.github.io/frankenstein-transformer/`, never synced from the repo)** | (a) `ft-diagram.js`: add `lsa_attn: { c: 'sparse', l: 'LSA Attn' },` to `CATS` after `sparda_attn` (line 39); add `SUBSTRUCT["lsa_attn"]` after `sparda_attn` (lines 325–332): nodes `x [B,n,H]` → `Q,K,V [B,n,H]` (edge `Wqkv`) → `Ward buckets c_b` (edge `cluster`) → `top-p buckets → k keys` (edge `select`, from QKV + buckets) → `sparse scores = QKᵀ` (edge `QKᵀ`) → `softmax` → `O = Wo·softmax·V` (edge `Wo`), all `c: 'sparse'` (existing STYLES key — no new class). (b) `ft-param-estimator.js`: add after the `sparda_attn` case (lines 192–195): `case 'lsa_attn': {` / `  return 2 * H * H + 2 * (H * kv); // N-LSA adds no parameters: q,o full-width + k,v to KV width` / `}`. (c) `index.html`: **no edit** — the mixer dropdown is schema-driven (`getMixerOptions()` reads the mirrored `schema.yaml`, lines 2624–2631). |

Rows 1–21 are the code+schema+test changes; 22–41 the docs/figures; 42 the site-only edits made
during the website sync (§6). `full_tests/param_truth.json` is **regenerated** (intentional diff)
by `param_count_check.py` — not hand-edited.

**Hard rules restated for the implementer**: the schema is the contract
(`additionalProperties: false` — every key above exists in `src/schema/_model/**` after rows 8–11);
`hidden_size % num_heads == 0`, `num_kv_heads` divides `num_heads`, `len(layer_pattern) ==
num_layers`; inference never fails because of the cache (D5 — the D5 defense in row 2 and the
resolve-slot fallback in `manager.py` lines 204–216 guarantee it); LSA is **not** training-free
(rule 5 n/a); `causal_lm` requires the decoder class (rule 6); the paper is the theory source of
truth (D11) — annex formulations must match arXiv:2610.42147 and `paper-es` must translate them
faithfully; every preset/example is smoke-tested (rule 8/9/D8).

---

## 5. Ordered execution checklist (commands run at implementation time, not now)

Preconditions verified today (2026-11-06): `pdflatex` ✓ `/usr/bin/pdflatex`, `bibtex` ✓
`/usr/bin/bibtex`, `chromium` ✓ `/usr/bin/chromium` (the `build-figures.sh` fallback chain,
lines 32–44, also accepts `google-chrome`/`chromium-browser`), `node` v25.9.0 ✓; mirror cloned at
`<repo>/erickfmm.github.io` (clean); main repo clean except the pre-existing untracked
`.opencode/plan/add-mixer-2609.28399.md`.

**Phase A — implementation (rows 1–16)**

- [ ] Row 1: `BucketKVSlot` in `src/model/cache/slots.py`
- [ ] Row 2: `src/model/attention/sparse/lsa_attn.py` (`LSAAttention`, cached branch, `make_cache_slot`)
- [ ] Rows 3–5: exports + `mixer_registry`
- [ ] Rows 6–7: `config.py` fields/validation + `_ATTENTION_MIXER_RENAMES`
- [ ] Rows 8–11: `_attention_lsa.yaml`, `_model.yaml` `$ref`, `_dims.yaml` enum + both tables,
      `_positional_encoding.yaml` `use_pe` entry + prose lists
- [ ] Rows 12–13: `capabilities.py` (spec + IMPLEMENTED_MIXERS + docstring count), `qk.py`
- [ ] Rows 14–15: `configs/examples/es_arch_lsa_adamw.yaml`, `configs/lsa.yaml`
- [ ] Row 16: `full_tests/run_e2e.py` `ATTENTIONS`

**Phase B — tests (rows 17–21)**

- [ ] `tests/test_schema_attention_layers.py` (expected set, per-mixer leaves, 46→47)
- [ ] `tests/test_schema_consistency.py` `_DEFAULT_PAIRS` (7 entries)
- [ ] `tests/test_sparse_attention.py` `LSAAttentionTests`
- [ ] `tests/test_bitnet_coverage.py` mixer list
- [ ] `tests/test_kv_cache_mixers.py` `LSABucketParityTests` (+ auto-joined default parity)

**Phase C — targeted gates, then full suite** (CPU-first; never run them in plan mode)

```bash
# targeted gates
uv run --extra cpu --extra train python -m pytest \
  tests/test_schema_attention_layers.py tests/test_schema_consistency.py \
  tests/test_yaml_examples.py tests/test_attention_refactor.py \
  tests/test_sparse_attention.py tests/test_bitnet_coverage.py \
  tests/test_kv_cache_mixers.py \
  -v --tb=short -p no:warnings

# full suite
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings
```

Expected: all green; in particular `test_all_presets_validate` (the two new YAMLs validate),
`test_layer_pattern_enum_matches_documentation` (both `_dims.yaml` tables contain `lsa_attn`),
`test_cache_specs_match_layer_pattern_enum` (47 == len(CACHE_SPECS)),
`test_every_implemented_mixer_parity` (lsa_attn × rope/nope/alibi PASS),
`LSABucketParityTests` PASS (finite logits, token-exact or within `rtol=1e-4, atol=1e-5`).

**Phase D — e2e smoke incl. targeted real-data training**

```bash
# enumerate the new mixer's combos (build-only, no training)
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --list-combos | grep -F lsa

# e2e TARGETED SMOKE — ONLY the new mixer: actual training on REAL data.
# --realonall is the DEFAULT: every selected combo trains on the real Borges
# corpus (not the synthetic toy one). First invocation downloads the
# tokenizer/corpus from the HF Hub. '--only lsa' substring-matches the
# auto-generated combo ids: attn_single_lsa_attn plus every pair
# attn_<a>__lsa_attn / attn_lsa_attn__<b>.
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only lsa

# minimal real-data run if the pair sweep is too heavy: the single solo combo
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only attn_single_lsa_attn

# offline synthetic fallback (no HF download), same selection
uv run --extra cpu --extra train python full_tests/run_e2e.py \
  --category attn --only lsa --no-realonall

# sizing knobs when needed: --max-samples N --max-epochs 1 --batch-size 2 --timeout 1800
# GPU smoke (real training box): swap the extra for --extra cu118 --extra train.
```

Expected evidence of the real-data smoke: every selected combo reports **PASS with a finite
training loss** over real corpus windows (28 toy / 32 real Borges windows per run, per the
harness table at `run_e2e.py` line 152), and the run directory keeps the generated `config.yaml`
+ metrics for inspection. A NaN/Inf loss, a schema rejection or a timeout is a **blocker**, not
a warning. If the HF Hub is unreachable, record the `--no-realonall` PASS and re-run the
real-data pass later.

**Phase E — figures + param gates + Sphinx**

```bash
# figures (needs Chrome/Chromium or PUPPETEER_EXECUTABLE)
( cd docs/paper/figures && ./build-figures.sh both )

# param ground truth + JS estimator (after the examples/presets land)
uv run --extra cpu --extra train python full_tests/param_count_check.py
node full_tests/param_estimate_check.mjs \
  erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js \
  full_tests/param_truth.json --tol 0.01

# Sphinx API docs
bash docs/build_docs.sh
```

Expected: `build-figures.sh` writes `mixers/lsa.{pdf,svg}` + refreshed
`mixer-taxonomy.{pdf,svg}` / `sparse-map.{pdf,svg}`; `param_count_check.py` prints
"`<N> configs counted, 0 failures"` and regenerates `param_truth.json` (intentional diff — the
two new configs appear); the estimator prints `ALL PASS` (≤ 1% drift, exact formula
`2H² + 2H·(nKv·hd)`); Sphinx builds under `-W --keep-going` (the new `lsa_attn.rst` resolves).

**Phase F — PDF rebuild (both papers)**

```bash
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

Post-compile verification: no `Fatal error`, no `Citation .* undefined` in the logs
(`kubicek_lsa_2026` must resolve), `paper.pdf`/`paper-es.pdf` newer than the edited
`.tex`/`.bbl`, and `git status --short docs/paper docs/paper-es` shows only intended
`.tex`/`.bbl`/`.pdf`/figure files. Fallbacks: `pdflatex`/`bibtex` **are** available today
(`/usr/bin`); if they become unavailable, leave the sources updated and note the rebuild must
happen on a TeX machine before the next release (the website sync copies prebuilt PDFs only). If
no Chrome is available for mermaid, commit the `.mmd` source and note the `.pdf`/`.svg` render
must be produced before the paper/website sync.

**Phase G — website sync** (§6) → **Phase H — commit/push** (§7).

---

## 6. Website sync (planned, not executed)

Follow `.opencode/command/sync-website.md` (mirror = `<repo>/erickfmm.github.io`, gitignored; all
git ops `git -C erickfmm.github.io …`; state script `.opencode/command/sync-website-state.sh`).

1. **Bootstrap/refresh**: mirror exists and is clean (`git -C erickfmm.github.io status --short`
   empty today). If `WEBSITE_REPO=MISSING` ever shows: `git clone
   https://github.com/erickfmm/erickfmm.github.io.git erickfmm.github.io` from the repo root.
2. **Copy per the file map** (main → mirror, plain `cp`; no rsync): `src/schema.yaml` →
   `schema.yaml`; `src/schema/**` → `schema/**`; `configs/*.yaml` + `configs/README.md` →
   `examples/*.yaml` + `examples/README.md` (top level only — includes the new `configs/lsa.yaml`);
   `configs/examples/**` → `examples/examples/**` (includes `es_arch_lsa_adamw.yaml`);
   `docs/paper/{paper.tex,paper.bbl,paper.pdf}` + `sections/*.tex` + `appendices/*.tex` →
   `paper/en/…`; `docs/paper-es/{…}` → `paper/es/…`; `docs/bibliography/**` →
   `paper/bibliography/**` (`.bib` only — includes the new `kubicek_lsa_2026`);
   `docs/architecture.md` → `paper/specs/architecture.md`; `docs/paper/figures/**`
   (`.mmd`, `.pdf`, `.svg` — includes the new `mixers/lsa.*` and the refreshed
   taxonomy/sparse-map) → `paper/figures/**`.
3. **Site-only edits** (mirror is the source of truth for these — never copy over them):
   `ft-diagram.js` + `ft-param-estimator.js` per row 42; `index.html` needs nothing
   (schema-driven dropdown).
4. **Paper tab HTML rebuild**:
   ```bash
   cd erickfmm.github.io/frankenstein-transformer/paper
   python3 flatten_tex.py en/paper.tex    en/paper.bbl    /tmp/paper-en-flat.tex --web --manifest /tmp/paper-en-figures.json
   python3 flatten_tex.py es/paper-es.tex es/paper-es.bbl /tmp/paper-es-flat.tex --web --manifest /tmp/paper-es-figures.json
   pandoc /tmp/paper-en-flat.tex -f latex -t html5 --standalone --mathjax --toc --toc-depth=3 -V lang=en -o paper-en.html
   pandoc /tmp/paper-es-flat.tex -f latex -t html5 --standalone --mathjax --toc --toc-depth=3 -V lang=es -o paper-es.html
   python3 postprocess_html.py --manifest /tmp/paper-en-figures.json paper-en.html
   python3 postprocess_html.py --manifest /tmp/paper-es-figures.json paper-es.html
   ```
   plus the top-level `paper/en.pdf` / `paper/es.pdf` copies of the rebuilt PDFs (`--web` +
   `postprocess_html.py` injects the SVG figures — mandatory, pandoc drops `\fitfigure`).
5. **Validation Gate** (all must pass before the commit phase):
   ```bash
   uv run --extra cpu --extra train python full_tests/param_count_check.py   # "0 failures"
   node full_tests/param_estimate_check.mjs \
     erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js \
     full_tests/param_truth.json --tol 0.01                                  # ALL PASS
   # JSON round-trip script over erickfmm.github.io/frankenstein-transformer/examples/**/*.yaml
   # CLI parse script (build_parser() over the 11 documented command lines)
   # state-script probes: schema-refs, example-yaml, cli-flags, … all PASS
   ```
6. Leave **both repos uncommitted** until the commit phase (§7).

---

## 7. Commit plan (planned, not executed)

Follow `/make-a-commit` conventions: English, conventional tag, structured body; never stage
`*.aux *.log *.out *.toc *.blg`, caches or mermaid temp files; `full_tests/param_truth.json`
changes are intentional.

1. **Main repo** — `feat(attention): add LSA mixer (arXiv:2610.42147)` (style of the last mixer
   commit, `4971c83 feat(attention): add Memory Attention mixer (arXiv:2609.28399)`). Body
   covers: module (`lsa_attn.py`, Ward clustering + flat semantic search, cached `BucketKVSlot`
   branch), registry/exports, schema (enum + both bilingual tables + `_attention_lsa.yaml` +
   `$ref` + renames + `config.py` fields/validation + `use_pe:false`), cache (capabilities +
   IMPLEMENTED_MIXERS + slots), qk-norm membership, example (`es_arch_lsa_adamw.yaml`) + preset
   (`configs/lsa.yaml`), tests (schema gates, `LSAAttentionTests`, bitnet coverage,
   `LSABucketParityTests`), e2e `ATTENTIONS` + real-data smoke result, figures (`lsa.mmd` +
   taxonomy/sparse-map bumps + rendered pdf/svg), docs (annex 4 EN/ES subsection, summary
   tables + census 47, schema annex rows + `lsa.*` block, section 04 + intro/discussion/
   conclusion EN/ES, annex-15 counts, README/architecture/configs-README, bibliography, Sphinx
   rst), rebuilt PDFs. Stage exactly:
   ```
   src/model/attention/sparse/lsa_attn.py src/model/attention/sparse/__init__.py
   src/model/attention/__init__.py src/model/hybrid_layer.py src/model/config.py
   src/utils/config_flatten.py src/schema/_model/_attention_lsa.yaml
   src/schema/_model.yaml src/schema/_model/_dims.yaml
   src/schema/_model/_positional_encoding.yaml src/model/cache/slots.py
   src/model/cache/capabilities.py src/model/norm/qk.py
   configs/examples/es_arch_lsa_adamw.yaml configs/lsa.yaml full_tests/run_e2e.py
   tests/test_schema_attention_layers.py tests/test_schema_consistency.py
   tests/test_sparse_attention.py tests/test_bitnet_coverage.py
   tests/test_kv_cache_mixers.py full_tests/param_truth.json
   docs/paper/figures/mixers/lsa.mmd docs/paper/figures/mixers/lsa.pdf
   docs/paper/figures/mixers/lsa.svg docs/paper/figures/mixer-taxonomy.mmd
   docs/paper/figures/mixer-taxonomy.pdf docs/paper/figures/mixer-taxonomy.svg
   docs/paper/figures/sparse-map.mmd docs/paper/figures/sparse-map.pdf
   docs/paper/figures/sparse-map.svg docs/bibliography/attention_types.bib
   docs/paper/appendices/annex-4-sparse-attention.tex
   docs/paper/appendices/annex-summary-tables.tex docs/paper/appendices/annex-schema.tex
   docs/paper/appendices/annex-15-kv-cache.tex
   docs/paper/sections/04-architecture-taxonomy.tex
   docs/paper/sections/01-introduction.tex docs/paper/sections/10-discussion.tex
   docs/paper/sections/11-conclusion.tex docs/paper/paper.bbl docs/paper/paper.pdf
   docs/paper-es/appendices/annex-4-sparse-attention.tex
   docs/paper-es/appendices/annex-summary-tables.tex docs/paper-es/appendices/annex-schema.tex
   docs/paper-es/appendices/annex-15-kv-cache.tex
   docs/paper-es/sections/04-taxonomia-arquitectura.tex
   docs/paper-es/sections/01-introduccion.tex docs/paper-es/sections/10-discusion.tex
   docs/paper-es/sections/11-conclusion.tex docs/paper-es/paper-es.bbl docs/paper-es/paper-es.pdf
   README.md docs/architecture.md configs/README.md
   docs/source/api/model/attention/sparse/lsa_attn.rst
   docs/source/api/model/attention/sparse/index.rst
   ```
   Then `git push origin HEAD:main`.
2. **Mirror** — `git -C erickfmm.github.io add frankenstein-transformer/` (schema, examples,
   paper tex/bbl/pdf/figures, bibliography, architecture.md, `ft-diagram.js`,
   `ft-param-estimator.js`, `paper/paper-en.html`, `paper/paper-es.html`, `paper/en.pdf`,
   `paper/es.pdf`), commit `docs(site): sync LSA mixer (arXiv:2610.42147)` (style of
   `bd19917 docs(site): sync Memory Attention mixer (arXiv:2609.28399)`), then
   `git -C erickfmm.github.io push origin HEAD:main`.

---

## 8. Verification matrix

| Gate | Command | Expected |
|---|---|---|
| Targeted pytest | `uv run --extra cpu --extra train python -m pytest tests/test_schema_attention_layers.py tests/test_schema_consistency.py tests/test_yaml_examples.py tests/test_attention_refactor.py tests/test_sparse_attention.py tests/test_bitnet_coverage.py tests/test_kv_cache_mixers.py -v --tb=short -p no:warnings` | All PASS |
| Full pytest | `uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings` | All PASS (76 files) |
| e2e combo enumeration | `… run_e2e.py --category attn --list-combos \| grep -F lsa` | `attn_single_lsa_attn` + every `attn_<a>__lsa_attn` / `attn_lsa_attn__<b>` pair id |
| **e2e targeted real-data training** | `… run_e2e.py --category attn --only lsa` | Every selected combo **PASS** with finite training loss on the real Borges corpus; run dir keeps `config.yaml` + metrics |
| e2e solo (light) | `… run_e2e.py --category attn --only attn_single_lsa_attn` | PASS |
| Offline fallback | `… run_e2e.py --category attn --only lsa --no-realonall` | PASS (synthetic corpus) |
| Figure render | `( cd docs/paper/figures && ./build-figures.sh both )` | `mixers/lsa.pdf` + `lsa.svg` written; taxonomy + sparse-map refreshed; no mermaid syntax error |
| Param count | `uv run --extra cpu --extra train python full_tests/param_count_check.py` | `… configs counted, 0 failures`; `param_truth.json` regenerated (intentional diff, 2 new configs) |
| JS estimator | `node full_tests/param_estimate_check.mjs erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js full_tests/param_truth.json --tol 0.01` | `ALL PASS` (≤ 1%; exact formula `2H² + 2H·(nKv·hd)`) |
| Sphinx | `bash docs/build_docs.sh` | HTML build OK under `-W --keep-going` (`lsa_attn.rst` resolves) |
| PDFs | `pdflatex`×2 + `bibtex` sequence (§5 Phase F) both papers | No `Fatal error`, no `Citation .* undefined` (`kubicek_lsa_2026` resolves), PDFs newer than sources, `git status --short docs/paper docs/paper-es` shows only intended files |
| KV parity (decision 1) | `pytest tests/test_kv_cache_mixers.py -v` | `lsa_attn` × {rope, nope, alibi} default parity PASS + `LSABucketParityTests` PASS |
| Mirror probes | Validation Gate of `sync-website.md` (§6.5) | `0 failures`, `ALL PASS`, JSON round-trip bad = 0, CLI parse all `OK`, state probes PASS |
| Census grep | `rg -n "forty-six\|cuarenta y seis\|(46 variants)\|Sparse (10)\|Disperso (10)\|46 sequence mixers\|46 mezcladores\|forty-five dispatchable" README.md docs/` after edits | No stale hits remain (pre-existing A-flags excluded) |

---

## 9. Risks, assumptions and open questions

**Assumptions to re-verify at implementation time**
- Line numbers cited above were read on 2026-11-06 at HEAD `4971c83`; re-anchor with grep before
  editing (the repo moves fast).
- `S % n_pool ≠ 0` and `T % n_cluster ≠ 0` handling (§2.5) are implementation decisions consistent
  with the paper's text but not literally specified by it — document them in the module docstring.
- Batch handling of Ward clustering (vectorize over the batch or loop) is an implementation
  choice; determinism only needs to hold within a process for parity (both paths run in-process).
- `configs/lsa.yaml` training keys must be schema-clean for `causal_lm` (mirror
  `configs/examples/causal_lm_decoder.yaml`); `vocab_size` is a placeholder the tokenizer
  pipeline overrides.
- HF Hub reachability for the real-data smoke; if down, `--no-realonall` PASS + rerun later.

**Risks**
- **BucketKVSlot complexity** is the largest new surface (user-chosen): truncate/rollback
  semantics (drop buckets intersecting the cut), full-precision centroid storage, and the D5
  defense (plain-KVSlot fallback ⇒ on-the-fly re-clustering) must each be covered by tests.
- Parity depends on deterministic, order-stable Ward merges (mutual-cheapest pairs, ties broken
  by index) — identical in legacy and cached paths within one process; keep the merge loop
  tie-breaking fixed (`argmin` on a Gram-derived cost with stable index order).
- `num_kv_heads > 1` generalizes eq. (3) per KV group (max over the group's query heads); the
  paper is MQA (`nKv = 1`). Default configs set 1; the e2e/bitnet/parity suites exercise `nKv=2`.
- PE + LSA: with `lsa_attn_use_pe: true` the module clusters post-PE keys (deviation from the
  paper's RoPE-free-part note); default `false` is paper-exact. PaPE layers never cache (D10) —
  handled by `manager.resolve_slot` lines 191–196, no new work.
- BitNet: no interaction beyond `BitLinear` projections (same shapes); covered by
  `test_bitnet_coverage.py`.
- Mode rules: none (works in encoder and decoder); no `config_loader` change.

**Open questions / out-of-scope variants (recorded, not planned)**
- **D-LSA** (eq. 1 index projections + eq. 2 forward-KL distillation against a frozen teacher)
  and **D-LSA (Training-Free)** (PCA directions) — need a distillation pipeline this repo lacks.
- **Tree search / beam search** (Algorithm 2, `n_{b,ℓ}`, `p_ℓ`) — the paper itself trains flat
  only ("We train N-LSA with flat search only, and leave native training with tree search to
  future work"). Could later become `lsa.tree_levels`-style knobs.
- The paper's Floyd–Rivest threshold selection, CuTe/Triton kernels and SGLang context
  parallelism — pure performance engineering, irrelevant for this repo's CPU-first reference
  implementation.

**Pre-existing drift found during the census (flagged; fix only with explicit user approval)**
- **A1** `mixer-taxonomy.mmd` line 9 lists only 9 sparse nodes — `swa_attn` is missing while
  the label says the family has more. Row 23 offers the coherent fix (add both `swa_attn` and
  `lsa_attn`, label `(11)`); the minimal alternative (add only `lsa_attn`, label `(10)`) leaves
  the diagram inconsistent with every other census surface.
- **A2** `annex-summary-tables.tex` line 81 (EN+ES) labels the latent block "Latent (9)" while
  listing 10 latent rows + ssog — unrelated to LSA; not fixed by this plan.
- **A3** `annex-schema.tex` line 209 (EN+ES) says "43 mixer codes" (stale before this change:
  46 actual). Row 30 updates it to 47 because we edit the same file/table anyway; revert to
  leave-stale if the user prefers.
- **A4** `README.md` line 180 still says "(43 variants / 8 families)" — untouched by this plan.
- **A5** `docs/build_docs.sh` lines 82–83 point at `docs/paper.tex` instead of
  `docs/paper/paper.tex` (LaTeX pre-check silently skips) — untouched; harmless to Sphinx.
- **A6** Stray typo'd mirror dir `erickfmm.github.io/frankestein-transformer/` (missing "t")
  exists on disk — do not delete without approval.

---

## 10. Out of scope

This command created, edited or deleted **nothing** outside the plan directory: no code, schema,
preset, test, figure, doc, PDF, mirror or git change was made in the repository. No `pytest`, no
e2e, no training, no `pdflatex`/`bibtex`/`pandoc`/`flatten_tex.py`/`build_docs.sh`/
`build-figures.sh`, no `uv sync`, no `git add/commit/push` was run. All commands above are quoted
for the implementer to run at implementation time. The two new YAMLs, the module, the slot and
every doc edit exist only as plans in this file.
