# Plan: add NorMuon optimizer (arXiv:2510.05491)

**Paper:** *NorMuon: Making Muon more efficient and scalable* — Zichong Li, Liming Liu
(Georgia Tech), Chen Liang, Weizhu Chen (Microsoft), Tuo Zhao (Georgia Tech).
arXiv:2510.05491 (Oct 2025).

**Summary.** Muon's Newton–Schulz orthogonalization equalizes the singular values of the
update (low condition number) but leaves the per-neuron (per-row) update norms highly
non-uniform, so a few neurons dominate training. NorMuon keeps Muon's orthogonalized path
and adds a **neuron-wise adaptive learning rate**: a per-row second-moment EMA
`v_t ∈ R^m` of the *orthogonalized* update, applied as a row-wise normalization
`O_t ⊘ (√V_t + ε)` after Newton–Schulz, with the update rescaled to Adam's empirical RMS
(`η̂ = 0.2η√(mn)/‖O_t‖_F`). State cost is `m(n+1)` — Muon's memory plus `m` scalars.
The paper reports 21.74% / 13.91% training-step reductions vs Adam at 1.1B / 5.4B and
6–15% vs Muon on Modded-NanoGPT 124M/350M. It belongs in this repo's *Orthogonalized*
family next to `muon`/`ada_muon`/`muon2`: it is the only member that adapts *rows*
(neurons) of the *orthogonalized* direction (AdaMuon is element-wise, Muon² preconditions
*before* orthogonalization).

---

## 2. Paper extract

### Update rule (Algorithm 1, verbatim)

For a hidden-layer weight matrix `W_t ∈ R^{m×n}`, gradient `G_t`, momentum coefficient
`β₁`, second-moment decay `β₂`, perturbation `ε`, weight decay `λ`, learning rate `η`
(lines refer to Algorithm 1 in the paper):

$$
\begin{aligned}
\text{(l.5)}\quad M_t &= \beta_1 M_{t-1} + (1-\beta_1) G_t, & M_0 &= 0 \\
\text{(l.6)}\quad O_t &= \mathrm{NS}_5(M_t) \\
\text{(l.7)}\quad v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,\mathrm{mean}_{\mathrm{cols}}(O_t \odot O_t), & v_0 &= 0,\; v_t \in \mathbb{R}^m \\
\text{(l.8)}\quad V_t &= \mathrm{ExpandRows}(v_t) \in \mathbb{R}^{m\times n} \\
\text{(l.9)}\quad O_t &\leftarrow O_t \oslash \big(\sqrt{V_t} + \epsilon\big) \\
\text{(l.10)}\quad \hat\eta &= 0.2\,\eta\,\sqrt{mn}\,/\,\lVert O_t\rVert_F \\
\text{(l.11)}\quad W_{t+1} &= W_t - \eta\lambda W_t - \hat\eta\, O_t
\end{aligned}
$$

`NS5` is Muon's five-iteration Newton–Schulz: `X_0 = M_t / ‖M_t‖_F`, quintic step
`X_{k+1} = aX_k + b(A_k X_k) + c(A_k^2 X_k)` with `A_k = X_k X_k^T`, coefficients
`(a,b,c) = (3.4445, −4.7750, 2.0315)`, `O_t = X_5` — **identical to this repo's
`muon.py`** (same triple, bfloat16, transpose-when-tall trick).

**Interpretation notes (needed for a faithful implementation):**

* `mean_cols(O ⊙ O)` is the per-row (per-neuron) mean of squared entries — exactly what
  `dion3.py`'s `normuon` flavor computes (`orth_f32.square().mean(dim=1)`,
  `src/model/optimizer/dion3.py:293`).
* ε placement: Algorithm 2 (distributed) shows `O ⊘ (√V + ε)` — epsilon **outside** the
  square root; the repo's two precedents (`dion3` normuon, `ada_muon`) also put it
  outside. Implement `O_t / (√V_t + ε)` (see Open questions for the OCR ambiguity).
* Line 5's `(1−β₁)` rescale vs this repo's Muon convention `M ← μM + G`: with `M_0 = 0`
  the two are exact multiples (`N_t = (1−β₁)M_t`), and NS5's Frobenius normalization
  cancels any constant scaling — so the repo convention is **mathematically
  equivalent**. Implement the repo form (`momentum` suffix, default 0.95 = paper's β₁)
  and note the equivalence in the module docstring (keeps D11 compliance: the annex must
  restate the paper's line 5 verbatim).
* No Nesterov in Algorithm 1; no bias correction on `v_t`.
* Paper's own hybrid routing: "scalar parameters, bias vectors, embeddings, and
  unembedding layers continue to use standard optimizers such as Adam" → this repo's
  `ADAMW_FALLBACK_GROUPS` + non-2-D fallback (`base.py:108`, `muon.py:130`).
* RMS alignment `η̂ = 0.2η√(mn)/‖O_t‖_F` makes the update RMS exactly `0.2η` — the same
  `_RMS_TARGET = 0.2` as `ada_muon.py:24`.

### Pseudocode (single-process form to implement)

```
for group in param_groups:                     # group carries lr, wd, betas, eps, name
    use_adamw = group.name in {embeddings, pos, head}
    beta2 = group.betas[1]                     # shared per-group suffix
    eps   = group.eps                          # shared per-group suffix
    for p in group.params:
        if p.grad is None: continue
        if use_adamw or p.ndim != 2:
            adamw_fallback_update(p, p.grad, state, lr, betas, eps, wd)   # base.py
            continue
        buf = state.momentum_buffer; buf.mul_(momentum).add_(grad)       # l.5
        x = NS5(buf)  # bfloat16, transpose if rows>cols, (a,b,c) Muon triple  # l.6
        o = x.to(p.dtype).to(float32)
        v = state.row_second_moment            # shape (rows,), float32
        v.mul_(beta2).add_(o.square().mean(dim=1), alpha=1-beta2)        # l.7
        o = o / (v.sqrt().unsqueeze(1) + eps)                            # l.8-9
        if wd > 0: p.mul_(1 - lr*wd)                                      # l.11 (decoupled)
        eta_hat = 0.2 * lr * sqrt(rows*cols) / max(o.norm(), ns_eps)     # l.10
        p.add_(o.to(p.dtype), alpha=-eta_hat)                             # l.11
```

### Hyperparameters

| Name (paper) | Key in repo | Type | Default | Range | Scope | Coercion |
|---|---|---|---|---|---|---|
| β₁ (momentum, l.5) | `nor_muon-momentum` | float | 0.95 (paper: 0.95, searched {0.9, 0.95}) | [0, 1) | global | `to_float` |
| β₂ (row-moment decay, l.7) | `nor_muon-betas_<group>` (index 1) | [float,float] | (0.9, **0.95**) | index 1 ∈ [0,1) | per group (shared suffix) | `to_betas` |
| ε (l.9) | `nor_muon-eps_<group>` | float | 1e-8 | ≥ 0 | per group (shared suffix) | `to_float` |
| NS iterations (l.6) | `nor_muon-ns_steps` | int | 5 (NS5) | ≥ 1 | global | `to_int` |
| NS Frobenius ε (l.6) | `nor_muon-ns_eps` | float | 1e-7 | ≥ 0 | global | `to_float` |
| λ (l.11) | `nor_muon-wd_<group>` | float | 0.0 (paper trains with 0.1 on 2-D hidden) | ≥ 0 | per group (shared suffix) | `to_float` |
| η | `nor_muon-lr_<group>` | float | group default | ≥ 0 | per group (shared suffix) | `to_float` |

Fallback path additionally consumes the group `betas[0]`, `betas[1]`, `eps` exactly as
`muon.py` does.

### State buffers & per-step cost

* NorMuon path, per 2-D parameter: `momentum_buffer` (m×n, param dtype) +
  `row_second_moment` (m, float32) → paper's `m(n+1)`; ~50% of Adam's `2mn`.
* AdamW-fallback path: `exp_avg` + `exp_avg_sq` (as in `base.adamw_fallback_update`).
* Per-step cost: 5 Newton–Schulz iterations `O(mn·k)` in the summary-table notation
  (k = NS steps), plus `O(mn)` row statistics/normalization/norm — the paper measures
  2.9% training-step time over AdamW at 5.4B (Table 2); optimizer memory 21.19 GB vs
  AdamW 40.56 GB.

### Category

Existing **Orthogonalized** bucket (`docs/architecture.md` §8.1, line 780):
`muon, turbo_muon, af_muon, ada_muon, dion3, root, muon2` → add `nor_muon`.
Summary-table family label: *Orthogonality + neuron-wise adaptive*.

### BibTeX data

```bibtex
@misc{li_normuon_2025,
      title={NorMuon: Making Muon more efficient and scalable},
      author={Zichong Li and Liming Liu and Chen Liang and Weizhu Chen and Tuo Zhao},
      year={2025},
      eprint={2510.05491},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.05491},
}
```

(Matches the Zotero arXiv style of `liu_muon2_2026`, `optimizers.bib:383-391`. No key
collision — no existing `li_normuon*`.)

---

## 3. Decisions

1. **Registry name `nor_muon`**, class `NorMuon`, file `src/model/optimizer/nor_muon.py`.
   Rationale: repo convention is `<qualifier>_<family>` (`ada_muon`, `af_muon`,
   `turbo_muon`); the bare short name `normuon` would collide conceptually with the
   existing `dion3-normuon` suffix (a Dion3 option implementing this same paper's row
   normalization). No syntactic collision either way (`dion3-normuon` keeps its prefix),
   but `nor_muon` is unambiguous. The `dion3` entry stays untouched.
2. **Closest peers and what is mirrored:**
   * `muon.py` — hybrid routing (`ADAMW_FALLBACK_GROUPS`, non-2-D fallback), momentum
     buffer convention, NS5 (bfloat16, transpose trick, `(3.4445,−4.7750,2.0315)`),
     decoupled wd as `p.mul_(1 − lr·wd)`.
   * `ada_muon.py` — post-orthogonalization second moment + `_RMS_TARGET = 0.2` RMS
     alignment (`gamma = 0.2·√(mn)/‖O‖_F`, `ada_muon.py:166-168`); constructor default
     `lr=1e-3` (Adam-scale, since the update RMS is 0.2η).
   * `dion3.py` normuon flavor (`dion3.py:289-306`) — the row-mean EMA wiring:
     `row_moment = orth_f32.square().mean(dim=1)`, `v.mul_(β2).add_(row_moment, 1−β2)`,
     `O / (v.sqrt().unsqueeze(1) + eps)`, float32 statistics; β2 = `group["betas"][1]`,
     ε = `group["eps"]`.
   * `muon2.py` — `__init__` validation style (ValueError ranges for lr/momentum/betas/
     eps/ns_steps/ns_eps), consumption of shared per-group `betas`/`eps`.
3. **Category:** existing Orthogonalized bucket (see §2). No new category; the "eight
   categories" count is unchanged.
4. **Suffix table** (specific global suffixes only; shared per-group suffixes apply):
   `momentum` (float, 0.95, `to_float`), `ns_steps` (int, 5, `to_int`),
   `ns_eps` (float, 1e-7, `to_float`). β₂/ε come from the shared per-group
   `betas_*`/`eps_*` (muon2 + dion3 normuon precedent; the shared betas default
   (0.9, 0.95) gives exactly the paper's β₂ = 0.95).
5. **`supports_shared_per_group_suffixes: true`** (the norm).
6. **No `nesterov` suffix** — Algorithm 1 uses plain momentum (precedent: `ada_muon`).
7. **Example preset:** `configs/examples/es_arch_hybrid_nor_muon.yaml` — a mirror of
   `es_arch_hybrid_muon2.yaml` (the smallest Muon-family example that passes
   `load_training_config`: `mlm` task, hybrid `retnet/titan_attn/mamba/ode` pattern,
   full per-group lr/wd/betas/eps block + the three specific globals).
8. **Census bump 31 → 32** — actual old number read from the repo (registry has 31
   entries; architecture.md §8.1 heading says "(31)"). Every occurrence is listed in
   row 18/13/16/14/17 of the table below.
9. **No annex figure** — `ada_muon`, `muon2`, `dion3` were all added without figures;
   only `muon/turbo-muon/af-muon/root` have `docs/paper/figures/optimizers/<name>.pdf`.
   Omit the figure (command's rule: only if one can be produced in the same style).
10. **Distributed variant out of scope** — Algorithm 2 (FSDP2, gather/scatter of
    orthogonalization, ZeRO-3) is a multi-GPU deployment strategy, not a different update
    rule; the annex mentions it in one sentence, the implementation stays single-device
    like every other family member.

---

## 4. File-by-file plan table

Hard rules that bind every row: the schema is the contract
(`additionalProperties: false`); optimizer keys are prefixed `<name>-<group>_<param>` /
`<name>-<param>`; never read an undocumented key; the papers are the theory source of
truth (AGENTS.md D11) so the annex formulations must match arXiv:2510.05491 and the
`paper-es` versions must translate them faithfully.

| # | File | Planned change (concrete) |
|---|---|---|
| 1 | `src/model/optimizer/nor_muon.py` | **New file.** `from __future__ import annotations`; Google-style docstrings; module docstring citing Li et al. 2025, arXiv:2510.05491 (draft below); `_RMS_TARGET = 0.2` (ada_muon precedent); module-level `def _newton_schulz(matrix, steps, eps, coefficients=(3.4445,-4.7750,2.0315))` helper (testable, mirrors `dion3._gram_newton_schulz`); class `NorMuon(Optimizer)` with `__init__(self, params, lr=1e-3, momentum=0.95, ns_steps=5, ns_eps=1e-7, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)` and muon2-style `ValueError` validation (lr ≥ 0, 0 ≤ momentum < 1, betas[i] ∈ [0,1), eps ≥ 0, ns_steps ≥ 1, ns_eps ≥ 0); `@torch.no_grad() step()` exactly as the pseudocode in §2 — hybrid routing via `ADAMW_FALLBACK_GROUPS` + `p.ndim != 2` → `adamw_fallback_update`; NS5 in bfloat16 with the transpose-when-tall trick; row stats in float32 (`state["row_second_moment"]`, shape `(p.size(0),)`); update `p.add_(normalized, alpha=-eta_hat)` with `eta_hat = 0.2·lr·√(mn)/max(‖O‖_F, ns_eps)`. CPU-safe (no `.cuda()`); no Nesterov; docstring notes the `(1−β₁)` equivalence argument of §2. |
| 2 | `src/model/optimizer/__init__.py` | After line 21 (`from .muon2 import Muon2`) add `from .nor_muon import NorMuon`; in `__all__` insert `"NorMuon",` after `"Muon2",` (line 53). |
| 3 | `src/model/optimizer/factory.py` | Import `from .nor_muon import NorMuon` after line 45 (`from .muon2 import Muon2`); registry entry `"nor_muon": NorMuon,` after `"muon2": Muon2,` (line 88); allowed keys `"nor_muon": _COMMON_PER_GROUP_KEYS \| {"momentum", "ns_steps", "ns_eps"},` after the `muon2` line (line 155); dispatch block after the muon2 block (after line 478), mirroring `ada_muon` (factory.py:339-348): `momentum = to_float(scoped.get("momentum"), 0.95)`, `ns_steps = to_int(scoped.get("ns_steps"), 5)`, `ns_eps = to_float(scoped.get("ns_eps"), 1e-7)`, `return optimizer_cls(converted_groups, momentum=momentum, ns_steps=ns_steps, ns_eps=ns_eps)` — per-group betas/eps already injected by `_build_param_groups_for_optimizer`. |
| 4 | `src/schema/_optimizer.yaml` | (a) `optimizer_class.enum`: append `- nor_muon` after `- muon2` (line 178). (b) EN use-case table: after the `muon2` row (line 65) add `\| \`nor_muon\` \| Balanced neuron updates \| Neuron-wise (row) second-moment normalization after Newton-Schulz; Muon memory + m scalars; Adam-RMS-aligned update (paper: arxiv:2510.05491) \|`. (c) ES use-case table: after line 123 add `\| \`nor_muon\` \| Actualizaciones por neurona equilibradas \| Normalización por segundo momento de fila (neurona) tras Newton-Schulz; memoria de Muon + m escalares; actualización alineada al RMS de Adam (paper: arxiv:2510.05491) \|`. (d) EN references paragraph: append ``; \`nor_muon\` (paper: arxiv:2510.05491)`` after the `adam_pm` reference (line 87); same for ES (line 146). (e) `parameters.description` suffix list: after the `muon2` bullet (line 211) add `- \`nor_muon\`: \`momentum\`, \`ns_steps\`, \`ns_eps\`. Muon orthogonalization followed by neuron-wise normalization: a per-row second-moment EMA v_t of the orthogonalized update rescales each neuron row (β2 from the shared \`betas_*\`, ε from the shared \`eps_*\`); the update is rescaled to RMS 0.2η via η̂ = 0.2η√(mn)/\|\|O_t\|\|_F; same hybrid routing as \`muon\``. (f) `parameters.description_es` suffix list: after line 255 add the faithful translation: `- \`nor_muon\`: \`momentum\`, \`ns_steps\`, \`ns_eps\`. Ortogonalización de Muon seguida de normalización por neurona: una EMA de segundo momento por fila v_t de la actualización ortogonalizada reescala cada fila-neurona (β2 de los \`betas_*\` compartidos, ε de los \`eps_*\` compartidos); la actualización se reescala a un RMS 0.2η mediante η̂ = 0.2η√(mn)/\|\|O_t\|\|_F; mismo enrutamiento híbrido que \`muon\``. (g) `x_parameter_reference.by_optimizer`: after the `muon2` block (lines 426-429) append: `nor_muon:\n          prefix: "nor_muon-"\n          supports_shared_per_group_suffixes: true\n          specific_global_suffixes: [momentum, ns_steps, ns_eps]`. |
| 5 | `src/schema/_training.yaml` | Append exactly one `if/then` block at the end of the `allOf` list, after the `adam_pm` block (file currently ends at line 1623 with `^adam_pm-: { type: [number, boolean, array, string] }`):<br>`- if:`<br>`    properties:`<br>`      optimizer:`<br>`        properties:`<br>`          optimizer_class:`<br>`            const: nor_muon`<br>`  then:`<br>`    properties:`<br>`      optimizer:`<br>`        properties:`<br>`          parameters:`<br>`            additionalProperties: false`<br>`            patternProperties:`<br>`              ^nor_muon-: { type: [number, boolean, array, string] }`<br>Parity test `test_optimizer_prefixes_match_conditional_rules` requires exactly one block per prefix. |
| 6 | `src/schema/_sbert.yaml` | `optimizer_kwargs.description` (line 400): extend the momentum list to `…/dion3/root/nor_muon)`, the ns_steps and ns_eps lists to `…/root/muon2/nor_muon)`, and append after the ada_muon sentence: ``For nor_muon the shared per-group betas[1] is the neuron-wise second-moment decay and the shared eps its row-normalization epsilon (defaults 0.95 / 1e-8).`` `optimizer_kwargs.description_es` (line 401): same three edits in Spanish (``momentum``/``ns_steps``/``ns_eps`` lists + ``Para `nor_muon`, el `betas[1]` por grupo compartido es el decaimiento del segundo momento por neurona y el `eps` compartido su épsilon de normalización por fila (por defecto 0.95 / 1e-8).``). |
| 7 | `configs/examples/es_arch_hybrid_nor_muon.yaml` | **New file**, a 97-line mirror of `configs/examples/es_arch_hybrid_muon2.yaml` with the `muon2-` prefix replaced by `nor_muon-` throughout (same model block: `frankenstein_text_encoder`, 768/12/12, layer_pattern `[retnet, titan_attn, mamba, ode]`, layer_norm, ffn 3072, ode rk4/2, retnet heads 12; same training block: `mlm`, seed 42, 4 epochs, batch 2, max_length 256, mlm_probability 0.15, same dataset/cache/parquet keys, `gradient_accumulation_steps: 12`, `scheduler_total_steps: 12000`, `csv_log_path: training_metrics_es_arch_hybrid_nor_muon.csv`), full per-group `lr_*`/`wd_*`/`betas_*`/`eps_*` block (identical values to the muon2 example) plus the three specific globals: `nor_muon-momentum: 0.95`, `nor_muon-ns_steps: 5`, `nor_muon-ns_eps: 1.0e-7`. Must pass `load_training_config` (auto-globbed by `tests/test_yaml_examples.py`). |
| 8 | `configs/README.md` | Chooser list: `- nor_muon` after `- muon2` (line 164). Specific-suffix list: after the `muon2` bullet (line 217) add ``- `nor_muon`: `momentum`, `ns_steps`, `ns_eps` (neuron-wise second-moment normalization after Newton-Schulz; β2 from the shared `betas_*`, ε from the shared `eps_*`; update rescaled to Adam's RMS via η̂ = 0.2η√(mn)/‖O_t‖_F; same hybrid routing as `muon`)``. Example keys: after `- muon2-ns_steps: 3` (line 231) add `- nor_muon-momentum: 0.95`. |
| 9 | `full_tests/run_e2e.py` | `OPTIMIZERS` list (line 267-273): append `"nor_muon"` after `"adam_pm"` (keeps newest-last, matches root/muon2/adam_pm precedent). Param map (after the `adam_pm` block, line 710-712) add: `if optimizer_class == "nor_muon":` → `params[f"{prefix}momentum"] = 0.95`, `params[f"{prefix}ns_steps"] = 5`, `params[f"{prefix}ns_eps"] = 1e-7`. New combo id: `opt_nor_muon` (ids look like `opt_muon2`, line 63). |
| 10 | `tests/test_optimizer_factory.py` | (a) Add `"nor_muon"` to the `required` set of `test_registry_contains_all_required` (line 30). (b) New class `BuildOptimizerNorMuonTests(unittest.TestCase)` with docstring `"""NorMuon defaults, paper invariants and hybrid routing (arXiv:2510.05491)."""`, inserted after `BuildOptimizerMuon2Tests` (after line 407), following the Muon2/ROOT/AdaMuon class patterns: `test_builds_without_error`; `test_step_runs`; `test_defaults_match_paper` (momentum 0.95, ns_steps 5, ns_eps 1e-7, betas (0.9, 0.95), eps 1e-8); `test_hidden_matrix_uses_normuon_state` (state has `momentum_buffer` + `row_second_moment` of shape `(rows,)`/float32, no `exp_avg*`); `test_row_second_moment_matches_first_step_ema` (recompute `O = _newton_schulz(momentum_buffer, 5, 1e-7)` on a clone and assert `v == (1-β2)·mean_cols(O²)`); `test_embedding_group_uses_adamw_fallback` (exp_avg/exp_avg_sq present, no momentum_buffer); `test_update_rms_matches_paper_alignment` (with wd=0, `‖Δp‖_F == 0.2·lr·√(mn)` within tolerance — mirrors `test_update_norm_matches_rms_alignment`, `ada_muon` tests at factory tests line 905); `test_normalized_rows_have_uniform_norms` (THE NorMuon property: after one step, per-row norms of the applied direction are near-uniform — ratio max/min ≤ 1.02, since row norms ≈ √(n/(1−β2)) after normalization); `test_beta2_from_shared_betas_routes` (`nor_muon-betas_attention: [0.9, 0.8]` → row EMA uses decay 0.8); `test_specific_suffixes_parsed` (string coercion + invalid fallback to defaults, mirroring factory tests lines 379-395); `test_unknown_group_suffix_raises` (`nor_muon-lr_bogus` → ValueError mentioning `lr_bogus`); `test_per_group_lr_routes` (`nor_muon-lr_ffn: 0.02`). |
| 11 | `docs/bibliography/optimizers.bib` | Append the `@misc{li_normuon_2025, …}` entry from §2 after the last entry (`ji_adampm_2026`, line 393+), keeping the Zotero arXiv field style. |
| 12 | `docs/paper/appendices/annex-1-optimizers.tex` | In `\subsection{Second-Order, Geometric, and Orthogonality Optimizers}` (line 895), insert a `\paragraph{NorMuon (Neuron-wise Normalized Muon).}` block between the Muon² Pros `\end{itemize}` (line 1523) and `\subsection{Evolutionary and Population-Based Hybrids}` (line 1525). Content (draft): intro citing `\citep{li_normuon_2025}` — Muon orthogonalization fixes the singular-value spectrum but leaves per-neuron L2 norms non-uniform (Fig. 1 of the paper); NorMuon tracks a per-row second-moment EMA of the orthogonalized update and normalizes rows before an Adam-RMS rescaling. `align` block restating Algorithm 1 lines 5-11 verbatim (with `M_t = β1 M_{t-1} + (1-β1) G_t`, `O_t = NS5(M_t)`, `v_t = β2 v_{t-1} + (1-β2) mean_cols(O_t ⊙ O_t)`, `V_t = ExpandRows(v_t)`, `O_t ← O_t ⊘ (√V_t + ε)`, `\hat η = 0.2 η √(mn)/\|O_t\|_F`, `W_{t+1} = W_t − ηλW_t − \hat η O_t`); `algorithm` float `\caption{NorMuon (Neuron-wise Normalized Muon)}\label{alg:normuon}` with the hybrid `If hidden-layer 2D matrix / Else AdamW fallback` structure of `alg:muon2`; one-sentence mention of the FSDP2 distributed variant (Alg. 2) as future deployment work; results sentence (21.74%/13.91% vs Adam at 1.1B/5.4B; 6-15% fewer iterations than Muon on 124M/350M Modded-NanoGPT; state `m(n+1)` ≈ Muon memory, 2.9% step-time overhead); paragraph noting the implementation mapping (momentum suffix = β1, shared per-group betas[1] = β2, shared eps = ε, and the `(1−β1)`-scaling/Frobenius-normalization equivalence); `\paragraph{Cons.}` (extra `m`-vector state; ε-placement follows Alg. 2; no Nesterov) and `\paragraph{Pros.}` (uniform neuron norms + low condition number; Adam-scale updates reuse Adam LR schedules; ~50% of Adam memory). **No figure** (see decision 9). |
| 13 | `docs/paper/appendices/annex-summary-tables.tex` | (a) Intro sentence line 111: "catalogs all **31** supported optimizers" → "all **32** supported optimizers". (b) In `tab:all-optimizers`, under the `\multicolumn{6}{c}{\textbf{Second-Order, Geometric \& Orthogonality}}` header (line 158), append after the Muon² row (line 169): `NorMuon & Orthogonality + neuron-wise adaptive & 2 ($m$, row $v$); + $v$ on the AdamW fallback & $\mathcal{O}(n \cdot k)$ & lr, momentum, $\beta_2$, eps, wd, NS steps $k$ & \citep{li_normuon_2025} \\`. |
| 14 | `docs/paper/sections/05-optimizer-families.tex` | Line 4: "The system supports **thirty-one** optimizer families in eight categories" → "**thirty-two**"; in the same sentence extend "geometry-oriented methods (Muon, Turbo-Muon, AF-Muon, AdaMuon, Dion3, ROOT, Muon²)" → "…, Muon², **NorMuon**)". |
| 15 | `docs/paper-es/appendices/annex-1-optimizadores.tex` | Faithful Spanish translation of row 12, inserted between the Muon² `Ventajas` `\end{itemize}` (line 1538) and `\subsection{Híbridos Evolutivos y Poblacionales}` (line 1539): `\paragraph{NorMuon (Muon Normalizado por Neurona).}` + the same `align` equations + `\caption{NorMuon (Muon Normalizado por Neurona)}\label{alg:normuon}` algorithm float + `Contras`/`Ventajas` + `\citep{li_normuon_2025}`. Terminology aligned with the existing ES annex (e.g. "ortogonalización Newton–Schulz", "segundo momento por fila", "respaldo AdamW", as in the Dion3 ES paragraph, line 1330). |
| 16 | `docs/paper-es/appendices/annex-summary-tables.tex` | (a) Line 111: "cataloga los **31** optimizadores soportados" → "los **32** optimizadores soportados". (b) After the Muon² row (line 169) under `Segundo Orden, Geométricos y Ortogonalidad` (line 158): `NorMuon & Ortogonalidad + adaptativa por neurona & 2 ($m$, $v$ por filas); + $v$ en el respaldo AdamW & $\mathcal{O}(n \cdot k)$ & lr, momentum, $\beta_2$, eps, wd, pasos NS $k$ & \citep{li_normuon_2025} \\`. |
| 17 | `docs/paper-es/sections/05-familias-optimizadores.tex` | Line 4: "soporta **treinta y una** familias de optimizadores" → "**treinta y dos**"; extend "métodos orientados a geometría (Muon, Turbo-Muon, AF-Muon, AdaMuon, Dion3, ROOT, Muon²)" → "…, Muon², **NorMuon**)". |
| 18 | `docs/architecture.md` + `README.md` (+ `configs/README.md`) | **Census pass, 31 → 32** (all occurrences found by grepping the old count): `docs/architecture.md` line 26 (`\| Optimizer families \| **31** \|` → `**32**`), line 87 (`\| src/model/optimizer/** \| 31 optimizers + registry + parameter routing` → `32 optimizers`), line 767 (`### 8.1 Optimizers (31)` → `(32)`), and line 780 (§8.1 family table "Orthogonalized" row: add `nor_muon` to the list and extend the character text with: "; `nor_muon` normalizes the orthogonalized update neuron-wise (row-level second-moment EMA of `O_t`, β2 from the shared per-group `betas_*`, Adam-RMS-aligned update)"). `README.md` line 20 (At a Glance table Optimizers column `**31**` → `**32**`), line 27 (mermaid "Train<br/>31 optimizer families" → `32 optimizer families`), line 98 (`\| Optimizer families \| 31 across 8 categories \|` → `32 across 8 categories`). `configs/README.md`: **n/a** for census — verified by grep that it carries no optimizer count (its "31 schema-compliant YAML presets" in `configs/examples/examples.md` is a stale curated count that recent optimizer additions (muon2/dion3/root/adam_pm) never touched — leave it; see Open questions). |

**Also read-only-verified, no change needed:** `tests/test_optimizer_base.py` (generic
helpers only), `tests/test_schema_attention_layers.py`, `src/utils/config_flatten.py`
(optimizers have no nested config), `src/model/config.py` (no optimizer fields),
`src/streamlit_gui/` (schema-driven). `docs/paper/appendices/annex-1-optimizers.tex`
end-of-file category longtable (line 1693) and its ES twin (line 1707) list the
geometry-oriented members ("Muon, Turbo-Muon, AF-Muon, AdaMuon, Dion3") — they predate
ROOT/Muon² (pre-existing drift). Recommended in-scope edit: add NorMuon to the row;
the missing ROOT/Muon² mentions are flagged under Open questions (needs user approval,
since it is unrelated drift).

---

## 5. Ordered execution checklist

Implementation-time only — **none of these are run by the planning session**.

- [ ] 1. `src/model/optimizer/nor_muon.py` (row 1)
- [ ] 2. `src/model/optimizer/__init__.py` (row 2)
- [ ] 3. `src/model/optimizer/factory.py` (row 3)
- [ ] 4. `src/schema/_optimizer.yaml` (row 4)
- [ ] 5. `src/schema/_training.yaml` (row 5)
- [ ] 6. `src/schema/_sbert.yaml` (row 6)
- [ ] 7. `configs/examples/es_arch_hybrid_nor_muon.yaml` (row 7)
- [ ] 8. `configs/README.md` (row 8)
- [ ] 9. `full_tests/run_e2e.py` (row 9)
- [ ] 10. `tests/test_optimizer_factory.py` (row 10) — then targeted gates:

```bash
# targeted gates (all CPU-first; never a bare `uv run` — it swaps the torch extra)
uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py tests/test_schema_consistency.py tests/test_yaml_examples.py -v --tb=short -p no:warnings

# full suite
uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings

# fast e2e smoke — nor_muon is the last OPTIMIZERS entry, so --limit 3 will NOT
# reach it; use the targeted combo id (or --category opt --limit 32)
uv run --extra cpu --extra train python full_tests/run_e2e.py --only opt_nor_muon
```

- [ ] 11. `docs/bibliography/optimizers.bib` (row 11)
- [ ] 12. `docs/paper/appendices/annex-1-optimizers.tex` (row 12)
- [ ] 13. `docs/paper/appendices/annex-summary-tables.tex` (row 13)
- [ ] 14. `docs/paper/sections/05-optimizer-families.tex` (row 14)
- [ ] 15. `docs/paper-es/appendices/annex-1-optimizadores.tex` (row 15)
- [ ] 16. `docs/paper-es/appendices/annex-summary-tables.tex` (row 16)
- [ ] 17. `docs/paper-es/sections/05-familias-optimizadores.tex` (row 17)
- [ ] 18. `docs/architecture.md` + `README.md` census (row 18)
- [ ] 19. Rebuild both PDFs (preconditions verified by the planning session:
      `pdflatex` = `/usr/bin/pdflatex`, `bibtex` = `/usr/bin/bibtex`, `pandoc`, `node`
      all present):

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

  **Fallback if `pdflatex`/`bibtex` are unavailable on the implementation machine:**
  leave the `.tex`/`.bib` sources updated and note in the commit body that the PDF
  rebuild must happen on a TeX machine before the next release (the website sync copies
  prebuilt PDFs only). Post-compilation verification:
  * no `Fatal error` in the logs (the `rm` above clears them — check before deleting,
    or run `pdflatex` and grep the `.log` for `Fatal` / `Error`);
  * no `Citation 'li_normuon_2025' undefined` and no `Warning: Citation.*undefined`
    in either `.blg`/`.log`;
  * `paper.pdf` / `paper-es.pdf` mtimes newer than every edited `.tex`/`.bbl`;
  * `git status --short docs/paper docs/paper-es` shows only intended `.tex`, `.bbl`,
    `.pdf` files (never stage `*.aux *.log *.out *.toc *.blg`).

- [ ] 20. Website sync (§6) — after both PDFs are rebuilt
- [ ] 21. Commit + push (§7)

---

## 6. Website sync (planned, not executed)

Run `/sync-website` (`.opencode/command/sync-website.md`) at implementation time. Its
live state report comes from the read-only
`bash .opencode/command/sync-website-state.sh`. Files this change must sync:

* `src/schema.yaml` + `src/schema/**` (the `_optimizer.yaml`, `_training.yaml`,
  `_sbert.yaml` edits) → mirror `schema.yaml` / `schema/**`.
* `configs/examples/es_arch_hybrid_nor_muon.yaml` → `examples/examples/**` (full
  recursion), `configs/README.md` → `examples/README.md`.
* `docs/paper/{paper.tex,paper.bbl,paper.pdf}` + `sections/*.tex` +
  `appendices/*.tex` → `paper/en/…`; `docs/paper-es/…` → `paper/es/…`.
* `docs/bibliography/**` (`.bib` only) → `paper/bibliography/**`.
* `docs/architecture.md` → `paper/specs/architecture.md`.
* No figure changes (no figure is added for nor_muon).
* Rebuild the Paper-tab HTML (`paper/flatten_tex.py --web` + `pandoc`, per the
  sync-website pipeline) and refresh the site copies `paper/en.pdf` / `paper/es.pdf`
  (state-script probes `paper-en-copy` / `paper-es-copy` enforce the copy equality).
* `index.html`: verified during planning that it carries **no hardcoded optimizer
  census** — no edit expected; the Streamlit builder is schema-driven.

**Validation Gate (in order, each must pass):**

```bash
uv run --extra cpu --extra train python full_tests/param_count_check.py
# → must print "… configs counted, 0 failures" (regenerates full_tests/param_truth.json;
#   the new es_arch_hybrid_nor_muon.yaml changes it — intentional, commit it)

node full_tests/param_estimate_check.mjs \
  erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js \
  full_tests/param_truth.json --tol 0.01
# → exit 0, "ALL PASS" (the estimator is optimizer-agnostic; only the new config is added)

# then re-run the state script and require PASS on: schema-refs, example-yaml,
# cli-flags probes and the estimator/summary sections:
bash .opencode/command/sync-website-state.sh
```

The mirror stays uncommitted until the commit phase (§7 step 2).

---

## 7. Commit plan (planned, not executed)

Follow `/make-a-commit` conventions: English, conventional tag, short first line +
structured body; never stage `*.aux *.log *.out *.toc *.blg` or caches;
`full_tests/param_truth.json` changes are intentional.

1. **Main repo** — stage exactly:

```bash
git add src/model/optimizer/nor_muon.py \
        src/model/optimizer/__init__.py \
        src/model/optimizer/factory.py \
        src/schema/_optimizer.yaml src/schema/_training.yaml src/schema/_sbert.yaml \
        configs/examples/es_arch_hybrid_nor_muon.yaml configs/README.md \
        full_tests/run_e2e.py full_tests/param_truth.json \
        tests/test_optimizer_factory.py \
        docs/bibliography/optimizers.bib \
        docs/paper/appendices/annex-1-optimizers.tex \
        docs/paper/appendices/annex-summary-tables.tex \
        docs/paper/sections/05-optimizer-families.tex \
        docs/paper/paper.bbl docs/paper/paper.pdf \
        docs/paper-es/appendices/annex-1-optimizadores.tex \
        docs/paper-es/appendices/annex-summary-tables.tex \
        docs/paper-es/sections/05-familias-optimizadores.tex \
        docs/paper-es/paper-es.bbl docs/paper-es/paper-es.pdf \
        docs/architecture.md README.md
git commit -m "feat(optimizer): add NorMuon (arXiv:2510.05491)" -m "<body>"
git push origin HEAD:main
```

   Body must cover: the neuron-wise normalization mechanism (row second-moment EMA of
   the orthogonalized update, `η̂ = 0.2η√(mn)/‖O‖_F` alignment); new suffixes
   (`momentum`, `ns_steps`, `ns_eps`; β2/ε from the shared per-group `betas_*`/`eps_*`);
   schema/registry/e2e wiring (enum, `by_optimizer`, single `if/then`, SBERT
   description, `OPTIMIZERS` list); docs (annex EN/ES + algorithm float, summary-table
   rows + census 31→32, family sections, bibliography `li_normuon_2025`, rebuilt PDFs);
   example preset; test class.
2. **Mirror** (only after §6's gate passes):

```bash
git -C erickfmm.github.io add frankenstein-transformer/schema.yaml \
    frankenstein-transformer/schema/ \
    frankenstein-transformer/examples/README.md \
    frankenstein-transformer/examples/examples/es_arch_hybrid_nor_muon.yaml \
    frankenstein-transformer/paper/en/ frankenstein-transformer/paper/es/ \
    frankenstein-transformer/paper/bibliography/ \
    frankenstein-transformer/paper/specs/architecture.md \
    frankenstein-transformer/paper/en.pdf frankenstein-transformer/paper/es.pdf
git -C erickfmm.github.io commit -m "docs(site): sync NorMuon optimizer (arXiv:2510.05491)" -m "<body>"
git -C erickfmm.github.io push origin HEAD:main
```

---

## 8. Verification matrix

| Gate | Command | Expected result |
|---|---|---|
| Optimizer factory/base tests | `uv run --extra cpu --extra train python -m pytest tests/test_optimizer_factory.py tests/test_optimizer_base.py -v` | All pass, incl. the new `BuildOptimizerNorMuonTests` (13 methods) and updated `test_registry_contains_all_required` |
| Schema consistency | `uv run --extra cpu --extra train python -m pytest tests/test_schema_consistency.py -v` | Presets validate; `nor_muon` enum ↔ tables; exactly one `^nor_muon-` if/then; SBERT enum resolves |
| YAML examples | `uv run --extra cpu --extra train python -m pytest tests/test_yaml_examples.py -v` | New example loads via `load_training_config` |
| Full suite | `uv run --extra cpu --extra train python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings` | Green (76+ files; naming conventions, schema gates unaffected) |
| e2e smoke | `uv run --extra cpu --extra train python full_tests/run_e2e.py --only opt_nor_muon` | `opt_nor_muon` combo trains a few steps without `ValueError` from `OPTIMIZER_ALLOWED_KEYS` or the factory |
| PDF EN | `pdflatex`/`bibtex` sequence (§5) | `paper.pdf` builds; no `Fatal error`; `li_normuon_2025` resolves (no `Citation … undefined`) |
| PDF ES | same for `paper-es.pdf` | same |
| PDF artifacts | `git status --short docs/paper docs/paper-es` | Only intended `.tex`/`.bbl`/`.pdf` changes |
| Param count | `uv run --extra cpu --extra train python full_tests/param_count_check.py` | `… configs counted, 0 failures`; `param_truth.json` gains one config |
| Estimator | `node full_tests/param_estimate_check.mjs erickfmm.github.io/frankenstein-transformer/ft-param-estimator.js full_tests/param_truth.json --tol 0.01` | Exit 0, `ALL PASS` (≤ 1% drift) |
| State probes | `bash .opencode/command/sync-website-state.sh` | `schema-refs`, `example-yaml`, `cli-flags`, `paper-en-copy`, `paper-es-copy`, paper-figure probes all PASS |

---

## 9. Risks, assumptions and open questions

1. **ε placement (OCR ambiguity).** Algorithm 1's line 9 renders ambiguously in the
   arXiv HTML (`√V_t + ε` vs `√(V_t + ε)`); Algorithm 2 and the repo's own dion3
   `normuon` + ada_muon implementations put ε **outside** the root. Plan assumes
   `O/(√V + ε)`. At implementation time, check the PDF source (LaTeX) of arXiv:2510.05491
   (`https://arxiv.org/e-print/2510.05491`) and the reference implementation
   (`https://github.com/zichongli5/NorMuon.git`); if the paper's TeX says inside, either
   match it in both code and annex or (preferred, for consistency with the registry)
   keep outside and note the deviation in the annex — user's call.
2. **`momentum` vs `(1−β₁)` scaling** — mathematically equivalent through NS5's
   Frobenius normalization (§2); assumption to re-verify with a one-line numeric check
   inside `test_row_second_moment_matches_first_step_ema` at implementation time.
3. **Naming.** `nor_muon` deliberately avoids the bare `normuon` (dion3's suffix). No
   collision with `dion3-normuon` keys or any `li_normuon*` bib key.
4. **β₂ wiring assumption.** The factory always injects per-group `betas`/`eps`
   (`_build_param_groups_for_optimizer`, factory.py:196-198), so `group["betas"][1]`
   reaches the row EMA without new plumbing — same assumption muon2/dion3 already
   rely on; covered by `test_beta2_from_shared_betas_routes`.
5. **Task/mixer compatibility.** No special `training.task` rules: `nor_muon` is a
   standard gradient optimizer (unlike `eve`/`ea4llm`, no `gradient_accumulation_steps`
   restriction, SBERT-legal). Hybrid routing means mixer choice is irrelevant to
   correctness; the example uses the standard Muon-family hybrid pattern.
6. **Open question (user approval needed — pre-existing drift, do not fix silently):**
   the end-of-annex category longtables (EN line 1693 / ES line 1707) still list the
   geometry-oriented row as "Muon, Turbo-Muon, AF-Muon, AdaMuon, Dion3" and predate
   ROOT and Muon². This plan adds NorMuon to the row; completing ROOT/Muon² there is
   unrelated drift and needs explicit approval.
7. **Open question:** `configs/examples/examples.md` counts "31 schema-compliant YAML
   presets" (stale; the directory has 118 YAMLs and the last four optimizer examples
   never updated it). This plan leaves it untouched, matching the muon2/dion3/root/
   adam_pm precedent; updating it is out of scope unless the user asks.
8. **Open question (paper scope):** the paper's FSDP2 distributed Algorithm 2 (gather/
   scatter orthogonalization, round-robin assignment, shard-local row normalization)
   is not implemented — single-device only, like every other registry member. The
   annex mentions it in one sentence as the paper's deployment contribution.
9. **`param_truth.json`** changes (new config) — intentional, part of the main commit.
10. **Census sweep completeness.** The ten "31" occurrences (README ×3,
    architecture ×3, family sections EN/ES, summary-table intros EN/ES) were found by
    grepping `31`, `thirty-one`, `treinta y una`, `31 optimiz`, `31 familias` across
    README.md, docs/, docs/paper/, docs/paper-es/. Re-run those greps at implementation
    time in case docs drifted since planning.

---

## 10. Out of scope

Explicitly: **no** code, schema, preset, test, doc, PDF, mirror or git change was made
by the planning command. This document is the deliverable; everything above is
planned content for a future, explicitly requested implementation session. The
untracked `.opencode/plan/add-mixer-2305.05084.md` from a previous session was left
untouched.
