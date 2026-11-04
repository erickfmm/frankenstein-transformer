# Plan: Paper Cleanup (English + Spanish)

## Scope
Modify both `docs/paper/` and `docs/paper-es/` in parallel. Every change to an English file must be mirrored in the corresponding Spanish file.

## Files to modify
- `paper.tex` / `paper-es.tex` (main files — abstract, appendix ordering)
- `sections/01-introduction.tex` / `01-introduccion.tex` (abstract + section 1.4)
- `sections/02-related-work.tex` / `02-trabajo-relacionado.tex` (complete rewrite)
- `sections/03-system-design.tex` / `03-diseno-arquitectura.tex` (remove mixer detail, remove inventory tables, remove optimizer prefix contract, fix Figure 1)
- `sections/04-architecture-taxonomy.tex` / `04-taxonomia-arquitectura.tex` (complete Figure 2, add new figures)
- `sections/05-optimizer-families.tex` / `05-familias-optimizadores.tex` (verify all optimizers)
- `sections/08-summary-tables.tex` / `08-tablas-resumen.tex` (massive expansion with 4 tables)
- `appendices/annex-1-optimizers.tex` / `annex-1-optimizadores.tex` (remove Table 5)
- `appendices/annex-2-recurrent-attention.tex` (remove last table)
- `appendices/annex-3-latent-attention.tex` (remove last table)
- `appendices/annex-4-sparse-attention.tex` (remove last table)
- `appendices/annex-5-gated-attention.tex` (remove last 3 tables)
- `appendices/annex-6-tutorial.tex` (move to first appendix)
- `appendices/annex-7-norm-bitnet-mod.tex` (keep as-is)
- **NEW**: `appendices/annex-schema.tex` (new appendix with schema structure)
- **NEW**: `appendices/annex-summary-tables.tex` (moved from section 8)

## Task Breakdown

### Phase 1: Research (read-only)
1. Search Tavily + arXiv for tools: Dash AI, Oumi, Llama Factory, Unsloth, PyTorch, TensorFlow, JAX, scikit-learn references
2. Verify all mixers in `src/model/attention/__init__.py` match what's in the paper
3. Verify all optimizers in `src/model/optimizer/` match what's in the paper
4. Verify all norms in `src/model/norm/` match what's in the paper
5. Verify all embeddings in `src/model/embeddings/` match what's in the paper

### Phase 2: Abstract (both languages)
- Rewrite abstract to ≤350 words, single paragraph
- Structure: global context → advances → related work → "we propose..."

### Phase 3: Section 1.4 (both languages)
- Remove `st.checkbox()`, `st.number_input()`, etc. implementation details
- Keep it short and succinct

### Phase 4: Section 2 (both languages)
- Complete rewrite focused on tools for configuring/training transformers
- Include: PyTorch, TensorFlow, JAX, scikit-learn as base frameworks
- Include: Dash AI, Oumi, Llama Factory as out-of-the-box tools
- Include: Unsloth and others as low-code tools
- Add proper references from Tavily/arXiv search

### Phase 5: Section 3 (both languages)
- Remove all mixer detail from layer_pattern (lines 33-57 in English)
- Replace with bold "Sequence Mixer Pattern Selection" + brief description
- Remove sections 3.3 (Model Feature Inventory table) and 3.5 (Training Feature Inventory table)
- Add link to new appendix for schema structure
- Remove section 3.6 (Optimizer Prefix Contract) and link to appendix
- Fix Figure 1: remove `finetune`, add `transformers-export`, `bitnet-gguf`

### Phase 6: Section 4 (both languages)
- Complete Figure 2 taxonomy with ALL mixers from `__init__.py`:
  - Dense: standard_attn, sigmoid_attn, gqa
  - Recurrent: retnet, retnet_attn, mamba, ode, titan_attn
  - Sparse: sparse_transformer_attn, longformer_attn, bigbird_attn, sparsek_attn, nsa_attn, sparge_attn, fasa_attn, msa_attn, sparda_attn
  - Gated: gla_attn, deltanet_attn, gated_deltanet_attn, gated_deltanet2_attn, hgrn2_attn, fox_attn, gated_softmax_attn, kda_attn
  - Latent: mla_attn, gqla_attn, mlra_attn, tucker_attn, iha_attn, gta_attn, mtla_attn, cca_attn, ccgqa_attn
  - Memory: engram_attn
- Add new figures for dense, recurrent, and latent attention families
- Update Figures 3 and 4 with current mixer info

### Phase 7: Section 5 (both languages)
- Verify all 23 optimizers are listed
- Update Figure to reflect all optimizers correctly

### Phase 8: Section 8 → New Appendix (both languages)
- Move section 8 content to new appendix `annex-summary-tables.tex`
- Create 4 comprehensive tables:
  1. **Sequence Mixers** (all ~30 mixers with references)
  2. **Optimizers** (all 23 with memory/complexity from old Table 5)
  3. **Normalization** (5 variants with references)
  4. **Embeddings** (RoPE, HoPE, Factorized, Conv)
- Multi-line cells, width-adjusted

### Phase 9: Appendices restructuring
- Remove Table 5 from annex-1 (optimizers) — data goes to Section 8 tables
- Remove last table from annex-2 (dense/recurrent) — data goes to Section 8 tables
- Remove last table from annex-3 (latent) — data goes to Section 8 tables
- Remove last table from annex-4 (sparse) — data goes to Section 8 tables
- Remove last 3 tables from annex-5 (gated) — data goes to Section 8 tables

### Phase 10: Appendix reordering (both main files)
New order:
1. annex-6-tutorial (conceptual intro) → renamed to annex-1
2. annex-schema (NEW — schema structure)
3. annex-summary-tables (moved from section 8)
4. annex-1-optimizers (was annex-1, now annex-4)
5. annex-2-recurrent-attention (was annex-2, now annex-5)
6. annex-3-latent-attention (was annex-3, now annex-6)
7. annex-4-sparse-attention (was annex-4, now annex-7)
8. annex-5-gated-attention (was annex-5, now annex-8)
9. annex-7-norm-bitnet-mod (was annex-7, now annex-9)

### Phase 11: Cross-reference updates
- Update all `\ref{}` labels throughout both papers
- Update all appendix references in sections

## Questions / Ambiguities

1. **"finetune" subcommand**: The paper mentions `finetune` as a CLI subcommand but `src/cli.py` doesn't have it. Should I remove it from the paper? → **Yes, remove it.**

2. **Mixer count**: The paper says "17 variants" but the actual code has ~30+ (including latent family, MSA, SparDA, KDA, Engram, GQA, CCA, CCGQA). Should I update the count? → **Yes, update to actual count.**

3. **Optimizer count**: Paper says 22-23. Code has 23 implementations. Should verify exact count.

4. **Section 8 tables**: The user wants 4 tables (mixers, optimizers, norms, embeddings). The old Section 8 only had 2 tables (mixers + optimizers). Need to create norms and embeddings tables from scratch.

5. **New appendix for schema**: Need to create from `src/schema.yaml` structure. Should I include the full schema or a summary? → **Summary with key fields and types.**

6. **Both languages**: Every change must be mirrored. The Spanish paper has identical structure but translated content. I'll process both in parallel.

7. **Figure 1 update**: Current figure shows "17 mixers" and "22 families". Need to update counts and add missing subcommands.

8. **The `retnet` vs `retnet_attn` distinction**: In the code, `retnet` and `retnet_attn` are separate entries. The gated family has `retnet_attn` (RetNetAttention in gated/). Need to clarify this in the taxonomy.
