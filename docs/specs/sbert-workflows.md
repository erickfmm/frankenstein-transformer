# SBERT Training and Inference Specification

> Cross-references: [Schema Reference](schema-reference.md) · [CLI Reference](cli-reference.md) · [Architecture](architecture.md)

## Overview

Sentence embedding workflows are built on Siamese-style training inspired by SBERT (Reimers & Gurevych, 2019 — arXiv:1908.10084). A shared encoder produces embeddings for sentence pairs, which are compared via cosine similarity.

## Siamese Training

Sentence embeddings are learned so that similar sentences map to nearby
points and dissimilar ones map far apart. All three dataset types feed a
Siamese (shared-encoder) setup.

### Cosine Similarity Loss

For sentence pair `(s₁, s₂)` with embeddings `(e₁, e₂)`:

```
cos(e₁, e₂) = e₁^⊤ e₂ / (‖e₁‖ · ‖e₂‖)
L_cos = (cos(e₁, e₂) − y)²
```

where `y ∈ [−1, 1]` is the ground-truth similarity score. This is a regression-style cosine loss, used for `paired_similarity` data.

### Triplet Loss

For a triplet `(anchor a, positive p, negative n)`, the goal is to pull the
anchor closer to the positive than to the negative by a margin `m`:

```
L_triplet = max(0, cos(a, n) − cos(a, p) + m)
```

Here `m` is the margin (a small positive constant, e.g. `0.3`). This is a
contrastive objective used for `triplets` data — it only pushes on the
hardest cases (when the margin is violated).

### QA (contrastive) Objective

For a `(question, answer)` pair, the objective encourages the question
embedding to be close to the correct answer embedding and far from other
answers in the batch:

```
L_qa = − log( exp(cos(q, a⁺)/τ) / Σ_{a∈batch} exp(cos(q, a)/τ) )
```

where `τ` is a temperature. This is standard in dense-retrieval training and
is used for `qa` data.

## Pooling Modes

| Mode | Description |
|---|---|
| `mean` | Average pooling over all token embeddings (recommended) |
| `cls` | Use the [CLS] token embedding only |
| `max` | Max pooling over the time dimension |

## Dataset Types

| Type | Format | Description |
|---|---|---|
| `paired_similarity` | `(s₁, s₂, score)` | Sentence pairs with similarity scores for cosine regression |
| `triplets` | `(anchor, positive, negative)` | Triplet loss for contrastive learning |
| `qa` | `(question, answer)` | Question-answer pairs for semantic search training |

## SBERT Configuration Block

When `training.task = sbert`, the `training.sbert` subsection is required:

| Field | Type | Default | Description |
|---|---|---|---|
| `epochs` | int | — | Training epochs |
| `batch_size` | int | — | Training batch size |
| `learning_rate` | float | — | Learning rate |
| `warmup_steps` | int | — | LR warmup steps |
| `evaluation_steps` | int | — | Evaluation frequency (steps) |
| `max_seq_length` | int | — | Max sequence length |
| `pooling_mode` | enum | `mean` | `mean`, `cls`, `max` |
| `dataset_name` | string | — | HuggingFace dataset identifier |
| `dataset_type` | enum | — | `paired_similarity`, `triplets`, `qa` |
| `max_train_samples` | int | — | Max training samples |
| `max_eval_samples` | int | — | Max evaluation samples |
| `output_dir` | string | — | Model output directory |

## Inference Modes

```
Shared encoder
├── Similarity → cosine score between two sentences
├── Search → top-k nearest neighbors over a corpus
├── Cluster → grouping embeddings (k-means)
└── Encode → persistent embedding export
```

### `similarity` — Pairwise Scoring

Computes cosine similarity between two input sentences.

```
Input: sentence1, sentence2
Output: cosine similarity score ∈ [−1, 1]
```

### `search` — Top-k Retrieval

Encodes a query and ranks corpus sentences by cosine similarity.

```
Input: query, corpus_file
Output: top-k results with scores
Parameters: --top_k (default 5)
```

### `cluster` — Embedding Clustering

Encodes all sentences and applies k-means clustering.

```
Input: sentences_file
Output: cluster assignments
Parameters: --n_clusters (default 5)
```

### `encode` — Embedding Export

Encodes sentences and serializes embeddings to disk.

```
Input: input_file
Output: output_file (NumPy .npy format)
```

## SBERT Inference Mode Router (Pseudocode)

```
if mode == similarity:
    return cos(E(x₁), E(x₂))
elif mode == search:
    return top-k by dot-product/cosine against corpus embeddings
elif mode == cluster:
    return clustering labels over E(X)
else:  # encode
    return serialized embeddings E(X)
```

## Evaluation

Because SBERT is about *relative* distances rather than exact outputs, it is
normally evaluated on **semantic textual similarity (STS)** benchmarks:

| Metric | What it measures | Where to look |
|---|---|---|
| Spearman rank correlation | Whether the model's cosine scores rank sentence pairs the same way humans do | STS-B, STS12–16 |
| Recall@k (retrieval) | Fraction of queries whose correct answer appears in the top-k | `search` mode against a corpus |
| Clustering metrics (ARI/NMI) | How well the embeddings group into true clusters | `cluster` mode on labeled sets |

For a quick internal check without external benchmarks, run the `similarity`
mode on a handful of hand-labeled pairs and confirm that semantically related
sentences score noticeably higher than unrelated ones.

## CLI Examples

```bash
# Train SBERT from a pretrained base model
frankenstein-transformer sbert-train \
  --base-model answerdotai/ModernBERT-base \
  --dataset_name erickfmm/agentlans__multilingual-sentences__paired_10_sts \
  --pooling_mode mean --epochs 4 --batch_size 16

# Train SBERT from a frankenstein checkpoint
frankenstein-transformer sbert-train \
  --pretrained checkpoints/model.pt \
  --hidden_size 768 --num_layers 12 \
  --pooling_mode cls

# Pairwise similarity
frankenstein-transformer sbert-infer \
  --model_path ./output/sbert --mode similarity \
  --sentence1 "Machine learning is fascinating" \
  --sentence2 "AI research is exciting"

# Semantic search
frankenstein-transformer sbert-infer \
  --model_path ./output/sbert --mode search \
  --query "transformer architecture" \
  --corpus_file papers.txt --top_k 10

# Clustering
frankenstein-transformer sbert-infer \
  --model_path ./output/sbert --mode cluster \
  --sentences_file reviews.txt --n_clusters 5

# Embedding export
frankenstein-transformer sbert-infer \
  --model_path ./output/sbert --mode encode \
  --input_file documents.txt --output_file embeddings.npy
```
