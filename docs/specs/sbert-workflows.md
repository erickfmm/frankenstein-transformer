# SBERT Training and Inference Specification

> Cross-references: [Schema Reference](schema-reference.md) · [CLI Reference](cli-reference.md) · [Architecture](architecture.md)

## Overview

Sentence embedding workflows are built on Siamese-style training inspired by SBERT (Reimers & Gurevych, 2019 — arXiv:1908.10084). A shared encoder produces embeddings for sentence pairs, which are compared via cosine similarity.

## Siamese Training

### Cosine Similarity Loss

For sentence pair `(s₁, s₂)` with embeddings `(e₁, e₂)`:

```
cos(e₁, e₂) = e₁^⊤ e₂ / (‖e₁‖ · ‖e₂‖)
L_cos = (cos(e₁, e₂) − y)²
```

where `y ∈ [−1, 1]` is the ground-truth similarity score. This is a regression-style cosine loss.

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

## CLI Examples

```bash
# Train SBERT from a pretrained base model
frankestein-transformer sbert-train \
  --base-model answerdotai/ModernBERT-base \
  --dataset_name erickfmm/agentlans__multilingual-sentences__paired_10_sts \
  --pooling_mode mean --epochs 4 --batch_size 16

# Train SBERT from a frankestein checkpoint
frankestein-transformer sbert-train \
  --pretrained checkpoints/model.pt \
  --hidden_size 768 --num_layers 12 \
  --pooling_mode cls

# Pairwise similarity
frankestein-transformer sbert-infer \
  --model_path ./output/sbert --mode similarity \
  --sentence1 "Machine learning is fascinating" \
  --sentence2 "AI research is exciting"

# Semantic search
frankestein-transformer sbert-infer \
  --model_path ./output/sbert --mode search \
  --query "transformer architecture" \
  --corpus_file papers.txt --top_k 10

# Clustering
frankestein-transformer sbert-infer \
  --model_path ./output/sbert --mode cluster \
  --sentences_file reviews.txt --n_clusters 5

# Embedding export
frankestein-transformer sbert-infer \
  --model_path ./output/sbert --mode encode \
  --input_file documents.txt --output_file embeddings.npy
```
