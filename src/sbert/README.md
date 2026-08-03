# SBERT Fine-tuning for TORMENTED-BERT-Frankenstein v2

This directory contains scripts for fine-tuning and using Sentence-BERT (SBERT) models based on the TORMENTED-BERT-Frankenstein architecture.

## Overview

SBERT is a modification of BERT that uses siamese and triplet network structures to derive semantically meaningful sentence embeddings. These scripts adapt the TORMENTED-BERT v2 model for sentence similarity tasks using the Spanish multilingual sentences dataset.

### Dataset

- **Name**: `erickfmm/agentlans__multilingual-sentences__paired_10_sts`
- **Type**: Sentence Similarity (STS)
- **Language**: Spanish
- **Size**: ~2M sentence pairs
- **Task**: Semantic Textual Similarity with cosine scores

## Files

- **`train_sbert.py`**: Training script for fine-tuning SBERT models
- **`inference_sbert.py`**: Inference script for computing embeddings and similarities
- **`README.md`**: This documentation file

## Installation

### Requirements

```bash
pip install sentence-transformers>=2.2.0
pip install datasets
pip install torch
pip install scipy
pip install scikit-learn
```

### Dependencies from Project

The scripts depend on:
- `src/model/v2/tormented_bert_frankestein.py` - Base model
- `src/tokenizer/spm_spa_redpajama35.py` - Tokenizer (optional)

## Training

### Basic Training

Train from scratch with default configuration:

```bash
python train_sbert.py \
    --output_dir ./output/sbert_tormented_v2 \
    --batch_size 16 \
    --epochs 4
```

### Training from Pretrained Checkpoint

Fine-tune from a pretrained TORMENTED-BERT checkpoint:

```bash
python train_sbert.py \
    --pretrained /path/to/checkpoint.pth \
    --output_dir ./output/sbert_finetuned \
    --batch_size 16 \
    --epochs 4 \
    --learning_rate 2e-5
```

### Advanced Options

```bash
python train_sbert.py \
    --pretrained /path/to/checkpoint.pth \
    --output_dir ./output/sbert_custom \
    --batch_size 32 \
    --epochs 10 \
    --learning_rate 2e-5 \
    --hidden_size 768 \
    --num_layers 12 \
    --pooling_mode mean \
    --max_train_samples 500000 \
    --max_eval_samples 10000 \
    --no_resample  # Don't balance the dataset distribution
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pretrained` | None | Path to pretrained checkpoint |
| `--output_dir` | `./output/sbert_tormented_v2` | Output directory |
| `--batch_size` | 16 | Training batch size |
| `--epochs` | 4 | Number of training epochs |
| `--learning_rate` | 2e-5 | Learning rate |
| `--max_train_samples` | None | Max training samples (None = all) |
| `--max_eval_samples` | 10000 | Max evaluation samples |
| `--hidden_size` | 768 | Model hidden size |
| `--num_layers` | 12 | Number of transformer layers |
| `--pooling_mode` | mean | Pooling strategy (mean/cls/max) |
| `--no_amp` | False | Disable automatic mixed precision |
| `--no_resample` | False | Don't resample dataset for balance |

### Training Features

1. **Dataset Resampling**: Automatically resamples the dataset to create a balanced distribution of similarity scores (recommended by dataset authors)

2. **Automatic Mixed Precision**: Uses AMP for faster training and reduced memory usage (P40 compatible)

3. **Evaluation**: Computes Spearman correlation on validation set during training

4. **Model Saving**: Automatically saves the best model based on validation performance

## Inference

### Modes

The inference script supports multiple modes:

#### 1. Similarity Computation

Compare two sentences:

```bash
python inference_sbert.py \
    --model_path ./output/sbert_tormented_v2 \
    --mode similarity \
    --sentence1 "El gato está en la casa" \
    --sentence2 "Un felino se encuentra en el hogar"
```

#### 2. Semantic Search

Find most similar sentences in a corpus:

```bash
python inference_sbert.py \
    --model_path ./output/sbert_tormented_v2 \
    --mode search \
    --query "Machine learning en español" \
    --corpus_file ./data/corpus.txt \
    --top_k 10
```

The corpus file should contain one sentence per line.

#### 3. Sentence Clustering

Cluster sentences by semantic similarity:

```bash
python inference_sbert.py \
    --model_path ./output/sbert_tormented_v2 \
    --mode cluster \
    --sentences_file ./data/sentences.txt \
    --n_clusters 5
```

#### 4. Encode and Save

Encode sentences and save embeddings for later use:

```bash
python inference_sbert.py \
    --model_path ./output/sbert_tormented_v2 \
    --mode encode \
    --input_file ./data/sentences.txt \
    --output_file ./data/embeddings.npz
```

### Programmatic Usage

Use the inference engine in your Python code:

```python
from inference_sbert import SBERTInference

# Initialize
inference = SBERTInference(
    model_path="./output/sbert_tormented_v2",
    device="cuda",  # or "cpu"
    batch_size=32
)

# Compute similarity
similarity = inference.compute_similarity(
    "El perro corre en el parque",
    "Un canino juega afuera"
)
print(f"Similarity: {similarity:.4f}")

# Encode sentences
sentences = [
    "Primera oración",
    "Segunda oración",
    "Tercera oración"
]
embeddings = inference.encode(sentences)
print(f"Embeddings shape: {embeddings.shape}")

# Find most similar
query = "aprendizaje automático"
candidates = [
    "inteligencia artificial",
    "redes neuronales",
    "cocina mexicana",
    "música clásica"
]
results = inference.find_most_similar(query, candidates, top_k=2)
for sentence, score in results:
    print(f"{score:.4f}: {sentence}")

# Semantic search
corpus = ["sentence1", "sentence2", ...]  # Large corpus
results = inference.semantic_search(
    queries=["query1", "query2"],
    corpus=corpus,
    top_k=5
)

# Batch comparison
results = inference.batch_compare(
    sentences1=["sent1", "sent2"],
    sentences2=["sent_a", "sent_b"]
)
for result in results:
    print(f"{result.sentence1} <-> {result.sentence2}: {result.similarity:.4f}")
```

## Model Architecture

### Pooling Strategies

Three pooling strategies are available:

1. **Mean Pooling** (default): Average all token embeddings
2. **CLS Pooling**: Use only the [CLS] token embedding
3. **Max Pooling**: Take maximum over all token embeddings

Mean pooling typically works best for sentence embeddings.

### Loss Function

The model uses **Cosine Similarity Loss**, which trains the model to predict the cosine similarity between sentence pairs directly. This is ideal for regression-style STS tasks.

### Base Model Features

TORMENTED-BERT-Frankenstein includes:
- **BitNet b1.58**: Ternary weight quantization for memory efficiency
- **Hybrid Architecture**: RetNet, Neural ODE, Mamba, and Attention layers
- **Dynamic Tanh Normalization**: Improved training stability
- **Sparse MoE**: Mixture of Experts for capacity scaling
- **HoPE Attention**: Real multi-head attention with HoPE applied to Q/K
- **Factorized Embeddings**: Lower parameter count with optional Conv1d

## Performance Tips

### Training

1. **Batch Size**: Start with 16-32. Increase if you have more VRAM.
2. **Mixed Precision**: Keep AMP enabled unless you have numerical issues.
3. **Dataset Resampling**: Keep enabled for better performance (balanced distribution).
4. **Learning Rate**: 2e-5 is a good default. Try 1e-5 or 5e-5 if needed.

### Inference

1. **Batch Processing**: Process multiple sentences at once for efficiency.
2. **Normalization**: Keep normalization enabled for cosine similarity.
3. **Device**: Use CUDA for large-scale inference.
4. **Precompute**: Save embeddings for frequently used sentences.

## Hardware Requirements

### Training

- **GPU**: NVIDIA Tesla P40 24GB or similar
- **RAM**: 32GB+ recommended
- **Storage**: ~10GB for dataset + checkpoints

### Inference

- **GPU**: Optional but recommended for batch processing
- **RAM**: 8GB+ 
- **CPU**: Modern multi-core processor

## Evaluation Metrics

The model is evaluated using:

- **Spearman Correlation**: Measures monotonic relationship between predicted and true similarities
- **Pearson Correlation**: Measures linear correlation
- **MSE**: Mean squared error for similarity predictions

Higher correlation values (closer to 1.0) indicate better performance.

## Example Results

Expected performance on the validation set:

- **Spearman Correlation**: 0.75-0.85
- **Pearson Correlation**: 0.73-0.83

(Actual results depend on training configuration and data quality)

## Troubleshooting

### Out of Memory (OOM)

- Reduce `batch_size`
- Enable AMP if disabled
- Reduce `hidden_size` or `num_layers`
- Use gradient checkpointing (requires code modification)

### Slow Training

- Increase `batch_size` if possible
- Reduce `ode_steps` in model config
- Set `num_loops=1` in model config
- Use fewer `max_train_samples`

### Poor Performance

- Train for more epochs
- Disable `no_resample` to balance data
- Try different `pooling_mode`
- Increase model size (`hidden_size`, `num_layers`)
- Fine-tune from a pretrained checkpoint

### Import Errors

```bash
# Install missing dependencies
pip install sentence-transformers datasets scipy scikit-learn

# Make sure project root is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/transformer-encoder-frankestein"
```

## Advanced Usage

### Custom Dataset

To train on a different dataset:

1. Modify `load_dataset()` in `train_sbert.py`
2. Ensure dataset has columns: `sentence1`, `sentence2`, `score`
3. Adjust score normalization if needed

### Custom Model Configuration

Create custom UltraConfig:

```python
from model.v2.tormented_bert_frankestein import UltraConfig

config = UltraConfig(
    vocab_size=50000,
    hidden_size=1024,
    num_layers=16,
    num_loops=2,
    ode_steps=2,
    retention_heads=8,
    num_heads=16,
    use_bitnet=True,
    norm_type="derf",
    use_factorized_embedding=True,
    factorized_embedding_dim=128,
    use_embedding_conv=True,
    hope_base=10_000.0,
    hope_damping=0.01
)
```

### Mini Preset

El preset `TormentedBertMini` también es compatible con SBERT cuando necesitas un backbone más ligero.

### Integration with Sentence-Transformers Ecosystem

The fine-tuned model is fully compatible with the sentence-transformers library:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('./output/sbert_tormented_v2')
embeddings = model.encode(["My sentence"])
```

## Citation

If you use this implementation, please cite:

```bibtex
@misc{tormented-bert-sbert-2026,
  title={SBERT Fine-tuning for TORMENTED-BERT-Frankenstein},
  author={Your Name},
  year={2026},
  howpublished={\\url{https://github.com/yourusername/transformer-encoder-frankestein}}
}
```

## License

See the main project LICENSE file.

## References

1. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019.
2. Dataset: [erickfmm/agentlans__multilingual-sentences__paired_10_sts](https://huggingface.co/datasets/erickfmm/agentlans__multilingual-sentences__paired_10_sts)
3. Base Model: TORMENTED-BERT-Frankenstein v2
