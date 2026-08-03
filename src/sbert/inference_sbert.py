#!/usr/bin/env python3
"""SBERT inference engine for TORMENTED-BERT-Frankenstein.

Provides :class:`SBERTInference` for computing sentence embeddings,
similarity scores, semantic search, clustering, and embedding
serialization using fine-tuned Sentence-BERT models.
"""

import os
import torch
import logging
import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Tuple
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
import json
try:
    from ..utils.device import SUPPORTED_DEVICE_CHOICES, resolve_torch_device
except ImportError:
    from utils.device import SUPPORTED_DEVICE_CHOICES, resolve_torch_device

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Result of a pairwise sentence similarity computation.

    Attributes:
        sentence1: First sentence text.
        sentence2: Second sentence text.
        similarity: Cosine similarity score in ``[0, 1]`` (normalized).
        distance: Cosine distance (``1 - similarity``).
    """

    sentence1: str
    sentence2: str
    similarity: float
    distance: float

    def to_dict(self):
        """Serialize the result to a plain dictionary."""
        return {
            'sentence1': self.sentence1,
            'sentence2': self.sentence2,
            'similarity': self.similarity,
            'distance': self.distance
        }


class SBERTInference:
    """Inference engine for fine-tuned SBERT models.

    Loads a :class:`SentenceTransformer` model and provides methods for
    encoding sentences to embeddings, computing pairwise similarity,
    semantic search over a corpus, clustering, and embedding serialization.

    Attributes:
        model_path: Path to the fine-tuned SBERT model directory.
        batch_size: Default batch size for encoding.
        device: Resolved PyTorch device string.
        model: Loaded :class:`SentenceTransformer` instance.
        max_seq_length: Maximum sequence length from the loaded model.
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """Initialize the SBERT inference engine.

        Args:
            model_path: Path to a fine-tuned SBERT model directory.
            device: Device string (``"cuda"``, ``"cpu"``, or ``None`` for
                auto-resolution).
            batch_size: Default batch size for encoding operations.
        """
        self.model_path = model_path
        self.batch_size = batch_size
        requested_device = device if device is not None else "auto"
        self.device = resolve_torch_device(requested_device)
        
        # Load model
        logger.info(f"Loading SBERT model from {model_path}")
        self.model = SentenceTransformer(model_path, device=self.device)
        logger.info(f"Model loaded on device: {self.device}")
        
        # Get model info
        self.max_seq_length = self.model.max_seq_length
        logger.info(f"Max sequence length: {self.max_seq_length}")
    
    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode sentences into embeddings.
        
        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size (uses default if None)
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length
        
        Returns:
            Embeddings array of shape (n_sentences, embedding_dim)
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        
        batch_size = batch_size or self.batch_size
        
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        return embeddings
    
    def compute_similarity(
        self,
        sentence1: Union[str, List[str]],
        sentence2: Union[str, List[str]],
        metric: str = "cosine"
    ) -> Union[float, np.ndarray]:
        """
        Compute similarity between sentence(s).
        
        Args:
            sentence1: First sentence(s)
            sentence2: Second sentence(s)
            metric: Similarity metric ('cosine' or 'dot')
        
        Returns:
            Similarity score(s) in range [0, 1] (or [-1, 1] for unnormalized)
        """
        # Encode sentences
        emb1 = self.encode(sentence1, normalize=(metric == "cosine"))
        emb2 = self.encode(sentence2, normalize=(metric == "cosine"))
        
        # Compute similarity
        if metric == "cosine":
            # For normalized embeddings, dot product = cosine similarity
            similarity = np.sum(emb1 * emb2, axis=1) if emb1.ndim > 1 else np.dot(emb1, emb2)
        elif metric == "dot":
            similarity = np.sum(emb1 * emb2, axis=1) if emb1.ndim > 1 else np.dot(emb1, emb2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarity
    
    def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find most similar sentences to query from candidates.
        
        Args:
            query: Query sentence
            candidates: List of candidate sentences
            top_k: Number of top results to return
        
        Returns:
            List of (sentence, similarity_score) tuples, sorted by similarity
        """
        # Encode query and candidates
        query_emb = self.encode(query)
        candidate_embs = self.encode(candidates, show_progress=True)
        
        # Compute similarities
        similarities = np.dot(candidate_embs, query_emb)
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (candidates[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def batch_compare(
        self,
        sentences1: List[str],
        sentences2: List[str]
    ) -> List[SimilarityResult]:
        """
        Batch comparison of sentence pairs.
        
        Args:
            sentences1: First sentences
            sentences2: Second sentences (must match length of sentences1)
        
        Returns:
            List of SimilarityResult objects
        """
        if len(sentences1) != len(sentences2):
            raise ValueError("sentences1 and sentences2 must have same length")
        
        # Encode both sets
        emb1 = self.encode(sentences1, show_progress=True)
        emb2 = self.encode(sentences2, show_progress=True)
        
        # Compute pairwise similarities
        similarities = np.sum(emb1 * emb2, axis=1)
        distances = 1 - similarities
        
        # Create results
        results = [
            SimilarityResult(
                sentence1=s1,
                sentence2=s2,
                similarity=float(sim),
                distance=float(dist)
            )
            for s1, s2, sim, dist in zip(sentences1, sentences2, similarities, distances)
        ]
        
        return results
    
    def semantic_search(
        self,
        queries: Union[str, List[str]],
        corpus: List[str],
        top_k: int = 5
    ) -> List[List[Tuple[int, float]]]:
        """
        Semantic search: find most similar corpus sentences for each query.
        
        Args:
            queries: Query sentence(s)
            corpus: Corpus of sentences to search
            top_k: Number of results per query
        
        Returns:
            List of results for each query, each result is (corpus_idx, score)
        """
        if isinstance(queries, str):
            queries = [queries]
        
        # Encode
        logger.info(f"Encoding {len(queries)} queries...")
        query_embs = self.encode(queries, show_progress=False)
        
        logger.info(f"Encoding corpus of {len(corpus)} sentences...")
        corpus_embs = self.encode(corpus, show_progress=True)
        
        # Compute similarities (queries x corpus)
        logger.info("Computing similarities...")
        similarities = np.dot(query_embs, corpus_embs.T)
        
        # Get top-k for each query
        results = []
        for query_sims in similarities:
            top_indices = np.argsort(query_sims)[::-1][:top_k]
            query_results = [
                (int(idx), float(query_sims[idx]))
                for idx in top_indices
            ]
            results.append(query_results)
        
        return results
    
    def cluster_sentences(
        self,
        sentences: List[str],
        n_clusters: int = 5,
        method: str = "kmeans"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster sentences by semantic similarity.

        Args:
            sentences: List of sentences to cluster
            n_clusters: Number of clusters
            method: Clustering method ('kmeans' or 'agglomerative')

        Returns:
            (cluster_labels, embeddings)
        """
        # Encode sentences
        logger.info(f"Encoding {len(sentences)} sentences for clustering...")
        embeddings = self.encode(sentences, show_progress=True)

        # Cluster
        logger.info(f"Clustering into {n_clusters} groups using {method}...")
        if method == "kmeans":
            labels = self._kmeans_clustering(embeddings, n_clusters=n_clusters)
        elif method == "agglomerative":
            labels = self._agglomerative_clustering(embeddings, n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown method: {method}")

        return labels, embeddings

    @staticmethod
    def _kmeans_clustering(embeddings: np.ndarray, n_clusters: int, max_iter: int = 300) -> np.ndarray:
        """PyTorch k-means clustering on sentence embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).
            n_clusters: Number of clusters.
            max_iter: Maximum number of Lloyd iterations.

        Returns:
            Cluster label array of shape (n_samples,).
        """
        n_samples = embeddings.shape[0]
        if n_samples <= n_clusters:
            return np.arange(n_samples)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.from_numpy(embeddings).to(device)

        # k-means++ initialization
        indices = [int(torch.randint(0, n_samples, (1,)).item())]
        for _ in range(1, n_clusters):
            dists = torch.cdist(x, x[indices])  # (n_samples, k)
            min_dists = dists.min(dim=1).values
            probs = min_dists / min_dists.sum()
            next_idx = int(torch.multinomial(probs, 1).item())
            indices.append(next_idx)
        centroids = x[indices].clone()

        for _ in range(max_iter):
            distances = torch.cdist(x, centroids)  # (n_samples, n_clusters)
            labels = distances.argmin(dim=1)
            new_centroids = torch.stack([
                x[labels == k].mean(dim=0) if (labels == k).any() else centroids[k]
                for k in range(n_clusters)
            ])
            if torch.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids

        return labels.cpu().numpy()

    @staticmethod
    def _agglomerative_clustering(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """PyTorch agglomerative (single-linkage) clustering on embeddings.

        This is a best-effort CPU/GPU implementation. It builds a full
        pairwise distance matrix and repeatedly merges the closest pair until
        the requested number of clusters is reached.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).
            n_clusters: Number of clusters.

        Returns:
            Cluster label array of shape (n_samples,).
        """
        n_samples = embeddings.shape[0]
        if n_samples <= n_clusters:
            return np.arange(n_samples)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.from_numpy(embeddings).to(device)

        # Full pairwise distance matrix
        dists = torch.cdist(x, x)  # (n_samples, n_samples)
        dists.fill_diagonal_(float('inf'))

        # Each sample starts as its own cluster; track active clusters.
        active = list(range(n_samples))
        cluster_of = list(range(n_samples))
        next_cluster_id = n_samples

        while len(active) > n_clusters:
            # Find minimum distance pair among active clusters.
            active_mask = torch.zeros(n_samples, dtype=torch.bool, device=device)
            for c in active:
                active_mask[c] = True
            sub_dists = dists[active_mask][:, active_mask]
            min_val, min_idx = sub_dists.view(-1).min(dim=0)
            size = len(active)
            i = int(min_idx // size)
            j = int(min_idx % size)
            if i == j:
                break
            ci = active[i]
            cj = active[j]

            # Merge cj into ci using single linkage (minimum of distances).
            merged = torch.minimum(dists[ci], dists[cj])
            dists[ci] = merged
            dists[:, ci] = merged
            dists[ci, ci] = float('inf')
            dists[cj] = float('inf')
            dists[:, cj] = float('inf')

            for idx, c in enumerate(cluster_of):
                if c == cj:
                    cluster_of[idx] = ci

            active.pop(j)
            next_cluster_id += 1

        # Remap active cluster ids to compact 0..n_clusters-1 labels.
        unique_active = sorted(set(cluster_of))
        remap = {c: i for i, c in enumerate(unique_active)}
        return np.array([remap[c] for c in cluster_of], dtype=np.int64)
    
    def save_embeddings(
        self,
        sentences: List[str],
        output_path: str,
        metadata: Optional[dict] = None
    ):
        """
        Encode sentences and save embeddings to file.
        
        Args:
            sentences: List of sentences
            output_path: Output file path (.npz format)
            metadata: Optional metadata to save
        """
        logger.info(f"Encoding {len(sentences)} sentences...")
        embeddings = self.encode(sentences, show_progress=True)
        
        # Prepare data
        data = {
            'embeddings': embeddings,
            'sentences': np.array(sentences, dtype=object)
        }
        
        if metadata:
            data['metadata'] = json.dumps(metadata)
        
        # Save
        np.savez_compressed(output_path, **data)
        logger.info(f"Embeddings saved to {output_path}")
    
    def load_embeddings(self, input_path: str) -> Tuple[np.ndarray, List[str], Optional[dict]]:
        """
        Load precomputed embeddings.
        
        Args:
            input_path: Path to .npz file
        
        Returns:
            (embeddings, sentences, metadata)
        """
        data = np.load(input_path, allow_pickle=True)
        
        embeddings = data['embeddings']
        sentences = data['sentences'].tolist()
        metadata = json.loads(data['metadata'].item()) if 'metadata' in data else None
        
        logger.info(f"Loaded {len(sentences)} embeddings from {input_path}")
        
        return embeddings, sentences, metadata


def main(argv=None):
    """Main inference script with CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SBERT Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["similarity", "search", "cluster", "encode"],
                       help="Inference mode")
    
    # Similarity mode
    parser.add_argument("--sentence1", type=str, help="First sentence (similarity mode)")
    parser.add_argument("--sentence2", type=str, help="Second sentence (similarity mode)")
    
    # Search mode
    parser.add_argument("--query", type=str, help="Query sentence (search mode)")
    parser.add_argument("--corpus_file", type=str, help="File with corpus sentences (one per line)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results (search mode)")
    
    # Cluster mode
    parser.add_argument("--sentences_file", type=str, help="File with sentences to cluster")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters")
    
    # Encode mode
    parser.add_argument("--input_file", type=str, help="Input sentences file")
    parser.add_argument("--output_file", type=str, help="Output embeddings file (.npz)")
    
    # General
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=SUPPORTED_DEVICE_CHOICES,
        help="Device to run SBERT inference on"
    )

    args = parser.parse_args(argv)
    resolved_device = resolve_torch_device(args.device)
    logger.info(f"SBERT infer device requested='{args.device}', resolved='{resolved_device}'")
    
    # Initialize inference engine
    inference = SBERTInference(
        model_path=args.model_path,
        device=resolved_device,
        batch_size=args.batch_size
    )
    
    # Execute based on mode
    if args.mode == "similarity":
        if not args.sentence1 or not args.sentence2:
            parser.error("--sentence1 and --sentence2 required for similarity mode")
        
        similarity = inference.compute_similarity(args.sentence1, args.sentence2)
        
        print(f"\nSentence 1: {args.sentence1}")
        print(f"Sentence 2: {args.sentence2}")
        print(f"Similarity: {similarity:.4f}")
    
    elif args.mode == "search":
        if not args.query or not args.corpus_file:
            parser.error("--query and --corpus_file required for search mode")
        
        # Load corpus
        with open(args.corpus_file, 'r', encoding='utf-8') as f:
            corpus = [line.strip() for line in f if line.strip()]
        
        print(f"\nSearching corpus of {len(corpus)} sentences...")
        results = inference.find_most_similar(args.query, corpus, top_k=args.top_k)
        
        print(f"\nQuery: {args.query}")
        print(f"\nTop {args.top_k} results:")
        for i, (sentence, score) in enumerate(results, 1):
            print(f"{i}. [{score:.4f}] {sentence}")
    
    elif args.mode == "cluster":
        if not args.sentences_file:
            parser.error("--sentences_file required for cluster mode")
        
        # Load sentences
        with open(args.sentences_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        labels, embeddings = inference.cluster_sentences(sentences, n_clusters=args.n_clusters)
        
        print(f"\nClustered {len(sentences)} sentences into {args.n_clusters} groups:")
        for cluster_id in range(args.n_clusters):
            cluster_sentences = [s for s, l in zip(sentences, labels) if l == cluster_id]
            print(f"\nCluster {cluster_id + 1} ({len(cluster_sentences)} sentences):")
            for sent in cluster_sentences[:5]:  # Show first 5
                print(f"  - {sent}")
            if len(cluster_sentences) > 5:
                print(f"  ... and {len(cluster_sentences) - 5} more")
    
    elif args.mode == "encode":
        if not args.input_file or not args.output_file:
            parser.error("--input_file and --output_file required for encode mode")
        
        # Load sentences
        with open(args.input_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        # Save embeddings
        inference.save_embeddings(sentences, args.output_file)


if __name__ == "__main__":
    main()
