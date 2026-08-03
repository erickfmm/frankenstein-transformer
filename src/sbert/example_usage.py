#!/usr/bin/env python3
"""
Example usage of SBERT fine-tuned TORMENTED-BERT model.

This script demonstrates:
1. Loading a fine-tuned SBERT model
2. Computing sentence embeddings
3. Calculating similarity between sentences
4. Performing semantic search
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sbert.inference_sbert import SBERTInference


def example_basic_similarity():
    """Example: Basic sentence similarity"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Sentence Similarity")
    print("=" * 80)
    
    # Initialize model (replace with your model path)
    model_path = "./output/sbert_tormented_v2"
    inference = SBERTInference(model_path=model_path, batch_size=32)
    
    # Example sentence pairs
    pairs = [
        (
            "El perro corre en el parque",
            "Un canino juega en el área verde"
        ),
        (
            "La inteligencia artificial está revolucionando el mundo",
            "Machine learning está cambiando todo"
        ),
        (
            "Me gusta la pizza",
            "Las matemáticas son complejas"
        )
    ]
    
    print("\nComputing similarities...")
    for s1, s2 in pairs:
        similarity = inference.compute_similarity(s1, s2)
        print(f"\nSentence 1: {s1}")
        print(f"Sentence 2: {s2}")
        print(f"Similarity: {similarity:.4f}")


def example_batch_encoding():
    """Example: Batch encoding of sentences"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Batch Encoding")
    print("=" * 80)
    
    model_path = "./output/sbert_tormented_v2"
    inference = SBERTInference(model_path=model_path)
    
    sentences = [
        "La ciencia es fascinante",
        "El deporte es saludable",
        "La música me relaja",
        "La programación es creativa",
        "La naturaleza es hermosa"
    ]
    
    print(f"\nEncoding {len(sentences)} sentences...")
    embeddings = inference.encode(sentences, show_progress=True)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Each sentence is represented as a {embeddings.shape[1]}-dimensional vector")


def example_semantic_search():
    """Example: Semantic search in a corpus"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Semantic Search")
    print("=" * 80)
    
    model_path = "./output/sbert_tormented_v2"
    inference = SBERTInference(model_path=model_path)
    
    # Define a corpus
    corpus = [
        "Python es un lenguaje de programación popular",
        "El machine learning requiere grandes datasets",
        "Las redes neuronales imitan el cerebro humano",
        "La pizza italiana es deliciosa",
        "El fútbol es el deporte más popular",
        "La música clásica de Mozart es atemporal",
        "El clima está cambiando rápidamente",
        "Los coches eléctricos son el futuro",
        "La inteligencia artificial transformará la sociedad",
        "Los pandas son animales en peligro de extinción"
    ]
    
    # Query
    query = "¿Qué es el deep learning?"
    
    print(f"\nQuery: {query}")
    print(f"Searching in corpus of {len(corpus)} sentences...\n")
    
    # Find most similar
    results = inference.find_most_similar(query, corpus, top_k=3)
    
    print("Top 3 most similar sentences:")
    for i, (sentence, score) in enumerate(results, 1):
        print(f"{i}. [Score: {score:.4f}] {sentence}")


def example_batch_comparison():
    """Example: Batch comparison of sentence pairs"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Batch Comparison")
    print("=" * 80)
    
    model_path = "./output/sbert_tormented_v2"
    inference = SBERTInference(model_path=model_path)
    
    sentences1 = [
        "El gato duerme en el sofá",
        "La computadora está encendida",
        "El sol brilla en el cielo"
    ]
    
    sentences2 = [
        "Un felino descansa en el mueble",
        "La laptop funciona correctamente",
        "Las estrellas iluminan la noche"
    ]
    
    print("\nComparing sentence pairs...")
    results = inference.batch_compare(sentences1, sentences2)
    
    print("\nResults:")
    for result in results:
        print(f"\nPair:")
        print(f"  1: {result.sentence1}")
        print(f"  2: {result.sentence2}")
        print(f"  Similarity: {result.similarity:.4f}")
        print(f"  Distance: {result.distance:.4f}")


def example_clustering():
    """Example: Cluster sentences by semantic similarity"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Sentence Clustering")
    print("=" * 80)
    
    model_path = "./output/sbert_tormented_v2"
    inference = SBERTInference(model_path=model_path)
    
    sentences = [
        # Tech cluster
        "Python es un lenguaje de programación",
        "JavaScript se usa en desarrollo web",
        "Los algoritmos son fundamentales en informática",
        
        # Sports cluster
        "El fútbol es popular en España",
        "El baloncesto requiere altura",
        "El tenis es un deporte individual",
        
        # Food cluster
        "La paella es un plato español",
        "La pasta italiana es deliciosa",
        "El sushi viene de Japón",
        
        # Music cluster
        "Beethoven fue un gran compositor",
        "El jazz nació en Estados Unidos",
        "La guitarra es versátil"
    ]
    
    print(f"\nClustering {len(sentences)} sentences into 4 groups...")
    labels, embeddings = inference.cluster_sentences(
        sentences,
        n_clusters=4,
        method="kmeans"
    )
    
    print("\nClustering results:")
    for cluster_id in range(4):
        cluster_sentences = [s for s, l in zip(sentences, labels) if l == cluster_id]
        print(f"\nCluster {cluster_id + 1} ({len(cluster_sentences)} sentences):")
        for sent in cluster_sentences:
            print(f"  - {sent}")


def example_save_load_embeddings():
    """Example: Save and load embeddings"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Save and Load Embeddings")
    print("=" * 80)
    
    model_path = "./output/sbert_tormented_v2"
    inference = SBERTInference(model_path=model_path)
    
    sentences = [
        "Primera oración de ejemplo",
        "Segunda oración de ejemplo",
        "Tercera oración de ejemplo"
    ]
    
    output_file = "./embeddings_example.npz"
    
    # Save embeddings
    print(f"\nSaving embeddings to {output_file}...")
    inference.save_embeddings(
        sentences,
        output_file,
        metadata={'created_by': 'example_script', 'version': '1.0'}
    )
    
    # Load embeddings
    print(f"\nLoading embeddings from {output_file}...")
    embeddings, loaded_sentences, metadata = inference.load_embeddings(output_file)
    
    print(f"\nLoaded:")
    print(f"  - {len(loaded_sentences)} sentences")
    print(f"  - Embeddings shape: {embeddings.shape}")
    print(f"  - Metadata: {metadata}")
    
    # Clean up
    import os
    os.remove(output_file)
    print(f"\nCleaned up temporary file: {output_file}")


def main():
    """Run all examples"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SBERT Examples")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output/sbert_tormented_v2",
        help="Path to fine-tuned SBERT model"
    )
    parser.add_argument(
        "--example",
        type=str,
        choices=["all", "similarity", "encoding", "search", "comparison", "clustering", "save_load"],
        default="all",
        help="Which example to run"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("SBERT TORMENTED-BERT-Frankenstein Examples")
    print("=" * 80)
    print(f"\nModel path: {args.model_path}")
    print(f"Example: {args.example}")
    
    # Update model path globally (for simplicity in examples)
    # In production, pass model_path as parameter
    
    examples = {
        "similarity": example_basic_similarity,
        "encoding": example_batch_encoding,
        "search": example_semantic_search,
        "comparison": example_batch_comparison,
        "clustering": example_clustering,
        "save_load": example_save_load_embeddings
    }
    
    try:
        if args.example == "all":
            for name, func in examples.items():
                func()
        else:
            examples[args.example]()
        
        print("\n" + "=" * 80)
        print("Examples completed successfully!")
        print("=" * 80 + "\n")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Model not found at {args.model_path}")
        print("Please train a model first using train_sbert.py")
        print("Or specify the correct path with --model_path")
    except Exception as e:
        print(f"\n❌ Error running example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
