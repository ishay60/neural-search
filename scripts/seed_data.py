#!/usr/bin/env python3
"""Seed the database with sample data.

This script creates sample collections and documents for testing.
"""

import argparse
import json
import logging

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample data
SAMPLE_DOCUMENTS = [
    {
        "id": "ai-ml-intro",
        "content": "Artificial Intelligence (AI) is a broad field of computer science focused on creating intelligent machines that can perform tasks typically requiring human intelligence. Machine Learning (ML) is a subset of AI that enables systems to learn from data without being explicitly programmed.",
        "metadata": {"category": "ai", "difficulty": "beginner", "source": "textbook"},
    },
    {
        "id": "deep-learning-basics",
        "content": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers (hence 'deep') to model complex patterns in data. It has revolutionized fields like computer vision, natural language processing, and speech recognition.",
        "metadata": {"category": "ai", "difficulty": "intermediate", "source": "course"},
    },
    {
        "id": "neural-networks",
        "content": "Neural networks are computing systems inspired by biological neural networks in the brain. They consist of interconnected nodes (neurons) organized in layers that process information using connectionist approaches to computation.",
        "metadata": {"category": "ai", "difficulty": "intermediate", "source": "wiki"},
    },
    {
        "id": "nlp-overview",
        "content": "Natural Language Processing (NLP) is a field of AI that gives machines the ability to read, understand, and derive meaning from human languages. Common NLP tasks include sentiment analysis, named entity recognition, and machine translation.",
        "metadata": {"category": "nlp", "difficulty": "beginner", "source": "blog"},
    },
    {
        "id": "transformers-architecture",
        "content": "The Transformer architecture, introduced in the 'Attention is All You Need' paper, revolutionized NLP. It uses self-attention mechanisms to process sequences in parallel, enabling faster training and better handling of long-range dependencies.",
        "metadata": {"category": "nlp", "difficulty": "advanced", "source": "paper"},
    },
    {
        "id": "bert-model",
        "content": "BERT (Bidirectional Encoder Representations from Transformers) is a language model developed by Google. It uses bidirectional training of Transformers to better understand context, achieving state-of-the-art results on many NLP benchmarks.",
        "metadata": {"category": "nlp", "difficulty": "advanced", "source": "paper"},
    },
    {
        "id": "vector-embeddings",
        "content": "Vector embeddings are numerical representations of data (text, images, etc.) in high-dimensional space. Similar items are positioned closer together in this space, enabling semantic search and similarity comparisons.",
        "metadata": {"category": "search", "difficulty": "intermediate", "source": "docs"},
    },
    {
        "id": "semantic-search",
        "content": "Semantic search goes beyond keyword matching to understand the intent and contextual meaning of search queries. It uses embeddings and vector databases to find conceptually similar content rather than exact text matches.",
        "metadata": {"category": "search", "difficulty": "intermediate", "source": "blog"},
    },
    {
        "id": "vector-databases",
        "content": "Vector databases are specialized databases designed for storing and querying high-dimensional vectors efficiently. They use approximate nearest neighbor (ANN) algorithms to enable fast similarity search at scale.",
        "metadata": {"category": "search", "difficulty": "intermediate", "source": "docs"},
    },
    {
        "id": "rag-systems",
        "content": "Retrieval-Augmented Generation (RAG) combines retrieval systems with language models. It retrieves relevant documents for a query and uses them as context for the language model, improving accuracy and reducing hallucinations.",
        "metadata": {"category": "ai", "difficulty": "advanced", "source": "paper"},
    },
]


def seed_data(base_url: str, collection: str, documents: list[dict]) -> None:
    """Seed data into Neural Search."""
    client = httpx.Client(base_url=base_url, timeout=60.0)

    # Create collection
    logger.info(f"Creating collection: {collection}")
    response = client.post(
        f"/api/v1/collections",
        json={"name": collection},
    )
    if response.status_code == 201:
        logger.info("  Collection created")
    elif response.status_code == 409:
        logger.info("  Collection already exists")
    else:
        logger.error(f"  Failed: {response.json()}")
        return

    # Index documents
    logger.info(f"Indexing {len(documents)} documents...")
    response = client.post(
        f"/api/v1/documents",
        json={
            "collection": collection,
            "documents": documents,
        },
    )

    if response.status_code == 202:
        result = response.json()
        logger.info(f"  Indexed: {result['documents_count']} documents")
    else:
        logger.error(f"  Failed: {response.json()}")
        return

    # Verify with search
    logger.info("Verifying with search...")
    response = client.post(
        f"/api/v1/search",
        json={
            "query": "machine learning artificial intelligence",
            "collection": collection,
            "top_k": 3,
        },
    )

    if response.status_code == 200:
        results = response.json()
        logger.info(f"  Search found {results['total_found']} results")
        for r in results["results"]:
            logger.info(f"    - [{r['score']:.3f}] {r['id']}")
    else:
        logger.error(f"  Search failed: {response.json()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Seed sample data")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Neural Search API URL",
    )
    parser.add_argument(
        "--collection",
        default="knowledge-base",
        help="Collection name",
    )
    parser.add_argument(
        "--file",
        help="JSON file with custom documents",
    )
    args = parser.parse_args()

    documents = SAMPLE_DOCUMENTS
    if args.file:
        logger.info(f"Loading documents from {args.file}")
        with open(args.file) as f:
            documents = json.load(f)

    seed_data(args.url, args.collection, documents)
    logger.info("Done!")


if __name__ == "__main__":
    main()
