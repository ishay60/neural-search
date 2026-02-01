#!/usr/bin/env python3
"""Benchmark search accuracy.

This script measures the quality of search results using
standard information retrieval metrics.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class AccuracyResult:
    """Results from accuracy benchmark."""

    search_type: str
    num_queries: int
    mrr: float  # Mean Reciprocal Rank
    recall_at_5: float
    recall_at_10: float
    ndcg_at_10: float


def calculate_mrr(rankings: list[list[int]], relevant: list[set[int]]) -> float:
    """Calculate Mean Reciprocal Rank."""
    reciprocal_ranks = []

    for ranking, rel_set in zip(rankings, relevant):
        for rank, doc_id in enumerate(ranking, 1):
            if doc_id in rel_set:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks)


def calculate_recall_at_k(
    rankings: list[list[int]],
    relevant: list[set[int]],
    k: int,
) -> float:
    """Calculate Recall@K."""
    recalls = []

    for ranking, rel_set in zip(rankings, relevant):
        top_k = set(ranking[:k])
        if rel_set:
            recall = len(top_k & rel_set) / len(rel_set)
            recalls.append(recall)

    return np.mean(recalls)


def calculate_ndcg_at_k(
    rankings: list[list[int]],
    relevance_scores: list[dict[int, float]],
    k: int,
) -> float:
    """Calculate NDCG@K."""
    ndcgs = []

    for ranking, rel_scores in zip(rankings, relevance_scores):
        # DCG
        dcg = 0.0
        for rank, doc_id in enumerate(ranking[:k], 1):
            rel = rel_scores.get(doc_id, 0.0)
            dcg += rel / np.log2(rank + 1)

        # Ideal DCG
        ideal_rels = sorted(rel_scores.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(ideal_rels, 1))

        if idcg > 0:
            ndcgs.append(dcg / idcg)
        else:
            ndcgs.append(0.0)

    return np.mean(ndcgs)


def create_test_dataset() -> tuple[list[str], list[str], list[set[int]], list[dict[int, float]]]:
    """Create a synthetic test dataset with queries and relevance judgments."""
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret images.",
        "Reinforcement learning trains agents through rewards.",
        "Supervised learning requires labeled training data.",
        "Unsupervised learning finds patterns in unlabeled data.",
        "Transfer learning reuses knowledge from one task to another.",
        "Neural networks are inspired by biological neurons.",
        "Backpropagation is used to train neural networks.",
    ]

    # Queries and relevance judgments
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain natural language processing",
        "What is deep learning?",
    ]

    # Relevant documents for each query (0-indexed)
    relevant_docs = [
        {0, 5, 6},  # ML query
        {1, 8, 9},  # Neural networks query
        {2},       # NLP query
        {1, 8},    # Deep learning query
    ]

    # Graded relevance scores
    relevance_scores = [
        {0: 3, 5: 2, 6: 2, 1: 1},  # ML
        {8: 3, 9: 3, 1: 2},        # Neural networks
        {2: 3},                    # NLP
        {1: 3, 8: 2},              # Deep learning
    ]

    return documents, queries, relevant_docs, relevance_scores


def simulate_search(
    query: str,
    documents: list[str],
    search_type: str = "dense",
) -> list[int]:
    """Simulate search results (in real usage, this would call the API)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode query and documents
    query_emb = model.encode([query])[0]
    doc_embs = model.encode(documents)

    # Calculate similarities
    similarities = np.dot(doc_embs, query_emb)

    # Return ranked document indices
    return list(np.argsort(similarities)[::-1])


def benchmark_search_type(
    search_type: str,
    documents: list[str],
    queries: list[str],
    relevant_docs: list[set[int]],
    relevance_scores: list[dict[int, float]],
) -> AccuracyResult:
    """Benchmark a search type."""
    print(f"\nBenchmarking {search_type} search...")

    rankings = []
    for query in queries:
        ranking = simulate_search(query, documents, search_type)
        rankings.append(ranking)

    # Calculate metrics
    mrr = calculate_mrr(rankings, relevant_docs)
    recall_5 = calculate_recall_at_k(rankings, relevant_docs, 5)
    recall_10 = calculate_recall_at_k(rankings, relevant_docs, 10)
    ndcg_10 = calculate_ndcg_at_k(rankings, relevance_scores, 10)

    result = AccuracyResult(
        search_type=search_type,
        num_queries=len(queries),
        mrr=mrr,
        recall_at_5=recall_5,
        recall_at_10=recall_10,
        ndcg_at_10=ndcg_10,
    )

    print(f"  MRR: {mrr:.4f}")
    print(f"  Recall@5: {recall_5:.4f}")
    print(f"  Recall@10: {recall_10:.4f}")
    print(f"  NDCG@10: {ndcg_10:.4f}")

    return result


def main():
    """Run search accuracy benchmarks."""
    print("=" * 60)
    print("Search Accuracy Benchmark")
    print("=" * 60)

    # Create test dataset
    documents, queries, relevant_docs, relevance_scores = create_test_dataset()

    print(f"\nDataset: {len(documents)} documents, {len(queries)} queries")

    # Benchmark different search types
    results = []

    # Dense search
    result = benchmark_search_type(
        "dense",
        documents,
        queries,
        relevant_docs,
        relevance_scores,
    )
    results.append(result)

    # Save results
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "search_accuracy.json"
    with open(output_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Search Type':<15} {'MRR':<10} {'R@5':<10} {'R@10':<10} {'NDCG@10':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r.search_type:<15} {r.mrr:<10.4f} {r.recall_at_5:<10.4f} {r.recall_at_10:<10.4f} {r.ndcg_at_10:<10.4f}")


if __name__ == "__main__":
    main()
