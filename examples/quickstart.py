#!/usr/bin/env python3
"""Quickstart example for Neural Search API.

This example demonstrates basic usage of the Neural Search API,
including creating collections, indexing documents, and searching.
"""

import httpx

# API base URL
BASE_URL = "http://localhost:8000/api/v1"


def main():
    """Run the quickstart example."""
    client = httpx.Client(base_url=BASE_URL, timeout=60.0)

    # 1. Create a collection
    print("Creating collection...")
    response = client.post(
        "/collections",
        json={
            "name": "quickstart-demo",
            "distance_metric": "cosine",
        },
    )
    if response.status_code == 201:
        print(f"  Created: {response.json()}")
    elif response.status_code == 409:
        print("  Collection already exists")
    else:
        print(f"  Error: {response.json()}")

    # 2. Index some documents
    print("\nIndexing documents...")
    documents = [
        {
            "id": "ml-intro",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "metadata": {"category": "ml", "difficulty": "beginner"},
        },
        {
            "id": "deep-learning",
            "content": "Deep learning is a type of machine learning that uses neural networks with multiple layers to progressively extract higher-level features from raw input.",
            "metadata": {"category": "ml", "difficulty": "intermediate"},
        },
        {
            "id": "nlp-basics",
            "content": "Natural Language Processing (NLP) is a field of AI that gives machines the ability to read, understand, and derive meaning from human languages.",
            "metadata": {"category": "nlp", "difficulty": "beginner"},
        },
        {
            "id": "transformers",
            "content": "Transformers are a type of neural network architecture that uses self-attention mechanisms to process sequential data, revolutionizing NLP tasks.",
            "metadata": {"category": "nlp", "difficulty": "advanced"},
        },
        {
            "id": "vector-search",
            "content": "Vector search uses embeddings to find semantically similar content by measuring distances in high-dimensional vector spaces.",
            "metadata": {"category": "search", "difficulty": "intermediate"},
        },
    ]

    response = client.post(
        "/documents",
        json={
            "collection": "quickstart-demo",
            "documents": documents,
        },
    )
    print(f"  Indexed: {response.json()}")

    # 3. Basic search
    print("\nSearching for 'neural networks and AI'...")
    response = client.post(
        "/search",
        json={
            "query": "neural networks and AI",
            "collection": "quickstart-demo",
            "top_k": 3,
        },
    )
    results = response.json()
    print(f"  Found {results['total_found']} results in {results['took_ms']:.2f}ms:")
    for r in results["results"]:
        print(f"    - [{r['score']:.3f}] {r['id']}: {r['content'][:80]}...")

    # 4. Search with filters
    print("\nSearching for 'learning' with filter category=ml...")
    response = client.post(
        "/search",
        json={
            "query": "learning",
            "collection": "quickstart-demo",
            "top_k": 3,
            "filters": {"category": "ml"},
        },
    )
    results = response.json()
    print(f"  Found {results['total_found']} results:")
    for r in results["results"]:
        print(f"    - [{r['score']:.3f}] {r['id']}")

    # 5. Hybrid search
    print("\nHybrid search for 'embeddings similarity'...")
    response = client.post(
        "/search",
        json={
            "query": "embeddings similarity",
            "collection": "quickstart-demo",
            "top_k": 3,
            "hybrid": True,
        },
    )
    results = response.json()
    print(f"  Found {results['total_found']} results in {results['took_ms']:.2f}ms:")
    for r in results["results"]:
        print(f"    - [{r['score']:.3f}] {r['id']}")

    # 6. Get collection stats
    print("\nCollection stats:")
    response = client.get("/collections/quickstart-demo")
    stats = response.json()
    print(f"  Name: {stats['name']}")
    print(f"  Documents: {stats['count']}")
    print(f"  Dimension: {stats['dimension']}")

    # 7. Cleanup (optional)
    # print("\nCleaning up...")
    # response = client.delete("/collections/quickstart-demo")
    # print("  Collection deleted")

    print("\nDone!")


if __name__ == "__main__":
    main()
