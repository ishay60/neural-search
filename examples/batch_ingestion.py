#!/usr/bin/env python3
"""Batch ingestion example for Neural Search API.

This example demonstrates how to efficiently ingest large numbers
of documents using async processing.
"""

import json
import time

import httpx

# API base URL
BASE_URL = "http://localhost:8000/api/v1"


def generate_sample_documents(n: int) -> list[dict]:
    """Generate sample documents for testing."""
    categories = ["technology", "science", "business", "health", "sports"]
    documents = []

    for i in range(n):
        documents.append({
            "id": f"doc-{i:05d}",
            "content": f"This is sample document number {i}. It contains information about {categories[i % len(categories)]} and various related topics that would be useful for semantic search testing.",
            "metadata": {
                "category": categories[i % len(categories)],
                "index": i,
                "batch": i // 100,
            },
        })

    return documents


def wait_for_job(client: httpx.Client, job_id: str, timeout: int = 300) -> dict:
    """Wait for an async job to complete."""
    start = time.time()
    while time.time() - start < timeout:
        response = client.get(f"/documents/jobs/{job_id}")
        status = response.json()

        if status["status"] == "completed":
            return status
        elif status["status"] == "failed":
            raise Exception(f"Job failed: {status.get('error')}")

        progress = status.get("progress", 0) or 0
        print(f"  Progress: {progress * 100:.0f}%")
        time.sleep(2)

    raise TimeoutError("Job timed out")


def main():
    """Run the batch ingestion example."""
    client = httpx.Client(base_url=BASE_URL, timeout=60.0)

    # Create collection
    print("Creating collection...")
    response = client.post(
        "/collections",
        json={"name": "batch-demo"},
    )
    if response.status_code == 201:
        print("  Created collection")
    elif response.status_code == 409:
        print("  Collection already exists")

    # Method 1: Synchronous batch ingestion (smaller batches)
    print("\nMethod 1: Synchronous ingestion (500 docs in batches of 100)...")
    documents = generate_sample_documents(500)

    start = time.time()
    for i in range(0, len(documents), 100):
        batch = documents[i:i + 100]
        response = client.post(
            "/documents",
            json={
                "collection": "batch-demo",
                "documents": batch,
            },
        )
        print(f"  Batch {i // 100 + 1}: {response.json()['documents_count']} docs")

    elapsed = time.time() - start
    print(f"  Total time: {elapsed:.2f}s ({len(documents) / elapsed:.0f} docs/s)")

    # Method 2: Async ingestion (for larger batches)
    print("\nMethod 2: Async ingestion (500 docs)...")
    documents = generate_sample_documents(500)

    start = time.time()
    response = client.post(
        "/documents?async_mode=true",
        json={
            "collection": "batch-demo",
            "documents": documents,
        },
    )
    result = response.json()
    print(f"  Job ID: {result['job_id']}")
    print("  Waiting for completion...")

    try:
        status = wait_for_job(client, result["job_id"])
        elapsed = time.time() - start
        print(f"  Completed in {elapsed:.2f}s")
    except Exception as e:
        print(f"  Error: {e}")

    # Check final stats
    print("\nFinal collection stats:")
    response = client.get("/collections/batch-demo")
    stats = response.json()
    print(f"  Total documents: {stats['count']}")

    # Search to verify
    print("\nVerifying with search...")
    response = client.post(
        "/search",
        json={
            "query": "technology information",
            "collection": "batch-demo",
            "top_k": 5,
            "filters": {"category": "technology"},
        },
    )
    results = response.json()
    print(f"  Found {results['total_found']} technology documents")

    print("\nDone!")


if __name__ == "__main__":
    main()
