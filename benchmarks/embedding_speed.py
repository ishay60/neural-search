#!/usr/bin/env python3
"""Benchmark embedding generation speed.

This script measures the throughput and latency of embedding generation
for different models and batch sizes.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    model: str
    batch_size: int
    num_texts: int
    total_time_s: float
    throughput_docs_per_s: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float


def generate_sample_texts(n: int, avg_length: int = 200) -> list[str]:
    """Generate sample texts for benchmarking."""
    words = [
        "machine", "learning", "artificial", "intelligence", "neural",
        "network", "deep", "algorithm", "data", "model", "training",
        "inference", "optimization", "gradient", "backpropagation",
        "transformer", "attention", "embedding", "vector", "semantic",
    ]

    texts = []
    for _ in range(n):
        # Generate random text
        num_words = avg_length // 7  # Average word length
        text_words = [words[i % len(words)] for i in np.random.randint(0, len(words), num_words)]
        texts.append(" ".join(text_words))

    return texts


def benchmark_model(
    model_name: str,
    texts: list[str],
    batch_sizes: list[int],
) -> list[BenchmarkResult]:
    """Benchmark a model with different batch sizes."""
    from sentence_transformers import SentenceTransformer

    print(f"\nBenchmarking model: {model_name}")
    model = SentenceTransformer(model_name)

    results = []

    for batch_size in batch_sizes:
        print(f"  Batch size: {batch_size}")

        # Warmup
        _ = model.encode(texts[:min(batch_size, len(texts))], batch_size=batch_size)

        # Benchmark
        latencies = []
        start_total = time.perf_counter()

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            start = time.perf_counter()
            _ = model.encode(batch, batch_size=batch_size)
            latencies.append((time.perf_counter() - start) * 1000)

        total_time = time.perf_counter() - start_total

        # Calculate metrics
        throughput = len(texts) / total_time
        latencies_array = np.array(latencies)

        result = BenchmarkResult(
            model=model_name,
            batch_size=batch_size,
            num_texts=len(texts),
            total_time_s=total_time,
            throughput_docs_per_s=throughput,
            latency_p50_ms=float(np.percentile(latencies_array, 50)),
            latency_p95_ms=float(np.percentile(latencies_array, 95)),
            latency_p99_ms=float(np.percentile(latencies_array, 99)),
        )
        results.append(result)

        print(f"    Throughput: {result.throughput_docs_per_s:.0f} docs/s")
        print(f"    Latency P50: {result.latency_p50_ms:.2f}ms")
        print(f"    Latency P99: {result.latency_p99_ms:.2f}ms")

    return results


def main():
    """Run embedding benchmarks."""
    # Configuration
    models = [
        "all-MiniLM-L6-v2",
        # "all-mpnet-base-v2",  # Uncomment for more models
    ]
    batch_sizes = [1, 8, 16, 32, 64]
    num_texts = 1000

    print("=" * 60)
    print("Embedding Speed Benchmark")
    print("=" * 60)

    # Generate test data
    print(f"\nGenerating {num_texts} sample texts...")
    texts = generate_sample_texts(num_texts)

    # Run benchmarks
    all_results = []
    for model_name in models:
        try:
            results = benchmark_model(model_name, texts, batch_sizes)
            all_results.extend(results)
        except Exception as e:
            print(f"  Error benchmarking {model_name}: {e}")

    # Save results
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "embedding_speed.json"
    with open(output_file, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Model':<30} {'Batch':<8} {'Throughput':<15} {'P99 Latency':<12}")
    print("-" * 60)
    for r in all_results:
        print(f"{r.model:<30} {r.batch_size:<8} {r.throughput_docs_per_s:<15.0f} {r.latency_p99_ms:<12.2f}ms")


if __name__ == "__main__":
    main()
