# Neural Search

Production-grade semantic search API with embeddings and vector search.

## Features

- **REST API with FastAPI** - Modern async API with automatic OpenAPI documentation
- **Text Embeddings** - Sentence-transformers for high-quality text embeddings
- **Vector Storage** - FAISS (in-memory) and Qdrant (persistent) backends
- **Hybrid Search** - Combine dense vectors with BM25 sparse retrieval
- **Re-ranking** - Cross-encoder models for improved result quality
- **Metadata Filtering** - Filter search results by document metadata
- **Async Processing** - Celery workers for background document ingestion
- **Caching** - Redis caching for embeddings and search results
- **Rate Limiting** - Protect API from abuse
- **Prometheus Metrics** - Full observability with Prometheus and Grafana

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Start the full stack
docker-compose up -d

# View logs
docker-compose logs -f api

# Access the API
curl http://localhost:8000/health
```

### Local Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Start Redis (required for caching/workers)
docker run -d -p 6379:6379 redis:7-alpine

# Run the API
make run

# In another terminal, start the worker
make worker
```

## API Usage

### Create a Collection

```bash
curl -X POST http://localhost:8000/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "my-collection"}'
```

### Index Documents

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "my-collection",
    "documents": [
      {
        "id": "doc1",
        "content": "Machine learning is a subset of AI...",
        "metadata": {"category": "technology"}
      }
    ]
  }'
```

### Search

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does neural network training work?",
    "collection": "my-collection",
    "top_k": 10,
    "hybrid": true,
    "rerank": true
  }'
```

## Configuration

Configuration is done via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEURAL_SEARCH_API_HOST` | `0.0.0.0` | API host |
| `NEURAL_SEARCH_API_PORT` | `8000` | API port |
| `NEURAL_SEARCH_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `NEURAL_SEARCH_VECTOR_STORE_TYPE` | `faiss` | `faiss` or `qdrant` |
| `NEURAL_SEARCH_QDRANT_HOST` | `localhost` | Qdrant host |
| `NEURAL_SEARCH_REDIS_URL` | `redis://localhost:6379/0` | Redis URL |

See `src/neural_search/config.py` for all options.

## Architecture

```
                     ┌─────────────────┐
                     │   FastAPI API   │
                     │   (Port 8000)   │
                     └────────┬────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    Embedding    │  │   Celery        │  │     Redis       │
│    Model        │  │   Workers       │  │   (Cache)       │
└─────────────────┘  └────────┬────────┘  └─────────────────┘
                              │
                     ┌────────▼────────┐
                     │  Vector Store   │
                     │ (FAISS/Qdrant)  │
                     └─────────────────┘
```

## Development

```bash
# Install dev dependencies
make dev

# Run tests
make test

# Run tests with coverage
make test-cov

# Lint code
make lint

# Format code
make format
```

## Benchmarks

Run benchmarks:

```bash
make benchmark
```

### Embedding Speed (all-MiniLM-L6-v2)

| Batch Size | Throughput | P99 Latency |
|------------|------------|-------------|
| 1 | ~50 docs/s | ~20ms |
| 32 | ~500 docs/s | ~65ms |
| 64 | ~600 docs/s | ~110ms |

### Search Latency

| Collection Size | Top-K | P50 | P99 |
|-----------------|-------|-----|-----|
| 10K | 10 | <5ms | <15ms |
| 100K | 10 | <10ms | <30ms |

## Examples

See the `examples/` directory:

- `quickstart.py` - Basic API usage
- `batch_ingestion.py` - Large-scale document ingestion
- `rag_example.py` - RAG pipeline integration

## License

MIT License - see LICENSE file.
