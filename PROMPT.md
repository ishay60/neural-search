# 🔍 neural-search - Semantic Search Engine

## Project Goal
Build a production-grade semantic search API that converts text to embeddings and enables similarity search. Demonstrates ML infrastructure knowledge, vector databases, and building AI-powered applications.

## Why This Project?
- Shows practical ML/AI engineering skills
- Demonstrates understanding of embeddings and vector search
- Highly relevant (RAG, search, recommendations all use this)
- Combines ML with solid backend engineering

---

## Core Features to Implement

### Phase 1: Foundation
- [ ] REST API with FastAPI
- [ ] Text embedding generation (sentence-transformers)
- [ ] In-memory vector store (FAISS)
- [ ] Basic CRUD for documents
- [ ] Similarity search endpoint

### Phase 2: Production Features
- [ ] Persistent vector storage (Qdrant/Milvus/Pinecone)
- [ ] Batch ingestion pipeline
- [ ] Async processing with Celery/RQ
- [ ] Caching layer (Redis)
- [ ] Rate limiting

### Phase 3: Advanced Search
- [ ] Hybrid search (vector + keyword BM25)
- [ ] Metadata filtering
- [ ] Multi-modal search (text + images with CLIP)
- [ ] Re-ranking with cross-encoders
- [ ] Query expansion/reformulation

### Phase 4: Operations
- [ ] Model serving optimization (ONNX, TensorRT)
- [ ] Horizontal scaling (multiple workers)
- [ ] A/B testing framework
- [ ] Search analytics
- [ ] Admin dashboard

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      API Gateway                             │
│                   (FastAPI + Auth)                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   Search    │   │   Index     │   │   Admin     │
│   Service   │   │   Service   │   │   Service   │
└──────┬──────┘   └──────┬──────┘   └─────────────┘
       │                 │
       │    ┌────────────┼────────────┐
       │    │            │            │
       ▼    ▼            ▼            ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  Embedding  │   │   Queue     │   │   Cache     │
│   Model     │   │  (Celery)   │   │  (Redis)    │
└─────────────┘   └─────────────┘   └─────────────┘
       │                 │
       └────────┬────────┘
                │
        ┌───────▼───────┐
        │ Vector Store  │
        │   (Qdrant)    │
        └───────────────┘
```

---

## API Design

### Document Ingestion
```python
# POST /api/v1/documents
{
    "documents": [
        {
            "id": "doc-001",
            "content": "Machine learning is a subset of artificial intelligence...",
            "metadata": {
                "source": "wikipedia",
                "category": "technology",
                "created_at": "2024-01-15"
            }
        }
    ],
    "collection": "knowledge-base"
}

# Response
{
    "status": "accepted",
    "job_id": "job-abc123",
    "documents_count": 1
}
```

### Search
```python
# POST /api/v1/search
{
    "query": "How does neural network training work?",
    "collection": "knowledge-base",
    "top_k": 10,
    "filters": {
        "category": "technology",
        "created_at": {"$gte": "2024-01-01"}
    },
    "include_metadata": true,
    "hybrid": true,  # Enable hybrid search
    "rerank": true   # Enable cross-encoder reranking
}

# Response
{
    "results": [
        {
            "id": "doc-001",
            "content": "Machine learning is a subset...",
            "score": 0.92,
            "metadata": {...}
        }
    ],
    "took_ms": 45,
    "total_found": 156
}
```

### Collections Management
```python
# POST /api/v1/collections
{
    "name": "knowledge-base",
    "embedding_model": "all-MiniLM-L6-v2",
    "dimension": 384,
    "distance_metric": "cosine"
}

# GET /api/v1/collections
# GET /api/v1/collections/{name}/stats
# DELETE /api/v1/collections/{name}
```

---

## Dependencies (requirements.txt)

```
# API
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0
python-multipart>=0.0.6

# ML/Embeddings
sentence-transformers>=2.2.2
transformers>=4.36.0
torch>=2.1.0
onnxruntime>=1.16.0

# Vector Store
qdrant-client>=1.7.0
faiss-cpu>=1.7.4

# Hybrid Search
rank-bm25>=0.2.2

# Queue/Cache
celery>=5.3.0
redis>=5.0.0

# Database
sqlalchemy>=2.0.0
asyncpg>=0.29.0
alembic>=1.13.0

# Monitoring
prometheus-client>=0.19.0
opentelemetry-api>=1.22.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.23.0
httpx>=0.26.0
```

---

## File Structure

```
neural-search/
├── pyproject.toml
├── README.md
├── LICENSE
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── docker.yml
│       └── benchmark.yml
├── src/
│   └── neural_search/
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── router.py
│       │   ├── documents.py
│       │   ├── search.py
│       │   ├── collections.py
│       │   └── schemas.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── embeddings.py
│       │   ├── search_engine.py
│       │   ├── reranker.py
│       │   └── hybrid.py
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── vector_store.py
│       │   ├── qdrant.py
│       │   └── faiss_store.py
│       ├── workers/
│       │   ├── __init__.py
│       │   ├── celery_app.py
│       │   └── tasks.py
│       └── utils/
│           ├── __init__.py
│           ├── cache.py
│           └── metrics.py
├── tests/
│   ├── conftest.py
│   ├── test_api/
│   ├── test_embeddings/
│   └── test_search/
├── benchmarks/
│   ├── embedding_speed.py
│   └── search_accuracy.py
├── scripts/
│   ├── download_models.py
│   └── seed_data.py
└── examples/
    ├── quickstart.py
    ├── batch_ingestion.py
    └── rag_example.py
```

---

## Embedding Models to Support

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | General purpose |
| all-mpnet-base-v2 | 768 | Medium | Better | Higher quality |
| e5-large-v2 | 1024 | Slow | Best | Maximum quality |
| CLIP ViT-B/32 | 512 | Medium | - | Multi-modal |

---

## Performance Benchmarks to Include

```markdown
## Benchmarks

### Embedding Generation
| Model | Batch Size | Throughput | Latency (p99) |
|-------|------------|------------|---------------|
| MiniLM-L6 | 32 | 1000 docs/s | 45ms |
| mpnet-base | 32 | 400 docs/s | 120ms |

### Search Latency
| Collection Size | Top-K | Latency (p50) | Latency (p99) |
|-----------------|-------|---------------|---------------|
| 100K | 10 | 5ms | 15ms |
| 1M | 10 | 12ms | 35ms |
| 10M | 10 | 25ms | 80ms |

### Search Quality (BEIR Benchmark)
| Dataset | BM25 | Dense | Hybrid |
|---------|------|-------|--------|
| MS MARCO | 0.228 | 0.334 | 0.358 |
| NFCorpus | 0.325 | 0.318 | 0.342 |
```

---

## RAG Integration Example

```python
from neural_search import NeuralSearchClient
from openai import OpenAI

search_client = NeuralSearchClient("http://localhost:8000")
llm_client = OpenAI()

def rag_query(question: str) -> str:
    # Retrieve relevant context
    results = search_client.search(
        query=question,
        collection="knowledge-base",
        top_k=5,
        hybrid=True,
        rerank=True
    )

    context = "\n\n".join([r.content for r in results])

    # Generate answer with context
    response = llm_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Answer based on this context:\n{context}"},
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message.content
```

---

## Success Criteria

This project is complete when:
1. Can ingest 1M documents and search in <50ms p99
2. Hybrid search improves quality over dense-only by 5%+
3. Has working RAG example
4. Comprehensive API documentation (OpenAPI/Swagger)
5. Docker Compose for full stack deployment
6. Published to PyPI
