# Week 2: RAG Pipeline
> Phase: 1 | Project: Forge | Estimated Duration: 7 days

## Context

Week 1 delivered a working inference server with OpenAI-compatible API. This week, you add Retrieval-Augmented Generation — the most deployed production AI pattern. You'll build a document ingestion pipeline, vector search with Qdrant, and a retrieval + generation endpoint.

**Prerequisites**: Week 1 complete — inference server running, Docker Compose working, PostgreSQL operational.

**Builds on**: The FastAPI server from Week 1. New endpoints will be added to the same server.

## Learning Goals

- [ ] Understand embedding models — how text becomes vectors, dimensionality, similarity metrics
- [ ] Understand vector search — approximate nearest neighbor (ANN), HNSW index, distance metrics
- [ ] Understand chunking tradeoffs — too small loses context, too large dilutes relevance
- [ ] Understand the RAG pipeline end-to-end: ingest → embed → store → retrieve → augment → generate
- [ ] Understand reranking — why initial retrieval isn't enough, cross-encoder vs bi-encoder

## Implementation Goals

- [ ] Set up Qdrant vector database in Docker Compose
- [ ] Build document ingestion API (upload, chunk, embed, store)
- [ ] Implement 3 chunking strategies (fixed-size, recursive, semantic)
- [ ] Integrate local embedding model (BGE-small or E5-small)
- [ ] Build retrieval endpoint (query → embed → search → return chunks)
- [ ] Build RAG generation endpoint (retrieve → augment prompt → generate with LLM)
- [ ] Implement reranking step (cross-encoder)
- [ ] Handle context window overflow (truncation strategy)
- [ ] Add metadata filtering to vector search

## Acceptance Criteria

1. **Qdrant running**: `curl localhost:6333/collections` returns valid JSON
2. **Document upload**: POST a PDF/markdown file → returns document ID and chunk count
3. **Chunks stored**: After uploading a 10-page document, Qdrant collection has 30+ vectors
4. **Search works**: Query endpoint returns top-5 relevant chunks with scores > 0.7
5. **RAG works**: RAG endpoint returns an answer that references information from uploaded documents
6. **Streaming RAG**: RAG endpoint supports streaming response
7. **Metadata filter**: Can filter search by document_id, upload_date, or custom tags
8. **Multiple strategies**: Config can switch between fixed/recursive/semantic chunking
9. **Reranking**: With reranking enabled, results are measurably more relevant (test with known-answer queries)
10. **Integration test**: `pytest tests/integration/test_rag.py` passes all cases

## Validation Commands

```bash
# Ensure Qdrant is running
curl http://localhost:6333/collections | python -m json.tool

# Upload a document
curl -X POST http://localhost:8000/v1/documents \
  -F "file=@test_docs/pytorch_tutorial.md" \
  -F "metadata={\"source\": \"pytorch\", \"tags\": [\"tutorial\"]}"

# List documents
curl http://localhost:8000/v1/documents | python -m json.tool

# Search (retrieval only)
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I create a tensor in PyTorch?", "top_k": 5}'

# RAG generation
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mistral-7b","messages":[{"role":"user","content":"How do I create a tensor in PyTorch?"}],"rag":{"enabled":true,"top_k":5},"stream":true}'

# Metadata-filtered search
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "tensor creation", "top_k": 5, "filter": {"source": "pytorch"}}'

# Run tests
pytest tests/integration/test_rag.py -v
```

## Technical Implementation Details

### Component 1: Qdrant Setup (Day 1)

Add to `docker-compose.yml`:
```yaml
qdrant:
  image: qdrant/qdrant:latest
  ports:
    - "6333:6333"
  volumes:
    - qdrant_data:/qdrant/storage
```

Create collection on startup with appropriate config:
- Vector size: 384 (for BGE-small) or 768 (for E5-base)
- Distance metric: Cosine
- HNSW index config: m=16, ef_construct=100

**File: `src/forge/vectordb.py`**
- Class `VectorStore` wrapping qdrant-client
- Methods: `create_collection()`, `upsert(vectors, metadata)`, `search(query_vector, top_k, filters)`

### Component 2: Embedding Model (Day 1-2)

**File: `src/forge/embeddings.py`**
- Load a local embedding model (sentence-transformers/bge-small-en-v1.5)
- Run on CPU (small enough, saves GPU VRAM for the LLM)
- Batch embedding for ingestion (embed many chunks at once)
- Single embedding for queries
- Class `EmbeddingModel` with methods: `embed_documents(texts: list[str]) -> list[list[float]]`, `embed_query(text: str) -> list[float]`

Dependencies: `uv add sentence-transformers qdrant-client`

### Component 3: Document Chunking (Day 2-3)

**File: `src/forge/chunking.py`**

Implement 3 strategies:

1. **FixedSizeChunker**: Split by character count with overlap
   - Parameters: chunk_size=512, overlap=50
   - Simple but doesn't respect document structure

2. **RecursiveChunker**: Split by separators in priority order
   - Split by: `\n\n` → `\n` → `. ` → ` ` → character limit
   - Respects paragraph boundaries where possible
   - This is what LangChain's RecursiveCharacterTextSplitter does

3. **SemanticChunker**: Split by meaning change
   - Embed each sentence, group consecutive sentences with high similarity
   - When similarity drops below threshold, start new chunk
   - Best quality but slowest

All chunkers implement: `chunk(text: str) -> list[Chunk]` where `Chunk` has: content, start_idx, end_idx, metadata

### Component 4: Document Ingestion Pipeline (Day 3-4)

**File: `src/forge/ingestion.py`**

Pipeline:
1. Accept file upload (PDF, markdown, plain text)
2. Extract text (use `pymupdf` for PDFs, direct read for markdown/text)
3. Chunk using configured strategy
4. Embed all chunks (batch)
5. Store vectors + metadata in Qdrant
6. Store document metadata in PostgreSQL (id, filename, chunk_count, upload_time)

**File: `src/forge/routes/documents.py`**

Endpoints:
- `POST /v1/documents` — upload + ingest (multipart form)
- `GET /v1/documents` — list all documents with metadata
- `GET /v1/documents/{id}` — get document details + chunk count
- `DELETE /v1/documents/{id}` — delete document and its vectors

### Component 5: Retrieval + RAG Generation (Day 4-5)

**File: `src/forge/routes/search.py`**

- `POST /v1/search` — pure retrieval (no generation)
  - Embed query → search Qdrant → return ranked chunks with scores
  - Support metadata filtering

**File: `src/forge/rag.py`**

RAG pipeline class:
1. Embed user query
2. Retrieve top-K chunks from Qdrant
3. (Optional) Rerank with cross-encoder
4. Assemble context: format chunks into a prompt section
5. Build augmented prompt: system message + context + user query
6. Generate with LLM (reuse existing inference engine)
7. Handle context overflow: if chunks exceed max context, truncate from bottom

Modify `POST /v1/chat/completions` to accept optional `rag` parameter:
```json
{
  "rag": {
    "enabled": true,
    "top_k": 5,
    "rerank": true,
    "filter": {"source": "pytorch"}
  }
}
```

### Component 6: Reranking (Day 5-6)

**File: `src/forge/reranker.py`**

- Load cross-encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2)
- Runs on CPU (small model)
- Takes query + candidate chunks → scores each pair → returns reordered list
- Significantly improves relevance over pure vector search

### Component 7: Tests (Day 6-7)

**File: `tests/integration/test_rag.py`**

- `test_upload_document` — upload returns 200 with document ID
- `test_upload_pdf` — PDF parsing works
- `test_search_returns_results` — after upload, search finds relevant chunks
- `test_search_with_filter` — metadata filter narrows results correctly
- `test_rag_generation` — RAG response references uploaded content
- `test_rag_streaming` — streaming works with RAG enabled
- `test_context_overflow` — large context is handled without crashing
- `test_reranking_improves_results` — reranked results are more relevant for a known-answer query

Prepare test fixtures: 3-4 markdown files with distinct content for testing retrieval quality.

## If You Get Stuck

**Embedding model won't load**: Use `sentence-transformers` library directly, not through HuggingFace pipeline. Ensure it runs on CPU: `model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')`

**Qdrant connection issues**: Check Docker network — services must be on same Docker network. Use service name `qdrant` not `localhost` from within Docker.

**Poor retrieval quality**: Start with recursive chunking at chunk_size=512 with 50 char overlap. If still poor, try smaller chunks (256) or switch to semantic chunking.

**Context window overflow**: Simple truncation works fine for now. Take top-K chunks, concatenate, if total tokens > (max_context - max_output - prompt_overhead), drop the last chunks until it fits.

## Agent Handoff Template

```
I'm on Week 2 of the Forge project — building a RAG pipeline.
The spec is at: /Users/jmalviya/Documents/zz/dev/plan_00/forge/specs/phase1/week02-rag-pipeline.md
Week 1 is complete: I have a working FastAPI inference server with vLLM at localhost:8000.
I need to add: document ingestion + vector search (Qdrant) + RAG generation.
The server codebase is at: [path to forge/src/]
Please implement the RAG components described in the spec, adding to the existing FastAPI server.
```

## Out of Scope

- Semantic caching (Week 3)
- Evaluation metrics for retrieval quality (Week 3)
- Hybrid search / BM25 (Week 3)
- Authentication for document endpoints (Week 5)
- Multi-tenant document isolation (Week 5)
