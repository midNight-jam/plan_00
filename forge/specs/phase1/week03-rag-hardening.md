# Week 3: RAG Production Hardening
> Phase: 1 | Project: Forge | Estimated Duration: 7 days

## Context

Weeks 1-2 delivered a working inference server with RAG. This week, you make it production-quality: caching for performance, evaluation metrics to measure quality, hybrid search for better retrieval, and prompt engineering patterns. This is where you differentiate from tutorial-level RAG implementations.

**Prerequisites**: Week 2 complete — RAG pipeline working (ingest, search, generate with retrieval).

**Builds on**: The RAG pipeline from Week 2. Adding caching layer, evaluation, and hybrid search.

## Learning Goals

- [ ] Understand semantic caching — how to cache by meaning, not just exact match
- [ ] Understand retrieval evaluation metrics — Recall@K, Precision@K, MRR, NDCG
- [ ] Understand hybrid search — why combining vector + keyword beats either alone
- [ ] Understand BM25 scoring and Reciprocal Rank Fusion (RRF)
- [ ] Understand prompt engineering patterns — few-shot, chain-of-thought, structured output

## Implementation Goals

- [ ] Build semantic cache layer (Redis-based, similarity threshold for cache hits)
- [ ] Build retrieval evaluation pipeline with test dataset and metrics
- [ ] Implement BM25 keyword search alongside vector search
- [ ] Implement Reciprocal Rank Fusion to combine vector + BM25 results
- [ ] Add prompt template management (versioned, configurable)
- [ ] Add structured output support (JSON mode)
- [ ] Implement incremental indexing (add docs without re-indexing everything)
- [ ] Add document processing for PDFs, code files, and markdown with proper handling

## Acceptance Criteria

1. **Cache hit**: Same query asked twice → second response returns in <100ms (vs >1s first time)
2. **Cache semantics**: Similar query (rephrased) also hits cache if similarity > 0.92
3. **Evaluation runs**: `python -m forge.eval.run` produces metrics report (Recall@5, MRR, etc.)
4. **Retrieval quality**: Recall@5 > 0.7 on the test dataset
5. **Hybrid search**: Hybrid (vector + BM25) achieves higher Recall@5 than vector-only
6. **Prompt templates**: Templates stored in config, switchable without code change
7. **Structured output**: Request with `response_format: {"type": "json_object"}` returns valid JSON
8. **Incremental index**: Adding a new document doesn't require re-embedding existing documents
9. **Multiple file types**: PDF, markdown, Python files all parse and chunk correctly
10. **Integration tests**: `pytest tests/integration/test_rag_advanced.py` passes

## Validation Commands

```bash
# Test semantic cache
time curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mistral-7b","messages":[{"role":"user","content":"What is PyTorch autograd?"}],"rag":{"enabled":true}}'
# Run same query again — should be much faster
time curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mistral-7b","messages":[{"role":"user","content":"What is PyTorch autograd?"}],"rag":{"enabled":true}}'

# Test similar query hits cache
time curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mistral-7b","messages":[{"role":"user","content":"Explain PyTorch automatic differentiation"}],"rag":{"enabled":true}}'

# Run evaluation
python -m forge.eval.run --dataset tests/eval/test_questions.json --output results/eval_report.json
cat results/eval_report.json | python -m json.tool

# Test hybrid search
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "CUDA memory allocation", "top_k": 5, "strategy": "hybrid"}'

# Test structured output
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mistral-7b","messages":[{"role":"user","content":"List 3 PyTorch functions"}],"response_format":{"type":"json_object"}}'

# Run advanced tests
pytest tests/integration/test_rag_advanced.py -v
```

## Technical Implementation Details

### Component 1: Semantic Cache (Day 1-2)

**File: `src/forge/cache.py`**

Architecture:
- Store (query_embedding, response) pairs in Redis
- On new query: embed it, search cache for similar embeddings (cosine > 0.92)
- If cache hit: return cached response immediately
- If miss: run full RAG pipeline, store result in cache
- TTL: cache entries expire after configurable duration (default 1 hour)
- Cache invalidation: when documents are added/deleted, invalidate relevant cache entries

Implementation:
- Use Redis with `redis-py` async client
- Store embeddings as Redis vectors (Redis supports vector similarity search)
- OR simpler: store embeddings as numpy arrays in Redis hash, do brute-force comparison (fine for <10K cache entries)
- Cache key: hash of the query embedding rounded to 4 decimal places
- Cache value: JSON with response, timestamp, source_chunks

Add to `docker-compose.yml`:
```yaml
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"
  volumes:
    - redis_data:/data
```

### Component 2: Retrieval Evaluation Pipeline (Day 2-3)

**File: `src/forge/eval/metrics.py`**

Metrics to implement:
- **Recall@K**: Of the relevant documents, what fraction was retrieved in top-K?
- **Precision@K**: Of the top-K retrieved, what fraction is relevant?
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank of first relevant result
- **NDCG@K**: Normalized Discounted Cumulative Gain (accounts for position)

**File: `src/forge/eval/dataset.py`**

Test dataset format (create manually with 20-30 question/answer pairs):
```json
[
  {
    "question": "How do I create a tensor in PyTorch?",
    "relevant_doc_ids": ["doc_pytorch_basics_chunk_3", "doc_pytorch_basics_chunk_4"],
    "expected_answer_contains": ["torch.tensor", "torch.zeros"]
  }
]
```

**File: `src/forge/eval/run.py`**

Evaluation runner:
1. Load test dataset
2. For each question: run retrieval, compare against ground truth
3. Compute metrics
4. Output report (JSON + human-readable summary)

### Component 3: Hybrid Search with BM25 + RRF (Day 3-4)

**File: `src/forge/search/bm25.py`**

BM25 implementation:
- Build inverted index from all document chunks (on ingestion)
- Score query against all documents using BM25 formula
- Store index in Redis or rebuild from Qdrant metadata

**File: `src/forge/search/hybrid.py`**

Reciprocal Rank Fusion:
```python
def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    scores = {}
    for rank, doc in enumerate(vector_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    for rank, doc in enumerate(bm25_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

Search strategies (configurable per request):
- `"vector"`: pure vector search (existing)
- `"bm25"`: pure keyword search
- `"hybrid"`: RRF combination of both

### Component 4: Prompt Template Management (Day 4-5)

**File: `src/forge/prompts/manager.py`**

Template system:
- Store templates as YAML files in `prompts/` directory
- Variables: `{context}`, `{query}`, `{history}`, `{format_instructions}`
- Version tracking: each template has a version number
- A/B support: can configure which template version to use per request

**File: `prompts/rag_default.yaml`**
```yaml
name: rag_default
version: 1
system: |
  You are a helpful assistant. Answer the user's question based on the provided context.
  If the context doesn't contain the answer, say "I don't have enough information to answer this."
  Always cite which part of the context you're drawing from.
context_prefix: |
  Here is the relevant context:
  ---
  {context}
  ---
format_instructions: |
  Respond in clear, concise language.
```

### Component 5: Structured Output (Day 5)

Modify inference to support JSON mode:
- When `response_format.type == "json_object"`, add instruction to system prompt
- Use constrained decoding if vLLM supports it, otherwise post-validate
- Parse response as JSON, retry once if malformed

### Component 6: Document Processing Improvements (Day 5-6)

**File: `src/forge/ingestion/parsers.py`**

Parsers:
- `MarkdownParser`: Preserves headers as metadata, chunks by section
- `PdfParser`: Uses pymupdf, handles tables and images (skip images, extract table text)
- `CodeParser`: Chunks by function/class for Python files, preserves imports as metadata
- `PlainTextParser`: Fallback for .txt files

### Component 7: Tests + Evaluation Dataset (Day 6-7)

Create a test corpus: 5-10 documents about PyTorch/CUDA (use official docs or tutorials).
Create evaluation dataset: 25-30 questions with known answers from those documents.
Run full evaluation, document baseline metrics.

## If You Get Stuck

**Redis vector search too complex**: Skip Redis vector similarity. Instead, store cache entries as a simple list and do brute-force numpy cosine similarity. For <10K entries, this is instant.

**BM25 implementation slow**: Use the `rank_bm25` Python package instead of implementing from scratch. Focus on the RRF fusion logic which is the important part.

**Evaluation dataset creation is tedious**: Use the LLM itself to help generate questions from the documents, then manually verify/edit the ground truth answers.

**Hybrid search doesn't improve over vector**: Make sure BM25 index is built on the same chunks as vector store. Try adjusting RRF k parameter (60 is standard but experiment with 20-100).

## Agent Handoff Template

```
I'm on Week 3 of Forge — hardening the RAG pipeline for production quality.
Spec: /Users/jmalviya/Documents/zz/dev/plan_00/forge/specs/phase1/week03-rag-hardening.md
Current state: Working RAG pipeline (Qdrant + embeddings + reranking) from Week 2.
I need to add: semantic caching (Redis), retrieval evaluation metrics, hybrid search (BM25 + RRF), prompt template management.
Codebase: [path to forge/src/]
Focus on: [specific component you're stuck on]
```

## Out of Scope

- Multi-tenant document isolation (Week 5)
- Document access control (Week 5)
- Real-time document updates/webhooks (not in plan)
- Multi-modal RAG / image understanding (not in plan)
- Complex agentic RAG with tool use (not in plan — keep it focused)
