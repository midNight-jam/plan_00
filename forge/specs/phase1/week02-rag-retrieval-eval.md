# Week 2: RAG + Retrieval Evaluation (v2 — compressed)
> Phase: 1 | Project: Forge | Estimated Duration: 7 days
> **v2 note:** Replaces v1 `week02-rag-pipeline.md` + `week03-rag-hardening.md` (both preserved in `original_artifacts/specs_v1/forge_phase1/`). RAG is the most commoditized pattern in AI engineering — one week, not two. The durable skill kept from both v1 weeks is the **retrieval evaluation harness**: metric-driven measurement generalizes to every benchmark you build in Phase 2. Cut: chunking-strategy comparison, BM25/hybrid/RRF, prompt template versioning, JSON mode, multi-filetype parser depth.

## Context

Week 1 delivered a working inference server with OpenAI-compatible API. This week adds Retrieval-Augmented Generation — but treated as a *solved pattern to implement quickly and measure rigorously*, not a specialty to develop. You build the standard pipeline in the first half of the week, then spend the second half on what tutorials skip: a labeled evaluation dataset, retrieval metrics, and a semantic cache with measured hit behavior. The eval harness you build here is the template for every measurement artifact in this project.

**Prerequisites**: Week 1 complete — inference server running, Docker Compose working, PostgreSQL operational.

**Builds on**: The FastAPI server from Week 1. New endpoints added to the same server.

## Learning Goals

- [ ] Understand embedding models — text → vectors, dimensionality, cosine similarity
- [ ] Understand vector search — ANN, HNSW index parameters, distance metrics
- [ ] Understand the RAG pipeline end-to-end: ingest → embed → store → retrieve → augment → generate
- [ ] Understand retrieval evaluation — Recall@K, Precision@K, MRR, NDCG, and why "it looks right" is not a metric
- [ ] Understand reranking (cross-encoder vs bi-encoder) and how to *prove* it helps with metrics
- [ ] Understand semantic caching — cache by meaning, invalidation, threshold tradeoffs

## Implementation Goals

- [ ] Set up Qdrant in Docker Compose (vector size 384 for BGE-small, Cosine, HNSW m=16/ef_construct=100)
- [ ] Document ingestion API: upload (markdown + PDF via pymupdf) → recursive chunking (512 chars, 50 overlap) → batch embed (BGE-small on CPU) → store in Qdrant + metadata in PostgreSQL
- [ ] Retrieval endpoint (`POST /v1/search`): embed query → search → ranked chunks with scores; metadata filtering
- [ ] RAG generation: optional `rag` parameter on `/v1/chat/completions` (retrieve → augment → generate, streaming supported, context-overflow truncation)
- [ ] Cross-encoder reranking (ms-marco-MiniLM-L-6-v2, CPU), toggleable per request
- [ ] **Evaluation harness** (`src/forge/eval/`): metrics module (Recall@K, Precision@K, MRR, NDCG@K), labeled dataset (25–30 Q/A pairs over a 5–10 doc test corpus), runner producing JSON + human-readable report
- [ ] Semantic cache (Redis, brute-force cosine over stored query embeddings, similarity > 0.92, TTL, invalidation on document change) — 1 day, no more

## Acceptance Criteria

1. **Qdrant running**: `curl localhost:6333/collections` returns valid JSON
2. **Document upload**: POST a markdown or PDF file → returns document ID and chunk count; 10-page doc yields 30+ vectors in Qdrant
3. **Search works**: `/v1/search` returns top-5 relevant chunks with scores; metadata filter (document_id / tags) narrows results correctly
4. **RAG works**: RAG-enabled chat completion returns an answer referencing uploaded-document content; streaming works with RAG enabled
5. **Eval harness runs**: `uv run python -m forge.eval.run --dataset tests/eval/questions.json` produces a report with Recall@5, Precision@5, MRR, NDCG@5
6. **Retrieval quality**: Recall@5 > 0.7 on the labeled dataset
7. **Reranking proven**: with reranking enabled, at least one headline metric (MRR or NDCG@5) improves vs vector-only, shown side-by-side in the eval report — not vibes
8. **Cache hit**: identical query twice → second response <100ms (vs >1s uncached); rephrased query with embedding similarity > 0.92 also hits
9. **Cache invalidation**: adding or deleting a document invalidates affected cache entries (stale answers not served)
10. **Integration tests**: `uv run pytest tests/integration/test_rag.py -v` passes (upload, search, filter, RAG, streaming, eval-metric sanity, cache hit/miss)

## Validation Commands

```bash
# Qdrant up
curl http://localhost:6333/collections | python -m json.tool

# Upload a document
curl -X POST http://localhost:8000/v1/documents \
  -F "file=@test_docs/pytorch_tutorial.md" \
  -F 'metadata={"source": "pytorch", "tags": ["tutorial"]}'

# Search (retrieval only)
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I create a tensor in PyTorch?", "top_k": 5}'

# RAG generation (streaming) — model name per config (qwen2.5-7b-awq)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b-awq","messages":[{"role":"user","content":"How do I create a tensor in PyTorch?"}],"rag":{"enabled":true,"top_k":5,"rerank":true},"stream":true}'

# Run evaluation (with and without reranking)
uv run python -m forge.eval.run --dataset tests/eval/questions.json --output results/eval_vector.json
uv run python -m forge.eval.run --dataset tests/eval/questions.json --rerank --output results/eval_rerank.json

# Cache behavior (second, rephrased query should be fast)
time curl -s -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b-awq","messages":[{"role":"user","content":"What is PyTorch autograd?"}],"rag":{"enabled":true}}' > /dev/null
time curl -s -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b-awq","messages":[{"role":"user","content":"Explain PyTorch automatic differentiation"}],"rag":{"enabled":true}}' > /dev/null

# Tests
uv run pytest tests/integration/test_rag.py -v
```

## Technical Implementation Details

### Day 1–2: Pipeline core
- `src/forge/vectordb.py` — `VectorStore` wrapping qdrant-client: `create_collection()`, `upsert()`, `search(query_vector, top_k, filters)`
- `src/forge/embeddings.py` — `SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')`; batch for ingestion, single for queries. CPU keeps GPU VRAM for the LLM.
- `src/forge/chunking.py` — **recursive chunker only** (`\n\n` → `\n` → `. ` → ` ` → char limit; size 512, overlap 50). One strategy, done well. (v1's fixed/semantic chunkers cut — no measurable payoff for the portfolio.)
- `src/forge/ingestion.py` + `src/forge/routes/documents.py` — upload (multipart), extract (pymupdf for PDF, direct read for md/txt), chunk, embed, store; `POST/GET/DELETE /v1/documents`. Document metadata in PostgreSQL.
- Dependencies: `uv add sentence-transformers qdrant-client pymupdf`

### Day 3: Retrieval + RAG generation
- `src/forge/routes/search.py` — `POST /v1/search` with metadata filtering
- `src/forge/rag.py` — retrieve → (optional rerank) → assemble context → augmented prompt via `apply_chat_template` → generate with existing engine; truncate lowest-ranked chunks when context would overflow (`max_context - max_output - overhead`)
- `src/forge/reranker.py` — `cross-encoder/ms-marco-MiniLM-L-6-v2` on CPU, reorders candidates

### Day 4–5: Evaluation harness (the point of the week)
- `src/forge/eval/metrics.py` — Recall@K, Precision@K, MRR, NDCG@K implemented from formulas (unit-tested against hand-computed examples)
- `src/forge/eval/dataset.py` — format: `{"question": ..., "relevant_doc_ids": [...], "expected_answer_contains": [...]}`
- `src/forge/eval/run.py` — loads dataset, runs retrieval per question, computes metrics, emits JSON + table; `--rerank` flag for side-by-side comparison
- Test corpus: 5–10 documents (PyTorch/CUDA docs work well); 25–30 labeled questions. Use the LLM to draft questions, then manually verify ground truth — the manual verification is what makes the numbers trustworthy.
- **Design it reusable**: the runner/report pattern (config → runs → JSON → comparison table) is exactly what Week 9's batching benchmarks and Week 12's quantization quality gates will reuse.

### Day 6: Semantic cache (timebox: one day)
- `src/forge/cache.py` — Redis (add `redis:7-alpine` to compose); store (query_embedding, response, source_doc_ids); on query: embed, brute-force cosine against cached embeddings (fine for <10K entries), hit if > 0.92; TTL 1h; invalidate entries whose `source_doc_ids` intersect changed documents.

### Day 7: Tests + report
- `tests/integration/test_rag.py` per AC list; fixtures = the test corpus.
- Short write-up in `docs/`: baseline metrics table (vector vs rerank), cache hit-rate observations. Feeds a later blog post.

## If You Get Stuck

**Embedding model won't load**: use sentence-transformers directly, `device='cpu'`.
**Qdrant connection from Docker**: use service name `qdrant`, not `localhost`, inside the network.
**Recall@5 below 0.7**: check that eval questions are answerable from the corpus; try chunk_size 256; verify the same chunk IDs exist in both the dataset labels and Qdrant.
**Reranker shows no improvement**: expected when vector search is already near-ceiling on a small corpus — add harder questions (paraphrases, multi-hop) so there's headroom to measure.
**Cache too eager/too shy**: sweep the threshold (0.88–0.95) against a handful of paraphrase pairs and pick from data, not defaults.

## Agent Handoff Template

```
I'm on Week 2 of the Forge project — RAG + retrieval evaluation (v2 compressed spec).
Spec: /home/zzjam/Documents/dev/plan_00/forge/specs/phase1/week02-rag-retrieval-eval.md
Week 1 is complete: FastAPI + vLLM (Qwen2.5-7B-Instruct-AWQ) server at localhost:8000, Postgres logging, Docker Compose.
This week: standard RAG pipeline fast (days 1–3), then the retrieval eval harness (days 4–5), semantic cache (day 6), tests (day 7).
Codebase: /home/zzjam/Documents/dev/plan_00/forge/src/forge/
Focus on: [specific component]
```

## Out of Scope

- Hybrid search / BM25 / RRF (v1 Week 3 content — cut; commoditized, low differentiation)
- Multiple chunking strategies (recursive only)
- Prompt template versioning, JSON/structured output (vLLM ships structured output natively — use it if needed, don't build it)
- Multi-tenant document isolation (Week 5)
- Agentic RAG / tool use (Week 3 covers agentic *serving*, deliberately not RAG-flavored)
