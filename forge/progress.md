# Forge Progress Tracker

> Last updated: 2026-06-13
> Current Phase: Phase 1
> Current Week: Week 1
> Week 1 status: environment + scaffolding complete (Component 1); next is Component 2 (FastAPI server).

## Phase 1: Platform Foundation (Weeks 1-7)

- [ ] **Week 1**: Inference Server — FastAPI + vLLM + OpenAI API + Docker
- [ ] **Week 2**: RAG Pipeline — Qdrant + chunking + embeddings + retrieval
- [ ] **Week 3**: RAG Hardening — caching + evaluation + hybrid search
- [ ] **Week 4**: Multi-Model — registry + router + LoRA swap + queue
- [ ] **Week 5**: API Gateway — auth + rate limiting + multi-tenancy
- [ ] **Week 6**: Deployment — Docker Compose + K3s + Helm + CI/CD
- [ ] **Week 7**: Consolidation — testing + chaos + documentation + blog

## Phase 2: Inference Depth (Weeks 8-14)

- [ ] **Week 8**: Transformer Internals — forward pass from scratch + KV-cache basics
- [ ] **Week 9**: Continuous Batching — custom scheduler + benchmarks
- [ ] **Week 10**: KV-Cache + Memory — block allocator + eviction + profiling
- [ ] **Week 11**: Speculative Decoding — draft/verify + prefix caching + scheduling
- [ ] **Week 12**: Quantization — GPTQ/AWQ/NF4 + TensorRT + quality gates
- [ ] **Week 13**: Observability — metrics + tracing + dashboards + load testing
- [ ] **Week 14**: Performance Engineering — profiling methodology + optimization pass

## Phase 3: Advanced Systems (Weeks 15-20)

- [ ] **Week 15**: K8s Operator — CRD + reconciliation + GPU scheduling
- [ ] **Week 16**: Triton Kernels — fused RMSNorm + attention + MLP
- [ ] **Week 17**: Training/Alignment — DPO + reward model + custom training loop
- [ ] **Weeks 18-20**: Portfolio Polish — blogs + benchmarks + ADRs + demo video

## Completion Log

| Week | Date Started | Date Completed | Notes |
|------|-------------|----------------|-------|
| 1 | 2026-06-13 | (in progress) | Env setup done: CUDA/nvcc, nvidia-container-toolkit, uv deps installed. Code stubs scaffolded, not yet implemented. |

## Decisions Made

| # | Decision | Date | Rationale |
|---|----------|------|-----------|
| 1 | Python 3.14 (spec said 3.11+) | 2026-06-13 | Latest toolchain; pinned in `.python-version` / `pyproject.toml`. |
| 2 | `uv` for dependency management | 2026-06-13 | Fast, reproducible; project convention (CLAUDE.md). |
| 3 | vLLM 0.22.x as inference backend | 2026-06-13 | Reference engine for Phase 1 before custom engine in Phase 2. |

## Blockers / Issues

| Issue | Week | Status | Resolution |
|-------|------|--------|------------|
| RTX 5080 is Blackwell (sm_120) — confirm installed vLLM/PyTorch CUDA build supports it | 1 | Watch | Validate at first model load; fall back to `transformers`+`accelerate` if vLLM won't load (per spec). |
