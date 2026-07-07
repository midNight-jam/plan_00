# Forge Progress Tracker

> Last updated: 2026-07-06
> Current Phase: Phase 1
> Current Week: Week 1
> Week 1 status: 9/10 ACs done. Only remaining: pytest-format integration tests (AC #9).
> **Plan v2 (2026-07-06):** W2/W3/W5/W6 specs regenerated per the strategic review (`original_artifacts/plan_evolution_v2_2026-07.md`); v1 originals archived in `original_artifacts/specs_v1/forge_phase1/`. Standing deltas for kept weeks + ⚠️ **re-evaluation gates (before W11 and before W15)** live in `specs/V2_DELTAS_AND_REEVAL.md` — read that file at every phase boundary.

## Phase 1: Platform Foundation (Weeks 1-7)

- [ ] **Week 1**: Inference Server — FastAPI + vLLM + OpenAI API + Docker
- [ ] **Week 2**: RAG + Retrieval Eval — compressed pipeline + eval harness (v2)
- [ ] **Week 3**: Agentic Serving — sessions + tool loop + trace suite + chat-vs-agent benchmark (v2, new)
- [ ] **Week 4**: Multi-Model — registry + router + LoRA swap + queue (v2 delta: +InferencePool-style adapter routing AC)
- [ ] **Week 5**: Gateway + Deployment — streaming metering + disconnect-abort + K3s/Helm/CI (v2, merged)
- [ ] **Week 6**: Buffer + OSS — AC sweep + blog #1 (sm_120/Blackwell) + upstream issue + paper sprint #0 (v2, new)
- [ ] **Week 7**: Consolidation — testing + chaos + documentation + blog (+ paper sprint #1)

## Phase 2: Inference Depth (Weeks 8-14)

> ⚠️ **Re-evaluation Gate #1 — when Week 10 completes, BEFORE starting Week 11**: re-check W11–W14 specs against the then-current landscape per `specs/V2_DELTAS_AND_REEVAL.md`, and capture which v2 forecasts held — that record is the input for the next high-level plan review. W9/W10/W12/W13 carry standing deltas from the same file.

- [ ] **Week 8**: Transformer Internals — forward pass from scratch + KV-cache basics
- [ ] **Week 9**: Continuous Batching — custom scheduler + benchmarks
- [ ] **Week 10**: KV-Cache + Memory — block allocator + eviction + profiling
- [ ] **Week 11**: Speculative Decoding — draft/verify + prefix caching + scheduling
- [ ] **Week 12**: Quantization — GPTQ/AWQ/NF4 + TensorRT + quality gates
- [ ] **Week 13**: Observability — metrics + tracing + dashboards + load testing
- [ ] **Week 14**: Performance Engineering — profiling methodology + optimization pass

## Phase 3: Advanced Systems (Weeks 15-20)

> ⚠️ **Re-evaluation Gate #2 — when Week 14 completes, BEFORE starting Week 15**: W15 must be rebuilt on DRA + Kueue + Gateway API Inference Extension (or become an llm-d contribution week) and W17 gains GRPO — re-validate both against the then-current ecosystem per `specs/V2_DELTAS_AND_REEVAL.md`. Applications start Jan–Feb 2027, so also re-check portfolio ACs (W18–20) against real job postings at this gate.

- [ ] **Week 15**: K8s Operator — CRD + reconciliation + GPU scheduling
- [ ] **Week 16**: Triton Kernels — fused RMSNorm + attention + MLP
- [ ] **Week 17**: Training/Alignment — DPO + reward model + custom training loop
- [ ] **Weeks 18-20**: Portfolio Polish — blogs + benchmarks + ADRs + demo video

## Completion Log

| Week | Date Started | Date Completed | Notes |
|------|-------------|----------------|-------|
| 1 | 2026-06-13 | (in progress) | 9/10 ACs done. Full stack live: vLLM (Qwen2.5-7B-Instruct-AWQ) + FastAPI (non-streaming + SSE streaming) + `/health` + `/v1/models` + DB logging (`database.py`, async SQLAlchemy, `request_logs` table auto-created via lifespan `init_db()`, fire-and-forget via BackgroundTasks) + config.yaml wired + Docker (GPU passthrough, HF cache mount, postgres:15). Logs confirmed landing in Postgres. Only remaining: pytest-format integration tests (AC #9). |

## Decisions Made

| # | Decision | Date | Rationale |
|---|----------|------|-----------|
| 1 | Python 3.14 (spec said 3.11+) | 2026-06-13 | Latest toolchain; pinned in `.python-version` / `pyproject.toml`. |
| 2 | `uv` for dependency management | 2026-06-13 | Fast, reproducible; project convention (CLAUDE.md). |
| 3 | vLLM 0.22.x as inference backend | 2026-06-13 | Reference engine for Phase 1 before custom engine in Phase 2. |
| 4 | Model: Qwen2.5-7B-Instruct-AWQ | 2026-06-13 | Llama-3-8B needs HuggingFace token (gated). Qwen2.5-7B (fp16) failed GPU load. AWQ variant loads cleanly on RTX 5080 within 16GB VRAM. |
| 5 | `apply_chat_template` for prompt formatting | 2026-06-13 | Replaced naive string-join prompt with HuggingFace tokenizer template. Required for correct instruction-following behavior. |
| 6 | Adopted plan evolution v2: W2/W3/W5/W6 specs regenerated; re-evaluation gates added before W11 and W15 | 2026-07-06 | Strategic review (`original_artifacts/plan_evolution_v2_2026-07.md`): RAG commoditized → 1 week; new agentic-serving week; gateway+deployment merged (senior-skill overlap); buffer+OSS week added (blog #1 + upstream issue). Later specs deliberately left for just-in-time re-evaluation. |

## Blockers / Issues

| Issue | Week | Status | Resolution |
|-------|------|--------|------------|
| RTX 5080 is Blackwell (sm_120) — confirm vLLM/PyTorch CUDA build supports it | 1 | Resolved | vLLM loads Qwen2.5-7B-Instruct-AWQ successfully on RTX 5080. CUDA build confirmed working. |
| Gated models require HuggingFace token | 1 | Resolved | Switched to Qwen2.5-7B-Instruct-AWQ (Apache 2.0, no token needed) instead of Llama. |
