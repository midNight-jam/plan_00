# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

A structured 10-month AI engineering study plan split across four parallel projects:

| Project | Track | Focus |
|---------|-------|-------|
| `forge/` | AI Platform + Inference | GPU inference server, vLLM, custom batching, quantization |
| `anvil/` | AI Infrastructure | Distributed systems, K8s, reliability, SRE |
| `crucible/` | Training + Alignment | RLHF, DPO, reward models, safety evaluation |
| `conduit/` | ML Systems | Data pipelines, feature store, MLOps, monitoring |

The chosen combination is **Forge + Anvil** (Months 1–5 and 6–10).

Only **Forge** has active code. The others contain planning docs (`MANIFEST.md`, `PLAN.md`, `progress.md`) but no implementation yet.

## Starting a Session on Any Week

Before touching code, always read two files:
1. `[project]/MANIFEST.md` — architecture, tech stack, decisions log
2. `[project]/progress.md` — current week and completion status

Then open the relevant week spec at `[project]/specs/phase[X]/week[NN]-[name].md`.

## Forge — Commands

Forge uses **uv** for dependency management (not pip, not poetry).

```bash
# Install deps
cd forge && uv sync

# Run the server
cd forge && uv run uvicorn src.forge.server:app --reload

# Run all tests
cd forge && uv run pytest

# Run a single test file
cd forge && uv run pytest tests/integration/test_inference.py -v

# Run with Docker
cd forge && docker-compose up --build
```

## Forge — Architecture

```
forge/
├── src/forge/
│   ├── server.py     # FastAPI app, OpenAI-compatible routes, SSE streaming
│   ├── engine.py     # Inference engine (vLLM wrapper, custom batching)
│   ├── models.py     # Pydantic request/response schemas
│   ├── config.py     # Config loading from config.yaml + .env
│   └── db.py         # SQLAlchemy async (asyncpg), PostgreSQL
├── tests/
│   └── integration/  # Integration tests (hit real services, not mocks)
├── config.yaml       # Runtime config (models, limits, infra endpoints)
├── .env              # Secrets (not committed)
└── docker-compose.yml # PostgreSQL, Qdrant, Redis, the Forge server
```

**Data flow**: request → FastAPI (`server.py`) → model router → inference engine (`engine.py`) → vLLM → streamed SSE response. PostgreSQL stores request logs and model registry metadata; Qdrant for vector search (RAG); Redis for job queue and rate limiting.

## Forge — Stack

- Python 3.14+, FastAPI, vLLM, PyTorch 2.x
- asyncpg + SQLAlchemy (async) for PostgreSQL
- Pydantic v2 for all schemas
- uv for package management
- Docker + K3s + Helm for deployment
- Prometheus + Grafana + OpenTelemetry for observability

## Planning Conventions

Each week spec contains **Acceptance Criteria** — 10 testable items that define "done." A week is not complete until all 10 pass. After completing a week, mark it in `[project]/progress.md` and log the date in the Completion Log table.

Specs are immutable once a week is marked complete. Bugs found later go in a new spec, not an edit to the old one.

## Hardware Context

ASUS ROG Strix SCAR 16, RTX 5080 (16GB VRAM), 32GB RAM, Ubuntu dual-boot. GPU is available locally — `nvidia-smi` should work. Design inference code around 16GB VRAM headroom.
