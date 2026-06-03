# Week 1: Inference Server
> Phase: 1 | Project: Forge | Estimated Duration: 7 days

## Context

This is the very first week of the Forge project. Nothing exists yet. The goal is to go from zero to a working GPU inference server that serves an LLM with an OpenAI-compatible API, containerized in Docker. This gives you a foundation that everything else builds on, and an immediate confidence boost — you'll have a working demo by day 5.

**Prerequisites**: ASUS ROG Strix with Ubuntu installed, internet connection for downloading models.

## Learning Goals

- [ ] Understand how vLLM loads a model into GPU memory and manages inference
- [ ] Understand the OpenAI Chat Completions API format (request/response structure)
- [ ] Understand Docker GPU passthrough (nvidia-container-toolkit)
- [ ] Understand VRAM constraints — what fits in 16GB and what doesn't
- [ ] Understand Server-Sent Events (SSE) for token streaming

## Implementation Goals

- [ ] Set up Ubuntu development environment with CUDA, Docker GPU support
- [ ] Build FastAPI server wrapping vLLM engine
- [ ] Implement OpenAI-compatible `/v1/chat/completions` endpoint (streaming + non-streaming)
- [ ] Implement `/v1/models` endpoint listing available models
- [ ] Implement `/health` endpoint with model load status
- [ ] Add request logging to PostgreSQL (request metadata, latency, token counts)
- [ ] Containerize with Docker (GPU-enabled)
- [ ] Add basic model configuration via YAML file
- [ ] Write integration test that verifies end-to-end inference

## Acceptance Criteria

1. **Environment**: `nvidia-smi` shows RTX 5080, `nvcc --version` shows CUDA 12.x
2. **Server starts**: `python -m forge.server` loads model and prints "Ready" within 90 seconds
3. **Inference works**: `curl -X POST localhost:8000/v1/chat/completions -d '{"model":"mistral-7b","messages":[{"role":"user","content":"Hello"}]}'` returns a valid response
4. **Streaming works**: Same curl with `"stream": true` returns SSE chunks with `data: {...}` format
5. **Models endpoint**: `curl localhost:8000/v1/models` returns JSON list of available models
6. **Health check**: `curl localhost:8000/health` returns `{"status": "healthy", "models_loaded": ["mistral-7b"]}`
7. **Logging**: After 5 requests, `SELECT count(*) FROM request_log` returns 5 in PostgreSQL
8. **Docker**: `docker compose up` starts the full stack (server + postgres) and model loads successfully
9. **Integration test**: `pytest tests/integration/test_inference.py` passes
10. **VRAM**: During inference, `nvidia-smi` shows VRAM usage (model loaded on GPU)

## Validation Commands

```bash
# Environment check
nvidia-smi
nvcc --version
python --version  # Should be 3.11+
docker --version
docker compose version

# Start services
docker compose up -d

# Wait for model to load (watch logs)
docker compose logs -f forge-server | grep -m1 "Ready"

# Test health
curl http://localhost:8000/health

# Test models list
curl http://localhost:8000/v1/models | python -m json.tool

# Test inference (non-streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mistral-7b","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50}' | python -m json.tool

# Test inference (streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mistral-7b","messages":[{"role":"user","content":"Write a haiku"}],"max_tokens":50,"stream":true}'

# Test logging
docker compose exec postgres psql -U forge -c "SELECT count(*) FROM request_log;"

# Run integration tests
pytest tests/integration/test_inference.py -v

# Check GPU utilization during inference
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 1
```

## Technical Implementation Details

### Component 1: Environment Setup (Day 1-2)

**Steps:**
1. Install NVIDIA drivers: `sudo apt install nvidia-driver-560` (or latest for RTX 5080)
2. Install CUDA toolkit 12.x from NVIDIA's repo
3. Install nvidia-container-toolkit for Docker GPU support
4. Install Python 3.11+ with `uv` package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`
5. Create project: `uv init forge && cd forge`
6. Add dependencies: `uv add fastapi uvicorn vllm pydantic sqlalchemy asyncpg python-dotenv pyyaml`

**Key files:**
```
forge/
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── config.yaml
├── src/
│   └── forge/
│       ├── __init__.py
│       ├── server.py          # FastAPI app
│       ├── engine.py          # vLLM wrapper
│       ├── models.py          # Pydantic models (OpenAI format)
│       ├── config.py          # Config loading
│       └── db.py              # PostgreSQL logging
├── tests/
│   └── integration/
│       └── test_inference.py
└── .env
```

### Component 2: FastAPI Server (Day 3-4)

**File: `src/forge/server.py`**

Core structure:
- FastAPI app with lifespan handler (load model on startup, cleanup on shutdown)
- POST `/v1/chat/completions` — accepts OpenAI format, returns OpenAI format
- GET `/v1/models` — returns list of loaded models
- GET `/health` — returns server + model health status
- Middleware: request timing, request ID generation, logging

**File: `src/forge/engine.py`**

Core structure:
- Class `ForgeEngine` that wraps vLLM's `AsyncLLMEngine`
- Methods: `generate(prompt, params) -> str`, `generate_stream(prompt, params) -> AsyncIterator[str]`
- Model loading on init with config-driven model selection
- Graceful handling of CUDA OOM (catch, log, return 503)

**File: `src/forge/models.py`**

Implement Pydantic models matching OpenAI API:
- `ChatCompletionRequest` (messages, model, max_tokens, temperature, stream, etc.)
- `ChatCompletionResponse` (id, object, created, model, choices, usage)
- `ChatCompletionChunk` (for streaming — delta format)
- `ModelList`, `ModelInfo`

### Component 3: Configuration (Day 4)

**File: `config.yaml`**
```yaml
server:
  host: "0.0.0.0"
  port: 8000

models:
  - name: "mistral-7b"
    path: "mistralai/Mistral-7B-Instruct-v0.3"  # or appropriate version
    max_model_len: 4096
    gpu_memory_utilization: 0.85
    dtype: "float16"

database:
  url: "postgresql+asyncpg://forge:forge@localhost:5432/forge"

logging:
  level: "INFO"
```

### Component 4: Request Logging (Day 5)

**File: `src/forge/db.py`**

- Async SQLAlchemy with asyncpg driver
- Table `request_log`: id, timestamp, model, messages (JSONB), response_tokens, total_tokens, latency_ms, status_code
- Non-blocking logging (don't slow down inference to write logs)
- Connection pooling

### Component 5: Docker (Day 5-6)

**File: `Dockerfile`**
- Base image: `nvidia/cuda:12.4.0-runtime-ubuntu22.04` (or appropriate for RTX 5080)
- Install Python, uv, project dependencies
- Multi-stage build to keep image small
- Non-root user for security

**File: `docker-compose.yml`**
- Service `forge-server`: GPU-enabled container, port 8000, depends on postgres
- Service `postgres`: PostgreSQL 16, persistent volume, init script for schema
- Shared network
- Environment variables from .env

### Component 6: Integration Test (Day 6-7)

**File: `tests/integration/test_inference.py`**

Tests (using httpx async client):
- `test_health_endpoint` — returns 200 with healthy status
- `test_models_list` — returns at least one model
- `test_chat_completion` — returns valid response with content
- `test_chat_completion_streaming` — returns SSE chunks, last chunk has finish_reason
- `test_request_logging` — after a request, log entry exists in DB
- `test_invalid_model` — returns 404 for non-existent model
- `test_empty_messages` — returns 400 for invalid input

## If You Get Stuck

**Model won't load (CUDA OOM)**:
- Try a smaller model: `mistralai/Mistral-7B-Instruct-v0.3` with `gpu_memory_utilization: 0.80`
- Try quantized: use a GPTQ 4-bit variant
- Check nothing else is using GPU: `nvidia-smi` should show minimal usage before loading

**Docker GPU not working**:
- Verify: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
- If fails: reinstall nvidia-container-toolkit, restart Docker daemon

**vLLM installation issues**:
- vLLM requires specific CUDA versions — check their docs for compatibility with RTX 5080
- Fallback: use `pip install vllm` in a conda environment if uv has issues

**Simplified fallback** (if vLLM has driver issues):
- Use `transformers` + `accelerate` directly for inference (slower but always works)
- Replace `ForgeEngine` internals, keep API layer identical

## Agent Handoff Template

```
I'm starting Week 1 of the Forge project — building a GPU inference server.
The spec is at: /Users/jmalviya/Documents/zz/dev/plan_00/forge/specs/phase1/week01-inference-server.md
Nothing has been built yet. I need a working FastAPI server that:
1. Wraps vLLM for GPU inference
2. Exposes OpenAI-compatible /v1/chat/completions (streaming + non-streaming)
3. Logs requests to PostgreSQL
4. Runs in Docker with GPU support
Target model: Mistral-7B-Instruct
Hardware: RTX 5080 (16GB VRAM), Ubuntu
Please implement the full project structure and all components described in the spec.
```

## Out of Scope

- Authentication (Week 5)
- Multi-model support beyond config (Week 4)
- Rate limiting (Week 5)
- RAG/retrieval (Week 2)
- Custom inference engine (Week 8-9, Phase 2)
- Kubernetes deployment (Week 6)
- Monitoring/metrics (Week 13)
