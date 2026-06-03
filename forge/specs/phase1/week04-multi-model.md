# Week 4: Multi-Model Orchestration
> Phase: 1 | Project: Forge | Estimated Duration: 7 days

## Context

Weeks 1-3 built a single-model inference server with production RAG. This week, you build the multi-model orchestration layer: model registry, VRAM-aware routing, LoRA adapter hot-swap, and a request queue with backpressure. This is the "platform" differentiator — you're not serving one model, you're managing a fleet.

**Prerequisites**: Weeks 1-3 complete — inference server + RAG working.

**Builds on**: Extends the engine layer to manage multiple models and adapters.

## Learning Goals

- [ ] Understand VRAM budgeting — how to estimate memory for model weights + KV-cache + activations
- [ ] Understand LoRA adapters — how they work (low-rank matrices), why they're tiny, how to swap them
- [ ] Understand request routing patterns — load balancing, priority queues, backpressure
- [ ] Understand model lifecycle — cold (disk) → warm (CPU RAM) → hot (GPU VRAM)

## Implementation Goals

- [ ] Build model registry (track models, states, VRAM requirements, capabilities)
- [ ] Build VRAM-aware model loader (check available memory before loading)
- [ ] Implement model state machine: cold → warm → hot → unloading
- [ ] Build LoRA adapter manager (load base once, swap adapters per-request)
- [ ] Build request queue with Redis (priority levels, backpressure, timeouts)
- [ ] Build model router (route requests to appropriate model/adapter)
- [ ] Implement graceful model unloading (drain requests, then unload)
- [ ] Add model health checking (detect degraded models)

## Acceptance Criteria

1. **Registry API**: `GET /v1/admin/models` returns all models with their state (cold/warm/hot)
2. **Load model**: `POST /v1/admin/models/mistral-7b/load` transitions model from cold → hot
3. **Unload model**: `POST /v1/admin/models/mistral-7b/unload` frees VRAM (verified with nvidia-smi)
4. **VRAM check**: Loading a model that won't fit returns error with available/required VRAM
5. **LoRA swap**: Same base model serves different LoRA adapters based on request parameter
6. **Adapter latency**: LoRA adapter swap takes < 200ms (measured)
7. **Queue works**: Under load, requests queue in Redis; queue depth visible via API
8. **Backpressure**: When queue exceeds max depth, new requests get HTTP 429 with retry-after header
9. **Priority**: High-priority requests are served before low-priority ones
10. **Health check**: Unhealthy model (produces garbage) is detected and marked as degraded

## Validation Commands

```bash
# List models and states
curl http://localhost:8000/v1/admin/models | python -m json.tool

# Load a model
curl -X POST http://localhost:8000/v1/admin/models/mistral-7b/load
nvidia-smi  # Verify VRAM increased

# Unload model
curl -X POST http://localhost:8000/v1/admin/models/mistral-7b/unload
nvidia-smi  # Verify VRAM freed

# Try loading model that won't fit
curl -X POST http://localhost:8000/v1/admin/models/llama-70b/load
# Should return: {"error": "Insufficient VRAM", "available_gb": 14.2, "required_gb": 40.0}

# Test LoRA routing
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"model":"mistral-7b","messages":[...],"adapter":"code-assistant"}'
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"model":"mistral-7b","messages":[...],"adapter":"writing-helper"}'

# Test queue backpressure (send many concurrent requests)
for i in {1..50}; do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/v1/chat/completions \
    -d '{"model":"mistral-7b","messages":[{"role":"user","content":"Count to 100"}],"max_tokens":200}' &
done
wait
# Some should return 429 if queue is full

# Check queue depth
curl http://localhost:8000/v1/admin/queue/status

# Run tests
pytest tests/integration/test_multi_model.py -v
```

## Technical Implementation Details

### Component 1: Model Registry (Day 1-2)

**File: `src/forge/registry.py`**

```python
class ModelState(Enum):
    COLD = "cold"       # On disk only
    WARMING = "warming" # Loading to CPU/GPU
    HOT = "hot"         # In VRAM, ready to serve
    DRAINING = "draining" # Serving remaining requests before unload
    ERROR = "error"     # Failed to load or unhealthy

class ModelInfo:
    name: str
    path: str           # HuggingFace ID or local path
    state: ModelState
    vram_required_gb: float
    max_context_length: int
    adapters: list[str] # Available LoRA adapter names
    loaded_at: datetime | None
    requests_served: int
    health: "healthy" | "degraded" | "unhealthy"

class ModelRegistry:
    def list_models() -> list[ModelInfo]
    def get_model(name: str) -> ModelInfo
    def register_model(config: ModelConfig) -> None
    def update_state(name: str, state: ModelState) -> None
```

Registry persists to PostgreSQL, loads from config on startup.

### Component 2: VRAM-Aware Model Loader (Day 2-3)

**File: `src/forge/loader.py`**

```python
class ModelLoader:
    def get_available_vram() -> float:
        # Use pynvml to query actual free VRAM
        
    def estimate_vram_needed(model_path: str, dtype: str) -> float:
        # Estimate: param_count * bytes_per_param + KV-cache buffer + activation buffer
        # FP16: params * 2 bytes, INT4: params * 0.5 bytes
        
    def can_load(model_name: str) -> tuple[bool, str]:
        # Check available vs required, return (can_load, reason)
        
    async def load_model(model_name: str) -> None:
        # 1. Check VRAM availability
        # 2. Set state to WARMING
        # 3. Load model (vLLM engine initialization)
        # 4. Set state to HOT
        # 5. On failure: set state to ERROR, log reason
        
    async def unload_model(model_name: str) -> None:
        # 1. Set state to DRAINING
        # 2. Wait for active requests to complete (with timeout)
        # 3. Delete model from vLLM engine
        # 4. Force CUDA garbage collection
        # 5. Set state to COLD
```

Use `pynvml` library for GPU memory queries. Add dependency: `uv add pynvml`

### Component 3: LoRA Adapter Manager (Day 3-4)

**File: `src/forge/adapters.py`**

How LoRA serving works:
- Base model loaded once (consumes most VRAM)
- LoRA adapters are tiny (few MB) — just low-rank delta matrices
- On request: merge adapter weights with base model temporarily
- vLLM supports multi-LoRA serving natively — use `--enable-lora` flag

```python
class AdapterManager:
    def __init__(self, base_model_engine):
        self.adapters: dict[str, str] = {}  # name -> path
        
    def register_adapter(self, name: str, path: str) -> None:
        # Validate adapter is compatible with base model
        
    def list_adapters(self) -> list[str]:
        # Return registered adapter names
        
    async def serve_with_adapter(self, request, adapter_name: str):
        # Route to vLLM with lora_request parameter
```

For this to work, you need LoRA adapters. Options:
- Download pre-made adapters from HuggingFace
- Fine-tune your own in Week 17 (for now, use public ones)
- Use adapters like: `predibase/customer_support_adapter` or similar

### Component 4: Request Queue (Day 4-5)

**File: `src/forge/queue.py`**

Redis-based priority queue:

```python
class RequestQueue:
    PRIORITY_HIGH = 0
    PRIORITY_NORMAL = 5
    PRIORITY_LOW = 10
    MAX_QUEUE_DEPTH = 100
    REQUEST_TIMEOUT = 60  # seconds
    
    async def enqueue(self, request: InferenceRequest, priority: int) -> str:
        # Add to Redis sorted set (score = priority * 1000 + timestamp)
        # If queue full, raise BackpressureError
        # Return request_id
        
    async def dequeue(self) -> InferenceRequest | None:
        # Pop highest priority (lowest score) from sorted set
        # Set processing timeout
        
    async def get_status(self) -> QueueStatus:
        # Return: depth, oldest_request_age, priority_breakdown
        
    async def cancel(self, request_id: str) -> None:
        # Remove from queue if still pending
```

Worker loop that drains the queue and dispatches to appropriate model/adapter.

### Component 5: Model Router (Day 5-6)

**File: `src/forge/router.py`**

```python
class ModelRouter:
    async def route(self, request: ChatCompletionRequest) -> ModelInfo:
        # 1. Check if requested model is loaded (HOT state)
        # 2. If not loaded, check if it CAN be loaded (VRAM available)
        # 3. If can load, enqueue model load + queue the request
        # 4. If can't load, check for fallback model
        # 5. Route to the resolved model
        
    def select_adapter(self, request) -> str | None:
        # If request specifies adapter, validate it exists
        # If no adapter specified, use base model
```

### Component 6: Health Checking (Day 6)

**File: `src/forge/health.py`**

Health check strategy:
- Periodic probe: send a simple prompt ("Hello") every 30 seconds to each loaded model
- Check: response is non-empty, latency is reasonable, no CUDA errors
- If 3 consecutive failures: mark model as degraded
- If degraded for 5 minutes: attempt unload + reload
- Expose health per-model via admin API

### Component 7: Admin API + Tests (Day 6-7)

**File: `src/forge/routes/admin.py`**

Endpoints:
- `GET /v1/admin/models` — list all models with full status
- `POST /v1/admin/models/{name}/load` — load model to GPU
- `POST /v1/admin/models/{name}/unload` — unload model from GPU
- `GET /v1/admin/queue/status` — queue depth and stats
- `GET /v1/admin/health` — overall platform health

## If You Get Stuck

**vLLM multi-model**: vLLM can only serve one model per engine instance. For multi-model, you need multiple engine instances. Use a process pool or run multiple vLLM workers.

**LoRA adapters not available**: Download any small LoRA from HuggingFace. Or create a dummy adapter using PEFT: `peft.get_peft_model(base, LoraConfig(r=8))` and save it.

**VRAM estimation inaccurate**: Start conservative. Measure actual VRAM after loading with `pynvml`. Build a calibration table: model X at dtype Y uses Z GB.

**Simplified fallback**: If multi-process vLLM is complex, start with a single model + multiple LoRA adapters. That alone is impressive and is the real production pattern.

## Agent Handoff Template

```
I'm on Week 4 of Forge — building multi-model orchestration.
Spec: /Users/jmalviya/Documents/zz/dev/plan_00/forge/specs/phase1/week04-multi-model.md
Current state: Working inference server with RAG (Weeks 1-3). Single model serving.
I need to add: model registry, VRAM-aware loading/unloading, LoRA adapter hot-swap, Redis request queue with priority and backpressure.
Key constraint: 16GB VRAM on RTX 5080.
Codebase: [path to forge/src/]
```

## Out of Scope

- Auto-scaling based on traffic (Week 15, K8s operator)
- Model download/pull from registry (keep local for now)
- A/B testing between models (Week 3 of Anvil project)
- Authentication for admin endpoints (Week 5)
- Distributed model serving across multiple GPUs (not applicable — single GPU)
