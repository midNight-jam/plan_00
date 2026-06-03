---
name: Forge Practical Build Plan
overview: "A 20-week (5-month) plan split into 3 phases: Platform Foundation (7 weeks), Inference Depth (7 weeks), and Advanced Systems (6 weeks, unchanged). Each phase has breathing room for deep understanding, experimentation, and production hardening. Targets AI Platform Engineer + Inference Specialist roles."
todos:
  - id: phase1-w1
    content: "Phase 1, Week 1: Environment setup + first inference server (vLLM, FastAPI, Docker GPU, OpenAI-compatible API)"
    status: pending
  - id: phase1-w2
    content: "Phase 1, Week 2: RAG pipeline — Qdrant, chunking strategies, embedding models, retrieval + reranking"
    status: pending
  - id: phase1-w3
    content: "Phase 1, Week 3: RAG production hardening — caching, evaluation (recall/precision), hybrid search, prompt engineering patterns"
    status: pending
  - id: phase1-w4
    content: "Phase 1, Week 4: Multi-model orchestration — model registry, VRAM-aware router, LoRA hot-swap, request queue + backpressure"
    status: pending
  - id: phase1-w5
    content: "Phase 1, Week 5: API gateway and platform patterns — auth, rate limiting, multi-tenancy, usage tracking, streaming middleware"
    status: pending
  - id: phase1-w6
    content: "Phase 1, Week 6: Deployment stack — Docker Compose, K3s + Helm chart, GPU scheduling, CI/CD, integration tests"
    status: pending
  - id: phase1-w7
    content: "Phase 1, Week 7: Consolidation — end-to-end testing, chaos scenarios, documentation, Phase 1 blog post, refactor"
    status: pending
  - id: phase2-w8
    content: "Phase 2, Week 8: Transformer internals study + naive inference implementation (no vLLM, raw PyTorch generate loop)"
    status: pending
  - id: phase2-w9
    content: "Phase 2, Week 9: Continuous batching — implement custom scheduler, understand prefill vs decode, benchmark against naive"
    status: pending
  - id: phase2-w10
    content: "Phase 2, Week 10: KV-cache and memory management — block allocator, eviction, GPU↔CPU swap, fragmentation profiling"
    status: pending
  - id: phase2-w11
    content: "Phase 2, Week 11: Speculative decoding + prefix caching + advanced scheduling (preemption, priority, fairness)"
    status: pending
  - id: phase2-w12
    content: "Phase 2, Week 12: Quantization pipeline — GPTQ/AWQ/NF4 comparison, TensorRT, quality gates, Pareto analysis"
    status: pending
  - id: phase2-w13
    content: "Phase 2, Week 13: Observability + load testing — Prometheus, OTel tracing, Grafana dashboards, stress tests, runbooks"
    status: pending
  - id: phase2-w14
    content: "Phase 2, Week 14: Performance engineering methodology — profiling, systematic bottleneck analysis, optimization iteration, Phase 2 blog post"
    status: pending
  - id: phase3-w15
    content: "Phase 3, Week 15: Kubernetes operator — CRD design, reconciliation, GPU-aware scheduling, rolling updates"
    status: pending
  - id: phase3-w16
    content: "Phase 3, Week 16: Triton GPU kernels — fused RMSNorm, tiled attention, fused MLP, benchmarking harness"
    status: pending
  - id: phase3-w17
    content: "Phase 3, Week 17: Training/alignment — custom training loop, LoRA fine-tuning, DPO, reward model basics"
    status: pending
  - id: phase3-w18-20
    content: "Phase 3, Weeks 18-20: Portfolio polish — 4 blog posts, benchmark report, ADRs, demo video, README, open-source release"
    status: pending
isProject: false
---

# Forge: The Practical Build Plan (20 Weeks / 5 Months)

## Philosophy: "Make It Work, Make It Right, Make It Fast"

- **Phase 1 (Weeks 1-7)**: Make It Work — Build a production-quality AI platform with depth
- **Phase 2 (Weeks 8-14)**: Make It Right — Deep inference understanding and optimization mastery
- **Phase 3 (Weeks 15-20)**: Make It Fast — Advanced systems, kernels, training, and portfolio polish

Each phase produces a DEPLOYABLE, DEMOABLE artifact. You never have "nothing to show." The extra weeks give you time to truly understand WHY things work, not just HOW to make them run.

---

## Phase 1: AI Platform Foundation (Weeks 1-7)

**Goal**: Build a working, production-quality local AI platform. Not a weekend hack — a system with proper error handling, testing, documentation, and deployment. By end of Week 7, you have a polished platform that could genuinely be used by a team.

### Week 1: Get Running Fast (The Confidence Builder)

**Day 1-2**: Environment setup
- Ubuntu dual-boot with NVIDIA drivers, CUDA toolkit, Docker with GPU support (nvidia-container-toolkit)
- Python environment with uv (modern package manager, not pip)
- Pull and run a model locally with vLLM (just get it working — understand what it does)
- Experiment: load different model sizes, see what fits in 16GB VRAM, understand the constraints

**Day 3-5**: Build your first serving layer
- FastAPI server that wraps vLLM (yes, start with the "wrapper" — but you'll replace the guts later)
- OpenAI-compatible API (POST /v1/chat/completions, streaming with SSE)
- Serve Mistral-7B or Llama-3-8B (whatever fits in 16GB VRAM)
- Basic request logging to PostgreSQL (timestamp, model, tokens, latency)
- Docker containerize the whole thing

**Day 6-7**: Multi-model support
- Config file that defines available models
- API routes for model selection
- Basic model loading/unloading (crude but functional)
- Write your first integration test (start a model, send a request, verify response)

**Deliverable**: A Docker-based API server that serves LLM inference with OpenAI-compatible endpoints. Push to GitHub with a clean README.

**What you learn**: How model serving actually works end-to-end, what VRAM constraints feel like, Docker GPU passthrough.

---

### Week 2: The Data Layer (RAG Pipeline — Core)

**Why this matters for interviews**: Every AI platform company needs people who understand retrieval + inference together. RAG is the most deployed pattern in production AI.

**Build**:
- Set up Qdrant (vector database) in Docker
- Build an ingestion pipeline:
  - Document chunking (implement multiple strategies: fixed-size, semantic, recursive)
  - Understand WHY different chunking strategies matter (demonstrate with examples)
  - Embedding generation using a local embedding model (e.g., BGE or E5)
  - Vector storage with metadata filtering
- Build the retrieval + generation pipeline:
  - Query embedding -> vector search -> context assembly -> LLM generation
  - Implement reranking (cross-encoder or LLM-based)
  - Handle context window management (what if retrieved docs exceed context?)
- API endpoints: upload document, query with retrieval, list documents
- Index a meaningful corpus (e.g., PyTorch docs, CUDA documentation, or Kubernetes docs)

**Deliverable**: A working RAG system with document ingestion, retrieval, and generation.

**What you learn**: Embedding models, vector search mechanics, chunking tradeoffs, context window management.

---

### Week 3: RAG Production Hardening (The Depth Most People Skip)

**Why this week exists**: Everyone builds a RAG demo. Almost nobody makes it production-quality. This week is where you differentiate.

**Build**:
- Semantic caching layer:
  - Cache frequent queries and their results (Redis)
  - Implement semantic cache: if a new query is similar enough to a cached query, return cached result
  - Measure cache hit rate and latency improvement
- Evaluation pipeline (critical — shows you think like a production engineer):
  - Build a test dataset with known-good answers
  - Measure retrieval quality: Recall@K, Precision@K, MRR (Mean Reciprocal Rank)
  - Measure generation quality: faithfulness (does the answer match the retrieved context?)
  - Automated regression testing: new chunking strategy must not reduce recall
- Hybrid search:
  - Combine vector search (semantic) with BM25 (keyword) using Reciprocal Rank Fusion
  - Show when hybrid beats pure vector (it almost always does)
- Prompt engineering patterns:
  - System prompt management (versioned, A/B testable)
  - Structured output (JSON mode, function calling patterns)
  - Chain-of-thought prompting with retrieval
- Document processing pipeline:
  - Handle PDFs, markdown, code files (not just plain text)
  - Metadata extraction and filtering
  - Incremental indexing (don't re-index everything on each document add)

**Deliverable**: A production-grade RAG system with evaluation metrics, caching, and hybrid search. Published evaluation results showing retrieval quality.

**What you learn**: How to evaluate AI systems rigorously, caching patterns, the gap between "demo" and "production."

---

### Week 4: Multi-Model Orchestration (The Platform Story)

**This is where you start differentiating from tutorials.**

**Build**:
- Model Registry:
  - Track available models (name, size, VRAM requirement, quantization level, capabilities)
  - Model states: loaded (GPU), cached (CPU RAM), available (disk)
  - Health checks per model (not just "is it running" but "is it generating coherent output")
  - Model metadata: supported context length, supported tasks, performance characteristics
- Smart Model Router:
  - Route requests to appropriate model based on request type / user preference
  - Implement model loading queue (don't load 3 models simultaneously — sequence them)
  - Memory-aware loading (check available VRAM before loading, estimate KV-cache needs)
  - Fallback chains: if primary model is busy, route to secondary
- LoRA Adapter Support:
  - Load a base model once
  - Swap LoRA adapters per-request (using PEFT library)
  - Measure adapter swap latency, optimize it
  - Serve concurrent requests with different adapters on same base model
- Request Queue:
  - Redis-based request queue with priority levels
  - Backpressure: reject requests when queue is full (with proper HTTP 429)
  - Request timeout handling
  - Dead letter queue for failed requests (retry later or alert)

**Deliverable**: A platform that intelligently manages multiple models and adapters on limited VRAM. Demo: serve 1 base model + 3 LoRA adapters, show routing between them.

**What you learn**: Resource management under constraints, the real-world pattern of base model + adapters, queuing theory basics.

---

### Week 5: API Gateway and Platform Patterns

**Why this week exists**: The difference between a "project" and a "platform" is everything around the core inference. This is what makes senior hiring managers take notice.

**Build**:
- Authentication and API keys:
  - API key management (create, revoke, rotate)
  - Per-key rate limiting and usage tracking
  - JWT-based auth for more complex scenarios
- Rate limiting:
  - Token bucket algorithm (implement yourself — it's a great interview question)
  - Per-user and per-model rate limits
  - Graceful degradation under load (queue vs reject vs throttle)
- Multi-tenancy:
  - Tenant isolation (separate API keys, separate usage tracking)
  - Per-tenant model access control (tenant A can use model X but not Y)
  - Usage metering and billing-ready metrics (tokens consumed per tenant)
- Streaming middleware:
  - SSE streaming with proper error handling mid-stream
  - Streaming token counting (track usage while streaming)
  - Client disconnect detection (cancel inference when client disconnects — saves GPU)
- Request/Response middleware:
  - Input validation and sanitization
  - Output filtering (basic content safety)
  - Request ID propagation for tracing
  - Structured logging with correlation IDs

**Deliverable**: A platform with proper auth, rate limiting, multi-tenancy, and production middleware. This looks like a real product, not a side project.

**What you learn**: Platform engineering patterns that every production AI system needs.

---

### Week 6: Deployment and Infrastructure

**Build**:
- Docker Compose stack (development):
  - Inference server (your FastAPI app)
  - Qdrant (vector DB)
  - PostgreSQL (metadata, logs, usage)
  - Redis (request queue, caching)
  - Prometheus + Grafana (basic monitoring)
- Kubernetes deployment (K3s — production-like):
  - Write Helm chart for your inference server
  - GPU resource requests/limits in pod spec
  - Liveness and readiness probes (important: readiness = model is loaded and healthy)
  - ConfigMap for model registry
  - Secrets management for API keys
  - Horizontal Pod Autoscaler (custom metric: queue depth)
  - PersistentVolumeClaim for model cache (don't re-download models on restart)
- CI/CD pipeline (GitHub Actions):
  - Lint (ruff), type check (mypy), unit tests
  - Build Docker image with layer caching
  - Integration test: spin up stack, load model, run test queries, verify outputs
  - Automated benchmark on PR (basic: latency/throughput regression detection)
  - Auto-publish Docker image on tag
- Infrastructure as Code:
  - All deployment configs in repo (no manual steps)
  - Environment-specific values (dev/staging)
  - One-command setup: `make dev` (Docker Compose) or `make k8s` (K3s)

**Deliverable**: One-command deployment, CI/CD pipeline, Helm chart. Push a commit and watch it build, test, and deploy.

**What you learn**: Production deployment patterns for GPU workloads, health checking for ML services, GPU scheduling in K8s.

---

### Week 7: Phase 1 Consolidation (Most People Skip This — Don't)

**Why this week exists**: A week to step back, harden everything, write tests, fix tech debt, and produce your first blog post. This is what separates "portfolio project" from "production system."

**Build**:
- End-to-end integration test suite:
  - Happy path: full request lifecycle from API key creation to inference
  - Error paths: model OOM, timeout, invalid input, auth failure
  - Performance assertion: "this endpoint responds in < X ms under Y concurrent requests"
- Chaos scenarios (manual, documented):
  - What happens when GPU OOM occurs? (graceful degradation)
  - What happens when Redis goes down? (queue fallback)
  - What happens when model produces garbage? (health check catches it)
  - Document each scenario and your system's behavior
- Code quality pass:
  - Type hints everywhere (mypy strict)
  - Docstrings on public APIs
  - Remove dead code, clean up TODOs
  - Consistent error handling patterns
- First blog post draft:
  - "Building a Multi-Model AI Platform from Scratch: Architecture Decisions"
  - Cover: why you built it, architecture overview, key tradeoffs, what you'd do differently
- Architecture Decision Records (first 3):
  - ADR-001: Why FastAPI over alternatives
  - ADR-002: Why Qdrant over Pinecone/Weaviate/pgvector
  - ADR-003: Why Redis for request queue over alternatives

**Deliverable**: A polished, tested, documented Phase 1 system. Clean git history. First blog post published (or ready to publish).

**Milestone check**: At this point, your GitHub repo already stands out. It has: clean code, tests, CI/CD, deployment configs, documentation, and a blog post. Most portfolio projects never reach this level.

---

## Phase 2: Inference Depth and Production Hardening (Weeks 8-14)

**Goal**: Go from "I use vLLM" to "I understand what vLLM does and can build the critical pieces myself." This phase transforms you from a platform user into an inference engineer.

### Week 8: Transformer Internals (Understand Before You Optimize)

**Why this week exists**: You cannot optimize what you don't understand. Before building a custom inference engine, you need to understand the forward pass at a code level.

**Build**:
- Implement a minimal transformer forward pass from scratch in PyTorch:
  - Multi-head attention (not using nn.MultiheadAttention — write the matmuls yourself)
  - Rotary Position Embedding (RoPE) — implement the rotation manually
  - RMSNorm (simpler than LayerNorm, used in Llama/Mistral)
  - SwiGLU activation (the MLP variant used in modern models)
  - Combine into a single transformer block
- Implement naive autoregressive generation:
  - The basic generate loop: forward pass -> sample token -> append -> repeat
  - Understand WHY this is slow (recomputing all attention for past tokens)
  - Implement basic KV-caching (the "aha moment" — cache past key/values, only compute new token's attention)
  - Measure: generation speed WITH vs WITHOUT KV-cache
- Load real model weights into your implementation:
  - Download Llama-3-8B or Mistral-7B weights
  - Map HuggingFace weight names to your implementation
  - Verify your implementation produces the same output as HuggingFace
- Profile the forward pass:
  - Use PyTorch profiler to see where time is spent
  - Identify: is it memory-bound or compute-bound?
  - Understand memory access patterns in attention

**Deliverable**: A from-scratch transformer inference implementation that loads real weights and generates text. Benchmarks showing KV-cache speedup. A profiling report showing bottlenecks.

**What you learn**: The complete inference pipeline at the code level. After this week, you can whiteboard the entire forward pass in an interview.

---

### Week 9: Continuous Batching (The Core Innovation)

**This is where you start replacing vLLM's internals with your own understanding.**

**Build**:
- First, implement static batching (the naive approach):
  - Pad all sequences to same length
  - Run entire batch through model
  - Wait for longest sequence to finish before accepting new requests
  - Measure wasted compute (GPU cycles spent on padding tokens)
- Then, implement continuous batching:
  - Maintain a running batch and a waiting queue
  - Each iteration: add new sequences to the batch if GPU memory allows
  - Remove completed sequences immediately (don't wait for the longest one)
  - Implement padding-free batching (concatenate sequences, use position offsets)
  - Handle variable sequence lengths in the attention mask
- Study vLLM's scheduler (read the code — `vllm/core/scheduler.py`):
  - How it tracks sequence groups
  - How it decides what to schedule next
  - How it handles preemption (when memory runs out)
- Build comparison benchmarks:
  - Static batching vs your continuous batching vs vLLM
  - Measure: throughput (tokens/sec), latency (time-to-first-token, inter-token), GPU utilization
  - Test with realistic load patterns (not just sustained max throughput)

**Key insight you'll gain**: WHY continuous batching matters. You'll see the 3-5x difference with your own eyes, and you'll understand it because you built the naive version first.

**Deliverable**: A custom scheduler implementation with benchmarks against static batching and vLLM. A write-up explaining the tradeoffs with diagrams.

---

### Week 10: KV-Cache and Memory Management (The Deep Dive)

**Build**:
- Block-based KV-Cache Manager (simplified PagedAttention):
  - Divide GPU memory into fixed-size blocks
  - Implement a block allocator (like malloc for GPU memory blocks)
  - Map logical sequences to physical blocks (the page table analogy)
  - Track memory usage per-sequence
  - Implement copy-on-write for parallel sampling (beam search uses this)
- Eviction and swap:
  - When VRAM is full, preempt lowest-priority sequences
  - Move preempted sequence's KV-cache to CPU RAM (swap out)
  - Bring it back when the sequence can be scheduled again (swap in)
  - Measure swap latency and its impact on tail latency
- GPU Memory Profiling tool:
  - Build a tool that shows: where is VRAM being used? (model weights, KV-cache, activations, fragmentation waste)
  - Visualize memory allocation over time during a load test
  - Detect memory leaks (VRAM that's allocated but never freed)
  - Show fragmentation (allocated but unusable gaps)
- Model Offloading:
  - Implement CPU offloading for inactive models (keep weights in system RAM)
  - Measure swap latency: CPU-to-GPU model load time vs GPU-to-GPU (if you had multiple)
  - Implement predictive loading (pre-load model to GPU when its request queue grows)

**Deliverable**: Memory profiler tool + KV-cache manager with benchmarks showing near-zero fragmentation. Memory breakdown charts. Eviction policy comparison.

**What you learn**: The exact same memory management concepts Anthropic/OpenAI deal with at their clusters — just on one GPU instead of thousands.

---

### Week 11: Advanced Scheduling — Speculative Decoding and Prefix Caching

**Why this week exists**: These are the optimizations that current-gen serving systems are actively building. Knowing them puts you at the cutting edge.

**Build**:
- Speculative Decoding:
  - Load a small "draft" model (e.g., 1B params) alongside the large model
  - Draft model generates N candidate tokens quickly
  - Large model verifies all N tokens in a single forward pass (parallel verification)
  - Accept correct tokens, reject where draft diverged
  - Implement the acceptance/rejection probability correctly (this is the tricky part)
  - Measure: speedup vs quality. Under what conditions does speculative decoding help most?
- Prefix Caching:
  - Cache KV-cache blocks for common prefixes (system prompts, few-shot examples)
  - When a new request starts with a cached prefix, reuse the KV-cache (skip prefill for that prefix)
  - Implement prefix matching (find the longest cached prefix that matches the new request)
  - Measure: how much prefill time is saved with common system prompts?
- Advanced Scheduling Policies:
  - Preemption strategies: swap (save KV to CPU) vs recompute (discard KV, regenerate later)
  - When is recompute cheaper than swap? (short sequences vs long sequences)
  - Fairness: prevent starvation of low-priority requests
  - SLO-aware scheduling: guarantee time-to-first-token under X ms for premium tier
  - Priority inheritance: if a high-priority request depends on context from a lower-priority one

**Deliverable**: Working speculative decoding with benchmark showing speedup. Prefix caching with system-prompt reuse demo. Scheduling comparison document.

**What you learn**: The exact optimizations being actively developed at vLLM, TGI, and internal systems at AI labs.

---

### Week 12: Quantization and Model Optimization

**Build** (using existing tools, but understanding them deeply):
- Quantization comparison pipeline:
  - Take one model (e.g., Llama-3-8B) -> quantize with: GPTQ, AWQ, bitsandbytes (NF4), GGUF
  - For each method, understand the algorithm at a high level:
    - GPTQ: layer-wise quantization minimizing reconstruction error (Hessian-based)
    - AWQ: protect "salient" weights (those multiplied by large activations)
    - NF4: normalized float 4-bit (information-theoretically optimal for normal distributions)
  - Benchmark each: latency, throughput, VRAM usage, quality (perplexity on WikiText)
  - Build automated quality gates: reject quantization if perplexity increases > X%
- TensorRT-LLM compilation:
  - Convert a model to TensorRT engine
  - Benchmark TensorRT vs PyTorch eager vs torch.compile
  - Understand what graph optimizations TensorRT applies (layer fusion, kernel selection)
- torch.compile deep dive:
  - Enable torch.compile on your inference engine
  - Understand compilation modes: default, reduce-overhead, max-autotune
  - Measure warmup time vs steady-state speedup
  - Understand when torch.compile helps and when it doesn't (dynamic shapes problem)
- Build a "Model Optimization Pipeline":
  - Input: HuggingFace model ID + target hardware specs
  - Output: Optimized, quantized, ready-to-serve model with benchmark report
  - Automated selection: "given your hardware, here's the best quantization method"
  - Version tracking: which optimization was applied, reproducibility

**Deliverable**: Automated optimization pipeline + Pareto frontier analysis (quality vs speed vs memory chart for each quantization method). Published as a reproducible notebook.

---

### Week 13: Observability and Production Operations

**Build**:
- Comprehensive metrics (Prometheus):
  - Request metrics: latency histogram (prefill, decode, total), throughput, error rate, queue depth
  - GPU metrics: VRAM usage by category (weights, KV-cache, activations), utilization %, temperature, power
  - Model metrics: tokens/sec per model, cache hit rate, active sequences, preemption rate
  - Business metrics: requests per model, per tenant, per adapter, token consumption
- Distributed tracing (OpenTelemetry):
  - Trace request lifecycle: receive -> auth -> queue -> schedule -> prefill -> decode -> respond
  - Add span attributes: batch size, sequence length, cache hits, quantization method
  - Identify bottlenecks visually in trace waterfall (Jaeger UI)
- Grafana dashboards (4 dashboards):
  1. Operations: request rate, error rate, latency SLOs (% requests under target)
  2. GPU: memory waterfall (stacked area chart), utilization, thermal
  3. Models: per-model performance comparison, adapter usage
  4. Platform: tenant usage, rate limit hits, queue depth trends
- Load testing:
  - Build a load testing tool with realistic LLM request patterns:
    - Variable input lengths (10 to 4000 tokens)
    - Variable output lengths (streaming responses)
    - Bursty traffic (not just constant rate)
  - Run stress tests: find the breaking point (max throughput before p99 degrades)
  - Run soak tests: sustained moderate load for hours (detect memory leaks, gradual degradation)
- Alerting and runbooks:
  - Alert on: p99 > threshold, VRAM > 90%, error rate > 1%, model unhealthy, queue backing up
  - For each alert, write a runbook: what to check, how to diagnose, how to fix
  - Include: "GPU OOM Recovery", "High Latency Investigation", "Model Health Failure"

**Deliverable**: Full observability stack with dashboards, tracing, load tests, and runbooks. Record a debugging session showing how you identify and fix a performance issue.

---

### Week 14: Performance Engineering Methodology (The Senior Engineer Week)

**Why this week exists**: Being able to systematically find and fix performance problems is THE skill that separates senior from mid-level engineers. This week you develop a methodology.

**Build**:
- Systematic profiling toolkit:
  - PyTorch Profiler integration (trace view, kernel time, memory timeline)
  - nvidia-smi monitoring script (GPU utilization over time)
  - Custom timing decorators for your own code (identify Python overhead vs GPU time)
  - Memory snapshot tool (dump VRAM allocation at any point)
- Performance investigation methodology (document this):
  1. Define the metric (what are we optimizing? throughput? p99 latency? TTFT?)
  2. Establish baseline (reproducible benchmark)
  3. Profile (where is time spent?)
  4. Hypothesize (is it memory-bound? compute-bound? CPU overhead? scheduling?)
  5. Implement fix
  6. Measure improvement (same benchmark as baseline)
  7. Check for regressions (did we make something else worse?)
- Apply methodology to your own system:
  - Find 3 real performance issues in your platform
  - Fix each one with documented before/after measurements
  - Write up each as a "performance investigation" mini-post
- End-to-end optimization pass:
  - Profile the full request lifecycle
  - Reduce Python overhead (async correctly, avoid unnecessary copies)
  - Optimize tokenization (batch tokenize, use fast tokenizers)
  - Memory optimization (avoid peak memory spikes, pre-allocate where possible)
- Phase 2 blog post:
  - "Deep Diving into LLM Inference: What I Learned Building a Custom Serving Engine"
  - Include profiling screenshots, before/after numbers, methodology description

**Deliverable**: A documented performance engineering methodology + 3 real performance investigations with fixes. Phase 2 blog post. Your inference engine is now measurably faster than where it started.

**Milestone check**: At this point you can confidently answer any interview question about inference serving, memory management, batching strategies, and quantization. You've built it, measured it, and optimized it.

---

## Phase 3: Going Deep + Portfolio Moat (Weeks 15-20)

**Goal**: Now you've built and operated the platform, go deep on the topics that separate you from other candidates.

### Week 15: Kubernetes Operator (The Staff Engineer Signal)

**Why this matters**: Writing a K8s operator signals you think at the "platform for platforms" level. Very few candidates have this.

**Build** (using Python `kopf` framework — approachable for Python devs):
- Custom Resource Definition: `InferenceService`
  - Spec: model name, VRAM limit, replicas, scaling policy
  - Status: ready/loading/error, current VRAM, active requests
- Operator reconciliation logic:
  - Watch for new InferenceService CRDs → create serving pods
  - Handle pod failures → restart with backoff
  - Scale based on queue depth (custom HPA logic)
  - Rolling updates: deploy new model version with zero downtime
- GPU-aware scheduling:
  - Track available VRAM per node
  - Bin-pack models onto GPUs (fit multiple small models on one GPU)
  - Prevent over-scheduling (don't assign 20GB of models to 16GB GPU)

**Deliverable**: Working operator that manages model serving on K3s. Demo: create an InferenceService YAML → operator provisions everything → model starts serving.

---

### Week 16: Custom Kernels (The 1% Differentiator — Approachable Version)

**By now you understand WHY kernels matter because you've hit performance walls.**

**Build** (start with Triton — it's Python-like, much more approachable than raw CUDA):
- Triton kernel #1: Fused RMSNorm
  - Combine normalization + residual add into one kernel (avoid extra memory reads)
  - Benchmark vs PyTorch's separate operations
- Triton kernel #2: Custom attention (simplified FlashAttention concept)
  - Implement tiled attention that operates on blocks
  - Understand the memory hierarchy argument (why reading from HBM is slow)
- Triton kernel #3: Fused MLP
  - Combine gate projection + up projection + activation into one kernel
- Benchmarking harness:
  - Measure: TFLOPS achieved vs theoretical peak
  - Memory bandwidth utilization
  - Compare your kernels against PyTorch defaults

**Key insight**: You don't need to be a CUDA wizard. You need to demonstrate you UNDERSTAND the principles (memory hierarchy, fusion, occupancy) and can apply them. Triton lets you do this in Python syntax.

**Deliverable**: 3 Triton kernels with benchmarks + a blog post "My First Custom GPU Kernels: What I Learned About Memory Hierarchy"

---

### Week 17: Training and Alignment Basics (The Anthropic/OpenAI Hook)

**You don't need to build a full RLHF pipeline. But you need to demonstrate you understand it.**

**Build**:
- Fine-tuning pipeline:
  - LoRA fine-tuning on a small model (3B params or less)
  - Custom training loop (not HuggingFace Trainer — use raw PyTorch)
  - Mixed precision, gradient accumulation, proper evaluation
- DPO implementation (simpler than full RLHF, equally impressive):
  - Implement DPO loss from the paper
  - Train on a preference dataset (UltraFeedback or similar)
  - Evaluate with win-rate comparison
- Basic RLHF understanding demo:
  - Implement reward model training (Bradley-Terry)
  - Show you understand the full pipeline conceptually (diagram + code for key components)
  - You don't need full PPO — but implement the reward scoring part

**Deliverable**: A working DPO fine-tuning pipeline + reward model. A technical write-up explaining RLHF vs DPO tradeoffs with code examples.

---

### Weeks 18-20: Portfolio Polish and Storytelling

**This is where most engineers fail. They build great things and present them poorly.**

**Build**:
- Technical blog posts (3-4, not 6 — quality over quantity):
  1. "Building a Multi-Model Inference Platform from Scratch" (the platform story)
  2. "Understanding Continuous Batching: Why Your LLM Server is Leaving Performance on the Table" (the deep dive)
  3. "A Practical Guide to LLM Quantization: Benchmarking Every Method on Consumer Hardware" (the data-driven post)
  4. "Writing My First GPU Kernels with Triton" (the learning journey — interviewers love humility + growth)
- Benchmark report:
  - Your platform vs Ollama vs vLLM on same hardware
  - Multiple dimensions: throughput, latency, memory efficiency, multi-model performance
  - Reproducible (provide scripts anyone can run)
- Architecture Decision Records (5-7):
  - Why block-based KV-cache over pre-allocated?
  - Why K3s over minikube?
  - Why Python kopf over Go for the operator?
  - Why DPO over full RLHF for the demo?
  - Each one: Context → Decision → Tradeoffs → Alternatives
- README as a pitch:
  - Problem → Architecture diagram → Key results (numbers!) → Quick start → Deep dive links
- Demo video (3-5 min):
  - Show: deploy → serve → load test → monitor → optimize → redeploy
  - Narrate your engineering decisions

---

## What You DON'T Need to Do (Cutting Scope Responsibly)

- **Skip**: Building FlashAttention from scratch (understand it, use it, don't reimplement it)
- **Skip**: Full distributed training on multiple GPUs (understand the concepts, simulate where needed)
- **Skip**: eBPF probes (regular Prometheus metrics are plenty impressive)
- **Skip**: gRPC dual-protocol (HTTP + SSE is fine for the demo)
- **Skip**: Full PPO implementation (DPO demonstrates the same understanding with 1/5 the complexity)
- **Skip**: Reading papers end-to-end (watch talks, read blog post summaries, read key sections only)
- **Skip**: Building a frontend/chat UI (zero signal for infra roles)

---

## The Resume Lines This Produces

> "Built Forge, an open-source GPU inference platform featuring multi-model orchestration with LoRA hot-swap, custom continuous batching scheduler, block-based KV-cache management, and a Kubernetes operator for GPU workload scheduling. Achieved 3.8x throughput over naive inference with comprehensive quantization pipeline and observability stack."

> "Implemented custom Triton GPU kernels for fused normalization and attention, demonstrating understanding of GPU memory hierarchy optimization. Built automated model quantization pipeline with quality gates, publishing Pareto-optimal configurations across GPTQ, AWQ, and NF4 methods."

---

## Daily Routine (Realistic for 5 Months)

- **1.5 hours**: Learn (watch a conference talk, study vLLM source code, read a blog post explaining a concept)
- **4-5 hours**: Build (write code, run experiments, debug)
- **30 min**: Document (update ADR, draft blog paragraph, clean up commit messages)
- **30 min**: Push (commit, update README, review what you built)

Weekends: optional catch-up or exploration. Don't burn out — this is a marathon.

---

## Phase Gates (How You Know You're Ready to Move On)

**End of Phase 1 (Week 7)**:
- Can deploy the full stack with one command
- Can explain model routing, LoRA serving, and RAG architecture to someone
- Have a clean GitHub repo with tests, CI, and documentation
- Have one published blog post

**End of Phase 2 (Week 14)**:
- Can explain continuous batching vs static batching with a whiteboard diagram
- Can explain KV-cache memory management and the PagedAttention concept
- Can run a load test and diagnose where the bottleneck is (and fix it)
- Have a second blog post and published benchmark results

**End of Phase 3 (Week 20)**:
- Can write a basic Triton kernel and explain GPU memory hierarchy
- Can explain DPO/RLHF and demonstrate understanding with code
- Have a polished portfolio: 4 blog posts, demo video, comprehensive README
- Can confidently walk through any part of your system in an interview

---

## Minimum Viable "Undeniable" Profile (If Life Happens)

If you can only complete Phase 1 + Phase 2 (14 weeks), the MINIMUM that still makes you stand out:

1. Working multi-model inference platform with LoRA support + RAG (Phase 1)
2. Custom batching scheduler + KV-cache management with benchmarks (Phase 2)
3. Observability stack with Grafana dashboards (Phase 2)
4. Two strong blog posts + clean README

This alone puts you above 90% of candidates. Phase 3 takes you to 99%.
