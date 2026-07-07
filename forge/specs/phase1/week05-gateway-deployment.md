# Week 5: Gateway Essentials + Deployment (v2 — merged)
> Phase: 1 | Project: Forge | Estimated Duration: 7 days
> **v2 note:** Merges v1 `week05-api-gateway.md` + `week06-deployment.md` into one week (originals in `original_artifacts/specs_v1/forge_phase1/`). Rationale: you have a decade of professional experience in auth, rate limiting, IaC, and CI/CD — re-proving it earns nothing. This week keeps only the **AI-specific** slices: token metering over SSE streams, client-disconnect → GPU cancellation, per-model rate limits, model-load-aware readiness probes, and GPU scheduling in K8s. Cut: key-rotation ceremony, multi-tenancy depth, admin dashboard breadth, Makefile/secrets ceremony as separate goals. The freed week becomes Week 6 (buffer + OSS).

## Context

Weeks 1–4 built the serving platform. This week wraps it with the minimum production armor (days 1–3) and ships it properly (days 4–7). The interview-relevant material here is what's *different about AI services*: metering tokens through a stream you don't buffer, killing GPU work when the client goes away, rate-limiting expensive models differently from cheap ones, and telling Kubernetes the pod isn't Ready until a 5-GB model is actually loaded. Everything else is deliberately done at professional speed, not learning speed.

**Prerequisites**: Weeks 1–4 complete — multi-model serving with sessions, RAG, request queue.

**Builds on**: Wraps existing endpoints with middleware; containerizes and deploys the full stack.

## Learning Goals

- [ ] Understand streaming middleware — metering/cancellation around SSE without buffering the stream
- [ ] Understand token-bucket rate limiting (implement it yourself — still a standard interview question) and per-model limit design
- [ ] Understand client-disconnect semantics — how FastAPI/Starlette detects it, how vLLM aborts a request
- [ ] Understand readiness vs liveness for ML services — why "server up" ≠ "model loaded"
- [ ] Understand K8s GPU scheduling as it exists on a single-node K3s (device plugin path; DRA — the 2026 replacement — is covered properly in Anvil Week 2)

## Implementation Goals

**Gateway essentials (Days 1–3):**
- [ ] API key auth: `sk-` keys, argon2 hash in PostgreSQL, Redis validation cache, create/revoke via admin endpoints (no rotation ceremony)
- [ ] Token-bucket rate limiter (Redis Lua/MULTI-EXEC atomic): per-key RPM + per-model RPM (expensive model = lower limit)
- [ ] Usage metering: input/output tokens per key/model recorded per request — **including streamed responses** (count as chunks flow, record in `finally`)
- [ ] Client-disconnect detection: mid-stream disconnect aborts the vLLM request (engine `abort()`), usage recorded for tokens actually generated
- [ ] Correlation IDs (`X-Request-Id`) end-to-end; light tenant isolation: `tenant_id` on keys filters RAG document access

**Deployment (Days 4–7):**
- [ ] Multi-stage Dockerfile (<5GB image; model weights in a volume, never in the image)
- [ ] Full Docker Compose: server + postgres + redis + qdrant with health checks, volumes, `depends_on: condition: service_healthy`, `start_period` sized for model load
- [ ] Single-node K3s with NVIDIA device plugin; pod requests `nvidia.com/gpu: 1`
- [ ] Helm chart (deployment, service, configmap, secret, PVC for model cache); readiness probe gated on model-loaded `/health`, `failureThreshold` sized for a 5-minute load
- [ ] GitHub Actions CI: ruff + mypy + unit tests + Docker build (GHA cache) + CPU-mode integration test (mock/tiny engine — CI has no GPU)
- [ ] Secrets hygiene: `.env.example` committed, `.env` ignored, K8s Secrets, GHA secrets; `make dev/test/lint/build/deploy` targets

## Acceptance Criteria

1. **Auth enforced**: no key → 401; valid `Authorization: Bearer sk-…` → 200; revoked key → 401 within 5s of revocation (cache invalidated)
2. **Rate limit enforced**: exceeding per-key RPM → 429 with `Retry-After`; per-model limits verified (low-limit model 429s while high-limit model still serves the same key)
3. **Streaming metering accurate**: for a streamed response, recorded output-token count matches the engine's own usage stats within ±5%
4. **Disconnect cancels GPU work**: killing the client mid-stream aborts the engine request (verified via engine logs / no further decode steps) and still records partial usage
5. **Correlation IDs**: every response carries `X-Request-Id`; the same ID appears in server logs and the request-log row
6. **Tenant isolation**: a key with tenant A cannot retrieve tenant B's documents via search or RAG
7. **Image + stack**: image <5GB; `make dev` brings up the full compose stack healthy; inference works end-to-end
8. **K8s deploy**: `helm install forge ./helm-charts/forge` on K3s succeeds; pod is scheduled with `nvidia.com/gpu: 1`; pod shows 0/1 Ready during model load and 1/1 only after `/health` reports the model loaded
9. **CI green**: push → GitHub Actions runs lint + typecheck + unit + build + CPU-mode integration, all passing
10. **Hygiene**: no secrets in git history for this week's changes; `make lint/test/build/deploy` all work; `pytest tests/integration/test_gateway.py -v` passes

## Validation Commands

```bash
# Create + use a key
curl -X POST http://localhost:8000/v1/admin/keys -H "X-Admin-Key: $ADMIN_KEY" \
  -d '{"name":"test","tenant":"team-a","rate_limit_rpm":30,"model_limits":{"qwen2.5-7b-awq":10}}'
export API_KEY="sk-…"

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b-awq","messages":[{"role":"user","content":"Hello"}]}'

# 401 without key; 429 after hammering
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" -d '{"model":"qwen2.5-7b-awq","messages":[{"role":"user","content":"Hi"}]}'
for i in {1..40}; do curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $API_KEY" \
  -X POST http://localhost:8000/v1/chat/completions \
  -d '{"model":"qwen2.5-7b-awq","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}' & done; wait

# Disconnect test: start a long stream, kill curl after 2s, watch server logs for abort
timeout 2 curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b-awq","messages":[{"role":"user","content":"Write a 2000 word story"}],"stream":true}'

# Usage
curl -H "X-Admin-Key: $ADMIN_KEY" "http://localhost:8000/v1/admin/usage?key=$API_KEY" | python -m json.tool

# Deployment
make dev && curl http://localhost:8000/health
docker images forge-server --format "{{.Size}}"
kubectl get nodes && kubectl describe node | grep nvidia.com/gpu
helm install forge ./helm-charts/forge -f helm-charts/forge/values-dev.yaml
kubectl get pods -w   # 0/1 during model load → 1/1 Ready

# Tests
uv run pytest tests/integration/test_gateway.py -v
```

## Technical Implementation Details

### Days 1–3: Gateway (professional speed)
- `src/forge/auth/keys.py` — `APIKeyManager` (create/validate/revoke); PG table `api_keys` (id, key_hash, name, tenant_id, rate_limit_rpm, model_limits JSONB, revoked_at); Redis cache TTL 5min, delete on revoke.
- `src/forge/auth/rate_limiter.py` — token bucket in Redis, atomic via Lua script; check global per-key bucket then per-model bucket; return remaining/reset for headers.
- `src/forge/metering.py` — per-request usage row (key, tenant, model, tokens in/out, latency, request_id); hourly rollups can wait (Week 13 dashboards will query raw).
- `src/forge/middleware/streaming.py` — the one genuinely interesting file:
  ```python
  async def stream_with_metering(request_id, key, engine_stream):
      out_tokens = 0
      try:
          async for chunk in engine_stream:
              out_tokens += chunk.token_count
              yield format_sse(chunk)
      except asyncio.CancelledError:      # client went away
          await engine.abort(request_id)  # free the GPU
          raise
      finally:
          await meter.record(key, ..., output_tokens=out_tokens)
  ```
- Middleware order: correlation ID → auth → rate limit → route. Auth as FastAPI `Depends`, metering inside the streaming generator.
- Tenant isolation = one Qdrant filter (`tenant_id`) applied in search/RAG paths. That's the whole feature; resist building more.

### Days 4–7: Deployment (also professional speed)
- Dockerfile: `nvidia/cuda:12.x-devel` builder stage (uv sync into venv) → `-runtime` final stage; `.dockerignore`; deps layer separate from code layer.
- Compose: all four services with healthchecks (`pg_isready`, `redis-cli ping`, qdrant `/healthz`, server `/health` with `start_period: 120s`).
- K3s: `curl -sfL https://get.k3s.io | sh -`; NVIDIA device plugin DaemonSet; confirm `nvidia.com/gpu` in node capacity. **Note for the write-up**: the device plugin is the legacy-but-practical path on single-node K3s; DRA (`ResourceClaim`/`DeviceClass`, GA in K8s 1.34) is the modern API and gets a full week in Anvil W2 — one paragraph acknowledging this shows you know where the ecosystem is.
- Helm: deployment/service/configmap/secret/PVC; readiness `initialDelaySeconds: 60, periodSeconds: 10, failureThreshold: 30`; liveness laxer than readiness.
- CI: lint/typecheck/unit on ubuntu-latest; build with GHA layer cache; integration job runs compose with a CPU-mock engine flag (add `FORGE_ENGINE=mock` supported by your engine factory — also useful for local fast tests).

## If You Get Stuck

**Rate-limiter race conditions**: Lua script doing INCR+EXPIRE+check atomically; don't read-then-write from Python.
**Disconnect not detected**: Starlette raises `CancelledError` into the generator when the client drops — make sure you don't swallow it; for vLLM use the engine's abort API with your request id.
**K3s GPU not visible**: nvidia-container-runtime must be containerd's default runtime for K3s; check `/var/lib/rancher/k3s/agent/etc/containerd/config.toml`.
**Image >5GB**: CUDA runtime base is ~3–4GB alone; ship no model weights, no dev tools, no test dirs; check with `docker history`.
**CI integration flaky**: mock-engine mode; `curl --retry` against `/health` before running tests.

## Agent Handoff Template

```
I'm on Week 5 of Forge — gateway essentials + deployment (v2 merged spec).
Spec: /home/zzjam/Documents/dev/plan_00/forge/specs/phase1/week05-gateway-deployment.md
Current state: Weeks 1–4 done — multi-model server with sessions, tools, RAG, LoRA routing.
Days 1–3: auth + token-bucket rate limits + streaming metering + disconnect-abort + correlation IDs.
Days 4–7: Dockerfile + Compose + K3s/device-plugin + Helm (model-load readiness) + GH Actions CI.
Codebase: /home/zzjam/Documents/dev/plan_00/forge/
Focus on: [specific component]
```

## Out of Scope

- Key rotation/expiry ceremony, OAuth/SSO, billing (metering only)
- Multi-tenancy beyond the document filter (quotas/priorities were v1 scope — cut)
- Admin dashboard breadth (usage endpoint only)
- Web UI (no interview signal for infra roles)
- DRA-based GPU scheduling (Anvil Week 2 — done properly there)
- ArgoCD/GitOps, multi-node, TLS/domains (Anvil territory / not portfolio-relevant)
