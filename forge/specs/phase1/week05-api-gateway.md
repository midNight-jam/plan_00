# Week 5: API Gateway and Platform Patterns
> Phase: 1 | Project: Forge | Estimated Duration: 7 days

## Context

Weeks 1-4 built the core serving platform. This week transforms it from a "project" into a "product" — adding authentication, rate limiting, multi-tenancy, usage tracking, and streaming middleware. These are the patterns that every production AI platform needs and that interviewers expect you to understand.

**Prerequisites**: Weeks 1-4 complete — multi-model serving with RAG and request queue working.

**Builds on**: Wraps existing endpoints with middleware layers for auth, rate limiting, and tenancy.

## Learning Goals

- [ ] Understand token bucket algorithm — how rate limiting works mathematically
- [ ] Understand API key management — generation, hashing, rotation, revocation
- [ ] Understand multi-tenancy — isolation, quotas, fair resource sharing
- [ ] Understand streaming middleware — how to add logic around SSE streams without breaking them
- [ ] Understand usage metering — tracking token consumption for billing/quotas

## Implementation Goals

- [ ] Implement API key authentication (create, validate, revoke, rotate)
- [ ] Implement token bucket rate limiter (per-key, per-model limits)
- [ ] Implement multi-tenancy (tenant isolation, per-tenant model access, usage tracking)
- [ ] Implement usage metering (tokens consumed per key/tenant/model)
- [ ] Implement streaming middleware (auth + metering work with SSE streams)
- [ ] Implement request/response middleware (validation, sanitization, correlation IDs)
- [ ] Implement client disconnect detection (cancel inference on disconnect)
- [ ] Build admin dashboard API for usage stats

## Acceptance Criteria

1. **Auth required**: Requests without API key return 401
2. **Valid key works**: Request with valid `Authorization: Bearer sk-xxx` returns 200
3. **Revoked key rejected**: After revoking a key, requests with it return 401
4. **Rate limit enforced**: After exceeding limit, requests return 429 with `Retry-After` header
5. **Tenant isolation**: Tenant A's API key cannot access Tenant B's documents
6. **Usage tracked**: After 10 requests, usage API shows accurate token counts per key
7. **Streaming + auth**: Streaming responses work correctly with authentication
8. **Disconnect cancels**: If client disconnects mid-stream, GPU inference is cancelled
9. **Correlation IDs**: Every response includes `X-Request-Id` header; logs contain this ID
10. **Admin usage**: `GET /v1/admin/usage?key=sk-xxx` returns token consumption breakdown

## Validation Commands

```bash
# Create API key
curl -X POST http://localhost:8000/v1/admin/keys \
  -H "X-Admin-Key: admin-secret" \
  -d '{"name": "test-key", "tenant": "team-alpha", "rate_limit_rpm": 60}'

# Use the returned key
export API_KEY="sk-returned-key-here"

# Authenticated request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"mistral-7b","messages":[{"role":"user","content":"Hello"}]}'

# Request without key (should fail)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mistral-7b","messages":[{"role":"user","content":"Hello"}]}'
# Should return 401

# Exceed rate limit
for i in {1..70}; do
  curl -s -o /dev/null -w "%{http_code}\n" \
    -H "Authorization: Bearer $API_KEY" \
    -X POST http://localhost:8000/v1/chat/completions \
    -d '{"model":"mistral-7b","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}' &
done
wait
# Should see some 429s

# Check usage
curl -H "X-Admin-Key: admin-secret" \
  "http://localhost:8000/v1/admin/usage?key=$API_KEY" | python -m json.tool

# Verify correlation ID
curl -v -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"model":"mistral-7b","messages":[{"role":"user","content":"Hello"}]}'
# Check for X-Request-Id in response headers

# Run tests
pytest tests/integration/test_gateway.py -v
```

## Technical Implementation Details

### Component 1: API Key Management (Day 1-2)

**File: `src/forge/auth/keys.py`**

```python
class APIKeyManager:
    def create_key(self, name: str, tenant: str, permissions: dict) -> APIKey:
        # Generate: sk- + 48 random chars
        # Store: hash of key (argon2), name, tenant, permissions, created_at
        # Return: full key (only time it's shown in plaintext)
        
    def validate_key(self, key: str) -> KeyInfo | None:
        # Hash the key, look up in DB
        # Check: not revoked, not expired
        # Return: tenant, permissions, rate_limit config
        
    def revoke_key(self, key_id: str) -> None:
        # Mark as revoked in DB
        # Remove from Redis cache
        
    def rotate_key(self, key_id: str) -> APIKey:
        # Create new key, mark old as revoked (with grace period)
```

Storage: PostgreSQL table `api_keys` (id, key_hash, name, tenant_id, permissions JSONB, rate_limit_rpm, created_at, revoked_at, expires_at)

Cache: Redis hash for fast validation (avoid DB hit on every request). TTL 5 minutes, invalidate on revoke.

### Component 2: Rate Limiter (Day 2-3)

**File: `src/forge/auth/rate_limiter.py`**

Token bucket algorithm (implement yourself — it's an interview question):

```python
class TokenBucketRateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def is_allowed(self, key: str, limit_rpm: int) -> tuple[bool, dict]:
        # Redis-based sliding window:
        # Key: f"ratelimit:{key_hash}:{current_minute}"
        # Increment counter, check against limit
        # Return: (allowed, {"remaining": N, "reset": timestamp})
        
    def get_retry_after(self, key: str) -> int:
        # Seconds until the rate limit resets
```

Also implement per-model limits (some models are more expensive):
- Per-key global RPM
- Per-model RPM (e.g., large model limited to 10 RPM, small model 100 RPM)
- Token-based limits (max tokens per hour per key)

### Component 3: Multi-Tenancy (Day 3-4)

**File: `src/forge/tenancy.py`**

```python
class Tenant:
    id: str
    name: str
    allowed_models: list[str]  # Which models this tenant can use
    max_documents: int          # Document quota for RAG
    gpu_priority: int           # Priority in request queue

class TenantManager:
    def get_tenant(self, tenant_id: str) -> Tenant
    def check_model_access(self, tenant: Tenant, model: str) -> bool
    def check_document_quota(self, tenant: Tenant) -> bool
```

Isolation rules:
- Documents uploaded by Tenant A are invisible to Tenant B (Qdrant filter by tenant_id)
- Tenant A's API keys can only access Tenant A's models (permission check)
- Usage is tracked per-tenant

### Component 4: Usage Metering (Day 4-5)

**File: `src/forge/metering.py`**

Track per request:
- Input tokens (prompt tokens)
- Output tokens (completion tokens)
- Model used
- Adapter used (if any)
- Latency
- Cache hit (if semantic cache was used)

Aggregation (stored in PostgreSQL):
- Hourly rollup per key: total_requests, total_input_tokens, total_output_tokens
- Daily summary per tenant
- Per-model usage breakdown

```python
class UsageMeter:
    async def record(self, key_id: str, tenant_id: str, model: str, 
                     input_tokens: int, output_tokens: int, latency_ms: int):
        # Write to usage_events table (async, non-blocking)
        # Update hourly rollup in Redis (fast increment)
        
    async def get_usage(self, key_id: str, start: datetime, end: datetime) -> UsageReport:
        # Query aggregated usage from PostgreSQL
```

### Component 5: Streaming Middleware (Day 5)

**File: `src/forge/middleware/streaming.py`**

Challenge: Auth and metering must work with SSE streams. You can't just wrap the response — you need to:
1. Authenticate BEFORE starting the stream
2. Count tokens AS they stream (each SSE chunk has content)
3. Record final usage AFTER stream completes
4. Cancel inference if client disconnects mid-stream

```python
async def stream_with_metering(request, engine_stream):
    total_tokens = 0
    try:
        async for chunk in engine_stream:
            total_tokens += count_tokens_in_chunk(chunk)
            yield format_sse(chunk)
    except asyncio.CancelledError:
        # Client disconnected — cancel inference
        await engine.cancel_request(request.id)
    finally:
        # Record usage regardless of how stream ended
        await meter.record(request.key_id, ..., output_tokens=total_tokens)
```

### Component 6: Request Middleware (Day 5-6)

**File: `src/forge/middleware/request.py`**

FastAPI middleware stack (order matters):
1. **Correlation ID**: Generate UUID, attach to request state, include in all logs
2. **Authentication**: Validate API key, attach tenant info to request
3. **Rate limiting**: Check rate limit for this key
4. **Input validation**: Validate request body, sanitize inputs
5. **Request logging**: Log request metadata (without sensitive content)

### Component 7: Admin Dashboard API (Day 6-7)

**File: `src/forge/routes/admin.py`** (extend from Week 4)

New endpoints:
- `GET /v1/admin/usage` — usage by key, tenant, model, time range
- `GET /v1/admin/keys` — list all API keys (masked)
- `POST /v1/admin/keys` — create new key
- `DELETE /v1/admin/keys/{id}` — revoke key
- `GET /v1/admin/tenants` — list tenants with usage summaries
- `GET /v1/admin/rate-limits` — current rate limit status per key

Admin auth: separate admin key (or basic auth) for admin endpoints.

## If You Get Stuck

**Rate limiter race conditions**: Use Redis MULTI/EXEC or Lua scripts for atomic increment + check. The `redis-py` library supports this.

**Streaming + auth complexity**: Implement auth as a dependency (FastAPI `Depends()`) that runs before the endpoint. The endpoint itself handles streaming. Metering is the tricky part — use a finally block.

**Client disconnect detection**: FastAPI/Starlette provides `request.is_disconnected()`. Check periodically during generation. For vLLM, use the abort API.

**Simplified fallback**: Start with just API key auth + basic rate limiting. Add multi-tenancy and metering as separate PRs.

## Agent Handoff Template

```
I'm on Week 5 of Forge — building API gateway patterns.
Spec: /Users/jmalviya/Documents/zz/dev/plan_00/forge/specs/phase1/week05-api-gateway.md
Current state: Multi-model inference server with RAG, request queue, LoRA support.
I need to add: API key auth, token bucket rate limiting, multi-tenancy, usage metering, streaming middleware with disconnect detection.
Codebase: [path to forge/src/]
Focus on: [specific component]
```

## Out of Scope

- OAuth2 / SSO integration (production concern, not portfolio)
- Payment/billing integration (just meter, don't bill)
- Web UI dashboard (API only — frontend adds no interview signal for infra roles)
- Complex RBAC beyond tenant-level access (keep it simple)
