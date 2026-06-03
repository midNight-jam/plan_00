# Week 7: Phase 1 Consolidation
> Phase: 1 | Project: Forge | Estimated Duration: 7 days

## Context

Weeks 1-6 built the entire platform. This week is for hardening, testing edge cases, writing documentation, and producing your first blog post. Most engineers skip this step, which is why their portfolio projects look unfinished. This week makes everything production-quality.

**Prerequisites**: Weeks 1-6 complete — deployed, CI passing, all features working.

**Builds on**: Polishes and hardens everything built so far.

## Learning Goals

- [ ] Understand chaos testing principles — what to break and how to observe recovery
- [ ] Understand ADR (Architecture Decision Record) format — context/decision/consequences
- [ ] Understand technical blog writing — structure, audience, what makes posts shareable
- [ ] Understand integration testing best practices — test real flows, not just units

## Implementation Goals

- [ ] Write comprehensive end-to-end integration test suite (happy path + error paths)
- [ ] Run chaos scenarios and document system behavior
- [ ] Clean up code quality (mypy strict, consistent patterns, remove dead code)
- [ ] Write 3 Architecture Decision Records
- [ ] Write and publish first technical blog post
- [ ] Update README with architecture diagram, quick start, and feature overview
- [ ] Record brief demo video (2-3 min showing the platform in action)

## Acceptance Criteria

1. **Test coverage**: Integration test suite covers 10+ end-to-end scenarios
2. **All tests pass**: `make test-all` exits 0 with no warnings
3. **mypy clean**: `mypy src/ --strict` passes with zero errors
4. **Chaos documented**: 4+ failure scenarios tested with documented behavior
5. **ADRs written**: 3 ADR files in `docs/adrs/` directory
6. **Blog post**: Draft complete (1500-2500 words), publishable quality
7. **README complete**: Architecture diagram, features list, quick start guide, API reference link
8. **Demo video**: 2-3 minute screen recording showing key features
9. **Clean git history**: Squash messy commits, clear commit messages telling a story
10. **Zero known bugs**: All known issues either fixed or documented in a KNOWN_ISSUES.md

## Validation Commands

```bash
# Full test suite
make test-all
echo $?  # Must be 0

# Type checking
uv run mypy src/ --strict

# Lint
uv run ruff check . && uv run ruff format --check .

# Run chaos scenario (example: kill Redis mid-request)
docker compose up -d
curl http://localhost:8000/v1/chat/completions -d '...' &  # Start request
docker compose stop redis  # Kill Redis mid-flight
# Observe: request should complete (graceful degradation) or return useful error
docker compose start redis  # Bring back
curl http://localhost:8000/health  # Should recover

# Verify README renders properly
python -m markdown README.md > /dev/null  # No errors

# Check for committed secrets
git log --all --full-history -p | grep -i "password\|secret\|api_key" | grep -v ".example"
# Should find nothing sensitive
```

## Technical Implementation Details

### Component 1: Integration Test Suite (Day 1-2)

**File: `tests/integration/test_e2e.py`**

Scenarios to test:
1. Full lifecycle: create key → upload doc → RAG query → verify answer uses doc
2. Streaming: start stream → receive chunks → verify complete response
3. Rate limiting: exceed limit → get 429 → wait → requests succeed again
4. Model loading: request unknown model → trigger load → request eventually succeeds (or useful error)
5. LoRA swap: same prompt with different adapters → different responses
6. Queue backpressure: flood with requests → verify 429s → drain → recover
7. Document deletion: delete doc → RAG no longer retrieves it
8. Auth failure: expired key, revoked key, malformed key → proper error codes
9. Context overflow: RAG with huge context → response still generated (truncation works)
10. Concurrent requests: 10 simultaneous requests → all get responses without errors

**File: `tests/integration/conftest.py`**

Shared fixtures:
- `api_client`: configured httpx client with auth
- `admin_client`: client with admin key
- `test_document`: uploaded test document (setup/teardown)
- `test_api_key`: created key for tests (cleaned up after)

### Component 2: Chaos Scenarios (Day 2-3)

**File: `docs/chaos/scenarios.md`**

Document each scenario:

**Scenario 1: Redis dies during request**
- Action: `docker compose stop redis` while requests are in-flight
- Expected: In-flight requests complete (they're already in the engine). New requests may fail fast (queue unavailable) with 503.
- Recovery: When Redis returns, system auto-recovers within 5 seconds.
- Improvement: Add circuit breaker — if Redis is down, bypass queue and serve directly (with reduced capacity).

**Scenario 2: GPU OOM**
- Action: Send request with very long input that exhausts VRAM
- Expected: Server catches CUDA OOM, returns 503 with message, continues serving other requests
- Recovery: Automatic — PyTorch clears failed allocation
- Improvement: Pre-check input length against available KV-cache space

**Scenario 3: Model produces garbage (health degradation)**
- Action: (Simulate by mocking model output with random tokens)
- Expected: Health check detects degraded output, marks model unhealthy
- Recovery: Auto-restart model after N failures
- Improvement: Define "garbage" detector (very high perplexity, very short response, etc.)

**Scenario 4: Disk full (model cache)**
- Action: Fill up the model cache volume
- Expected: New model loads fail gracefully with clear error message
- Recovery: Manual — alert admin to clean cache
- Improvement: LRU eviction of old models from cache

### Component 3: Code Quality Pass (Day 3-4)

Checklist:
- [ ] Run `mypy src/ --strict` — fix all type errors
- [ ] Run `ruff check . --fix` — fix all lint issues
- [ ] Remove all TODO comments (either do them or create issues)
- [ ] Ensure consistent error handling pattern (custom exception hierarchy)
- [ ] Ensure all public functions have type hints and docstrings
- [ ] Remove unused imports, dead code paths
- [ ] Ensure consistent naming conventions across all modules
- [ ] Add `py.typed` marker for the package

### Component 4: Architecture Decision Records (Day 4-5)

**File: `docs/adrs/001-api-framework.md`**
```markdown
# ADR-001: FastAPI as API Framework

## Status: Accepted

## Context
We needed an async Python web framework for serving LLM inference with streaming.

## Decision
Use FastAPI with uvicorn.

## Consequences
- Good: Native async, auto-generated OpenAPI docs, Pydantic validation, SSE support
- Good: Large ecosystem, easy dependency injection
- Bad: Single-process by default (need uvicorn workers or separate processes for multi-model)
- Alternatives considered: Flask (no native async), gRPC (overkill for demo), Starlette (less batteries)
```

Write 3 ADRs covering your most significant decisions.

### Component 5: Blog Post (Day 5-6)

**File: `docs/blog/01-building-ai-platform.md`**

Structure:
1. **Hook** (2-3 sentences): "Everyone's building RAG apps. Here's what most of them get wrong — and how to build a real platform."
2. **Problem statement**: Why a simple wrapper isn't enough
3. **Architecture overview** with diagram
4. **Key decisions** (pick 2-3 most interesting):
   - Multi-model VRAM management
   - Semantic caching for RAG
   - Rate limiting with token buckets
5. **What I learned**: Honest reflection on surprises and mistakes
6. **What's next**: Tease Phase 2 (inference optimization)

Target: 1500-2500 words. Publish on dev.to, Hashnode, or personal blog.

### Component 6: README Overhaul (Day 6)

**File: `README.md`**

Sections:
- Project title + one-line description
- Architecture diagram (Mermaid or ASCII)
- Key features (bullet list with brief descriptions)
- Quick start (copy-paste commands to get running)
- API reference (link to auto-generated OpenAPI docs)
- Configuration reference
- Benchmarks preview (throughput number)
- Blog posts (links)
- ADRs (links)
- Contributing guidelines (brief)
- License

### Component 7: Demo Video (Day 7)

Record with OBS or simple screen recorder. Show:
1. `make dev` → stack starts up
2. Model loads (show logs)
3. Send inference request → get streaming response
4. Upload document → RAG query → answer references document
5. Show Grafana dashboard (even if basic at this stage)
6. Show rate limiting in action (burst requests, see 429s)

Keep it 2-3 minutes. No need for polish — just clear narration of what's happening.

## If You Get Stuck

**mypy strict is hard**: Start with `mypy src/ --ignore-missing-imports`. Then enable one strict flag at a time. Most common fixes: add return types, annotate dict/list types explicitly.

**Blog writing is slow**: Write the outline first (headings only). Then fill in each section in any order. Don't edit while writing — write first, edit after.

**Demo video anxiety**: Nobody expects production quality. Record in one take. If you mess up, start that section over. Cut with a simple editor if needed.

## Agent Handoff Template

```
I'm on Week 7 of Forge — consolidation and documentation.
Spec: /Users/jmalviya/Documents/zz/dev/plan_00/forge/specs/phase1/week07-consolidation.md
Current state: Full platform deployed and working (Weeks 1-6 complete).
I need help with: [writing integration tests / fixing mypy errors / writing the blog post / etc.]
The codebase is at: [path]
All features work but need hardening and documentation.
```

## Out of Scope

- Performance optimization (Phase 2)
- New features (resist the urge!)
- Kubernetes operator (Phase 3)
- Fancy UI or dashboard (not needed)
