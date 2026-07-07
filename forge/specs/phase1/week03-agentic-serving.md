# Week 3: Agent-Native Serving Workloads (v2 — new week)
> Phase: 1 | Project: Forge | Estimated Duration: 7 days
> **v2 note:** New week, replacing v1 `week03-rag-hardening.md` (preserved in `original_artifacts/specs_v1/forge_phase1/`; its eval + cache content moved into Week 2). Rationale from `original_artifacts/plan_evolution_v2_2026-07.md`: agentic traffic (multi-turn tool loops, massive prefix reuse, tool-call dead time) is reshaping inference serving, and almost nobody has published rigorous agentic-workload serving benchmarks. This week builds the **trace suite and measurement vocabulary** that Phase 2 (Weeks 9–13) benchmarks against — it is the differentiation seed for the whole project.

## Context

Weeks 1–2 serve stateless chat requests. Real 2026 traffic increasingly looks different: an *agent session* is a long shared prefix (system prompt + tool definitions), many sequential turns, interleaved tool calls with GPU-idle dead time, and branching (retries, best-of-n). This inverts serving priorities — prefix-cache hit rate and per-*step* TTFT dominate, and the KV cache becomes the contended resource.

This week you (a) make the server session- and tool-aware, (b) build a reproducible agentic-workload generator, and (c) *measure* how agent traffic differs from chat traffic on your own hardware. You are not building an agent framework — you are building the serving-side substrate and the benchmark.

**Prerequisites**: Weeks 1–2 complete — inference server + RAG + eval harness pattern.

**Builds on**: Week 1's server and vLLM engine; Week 2's eval-harness report pattern. Produces the trace suite reused by Weeks 9 (batching benchmarks) and 13 (load testing).

## Learning Goals

- [ ] Understand how agentic traffic differs from chat: prefix reuse, turn cadence, tool dead time, branching
- [ ] Understand OpenAI-compatible tool calling — `tools` parameter, `tool_calls` in responses, `role: "tool"` messages
- [ ] Understand vLLM automatic prefix caching — what gets cached, when it evicts, how to observe hits
- [ ] Understand per-session vs per-request metrics — TTFT-per-step, end-to-end task latency, tokens/session
- [ ] Understand why tool-call dead time matters for scheduling (GPU idle while the "user" computes)

## Implementation Goals

- [ ] Session layer: `POST /v1/sessions` + session-scoped chat endpoint that maintains conversation history server-side (PostgreSQL/Redis), with concurrent-session isolation
- [ ] Tool-calling support: accept `tools` in chat completions, render them via the model's chat template (Qwen2.5 supports tool calling), parse `tool_calls` from output, accept `role:"tool"` result messages to continue the loop
- [ ] Mock tool executor: configurable latency distribution (e.g., lognormal 50ms–5s) and deterministic outputs, so traces are reproducible
- [ ] Agentic trace generator (`src/forge/traces/`): seeded, with knobs for sessions, turns/session, shared-prefix length, prefix-share ratio, tool-latency distribution, branching factor (best-of-n), and session arrival process (Poisson)
- [ ] Trace format spec: versioned JSON schema + fixture files in `traces/` — the contract Weeks 9/13 consume
- [ ] Measurement harness: run chat-style vs agentic workloads against the server; record TTFT-per-step, end-to-end session latency, throughput, GPU util; toggle vLLM prefix caching on/off
- [ ] Per-session metrics persisted (session_id on every request log row; rollups per session)
- [ ] Mini-report: "Chat vs agent traffic — measured" with charts

## Acceptance Criteria

1. **Sessions work**: create session → 3+ turns via session endpoint → server maintains history (verified: turn N includes context from turn 1 without client resending it); 20 concurrent sessions stay isolated (no history bleed)
2. **Tool loop works**: request with `tools` → response contains parsed `tool_calls` (OpenAI-compatible shape); submitting a `role:"tool"` result continues the turn to a final answer — full loop demonstrated end-to-end
3. **Traces reproducible**: same seed → byte-identical trace file; schema documented in `traces/SCHEMA.md` with version field
4. **Baseline comparison**: chat workload (Poisson, independent requests) vs agentic workload (same total token volume) benchmarked on the same server; TTFT and throughput compared in ≥3 charts
5. **Prefix caching measured**: with vLLM prefix caching enabled vs disabled, TTFT-per-step on high-prefix-reuse traces (≥1500-token shared prefix, ≥5 turns) improves ≥30%; the delta is charted across turn number
6. **Per-session metrics**: TTFT-per-step, end-to-end session latency, and tokens/session recorded in PostgreSQL and queryable per session_id
7. **Dead-time quantified**: during tool execution windows, GPU idle time is measured and reported (e.g., % of wall-clock the GPU is idle at concurrency 1 vs 16); one paragraph documents the scheduling implication (why engines want other sessions' work to fill the gap)
8. **Branching measured**: best-of-n step (n=3) benchmarked vs n=1 — marginal cost per extra branch reported, demonstrating shared-prefix amortization
9. **Trace suite versioned**: ≥4 named workload profiles committed (`chat_poisson`, `agent_light`, `agent_heavy_prefix`, `agent_branching`) with fixture files + generator configs
10. **Report + tests**: mini-report (`docs/agent_vs_chat.md`, ≥3 charts, numbers tabulated) committed; `uv run pytest tests/integration/test_sessions.py tests/integration/test_traces.py -v` passes

## Validation Commands

```bash
# Create a session and converse
SESSION=$(curl -s -X POST http://localhost:8000/v1/sessions -d '{"system":"You are a helpful assistant."}' | jq -r .id)
curl -X POST http://localhost:8000/v1/sessions/$SESSION/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"My name is Jayam. Remember it."}]}'
curl -X POST http://localhost:8000/v1/sessions/$SESSION/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is my name?"}]}'   # must answer from history

# Tool-calling loop
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b-awq","messages":[{"role":"user","content":"What is the weather in Berlin?"}],
       "tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}'
# → expect tool_calls in response; then POST the tool result to continue

# Generate traces (seeded)
uv run python -m forge.traces.generate --profile agent_heavy_prefix --seed 42 --out traces/agent_heavy_prefix.json
uv run python -m forge.traces.generate --profile agent_heavy_prefix --seed 42 --out /tmp/repro.json
diff traces/agent_heavy_prefix.json /tmp/repro.json   # must be identical

# Run the benchmark (prefix caching on vs off — restart engine between runs)
uv run python -m forge.traces.bench --trace traces/agent_heavy_prefix.json --out results/agent_apc_on.json
# restart server with --no-enable-prefix-caching, then:
uv run python -m forge.traces.bench --trace traces/agent_heavy_prefix.json --out results/agent_apc_off.json
uv run python -m forge.traces.report results/agent_apc_on.json results/agent_apc_off.json

# Tests
uv run pytest tests/integration/test_sessions.py tests/integration/test_traces.py -v
```

## Technical Implementation Details

### Component 1: Session layer (Day 1–2)
**Files: `src/forge/sessions.py`, `src/forge/routes/sessions.py`**
- Session row in PostgreSQL (id, system_prompt, created_at, metadata JSONB); message history either in PG (simple, durable) or Redis list (fast) — pick one, document why in an ADR-style note.
- `POST /v1/sessions` → id; `POST /v1/sessions/{id}/chat` → appends user message, assembles full history, calls the engine, appends assistant message, returns OpenAI-shaped response (streaming supported).
- Keep the stateless `/v1/chat/completions` untouched — sessions are additive.

### Component 2: Tool calling (Day 2–3)
**File: `src/forge/tools.py`**
- Qwen2.5's chat template supports tools natively: pass `tools` to `apply_chat_template`. Parse the model's tool-call output format (Hermes-style JSON in Qwen2.5) into OpenAI `tool_calls` structure. Handle malformed JSON with one reparse attempt, then surface as content.
- Mock executor: registry of fake tools (`get_weather`, `search`, `calculator`) returning deterministic outputs; latency sampled from a configurable distribution with a fixed seed.
- vLLM note: recent vLLM versions have `--enable-auto-tool-choice --tool-call-parser hermes` for served models; since Week 1 wraps `AsyncLLMEngine` directly, parsing in your own layer is the appropriate path — and teaches you what the flag does.

### Component 3: Trace generator (Day 3–4)
**Files: `src/forge/traces/generate.py`, `traces/SCHEMA.md`**
- Trace = list of sessions; session = arrival_time + shared prefix + list of steps; step = user tokens (synthetic text of target length), expects_tool_call flag, tool_latency_ms, branching factor.
- Profiles as dataclasses/YAML: `chat_poisson` (1 turn, no prefix share), `agent_light` (5 turns, 500-token prefix), `agent_heavy_prefix` (10 turns, 2000-token prefix, tools every other turn), `agent_branching` (n=3 best-of-n steps).
- Everything seeded (`random.Random(seed)`); token lengths matter more than token content — use repeatable filler text with distinct prefixes per session so prefix caching behaves realistically (identical prefix *within* a session, distinct *across* sessions).

### Component 4: Benchmark + metrics (Day 4–6)
**Files: `src/forge/traces/bench.py`, `src/forge/traces/report.py`**
- Async replayer: honors arrival times, plays sessions concurrently, waits tool_latency before submitting the next step (this creates the dead time you measure).
- Record per step: TTFT, completion time, tokens in/out; per run: aggregate percentiles, GPU utilization sampled via pynvml at 1Hz, wall-clock GPU-idle fraction.
- Reuse Week 2's report pattern: JSON results → comparison tables/charts (matplotlib).
- vLLM: enable `enable_prefix_caching=True` in engine args for the "on" runs; verify hits via vLLM's logged cache stats or TTFT deltas.

### Component 5: Report + tests (Day 6–7)
- `docs/agent_vs_chat.md`: the three headline charts — (1) TTFT-per-step vs turn number, caching on/off; (2) throughput chat vs agent at equal token volume; (3) GPU idle % vs concurrency during tool waits. Table of all numbers. Two paragraphs of interpretation — this seeds a blog post and the Phase 2 benchmark baseline.
- Tests: session isolation (parallel sessions, assert no bleed), tool-loop e2e, trace reproducibility, bench smoke run on a tiny trace.

## If You Get Stuck

**Model won't emit tool calls**: verify the chat template receives `tools` (print the rendered prompt); Qwen2.5-7B-Instruct handles Hermes-format tools — few-shot the system prompt if reluctance persists.
**Prefix caching shows no benefit**: ensure the shared prefix is genuinely identical tokens (same rendered template) and long enough (≥ several hundred tokens); confirm `enable_prefix_caching=True` reached the engine; distinct-session prefixes must differ from token 0.
**Bench numbers noisy**: fix seeds, warm up the engine (discard first N requests), run 3 repetitions and report mean±std — the eval-harness discipline from Week 2 applies.
**Sessions + streaming awkward**: stream the response through, append the full assistant message to history in a `finally` block after the stream completes.
**Scope creep into agent frameworks**: you are not building planning/reasoning — the "agent" is the trace generator. If you're writing agent logic, stop.

## Agent Handoff Template

```
I'm on Week 3 of Forge — agent-native serving workloads (v2 spec, new week).
Spec: /home/zzjam/Documents/dev/plan_00/forge/specs/phase1/week03-agentic-serving.md
Current state: Weeks 1–2 done — vLLM (Qwen2.5-7B-Instruct-AWQ) server with RAG + retrieval eval harness.
This week: session layer + tool-calling parse loop + seeded agentic trace generator + chat-vs-agent benchmark (prefix caching on/off) + mini-report.
Codebase: /home/zzjam/Documents/dev/plan_00/forge/src/forge/
Focus on: [specific component]
```

## Out of Scope

- Agent frameworks, planning, reasoning loops (the trace generator simulates the client side)
- Real tools / external APIs (mock executor only — determinism is the point)
- MCP protocol support (concept worth knowing; not needed for the measurement)
- Custom scheduling changes based on the findings (that's Week 9 — this week produces the workload and the baseline)
- Multi-model routing of sessions (Week 4)
