# Week 6: Buffer + OSS Week (v2 — new)
> Phase: 1 | Project: Forge | Estimated Duration: 7 days
> **v2 note:** This week exists because v1's Weeks 5+6 merged into the new Week 5 (v1 originals in `original_artifacts/specs_v1/forge_phase1/`). It is deliberately *not* a themed build week. Per `original_artifacts/plan_evolution_v2_2026-07.md`: (a) the specs are dense and you work alongside a full-time job — Phase 2's deep weeks (transformer internals, batching, KV-cache) must not start with debt; (b) the portfolio needs an OSS-contribution thread and original published measurements, which need dedicated time, not margins.
> **Format deviation, on purpose:** ACs 1–5 are mandatory; 6–10 are ranked stretch goals. A buffer week with ten hard criteria isn't a buffer.

## Context

Five build weeks are done. Before Phase 2 goes deep, this week: clears every unfinished acceptance criterion from Weeks 1–5, converts your consumer-Blackwell (RTX 5080 / sm_120) experience into public artifacts — the blog post and upstream issue that start your OSS thread — and, if time remains, runs paper sprint #0 to build the paper-to-code muscle the boundary-layer strategy calls for.

**Prerequisites**: Weeks 1–5 attempted (not necessarily 100% green — that's what this week fixes).

**Builds on**: Everything. Produces: a clean Phase-1 platform, publication #1, upstream contribution #1.

## Learning Goals

- [ ] Learn the upstream-contribution workflow of a major OSS project (vLLM or similar): issue etiquette, repro cases, contribution guidelines
- [ ] Learn to write a technical blog post around *measurements you own* (numbers first, narrative second)
- [ ] (Stretch) Learn to go paper → minimal working implementation in 2–3 days, cutting the right corners

## Implementation Goals

- [ ] Sweep Weeks 1–5 acceptance criteria; fix every red one (start with Week 1 AC #9/#10: pytest integration tests, if still open)
- [ ] Write and publish blog post #1: "Serving LLMs on consumer Blackwell: vLLM + sm_120 on a 16GB RTX 5080" — the Week 1 pain (fp16 didn't fit, AWQ path, gated-model detours), with load-time/throughput/VRAM numbers
- [ ] File ≥1 well-formed upstream issue (vLLM or adjacent) from real friction on sm_120 / consumer Blackwell, with a minimal reproduction — or, if everything now works, a docs/test-matrix PR adding RTX 5080 datapoints
- [ ] (Stretch) Paper sprint #0: implement one small recent technique against your stack (candidate: a decoding or prefix-caching tweak measurable with the Week 3 trace suite) + 500-word write-up
- [ ] (Stretch) Pull forward Week 7 consolidation items (ADRs, chaos scenarios)

## Acceptance Criteria

**Mandatory (1–5):**
1. **Phase 1 green**: every AC from Weeks 1–5 passes, or is logged in `KNOWN_ISSUES.md` with cause and a decision (fix-by date or won't-fix rationale) — nothing silently red
2. **Full-stack proof**: `make dev` from a clean clone → healthy stack → authenticated, rate-limited, streamed, metered RAG request succeeds; `uv run pytest tests/ -v` fully green
3. **Blog post #1 published** (external platform — personal site, dev.to, or Medium) with ≥3 original measurements (e.g., model load time, tokens/s, VRAM headroom on the 5080); linked from the repo README
4. **Upstream artifact exists**: issue (with minimal repro) or PR submitted and linked in `progress.md`; quality bar = a maintainer could act on it without asking for more info
5. **progress.md current**: Weeks 1–5 statuses, decisions, and blockers all recorded; completion log dated

**Stretch, in priority order (6–10):**
6. **Paper sprint #0 done**: technique implemented, measured against a Week 3 trace or Week 2 eval, 500-word write-up in `docs/`
7. **W7 pull-forward**: ≥2 ADRs drafted (model choice, session-store choice are natural candidates)
8. **Chaos preview**: one failure scenario scripted (Redis death or GPU OOM) with observed behavior documented
9. **Bench re-run**: Week 3 chat-vs-agent benchmark re-run on the hardened stack; results appended to `docs/agent_vs_chat.md`
10. **Upstream follow-through**: engage with maintainer feedback on the issue/PR (respond within the week if feedback arrives)

## Validation Commands

```bash
# The whole point, end to end
git clone <repo> /tmp/forge-clean && cd /tmp/forge-clean && make dev
export API_KEY=$(./scripts/dev-key.sh)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b-awq","messages":[{"role":"user","content":"test"}],"rag":{"enabled":true},"stream":true}'

# Everything green
uv run pytest tests/ -v
uv run ruff check . && uv run mypy src/

# AC sweep helper — grep each week's spec ACs and check them off explicitly
ls specs/phase1/ && cat KNOWN_ISSUES.md 2>/dev/null
```

## Technical Implementation Details

### Day 1–2: AC sweep
Walk Weeks 1–5 specs top to bottom, re-run every validation command, fix or log. Week 1 AC #9 (pytest integration tests) is known-open as of v2 writing — it goes first.

### Day 3–4: Blog post #1
You already have the material in `progress.md` (Decisions #3–5, both resolved blockers). Structure: the constraint (16GB, sm_120) → what broke (fp16 OOM, gated models) → what worked (AWQ, Qwen2.5) → numbers (load time, tokens/s at batch 1/8, VRAM headroom) → what this means for consumer-Blackwell users. 1500–2500 words. Numbers you can defend; no fluff.

### Day 4–5: Upstream artifact
Check vLLM's issue tracker for existing sm_120/RTX 5080 threads first — commenting with a solid additional datapoint on an existing issue also counts, a duplicate issue doesn't. A good issue: environment block, minimal repro script, expected vs actual, version matrix. If Week 1's friction is already fixed upstream, pivot: PR adding your hardware to a compatibility table/docs, or a failing-then-passing test datapoint.

### Day 6–7: Stretch goals in listed order
Paper sprint #0 candidates (pick something measurable in ≤2 days against your existing harnesses — a sampling/decoding tweak or cache-policy variant beats anything requiring training).

## If You Get Stuck

**AC sweep uncovers something big**: that *is* the week's work — fix it; drop all stretch goals without guilt.
**Blog feels thin**: it isn't — consumer-Blackwell datapoints are scarce; publish the numbers table even if the prose is short.
**Nothing to file upstream**: the docs/test-matrix PR route always exists; scarce-hardware datapoints are genuinely welcomed.
**Everything was already green**: congratulations — this week is paper sprint #0 plus W7 pull-forward, and Phase 2 starts from a running start.

## Agent Handoff Template

```
I'm on Week 6 of Forge — buffer + OSS week (v2 spec).
Spec: /home/zzjam/Documents/dev/plan_00/forge/specs/phase1/week06-buffer-oss.md
Current state: [which Weeks 1–5 ACs are red / what's in KNOWN_ISSUES.md]
Mandatory: AC sweep green, blog post #1 (sm_120/Blackwell), upstream issue or PR, progress.md current.
Stretch (in order): paper sprint #0, ADRs, chaos preview, bench re-run.
Codebase: /home/zzjam/Documents/dev/plan_00/forge/
```

## Out of Scope

- New platform features (this week hardens and publishes; it does not extend)
- Starting Phase 2 content early (transformer internals deserve a fresh week)
- More than one paper sprint (timebox: 2–3 days max)
- Blog posts #2+ (Week 7 consolidation owns the phase retrospective)
