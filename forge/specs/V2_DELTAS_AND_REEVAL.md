# Forge Specs — v2 Deltas & Re-evaluation Gates
> Source of truth for *why*: `original_artifacts/plan_evolution_v2_2026-07.md` (2026-07-06 strategic review).
> This file tracks (a) which specs were regenerated, (b) standing deltas for specs kept as-written, and (c) **when the remaining v1 specs must be re-evaluated before use**.

## Regenerated (v2 specs live in `specs/phase1/`, v1 originals in `original_artifacts/specs_v1/forge_phase1/`)

| Week | v1 file (superseded) | v2 file | Change |
|---|---|---|---|
| W2 | `week02-rag-pipeline.md` + `week03-rag-hardening.md` | `week02-rag-retrieval-eval.md` | Two RAG weeks compressed to one; eval harness is the keeper |
| W3 | `week03-rag-hardening.md` | `week03-agentic-serving.md` | **New week** — agentic trace suite + chat-vs-agent measurements |
| W5 | `week05-api-gateway.md` + `week06-deployment.md` | `week05-gateway-deployment.md` | Merged; AI-specific slices only (streaming metering, disconnect-abort, model-load readiness) |
| W6 | `week06-deployment.md` | `week06-buffer-oss.md` | **New week** — AC sweep + blog #1 (sm_120/Blackwell) + upstream issue/PR + paper sprint #0 |

## Standing deltas for kept v1 specs (apply when you reach the week; small enough to not warrant regeneration yet)

- **W4 multi-model**: +1 AC — route by adapter name with Gateway API Inference Extension `InferencePool` semantics (concept parity on one node). Sessions from new W3 must route consistently (session affinity to a model).
- **W7 consolidation**: + paper sprint #1 (2–3 days, pick from the reading queue in the v2 doc §A3); blog post #2 = Phase 1 retrospective as specced.
- **W8 transformer internals**: unchanged. ✅ Keep as written.
- **W9 continuous batching**: benchmarks must run on the W3 agentic trace suite *in addition to* Poisson chat; report **goodput** (SLO-attainment throughput), not only tokens/s.
- **W10 KV-cache**: + radix-tree cross-request prefix reuse (pulled from v1 W11), + hierarchical offload GPU→CPU(pinned)→NVMe with restore-latency numbers, + per-session hit-rate telemetry.
- **W11 speculative decoding**: prefix caching moved to W10. + 2-process prefill/decode **disaggregation simulation** on one GPU with measured KV-transfer overhead vs recompute (NIXL-style interface).
- **W12 quantization**: + **FP4/NVFP4 on the RTX 5080** (Blackwell supports it — rare public data) with task-eval quality gates, not just perplexity; GGUF demoted to footnote; blog post #3 = "FP4 on consumer Blackwell".
- **W13 observability**: dashboards framed around goodput/SLO-attainment + per-session prefix-hit-rate; load profiles include the W3 agentic traces.
- **W14 perf methodology**: + paper sprint #2.

## ⚠️ RE-EVALUATION GATE #1 — before starting Week 11 (DO NOT SKIP)

**Trigger:** Week 10 marked complete in `progress.md`.
**Why:** W11+ specs were written mid-2026 assuming a mid-2026 landscape; by the time you arrive (~Oct–Nov 2026) the inference stack will have moved again (this is exactly what happened to the v1 specs between June 2026 and the v2 review — DRA went GA and invalidated W15's design in three months).

**Action checklist at the gate:**
1. Re-read `original_artifacts/plan_evolution_v2_2026-07.md` §A (forecast) and §B1 (Forge risk table) — check which predictions landed.
2. Run a fresh landscape check (web) on, minimally: vLLM/SGLang release notes since July 2026; speculative-decoding as built-in engine feature (does W11's build still differentiate?); FP4/NVFP4 tooling maturity on consumer Blackwell (does W12's plan need new tools?); NIXL/LMCache/KV-transfer standardization (does W11's disagg sim match current interfaces?).
3. Regenerate or amend `week11`/`week12`/`week13`/`week14` specs against findings; record decisions in `progress.md`.
4. **Capture for the high-level plan review:** which v2 §A predictions were right/wrong, and what that implies for the W15–20 bets (DRA operator, Triton-on-Blackwell, GRPO bridge) and for Anvil.

## ⚠️ RE-EVALUATION GATE #2 — before starting Week 15 (Phase 3)

**Trigger:** Week 14 complete (~Dec 2026–Jan 2027).
**Action:** Same drill, focused on: K8s DRA/Kueue/Gateway-API-Inference-Extension versions (W15 rebuild targets); llm-d/KubeAI contribution surface (is a contribution week still the higher-value W15?); GRPO/RLVR tooling for W17; and — because applications start Jan–Feb 2027 — re-check the v2 doc §D niches against what job postings actually ask for. Regenerate `week15`/`week17` specs; amend `week18-20` portfolio ACs if the OSS/blog targets shifted.

## Bookkeeping

- Spec-immutability convention: applies to *completed* weeks. W1 (in progress) keeps its v1 spec. Superseded-but-never-executed v1 specs are archived, not history-rewritten.
- `progress.md` week names updated to match v2; decision logged there (2026-07-06).
