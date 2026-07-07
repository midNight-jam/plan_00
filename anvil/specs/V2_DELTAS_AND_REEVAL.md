# Anvil Specs — v2 Deltas & Re-evaluation Gates
> Source of truth for *why*: `original_artifacts/plan_evolution_v2_2026-07.md` (2026-07-06 strategic review).
> Anvil starts ~Nov 2026 — **four months after this review**. Treat every spec in this project as provisional until it passes a gate below. The v1 W2 spec went stale in ~3 months (DRA GA); assume similar drift.

## Regenerated now (v2 specs in `specs/phaseA/`, v1 originals in `original_artifacts/specs_v1/anvil_phaseA/`)

| Week | v1 file (superseded) | v2 file | Change |
|---|---|---|---|
| W1 | `week01-distributed-systems.md` | `week01-raft-and-collectives.md` | Raft compressed to 3 days; +ring allreduce/NCCL concepts (absorbed from Crucible W6) |
| W2 | `week02-k8s-deep-dive.md` | `week02-k8s-dra-kueue.md` | Scheduler-extender/device-plugin pattern → DRA + Kueue (2026 APIs); webhooks kept |

## Standing deltas for kept Phase A specs (apply on arrival, re-validate at Gate #1)

- **W3 training orchestrator** (flagship — keep): + elastic semantics (grow/shrink world size via checkpoint-restore); benchmark against **Kueue + JobSet**, not only standalone; SLIs framed as researcher experience (time-to-first-batch, queue wait, resume latency); submit through W2's Kueue queues.
- **W4 IaC**: compress to 2–3 days — keep the spot-eviction→checkpoint drill + ArgoCD sync for the platform; skip Terraform-module ceremony (professional experience covers it). Freed days → W3 spillover.
- **W5 networking**: keep NCCL/topology-aware placement half (feeds W3); cut Cilium/WireGuard builds to reading.
- **W6 storage**: narrow to checkpoint-I/O engineering — async checkpointing (<100ms stall), tiering, restore-latency benchmarks, CAS dedup.
- **W7 consolidation**: + paper sprint #3.

## ⚠️ RE-EVALUATION GATE #1 — before starting Anvil at all (~Nov 2026, after Forge W20 or overlapping W18–20)

**Trigger:** Forge Phase 3 winding down; before Anvil W1 begins.
**Action checklist:**
1. Re-read `original_artifacts/plan_evolution_v2_2026-07.md` §B2 (Anvil risk table) + §E2 (revised map); score §A's predictions with 4 months of hindsight.
2. Fresh landscape check (web): Kueue/KAI/JobSet releases; DRA ecosystem maturity (NVIDIA DRA driver status — does W2's simulated-device plan still make sense, or is real-GPU DRA on the host now the better lab?); K3s/K8s versions; anything new in training-orchestration land that changes W3's benchmark baseline.
3. Amend W1–W7 specs as needed (the two v2 specs included — they were written in July for a November start).
4. **Capture for the high-level plan review:** confirm or revise the Phase B surgery (W8–10 RL flagship, W11–13 replacements) before writing those specs.

## ⚠️ RE-EVALUATION GATE #2 — end of Phase A / before Week 8 (~Dec 2026–Jan 2027) — the big one

**Trigger:** W7 consolidation complete.
**Why this gate matters most:** the v2 plan's centerpiece — **W8–10 RL post-training infrastructure flagship** (environment API + rollout workers on the Forge engine + GRPO trainer + async weight sync + reward-hacking telemetry) — deliberately has **no specs yet**. They must be written at this gate, against the RL-infra landscape of early 2027, not July 2026. This is also when applications begin (Jan–Feb 2027), so scope W8–10 to produce demo-able artifacts fast.

**Action checklist:**
1. Fresh landscape check: verl/OpenRLHF/RLinf/OpenTinker architectures as of then; GRPO/DAPO successor algorithms; RL-environment standards (HUD-style SDKs, MCP-based envs); trainer↔engine weight-sync and logprob-consistency practice (importance-sampling corrections — still the live issue?).
2. Write `specs/phaseB/week08/09/10` (RL flagship, per v2 doc §E2 W8–10 detail) + regenerate W11 (GPU-failure chaos + goodput SLOs), W12 (cost-aware routing + queueing/capacity), W13 (eval-gated lifecycle + judge rigor).
3. Re-check Phase C bets (W15 compress, W16 GPU-health flagship, W17–18) — amend the standing deltas below if the fleet-reliability landscape moved (DRA health signaling, DCGM changes).
4. **Capture for the high-level plan review:** portfolio-readiness vs application timeline; which flagships are demo-ready; what W19–20 must prioritize.

## Standing deltas for Phase B/C (provisional — finalize at Gate #2)

- **W8–10**: replaced wholesale by the RL post-training infrastructure flagship (specs to be written at Gate #2). v1 SRE/security/cost content: goodput SLOs + error budgets fold into new W11; cost attribution folds into new W12; Vault depth cut to ~2 days (model signing + GPU quotas only) inside a consolidation week.
- **W11**: GPU/training-specific chaos only (Xid, ECC, thermal, NCCL timeout, node death mid-training → checkpoint recovery) + the folded SRE content.
- **W12**: multi-cluster federation replaced by cost-aware routing + M/M/c capacity modeling validated against measurement (absorbs v1 W18's queueing content).
- **W13**: compress to eval-gated promotion + LLM-judge rigor (paired bootstrap, position-bias controls) + canary/rollback; drop feature store.
- **W15**: compress CLI/IDP to 2–3 days; freed time → W16.
- **W16 GPU health controller** (flagship — keep/expand): DRA-native device health, DCGM Xid/ECC/thermal taxonomy, cordon→checkpoint→migrate integrated with W3.
- **W17**: keep AI-specific observability (train→serve correlation IDs, drift detection, Xid/OOM log alerts); trim Thanos depth.
- **W18**: perf-regression CI gates + hardening (queueing content moved to W12).
- **W19–20**: + combined Forge+Anvil+RL demo (code → RL training with rollouts on own engine → checkpoint → health-event migration → deploy → serve); applications launched by early Feb 2027.

## Bookkeeping

- `progress.md` updated with v2 week names for W1–W2 and gate reminders; log all gate outcomes there.
- Convention: specs are immutable once a week is *complete*; unexecuted specs are freely regenerable at gates.
