# Anvil Progress Tracker

> Last updated: 2026-07-06
> Current Phase: Phase A (not started — begins ~Nov 2026 after Forge)
> Current Week: Week 1
> **Plan v2 (2026-07-06):** W1/W2 specs regenerated per the strategic review (`original_artifacts/plan_evolution_v2_2026-07.md`); v1 originals archived in `original_artifacts/specs_v1/anvil_phaseA/`. ⚠️ **Two re-evaluation gates in `specs/V2_DELTAS_AND_REEVAL.md`: Gate #1 before starting Anvil at all (~Nov 2026 — re-validate even the v2 specs, written 4 months early), Gate #2 before Phase B (write the W8–10 RL-infrastructure flagship specs then, against the early-2027 landscape).** Do not start any week without checking that file.

## Phase A: Distributed Systems + Cluster Orchestration (Weeks 1-7)

- [ ] **Week 1**: Raft Essentials + Collectives — Raft (3 days) + ring allreduce/NCCL concepts (v2)
- [ ] **Week 2**: K8s for AI — DRA + Kueue + admission webhooks (v2, replaces scheduler-extender/device-plugin pattern)
- [ ] **Week 3**: Training Orchestrator — job CRD, gang scheduling, checkpointing, recovery
- [ ] **Week 4**: Infrastructure as Code — Terraform modules, ArgoCD GitOps, spot handling
- [ ] **Week 5**: Networking — topology-aware scheduling, network policies, CNI
- [ ] **Week 6**: Storage Systems — model registry, checkpoint store, dataset versioning
- [ ] **Week 7**: Consolidation — integration tests, architecture docs, blog post

## Phase B: Reliability, Cost, Security (Weeks 8-14)

> ⚠️ **Provisional — restructured by plan v2, specs to be written at Gate #2 (end of Phase A).** W8–10 become the **RL post-training infrastructure flagship** (environments + rollout workers on the Forge engine + GRPO trainer + weight sync + reward-hacking telemetry); W11 → GPU-failure chaos + goodput SLOs; W12 → cost-aware routing + queueing/capacity modeling; W13 → eval-gated lifecycle + LLM-judge rigor. The v1 list below stands only until that gate; capture the gate's findings for the next high-level plan review.

- [ ] **Week 8**: SRE Practices — SLIs/SLOs, error budgets, incident management, capacity planning
- [ ] **Week 9**: Cost Optimization — GPU utilization tracking, sharing, spot management
- [ ] **Week 10**: Security — Vault, network policies, model signing, RBAC, audit
- [ ] **Week 11**: Chaos Engineering — failure injection, gameday exercises, resilience scoring
- [ ] **Week 12**: Multi-Cluster — cross-cluster routing, failover, progressive rollout
- [ ] **Week 13**: ML Lifecycle — experiment tracking, model promotion, A/B testing
- [ ] **Week 14**: Consolidation — full lifecycle demo, blog post, security audit

## Phase C: Advanced Platform Engineering (Weeks 15-20)

- [ ] **Week 15**: Developer Platform — self-service CLI, abstractions, documentation
- [ ] **Week 16**: Advanced K8s — GPU health controller, VPA, disruption budgets
- [ ] **Week 17**: Observability at Scale — high-cardinality metrics, log aggregation, model monitoring
- [ ] **Week 18**: Infrastructure Performance — benchmarks, optimization, capacity modeling
- [ ] **Weeks 19-20**: Portfolio Integration — unified Forge+Anvil architecture, blogs, demo video

## Completion Log

| Week | Date Started | Date Completed | Notes |
|------|-------------|----------------|-------|
| | | | |

## Decisions Made

| # | Decision | Date | Rationale |
|---|----------|------|-----------|
| 1 | Adopted plan evolution v2: W1/W2 specs regenerated (Raft compressed + collectives added; scheduler-extender → DRA/Kueue); Phase B restructure planned; re-eval gates before Anvil start and before Phase B | 2026-07-06 | Strategic review (`original_artifacts/plan_evolution_v2_2026-07.md`): DRA went GA in K8s 1.34 (Mar 2026), invalidating the v1 device-plugin/extender design; RL post-training infra identified as the flagship gap. Phase B/C specs deliberately deferred to just-in-time gates. |

## Blockers / Issues

| Issue | Week | Status | Resolution |
|-------|------|--------|------------|
| | | | |
