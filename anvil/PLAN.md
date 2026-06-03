# Anvil: Full Narrative Plan

The detailed narrative plan for Anvil lives in:
`/Users/jmalviya/.cursor/plans/anvil_ai_infrastructure_plan_3142860d.plan.md`

It contains the full week-by-week philosophy, deliverables, technology breakdown, and interview mapping.

For quick reference, see MANIFEST.md in this directory for architecture and session start templates.
For week-by-week execution, see the specs/ directory.

## Quick Summary

**Philosophy**: "Think Distributed, Build Locally, Prove Rigor"

- **Phase A (Weeks 1-7)**: Distributed systems + cluster orchestration (Raft, custom K8s scheduler, training job orchestrator, IaC, GitOps)
- **Phase B (Weeks 8-14)**: Reliability, cost, security (SRE, chaos engineering, multi-cluster, ML lifecycle)
- **Phase C (Weeks 15-20)**: Advanced platform engineering (developer platform, advanced K8s, observability at scale, portfolio)

## Resume Line

> "Built Anvil, a distributed AI infrastructure platform featuring Raft-based consensus, custom K8s GPU scheduler, training job orchestrator with checkpoint recovery, multi-cluster federation, chaos engineering suite, and a self-service developer CLI. Demonstrated SRE practices with error budgets, achieving 40% GPU utilization improvement through automated cost optimization."

## Interview Mapping

| Round | What They Ask | Where You Built It |
|-------|--------------|-------------------|
| Distributed Systems | "Design a training job scheduler" | Weeks 1, 3 |
| Infrastructure Design | "Design multi-region serving" | Weeks 4, 5, 12 |
| Reliability | "Make training resilient to GPU failures" | Weeks 3, 8, 11 |
| Cost/Efficiency | "GPU utilization is 30%, improve it" | Week 9 |
| Security | "Secure model artifacts in multi-tenant" | Week 10 |

## Daily Routine

- **1.5 hours**: Learn (distributed systems talks, K8s source code, SRE books)
- **4-5 hours**: Build (code, deploy, test failure scenarios)
- **30 min**: Document (architecture diagrams, ADRs, runbooks)
- **30 min**: Push (commit, update progress)
