# Anvil: AI Infrastructure Platform Engineering

## Vision

Anvil is a production-grade AI infrastructure platform that demonstrates mastery of distributed systems, GPU cluster orchestration, reliability engineering, cost optimization, and platform engineering. While Forge proves you can build fast AI systems, Anvil proves you can RUN them reliably at scale.

Together, Forge + Anvil make the engineer a full-stack AI infrastructure candidate — the rarest and most in-demand profile at top AI firms.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Anvil Platform                                │
├─────────────────────────────────────────────────────────────────┤
│  Developer Portal (CLI, Self-Service, Documentation)             │
├─────────────────────────────────────────────────────────────────┤
│  Control Plane                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Job          │  │ Model        │  │ Experiment           │  │
│  │ Orchestrator │  │ Lifecycle    │  │ Tracking             │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Scheduling Layer                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ GPU-Aware    │  │ Gang         │  │ Network-Aware        │  │
│  │ Scheduler    │  │ Scheduling   │  │ Placement            │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ K8s Cluster  │  │ Storage      │  │ Networking           │  │
│  │ (Multi-Node) │  │ (MinIO+PG)   │  │ (CNI+Policies)       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Reliability Layer                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ SLOs + Error │  │ Chaos        │  │ Cost                 │  │
│  │ Budgets      │  │ Engineering  │  │ Optimization         │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Security (Vault, RBAC, Network Policies, Model Signing)         │
├─────────────────────────────────────────────────────────────────┤
│  Observability (Prometheus, Grafana, Loki, OTel, Alerting)       │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

- **Language**: Python (primary), Go (K8s controllers if desired), Bash
- **Orchestration**: Kubernetes (K3s multi-node), custom schedulers, CRDs, operators (kopf)
- **IaC**: Terraform or Pulumi, ArgoCD (GitOps)
- **Distributed Systems**: Raft consensus, distributed KV store, leader election
- **Observability**: Prometheus, Grafana, Loki, OpenTelemetry, Jaeger
- **Security**: HashiCorp Vault, K8s NetworkPolicies, RBAC, cosign (image signing)
- **Storage**: MinIO (S3-compatible), PostgreSQL, Redis
- **Networking**: Calico/Cilium CNI, service mesh concepts
- **ML Lifecycle**: MLflow (or custom), experiment tracking, model registry
- **Testing**: Chaos Mesh, Locust, pytest, property-based testing

## Phases

| Phase | Weeks | Focus | Outcome |
|-------|-------|-------|---------|
| Phase A | 1-7 | Distributed Systems + Cluster Orchestration | Raft, custom scheduler, job orchestrator, IaC, GitOps |
| Phase B | 8-14 | Reliability, Cost, Security | SRE practices, chaos engineering, multi-cluster, ML lifecycle |
| Phase C | 15-20 | Advanced Platform Engineering | Developer platform, advanced K8s, observability at scale |

## Hardware

- ASUS ROG Strix SCAR 16
- Intel Core Ultra 9 HX (for running multiple VMs/containers)
- NVIDIA RTX 5080 (16GB VRAM)
- 32GB RAM (for multi-node K3s cluster via VMs)
- 2TB SSD (for model/checkpoint storage)
- Ubuntu (dual-boot)

## Key Architectural Decisions

(Updated as decisions are made during implementation)

1. [TBD] — Will be filled as you progress

## How to Start a New Session

Paste this into any new agent/chat session:

```
I'm working on Anvil, an AI infrastructure platform.
Read the manifest at /Users/jmalviya/Documents/zz/dev/plan_00/anvil/MANIFEST.md
Check progress at /Users/jmalviya/Documents/zz/dev/plan_00/anvil/progress.md
I'm currently on Week [N]. The spec is at /Users/jmalviya/Documents/zz/dev/plan_00/anvil/specs/phase[X]/week[NN]-[name].md
I need help with: [specific ask]
```

## Related Project

Forge (GPU Inference Orchestrator) at /Users/jmalviya/Documents/zz/dev/plan_00/forge/ — the complementary inference project from Months 1-5.
