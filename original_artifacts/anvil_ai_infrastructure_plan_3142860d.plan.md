---
name: Anvil AI Infrastructure Plan
overview: "A 20-week (5-month) plan for Months 6-10 covering AI Infrastructure Engineering: distributed systems, GPU cluster orchestration, reliability engineering, networking, storage, cost optimization, and advanced platform engineering. Complements the Forge inference plan to create a complete AI systems profile."
todos:
  - id: anvil-w1
    content: "Phase A, Week 1: Implement Raft consensus + distributed KV store with failure scenarios"
    status: pending
  - id: anvil-w2
    content: "Phase A, Week 2: K8s deep dive — custom scheduler, admission webhooks, multi-node K3s"
    status: pending
  - id: anvil-w3
    content: "Phase A, Week 3: Training job orchestrator — CRD, gang scheduling, checkpointing, failure recovery"
    status: pending
  - id: anvil-w4
    content: "Phase A, Week 4: Infrastructure as Code — Terraform modules, GitOps with ArgoCD, spot instance handling"
    status: pending
  - id: anvil-w5
    content: "Phase A, Week 5: Networking — topology-aware scheduling, network policies, service mesh, observability"
    status: pending
  - id: anvil-w6
    content: "Phase A, Week 6: Storage systems — model registry, checkpoint storage, dataset versioning, CLI tool"
    status: pending
  - id: anvil-w7
    content: "Phase A, Week 7: Phase A consolidation — integration tests, architecture docs, blog post"
    status: pending
  - id: anvil-w8
    content: "Phase B, Week 8: SRE practices — SLIs/SLOs, error budgets, incident management, capacity planning"
    status: pending
  - id: anvil-w9
    content: "Phase B, Week 9: Cost optimization — GPU utilization tracking, sharing/time-slicing, spot management, right-sizing"
    status: pending
  - id: anvil-w10
    content: "Phase B, Week 10: Security — Vault, network policies, model signing, RBAC, audit logging"
    status: pending
  - id: anvil-w11
    content: "Phase B, Week 11: Chaos engineering — failure injection, gameday exercises, resilience scorecard"
    status: pending
  - id: anvil-w12
    content: "Phase B, Week 12: Multi-cluster federation — cross-cluster routing, failover, progressive model rollout"
    status: pending
  - id: anvil-w13
    content: "Phase B, Week 13: ML lifecycle — experiment tracking, model promotion, A/B testing, feature store"
    status: pending
  - id: anvil-w14
    content: "Phase B, Week 14: Phase B consolidation — full lifecycle demo, blog post, security audit"
    status: pending
  - id: anvil-w15
    content: "Phase C, Week 15: Internal developer platform — self-service CLI, abstractions, documentation"
    status: pending
  - id: anvil-w16
    content: "Phase C, Week 16: Advanced K8s — GPU health controller, VPA for AI, disruption budgets"
    status: pending
  - id: anvil-w17
    content: "Phase C, Week 17: Observability at scale — high-cardinality metrics, log aggregation, model monitoring"
    status: pending
  - id: anvil-w18
    content: "Phase C, Week 18: Infrastructure performance engineering — benchmarks, optimization, capacity modeling"
    status: pending
  - id: anvil-w19-20
    content: "Phase C, Weeks 19-20: Portfolio integration — unified Forge+Anvil architecture, 4 blog posts, demo video, open-source release"
    status: pending
isProject: false
---

# Anvil: AI Infrastructure Platform Engineering (20 Weeks / Months 6-10)

## Strategic Positioning

**Forge** (Months 1-5) answers: "Can you build fast AI systems?"
**Anvil** (Months 6-10) answers: "Can you run them reliably at scale?"

Together, they make you a full-stack AI infrastructure engineer — the rarest and most in-demand profile at Anthropic, OpenAI, xAI, and peer firms.

## Philosophy: "Think Distributed, Build Locally, Prove Rigor"

You have one machine. These companies have thousands of GPUs. The trick: build systems that DEMONSTRATE distributed thinking even on local hardware (multi-process, multi-container, simulated failures). The concepts transfer directly — scale is just a multiplier on the same patterns.

---

## Phase A: Distributed Systems + Cluster Orchestration (Weeks 1-7)

### Week 1: Distributed Systems Fundamentals (The Foundation)

**Why this matters**: Every infrastructure interview at these companies tests distributed systems. Not textbook knowledge — practical understanding of consensus, failure modes, and coordination.

**Build**:
- Implement a simplified Raft consensus algorithm in Python:
  - Leader election with term numbers and timeouts
  - Log replication across 3-5 nodes (use separate processes on localhost)
  - Handle leader failure and re-election
  - Implement basic log compaction (snapshotting)
- Build a distributed key-value store on top of your Raft implementation:
  - Put/Get/Delete operations that go through consensus
  - Linearizable reads (read from leader only)
  - Handle network partitions (simulate with dropped messages)
- Run failure scenarios:
  - Kill the leader, observe re-election
  - Partition 2 nodes from the other 3, observe split-brain prevention
  - Bring partitioned nodes back, observe log catch-up

**Deliverable**: A working Raft implementation + KV store that survives node failures. Test suite proving correctness under failures.

**What you learn**: Consensus, leader election, fault tolerance, the CAP theorem in practice. This is the #1 topic in distributed systems interviews.

---

### Week 2: Container Orchestration Deep Dive (Beyond kubectl)

**Why this matters**: Everyone "uses" Kubernetes. Almost nobody understands HOW it works internally. This week you go from user to operator to extender.

**Build**:
- Study and document the K8s control plane internals:
  - How the scheduler actually works (predicates + priorities, scoring)
  - How the controller manager reconciles desired vs actual state
  - How etcd stores cluster state (watch mechanism)
  - How the kubelet manages pods on a node
- Build a custom K8s scheduler (Python, using the scheduler extender API):
  - GPU-topology-aware scheduling: prefer co-locating related pods on same node
  - VRAM-aware scoring: don't over-commit GPU memory
  - Anti-affinity: spread redundant model replicas across nodes
  - Priority classes: training jobs can preempt inference if needed
- Admission webhooks:
  - Mutating webhook: automatically inject GPU monitoring sidecar into AI pods
  - Validating webhook: reject pods that request more VRAM than any node has
  - Resource quota enforcement: per-namespace GPU limits
- Deploy on a multi-node K3s cluster:
  - Set up 3 K3s nodes (can be VMs with multipass or LXD on your machine)
  - Demonstrate scheduling decisions across nodes
  - Show preemption behavior

**Deliverable**: Custom K8s scheduler + admission webhooks deployed on multi-node K3s. Demo showing GPU-aware scheduling decisions.

---

### Week 3: Training Job Orchestration

**Why this matters**: At AI labs, the most critical infrastructure is the training platform. Training runs cost millions and last weeks. Making them reliable is THE hard problem.

**Build**:
- Training Job Manager (like a simplified Volcano/Kubeflow Training Operator):
  - Define a `TrainingJob` CRD: model config, dataset, resources, checkpoint policy
  - Job lifecycle: Pending -> Scheduling -> Running -> Checkpointing -> Completed/Failed
  - Gang scheduling: all workers for a job must be scheduled together (or none)
  - Preemption: lower-priority jobs yield resources to higher-priority ones
- Checkpointing and resumption:
  - Periodic checkpointing (save model state, optimizer state, step number)
  - On failure: detect failure, find latest checkpoint, resume from that point
  - Async checkpointing (don't block training to save checkpoint)
  - Checkpoint garbage collection (keep last N, delete older)
- Failure handling:
  - Detect node failure (heartbeat timeout)
  - Detect GPU failure (ECC errors, thermal shutdown)
  - Automatic rescheduling on healthy nodes
  - Exponential backoff for repeated failures
  - Dead-letter notification for permanently failed jobs
- Job queue and fairness:
  - Multi-tenant job queue (team A and team B share cluster)
  - Fair-share scheduling (proportional to allocated quota)
  - Priority with preemption (urgent job bumps lower priority)
  - Backfill: use idle resources for low-priority jobs without blocking high-priority

**Deliverable**: Training job orchestrator with CRD, checkpointing, failure recovery, and fair-share scheduling. Demo: start training, kill a worker mid-run, observe automatic recovery from checkpoint.

---

### Week 4: Infrastructure as Code for GPU Clusters

**Why this matters**: Real AI infra teams manage hundreds of GPU instances across clouds. Doing it manually is impossible.

**Build** (simulate with local VMs + cloud-like abstractions):
- Terraform/Pulumi modules for GPU infrastructure:
  - Define reusable modules: GPU node pool, networking, storage, monitoring
  - Environment management: dev/staging/prod with variable files
  - State management: remote state with locking
  - Plan/Apply workflow with PR-based review
- Auto-scaling for AI workloads:
  - Scale-up: when job queue depth exceeds threshold, provision more GPU nodes
  - Scale-down: when GPU utilization < 30% for 15 min, drain and terminate
  - Spot/Preemptible instance handling:
    - Request spot instances for fault-tolerant training (with checkpointing)
    - Detect preemption notice, checkpoint immediately, reschedule
  - Cost tracking per job/team
- Configuration management:
  - NVIDIA driver installation automation
  - CUDA toolkit version pinning per cluster
  - Container runtime configuration (nvidia-container-runtime)
  - Network configuration (high-bandwidth between GPU nodes)
- GitOps deployment (ArgoCD):
  - Cluster config stored in Git
  - ArgoCD watches repo, applies changes automatically
  - Progressive rollout: canary -> percentage -> full
  - Automatic rollback on health check failure

**Deliverable**: IaC modules for GPU cluster provisioning + ArgoCD GitOps pipeline. Demo: merge a PR, watch ArgoCD deploy the change, verify with health checks.

---

### Week 5: Networking for AI Workloads

**Why this matters**: Network is the #1 bottleneck in distributed training. Understanding it separates AI infra engineers from generic DevOps.

**Build/Study**:
- Understand the networking stack for AI (conceptual + practical):
  - Why RDMA/InfiniBand matters (bypass kernel, direct memory access between GPUs)
  - NCCL (NVIDIA Collective Communications Library): what it does, how to configure it
  - Network topologies: fat-tree, rail-optimized, fully-connected
  - Why "GPU locality" matters: NVLink (intra-node) vs InfiniBand (inter-node)
- Build a network-aware job scheduler:
  - Model the cluster topology (which nodes share a switch, which share a rack)
  - Schedule multi-node jobs to minimize cross-rack communication
  - Implement bandwidth reservation (don't over-subscribe inter-rack links)
  - Show: same training job is faster when scheduled on co-located nodes
- Practical networking for your K3s cluster:
  - Configure Calico/Cilium CNI with network policies
  - Implement pod-to-pod encryption for model weights in transit
  - Service mesh basics (Istio/Linkerd) — traffic management, mTLS
  - DNS-based service discovery for distributed training workers
- Network observability:
  - Measure inter-pod bandwidth and latency
  - Detect network congestion (packet drops, retransmits)
  - Alert on degraded network performance affecting training jobs

**Deliverable**: Network-aware scheduler + network observability dashboard. Document explaining why network topology matters for AI workloads with diagrams.

---

### Week 6: Storage Systems for ML

**Why this matters**: AI workloads have unique storage patterns — massive datasets, huge checkpoints, model artifacts. Generic storage solutions don't work.

**Build**:
- Model Registry and Artifact Store:
  - Version-controlled model storage (like MLflow model registry)
  - Metadata: training config, metrics, lineage (which dataset, which code commit)
  - Model promotion workflow: staging -> canary -> production
  - Artifact deduplication (don't store the same base model 50 times)
- Checkpoint storage system:
  - Fast checkpoint writes (async, parallel, compressed)
  - Tiered storage: hot (local NVMe) -> warm (NFS/shared) -> cold (object store)
  - Lifecycle management: promote successful checkpoints, garbage-collect failed ones
  - Checkpoint sharing: multiple jobs can resume from same base checkpoint
- Dataset management:
  - Dataset versioning (track which data was used for which training run)
  - Distributed caching: pre-load dataset shards to nodes before training starts
  - Streaming data loading (don't require full dataset download before training)
- Implement using:
  - MinIO (S3-compatible object store, runs locally)
  - PostgreSQL (metadata)
  - Redis (caching layer)
  - Build CLI tool: `anvil model push`, `anvil model pull`, `anvil checkpoint restore`

**Deliverable**: Model registry + checkpoint storage system with CLI. Demo: push model, version it, promote to production, restore from checkpoint.

---

### Week 7: Phase A Consolidation

**Build**:
- Integration testing across all Phase A components:
  - Submit training job -> scheduler places it -> training runs -> checkpoints -> completes
  - Simulate failure mid-training -> observe recovery
  - Show GitOps deploy of new scheduler config -> ArgoCD applies it
- Architecture documentation:
  - System design document for the full orchestration layer
  - Sequence diagrams for key flows (job submission, failure recovery, scaling)
  - ADRs for Phase A decisions
- Blog post: "Building a Training Job Orchestrator: Lessons in Distributed Systems"
- Code quality:
  - Tests for failure scenarios (property-based testing for distributed correctness)
  - Clean interfaces between components
  - README with architecture overview

**Phase A Milestone**: You can now discuss distributed systems, cluster scheduling, and training infrastructure with confidence. You understand the problems these companies solve daily.

---

## Phase B: Reliability, Cost, and Security (Weeks 8-14)

### Week 8: SRE Practices for AI Systems

**Build**:
- Define SLIs/SLOs for AI services:
  - Inference: availability (% requests served), latency (p99 < Xms), quality (no degraded output)
  - Training: job completion rate, checkpoint success rate, GPU utilization target
  - Platform: API uptime, scheduling latency, storage throughput
- Error budget implementation:
  - Calculate error budget: if SLO is 99.9%, budget is 0.1% failure per month
  - Track budget consumption in real-time
  - Policy: when budget exhausted, freeze feature deployments until reliability improves
- Incident management:
  - Define severity levels (P1-P4) with response time expectations
  - Build automated incident detection (alert -> PagerDuty-style notification -> runbook link)
  - Post-incident review template (timeline, root cause, action items)
  - Implement automated remediation for known issues (self-healing)
- Capacity planning:
  - Forecast GPU demand based on historical job submissions
  - Alert when projected demand exceeds capacity in N days
  - Recommend: add nodes, optimize existing workloads, or queue lower-priority work

**Deliverable**: SLO dashboard, error budget tracker, incident management system, capacity planning tool.

---

### Week 9: Cost Optimization (The $M Problem)

**Build**:
- GPU utilization tracker:
  - Per-GPU utilization over time (compute %, memory %)
  - Per-job cost attribution (this training run cost $X based on GPU-hours consumed)
  - Idle GPU detection: alert when GPUs are allocated but underutilized
  - Team/project cost breakdown dashboards
- GPU sharing and time-slicing:
  - MPS (Multi-Process Service): run multiple inference workloads on one GPU
  - MIG (Multi-Instance GPU): partition a GPU into isolated instances (concept — A100/H100 specific, but understand the API)
  - Time-slicing: round-robin GPU access for multiple pods
  - Implement and benchmark: when does sharing help vs hurt?
- Spot/Preemptible instance management:
  - Spot instance bid strategy (which instance types, which AZs)
  - Preemption handler: detect 2-minute warning, checkpoint, gracefully terminate
  - Fallback to on-demand if spot unavailable (with cost alerting)
  - Measure: cost savings vs interruption rate
- Right-sizing recommendations:
  - Analyze historical GPU utilization per job type
  - Recommend: "this job only uses 8GB VRAM, move from A100 to L4"
  - Automated enforcement: reject over-provisioned resource requests

**Deliverable**: Cost dashboard with per-team attribution, GPU sharing implementation, spot instance handler. Show: "this optimization saved X% on GPU costs."

---

### Week 10: Security for AI Infrastructure

**Build**:
- Secrets management:
  - Vault (HashiCorp) or Sealed Secrets integration
  - Automatic secret rotation (API keys, DB credentials, model access tokens)
  - Secret injection into pods (no secrets in environment variables or config maps)
  - Audit log: who accessed which secret, when
- Network security:
  - Kubernetes NetworkPolicies: isolate namespaces (training can't talk to production inference)
  - Pod Security Standards: restricted profile (no privileged containers, no host network)
  - mTLS between all services (automatic with service mesh)
  - Egress filtering: pods can only reach approved external endpoints
- Model supply chain security:
  - Model artifact signing (verify model wasn't tampered with between training and deployment)
  - Container image scanning (no known vulnerabilities in inference images)
  - SBOM (Software Bill of Materials) for ML containers
  - Provenance tracking: which code + data produced this model?
- Access control:
  - RBAC: team A can deploy to their namespace only
  - GPU quotas: enforce per-team GPU limits
  - Audit logging: all kubectl commands, all model deployments
  - Break-glass procedure: emergency access with full audit trail

**Deliverable**: Secured cluster with Vault, network policies, signed model artifacts, RBAC, and audit logging. Security architecture document.

---

### Week 11: Chaos Engineering and Resilience Testing

**Build**:
- Chaos framework (using Chaos Mesh or LitmusChaos on your K3s cluster):
  - GPU failure simulation: make a GPU "unavailable" (unbind driver)
  - Node failure: kill a K3s agent node, observe rescheduling
  - Network chaos: inject latency, packet loss, partition between nodes
  - Storage failure: make checkpoint storage temporarily unavailable
  - CPU/Memory pressure: starve pods of resources
- Gameday exercises (documented scenarios):
  - Scenario 1: "GPU node dies during training" -> observe checkpoint recovery
  - Scenario 2: "Network partition between training workers" -> observe NCCL timeout handling
  - Scenario 3: "Model registry is down during deployment" -> observe graceful degradation
  - Scenario 4: "Disk full on checkpoint storage" -> observe alerting and cleanup
- Resilience scoring:
  - For each chaos experiment: did the system recover? How long? Any data loss?
  - Build a "resilience scorecard" showing system behavior under each failure type
  - Identify gaps and fix them
- Circuit breakers and bulkheads:
  - Implement circuit breaker between inference service and model loading
  - Bulkhead: isolate tenant workloads (one tenant's OOM doesn't affect others)
  - Implement graceful degradation: serve cached/smaller model when primary fails

**Deliverable**: Chaos experiment suite with documented results, resilience scorecard, and improvements made. Blog post: "Chaos Engineering for GPU Workloads: What Breaks and How to Fix It."

---

### Week 12: Multi-Cluster and Federation

**Build**:
- Multi-cluster setup (3 K3s clusters representing "regions"):
  - Cluster A: "us-west" (primary training)
  - Cluster B: "us-east" (primary inference)
  - Cluster C: "eu" (inference with data locality)
- Cross-cluster service discovery:
  - Services in one cluster can discover and call services in another
  - Implement with DNS-based federation or service mesh (Istio multi-cluster)
- Workload distribution:
  - Route inference requests to nearest cluster (latency-based)
  - Failover: if one cluster's inference is unhealthy, route to another
  - Training: schedule on cluster with most available GPU capacity
- Model distribution:
  - Train in cluster A -> push model to registry -> deploy to clusters B and C
  - Implement progressive rollout: deploy to one cluster first, verify, then expand
  - Rollback: if metrics degrade in cluster B, revert to previous model
- Global observability:
  - Aggregate metrics across clusters into single dashboard
  - Cross-cluster tracing (request starts in cluster A, calls service in cluster B)
  - Alert correlation: "high latency in eu-cluster correlates with network issue"

**Deliverable**: Multi-cluster deployment with cross-cluster routing, failover, and progressive model rollout. Global observability dashboard.

---

### Week 13: Experiment Tracking and ML Lifecycle

**Build**:
- Experiment tracking system (custom or MLflow-based):
  - Track: hyperparameters, metrics (loss curves), artifacts (checkpoints, plots)
  - Compare experiments: side-by-side metric comparison
  - Reproducibility: log exact code commit, dependencies, data version for each run
  - Integration with your training job orchestrator (auto-log everything)
- Model lifecycle management:
  - Stages: experiment -> candidate -> staging -> production -> retired
  - Promotion gates: candidate must pass eval suite before staging
  - A/B testing in production: route 5% traffic to candidate, compare metrics
  - Automatic rollback: if candidate metrics worse than production, revert
- Feature store (simplified):
  - Centralized storage for processed features used across models
  - Point-in-time correctness (avoid data leakage in training)
  - Serve features at inference time (low-latency lookup)
  - Feature versioning and lineage
- Data versioning:
  - Track dataset versions used for each training run
  - Dataset comparison: what changed between v1 and v2?
  - Data quality checks before training (schema validation, distribution checks)

**Deliverable**: Experiment tracking + model lifecycle system integrated with the training orchestrator. Demo: run experiment -> promote winner -> deploy to production -> monitor.

---

### Week 14: Phase B Consolidation

**Build**:
- Integration scenario: Full lifecycle demo
  - Submit training job -> runs with checkpointing -> completes
  - Model passes eval gate -> promoted to staging
  - Progressive rollout to production across clusters
  - Monitor with SLOs, detect degradation, auto-rollback
  - Cost report generated for the entire workflow
- Phase B blog post: "Building Reliable AI Infrastructure: SRE, Security, and Cost at Scale"
- Architecture Decision Records for Phase B
- Security audit of entire system (document findings and fixes)
- Update portfolio README with infrastructure architecture diagrams

---

## Phase C: Advanced Platform Engineering (Weeks 15-20)

### Week 15: Internal Developer Platform (The Platform-for-Platforms)

**Build**:
- Self-service portal for AI engineers (CLI + simple web API):
  - `anvil train submit job.yaml` — submit training job
  - `anvil model deploy my-model --replicas 3` — deploy model to inference
  - `anvil cluster status` — GPU availability across clusters
  - `anvil cost report --team my-team --month june` — cost breakdown
- Platform abstractions:
  - AI engineers shouldn't need to know K8s YAML
  - Abstract away: GPU types, node selection, storage classes, networking
  - Expose only: model path, resource requirements, scaling policy
- Developer documentation:
  - Getting started guide (0 to serving in 15 min)
  - Architecture overview for platform team
  - Troubleshooting guide (common issues and solutions)
  - API reference for the CLI

**Deliverable**: Developer platform CLI + documentation. A platform that OTHER engineers could use without knowing the infrastructure details.

---

### Week 16: Advanced Kubernetes Patterns

**Build**:
- Custom controllers beyond the operator:
  - GPU health monitor: watch for ECC errors, thermal throttling, driver crashes
  - Automatic node drain on GPU degradation
  - Capacity reporter: custom metric for "available VRAM" per node
- Vertical Pod Autoscaler for AI:
  - Monitor actual VRAM usage vs requested
  - Recommend right-sized requests
  - Auto-resize in place (without pod restart) where possible
- Pod disruption budgets for AI:
  - Training: never evict more than 1 worker at a time (maintain training progress)
  - Inference: always maintain N-1 replicas during rolling updates
- Custom Resource lifecycle:
  - Implement finalizers properly (cleanup GPU resources on deletion)
  - Status conditions (rich status reporting for debugging)
  - Events (emit K8s events for job state transitions)

**Deliverable**: Advanced K8s controllers for GPU health, autoscaling, and disruption management.

---

### Week 17: Observability at Scale

**Build**:
- High-cardinality metrics handling:
  - Per-request metrics without blowing up Prometheus (use exemplars)
  - Metrics aggregation for 1000s of pods (recording rules, federation)
  - Long-term metric storage (Thanos or Mimir)
- Log aggregation:
  - Centralized logging (Loki or ELK)
  - Structured logging with correlation IDs across all services
  - Log-based alerting (detect error patterns)
- AI-specific observability:
  - Model performance monitoring (detect output quality degradation)
  - Data drift detection (input distribution changing over time)
  - Inference anomaly detection (sudden latency spike, output length change)
- Dashboard-as-code:
  - Grafana dashboards defined in JSON/YAML, deployed via GitOps
  - Per-team dashboard templates (self-service)
  - Executive dashboard: cluster health, cost, utilization at a glance

**Deliverable**: Production-grade observability stack with log aggregation, model monitoring, and dashboard-as-code.

---

### Week 18: Performance Engineering for Infrastructure

**Build**:
- Benchmark suite for infrastructure components:
  - Scheduler latency: time from job submit to pod running
  - Checkpoint write/restore speed
  - Model download and load time
  - Cross-cluster communication latency
- Optimization pass:
  - Reduce scheduler decision time (cache node state, batch scheduling decisions)
  - Parallel checkpoint writes (shard across multiple storage backends)
  - Model preloading (predictively pull models before they're needed)
  - Connection pooling and keep-alive for inter-service communication
- Capacity modeling:
  - Build a capacity model: given N GPUs, how many concurrent training + inference jobs?
  - Queuing theory: model the system as M/M/c queue, predict wait times
  - What-if analysis: "if we add 10 more GPUs, how does wait time change?"

**Deliverable**: Infrastructure benchmark suite + optimization documentation + capacity model.

---

### Weeks 19-20: Portfolio Integration and Final Polish

**Build**:
- Unified architecture document showing Forge + Anvil together:
  - How inference (Forge) and infrastructure (Anvil) connect
  - Full system architecture diagram (from GPU kernel to Kubernetes operator)
  - Data flow diagram for complete lifecycle
- Blog posts for Anvil (3-4):
  1. "Implementing Raft: What I Learned About Distributed Consensus"
  2. "Training Job Orchestration: Making Million-Dollar GPU Jobs Reliable"
  3. "Chaos Engineering for GPU Infrastructure: A Practical Guide"
  4. "Building an Internal AI Platform: From kubectl to Self-Service"
- Combined demo video (5-7 min):
  - Show: submit training -> checkpoint -> model ready -> deploy for inference -> scale -> survive failure -> cost report
  - This single demo covers BOTH plans end-to-end
- Open-source release:
  - Clean up both repos
  - Write CONTRIBUTING.md
  - Tag v1.0 releases
  - Submit to relevant awesome-lists and HN/Reddit

---

## The Combined Resume After 10 Months

> "Built Forge + Anvil: an open-source AI infrastructure platform featuring a custom inference engine with continuous batching and KV-cache management (Forge), and a distributed training orchestrator with Raft-based consensus, checkpoint recovery, multi-cluster federation, and a Kubernetes operator for GPU workload scheduling (Anvil). Demonstrated SRE practices with error budgets, chaos engineering for GPU failures, and cost optimization achieving 40% GPU utilization improvement."

---

## Technologies Covered Across Anvil

```
Orchestration:  Kubernetes (deep), K3s, custom schedulers, CRDs, operators
IaC:            Terraform or Pulumi, ArgoCD (GitOps)
Distributed:    Raft consensus, distributed KV store, leader election
Observability:  Prometheus, Grafana, Loki, OpenTelemetry, Jaeger
Security:       Vault, NetworkPolicies, RBAC, image signing, mTLS
Storage:        MinIO (S3), PostgreSQL, Redis, distributed caching
Networking:     Calico/Cilium, service mesh concepts, network-aware scheduling
Cost:           GPU utilization tracking, spot management, time-slicing
ML Lifecycle:   MLflow/custom, experiment tracking, model registry, feature store
Languages:      Python (primary), Go (for K8s controllers if desired), Bash
```

---

## How This Maps to Interview Rounds at AI Firms

| Round | What They Ask | Where You Built It |
|-------|--------------|-------------------|
| Distributed Systems | "Design a training job scheduler that handles failures" | Weeks 1, 3 |
| Infrastructure Design | "Design a multi-region model serving platform" | Weeks 4, 5, 12 |
| Reliability | "How would you make training resilient to GPU failures?" | Weeks 3, 8, 11 |
| Coding | "Implement a leader election algorithm" | Week 1 |
| Production Operations | "Walk me through debugging a training slowdown" | Weeks 8, 13, 17 |
| Cost/Efficiency | "GPU utilization is 30%, how do you improve it?" | Week 9 |
| Security | "How do you secure model artifacts in a multi-tenant cluster?" | Week 10 |
