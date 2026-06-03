# Week 7: Phase A Consolidation

## Context

**Where it fits:** Phase A, Week 7 — the capstone week. No new features; instead, prove that everything from Weeks 1-6 works together as a cohesive system. Document decisions, write tests, and prepare for Phase B.

**Prerequisites:**
- Weeks 1-6 completed (all components functional individually)
- K3s cluster running with operator, scheduler, storage, networking
- All CLI tools working (`anvil model`, `anvil checkpoint`)
- Hardware: ASUS ROG Strix SCAR 16 (RTX 5080 16GB, 32GB RAM, 2TB SSD, Ubuntu)

**What it builds on:** Every prior week. This week validates the full pipeline end-to-end and produces documentation that makes the system understandable to others (and future-you).

---

## Learning Goals

- [ ] Explain Anvil's architecture end-to-end (from job submission to completion)
- [ ] Articulate design tradeoffs made in Phase A (captured in ADRs)
- [ ] Describe the failure modes and how each is handled
- [ ] Explain what was deliberately deferred to Phase B and why
- [ ] Demonstrate integration testing strategy for distributed systems
- [ ] Articulate lessons learned about building a training orchestrator

---

## Implementation Goals

- [ ] End-to-end integration test: submit → schedule → train → checkpoint → complete
- [ ] Failure recovery integration test: kill worker → checkpoint restore → resume
- [ ] GitOps integration test: merge → ArgoCD sync → verify deployment
- [ ] Network isolation integration test: verify cross-job traffic blocked
- [ ] Architecture documentation with sequence diagrams (Mermaid)
- [ ] Write 3-4 Architecture Decision Records (ADRs)
- [ ] Blog post draft: "Building a Training Job Orchestrator"
- [ ] Code quality pass: linting, type hints, docstrings, dead code removal
- [ ] Performance benchmarks: scheduling latency, checkpoint throughput, recovery time
- [ ] Phase B readiness checklist

---

## Acceptance Criteria

1. Integration test passes: `anvil job submit` → scheduler places workers → simulated training runs → checkpoint saved → job completes with status "Completed".
2. Failure recovery test passes: worker killed mid-training → operator detects within 10s → restores from checkpoint → resumes from correct step.
3. GitOps test passes: config change committed → ArgoCD syncs within 3 minutes → `kubectl get` confirms new config applied.
4. Network test passes: workers in job A cannot reach workers in job B (curl times out).
5. Architecture doc contains: system overview diagram, component interaction sequence diagram, data flow diagram.
6. At least 3 ADRs written covering: consensus approach (Week 1), CRD design (Week 3), storage tiering (Week 6).
7. Blog post is 1500-2500 words, technically substantive, and includes at least one code snippet and one architecture diagram.
8. All Python code passes `ruff check` with zero errors and `mypy --strict` with <10 errors.
9. Benchmark results documented: scheduling decision <500ms, checkpoint save <5s for 1GB, recovery time <30s.
10. Phase B readiness checklist has all items checked: code merged, docs written, infra stable, no critical bugs.

---

## Validation Commands

```bash
# Run full integration test suite
cd ~/anvil
python -m pytest tests/integration/ -v --timeout=300

# End-to-end happy path
python tests/integration/test_e2e_happy_path.py

# Failure recovery test
python tests/integration/test_failure_recovery.py

# GitOps test
python tests/integration/test_gitops_sync.py

# Network isolation test
python tests/integration/test_network_isolation.py

# Code quality
cd ~/anvil
ruff check . --fix
mypy . --strict 2>&1 | tail -5  # Count remaining errors

# Benchmarks
python benchmarks/scheduling_latency.py
python benchmarks/checkpoint_throughput.py
python benchmarks/recovery_time.py

# Generate architecture diagrams
python docs/generate_diagrams.py  # Outputs Mermaid .md files

# Verify all components running
kubectl get pods -A | grep anvil
kubectl get trainingjobs -A
argocd app list

# Build blog post preview
cd docs/blog && python -m http.server 8080  # View at localhost:8080
```

---

## Technical Implementation Details

### Project Structure

```
~/anvil/
├── tests/
│   └── integration/
│       ├── conftest.py                # Shared fixtures
│       ├── test_e2e_happy_path.py     # Full lifecycle
│       ├── test_failure_recovery.py   # Kill + restore
│       ├── test_gitops_sync.py        # Git → ArgoCD → cluster
│       ├── test_network_isolation.py  # Cross-job blocked
│       └── helpers.py                 # Test utilities
├── benchmarks/
│   ├── scheduling_latency.py
│   ├── checkpoint_throughput.py
│   └── recovery_time.py
├── docs/
│   ├── architecture/
│   │   ├── overview.md               # System architecture
│   │   ├── sequence-diagrams.md      # Mermaid sequences
│   │   └── data-flow.md              # Data movement
│   ├── adrs/
│   │   ├── 001-raft-for-consensus.md
│   │   ├── 002-crd-based-job-model.md
│   │   ├── 003-tiered-checkpoint-storage.md
│   │   └── 004-topology-aware-scheduling.md
│   ├── blog/
│   │   └── building-training-orchestrator.md
│   └── phase-b-readiness.md
└── Makefile                           # Top-level commands
```

### End-to-End Integration Test

```python
# tests/integration/test_e2e_happy_path.py
import pytest
import time
import subprocess
import json

TIMEOUT = 180  # 3 minutes max

class TestEndToEndHappyPath:
    """Verify the full job lifecycle works end-to-end."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Ensure clean state before test."""
        subprocess.run(["kubectl", "delete", "trainingjobs", "--all"], capture_output=True)
        time.sleep(5)
        yield
        subprocess.run(["kubectl", "delete", "trainingjobs", "--all"], capture_output=True)

    def test_job_completes_successfully(self):
        """Submit job → schedule → train → checkpoint → complete."""
        # Submit
        result = subprocess.run(
            ["kubectl", "apply", "-f", "tests/integration/manifests/simple-job.yaml"],
            capture_output=True, text=True
        )
        assert result.returncode == 0

        # Wait for scheduling (phase: Pending → Running)
        self._wait_for_phase("integration-test-job", "Running", timeout=60)

        # Verify workers are placed
        pods = self._get_job_pods("integration-test-job")
        assert len(pods) == 2, f"Expected 2 workers, got {len(pods)}"

        # Verify all workers are on different nodes (anti-affinity)
        nodes = set(p["spec"]["nodeName"] for p in pods)
        assert len(nodes) >= 2 or len(nodes) == len(pods)

        # Wait for checkpoint (should happen within training)
        time.sleep(30)
        job = self._get_job("integration-test-job")
        assert job["status"].get("lastCheckpointStep", 0) > 0

        # Wait for completion
        self._wait_for_phase("integration-test-job", "Completed", timeout=TIMEOUT)

        # Verify final state
        job = self._get_job("integration-test-job")
        assert job["status"]["phase"] == "Completed"
        assert job["status"]["currentStep"] > 0

    def _wait_for_phase(self, job_name: str, phase: str, timeout: int):
        start = time.time()
        while time.time() - start < timeout:
            job = self._get_job(job_name)
            if job and job.get("status", {}).get("phase") == phase:
                return
            time.sleep(5)
        raise TimeoutError(f"Job {job_name} did not reach phase {phase} within {timeout}s")

    def _get_job(self, name: str) -> dict:
        result = subprocess.run(
            ["kubectl", "get", "trainingjob", name, "-o", "json"],
            capture_output=True, text=True
        )
        return json.loads(result.stdout) if result.returncode == 0 else {}

    def _get_job_pods(self, job_name: str) -> list:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-l", f"anvil.io/job-id={job_name}", "-o", "json"],
            capture_output=True, text=True
        )
        return json.loads(result.stdout).get("items", [])
```

### Failure Recovery Test

```python
# tests/integration/test_failure_recovery.py
import pytest
import time
import subprocess
import json

class TestFailureRecovery:
    """Verify checkpoint-based recovery after worker failure."""

    def test_worker_kill_triggers_recovery(self):
        # Submit job with checkpointing enabled
        subprocess.run(
            ["kubectl", "apply", "-f", "tests/integration/manifests/checkpoint-job.yaml"],
            check=True, capture_output=True
        )
        self._wait_for_phase("recovery-test-job", "Running", timeout=60)

        # Wait for at least one checkpoint
        self._wait_for_checkpoint("recovery-test-job", min_step=100, timeout=90)

        # Record the checkpoint step
        job = self._get_job("recovery-test-job")
        checkpoint_step = job["status"]["lastCheckpointStep"]
        assert checkpoint_step > 0

        # Kill a worker
        pods = self._get_job_pods("recovery-test-job")
        victim = pods[1]["metadata"]["name"]
        subprocess.run(
            ["kubectl", "delete", "pod", victim, "--force", "--grace-period=0"],
            check=True, capture_output=True
        )

        # Verify operator detects and recovers
        time.sleep(15)
        new_pods = self._get_job_pods("recovery-test-job")
        new_names = [p["metadata"]["name"] for p in new_pods]
        assert victim not in new_names, "Old pod should be replaced"
        assert len(new_pods) == len(pods), "Worker count should be restored"

        # Verify resumed from checkpoint (not from step 0)
        time.sleep(10)
        for pod in self._get_job_pods("recovery-test-job"):
            logs = subprocess.run(
                ["kubectl", "logs", pod["metadata"]["name"], "--tail=20"],
                capture_output=True, text=True
            ).stdout
            assert f"Resuming from step {checkpoint_step}" in logs or \
                   int(self._extract_step(logs)) >= checkpoint_step
```

### Architecture Decision Record Template

```markdown
# ADR-001: Raft Consensus for Distributed Coordination

## Status
Accepted

## Context
Anvil's training orchestrator needs fault-tolerant coordination between components.
Options considered:
1. Use etcd directly (already in K8s)
2. Implement Raft from scratch
3. Use a library (hashicorp/raft Go binding)

## Decision
Implement simplified Raft from scratch in Python for Week 1 learning,
then rely on K8s/etcd for production coordination in the orchestrator.

## Consequences
- (+) Deep understanding of consensus, directly applicable to interviews
- (+) Ability to reason about etcd behavior from first principles
- (+) Custom implementation teaches failure modes experientially
- (-) Not production-grade; real workloads use etcd through K8s API
- (-) Time investment in reimplementing solved problem

## Alternatives Rejected
- etcd direct: Good for prod, doesn't teach internals
- hashicorp/raft: Go-only, language mismatch with rest of project
```

### Architecture Sequence Diagram

```markdown
# docs/architecture/sequence-diagrams.md

## Job Submission to Completion

​```mermaid
sequenceDiagram
    participant User
    participant API as K8s API Server
    participant Op as Training Operator
    participant Sched as GPU Scheduler
    participant Worker as Worker Pods
    participant Store as Checkpoint Store

    User->>API: kubectl apply trainingjob.yaml
    API->>Op: Watch event (CREATE)
    Op->>Op: Validate spec, set phase=Pending
    Op->>Sched: Request gang scheduling (N workers)
    Sched->>Sched: Score nodes (GPU, topology, VRAM)
    Sched-->>Op: Placement: [node1, node2, ...]
    Op->>API: Create N worker pods
    API->>Worker: Schedule on assigned nodes
    Op->>Op: Set phase=Running

    loop Every checkpoint_interval steps
        Worker->>Store: Save checkpoint (async)
        Store-->>Op: Update lastCheckpointStep
    end

    Worker->>Op: Training complete
    Op->>Op: Set phase=Completed
    Op->>API: Update status
​```

## Failure Recovery Flow

​```mermaid
sequenceDiagram
    participant Op as Training Operator
    participant K8s as K8s API
    participant Store as Checkpoint Store
    participant Worker as New Worker

    Note over Op: Periodic health check (10s interval)
    Op->>K8s: List pods for job
    K8s-->>Op: Worker-2 missing/Failed
    Op->>Op: Set phase=Recovering
    Op->>Store: Get latest checkpoint
    Store-->>Op: checkpoint at step 5000
    Op->>K8s: Delete remaining workers
    Op->>K8s: Create N new workers with checkpoint reference
    K8s->>Worker: Start with --resume-from=/checkpoints/step-5000
    Worker->>Worker: Load checkpoint, resume training
    Op->>Op: Set phase=Running
​```
```

### Benchmark Scripts

```python
# benchmarks/scheduling_latency.py
import time
import subprocess
import json
import statistics

def measure_scheduling_latency(num_trials: int = 10) -> dict:
    """Measure time from job submission to first pod scheduled."""
    latencies = []

    for i in range(num_trials):
        # Submit job
        start = time.time()
        subprocess.run(
            ["kubectl", "apply", "-f", "benchmarks/manifests/bench-job.yaml"],
            check=True, capture_output=True
        )

        # Poll until pod is scheduled
        while time.time() - start < 30:
            result = subprocess.run(
                ["kubectl", "get", "pods", "-l", "anvil.io/job-id=bench-job",
                 "-o", "jsonpath={.items[0].status.conditions[?(@.type=='PodScheduled')].status}"],
                capture_output=True, text=True
            )
            if result.stdout.strip() == "True":
                latency = time.time() - start
                latencies.append(latency)
                break
            time.sleep(0.5)

        # Cleanup
        subprocess.run(["kubectl", "delete", "trainingjob", "bench-job"],
                      capture_output=True)
        time.sleep(5)

    results = {
        "trials": num_trials,
        "mean_ms": statistics.mean(latencies) * 1000,
        "p50_ms": statistics.median(latencies) * 1000,
        "p99_ms": sorted(latencies)[int(0.99 * len(latencies))] * 1000 if latencies else 0,
        "max_ms": max(latencies) * 1000 if latencies else 0,
    }

    print(f"\n{'='*50}")
    print(f"  SCHEDULING LATENCY BENCHMARK")
    print(f"{'='*50}")
    print(f"  Trials:  {results['trials']}")
    print(f"  Mean:    {results['mean_ms']:.0f} ms")
    print(f"  P50:     {results['p50_ms']:.0f} ms")
    print(f"  P99:     {results['p99_ms']:.0f} ms")
    print(f"  Max:     {results['max_ms']:.0f} ms")
    print(f"  Target:  < 500 ms")
    print(f"  Status:  {'PASS' if results['mean_ms'] < 500 else 'FAIL'}")
    print(f"{'='*50}\n")

    return results

if __name__ == "__main__":
    measure_scheduling_latency()
```

### Blog Post Outline

```markdown
# docs/blog/building-training-orchestrator.md

# Building a Training Job Orchestrator: Lessons in Distributed Systems

## Introduction (200 words)
- The problem: orchestrating GPU training at scale
- Why existing tools (Slurm, K8s vanilla) fall short
- What we built: Anvil, a K8s-native training orchestrator

## Architecture Overview (400 words)
- Component diagram
- Design principles: declarative, fault-tolerant, topology-aware
- Why CRDs over custom APIs

## The Hard Problems (600 words)
### Gang Scheduling
- Why all-or-nothing matters for distributed training
- Implementation: atomic reservation with rollback

### Checkpoint-Based Recovery
- The cost of restarting from scratch at scale
- Async checkpointing without blocking training
- Tiered storage for cost efficiency

### Network-Aware Placement
- Why topology dominates training speed
- Modeling bandwidth hierarchy
- Scheduler scoring for locality

## Lessons Learned (400 words)
1. Idempotency is everything in controllers
2. Level-triggered > edge-triggered for reliability
3. Fair-share is harder than it looks (DRF complexity)
4. Test failure modes first, happy path second

## Results & Benchmarks (200 words)
- Scheduling latency: X ms
- Recovery time: Y seconds
- Checkpoint overhead: Z%

## What's Next (100 words)
- Phase B: elastic training, multi-cluster, real GPU workloads
```

### Makefile (Top-Level)

```makefile
# ~/anvil/Makefile
.PHONY: test lint typecheck bench docs clean

test:
	python -m pytest tests/ -v --timeout=300

integration:
	python -m pytest tests/integration/ -v --timeout=300

lint:
	ruff check . --fix
	ruff format .

typecheck:
	mypy . --strict

bench:
	python benchmarks/scheduling_latency.py
	python benchmarks/checkpoint_throughput.py
	python benchmarks/recovery_time.py

docs:
	python docs/generate_diagrams.py

clean:
	kubectl delete trainingjobs --all
	kubectl delete pods -l anvil.io/component --all

status:
	@echo "=== Cluster ==="
	kubectl get nodes
	@echo "\n=== Anvil Components ==="
	kubectl get pods -n anvil-system
	@echo "\n=== Training Jobs ==="
	kubectl get trainingjobs -A
	@echo "\n=== ArgoCD Apps ==="
	argocd app list 2>/dev/null || echo "ArgoCD not configured"
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Integration test times out | Check operator logs: `kubectl logs -n anvil-system deploy/training-operator`. Likely a scheduling failure — verify node resources. |
| Recovery test: new worker doesn't resume | Verify checkpoint path is mounted in new pod spec. Check operator passes `--resume-from` arg. |
| GitOps test: ArgoCD never syncs | Check `argocd app get <app>` — look for "ComparisonError". Often a schema mismatch or RBAC issue. |
| Mypy too many errors | Start with `--strict` on core modules only. Use `# type: ignore` sparingly for external libs. |
| Benchmark results inconsistent | Run more trials (20+). Ensure no other workloads on cluster during bench. Warm up with a throwaway run. |
| Mermaid diagrams not rendering | Use mermaid.live or VS Code Mermaid extension to preview. Check syntax at mermaid-js.github.io. |

---

## Agent Handoff Template

```
Resume Anvil Phase A, Week 7: Phase A Consolidation.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
Project root: ~/anvil/
Cluster: 3-node K3s. All components from Weeks 1-6 deployed.

Current state: [DESCRIBE - e.g., "Integration tests written but recovery test fails — operator doesn't detect pod death"]

What's done:
- [x/blank] E2E integration test (happy path)
- [x/blank] Failure recovery integration test
- [x/blank] GitOps integration test
- [x/blank] Network isolation integration test
- [x/blank] Architecture documentation + diagrams
- [x/blank] ADRs (3-4 written)
- [x/blank] Blog post draft
- [x/blank] Code quality pass (ruff + mypy)
- [x/blank] Performance benchmarks
- [x/blank] Phase B readiness checklist

Next task: [SPECIFIC NEXT STEP]

Key files:
- tests/integration/ — integration test suite
- docs/architecture/ — system documentation
- docs/adrs/ — decision records
- docs/blog/ — blog post draft
- benchmarks/ — performance measurement scripts
- Makefile — top-level commands

All weekly project roots:
- ~/anvil/distributed-systems/ (Week 1)
- ~/anvil/k8s-platform/ (Week 2)
- ~/anvil/training-orchestrator/ (Week 3)
- ~/anvil/infrastructure/ (Week 4)
- ~/anvil/networking/ (Week 5)
- ~/anvil/storage/ (Week 6)
```

---

## Out of Scope

- New feature development (this is consolidation only)
- Phase B implementation (elastic training, real GPU workloads)
- Production deployment to cloud
- CI/CD pipeline setup (GitHub Actions, etc.)
- Load testing at scale (>10 concurrent jobs)
- Security audit
- User-facing web dashboard
- API documentation (OpenAPI/Swagger)
- Video demo or presentation slides
- Comparison benchmarks against Volcano/Kueue (qualitative only in blog)
