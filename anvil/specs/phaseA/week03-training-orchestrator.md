# Week 3: Training Job Orchestrator

## Context

**Where it fits:** Phase A, Week 3 — the core product of Anvil. This is what users interact with: submit a training job, and the orchestrator handles scheduling, execution, checkpointing, and recovery.

**Prerequisites:**
- Week 1 completed (consensus, fault tolerance patterns)
- Week 2 completed (K3s cluster running, CRD concepts, scheduler extender)
- Python `kopf` framework basics (Kubernetes operator framework)
- Hardware: ASUS ROG Strix SCAR 16 (RTX 5080 16GB, 32GB RAM, 2TB SSD, Ubuntu)

**What it builds on:** Week 1's fault tolerance informs checkpoint/recovery design. Week 2's scheduler extender is invoked by this orchestrator for placement. Week 4's GitOps deploys this operator. Week 6's storage backs checkpoints.

---

## Learning Goals

- [ ] Explain the Kubernetes Operator pattern and why CRDs + controllers are powerful
- [ ] Describe gang scheduling and why it's critical for distributed training (all-or-nothing)
- [ ] Articulate checkpoint strategies: synchronous vs asynchronous, frequency tradeoffs
- [ ] Explain fair-share scheduling: DRF (Dominant Resource Fairness) algorithm
- [ ] Describe preemption mechanics: graceful termination, checkpoint-before-kill
- [ ] Understand job lifecycle state machines and idempotent reconciliation
- [ ] Compare with Volcano, Kueue, and Ray for training orchestration

---

## Implementation Goals

- [ ] Define TrainingJob CRD with full spec (model, dataset, resources, checkpoint policy)
- [ ] Implement kopf-based operator with reconciliation loop
- [ ] Job lifecycle: Pending → Scheduling → Running → Checkpointing → Completed/Failed
- [ ] Gang scheduling: atomic placement of all workers or rollback
- [ ] Periodic checkpointing with configurable interval and async write
- [ ] Checkpoint garbage collection (keep last N checkpoints)
- [ ] Failure detection: pod failure → find latest checkpoint → recreate workers → resume
- [ ] Fair-share queue: multiple tenants with guaranteed resource shares
- [ ] Priority levels with preemption (high-priority job evicts low-priority)
- [ ] Metrics emission (Prometheus): queue depth, job duration, GPU utilization

---

## Acceptance Criteria

1. `kubectl apply -f training-job.yaml` creates a TrainingJob CR and the operator picks it up within 5 seconds.
2. A 4-worker training job is gang-scheduled: all 4 pods are created atomically, or none are (verified by partially-satisfiable resource scenario).
3. Job status transitions are recorded: `kubectl get trainingjob my-job -o jsonpath='{.status.phase}'` shows correct phase.
4. Checkpoints are saved every N steps (configurable) to shared storage without blocking training (async).
5. Old checkpoints are garbage-collected, keeping only the last 3 (configurable).
6. When a worker pod is killed, the operator detects failure within 10 seconds and initiates recovery from the latest checkpoint.
7. After recovery, the job resumes from the checkpointed step (not from scratch) — verified by log output showing resumed step number.
8. Fair-share scheduling allocates GPUs proportionally: team with 60% share gets 3 of 5 available GPUs.
9. A priority-2 job preempts a priority-1 job: low-priority job is checkpointed and suspended, high-priority starts.
10. Prometheus metrics endpoint exposes `anvil_training_jobs_total`, `anvil_queue_depth`, `anvil_job_duration_seconds`.

---

## Validation Commands

```bash
# Deploy the operator
cd ~/anvil/training-orchestrator
kubectl apply -f deploy/crds/
kubectl apply -f deploy/operator/
kubectl logs -n anvil-system deploy/training-operator -f &

# Submit a training job
kubectl apply -f examples/simple-training-job.yaml
kubectl get trainingjobs -w

# Verify gang scheduling (request more workers than available, should stay Pending)
kubectl apply -f tests/manifests/gang-overcommit.yaml
sleep 10
kubectl get trainingjob gang-test -o jsonpath='{.status.phase}'  # Should be "Pending"

# Test checkpointing
kubectl apply -f examples/checkpoint-job.yaml
sleep 60
kubectl exec checkpoint-job-worker-0 -- ls /checkpoints/  # Should show checkpoint files

# Test failure recovery
kubectl delete pod checkpoint-job-worker-1 --force
sleep 15
kubectl get pods | grep checkpoint-job  # New worker-1 should be Running
kubectl logs checkpoint-job-worker-1 | grep "Resuming from step"

# Test fair-share
kubectl apply -f tests/manifests/fairshare-scenario.yaml
python tests/verify_fairshare.py

# Test preemption
kubectl apply -f tests/manifests/preemption-scenario.yaml
sleep 10
kubectl get trainingjobs  # Low-priority should be Suspended

# Metrics
curl http://localhost:9090/metrics | grep anvil_
```

---

## Technical Implementation Details

### Project Structure

```
~/anvil/training-orchestrator/
├── operator/
│   ├── __init__.py
│   ├── main.py              # kopf handlers entry point
│   ├── handlers.py          # CRD event handlers
│   ├── reconciler.py        # Core reconciliation logic
│   ├── scheduler.py         # Gang scheduling + fair-share
│   ├── checkpoint.py        # Checkpoint management
│   ├── recovery.py          # Failure detection + recovery
│   ├── preemption.py        # Priority-based preemption
│   ├── metrics.py           # Prometheus metrics
│   └── models.py            # Pydantic models for CRD spec
├── deploy/
│   ├── crds/
│   │   └── trainingjob-crd.yaml
│   ├── operator/
│   │   ├── deployment.yaml
│   │   ├── rbac.yaml
│   │   └── service.yaml
│   └── scheduler-config.yaml
├── examples/
│   ├── simple-training-job.yaml
│   └── checkpoint-job.yaml
├── tests/
│   ├── manifests/
│   ├── test_reconciler.py
│   ├── test_scheduler.py
│   ├── test_checkpoint.py
│   └── test_recovery.py
├── Dockerfile
└── pyproject.toml
```

### TrainingJob CRD Definition

```yaml
# deploy/crds/trainingjob-crd.yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: trainingjobs.anvil.io
spec:
  group: anvil.io
  versions:
    - name: v1alpha1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                model:
                  type: object
                  properties:
                    name: {type: string}
                    framework: {type: string, enum: [pytorch, jax]}
                    image: {type: string}
                dataset:
                  type: object
                  properties:
                    path: {type: string}
                    version: {type: string}
                workers:
                  type: object
                  properties:
                    count: {type: integer, minimum: 1}
                    gpusPerWorker: {type: integer, minimum: 1}
                    vramGb: {type: integer}
                checkpoint:
                  type: object
                  properties:
                    enabled: {type: boolean}
                    intervalSteps: {type: integer}
                    maxKeep: {type: integer, default: 3}
                    storagePath: {type: string}
                    async: {type: boolean, default: true}
                scheduling:
                  type: object
                  properties:
                    priority: {type: integer, minimum: 0, maximum: 10}
                    team: {type: string}
                    queue: {type: string, default: default}
            status:
              type: object
              properties:
                phase: {type: string, enum: [Pending, Scheduling, Running, Checkpointing, Completed, Failed, Suspended]}
                currentStep: {type: integer}
                lastCheckpointStep: {type: integer}
                startTime: {type: string}
                workers: {type: array, items: {type: object}}
      subresources:
        status: {}
  scope: Namespaced
  names:
    plural: trainingjobs
    singular: trainingjob
    kind: TrainingJob
    shortNames: [tj]
```

### Operator Core Logic (kopf)

```python
# operator/main.py
import kopf
import kubernetes
from reconciler import TrainingJobReconciler
from scheduler import GangScheduler, FairShareQueue
from checkpoint import CheckpointManager
from recovery import RecoveryManager

reconciler = TrainingJobReconciler()
scheduler = GangScheduler()
queue = FairShareQueue()

@kopf.on.create("anvil.io", "v1alpha1", "trainingjobs")
async def on_create(spec, name, namespace, status, patch, **kwargs):
    """Handle new TrainingJob submission."""
    patch.status["phase"] = "Pending"
    queue.enqueue(name, namespace, spec)
    await reconciler.reconcile(name, namespace, spec, patch)

@kopf.on.update("anvil.io", "v1alpha1", "trainingjobs")
async def on_update(spec, name, namespace, status, patch, **kwargs):
    await reconciler.reconcile(name, namespace, spec, patch)

@kopf.on.timer("anvil.io", "v1alpha1", "trainingjobs", interval=10.0)
async def periodic_reconcile(spec, name, namespace, status, patch, **kwargs):
    """Periodic reconciliation for failure detection."""
    if status.get("phase") == "Running":
        await reconciler.check_health(name, namespace, spec, patch)

@kopf.on.field("anvil.io", "v1alpha1", "trainingjobs", field="status.phase")
async def on_phase_change(old, new, name, namespace, **kwargs):
    """Log and emit metrics on phase transitions."""
    print(f"TrainingJob {namespace}/{name}: {old} → {new}")
```

### Gang Scheduler

```python
# operator/scheduler.py
from dataclasses import dataclass
from kubernetes import client

@dataclass
class SchedulingRequest:
    job_name: str
    namespace: str
    worker_count: int
    gpus_per_worker: int
    vram_gb: int
    priority: int
    team: str

class GangScheduler:
    def __init__(self):
        self.v1 = client.CoreV1Api()

    async def try_schedule(self, request: SchedulingRequest) -> list[str] | None:
        """Attempt atomic scheduling. Returns node assignments or None."""
        nodes = self._get_available_nodes()
        assignments = self._find_placement(nodes, request)

        if assignments is None:
            return None  # Cannot satisfy — all or nothing

        # Reserve resources atomically (optimistic locking via resource version)
        if not await self._reserve_all(assignments, request):
            return None  # Conflict — retry next reconciliation

        return assignments

    def _find_placement(self, nodes: list, request: SchedulingRequest) -> list[str] | None:
        """Find placement for all workers. Returns list of node names or None."""
        assignments = []
        available = {n.metadata.name: self._free_gpus(n) for n in nodes}

        for _ in range(request.worker_count):
            best_node = None
            best_score = -1
            for node_name, free in available.items():
                if free >= request.gpus_per_worker:
                    score = self._score(node_name, request)
                    if score > best_score:
                        best_score = score
                        best_node = node_name
            if best_node is None:
                return None  # Can't place all workers
            assignments.append(best_node)
            available[best_node] -= request.gpus_per_worker

        return assignments


class FairShareQueue:
    """Multi-tenant queue with Dominant Resource Fairness."""

    def __init__(self):
        self.team_shares: dict[str, float] = {}  # team -> guaranteed share (0-1)
        self.team_usage: dict[str, float] = {}   # team -> current usage
        self.pending: list[SchedulingRequest] = []

    def set_shares(self, shares: dict[str, float]):
        self.team_shares = shares

    def next_job(self) -> SchedulingRequest | None:
        """Return the next job to schedule based on fair-share."""
        if not self.pending:
            return None
        # Sort by (current_usage / share) ascending — most underserved first
        self.pending.sort(key=lambda r: self._fairness_score(r.team))
        return self.pending[0]

    def _fairness_score(self, team: str) -> float:
        share = self.team_shares.get(team, 0.1)
        usage = self.team_usage.get(team, 0.0)
        return usage / share if share > 0 else float("inf")
```

### Checkpoint Manager

```python
# operator/checkpoint.py
import asyncio
from datetime import datetime
from pathlib import Path

class CheckpointManager:
    def __init__(self, storage_backend):
        self.storage = storage_backend

    async def save_checkpoint(self, job_name: str, step: int, data_path: str, async_write: bool = True):
        """Save checkpoint, optionally non-blocking."""
        checkpoint_id = f"{job_name}/step-{step}-{datetime.utcnow().isoformat()}"

        if async_write:
            asyncio.create_task(self._async_save(checkpoint_id, data_path))
        else:
            await self._async_save(checkpoint_id, data_path)

        return checkpoint_id

    async def _async_save(self, checkpoint_id: str, data_path: str):
        await self.storage.upload(data_path, f"checkpoints/{checkpoint_id}")

    async def garbage_collect(self, job_name: str, max_keep: int = 3):
        """Remove old checkpoints, keeping only the most recent max_keep."""
        checkpoints = await self.storage.list(f"checkpoints/{job_name}/")
        checkpoints.sort(key=lambda c: c.created_at, reverse=True)

        for old in checkpoints[max_keep:]:
            await self.storage.delete(old.path)

    async def get_latest(self, job_name: str) -> str | None:
        """Get path to most recent checkpoint for recovery."""
        checkpoints = await self.storage.list(f"checkpoints/{job_name}/")
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda c: c.created_at, reverse=True)
        return checkpoints[0].path
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| kopf operator doesn't start | Check RBAC: operator needs get/list/watch/patch on trainingjobs and pods. Verify `ClusterRole` applied. |
| CRD not appearing | Run `kubectl get crd trainingjobs.anvil.io`. If missing, check YAML syntax. `kubectl apply --dry-run=server -f crd.yaml`. |
| Gang scheduling always fails | Verify node resources: `kubectl describe nodes`. Reduce worker count or GPU request for testing. |
| Checkpoint not writing | Check shared volume mount exists in worker pods. Test storage path manually: `kubectl exec worker -- ls /checkpoints/`. |
| Recovery creates duplicate workers | Ensure reconciler is idempotent. Use `metadata.ownerReferences` so old pods are garbage-collected. |
| Fair-share not balanced | Print debug: log team usage/share ratios each scheduling cycle. Check share config is loaded. |

---

## Agent Handoff Template

```
Resume Anvil Phase A, Week 3: Training Job Orchestrator.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
Project root: ~/anvil/training-orchestrator/
Cluster: 3-node K3s (from Week 2). Kubeconfig: ~/.kube/anvil-config

Current state: [DESCRIBE - e.g., "CRD deployed, operator runs but gang scheduling doesn't work"]

What's done:
- [x/blank] TrainingJob CRD defined and applied
- [x/blank] kopf operator running in cluster
- [x/blank] Job lifecycle state transitions
- [x/blank] Gang scheduling (all-or-nothing)
- [x/blank] Periodic checkpointing
- [x/blank] Checkpoint garbage collection
- [x/blank] Failure recovery from checkpoint
- [x/blank] Fair-share multi-tenant queue
- [x/blank] Priority preemption
- [x/blank] Prometheus metrics

Next task: [SPECIFIC NEXT STEP]

Key files:
- operator/main.py — kopf event handlers
- operator/scheduler.py — gang scheduling + fair-share
- operator/checkpoint.py — checkpoint lifecycle
- operator/recovery.py — failure detection + restart
- deploy/crds/trainingjob-crd.yaml — CRD schema

Dependencies: kopf, kubernetes, pydantic, prometheus-client.
```

---

## Out of Scope

- Actual model training (we simulate with sleep + step counter)
- Distributed training frameworks (PyTorch DDP, DeepSpeed) — Phase B
- Elastic training (scaling workers up/down mid-job) — Phase B
- Multi-cluster scheduling
- Spot instance interruption handling — covered in Week 4
- Network-aware placement — covered in Week 5
- Persistent checkpoint storage backend — covered in Week 6
- Web UI for job monitoring
