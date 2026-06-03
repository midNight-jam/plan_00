# Week 11: Chaos Engineering

## Context

**Where it fits:** Phase B, Week 11 — testing resilience of the secured, SRE-managed platform.
**Prerequisites:** Phase A complete. Weeks 8-10 complete (SRE practices, cost optimization, security hardening).
**What it builds on:** The self-healing from Week 8 and security from Week 10 need to be tested under real failure conditions. Chaos engineering validates that our SLOs survive real-world failures. This week deliberately breaks things to find weaknesses before production does.

Your 3-node K3s cluster is the blast radius. The RTX 5080 GPU provides a real GPU failure scenario (driver unbind). After chaos experiments, you fix the weaknesses — the system should be measurably more resilient by week's end.

---

## Learning Goals

- [ ] Understand chaos engineering principles (steady state hypothesis, minimize blast radius, run in production)
- [ ] Know how to design chaos experiments with clear hypotheses and success criteria
- [ ] Use Chaos Mesh to inject infrastructure failures in Kubernetes
- [ ] Implement circuit breakers and bulkheads for tenant isolation
- [ ] Conduct gameday exercises with documented observations
- [ ] Build a resilience scorecard that quantifies system reliability

---

## Implementation Goals

- [ ] Install and configure Chaos Mesh on K3s cluster
- [ ] Design 6+ failure scenarios covering GPU, node, network, storage, CPU, and memory
- [ ] Execute each scenario with documented hypothesis, observations, and results
- [ ] Conduct 4+ full gameday exercises with expected vs actual behavior comparison
- [ ] Build resilience scorecard measuring recovery time, data loss, and availability impact
- [ ] Implement circuit breakers for inference service (fail fast when backend is down)
- [ ] Implement bulkheads isolating tenant workloads (one tenant's failure doesn't affect others)
- [ ] Fix at least 3 weaknesses discovered during chaos experiments
- [ ] Verify fixes by re-running the experiments that originally failed

---

## Acceptance Criteria

1. Chaos Mesh is installed and can inject PodChaos, NetworkChaos, StressChaos, and IOChaos
2. GPU failure experiment (driver unbind) is detected within 60 seconds and workload migrates to healthy node
3. Node death (VM shutdown) triggers pod rescheduling within 90 seconds
4. Network partition between nodes causes graceful degradation, not cascading failure
5. Storage unavailability triggers checkpoint-based recovery without data loss
6. Circuit breaker opens after 5 consecutive failures and returns fast-fail within 10ms
7. Bulkhead isolation prevents tenant-A's runaway job from consuming tenant-B's GPU allocation
8. Resilience scorecard documents 6+ experiments with pass/fail and recovery metrics
9. At least 3 identified weaknesses are fixed and verified by re-running experiments
10. Gameday report includes timeline, screenshots/logs, and improvement recommendations

---

## Validation Commands

```bash
# Verify Chaos Mesh installation
kubectl get pods -n chaos-testing | grep -c Running

# Run GPU failure experiment
kubectl apply -f chaos/experiments/gpu-failure.yaml && \
  sleep 120 && kubectl get pods -n inference -o wide | grep -v "gpu-node-1"

# Run node death experiment
multipass stop k3s-worker-1 && sleep 100 && \
  kubectl get pods -n inference -o wide | grep Running

# Test circuit breaker behavior
kubectl apply -f chaos/experiments/backend-failure.yaml && \
  sleep 30 && curl -w "%{time_total}" http://inference-gateway:8080/predict | grep "circuit_open"

# Verify bulkhead isolation
kubectl apply -f chaos/experiments/noisy-neighbor.yaml && \
  sleep 60 && kubectl exec -n tenant-b deploy/inference -- curl -s localhost:8080/metrics | \
  grep 'request_latency.*quantile="0.99"' | awk '{print $2 < 0.5 ? "PASS" : "FAIL"}'

# Check resilience scorecard
cat reports/resilience-scorecard.json | jq '.experiments | map(select(.passed == false)) | length'

# Re-run fixed experiments
kubectl apply -f chaos/experiments/regression-suite.yaml && \
  kubectl wait --for=condition=complete job/chaos-regression -n chaos-testing --timeout=600s

# Restore worker node
multipass start k3s-worker-1 && sleep 30 && kubectl get nodes | grep Ready
```

---

## Technical Implementation Details

### Chaos Mesh Installation

```bash
# File: scripts/install-chaos-mesh.sh
#!/bin/bash
set -euo pipefail

helm repo add chaos-mesh https://charts.chaos-mesh.org
helm repo update

helm install chaos-mesh chaos-mesh/chaos-mesh \
  -n chaos-testing --create-namespace \
  --set chaosDaemon.runtime=containerd \
  --set chaosDaemon.socketPath=/run/containerd/containerd.sock \
  --set dashboard.securityMode=false

kubectl wait --for=condition=ready pods -l app.kubernetes.io/instance=chaos-mesh \
  -n chaos-testing --timeout=120s

echo "Chaos Mesh installed. Dashboard: kubectl port-forward -n chaos-testing svc/chaos-dashboard 2333:2333"
```

### GPU Failure Experiment

```yaml
# File: chaos/experiments/gpu-failure.yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: gpu-failure-simulation
  namespace: chaos-testing
spec:
  action: pod-kill
  mode: one
  selector:
    namespaces:
      - inference
    labelSelectors:
      app: model-server
      gpu-node: "true"
  duration: "5m"
---
# Companion: simulate GPU driver unbind on specific node
apiVersion: chaos-mesh.org/v1alpha1
kind: StressChaos
metadata:
  name: gpu-memory-pressure
  namespace: chaos-testing
spec:
  mode: one
  selector:
    namespaces:
      - inference
    labelSelectors:
      app: model-server
  stressors:
    memory:
      workers: 4
      size: "14GB"  # Near the 16GB GPU memory limit
  duration: "3m"
```

### Node Death Experiment

```yaml
# File: chaos/experiments/node-death.yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: PhysicalMachineChaos
metadata:
  name: node-shutdown
  namespace: chaos-testing
spec:
  action: shutdown
  address: "k3s-worker-1.local:31767"
  duration: "5m"
---
# Alternative: use pod-kill on all pods on a specific node
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: node-death-sim
  namespace: chaos-testing
spec:
  action: pod-kill
  mode: all
  selector:
    namespaces:
      - inference
      - training
    fieldSelectors:
      spec.nodeName: k3s-worker-1
  duration: "5m"
```

### Network Partition Experiment

```yaml
# File: chaos/experiments/network-partition.yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: partition-worker-1
  namespace: chaos-testing
spec:
  action: partition
  mode: all
  selector:
    namespaces:
      - inference
    fieldSelectors:
      spec.nodeName: k3s-worker-1
  direction: both
  target:
    selector:
      namespaces:
        - inference
      fieldSelectors:
        spec.nodeName: k3s-worker-2
    mode: all
  duration: "3m"
---
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: high-latency-injection
  namespace: chaos-testing
spec:
  action: delay
  mode: all
  selector:
    namespaces:
      - inference
    labelSelectors:
      app: model-server
  delay:
    latency: "200ms"
    jitter: "50ms"
    correlation: "75"
  duration: "5m"
```

### Circuit Breaker Implementation

```python
# File: src/resilience/circuit_breaker.py
import time
from enum import Enum
from threading import Lock
from functools import wraps

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0,
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self._lock = Lock()

    def call(self, func, *args, **kwargs):
        with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitOpenError(
                        f"Circuit is OPEN. Retry after {self.recovery_timeout}s"
                    )
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            else:
                self.failure_count = 0

    def _on_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN

class CircuitOpenError(Exception):
    pass

# Usage as decorator
def circuit_protected(breaker: CircuitBreaker):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator
```

### Bulkhead Implementation

```yaml
# File: k8s/resilience/bulkhead-quotas.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: tenant-a-quota
  namespace: tenant-a
spec:
  hard:
    requests.nvidia.com/gpu: "1"
    limits.nvidia.com/gpu: "1"
    requests.cpu: "4"
    requests.memory: "8Gi"
    pods: "10"
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: tenant-b-quota
  namespace: tenant-b
spec:
  hard:
    requests.nvidia.com/gpu: "1"
    limits.nvidia.com/gpu: "1"
    requests.cpu: "4"
    requests.memory: "8Gi"
    pods: "10"
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: inference-pdb
  namespace: tenant-a
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: model-server
```

### Resilience Scorecard

```python
# File: src/resilience/scorecard.py
import json
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class ExperimentResult:
    name: str
    hypothesis: str
    chaos_type: str
    duration_seconds: int
    detected_in_seconds: float
    recovered_in_seconds: float
    data_loss: bool
    availability_impact_pct: float
    passed: bool
    notes: str
    fix_applied: str = ""

class ResilienceScorecard:
    def __init__(self):
        self.experiments: list[ExperimentResult] = []

    def add_result(self, result: ExperimentResult):
        self.experiments.append(result)

    def summary(self) -> dict:
        total = len(self.experiments)
        passed = sum(1 for e in self.experiments if e.passed)
        avg_detection = sum(e.detected_in_seconds for e in self.experiments) / total if total else 0
        avg_recovery = sum(e.recovered_in_seconds for e in self.experiments) / total if total else 0
        data_loss_count = sum(1 for e in self.experiments if e.data_loss)
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "total_experiments": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total else 0,
            "avg_detection_time_seconds": avg_detection,
            "avg_recovery_time_seconds": avg_recovery,
            "data_loss_incidents": data_loss_count,
            "resilience_score": (passed / total * 100) if total else 0,
            "experiments": [asdict(e) for e in self.experiments],
        }

    def save(self, path: str = "reports/resilience-scorecard.json"):
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)

# Example usage:
# scorecard = ResilienceScorecard()
# scorecard.add_result(ExperimentResult(
#     name="gpu-failure",
#     hypothesis="Inference survives single GPU failure with <60s detection",
#     chaos_type="PodChaos",
#     duration_seconds=300,
#     detected_in_seconds=45,
#     recovered_in_seconds=90,
#     data_loss=False,
#     availability_impact_pct=2.1,
#     passed=True,
#     notes="Pod rescheduled to worker-2 successfully"
# ))
```

### Gameday Exercise Template

```markdown
# File: docs/gameday/template.md
## Gameday Exercise: [SCENARIO NAME]
**Date:** YYYY-MM-DD | **Duration:** X minutes | **Participants:** [names]

### Hypothesis
"When [failure condition], we expect [system behavior] within [time bound]."

### Pre-conditions
- [ ] All nodes healthy: `kubectl get nodes`
- [ ] All critical pods running: `kubectl get pods -A | grep -v Running`
- [ ] Monitoring dashboards open
- [ ] Incident channel ready

### Execution Steps
1. [Step 1 - inject chaos]
2. [Step 2 - observe]
3. [Step 3 - validate recovery]
4. [Step 4 - cleanup]

### Expected Behavior
| Metric | Expected | Actual | Pass? |
|--------|----------|--------|-------|
| Detection time | <60s | | |
| Recovery time | <120s | | |
| Data loss | None | | |
| Error rate spike | <5% | | |

### Observations
[What actually happened, screenshots, log snippets]

### Surprises
[Anything unexpected — both good and bad]

### Action Items
| Action | Owner | Priority |
|--------|-------|----------|
| [fix] | [name] | P[1-3] |
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Chaos Mesh DaemonSet fails to start | Check container runtime socket path; K3s uses containerd at `/run/k3s/containerd/containerd.sock` |
| PodChaos has no effect | Verify Chaos Mesh has RBAC to the target namespace: `kubectl get clusterrole chaos-mesh-manager -o yaml` |
| Network partition doesn't work | K3s with Flannel may not support all NetworkChaos modes; use pod-level injection instead |
| GPU driver unbind crashes the node | Use pod-kill to simulate GPU failure instead of actual driver manipulation |
| Circuit breaker too aggressive | Tune failure_threshold and recovery_timeout based on observed false-positive rate |
| Multipass VM won't restart after stop | `multipass recover k3s-worker-1` then `multipass start k3s-worker-1` |

---

## Agent Handoff Template

```
Resume Week 11: Chaos Engineering for AI Infrastructure.

Environment: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
K3s cluster: 3 multipass nodes. Phase A + Weeks 8-10 complete.

Current state: [describe what's done and what's next]

Tasks remaining:
- [ ] [list incomplete items from Implementation Goals]

Key files:
- scripts/install-chaos-mesh.sh
- chaos/experiments/gpu-failure.yaml
- chaos/experiments/node-death.yaml
- chaos/experiments/network-partition.yaml
- src/resilience/circuit_breaker.py
- k8s/resilience/bulkhead-quotas.yaml
- src/resilience/scorecard.py
- docs/gameday/template.md

IMPORTANT: K3s containerd socket is at /run/k3s/containerd/containerd.sock (not default path).
After node-death experiments, restore with: multipass start k3s-worker-1
Validate with the validation commands in the spec.
```

---

## Out of Scope

- Chaos in production (this is a dev/staging cluster)
- Automated continuous chaos (run experiments manually first)
- Chaos for security (e.g., privilege escalation testing — covered in Week 10)
- Multi-cluster chaos (covered in Week 12)
- Application-level fault injection (HTTP error injection) — focus is infrastructure
- LitmusChaos or other chaos tools (standardize on Chaos Mesh)
- Formal verification or TLA+ modeling
