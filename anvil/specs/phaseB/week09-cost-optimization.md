# Week 9: Cost Optimization

## Context

**Where it fits:** Phase B, Week 9 — building on SRE practices from Week 8 to add financial observability.
**Prerequisites:** Phase A complete (K3s cluster, GPU operator, monitoring). Week 8 SLO framework in place.
**What it builds on:** The Prometheus metrics and Grafana dashboards from Phase A and Week 8 now get extended with cost attribution. GPU scheduling from Phase A gets optimized with sharing and right-sizing.

Your RTX 5080 16GB simulates expensive GPU infrastructure. Every optimization technique here translates directly to cloud GPU costs ($2-8/hr per GPU). The goal is demonstrating 30%+ cost reduction while maintaining SLOs.

---

## Learning Goals

- [ ] Understand GPU utilization patterns and why most AI workloads waste 40-60% of GPU capacity
- [ ] Know how NVIDIA MPS and time-slicing work and when each is appropriate
- [ ] Implement cost attribution models that map resource usage to teams/projects
- [ ] Design spot/preemptible instance strategies with graceful preemption handling
- [ ] Build right-sizing recommendations from historical usage data
- [ ] Create chargeback dashboards that drive behavioral change

---

## Implementation Goals

- [ ] Deploy GPU utilization tracker exporting per-pod, per-namespace GPU metrics every 30s
- [ ] Implement per-job cost calculator (GPU-hours × rate = cost per job)
- [ ] Configure NVIDIA MPS for inference workloads sharing a single GPU
- [ ] Configure time-slicing for batch workloads and benchmark vs MPS
- [ ] Build spot instance preemption handler with 2-minute checkpoint-and-migrate
- [ ] Create right-sizing recommendation engine analyzing 7-day usage history
- [ ] Deploy per-team cost dashboard with daily/weekly/monthly aggregations
- [ ] Implement idle GPU detection with auto-scaledown after 15 minutes

---

## Acceptance Criteria

1. GPU utilization is tracked per-pod at 30-second granularity and stored in Prometheus for 30 days
2. Per-job cost is calculated within 5% accuracy of actual GPU-time consumed
3. MPS configuration allows 3+ inference pods to share one GPU with <10% latency overhead
4. Time-slicing allows 2+ training jobs to share one GPU with measured throughput trade-off documented
5. Preemption handler checkpoints a running training job within 90 seconds of receiving termination signal
6. Checkpointed job resumes from checkpoint on a different node within 2 minutes
7. Right-sizing engine identifies at least 2 workloads that are over-provisioned by >30%
8. Idle GPU detection correctly identifies GPUs with <5% utilization for 15+ minutes
9. Cost dashboard shows per-team breakdown and month-over-month trend
10. End-to-end demo shows a measurable cost reduction (documented before/after comparison)

---

## Validation Commands

```bash
# Verify GPU metrics are being collected per-pod
kubectl exec -n monitoring prometheus-0 -- promtool query instant \
  'DCGM_FI_DEV_GPU_UTIL{pod!=""}'

# Check MPS is active on GPU node
kubectl exec -n gpu-operator -l app=nvidia-mps -- nvidia-smi | grep MPS

# Run benchmark: MPS shared inference vs dedicated
kubectl apply -f tests/benchmark-mps-shared.yaml && \
  kubectl wait --for=condition=complete job/mps-benchmark -n cost --timeout=300s && \
  kubectl logs job/mps-benchmark -n cost | grep "latency_overhead"

# Test preemption handler
kubectl exec -n training deploy/training-job -- kill -SIGTERM 1 && \
  sleep 95 && kubectl exec -n training deploy/training-job -- \
  ls /checkpoints/ | grep "preempt-checkpoint"

# Verify idle detection
kubectl apply -f tests/idle-gpu-test.yaml && sleep 900 && \
  kubectl get events -n cost | grep "IdleGPUDetected"

# Check cost attribution
curl -s http://cost-service.cost:8080/api/v1/costs/team/ml-team/weekly | jq '.total_gpu_hours'

# Verify right-sizing recommendations
kubectl exec -n cost deploy/rightsizer -- cat /reports/recommendations-latest.json | \
  jq '.recommendations | length'
```

---

## Technical Implementation Details

### GPU Utilization Tracker

```yaml
# File: k8s/cost/gpu-metrics-exporter.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: gpu-cost-exporter
  namespace: cost
spec:
  selector:
    matchLabels:
      app: gpu-cost-exporter
  template:
    metadata:
      labels:
        app: gpu-cost-exporter
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9400"
    spec:
      nodeSelector:
        nvidia.com/gpu.present: "true"
      containers:
        - name: exporter
          image: anvil/gpu-cost-exporter:latest
          ports:
            - containerPort: 9400
          env:
            - name: COLLECTION_INTERVAL
              value: "30"
            - name: NODE_NAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName
          volumeMounts:
            - name: pod-resources
              mountPath: /var/lib/kubelet/pod-resources
              readOnly: true
      volumes:
        - name: pod-resources
          hostPath:
            path: /var/lib/kubelet/pod-resources
```

### Cost Calculator Service

```python
# File: src/cost/calculator.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from prometheus_api_client import PrometheusConnect

@dataclass
class GPURate:
    gpu_type: str
    cost_per_hour: float  # USD

GPU_RATES = {
    "RTX_5080": GPURate("RTX_5080", 1.20),
    "A100_40GB": GPURate("A100_40GB", 3.50),
    "H100_80GB": GPURate("H100_80GB", 8.00),
}

@dataclass
class JobCost:
    job_name: str
    namespace: str
    team: str
    gpu_type: str
    gpu_hours: float
    cost_usd: float
    avg_utilization: float
    waste_usd: float  # cost of idle GPU time

class CostCalculator:
    def __init__(self, prom_url: str, gpu_type: str = "RTX_5080"):
        self.prom = PrometheusConnect(url=prom_url)
        self.rate = GPU_RATES[gpu_type]

    def calculate_job_cost(self, job_name: str, namespace: str) -> JobCost:
        duration_hours = self._get_job_duration_hours(job_name, namespace)
        avg_util = self._get_avg_utilization(job_name, namespace)
        total_cost = duration_hours * self.rate.cost_per_hour
        effective_cost = total_cost * avg_util
        waste = total_cost - effective_cost

        team = self._get_team_label(namespace)
        return JobCost(
            job_name=job_name,
            namespace=namespace,
            team=team,
            gpu_type=self.rate.gpu_type,
            gpu_hours=duration_hours,
            cost_usd=total_cost,
            avg_utilization=avg_util,
            waste_usd=waste,
        )

    def get_team_costs(self, team: str, days: int = 7) -> dict:
        query = f'sum(rate(gpu_usage_seconds_total{{team="{team}"}}[{days}d])) / 3600'
        result = self.prom.custom_query(query)
        gpu_hours = float(result[0]["value"][1]) if result else 0
        return {
            "team": team,
            "period_days": days,
            "total_gpu_hours": gpu_hours,
            "total_cost_usd": gpu_hours * self.rate.cost_per_hour,
        }

    def _get_job_duration_hours(self, job: str, ns: str) -> float:
        query = f'(time() - kube_job_status_start_time{{job_name="{job}",namespace="{ns}"}}) / 3600'
        result = self.prom.custom_query(query)
        return float(result[0]["value"][1]) if result else 0

    def _get_avg_utilization(self, job: str, ns: str) -> float:
        query = f'avg_over_time(DCGM_FI_DEV_GPU_UTIL{{pod=~"{job}.*",namespace="{ns}"}}[1h]) / 100'
        result = self.prom.custom_query(query)
        return float(result[0]["value"][1]) if result else 0

    def _get_team_label(self, namespace: str) -> str:
        ns_team_map = {"ml-team": "ml-team", "data-eng": "data-eng", "research": "research"}
        return ns_team_map.get(namespace, "unknown")
```

### NVIDIA MPS Configuration

```yaml
# File: k8s/cost/mps-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mps-config
  namespace: gpu-operator
data:
  mps-entrypoint.sh: |
    #!/bin/bash
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
    mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
    # Start MPS daemon
    nvidia-cuda-mps-control -d
    # Set active thread percentage per client (33% for 3 clients)
    echo "set_active_thread_percentage 33" | nvidia-cuda-mps-control
    echo "MPS daemon started with 33% thread allocation per client"
    # Keep running
    wait
---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-mps-server
  namespace: gpu-operator
spec:
  selector:
    matchLabels:
      app: nvidia-mps
  template:
    metadata:
      labels:
        app: nvidia-mps
    spec:
      nodeSelector:
        nvidia.com/gpu.present: "true"
      containers:
        - name: mps
          image: nvidia/cuda:12.4-base-ubuntu22.04
          command: ["/bin/bash", "/config/mps-entrypoint.sh"]
          securityContext:
            privileged: true
          volumeMounts:
            - name: config
              mountPath: /config
            - name: mps-pipe
              mountPath: /tmp/nvidia-mps
          resources:
            limits:
              nvidia.com/gpu: "1"
      volumes:
        - name: config
          configMap:
            name: mps-config
        - name: mps-pipe
          hostPath:
            path: /tmp/nvidia-mps
```

### Preemption Handler

```python
# File: src/cost/preemption_handler.py
import signal
import sys
import time
import torch
import os
from pathlib import Path

class PreemptionHandler:
    def __init__(self, checkpoint_dir: str = "/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.should_checkpoint = False
        signal.signal(signal.SIGTERM, self._handle_sigterm)

    def _handle_sigterm(self, signum, frame):
        print(f"[PREEMPTION] Received SIGTERM at {time.time():.0f}, initiating checkpoint...")
        self.should_checkpoint = True

    def checkpoint_if_needed(self, model, optimizer, epoch: int, step: int, loss: float) -> bool:
        if not self.should_checkpoint:
            return False
        start = time.time()
        checkpoint_path = self.checkpoint_dir / f"preempt-checkpoint-{int(time.time())}.pt"
        torch.save({
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "timestamp": time.time(),
            "preempted": True,
        }, checkpoint_path)
        elapsed = time.time() - start
        print(f"[PREEMPTION] Checkpoint saved to {checkpoint_path} in {elapsed:.1f}s")
        self._notify_scheduler(checkpoint_path)
        sys.exit(0)

    def resume_from_checkpoint(self, model, optimizer) -> dict:
        checkpoints = sorted(self.checkpoint_dir.glob("preempt-checkpoint-*.pt"))
        if not checkpoints:
            return None
        latest = checkpoints[-1]
        print(f"[RESUME] Loading checkpoint from {latest}")
        checkpoint = torch.load(latest, map_location="cuda")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    def _notify_scheduler(self, checkpoint_path: Path):
        os.environ["LAST_CHECKPOINT"] = str(checkpoint_path)
```

### Right-Sizing Engine

```python
# File: src/cost/rightsizer.py
from dataclasses import dataclass
from prometheus_api_client import PrometheusConnect

@dataclass
class Recommendation:
    workload: str
    namespace: str
    current_gpu_request: str
    recommended: str
    reason: str
    potential_savings_pct: float

class RightSizer:
    def __init__(self, prom_url: str, lookback_days: int = 7):
        self.prom = PrometheusConnect(url=prom_url)
        self.lookback = f"{lookback_days}d"

    def analyze(self) -> list[Recommendation]:
        recommendations = []
        workloads = self._get_gpu_workloads()
        for w in workloads:
            p95_util = self._get_p95_utilization(w["pod_prefix"], w["namespace"])
            p95_memory = self._get_p95_gpu_memory(w["pod_prefix"], w["namespace"])
            if p95_util < 0.30:
                recommendations.append(Recommendation(
                    workload=w["pod_prefix"],
                    namespace=w["namespace"],
                    current_gpu_request="1 GPU",
                    recommended="GPU time-slice (50%)" if p95_util > 0.15 else "CPU-only or shared MPS",
                    reason=f"P95 GPU utilization is only {p95_util*100:.0f}% over {self.lookback}",
                    potential_savings_pct=(1 - p95_util) * 100,
                ))
            if p95_memory < 0.40:
                recommendations.append(Recommendation(
                    workload=w["pod_prefix"],
                    namespace=w["namespace"],
                    current_gpu_request="16GB GPU",
                    recommended="8GB GPU or shared",
                    reason=f"P95 GPU memory usage is only {p95_memory*100:.0f}%",
                    potential_savings_pct=(1 - p95_memory) * 50,
                ))
        return recommendations

    def _get_gpu_workloads(self) -> list[dict]:
        query = 'count by (pod, namespace) (DCGM_FI_DEV_GPU_UTIL)'
        results = self.prom.custom_query(query)
        return [{"pod_prefix": r["metric"]["pod"], "namespace": r["metric"]["namespace"]}
                for r in results]

    def _get_p95_utilization(self, pod: str, ns: str) -> float:
        query = f'quantile_over_time(0.95, DCGM_FI_DEV_GPU_UTIL{{pod=~"{pod}.*",namespace="{ns}"}}[{self.lookback}]) / 100'
        result = self.prom.custom_query(query)
        return float(result[0]["value"][1]) if result else 0

    def _get_p95_gpu_memory(self, pod: str, ns: str) -> float:
        query = f'quantile_over_time(0.95, DCGM_FI_DEV_MEM_COPY_UTIL{{pod=~"{pod}.*",namespace="{ns}"}}[{self.lookback}]) / 100'
        result = self.prom.custom_query(query)
        return float(result[0]["value"][1]) if result else 0
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| DCGM metrics not showing per-pod | Ensure DCGM exporter has pod-resources socket mounted; check `nvidia-smi` on node |
| MPS daemon fails to start | Check GPU compute mode: `nvidia-smi -i 0 -c EXCLUSIVE_PROCESS` must NOT be set |
| Time-slicing not working | Requires GPU operator config: `kubectl edit clusterpolicy gpu-cluster-policy` set `timeSlicing.enabled: true` |
| Preemption handler misses SIGTERM | K8s sends SIGTERM to PID 1; if using shell wrapper, trap and forward the signal |
| Cost calculation seems wrong | Verify job start time with `kubectl get job -o jsonpath='{.status.startTime}'` |
| Right-sizer returns empty | Need 7 days of data; create synthetic workloads or reduce lookback window for testing |

---

## Agent Handoff Template

```
Resume Week 9: Cost Optimization for AI Infrastructure.

Environment: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
K3s cluster: 3 multipass nodes. Phase A + Week 8 SRE practices complete.

Current state: [describe what's done and what's next]

Tasks remaining:
- [ ] [list incomplete items from Implementation Goals]

Key files:
- k8s/cost/gpu-metrics-exporter.yaml
- src/cost/calculator.py
- k8s/cost/mps-config.yaml
- src/cost/preemption_handler.py
- src/cost/rightsizer.py

GPU rate for RTX 5080: $1.20/hr equivalent.
MPS target: 3 inference pods per GPU with <10% latency overhead.
Preemption checkpoint must complete within 90 seconds.
Validate with the validation commands in the spec.
```

---

## Out of Scope

- Cloud provider billing API integration (AWS/GCP cost explorer)
- Multi-GPU (NVLink) cost modeling
- Network egress cost tracking
- Storage tiering cost optimization
- License cost tracking (CUDA, cuDNN)
- FinOps team organizational processes
- Commitment/reservation discount strategies (reserved instances)
