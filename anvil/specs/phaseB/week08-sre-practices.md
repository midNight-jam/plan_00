# Week 8: SRE Practices for AI Systems

## Context

**Where it fits:** Phase B, Week 8 — the first week of the reliability and operations arc.
**Prerequisites:** Phase A complete (K3s multi-node cluster, GPU operator, monitoring with Prometheus/Grafana, model serving with KServe/Triton).
**What it builds on:** The monitoring stack from Phase A now gets formalized into SLIs/SLOs with error budgets. The alerting moves from ad-hoc thresholds to structured incident management with automated remediation.

Your 3-node K3s cluster (multipass VMs on ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM) is the production-like environment. Treat it as a real production system — every SLO decision should reflect real trade-offs.

---

## Learning Goals

- [ ] Understand the difference between SLIs, SLOs, and SLAs in AI/ML context
- [ ] Know how to calculate error budgets and make release decisions based on them
- [ ] Design meaningful SLIs for inference, training, and GPU workloads
- [ ] Implement automated incident detection and escalation
- [ ] Build self-healing systems that reduce toil
- [ ] Create capacity planning models that predict resource exhaustion

---

## Implementation Goals

- [ ] Define 4+ SLIs covering inference latency, availability, training success, and GPU utilization
- [ ] Implement error budget tracking with Prometheus recording rules and Grafana dashboard
- [ ] Build incident management system with severity levels (P1-P4), auto-detection, and notification
- [ ] Create post-incident review template and conduct at least one simulated incident review
- [ ] Implement capacity planning that forecasts GPU demand 7 days ahead
- [ ] Deploy 3+ self-healing automations for known failure modes
- [ ] Write runbooks for the top 5 operational scenarios

---

## Acceptance Criteria

1. SLI metrics are exported to Prometheus and correctly calculate availability over rolling 30-day windows
2. Error budget dashboard shows remaining budget as percentage, burn rate, and projected exhaustion date
3. When error budget drops below 25%, an alert fires and a "deployment freeze" annotation appears
4. Incident auto-detection triggers within 60 seconds of SLO violation
5. Notification system sends alerts to at least 2 channels (webhook + log) with severity-appropriate routing
6. Runbooks are linked from alerts and contain step-by-step remediation
7. Self-healing restarts crashed inference pods within 30 seconds without human intervention
8. Self-healing scales up replicas when p99 latency exceeds SLO for 2 minutes
9. Capacity forecast produces a report showing projected GPU hours needed for next 7 days
10. Post-incident review template is filled out for a simulated GPU OOM incident

---

## Validation Commands

```bash
# Verify SLI metrics are being collected
kubectl exec -n monitoring prometheus-0 -- promtool query instant \
  'sli_inference_availability_ratio[30d]'

# Check error budget recording rules
kubectl exec -n monitoring prometheus-0 -- promtool check rules /etc/prometheus/rules/slo-rules.yaml

# Trigger a test incident and verify detection time
time curl -X POST http://inference-service:8080/health/fail && \
  kubectl logs -n sre -l app=incident-detector --since=2m | grep "INCIDENT_DETECTED"

# Verify self-healing restarts a killed pod
POD=$(kubectl get pod -n inference -l app=model-server -o name | head -1)
kubectl delete $POD -n inference && sleep 35 && \
  kubectl get pod -n inference -l app=model-server | grep Running

# Check capacity forecast output
kubectl exec -n sre deploy/capacity-planner -- cat /reports/forecast-latest.json | jq '.projected_gpu_hours_7d'

# Verify notification was sent
kubectl logs -n sre -l app=incident-notifier --since=5m | grep "notification_sent"

# Run the full SRE validation suite
kubectl apply -f tests/sre-integration-test.yaml && \
  kubectl wait --for=condition=complete job/sre-test -n sre --timeout=120s
```

---

## Technical Implementation Details

### SLI/SLO Definitions

```yaml
# File: k8s/sre/slo-definitions.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: slo-definitions
  namespace: sre
data:
  slos.yaml: |
    slos:
      - name: inference-availability
        sli: "sum(rate(inference_requests_total{code!~'5..'}[5m])) / sum(rate(inference_requests_total[5m]))"
        target: 0.999
        window: 30d
        budget_alert_threshold: 0.25

      - name: inference-latency-p99
        sli: "histogram_quantile(0.99, sum(rate(inference_latency_seconds_bucket[5m])) by (le))"
        target_max: 0.5  # 500ms
        window: 30d

      - name: training-completion-rate
        sli: "sum(rate(training_jobs_succeeded_total[24h])) / sum(rate(training_jobs_total[24h]))"
        target: 0.95
        window: 7d

      - name: gpu-utilization-target
        sli: "avg(DCGM_FI_DEV_GPU_UTIL)"
        target_min: 0.60
        window: 7d
```

### Prometheus Recording Rules for Error Budgets

```yaml
# File: k8s/sre/prometheus-rules/slo-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: slo-error-budget
  namespace: monitoring
spec:
  groups:
    - name: slo.error_budget
      interval: 60s
      rules:
        - record: slo:inference_availability:ratio
          expr: |
            sum(rate(inference_requests_total{code!~"5.."}[30d]))
            / sum(rate(inference_requests_total[30d]))

        - record: slo:error_budget:remaining_ratio
          expr: |
            1 - (
              (1 - slo:inference_availability:ratio)
              / (1 - 0.999)
            )

        - record: slo:error_budget:burn_rate_1h
          expr: |
            (1 - (sum(rate(inference_requests_total{code!~"5.."}[1h])) / sum(rate(inference_requests_total[1h]))))
            / (1 - 0.999)

        - alert: ErrorBudgetNearlyExhausted
          expr: slo:error_budget:remaining_ratio < 0.25
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "Error budget below 25% — consider deployment freeze"
            runbook_url: "https://runbooks.internal/slo/budget-exhausted"
```

### Incident Detector

```python
# File: src/sre/incident_detector.py
import time
from prometheus_api_client import PrometheusConnect
from dataclasses import dataclass
from enum import IntEnum

class Severity(IntEnum):
    P1 = 1  # Service down, all users affected
    P2 = 2  # Degraded, >10% users affected
    P3 = 3  # Minor degradation, <10% users
    P4 = 4  # Cosmetic or low-impact

@dataclass
class Incident:
    id: str
    title: str
    severity: Severity
    detected_at: float
    slo_violated: str
    current_value: float
    target_value: float
    runbook_url: str

class IncidentDetector:
    def __init__(self, prom_url: str, check_interval: int = 30):
        self.prom = PrometheusConnect(url=prom_url)
        self.check_interval = check_interval
        self.active_incidents = {}

    def check_slos(self) -> list[Incident]:
        incidents = []
        availability = self._query_scalar("slo:inference_availability:ratio")
        if availability < 0.999:
            incidents.append(Incident(
                id=f"inc-avail-{int(time.time())}",
                title="Inference availability below SLO",
                severity=Severity.P1 if availability < 0.99 else Severity.P2,
                detected_at=time.time(),
                slo_violated="inference-availability",
                current_value=availability,
                target_value=0.999,
                runbook_url="https://runbooks.internal/inference/availability"
            ))
        burn_rate = self._query_scalar("slo:error_budget:burn_rate_1h")
        if burn_rate > 10:
            incidents.append(Incident(
                id=f"inc-burn-{int(time.time())}",
                title=f"Error budget burn rate critical: {burn_rate:.1f}x",
                severity=Severity.P2,
                detected_at=time.time(),
                slo_violated="error-budget-burn",
                current_value=burn_rate,
                target_value=1.0,
                runbook_url="https://runbooks.internal/slo/high-burn"
            ))
        return incidents

    def _query_scalar(self, query: str) -> float:
        result = self.prom.custom_query(query)
        if result:
            return float(result[0]["value"][1])
        return 0.0
```

### Self-Healing Controller

```python
# File: src/sre/self_healing.py
import subprocess
import json
from kubernetes import client, config

class SelfHealingController:
    REMEDIATIONS = {
        "pod_crash_loop": "restart_pod",
        "high_latency": "scale_up",
        "gpu_oom": "reduce_batch_size",
        "disk_pressure": "cleanup_old_checkpoints",
    }

    def __init__(self):
        config.load_incluster_config()
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()

    def handle(self, issue_type: str, context: dict) -> dict:
        action = self.REMEDIATIONS.get(issue_type)
        if not action:
            return {"status": "no_remediation", "issue": issue_type}
        handler = getattr(self, f"_do_{action}")
        return handler(context)

    def _do_scale_up(self, ctx: dict) -> dict:
        namespace = ctx["namespace"]
        deployment = ctx["deployment"]
        current = self.apps_v1.read_namespaced_deployment(deployment, namespace)
        new_replicas = min(current.spec.replicas + 1, ctx.get("max_replicas", 5))
        current.spec.replicas = new_replicas
        self.apps_v1.patch_namespaced_deployment(deployment, namespace, current)
        return {"status": "scaled", "replicas": new_replicas}

    def _do_restart_pod(self, ctx: dict) -> dict:
        self.core_v1.delete_namespaced_pod(ctx["pod_name"], ctx["namespace"])
        return {"status": "restarted", "pod": ctx["pod_name"]}

    def _do_reduce_batch_size(self, ctx: dict) -> dict:
        cm = self.core_v1.read_namespaced_config_map("model-config", ctx["namespace"])
        config_data = json.loads(cm.data["config.json"])
        config_data["max_batch_size"] = max(1, config_data["max_batch_size"] // 2)
        cm.data["config.json"] = json.dumps(config_data)
        self.core_v1.patch_namespaced_config_map("model-config", ctx["namespace"], cm)
        return {"status": "batch_reduced", "new_size": config_data["max_batch_size"]}

    def _do_cleanup_old_checkpoints(self, ctx: dict) -> dict:
        result = subprocess.run(
            ["find", ctx["checkpoint_path"], "-mtime", "+7", "-delete"],
            capture_output=True, text=True
        )
        return {"status": "cleaned", "output": result.stdout}
```

### Capacity Planning

```yaml
# File: k8s/sre/capacity-planner-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: capacity-planner
  namespace: sre
spec:
  schedule: "0 6 * * *"  # Daily at 6 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: planner
              image: anvil/capacity-planner:latest
              env:
                - name: PROMETHEUS_URL
                  value: "http://prometheus.monitoring:9090"
                - name: FORECAST_DAYS
                  value: "7"
                - name: ALERT_THRESHOLD_PERCENT
                  value: "80"
              volumeMounts:
                - name: reports
                  mountPath: /reports
          volumes:
            - name: reports
              persistentVolumeClaim:
                claimName: sre-reports
          restartPolicy: OnFailure
```

### Post-Incident Review Template

```markdown
# File: docs/templates/post-incident-review.md
## Incident: [TITLE]
**Date:** YYYY-MM-DD | **Duration:** X minutes | **Severity:** P[1-4]
**Author:** [name] | **Reviewers:** [names]

### Timeline
| Time | Event |
|------|-------|
| HH:MM | First alert fired |
| HH:MM | On-call acknowledged |
| HH:MM | Root cause identified |
| HH:MM | Mitigation applied |
| HH:MM | Service restored |

### Impact
- Users affected: [number/percentage]
- Error budget consumed: [X%]
- Revenue impact: [if applicable]

### Root Cause
[2-3 paragraphs explaining the technical root cause]

### What Went Well
- [item]

### What Went Wrong
- [item]

### Action Items
| Action | Owner | Priority | Due Date |
|--------|-------|----------|----------|
| [action] | [name] | P[1-3] | YYYY-MM-DD |

### Lessons Learned
[Paragraph summarizing key takeaways]
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Prometheus not collecting custom metrics | Ensure ServiceMonitor matches your service labels; check `kubectl get servicemonitor -A` |
| Error budget shows NaN | Your SLI query returns no data — ensure inference service is generating traffic |
| Self-healing controller can't access K8s API | Check RBAC: `kubectl auth can-i patch deployments --as=system:serviceaccount:sre:self-healer` |
| Alerts not firing | Check `kubectl exec prometheus-0 -- promtool check rules` and verify alertmanager config |
| Capacity planner forecast is flat | Need at least 7 days of historical data; generate synthetic load to bootstrap |
| Incident detector has high latency | Reduce Prometheus query complexity; use recording rules instead of raw queries |

---

## Agent Handoff Template

```
Resume Week 8: SRE Practices for AI Systems.

Environment: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
K3s cluster: 3 multipass nodes (k3s-master, k3s-worker-1, k3s-worker-2).
Phase A complete: GPU operator, Prometheus/Grafana, KServe deployed.

Current state: [describe what's done and what's next]

Tasks remaining:
- [ ] [list incomplete items from Implementation Goals]

Key files:
- k8s/sre/slo-definitions.yaml
- k8s/sre/prometheus-rules/slo-rules.yaml
- src/sre/incident_detector.py
- src/sre/self_healing.py
- k8s/sre/capacity-planner-cronjob.yaml

The SLO target for inference availability is 99.9% (0.1% error budget per 30-day window).
Self-healing must handle: pod crashes, high latency, GPU OOM, disk pressure.
Validate with: kubectl and the validation commands in the spec.
```

---

## Out of Scope

- Multi-region SLOs (covered in Week 12)
- Cost-based SLOs (covered in Week 9)
- Security incident management (covered in Week 10)
- Chaos testing of SRE systems (covered in Week 11)
- ML-specific lifecycle SLOs (covered in Week 13)
- Production SLA contracts with external customers
- PagerDuty/Opsgenie integration (use webhook-based notifications instead)
