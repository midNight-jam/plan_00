# Week 17: Observability at Scale

## Context

**Where it fits:** Phase C, Week 17 of the Anvil AI Infrastructure project. With the platform hardened (Week 16), this week builds enterprise-grade observability: high-cardinality metrics, centralized logging, AI-specific monitoring, and dashboard-as-code.

**Prerequisites:**
- Prometheus and Grafana deployed and scraping GPU/cluster metrics (Phase A)
- All custom controllers emitting events and metrics (Week 16)
- Anvil CLI operational for submitting jobs and querying status (Week 15)
- Basic alerting rules configured from Phase B
- Understanding of PromQL and Grafana dashboard JSON model

**What it builds on:** Phase A deployed basic Prometheus/Grafana. Phase B added per-job metrics. This week scales observability to handle thousands of concurrent jobs without metric explosion, adds structured logging, and creates AI-specific anomaly detection dashboards.

---

## Learning Goals

- [ ] Understand high-cardinality metric problems and solutions (exemplars, recording rules, metric relabeling)
- [ ] Learn Loki architecture: ingester, distributor, querier, chunk storage
- [ ] Study OpenTelemetry correlation: trace IDs linking logs → metrics → traces
- [ ] Understand Thanos/Mimir for long-term metric storage and global query
- [ ] Learn Grafana dashboard provisioning via JSON/YAML and GitOps workflows
- [ ] Study ML model monitoring: concept drift, data drift, prediction quality degradation
- [ ] Understand recording rules and aggregation strategies for cost-effective retention

---

## Implementation Goals

- [ ] Deploy Loki stack for centralized log aggregation with structured logging
- [ ] Implement correlation IDs across all Anvil components (controllers, CLI, jobs)
- [ ] Create recording rules to pre-aggregate high-cardinality GPU metrics
- [ ] Configure exemplars on key metrics linking to specific trace spans
- [ ] Build AI monitoring module: track inference latency percentiles, prediction distributions, input feature drift
- [ ] Deploy Thanos sidecar for long-term metric storage (90-day retention)
- [ ] Create dashboard-as-code repository with CI/CD pipeline to Grafana
- [ ] Build executive dashboard: cluster health, cost trends, GPU utilization heatmap
- [ ] Implement log-based alerting for GPU driver errors and OOM kills
- [ ] Create runbook links in all alerts

---

## Acceptance Criteria

1. Loki ingests logs from all Anvil components with structured JSON format including fields: `timestamp`, `level`, `component`, `correlation_id`, `job_id`, `node`, `gpu_index`.
2. A single correlation ID traces a training job from CLI submission through scheduler, controller, and worker pods — queryable in Grafana via `{correlation_id="X"}`.
3. Recording rules reduce raw GPU metric cardinality from 50k+ series to under 5k aggregated series while preserving per-job granularity for the last 2 hours.
4. Exemplars on `anvil_inference_latency_seconds` histogram link to specific OpenTelemetry trace spans viewable in Grafana Tempo/Jaeger.
5. AI drift detection dashboard shows: prediction distribution shift (KL divergence), input feature statistics (mean, stddev) over time, and alerts when drift exceeds threshold.
6. Thanos query can retrieve metrics from 90 days ago with query latency under 10 seconds for common dashboard queries.
7. All dashboards are version-controlled in `~/anvil/dashboards/` and deployed automatically via `make dashboards-deploy` or GitOps.
8. Executive dashboard loads in under 3 seconds showing: total GPU count, utilization %, cost burn rate, active jobs, queue depth, top-5 resource consumers.
9. Log-based alert fires within 2 minutes of a GPU Xid error appearing in pod logs, with runbook link in the alert annotation.
10. `anvil cluster observability-check` validates all monitoring components are healthy: Prometheus scraping, Loki ingesting, Thanos connected, dashboards loaded.

---

## Validation Commands

```bash
# Deploy Loki stack
cd ~/anvil/observability && helm upgrade --install loki grafana/loki-stack \
  -f values-loki.yaml -n monitoring

# Verify Loki is receiving logs
kubectl logs -n monitoring -l app=loki --tail=5
logcli query '{namespace="training"}' --limit=5

# Check correlation ID propagation
JOB_ID=$(anvil train submit --model ./examples/mnist --gpu 1 | grep "job-id" | awk '{print $2}')
logcli query "{correlation_id=\"$JOB_ID\"}" --limit=50

# Verify recording rules are active
curl -s http://localhost:9090/api/v1/rules | jq '.data.groups[] | select(.name=="anvil-gpu-aggregations")'

# Check metric cardinality
curl -s http://localhost:9090/api/v1/label/__name__/values | jq '. | length'

# Query Thanos for historical data
curl -s "http://localhost:10902/api/v1/query?query=anvil_gpu_utilization_percent[90d]" | jq '.status'

# Deploy dashboards
cd ~/anvil/dashboards && make deploy
curl -s http://localhost:3000/api/search | jq '.[].title'

# Test drift detection
cd ~/anvil/monitoring/drift && python simulate_drift.py --magnitude 0.3
sleep 120
curl -s http://localhost:9093/api/v2/alerts | jq '.[] | select(.labels.alertname=="ModelDriftDetected")'

# Executive dashboard load time
time curl -s "http://localhost:3000/api/dashboards/uid/executive" > /dev/null

# Run observability health check
anvil cluster observability-check
```

---

## Technical Implementation Details

### Project Structure

```
~/anvil/observability/
├── loki/
│   ├── values-loki.yaml
│   ├── promtail-config.yaml
│   └── alert-rules.yaml
├── thanos/
│   ├── sidecar.yaml
│   ├── store-gateway.yaml
│   ├── compactor.yaml
│   └── query-frontend.yaml
├── recording-rules/
│   ├── gpu-aggregations.yaml
│   ├── job-summaries.yaml
│   └── cost-attribution.yaml
├── dashboards/
│   ├── executive/
│   │   └── cluster-overview.json
│   ├── training/
│   │   ├── job-detail.json
│   │   └── gpu-utilization.json
│   ├── inference/
│   │   ├── latency-overview.json
│   │   └── drift-detection.json
│   └── Makefile
├── drift-detection/
│   ├── detector.py
│   ├── metrics_exporter.py
│   └── config.yaml
└── correlation/
    ├── middleware.go
    └── propagation.go
```

### Correlation ID Middleware

```go
// observability/correlation/middleware.go
package correlation

import (
    "context"
    "github.com/google/uuid"
    "go.opentelemetry.io/otel/trace"
)

const CorrelationIDKey = "X-Correlation-ID"

type correlationKey struct{}

func NewCorrelationID() string {
    return uuid.New().String()
}

func WithCorrelationID(ctx context.Context, id string) context.Context {
    return context.WithValue(ctx, correlationKey{}, id)
}

func FromContext(ctx context.Context) string {
    if id, ok := ctx.Value(correlationKey{}).(string); ok {
        return id
    }
    if span := trace.SpanFromContext(ctx); span.SpanContext().IsValid() {
        return span.SpanContext().TraceID().String()
    }
    return NewCorrelationID()
}

func StructuredLogger(ctx context.Context, component string) *slog.Logger {
    return slog.Default().With(
        "correlation_id", FromContext(ctx),
        "component", component,
        "trace_id", trace.SpanFromContext(ctx).SpanContext().TraceID().String(),
    )
}
```

### Recording Rules for GPU Metrics

```yaml
# recording-rules/gpu-aggregations.yaml
groups:
  - name: anvil-gpu-aggregations
    interval: 30s
    rules:
      - record: anvil:gpu_utilization:avg_by_node
        expr: avg by (node) (DCGM_FI_DEV_GPU_UTIL)

      - record: anvil:gpu_memory_used:sum_by_job
        expr: |
          sum by (job_name, namespace) (
            DCGM_FI_DEV_FB_USED * on(pod) group_left(job_name)
            label_replace(kube_pod_labels{label_anvil_io_job!=""}, "job_name", "$1", "label_anvil_io_job", "(.*)")
          )

      - record: anvil:gpu_cost_per_hour:by_team
        expr: |
          sum by (team) (
            anvil:gpu_utilization:avg_by_node * on(node) group_left(team)
            kube_node_labels
          ) * 2.50 / 100
```

### AI Drift Detection

```python
# drift-detection/detector.py
import numpy as np
from scipy.stats import ks_2samp, entropy
from prometheus_client import Gauge, start_http_server

drift_score = Gauge('anvil_model_drift_score', 'KL divergence from reference distribution', ['model', 'feature'])
prediction_shift = Gauge('anvil_prediction_distribution_shift', 'Prediction distribution shift', ['model'])

class DriftDetector:
    def __init__(self, reference_data: dict, window_size: int = 1000):
        self.reference = reference_data
        self.window_size = window_size
        self.current_window = []

    def observe(self, prediction: dict):
        self.current_window.append(prediction)
        if len(self.current_window) >= self.window_size:
            self._compute_drift()
            self.current_window = self.current_window[self.window_size // 2:]

    def _compute_drift(self):
        for feature in self.reference:
            ref_dist = np.histogram(self.reference[feature], bins=50, density=True)[0]
            cur_dist = np.histogram(
                [p[feature] for p in self.current_window], bins=50, density=True
            )[0]
            ref_dist = ref_dist + 1e-10
            cur_dist = cur_dist + 1e-10
            kl_div = entropy(cur_dist, ref_dist)
            drift_score.labels(model=self.model_name, feature=feature).set(kl_div)
```

### Executive Dashboard (Grafana JSON snippet)

```json
{
  "title": "Anvil Executive Dashboard",
  "uid": "executive",
  "panels": [
    {
      "title": "GPU Fleet Utilization",
      "type": "gauge",
      "targets": [{"expr": "avg(anvil:gpu_utilization:avg_by_node) * 100"}],
      "fieldConfig": {"defaults": {"thresholds": {"steps": [
        {"value": 0, "color": "red"},
        {"value": 50, "color": "yellow"},
        {"value": 75, "color": "green"}
      ]}}}
    },
    {
      "title": "Daily Cost Burn Rate",
      "type": "stat",
      "targets": [{"expr": "sum(anvil:gpu_cost_per_hour:by_team) * 24"}]
    },
    {
      "title": "Active Training Jobs",
      "type": "stat",
      "targets": [{"expr": "count(anvil_training_job_status{phase='Running'})"}]
    }
  ]
}
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Loki not receiving logs | Check Promtail DaemonSet is running on all nodes: `kubectl get ds -n monitoring` |
| Correlation ID not propagating | Ensure middleware is injected in all HTTP handlers and gRPC interceptors |
| Recording rules not evaluating | Check `kubectl port-forward svc/prometheus 9090` then `/api/v1/rules` for errors |
| Thanos sidecar not connecting | Verify object storage credentials in secret; check sidecar logs |
| Dashboard provisioning fails | Validate JSON with `jq . < dashboard.json`; check Grafana provisioning logs |
| High cardinality alert firing | Review `prometheus_tsdb_head_series` and add metric relabeling to drop unused labels |
| Drift detection false positives | Increase window size or KL divergence threshold in config.yaml |

---

## Agent Handoff Template

```
Resume Anvil Phase C, Week 17: Observability at Scale.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
State: Phases A+B complete, Week 15-16 done. Prometheus/Grafana running, GPU health controller deployed, CLI operational.

Current goal: Build enterprise observability - Loki for logs, Thanos for long-term metrics, recording rules for cardinality control, AI drift detection, dashboard-as-code.
Key files: ~/anvil/observability/ (Loki, Thanos, recording rules, dashboards, drift detection)
Test with: `anvil cluster observability-check` and `make dashboards-deploy`.

Specific task: [DESCRIBE WHAT TO DO NEXT]
Constraints: Metric cardinality must stay under 5k series after aggregation. All dashboards must be version-controlled. Correlation IDs must span the full request lifecycle.
```

---

## Out of Scope

- Distributed tracing infrastructure (Jaeger/Tempo deployment — assume it exists)
- Log retention policies and compliance (use sensible defaults)
- PagerDuty/Opsgenie integration (alert routing is ops concern)
- Custom Grafana plugin development
- Real-time streaming analytics (batch aggregation is sufficient)
- User-facing dashboards (these are internal/operator dashboards)
- APM for the CLI itself (focus is on cluster workloads)
