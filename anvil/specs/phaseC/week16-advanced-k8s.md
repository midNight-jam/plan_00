# Week 16: Advanced Kubernetes Patterns

## Context

**Where it fits:** Phase C, Week 16 of the Anvil AI Infrastructure project. With the developer platform CLI in place (Week 15), this week hardens the underlying Kubernetes infrastructure with GPU-aware controllers, custom metrics, and advanced resource management.

**Prerequisites:**
- Anvil CLI operational from Week 15
- GPU Operator running and exposing basic GPU metrics via DCGM exporter
- Kubernetes cluster with at least 1 GPU node (RTX 5080)
- Prometheus + custom metrics adapter deployed
- Familiarity with controller-runtime and Kubebuilder from Phase A

**What it builds on:** Phase A established basic GPU scheduling. This week makes it production-grade: automatic detection and response to GPU hardware failures, right-sizing via VPA, and safe disruption handling that protects long-running training jobs.

---

## Learning Goals

- [ ] Understand GPU failure modes: ECC errors (correctable vs uncorrectable), thermal throttling, Xid errors, driver hangs
- [ ] Learn Kubernetes custom metrics API and how HPA/VPA consume external metrics
- [ ] Study controller finalizer patterns for safe resource cleanup
- [ ] Understand Pod Disruption Budgets and their interaction with cluster autoscaler
- [ ] Learn Vertical Pod Autoscaler architecture (recommender, updater, admission controller)
- [ ] Study in-place pod resource resizing (KEP-1287, K8s 1.27+)
- [ ] Understand Kubernetes Event emission and status condition patterns

---

## Implementation Goals

- [ ] Build GPU Health Monitor controller that watches DCGM metrics for degradation
- [ ] Implement automatic node cordoning and draining when GPU health drops below threshold
- [ ] Register custom metrics server exposing `gpu_vram_available_bytes` as a K8s metric
- [ ] Deploy and configure VPA for AI workloads with GPU-aware recommendations
- [ ] Create PodDisruptionBudgets that protect training jobs (max 1 unavailable worker)
- [ ] Implement finalizers on TrainingJob CR for graceful checkpoint-on-delete
- [ ] Add comprehensive status conditions to all custom resources
- [ ] Emit Kubernetes Events for all significant state transitions
- [ ] Implement in-place pod resizing for inference workloads where supported
- [ ] Write comprehensive e2e tests simulating GPU failures

---

## Acceptance Criteria

1. When a GPU reports uncorrectable ECC errors (simulated via metric injection), the controller cordons the node within 30 seconds and begins drain within 60 seconds.
2. Thermal throttling detection triggers a warning event and reduces scheduling weight for the affected node within 1 minute.
3. `kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1/namespaces/*/pods/*/gpu_vram_available_bytes` returns current available VRAM for each GPU pod.
4. VPA recommendations for training pods reflect actual GPU memory usage patterns after 24 hours of observation (tested with simulated historical data).
5. A PDB exists for every multi-worker TrainingJob ensuring `maxUnavailable: 1`, and `kubectl drain` respects this by draining one worker at a time.
6. Deleting a TrainingJob triggers the finalizer which initiates a checkpoint save; the CR is not removed until checkpoint completes.
7. All custom resources (TrainingJob, InferenceService, GPUPool) have status conditions: `Ready`, `Progressing`, `Degraded` with human-readable messages.
8. Events are emitted for: job started, checkpoint saved, GPU error detected, node drained, scaling event, job completed/failed.
9. In-place resize of an InferenceService pod (CPU/memory, not GPU) completes without pod restart on K8s 1.27+.
10. Full e2e test suite passes in CI: simulated GPU failure → detection → drain → job migration → recovery, completing in under 5 minutes.

---

## Validation Commands

```bash
# Build and deploy the GPU health monitor controller
cd ~/anvil/controllers/gpu-health && make docker-build docker-push deploy

# Verify controller is running
kubectl get pods -n anvil-system -l app=gpu-health-monitor

# Simulate ECC error by injecting metric
kubectl apply -f test/fixtures/gpu-ecc-error-metric.yaml
sleep 30
kubectl get nodes -o wide | grep -i cordon

# Check custom metrics API
kubectl get --raw '/apis/custom.metrics.k8s.io/v1beta1/namespaces/default/pods/*/gpu_vram_available_bytes' | jq .

# Verify VPA is recommending
kubectl get vpa -n training
kubectl describe vpa training-vpa -n training

# Test PDB protection
kubectl get pdb -n training
kubectl drain <node> --ignore-daemonsets --dry-run=server 2>&1 | grep -i disruption

# Test finalizer (delete a job and watch checkpoint)
kubectl delete trainingjob test-job -n training
kubectl get trainingjob test-job -n training -o jsonpath='{.metadata.finalizers}'

# Check events
kubectl get events -n training --sort-by='.lastTimestamp' | tail -20

# Check status conditions
kubectl get trainingjob -n training -o jsonpath='{.items[*].status.conditions}' | jq .

# Run e2e tests
cd ~/anvil && make test-e2e-gpu-health
```

---

## Technical Implementation Details

### Project Structure

```
~/anvil/controllers/gpu-health/
├── cmd/
│   └── manager/
│       └── main.go
├── internal/
│   ├── controller/
│   │   ├── gpuhealth_controller.go
│   │   ├── gpuhealth_controller_test.go
│   │   ├── node_drainer.go
│   │   └── metrics_registrar.go
│   ├── metrics/
│   │   ├── dcgm_collector.go        # Polls DCGM exporter
│   │   ├── health_scorer.go          # Computes health score from signals
│   │   └── custom_metrics_server.go  # Serves custom.metrics.k8s.io
│   ├── vpa/
│   │   ├── gpu_recommender.go        # GPU-aware VPA recommendations
│   │   └── policy.go                 # Update policies for AI workloads
│   └── disruption/
│       ├── pdb_manager.go            # Creates/updates PDBs for training jobs
│       └── drain_coordinator.go      # Coordinates multi-node drains
├── api/
│   └── v1alpha1/
│       ├── gpuhealth_types.go
│       └── zz_generated.deepcopy.go
├── config/
│   ├── crd/
│   ├── rbac/
│   └── manager/
└── test/
    ├── e2e/
    └── fixtures/
```

### GPU Health Monitor Controller

```go
// internal/controller/gpuhealth_controller.go
package controller

import (
    "context"
    "time"

    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/client"
    corev1 "k8s.io/api/core/v1"
)

type GPUHealthReconciler struct {
    client.Client
    MetricsCollector *metrics.DCGMCollector
    HealthScorer     *metrics.HealthScorer
    NodeDrainer      *NodeDrainer
}

type GPUHealthScore struct {
    NodeName      string
    Score         float64 // 0.0 (dead) to 1.0 (perfect)
    ECCErrors     int64
    Temperature   float64
    Throttled     bool
    XidErrors     []int
}

func (r *GPUHealthReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    node := &corev1.Node{}
    if err := r.Get(ctx, req.NamespacedName, node); err != nil {
        return ctrl.Result{}, client.IgnoreNotFound(err)
    }

    health, err := r.MetricsCollector.CollectForNode(ctx, node.Name)
    if err != nil {
        return ctrl.Result{RequeueAfter: 10 * time.Second}, err
    }

    score := r.HealthScorer.Score(health)

    switch {
    case score.Score < 0.3:
        // Critical: uncorrectable ECC or dead GPU
        r.emitEvent(node, "GPUCritical", "GPU health critical, initiating drain")
        if err := r.NodeDrainer.CordonAndDrain(ctx, node); err != nil {
            return ctrl.Result{RequeueAfter: 5 * time.Second}, err
        }
    case score.Score < 0.6:
        // Warning: throttling or correctable errors accumulating
        r.emitEvent(node, "GPUDegraded", "GPU performance degraded, reducing scheduling weight")
        r.reduceSchedulingWeight(ctx, node)
    case score.Throttled:
        r.emitEvent(node, "GPUThrottled", "GPU thermal throttling detected")
    }

    return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
}
```

### Custom Metrics Server

```go
// internal/metrics/custom_metrics_server.go
package metrics

import (
    "k8s.io/metrics/pkg/apis/custom_metrics"
    "sigs.k8s.io/custom-metrics-apiserver/pkg/provider"
)

type GPUMetricsProvider struct {
    dcgmCollector *DCGMCollector
}

func (p *GPUMetricsProvider) GetMetricByName(ctx context.Context, name types.NamespacedName, info provider.CustomMetricInfo, metricSelector labels.Selector) (*custom_metrics.MetricValue, error) {
    switch info.Metric {
    case "gpu_vram_available_bytes":
        total, used, err := p.dcgmCollector.GetVRAM(ctx, name)
        if err != nil {
            return nil, err
        }
        return &custom_metrics.MetricValue{
            Value: *resource.NewQuantity(total-used, resource.BinarySI),
        }, nil
    }
    return nil, provider.NewMetricNotFoundError(info.GroupResource, info.Metric)
}
```

### PDB Manager

```go
// internal/disruption/pdb_manager.go
package disruption

func (m *PDBManager) EnsurePDBForTrainingJob(ctx context.Context, job *v1alpha1.TrainingJob) error {
    pdb := &policyv1.PodDisruptionBudget{
        ObjectMeta: metav1.ObjectMeta{
            Name:      fmt.Sprintf("%s-pdb", job.Name),
            Namespace: job.Namespace,
            OwnerReferences: []metav1.OwnerReference{
                *metav1.NewControllerRef(job, v1alpha1.GroupVersion.WithKind("TrainingJob")),
            },
        },
        Spec: policyv1.PodDisruptionBudgetSpec{
            MaxUnavailable: &intstr.IntOrString{IntVal: 1},
            Selector: &metav1.LabelSelector{
                MatchLabels: map[string]string{
                    "anvil.io/training-job": job.Name,
                },
            },
        },
    }
    return m.Client.Patch(ctx, pdb, client.Apply, client.FieldOwner("anvil-pdb-manager"))
}
```

### Finalizer Pattern

```go
// internal/controller/finalizer.go
const checkpointFinalizer = "anvil.io/checkpoint-on-delete"

func (r *TrainingJobReconciler) handleDeletion(ctx context.Context, job *v1alpha1.TrainingJob) (ctrl.Result, error) {
    if !controllerutil.ContainsFinalizer(job, checkpointFinalizer) {
        return ctrl.Result{}, nil
    }

    if job.Status.CheckpointStatus != "Saved" {
        if err := r.CheckpointManager.TriggerCheckpoint(ctx, job); err != nil {
            return ctrl.Result{RequeueAfter: 5 * time.Second}, err
        }
        r.emitEvent(job, "CheckpointTriggered", "Saving checkpoint before deletion")
        return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
    }

    controllerutil.RemoveFinalizer(job, checkpointFinalizer)
    return ctrl.Result{}, r.Update(ctx, job)
}
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| DCGM metrics not available | Ensure `dcgm-exporter` DaemonSet is running: `kubectl get ds -n gpu-operator` |
| Custom metrics API returns 404 | Check APIService registration: `kubectl get apiservice v1beta1.custom.metrics.k8s.io` |
| Node drain hangs | Check PDB is not blocking all evictions; verify `maxUnavailable > 0` |
| VPA not updating | VPA needs `updater` component running and pod restart policy set to `Auto` |
| Finalizer preventing deletion | Manually remove finalizer if stuck: `kubectl patch trainingjob X -p '{"metadata":{"finalizers":[]}}' --type=merge` |
| In-place resize not working | Requires K8s 1.27+ with `InPlacePodVerticalScaling` feature gate enabled |
| Controller not reconciling | Check RBAC: controller needs `get/list/watch` on nodes, pods, and custom resources |

---

## Agent Handoff Template

```
Resume Anvil Phase C, Week 16: Advanced Kubernetes Patterns.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
State: Phase A+B complete. Week 15 CLI operational. Cluster running with GPU operator and DCGM exporter.

Current goal: Build GPU health monitoring controller with automatic node drain, custom metrics API, VPA integration, and PDB protection for training jobs.
Key files: ~/anvil/controllers/gpu-health/internal/
Test with: `make test-e2e-gpu-health` (simulates GPU failures in kind cluster).

Specific task: [DESCRIBE WHAT TO DO NEXT]
Constraints: Training jobs must never lose more than 1 worker simultaneously. Checkpoint must complete before any resource is deleted.
```

---

## Out of Scope

- Multi-cluster GPU health federation (single cluster focus)
- Hardware replacement automation (alerts ops team, doesn't auto-replace)
- GPU driver updates or firmware management
- Cluster autoscaler integration (adding/removing nodes)
- Cost optimization based on spot instances
- GPU sharing (MIG/MPS) — addressed in earlier phases
- Network-level disruption handling (focus is GPU/node level)
