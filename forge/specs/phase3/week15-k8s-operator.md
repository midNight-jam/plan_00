# Week 15: Kubernetes Operator for GPU Workloads

## Context

**Phase:** 3 — Production Infrastructure & Advanced Systems
**Prerequisites:** Full platform deployed on K3s (Phase 1), familiarity with Kubernetes primitives (Pods, Services, Deployments), working inference server from earlier weeks.
**Duration:** 1 week
**Difficulty:** Advanced

You've built a working inference platform. Now you'll build the control plane that manages it declaratively. A Kubernetes operator watches for custom resources and reconciles actual state to desired state — this is how production ML platforms (KServe, Seldon, Ray Serve) manage GPU workloads at scale. Building one yourself teaches you the exact mechanics of how GPU scheduling, rolling updates, and auto-scaling work in production clusters.

---

## Learning Goals

- Understand the Kubernetes operator pattern: desired state → observe → diff → act
- Learn how Custom Resource Definitions extend the Kubernetes API
- Grasp GPU-aware scheduling constraints (VRAM is non-shareable, fragmentation matters)
- Implement leader election and reconciliation loop semantics
- Understand finalizers, owner references, and garbage collection in K8s
- Learn how HPA works internally by implementing custom scaling logic

---

## Implementation Goals

- Build a fully functional K8s operator using the `kopf` Python framework
- Define and apply an `InferenceService` CRD with proper validation
- Implement a reconciliation loop that creates/updates/deletes serving pods
- Build GPU-aware scheduling that tracks VRAM per node and bin-packs models
- Implement rolling updates with zero-downtime model version changes
- Build auto-scaling based on request queue depth
- Handle pod failures with exponential backoff restart logic
- Use finalizers for proper resource cleanup on deletion

---

## Acceptance Criteria

1. The `InferenceService` CRD applies cleanly to the cluster with `kubectl apply` and is visible via `kubectl get inferenceservices`.
2. Creating an `InferenceService` resource causes the operator to spin up the correct number of serving pods within 30 seconds.
3. Deleting an `InferenceService` resource triggers finalizer logic that cleanly removes all child pods, services, and configmaps with no orphans.
4. The operator tracks VRAM usage per node and refuses to schedule a model if it would exceed available VRAM (returns an event with reason `InsufficientVRAM`).
5. Updating the `spec.modelVersion` field triggers a rolling update where at least one pod remains serving traffic throughout (zero dropped requests verified by load test).
6. When request queue depth exceeds the configured threshold for 30+ seconds, the operator scales up replicas (up to `spec.maxReplicas`).
7. When request queue depth drops below threshold for 60+ seconds, the operator scales down replicas (down to `spec.minReplicas`).
8. If a serving pod crashes, the operator detects failure within 10 seconds and restarts it with exponential backoff (1s, 2s, 4s, 8s, max 60s).
9. The operator survives its own restart — on startup it re-syncs all existing `InferenceService` resources and reconciles any drift.
10. `kubectl describe inferenceservice <name>` shows accurate status including `readyReplicas`, `vramUsageMB`, `activeRequests`, and condition transitions with timestamps.

---

## Validation Commands

```bash
# Apply the CRD
kubectl apply -f manifests/crd-inferenceservice.yaml
kubectl get crd inferenceservices.forge.ai

# Deploy the operator
kubectl apply -f manifests/operator-deployment.yaml
kubectl logs -f deployment/forge-operator -n forge-system

# Create an InferenceService
kubectl apply -f examples/llama3-service.yaml
kubectl get inferenceservices
kubectl get pods -l forge.ai/managed-by=operator

# Test VRAM tracking
kubectl get nodes -o custom-columns=NAME:.metadata.name,VRAM:.status.allocatable.nvidia\\.com/vram

# Test rolling update
kubectl patch inferenceservice llama3 --type merge -p '{"spec":{"modelVersion":"v2"}}'
# In another terminal, run continuous requests and verify zero 5xx errors

# Test auto-scaling
hey -z 60s -c 50 http://llama3-service:8080/v1/completions
kubectl get inferenceservice llama3 -w  # watch replicas increase

# Test failure recovery
kubectl delete pod -l forge.ai/model=llama3 --grace-period=0
kubectl get events --field-selector reason=PodRestarted

# Test cleanup
kubectl delete inferenceservice llama3
kubectl get pods -l forge.ai/model=llama3  # should be empty
```

---

## Technical Implementation Details

### CRD Definition

```yaml
# manifests/crd-inferenceservice.yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: inferenceservices.forge.ai
spec:
  group: forge.ai
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
              required: [modelName, vramLimitMB, replicas]
              properties:
                modelName:
                  type: string
                modelVersion:
                  type: string
                  default: "latest"
                vramLimitMB:
                  type: integer
                  minimum: 512
                replicas:
                  type: integer
                  minimum: 1
                  maximum: 32
                minReplicas:
                  type: integer
                  default: 1
                maxReplicas:
                  type: integer
                  default: 8
                scalingPolicy:
                  type: object
                  properties:
                    metric:
                      type: string
                      enum: [queueDepth, latencyP99, gpuUtilization]
                    targetValue:
                      type: integer
                    scaleUpStabilization:
                      type: integer
                      default: 30
                    scaleDownStabilization:
                      type: integer
                      default: 60
            status:
              type: object
              properties:
                phase:
                  type: string
                  enum: [Ready, Loading, Error, Scaling]
                readyReplicas:
                  type: integer
                vramUsageMB:
                  type: integer
                activeRequests:
                  type: integer
                conditions:
                  type: array
                  items:
                    type: object
                    properties:
                      type:
                        type: string
                      status:
                        type: string
                      lastTransitionTime:
                        type: string
                      reason:
                        type: string
                      message:
                        type: string
      subresources:
        status: {}
      additionalPrinterColumns:
        - name: Model
          type: string
          jsonPath: .spec.modelName
        - name: Phase
          type: string
          jsonPath: .status.phase
        - name: Ready
          type: integer
          jsonPath: .status.readyReplicas
        - name: VRAM
          type: integer
          jsonPath: .status.vramUsageMB
  scope: Namespaced
  names:
    plural: inferenceservices
    singular: inferenceservice
    kind: InferenceService
    shortNames: [isvc]
```

### Operator Core (kopf)

```python
# operator/main.py
import kopf
import kubernetes
import time
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class NodeGPUState:
    total_vram_mb: int
    allocated_vram_mb: int
    models: list[str]

    @property
    def available_vram_mb(self) -> int:
        return self.total_vram_mb - self.allocated_vram_mb

# In-memory state (rebuilt on startup from cluster state)
gpu_registry: Dict[str, NodeGPUState] = {}
restart_backoff: Dict[str, int] = {}  # pod_name → last backoff seconds


@kopf.on.startup()
async def startup(settings: kopf.OperatorSettings, **kwargs):
    """Re-sync GPU state from existing pods on operator restart."""
    settings.posting.level = logging.INFO
    v1 = kubernetes.client.CoreV1Api()

    # Rebuild GPU registry from node annotations
    nodes = v1.list_node()
    for node in nodes.items:
        vram = int(node.status.allocatable.get("nvidia.com/vram", "0"))
        if vram > 0:
            gpu_registry[node.metadata.name] = NodeGPUState(
                total_vram_mb=vram, allocated_vram_mb=0, models=[]
            )

    # Rebuild allocations from existing managed pods
    pods = v1.list_pod_for_all_namespaces(
        label_selector="forge.ai/managed-by=operator"
    )
    for pod in pods.items:
        if pod.status.phase == "Running":
            node = pod.spec.node_name
            vram = int(pod.metadata.annotations.get("forge.ai/vram-mb", "0"))
            if node in gpu_registry:
                gpu_registry[node].allocated_vram_mb += vram


@kopf.on.create("forge.ai", "v1alpha1", "inferenceservices")
async def on_create(spec, name, namespace, status, patch, **kwargs):
    """Handle new InferenceService creation."""
    model_name = spec["modelName"]
    vram_limit = spec["vramLimitMB"]
    replicas = spec["replicas"]

    patch.status["phase"] = "Loading"
    patch.status["readyReplicas"] = 0

    # Find nodes with sufficient VRAM (bin-packing: pick node with least available)
    for i in range(replicas):
        node = select_node_binpack(vram_limit)
        if node is None:
            patch.status["phase"] = "Error"
            kopf.event(
                {"apiVersion": "forge.ai/v1alpha1", "kind": "InferenceService",
                 "metadata": {"name": name, "namespace": namespace}},
                type="Warning", reason="InsufficientVRAM",
                message=f"No node with {vram_limit}MB VRAM available for replica {i}"
            )
            return

        create_serving_pod(name, namespace, spec, node, replica_index=i)
        gpu_registry[node].allocated_vram_mb += vram_limit


def select_node_binpack(required_vram_mb: int) -> Optional[str]:
    """Select node with least available VRAM that still fits the model."""
    candidates = [
        (name, state) for name, state in gpu_registry.items()
        if state.available_vram_mb >= required_vram_mb
    ]
    if not candidates:
        return None
    # Bin-pack: choose the tightest fit
    candidates.sort(key=lambda x: x[1].available_vram_mb)
    return candidates[0][0]


@kopf.on.update("forge.ai", "v1alpha1", "inferenceservices", field="spec.modelVersion")
async def on_model_update(old, new, name, namespace, spec, patch, **kwargs):
    """Rolling update: replace pods one at a time."""
    patch.status["phase"] = "Scaling"
    replicas = spec["replicas"]

    for i in range(replicas):
        pod_name = f"{name}-{i}"
        # Create new pod with updated version
        create_serving_pod(name, namespace, spec, node=None, replica_index=i, suffix="-canary")
        # Wait for new pod to be ready
        await wait_for_pod_ready(f"{pod_name}-canary", namespace, timeout=120)
        # Delete old pod
        delete_pod(pod_name, namespace)
        # Rename canary (or just keep the new pod)

    patch.status["phase"] = "Ready"


@kopf.on.delete("forge.ai", "v1alpha1", "inferenceservices")
async def on_delete(spec, name, namespace, **kwargs):
    """Finalizer: clean up all child resources."""
    v1 = kubernetes.client.CoreV1Api()

    # Delete all pods with matching labels
    v1.delete_collection_namespaced_pod(
        namespace=namespace,
        label_selector=f"forge.ai/model={name}"
    )

    # Release VRAM allocations
    vram_limit = spec["vramLimitMB"]
    pods = v1.list_namespaced_pod(namespace, label_selector=f"forge.ai/model={name}")
    for pod in pods.items:
        node = pod.spec.node_name
        if node in gpu_registry:
            gpu_registry[node].allocated_vram_mb -= vram_limit

    # Delete associated service
    try:
        v1.delete_namespaced_service(f"{name}-service", namespace)
    except kubernetes.client.exceptions.ApiException:
        pass


@kopf.timer("forge.ai", "v1alpha1", "inferenceservices", interval=10.0)
async def autoscale_check(spec, name, namespace, status, patch, **kwargs):
    """Periodic check for auto-scaling decisions."""
    policy = spec.get("scalingPolicy", {})
    if not policy:
        return

    current_replicas = status.get("readyReplicas", spec["replicas"])
    min_replicas = spec.get("minReplicas", 1)
    max_replicas = spec.get("maxReplicas", 8)
    target = policy.get("targetValue", 10)

    # Get current queue depth from metrics endpoint
    queue_depth = await get_queue_depth(name, namespace)

    if queue_depth > target and current_replicas < max_replicas:
        # Scale up
        new_replicas = min(current_replicas + 1, max_replicas)
        patch.spec["replicas"] = new_replicas
        patch.status["phase"] = "Scaling"
    elif queue_depth < target * 0.5 and current_replicas > min_replicas:
        # Scale down
        new_replicas = max(current_replicas - 1, min_replicas)
        patch.spec["replicas"] = new_replicas
        patch.status["phase"] = "Scaling"
```

### Pod Template Builder

```python
# operator/pod_builder.py
def build_serving_pod(name, namespace, spec, node, replica_index):
    """Construct the Pod manifest for a serving replica."""
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": f"{name}-{replica_index}",
            "namespace": namespace,
            "labels": {
                "forge.ai/managed-by": "operator",
                "forge.ai/model": name,
                "forge.ai/replica": str(replica_index),
            },
            "annotations": {
                "forge.ai/vram-mb": str(spec["vramLimitMB"]),
                "forge.ai/model-version": spec.get("modelVersion", "latest"),
            },
            "ownerReferences": [{
                "apiVersion": "forge.ai/v1alpha1",
                "kind": "InferenceService",
                "name": name,
                "uid": "...",  # filled at creation time
                "controller": True,
                "blockOwnerDeletion": True,
            }],
        },
        "spec": {
            "nodeName": node,
            "containers": [{
                "name": "inference",
                "image": f"forge/serving:{spec.get('modelVersion', 'latest')}",
                "env": [
                    {"name": "MODEL_NAME", "value": spec["modelName"]},
                    {"name": "VRAM_LIMIT_MB", "value": str(spec["vramLimitMB"])},
                ],
                "resources": {
                    "limits": {
                        "nvidia.com/gpu": "1",
                        "nvidia.com/vram": f"{spec['vramLimitMB']}Mi",
                    }
                },
                "ports": [{"containerPort": 8080, "name": "http"}],
                "readinessProbe": {
                    "httpGet": {"path": "/health", "port": 8080},
                    "initialDelaySeconds": 10,
                    "periodSeconds": 5,
                },
                "livenessProbe": {
                    "httpGet": {"path": "/health", "port": 8080},
                    "initialDelaySeconds": 30,
                    "periodSeconds": 10,
                    "failureThreshold": 3,
                },
            }],
            "restartPolicy": "Never",  # Operator manages restarts with backoff
        },
    }
```

### Project Structure

```
forge-operator/
├── manifests/
│   ├── crd-inferenceservice.yaml
│   ├── operator-deployment.yaml
│   ├── rbac.yaml
│   └── namespace.yaml
├── operator/
│   ├── __init__.py
│   ├── main.py              # kopf handlers
│   ├── pod_builder.py       # pod manifest construction
│   ├── gpu_tracker.py       # VRAM accounting
│   ├── scaler.py            # auto-scaling logic
│   └── metrics.py           # Prometheus metrics exposition
├── examples/
│   ├── llama3-service.yaml
│   └── mixtral-service.yaml
├── tests/
│   ├── test_reconcile.py
│   ├── test_gpu_binpack.py
│   └── test_scaling.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| kopf handler not firing | Check RBAC — operator needs `watch`, `list`, `create`, `update`, `delete` on your CRD and pods |
| CRD not showing custom columns | Ensure `additionalPrinterColumns` jsonPath starts with `.` |
| Status subresource not updating | Must use `patch.status[...]` in kopf, not `patch[...]` |
| Operator crash loop | Check for unhandled exceptions in handlers — kopf swallows some errors silently |
| GPU tracking drift over time | Add a periodic re-sync timer that rebuilds state from actual pods |
| Rolling update drops requests | Ensure readiness probe passes on new pod before deleting old pod |
| Finalizer blocks deletion | Finalizer must not raise exceptions — catch all errors and log them |
| Can't test without real GPU | Mock the GPU node labels and use `fake-gpu-operator` for testing |

**Key Resources:**
- [kopf documentation](https://kopf.readthedocs.io/)
- [Kubernetes CRD docs](https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/)
- [Operator pattern explained](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)
- Study KServe's controller for production-grade GPU operator patterns

---

## Agent Handoff Template

```
## Session State
- Phase: 3 / Week 15
- Current task: [what you're working on]
- Branch: forge/week15-k8s-operator

## What's Done
- [ ] CRD defined and applies cleanly
- [ ] kopf operator scaffolded with startup handler
- [ ] Create handler: pods spin up on CRD creation
- [ ] GPU tracking: VRAM bin-packing works
- [ ] Rolling update with readiness gates
- [ ] Auto-scaling timer logic
- [ ] Failure detection and backoff restarts
- [ ] Finalizer cleanup
- [ ] Integration test with K3s cluster

## Current Blocker
[Describe the exact error/issue]

## Key Files
- operator/main.py — core reconciliation handlers
- manifests/crd-inferenceservice.yaml — CRD definition
- tests/test_reconcile.py — unit tests

## Next Step
[Exact next action to take]
```

---

## Out of Scope

- Multi-cluster federation (single cluster only)
- Service mesh integration (Istio/Linkerd)
- GPU time-slicing or MIG partitioning
- Integration with cloud provider managed K8s (GKE Autopilot, EKS)
- Production-grade leader election (use kopf's built-in peering)
- Webhook admission controllers for CRD validation
- Prometheus ServiceMonitor setup (just expose /metrics endpoint)
- Network policies between serving pods
