# Week 2: Kubernetes Deep Dive

## Context

**Where it fits:** Phase A, Week 2 — transitions from raw distributed systems to the orchestration platform everything runs on. The custom scheduler and webhooks built here are directly used in Weeks 3-4.

**Prerequisites:**
- Week 1 completed (understand consensus, fault tolerance, leader election)
- Docker installed and working
- `kubectl` CLI familiarity (basic pod/deployment operations)
- `multipass` or LXD installed for multi-node VMs
- Hardware: ASUS ROG Strix SCAR 16 (RTX 5080 16GB, 32GB RAM, 2TB SSD, Ubuntu)

**What it builds on:** Week 1's consensus understanding maps directly to etcd (K8s backing store). The scheduler extender built here becomes the foundation for Week 3's gang scheduling and Week 5's topology-aware placement.

---

## Learning Goals

- [ ] Explain K8s control plane components and their interactions (API server, scheduler, controller manager, etcd, kubelet)
- [ ] Describe the scheduler's filtering and scoring pipeline
- [ ] Articulate how the watch mechanism works (informers, reflectors, work queues)
- [ ] Explain CRD lifecycle and controller patterns (level-triggered vs edge-triggered)
- [ ] Describe how the device plugin framework exposes GPUs to pods
- [ ] Understand admission webhook flow (mutating before validating)
- [ ] Explain resource requests vs limits and how they affect scheduling

---

## Implementation Goals

- [ ] Set up 3-node K3s cluster using multipass VMs (1 control-plane, 2 workers)
- [ ] Configure NVIDIA device plugin on worker nodes (GPU visible in `kubectl describe node`)
- [ ] Build custom scheduler extender in Python: GPU topology scoring
- [ ] Implement VRAM-aware scoring (prefer nodes with sufficient contiguous VRAM)
- [ ] Implement GPU anti-affinity (spread training jobs across GPUs)
- [ ] Build mutating admission webhook: auto-inject GPU monitoring sidecar
- [ ] Build validating admission webhook: reject pods requesting more VRAM than available
- [ ] Deploy webhooks with TLS certificates (cert-manager or self-signed)
- [ ] Write integration tests that submit pods and verify placement decisions
- [ ] Document the scheduling decision flow with diagrams

---

## Acceptance Criteria

1. A 3-node K3s cluster is running with nodes visible via `kubectl get nodes` showing Ready status.
2. GPU resources appear in node capacity: `kubectl describe node worker-1 | grep nvidia.com/gpu` shows available GPUs.
3. The custom scheduler extender is called for pods with `schedulerName: anvil-gpu-scheduler` and logs scoring decisions.
4. A pod requesting 8GB VRAM is scheduled to a node with sufficient free VRAM (not just any GPU node).
5. Two pods with anti-affinity labels are placed on different nodes when capacity allows.
6. The mutating webhook injects a `gpu-metrics-exporter` sidecar container into any pod requesting GPU resources.
7. The validating webhook rejects a pod requesting `nvidia.com/gpu: 4` when no node has 4 available GPUs, returning a clear error message.
8. Webhook TLS certificates are valid and the API server can reach the webhook service.
9. Scheduler extender handles node failures gracefully — if a scored node becomes unavailable, the pod is rescheduled.
10. End-to-end test: submit 5 GPU pods with varying VRAM requirements, verify all placed optimally within 30 seconds.

---

## Validation Commands

```bash
# Verify cluster health
kubectl get nodes -o wide
kubectl get pods -n kube-system

# Check GPU visibility
kubectl describe node worker-1 | grep -A5 "Allocated resources"
kubectl describe node worker-1 | grep "nvidia.com/gpu"

# Deploy scheduler extender
kubectl apply -f deploy/scheduler-extender/
kubectl logs -n anvil-system deploy/anvil-scheduler -f

# Test scheduling decision
kubectl apply -f tests/manifests/gpu-pod-8gb.yaml
kubectl get pod gpu-test -o jsonpath='{.spec.nodeName}'

# Test mutating webhook
kubectl apply -f tests/manifests/training-pod.yaml
kubectl get pod training-test -o jsonpath='{.spec.containers[*].name}' | grep gpu-metrics

# Test validating webhook
kubectl apply -f tests/manifests/overcommit-pod.yaml 2>&1 | grep "denied"

# Anti-affinity test
kubectl apply -f tests/manifests/spread-jobs.yaml
kubectl get pods -o wide | grep spread-

# Run full test suite
python -m pytest tests/k8s/ -v --timeout=60
```

---

## Technical Implementation Details

### Project Structure

```
~/anvil/k8s-platform/
├── cluster/
│   ├── setup-multipass.sh    # Create 3 VMs with multipass
│   ├── install-k3s.sh        # Install K3s on VMs
│   └── gpu-plugin.yaml       # NVIDIA device plugin DaemonSet
├── scheduler-extender/
│   ├── main.py               # Flask app handling scheduler callbacks
│   ├── scoring.py            # GPU-aware scoring logic
│   ├── topology.py           # Node GPU topology model
│   ├── Dockerfile
│   └── deploy/
│       ├── deployment.yaml
│       ├── service.yaml
│       └── scheduler-config.yaml
├── webhooks/
│   ├── mutating/
│   │   ├── main.py           # Sidecar injection webhook
│   │   └── Dockerfile
│   ├── validating/
│   │   ├── main.py           # Resource validation webhook
│   │   └── Dockerfile
│   └── deploy/
│       ├── webhook-configs.yaml
│       └── certificates.yaml
├── tests/
│   ├── manifests/
│   └── k8s/
│       ├── test_scheduler.py
│       └── test_webhooks.py
└── docs/
    └── scheduling-flow.md
```

### Multipass Cluster Setup

```bash
#!/bin/bash
# cluster/setup-multipass.sh

# Create VMs
multipass launch --name k3s-master --cpus 4 --memory 8G --disk 40G 22.04
multipass launch --name k3s-worker-1 --cpus 4 --memory 10G --disk 40G 22.04
multipass launch --name k3s-worker-2 --cpus 4 --memory 10G --disk 40G 22.04

# Install K3s on master
MASTER_IP=$(multipass info k3s-master | grep IPv4 | awk '{print $2}')
multipass exec k3s-master -- bash -c "
  curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC='--disable traefik' sh -
"

# Get join token
TOKEN=$(multipass exec k3s-master -- sudo cat /var/lib/rancher/k3s/server/node-token)

# Join workers
for WORKER in k3s-worker-1 k3s-worker-2; do
  multipass exec $WORKER -- bash -c "
    curl -sfL https://get.k3s.io | K3S_URL=https://$MASTER_IP:6443 K3S_TOKEN=$TOKEN sh -
  "
done

# Copy kubeconfig locally
multipass exec k3s-master -- sudo cat /etc/rancher/k3s/k3s.yaml | \
  sed "s/127.0.0.1/$MASTER_IP/" > ~/.kube/anvil-config
export KUBECONFIG=~/.kube/anvil-config
```

### Custom Scheduler Extender

```python
# scheduler-extender/main.py
from flask import Flask, request, jsonify
from scoring import GPUScorer

app = Flask(__name__)
scorer = GPUScorer()

@app.route("/filter", methods=["POST"])
def filter_nodes():
    """Remove nodes that can't satisfy GPU requirements."""
    data = request.json
    pod = data["pod"]
    nodes = data["nodes"]["items"]

    gpu_request = _extract_gpu_request(pod)
    if not gpu_request:
        return jsonify({"nodes": {"items": nodes}})

    eligible = [n for n in nodes if scorer.node_can_fit(n, gpu_request)]
    return jsonify({"nodes": {"items": eligible}})

@app.route("/prioritize", methods=["POST"])
def prioritize_nodes():
    """Score nodes based on GPU topology and VRAM availability."""
    data = request.json
    pod = data["pod"]
    nodes = data["nodes"]["items"]

    gpu_request = _extract_gpu_request(pod)
    scores = []
    for node in nodes:
        score = scorer.score_node(node, gpu_request)
        scores.append({
            "host": node["metadata"]["name"],
            "score": score  # 0-100
        })
    return jsonify(scores)

def _extract_gpu_request(pod: dict) -> dict | None:
    for container in pod["spec"]["containers"]:
        resources = container.get("resources", {}).get("requests", {})
        if "nvidia.com/gpu" in resources:
            return {
                "count": int(resources["nvidia.com/gpu"]),
                "vram_gb": int(container.get("resources", {})
                    .get("requests", {})
                    .get("anvil.io/vram-gb", "0"))
            }
    return None
```

### GPU Scoring Logic

```python
# scheduler-extender/scoring.py
from dataclasses import dataclass

@dataclass
class GPUInfo:
    gpu_id: int
    vram_total_gb: int
    vram_used_gb: int
    pcie_bus: str
    numa_node: int

class GPUScorer:
    def __init__(self):
        self.node_gpus: dict[str, list[GPUInfo]] = {}

    def score_node(self, node: dict, gpu_request: dict) -> int:
        node_name = node["metadata"]["name"]
        gpus = self._get_node_gpus(node_name)

        score = 0
        # Prefer nodes with exact GPU count match (bin-packing)
        available = sum(1 for g in gpus if g.vram_used_gb == 0)
        if available == gpu_request["count"]:
            score += 40  # Exact fit bonus

        # Prefer nodes with GPUs on same NUMA node (topology)
        numa_groups = self._group_by_numa(gpus)
        for numa_gpus in numa_groups.values():
            free_in_group = [g for g in numa_gpus if g.vram_used_gb == 0]
            if len(free_in_group) >= gpu_request["count"]:
                score += 30  # Same NUMA bonus
                break

        # VRAM headroom score (prefer tight fit)
        if gpu_request.get("vram_gb"):
            headroom = self._vram_headroom(gpus, gpu_request["vram_gb"])
            score += max(0, 30 - headroom)  # Less waste = higher score

        return min(score, 100)
```

### Mutating Admission Webhook

```python
# webhooks/mutating/main.py
import json
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)

SIDECAR = {
    "name": "gpu-metrics-exporter",
    "image": "anvil/gpu-metrics:latest",
    "resources": {"requests": {"cpu": "100m", "memory": "64Mi"}},
    "volumeMounts": [{"name": "gpu-metrics", "mountPath": "/metrics"}]
}

@app.route("/mutate", methods=["POST"])
def mutate():
    admission_review = request.json
    pod = admission_review["request"]["object"]

    patches = []
    if _requests_gpu(pod):
        containers = pod["spec"].get("containers", [])
        patches.append({
            "op": "add",
            "path": f"/spec/containers/{len(containers)}",
            "value": SIDECAR
        })

    response = {
        "apiVersion": "admission.k8s.io/v1",
        "kind": "AdmissionReview",
        "response": {
            "uid": admission_review["request"]["uid"],
            "allowed": True,
            "patchType": "JSONPatch",
            "patch": base64.b64encode(json.dumps(patches).encode()).decode()
        }
    }
    return jsonify(response)
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| multipass VMs can't reach each other | Check `multipass list` for IPs. Ensure VMs are on same bridge network. Try `multipass exec vm -- ping <other-ip>`. |
| K3s worker won't join | Verify token is correct, master IP reachable from worker. Check `journalctl -u k3s-agent` on worker. |
| GPU not visible in K8s | Ensure NVIDIA drivers installed in VM, device plugin DaemonSet running. Check `kubectl logs -n kube-system <nvidia-device-plugin-pod>`. |
| Scheduler extender not called | Verify `KubeSchedulerConfiguration` references your extender URL. Check scheduler logs for HTTP errors. |
| Webhook connection refused | Verify Service selector matches webhook pod labels. Check TLS cert CN matches service DNS name. |
| Webhook cert issues | Use `openssl s_client -connect <svc-ip>:443` to debug. Ensure `caBundle` in webhook config matches CA. |

---

## Agent Handoff Template

```
Resume Anvil Phase A, Week 2: Kubernetes Deep Dive.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
Project root: ~/anvil/k8s-platform/
Cluster: 3-node K3s via multipass (k3s-master, k3s-worker-1, k3s-worker-2)
Kubeconfig: ~/.kube/anvil-config

Current state: [DESCRIBE - e.g., "Cluster running, scheduler extender deployed but scoring returns 0 for all nodes"]

What's done:
- [x/blank] 3-node K3s cluster running
- [x/blank] NVIDIA device plugin deployed
- [x/blank] Custom scheduler extender (filter + prioritize)
- [x/blank] VRAM-aware scoring
- [x/blank] Mutating webhook (sidecar injection)
- [x/blank] Validating webhook (resource rejection)
- [x/blank] Integration tests passing

Next task: [SPECIFIC NEXT STEP]

Key files:
- scheduler-extender/main.py — scheduler HTTP endpoints
- scheduler-extender/scoring.py — GPU scoring algorithm
- webhooks/mutating/main.py — sidecar injection
- webhooks/validating/main.py — resource validation

Dependencies: Flask, kubernetes Python client, cert-manager (or self-signed certs).
```

---

## Out of Scope

- Production multi-GPU servers (we simulate GPU topology with annotations)
- MIG (Multi-Instance GPU) partitioning — covered in Phase B
- Cluster autoscaler — fixed node count for now
- Helm chart packaging — raw manifests are fine
- Service mesh (Istio/Linkerd) — Week 5 covers networking
- etcd backup/restore procedures — trust K3s embedded etcd
- RBAC/multi-tenancy — Week 3 handles fair-share per team
- Windows node support
