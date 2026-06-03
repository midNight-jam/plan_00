# Week 5: Networking for AI Workloads

## Context

**Where it fits:** Phase A, Week 5 — networking is the #1 bottleneck for distributed training at scale. Understanding topology, bandwidth, and latency directly impacts scheduler decisions from Week 3.

**Prerequisites:**
- Weeks 1-3 completed (cluster running, training orchestrator functional)
- Basic TCP/IP networking understanding
- K3s cluster from Week 2 running
- CNI concepts (what a CNI plugin does)
- Hardware: ASUS ROG Strix SCAR 16 (RTX 5080 16GB, 32GB RAM, 2TB SSD, Ubuntu)

**What it builds on:** Week 2's scheduler is extended with network topology awareness. Week 3's gang scheduler uses rack-locality for placement. Week 7's integration tests verify end-to-end bandwidth between training workers.

---

## Learning Goals

- [ ] Explain RDMA, InfiniBand, and RoCE — why they matter for GPU communication
- [ ] Describe NCCL (NVIDIA Collective Communications Library) and its topology discovery
- [ ] Articulate why network topology matters: NVLink > PCIe > InfiniBand > Ethernet
- [ ] Explain CNI plugin architecture (Calico vs Cilium vs Flannel tradeoffs)
- [ ] Describe eBPF and how Cilium uses it for network policy enforcement
- [ ] Understand service discovery patterns for distributed training (headless services, DNS)
- [ ] Explain network policy: deny-by-default, allow specific traffic patterns

---

## Implementation Goals

- [ ] Document RDMA/InfiniBand/NCCL concepts and topology hierarchy (written explanation)
- [ ] Model cluster network topology: nodes, racks, switches, bandwidth between levels
- [ ] Build topology-aware scheduler extension: minimize cross-rack communication
- [ ] Install and configure Cilium CNI on K3s cluster (replace default Flannel)
- [ ] Implement network policies: isolate training job namespaces, allow only intra-job traffic
- [ ] Pod-to-pod encryption (WireGuard via Cilium or manual)
- [ ] Network observability: bandwidth measurement between pods, latency heatmap
- [ ] Service discovery for training workers (headless Service + DNS SRV records)
- [ ] Congestion detection: identify slow links affecting training performance
- [ ] Bandwidth reservation simulation for high-priority jobs

---

## Acceptance Criteria

1. A written document explains RDMA/InfiniBand/NCCL concepts with diagrams showing the bandwidth hierarchy (NVLink 900GB/s → PCIe 64GB/s → InfiniBand 400Gb/s → Ethernet 100Gb/s).
2. Cluster topology is modeled in code: nodes belong to racks, racks connect via simulated switches with defined bandwidth.
3. The topology-aware scheduler places multi-worker jobs on the same rack (or closest nodes) when possible — verified by placement test.
4. Cilium is installed and functional: `cilium status` shows all agents healthy, `cilium connectivity test` passes.
5. Network policy denies cross-namespace pod traffic by default — verified by failed curl between namespaces.
6. Traffic within a training job's worker group is allowed — workers can communicate freely.
7. Pod-to-pod encryption is active — `tcpdump` between nodes shows encrypted traffic (WireGuard).
8. `iperf3` between worker pods shows measured bandwidth; results stored in a metrics database.
9. Headless Service for a training job resolves all worker pod IPs — `nslookup` returns correct pod count.
10. Congestion detection identifies when inter-pod bandwidth drops below threshold and emits an alert.

---

## Validation Commands

```bash
# Install Cilium (replacing flannel)
cd ~/anvil/networking
helm repo add cilium https://helm.cilium.io/
helm install cilium cilium/cilium --namespace kube-system \
  --set encryption.enabled=true \
  --set encryption.type=wireguard

# Verify Cilium
cilium status
cilium connectivity test

# Apply network policies
kubectl apply -f policies/deny-all-cross-namespace.yaml
kubectl apply -f policies/allow-training-workers.yaml

# Test network isolation
kubectl exec -n team-a pod/test -- curl -s --max-time 3 http://svc.team-b.svc:8080
# Should timeout/fail

# Test intra-job communication
kubectl exec -n training-job-123 pod/worker-0 -- curl -s http://worker-1.workers.training-job-123.svc:29500
# Should succeed

# Measure bandwidth
kubectl apply -f tests/iperf-server.yaml -n perf-test
kubectl apply -f tests/iperf-client.yaml -n perf-test
kubectl logs -n perf-test job/iperf-client

# Service discovery test
kubectl exec worker-0 -- nslookup workers.my-training-job.default.svc.cluster.local

# Topology-aware scheduling test
python -m tests.topology_placement --workers 4 --verify-rack-locality

# Congestion detection
python scripts/bandwidth_monitor.py --threshold 1Gbps --alert-on-drop

# Latency heatmap
python scripts/latency_heatmap.py --namespace default --output heatmap.html
```

---

## Technical Implementation Details

### Project Structure

```
~/anvil/networking/
├── topology/
│   ├── __init__.py
│   ├── model.py              # Topology data model (nodes, racks, switches)
│   ├── discovery.py          # Auto-discover topology from node labels
│   ├── scheduler_plugin.py   # Topology-aware scoring for scheduler
│   └── visualize.py          # Generate topology diagram
├── policies/
│   ├── deny-all-cross-namespace.yaml
│   ├── allow-training-workers.yaml
│   ├── allow-monitoring.yaml
│   └── allow-dns.yaml
├── observability/
│   ├── bandwidth_monitor.py  # Continuous bandwidth measurement
│   ├── latency_heatmap.py    # Generate latency matrix
│   ├── congestion_detector.py
│   └── dashboards/
│       └── network-overview.json  # Grafana dashboard
├── service-discovery/
│   ├── headless-service-template.yaml
│   └── worker_discovery.py   # Helper for workers to find peers
├── tests/
│   ├── test_topology_placement.py
│   ├── test_network_policies.py
│   ├── test_bandwidth.py
│   ├── iperf-server.yaml
│   └── iperf-client.yaml
├── docs/
│   ├── rdma-infiniband-nccl.md
│   └── topology-matters.md
└── scripts/
    └── setup-cilium.sh
```

### Network Topology Model

```python
# topology/model.py
from dataclasses import dataclass, field
from enum import Enum

class LinkType(Enum):
    NVLINK = "nvlink"      # 900 GB/s (intra-node GPU-GPU)
    PCIE = "pcie"          # 64 GB/s (intra-node CPU-GPU)
    INFINIBAND = "ib"      # 50 GB/s (inter-node, same rack)
    ETHERNET_25G = "eth25" # 3.1 GB/s (inter-rack)
    ETHERNET_100G = "eth100"  # 12.5 GB/s (inter-rack spine)

@dataclass
class NetworkLink:
    link_type: LinkType
    bandwidth_gbps: float
    latency_us: float

@dataclass
class Node:
    name: str
    rack_id: str
    gpu_count: int
    labels: dict = field(default_factory=dict)

@dataclass
class Rack:
    rack_id: str
    switch_bandwidth_gbps: float = 100.0
    nodes: list[Node] = field(default_factory=list)

@dataclass
class ClusterTopology:
    racks: list[Rack] = field(default_factory=list)
    spine_bandwidth_gbps: float = 400.0

    def distance(self, node_a: str, node_b: str) -> int:
        """Return topology distance: 0=same node, 1=same rack, 2=cross-rack."""
        rack_a = self._find_rack(node_a)
        rack_b = self._find_rack(node_b)
        if node_a == node_b:
            return 0
        if rack_a == rack_b:
            return 1
        return 2

    def bandwidth_between(self, node_a: str, node_b: str) -> float:
        """Effective bandwidth in GB/s between two nodes."""
        dist = self.distance(node_a, node_b)
        if dist == 0:
            return float("inf")
        elif dist == 1:
            return self._rack_bandwidth(self._find_rack(node_a))
        else:
            return self.spine_bandwidth_gbps / 8  # Convert to GB/s

    def _find_rack(self, node_name: str) -> str:
        for rack in self.racks:
            for node in rack.nodes:
                if node.name == node_name:
                    return rack.rack_id
        raise ValueError(f"Node {node_name} not found in topology")

    def _rack_bandwidth(self, rack_id: str) -> float:
        for rack in self.racks:
            if rack.rack_id == rack_id:
                return rack.switch_bandwidth_gbps / 8
        return 0.0
```

### Topology-Aware Scheduler Extension

```python
# topology/scheduler_plugin.py
from model import ClusterTopology

class TopologyAwareScorer:
    def __init__(self, topology: ClusterTopology):
        self.topology = topology

    def score_placement(self, candidate_nodes: list[str], worker_count: int) -> dict[str, int]:
        """Score each candidate node for placing a worker group.
        Prefer nodes that minimize total cross-rack communication."""
        scores = {}
        for node in candidate_nodes:
            locality_score = self._locality_score(node, candidate_nodes, worker_count)
            bandwidth_score = self._bandwidth_score(node, candidate_nodes)
            scores[node] = int(0.6 * locality_score + 0.4 * bandwidth_score)
        return scores

    def _locality_score(self, node: str, all_candidates: list[str], worker_count: int) -> int:
        """How many co-workers can fit in the same rack as this node?"""
        same_rack = [n for n in all_candidates
                     if self.topology.distance(node, n) <= 1]
        # Score 0-100: higher if more workers fit in same rack
        return min(100, int(len(same_rack) / worker_count * 100))

    def _bandwidth_score(self, node: str, all_candidates: list[str]) -> int:
        """Average bandwidth to other candidates."""
        if len(all_candidates) <= 1:
            return 100
        bandwidths = [self.topology.bandwidth_between(node, other)
                      for other in all_candidates if other != node]
        avg_bw = sum(bandwidths) / len(bandwidths)
        # Normalize: 12.5 GB/s = 100, 3.1 GB/s = 25
        return min(100, int(avg_bw / 12.5 * 100))
```

### Network Policies

```yaml
# policies/deny-all-cross-namespace.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-cross-namespace
  namespace: default
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector: {}  # Same namespace only
  egress:
    - to:
        - podSelector: {}  # Same namespace only
    - to:  # Allow DNS
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
```

```yaml
# policies/allow-training-workers.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-training-workers
spec:
  podSelector:
    matchLabels:
      anvil.io/component: training-worker
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchExpressions:
              - key: anvil.io/job-id
                operator: In
                values: ["${JOB_ID}"]  # Templated per job
      ports:
        - protocol: TCP
          port: 29500  # PyTorch distributed port
        - protocol: TCP
          port: 29501  # NCCL port
  egress:
    - to:
        - podSelector:
            matchExpressions:
              - key: anvil.io/job-id
                operator: In
                values: ["${JOB_ID}"]
```

### Bandwidth Monitor

```python
# observability/bandwidth_monitor.py
import asyncio
import subprocess
import json
from datetime import datetime

class BandwidthMonitor:
    def __init__(self, namespace: str, threshold_mbps: float = 1000):
        self.namespace = namespace
        self.threshold = threshold_mbps
        self.measurements: list[dict] = []

    async def measure_pair(self, pod_a: str, pod_b: str) -> float:
        """Measure bandwidth between two pods using iperf3."""
        # Start iperf server on pod_b
        server_cmd = f"kubectl exec -n {self.namespace} {pod_b} -- iperf3 -s -1 -D"
        subprocess.run(server_cmd.split(), capture_output=True)
        await asyncio.sleep(1)

        # Run iperf client on pod_a
        client_cmd = (
            f"kubectl exec -n {self.namespace} {pod_a} -- "
            f"iperf3 -c {pod_b}.workers -t 5 -J"
        )
        result = subprocess.run(client_cmd.split(), capture_output=True, text=True)

        try:
            data = json.loads(result.stdout)
            bw_mbps = data["end"]["sum_received"]["bits_per_second"] / 1_000_000
        except (json.JSONDecodeError, KeyError):
            bw_mbps = 0.0

        self.measurements.append({
            "time": datetime.utcnow().isoformat(),
            "from": pod_a, "to": pod_b,
            "bandwidth_mbps": bw_mbps
        })

        if bw_mbps < self.threshold:
            print(f"[ALERT] Low bandwidth: {pod_a} → {pod_b}: {bw_mbps:.0f} Mbps (threshold: {self.threshold})")

        return bw_mbps

    async def full_mesh_measurement(self, pods: list[str]):
        """Measure bandwidth between all pod pairs."""
        for i, pod_a in enumerate(pods):
            for pod_b in pods[i+1:]:
                bw = await self.measure_pair(pod_a, pod_b)
                print(f"  {pod_a} ↔ {pod_b}: {bw:.0f} Mbps")
```

### Worker Service Discovery

```python
# service-discovery/worker_discovery.py
import socket
import os

def discover_workers(job_name: str, namespace: str = "default") -> list[str]:
    """Discover all worker pod IPs via headless service DNS.
    Each worker pod is addressable as: worker-N.workers.JOB.NAMESPACE.svc.cluster.local
    """
    service_dns = f"workers.{job_name}.{namespace}.svc.cluster.local"
    try:
        _, _, ips = socket.gethostbyname_ex(service_dns)
        return sorted(ips)
    except socket.gaierror:
        return []

def get_rank() -> int:
    """Get this worker's rank from hostname (worker-N)."""
    hostname = os.environ.get("HOSTNAME", socket.gethostname())
    return int(hostname.rsplit("-", 1)[1])

def get_world_size(job_name: str) -> int:
    return len(discover_workers(job_name))

def get_master_addr(job_name: str, namespace: str = "default") -> str:
    """Return the address of worker-0 (master for distributed training)."""
    return f"worker-0.workers.{job_name}.{namespace}.svc.cluster.local"
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Cilium fails to install on K3s | Disable K3s default Flannel: reinstall with `--flannel-backend=none --disable-network-policy`. |
| Network policy not enforced | Verify Cilium is the active CNI: `kubectl get pods -n kube-system -l k8s-app=cilium`. Check `cilium status`. |
| Pods can't resolve DNS after policy | Ensure DNS egress rule exists (allow UDP/53 to kube-dns). See `allow-dns` policy. |
| iperf3 not available in pods | Build a test image with iperf3 installed, or use `networkstatic/iperf3` image. |
| WireGuard encryption not working | Check kernel module: `modprobe wireguard`. Verify with `cilium encrypt status`. |
| Topology labels missing | Manually label nodes: `kubectl label node worker-1 topology.kubernetes.io/rack=rack-1`. |

---

## Agent Handoff Template

```
Resume Anvil Phase A, Week 5: Networking for AI Workloads.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
Project root: ~/anvil/networking/
Cluster: 3-node K3s. Kubeconfig: ~/.kube/anvil-config
CNI: [Cilium installed? Flannel default?]

Current state: [DESCRIBE - e.g., "Cilium installed, network policies applied, but bandwidth monitor shows 0"]

What's done:
- [x/blank] RDMA/InfiniBand/NCCL documentation
- [x/blank] Topology model (nodes, racks, bandwidth)
- [x/blank] Topology-aware scheduler extension
- [x/blank] Cilium CNI installed and healthy
- [x/blank] Network policies (isolation + allow training)
- [x/blank] Pod-to-pod encryption (WireGuard)
- [x/blank] Bandwidth monitoring (iperf3-based)
- [x/blank] Service discovery for workers
- [x/blank] Congestion detection + alerting

Next task: [SPECIFIC NEXT STEP]

Key files:
- topology/model.py — network topology data model
- topology/scheduler_plugin.py — scoring based on locality
- policies/ — NetworkPolicy YAML files
- observability/bandwidth_monitor.py — measurement tool

Dependencies: Cilium Helm chart, iperf3, Prometheus (for metrics).
```

---

## Out of Scope

- Actual RDMA/InfiniBand hardware setup (we simulate topology with labels and annotations)
- GPU Direct RDMA configuration
- SR-IOV network virtualization
- Multi-cluster networking (Cilium ClusterMesh)
- BGP peering configuration
- Hardware load balancers
- IPv6 dual-stack
- DPDK (Data Plane Development Kit)
- Production NCCL tuning (NCCL_TREE_THRESHOLD, etc.) — Phase B
