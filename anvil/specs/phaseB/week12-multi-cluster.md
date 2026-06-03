# Week 12: Multi-Cluster Federation

## Context

**Where it fits:** Phase B, Week 12 — scaling from single cluster to multi-region architecture.
**Prerequisites:** Phase A complete. Weeks 8-11 complete (SRE, cost, security, chaos resilience proven).
**What it builds on:** The single K3s cluster from Phase A is now hardened and resilient. This week extends to 3 clusters simulating geographic regions. The model serving, monitoring, and deployment patterns from previous weeks get federated across clusters.

You'll simulate 3 regions using either additional multipass VMs or namespace-based isolation on existing nodes. The RTX 5080 GPU is shared across "regions" via time-slicing or MPS. The goal is demonstrating geographic distribution patterns that translate to cloud multi-region deployments.

---

## Learning Goals

- [ ] Understand multi-cluster architectures: federation, mesh, and independent with sync
- [ ] Know how cross-cluster service discovery works (DNS, service mesh, custom)
- [ ] Implement latency-based routing and geographic failover
- [ ] Design model distribution pipelines across clusters
- [ ] Build progressive rollout strategies spanning multiple clusters
- [ ] Aggregate observability data from multiple clusters into a single pane

---

## Implementation Goals

- [ ] Set up 3 K3s clusters representing us-west, us-east, and eu-west regions
- [ ] Implement cross-cluster service discovery so services can find peers in other clusters
- [ ] Build latency-based request routing (measure RTT, route to nearest healthy cluster)
- [ ] Implement failover: detect unhealthy cluster and reroute traffic within 30 seconds
- [ ] Create model distribution pipeline: train in one cluster, push to shared registry, deploy to all
- [ ] Implement progressive rollout: deploy to us-west first, verify metrics, expand to others
- [ ] Deploy global observability aggregating metrics and logs from all 3 clusters
- [ ] Test failover under load (combine with chaos from Week 11)

---

## Acceptance Criteria

1. Three distinct K3s clusters are running and independently functional (each can serve inference)
2. Service in cluster-A can discover and call service in cluster-B by name (not IP)
3. Request routing sends traffic to the cluster with lowest measured latency
4. When one cluster is marked unhealthy, traffic shifts to remaining clusters within 30 seconds
5. Model trained in us-west is automatically available for deployment in us-east and eu-west
6. Progressive rollout deploys to us-west, waits for 5-minute metric validation, then expands
7. Rollout automatically halts if error rate exceeds 1% during canary phase
8. Global Grafana dashboard shows metrics from all 3 clusters in unified view
9. Cross-cluster failover maintains <1 second of request failures during transition
10. Full end-to-end test: train → distribute → progressive deploy → failover → recovery passes

---

## Validation Commands

```bash
# Verify all 3 clusters are running
for cluster in us-west us-east eu-west; do
  kubectl --context k3s-$cluster get nodes | grep Ready
done

# Test cross-cluster service discovery
kubectl --context k3s-us-west exec deploy/test-client -- \
  nslookup inference.us-east.svc.clusterset.local

# Measure cross-cluster latency
kubectl --context k3s-us-west exec deploy/latency-probe -- \
  curl -w "%{time_total}" -s http://inference.us-east.svc.clusterset.local:8080/health

# Test failover
kubectl --context k3s-us-east scale deploy/inference --replicas=0 && sleep 35 && \
  kubectl --context k3s-us-west exec deploy/test-client -- \
  curl -s http://inference-global:8080/predict | jq '.served_by'

# Verify model distribution
kubectl --context k3s-us-west apply -f models/bert-v2-training-job.yaml && \
  kubectl --context k3s-us-west wait --for=condition=complete job/train-bert-v2 --timeout=600s && \
  kubectl --context k3s-eu-west get inferenceservice bert-v2 | grep Ready

# Test progressive rollout
kubectl apply -f rollouts/progressive-bert-v3.yaml --context k3s-us-west && \
  sleep 300 && kubectl --context k3s-us-east get inferenceservice bert-v3 | grep Ready

# Verify global metrics
curl -s http://global-grafana:3000/api/datasources | jq '.[].name' | grep -c "cluster"

# Full integration test
kubectl apply -f tests/multi-cluster-e2e.yaml --context k3s-us-west && \
  kubectl wait --for=condition=complete job/e2e-multi-cluster --timeout=900s --context k3s-us-west
```

---

## Technical Implementation Details

### Multi-Cluster Setup

```bash
# File: scripts/setup-multi-cluster.sh
#!/bin/bash
set -euo pipefail

CLUSTERS=("us-west" "us-east" "eu-west")

for region in "${CLUSTERS[@]}"; do
  echo "Creating cluster: k3s-$region"
  multipass launch --name "k3s-$region-master" --cpus 4 --memory 8G --disk 40G
  multipass exec "k3s-$region-master" -- bash -c "
    curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC='server --cluster-init' sh -
  "
  # Extract kubeconfig
  multipass exec "k3s-$region-master" -- sudo cat /etc/rancher/k3s/k3s.yaml | \
    sed "s/127.0.0.1/$(multipass info k3s-$region-master | grep IPv4 | awk '{print $2}')/g" | \
    sed "s/default/k3s-$region/g" > ~/.kube/k3s-$region.yaml
done

# Merge kubeconfigs
KUBECONFIG=$(echo ~/.kube/k3s-*.yaml | tr ' ' ':') kubectl config view --flatten > ~/.kube/config
echo "All clusters ready. Contexts: k3s-us-west, k3s-us-east, k3s-eu-west"
```

### Cross-Cluster Service Discovery

```yaml
# File: k8s/federation/service-export.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-endpoints
  namespace: federation
data:
  clusters.yaml: |
    clusters:
      - name: us-west
        endpoint: https://10.0.1.10:6443
        inference_url: http://10.0.1.10:30080
        region: us-west-1
        weight: 100
      - name: us-east
        endpoint: https://10.0.2.10:6443
        inference_url: http://10.0.2.10:30080
        region: us-east-1
        weight: 100
      - name: eu-west
        endpoint: https://10.0.3.10:6443
        inference_url: http://10.0.3.10:30080
        region: eu-west-1
        weight: 100
---
# CoreDNS plugin for cross-cluster resolution
apiVersion: v1
kind: ConfigMap
metadata:
  name: coredns-custom
  namespace: kube-system
data:
  clusterset.server: |
    clusterset.local:53 {
        forward . 10.0.1.53 10.0.2.53 10.0.3.53
        log
        errors
    }
```

### Latency-Based Router

```python
# File: src/federation/router.py
import time
import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Optional

@dataclass
class ClusterHealth:
    name: str
    url: str
    latency_ms: float
    healthy: bool
    last_check: float
    consecutive_failures: int = 0

class LatencyBasedRouter:
    def __init__(self, clusters: list[dict], check_interval: int = 10,
                 unhealthy_threshold: int = 3):
        self.clusters = {c["name"]: ClusterHealth(
            name=c["name"], url=c["inference_url"],
            latency_ms=float("inf"), healthy=True, last_check=0
        ) for c in clusters}
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold

    async def start_health_checks(self):
        while True:
            await asyncio.gather(*[
                self._check_cluster(name) for name in self.clusters
            ])
            await asyncio.sleep(self.check_interval)

    async def _check_cluster(self, name: str):
        cluster = self.clusters[name]
        try:
            start = time.monotonic()
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{cluster.url}/health") as resp:
                    if resp.status == 200:
                        cluster.latency_ms = (time.monotonic() - start) * 1000
                        cluster.healthy = True
                        cluster.consecutive_failures = 0
                    else:
                        self._mark_failure(cluster)
        except Exception:
            self._mark_failure(cluster)
        cluster.last_check = time.time()

    def _mark_failure(self, cluster: ClusterHealth):
        cluster.consecutive_failures += 1
        if cluster.consecutive_failures >= self.unhealthy_threshold:
            cluster.healthy = False

    def get_best_cluster(self) -> Optional[ClusterHealth]:
        healthy = [c for c in self.clusters.values() if c.healthy]
        if not healthy:
            return None
        return min(healthy, key=lambda c: c.latency_ms)

    def get_failover_order(self) -> list[ClusterHealth]:
        healthy = [c for c in self.clusters.values() if c.healthy]
        return sorted(healthy, key=lambda c: c.latency_ms)

    async def route_request(self, request_data: dict) -> dict:
        for cluster in self.get_failover_order():
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.post(
                        f"{cluster.url}/predict", json=request_data
                    ) as resp:
                        result = await resp.json()
                        result["served_by"] = cluster.name
                        return result
            except Exception:
                self._mark_failure(cluster)
                continue
        raise Exception("All clusters unavailable")
```

### Model Distribution Pipeline

```yaml
# File: k8s/federation/model-distribution.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: distribute-model
  namespace: federation
spec:
  template:
    spec:
      containers:
        - name: distributor
          image: anvil/model-distributor:latest
          env:
            - name: MODEL_NAME
              value: "bert-v2"
            - name: SOURCE_REGISTRY
              value: "registry.us-west:5000"
            - name: TARGET_REGISTRIES
              value: "registry.us-east:5000,registry.eu-west:5000"
            - name: VERIFY_SIGNATURE
              value: "true"
          command:
            - /bin/sh
            - -c
            - |
              # Pull from source
              skopeo copy --src-tls-verify=false \
                docker://${SOURCE_REGISTRY}/models/${MODEL_NAME}:latest \
                dir:/tmp/model

              # Verify signature
              cosign verify --key /keys/cosign.pub \
                ${SOURCE_REGISTRY}/models/${MODEL_NAME}:latest

              # Push to all targets
              for registry in $(echo $TARGET_REGISTRIES | tr ',' ' '); do
                skopeo copy --dest-tls-verify=false \
                  dir:/tmp/model \
                  docker://${registry}/models/${MODEL_NAME}:latest
                echo "Distributed to $registry"
              done
      restartPolicy: OnFailure
```

### Progressive Rollout Controller

```python
# File: src/federation/progressive_rollout.py
import asyncio
import time
from dataclasses import dataclass
from enum import Enum

class RolloutPhase(Enum):
    CANARY = "canary"
    EXPANDING = "expanding"
    COMPLETE = "complete"
    ROLLED_BACK = "rolled_back"

@dataclass
class RolloutConfig:
    model_name: str
    model_version: str
    cluster_order: list[str]  # Deploy in this order
    canary_duration_seconds: int = 300  # 5 minutes per cluster
    error_rate_threshold: float = 0.01  # 1% max error rate
    latency_p99_threshold_ms: float = 500

class ProgressiveRollout:
    def __init__(self, config: RolloutConfig, metrics_client, deploy_client):
        self.config = config
        self.metrics = metrics_client
        self.deployer = deploy_client
        self.phase = RolloutPhase.CANARY
        self.deployed_clusters: list[str] = []

    async def execute(self) -> dict:
        for cluster in self.config.cluster_order:
            print(f"[ROLLOUT] Deploying {self.config.model_name}:{self.config.model_version} to {cluster}")
            await self.deployer.deploy(
                cluster=cluster,
                model=self.config.model_name,
                version=self.config.model_version,
                traffic_percent=10,  # Start with 10% canary
            )
            healthy = await self._wait_and_validate(cluster)
            if not healthy:
                await self._rollback()
                return {"status": "rolled_back", "failed_at": cluster}

            await self.deployer.promote(cluster, traffic_percent=100)
            self.deployed_clusters.append(cluster)
            print(f"[ROLLOUT] {cluster} promoted to 100% traffic")

        self.phase = RolloutPhase.COMPLETE
        return {"status": "complete", "clusters": self.deployed_clusters}

    async def _wait_and_validate(self, cluster: str) -> bool:
        print(f"[ROLLOUT] Validating {cluster} for {self.config.canary_duration_seconds}s...")
        await asyncio.sleep(self.config.canary_duration_seconds)
        error_rate = await self.metrics.get_error_rate(cluster, self.config.model_version)
        latency_p99 = await self.metrics.get_latency_p99(cluster, self.config.model_version)
        if error_rate > self.config.error_rate_threshold:
            print(f"[ROLLOUT] FAILED: error rate {error_rate:.3f} > {self.config.error_rate_threshold}")
            return False
        if latency_p99 > self.config.latency_p99_threshold_ms:
            print(f"[ROLLOUT] FAILED: p99 latency {latency_p99:.0f}ms > {self.config.latency_p99_threshold_ms}ms")
            return False
        return True

    async def _rollback(self):
        self.phase = RolloutPhase.ROLLED_BACK
        for cluster in self.deployed_clusters:
            await self.deployer.rollback(cluster, self.config.model_name)
            print(f"[ROLLOUT] Rolled back {cluster}")
```

### Global Observability

```yaml
# File: k8s/federation/global-monitoring.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: thanos-sidecar-config
  namespace: monitoring
data:
  thanos.yaml: |
    type: S3
    config:
      bucket: "anvil-metrics"
      endpoint: "minio.storage:9000"
      insecure: true
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: thanos-query
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: thanos-query
  template:
    metadata:
      labels:
        app: thanos-query
    spec:
      containers:
        - name: thanos-query
          image: quay.io/thanos/thanos:latest
          args:
            - query
            - --store=prometheus-us-west.monitoring:10901
            - --store=prometheus-us-east.monitoring:10901
            - --store=prometheus-eu-west.monitoring:10901
            - --query.replica-label=cluster
          ports:
            - containerPort: 9090
              name: http
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Can't run 3 full clusters on single machine | Use namespace-based isolation instead: create namespaces `us-west`, `us-east`, `eu-west` on existing cluster |
| Cross-cluster DNS not resolving | Use NodePort services + explicit IP routing instead of DNS federation |
| Latency measurements all the same (localhost) | Add artificial latency with `tc qdisc add dev eth0 root netem delay 50ms` on VMs |
| Thanos sidecar can't connect to Prometheus | Check Prometheus is exposing gRPC on port 10901: add `--storage.tsdb.min-block-duration=2h` flag |
| Model distribution fails with auth error | Configure registry mirrors or use `--src-tls-verify=false` for local registries |
| Progressive rollout too slow for testing | Reduce `canary_duration_seconds` to 30 for development, restore to 300 for validation |

---

## Agent Handoff Template

```
Resume Week 12: Multi-Cluster Federation.

Environment: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
Original K3s cluster: 3 multipass nodes. Phase A + Weeks 8-11 complete.

Current state: [describe what's done and what's next]

Tasks remaining:
- [ ] [list incomplete items from Implementation Goals]

Key files:
- scripts/setup-multi-cluster.sh
- k8s/federation/service-export.yaml
- src/federation/router.py
- k8s/federation/model-distribution.yaml
- src/federation/progressive_rollout.py
- k8s/federation/global-monitoring.yaml

IMPORTANT: If machine can't handle 3 full clusters, use namespace-based simulation.
RTT between clusters should be simulated with tc netem for realistic routing tests.
Progressive rollout: us-west first, then us-east, then eu-west.
Validate with the validation commands in the spec.
```

---

## Out of Scope

- Actual cloud multi-region deployment (AWS/GCP/Azure)
- Global load balancer (use custom router instead)
- Data sovereignty and GDPR region restrictions
- Multi-cluster Istio service mesh
- Cluster API (CAPI) for cluster lifecycle management
- BGP-based anycast routing
- Cross-cluster persistent volume replication
