# Week 4: Infrastructure as Code

## Context

**Where it fits:** Phase A, Week 4 — makes everything deployable, reproducible, and auditable. Moves from manual `kubectl apply` to GitOps-driven infrastructure management.

**Prerequisites:**
- Weeks 1-3 completed (cluster running, operator deployed)
- Terraform CLI installed
- ArgoCD concepts (declarative GitOps)
- Git repository for infrastructure state
- Hardware: ASUS ROG Strix SCAR 16 (RTX 5080 16GB, 32GB RAM, 2TB SSD, Ubuntu)

**What it builds on:** Week 2's cluster is now managed by Terraform. Week 3's operator is deployed via ArgoCD. Week 7's integration tests exercise the full GitOps pipeline.

---

## Learning Goals

- [ ] Explain Terraform's plan/apply lifecycle and state management
- [ ] Articulate why state locking matters (concurrent applies, corruption)
- [ ] Describe GitOps principles: Git as single source of truth, reconciliation loops
- [ ] Explain progressive rollout strategies: canary, blue-green, percentage-based
- [ ] Understand spot/preemptible instance economics and interruption patterns
- [ ] Describe Terraform module composition and dependency management
- [ ] Explain ArgoCD's sync waves and resource hooks

---

## Implementation Goals

- [ ] Terraform modules for local infrastructure (Docker containers, K3s nodes, networking)
- [ ] State backend with locking (local with file lock, or MinIO + DynamoDB-compatible)
- [ ] Environment management: dev/staging variable files with workspace isolation
- [ ] ArgoCD installation and configuration on the K3s cluster
- [ ] GitOps workflow: push to Git → ArgoCD detects → syncs to cluster
- [ ] Progressive rollout: canary deployment with health checks before full rollout
- [ ] Spot instance simulation: random pod eviction with checkpoint-before-kill
- [ ] Cost tracking: label-based resource accounting per job and team
- [ ] Drift detection: alert when cluster state diverges from Git
- [ ] Automated rollback on failed health checks

---

## Acceptance Criteria

1. `terraform plan` shows a clear diff of infrastructure changes; `terraform apply` creates the defined resources (Docker networks, containers simulating nodes).
2. Terraform state is locked during apply — a concurrent `terraform apply` fails with a lock error.
3. Separate `dev` and `staging` workspaces produce isolated infrastructure with different variable values.
4. ArgoCD is deployed and its UI is accessible at `https://argocd.local:8443`.
5. Pushing a change to the `infrastructure` Git repo triggers ArgoCD sync within 3 minutes (polling) or 30 seconds (webhook).
6. A canary deployment rolls out to 1 pod first; if health check passes after 60s, proceeds to full rollout.
7. A failed canary (pod crashes) triggers automatic rollback — the previous version remains running.
8. Simulated spot eviction: a pod receives SIGTERM, checkpoints within the grace period, and the orchestrator reschedules it.
9. Cost tracking labels are present on all pods: `anvil.io/team`, `anvil.io/job-id`, `anvil.io/cost-center`.
10. `terraform destroy` cleanly removes all created resources with no orphans.

---

## Validation Commands

```bash
# Initialize Terraform
cd ~/anvil/infrastructure/terraform
terraform init
terraform workspace new dev
terraform plan -var-file=envs/dev.tfvars

# Apply infrastructure
terraform apply -var-file=envs/dev.tfvars -auto-approve
terraform state list

# Test state locking (run in two terminals)
terraform apply -var-file=envs/dev.tfvars &  # Terminal 1
terraform apply -var-file=envs/dev.tfvars     # Terminal 2 — should fail with lock

# Deploy ArgoCD
kubectl apply -n argocd -f deploy/argocd/install.yaml
kubectl wait --for=condition=available deploy/argocd-server -n argocd --timeout=120s
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d

# Create ArgoCD application
kubectl apply -f deploy/argocd/applications/training-operator.yaml
argocd app sync training-operator

# Test GitOps flow
git commit --allow-empty -m "trigger sync" && git push
sleep 60
argocd app get training-operator --show-operation

# Test canary rollout
kubectl apply -f deploy/rollouts/canary-training-operator.yaml
kubectl argo rollouts status training-operator -w

# Spot eviction simulation
python scripts/simulate_spot_eviction.py --pod training-job-worker-2 --grace-period 30

# Cost report
python scripts/cost_report.py --period 24h --group-by team

# Destroy
terraform destroy -var-file=envs/dev.tfvars -auto-approve
```

---

## Technical Implementation Details

### Project Structure

```
~/anvil/infrastructure/
├── terraform/
│   ├── main.tf                # Root module
│   ├── variables.tf           # Input variables
│   ├── outputs.tf             # Output values
│   ├── backend.tf             # State backend config
│   ├── envs/
│   │   ├── dev.tfvars
│   │   └── staging.tfvars
│   └── modules/
│       ├── k3s-cluster/
│       │   ├── main.tf
│       │   ├── variables.tf
│       │   └── outputs.tf
│       ├── gpu-node-pool/
│       │   ├── main.tf
│       │   └── variables.tf
│       ├── networking/
│       │   ├── main.tf
│       │   └── variables.tf
│       └── storage/
│           ├── main.tf
│           └── variables.tf
├── deploy/
│   ├── argocd/
│   │   ├── install.yaml
│   │   └── applications/
│   │       ├── training-operator.yaml
│   │       ├── scheduler-extender.yaml
│   │       └── monitoring.yaml
│   ├── rollouts/
│   │   └── canary-training-operator.yaml
│   └── base/                  # Kustomize base manifests
│       ├── kustomization.yaml
│       └── overlays/
│           ├── dev/
│           └── staging/
├── scripts/
│   ├── simulate_spot_eviction.py
│   ├── cost_report.py
│   └── drift_detector.py
├── gitops-repo/               # Separate repo ArgoCD watches
│   ├── apps/
│   │   ├── training-operator/
│   │   └── monitoring/
│   └── cluster-config/
└── docs/
    └── gitops-workflow.md
```

### Terraform Root Module

```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
}

provider "docker" {}

provider "kubernetes" {
  config_path = var.kubeconfig_path
}

module "networking" {
  source       = "./modules/networking"
  network_name = "${var.environment}-anvil-net"
  subnet_cidr  = var.pod_subnet_cidr
}

module "k3s_cluster" {
  source          = "./modules/k3s-cluster"
  cluster_name    = "${var.environment}-anvil"
  node_count      = var.worker_node_count
  network_id      = module.networking.network_id
  server_memory   = var.server_memory
  worker_memory   = var.worker_memory
}

module "gpu_node_pool" {
  source              = "./modules/gpu-node-pool"
  cluster_id          = module.k3s_cluster.cluster_id
  gpu_node_count      = var.gpu_node_count
  gpu_type            = var.gpu_type
  spot_percentage     = var.spot_percentage
  network_id          = module.networking.network_id
}

module "storage" {
  source          = "./modules/storage"
  cluster_id      = module.k3s_cluster.cluster_id
  minio_enabled   = true
  nfs_enabled     = true
  storage_size_gb = var.storage_size_gb
}
```

### Environment Variables

```hcl
# terraform/envs/dev.tfvars
environment        = "dev"
kubeconfig_path    = "~/.kube/anvil-config"
worker_node_count  = 2
gpu_node_count     = 2
gpu_type           = "simulated-rtx5080"
spot_percentage    = 50
server_memory      = "2048m"
worker_memory      = "4096m"
pod_subnet_cidr    = "10.42.0.0/16"
storage_size_gb    = 50

# terraform/envs/staging.tfvars
environment        = "staging"
kubeconfig_path    = "~/.kube/anvil-staging"
worker_node_count  = 3
gpu_node_count     = 3
gpu_type           = "simulated-rtx5080"
spot_percentage    = 30
server_memory      = "4096m"
worker_memory      = "8192m"
pod_subnet_cidr    = "10.43.0.0/16"
storage_size_gb    = 100
```

### ArgoCD Application

```yaml
# deploy/argocd/applications/training-operator.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: training-operator
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/your-user/anvil-gitops.git
    targetRevision: main
    path: apps/training-operator
    kustomize:
      namePrefix: ""
  destination:
    server: https://kubernetes.default.svc
    namespace: anvil-system
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
    retry:
      limit: 3
      backoff:
        duration: 5s
        maxDuration: 3m
```

### Canary Rollout

```yaml
# deploy/rollouts/canary-training-operator.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: training-operator
  namespace: anvil-system
spec:
  replicas: 3
  strategy:
    canary:
      steps:
        - setWeight: 10
        - pause: {duration: 60s}
        - analysis:
            templates:
              - templateName: success-rate
        - setWeight: 50
        - pause: {duration: 60s}
        - setWeight: 100
      canaryService: training-operator-canary
      stableService: training-operator-stable
  selector:
    matchLabels:
      app: training-operator
  template:
    metadata:
      labels:
        app: training-operator
        anvil.io/team: platform
        anvil.io/cost-center: infra
    spec:
      containers:
        - name: operator
          image: anvil/training-operator:latest
          resources:
            requests:
              cpu: 200m
              memory: 256Mi
```

### Spot Eviction Simulator

```python
# scripts/simulate_spot_eviction.py
import argparse
import time
import subprocess
import json

def simulate_eviction(pod_name: str, namespace: str, grace_period: int):
    """Simulate spot instance preemption with graceful shutdown."""
    print(f"[SPOT] Simulating preemption notice for {pod_name}")
    print(f"[SPOT] Grace period: {grace_period}s")

    # Annotate pod with eviction notice (operator watches for this)
    subprocess.run([
        "kubectl", "annotate", "pod", pod_name,
        f"anvil.io/spot-eviction-time={int(time.time()) + grace_period}",
        "-n", namespace, "--overwrite"
    ], check=True)

    # Wait for grace period (operator should checkpoint during this time)
    print(f"[SPOT] Waiting {grace_period}s for checkpoint...")
    time.sleep(grace_period)

    # Evict the pod
    print(f"[SPOT] Evicting pod {pod_name}")
    subprocess.run([
        "kubectl", "delete", "pod", pod_name,
        "-n", namespace, "--grace-period=5"
    ], check=True)

    # Verify rescheduling
    time.sleep(10)
    result = subprocess.run(
        ["kubectl", "get", "pods", "-n", namespace, "-l",
         f"job-name={pod_name.rsplit('-', 1)[0]}", "-o", "json"],
        capture_output=True, text=True
    )
    pods = json.loads(result.stdout)
    running = sum(1 for p in pods["items"] if p["status"]["phase"] == "Running")
    print(f"[SPOT] Post-eviction: {running} workers running")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--grace-period", type=int, default=30)
    args = parser.parse_args()
    simulate_eviction(args.pod, args.namespace, args.grace_period)
```

### Cost Tracking Script

```python
# scripts/cost_report.py
import subprocess
import json
from collections import defaultdict
from datetime import datetime, timedelta

GPU_COST_PER_HOUR = 2.50  # Simulated cost for RTX 5080
CPU_COST_PER_HOUR = 0.05

def generate_report(period_hours: int, group_by: str):
    result = subprocess.run(
        ["kubectl", "get", "pods", "--all-namespaces", "-o", "json"],
        capture_output=True, text=True
    )
    pods = json.loads(result.stdout)["items"]

    costs = defaultdict(lambda: {"gpu_hours": 0, "cpu_hours": 0, "cost": 0.0})

    for pod in pods:
        labels = pod["metadata"].get("labels", {})
        group_key = labels.get(f"anvil.io/{group_by}", "untracked")
        start = pod["status"].get("startTime")
        if not start:
            continue
        runtime_hours = min(
            (datetime.utcnow() - datetime.fromisoformat(start.rstrip("Z"))).total_seconds() / 3600,
            period_hours
        )
        for container in pod["spec"]["containers"]:
            requests = container.get("resources", {}).get("requests", {})
            gpus = int(requests.get("nvidia.com/gpu", 0))
            cpus = float(requests.get("cpu", "0").rstrip("m")) / 1000

            costs[group_key]["gpu_hours"] += gpus * runtime_hours
            costs[group_key]["cpu_hours"] += cpus * runtime_hours
            costs[group_key]["cost"] += (gpus * GPU_COST_PER_HOUR + cpus * CPU_COST_PER_HOUR) * runtime_hours

    print(f"\n{'='*60}")
    print(f"  ANVIL COST REPORT — Last {period_hours}h (grouped by {group_by})")
    print(f"{'='*60}")
    for group, data in sorted(costs.items(), key=lambda x: x[1]["cost"], reverse=True):
        print(f"  {group:<20} GPU-hrs: {data['gpu_hours']:>6.1f}  CPU-hrs: {data['cpu_hours']:>6.1f}  Cost: ${data['cost']:>8.2f}")
    print(f"{'='*60}")
    total = sum(d["cost"] for d in costs.values())
    print(f"  {'TOTAL':<20} {'':>28} Cost: ${total:>8.2f}\n")
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Terraform can't find Docker provider | Run `terraform init` again. Check `~/.terraform.d/plugins/` or use `terraform providers mirror`. |
| State lock stuck | Find lock file in backend. For local: remove `.terraform.tfstate.lock.info`. For remote: use `terraform force-unlock <ID>`. |
| ArgoCD can't reach Git repo | Use local Git server (gitea in Docker) or file-based repo. Check ArgoCD repo credentials. |
| Sync stuck in "Progressing" | Check `argocd app get <app> --show-operation`. Often a resource is failing to create — check events. |
| Canary never progresses | Verify AnalysisTemplate exists and Prometheus is reachable. Check `kubectl argo rollouts status`. |
| Terraform destroy leaves orphans | Use `terraform state list` to find resources, then `terraform state rm` for orphans. |

---

## Agent Handoff Template

```
Resume Anvil Phase A, Week 4: Infrastructure as Code.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
Project root: ~/anvil/infrastructure/
Cluster: 3-node K3s. Kubeconfig: ~/.kube/anvil-config
Git repo for GitOps: [LOCAL PATH or URL]

Current state: [DESCRIBE - e.g., "Terraform modules work, ArgoCD installed but sync fails"]

What's done:
- [x/blank] Terraform modules (networking, cluster, GPU pool, storage)
- [x/blank] State backend with locking
- [x/blank] Dev/staging environments
- [x/blank] ArgoCD deployed and accessible
- [x/blank] GitOps sync working (push → apply)
- [x/blank] Canary rollout with health checks
- [x/blank] Spot eviction simulation
- [x/blank] Cost tracking per team/job
- [x/blank] Drift detection

Next task: [SPECIFIC NEXT STEP]

Key files:
- terraform/main.tf — root module
- terraform/modules/ — reusable modules
- deploy/argocd/ — ArgoCD config
- scripts/ — operational scripts

Dependencies: Terraform 1.5+, ArgoCD, Argo Rollouts, Docker provider.
```

---

## Out of Scope

- Real cloud provider resources (AWS/GCP/Azure) — we use Docker + K3s as targets
- Terraform Cloud / Spacelift — local state with file locking
- Pulumi or CDK alternatives
- Multi-region deployment
- Secrets management (Vault, SOPS) — use plain ConfigMaps for now
- Compliance frameworks (SOC2, HIPAA)
- Cost optimization recommendations (right-sizing) — just tracking for now
- Production-grade CI/CD (GitHub Actions, GitLab CI) — manual push for now
