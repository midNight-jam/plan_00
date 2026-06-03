# Week 10: Security for AI Infrastructure

## Context

**Where it fits:** Phase B, Week 10 — securing the platform before chaos testing and multi-cluster expansion.
**Prerequisites:** Phase A complete (K3s cluster, GPU operator, model serving). Weeks 8-9 complete (SRE practices, cost tracking).
**What it builds on:** The running K3s cluster needs hardening. Models are being served but without supply chain verification. RBAC exists at basic level but needs team-level granularity. Secrets are in plain ConfigMaps that need Vault protection.

Security is non-negotiable for AI infrastructure — models are high-value intellectual property, GPUs are expensive attack targets, and training data may contain sensitive information. This week locks everything down.

---

## Learning Goals

- [ ] Understand secrets management with HashiCorp Vault and the sidecar injector pattern
- [ ] Know how Kubernetes NetworkPolicies enforce microsegmentation
- [ ] Implement Pod Security Standards (restricted profile) without breaking GPU workloads
- [ ] Design model supply chain security with signing and verification
- [ ] Build comprehensive RBAC with team-scoped GPU quotas
- [ ] Create audit trails for compliance and forensics
- [ ] Understand break-glass procedures for emergency access

---

## Implementation Goals

- [ ] Deploy HashiCorp Vault on K3s with auto-unseal, secret rotation, and audit logging
- [ ] Migrate all secrets from ConfigMaps/environment variables to Vault
- [ ] Implement NetworkPolicies isolating namespaces (inference, training, monitoring, cost)
- [ ] Enable Pod Security Standards (restricted) with exceptions for GPU pods only
- [ ] Sign model artifacts with cosign and verify before deployment
- [ ] Scan container images for CVEs on every deployment
- [ ] Configure team-level RBAC with GPU quota enforcement
- [ ] Deploy audit logging capturing all kubectl commands and model deployments
- [ ] Document and test break-glass procedure for emergency cluster access

---

## Acceptance Criteria

1. Vault is running in HA mode with auto-unseal and all application secrets stored in Vault paths
2. Secret rotation occurs automatically every 24 hours without service disruption
3. NetworkPolicies deny all inter-namespace traffic by default; only explicitly allowed paths work
4. A pod without the correct network policy cannot reach the inference service (verified by test)
5. Pod Security Standards reject any pod requesting privileged access (except GPU operator DaemonSet)
6. Model artifacts are signed with cosign and deployment fails if signature verification fails
7. Container image scan blocks deployment of images with critical/high CVEs
8. Team RBAC prevents ml-team from accessing data-eng namespace resources
9. Audit log captures who deployed which model at what time, queryable via CLI
10. Break-glass procedure grants temporary admin access with automatic revocation after 1 hour

---

## Validation Commands

```bash
# Verify Vault is running and unsealed
kubectl exec -n vault vault-0 -- vault status | grep "Sealed.*false"

# Test secret injection into a pod
kubectl apply -f tests/secret-test-pod.yaml -n inference && \
  kubectl exec -n inference secret-test-pod -- cat /vault/secrets/db-password

# Test NetworkPolicy enforcement
kubectl run nettest --image=busybox -n default --rm -it --restart=Never -- \
  wget -qO- --timeout=3 http://inference-service.inference:8080/health && echo "FAIL: should be blocked"

# Verify Pod Security Standards
kubectl apply -f tests/privileged-pod.yaml -n inference 2>&1 | grep "forbidden"

# Verify cosign signature on model artifact
cosign verify --key k8s://cosign-system/cosign-key \
  registry.local:5000/models/bert-base:latest

# Test RBAC isolation
kubectl auth can-i get pods --namespace=data-eng --as=system:serviceaccount:ml-team:ml-deployer | grep "no"

# Check audit log for recent deployments
kubectl logs -n audit -l app=audit-logger --since=1h | jq 'select(.verb=="create" and .resource=="deployments")'

# Test break-glass procedure
kubectl apply -f k8s/security/break-glass-request.yaml && \
  sleep 5 && kubectl auth can-i '*' '*' --as=system:serviceaccount:security:break-glass | grep "yes" && \
  sleep 3605 && kubectl auth can-i '*' '*' --as=system:serviceaccount:security:break-glass | grep "no"

# Run full security scan
kubectl apply -f tests/security-audit-job.yaml && \
  kubectl wait --for=condition=complete job/security-audit -n security --timeout=300s
```

---

## Technical Implementation Details

### HashiCorp Vault Deployment

```yaml
# File: k8s/security/vault-helm-values.yaml
server:
  ha:
    enabled: true
    replicas: 3
    raft:
      enabled: true
  dataStorage:
    size: 5Gi
  auditStorage:
    enabled: true
    size: 5Gi
  extraEnvironmentVars:
    VAULT_LOG_LEVEL: info

injector:
  enabled: true
  replicas: 2

ui:
  enabled: true
  serviceType: ClusterIP
```

```bash
# File: scripts/setup-vault.sh
#!/bin/bash
set -euo pipefail

helm repo add hashicorp https://helm.releases.hashicorp.com
helm install vault hashicorp/vault -n vault --create-namespace \
  -f k8s/security/vault-helm-values.yaml

# Wait for pods
kubectl wait --for=condition=ready pod/vault-0 -n vault --timeout=120s

# Initialize Vault
kubectl exec -n vault vault-0 -- vault operator init \
  -key-shares=5 -key-threshold=3 -format=json > vault-keys.json

# Auto-unseal (for dev - use KMS in production)
for i in 0 1 2; do
  KEY=$(jq -r ".unseal_keys_b64[$i]" vault-keys.json)
  kubectl exec -n vault vault-0 -- vault operator unseal "$KEY"
done

# Enable audit logging
ROOT_TOKEN=$(jq -r '.root_token' vault-keys.json)
kubectl exec -n vault vault-0 -- vault login "$ROOT_TOKEN"
kubectl exec -n vault vault-0 -- vault audit enable file file_path=/vault/audit/audit.log

# Enable Kubernetes auth
kubectl exec -n vault vault-0 -- vault auth enable kubernetes
kubectl exec -n vault vault-0 -- vault write auth/kubernetes/config \
  kubernetes_host="https://kubernetes.default.svc:443"
```

### Secret Rotation Policy

```python
# File: src/security/secret_rotator.py
import hvac
import schedule
import time
import secrets
import string
from datetime import datetime

class SecretRotator:
    def __init__(self, vault_url: str, token: str):
        self.client = hvac.Client(url=vault_url, token=token)
        self.rotation_policies = {
            "database-credentials": {"interval_hours": 24, "length": 32},
            "api-keys": {"interval_hours": 24, "length": 48},
            "tls-certificates": {"interval_hours": 720, "length": None},  # 30 days
        }

    def rotate_secret(self, path: str, policy: dict):
        new_secret = self._generate_secret(policy["length"])
        old_version = self.client.secrets.kv.v2.read_secret_version(path=path)
        self.client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret={"value": new_secret, "rotated_at": datetime.utcnow().isoformat()},
        )
        print(f"[ROTATION] Rotated secret at {path}, old version: {old_version['data']['metadata']['version']}")

    def _generate_secret(self, length: int) -> str:
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    def start_rotation_scheduler(self):
        for path, policy in self.rotation_policies.items():
            schedule.every(policy["interval_hours"]).hours.do(
                self.rotate_secret, path=f"ai-infra/{path}", policy=policy
            )
        while True:
            schedule.run_pending()
            time.sleep(60)
```

### NetworkPolicies

```yaml
# File: k8s/security/network-policies/default-deny.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: inference
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-inference-ingress
  namespace: inference
spec:
  podSelector:
    matchLabels:
      app: model-server
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: istio-system
        - namespaceSelector:
            matchLabels:
              name: monitoring
          podSelector:
            matchLabels:
              app: prometheus
      ports:
        - protocol: TCP
          port: 8080
        - protocol: TCP
          port: 9090
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns-egress
  namespace: inference
spec:
  podSelector: {}
  policyTypes:
    - Egress
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: kube-system
      ports:
        - protocol: UDP
          port: 53
```

### Model Supply Chain Security

```bash
# File: scripts/sign-model.sh
#!/bin/bash
set -euo pipefail

MODEL_IMAGE=$1  # e.g., registry.local:5000/models/bert-base:v1.2

# Generate SBOM
syft "$MODEL_IMAGE" -o spdx-json > sbom.spdx.json

# Scan for vulnerabilities
grype "$MODEL_IMAGE" --fail-on critical

# Sign the image
cosign sign --key k8s://cosign-system/cosign-key "$MODEL_IMAGE"

# Attach SBOM
cosign attach sbom --sbom sbom.spdx.json "$MODEL_IMAGE"

# Verify (as deployment would)
cosign verify --key k8s://cosign-system/cosign-key "$MODEL_IMAGE"

echo "Model $MODEL_IMAGE signed and verified successfully"
```

```yaml
# File: k8s/security/admission-policy.yaml
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: require-image-signature
spec:
  failurePolicy: Fail
  matchConstraints:
    resourceRules:
      - apiGroups: ["apps"]
        apiVersions: ["v1"]
        operations: ["CREATE", "UPDATE"]
        resources: ["deployments"]
    namespaceSelector:
      matchLabels:
        require-signature: "true"
  validations:
    - expression: "object.spec.template.metadata.annotations['cosign.sigstore.dev/verified'] == 'true'"
      message: "All model images must be signed with cosign"
```

### Team RBAC with GPU Quotas

```yaml
# File: k8s/security/rbac/ml-team.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml-team
  labels:
    team: ml-team
    require-signature: "true"
    pod-security.kubernetes.io/enforce: restricted
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: ml-team-role
  namespace: ml-team
rules:
  - apiGroups: ["", "apps", "batch"]
    resources: ["pods", "deployments", "jobs", "services", "configmaps"]
    verbs: ["get", "list", "watch", "create", "update", "delete"]
  - apiGroups: [""]
    resources: ["secrets"]
    verbs: ["get", "list"]  # no create/update — use Vault
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: ml-team-binding
  namespace: ml-team
subjects:
  - kind: ServiceAccount
    name: ml-deployer
    namespace: ml-team
roleRef:
  kind: Role
  name: ml-team-role
  apiGroup: rbac.authorization.k8s.io
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ml-team-gpu-quota
  namespace: ml-team
spec:
  hard:
    requests.nvidia.com/gpu: "2"
    limits.nvidia.com/gpu: "2"
    requests.memory: "16Gi"
    requests.cpu: "8"
```

### Break-Glass Procedure

```yaml
# File: k8s/security/break-glass.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: break-glass-admin
rules:
  - apiGroups: ["*"]
    resources: ["*"]
    verbs: ["*"]
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: break-glass
  namespace: security
---
# CronJob that revokes break-glass access after 1 hour
apiVersion: batch/v1
kind: CronJob
metadata:
  name: break-glass-revoker
  namespace: security
spec:
  schedule: "*/5 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: break-glass-admin-sa
          containers:
            - name: revoker
              image: bitnami/kubectl:latest
              command:
                - /bin/sh
                - -c
                - |
                  # Find and delete expired break-glass bindings
                  kubectl get clusterrolebindings -l break-glass=true -o json | \
                    jq -r '.items[] | select(.metadata.annotations["expires"] < now | todate) | .metadata.name' | \
                    xargs -I {} kubectl delete clusterrolebinding {}
          restartPolicy: OnFailure
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Vault pods stuck in Init | Check PVC provisioning: `kubectl get pvc -n vault`; may need to create StorageClass |
| Vault injector not injecting secrets | Check mutating webhook: `kubectl get mutatingwebhookconfigurations` and pod annotations |
| NetworkPolicy not blocking traffic | K3s uses Flannel by default which doesn't support NetworkPolicy; install Calico: `kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml` |
| Pod Security Standards blocking GPU pods | Add exemption: `pod-security.kubernetes.io/enforce: baseline` for gpu-operator namespace |
| cosign verify fails | Ensure the public key matches; re-generate with `cosign generate-key-pair` |
| RBAC too restrictive for testing | Use `kubectl auth can-i --list --as=system:serviceaccount:ns:sa` to debug permissions |

---

## Agent Handoff Template

```
Resume Week 10: Security for AI Infrastructure.

Environment: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
K3s cluster: 3 multipass nodes. Phase A + Weeks 8-9 complete.

Current state: [describe what's done and what's next]

Tasks remaining:
- [ ] [list incomplete items from Implementation Goals]

Key files:
- k8s/security/vault-helm-values.yaml
- scripts/setup-vault.sh
- src/security/secret_rotator.py
- k8s/security/network-policies/default-deny.yaml
- scripts/sign-model.sh
- k8s/security/rbac/ml-team.yaml
- k8s/security/break-glass.yaml

IMPORTANT: K3s default CNI (Flannel) does NOT support NetworkPolicies.
Must install Calico or Cilium first.
GPU operator namespace needs Pod Security exemption (baseline, not restricted).
Validate with the validation commands in the spec.
```

---

## Out of Scope

- SOC2/HIPAA compliance frameworks
- Data encryption at rest for training data (focus is on secrets and supply chain)
- Identity federation (OIDC/SAML for human users)
- DLP (Data Loss Prevention) for model weights
- Hardware security modules (HSM) — Vault auto-unseal uses simpler approach for dev
- Penetration testing by external team
- Zero-trust network mesh (Istio mTLS — keep it to NetworkPolicies for now)
