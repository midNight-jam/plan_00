# Week 2: Kubernetes for AI — DRA, Kueue, and Admission Control (v2 — modernized)

> **v2 note:** Replaces v1 `week02-k8s-deep-dive.md` (preserved in `original_artifacts/specs_v1/anvil_phaseA/`). Rationale from `original_artifacts/plan_evolution_v2_2026-07.md`: the v1 spec was built on the scheduler-extender + device-plugin + custom-`vram` extended-resource pattern — the **2024 stack**. Dynamic Resource Allocation (DRA) went GA in Kubernetes 1.34 (March 2026) and replaces the device-plugin model; Kueue owns queue admission/quota/preemption for batch workloads. Same learning goals as v1 — how GPU-aware scheduling actually works — on the APIs the industry now runs. The admission-webhook material survives unchanged (still-current API).

## Context

**Where it fits:** Phase A, Week 2 — the orchestration substrate for everything after. Week 3's training orchestrator submits jobs *through Kueue* onto *DRA-claimed devices*; Week 16's GPU health controller publishes device health that this week's machinery consumes.

**Prerequisites:**
- Week 1 completed; Docker + kubectl familiarity; `multipass` installed
- Hardware: ASUS ROG Strix SCAR 16 (RTX 5080 16GB, 32GB RAM, Ubuntu)

**What it builds on:** Week 1's Raft maps to etcd (the consensus store all of this state lives in). This week's claims/quotas/gang admission become Week 3's foundation.

---

## Learning Goals

- [ ] Explain the K8s control plane (API server, scheduler, controller manager, etcd, kubelet) and the watch/informer mechanism
- [ ] Explain **DRA**: DeviceClass, ResourceClaim, ResourceClaimTemplate, ResourceSlice, structured parameters, and the allocation flow (who writes what, when a claim binds)
- [ ] Explain what DRA fixes vs the device-plugin/extended-resource model (`nvidia.com/gpu: 1`): parameterized selection, attribute-based matching (CEL), fine-grained sharing, topology constraints
- [ ] Explain **Kueue**: ClusterQueue/LocalQueue/ResourceFlavor, admission vs scheduling, quota borrowing, preemption — and why "don't start a job you can't fully place" (gang admission) matters for training
- [ ] Explain admission webhooks (mutating before validating) — unchanged from v1
- [ ] Articulate the migration story: device plugin → DRA (you deployed the device plugin in Forge W5 — connect the two)

## Implementation Goals

- [ ] 3-node K3s cluster via multipass on a K8s version ≥1.34 (DRA GA); verify `resource.k8s.io/v1` API group is served
- [ ] Deploy the **dra-example-driver** (kubernetes-sigs) so each worker advertises *simulated* GPUs via ResourceSlices (multipass VMs have no real GPU — simulation is the honest path; the real RTX 5080 stays on the host for Forge)
- [ ] Extend/configure the simulated devices with attributes (`vramGB`, `numaNode`, `deviceIndex`) so CEL selectors have something to match
- [ ] Schedule pods via ResourceClaimTemplates: attribute-selected placement (e.g., `device.attributes["vramGB"] >= 8`)
- [ ] Install **Kueue**; define ResourceFlavors + ClusterQueue/LocalQueue with device quotas; gate Jobs through queue admission
- [ ] Gang admission: multi-pod Job admitted all-or-nothing; priority-based **preemption** demonstrated
- [ ] Admission webhooks (kept from v1): validating webhook rejects claims exceeding cluster device capacity with a clear message; mutating webhook injects a metrics sidecar into device-requesting pods (TLS via cert-manager or self-signed)
- [ ] Written migration doc: device-plugin vs DRA, with the scheduling-flow diagram redrawn for claims
- [ ] Integration tests submitting varied workloads and asserting placement/admission decisions

## Acceptance Criteria

1. **Cluster + API**: 3-node K3s cluster Ready on K8s ≥1.34; `kubectl api-resources --api-group=resource.k8s.io` lists deviceclasses, resourceclaims, resourceclaimtemplates, resourceslices
2. **Devices advertised**: dra-example-driver runs on both workers; `kubectl get resourceslices` shows ≥2 simulated devices per worker with your custom attributes (vramGB, numaNode)
3. **Claim allocation**: a pod using a ResourceClaimTemplate schedules, the claim shows `allocated`, and the device attribute set is visible in the claim status
4. **Attribute selection**: a claim with CEL selector `vramGB >= 8` lands only on qualifying devices; requesting an attribute no device has leaves the pod Pending with an explanatory event (verified via `kubectl describe`)
5. **Multi-device claim**: a pod requesting 2 devices in one claim gets both on the same node, or stays Pending when only fragmented capacity exists — behavior demonstrated and documented
6. **Kueue admission**: with quota for N devices, submitting jobs totaling >N leaves the excess **suspended by Kueue** (not Pending at the scheduler); they admit automatically as capacity frees
7. **Gang semantics**: a 4-pod Job admits all-or-nothing — with quota for only 2 of its pods it stays fully suspended; no partial start ever observed
8. **Preemption**: a high-priority workload preempts an admitted lower-priority job (eviction observed, low-priority job requeued and re-admitted later)
9. **Webhooks work**: validating webhook rejects an over-capacity request with a clear denial message; mutating webhook injects the `gpu-metrics-exporter` sidecar into device-requesting pods; TLS verified
10. **Docs + tests**: `docs/dra-vs-device-plugin.md` (what changed, migration path, flow diagram) committed; `pytest tests/k8s/ -v` passes — ≥5 workloads with varied device needs, all placement/admission assertions green

## Validation Commands

```bash
# Cluster + DRA API present
kubectl get nodes -o wide
kubectl version | grep -i server        # ≥ v1.34
kubectl api-resources --api-group=resource.k8s.io

# Simulated devices
kubectl get resourceslices -o yaml | head -50
kubectl get deviceclasses

# Claim-based scheduling
kubectl apply -f tests/manifests/pod-with-claim.yaml
kubectl get resourceclaims -A
kubectl describe pod claim-test | grep -A5 Events

# CEL selector — should stay Pending (no 80GB device simulated)
kubectl apply -f tests/manifests/pod-claim-vram80.yaml
kubectl describe pod vram80-test | grep -A3 Events

# Kueue
kubectl apply -f deploy/kueue/cluster-queue.yaml
kubectl create -f tests/manifests/batch-job-gang4.yaml
kubectl get workloads -A          # suspended vs admitted
kubectl get jobs -w

# Preemption
kubectl create -f tests/manifests/high-priority-job.yaml
kubectl get workloads -A -w       # watch eviction + requeue

# Webhooks
kubectl apply -f tests/manifests/overcommit-claim.yaml 2>&1 | grep -i denied
kubectl get pod training-test -o jsonpath='{.spec.containers[*].name}' | grep gpu-metrics

# Tests
python -m pytest tests/k8s/ -v --timeout=120
```

## Technical Implementation Details

### Project structure
```
~/anvil/k8s-platform/
├── cluster/            # multipass + K3s setup scripts (reuse v1's, pin K8s ≥1.34)
├── dra/
│   ├── driver/         # dra-example-driver deployment + device config (attributes)
│   ├── deviceclasses/  # DeviceClass definitions
│   └── claims/         # example ResourceClaimTemplates with CEL selectors
├── kueue/
│   ├── install.yaml    # pinned Kueue release
│   ├── flavors.yaml    # ResourceFlavors
│   └── queues.yaml     # ClusterQueue + LocalQueues (team-a, team-b)
├── webhooks/           # mutating (sidecar) + validating (capacity) — v1 code carries over
├── tests/{manifests,k8s}/
└── docs/dra-vs-device-plugin.md
```

### Cluster setup (Day 1)
Reuse v1's multipass scripts (in the preserved spec) with one change: pin the K3s channel/version to a K8s ≥1.34 release (`INSTALL_K3S_VERSION` env or `INSTALL_K3S_CHANNEL=latest`; verify with `kubectl version`). If the packaged K3s lags, use `kubeadm` on the VMs or `kind` on the host as fallback — the APIs matter, not the distro.

### DRA hands-on (Days 2–4)
- Deploy kubernetes-sigs **dra-example-driver** (it exists precisely for clusters without real accelerators). Configure its simulated devices per node — number of devices and attribute values (set two devices with `vramGB: 16` and one with `vramGB: 8` per worker so selectors have real work).
- Write DeviceClass(es) targeting the driver; ResourceClaimTemplates exercising: single device, attribute CEL selection, multi-device.
- Trace one allocation end to end in `docs/`: claim created → scheduler + driver negotiation (structured parameters) → allocation written to claim status → kubelet prepares device → pod starts. This narrative is the interview artifact.
- Optional stretch (host GPU, no VMs): try the **NVIDIA DRA driver** against the real RTX 5080 on a single-node K3s on the host; one paragraph of findings. Don't sink >half a day — the sm_120 datapoint is blog material if it works, skippable if not.

### Kueue (Days 4–6)
- Install pinned release; ResourceFlavor for the simulated device type; ClusterQueue with quota (e.g., 4 devices), two LocalQueues (team-a, team-b) for the fair-sharing story Week 3 builds on.
- Jobs with `kueue.x-k8s.io/queue-name` labels; `suspend: true` flow — watch Kueue unsuspend on admission.
- Gang: a 4-pod Job (completions/parallelism 4) with quota 2 → fully suspended; raise quota → admits whole. Document that Kueue admission ≠ kube-scheduler placement, and how the two layers divide the "don't start what you can't finish" problem.
- Preemption: WorkloadPriorityClass high/low; verify eviction + requeue of the low-priority workload.

### Webhooks (Day 6, carried from v1)
The v1 webhook code (Flask, AdmissionReview, JSONPatch sidecar injection) carries over nearly verbatim — the API is still current. Update the validating webhook's logic to read ResourceSlices (sum simulated capacity) instead of node extended-resources.

### Tests + docs (Day 7)
- pytest via the kubernetes Python client: apply manifest → poll → assert node/claim/suspension state; ≥5 scenarios (single, CEL-selected, multi-device, over-quota gang, preemption).
- `docs/dra-vs-device-plugin.md`: side-by-side (advertising, requesting, selecting, sharing, topology), migration path, updated flow diagram. Close with the Forge tie-in: what Forge W5's device-plugin deploy would look like ported to DRA (that port is Forge W15's job).

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| K3s version lacks `resource.k8s.io/v1` | Pin newer K3s (`INSTALL_K3S_VERSION`); fallback: `kind` cluster on host with a ≥1.34 node image |
| dra-example-driver pods CrashLoop | Check the driver's kubelet-plugin socket mount and node feature requirements; read its README version matrix against your K8s minor |
| Claim never allocates | `kubectl describe resourceclaim` + scheduler logs; usually DeviceClass selector doesn't match any ResourceSlice attribute — print the slices and eyeball attribute names/casing |
| CEL selector errors | Validate expression syntax against the DRA docs examples; attributes are typed (int vs string) — `vramGB` must be numeric in the slice to compare with `>=` |
| Kueue admits partially (looks non-gang) | Ensure the Job is one Workload (single Job object, not N Jobs); check `waitForPodsReady`/admission config |
| Webhook cert issues | Same as ever: CN must match service DNS name; `caBundle` must match the CA; `openssl s_client` to debug |
| Multipass VMs starve the laptop | 3 VMs × 8GB is most of your 32GB — trim VM memory to 4–6GB each; K3s runs fine |

## Agent Handoff Template

```
Resume Anvil Phase A, Week 2: K8s DRA + Kueue + admission control (v2 spec).
Spec: /home/zzjam/Documents/dev/plan_00/anvil/specs/phaseA/week02-k8s-dra-kueue.md
Hardware: RTX 5080 16GB, 32GB RAM, Ubuntu. Cluster: 3-node K3s via multipass (K8s ≥1.34).
Project root: ~/anvil/k8s-platform/  Kubeconfig: ~/.kube/anvil-config

Current state: [DESCRIBE]
What's done:
- [ ] cluster ≥1.34 + resource.k8s.io/v1 served
- [ ] dra-example-driver + attributed ResourceSlices
- [ ] claim-based scheduling + CEL selection
- [ ] Kueue queues/quotas + gang admission
- [ ] preemption demo
- [ ] webhooks (validating capacity / mutating sidecar)
- [ ] docs + pytest suite

Next task: [SPECIFIC NEXT STEP]
```

## Out of Scope

- Custom scheduler extender / custom `vram` extended resources (the v1 approach — superseded by DRA; v1 spec preserved for reference)
- Writing a production DRA driver from scratch (configure/extend the example driver; a from-scratch driver is a possible Phase C stretch)
- Real GPU virtualization (MIG/MPS/time-slicing — Phase B Week 9, DRA-native where possible)
- KAI scheduler / Volcano (Kueue is the chosen tool; others are compared in the write-up only)
- Gateway API Inference Extension (serving-side routing — Forge W15's territory)
- Cluster autoscaling, service mesh, Helm packaging
