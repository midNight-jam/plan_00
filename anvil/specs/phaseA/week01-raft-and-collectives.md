# Week 1: Raft Essentials + Collective Communication (v2 — rebalanced)

> **v2 note:** Replaces v1 `week01-distributed-systems.md` (preserved in `original_artifacts/specs_v1/anvil_phaseA/`). Rationale from `original_artifacts/plan_evolution_v2_2026-07.md`: Raft-from-scratch is classic but low-differentiation for an already-senior platform engineer — compressed to 3 days at whiteboard-mastery depth. The reclaimed 4 days go to **collective communication (ring allreduce from scratch, NCCL concepts)** — the AI-native distributed fundamental your profile is missing, absorbed from Crucible Week 6. This feeds Week 3 (gang scheduling), Week 5 (topology-aware placement), and the Phase B RL-infrastructure flagship.

## Context

**Where it fits:** Phase A, Week 1 — two foundations in one week. Raft gives you the consensus vocabulary every control plane (etcd, hence Kubernetes) is built on. Collectives give you the data-plane vocabulary every distributed training and RL system is built on. For AI-infrastructure interviews, the second is the rarer signal.

**Prerequisites:**
- Python 3.11+, asyncio basics, PyTorch installed (`torch.distributed` with gloo backend — no GPU needed for Part B correctness work)
- Hardware: ASUS ROG Strix SCAR 16 (RTX 5080 16GB, 32GB RAM, Ubuntu)

**What it builds on:** Starting point of Anvil. Raft → etcd intuition for Weeks 2–3. Collectives → Week 3 gang scheduling ("why do all workers need to start together?" — because allreduce blocks on the slowest), Week 5 topology-awareness, Phase B RL infrastructure.

---

## Learning Goals

- [ ] Explain Raft: leader election, log replication, safety; terms, quorum, committed entries, split-brain prevention
- [ ] Explain linearizable reads and why follower reads are unsafe without read-index/lease
- [ ] Map Raft → etcd → Kubernetes control plane (what actually goes through consensus in a cluster)
- [ ] Explain ring allreduce: reduce-scatter + all-gather phases, why it's bandwidth-optimal, the 2(N−1)/N·S bytes-per-node result
- [ ] Explain what NCCL provides beyond the algorithm: topology detection, ring/tree selection, NVLink/PCIe/IB hierarchy
- [ ] Explain floating-point non-associativity in reductions and why it matters for reproducibility (and later, for RL determinism)
- [ ] Whiteboard the memory/communication tradeoffs: DDP vs ZeRO-1/2/3 vs FSDP vs tensor/pipeline parallelism

## Implementation Goals

**Part A — Raft essentials (Days 1–3):**
- [ ] Raft node state machine (Follower/Candidate/Leader), randomized election timeouts, AppendEntries + RequestVote over asyncio TCP
- [ ] In-memory KV store on top; client routing to leader
- [ ] 3- and 5-node localhost clusters; kill-leader and partition scripts

**Part B — Collectives from scratch (Days 4–7):**
- [ ] Ring allreduce using only `torch.distributed` point-to-point ops (`send`/`recv` or `isend`/`irecv`), gloo backend, 4 processes on localhost
- [ ] Implemented as explicit reduce-scatter + all-gather phases
- [ ] Correctness check vs `dist.all_reduce` (bit-exact fp32); non-associativity study in bf16
- [ ] Bandwidth benchmark: latency vs tensor size (1KB → 1GB), measured vs the 2(N−1)/N model
- [ ] Written parallelism cheat-sheet with *computed* numbers for a 7B model (memory per GPU and comm volume per step under DDP/ZeRO-1/2/3/FSDP/TP/PP)

## Acceptance Criteria

1. **Election**: a 5-node cluster elects a leader within 3s of startup; killing the leader yields a new leader within 2s and the cluster keeps serving
2. **Replication**: `put(k,v)` acks only after replication to a majority; a restarted follower catches up and converges
3. **Linearizable reads**: `get(k)` returns the most recently committed value; the read path (leader read-index or equivalent) is documented in the README
4. **Partition safety**: isolating the leader forces step-down; the majority partition elects and serves; with 3 of 5 nodes down, writes are rejected (no quorum); after healing, logs converge within 5s
5. **Allreduce correct**: your ring allreduce matches `dist.all_reduce` bit-exact for fp32 across 4 processes on tensors from 1K to 100M elements
6. **Decomposition explicit**: reduce-scatter and all-gather are separate, individually-tested functions; allreduce composes them
7. **Bandwidth model validated**: measured bytes sent per node matches 2(N−1)/N·S within 10%; a latency-vs-size chart (log-log, 1KB→1GB) is committed, with the latency-dominated vs bandwidth-dominated regimes annotated
8. **Non-associativity quantified**: bf16 allreduce max elementwise divergence vs fp64 reference reported for 2/4/8 processes, with a paragraph on why reduction order matters for reproducibility
9. **Cheat-sheet computed**: parallelism table for a 7B model (fp16 weights 14GB, AdamW states, gradients) — per-GPU memory and per-step comm volume for DDP, ZeRO-1/2/3, FSDP, TP, PP — every number derived, not quoted
10. **Tests green**: `pytest tests/ -v` covers election, replication, partition (Part A) and RS/AG/allreduce correctness (Part B)

## Validation Commands

```bash
cd ~/anvil/distributed-foundations

# Part A: cluster up, elect, kill, re-elect
python -m raft.cluster --nodes 5 --base-port 5000 &
sleep 3 && for p in 5000 5001 5002 5003 5004; do curl -s localhost:$p/status | jq -r '.role'; done
curl -X PUT localhost:5000/kv/foo -d '{"value":"bar"}' && curl localhost:5001/kv/foo
kill $(curl -s localhost:5000/status | jq -r '.pid') && sleep 3
python -m tests.partition --isolate-leader --duration 10

# Part B: allreduce correctness + benchmark (4 procs, gloo)
torchrun --nproc_per_node=4 -m collectives.test_correctness
torchrun --nproc_per_node=4 -m collectives.bench --sizes 1KB,64KB,1MB,64MB,1GB --out results/allreduce_bench.json
python -m collectives.report results/allreduce_bench.json   # produces the chart + model comparison

# Everything
pytest tests/ -v
```

## Technical Implementation Details

### Project structure
```
~/anvil/distributed-foundations/
├── raft/                 # Part A (node.py, rpc.py, state_machine.py, cluster.py)
├── collectives/          # Part B
│   ├── ring.py           # reduce_scatter(), all_gather(), ring_allreduce()
│   ├── test_correctness.py
│   ├── bench.py
│   └── report.py
├── docs/
│   ├── raft-notes.md     # incl. linearizable-read design
│   └── parallelism-cheatsheet.md
└── tests/
```

### Part A: Raft (timebox — 3 days, cut where v1 didn't)
Keep v1's core skeleton (state machine + asyncio RPC; see the preserved v1 spec for reference code). Explicit cuts vs v1: **no WAL/persistence** (in-memory; restart-rejoin via snapshot-from-leader is out), **no Jepsen-style checker** (concurrent-client sanity test suffices), no membership changes. If Day 3 ends and partition tests pass, Part A is done — resist gold-plating; the v1 spec remains available if you ever want the deep version.

### Part B: Ring allreduce (the new material)
```python
# collectives/ring.py — the shape of it
def ring_allreduce(tensor: torch.Tensor) -> torch.Tensor:
    rank, world = dist.get_rank(), dist.get_world_size()
    chunks = list(tensor.chunk(world))
    # Phase 1: reduce-scatter — after world-1 steps, chunk[(rank+1) % world] is fully reduced
    for step in range(world - 1):
        send_idx = (rank - step) % world
        recv_idx = (rank - step - 1) % world
        # isend chunks[send_idx] to (rank+1)%world; irecv into buffer from (rank-1)%world
        # chunks[recv_idx] += received
    # Phase 2: all-gather — circulate the reduced chunks world-1 more steps
    ...
```
- Use `dist.isend`/`dist.irecv` with pre-allocated recv buffers; wait both before touching data (deadlock note: post recv before send, or alternate by parity).
- Correctness: seed per-rank tensors deterministically (`torch.manual_seed(rank)`), compare against `dist.all_reduce` on a clone.
- Benchmark: time N repetitions per size (discard warmup); bytes-per-node = 2·(world−1)/world·size; chart measured vs model. On localhost gloo you're measuring loopback/memcpy, not a network — say so in the report and reason about where the model would bend on real NICs (this reasoning is the interview content).
- bf16 study: same experiment in bf16, compare against fp64 ground truth; vary process count to change reduction order.

### The cheat-sheet (Day 7, ~half day)
For a 7B model: weights 14GB fp16; AdamW = params + m + v in fp32 (or master weights) → the famous ~16 bytes/param; gradients 2 bytes/param. Then per strategy: what's replicated, what's sharded, what moves per step (DDP: allreduce grads = 2·2·(N−1)/N·P bytes; ZeRO-3/FSDP: all-gather weights forward+backward + reduce-scatter grads; TP: activations per layer; PP: activations at stage boundaries + bubble fraction). One table, every cell computed in an accompanying script (`docs/parallelism_math.py`) so the numbers are auditable.

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Election cycles endlessly | Widen the randomized timeout range (150–300ms); check that higher-term RPCs force step-down |
| isend/irecv deadlock | Post the irecv before isend on every rank, or split by rank parity; always `wait()` both handles |
| Bit-exact check fails in fp32 | Ensure your manual reduction adds in the same order as ring order every time; compare against `dist.all_reduce` (which is also ring on gloo for large tensors) — if still off, check for in-place aliasing of send buffers |
| gloo slower than expected at large sizes | Expected on loopback; report what you see and note the caveat — the *shape* of the curve is the deliverable |
| Raft eats into Day 4+ | Stop at "partition tests pass"; log remaining polish in the README and move on — Part B is the differentiating half |

## Agent Handoff Template

```
Resume Anvil Phase A, Week 1: Raft essentials + collective communication (v2 spec).
Spec: /home/zzjam/Documents/dev/plan_00/anvil/specs/phaseA/week01-raft-and-collectives.md
Hardware: RTX 5080 16GB, 32GB RAM, Ubuntu. Project root: ~/anvil/distributed-foundations/

Current state: [DESCRIBE]
Part A (Days 1–3): [ ] election [ ] replication [ ] linearizable reads [ ] partition tests
Part B (Days 4–7): [ ] ring allreduce (RS+AG) [ ] bit-exact vs dist.all_reduce [ ] bandwidth bench + chart [ ] bf16 study [ ] parallelism cheat-sheet

Next task: [SPECIFIC NEXT STEP]
Constraints: localhost processes only; gloo backend for Part B (no GPU required for correctness).
```

## Out of Scope

- Raft persistence/WAL, snapshots, membership changes, Jepsen-grade checking (v1 spec preserved if ever wanted)
- NCCL-backend benchmarking on a single GPU (meaningless with one device — concepts are covered in the write-up)
- Actual multi-node networking (Week 5 models topology; real RDMA/IB is narrated, not built)
- Tree allreduce / hierarchical algorithms (mention in the write-up; implement only ring)
- Training a model with your allreduce (Phase B RL block wires collectives to real work)
