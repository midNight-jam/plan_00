# Week 1: Distributed Systems Fundamentals

## Context

**Where it fits:** Phase A, Week 1 — the foundational building block for everything that follows. Every component in Anvil (scheduler, job orchestrator, storage) depends on consensus, leader election, and fault tolerance.

**Prerequisites:**
- Python 3.11+ installed
- Familiarity with TCP/UDP sockets and asyncio basics
- Understanding of state machines
- Hardware: ASUS ROG Strix SCAR 16 (RTX 5080 16GB, 32GB RAM, 2TB SSD, Ubuntu)

**What it builds on:** This is the starting point. Concepts here (leader election, log replication, split-brain prevention) directly apply to Week 3 (job orchestrator fault tolerance) and Week 5 (network partition handling).

---

## Learning Goals

- [ ] Explain Raft consensus algorithm phases: leader election, log replication, safety
- [ ] Define terms: term number, committed entry, log index, quorum
- [ ] Articulate why consensus is needed vs. eventual consistency
- [ ] Describe split-brain scenarios and how Raft prevents them
- [ ] Compare Raft to Paxos and explain Raft's design tradeoffs
- [ ] Explain linearizable reads and how they're achieved
- [ ] Understand membership changes (joint consensus)

---

## Implementation Goals

- [ ] Implement Raft node state machine (Follower, Candidate, Leader)
- [ ] Implement leader election with randomized timeouts
- [ ] Implement AppendEntries RPC for log replication
- [ ] Implement RequestVote RPC with term comparison
- [ ] Build distributed KV store on top of Raft (put/get/delete)
- [ ] Client request routing to leader
- [ ] Run 3-node and 5-node clusters as separate processes on localhost
- [ ] Failure scenario: kill leader, observe re-election within 2 seconds
- [ ] Failure scenario: simulate network partition, verify split-brain prevention
- [ ] Persistent state (currentTerm, votedFor, log) to disk with WAL

---

## Acceptance Criteria

1. A 5-node Raft cluster elects a leader within 3 seconds of startup with no human intervention.
2. After killing the leader process, a new leader is elected within 2 seconds and the cluster continues serving requests.
3. A `put(key, value)` operation committed on the leader is replicated to a majority before returning success to the client.
4. A `get(key)` operation returns the most recently committed value (linearizable read).
5. When 2 of 5 nodes are killed, the cluster remains available (quorum of 3 maintained).
6. When 3 of 5 nodes are killed, the cluster stops accepting writes (no quorum).
7. A network partition isolating the leader causes it to step down and a new leader elected in the majority partition.
8. After healing a partition, the minority nodes catch up their logs from the new leader within 5 seconds.
9. Persistent state survives process restart — a restarted node rejoins with its previous term and log intact.
10. The KV store passes a linearizability checker (Jepsen-style) under concurrent client operations.

---

## Validation Commands

```bash
# Start a 5-node cluster
cd ~/anvil/distributed-systems
python -m raft.cluster --nodes 5 --base-port 5000 &

# Verify leader elected
sleep 3
curl http://localhost:5000/status | jq '.role' # Should show "leader" on one node

# Write and read
curl -X PUT http://localhost:5000/kv/foo -d '{"value": "bar"}'
curl http://localhost:5001/kv/foo  # Should return "bar" (redirects to leader)

# Kill leader, verify re-election
LEADER_PID=$(curl -s http://localhost:5000/status | jq -r '.pid')
kill $LEADER_PID
sleep 3
# Check remaining nodes for new leader
for port in 5001 5002 5003 5004; do
  echo "Port $port: $(curl -s http://localhost:$port/status | jq -r '.role')"
done

# Run linearizability test
python -m tests.linearizability --nodes 5 --duration 30 --clients 10

# Partition test
python -m tests.partition --isolate 5000,5001 --duration 10
```

---

## Technical Implementation Details

### Project Structure

```
~/anvil/distributed-systems/
├── raft/
│   ├── __init__.py
│   ├── node.py          # Core Raft state machine
│   ├── rpc.py           # AsyncIO RPC layer
│   ├── log.py           # Replicated log with WAL
│   ├── state_machine.py # KV store state machine
│   ├── cluster.py       # Process launcher
│   └── client.py        # Client library with leader discovery
├── tests/
│   ├── test_election.py
│   ├── test_replication.py
│   ├── test_partition.py
│   └── linearizability.py
├── pyproject.toml
└── README.md
```

### Core Raft Node Implementation

```python
# raft/node.py
import asyncio
import random
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

class Role(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"

@dataclass
class LogEntry:
    term: int
    index: int
    command: dict  # {"op": "put", "key": "x", "value": "1"}

@dataclass
class RaftNode:
    node_id: int
    peers: list[int]
    role: Role = Role.FOLLOWER
    current_term: int = 0
    voted_for: Optional[int] = None
    log: list[LogEntry] = field(default_factory=list)
    commit_index: int = 0
    last_applied: int = 0
    # Leader state
    next_index: dict[int, int] = field(default_factory=dict)
    match_index: dict[int, int] = field(default_factory=dict)

    ELECTION_TIMEOUT_MIN = 150  # ms
    ELECTION_TIMEOUT_MAX = 300  # ms
    HEARTBEAT_INTERVAL = 50     # ms

    async def run(self):
        """Main event loop."""
        while True:
            if self.role == Role.FOLLOWER:
                await self._run_follower()
            elif self.role == Role.CANDIDATE:
                await self._run_candidate()
            elif self.role == Role.LEADER:
                await self._run_leader()

    async def _run_follower(self):
        timeout = random.randint(
            self.ELECTION_TIMEOUT_MIN, self.ELECTION_TIMEOUT_MAX
        ) / 1000.0
        try:
            await asyncio.wait_for(self._wait_for_heartbeat(), timeout)
        except asyncio.TimeoutError:
            self.role = Role.CANDIDATE

    async def _run_candidate(self):
        self.current_term += 1
        self.voted_for = self.node_id
        votes = 1  # Vote for self
        # Request votes from all peers concurrently
        results = await asyncio.gather(
            *[self._request_vote(peer) for peer in self.peers],
            return_exceptions=True
        )
        votes += sum(1 for r in results if r is True)
        if votes > (len(self.peers) + 1) // 2:
            self._become_leader()
        else:
            self.role = Role.FOLLOWER

    def _become_leader(self):
        self.role = Role.LEADER
        last_log_index = len(self.log)
        for peer in self.peers:
            self.next_index[peer] = last_log_index + 1
            self.match_index[peer] = 0
```

### RPC Layer with asyncio

```python
# raft/rpc.py
import asyncio
import json

class RaftRPC:
    def __init__(self, node: "RaftNode", host: str, port: int):
        self.node = node
        self.host = host
        self.port = port

    async def start_server(self):
        server = await asyncio.start_server(
            self._handle_connection, self.host, self.port
        )
        async with server:
            await server.serve_forever()

    async def _handle_connection(self, reader, writer):
        data = await reader.readline()
        message = json.loads(data.decode())
        response = await self._dispatch(message)
        writer.write(json.dumps(response).encode() + b"\n")
        await writer.drain()
        writer.close()

    async def send_rpc(self, peer_port: int, message: dict) -> dict:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", peer_port),
                timeout=0.5
            )
            writer.write(json.dumps(message).encode() + b"\n")
            await writer.drain()
            data = await asyncio.wait_for(reader.readline(), timeout=0.5)
            writer.close()
            return json.loads(data.decode())
        except (asyncio.TimeoutError, ConnectionRefusedError):
            return {"success": False, "reason": "unreachable"}
```

### KV State Machine

```python
# raft/state_machine.py
class KVStateMachine:
    def __init__(self):
        self.store: dict[str, str] = {}

    def apply(self, command: dict) -> dict:
        op = command["op"]
        if op == "put":
            self.store[command["key"]] = command["value"]
            return {"status": "ok"}
        elif op == "get":
            value = self.store.get(command["key"])
            return {"status": "ok", "value": value}
        elif op == "delete":
            self.store.pop(command["key"], None)
            return {"status": "ok"}
        return {"status": "error", "reason": "unknown op"}
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Election keeps cycling (no stable leader) | Check randomized timeout range — min/max too close causes contention. Use 150-300ms range. |
| Log replication never converges | Verify `nextIndex` decrements on AppendEntries rejection. Check term comparison logic. |
| Split-brain after partition heal | Ensure nodes with stale terms step down when they receive a higher term in any RPC. |
| Processes can't connect on localhost | Verify ports aren't in use: `lsof -i :5000-5004`. Kill stale processes. |
| asyncio deadlock | Never `await` inside a lock. Use `asyncio.Queue` for cross-coroutine communication. |
| WAL corruption on crash | Use `fsync()` after each write. Consider using `aiofiles` with flush. |

---

## Agent Handoff Template

```
Resume Anvil Phase A, Week 1: Distributed Systems Fundamentals.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
Project root: ~/anvil/distributed-systems/

Current state: [DESCRIBE - e.g., "Leader election works for 3 nodes, but 5-node cluster has term cycling"]

What's done:
- [x/blank] Raft node state machine (Follower/Candidate/Leader)
- [x/blank] Leader election with randomized timeouts
- [x/blank] Log replication (AppendEntries)
- [x/blank] KV store on top of Raft
- [x/blank] Partition tolerance tests
- [x/blank] Linearizability checker

Next task: [SPECIFIC NEXT STEP]

Key files:
- raft/node.py — core state machine
- raft/rpc.py — asyncio networking
- raft/log.py — replicated log
- tests/ — test suite

Constraints: All nodes run as localhost processes. No external dependencies beyond Python stdlib + aiofiles.
```

---

## Out of Scope

- Production-grade Raft libraries (etcd/raft, hashicorp/raft) — we're building from scratch to learn
- Multi-Raft (multiple Raft groups) — that's Phase B
- Snapshot/compaction — keep full log for this week
- TLS between nodes — plain TCP is fine for localhost
- Formal verification (TLA+) — understanding is enough
- Read replicas / follower reads — only linearizable reads through leader
- Dynamic membership changes — fixed cluster size for now
