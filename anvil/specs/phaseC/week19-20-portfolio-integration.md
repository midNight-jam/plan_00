# Weeks 19-20: Portfolio Integration

## Context

**Where it fits:** Phase C, Weeks 19-20 of the Anvil AI Infrastructure project. This is the capstone — integrating Anvil (infrastructure) with Forge (inference engine), creating polished portfolio artifacts, and preparing for open-source release.

**Prerequisites:**
- All Anvil phases complete: infrastructure (A), advanced systems (B), platform maturity (C Weeks 15-18)
- Forge inference engine operational with Raft consensus, model serving, routing
- Working end-to-end flow: model training → checkpoint → deployment → inference
- Benchmark data and observability dashboards populated
- All components tested and documented

**What it builds on:** The entire Anvil + Forge ecosystem. This phase synthesizes everything into a unified architecture document, compelling blog posts, a demo video, and open-source release — transforming a learning project into a portfolio piece that demonstrates systems engineering depth.

---

## Learning Goals

- [ ] Understand technical writing for engineering blogs: narrative structure, code examples, lessons learned
- [ ] Learn open-source release best practices: licensing, CONTRIBUTING.md, semantic versioning, changelog
- [ ] Study demo video creation: scripting, screen recording, narration pacing
- [ ] Understand architecture documentation: C4 model, decision records, component diagrams
- [ ] Learn community engagement: awesome-lists, Hacker News, Reddit posting strategies
- [ ] Study portfolio presentation: how to frame infrastructure work for hiring managers

---

## Implementation Goals

- [ ] Write unified Forge + Anvil architecture document with C4 diagrams
- [ ] Create integration layer showing how Forge inference connects to Anvil infrastructure
- [ ] Write blog post 1: "Implementing Raft: What I Learned About Distributed Consensus"
- [ ] Write blog post 2: "Training Job Orchestration: Making GPU Jobs Reliable"
- [ ] Write blog post 3: "Chaos Engineering for GPU Infrastructure"
- [ ] Write blog post 4: "Building an Internal AI Platform: From kubectl to Self-Service"
- [ ] Record combined demo video (5-7 minutes): full lifecycle from training to serving
- [ ] Prepare repos for open-source: LICENSE, CONTRIBUTING.md, semantic versioning, CI badges
- [ ] Submit to awesome-kubernetes, awesome-machine-learning, relevant awesome-lists
- [ ] Create HN/Reddit launch posts with compelling narratives

---

## Acceptance Criteria

1. Architecture document clearly shows how a model goes from code → training job → checkpoint → inference endpoint, with component diagram showing Forge ↔ Anvil boundaries.
2. Blog post 1 (Raft) includes: motivation, implementation journey, at least 2 code snippets, performance numbers, and a "what I'd do differently" section. 1500-2500 words.
3. Blog post 2 (Orchestration) covers: scheduler design, fault tolerance, checkpoint recovery with real metrics from benchmarks. 1500-2500 words.
4. Blog post 3 (Chaos Engineering) includes: failure injection methodology, actual failure scenarios tested, recovery times measured, lessons learned. 1500-2500 words.
5. Blog post 4 (Platform) shows: before/after UX comparison (raw YAML vs CLI), design decisions, user feedback (simulated), metrics on adoption. 1500-2500 words.
6. Demo video is 5-7 minutes, shows: `anvil train submit` → training running → checkpoint → `anvil model deploy` → inference request → monitoring dashboard — all in real-time or light editing.
7. Both repos have: Apache-2.0 LICENSE, CONTRIBUTING.md with setup instructions, semantic version tags (v1.0.0), GitHub Actions CI, README with badges.
8. CONTRIBUTING.md enables a new developer to clone, build, and run tests within 30 minutes.
9. At least 3 awesome-list PRs submitted with correct formatting and category placement.
10. HN/Reddit posts have compelling titles, concise descriptions, and link to live demos or blog posts.

---

## Validation Commands

```bash
# Verify architecture document renders correctly
cd ~/anvil/docs && mkdocs serve &
open http://localhost:8000/architecture

# Build both repos cleanly
cd ~/anvil && make build test
cd ~/forge && make build test

# Verify integration: training to inference end-to-end
anvil train submit --model ./examples/sentiment --gpu 1 --name integration-test
anvil train wait --name integration-test --timeout 10m
CHECKPOINT=$(anvil train get --name integration-test -o json | jq -r '.status.latestCheckpoint')
anvil model deploy --checkpoint "$CHECKPOINT" --name sentiment-serve --replicas 1
sleep 30
curl -X POST http://localhost:8080/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This integration works perfectly!"}'

# Validate blog posts (word count, structure)
for post in ~/anvil/blog/*.md; do
  echo "$post: $(wc -w < "$post") words"
  grep -c "^##" "$post" | xargs -I{} echo "  sections: {}"
done

# Check open-source readiness
cd ~/anvil
test -f LICENSE && echo "LICENSE: OK" || echo "LICENSE: MISSING"
test -f CONTRIBUTING.md && echo "CONTRIBUTING: OK" || echo "CONTRIBUTING: MISSING"
git tag -l 'v*' | head -5
grep -l "badge" README.md && echo "Badges: OK"

# Verify CI passes
gh workflow run ci.yml && gh run watch

# Demo video script check
cat ~/anvil/docs/demo-script.md | wc -l

# Semantic versioning
cd ~/anvil && git tag v1.0.0 && git tag -v v1.0.0
```

---

## Technical Implementation Details

### Project Structure

```
~/anvil/
├── docs/
│   ├── architecture/
│   │   ├── overview.md            # C4 context + container diagrams
│   │   ├── forge-integration.md   # Forge ↔ Anvil boundary
│   │   ├── decisions/             # Architecture Decision Records
│   │   │   ├── 001-scheduler-design.md
│   │   │   ├── 002-checkpoint-strategy.md
│   │   │   └── 003-federation-protocol.md
│   │   └── diagrams/
│   │       ├── c4-context.mmd     # Mermaid C4 context
│   │       ├── c4-container.mmd   # Mermaid C4 container
│   │       └── sequence-training.mmd
│   ├── demo-script.md             # Video recording script
│   └── mkdocs.yml
├── blog/
│   ├── 01-implementing-raft.md
│   ├── 02-training-orchestration.md
│   ├── 03-chaos-engineering-gpu.md
│   └── 04-internal-ai-platform.md
├── CONTRIBUTING.md
├── LICENSE                         # Apache-2.0
├── CHANGELOG.md
├── README.md                       # With badges, quick start, architecture diagram
└── .github/
    ├── workflows/
    │   ├── ci.yml
    │   └── release.yml
    ├── ISSUE_TEMPLATE/
    │   ├── bug_report.md
    │   └── feature_request.md
    └── PULL_REQUEST_TEMPLATE.md
```

### Architecture Document Structure

```markdown
<!-- docs/architecture/overview.md -->
# Anvil + Forge: AI Infrastructure Architecture

## System Context (C4 Level 1)
- AI Engineers interact with Anvil CLI
- Anvil manages K8s cluster with GPU resources
- Forge handles inference routing and Raft consensus
- External: model registry, object storage, monitoring

## Container Diagram (C4 Level 2)
- Anvil CLI → Platform API → K8s API
- Scheduler Controller → GPU Allocator → Node Affinity
- Checkpoint Manager → Object Storage (S3/GCS)
- Forge Router → Raft Cluster → Model Workers
- Federation Controller → Remote Clusters

## Key Flows
1. Training: submit → schedule → allocate GPU → train → checkpoint → complete
2. Deployment: checkpoint → load model → health check → register endpoint
3. Inference: request → route → predict → respond (with Raft-based leader election)
4. Recovery: failure detected → checkpoint loaded → job resumed on healthy node
```

### Blog Post Template

```markdown
<!-- blog/01-implementing-raft.md -->
# Implementing Raft: What I Learned About Distributed Consensus

## Why Raft?
[Motivation: inference service needs leader election for routing decisions.
Compare to alternatives: ZooKeeper (operational overhead), etcd (over-engineered for this use case)]

## The Implementation Journey

### Phase 1: Leader Election
[Code snippet showing RequestVote RPC implementation]
[Diagram: election timeout → candidate → collect votes → leader]

### Phase 2: Log Replication
[Code snippet showing AppendEntries with conflict resolution]
[Metrics: replication latency P50/P99]

### Phase 3: The Hard Parts
- Split brain during network partition (how we handled it)
- Log compaction and snapshotting
- Membership changes (joint consensus)

## Performance Numbers
- Election converges in < 500ms (measured across 5-node cluster)
- Log replication: 10k entries/sec sustained throughput
- Recovery from leader failure: < 2 seconds to new leader

## What I'd Do Differently
1. Start with a formal TLA+ spec before coding
2. Use deterministic simulation testing earlier
3. Build the membership change protocol from day 1

## Key Takeaways
[3-4 bullet points summarizing the most valuable lessons]
```

### Demo Video Script

```markdown
<!-- docs/demo-script.md -->
# Anvil Demo Video Script (5-7 minutes)

## Opening (30s)
- Show terminal with `anvil version`
- "Anvil is an AI infrastructure platform that makes GPU training and serving self-service"

## Act 1: Training (2 min)
- Show model code briefly (simple PyTorch)
- `anvil train submit --model ./sentiment --gpu 2 --name demo`
- Show scheduler assigning GPUs in real-time
- Dashboard: GPU utilization climbing
- `anvil train status --name demo` showing progress

## Act 2: Resilience (1.5 min)
- Inject GPU failure: `anvil chaos inject gpu-error --node worker-1`
- Show health monitor detecting (events stream)
- Checkpoint auto-saves, pod migrates
- Training resumes on healthy node
- "Zero data loss, zero manual intervention"

## Act 3: Deployment (1.5 min)
- Training completes, checkpoint saved
- `anvil model deploy --checkpoint <path> --name sentiment-api --replicas 2`
- Show Raft leader election in Forge
- `curl /v1/predict` with sample request
- Dashboard: inference latency, request rate

## Act 4: Platform View (1 min)
- `anvil cluster status` — fleet overview
- `anvil cost report` — GPU hours by team
- Executive dashboard in Grafana
- "From kubectl expert to self-service in one CLI"

## Closing (30s)
- GitHub links, blog posts
- Architecture diagram flash
- "Built to learn distributed systems. Designed to be production-ready."
```

### CONTRIBUTING.md Structure

```markdown
<!-- CONTRIBUTING.md -->
# Contributing to Anvil

## Quick Start (< 30 minutes)

### Prerequisites
- Go 1.21+
- Docker
- kind (Kubernetes in Docker)
- kubectl

### Setup
git clone https://github.com/<user>/anvil.git
cd anvil
make setup          # Installs tools, creates kind cluster
make build          # Builds all components
make test           # Runs unit tests
make test-e2e       # Runs e2e tests (requires kind cluster)

## Development Workflow
1. Fork and clone
2. Create feature branch: `git checkout -b feat/my-feature`
3. Make changes with tests
4. Run `make lint test`
5. Commit with conventional commits: `feat:`, `fix:`, `docs:`
6. Push and create PR

## Architecture Overview
[Brief component map pointing to docs/architecture/]

## Code Style
- Go: follow `gofumpt`, run `make lint`
- Python: `ruff` for formatting, `mypy` for types
- YAML: 2-space indent, `yamllint`

## Testing
- Unit tests: `make test` (must pass, >80% coverage for new code)
- Integration tests: `make test-integration` (requires kind)
- E2E tests: `make test-e2e` (full cluster simulation)

## Release Process
- Semantic versioning (semver.org)
- CHANGELOG.md updated with each PR
- Tags trigger release workflow
```

### Open-Source CI Workflow

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with: { go-version: '1.21' }
      - run: make lint

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with: { go-version: '1.21' }
      - run: make test
      - uses: codecov/codecov-action@v3

  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with: { go-version: '1.21' }
      - uses: helm/kind-action@v1
      - run: make test-e2e

  benchmark:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with: { go-version: '1.21' }
      - uses: helm/kind-action@v1
      - run: make ci-perf-gate
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Architecture diagram too complex | Use C4 model layers: start with context, zoom into containers only for key flows |
| Blog post too long/unfocused | Each post has ONE thesis. Cut anything that doesn't support it. Target 2000 words. |
| Demo video has dead time | Pre-run commands once to know timing. Use `asciinema` for terminal recording, edit pauses |
| CI badges not showing | Check badge URLs use the correct repo name and branch (main vs master) |
| Awesome-list PR rejected | Read their CONTRIBUTING.md carefully; ensure alphabetical order and correct format |
| HN post gets no traction | Post Tuesday-Thursday 9-11am ET. Title should be specific, not generic. |
| Integration test flaky | Add retry logic and increase timeouts for E2E; use `Eventually` pattern |

---

## Agent Handoff Template

```
Resume Anvil Phase C, Weeks 19-20: Portfolio Integration.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
State: ALL Anvil phases complete (A, B, C Weeks 15-18). Forge inference engine operational. Full platform running.

Current goal: Create portfolio artifacts — architecture doc, 4 blog posts, demo video script, open-source release preparation.
Key files: ~/anvil/docs/architecture/, ~/anvil/blog/, ~/anvil/CONTRIBUTING.md, ~/anvil/.github/
Test with: `make build test` for code health, `mkdocs serve` for docs.

Specific task: [DESCRIBE WHAT TO DO NEXT]
Constraints: Blog posts must be 1500-2500 words each with code examples. Architecture doc must use C4 model. CONTRIBUTING.md must enable 30-minute onboarding.
```

---

## Out of Scope

- Actually deploying to a public cloud (local/kind cluster is sufficient for demos)
- Paid hosting for blog posts (use GitHub Pages or dev.to)
- Video editing beyond basic cuts (no animations, motion graphics)
- Marketing strategy beyond initial launch posts
- Community management after launch (responding to issues, PRs)
- Conference talk preparation (separate effort)
- Monetization or SaaS wrapper
- Mobile-responsive documentation (desktop-first is fine)
