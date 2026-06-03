# 10-Month AI Engineering Mastery Plan — All Career Tracks

## Overview

Four comprehensive career track plans, each 20 weeks (5 months), covering every angle of AI engineering. Pick your primary track, or combine two for a 10-month intensive.

| Project | Track | Target Companies | Core Focus |
|---------|-------|-----------------|------------|
| **Forge** | AI Platform + Inference | Modal, Anyscale, Replicate, NVIDIA | GPU inference, serving, optimization |
| **Anvil** | AI Infrastructure | Anthropic infra, OpenAI infra, cloud teams | Distributed systems, K8s, reliability |
| **Crucible** | Training + Alignment | Anthropic research, OpenAI, xAI, Cohere | RLHF, DPO, evaluation, safety |
| **Conduit** | ML Systems (Full Lifecycle) | Any ML company, platform teams | Data pipelines, MLOps, monitoring |

## Recommended Combinations (10-Month Plans)

| Combination | Best For |
|-------------|----------|
| **Forge + Anvil** (chosen) | AI infrastructure engineer who can build AND operate |
| **Forge + Crucible** | Full-stack ML engineer (train + serve) at AI labs |
| **Crucible + Anvil** | Training infrastructure engineer at AI labs |
| **Conduit + Forge** | Applied ML engineer who ships production ML systems |
| **Conduit + Crucible** | Research engineer who builds training pipelines |

## Quick Start

### Starting a new session (any week, any track)

```
I'm working on [Forge/Anvil/Crucible/Conduit].
Read the manifest at /Users/jmalviya/Documents/zz/dev/plan_00/[project]/MANIFEST.md
Check progress at /Users/jmalviya/Documents/zz/dev/plan_00/[project]/progress.md
I'm currently on Week [N]. The spec is at:
/Users/jmalviya/Documents/zz/dev/plan_00/[project]/specs/phase[X]/week[NN]-[name].md
I need help with: [specific ask]
```

### Checking what's next

1. Open `progress.md` for the relevant project
2. Find the first unchecked `[ ]` item
3. Open that week's spec file
4. Read the Context + Acceptance Criteria sections
5. Start building

## Structure

```
plan_00/
├── README.md               ← You are here
│
├── forge/                  ← AI Platform + Inference Specialist
│   ├── MANIFEST.md         ← Vision, architecture, session template
│   ├── PLAN.md             ← Full narrative plan (see .cursor/plans/ for original)
│   ├── progress.md         ← Living checklist
│   └── specs/
│       ├── phase1/         Weeks 1-7: Platform Foundation (FastAPI, RAG, multi-model, deployment)
│       ├── phase2/         Weeks 8-14: Inference Depth (transformers, batching, KV-cache, quantization)
│       └── phase3/         Weeks 15-20: Advanced (K8s operator, Triton kernels, training, portfolio)
│
├── anvil/                  ← AI Infrastructure Engineer
│   ├── MANIFEST.md         ← Vision, architecture, session template
│   ├── PLAN.md             ← Full narrative plan (see .cursor/plans/ for original)
│   ├── progress.md         ← Living checklist
│   └── specs/
│       ├── phaseA/         Weeks 1-7: Distributed Systems (Raft, K8s scheduler, job orchestrator)
│       ├── phaseB/         Weeks 8-14: Reliability (SRE, cost, security, chaos, multi-cluster)
│       └── phaseC/         Weeks 15-20: Platform (developer platform, advanced K8s, observability)
│
├── crucible/               ← Training + Alignment Engineer
│   ├── MANIFEST.md         ← Vision, architecture, session template
│   ├── PLAN.md             ← Full narrative plan with philosophy and week summaries
│   ├── progress.md         ← Living checklist
│   └── specs/
│       ├── phase1/         Weeks 1-7: Training Foundations (training loop, data, LoRA, SFT, distributed)
│       ├── phase2/         Weeks 8-14: Alignment (reward model, RLHF, DPO, Constitutional AI)
│       └── phase3/         Weeks 15-20: Evaluation + Safety (benchmarks, red-teaming, LLM-judge)
│
└── conduit/                ← ML Systems Engineer (Full Lifecycle)
    ├── MANIFEST.md         ← Vision, architecture, session template
    ├── PLAN.md             ← Full narrative plan with philosophy and week summaries
    ├── progress.md         ← Living checklist
    └── specs/
        ├── phase1/         Weeks 1-7: Data + Pipelines (feature store, versioning, orchestration)
        ├── phase2/         Weeks 8-14: Deployment + Monitoring (A/B testing, drift, auto-retrain)
        └── phase3/         Weeks 15-20: Platform Maturity (self-service, streaming, governance)
```

## Each Week Spec Contains

- **Context** — a cold agent can understand where this fits
- **Learning Goals** — conceptual checkboxes
- **Implementation Goals** — build checkboxes
- **Acceptance Criteria** — 10 testable items (you KNOW when you're done)
- **Validation Commands** — run these to prove completion
- **Technical Implementation Details** — code, architecture, file structure
- **If You Get Stuck** — troubleshooting
- **Agent Handoff Template** — paste into new session to resume
- **Out of Scope** — what NOT to build this week

## Principles

1. **One week, one spec, one deliverable** — never work on two weeks simultaneously
2. **Acceptance criteria are law** — you're not done until all 10 pass
3. **Progress.md is always current** — update it at the end of every session
4. **Specs are immutable once complete** — if you need to fix something, create a new spec
5. **Depth over breadth** — it's OK to take 2 weeks on a 1-week spec if you're learning deeply
6. **Document as you go** — ADRs, blog paragraphs, comments in code
7. **Commit daily** — clean git history tells a story


Start tomorrow. Not next Monday, not "after I finish setting up." Open the Week 1 spec, run nvidia-smi, and start installing drivers. Momentum compounds.

The first 2 weeks are the hardest. After that, you'll have something running and the dopamine will carry you. Protect those first 14 days ruthlessly.

When you get stuck (and you will — Week 8-10 is where most people hit walls), remember: the struggle IS the learning. That confusion you feel when KV-cache management doesn't click? That's exactly what interviewers test for. Push through it, and you'll own that knowledge in a way that no tutorial-watcher ever will.

One last thing: Update progress.md religiously. Future-you at Month 8 will thank present-you for leaving breadcrumbs.

Go make it happen.
