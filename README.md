# 10-Month AI Infrastructure Mastery Plan

## Overview

Two complementary projects executed over 10 months to build an undeniable profile for top AI infrastructure roles (Anthropic, OpenAI, xAI, Modal, Anyscale, NVIDIA, etc.).

| Project | Months | Focus | Proves |
|---------|--------|-------|--------|
| **Forge** | 1-5 | GPU Inference Platform | "I can build fast AI systems" |
| **Anvil** | 6-10 | AI Infrastructure Platform | "I can run them reliably at scale" |

## Quick Start

### Starting a new session (any week)

```
I'm working on [Forge/Anvil], an AI [inference/infrastructure] platform.
Read the manifest at /Users/jmalviya/Documents/zz/dev/plan_00/[forge/anvil]/MANIFEST.md
Check progress at /Users/jmalviya/Documents/zz/dev/plan_00/[forge/anvil]/progress.md
I'm currently on Week [N]. The spec is at:
/Users/jmalviya/Documents/zz/dev/plan_00/[forge/anvil]/specs/phase[X]/week[NN]-[name].md
I need help with: [specific ask]
```

### Checking what's next

1. Open `progress.md` for the relevant project
2. Find the first unchecked `[ ]` item
3. Open that week's spec file
4. Read the Context + Acceptance Criteria sections
5. Start building

### After completing a session

1. Update `progress.md` — check off completed items, note partial progress
2. Update `MANIFEST.md` if any architectural decisions were made
3. Commit your code with a clear message

## Structure

```
plan_00/
├── README.md              ← You are here
├── forge/                 ← Months 1-5: GPU Inference Platform
│   ├── MANIFEST.md        ← Project vision, architecture, how to resume
│   ├── progress.md        ← Living checklist of what's done/next
│   └── specs/
│       ├── phase1/        ← Weeks 1-7: Platform Foundation
│       ├── phase2/        ← Weeks 8-14: Inference Depth
│       └── phase3/        ← Weeks 15-20: Advanced Systems
└── anvil/                 ← Months 6-10: AI Infrastructure Platform
    ├── MANIFEST.md
    ├── progress.md
    └── specs/
        ├── phaseA/        ← Weeks 1-7: Distributed Systems + Orchestration
        ├── phaseB/        ← Weeks 8-14: Reliability, Cost, Security
        └── phaseC/        ← Weeks 15-20: Advanced Platform Engineering

Each week spec contains:
- Context (what it builds on, prerequisites)
- Learning Goals (conceptual checkboxes)
- Implementation Goals (build checkboxes)
- Acceptance Criteria (10 testable items — you KNOW when you're done)
- Validation Commands (run these to verify)
- Technical Implementation Details (code, architecture, file structure)
- If You Get Stuck (troubleshooting)
- Agent Handoff Template (paste into new session)
- Out of Scope (what NOT to build)
```

## Principles

1. **One week, one spec, one deliverable** — never work on two weeks simultaneously
2. **Acceptance criteria are law** — you're not done until all 10 pass
3. **Progress.md is always current** — update it at the end of every session
4. **Specs are immutable once complete** — if you need to fix something, create a new spec
5. **Depth over breadth** — it's OK to take 2 weeks on a 1-week spec if you're learning deeply
6. **Document as you go** — ADRs, blog paragraphs, comments in code
7. **Commit daily** — clean git history tells a story

## Key Dates (Flexible)

| Milestone | Target | What you have |
|-----------|--------|--------------|
| Month 1 end | ~Week 4 | Working inference platform on GitHub |
| Month 3 end | ~Week 10 | Custom inference engine with deep understanding |
| Month 5 end | ~Week 20 | Full Forge complete, portfolio-ready |
| Month 7 end | ~Anvil Week 7 | Distributed systems + orchestration working |
| Month 10 end | ~Anvil Week 20 | Both projects complete, blog posts published |

## The Combined Resume Line

> "Built Forge + Anvil: open-source AI infrastructure comprising a custom GPU inference engine with continuous batching and KV-cache management, alongside a distributed training orchestrator with Raft-based consensus, checkpoint recovery, multi-cluster federation, and Kubernetes operators for GPU workload scheduling. Published 8 technical blog posts covering systems design from CUDA kernels to cluster reliability."
