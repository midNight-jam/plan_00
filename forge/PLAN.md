# Forge: Full Narrative Plan

The detailed narrative plan for Forge lives in:
`/home/zzjam/Documents/dev/plan_00/original_artifacts/forge_practical_build_plan_615ca283.plan.md`

It contains the full week-by-week philosophy, deliverables, daily routine, resume lines, and phase gates.

For quick reference, see MANIFEST.md in this directory for architecture and session start templates.
For week-by-week execution, see the specs/ directory.

## Quick Summary

**Philosophy**: "Make It Work, Make It Right, Make It Fast"

- **Phase 1 (Weeks 1-7)**: Build a production-quality AI platform (FastAPI, RAG, multi-model, auth, deployment)
- **Phase 2 (Weeks 8-14)**: Go deep on inference (transformer internals, continuous batching, KV-cache, quantization, observability)
- **Phase 3 (Weeks 15-20)**: Advanced systems (K8s operator, Triton kernels, training/alignment, portfolio polish)

## Resume Line

> "Built Forge, an open-source GPU inference platform featuring multi-model orchestration with LoRA hot-swap, custom continuous batching scheduler, block-based KV-cache management, and a Kubernetes operator for GPU workload scheduling. Achieved 3.8x throughput over naive inference with comprehensive quantization pipeline and observability stack."

## Daily Routine

- **1.5 hours**: Learn (watch talks, study vLLM source, read blog posts)
- **4-5 hours**: Build (code, experiments, debug)
- **30 min**: Document (ADRs, blog paragraphs, clean commits)
- **30 min**: Push (commit, update README)
