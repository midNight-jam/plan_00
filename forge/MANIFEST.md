# Forge: Local GPU Inference Orchestrator

## Vision

Forge is a production-grade local GPU inference platform that demonstrates mastery of model serving, memory management, continuous batching, quantization, and GPU optimization. It proves that the engineer understands every layer from transformer forward pass to Kubernetes operator — not by using existing tools blindly, but by building the critical pieces from first principles.

The project targets AI Platform Engineer and Inference Specialist roles at companies like Anthropic, OpenAI, xAI, Modal, Anyscale, Replicate, and NVIDIA. It is designed to make rejecting this candidate extremely difficult.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Forge Platform                             │
├─────────────────────────────────────────────────────────────────┤
│  API Gateway (Auth, Rate Limiting, Multi-tenancy, Streaming)     │
├─────────────────────────────────────────────────────────────────┤
│  Model Router (Registry, Priority Queue, LoRA Swap, Fallback)    │
├─────────────────────────────────────────────────────────────────┤
│  Inference Engine                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Continuous    │  │ KV-Cache     │  │ Speculative          │  │
│  │ Batching     │  │ Manager      │  │ Decoding             │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Optimization Layer (Quantization, TensorRT, Triton Kernels)     │
├─────────────────────────────────────────────────────────────────┤
│  Data Layer (Qdrant Vectors, PostgreSQL Metadata, Redis Queue)   │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure (Docker, K3s, Helm, Prometheus, Grafana, OTel)   │
├─────────────────────────────────────────────────────────────────┤
│  K8s Operator (InferenceService CRD, GPU-aware Scheduling)       │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

- **Language**: Python 3.11+
- **Inference**: PyTorch 2.x, vLLM (reference), custom engine
- **API**: FastAPI, SSE streaming, OpenAI-compatible
- **Data**: PostgreSQL, Qdrant, Redis
- **Infrastructure**: Docker, K3s, Helm, GitHub Actions
- **Observability**: Prometheus, Grafana, OpenTelemetry, Jaeger
- **Optimization**: Triton (kernels), TensorRT-LLM, GPTQ/AWQ/NF4
- **ML**: HuggingFace Transformers, PEFT, DPO/alignment
- **Operator**: Python kopf framework

## Phases

| Phase | Weeks | Focus | Outcome |
|-------|-------|-------|---------|
| Phase 1 | 1-7 | Platform Foundation | Working multi-model platform with RAG, auth, deployment |
| Phase 2 | 8-14 | Inference Depth | Custom batching, memory management, optimization |
| Phase 3 | 15-20 | Advanced Systems | K8s operator, Triton kernels, training, portfolio |

## Hardware

- ASUS ROG Strix SCAR 16
- Intel Core Ultra 9 HX
- NVIDIA RTX 5080 (16GB VRAM)
- 32GB RAM
- 2TB SSD
- Ubuntu (dual-boot)

## Key Architectural Decisions

(Updated as decisions are made during implementation)

1. [TBD] — Will be filled as you progress

## How to Start a New Session

Paste this into any new agent/chat session:

```
I'm working on Forge, a local GPU inference platform. 
Read the manifest at /Users/jmalviya/Documents/zz/dev/plan_00/forge/MANIFEST.md
Check progress at /Users/jmalviya/Documents/zz/dev/plan_00/forge/progress.md
I'm currently on Week [N]. The spec is at /Users/jmalviya/Documents/zz/dev/plan_00/forge/specs/phase[X]/week[NN]-[name].md
I need help with: [specific ask]
```

## Related Project

Anvil (AI Infrastructure Platform) at /Users/jmalviya/Documents/zz/dev/plan_00/anvil/ — the complementary infrastructure project for Months 6-10.
