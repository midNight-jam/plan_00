# Crucible: Training and Alignment Engineering

## Vision

Crucible is a comprehensive training and alignment platform that demonstrates mastery of the full model training lifecycle — from pre-training data pipelines to RLHF/DPO alignment, evaluation, and safety. It proves that the engineer understands how models are MADE, not just how they're served.

This is the track that targets roles at Anthropic, OpenAI, xAI, Cohere, and DeepMind — the companies that train frontier models. Their interview loops test deep understanding of training dynamics, alignment methods, evaluation rigor, and the infrastructure that makes these work at scale.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Crucible Platform                             │
├─────────────────────────────────────────────────────────────────┤
│  Evaluation Layer                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Benchmark    │  │ Safety       │  │ Human Preference     │  │
│  │ Harness      │  │ Evaluation   │  │ Simulation           │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Alignment Layer                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ RLHF (PPO)  │  │ DPO / IPO   │  │ Constitutional AI    │  │
│  │ Pipeline     │  │ Training     │  │ (Self-Critique)      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Training Layer                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Custom       │  │ Distributed  │  │ Data Pipeline        │  │
│  │ Training Loop│  │ Training Sim │  │ + Curation           │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Model Layer                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Transformer  │  │ LoRA / QLoRA │  │ Reward Model         │  │
│  │ From Scratch │  │ Fine-tuning  │  │ Training             │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Mixed        │  │ Gradient     │  │ Checkpointing        │  │
│  │ Precision    │  │ Checkpointing│  │ + Experiment Track   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

- **Language**: Python 3.11+
- **Framework**: PyTorch 2.x (raw, not HuggingFace Trainer for core work)
- **Models**: HuggingFace Transformers, PEFT (LoRA/QLoRA)
- **Alignment**: Custom RLHF (PPO), DPO, IPO, KTO implementations
- **Data**: Datasets library, custom data pipelines, preference data curation
- **Evaluation**: lm-evaluation-harness, custom benchmarks, LLM-as-judge
- **Experiment Tracking**: Weights & Biases (or MLflow)
- **Distributed**: torch.distributed, FSDP concepts, DeepSpeed concepts
- **Safety**: Custom toxicity detection, jailbreak testing, refusal evaluation

## Phases

| Phase | Weeks | Focus | Outcome |
|-------|-------|-------|---------|
| Phase 1 | 1-7 | Training Foundations | Custom training loop, fine-tuning, data pipelines, basic alignment |
| Phase 2 | 8-14 | Alignment Deep Dive | Full RLHF, DPO, reward modeling, Constitutional AI, multi-method comparison |
| Phase 3 | 15-20 | Evaluation + Safety + Portfolio | Comprehensive eval, safety testing, distributed concepts, blog posts |

## Hardware

- ASUS ROG Strix SCAR 16
- NVIDIA RTX 5080 (16GB VRAM) — sufficient for training 1-3B models, fine-tuning 7B with QLoRA
- 32GB RAM
- 2TB SSD
- Ubuntu

## Key Insight

On 16GB VRAM, you can:
- Train models up to 1-3B parameters (full fine-tune)
- Fine-tune models up to 7B with LoRA/QLoRA (4-bit base + LoRA adapters)
- Train reward models up to 3B
- Run DPO on 7B models with gradient checkpointing + QLoRA

You CANNOT pre-train a 7B model from scratch. But you can demonstrate all the CONCEPTS and INFRASTRUCTURE needed for it. The understanding transfers directly.

## How to Start a New Session

```
I'm working on Crucible, a training and alignment platform.
Read the manifest at /Users/jmalviya/Documents/zz/dev/plan_00/crucible/MANIFEST.md
Check progress at /Users/jmalviya/Documents/zz/dev/plan_00/crucible/progress.md
I'm currently on Week [N]. The spec is at:
/Users/jmalviya/Documents/zz/dev/plan_00/crucible/specs/phase[X]/week[NN]-[name].md
I need help with: [specific ask]
```

## Target Companies

- **Anthropic**: Constitutional AI, RLHF at scale, safety evaluation
- **OpenAI**: Training infrastructure, alignment research, evaluation
- **xAI**: Training efficiency, novel alignment approaches
- **Cohere**: Fine-tuning pipelines, instruction tuning, evaluation
- **DeepMind**: Training dynamics, reward modeling, safety

## Related Projects

- Forge (Inference) at /Users/jmalviya/Documents/zz/dev/plan_00/forge/
- Anvil (Infrastructure) at /Users/jmalviya/Documents/zz/dev/plan_00/anvil/
- Conduit (ML Systems) at /Users/jmalviya/Documents/zz/dev/plan_00/conduit/
