# Conduit: ML Systems Engineering

## Vision

Conduit is a full-lifecycle ML platform that demonstrates mastery of the glue between research and production — data pipelines, feature engineering, experiment tracking, model training orchestration, deployment automation, monitoring, and feedback loops. It proves that the engineer can take a model from "works on my laptop" to "serving 10M users reliably."

This is the broadest track — applicable to virtually any AI company. ML Systems Engineers are the people who make the entire machine work. They understand enough ML to talk to researchers, enough infrastructure to talk to DevOps, and enough data engineering to talk to data teams.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Conduit Platform                             │
├─────────────────────────────────────────────────────────────────┤
│  Feedback Loop (Production → Data → Retraining → Deployment)     │
├─────────────────────────────────────────────────────────────────┤
│  Deployment Layer                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Model        │  │ A/B Testing  │  │ Canary + Progressive │  │
│  │ Serving      │  │ Framework    │  │ Rollout              │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Training Layer                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Experiment   │  │ Hyperparameter│  │ Training             │  │
│  │ Tracking     │  │ Optimization │  │ Orchestration        │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Data Layer                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Feature      │  │ Data         │  │ Data Quality         │  │
│  │ Store        │  │ Versioning   │  │ + Validation         │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Monitoring Layer                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Model        │  │ Data Drift   │  │ Performance          │  │
│  │ Monitoring   │  │ Detection    │  │ Degradation          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Pipeline Orchestration (DAGs, Scheduling, Dependencies)         │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure (Docker, K8s, Object Storage, Message Queues)    │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

- **Language**: Python 3.11+
- **Pipeline Orchestration**: Prefect or Dagster (modern alternatives to Airflow)
- **Feature Store**: Feast (or custom)
- **Experiment Tracking**: MLflow + Weights & Biases
- **Data**: DuckDB (local analytics), PostgreSQL, MinIO (S3-compatible)
- **Data Quality**: Great Expectations (or custom validators)
- **Model Serving**: FastAPI + custom serving (builds on Forge concepts)
- **Monitoring**: Prometheus, Grafana, custom drift detection
- **ML Framework**: PyTorch, scikit-learn, HuggingFace
- **Deployment**: Docker, K3s, Helm, GitHub Actions
- **Message Queue**: Redis Streams or Kafka (for event-driven pipelines)
- **Data Versioning**: DVC or custom

## Phases

| Phase | Weeks | Focus | Outcome |
|-------|-------|-------|---------|
| Phase 1 | 1-7 | Data + Training Pipelines | Feature store, data versioning, training orchestration, experiment tracking |
| Phase 2 | 8-14 | Deployment + Monitoring | Model serving, A/B testing, drift detection, feedback loops, auto-retraining |
| Phase 3 | 15-20 | Platform Maturity + Portfolio | Self-service platform, advanced monitoring, pipeline optimization, blog posts |

## Hardware

- ASUS ROG Strix SCAR 16
- NVIDIA RTX 5080 (16GB VRAM)
- 32GB RAM
- 2TB SSD
- Ubuntu

## Key Differentiator

While other tracks go deep on ONE thing (inference, training, or infra), Conduit goes WIDE across the full lifecycle. The signal to hiring managers: "This person can own the entire ML pipeline from data to production, and knows where all the bodies are buried."

Best for: ML Platform teams, MLOps roles, Applied ML teams at any company that deploys models.

## How to Start a New Session

```
I'm working on Conduit, an end-to-end ML systems platform.
Read the manifest at /Users/jmalviya/Documents/zz/dev/plan_00/conduit/MANIFEST.md
Check progress at /Users/jmalviya/Documents/zz/dev/plan_00/conduit/progress.md
I'm currently on Week [N]. The spec is at:
/Users/jmalviya/Documents/zz/dev/plan_00/conduit/specs/phase[X]/week[NN]-[name].md
I need help with: [specific ask]
```

## Target Companies

- **Any company deploying ML**: Stripe, Spotify, Netflix, Uber, Airbnb
- **ML Platform companies**: Weights & Biases, MLflow/Databricks, Neptune, Comet
- **AI startups**: Any startup where you'd be the "ML engineer who makes it work"
- **Big tech ML teams**: Google, Meta, Amazon ML platform teams
- **AI labs (applied)**: Anthropic applied team, OpenAI applied, Cohere

## Related Projects

- Forge (Inference) at /Users/jmalviya/Documents/zz/dev/plan_00/forge/
- Anvil (Infrastructure) at /Users/jmalviya/Documents/zz/dev/plan_00/anvil/
- Crucible (Training/Alignment) at /Users/jmalviya/Documents/zz/dev/plan_00/crucible/
