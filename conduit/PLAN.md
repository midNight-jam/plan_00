# Conduit: ML Systems Engineering — The Full Plan (20 Weeks)

## Philosophy: "Own the Full Lifecycle, End to End"

- **Phase 1 (Weeks 1-7)**: Build the Data Foundation — Feature stores, data versioning, pipeline orchestration, experiment tracking
- **Phase 2 (Weeks 8-14)**: Build the Production Machine — Model serving, A/B testing, drift detection, auto-retraining
- **Phase 3 (Weeks 15-20)**: Build the Platform — Self-service tools, streaming, governance, polish

The key differentiator: While other tracks go DEEP on one thing, this track goes WIDE across the full lifecycle. You become the person who can own an ML system from data ingestion to production monitoring — the rarest and most valuable engineer on any applied ML team.

---

## Why This Track Matters

Most companies deploying ML don't need someone who can write CUDA kernels or implement RLHF. They need someone who can answer:
- "Our model's accuracy dropped 5% this week. What happened?"
- "How do we know when to retrain?"
- "Can we safely deploy this new model without breaking things?"
- "How do we track which data was used for which model?"
- "Our features are stale. How do we build real-time feature serving?"

These are the problems that 95% of ML teams face daily. Solving them well pays more than most people expect.

---

## Who This Track Is For

- Applied ML teams at any company (Stripe, Spotify, Netflix, Uber, Airbnb)
- ML Platform teams (Databricks, W&B, Neptune, Comet)
- AI startups where you'd be "the ML engineer who makes it work end-to-end"
- Big tech ML platform teams (Google, Meta, Amazon)
- Anyone who wants to be the indispensable "full-stack ML" person

---

## Phase 1: Data + Training Pipelines (Weeks 1-7)

### Week 1: Data Engineering Foundations

**Everything starts with data.** Bad data pipelines = bad models. No exceptions.

Build:
- DuckDB for local analytics (blazing fast, columnar, SQL — the modern data stack)
- Data ingestion pipeline: accept CSV/JSON/Parquet, validate schema, store in MinIO
- Data quality validation: null checks, range checks, uniqueness, referential integrity
- Data profiling: automated statistics on every dataset (distributions, missing rates, outliers)
- Schema evolution: handle new columns and type changes gracefully
- CLI: `conduit data ingest`, `conduit data validate`, `conduit data profile`

**Deliverable**: A data ingestion + validation system that catches bad data before it reaches models.

---

### Week 2: Feature Store

**The most underrated component in ML systems.** Prevents the #1 production ML bug: training-serving skew.

Build:
- Set up Feast (feature store) with offline (PostgreSQL) and online (Redis) backends
- Define features: entities, feature views, data sources
- Feature engineering pipeline: raw data → computed features → store
- Online serving: <10ms feature lookup at inference time
- Offline serving: point-in-time-correct feature retrieval for training (no future data leakage!)
- Feature registry: catalog all features with descriptions, owners, freshness SLAs

**Deliverable**: Working feature store serving features for both training and inference with guaranteed consistency.

**Interview signal**: "I understand training-serving skew and built a feature store that prevents it."

---

### Week 3: Data Versioning and Lineage

**Reproducibility.** "Which data trained this model?" must have an instant answer.

Build:
- DVC (Data Version Control): track datasets alongside code in Git
- Dataset versioning: version training/eval data, roll back when needed
- Lineage graph: raw data → transformations → features → model (fully traceable)
- Reproducibility guarantee: given any model, trace back to exact data + code + config
- Dataset diffing: compare v1 vs v2 (rows added/removed, distribution shifts)
- Metadata catalog: searchable registry of all datasets

**Deliverable**: Full lineage system — from any production model, trace back to the exact data that produced it.

---

### Week 4: Pipeline Orchestration

**Automation.** ML pipelines shouldn't need a human clicking buttons.

Build:
- Prefect (or Dagster) for workflow orchestration
- DAG pipelines: ingest → validate → feature engineer → train → evaluate
- Scheduling: cron, event-triggered, manual
- Retry logic with configurable backoff
- Parameterized pipelines: same DAG, different configs (different datasets, models)
- Monitoring: track pipeline runs, success/failure rates, duration
- Alerting: notify on pipeline failure

**Deliverable**: Automated pipeline that runs end-to-end without human intervention.

---

### Week 5: Experiment Tracking

**The Lab Notebook.** Every training run logged, comparable, reproducible.

Build:
- MLflow tracking server (local)
- W&B integration for rich visualizations
- Log everything: hyperparameters, per-step metrics, artifacts, code version, data version
- Experiment comparison: side-by-side plots, hyperparameter importance analysis
- Artifact management: checkpoints, configs, evaluation results
- Reproduce any run: from logged metadata, recreate exact training setup

**Deliverable**: Experiment tracking system where any historical run can be inspected and reproduced.

---

### Week 6: Training Orchestration

**Scale the experiments.** Automated hyperparameter search, early stopping, job management.

Build:
- Automated training pipeline: trigger on new data or schedule
- Hyperparameter optimization: Optuna integration (Bayesian optimization)
- Early stopping: monitor validation metric, halt when plateauing
- Training job queue: submit, monitor, cancel/restart
- Evaluation gates: after training, auto-run eval suite, log results
- Model selection: automatically pick best model from HPO sweep

**Deliverable**: Submit a training config → system finds best hyperparameters → evaluates → registers best model. Fully automated.

---

### Week 7: Phase 1 Consolidation

- End-to-end: raw data → validate → features → train → evaluate → register model
- Trigger by data arrival: new data → pipeline runs → new model candidate
- Blog post: "Building an End-to-End ML Pipeline: From Raw Data to Model Registry"
- Integration tests, documentation, ADRs

**Phase 1 Milestone**: A complete automated ML pipeline from data to registered model.

---

## Phase 2: Deployment + Monitoring (Weeks 8-14)

**The production side.** Getting a model trained is half the battle. Keeping it healthy in production is the other half.

### Week 8: Model Registry and Packaging

**The handoff point between training and serving.**

Build:
- MLflow Model Registry: staging → production transitions
- Model packaging: model + dependencies + serving code in one artifact
- Model signatures: input/output schema validation (catch bad inputs before inference)
- Containerized model serving: model → Docker → serve endpoint
- Promotion workflow: auto-checks before promotion (eval passes, no regressions)

**Deliverable**: Models promoted through stages with automated checks. One command to deploy.

---

### Week 9: Model Serving Patterns

**Not all serving is the same.** Different patterns for different needs.

Build:
- Online serving: FastAPI with feature lookup → predict → respond (<100ms)
- Batch serving: scheduled prediction on large datasets (daily scoring)
- Shadow mode: new model runs alongside production, compare outputs silently
- Feature serving integration: real-time features from feature store at inference
- Response caching: cache predictions for identical inputs
- Latency budget decomposition: feature lookup + inference + postprocessing

**Deliverable**: Multiple serving modes working, with shadow mode for safe testing.

---

### Week 10: A/B Testing Framework

**The decision engine.** How do you know if a new model is actually better?

Build:
- Traffic splitting: consistent user assignment (user X always sees model A)
- Metric collection: define success metrics, collect per-variant
- Statistical significance: p-values, confidence intervals, required sample size calculation
- Guardrail metrics: safety metrics that auto-stop experiments if violated
- Multi-armed bandits: Thompson sampling for faster convergence
- Experiment dashboard: live metrics, significance indicators, time to decision

**Deliverable**: Deploy two model variants, split traffic, get statistically significant result automatically.

**Interview signal**: "I can design and run ML experiments with proper statistical rigor."

---

### Week 11: Model Monitoring and Drift Detection

**The early warning system.** Models degrade silently. You need to catch it.

Build:
- Data drift: monitor input feature distributions (KS test, PSI, JS divergence)
- Prediction drift: monitor output distribution changes
- Performance monitoring: track accuracy when ground truth arrives (often delayed)
- Feature importance drift: which features shifted contribution?
- Alert pipeline: drift detected → alert → investigation workflow
- Dashboard: distributions, drift scores, performance trends

**Deliverable**: Monitoring system that detects when a model starts to degrade and alerts before users notice.

---

### Week 12: Feedback Loops and Active Learning

**The data flywheel.** Production data makes models better, which makes products better, which generates more data.

Build:
- Production label collection: user actions → ground truth labels
- Active learning: identify high-uncertainty samples, prioritize for labeling
- Human-in-the-loop: route low-confidence predictions to reviewers
- Feedback data pipeline: events → labels → feature store enrichment
- Retraining triggers: accumulated labels > threshold OR drift detected

**Deliverable**: System that automatically collects production labels and identifies the most valuable data for retraining.

---

### Week 13: Auto-Retraining Pipeline

**Self-healing models.** Drift detected → retrain → evaluate → deploy (or reject).

Build:
- Trigger conditions: scheduled, drift-triggered, data-volume-triggered
- Automated retraining: new data → train → evaluate → compare to production
- Promotion decision: new model beats production → auto-promote
- Rejection: new model worse → reject, alert team
- Canary deployment: 5% traffic → monitor → expand or rollback
- Automatic rollback: metrics degrade → revert within 5 minutes
- Full audit trail: every retraining logged end-to-end

**Deliverable**: The full loop — from drift detection to automatic redeployment. Self-healing ML.

---

### Week 14: Phase 2 Consolidation

- Full lifecycle demo: data → features → drift → retrain → deploy → A/B → promote
- Incident response playbook: "model degraded, now what?"
- Blog post: "Building Self-Healing ML Systems"
- Integration tests, documentation

**Phase 2 Milestone**: A production ML system that monitors itself, detects problems, and fixes them automatically.

---

## Phase 3: Platform Maturity + Portfolio (Weeks 15-20)

### Week 15: Self-Service ML Platform

Build CLI/SDK so data scientists can use the platform without knowing the infrastructure. Templates, guardrails, validation. Documentation that gets someone productive in 15 minutes.

### Week 16: Streaming and Real-Time Patterns

Event-driven pipelines (Redis Streams), real-time feature computation, online prediction <100ms, windowed aggregations. Build a real-time recommendation or fraud detection system.

### Week 17: Cost and Efficiency

Training cost tracking (GPU-hours per experiment), inference cost per request, model distillation (train small from large), resource utilization dashboards, budget alerts.

### Week 18: ML Governance

Model cards, audit trails (complete lineage), bias detection, compliance documentation (GDPR, AI Act awareness), approval workflows, risk scoring for model changes.

### Weeks 19-20: Portfolio Polish

4 blog posts, architecture documentation, demo video (full lifecycle in 5-7 minutes), benchmark report, comparison against Vertex AI/SageMaker, open-source release.

---

## What You DON'T Need to Do

- **Skip**: Building a custom inference engine (that's Forge territory)
- **Skip**: Implementing alignment algorithms (that's Crucible territory)
- **Skip**: Kubernetes operator development (that's Anvil territory)
- **Skip**: Custom CUDA kernels (not needed for ML systems roles)
- **Skip**: Distributed training implementation (understand concepts, use existing tools)
- **Skip**: Building a fancy UI (CLI + API is sufficient)

---

## The Resume Lines This Produces

> "Built Conduit, an end-to-end ML platform featuring automated data pipelines, feature store with point-in-time correctness, experiment tracking, A/B testing with statistical significance, data drift detection (KS/PSI), and auto-retraining with canary deployment. System detected model degradation within 30 minutes and auto-recovered without human intervention."

> "Implemented ML governance framework with complete model lineage (data → features → training → deployment), automated bias detection, model cards, and audit trails. Built self-service CLI enabling data scientists to deploy models with zero infrastructure knowledge."

---

## Daily Routine

- **1 hour**: Study — read ML system design case studies, study open-source ML platforms
- **4-5 hours**: Build — implement components, wire integrations, run experiments
- **30 min**: Document — architecture notes, decision records, blog paragraphs
- **30 min**: Test/Polish — integration tests, CLI UX improvements, error messages

---

## Key Resources

1. "Designing Machine Learning Systems" by Chip Huyen — THE book for this track
2. "Machine Learning Engineering" by Andriy Burkov — practical production ML
3. Google's "Rules of ML" — lessons from decades of production ML
4. Netflix's ML platform blog posts — real-world platform engineering
5. Uber Michelangelo papers — feature store and platform design
6. Feast documentation — feature store patterns
7. MLflow documentation — experiment tracking and model registry

---

## Phase Gates

**End of Phase 1 (Week 7)**: Complete automated pipeline from data to registered model. Can explain feature stores, data versioning, and pipeline orchestration.

**End of Phase 2 (Week 14)**: Self-healing ML system — detects drift, retrains, deploys with canary, rolls back if bad. Can run A/B tests with statistical rigor.

**End of Phase 3 (Week 20)**: Full platform with CLI, governance, streaming, and documentation. Blog posts published. Can own any ML system end-to-end.

---

## How This Differs from Other Tracks

| Dimension | Forge | Anvil | Crucible | **Conduit** |
|-----------|-------|-------|----------|-------------|
| Depth | GPU inference | Distributed infra | Alignment math | **Breadth across lifecycle** |
| Core skill | Optimization | Reliability | Training theory | **Integration & systems thinking** |
| Interview Q | "Design model serving" | "Design fault-tolerant scheduler" | "Explain DPO derivation" | **"Design ML pipeline for X"** |
| Best for | Inference teams | Infra teams | Research/alignment teams | **Applied ML / Platform teams** |
