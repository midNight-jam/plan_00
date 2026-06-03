# Week 7: Phase 1 Consolidation

## Context

**Where it fits:** Week 7 is the final week of Phase 1 (Data + Training Pipelines). This week integrates all components from Weeks 1-6 into a cohesive, end-to-end ML pipeline that runs autonomously.

**Prerequisites:**
- Weeks 1-6 complete and individually functional:
  - W1: Data ingestion + validation + profiling
  - W2: Feature store (Feast + Redis)
  - W3: Data versioning + lineage (DVC)
  - W4: Pipeline orchestration (Prefect)
  - W5: Experiment tracking (MLflow + W&B)
  - W6: Training orchestration (Optuna + evaluation gates)

**What it builds on:** Every previous week. This is the integration week — the system should work end-to-end without manual intervention.

**What comes next:** Phase 2 will add model serving, monitoring, and production deployment on top of this foundation.

---

## Learning Goals

- [ ] Understand end-to-end ML pipeline architecture: how components interact in production
- [ ] Understand integration testing for ML systems: testing data flow, not just unit logic
- [ ] Understand ADR (Architecture Decision Records): documenting why, not just what
- [ ] Understand pipeline reliability: what makes a pipeline robust vs fragile
- [ ] Understand documentation as a first-class deliverable in ML engineering

---

## Implementation Goals

- [ ] Wire up end-to-end pipeline: raw data → validate → features → train → evaluate → register
- [ ] Implement data-arrival trigger: new data in MinIO → pipeline kicks off automatically
- [ ] Write integration tests covering the full pipeline path
- [ ] Write comprehensive documentation: architecture, adding features, adding models
- [ ] Write blog post: "Building an End-to-End ML Pipeline"
- [ ] Write ADRs: Prefect vs Airflow, Feast vs custom, DVC vs LakeFS
- [ ] Ensure code quality: type hints, linting, formatting, pre-commit hooks
- [ ] Set up CI/CD: automated tests on push, linting, type checking

---

## Acceptance Criteria

1. Dropping a new data file into MinIO's `raw/` prefix automatically triggers the full pipeline within 60 seconds, ending with a new model candidate registered in MLflow.
2. The end-to-end pipeline (ingest → validate → features → train 5 epochs → evaluate → register) completes in under 5 minutes for a 100K-row dataset.
3. Integration tests cover at least 5 critical paths: happy path, validation failure, feature computation, training with early stopping, and evaluation gate rejection.
4. All integration tests pass in CI with `pytest tests/integration/ -v` and complete in under 10 minutes.
5. Architecture documentation includes a system diagram (Mermaid), component descriptions, and data flow explanation.
6. "How to add a new feature" guide successfully allows a new feature to be added by following the steps without external help.
7. Blog post draft covers motivation, architecture, key decisions, lessons learned, and is at least 2000 words.
8. At least 3 ADRs are written (orchestration, feature store, versioning) with context, decision, and consequences sections.
9. Pre-commit hooks enforce: black formatting, ruff linting, mypy type checking, and all pass on the current codebase.
10. CI pipeline (GitHub Actions or GitLab CI) runs tests, linting, and type checking on every push, completing in under 5 minutes.

---

## Validation Commands

```bash
# Run the full end-to-end pipeline
conduit pipeline run --name end_to_end --params '{"dataset": "transactions", "source": "data/raw/transactions.parquet"}'

# Test data-arrival trigger (drop file and watch)
mc cp data/raw/new_transactions.parquet minio/conduit-data/raw/transactions/
# Wait and verify pipeline started:
prefect flow-run ls --flow-name end_to_end --limit 1

# Run integration tests
pytest tests/integration/ -v --timeout=600

# Run full test suite
pytest tests/ -v --timeout=600 -x

# Lint and type check
ruff check src/
mypy src/conduit/ --strict
black --check src/ tests/

# Run pre-commit on all files
pre-commit run --all-files

# Build documentation
mkdocs build --strict

# Verify CI locally
act push  # (if using GitHub Actions with `act`)

# Check model was registered
mlflow models list | grep "fraud_detection"

# Verify lineage for latest model
conduit lineage show --model fraud_detection_latest --format mermaid
```

---

## Technical Implementation Details

### Project Structure (final)

```
conduit/
├── src/conduit/
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── features.py
│   │   ├── pipeline.py
│   │   ├── train.py
│   │   └── experiment.py
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── validation.py
│   │   ├── profiling.py
│   │   ├── schema.py
│   │   └── storage.py
│   ├── features/
│   │   ├── engineering.py
│   │   ├── registry.py
│   │   └── serving.py
│   ├── versioning/
│   │   ├── dvc_manager.py
│   │   ├── lineage.py
│   │   ├── diff.py
│   │   └── catalog.py
│   ├── pipelines/
│   │   ├── data_pipeline.py
│   │   ├── training_pipeline.py
│   │   ├── end_to_end.py      # NEW: full pipeline
│   │   ├── triggers.py         # NEW: event triggers
│   │   └── tasks/
│   ├── experiments/
│   │   ├── tracker.py
│   │   ├── mlflow_backend.py
│   │   └── wandb_backend.py
│   └── training/
│       ├── orchestrator.py
│       ├── hpo.py
│       ├── early_stopping.py
│       └── resources.py
├── tests/
│   ├── unit/
│   ├── integration/
│   │   ├── test_end_to_end.py  # NEW
│   │   ├── test_data_trigger.py # NEW
│   │   └── conftest.py
│   └── conftest.py
├── docs/
│   ├── architecture.md         # NEW
│   ├── adding-features.md      # NEW
│   ├── adding-models.md        # NEW
│   └── adr/
│       ├── 001-orchestration.md
│       ├── 002-feature-store.md
│       └── 003-versioning.md
├── blog/
│   └── end-to-end-ml-pipeline.md # NEW
├── .pre-commit-config.yaml     # NEW
├── .github/workflows/ci.yaml   # NEW
├── pyproject.toml
├── docker-compose.yaml
└── Makefile
```

### End-to-End Pipeline

```python
# src/conduit/pipelines/end_to_end.py
from prefect import flow, task, get_run_logger
from conduit.data.ingestion import DataIngestionPipeline
from conduit.data.validation import DataValidator
from conduit.features.engineering import FeatureEngineer
from conduit.versioning.dvc_manager import DVCManager
from conduit.training.orchestrator import TrainingOrchestrator
from conduit.experiments.tracker import TrackerFactory

@task(retries=3, retry_delay_seconds=[2, 4, 8])
def ingest(source: str, dataset: str) -> dict:
    pipeline = DataIngestionPipeline.from_config()
    result = pipeline.ingest(source, dataset)
    return {"dataset": dataset, "rows": result.rows_ingested, "path": result.storage_path}

@task(retries=2, retry_delay_seconds=[5, 10])
def validate(ingestion_result: dict) -> dict:
    validator = DataValidator.from_config()
    report = validator.validate_all(ingestion_result["dataset"])
    if not report.all_passed:
        raise ValueError(f"Validation failed: {report.failed_checks}")
    return ingestion_result

@task
def compute_features(validated: dict) -> dict:
    engineer = FeatureEngineer.from_config()
    features = engineer.compute_all(validated["dataset"])
    return {"dataset": validated["dataset"], "features": features}

@task
def version_data(feature_result: dict) -> dict:
    dvc = DVCManager.from_config()
    version = dvc.track_and_push(feature_result["dataset"])
    return {"version": version, **feature_result}

@task(retries=1)
def train_model(versioned: dict, training_config: dict) -> dict:
    orchestrator = TrainingOrchestrator.from_config()
    job_id = orchestrator.submit(training_config)
    orchestrator.wait_for_completion(job_id)
    job = orchestrator.get_job(job_id)
    return {"job_id": job_id, "best_metric": job.best_metric, "run_id": job.mlflow_run_id}

@task
def evaluate_model(training_result: dict, gate_config: dict) -> dict:
    from conduit.training.evaluation import EvaluationGate
    gate = EvaluationGate(gate_config)
    passed = gate.evaluate(training_result["run_id"])
    if not passed:
        raise ValueError(f"Model failed evaluation gate")
    return {**training_result, "gate_passed": True}

@task
def register_model(eval_result: dict) -> dict:
    import mlflow
    model_uri = f"runs:/{eval_result['run_id']}/model"
    registered = mlflow.register_model(model_uri, "fraud_detection")
    return {"model_name": "fraud_detection", "version": registered.version}

@flow(name="end_to_end")
def end_to_end_pipeline(dataset: str, source: str, training_config: dict | None = None):
    logger = get_run_logger()
    logger.info(f"Starting end-to-end pipeline for {dataset}")

    if training_config is None:
        training_config = load_default_training_config(dataset)

    ingested = ingest(source=source, dataset=dataset)
    validated = validate(ingested)
    features = compute_features(validated)
    versioned = version_data(features)
    trained = train_model(versioned, training_config)
    evaluated = evaluate_model(trained, gate_config=load_gate_config())
    registered = register_model(evaluated)

    logger.info(f"Pipeline complete. Model registered: {registered}")
    return registered
```

### Data-Arrival Trigger

```python
# src/conduit/pipelines/triggers.py
import time
from minio import Minio
from prefect.deployments import run_deployment

class MinIOEventTrigger:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str, prefix: str):
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)
        self.bucket = bucket
        self.prefix = prefix
        self.seen_objects: set[str] = set()

    def poll(self, interval_seconds: int = 30):
        """Poll MinIO for new objects and trigger pipeline."""
        while True:
            objects = self.client.list_objects(self.bucket, prefix=self.prefix, recursive=True)
            for obj in objects:
                if obj.object_name not in self.seen_objects:
                    self.seen_objects.add(obj.object_name)
                    self._trigger_pipeline(obj.object_name)
            time.sleep(interval_seconds)

    def _trigger_pipeline(self, object_name: str):
        dataset = object_name.split("/")[1]  # raw/<dataset>/file.parquet
        source = f"s3://{self.bucket}/{object_name}"
        run_deployment(
            name="end_to_end/data-arrival",
            parameters={"dataset": dataset, "source": source},
        )
```

### Integration Test

```python
# tests/integration/test_end_to_end.py
import pytest
from pathlib import Path
from conduit.pipelines.end_to_end import end_to_end_pipeline

@pytest.fixture
def sample_data(tmp_path):
    import pandas as pd
    import numpy as np
    np.random.seed(42)
    df = pd.DataFrame({
        "transaction_id": [f"txn_{i}" for i in range(1000)],
        "user_id": [f"user_{i % 100}" for i in range(1000)],
        "amount": np.random.exponential(50, 1000),
        "timestamp": pd.date_range("2024-01-01", periods=1000, freq="h"),
        "is_fraud": np.random.choice([0, 1], 1000, p=[0.95, 0.05]),
    })
    path = tmp_path / "transactions.parquet"
    df.to_parquet(path)
    return str(path)

def test_happy_path(sample_data):
    """Full pipeline completes successfully with valid data."""
    result = end_to_end_pipeline(dataset="test_transactions", source=sample_data)
    assert result["model_name"] == "fraud_detection"
    assert result["version"] is not None

def test_validation_failure(tmp_path):
    """Pipeline stops at validation when data has quality issues."""
    import pandas as pd
    df = pd.DataFrame({
        "transaction_id": [None] * 100,  # all nulls — fails not_null check
        "amount": [-5] * 100,             # negative — fails range check
    })
    path = tmp_path / "bad_data.parquet"
    df.to_parquet(path)
    with pytest.raises(ValueError, match="Validation failed"):
        end_to_end_pipeline(dataset="test_bad", source=str(path))

def test_evaluation_gate_rejection(sample_data, monkeypatch):
    """Pipeline stops at evaluation gate when model is below threshold."""
    from conduit.training.evaluation import EvaluationGate
    monkeypatch.setattr(EvaluationGate, "evaluate", lambda self, run_id: False)
    with pytest.raises(ValueError, match="failed evaluation gate"):
        end_to_end_pipeline(dataset="test_reject", source=sample_data)
```

### CI Configuration

```yaml
# .github/workflows/ci.yaml
name: CI
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff black mypy
      - run: ruff check src/
      - run: black --check src/ tests/
      - run: mypy src/conduit/ --strict

  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: conduit
          POSTGRES_USER: conduit
          POSTGRES_PASSWORD: conduit_dev
        ports: ["5432:5432"]
      redis:
        image: redis:7
        ports: ["6379:6379"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit/ -v --timeout=60
      - run: pytest tests/integration/ -v --timeout=600
```

### Pre-commit Config

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML, types-redis]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=1000]
```

### ADR Template

```markdown
# ADR-001: Prefect for Pipeline Orchestration

## Status: Accepted

## Context
We need a workflow orchestration tool to manage our ML pipelines (data ingestion,
feature engineering, training). Options considered: Airflow, Prefect, Dagster, Argo.

## Decision
We chose Prefect because:
- Pure Python API (no DSL, no YAML DAGs)
- Local-first development (no scheduler process required for testing)
- Native async support for I/O-bound tasks
- Built-in retry logic with exponential backoff
- Lightweight deployment model (server + worker)

## Consequences
- Positive: Fast iteration during development, easy testing
- Positive: Prefect Cloud available if we need managed infrastructure later
- Negative: Smaller community than Airflow (fewer plugins, less SO answers)
- Negative: Less battle-tested at extreme scale (10K+ daily runs)
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| End-to-end test hangs | Check for deadlocks in sequential task dependencies. Add timeouts to each task. Verify Docker services are responsive. |
| Trigger not detecting new files | Verify MinIO bucket notifications are configured. Check polling interval. Manually test with `mc cp`. |
| CI tests fail but local pass | Ensure all Docker services are listed in CI config. Check for hardcoded `localhost` paths. Use environment variables. |
| Pre-commit mypy fails | Add type stubs: `pip install types-PyYAML types-redis types-requests`. Use `# type: ignore` sparingly for third-party libs. |
| Integration tests slow | Use smaller datasets in tests (100-1000 rows). Mock external services where possible. Parallelize independent tests. |
| Model registration fails | Ensure MLflow server is running and reachable. Check the `registered_model` doesn't already exist at same version. |

---

## Agent Handoff Template

```
I'm working on the Conduit project, Week 7: Phase 1 Consolidation.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
Project root: ~/conduit/

Current state: [describe what's working/broken]

What I need help with: [specific issue]

Key files:
- src/conduit/pipelines/end_to_end.py — Full pipeline integration
- src/conduit/pipelines/triggers.py — Data-arrival trigger
- tests/integration/test_end_to_end.py — Integration tests
- .github/workflows/ci.yaml — CI configuration
- .pre-commit-config.yaml — Code quality hooks
- docs/ — Architecture and guides
- docs/adr/ — Architecture Decision Records

Infrastructure: Full stack — PostgreSQL, MinIO, Redis, Prefect, MLflow, DuckDB, Feast.
Pipeline: raw data arrival → ingest → validate → features → version → train → evaluate → register model.
This is the integration week — all components from Weeks 1-6 working together.
```

---

## Out of Scope

- Production deployment (Kubernetes, cloud) — covered in Phase 2
- Model serving (REST API, gRPC) — Phase 2
- Model monitoring (drift, performance degradation) — Phase 2
- Advanced CI/CD (canary deployments, blue-green) — Phase 2
- Security hardening (secrets management, network policies)
- Load testing and performance optimization beyond basic benchmarks
- Multi-model pipeline (ensemble, cascade) — single model path only
