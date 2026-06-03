# Week 4: Pipeline Orchestration

## Context

**Where it fits:** Week 4 of Phase 1 (Data + Training Pipelines). Orchestration ties together all the pieces from Weeks 1-3 into automated, reliable, observable DAG pipelines. This is the "glue" that makes everything production-grade.

**Prerequisites:**
- Weeks 1-3 complete: ingestion, feature store, and versioning all working
- Understanding of DAGs (directed acyclic graphs)
- Docker Compose infrastructure running

**What it builds on:** Orchestrates the individual components built in Weeks 1-3: data ingestion (W1) → validation (W1) → feature engineering (W2) → versioning (W3) into automated pipelines with scheduling, retries, and monitoring.

**What comes next:** Week 5 (Experiment Tracking) and Week 6 (Training Orchestration) will add ML-specific tasks to these pipelines. Week 7 (Consolidation) integrates everything end-to-end.

---

## Learning Goals

- [ ] Understand workflow orchestration: why cron is insufficient for data pipelines
- [ ] Understand DAG semantics: dependencies, parallelism, conditional execution
- [ ] Understand retry strategies: exponential backoff, jitter, idempotency requirements
- [ ] Understand observability for pipelines: logging, metrics, alerting
- [ ] Understand parameterized pipelines: configuration-driven execution

---

## Implementation Goals

- [ ] Set up Prefect server locally with UI dashboard
- [ ] Build core data pipeline DAG: ingest → validate → feature engineer → version → store
- [ ] Implement scheduling: cron-based and event-triggered pipeline runs
- [ ] Implement dependency management: tasks respect upstream success/failure
- [ ] Add retry logic with configurable exponential backoff
- [ ] Build parameterized pipelines: same flow, different dataset/model configs
- [ ] Set up pipeline monitoring: run history, duration tracking, success rates
- [ ] Implement failure alerting: webhook notifications on pipeline failures

---

## Acceptance Criteria

1. Prefect server UI is accessible at `http://localhost:4200` and shows all registered flows.
2. The data pipeline DAG executes tasks in correct dependency order: ingest completes before validate starts, validate before feature engineering.
3. A scheduled pipeline triggers automatically at the configured cron interval (verified by checking run history).
4. When a task fails, it retries up to the configured max with exponential backoff (visible in logs: attempt 1, wait 2s, attempt 2, wait 4s, attempt 3).
5. A failed upstream task correctly prevents all downstream tasks from executing, with proper FAILED/SKIPPED status propagation.
6. The same pipeline definition runs with different parameters (dataset name, date range) producing separate tracked runs.
7. Pipeline run duration is logged and visible in the monitoring dashboard, with p50/p95 statistics over the last 7 days.
8. A failure notification webhook fires within 60 seconds of a pipeline failure (verified by a test webhook endpoint).
9. Parallel tasks within the same pipeline (independent feature computations) execute concurrently, reducing total wall-clock time.
10. A full ingest-to-store pipeline run completes in under 60 seconds for a 100K-row dataset and the run is fully observable in the Prefect UI.

---

## Validation Commands

```bash
# Start Prefect server
prefect server start &

# Deploy the pipeline
conduit pipeline deploy --config configs/pipelines/data_pipeline.yaml

# Trigger a manual run
conduit pipeline run --name data_pipeline --params '{"dataset": "transactions", "date": "2024-12-01"}'

# Check run status
prefect flow-run ls --flow-name data_pipeline --limit 5

# View scheduled deployments
prefect deployment ls

# Test retry behavior (inject failure)
conduit pipeline run --name data_pipeline --params '{"dataset": "transactions", "simulate_failure": true}'

# Check logs for retry attempts
prefect flow-run logs <run-id> | grep "Retry"

# Test parameterized runs
conduit pipeline run --name data_pipeline --params '{"dataset": "users"}'
conduit pipeline run --name data_pipeline --params '{"dataset": "merchants"}'

# Verify monitoring metrics
curl http://localhost:4200/api/flow_runs/filter -X POST -H "Content-Type: application/json" \
  -d '{"flow_runs": {"operator": "and_", "name": {"like_": "data_pipeline%"}}}'

# Run tests
pytest tests/unit/pipelines/ -v
pytest tests/integration/pipelines/ -v --timeout=120
```

---

## Technical Implementation Details

### Project Structure (additions)

```
conduit/
├── src/conduit/
│   └── pipelines/
│       ├── __init__.py
│       ├── data_pipeline.py    # Main data pipeline DAG
│       ├── tasks/
│       │   ├── __init__.py
│       │   ├── ingest.py       # Ingestion task
│       │   ├── validate.py     # Validation task
│       │   ├── features.py     # Feature engineering task
│       │   └── version.py      # Versioning task
│       ├── config.py           # Pipeline configuration
│       ├── monitoring.py       # Run metrics + alerting
│       └── notifications.py    # Webhook/Slack alerts
├── configs/
│   └── pipelines/
│       ├── data_pipeline.yaml
│       └── schedules.yaml
└── prefect.yaml                # Prefect deployment config
```

### Main Pipeline DAG

```python
# src/conduit/pipelines/data_pipeline.py
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta
from conduit.data.ingestion import DataIngestionPipeline
from conduit.data.validation import DataValidator
from conduit.features.engineering import FeatureEngineer
from conduit.versioning.dvc_manager import DVCManager

@task(retries=3, retry_delay_seconds=[2, 4, 8], cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def ingest_data(source: str, dataset: str) -> dict:
    logger = get_run_logger()
    logger.info(f"Ingesting {source} into dataset {dataset}")
    pipeline = DataIngestionPipeline.from_config()
    result = pipeline.ingest(source, dataset)
    return {"dataset": dataset, "rows": result.rows_ingested, "path": result.storage_path}

@task(retries=2, retry_delay_seconds=[5, 10])
def validate_data(ingestion_result: dict) -> dict:
    logger = get_run_logger()
    dataset = ingestion_result["dataset"]
    logger.info(f"Validating dataset {dataset}")
    validator = DataValidator.from_config()
    report = validator.validate_all(dataset)
    if not report.all_passed:
        failed = [c.check_name for c in report.results if not c.passed]
        raise ValueError(f"Validation failed for checks: {failed}")
    return {"dataset": dataset, "validation": "passed", "checks_run": len(report.results)}

@task(retries=2, retry_delay_seconds=[5, 10])
def compute_features(validation_result: dict) -> dict:
    logger = get_run_logger()
    dataset = validation_result["dataset"]
    logger.info(f"Computing features for {dataset}")
    engineer = FeatureEngineer.from_config()
    features = engineer.compute_all(dataset)
    return {"dataset": dataset, "features_computed": len(features)}

@task(retries=1)
def version_dataset(feature_result: dict) -> dict:
    logger = get_run_logger()
    dataset = feature_result["dataset"]
    logger.info(f"Versioning dataset {dataset}")
    dvc = DVCManager.from_config()
    version = dvc.track_and_push(dataset)
    return {"dataset": dataset, "version": version}

@flow(name="data_pipeline", log_prints=True)
def data_pipeline(dataset: str, source: str, date: str | None = None, simulate_failure: bool = False):
    """End-to-end data pipeline: ingest → validate → features → version."""
    logger = get_run_logger()
    logger.info(f"Starting data pipeline for {dataset} from {source}")

    if simulate_failure:
        raise RuntimeError("Simulated failure for testing")

    ingestion = ingest_data(source=source, dataset=dataset)
    validation = validate_data(ingestion)
    features = compute_features(validation)
    version = version_dataset(features)

    logger.info(f"Pipeline complete. Dataset {dataset} at version {version['version']}")
    return version
```

### Parameterized Pipeline Config

```yaml
# configs/pipelines/data_pipeline.yaml
name: data_pipeline
description: "End-to-end data pipeline: ingest → validate → features → version"

schedule:
  cron: "0 6 * * *"  # Daily at 6 AM
  timezone: "UTC"

parameters:
  defaults:
    dataset: "transactions"
    source: "s3://conduit-data/raw/transactions/latest.parquet"
  overrides:
    - name: "users_daily"
      dataset: "users"
      source: "s3://conduit-data/raw/users/latest.parquet"
    - name: "merchants_weekly"
      dataset: "merchants"
      source: "s3://conduit-data/raw/merchants/latest.parquet"
      schedule:
        cron: "0 6 * * 0"  # Weekly on Sunday

retry_policy:
  max_retries: 3
  backoff_factor: 2
  initial_delay_seconds: 2

notifications:
  on_failure:
    - type: webhook
      url: "http://localhost:8080/webhook/pipeline-failure"
    - type: slack
      channel: "#ml-alerts"
  on_success:
    - type: webhook
      url: "http://localhost:8080/webhook/pipeline-success"
```

### Monitoring and Alerting

```python
# src/conduit/pipelines/monitoring.py
import httpx
from dataclasses import dataclass
from datetime import datetime
from prefect import get_run_logger

@dataclass
class PipelineMetrics:
    flow_name: str
    run_id: str
    status: str
    duration_seconds: float
    tasks_completed: int
    tasks_failed: int
    timestamp: datetime

class PipelineMonitor:
    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url

    async def on_failure(self, flow, flow_run, state):
        logger = get_run_logger()
        logger.error(f"Pipeline {flow.name} failed: {state.message}")
        if self.webhook_url:
            payload = {
                "flow_name": flow.name,
                "run_id": str(flow_run.id),
                "status": "FAILED",
                "message": state.message,
                "timestamp": datetime.utcnow().isoformat(),
            }
            async with httpx.AsyncClient() as client:
                await client.post(self.webhook_url, json=payload, timeout=10)

    async def on_success(self, flow, flow_run, state):
        logger = get_run_logger()
        duration = (flow_run.end_time - flow_run.start_time).total_seconds()
        logger.info(f"Pipeline {flow.name} succeeded in {duration:.1f}s")
```

### Prefect Deployment

```yaml
# prefect.yaml
name: conduit
prefect-version: 2.14.0

deployments:
  - name: data-pipeline-daily
    entrypoint: src/conduit/pipelines/data_pipeline.py:data_pipeline
    work_pool:
      name: default-agent-pool
    schedule:
      cron: "0 6 * * *"
    parameters:
      dataset: "transactions"
      source: "s3://conduit-data/raw/transactions/latest.parquet"

  - name: data-pipeline-users
    entrypoint: src/conduit/pipelines/data_pipeline.py:data_pipeline
    work_pool:
      name: default-agent-pool
    schedule:
      cron: "0 7 * * *"
    parameters:
      dataset: "users"
      source: "s3://conduit-data/raw/users/latest.parquet"
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Prefect server won't start | Check port 4200 is free: `lsof -i :4200`. Try `prefect server start --host 0.0.0.0`. |
| Tasks not retrying | Verify `retries` and `retry_delay_seconds` are set on the `@task` decorator, not the `@flow`. |
| Schedule not triggering | Ensure a Prefect agent/worker is running: `prefect worker start --pool default-agent-pool`. |
| Dependency order wrong | Tasks must pass return values to create implicit dependencies. Use `wait_for=[task_a]` for explicit deps without data. |
| Webhook not receiving | Test the endpoint directly: `curl -X POST http://localhost:8080/webhook/test -d '{}'`. Check firewall. |
| Parallel tasks not parallel | Prefect runs tasks concurrently by default if they don't have data dependencies. Ensure independent tasks don't reference each other's results. |

---

## Agent Handoff Template

```
I'm working on the Conduit project, Week 4: Pipeline Orchestration.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
Project root: ~/conduit/

Current state: [describe what's working/broken]

What I need help with: [specific issue]

Key files:
- src/conduit/pipelines/data_pipeline.py — Main pipeline DAG
- src/conduit/pipelines/tasks/ — Individual task implementations
- src/conduit/pipelines/monitoring.py — Metrics and alerting
- configs/pipelines/data_pipeline.yaml — Pipeline configuration
- prefect.yaml — Deployment configuration

Infrastructure: Prefect server (localhost:4200), PostgreSQL, MinIO, Redis.
Pipeline flow: ingest → validate → compute features → version dataset.
All tasks have retries with exponential backoff. Pipeline is parameterized by dataset name.
```

---

## Out of Scope

- Distributed execution (Dask, Ray, Kubernetes) — local Prefect only
- Complex branching logic (if/else in DAGs) — linear pipeline with parallel leaves
- Data-aware scheduling (trigger on new file arrival) — cron only for now
- Multi-tenant pipeline isolation
- Cost tracking per pipeline run
- Pipeline versioning (different DAG versions running simultaneously)
- Cross-pipeline dependencies (pipeline A triggers pipeline B)
