# Week 5: Experiment Tracking

## Context

**Where it fits:** Week 5 of Phase 1 (Data + Training Pipelines). Experiment tracking is the memory of your ML system — every training run, hyperparameter choice, and result is logged, searchable, and reproducible.

**Prerequisites:**
- Weeks 1-4 complete: data pipeline running end-to-end with orchestration
- Basic ML training familiarity (PyTorch or sklearn)
- Understanding of hyperparameters, metrics, and model artifacts

**What it builds on:** Integrates with the orchestrated pipelines (Week 4), versioned datasets (Week 3), and features (Week 2). Every pipeline run that trains a model logs its experiment here.

**What comes next:** Week 6 (Training Orchestration) will use experiment tracking to manage HPO runs and model selection. Week 7 (Consolidation) ties it all together with automated model registration.

---

## Learning Goals

- [ ] Understand why experiment tracking is essential: reproducibility, comparison, collaboration, auditing
- [ ] Understand the anatomy of an experiment: run, parameters, metrics, artifacts, tags
- [ ] Understand MLflow's architecture: tracking server, artifact store, model registry
- [ ] Understand W&B's approach: richer visualizations, sweeps, team collaboration
- [ ] Understand how to structure experiments for meaningful comparison

---

## Implementation Goals

- [ ] Set up MLflow tracking server with PostgreSQL backend and MinIO artifact store
- [ ] Set up W&B (Weights & Biases) integration for enhanced visualization
- [ ] Log complete training metadata: hyperparams, per-step metrics, code version, data version
- [ ] Build experiment comparison: side-by-side metric plots, parameter importance
- [ ] Implement artifact management: model checkpoints, configs, evaluation results
- [ ] Organize runs with projects, tags, and searchable notes
- [ ] Implement run reproduction: from any logged run, recreate exact training setup
- [ ] Define and log custom domain-specific metrics

---

## Acceptance Criteria

1. MLflow UI at `http://localhost:5000` shows all experiments organized by project with searchable tags.
2. A training run logs at least: all hyperparameters, per-epoch train/val loss, per-epoch accuracy, final test metrics, training duration, GPU utilization.
3. `mlflow.search_runs()` with a filter query (e.g., `metrics.val_accuracy > 0.9 AND params.learning_rate < 0.01`) returns matching runs in under 2 seconds.
4. Side-by-side comparison of two runs shows metric curves overlaid on the same plot with clear visual differentiation.
5. Artifact store contains model checkpoints (`.pt` files), training configs (YAML), and evaluation reports (JSON) for every run.
6. Given a run ID, `conduit experiment reproduce --run-id <id>` outputs the exact command, config, data version, and code commit needed to reproduce it.
7. W&B integration logs the same data as MLflow and produces interactive dashboards with hyperparameter importance plots.
8. Custom metrics (e.g., calibration error, fairness metrics) are logged alongside standard metrics and appear in comparison views.
9. Runs are auto-tagged with git commit hash, dataset version, and feature set version for full traceability.
10. Deleting an experiment's artifacts from the artifact store is prevented if any downstream model in the registry references that run.

---

## Validation Commands

```bash
# Start MLflow tracking server
mlflow server \
  --backend-store-uri postgresql://conduit:conduit_dev@localhost:5432/mlflow \
  --default-artifact-root s3://conduit-artifacts/ \
  --host 0.0.0.0 --port 5000 &

# Run a training experiment
conduit experiment run --config configs/experiments/baseline.yaml

# List experiments
mlflow experiments search --filter "name LIKE '%fraud%'"

# Search runs
conduit experiment search --filter "metrics.val_accuracy > 0.85" --order-by "metrics.val_accuracy DESC"

# Compare runs
conduit experiment compare --run-ids "run_abc123,run_def456" --metrics "val_loss,val_accuracy"

# Reproduce a run
conduit experiment reproduce --run-id run_abc123 --dry-run

# Log a run with W&B
WANDB_MODE=online conduit experiment run --config configs/experiments/baseline.yaml --tracker wandb

# Check artifacts
mlflow artifacts list --run-id run_abc123

# Run tests
pytest tests/unit/experiments/ -v
pytest tests/integration/experiments/ -v
```

---

## Technical Implementation Details

### Project Structure (additions)

```
conduit/
├── src/conduit/
│   └── experiments/
│       ├── __init__.py
│       ├── tracker.py          # Unified tracking interface
│       ├── mlflow_backend.py   # MLflow implementation
│       ├── wandb_backend.py    # W&B implementation
│       ├── comparison.py       # Run comparison logic
│       ├── reproduce.py        # Reproducibility engine
│       └── metrics.py          # Custom metric definitions
├── configs/
│   └── experiments/
│       ├── baseline.yaml
│       └── experiment_template.yaml
└── mlflow/
    └── Dockerfile              # MLflow server container
```

### Unified Tracking Interface

```python
# src/conduit/experiments/tracker.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import time

@dataclass
class RunContext:
    experiment_name: str
    run_name: str
    tags: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    git_commit: str = ""
    data_version: str = ""
    feature_version: str = ""

class ExperimentTracker(ABC):
    @abstractmethod
    def start_run(self, context: RunContext) -> str:
        """Start a new tracked run, returns run_id."""

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""

    @abstractmethod
    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric value, optionally at a step."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at once."""

    @abstractmethod
    def log_artifact(self, local_path: Path, artifact_path: str | None = None) -> None:
        """Log a file as an artifact."""

    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""

    @abstractmethod
    def search_runs(self, experiment_name: str, filter_string: str) -> list[dict]:
        """Search runs matching filter criteria."""

class TrackerFactory:
    @staticmethod
    def create(backend: str, **kwargs) -> ExperimentTracker:
        match backend:
            case "mlflow":
                from conduit.experiments.mlflow_backend import MLflowTracker
                return MLflowTracker(**kwargs)
            case "wandb":
                from conduit.experiments.wandb_backend import WandbTracker
                return WandbTracker(**kwargs)
            case _:
                raise ValueError(f"Unknown tracking backend: {backend}")
```

### MLflow Backend

```python
# src/conduit/experiments/mlflow_backend.py
import mlflow
from pathlib import Path
from typing import Any
from conduit.experiments.tracker import ExperimentTracker, RunContext

class MLflowTracker(ExperimentTracker):
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self._run_id: str | None = None

    def start_run(self, context: RunContext) -> str:
        mlflow.set_experiment(context.experiment_name)
        run = mlflow.start_run(run_name=context.run_name, tags={
            **context.tags,
            "git_commit": context.git_commit,
            "data_version": context.data_version,
            "feature_version": context.feature_version,
        })
        self._run_id = run.info.run_id
        if context.params:
            mlflow.log_params(context.params)
        return self._run_id

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: Path, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(str(local_path), artifact_path)

    def end_run(self, status: str = "FINISHED") -> None:
        mlflow.end_run(status=status)
        self._run_id = None

    def search_runs(self, experiment_name: str, filter_string: str) -> list[dict]:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return []
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            output_format="list",
        )
        return [{"run_id": r.info.run_id, "params": r.data.params,
                 "metrics": r.data.metrics, "tags": r.data.tags} for r in runs]
```

### Training Loop with Tracking

```python
# src/conduit/experiments/example_training.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from conduit.experiments.tracker import TrackerFactory, RunContext

def train_model(config: dict):
    tracker = TrackerFactory.create(config["tracker_backend"])

    context = RunContext(
        experiment_name=config["experiment_name"],
        run_name=config["run_name"],
        params=config["hyperparameters"],
        git_commit=get_git_commit(),
        data_version=config["data_version"],
        feature_version=config["feature_version"],
    )
    run_id = tracker.start_run(context)

    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["hyperparameters"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["hyperparameters"]["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        tracker.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }, step=epoch)

        if epoch % config.get("checkpoint_every", 5) == 0:
            checkpoint_path = save_checkpoint(model, epoch, config)
            tracker.log_artifact(checkpoint_path, "checkpoints")

    # Final evaluation
    test_metrics = evaluate_final(model, test_loader)
    tracker.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    # Log custom metrics
    calibration = compute_calibration_error(model, test_loader)
    tracker.log_metric("calibration_error", calibration)

    # Save final model
    model_path = save_model(model, config)
    tracker.log_artifact(model_path, "models")
    tracker.log_artifact(Path(config["config_path"]), "configs")

    tracker.end_run()
    return run_id
```

### Experiment Config

```yaml
# configs/experiments/baseline.yaml
experiment_name: "fraud_detection"
run_name: "baseline_v1"
tracker_backend: "mlflow"

data_version: "v3"
feature_version: "user_features_v2"

hyperparameters:
  model_type: "transformer"
  learning_rate: 0.001
  batch_size: 256
  epochs: 50
  hidden_dim: 128
  num_layers: 4
  dropout: 0.1
  weight_decay: 0.0001

checkpoint_every: 10
early_stopping:
  patience: 5
  metric: "val_loss"
  mode: "min"
```

### Reproducibility Engine

```python
# src/conduit/experiments/reproduce.py
from dataclasses import dataclass
import mlflow

@dataclass
class ReproductionSpec:
    git_commit: str
    data_version: str
    feature_version: str
    config: dict
    command: str

def get_reproduction_spec(run_id: str) -> ReproductionSpec:
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    return ReproductionSpec(
        git_commit=run.data.tags.get("git_commit", "unknown"),
        data_version=run.data.tags.get("data_version", "unknown"),
        feature_version=run.data.tags.get("feature_version", "unknown"),
        config=run.data.params,
        command=f"git checkout {run.data.tags['git_commit']} && "
                f"dvc checkout data/ --rev {run.data.tags['data_version']} && "
                f"conduit experiment run --config <reconstructed_config>",
    )
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| MLflow server won't start | Ensure PostgreSQL has `mlflow` database: `createdb -h localhost -U conduit mlflow`. Check port 5000. |
| Artifacts not saving | Verify MinIO bucket `conduit-artifacts` exists. Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` env vars. |
| W&B login issues | Run `wandb login` with API key from wandb.ai/settings. For offline: `WANDB_MODE=offline`. |
| Metrics not showing per-step | Ensure `step` parameter is passed to `log_metric`. MLflow requires explicit step values. |
| Search query syntax | MLflow uses SQL-like syntax: `metrics.val_accuracy > 0.9 AND params.lr = '0.001'` (params are strings). |
| Run comparison slow | Limit search scope with experiment ID. Index frequently queried metrics. |

---

## Agent Handoff Template

```
I'm working on the Conduit project, Week 5: Experiment Tracking.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
Project root: ~/conduit/

Current state: [describe what's working/broken]

What I need help with: [specific issue]

Key files:
- src/conduit/experiments/tracker.py — Unified tracking interface (ABC)
- src/conduit/experiments/mlflow_backend.py — MLflow implementation
- src/conduit/experiments/wandb_backend.py — W&B implementation
- src/conduit/experiments/reproduce.py — Reproducibility engine
- configs/experiments/ — Experiment configuration files

Infrastructure: MLflow server (localhost:5000), PostgreSQL (backend store), MinIO (artifact store).
Tracking flow: start_run → log_params → [training loop: log_metrics per step] → log_artifacts → end_run.
```

---

## Out of Scope

- Model registry (covered in Week 7 consolidation)
- Automated hyperparameter tuning (Week 6: Training Orchestration)
- Multi-user collaboration features (team permissions, shared experiments)
- Experiment scheduling and automation (handled by Week 4 orchestration)
- A/B testing and online experiments
- Cost tracking per experiment (GPU hours, cloud spend)
- Custom MLflow plugins or UI extensions
