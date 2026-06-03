# Week 6: Training Orchestration

## Context

**Where it fits:** Week 6 of Phase 1 (Data + Training Pipelines). Training orchestration automates the process of finding optimal models — triggering training, tuning hyperparameters, managing resources, and selecting winners.

**Prerequisites:**
- Week 4 complete: pipeline orchestration with Prefect
- Week 5 complete: experiment tracking with MLflow/W&B
- PyTorch installed with CUDA support (RTX 5080)
- Understanding of hyperparameter optimization concepts

**What it builds on:** Combines pipeline orchestration (Week 4) with experiment tracking (Week 5). Training jobs are orchestrated pipelines that log results to the experiment tracker. Uses versioned data (Week 3) and computed features (Week 2).

**What comes next:** Week 7 (Consolidation) integrates training orchestration into the full end-to-end pipeline with automated model registration and evaluation gates.

---

## Learning Goals

- [ ] Understand automated training: why manual training doesn't scale and how automation enables iteration velocity
- [ ] Understand hyperparameter optimization: grid search vs random vs Bayesian (Optuna's TPE sampler)
- [ ] Understand early stopping: preventing overfitting and wasted compute
- [ ] Understand resource management: GPU scheduling, memory constraints, concurrent job limits
- [ ] Understand model selection: how to automatically pick the best model from many candidates

---

## Implementation Goals

- [ ] Build automated training pipeline: data-arrival-triggered or scheduled training
- [ ] Integrate Optuna for hyperparameter optimization with pruning
- [ ] Implement early stopping with configurable patience and delta
- [ ] Build resource manager: track GPU utilization, limit concurrent jobs, queue overflow
- [ ] Implement training configuration management: YAML hierarchy with overrides
- [ ] Build training job queue: submit, track progress, cancel, restart
- [ ] Implement evaluation gates: post-training automatic evaluation with pass/fail criteria
- [ ] Implement model selection: pick best model from HPO study based on configurable criteria

---

## Acceptance Criteria

1. A new training job can be submitted via `conduit train submit --config configs/training/model_v1.yaml` and begins executing within 5 seconds.
2. Optuna HPO study with 20 trials completes and identifies the best hyperparameter combination, logged to MLflow with all trial results.
3. Early stopping correctly halts training when validation loss hasn't improved for `patience` epochs (verified by checking epoch count < max_epochs).
4. Resource manager prevents launching a second GPU training job when GPU memory utilization exceeds 80%.
5. Training config override works: `conduit train submit --config base.yaml --override lr=0.0001,epochs=100` correctly merges parameters.
6. Job queue shows all submitted, running, and completed jobs with their status, GPU usage, and ETA.
7. Evaluation gate after training automatically runs the test suite and blocks model registration if accuracy < threshold (configurable).
8. Model selection from a 20-trial HPO study picks the trial with best val_accuracy and registers it as the candidate model.
9. A cancelled training job (`conduit train cancel --job-id <id>`) cleanly stops, saves the last checkpoint, and logs partial results.
10. End-to-end: submit HPO study → 20 trials with pruning → early stopping → evaluate best → register winner, all in under 30 minutes on RTX 5080.

---

## Validation Commands

```bash
# Submit a single training job
conduit train submit --config configs/training/fraud_model.yaml

# Submit HPO study
conduit train hpo --config configs/training/fraud_model.yaml \
  --n-trials 20 --study-name fraud_hpo_v1

# Check job status
conduit train status
conduit train status --job-id <job-id>

# View GPU utilization
conduit resources gpu-status

# Cancel a running job
conduit train cancel --job-id <job-id>

# View HPO study results
conduit train hpo-results --study-name fraud_hpo_v1

# Trigger evaluation gate
conduit train evaluate --job-id <best-job-id> --gate configs/gates/production_gate.yaml

# Select best model from study
conduit train select-best --study-name fraud_hpo_v1 --metric val_accuracy --direction maximize

# View training queue
conduit train queue

# Run tests
pytest tests/unit/training/ -v
pytest tests/integration/training/ -v --timeout=300
```

---

## Technical Implementation Details

### Project Structure (additions)

```
conduit/
├── src/conduit/
│   └── training/
│       ├── __init__.py
│       ├── orchestrator.py     # Training job orchestration
│       ├── hpo.py              # Hyperparameter optimization (Optuna)
│       ├── early_stopping.py   # Early stopping logic
│       ├── resources.py        # GPU/resource management
│       ├── config.py           # Config management with overrides
│       ├── queue.py            # Job queue management
│       ├── evaluation.py       # Evaluation gates
│       └── selection.py        # Model selection
├── configs/
│   └── training/
│       ├── fraud_model.yaml    # Model training config
│       ├── hpo_space.yaml      # HPO search space
│       └── base.yaml           # Base config (inherited)
│   └── gates/
│       └── production_gate.yaml
```

### Training Orchestrator

```python
# src/conduit/training/orchestrator.py
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import torch
from conduit.experiments.tracker import TrackerFactory, RunContext
from conduit.training.early_stopping import EarlyStopping
from conduit.training.resources import ResourceManager

class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TrainingJob:
    job_id: str
    config: dict
    status: JobStatus
    gpu_id: int | None = None
    current_epoch: int = 0
    best_metric: float | None = None

class TrainingOrchestrator:
    def __init__(self, resource_manager: ResourceManager, tracker_factory: TrackerFactory):
        self.resource_manager = resource_manager
        self.tracker_factory = tracker_factory
        self.jobs: dict[str, TrainingJob] = {}

    def submit(self, config: dict) -> str:
        job_id = self._generate_job_id()
        job = TrainingJob(job_id=job_id, config=config, status=JobStatus.QUEUED)
        self.jobs[job_id] = job

        gpu_id = self.resource_manager.acquire_gpu(
            memory_required_gb=config.get("gpu_memory_gb", 8)
        )
        if gpu_id is not None:
            job.gpu_id = gpu_id
            job.status = JobStatus.RUNNING
            self._execute_training(job)
        return job_id

    def _execute_training(self, job: TrainingJob):
        config = job.config
        tracker = self.tracker_factory.create(config.get("tracker_backend", "mlflow"))

        context = RunContext(
            experiment_name=config["experiment_name"],
            run_name=f"job_{job.job_id}",
            params=config["hyperparameters"],
        )
        tracker.start_run(context)

        model = self._build_model(config)
        optimizer = self._build_optimizer(model, config)
        early_stopping = EarlyStopping(
            patience=config.get("early_stopping_patience", 5),
            min_delta=config.get("early_stopping_delta", 0.001),
        )

        device = torch.device(f"cuda:{job.gpu_id}")
        model = model.to(device)

        for epoch in range(config["hyperparameters"]["epochs"]):
            if job.status == JobStatus.CANCELLED:
                break
            train_loss = self._train_epoch(model, optimizer, device, config)
            val_loss, val_acc = self._validate(model, device, config)

            job.current_epoch = epoch
            job.best_metric = max(job.best_metric or 0, val_acc)
            tracker.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc}, step=epoch)

            if early_stopping.should_stop(val_loss):
                tracker.log_metric("stopped_epoch", epoch)
                break

        tracker.end_run()
        job.status = JobStatus.COMPLETED
        self.resource_manager.release_gpu(job.gpu_id)
```

### Hyperparameter Optimization

```python
# src/conduit/training/hpo.py
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from typing import Callable

class HPOStudy:
    def __init__(self, study_name: str, storage_url: str = "sqlite:///optuna.db"):
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            load_if_exists=True,
        )

    def define_search_space(self, trial: optuna.Trial, space_config: dict) -> dict:
        params = {}
        for name, spec in space_config.items():
            match spec["type"]:
                case "float":
                    params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
                case "int":
                    params[name] = trial.suggest_int(name, spec["low"], spec["high"])
                case "categorical":
                    params[name] = trial.suggest_categorical(name, spec["choices"])
        return params

    def optimize(self, objective: Callable, n_trials: int, space_config: dict):
        def wrapped_objective(trial):
            params = self.define_search_space(trial, space_config)
            return objective(params, trial)

        self.study.optimize(wrapped_objective, n_trials=n_trials, show_progress_bar=True)
        return self.study.best_trial

    def get_results(self) -> dict:
        return {
            "best_params": self.study.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials),
            "trials": [
                {"number": t.number, "value": t.value, "params": t.params, "state": t.state.name}
                for t in self.study.trials
            ],
        }
```

### Early Stopping

```python
# src/conduit/training/early_stopping.py
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: float | None = None

    def should_stop(self, current_value: float) -> bool:
        if self.best_value is None:
            self.best_value = current_value
            return False

        improved = (
            current_value < self.best_value - self.min_delta if self.mode == "min"
            else current_value > self.best_value + self.min_delta
        )

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
```

### Resource Manager

```python
# src/conduit/training/resources.py
import subprocess
import json
from dataclasses import dataclass

@dataclass
class GPUStatus:
    id: int
    name: str
    memory_total_gb: float
    memory_used_gb: float
    memory_free_gb: float
    utilization_pct: float

class ResourceManager:
    def __init__(self, max_utilization_pct: float = 80.0, max_concurrent_jobs: int = 2):
        self.max_utilization_pct = max_utilization_pct
        self.max_concurrent_jobs = max_concurrent_jobs
        self.active_jobs: dict[int, str] = {}  # gpu_id -> job_id

    def get_gpu_status(self) -> list[GPUStatus]:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            gpus.append(GPUStatus(
                id=int(parts[0]), name=parts[1],
                memory_total_gb=float(parts[2]) / 1024,
                memory_used_gb=float(parts[3]) / 1024,
                memory_free_gb=float(parts[4]) / 1024,
                utilization_pct=float(parts[5]),
            ))
        return gpus

    def acquire_gpu(self, memory_required_gb: float = 8.0) -> int | None:
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            return None
        for gpu in self.get_gpu_status():
            if gpu.id not in self.active_jobs and gpu.memory_free_gb >= memory_required_gb:
                if gpu.utilization_pct < self.max_utilization_pct:
                    return gpu.id
        return None

    def release_gpu(self, gpu_id: int) -> None:
        self.active_jobs.pop(gpu_id, None)
```

### HPO Search Space Config

```yaml
# configs/training/hpo_space.yaml
study_name: fraud_detection_hpo
n_trials: 20
direction: maximize
metric: val_accuracy

search_space:
  learning_rate:
    type: float
    low: 0.00001
    high: 0.01
    log: true
  batch_size:
    type: categorical
    choices: [64, 128, 256, 512]
  hidden_dim:
    type: categorical
    choices: [64, 128, 256, 512]
  num_layers:
    type: int
    low: 2
    high: 8
  dropout:
    type: float
    low: 0.0
    high: 0.5
  weight_decay:
    type: float
    low: 0.000001
    high: 0.01
    log: true
```

### Evaluation Gate

```yaml
# configs/gates/production_gate.yaml
name: production_readiness
description: "Gate checks before a model can be registered for production"

checks:
  - metric: test_accuracy
    operator: ">="
    threshold: 0.85
  - metric: test_f1
    operator: ">="
    threshold: 0.80
  - metric: calibration_error
    operator: "<="
    threshold: 0.05
  - metric: inference_latency_p99_ms
    operator: "<="
    threshold: 50
  - metric: model_size_mb
    operator: "<="
    threshold: 500
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce batch size or model size. Check `nvidia-smi` for other processes. Use `torch.cuda.empty_cache()`. |
| Optuna study not saving | Verify SQLite path is writable. Try `storage="sqlite:///$(pwd)/optuna.db"` for absolute path. |
| Early stopping too aggressive | Increase patience (try 10-15) or decrease min_delta. Check if validation set is too small (noisy metrics). |
| GPU not detected | Verify `torch.cuda.is_available()`. Check CUDA drivers: `nvidia-smi`. Reinstall PyTorch with CUDA. |
| HPO trials all pruned | Reduce pruner aggressiveness: increase `n_warmup_steps`. Ensure objective function returns intermediate values with `trial.report()`. |
| Job queue stuck | Check for zombie processes: `ps aux | grep conduit`. Kill and restart the orchestrator. |

---

## Agent Handoff Template

```
I'm working on the Conduit project, Week 6: Training Orchestration.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB VRAM, 32GB RAM, Ubuntu.
Project root: ~/conduit/

Current state: [describe what's working/broken]

What I need help with: [specific issue]

Key files:
- src/conduit/training/orchestrator.py — Training job management
- src/conduit/training/hpo.py — Optuna HPO integration
- src/conduit/training/early_stopping.py — Early stopping logic
- src/conduit/training/resources.py — GPU resource management
- configs/training/ — Training and HPO configs
- configs/gates/ — Evaluation gate definitions

Infrastructure: RTX 5080 (16GB VRAM), Prefect (orchestration), MLflow (tracking), Optuna (HPO), PostgreSQL, MinIO.
Flow: submit job → acquire GPU → train with early stopping → evaluate → pass gate → register model.
```

---

## Out of Scope

- Multi-GPU training (DistributedDataParallel) — single GPU only
- Cloud training (AWS SageMaker, GCP Vertex) — local only
- Custom Optuna samplers or pruners
- Neural Architecture Search (NAS)
- Training on TPUs or other accelerators
- Federated learning or privacy-preserving training
- AutoML (automated feature selection + model selection + HPO combined)
