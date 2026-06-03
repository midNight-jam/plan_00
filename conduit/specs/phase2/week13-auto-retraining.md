# Week 13: Auto-Retraining Pipeline

## Context

**Where it fits:** Week 13 of Phase 2 (Deployment + Monitoring). Auto-retraining closes the loop — when monitoring detects degradation and feedback provides new labels, this pipeline automatically retrains, evaluates, and deploys an improved model without human intervention.

**Prerequisites:**
- Week 11 complete: drift detection triggers available
- Week 12 complete: feedback loops providing new labeled data
- Week 6 complete: training orchestration (HPO, evaluation gates)
- Week 8 complete: model registry with promotion workflow

**What it builds on:** Uses retraining triggers (Week 12) to know when to retrain. Uses training orchestration (Week 6) to execute training. Uses the model registry (Week 8) for promotion. Uses serving patterns (Week 9) for canary deployment. Uses monitoring (Week 11) to validate the new model in production.

**What comes next:** Week 14 (Consolidation) integrates all components into a seamless end-to-end demonstration.

---

## Learning Goals

- [ ] Understand automated ML operations: why human-triggered retraining is a bottleneck at scale
- [ ] Understand canary deployments: gradually rolling out a new model to limit blast radius
- [ ] Understand automated rollback: detecting failure fast and reverting without human intervention
- [ ] Understand the promotion decision: comparing a freshly trained model against the current production model
- [ ] Understand audit trails: why every automated decision needs to be logged and reproducible

---

## Implementation Goals

- [ ] Build trigger-to-training connector: drift/schedule/volume triggers automatically start training
- [ ] Implement automated training pipeline: new data → train → evaluate → register candidate
- [ ] Build promotion decision logic: candidate beats production on eval suite → auto-promote
- [ ] Implement rejection logic: candidate is worse → reject, keep production, alert team
- [ ] Build canary deployment: deploy new model to 5% traffic, expand gradually
- [ ] Implement automated rollback: metrics degrade during canary → rollback within 5 minutes
- [ ] Build full audit trail: every retraining logged with data version, config, results, decision
- [ ] Implement end-to-end orchestration: drift → retrain → deploy → monitor → confirm (or rollback)

---

## Acceptance Criteria

1. When drift trigger fires, a retraining job starts automatically within 2 minutes, using the latest labeled data.
2. Automated training uses the same HPO configuration as the original model (reproducibility from training config stored in registry).
3. Promotion decision correctly auto-promotes when candidate model beats production by >1% on primary metric across the evaluation dataset.
4. Rejection logic correctly rejects a worse model and sends an alert with the comparison report showing where it underperformed.
5. Canary deployment starts at 5% traffic and expands to 25% → 50% → 100% over 1 hour with no manual intervention (if metrics hold).
6. Automated rollback triggers within 5 minutes when canary metrics degrade by more than 10% relative to production baseline.
7. Audit trail for a complete retraining cycle contains: trigger reason, data version, training config, all trial metrics, promotion decision, deployment timeline, and monitoring confirmation.
8. Scheduled retraining (weekly) executes even when no drift is detected, training on the latest accumulated data.
9. Concurrent retraining requests are serialized — only one retraining pipeline runs at a time per model.
10. End-to-end: inject drift → trigger detected → auto-retrain → candidate wins evaluation → canary deploy → metrics stable → full promotion, all without manual intervention.

---

## Validation Commands

```bash
# Manually trigger a retraining pipeline
conduit retrain trigger --model fraud_detector --reason manual

# View retraining pipeline status
conduit retrain status --model fraud_detector

# View retraining history
conduit retrain history --model fraud_detector --last 10

# Start canary deployment
conduit retrain canary-start --model fraud_detector --version 4 --initial-pct 5

# Check canary status
conduit retrain canary-status --model fraud_detector

# Expand canary
conduit retrain canary-expand --model fraud_detector --target-pct 50

# Rollback canary
conduit retrain canary-rollback --model fraud_detector

# View audit trail
conduit retrain audit --model fraud_detector --run-id retrain_20250115

# View promotion decision
conduit retrain decision --model fraud_detector --run-id retrain_20250115

# Simulate drift trigger (for testing)
conduit retrain simulate --model fraud_detector --trigger drift

# Run tests
pytest tests/unit/retraining/ -v
pytest tests/integration/retraining/ -v --timeout=600
```

---

## Technical Implementation Details

### Project Structure (additions)

```
conduit/
├── src/conduit/
│   └── retraining/
│       ├── __init__.py
│       ├── pipeline.py          # End-to-end retraining orchestration
│       ├── trigger_handler.py   # Converts triggers into training jobs
│       ├── evaluator.py         # Candidate vs production comparison
│       ├── promoter.py          # Auto-promotion decision logic
│       ├── canary.py            # Canary deployment management
│       ├── rollback.py          # Automated rollback logic
│       └── audit.py             # Audit trail logging
├── configs/
│   └── retraining/
│       ├── pipeline.yaml
│       ├── canary.yaml
│       └── promotion_criteria.yaml
```

### Retraining Pipeline Orchestrator

```python
# src/conduit/retraining/pipeline.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from conduit.feedback.triggers import TriggerDecision, TriggerReason
from conduit.retraining.evaluator import CandidateEvaluator
from conduit.retraining.promoter import PromotionDecider
from conduit.retraining.canary import CanaryDeployment
from conduit.retraining.audit import AuditTrail

logger = logging.getLogger(__name__)

class PipelineStatus(Enum):
    TRIGGERED = "triggered"
    TRAINING = "training"
    EVALUATING = "evaluating"
    PROMOTING = "promoting"
    CANARY = "canary"
    COMPLETED = "completed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"

@dataclass
class RetrainingRun:
    run_id: str
    model_name: str
    trigger: TriggerDecision
    status: PipelineStatus
    started_at: datetime
    data_version: str | None = None
    candidate_version: int | None = None
    metrics: dict = field(default_factory=dict)
    decision: str | None = None
    completed_at: datetime | None = None

class RetrainingPipeline:
    def __init__(self, model_name: str, training_orchestrator, registry_manager,
                 evaluator: CandidateEvaluator, promoter: PromotionDecider,
                 canary: CanaryDeployment, audit: AuditTrail):
        self.model_name = model_name
        self.training = training_orchestrator
        self.registry = registry_manager
        self.evaluator = evaluator
        self.promoter = promoter
        self.canary = canary
        self.audit = audit
        self.active_run: RetrainingRun | None = None
        self._lock = False

    def execute(self, trigger: TriggerDecision) -> RetrainingRun:
        if self._lock:
            raise RuntimeError(f"Retraining already in progress for {self.model_name}")

        self._lock = True
        run = RetrainingRun(
            run_id=f"retrain_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            model_name=self.model_name,
            trigger=trigger,
            status=PipelineStatus.TRIGGERED,
            started_at=datetime.utcnow(),
        )
        self.active_run = run
        self.audit.log_event(run.run_id, "pipeline_started", {"trigger": trigger.reason.value})

        try:
            self._train(run)
            self._evaluate(run)
            decision = self._decide(run)

            if decision == "promote":
                self._deploy_canary(run)
            else:
                run.status = PipelineStatus.REJECTED
                run.decision = "rejected"
                self.audit.log_event(run.run_id, "candidate_rejected", run.metrics)
                logger.info(f"Candidate rejected: {run.metrics}")
        except Exception as e:
            run.status = PipelineStatus.FAILED
            self.audit.log_event(run.run_id, "pipeline_failed", {"error": str(e)})
            raise
        finally:
            run.completed_at = datetime.utcnow()
            self._lock = False

        return run

    def _train(self, run: RetrainingRun):
        run.status = PipelineStatus.TRAINING
        data_version = self._get_latest_data_version()
        run.data_version = data_version

        training_config = self._build_training_config(data_version)
        job_id = self.training.submit(training_config)
        result = self.training.wait_for_completion(job_id)

        candidate = self.registry.register(
            run_id=result.mlflow_run_id,
            model_name=self.model_name,
            description=f"Auto-retrained: {run.run_id}",
        )
        run.candidate_version = candidate.version
        self.audit.log_event(run.run_id, "training_complete", {
            "candidate_version": candidate.version, "data_version": data_version,
        })

    def _evaluate(self, run: RetrainingRun):
        run.status = PipelineStatus.EVALUATING
        production = self.registry.get_production_version(self.model_name)
        candidate = self.registry.get_version(self.model_name, run.candidate_version)

        comparison = self.evaluator.compare(candidate, production)
        run.metrics = comparison.to_dict()
        self.audit.log_event(run.run_id, "evaluation_complete", run.metrics)

    def _decide(self, run: RetrainingRun) -> str:
        decision = self.promoter.decide(run.metrics)
        run.decision = decision
        self.audit.log_event(run.run_id, "promotion_decision", {"decision": decision})
        return decision

    def _deploy_canary(self, run: RetrainingRun):
        run.status = PipelineStatus.CANARY
        self.canary.start(
            model_name=self.model_name,
            candidate_version=run.candidate_version,
            initial_pct=5,
        )
        self.audit.log_event(run.run_id, "canary_started", {"initial_pct": 5})

    def _get_latest_data_version(self) -> str:
        return f"v{datetime.utcnow().strftime('%Y%m%d')}"

    def _build_training_config(self, data_version: str) -> dict:
        prod_version = self.registry.get_production_version(self.model_name)
        base_config = prod_version.params if prod_version else {}
        return {
            **base_config,
            "data_version": data_version,
            "experiment_name": f"{self.model_name}_retrain",
            "hyperparameters": base_config.get("hyperparameters", {}),
        }
```

### Canary Deployment

```python
# src/conduit/retraining/canary.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)

class CanaryStatus(Enum):
    STARTING = "starting"
    RUNNING = "running"
    EXPANDING = "expanding"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"

@dataclass
class CanaryStage:
    traffic_pct: int
    duration_minutes: int
    started_at: datetime | None = None
    metrics: dict | None = None

@dataclass
class CanaryState:
    model_name: str
    candidate_version: int
    production_version: int
    status: CanaryStatus
    current_stage_idx: int
    stages: list[CanaryStage]
    started_at: datetime

class CanaryDeployment:
    def __init__(self, traffic_router, monitoring, rollback_threshold: float = 0.10):
        self.traffic_router = traffic_router
        self.monitoring = monitoring
        self.rollback_threshold = rollback_threshold
        self.state: CanaryState | None = None

    def start(self, model_name: str, candidate_version: int, initial_pct: int = 5):
        production = self.traffic_router.get_current_production(model_name)

        stages = [
            CanaryStage(traffic_pct=5, duration_minutes=15),
            CanaryStage(traffic_pct=25, duration_minutes=15),
            CanaryStage(traffic_pct=50, duration_minutes=15),
            CanaryStage(traffic_pct=100, duration_minutes=0),
        ]

        self.state = CanaryState(
            model_name=model_name,
            candidate_version=candidate_version,
            production_version=production.version,
            status=CanaryStatus.STARTING,
            current_stage_idx=0,
            stages=stages,
            started_at=datetime.utcnow(),
        )

        self._apply_traffic_split(initial_pct)
        self.state.status = CanaryStatus.RUNNING
        self.state.stages[0].started_at = datetime.utcnow()
        logger.info(f"Canary started: {model_name} v{candidate_version} at {initial_pct}%")

    def check_and_advance(self) -> CanaryStatus:
        if not self.state or self.state.status not in (CanaryStatus.RUNNING, CanaryStatus.EXPANDING):
            return self.state.status if self.state else CanaryStatus.ROLLED_BACK

        current_stage = self.state.stages[self.state.current_stage_idx]
        metrics = self.monitoring.get_canary_metrics(
            self.state.model_name, self.state.candidate_version,
        )

        if self._should_rollback(metrics):
            self.rollback()
            return CanaryStatus.ROLLED_BACK

        elapsed = datetime.utcnow() - current_stage.started_at
        if elapsed >= timedelta(minutes=current_stage.duration_minutes):
            current_stage.metrics = metrics
            self._advance_to_next_stage()

        return self.state.status

    def _should_rollback(self, canary_metrics: dict) -> bool:
        baseline = self.monitoring.get_production_baseline(self.state.model_name)
        for metric_name, canary_value in canary_metrics.items():
            baseline_value = baseline.get(metric_name, 0)
            if baseline_value > 0:
                degradation = (baseline_value - canary_value) / baseline_value
                if degradation > self.rollback_threshold:
                    logger.warning(
                        f"Canary degradation: {metric_name} dropped {degradation:.1%} "
                        f"(threshold: {self.rollback_threshold:.1%})"
                    )
                    return True
        return False

    def _advance_to_next_stage(self):
        self.state.current_stage_idx += 1
        if self.state.current_stage_idx >= len(self.state.stages):
            self.state.status = CanaryStatus.COMPLETED
            self._apply_traffic_split(100)
            logger.info(f"Canary completed: {self.state.model_name} v{self.state.candidate_version} fully deployed")
            return

        next_stage = self.state.stages[self.state.current_stage_idx]
        next_stage.started_at = datetime.utcnow()
        self._apply_traffic_split(next_stage.traffic_pct)
        self.state.status = CanaryStatus.EXPANDING
        logger.info(f"Canary expanding to {next_stage.traffic_pct}%")

    def _apply_traffic_split(self, candidate_pct: int):
        self.traffic_router.set_split(
            model_name=self.state.model_name,
            production_version=self.state.production_version,
            candidate_version=self.state.candidate_version,
            candidate_pct=candidate_pct,
        )

    def rollback(self):
        self.state.status = CanaryStatus.ROLLING_BACK
        self._apply_traffic_split(0)
        self.state.status = CanaryStatus.ROLLED_BACK
        logger.warning(f"Canary rolled back: {self.state.model_name} v{self.state.candidate_version}")
```

### Audit Trail

```python
# src/conduit/retraining/audit.py
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

@dataclass
class AuditEvent:
    run_id: str
    event_type: str
    timestamp: datetime
    data: dict

@dataclass
class AuditRecord:
    run_id: str
    model_name: str
    events: list[AuditEvent] = field(default_factory=list)

class AuditTrail:
    def __init__(self, storage_path: str = "data/audit"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, AuditRecord] = {}

    def log_event(self, run_id: str, event_type: str, data: dict):
        if run_id not in self.records:
            self.records[run_id] = AuditRecord(run_id=run_id, model_name="")
        event = AuditEvent(
            run_id=run_id,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            data=data,
        )
        self.records[run_id].events.append(event)
        self._persist(run_id)

    def get_record(self, run_id: str) -> AuditRecord | None:
        return self.records.get(run_id)

    def get_full_trail(self, run_id: str) -> list[dict]:
        record = self.records.get(run_id)
        if not record:
            return []
        return [
            {"event_type": e.event_type, "timestamp": e.timestamp.isoformat(), "data": e.data}
            for e in record.events
        ]

    def _persist(self, run_id: str):
        record = self.records[run_id]
        filepath = self.storage_path / f"{run_id}.json"
        data = {
            "run_id": run_id,
            "model_name": record.model_name,
            "events": [
                {"event_type": e.event_type, "timestamp": e.timestamp.isoformat(), "data": e.data}
                for e in record.events
            ],
        }
        filepath.write_text(json.dumps(data, indent=2))
```

### Promotion Criteria Config

```yaml
# configs/retraining/promotion_criteria.yaml
model: fraud_detector

comparison:
  primary_metric: test_accuracy
  direction: maximize
  min_improvement: 0.01  # Candidate must beat production by at least 1%

  secondary_metrics:
    - name: test_f1
      direction: maximize
      must_not_regress: true
    - name: inference_latency_p99_ms
      direction: minimize
      max_regression: 10  # Can be up to 10ms slower

canary:
  stages:
    - traffic_pct: 5
      duration_minutes: 15
    - traffic_pct: 25
      duration_minutes: 15
    - traffic_pct: 50
      duration_minutes: 15
    - traffic_pct: 100
      duration_minutes: 0
  rollback_threshold: 0.10  # 10% degradation triggers rollback
  rollback_timeout_minutes: 5

rejection:
  alert_on_rejection: true
  alert_channel: ml-team
  include_comparison_report: true
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Retraining never triggers | Check trigger thresholds — they may be too high. Use `conduit retrain simulate` to test. Verify monitoring is running. |
| Canary metrics unstable | Increase observation window (don't judge on 1 minute of data). Use rolling averages over 5-10 minutes. |
| Rollback too slow | Reduce monitoring check interval. Pre-compute rollback action (just flip traffic). Ensure traffic router responds instantly. |
| Candidate always worse than production | Check if training data includes recent drift. Verify evaluation dataset is representative. Model may need architecture changes (manual). |
| Concurrent retrain conflict | The pipeline uses a lock. If stuck, check for zombie processes. Add timeout to lock acquisition. |
| Audit trail incomplete | Ensure all pipeline stages log events, even on failure. Wrap each stage in try/finally. |

---

## Agent Handoff Template

```
I'm working on the Conduit project, Week 13: Auto-Retraining Pipeline.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB VRAM, 32GB RAM, Ubuntu.
Project root: ~/conduit/

Current state: [describe what's working/broken]

What I need help with: [specific issue]

Key files:
- src/conduit/retraining/pipeline.py — End-to-end retraining orchestration
- src/conduit/retraining/canary.py — Canary deployment management
- src/conduit/retraining/rollback.py — Automated rollback logic
- src/conduit/retraining/audit.py — Full audit trail
- src/conduit/retraining/evaluator.py — Candidate vs production comparison
- configs/retraining/ — Pipeline, canary, and promotion config

Infrastructure: Prefect (orchestration), MLflow (registry), FastAPI (serving), PostgreSQL, monitoring stack.
Flow: trigger → train → evaluate → decide → canary deploy → monitor → promote/rollback.
```

---

## Out of Scope

- Blue-green deployment (canary with gradual rollout only)
- Multi-model retraining coordination (one model at a time)
- Transfer learning from production model (train from scratch each time)
- Automated hyperparameter re-optimization (uses same config as original)
- Retraining on distributed compute (single machine only)
- Shadow deployment as alternative to canary
- Cost-aware retraining (training budget constraints)
