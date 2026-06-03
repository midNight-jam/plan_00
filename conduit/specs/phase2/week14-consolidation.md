# Week 14: Phase 2 Consolidation

## Context

**Where it fits:** Week 14, the final week of Phase 2 (Deployment + Monitoring). This is the integration and consolidation week — proving that all components work together as a self-healing ML system, documenting decisions, and creating a full lifecycle demonstration.

**Prerequisites:**
- Weeks 8–13 complete: model registry, serving, A/B testing, monitoring, feedback loops, auto-retraining
- All individual components passing their unit and integration tests
- At least one model trained, registered, and serving predictions

**What it builds on:** Every component from Phase 2 is exercised in the full lifecycle demo. Monitoring (Week 11) detects problems, feedback (Week 12) provides data, and auto-retraining (Week 13) fixes the model — all without human intervention.

**What comes next:** Phase 3 (if planned) would cover advanced topics like distributed serving, multi-model systems, or ML platform engineering.

---

## Learning Goals

- [ ] Understand end-to-end ML lifecycle: how all components interact in a production system
- [ ] Understand incident response for ML: systematically investigating and resolving model degradation
- [ ] Understand system resilience: how a self-healing ML pipeline recovers from data drift autonomously
- [ ] Understand architecture documentation: capturing decisions, tradeoffs, and rationale in ADRs
- [ ] Understand operational readiness: what it takes to run an ML system reliably

---

## Implementation Goals

- [ ] Build full lifecycle integration test: data → features → drift → retrain → deploy → verify
- [ ] Implement ML incident response runbook: steps to investigate and fix model performance issues
- [ ] Write technical blog post: "Building Self-Healing ML Systems"
- [ ] Build comprehensive integration test suite for the deployment pipeline
- [ ] Write ADRs: monitoring thresholds, canary vs blue-green, serving pattern choice
- [ ] Complete documentation and code cleanup across all Phase 2 modules
- [ ] Create system architecture diagram showing all component interactions
- [ ] Implement health check dashboard: single view of entire ML pipeline status

---

## Acceptance Criteria

1. Full lifecycle demo runs end-to-end unattended: inject drift → monitoring detects it → retraining triggers → new model trains → passes evaluation → canary deploys → metrics hold → full promotion (under 2 hours total).
2. ML incident response runbook successfully guides resolution of a simulated "model accuracy dropped 10%" incident within 30 minutes.
3. Blog post is complete (2000+ words) covering architecture, design decisions, and lessons learned with code examples and diagrams.
4. Integration test suite covers the complete deployment pipeline (registry → serve → monitor → retrain → promote) and passes reliably.
5. At least 3 ADRs are written: (a) why these drift thresholds, (b) canary vs blue-green deployment, (c) online vs batch serving choice.
6. All Phase 2 modules have docstrings, type hints, and a module-level README explaining purpose and usage.
7. System architecture diagram correctly shows all 7 components (registry, serving, A/B, monitoring, feedback, retraining, audit) and their interactions.
8. Health check dashboard shows green/red status for each pipeline component and last-known-good timestamps.
9. `conduit system status` command outputs the health of all subsystems in a single view.
10. Clean git history: each Phase 2 week is a logical commit group, no dead code, no TODO comments without linked issues.

---

## Validation Commands

```bash
# Run full lifecycle demo
conduit demo lifecycle --model fraud_detector --inject-drift --unattended

# Run integration test suite
pytest tests/integration/lifecycle/ -v --timeout=7200

# Check system health
conduit system status

# Run incident response simulation
conduit incident simulate --scenario accuracy_drop_10pct

# Validate all components are connected
conduit system connectivity-check

# Generate architecture diagram
conduit docs generate-architecture --output docs/architecture.svg

# Validate documentation coverage
conduit docs check-coverage --modules registry,serving,experiments,monitoring,feedback,retraining

# Run all Phase 2 tests
pytest tests/ -v --timeout=3600 -m "phase2"

# Lint and type check
ruff check src/conduit/
mypy src/conduit/ --strict

# Generate test coverage report
pytest tests/ --cov=src/conduit --cov-report=html
```

---

## Technical Implementation Details

### Project Structure (additions)

```
conduit/
├── src/conduit/
│   └── system/
│       ├── __init__.py
│       ├── lifecycle.py         # Full lifecycle orchestration
│       ├── health.py            # Health check aggregation
│       ├── incident.py          # Incident response automation
│       └── status.py            # System status reporting
├── docs/
│   ├── architecture.md          # System architecture
│   ├── adrs/
│   │   ├── 001-drift-thresholds.md
│   │   ├── 002-canary-vs-bluegreen.md
│   │   └── 003-serving-pattern.md
│   ├── runbooks/
│   │   └── model-performance-degradation.md
│   └── blog/
│       └── self-healing-ml-systems.md
├── tests/
│   └── integration/
│       └── lifecycle/
│           ├── test_full_cycle.py
│           ├── test_incident_response.py
│           └── conftest.py
```

### Full Lifecycle Test

```python
# tests/integration/lifecycle/test_full_cycle.py
import pytest
import time
from datetime import datetime, timedelta
from conduit.monitoring.drift import DriftDetector
from conduit.retraining.pipeline import RetrainingPipeline, PipelineStatus
from conduit.registry.manager import ModelRegistryManager, ModelStage
from conduit.serving.online import app as serving_app
from conduit.feedback.triggers import TriggerReason

@pytest.fixture
def ml_system(configured_system):
    """Provides a fully configured ML system with all components wired together."""
    return configured_system

class TestFullLifecycle:
    @pytest.mark.timeout(7200)
    def test_drift_to_promotion(self, ml_system):
        """End-to-end: drift → retrain → promote."""
        registry = ml_system.registry
        monitoring = ml_system.monitoring
        retraining = ml_system.retraining

        initial_prod = registry.get_production_version("fraud_detector")
        assert initial_prod is not None

        ml_system.inject_drift(feature="amount_mean_7d", shift_std=1.0)

        drift_detected = self._wait_for_condition(
            lambda: monitoring.latest_drift_report.overall_drifted,
            timeout_seconds=300,
            poll_interval=10,
        )
        assert drift_detected, "Drift not detected within 5 minutes"

        retrain_started = self._wait_for_condition(
            lambda: retraining.active_run is not None,
            timeout_seconds=180,
            poll_interval=5,
        )
        assert retrain_started, "Retraining not triggered within 3 minutes"

        retrain_complete = self._wait_for_condition(
            lambda: retraining.active_run.status in (
                PipelineStatus.COMPLETED, PipelineStatus.REJECTED, PipelineStatus.ROLLED_BACK
            ),
            timeout_seconds=3600,
            poll_interval=30,
        )
        assert retrain_complete, "Retraining not completed within 1 hour"

        final_prod = registry.get_production_version("fraud_detector")
        run = retraining.active_run

        if run.status == PipelineStatus.COMPLETED:
            assert final_prod.version > initial_prod.version
        elif run.status == PipelineStatus.REJECTED:
            assert final_prod.version == initial_prod.version

    def test_canary_rollback_on_degradation(self, ml_system):
        """Canary deployment rolls back when metrics degrade."""
        canary = ml_system.canary

        canary.start(model_name="fraud_detector", candidate_version=99, initial_pct=5)

        ml_system.inject_metric_degradation(metric="accuracy", drop=0.15)

        rolled_back = self._wait_for_condition(
            lambda: canary.state.status.value == "rolled_back",
            timeout_seconds=300,
            poll_interval=5,
        )
        assert rolled_back, "Canary not rolled back within 5 minutes"

    def _wait_for_condition(self, condition_fn, timeout_seconds: int, poll_interval: int) -> bool:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if condition_fn():
                return True
            time.sleep(poll_interval)
        return False
```

### System Health Check

```python
# src/conduit/system/health.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class ComponentStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class ComponentHealth:
    name: str
    status: ComponentStatus
    last_check: datetime
    message: str
    details: dict | None = None

@dataclass
class SystemHealth:
    overall_status: ComponentStatus
    components: list[ComponentHealth]
    checked_at: datetime

class HealthChecker:
    def __init__(self, registry, serving, monitoring, feedback, retraining):
        self.checks = {
            "model_registry": self._check_registry,
            "serving_endpoint": self._check_serving,
            "monitoring": self._check_monitoring,
            "feedback_pipeline": self._check_feedback,
            "retraining_pipeline": self._check_retraining,
            "feature_store": self._check_feature_store,
            "experiment_tracker": self._check_tracker,
        }
        self.registry = registry
        self.serving = serving
        self.monitoring = monitoring
        self.feedback = feedback
        self.retraining = retraining

    def check_all(self) -> SystemHealth:
        components = []
        for name, check_fn in self.checks.items():
            try:
                health = check_fn()
            except Exception as e:
                health = ComponentHealth(
                    name=name, status=ComponentStatus.UNHEALTHY,
                    last_check=datetime.utcnow(), message=str(e),
                )
            components.append(health)

        overall = self._compute_overall(components)
        return SystemHealth(overall_status=overall, components=components, checked_at=datetime.utcnow())

    def _check_registry(self) -> ComponentHealth:
        try:
            models = self.registry.client.search_registered_models()
            return ComponentHealth(
                name="model_registry", status=ComponentStatus.HEALTHY,
                last_check=datetime.utcnow(), message=f"{len(list(models))} models registered",
            )
        except Exception as e:
            return ComponentHealth(
                name="model_registry", status=ComponentStatus.UNHEALTHY,
                last_check=datetime.utcnow(), message=f"Registry unreachable: {e}",
            )

    def _check_serving(self) -> ComponentHealth:
        import httpx
        try:
            resp = httpx.get("http://localhost:8080/health", timeout=5)
            if resp.status_code == 200:
                return ComponentHealth(
                    name="serving_endpoint", status=ComponentStatus.HEALTHY,
                    last_check=datetime.utcnow(), message="Endpoint responding",
                )
            return ComponentHealth(
                name="serving_endpoint", status=ComponentStatus.DEGRADED,
                last_check=datetime.utcnow(), message=f"Status {resp.status_code}",
            )
        except Exception as e:
            return ComponentHealth(
                name="serving_endpoint", status=ComponentStatus.UNHEALTHY,
                last_check=datetime.utcnow(), message=f"Endpoint unreachable: {e}",
            )

    def _check_monitoring(self) -> ComponentHealth:
        last_report = self.monitoring.get_latest_report()
        if last_report and (datetime.utcnow() - last_report.timestamp) < timedelta(hours=2):
            return ComponentHealth(
                name="monitoring", status=ComponentStatus.HEALTHY,
                last_check=datetime.utcnow(), message=f"Last check: {last_report.timestamp.isoformat()}",
            )
        return ComponentHealth(
            name="monitoring", status=ComponentStatus.DEGRADED,
            last_check=datetime.utcnow(), message="No recent drift checks",
        )

    def _check_feedback(self) -> ComponentHealth:
        stats = self.feedback.get_stats(window_hours=24)
        if stats.total_labels > 0:
            return ComponentHealth(
                name="feedback_pipeline", status=ComponentStatus.HEALTHY,
                last_check=datetime.utcnow(), message=f"{stats.total_labels} labels in last 24h",
            )
        return ComponentHealth(
            name="feedback_pipeline", status=ComponentStatus.DEGRADED,
            last_check=datetime.utcnow(), message="No labels collected in 24h",
        )

    def _check_retraining(self) -> ComponentHealth:
        return ComponentHealth(
            name="retraining_pipeline", status=ComponentStatus.HEALTHY,
            last_check=datetime.utcnow(), message="Ready (no active run)",
        )

    def _check_feature_store(self) -> ComponentHealth:
        return ComponentHealth(
            name="feature_store", status=ComponentStatus.HEALTHY,
            last_check=datetime.utcnow(), message="Feature store operational",
        )

    def _check_tracker(self) -> ComponentHealth:
        return ComponentHealth(
            name="experiment_tracker", status=ComponentStatus.HEALTHY,
            last_check=datetime.utcnow(), message="MLflow reachable",
        )

    def _compute_overall(self, components: list[ComponentHealth]) -> ComponentStatus:
        statuses = [c.status for c in components]
        if any(s == ComponentStatus.UNHEALTHY for s in statuses):
            return ComponentStatus.UNHEALTHY
        if any(s == ComponentStatus.DEGRADED for s in statuses):
            return ComponentStatus.DEGRADED
        return ComponentStatus.HEALTHY
```

### Incident Response

```python
# src/conduit/system/incident.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class IncidentSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IncidentStep(Enum):
    DETECT = "detect"
    TRIAGE = "triage"
    INVESTIGATE = "investigate"
    MITIGATE = "mitigate"
    RESOLVE = "resolve"
    POSTMORTEM = "postmortem"

@dataclass
class Investigation:
    step: IncidentStep
    finding: str
    data: dict
    timestamp: datetime = field(default_factory=datetime.utcnow)

class MLIncidentResponse:
    def __init__(self, monitoring, registry, serving, audit):
        self.monitoring = monitoring
        self.registry = registry
        self.serving = serving
        self.audit = audit

    def investigate_performance_drop(self, model_name: str) -> list[Investigation]:
        findings = []

        drift_report = self.monitoring.get_latest_report()
        findings.append(Investigation(
            step=IncidentStep.DETECT,
            finding=f"Drift detected: {drift_report.drifted_features}" if drift_report.overall_drifted else "No drift detected",
            data={"drifted_features": drift_report.drifted_features},
        ))

        prod_model = self.registry.get_production_version(model_name)
        findings.append(Investigation(
            step=IncidentStep.TRIAGE,
            finding=f"Production model: v{prod_model.version}, deployed {prod_model.tags.get('deployed_at', 'unknown')}",
            data={"version": prod_model.version, "metrics": prod_model.metrics},
        ))

        serving_health = self.serving.get_health()
        findings.append(Investigation(
            step=IncidentStep.INVESTIGATE,
            finding=f"Serving latency p99: {serving_health.get('latency_p99_ms', 'unknown')}ms",
            data=serving_health,
        ))

        if drift_report.overall_drifted:
            findings.append(Investigation(
                step=IncidentStep.MITIGATE,
                finding="Recommendation: trigger retraining with latest data",
                data={"action": "retrain", "priority": "high"},
            ))
        else:
            findings.append(Investigation(
                step=IncidentStep.MITIGATE,
                finding="Recommendation: check for upstream data quality issues",
                data={"action": "investigate_data", "priority": "medium"},
            ))

        return findings
```

### ADR Template

```markdown
# ADR-001: Drift Detection Thresholds

## Status
Accepted

## Context
We need to decide on thresholds for triggering drift alerts. Too sensitive means
alert fatigue; too lenient means missing real drift until model performance degrades.

## Decision
- PSI threshold: 0.2 (industry standard for "significant change")
- KS test: p < 0.05 but only alert if 3+ features drift simultaneously
- Prediction drift (JS divergence): 0.1
- Check frequency: hourly
- Alert only after 2 consecutive violations (debounce)

## Rationale
- PSI 0.2 is well-established in credit risk modeling as the boundary for action
- Requiring multiple features prevents false positives from single noisy features
- Hourly checks balance detection speed with computational cost
- Debouncing prevents alerts from transient anomalies (batch jobs, time-of-day effects)

## Consequences
- May miss slow drift that affects individual features below threshold
- 2-hour minimum detection time (2 consecutive hourly checks)
- Acceptable tradeoff: prefer fewer false alarms over faster detection
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Lifecycle test flaky | Add longer timeouts and retry logic. ML operations are inherently variable. Use `pytest-rerunfailures` for CI. |
| Components not connecting | Run `conduit system connectivity-check`. Verify all services are running. Check environment variables for URIs. |
| Incident simulation not realistic | Use recorded production traces instead of synthetic data. Replay actual drift events from monitoring history. |
| Architecture diagram out of date | Generate from code annotations. Use `conduit docs generate-architecture` which introspects imports and config. |
| Tests too slow | Parallelize independent tests with `pytest-xdist`. Mock external services (MLflow, feature store) in unit tests. |
| Documentation incomplete | Use `conduit docs check-coverage` to find undocumented modules. Prioritize public APIs and config files. |

---

## Agent Handoff Template

```
I'm working on the Conduit project, Week 14: Phase 2 Consolidation.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB VRAM, 32GB RAM, Ubuntu.
Project root: ~/conduit/

Current state: [describe what's working/broken]

What I need help with: [specific issue]

Key files:
- src/conduit/system/lifecycle.py — Full lifecycle orchestration
- src/conduit/system/health.py — Health check aggregation
- src/conduit/system/incident.py — Incident response automation
- tests/integration/lifecycle/ — End-to-end integration tests
- docs/adrs/ — Architecture Decision Records
- docs/runbooks/ — Operational runbooks

Infrastructure: All Phase 2 components (MLflow, FastAPI, PostgreSQL, Redis, Docker, Prefect).
Goal: Prove all components work together as a self-healing ML system.
```

---

## Out of Scope

- Production-grade observability stack (Grafana, Prometheus, Datadog)
- SLA/SLO definition and enforcement
- On-call rotation and escalation policies
- Chaos engineering (fault injection beyond drift simulation)
- Performance benchmarking against industry baselines
- Security audit (model access control, data privacy)
- Cost optimization (compute budget management)
- Multi-team collaboration workflows (model review boards)
