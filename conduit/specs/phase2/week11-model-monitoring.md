# Week 11: Model Monitoring and Drift Detection

## Context

**Where it fits:** Week 11 of Phase 2 (Deployment + Monitoring). Once a model is in production, its performance silently degrades as the world changes. Monitoring detects this degradation before it causes business impact.

**Prerequisites:**
- Week 9 complete: model serving operational with prediction logging
- Week 10 complete: metric collection infrastructure from A/B testing
- Understanding of statistical distributions and hypothesis testing
- Training data distributions accessible for comparison

**What it builds on:** Uses prediction logs from the serving layer (Week 9). Uses metric infrastructure from A/B testing (Week 10). References training data from the feature store (Week 2) and data versioning (Week 3) for baseline distributions.

**What comes next:** Week 12 (Feedback Loops) collects ground truth labels to confirm whether detected drift actually impacts model performance.

---

## Learning Goals

- [ ] Understand data drift: why input distributions change and how it impacts model predictions
- [ ] Understand prediction drift: monitoring output distributions as an early warning signal
- [ ] Understand statistical tests for drift: KS test, PSI, JS divergence, and when to use each
- [ ] Understand monitoring architecture: what to monitor, at what frequency, and how to alert
- [ ] Understand the difference between drift detection and performance degradation

---

## Implementation Goals

- [ ] Build data drift detector: monitor input feature distributions vs training baseline
- [ ] Build prediction drift detector: monitor output distribution changes over time
- [ ] Implement performance tracker: track accuracy/precision/recall when ground truth arrives
- [ ] Implement feature importance monitoring: track which features drive predictions over time
- [ ] Build statistical test suite: KS test, PSI, JS divergence with configurable thresholds
- [ ] Build alert pipeline: drift detected → severity classification → alert → investigation workflow
- [ ] Create monitoring dashboard: distributions, drift scores, alerts timeline
- [ ] Implement baseline management: store and update reference distributions

---

## Acceptance Criteria

1. Data drift detector identifies synthetic drift injection within 1 hour (feature distribution shifted by 0.5 standard deviations).
2. KS test correctly detects distribution shift with p-value < 0.05 when applied to drifted feature data vs training baseline.
3. PSI calculation returns values >0.2 (significant drift) when input feature distribution changes substantially from training.
4. Prediction drift alert fires when output distribution diverges from baseline by more than the configured JS divergence threshold.
5. Feature importance monitoring tracks top-10 feature importances weekly and alerts when rankings change significantly.
6. Alert pipeline delivers notification within 5 minutes of drift threshold breach, including affected features and severity level.
7. Monitoring dashboard displays feature distribution histograms (current vs training), drift score timelines, and alert history.
8. Performance tracker correctly computes accuracy degradation when delayed ground truth labels arrive (comparing predicted vs actual).
9. Baseline management stores training distributions and supports manual re-baselining after approved model updates.
10. End-to-end: inject drift into input data → detector fires → alert created → dashboard shows drift → investigation workflow triggered.

---

## Validation Commands

```bash
# Start monitoring for a model
conduit monitor start --model fraud_detector --stage production

# View current drift scores
conduit monitor drift-report --model fraud_detector

# Check specific feature drift
conduit monitor feature-drift --model fraud_detector --feature amount_mean_7d

# View prediction distribution
conduit monitor prediction-dist --model fraud_detector --window 24h

# Run drift detection manually
conduit monitor detect --model fraud_detector --reference training_v3

# View feature importance changes
conduit monitor feature-importance --model fraud_detector --compare-to baseline

# List active alerts
conduit monitor alerts --status open

# Acknowledge an alert
conduit monitor alert-ack --alert-id drift_001

# Inject synthetic drift (for testing)
conduit monitor inject-drift --feature amount_mean_7d --shift 0.5 --duration 2h

# Update baseline
conduit monitor update-baseline --model fraud_detector --reference latest_training_data

# Run tests
pytest tests/unit/monitoring/ -v
pytest tests/integration/monitoring/ -v --timeout=180
```

---

## Technical Implementation Details

### Project Structure (additions)

```
conduit/
├── src/conduit/
│   └── monitoring/
│       ├── __init__.py
│       ├── drift.py             # Drift detection engine
│       ├── statistical_tests.py # KS, PSI, JS divergence
│       ├── prediction_monitor.py # Output distribution monitoring
│       ├── performance.py       # Accuracy tracking with ground truth
│       ├── feature_importance.py # Feature importance over time
│       ├── alerts.py            # Alert pipeline
│       ├── baseline.py          # Reference distribution management
│       └── dashboard.py         # Monitoring dashboard data
├── configs/
│   └── monitoring/
│       ├── drift_thresholds.yaml
│       └── alert_rules.yaml
```

### Statistical Tests

```python
# src/conduit/monitoring/statistical_tests.py
import numpy as np
from dataclasses import dataclass
from scipy import stats

@dataclass
class DriftTestResult:
    test_name: str
    statistic: float
    p_value: float | None
    drifted: bool
    severity: str  # "none", "low", "medium", "high"
    feature_name: str

class StatisticalTests:
    def ks_test(self, reference: np.ndarray, current: np.ndarray, feature_name: str,
                threshold: float = 0.05) -> DriftTestResult:
        statistic, p_value = stats.ks_2samp(reference, current)
        drifted = p_value < threshold
        severity = self._classify_severity_by_pvalue(p_value)
        return DriftTestResult(
            test_name="kolmogorov_smirnov",
            statistic=statistic,
            p_value=p_value,
            drifted=drifted,
            severity=severity,
            feature_name=feature_name,
        )

    def psi(self, reference: np.ndarray, current: np.ndarray, feature_name: str,
            n_bins: int = 10) -> DriftTestResult:
        """Population Stability Index."""
        breakpoints = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            n_bins + 1,
        )
        ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
        cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current)

        ref_counts = np.clip(ref_counts, 1e-6, None)
        cur_counts = np.clip(cur_counts, 1e-6, None)

        psi_value = float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))

        if psi_value < 0.1:
            severity = "none"
        elif psi_value < 0.2:
            severity = "low"
        elif psi_value < 0.3:
            severity = "medium"
        else:
            severity = "high"

        return DriftTestResult(
            test_name="psi",
            statistic=psi_value,
            p_value=None,
            drifted=psi_value >= 0.2,
            severity=severity,
            feature_name=feature_name,
        )

    def js_divergence(self, reference: np.ndarray, current: np.ndarray, feature_name: str,
                      n_bins: int = 50, threshold: float = 0.1) -> DriftTestResult:
        """Jensen-Shannon divergence."""
        breakpoints = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            n_bins + 1,
        )
        ref_hist = np.histogram(reference, bins=breakpoints, density=True)[0]
        cur_hist = np.histogram(current, bins=breakpoints, density=True)[0]

        ref_hist = ref_hist / ref_hist.sum()
        cur_hist = cur_hist / cur_hist.sum()

        m = 0.5 * (ref_hist + cur_hist)
        js = 0.5 * (stats.entropy(ref_hist, m) + stats.entropy(cur_hist, m))
        js_value = float(np.sqrt(js))  # JS distance

        drifted = js_value > threshold
        severity = self._classify_severity_continuous(js_value, [0.05, 0.1, 0.2])

        return DriftTestResult(
            test_name="js_divergence",
            statistic=js_value,
            p_value=None,
            drifted=drifted,
            severity=severity,
            feature_name=feature_name,
        )

    def _classify_severity_by_pvalue(self, p_value: float) -> str:
        if p_value >= 0.05:
            return "none"
        elif p_value >= 0.01:
            return "low"
        elif p_value >= 0.001:
            return "medium"
        return "high"

    def _classify_severity_continuous(self, value: float, thresholds: list[float]) -> str:
        levels = ["none", "low", "medium", "high"]
        for i, t in enumerate(thresholds):
            if value < t:
                return levels[i]
        return levels[-1]
```

### Drift Detection Engine

```python
# src/conduit/monitoring/drift.py
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from conduit.monitoring.statistical_tests import StatisticalTests, DriftTestResult
from conduit.monitoring.baseline import BaselineStore

@dataclass
class DriftReport:
    model_name: str
    timestamp: datetime
    window_hours: int
    feature_results: list[DriftTestResult]
    overall_drifted: bool
    drifted_features: list[str]

class DriftDetector:
    def __init__(self, model_name: str, baseline_store: BaselineStore,
                 window_hours: int = 24, check_interval_minutes: int = 60):
        self.model_name = model_name
        self.baseline_store = baseline_store
        self.window_hours = window_hours
        self.check_interval_minutes = check_interval_minutes
        self.tests = StatisticalTests()
        self.history: list[DriftReport] = []

    def detect(self, current_data: pd.DataFrame) -> DriftReport:
        baseline = self.baseline_store.get_baseline(self.model_name)
        feature_results = []

        for feature_name in baseline.feature_names:
            if feature_name not in current_data.columns:
                continue

            ref_values = baseline.get_feature_distribution(feature_name)
            cur_values = current_data[feature_name].dropna().values

            if len(cur_values) < 100:
                continue

            ks_result = self.tests.ks_test(ref_values, cur_values, feature_name)
            psi_result = self.tests.psi(ref_values, cur_values, feature_name)

            worst = max([ks_result, psi_result], key=lambda r: r.drifted)
            feature_results.append(worst)

        drifted_features = [r.feature_name for r in feature_results if r.drifted]

        report = DriftReport(
            model_name=self.model_name,
            timestamp=datetime.utcnow(),
            window_hours=self.window_hours,
            feature_results=feature_results,
            overall_drifted=len(drifted_features) > 0,
            drifted_features=drifted_features,
        )
        self.history.append(report)
        return report

    def get_drift_trend(self, feature_name: str, lookback_days: int = 7) -> list[dict]:
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        trend = []
        for report in self.history:
            if report.timestamp < cutoff:
                continue
            for result in report.feature_results:
                if result.feature_name == feature_name:
                    trend.append({
                        "timestamp": report.timestamp.isoformat(),
                        "statistic": result.statistic,
                        "drifted": result.drifted,
                        "severity": result.severity,
                    })
        return trend
```

### Alert Pipeline

```python
# src/conduit/monitoring/alerts.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertStatus(Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"

@dataclass
class Alert:
    alert_id: str
    model_name: str
    severity: AlertSeverity
    title: str
    description: str
    created_at: datetime
    status: AlertStatus = AlertStatus.OPEN
    metadata: dict = field(default_factory=dict)

class AlertPipeline:
    def __init__(self, notification_backends: list | None = None):
        self.alerts: list[Alert] = []
        self.backends = notification_backends or []

    def create_alert(self, model_name: str, severity: AlertSeverity,
                     title: str, description: str, metadata: dict = None) -> Alert:
        alert = Alert(
            alert_id=f"alert_{len(self.alerts):04d}",
            model_name=model_name,
            severity=severity,
            title=title,
            description=description,
            created_at=datetime.utcnow(),
            metadata=metadata or {},
        )
        self.alerts.append(alert)
        self._notify(alert)
        logger.warning(f"Alert created: [{severity.value}] {title}")
        return alert

    def from_drift_report(self, report) -> Alert | None:
        if not report.overall_drifted:
            return None

        n_drifted = len(report.drifted_features)
        severity = AlertSeverity.CRITICAL if n_drifted > 3 else AlertSeverity.WARNING

        return self.create_alert(
            model_name=report.model_name,
            severity=severity,
            title=f"Data drift detected: {n_drifted} features drifted",
            description=f"Features: {', '.join(report.drifted_features)}",
            metadata={"drifted_features": report.drifted_features, "window_hours": report.window_hours},
        )

    def acknowledge(self, alert_id: str) -> None:
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                break

    def get_open_alerts(self, model_name: str = None) -> list[Alert]:
        alerts = [a for a in self.alerts if a.status == AlertStatus.OPEN]
        if model_name:
            alerts = [a for a in alerts if a.model_name == model_name]
        return alerts

    def _notify(self, alert: Alert):
        for backend in self.backends:
            backend.send(alert)
```

### Baseline Store

```python
# src/conduit/monitoring/baseline.py
from dataclasses import dataclass
import numpy as np
import json
from pathlib import Path

@dataclass
class FeatureBaseline:
    feature_names: list[str]
    distributions: dict[str, np.ndarray]
    metadata: dict

    def get_feature_distribution(self, feature_name: str) -> np.ndarray:
        return self.distributions[feature_name]

class BaselineStore:
    def __init__(self, storage_path: str = "data/baselines"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save_baseline(self, model_name: str, data: dict[str, np.ndarray], metadata: dict = None):
        model_dir = self.storage_path / model_name
        model_dir.mkdir(exist_ok=True)

        for feature_name, values in data.items():
            np.save(model_dir / f"{feature_name}.npy", values)

        meta = {"feature_names": list(data.keys()), **(metadata or {})}
        (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    def get_baseline(self, model_name: str) -> FeatureBaseline:
        model_dir = self.storage_path / model_name
        meta = json.loads((model_dir / "metadata.json").read_text())

        distributions = {}
        for feature_name in meta["feature_names"]:
            distributions[feature_name] = np.load(model_dir / f"{feature_name}.npy")

        return FeatureBaseline(
            feature_names=meta["feature_names"],
            distributions=distributions,
            metadata=meta,
        )
```

### Monitoring Config

```yaml
# configs/monitoring/drift_thresholds.yaml
model: fraud_detector
check_interval_minutes: 60
window_hours: 24

feature_thresholds:
  default:
    ks_pvalue: 0.05
    psi: 0.2
    js_divergence: 0.1
  amount_mean_7d:
    psi: 0.15  # More sensitive for this critical feature

prediction_thresholds:
  js_divergence: 0.08
  mean_shift_std: 1.5

alert_rules:
  - condition: "drifted_features_count > 3"
    severity: critical
    action: page_oncall
  - condition: "drifted_features_count > 0"
    severity: warning
    action: slack_channel
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| KS test always significant | With large sample sizes, KS test is overly sensitive. Use PSI or JS divergence with practical thresholds instead of p-values alone. |
| PSI returns infinity | One bin has zero count. Apply Laplace smoothing (add small epsilon to all bins). Check `np.clip(counts, 1e-6, None)`. |
| Too many false alerts | Increase window size (more data = more stable distributions). Require multiple consecutive violations before alerting. |
| Baseline stale after model update | Always update baseline when a new model is promoted. Link baseline version to model version in registry. |
| Feature store values differ from training | Training-serving skew. Ensure online feature computation matches offline. Log and compare distributions. |
| Monitoring too slow | Subsample current data for drift checks. Don't need all records — 10,000 samples is sufficient for statistical tests. |

---

## Agent Handoff Template

```
I'm working on the Conduit project, Week 11: Model Monitoring and Drift Detection.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB VRAM, 32GB RAM, Ubuntu.
Project root: ~/conduit/

Current state: [describe what's working/broken]

What I need help with: [specific issue]

Key files:
- src/conduit/monitoring/drift.py — Drift detection engine
- src/conduit/monitoring/statistical_tests.py — KS, PSI, JS divergence
- src/conduit/monitoring/alerts.py — Alert pipeline
- src/conduit/monitoring/baseline.py — Reference distribution storage
- src/conduit/monitoring/performance.py — Accuracy tracking
- configs/monitoring/ — Thresholds and alert rules

Infrastructure: PostgreSQL (metrics storage), prediction logs from serving layer, training data baselines.
Flow: prediction logged → aggregate window → compare to baseline → statistical test → drift score → alert if threshold breached.
```

---

## Out of Scope

- Concept drift (changes in P(Y|X)) without ground truth — only data drift (P(X)) and prediction drift
- Real-time streaming drift detection (Apache Kafka/Flink) — batch window-based only
- Automated root cause analysis (why drift happened)
- Multi-model correlated drift detection
- Drift detection for unstructured data (images, text)
- Integration with PagerDuty, Opsgenie, or cloud alerting services
- Time-series anomaly detection for model metrics
