# Week 10: A/B Testing Framework

## Context

**Where it fits:** Week 10 of Phase 2 (Deployment + Monitoring). A/B testing is how you make data-driven decisions about model deployments — proving that a new model actually improves business outcomes before fully committing to it.

**Prerequisites:**
- Week 9 complete: model serving patterns (online endpoint operational)
- Understanding of basic statistics (p-values, confidence intervals)
- Feature store and model registry operational
- Multiple model versions available for comparison

**What it builds on:** Uses the serving layer (Week 9) to route traffic between model variants. Uses the model registry (Week 8) to load specific model versions. Experiments are logged to the experiment tracker (Week 5).

**What comes next:** Week 11 (Model Monitoring) monitors the winning model's health in production after the A/B test concludes.

---

## Learning Goals

- [ ] Understand A/B testing for ML: why offline metrics aren't sufficient and online experiments are needed
- [ ] Understand statistical significance: p-values, confidence intervals, sample size requirements, and power analysis
- [ ] Understand traffic splitting: consistent hashing for stable user assignment across requests
- [ ] Understand guardrail metrics: safety checks that protect against degraded user experience
- [ ] Understand multi-armed bandits: adaptive allocation vs fixed splits for faster convergence

---

## Implementation Goals

- [ ] Build traffic router: consistently assign users to model variants based on experiment config
- [ ] Implement metric collection: track success metrics per variant (click-through, conversion, quality)
- [ ] Implement statistical analysis: p-value calculation, confidence intervals, sample size estimation
- [ ] Build experiment lifecycle: create → configure → run → analyze → decide (promote or rollback)
- [ ] Implement guardrail metrics: auto-stop experiment if safety metrics degrade beyond threshold
- [ ] Build multi-armed bandit: Thompson sampling for adaptive traffic allocation
- [ ] Create experiment dashboard: metrics over time, significance indicators, winner detection
- [ ] Implement experiment configuration: YAML-based experiment definitions

---

## Acceptance Criteria

1. Traffic router consistently assigns the same user to the same variant across multiple requests (verified with 1000 requests for the same user_id).
2. Traffic split matches configured ratio within 2% tolerance (e.g., 50/50 split verified over 10,000 requests).
3. Statistical significance calculator correctly identifies a winner when the true effect size is 5% with 95% confidence (verified with synthetic data).
4. Sample size estimator correctly calculates required samples given effect size, alpha, and power parameters.
5. Guardrail metric stops the experiment within 60 seconds when a safety metric degrades by more than the configured threshold.
6. Multi-armed bandit allocates more traffic to the better-performing variant over time (verified: after 5000 samples, >70% traffic goes to the true winner).
7. Experiment lifecycle completes end-to-end: create experiment → run for N samples → analyze → auto-declare winner.
8. Experiment dashboard shows per-variant metrics, cumulative significance chart, and current allocation percentages.
9. Concurrent experiments work correctly: two experiments on different user segments don't interfere with each other.
10. End-to-end: deploy two model versions → create A/B experiment → collect 10,000 predictions → compute significance → promote winner to production.

---

## Validation Commands

```bash
# Create an experiment
conduit experiment create --name fraud_v2_vs_v3 \
  --control fraud_detector:v2 --treatment fraud_detector:v3 \
  --traffic-split 50:50 --metric conversion_rate

# Start the experiment
conduit experiment start --name fraud_v2_vs_v3

# Check experiment status
conduit experiment status --name fraud_v2_vs_v3

# View live metrics
conduit experiment metrics --name fraud_v2_vs_v3

# Run significance analysis
conduit experiment analyze --name fraud_v2_vs_v3

# Calculate required sample size
conduit experiment sample-size --baseline-rate 0.05 \
  --min-effect 0.01 --alpha 0.05 --power 0.80

# Stop experiment and declare winner
conduit experiment conclude --name fraud_v2_vs_v3

# List active experiments
conduit experiment list --status active

# Run bandit experiment
conduit experiment create --name fraud_bandit \
  --variants fraud_v1,fraud_v2,fraud_v3 \
  --strategy thompson_sampling --metric quality_score

# Run tests
pytest tests/unit/experiments/ -v
pytest tests/integration/experiments/ -v --timeout=120
```

---

## Technical Implementation Details

### Project Structure (additions)

```
conduit/
├── src/conduit/
│   └── experiments/
│       ├── __init__.py
│       ├── router.py            # Traffic routing and assignment
│       ├── metrics.py           # Metric collection and storage
│       ├── statistics.py        # Statistical tests and analysis
│       ├── lifecycle.py         # Experiment lifecycle management
│       ├── guardrails.py        # Safety metric monitoring
│       ├── bandit.py            # Multi-armed bandit strategies
│       └── dashboard.py         # Experiment reporting
├── configs/
│   └── experiments/
│       ├── fraud_v2_vs_v3.yaml
│       └── guardrails.yaml
```

### Traffic Router

```python
# src/conduit/experiments/router.py
import hashlib
from dataclasses import dataclass

@dataclass
class Variant:
    name: str
    model_name: str
    model_version: str
    weight: float

@dataclass
class Assignment:
    variant: Variant
    experiment_name: str
    user_id: str

class TrafficRouter:
    def __init__(self, experiment_name: str, variants: list[Variant]):
        self.experiment_name = experiment_name
        self.variants = variants
        self._validate_weights()

    def _validate_weights(self):
        total = sum(v.weight for v in self.variants)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def assign(self, user_id: str) -> Assignment:
        hash_input = f"{self.experiment_name}:{user_id}"
        hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0

        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant.weight
            if bucket < cumulative:
                return Assignment(
                    variant=variant,
                    experiment_name=self.experiment_name,
                    user_id=user_id,
                )
        return Assignment(variant=self.variants[-1], experiment_name=self.experiment_name, user_id=user_id)

    def update_weights(self, new_weights: dict[str, float]):
        for variant in self.variants:
            if variant.name in new_weights:
                variant.weight = new_weights[variant.name]
        self._validate_weights()
```

### Statistical Analysis

```python
# src/conduit/experiments/statistics.py
import numpy as np
from dataclasses import dataclass
from scipy import stats

@dataclass
class SignificanceResult:
    significant: bool
    p_value: float
    confidence_interval: tuple[float, float]
    effect_size: float
    control_mean: float
    treatment_mean: float
    relative_lift: float

class ExperimentStatistics:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def test_proportions(self, control_successes: int, control_total: int,
                         treatment_successes: int, treatment_total: int) -> SignificanceResult:
        control_rate = control_successes / control_total
        treatment_rate = treatment_successes / treatment_total

        pooled_rate = (control_successes + treatment_successes) / (control_total + treatment_total)
        se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/control_total + 1/treatment_total))

        z_stat = (treatment_rate - control_rate) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        diff = treatment_rate - control_rate
        ci = (diff - z_crit * se, diff + z_crit * se)

        relative_lift = (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0

        return SignificanceResult(
            significant=p_value < self.alpha,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=diff,
            control_mean=control_rate,
            treatment_mean=treatment_rate,
            relative_lift=relative_lift,
        )

    def required_sample_size(self, baseline_rate: float, min_effect: float,
                             alpha: float = 0.05, power: float = 0.80) -> int:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        p1 = baseline_rate
        p2 = baseline_rate + min_effect
        pooled = (p1 + p2) / 2

        n = ((z_alpha * np.sqrt(2 * pooled * (1 - pooled)) +
              z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (min_effect ** 2)

        return int(np.ceil(n))

    def sequential_test(self, control_data: list[float], treatment_data: list[float],
                        spending_func: str = "obrien_fleming") -> SignificanceResult:
        """Sequential testing with alpha spending for early stopping."""
        n_looks = 5
        current_fraction = len(control_data) / (len(control_data) * n_looks)
        adjusted_alpha = self._alpha_spending(current_fraction, spending_func)

        t_stat, p_value = stats.ttest_ind(treatment_data, control_data)
        diff = np.mean(treatment_data) - np.mean(control_data)
        se = np.sqrt(np.var(treatment_data)/len(treatment_data) + np.var(control_data)/len(control_data))
        z_crit = stats.norm.ppf(1 - adjusted_alpha / 2)
        ci = (diff - z_crit * se, diff + z_crit * se)

        return SignificanceResult(
            significant=p_value < adjusted_alpha,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=diff,
            control_mean=float(np.mean(control_data)),
            treatment_mean=float(np.mean(treatment_data)),
            relative_lift=diff / np.mean(control_data) if np.mean(control_data) != 0 else 0,
        )

    def _alpha_spending(self, fraction: float, func: str) -> float:
        if func == "obrien_fleming":
            return 2 * (1 - stats.norm.cdf(stats.norm.ppf(1 - self.alpha/2) / np.sqrt(fraction)))
        return self.alpha * fraction
```

### Multi-Armed Bandit

```python
# src/conduit/experiments/bandit.py
import numpy as np
from dataclasses import dataclass, field

@dataclass
class ArmState:
    name: str
    successes: int = 0
    failures: int = 0
    total_reward: float = 0.0
    pulls: int = 0

class ThompsonSampling:
    def __init__(self, arm_names: list[str]):
        self.arms = {name: ArmState(name=name) for name in arm_names}

    def select_arm(self) -> str:
        samples = {}
        for name, arm in self.arms.items():
            alpha = arm.successes + 1
            beta = arm.failures + 1
            samples[name] = np.random.beta(alpha, beta)
        return max(samples, key=samples.get)

    def update(self, arm_name: str, reward: float):
        arm = self.arms[arm_name]
        arm.pulls += 1
        arm.total_reward += reward
        if reward > 0.5:
            arm.successes += 1
        else:
            arm.failures += 1

    def get_allocation_weights(self) -> dict[str, float]:
        n_simulations = 10000
        wins = {name: 0 for name in self.arms}
        for _ in range(n_simulations):
            samples = {}
            for name, arm in self.arms.items():
                alpha = arm.successes + 1
                beta = arm.failures + 1
                samples[name] = np.random.beta(alpha, beta)
            winner = max(samples, key=samples.get)
            wins[winner] += 1
        total = sum(wins.values())
        return {name: count / total for name, count in wins.items()}

    def get_stats(self) -> dict:
        return {
            name: {
                "pulls": arm.pulls,
                "success_rate": arm.successes / arm.pulls if arm.pulls > 0 else 0,
                "mean_reward": arm.total_reward / arm.pulls if arm.pulls > 0 else 0,
            }
            for name, arm in self.arms.items()
        }
```

### Guardrail Metrics

```python
# src/conduit/experiments/guardrails.py
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class GuardrailAction(Enum):
    CONTINUE = "continue"
    ALERT = "alert"
    STOP = "stop"

@dataclass
class GuardrailCheck:
    metric_name: str
    operator: str  # ">=", "<=", "<", ">"
    threshold: float
    action: GuardrailAction = GuardrailAction.STOP

@dataclass
class GuardrailResult:
    check: GuardrailCheck
    current_value: float
    violated: bool
    action: GuardrailAction

class GuardrailMonitor:
    def __init__(self, checks: list[GuardrailCheck]):
        self.checks = checks
        self.violations: list[GuardrailResult] = []

    def evaluate(self, metrics: dict[str, float]) -> list[GuardrailResult]:
        results = []
        for check in self.checks:
            value = metrics.get(check.metric_name)
            if value is None:
                continue
            violated = self._check_violation(value, check.operator, check.threshold)
            result = GuardrailResult(
                check=check, current_value=value, violated=violated,
                action=check.action if violated else GuardrailAction.CONTINUE,
            )
            results.append(result)
            if violated:
                self.violations.append(result)
                logger.warning(f"Guardrail violated: {check.metric_name}={value} {check.operator} {check.threshold}")
        return results

    def should_stop(self, metrics: dict[str, float]) -> bool:
        results = self.evaluate(metrics)
        return any(r.violated and r.action == GuardrailAction.STOP for r in results)

    def _check_violation(self, value: float, operator: str, threshold: float) -> bool:
        ops = {">=": lambda v, t: v < t, "<=": lambda v, t: v > t, ">": lambda v, t: v <= t, "<": lambda v, t: v >= t}
        return ops[operator](value, threshold)
```

### Experiment Config

```yaml
# configs/experiments/fraud_v2_vs_v3.yaml
name: fraud_v2_vs_v3
description: "Compare fraud detector v3 against production v2"
status: active

variants:
  - name: control
    model_name: fraud_detector
    model_version: "2"
    weight: 0.5
  - name: treatment
    model_name: fraud_detector
    model_version: "3"
    weight: 0.5

metrics:
  primary:
    name: fraud_detection_rate
    type: proportion
  secondary:
    - name: false_positive_rate
      type: proportion
    - name: latency_p99_ms
      type: continuous

guardrails:
  - metric_name: false_positive_rate
    operator: "<="
    threshold: 0.05
    action: stop
  - metric_name: latency_p99_ms
    operator: "<="
    threshold: 200
    action: alert

settings:
  min_samples_per_variant: 5000
  max_duration_days: 14
  alpha: 0.05
  power: 0.80
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Inconsistent user assignment | Verify hash function is deterministic. Check that experiment name is included in hash input (prevents cross-experiment leakage). |
| Test never reaches significance | Check sample size calculator — you may need more traffic. Reduce min_effect size or increase experiment duration. |
| Guardrail false positives | Increase evaluation window (don't trigger on single data points). Use rolling average over last N minutes. |
| Bandit converging too slowly | Increase the reward signal strength. Ensure rewards are between 0 and 1 for Beta distribution. |
| Metrics not collecting | Check that the serving layer is logging variant assignments. Verify metric event pipeline is connected. |
| Multiple experiments conflicting | Ensure user segments don't overlap. Use experiment layers (separate hash salts per layer). |

---

## Agent Handoff Template

```
I'm working on the Conduit project, Week 10: A/B Testing Framework.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB VRAM, 32GB RAM, Ubuntu.
Project root: ~/conduit/

Current state: [describe what's working/broken]

What I need help with: [specific issue]

Key files:
- src/conduit/experiments/router.py — Traffic routing (consistent hashing)
- src/conduit/experiments/statistics.py — Statistical significance tests
- src/conduit/experiments/bandit.py — Thompson sampling multi-armed bandit
- src/conduit/experiments/guardrails.py — Safety metric monitoring
- src/conduit/experiments/lifecycle.py — Experiment state management
- configs/experiments/ — Experiment YAML definitions

Infrastructure: FastAPI (serving), PostgreSQL (metric storage), Redis (assignment cache).
Flow: request → router assigns variant → model predicts → metric logged → periodic analysis → significance reached → conclude.
```

---

## Out of Scope

- Bayesian A/B testing (frequentist approach only, except for bandits)
- Multi-variate testing (more than two factors simultaneously)
- Interleaving experiments (for ranking models)
- Network effects or interference between users
- Long-term holdout groups
- Causal inference beyond basic A/B (instrumental variables, diff-in-diff)
- Integration with third-party experimentation platforms (LaunchDarkly, Optimizely)
