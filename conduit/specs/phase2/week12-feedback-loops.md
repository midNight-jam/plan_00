# Week 12: Feedback Loops and Active Learning

## Context

**Where it fits:** Week 12 of Phase 2 (Deployment + Monitoring). Feedback loops close the gap between model predictions and real-world outcomes — collecting ground truth labels from production, identifying where the model struggles, and routing uncertain cases for human review.

**Prerequisites:**
- Week 11 complete: monitoring infrastructure detects drift and performance changes
- Week 9 complete: serving layer logs predictions with metadata
- Feature store operational (for enriching feedback data)
- Understanding of uncertainty quantification and active learning concepts

**What it builds on:** Uses prediction logs from serving (Week 9). Uses drift detection (Week 11) to identify when new labeled data is most needed. Enriches the feature store (Week 2) with production feedback. Feeds data versioning (Week 3) with new training data.

**What comes next:** Week 13 (Auto-Retraining) uses accumulated feedback data to trigger and execute automated model retraining.

---

## Learning Goals

- [ ] Understand the feedback loop: predictions → user actions → labels → retraining → better predictions
- [ ] Understand delayed feedback: ground truth may arrive hours/days/weeks after prediction
- [ ] Understand active learning: strategically selecting which data points to label for maximum model improvement
- [ ] Understand human-in-the-loop: when and how to involve humans in the prediction pipeline
- [ ] Understand data flywheels: the virtuous cycle of better models creating more data

---

## Implementation Goals

- [ ] Build label collection pipeline: capture ground truth from user actions and delayed events
- [ ] Implement label extraction: transform raw events into structured labels with quality checks
- [ ] Build active learning selector: identify uncertain predictions for prioritized human labeling
- [ ] Implement human-in-the-loop routing: send low-confidence predictions to human reviewers
- [ ] Build feedback-to-feature-store pipeline: enrich features with production outcome data
- [ ] Implement retraining trigger logic: fire when enough new labels accumulate or drift exceeds threshold
- [ ] Build data quality validation: ensure collected labels meet quality standards before training
- [ ] Implement the data flywheel: production data → model improvement → product improvement → more data

---

## Acceptance Criteria

1. Label collection pipeline ingests user action events and produces structured labels within 5 minutes of event arrival.
2. Delayed feedback labels (arriving up to 7 days after prediction) are correctly joined back to their original prediction by prediction_id.
3. Active learning selector identifies the top-100 most uncertain predictions from the last 24 hours using model confidence scores.
4. Human-in-the-loop routing sends predictions with confidence < 0.6 to a review queue and returns within 30 seconds for synchronous use cases.
5. Label quality validation rejects labels that fail consistency checks (contradicts known facts, duplicate submission, obvious spam).
6. Feedback data correctly flows into the feature store — enriched features are available for subsequent predictions within 1 hour.
7. Retraining trigger fires automatically when 10,000 new labeled samples accumulate since last training run.
8. Active learning achieves 15% better model improvement per labeled sample compared to random sampling (verified on held-out test set).
9. Data flywheel metrics show: more predictions → more labels → better model → more user engagement → more predictions.
10. End-to-end: model predicts → user acts → label extracted → quality validated → stored → triggers retrain signal → new data version created.

---

## Validation Commands

```bash
# Start label collection pipeline
conduit feedback start --model fraud_detector

# View label collection stats
conduit feedback stats --model fraud_detector --window 24h

# Run active learning selection
conduit feedback select-uncertain --model fraud_detector \
  --n-samples 100 --strategy uncertainty_sampling

# View human review queue
conduit feedback review-queue --status pending

# Submit a human review
conduit feedback review-submit --prediction-id pred_123 --label fraud --reviewer jm

# Check label quality
conduit feedback quality-report --model fraud_detector --window 7d

# Trigger feedback → feature store sync
conduit feedback sync-features --model fraud_detector

# Check retraining trigger status
conduit feedback trigger-status --model fraud_detector

# View data flywheel metrics
conduit feedback flywheel-metrics --model fraud_detector

# Backfill delayed labels
conduit feedback backfill --model fraud_detector \
  --source events/chargebacks_2025_01.parquet --join-key prediction_id

# Run tests
pytest tests/unit/feedback/ -v
pytest tests/integration/feedback/ -v --timeout=180
```

---

## Technical Implementation Details

### Project Structure (additions)

```
conduit/
├── src/conduit/
│   └── feedback/
│       ├── __init__.py
│       ├── collector.py         # Label collection pipeline
│       ├── label_extraction.py  # Event → label transformation
│       ├── active_learning.py   # Uncertainty-based sample selection
│       ├── human_loop.py        # Human-in-the-loop routing
│       ├── quality.py           # Label quality validation
│       ├── triggers.py          # Retraining trigger logic
│       ├── sync.py              # Feedback → feature store sync
│       └── flywheel.py          # Data flywheel metrics
├── configs/
│   └── feedback/
│       ├── collection.yaml
│       ├── active_learning.yaml
│       └── triggers.yaml
```

### Label Collection Pipeline

```python
# src/conduit/feedback/collector.py
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd

class LabelSource(Enum):
    USER_ACTION = "user_action"
    DELAYED_EVENT = "delayed_event"
    HUMAN_REVIEW = "human_review"
    AUTOMATED = "automated"

@dataclass
class Label:
    prediction_id: str
    label_value: float | str
    source: LabelSource
    timestamp: datetime
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

@dataclass
class CollectionStats:
    total_labels: int
    labels_by_source: dict[str, int]
    avg_delay_hours: float
    label_rate_per_hour: float

class LabelCollector:
    def __init__(self, prediction_store, label_store, max_delay_days: int = 30):
        self.prediction_store = prediction_store
        self.label_store = label_store
        self.max_delay_days = max_delay_days

    def collect_from_events(self, events: pd.DataFrame, extraction_config: dict) -> list[Label]:
        labels = []
        for _, event in events.iterrows():
            prediction_id = event.get("prediction_id")
            if not prediction_id:
                prediction_id = self._match_event_to_prediction(event)
                if not prediction_id:
                    continue

            prediction = self.prediction_store.get(prediction_id)
            if not prediction:
                continue

            delay = event["timestamp"] - prediction["timestamp"]
            if delay > timedelta(days=self.max_delay_days):
                continue

            label_value = self._extract_label(event, extraction_config)
            label = Label(
                prediction_id=prediction_id,
                label_value=label_value,
                source=LabelSource.DELAYED_EVENT,
                timestamp=event["timestamp"],
                metadata={"delay_hours": delay.total_seconds() / 3600, "event_type": event.get("event_type")},
            )
            labels.append(label)

        self.label_store.save_batch(labels)
        return labels

    def _match_event_to_prediction(self, event: pd.Series) -> str | None:
        entity_id = event.get("entity_id")
        event_time = event["timestamp"]
        window = timedelta(hours=24)
        candidates = self.prediction_store.query(
            entity_id=entity_id,
            time_range=(event_time - window, event_time),
        )
        if candidates:
            return candidates[-1]["prediction_id"]
        return None

    def _extract_label(self, event: pd.Series, config: dict) -> float | str:
        event_type = event.get("event_type")
        label_map = config.get("event_to_label", {})
        return label_map.get(event_type, 0)

    def get_stats(self, window_hours: int = 24) -> CollectionStats:
        recent_labels = self.label_store.query_recent(hours=window_hours)
        by_source = {}
        delays = []
        for label in recent_labels:
            source_name = label.source.value
            by_source[source_name] = by_source.get(source_name, 0) + 1
            if "delay_hours" in label.metadata:
                delays.append(label.metadata["delay_hours"])

        return CollectionStats(
            total_labels=len(recent_labels),
            labels_by_source=by_source,
            avg_delay_hours=sum(delays) / len(delays) if delays else 0,
            label_rate_per_hour=len(recent_labels) / window_hours,
        )
```

### Active Learning

```python
# src/conduit/feedback/active_learning.py
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

class SelectionStrategy(Enum):
    UNCERTAINTY_SAMPLING = "uncertainty_sampling"
    MARGIN_SAMPLING = "margin_sampling"
    ENTROPY_SAMPLING = "entropy_sampling"
    DIVERSITY_SAMPLING = "diversity_sampling"

@dataclass
class SelectionResult:
    prediction_ids: list[str]
    scores: list[float]
    strategy: SelectionStrategy
    pool_size: int

class ActiveLearningSelector:
    def __init__(self, strategy: SelectionStrategy = SelectionStrategy.UNCERTAINTY_SAMPLING):
        self.strategy = strategy

    def select(self, predictions: pd.DataFrame, n_samples: int) -> SelectionResult:
        if self.strategy == SelectionStrategy.UNCERTAINTY_SAMPLING:
            scores = self._uncertainty_scores(predictions)
        elif self.strategy == SelectionStrategy.MARGIN_SAMPLING:
            scores = self._margin_scores(predictions)
        elif self.strategy == SelectionStrategy.ENTROPY_SAMPLING:
            scores = self._entropy_scores(predictions)
        elif self.strategy == SelectionStrategy.DIVERSITY_SAMPLING:
            return self._diversity_select(predictions, n_samples)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        top_indices = np.argsort(scores)[-n_samples:][::-1]

        return SelectionResult(
            prediction_ids=predictions.iloc[top_indices]["prediction_id"].tolist(),
            scores=[scores[i] for i in top_indices],
            strategy=self.strategy,
            pool_size=len(predictions),
        )

    def _uncertainty_scores(self, predictions: pd.DataFrame) -> np.ndarray:
        confidence = predictions["confidence"].values
        return 1.0 - np.abs(2 * confidence - 1.0)

    def _margin_scores(self, predictions: pd.DataFrame) -> np.ndarray:
        if "prob_class_0" in predictions.columns and "prob_class_1" in predictions.columns:
            margin = np.abs(predictions["prob_class_0"].values - predictions["prob_class_1"].values)
            return 1.0 - margin
        return self._uncertainty_scores(predictions)

    def _entropy_scores(self, predictions: pd.DataFrame) -> np.ndarray:
        prob_cols = [c for c in predictions.columns if c.startswith("prob_class_")]
        if not prob_cols:
            return self._uncertainty_scores(predictions)
        probs = predictions[prob_cols].values
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log2(probs), axis=1)
        return entropy / np.log2(len(prob_cols))

    def _diversity_select(self, predictions: pd.DataFrame, n_samples: int) -> SelectionResult:
        feature_cols = [c for c in predictions.columns if c.startswith("feature_")]
        if not feature_cols:
            return self.select(predictions.assign(confidence=0.5), n_samples)

        features = predictions[feature_cols].values
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

        selected_indices = [np.random.randint(len(features))]
        for _ in range(n_samples - 1):
            distances = np.min([
                np.linalg.norm(features - features[idx], axis=1)
                for idx in selected_indices
            ], axis=0)
            next_idx = np.argmax(distances)
            selected_indices.append(next_idx)

        return SelectionResult(
            prediction_ids=predictions.iloc[selected_indices]["prediction_id"].tolist(),
            scores=[1.0] * len(selected_indices),
            strategy=SelectionStrategy.DIVERSITY_SAMPLING,
            pool_size=len(predictions),
        )
```

### Human-in-the-Loop

```python
# src/conduit/feedback/human_loop.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import PriorityQueue

class ReviewStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"

@dataclass(order=True)
class ReviewTask:
    priority: float
    prediction_id: str = field(compare=False)
    features: dict = field(compare=False)
    model_prediction: float = field(compare=False)
    model_confidence: float = field(compare=False)
    created_at: datetime = field(compare=False, default_factory=datetime.utcnow)
    status: ReviewStatus = field(compare=False, default=ReviewStatus.PENDING)
    reviewer: str | None = field(compare=False, default=None)
    human_label: float | str | None = field(compare=False, default=None)

class HumanInTheLoop:
    def __init__(self, confidence_threshold: float = 0.6, max_queue_size: int = 1000):
        self.confidence_threshold = confidence_threshold
        self.max_queue_size = max_queue_size
        self.queue: PriorityQueue[ReviewTask] = PriorityQueue()
        self.completed: list[ReviewTask] = []

    def should_route_to_human(self, confidence: float) -> bool:
        return confidence < self.confidence_threshold

    def add_to_queue(self, prediction_id: str, features: dict,
                     model_prediction: float, model_confidence: float) -> ReviewTask:
        priority = model_confidence  # Lower confidence = higher priority (processed first)
        task = ReviewTask(
            priority=priority,
            prediction_id=prediction_id,
            features=features,
            model_prediction=model_prediction,
            model_confidence=model_confidence,
        )
        if self.queue.qsize() < self.max_queue_size:
            self.queue.put(task)
        return task

    def get_next_task(self, reviewer: str) -> ReviewTask | None:
        if self.queue.empty():
            return None
        task = self.queue.get()
        task.status = ReviewStatus.IN_PROGRESS
        task.reviewer = reviewer
        return task

    def submit_review(self, task: ReviewTask, label: float | str) -> None:
        task.human_label = label
        task.status = ReviewStatus.COMPLETED
        self.completed.append(task)

    def get_agreement_rate(self) -> float:
        if not self.completed:
            return 0.0
        agreements = sum(
            1 for t in self.completed
            if self._labels_agree(t.model_prediction, t.human_label)
        )
        return agreements / len(self.completed)

    def _labels_agree(self, model_pred, human_label) -> bool:
        if isinstance(human_label, str):
            return str(int(round(model_pred))) == human_label
        return abs(model_pred - float(human_label)) < 0.5
```

### Retraining Triggers

```python
# src/conduit/feedback/triggers.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class TriggerReason(Enum):
    SCHEDULED = "scheduled"
    DATA_VOLUME = "data_volume"
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADED = "performance_degraded"

@dataclass
class TriggerDecision:
    should_trigger: bool
    reason: TriggerReason | None
    details: dict

class RetrainingTrigger:
    def __init__(self, min_new_samples: int = 10000, max_days_since_training: int = 7,
                 drift_threshold: float = 0.2, performance_drop_threshold: float = 0.05):
        self.min_new_samples = min_new_samples
        self.max_days_since_training = max_days_since_training
        self.drift_threshold = drift_threshold
        self.performance_drop_threshold = performance_drop_threshold
        self.last_training_time: datetime | None = None
        self.last_training_sample_count: int = 0

    def evaluate(self, current_sample_count: int, drift_score: float,
                 current_performance: float, baseline_performance: float) -> TriggerDecision:
        new_samples = current_sample_count - self.last_training_sample_count
        if new_samples >= self.min_new_samples:
            return TriggerDecision(
                should_trigger=True,
                reason=TriggerReason.DATA_VOLUME,
                details={"new_samples": new_samples, "threshold": self.min_new_samples},
            )

        if self.last_training_time:
            days_elapsed = (datetime.utcnow() - self.last_training_time).days
            if days_elapsed >= self.max_days_since_training:
                return TriggerDecision(
                    should_trigger=True,
                    reason=TriggerReason.SCHEDULED,
                    details={"days_elapsed": days_elapsed, "max_days": self.max_days_since_training},
                )

        if drift_score >= self.drift_threshold:
            return TriggerDecision(
                should_trigger=True,
                reason=TriggerReason.DRIFT_DETECTED,
                details={"drift_score": drift_score, "threshold": self.drift_threshold},
            )

        perf_drop = baseline_performance - current_performance
        if perf_drop >= self.performance_drop_threshold:
            return TriggerDecision(
                should_trigger=True,
                reason=TriggerReason.PERFORMANCE_DEGRADED,
                details={"performance_drop": perf_drop, "threshold": self.performance_drop_threshold},
            )

        return TriggerDecision(should_trigger=False, reason=None, details={"new_samples": new_samples})

    def record_training(self, sample_count: int):
        self.last_training_time = datetime.utcnow()
        self.last_training_sample_count = sample_count
```

### Feedback Config

```yaml
# configs/feedback/collection.yaml
model: fraud_detector

label_sources:
  user_action:
    events: ["transaction_reported_fraud", "transaction_confirmed_legit"]
    event_to_label:
      transaction_reported_fraud: 1
      transaction_confirmed_legit: 0
    max_delay_hours: 1

  delayed_event:
    events: ["chargeback_filed", "dispute_resolved"]
    event_to_label:
      chargeback_filed: 1
      dispute_resolved: 0
    max_delay_days: 30
    join_key: prediction_id

active_learning:
  strategy: uncertainty_sampling
  daily_budget: 100
  confidence_threshold: 0.6

human_review:
  confidence_threshold: 0.5
  max_queue_size: 500
  timeout_hours: 4

triggers:
  min_new_samples: 10000
  max_days_since_training: 7
  drift_threshold: 0.2
  performance_drop_threshold: 0.05
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Cannot join delayed labels to predictions | Ensure prediction_id is logged with every prediction and included in downstream events. Use entity_id + time window as fallback. |
| Active learning not improving model | Check that selected samples are actually from the decision boundary. Verify uncertainty scores correlate with actual errors. |
| Human review queue growing unbounded | Lower the confidence threshold (route fewer items). Add timeout and auto-skip for stale items. |
| Label quality poor | Add validation rules (e.g., no contradictions within 24h for same entity). Require minimum reviewer agreement. |
| Feedback data not reaching feature store | Check the sync pipeline. Verify feature store write permissions. Check that entity_id mapping is correct. |
| Retraining triggered too often | Increase `min_new_samples` or `max_days_since_training`. Add cooldown period between triggers. |

---

## Agent Handoff Template

```
I'm working on the Conduit project, Week 12: Feedback Loops and Active Learning.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB VRAM, 32GB RAM, Ubuntu.
Project root: ~/conduit/

Current state: [describe what's working/broken]

What I need help with: [specific issue]

Key files:
- src/conduit/feedback/collector.py — Label collection from events
- src/conduit/feedback/active_learning.py — Uncertainty-based sample selection
- src/conduit/feedback/human_loop.py — Human-in-the-loop routing
- src/conduit/feedback/triggers.py — Retraining trigger logic
- src/conduit/feedback/quality.py — Label quality validation
- configs/feedback/ — Collection and trigger configuration

Infrastructure: PostgreSQL (label store), prediction log store, feature store, event queue.
Flow: user action → event → label extraction → quality check → store → active learning selection → trigger evaluation.
```

---

## Out of Scope

- Crowdsourcing platforms (Amazon Mechanical Turk, Labelbox integration)
- Semi-supervised learning (using unlabeled data for training)
- Weak supervision (programmatic labeling with Snorkel-style label functions)
- Multi-annotator disagreement resolution (inter-rater reliability)
- Reinforcement learning from human feedback (RLHF)
- Real-time streaming label collection (Kafka consumers)
- Privacy-preserving feedback (differential privacy, federated collection)
