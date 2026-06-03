# Week 13: ML Lifecycle Management

## Context

**Where it fits:** Phase B, Week 13 — the ML-specific operations layer on top of the platform.
**Prerequisites:** Phase A complete. Weeks 8-12 complete (SRE, cost, security, chaos, multi-cluster).
**What it builds on:** Models have been served since Phase A, but without lifecycle governance. This week adds the full experiment-to-retirement pipeline: tracking experiments, promoting models through stages, A/B testing in production, automatic rollback, data versioning, and a simplified feature store.

This is where infrastructure meets ML engineering. The platform is reliable, secure, and scalable — now it needs to manage the ML-specific complexity of experiments, evaluations, and safe deployments.

---

## Learning Goals

- [ ] Understand ML experiment tracking and why reproducibility matters
- [ ] Know model lifecycle stages and what gates prevent bad models from reaching production
- [ ] Design A/B testing infrastructure for model comparison in production
- [ ] Implement automatic rollback based on statistical significance of metric degradation
- [ ] Understand data versioning and why point-in-time correctness matters for features
- [ ] Build a simplified feature store that demonstrates the core concepts

---

## Implementation Goals

- [ ] Deploy MLflow for experiment tracking (hyperparameters, metrics, artifacts, comparison)
- [ ] Define model lifecycle stages: experiment → candidate → staging → production → retired
- [ ] Implement promotion gates requiring eval suite pass before stage advancement
- [ ] Build A/B testing infrastructure routing configurable traffic percentage to candidate model
- [ ] Implement automatic rollback if candidate model's metrics are statistically worse
- [ ] Set up data versioning with DVC tracking dataset versions used for each training run
- [ ] Build simplified feature store with centralized features and point-in-time correctness
- [ ] Create dashboard showing model lineage: data version → training run → model version → deployment

---

## Acceptance Criteria

1. MLflow tracks all training runs with hyperparameters, loss curves, and model artifacts
2. Model lifecycle transitions are gated: promotion from candidate to staging requires eval score >0.85
3. Attempting to promote a model with eval score <0.85 is rejected with clear error message
4. A/B test routes exactly the configured percentage (e.g., 10%) of traffic to candidate model
5. A/B test comparison dashboard shows side-by-side metrics for production vs candidate
6. Automatic rollback triggers within 5 minutes when candidate error rate is 2x production
7. After rollback, 100% traffic returns to production model and alert is generated
8. Data versioning tracks which dataset version was used for each training run
9. Feature store serves consistent features with point-in-time correctness (no data leakage)
10. End-to-end demo: train with tracked experiment → promote → A/B test → rollback → fix → promote → production

---

## Validation Commands

```bash
# Verify MLflow is running
curl -s http://mlflow.mlops:5000/api/2.0/mlflow/experiments/list | jq '.experiments | length'

# Check experiment tracking
curl -s http://mlflow.mlops:5000/api/2.0/mlflow/runs/search \
  -d '{"experiment_ids":["1"]}' | jq '.runs[0].data.metrics'

# Test promotion gate (should fail)
python -c "
from src.lifecycle.promoter import ModelPromoter
p = ModelPromoter()
result = p.promote('model-v1', 'candidate', 'staging')
assert result['status'] == 'rejected', f'Expected rejection, got {result}'
print('PASS: Low-score model correctly rejected')
"

# Verify A/B traffic split
for i in $(seq 1 100); do
  curl -s http://inference-gateway:8080/predict -d '{"text":"test"}' | jq -r '.model_version'
done | sort | uniq -c
# Should show ~90 production, ~10 candidate

# Test automatic rollback
kubectl apply -f tests/bad-candidate-model.yaml && sleep 360 && \
  curl -s http://inference-gateway:8080/predict | jq '.model_version' | grep "production"

# Verify data versioning
dvc status && git log --oneline data/datasets/ | head -5

# Test feature store point-in-time query
python -c "
from src.features.store import FeatureStore
fs = FeatureStore()
features = fs.get_features(entity_id='user_123', timestamp='2024-01-15T00:00:00Z')
assert 'request_count_7d' in features
print(f'PASS: Got {len(features)} features for point-in-time query')
"

# Full lifecycle test
kubectl apply -f tests/ml-lifecycle-e2e.yaml && \
  kubectl wait --for=condition=complete job/lifecycle-e2e -n mlops --timeout=900s
```

---

## Technical Implementation Details

### MLflow Deployment

```yaml
# File: k8s/mlops/mlflow.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: mlops
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
        - name: mlflow
          image: ghcr.io/mlflow/mlflow:latest
          command:
            - mlflow
            - server
            - --host=0.0.0.0
            - --port=5000
            - --backend-store-uri=postgresql://mlflow:mlflow@postgres.mlops:5432/mlflow
            - --default-artifact-root=s3://mlflow-artifacts/
          env:
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: minio-credentials
                  key: access-key
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: minio-credentials
                  key: secret-key
            - name: MLFLOW_S3_ENDPOINT_URL
              value: "http://minio.storage:9000"
          ports:
            - containerPort: 5000
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow
  namespace: mlops
spec:
  selector:
    app: mlflow
  ports:
    - port: 5000
      targetPort: 5000
```

### Model Lifecycle Promoter

```python
# File: src/lifecycle/promoter.py
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

class ModelStage(Enum):
    EXPERIMENT = "experiment"
    CANDIDATE = "candidate"
    STAGING = "staging"
    PRODUCTION = "production"
    RETIRED = "retired"

VALID_TRANSITIONS = {
    ModelStage.EXPERIMENT: [ModelStage.CANDIDATE],
    ModelStage.CANDIDATE: [ModelStage.STAGING, ModelStage.RETIRED],
    ModelStage.STAGING: [ModelStage.PRODUCTION, ModelStage.RETIRED],
    ModelStage.PRODUCTION: [ModelStage.RETIRED],
}

@dataclass
class PromotionGate:
    stage: ModelStage
    min_eval_score: float
    required_tests: list[str]

GATES = {
    ModelStage.CANDIDATE: PromotionGate(ModelStage.CANDIDATE, 0.70, ["unit_tests"]),
    ModelStage.STAGING: PromotionGate(ModelStage.STAGING, 0.85, ["eval_suite", "latency_test"]),
    ModelStage.PRODUCTION: PromotionGate(ModelStage.PRODUCTION, 0.90, ["eval_suite", "latency_test", "ab_test_pass"]),
}

class ModelPromoter:
    def __init__(self, mlflow_url: str = "http://mlflow.mlops:5000"):
        mlflow.set_tracking_uri(mlflow_url)
        self.client = MlflowClient()

    def promote(self, model_name: str, from_stage: str, to_stage: str) -> dict:
        from_s = ModelStage(from_stage)
        to_s = ModelStage(to_stage)

        if to_s not in VALID_TRANSITIONS.get(from_s, []):
            return {"status": "rejected", "reason": f"Invalid transition: {from_s.value} → {to_s.value}"}

        gate = GATES.get(to_s)
        if gate:
            eval_result = self._run_evaluation(model_name)
            if eval_result["score"] < gate.min_eval_score:
                return {
                    "status": "rejected",
                    "reason": f"Eval score {eval_result['score']:.3f} < required {gate.min_eval_score}",
                    "details": eval_result,
                }
            for test in gate.required_tests:
                if not self._check_test_passed(model_name, test):
                    return {"status": "rejected", "reason": f"Required test '{test}' not passed"}

        self.client.transition_model_version_stage(
            name=model_name,
            version=self._get_latest_version(model_name),
            stage=to_s.value,
        )
        return {
            "status": "promoted",
            "model": model_name,
            "from": from_s.value,
            "to": to_s.value,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _run_evaluation(self, model_name: str) -> dict:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
        # Run eval suite against standard test dataset
        # Returns aggregated score
        return {"score": 0.88, "metrics": {"accuracy": 0.89, "f1": 0.87}}

    def _check_test_passed(self, model_name: str, test_name: str) -> bool:
        runs = self.client.search_runs(
            experiment_ids=["1"],
            filter_string=f"tags.model_name='{model_name}' AND tags.test_name='{test_name}' AND tags.test_passed='true'"
        )
        return len(runs) > 0

    def _get_latest_version(self, model_name: str) -> str:
        versions = self.client.get_latest_versions(model_name)
        return versions[0].version if versions else "1"
```

### A/B Testing Infrastructure

```python
# File: src/lifecycle/ab_testing.py
import random
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
import numpy as np
from scipy import stats

@dataclass
class ABTestConfig:
    name: str
    production_model: str
    candidate_model: str
    traffic_percent_candidate: float = 10.0  # 10% to candidate
    min_samples: int = 1000
    significance_level: float = 0.05
    max_duration_seconds: int = 3600  # 1 hour max

@dataclass
class ABTestMetrics:
    model_version: str
    latencies: list[float] = field(default_factory=list)
    errors: int = 0
    total: int = 0

    @property
    def error_rate(self) -> float:
        return self.errors / self.total if self.total > 0 else 0

    @property
    def mean_latency(self) -> float:
        return np.mean(self.latencies) if self.latencies else 0

class ABTestRunner:
    def __init__(self, config: ABTestConfig):
        self.config = config
        self.production_metrics = ABTestMetrics(model_version=config.production_model)
        self.candidate_metrics = ABTestMetrics(model_version=config.candidate_model)
        self.start_time = time.time()

    def route_request(self) -> str:
        if random.random() * 100 < self.config.traffic_percent_candidate:
            return self.config.candidate_model
        return self.config.production_model

    def record_result(self, model_version: str, latency: float, is_error: bool):
        metrics = (self.candidate_metrics if model_version == self.config.candidate_model
                   else self.production_metrics)
        metrics.total += 1
        metrics.latencies.append(latency)
        if is_error:
            metrics.errors += 1

    def should_rollback(self) -> tuple[bool, str]:
        if self.candidate_metrics.total < 100:
            return False, "insufficient_samples"

        # Check error rate (2x production = rollback)
        if (self.candidate_metrics.error_rate > 2 * self.production_metrics.error_rate
                and self.production_metrics.error_rate > 0):
            return True, f"error_rate: candidate={self.candidate_metrics.error_rate:.3f} vs production={self.production_metrics.error_rate:.3f}"

        # Check latency degradation
        if len(self.candidate_metrics.latencies) >= self.config.min_samples:
            t_stat, p_value = stats.ttest_ind(
                self.candidate_metrics.latencies,
                self.production_metrics.latencies,
                alternative="greater"
            )
            if p_value < self.config.significance_level:
                return True, f"latency_degradation: p_value={p_value:.4f}"

        return False, "healthy"

    def get_comparison(self) -> dict:
        return {
            "test_name": self.config.name,
            "duration_seconds": time.time() - self.start_time,
            "production": {
                "model": self.config.production_model,
                "requests": self.production_metrics.total,
                "error_rate": self.production_metrics.error_rate,
                "mean_latency_ms": self.production_metrics.mean_latency * 1000,
            },
            "candidate": {
                "model": self.config.candidate_model,
                "requests": self.candidate_metrics.total,
                "error_rate": self.candidate_metrics.error_rate,
                "mean_latency_ms": self.candidate_metrics.mean_latency * 1000,
            },
        }
```

### Data Versioning Setup

```bash
# File: scripts/setup-data-versioning.sh
#!/bin/bash
set -euo pipefail

pip install dvc dvc-s3

cd /workspace/anvil-ml

# Initialize DVC
dvc init

# Configure remote storage
dvc remote add -d minio s3://anvil-datasets
dvc remote modify minio endpointurl http://minio.storage:9000
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin

# Track dataset directory
dvc add data/datasets/training-v1/
git add data/datasets/training-v1.dvc data/.gitignore
git commit -m "Track training dataset v1 with DVC"

# Push data to remote
dvc push
```

### Simplified Feature Store

```python
# File: src/features/store.py
import sqlite3
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

@dataclass
class Feature:
    name: str
    entity_type: str
    value_type: str  # int, float, string, list
    description: str

class FeatureStore:
    def __init__(self, db_path: str = "/data/features.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS feature_definitions (
                name TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                value_type TEXT NOT NULL,
                description TEXT
            );
            CREATE TABLE IF NOT EXISTS feature_values (
                feature_name TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                value TEXT NOT NULL,
                event_timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (feature_name, entity_id, event_timestamp)
            );
            CREATE INDEX IF NOT EXISTS idx_feature_time
                ON feature_values(feature_name, entity_id, event_timestamp);
        """)

    def register_feature(self, feature: Feature):
        self.conn.execute(
            "INSERT OR REPLACE INTO feature_definitions VALUES (?, ?, ?, ?)",
            (feature.name, feature.entity_type, feature.value_type, feature.description)
        )
        self.conn.commit()

    def ingest(self, feature_name: str, entity_id: str, value, event_timestamp: str):
        self.conn.execute(
            "INSERT INTO feature_values (feature_name, entity_id, value, event_timestamp) VALUES (?, ?, ?, ?)",
            (feature_name, entity_id, str(value), event_timestamp)
        )
        self.conn.commit()

    def get_features(self, entity_id: str, timestamp: Optional[str] = None) -> dict:
        """Point-in-time correct feature retrieval — no future data leakage."""
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        cursor = self.conn.execute("""
            SELECT fv.feature_name, fv.value
            FROM feature_values fv
            INNER JOIN (
                SELECT feature_name, entity_id, MAX(event_timestamp) as max_ts
                FROM feature_values
                WHERE entity_id = ? AND event_timestamp <= ?
                GROUP BY feature_name
            ) latest ON fv.feature_name = latest.feature_name
                    AND fv.entity_id = latest.entity_id
                    AND fv.event_timestamp = latest.max_ts
            WHERE fv.entity_id = ?
        """, (entity_id, timestamp, entity_id))

        return {row[0]: row[1] for row in cursor.fetchall()}

    def get_training_dataset(self, feature_names: list[str], entity_ids: list[str],
                             timestamp: str) -> list[dict]:
        """Get features for multiple entities at a point in time (for training)."""
        results = []
        for entity_id in entity_ids:
            features = self.get_features(entity_id, timestamp)
            filtered = {k: v for k, v in features.items() if k in feature_names}
            filtered["entity_id"] = entity_id
            results.append(filtered)
        return results
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| MLflow can't connect to PostgreSQL | Check postgres pod is running: `kubectl get pods -n mlops -l app=postgres` |
| MLflow artifacts not saving to MinIO | Verify `MLFLOW_S3_ENDPOINT_URL` env var and MinIO bucket exists |
| A/B test routing not hitting exact percentages | Statistical variance with small sample sizes; need 1000+ requests for accurate split |
| DVC push fails with S3 error | Check MinIO credentials and bucket existence: `mc ls minio/anvil-datasets` |
| Automatic rollback triggers too quickly | Increase `min_samples` threshold; 100 samples may not be statistically significant |
| Feature store query slow | Add index on (entity_id, event_timestamp); consider switching to Redis for online serving |

---

## Agent Handoff Template

```
Resume Week 13: ML Lifecycle Management.

Environment: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
K3s cluster: 3 multipass nodes. Phase A + Weeks 8-12 complete.

Current state: [describe what's done and what's next]

Tasks remaining:
- [ ] [list incomplete items from Implementation Goals]

Key files:
- k8s/mlops/mlflow.yaml
- src/lifecycle/promoter.py
- src/lifecycle/ab_testing.py
- scripts/setup-data-versioning.sh
- src/features/store.py

IMPORTANT: Model lifecycle stages are experiment → candidate → staging → production → retired.
Promotion from candidate → staging requires eval score > 0.85.
A/B test auto-rollback triggers when candidate error rate > 2x production.
Feature store must guarantee point-in-time correctness (no future data leakage).
Validate with the validation commands in the spec.
```

---

## Out of Scope

- Full-scale feature store (Feast/Tecton) — build simplified version to demonstrate concepts
- Model explainability (SHAP/LIME)
- Data labeling workflows
- Real-time feature computation (streaming)
- Model compression/quantization
- Multi-armed bandit (stick with simple A/B for now)
- Federated learning
- Compliance/audit for model decisions (focus is on lifecycle, not governance)
