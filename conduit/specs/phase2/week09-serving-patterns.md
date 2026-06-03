# Week 9: Model Serving Patterns

## Context

**Where it fits:** Week 9 of Phase 2 (Deployment + Monitoring). Model serving is how ML models deliver value — translating registered model artifacts into live predictions that applications consume. This week covers the full spectrum of serving patterns.

**Prerequisites:**
- Week 8 complete: model registry with versioned, packaged models
- Phase 1 feature store operational (feature retrieval at inference time)
- FastAPI and Docker experience from Week 8
- Understanding of REST APIs and async processing

**What it builds on:** Loads models from the registry (Week 8). Retrieves features from the feature store (Week 2) at inference time. Uses experiment tracking metadata to know which model version to serve.

**What comes next:** Week 10 (A/B Testing) uses the serving layer to route traffic between model variants for experimentation.

---

## Learning Goals

- [ ] Understand online vs batch serving: when to use each pattern and their tradeoffs (latency vs throughput)
- [ ] Understand shadow mode: safely testing new models in production without user impact
- [ ] Understand feature serving: bridging the training-serving skew problem
- [ ] Understand model ensembles: combining predictions for improved accuracy or robustness
- [ ] Understand latency budgets: decomposing end-to-end latency into components

---

## Implementation Goals

- [ ] Build online serving endpoint: FastAPI with feature lookup, model inference, and response formatting
- [ ] Build batch serving pipeline: scheduled prediction jobs over large datasets
- [ ] Implement shadow mode: new model runs alongside production, outputs logged but not returned
- [ ] Implement feature serving: real-time feature computation and retrieval at inference time
- [ ] Build model ensemble: weighted combination of multiple model predictions
- [ ] Implement preprocessing/postprocessing pipelines: input validation, feature transformation, output formatting
- [ ] Add response caching: cache predictions for identical feature vectors
- [ ] Implement latency tracking: decompose and monitor each stage of the serving pipeline

---

## Acceptance Criteria

1. Online endpoint returns predictions within 100ms p99 latency (including feature lookup from feature store).
2. Batch serving processes 1M records in under 30 minutes on local hardware, writing predictions to configured output sink.
3. Shadow mode runs a candidate model on 100% of production traffic and logs predictions without affecting response to the caller.
4. Feature serving retrieves real-time features from the feature store with under 10ms latency for online requests.
5. Model ensemble combines 3 models with configurable weights and returns a blended prediction that outperforms any single model on the evaluation set.
6. Preprocessing pipeline validates input schema, handles missing values, and applies feature transformations consistent with training.
7. Response cache achieves a 40%+ hit rate on repeated inference requests, reducing average latency by at least 30%.
8. Latency budget dashboard shows breakdown: feature_lookup_ms + preprocessing_ms + inference_ms + postprocessing_ms = total_ms.
9. Shadow mode comparison report shows agreement rate between production and shadow model, plus per-metric divergence.
10. End-to-end: API request → feature lookup → preprocess → ensemble inference → postprocess → cached response, all within latency budget.

---

## Validation Commands

```bash
# Start online serving endpoint
conduit serve online --model fraud_detector --stage production --port 8080

# Make a prediction (online)
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "user_123", "context": {"timestamp": "2025-01-15T10:30:00Z"}}'

# Run batch prediction
conduit serve batch --model fraud_detector --stage production \
  --input s3://data/transactions_2025_01.parquet \
  --output s3://predictions/fraud_scores_2025_01.parquet

# Enable shadow mode
conduit serve shadow --production fraud_detector_v2 --candidate fraud_detector_v3 --port 8080

# View shadow comparison
conduit serve shadow-report --since 2025-01-15

# Start ensemble serving
conduit serve ensemble --models fraud_v1:0.3,fraud_v2:0.5,fraud_v3:0.2 --port 8080

# Check latency breakdown
conduit serve latency-report --last 1h

# View cache stats
conduit serve cache-stats

# Load test
conduit serve load-test --endpoint http://localhost:8080/predict \
  --concurrency 50 --duration 60s --input-file tests/fixtures/load_test_data.json

# Run tests
pytest tests/unit/serving/ -v
pytest tests/integration/serving/ -v --timeout=180
```

---

## Technical Implementation Details

### Project Structure (additions)

```
conduit/
├── src/conduit/
│   └── serving/
│       ├── __init__.py
│       ├── online.py            # FastAPI online serving
│       ├── batch.py             # Batch prediction pipeline
│       ├── shadow.py            # Shadow mode implementation
│       ├── features.py          # Feature serving integration
│       ├── ensemble.py          # Model ensemble logic
│       ├── pipeline.py          # Pre/post-processing pipelines
│       ├── cache.py             # Response caching
│       └── latency.py           # Latency tracking and budgets
├── configs/
│   └── serving/
│       ├── online.yaml
│       ├── batch.yaml
│       ├── ensemble.yaml
│       └── latency_budget.yaml
```

### Online Serving Endpoint

```python
# src/conduit/serving/online.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import time
import mlflow
from conduit.serving.features import FeatureServer
from conduit.serving.pipeline import PreprocessPipeline, PostprocessPipeline
from conduit.serving.cache import PredictionCache
from conduit.serving.latency import LatencyTracker

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = mlflow.pyfunc.load_model(f"models:/{app.state.model_name}/Production")
    app.state.feature_server = FeatureServer()
    app.state.cache = PredictionCache(max_size=10000, ttl_seconds=300)
    app.state.preprocessor = PreprocessPipeline.from_config("configs/serving/online.yaml")
    app.state.postprocessor = PostprocessPipeline.from_config("configs/serving/online.yaml")
    yield

app = FastAPI(title="Conduit Online Serving", lifespan=lifespan)

class PredictRequest(BaseModel):
    entity_id: str
    context: dict = {}

class PredictResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str
    latency_ms: float
    cached: bool = False

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    tracker = LatencyTracker()

    cache_key = app.state.cache.make_key(request.entity_id, request.context)
    cached = app.state.cache.get(cache_key)
    if cached:
        return PredictResponse(**cached, cached=True)

    with tracker.stage("feature_lookup"):
        features = await app.state.feature_server.get_features(
            entity_id=request.entity_id,
            feature_names=["amount_mean_7d", "tx_count_1h", "merchant_risk_score"],
            context=request.context,
        )

    with tracker.stage("preprocessing"):
        processed = app.state.preprocessor.transform(features)

    with tracker.stage("inference"):
        raw_prediction = app.state.model.predict(processed)

    with tracker.stage("postprocessing"):
        result = app.state.postprocessor.transform(raw_prediction, request.context)

    response_data = {
        "prediction": result["score"],
        "confidence": result["confidence"],
        "model_version": app.state.model.metadata.run_id,
        "latency_ms": tracker.total_ms(),
    }
    app.state.cache.put(cache_key, response_data)
    return PredictResponse(**response_data)
```

### Batch Serving Pipeline

```python
# src/conduit/serving/batch.py
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import mlflow
from conduit.serving.pipeline import PreprocessPipeline, PostprocessPipeline
from conduit.serving.features import FeatureServer

@dataclass
class BatchConfig:
    model_name: str
    model_stage: str
    input_path: str
    output_path: str
    batch_size: int = 10000
    feature_names: list[str] | None = None

class BatchServingPipeline:
    def __init__(self, config: BatchConfig):
        self.config = config
        self.model = mlflow.pyfunc.load_model(f"models:/{config.model_name}/{config.model_stage}")
        self.feature_server = FeatureServer()
        self.preprocessor = PreprocessPipeline.from_config("configs/serving/batch.yaml")

    def run(self) -> dict:
        input_df = pd.read_parquet(self.config.input_path)
        total_rows = len(input_df)
        predictions = []

        for start_idx in range(0, total_rows, self.config.batch_size):
            batch = input_df.iloc[start_idx:start_idx + self.config.batch_size]

            if self.config.feature_names:
                features = self.feature_server.get_features_batch(
                    entity_ids=batch["entity_id"].tolist(),
                    feature_names=self.config.feature_names,
                )
                batch = batch.join(features, on="entity_id")

            processed = self.preprocessor.transform(batch)
            batch_preds = self.model.predict(processed)
            predictions.extend(batch_preds.tolist())

        output_df = input_df[["entity_id", "timestamp"]].copy()
        output_df["prediction"] = predictions
        output_df["model_name"] = self.config.model_name
        output_df["model_stage"] = self.config.model_stage

        output_df.to_parquet(self.config.output_path, index=False)

        return {
            "total_rows": total_rows,
            "output_path": self.config.output_path,
            "model": f"{self.config.model_name}/{self.config.model_stage}",
        }
```

### Shadow Mode

```python
# src/conduit/serving/shadow.py
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from conduit.serving.online import PredictRequest

logger = logging.getLogger(__name__)

@dataclass
class ShadowResult:
    production_prediction: float
    shadow_prediction: float
    agreement: bool
    delta: float
    timestamp: datetime

class ShadowMode:
    def __init__(self, production_model, shadow_model, threshold: float = 0.1):
        self.production_model = production_model
        self.shadow_model = shadow_model
        self.threshold = threshold
        self.results: list[ShadowResult] = []

    async def predict(self, features: pd.DataFrame) -> float:
        prod_task = asyncio.to_thread(self.production_model.predict, features)
        shadow_task = asyncio.to_thread(self.shadow_model.predict, features)

        prod_pred, shadow_pred = await asyncio.gather(prod_task, shadow_task)

        prod_val = float(prod_pred[0])
        shadow_val = float(shadow_pred[0])
        delta = abs(prod_val - shadow_val)

        self.results.append(ShadowResult(
            production_prediction=prod_val,
            shadow_prediction=shadow_val,
            agreement=delta < self.threshold,
            delta=delta,
            timestamp=datetime.utcnow(),
        ))

        if delta >= self.threshold:
            logger.warning(f"Shadow divergence: prod={prod_val:.4f}, shadow={shadow_val:.4f}, delta={delta:.4f}")

        return prod_val  # Always return production prediction

    def get_comparison_report(self) -> dict:
        if not self.results:
            return {"status": "no_data"}
        agreement_rate = sum(1 for r in self.results if r.agreement) / len(self.results)
        deltas = [r.delta for r in self.results]
        return {
            "total_predictions": len(self.results),
            "agreement_rate": agreement_rate,
            "mean_delta": sum(deltas) / len(deltas),
            "max_delta": max(deltas),
            "p95_delta": sorted(deltas)[int(len(deltas) * 0.95)],
        }
```

### Model Ensemble

```python
# src/conduit/serving/ensemble.py
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class EnsembleMember:
    model: object
    weight: float
    name: str

class ModelEnsemble:
    def __init__(self, members: list[EnsembleMember]):
        total_weight = sum(m.weight for m in members)
        self.members = [
            EnsembleMember(model=m.model, weight=m.weight / total_weight, name=m.name)
            for m in members
        ]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        weighted_predictions = []
        for member in self.members:
            pred = member.model.predict(features)
            weighted_predictions.append(pred * member.weight)
        return np.sum(weighted_predictions, axis=0)

    def predict_with_details(self, features: pd.DataFrame) -> dict:
        individual = {}
        for member in self.members:
            individual[member.name] = member.model.predict(features)

        ensemble_pred = sum(individual[m.name] * m.weight for m in self.members)
        disagreement = np.std([individual[m.name] for m in self.members], axis=0)

        return {
            "ensemble_prediction": ensemble_pred,
            "individual_predictions": individual,
            "disagreement": disagreement,
            "weights": {m.name: m.weight for m in self.members},
        }
```

### Response Cache

```python
# src/conduit/serving/cache.py
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass

@dataclass
class CacheEntry:
    value: dict
    created_at: float
    hits: int = 0

class PredictionCache:
    def __init__(self, max_size: int = 10000, ttl_seconds: float = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}

    def make_key(self, entity_id: str, context: dict) -> str:
        raw = json.dumps({"entity_id": entity_id, "context": context}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, key: str) -> dict | None:
        if key not in self.cache:
            self.stats["misses"] += 1
            return None
        entry = self.cache[key]
        if time.time() - entry.created_at > self.ttl_seconds:
            del self.cache[key]
            self.stats["misses"] += 1
            return None
        entry.hits += 1
        self.stats["hits"] += 1
        self.cache.move_to_end(key)
        return entry.value

    def put(self, key: str, value: dict) -> None:
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
            self.stats["evictions"] += 1
        self.cache[key] = CacheEntry(value=value, created_at=time.time())

    def get_stats(self) -> dict:
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {**self.stats, "size": len(self.cache), "hit_rate": hit_rate}
```

### Latency Budget Config

```yaml
# configs/serving/latency_budget.yaml
total_budget_ms: 100

stages:
  feature_lookup:
    budget_ms: 20
    alert_threshold_ms: 30
  preprocessing:
    budget_ms: 5
    alert_threshold_ms: 10
  inference:
    budget_ms: 50
    alert_threshold_ms: 70
  postprocessing:
    budget_ms: 5
    alert_threshold_ms: 10
  overhead:
    budget_ms: 20
    alert_threshold_ms: 30
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Feature store latency too high | Add local caching to feature server. Check if feature store index is built. Use batch lookups instead of one-by-one. |
| Model loading slow | Load model once at startup (not per request). Use `mlflow.pyfunc.load_model` in lifespan handler. |
| Shadow mode slowing production | Run shadow prediction asynchronously with `asyncio.create_task`. Don't await if latency critical. |
| Ensemble predictions don't improve | Check that models are diverse (different architectures or data). Correlated models don't benefit from ensembling. |
| Cache not helping latency | Check TTL isn't too short. Monitor unique entity IDs — high cardinality means fewer cache hits. |
| Batch OOM on large datasets | Reduce `batch_size` in config. Process in chunks and write incrementally. |

---

## Agent Handoff Template

```
I'm working on the Conduit project, Week 9: Model Serving Patterns.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB VRAM, 32GB RAM, Ubuntu.
Project root: ~/conduit/

Current state: [describe what's working/broken]

What I need help with: [specific issue]

Key files:
- src/conduit/serving/online.py — FastAPI online serving endpoint
- src/conduit/serving/batch.py — Batch prediction pipeline
- src/conduit/serving/shadow.py — Shadow mode (candidate testing)
- src/conduit/serving/ensemble.py — Multi-model ensemble
- src/conduit/serving/cache.py — Response caching
- src/conduit/serving/latency.py — Latency tracking and budgets
- configs/serving/ — Serving configuration files

Infrastructure: FastAPI, MLflow (model loading), Redis (feature store), Docker, PostgreSQL.
Flow: request → feature lookup → preprocess → inference → postprocess → cache → respond.
```

---

## Out of Scope

- gRPC serving (REST/HTTP only)
- GPU-accelerated inference (Triton, TensorRT) — CPU inference only
- Kubernetes autoscaling (HPA, VPA)
- Multi-region serving or CDN caching
- Streaming inference (real-time event streams)
- Model compilation (TorchScript, ONNX Runtime optimization)
- WebSocket-based serving for real-time applications
