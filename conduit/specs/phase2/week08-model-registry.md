# Week 8: Model Registry and Packaging

## Context

**Where it fits:** Week 8 of Phase 2 (Deployment + Monitoring). The model registry is the bridge between training and serving — it provides versioned, immutable model artifacts with metadata, signatures, and promotion workflows.

**Prerequisites:**
- Phase 1 complete: data pipelines, feature store, experiment tracking, training orchestration
- MLflow running with tracking server configured
- Docker installed and working (`docker build` / `docker run`)
- Understanding of model serialization (pickle, ONNX, TorchScript)

**What it builds on:** Uses experiment tracking (Week 5) to link models to their training runs. Uses training orchestration (Week 6) to register winning models from HPO studies. The registry becomes the single source of truth for all production-ready models.

**What comes next:** Week 9 (Serving Patterns) deploys models from the registry to online/batch serving endpoints.

---

## Learning Goals

- [ ] Understand model registries: why versioned model storage with metadata is essential for production ML
- [ ] Understand stage transitions: the lifecycle of a model from experimental to production
- [ ] Understand model signatures: input/output schemas that enforce contracts at inference time
- [ ] Understand model packaging: bundling code, dependencies, and artifacts for reproducible serving
- [ ] Understand promotion workflows: automated checks that gate model deployment

---

## Implementation Goals

- [ ] Integrate MLflow Model Registry with stage management (None → Staging → Production → Archived)
- [ ] Build model packaging: save model with conda env, requirements.txt, and custom code
- [ ] Implement model signature enforcement: validate inputs/outputs against declared schema
- [ ] Build Docker-based model serving: containerize model with REST endpoint
- [ ] Store rich metadata: training config, data version, metrics, experiment link, git commit
- [ ] Implement automated promotion workflow: eval gates must pass before stage transition
- [ ] Build CLI commands: `conduit model register`, `conduit model promote`, `conduit model serve`
- [ ] Implement model comparison: compare candidate vs production on evaluation dataset

---

## Acceptance Criteria

1. `conduit model register --run-id <id> --name fraud_detector` creates a new model version in MLflow Model Registry with all metadata (metrics, params, data version, git SHA).
2. Model signature is automatically inferred from training data and saved with the model; serving rejects inputs that don't match the signature schema.
3. `conduit model promote --name fraud_detector --version 3 --stage staging` transitions the model and runs all promotion checks before completing.
4. Promotion from Staging → Production fails if the candidate model scores lower than the current production model on the evaluation dataset.
5. `conduit model serve --name fraud_detector --stage production` starts a Docker container with a `/predict` endpoint that returns predictions within 100ms.
6. Model package includes all dependencies (requirements.txt with pinned versions) and can be loaded in a fresh environment without additional installs.
7. Model metadata query works: `conduit model info --name fraud_detector --version 3` returns training config, data version, metrics, and lineage.
8. Archiving a model (`--stage archived`) removes it from serving eligibility but preserves all artifacts and metadata.
9. Model comparison report (`conduit model compare --name fraud_detector --versions 2,3`) shows metric differences, latency comparison, and size delta.
10. End-to-end: train model → register → promote to staging → pass eval gate → promote to production → serve in container, all via CLI commands.

---

## Validation Commands

```bash
# Register a model from a completed training run
conduit model register --run-id abc123 --name fraud_detector \
  --description "Fraud detection model v3 with improved recall"

# List all model versions
conduit model list --name fraud_detector

# Get detailed info about a version
conduit model info --name fraud_detector --version 3

# Promote to staging (runs checks)
conduit model promote --name fraud_detector --version 3 --stage staging

# Promote to production (compares against current production)
conduit model promote --name fraud_detector --version 3 --stage production

# Compare two versions
conduit model compare --name fraud_detector --versions 2,3

# Serve the production model in Docker
conduit model serve --name fraud_detector --stage production --port 8080

# Test the served model
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"amount": 150.0, "merchant_category": "electronics", "hour": 14}}'

# Validate model signature
conduit model validate-signature --name fraud_detector --version 3 \
  --input-file tests/fixtures/sample_input.json

# Run tests
pytest tests/unit/registry/ -v
pytest tests/integration/registry/ -v --timeout=120
```

---

## Technical Implementation Details

### Project Structure (additions)

```
conduit/
├── src/conduit/
│   └── registry/
│       ├── __init__.py
│       ├── manager.py           # Model registry operations
│       ├── packaging.py         # Model packaging and dependencies
│       ├── signatures.py        # Input/output schema validation
│       ├── promotion.py         # Promotion workflow and checks
│       ├── serving.py           # Docker-based model serving
│       └── metadata.py          # Model metadata management
├── configs/
│   └── registry/
│       ├── promotion_checks.yaml
│       └── serving.yaml
├── dockerfiles/
│   └── model_server/
│       ├── Dockerfile
│       └── serve.py
```

### Model Registry Manager

```python
# src/conduit/registry/manager.py
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

class ModelStage(Enum):
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"

@dataclass
class ModelVersion:
    name: str
    version: int
    stage: ModelStage
    run_id: str
    metrics: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)
    description: str = ""

class ModelRegistryManager:
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def register(self, run_id: str, model_name: str, description: str = "") -> ModelVersion:
        run = self.client.get_run(run_id)
        model_uri = f"runs:/{run_id}/model"

        result = mlflow.register_model(model_uri=model_uri, name=model_name)

        self.client.update_model_version(
            name=model_name,
            version=result.version,
            description=description,
        )

        self._attach_metadata(model_name, result.version, run)

        return ModelVersion(
            name=model_name,
            version=int(result.version),
            stage=ModelStage.NONE,
            run_id=run_id,
            metrics=run.data.metrics,
            params=run.data.params,
        )

    def _attach_metadata(self, name: str, version: str, run) -> None:
        tags = {
            "data_version": run.data.params.get("data_version", "unknown"),
            "git_sha": run.data.tags.get("mlflow.source.git.commit", "unknown"),
            "training_config": run.data.params.get("config_path", ""),
            "registered_by": "conduit_cli",
        }
        for key, value in tags.items():
            self.client.set_model_version_tag(name, version, key, value)

    def get_version(self, name: str, version: int) -> ModelVersion:
        mv = self.client.get_model_version(name, str(version))
        run = self.client.get_run(mv.run_id)
        return ModelVersion(
            name=name,
            version=version,
            stage=ModelStage(mv.current_stage),
            run_id=mv.run_id,
            metrics=run.data.metrics,
            params=run.data.params,
            tags=mv.tags,
            description=mv.description,
        )

    def get_production_version(self, name: str) -> ModelVersion | None:
        versions = self.client.get_latest_versions(name, stages=["Production"])
        if not versions:
            return None
        mv = versions[0]
        return self.get_version(name, int(mv.version))
```

### Model Signatures

```python
# src/conduit/registry/signatures.py
from dataclasses import dataclass
import pandas as pd
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec

@dataclass
class SignatureValidationResult:
    valid: bool
    errors: list[str]

class SignatureManager:
    def infer_from_data(self, input_df: pd.DataFrame, output_df: pd.DataFrame) -> ModelSignature:
        return infer_signature(input_df, output_df)

    def create_explicit(self, input_schema: list[dict], output_schema: list[dict]) -> ModelSignature:
        input_cols = [ColSpec(type=col["type"], name=col["name"]) for col in input_schema]
        output_cols = [ColSpec(type=col["type"], name=col["name"]) for col in output_schema]
        return ModelSignature(
            inputs=Schema(input_cols),
            outputs=Schema(output_cols),
        )

    def validate_input(self, signature: ModelSignature, input_data: pd.DataFrame) -> SignatureValidationResult:
        errors = []
        expected_cols = {col.name for col in signature.inputs.inputs}
        actual_cols = set(input_data.columns)

        missing = expected_cols - actual_cols
        if missing:
            errors.append(f"Missing columns: {missing}")

        extra = actual_cols - expected_cols
        if extra:
            errors.append(f"Unexpected columns: {extra}")

        for col_spec in signature.inputs.inputs:
            if col_spec.name in input_data.columns:
                expected_dtype = self._mlflow_type_to_pandas(col_spec.type)
                actual_dtype = str(input_data[col_spec.name].dtype)
                if not self._types_compatible(expected_dtype, actual_dtype):
                    errors.append(f"Column '{col_spec.name}': expected {expected_dtype}, got {actual_dtype}")

        return SignatureValidationResult(valid=len(errors) == 0, errors=errors)

    def _mlflow_type_to_pandas(self, mlflow_type) -> str:
        mapping = {"double": "float64", "float": "float32", "long": "int64", "integer": "int32", "string": "object"}
        return mapping.get(str(mlflow_type), str(mlflow_type))

    def _types_compatible(self, expected: str, actual: str) -> bool:
        numeric_types = {"float32", "float64", "int32", "int64"}
        if expected in numeric_types and actual in numeric_types:
            return True
        return expected == actual
```

### Promotion Workflow

```python
# src/conduit/registry/promotion.py
from dataclasses import dataclass
from conduit.registry.manager import ModelRegistryManager, ModelStage, ModelVersion

@dataclass
class PromotionCheck:
    name: str
    passed: bool
    message: str

@dataclass
class PromotionResult:
    approved: bool
    checks: list[PromotionCheck]
    candidate: ModelVersion
    target_stage: ModelStage

class PromotionWorkflow:
    def __init__(self, registry: ModelRegistryManager, eval_dataset_path: str):
        self.registry = registry
        self.eval_dataset_path = eval_dataset_path

    def promote(self, name: str, version: int, target_stage: ModelStage) -> PromotionResult:
        candidate = self.registry.get_version(name, version)
        checks = []

        checks.append(self._check_metrics_above_threshold(candidate))
        checks.append(self._check_no_regression(candidate, target_stage))
        checks.append(self._check_signature_exists(candidate))
        checks.append(self._check_metadata_complete(candidate))

        approved = all(c.passed for c in checks)

        if approved:
            self.registry.client.transition_model_version_stage(
                name=name,
                version=str(version),
                stage=target_stage.value,
            )

        return PromotionResult(approved=approved, checks=checks, candidate=candidate, target_stage=target_stage)

    def _check_metrics_above_threshold(self, candidate: ModelVersion) -> PromotionCheck:
        accuracy = candidate.metrics.get("test_accuracy", 0)
        threshold = 0.85
        passed = accuracy >= threshold
        return PromotionCheck(
            name="metrics_threshold",
            passed=passed,
            message=f"Accuracy {accuracy:.4f} {'≥' if passed else '<'} {threshold}",
        )

    def _check_no_regression(self, candidate: ModelVersion, target_stage: ModelStage) -> PromotionCheck:
        if target_stage != ModelStage.PRODUCTION:
            return PromotionCheck(name="no_regression", passed=True, message="Skipped (not promoting to production)")

        current_prod = self.registry.get_production_version(candidate.name)
        if current_prod is None:
            return PromotionCheck(name="no_regression", passed=True, message="No current production model")

        candidate_acc = candidate.metrics.get("test_accuracy", 0)
        prod_acc = current_prod.metrics.get("test_accuracy", 0)
        passed = candidate_acc >= prod_acc
        return PromotionCheck(
            name="no_regression",
            passed=passed,
            message=f"Candidate {candidate_acc:.4f} vs Production {prod_acc:.4f}",
        )

    def _check_signature_exists(self, candidate: ModelVersion) -> PromotionCheck:
        has_sig = candidate.tags.get("has_signature", "false") == "true"
        return PromotionCheck(name="signature_exists", passed=has_sig, message="Model has input/output signature" if has_sig else "Missing signature")

    def _check_metadata_complete(self, candidate: ModelVersion) -> PromotionCheck:
        required_tags = ["data_version", "git_sha"]
        missing = [t for t in required_tags if t not in candidate.tags]
        passed = len(missing) == 0
        return PromotionCheck(name="metadata_complete", passed=passed, message=f"Missing: {missing}" if not passed else "All metadata present")
```

### Docker Model Server

```python
# dockerfiles/model_server/serve.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
import os
import logging

app = FastAPI(title="Conduit Model Server")
logger = logging.getLogger(__name__)

MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")

model = None

@app.on_event("startup")
def load_model():
    global model
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info(f"Loaded model {MODEL_NAME} ({MODEL_STAGE})")

class PredictRequest(BaseModel):
    features: dict

class PredictResponse(BaseModel):
    prediction: float
    model_name: str
    model_version: str

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    input_df = pd.DataFrame([request.features])
    prediction = model.predict(input_df)
    return PredictResponse(
        prediction=float(prediction[0]),
        model_name=MODEL_NAME,
        model_version=MODEL_STAGE,
    )

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME, "stage": MODEL_STAGE}
```

```dockerfile
# dockerfiles/model_server/Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serve.py .

EXPOSE 8080
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| MLflow registry not found | Ensure `MLFLOW_TRACKING_URI` is set. Start MLflow server: `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts`. |
| Model signature mismatch | Check column names and types with `model.metadata.signature`. Use `infer_signature` on actual training data. |
| Docker container can't reach MLflow | Use `--network host` or set `MLFLOW_TRACKING_URI` to host IP (not localhost inside container). |
| Promotion check failing | Run `conduit model info` to see current metrics. Adjust threshold in `configs/registry/promotion_checks.yaml`. |
| Model loading slow in container | Pre-download model artifacts into the Docker image at build time instead of fetching at startup. |
| Version conflict | MLflow auto-increments versions. Use `conduit model list` to see current versions. Don't hardcode version numbers. |

---

## Agent Handoff Template

```
I'm working on the Conduit project, Week 8: Model Registry and Packaging.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB VRAM, 32GB RAM, Ubuntu.
Project root: ~/conduit/

Current state: [describe what's working/broken]

What I need help with: [specific issue]

Key files:
- src/conduit/registry/manager.py — MLflow Model Registry integration
- src/conduit/registry/packaging.py — Model packaging with dependencies
- src/conduit/registry/signatures.py — Input/output schema validation
- src/conduit/registry/promotion.py — Promotion workflow and checks
- src/conduit/registry/serving.py — Docker container serving
- dockerfiles/model_server/ — Dockerfile and serve.py
- configs/registry/ — Promotion checks and serving config

Infrastructure: MLflow (tracking + registry), Docker, FastAPI, PostgreSQL, MinIO (artifact storage).
Flow: train → register with signature + metadata → promote through stages → serve in container.
```

---

## Out of Scope

- Kubernetes deployment (Helm charts, pod autoscaling) — Docker only
- Model format conversion (ONNX, TensorRT, TorchScript optimization)
- Multi-model endpoints (serving multiple models from one container)
- GPU-accelerated serving (Triton Inference Server)
- Model encryption or signing for security
- A/B testing between model versions (covered in Week 10)
- Cloud model registries (SageMaker Model Registry, Vertex AI Model Registry)
