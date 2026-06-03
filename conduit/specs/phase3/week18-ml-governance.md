# Week 18: ML Governance and Compliance

## Context

**Where it fits:** Phase 3, Week 18 — Platform Maturity + Portfolio
**Prerequisites:** Phases 1+2 complete (full ML lifecycle). Weeks 15-17 (CLI/SDK, streaming, cost tracking) operational.
**What it builds on:** The platform can train, serve, and monitor models efficiently. But there's no formal governance: no documentation of what models do, no audit trail for decisions, no bias detection, no approval process for deploying to production. This week adds the governance layer required for responsible ML in regulated environments.

**Hardware:** ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu

---

## Learning Goals

- [ ] Understand model cards: standardized model documentation (Google's Model Cards for Model Reporting)
- [ ] Learn ML audit trails: data lineage, model provenance, deployment decisions
- [ ] Study fairness metrics: demographic parity, equalized odds, predictive equality
- [ ] Explore GDPR requirements for ML: right to explanation, data minimization, purpose limitation
- [ ] Understand the EU AI Act: risk classification, high-risk requirements, transparency obligations
- [ ] Learn approval workflow patterns: multi-stage gates, role-based sign-off
- [ ] Study model deprecation strategies: graceful retirement, traffic shifting, fallback models

---

## Implementation Goals

- [ ] Build model card generator: auto-populate from training metadata + manual sections
- [ ] Implement complete audit trail: data → features → training → evaluation → deployment
- [ ] Create bias detection system: compute fairness metrics across demographic groups
- [ ] Build compliance documentation generator: data usage, retention, consent tracking
- [ ] Implement model deprecation workflow: sunset timeline, migration plan, fallback routing
- [ ] Create approval workflow engine: configurable multi-stage gates with role-based sign-off
- [ ] Build automated risk scoring: assess risk level of model changes based on impact and data sensitivity
- [ ] Add regulatory report generator: GDPR Article 22, AI Act conformity assessment

---

## Acceptance Criteria

1. Model cards generate automatically from training metadata and include: purpose, architecture, training data description, performance metrics, known limitations, and ethical considerations
2. Audit trail captures every step from raw data ingestion through production deployment with timestamps, actors, and rationale — queryable via `conduit audit show <model-id>`
3. Bias detection computes demographic parity difference, equalized odds ratio, and disparate impact across at least 3 protected attributes with results in <30 seconds
4. Compliance report documents all training data sources, their consent basis, retention policy, and data minimization justification for each feature
5. Model deprecation workflow sends notifications 30/14/7 days before sunset, shifts traffic gradually, and maintains fallback model until retirement completes
6. Approval workflow blocks production deployment until all required approvals (ML engineer, data scientist, compliance officer) are recorded
7. Risk scoring evaluates model changes on 5 dimensions (data sensitivity, population impact, accuracy change, fairness delta, reversibility) and produces a score from 1-5
8. GDPR explanation endpoint generates human-readable explanations for individual predictions using SHAP values within 2 seconds
9. Full governance metadata is versioned in Git alongside model artifacts — `conduit governance history <model-id>` shows complete changelog
10. End-to-end demo: model trained → bias checked → model card generated → approval requested → approved → deployed → audit trail queryable

---

## Validation Commands

```bash
# Generate model card for an existing model
conduit governance model-card generate \
  --model models/fraud_detector_v3 \
  --output docs/model_cards/fraud_detector_v3.md
cat docs/model_cards/fraud_detector_v3.md | head -50

# Query audit trail
conduit audit show fraud_detector_v3 --format json | python -m json.tool
conduit audit show fraud_detector_v3 --format timeline

# Run bias detection
conduit governance bias-check \
  --model models/fraud_detector_v3 \
  --test-data data/test_with_demographics.parquet \
  --protected-attrs gender,age_group,ethnicity \
  --output reports/bias_report.html

# Verify fairness metrics
python -c "
from conduit.governance.fairness import FairnessEvaluator
evaluator = FairnessEvaluator()
report = evaluator.evaluate('models/fraud_detector_v3', 'data/test_with_demographics.parquet')
print(f'Demographic parity diff: {report.demographic_parity_diff:.4f}')
print(f'Equalized odds ratio: {report.equalized_odds_ratio:.4f}')
assert report.demographic_parity_diff < 0.1, 'Fairness threshold exceeded'
"

# Test approval workflow
conduit governance request-approval \
  --model fraud_detector_v3 \
  --stage production \
  --reviewer ml-lead,compliance

conduit governance approve \
  --model fraud_detector_v3 \
  --stage production \
  --reviewer ml-lead \
  --comment "Metrics look good, bias within thresholds"

conduit governance status fraud_detector_v3

# Test risk scoring
conduit governance risk-score \
  --model fraud_detector_v3 \
  --change "retrained on 2 months additional data"

# Generate compliance report
conduit governance compliance-report \
  --model fraud_detector_v3 \
  --regulation gdpr \
  --output reports/gdpr_compliance.pdf

# Test model deprecation
conduit governance deprecate \
  --model fraud_detector_v2 \
  --sunset-date 2026-07-01 \
  --fallback fraud_detector_v3 \
  --notify team-ml@company.com

# Test GDPR explanation endpoint
python -c "
from conduit.governance.explanations import ExplainabilityEngine
engine = ExplainabilityEngine('models/fraud_detector_v3')
explanation = engine.explain(sample_input={'amount': 5000, 'country': 'NG', 'hour': 3})
print(f'Decision: {explanation.decision}')
print(f'Top factors: {explanation.top_factors[:3]}')
print(f'Explanation: {explanation.natural_language}')
"
```

---

## Technical Implementation Details

### Model Card Generator

```python
# src/conduit/governance/model_card.py
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml

@dataclass
class ModelCard:
    model_name: str
    version: str
    created_at: datetime
    # Overview
    purpose: str = ""
    intended_use: str = ""
    out_of_scope_use: str = ""
    # Architecture
    model_type: str = ""
    architecture_details: str = ""
    input_format: str = ""
    output_format: str = ""
    # Training
    training_data_description: str = ""
    training_data_size: int = 0
    training_date: Optional[datetime] = None
    training_duration_hours: float = 0
    hyperparameters: dict = field(default_factory=dict)
    # Performance
    metrics: dict = field(default_factory=dict)
    performance_by_group: dict = field(default_factory=dict)
    # Limitations
    known_limitations: list[str] = field(default_factory=list)
    ethical_considerations: list[str] = field(default_factory=list)
    # Governance
    owner: str = ""
    contact: str = ""
    review_date: Optional[datetime] = None
    risk_level: str = "medium"

    @classmethod
    def from_training_metadata(cls, model_path: str) -> "ModelCard":
        meta_path = Path(model_path) / "metadata.yaml"
        meta = yaml.safe_load(meta_path.read_text())
        return cls(
            model_name=meta["model_name"],
            version=meta["version"],
            created_at=datetime.fromisoformat(meta["created_at"]),
            model_type=meta.get("model_type", ""),
            training_data_size=meta.get("training_samples", 0),
            training_duration_hours=meta.get("training_hours", 0),
            hyperparameters=meta.get("hyperparameters", {}),
            metrics=meta.get("final_metrics", {}),
        )

    def render_markdown(self) -> str:
        sections = [
            f"# Model Card: {self.model_name} v{self.version}\n",
            f"**Created:** {self.created_at.isoformat()}  ",
            f"**Owner:** {self.owner}  ",
            f"**Risk Level:** {self.risk_level}\n",
            "## Purpose and Intended Use\n",
            f"{self.purpose}\n",
            f"**Intended use:** {self.intended_use}  ",
            f"**Out-of-scope use:** {self.out_of_scope_use}\n",
            "## Architecture\n",
            f"- **Type:** {self.model_type}",
            f"- **Input:** {self.input_format}",
            f"- **Output:** {self.output_format}",
            f"- **Details:** {self.architecture_details}\n",
            "## Training Data\n",
            f"{self.training_data_description}\n",
            f"- **Samples:** {self.training_data_size:,}",
            f"- **Duration:** {self.training_duration_hours:.1f} hours\n",
            "## Performance Metrics\n",
            *[f"- **{k}:** {v}" for k, v in self.metrics.items()],
            "\n## Known Limitations\n",
            *[f"- {lim}" for lim in self.known_limitations],
            "\n## Ethical Considerations\n",
            *[f"- {eth}" for eth in self.ethical_considerations],
        ]
        return "\n".join(sections)
```

### Audit Trail System

```python
# src/conduit/governance/audit.py
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

class AuditEventType(Enum):
    DATA_INGESTED = "data_ingested"
    FEATURES_COMPUTED = "features_computed"
    TRAINING_STARTED = "training_started"
    TRAINING_COMPLETED = "training_completed"
    EVALUATION_RUN = "evaluation_run"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    DEPLOYMENT_STARTED = "deployment_started"
    DEPLOYMENT_COMPLETED = "deployment_completed"
    MODEL_DEPRECATED = "model_deprecated"

@dataclass
class AuditEvent:
    event_type: AuditEventType
    model_id: str
    timestamp: datetime
    actor: str
    details: dict
    parent_event_hash: str = ""

    @property
    def event_hash(self) -> str:
        content = json.dumps({
            "type": self.event_type.value,
            "model": self.model_id,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "parent": self.parent_event_hash,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

class AuditTrail:
    def __init__(self, storage_dir: str = "~/.conduit/audit"):
        self.storage_dir = Path(storage_dir).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def record(self, event: AuditEvent):
        model_dir = self.storage_dir / event.model_id
        model_dir.mkdir(exist_ok=True)
        trail = self._load_trail(event.model_id)
        if trail:
            event.parent_event_hash = trail[-1].event_hash
        trail.append(event)
        self._save_trail(event.model_id, trail)

    def get_trail(self, model_id: str) -> list[AuditEvent]:
        return self._load_trail(model_id)

    def verify_integrity(self, model_id: str) -> bool:
        trail = self._load_trail(model_id)
        for i, event in enumerate(trail):
            if i > 0:
                expected_parent = trail[i-1].event_hash
                if event.parent_event_hash != expected_parent:
                    return False
        return True

    def _load_trail(self, model_id: str) -> list[AuditEvent]:
        path = self.storage_dir / model_id / "trail.jsonl"
        if not path.exists():
            return []
        events = []
        for line in path.read_text().strip().split("\n"):
            data = json.loads(line)
            data["event_type"] = AuditEventType(data["event_type"])
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            events.append(AuditEvent(**data))
        return events

    def _save_trail(self, model_id: str, trail: list[AuditEvent]):
        path = self.storage_dir / model_id / "trail.jsonl"
        lines = []
        for event in trail:
            data = asdict(event)
            data["event_type"] = event.event_type.value
            data["timestamp"] = event.timestamp.isoformat()
            lines.append(json.dumps(data))
        path.write_text("\n".join(lines))
```

### Bias Detection / Fairness Evaluator

```python
# src/conduit/governance/fairness.py
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class FairnessReport:
    model_id: str
    protected_attributes: list[str]
    demographic_parity_diff: float
    equalized_odds_ratio: float
    disparate_impact: float
    group_metrics: dict
    recommendations: list[str]
    passes_threshold: bool

class FairnessEvaluator:
    def __init__(self, dp_threshold: float = 0.1, di_threshold: float = 0.8):
        self.dp_threshold = dp_threshold
        self.di_threshold = di_threshold

    def evaluate(self, model_path: str, test_data_path: str,
                 protected_attrs: list[str] = None) -> FairnessReport:
        import torch
        model = torch.load(f"{model_path}/model.pt", map_location="cpu")
        model.eval()
        df = pd.read_parquet(test_data_path)
        protected_attrs = protected_attrs or ["gender", "age_group", "ethnicity"]
        predictions = self._get_predictions(model, df)
        group_metrics = {}
        max_dp_diff = 0

        for attr in protected_attrs:
            if attr not in df.columns:
                continue
            groups = df[attr].unique()
            group_rates = {}
            for group in groups:
                mask = df[attr] == group
                group_preds = predictions[mask]
                positive_rate = (group_preds > 0.5).mean()
                group_rates[str(group)] = float(positive_rate)
            group_metrics[attr] = group_rates
            rates = list(group_rates.values())
            dp_diff = max(rates) - min(rates)
            max_dp_diff = max(max_dp_diff, dp_diff)

        min_rate = min(min(gm.values()) for gm in group_metrics.values())
        max_rate = max(max(gm.values()) for gm in group_metrics.values())
        disparate_impact = min_rate / max_rate if max_rate > 0 else 0
        equalized_odds = self._compute_equalized_odds(df, predictions, protected_attrs)

        recommendations = []
        if max_dp_diff > self.dp_threshold:
            recommendations.append(f"Demographic parity difference ({max_dp_diff:.3f}) exceeds threshold ({self.dp_threshold}). Consider resampling or reweighting training data.")
        if disparate_impact < self.di_threshold:
            recommendations.append(f"Disparate impact ({disparate_impact:.3f}) below {self.di_threshold}. Model may be discriminatory under four-fifths rule.")

        return FairnessReport(
            model_id=model_path,
            protected_attributes=protected_attrs,
            demographic_parity_diff=max_dp_diff,
            equalized_odds_ratio=equalized_odds,
            disparate_impact=disparate_impact,
            group_metrics=group_metrics,
            recommendations=recommendations,
            passes_threshold=(max_dp_diff <= self.dp_threshold and disparate_impact >= self.di_threshold),
        )

    def _compute_equalized_odds(self, df, predictions, protected_attrs) -> float:
        # Simplified: ratio of true positive rates across groups
        labels = df["label"].values
        tpr_ratios = []
        for attr in protected_attrs:
            if attr not in df.columns:
                continue
            groups = df[attr].unique()
            tprs = []
            for group in groups:
                mask = (df[attr] == group) & (labels == 1)
                if mask.sum() > 0:
                    tprs.append((predictions[mask] > 0.5).mean())
            if len(tprs) >= 2:
                tpr_ratios.append(min(tprs) / max(tprs) if max(tprs) > 0 else 0)
        return float(np.mean(tpr_ratios)) if tpr_ratios else 1.0

    def _get_predictions(self, model, df) -> np.ndarray:
        import torch
        feature_cols = [c for c in df.columns if c not in ["label", "gender", "age_group", "ethnicity"]]
        X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        with torch.no_grad():
            preds = torch.sigmoid(model(X)).numpy().flatten()
        return preds
```

### Approval Workflow Engine

```python
# src/conduit/governance/approvals.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import json
from pathlib import Path

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class ApprovalRequest:
    model_id: str
    stage: str  # "staging", "production"
    requester: str
    required_reviewers: list[str]
    approvals: dict = field(default_factory=dict)  # reviewer -> (status, comment, timestamp)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    @property
    def status(self) -> ApprovalStatus:
        if self.expires_at and datetime.now() > self.expires_at:
            return ApprovalStatus.EXPIRED
        for reviewer, (status, _, _) in self.approvals.items():
            if status == ApprovalStatus.REJECTED:
                return ApprovalStatus.REJECTED
        if all(r in self.approvals for r in self.required_reviewers):
            if all(self.approvals[r][0] == ApprovalStatus.APPROVED for r in self.required_reviewers):
                return ApprovalStatus.APPROVED
        return ApprovalStatus.PENDING

    def approve(self, reviewer: str, comment: str = ""):
        if reviewer not in self.required_reviewers:
            raise ValueError(f"{reviewer} is not a required reviewer")
        self.approvals[reviewer] = (ApprovalStatus.APPROVED, comment, datetime.now())

    def reject(self, reviewer: str, comment: str = ""):
        if reviewer not in self.required_reviewers:
            raise ValueError(f"{reviewer} is not a required reviewer")
        self.approvals[reviewer] = (ApprovalStatus.REJECTED, comment, datetime.now())

class ApprovalWorkflow:
    def __init__(self, storage_dir: str = "~/.conduit/approvals"):
        self.storage_dir = Path(storage_dir).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def request_approval(self, model_id: str, stage: str, requester: str, reviewers: list[str]) -> ApprovalRequest:
        request = ApprovalRequest(
            model_id=model_id,
            stage=stage,
            requester=requester,
            required_reviewers=reviewers,
        )
        self._save(request)
        self._notify_reviewers(request)
        return request

    def can_deploy(self, model_id: str, stage: str) -> bool:
        request = self._load(model_id, stage)
        if request is None:
            return False
        return request.status == ApprovalStatus.APPROVED

    def _notify_reviewers(self, request: ApprovalRequest):
        for reviewer in request.required_reviewers:
            print(f"[NOTIFY] {reviewer}: Approval requested for "
                  f"{request.model_id} → {request.stage} by {request.requester}")

    def _save(self, request: ApprovalRequest):
        path = self.storage_dir / f"{request.model_id}_{request.stage}.json"
        path.write_text(json.dumps({
            "model_id": request.model_id,
            "stage": request.stage,
            "requester": request.requester,
            "required_reviewers": request.required_reviewers,
            "created_at": request.created_at.isoformat(),
        }))

    def _load(self, model_id: str, stage: str) -> Optional[ApprovalRequest]:
        path = self.storage_dir / f"{model_id}_{stage}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return ApprovalRequest(**{k: v for k, v in data.items() if k != "created_at"},
                               created_at=datetime.fromisoformat(data["created_at"]))
```

### Project file structure:
```
~/conduit/src/conduit/governance/
├── __init__.py
├── model_card.py       # Model card generation and rendering
├── audit.py            # Immutable audit trail with hash chaining
├── fairness.py         # Bias detection and fairness metrics
├── approvals.py        # Multi-stage approval workflow
├── risk.py             # Automated risk scoring
├── deprecation.py      # Model sunset and fallback routing
├── explanations.py     # SHAP-based prediction explanations
├── compliance.py       # GDPR/AI Act report generation
└── templates/
    ├── model_card.md.j2
    ├── compliance_report.html.j2
    └── bias_report.html.j2
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| SHAP explanations too slow for real-time | Use `shap.KernelExplainer` with `nsamples=100` for speed, or pre-compute SHAP for common input patterns and cache results |
| Audit trail hash chain broken after crash | Implement write-ahead logging: write event to WAL first, then append to trail. On recovery, replay WAL |
| Fairness metrics undefined for small groups | Set minimum group size threshold (e.g., n>30). Report "insufficient data" rather than unreliable metrics |
| Approval workflow blocking CI/CD | Add `--auto-approve` for non-production stages. Production always requires explicit approval |
| Model card too verbose for quick review | Generate both full model card and a 1-page summary. Default CLI shows summary with `--full` flag for details |
| Compliance report missing data sources | Enforce data registration at ingestion time via CLI guardrails. Block training if data lineage is incomplete |

---

## Agent Handoff Template

```
I'm building Week 18 of the Conduit ML platform: ML Governance and Compliance.

Current state:
- Full ML lifecycle from Phases 1+2 operational
- CLI/SDK, streaming, cost tracking from Weeks 15-17 working
- No governance, audit trail, or compliance documentation exists yet

What I need help with:
- [specific task: e.g., "implementing fairness evaluation with demographic parity and equalized odds"]

Key files:
- Model cards: src/conduit/governance/model_card.py
- Audit trail: src/conduit/governance/audit.py
- Fairness: src/conduit/governance/fairness.py
- Approvals: src/conduit/governance/approvals.py
- Risk scoring: src/conduit/governance/risk.py

Tech stack: Python 3.11, SHAP, Pydantic v2, Jinja2 (report templates)
Hardware: RTX 5080 16GB, 32GB RAM, Ubuntu

The goal is enterprise-grade ML governance: model cards, audit trails,
bias detection, approval workflows, and regulatory compliance documentation.
```

---

## Out of Scope

- Legal review of compliance reports (engineering builds the tools, legal validates content)
- Real GDPR data subject access request (DSAR) processing
- EU AI Act conformity assessment submission
- Integration with legal/compliance ticketing systems (ServiceNow, etc.)
- Privacy-preserving ML (federated learning, differential privacy)
- Responsible AI committee formation and processes
- Third-party model audit certification
