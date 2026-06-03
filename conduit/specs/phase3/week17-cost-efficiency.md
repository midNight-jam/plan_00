# Week 17: Cost and Efficiency

## Context

**Where it fits:** Phase 3, Week 17 — Platform Maturity + Portfolio
**Prerequisites:** Phases 1+2 complete (training pipelines, model serving, monitoring). Week 15 CLI/SDK, Week 16 streaming operational.
**What it builds on:** The platform trains and serves models but has no visibility into costs. GPU time is expensive — a single training run can cost the equivalent of hours of compute. This week adds cost tracking, optimization strategies, and a framework for making cost-aware model decisions.

**Hardware:** ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu

---

## Learning Goals

- [ ] Understand GPU cost modeling: utilization, idle time, memory allocation waste
- [ ] Learn model distillation: training smaller models from larger model outputs
- [ ] Study inference optimization: batching, request coalescing, response caching
- [ ] Explore cost attribution: allocating shared resource costs to individual predictions
- [ ] Understand cost-benefit analysis frameworks for ML model selection
- [ ] Learn resource profiling: nvidia-smi metrics, power draw, thermal throttling
- [ ] Study efficient training techniques: mixed precision, gradient checkpointing, data parallelism tradeoffs

---

## Implementation Goals

- [ ] Build GPU cost tracker: measure GPU-hours per experiment with dollar-equivalent estimates
- [ ] Implement inference cost attribution: compute cost-per-request for each model endpoint
- [ ] Create model distillation pipeline: train student model from teacher model outputs
- [ ] Build inference optimizer: dynamic batching, request coalescing, prediction caching
- [ ] Implement resource utilization dashboard with real-time GPU metrics
- [ ] Add budget alerting: notifications when teams exceed GPU budget thresholds
- [ ] Build cost-aware model selection framework: automated cost-benefit analysis
- [ ] Create efficiency reporting: weekly summaries of waste and optimization opportunities

---

## Acceptance Criteria

1. Every training run records GPU-hours consumed (accurate to 0.01 hours) and estimated dollar cost based on configurable rate cards
2. Inference cost tracker attributes per-request cost within 10% accuracy compared to actual resource consumption measured by nvidia-smi
3. Model distillation pipeline produces a student model that achieves >95% of teacher performance at <30% of inference cost
4. Dynamic batching increases inference throughput by at least 3x compared to single-request processing at p99 latency <200ms
5. Prediction cache achieves >40% hit rate on repeated/similar inputs, reducing GPU compute by equivalent amount
6. Resource utilization dashboard updates every 10 seconds showing GPU utilization %, memory usage, power draw, and idle detection
7. Budget alerts fire within 60 seconds of a team exceeding their configured daily GPU-hour threshold
8. Cost-benefit analysis report quantifies the accuracy-vs-cost tradeoff for model variants and recommends optimal selection
9. Efficiency report identifies at least 3 categories of waste: idle GPU time, over-provisioned memory, redundant retraining
10. End-to-end demo shows training cost tracking → distillation → optimized inference → cost comparison proving >50% cost reduction

---

## Validation Commands

```bash
# Start GPU metrics collector
cd ~/conduit && python -m conduit.cost.collector --interval 1 &
COLLECTOR_PID=$!

# Run a training job and verify cost tracking
conduit model train --config configs/train_cost_demo.yaml
python -c "
from conduit.cost.tracker import CostTracker
tracker = CostTracker()
latest = tracker.get_latest_run()
print(f'GPU-hours: {latest.gpu_hours:.3f}')
print(f'Estimated cost: \${latest.estimated_cost:.2f}')
print(f'Peak memory: {latest.peak_memory_gb:.1f} GB')
assert latest.gpu_hours > 0
"

# Test model distillation
python -m conduit.cost.distillation \
  --teacher models/teacher_large.pt \
  --student-config configs/student_small.yaml \
  --data data/distillation_set.parquet \
  --output models/student_distilled.pt

# Compare teacher vs student
python -m conduit.cost.compare \
  --models models/teacher_large.pt models/student_distilled.pt \
  --test-data data/test.parquet \
  --metrics accuracy,latency,cost_per_1k

# Test dynamic batching
python -m conduit.cost.bench_batching \
  --model models/student_distilled.pt \
  --requests 5000 \
  --concurrency 50 \
  --batch-sizes 1,4,8,16,32

# Test prediction cache
python -m conduit.cost.bench_cache \
  --endpoint http://localhost:8080/predict \
  --requests 10000 \
  --cache-enabled true
echo "Cache hit rate:" && redis-cli INFO stats | grep keyspace_hits

# Check resource utilization
python -m conduit.cost.utilization --format table

# Test budget alerts
python -m conduit.cost.budget \
  --team ml-team \
  --daily-limit 2.0 \
  --simulate-usage 2.5 2>&1 | grep "ALERT"

# Generate cost-benefit report
python -m conduit.cost.analysis \
  --models models/ \
  --test-data data/test.parquet \
  --output reports/cost_benefit.html

# Cleanup
kill $COLLECTOR_PID
```

---

## Technical Implementation Details

### GPU Cost Tracker

```python
# src/conduit/cost/tracker.py
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class TrainingCostRecord:
    run_id: str
    start_time: float
    end_time: float
    gpu_hours: float
    peak_memory_gb: float
    avg_utilization_pct: float
    estimated_cost: float
    gpu_model: str

class CostTracker:
    RATE_CARDS = {
        "NVIDIA GeForce RTX 5080": 0.80,  # $/GPU-hour (local equivalent)
        "NVIDIA A100": 3.50,
        "NVIDIA H100": 5.00,
    }

    def __init__(self, storage_path: str = "~/.conduit/costs"):
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._samples: list[dict] = []

    def start_tracking(self, run_id: str):
        self._current_run_id = run_id
        self._start_time = time.time()
        self._samples = []

    def sample(self):
        metrics = self._query_nvidia_smi()
        self._samples.append({
            "timestamp": time.time(),
            **metrics,
        })

    def stop_tracking(self) -> TrainingCostRecord:
        end_time = time.time()
        duration_hours = (end_time - self._start_time) / 3600
        avg_util = sum(s["gpu_utilization"] for s in self._samples) / len(self._samples) if self._samples else 0
        peak_mem = max((s["memory_used_mb"] for s in self._samples), default=0) / 1024
        gpu_model = self._samples[0]["gpu_name"] if self._samples else "unknown"
        gpu_hours = duration_hours * (avg_util / 100)
        rate = self.RATE_CARDS.get(gpu_model, 1.00)
        estimated_cost = gpu_hours * rate

        record = TrainingCostRecord(
            run_id=self._current_run_id,
            start_time=self._start_time,
            end_time=end_time,
            gpu_hours=round(gpu_hours, 4),
            peak_memory_gb=round(peak_mem, 2),
            avg_utilization_pct=round(avg_util, 1),
            estimated_cost=round(estimated_cost, 4),
            gpu_model=gpu_model,
        )
        self._save_record(record)
        return record

    def _query_nvidia_smi(self) -> dict:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        parts = result.stdout.strip().split(", ")
        return {
            "gpu_name": parts[0],
            "gpu_utilization": float(parts[1]),
            "memory_used_mb": float(parts[2]),
            "memory_total_mb": float(parts[3]),
            "power_draw_w": float(parts[4]),
        }

    def _save_record(self, record: TrainingCostRecord):
        path = self.storage_path / f"{record.run_id}.json"
        path.write_text(json.dumps(record.__dict__, default=str))

    def get_latest_run(self) -> TrainingCostRecord:
        files = sorted(self.storage_path.glob("*.json"), key=lambda p: p.stat().st_mtime)
        data = json.loads(files[-1].read_text())
        return TrainingCostRecord(**data)
```

### Model Distillation Pipeline

```python
# src/conduit/cost/distillation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from conduit.models.base import ConduitModel

class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels):
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction="batchmean",
        ) * (self.temperature ** 2)

        hard_loss = F.cross_entropy(student_logits, labels)
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


def distill_model(
    teacher: ConduitModel,
    student: ConduitModel,
    train_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    temperature: float = 4.0,
) -> ConduitModel:
    teacher.eval()
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    criterion = DistillationLoss(temperature=temperature)

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch["features"].cuda(), batch["labels"].cuda()
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            student_logits = student(inputs)
            loss = criterion(student_logits, teacher_logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Distillation loss: {avg_loss:.4f}")

    return student
```

### Dynamic Batching for Inference

```python
# src/conduit/cost/batching.py
import asyncio
import time
from dataclasses import dataclass
from typing import Any

@dataclass
class PredictionRequest:
    request_id: str
    input_data: Any
    future: asyncio.Future
    arrived_at: float

class DynamicBatcher:
    def __init__(self, model, max_batch_size: int = 32, max_wait_ms: float = 10.0):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def start(self):
        self._running = True
        asyncio.create_task(self._batch_loop())

    async def predict(self, input_data: Any) -> Any:
        future = asyncio.get_event_loop().create_future()
        request = PredictionRequest(
            request_id=f"{time.time_ns()}",
            input_data=input_data,
            future=future,
            arrived_at=time.time(),
        )
        await self._queue.put(request)
        return await future

    async def _batch_loop(self):
        while self._running:
            batch: list[PredictionRequest] = []
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                batch.append(first)
            except asyncio.TimeoutError:
                continue

            deadline = time.time() + (self.max_wait_ms / 1000)
            while len(batch) < self.max_batch_size and time.time() < deadline:
                try:
                    remaining = deadline - time.time()
                    item = await asyncio.wait_for(self._queue.get(), timeout=max(0, remaining))
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            await self._process_batch(batch)

    async def _process_batch(self, batch: list[PredictionRequest]):
        import torch
        inputs = torch.stack([r.input_data for r in batch])
        with torch.no_grad():
            outputs = self.model(inputs.cuda())
        for request, output in zip(batch, outputs):
            request.future.set_result(output.cpu())
```

### Cost-Benefit Analysis Framework

```python
# src/conduit/cost/analysis.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelCostProfile:
    model_name: str
    accuracy: float
    latency_p50_ms: float
    latency_p99_ms: float
    cost_per_1k_requests: float
    gpu_memory_mb: float
    training_gpu_hours: float
    training_cost: float

@dataclass
class CostBenefitResult:
    recommended_model: str
    reason: str
    accuracy_delta: float
    cost_delta: float
    cost_per_accuracy_point: float

class CostBenefitAnalyzer:
    def __init__(self, budget_per_1k: Optional[float] = None, min_accuracy: Optional[float] = None):
        self.budget_per_1k = budget_per_1k
        self.min_accuracy = min_accuracy

    def analyze(self, profiles: list[ModelCostProfile]) -> CostBenefitResult:
        viable = profiles
        if self.min_accuracy:
            viable = [p for p in viable if p.accuracy >= self.min_accuracy]
        if self.budget_per_1k:
            viable = [p for p in viable if p.cost_per_1k_requests <= self.budget_per_1k]

        if not viable:
            best = max(profiles, key=lambda p: p.accuracy)
            return CostBenefitResult(
                recommended_model=best.model_name,
                reason="No model meets constraints. Showing best accuracy.",
                accuracy_delta=0, cost_delta=0, cost_per_accuracy_point=0,
            )

        # Pareto optimal: best accuracy at each cost level
        viable.sort(key=lambda p: p.cost_per_1k_requests)
        pareto = []
        max_acc = 0
        for p in viable:
            if p.accuracy > max_acc:
                pareto.append(p)
                max_acc = p.accuracy

        # Find best value: highest accuracy / cost ratio
        best = max(pareto, key=lambda p: p.accuracy / max(p.cost_per_1k_requests, 0.001))
        most_accurate = max(pareto, key=lambda p: p.accuracy)

        if best.model_name == most_accurate.model_name:
            reason = "Best model is also most cost-efficient."
        else:
            acc_gain = most_accurate.accuracy - best.accuracy
            cost_increase = most_accurate.cost_per_1k_requests - best.cost_per_1k_requests
            reason = (f"'{most_accurate.model_name}' is {acc_gain:.1%} more accurate but "
                      f"${cost_increase:.3f}/1k more expensive. Recommending '{best.model_name}' for value.")

        return CostBenefitResult(
            recommended_model=best.model_name,
            reason=reason,
            accuracy_delta=most_accurate.accuracy - best.accuracy,
            cost_delta=most_accurate.cost_per_1k_requests - best.cost_per_1k_requests,
            cost_per_accuracy_point=(most_accurate.cost_per_1k_requests - best.cost_per_1k_requests)
                                    / max(most_accurate.accuracy - best.accuracy, 0.001),
        )
```

### Project file structure:
```
~/conduit/src/conduit/cost/
├── __init__.py
├── tracker.py          # GPU cost tracking per training run
├── collector.py        # Background nvidia-smi metrics collector
├── distillation.py     # Model distillation pipeline
├── batching.py         # Dynamic batching for inference
├── cache.py            # Prediction caching layer
├── analysis.py         # Cost-benefit analysis framework
├── budget.py           # Budget alerts and thresholds
├── utilization.py      # Resource utilization dashboard data
├── compare.py          # Model comparison (accuracy vs cost)
└── bench_batching.py   # Batching benchmark tool
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| `nvidia-smi` not returning data in containers | Mount `/dev/nvidia*` devices and install `nvidia-container-toolkit`. For WSL2: ensure NVIDIA drivers are installed on the Windows host |
| Distillation loss diverging | Lower learning rate to 1e-4, increase temperature to 6.0, ensure teacher model is in eval mode with `torch.no_grad()` |
| Dynamic batching adds too much latency | Reduce `max_wait_ms` to 5ms. For low-traffic endpoints, set `max_batch_size=1` to disable batching |
| Cost calculations inconsistent | Ensure nvidia-smi sampling interval matches actual compute (sample every 1s, not 10s). Account for multi-GPU by summing across devices |
| Cache invalidation after model update | Version cache keys with model version hash: `f"{model_version}:{input_hash}"`. Clear on deploy |
| Budget alerts firing too frequently | Add cooldown period (e.g., 1 alert per hour). Use exponential smoothing for usage estimates |

---

## Agent Handoff Template

```
I'm building Week 17 of the Conduit ML platform: Cost and Efficiency.

Current state:
- Full ML lifecycle from Phases 1+2 operational
- CLI/SDK (Week 15) and streaming (Week 16) working
- No cost visibility or optimization exists yet

What I need help with:
- [specific task: e.g., "implementing model distillation pipeline with knowledge distillation loss"]

Key files:
- GPU tracker: src/conduit/cost/tracker.py
- Distillation: src/conduit/cost/distillation.py
- Dynamic batching: src/conduit/cost/batching.py
- Cost analysis: src/conduit/cost/analysis.py
- Budget alerts: src/conduit/cost/budget.py

Tech stack: Python 3.11, PyTorch, nvidia-smi, Redis (caching), Rich (dashboards)
Hardware: RTX 5080 16GB, 32GB RAM, Ubuntu

The goal is full cost visibility per training run and inference request,
plus optimization via distillation, batching, and caching to reduce GPU costs by >50%.
```

---

## Out of Scope

- Cloud cost APIs (AWS/GCP billing integration — local only)
- Multi-GPU distributed training cost splitting
- Spot instance / preemptible VM scheduling
- Hardware procurement recommendations
- Carbon footprint / energy efficiency tracking
- Financial forecasting / capacity planning
- Chargeback billing system between teams
