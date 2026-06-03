# Week 18: Infrastructure Performance Engineering

## Context

**Where it fits:** Phase C, Week 18 of the Anvil AI Infrastructure project. With observability in place (Week 17), this week uses that data to systematically benchmark, optimize, and model infrastructure performance.

**Prerequisites:**
- Full observability stack operational (Prometheus, Loki, Thanos, Grafana) from Week 17
- Historical metrics available for at least simulated workload patterns
- All controllers running with measurable latency (scheduler, checkpoint, health monitor)
- CI/CD pipeline from Phase B capable of running benchmarks
- Basic queueing theory understanding

**What it builds on:** Weeks 15-17 built the platform, hardened it, and made it observable. This week answers "how fast is it?" and "how do we make it faster?" — turning observability data into actionable performance insights and establishing performance regression gates.

---

## Learning Goals

- [ ] Understand infrastructure benchmarking methodology: controlled experiments, statistical significance, warmup
- [ ] Learn queueing theory for capacity planning: M/M/c models, Little's Law, utilization vs wait time curves
- [ ] Study checkpoint I/O optimization: parallel writes, compression, incremental checkpoints
- [ ] Understand Kubernetes scheduler performance: scheduling throughput, queue depth impact
- [ ] Learn performance regression detection: statistical comparison, confidence intervals, effect size
- [ ] Study model preloading and warm-start techniques for reducing cold start latency
- [ ] Understand connection pooling patterns for high-throughput gRPC/HTTP services

---

## Implementation Goals

- [ ] Build benchmark suite measuring: scheduler latency, checkpoint speed, model load time, cross-component latency
- [ ] Optimize scheduler: reduce decision time from P99 to under 100ms for 1000-job queue
- [ ] Implement parallel checkpoint writes with configurable worker count
- [ ] Build model preloading system that pre-fetches models based on schedule predictions
- [ ] Implement connection pooling for all inter-component gRPC connections
- [ ] Create capacity model: given N GPUs, predict max concurrent jobs and expected wait times
- [ ] Build what-if analysis tool: simulate adding/removing GPUs, changing job mix
- [ ] Implement performance regression CI gate: block merges that regress P99 latency by >10%
- [ ] Create benchmark dashboard showing historical performance trends
- [ ] Document performance tuning knobs and their expected impact

---

## Acceptance Criteria

1. Benchmark suite runs reproducibly and outputs results in structured format (JSON + human-readable), covering scheduler latency, checkpoint throughput, model load time, and end-to-end job submission latency.
2. Scheduler P99 latency is under 100ms when processing a queue of 1000 pending jobs (measured via benchmark, down from baseline).
3. Checkpoint write throughput improves by at least 2x with parallel writes enabled (4 workers) compared to sequential baseline for a 2GB model.
4. Model preloading reduces cold-start inference latency by at least 50% for models in the prediction window.
5. Connection pooling reduces gRPC connection establishment overhead: P99 latency for inter-component calls drops below 5ms.
6. Capacity model accurately predicts (within 15% error) the queue wait time for a given GPU count and job arrival rate, validated against simulation.
7. What-if analysis tool outputs: "Adding 10 RTX 5080 GPUs reduces average wait time from X to Y minutes" with confidence interval.
8. CI performance gate blocks PRs that regress scheduler P99 by more than 10%, with clear diff report showing before/after.
9. Benchmark dashboard shows 30-day trend of all key latency metrics with automated annotations for deployments.
10. Performance tuning guide documents at least 8 configuration knobs with measured impact (e.g., "increasing scheduler workers from 4 to 8 reduces P99 by 30%").

---

## Validation Commands

```bash
# Run the full benchmark suite
cd ~/anvil/benchmarks && make run-all

# Individual benchmarks
make bench-scheduler     # Scheduler latency under load
make bench-checkpoint    # Checkpoint I/O throughput
make bench-model-load    # Model loading cold/warm start
make bench-e2e           # End-to-end job submission to running

# View results
cat results/latest/summary.json | jq .
cat results/latest/report.md

# Compare against baseline
make compare BASELINE=results/v0.15.0/summary.json CURRENT=results/latest/summary.json

# Run capacity model
cd ~/anvil/capacity && python model.py \
  --gpus 8 \
  --gpu-type rtx5080 \
  --arrival-rate 5 \
  --avg-job-duration 2h

# What-if analysis
python what_if.py \
  --current-gpus 8 \
  --add-gpus 10 \
  --current-wait-p50 "$(cat results/latest/summary.json | jq '.queue_wait_p50')"

# Test parallel checkpoint
anvil train submit --model ./examples/large-model --gpu 2 --checkpoint-workers 4
anvil train logs --name <job> | grep "checkpoint_duration"

# Performance regression CI check
cd ~/anvil && make ci-perf-gate

# Check connection pool metrics
curl -s http://localhost:9090/api/v1/query?query=anvil_grpc_pool_active_connections | jq .

# Model preloading verification
anvil model preload --name my-model --checkpoint /mnt/checkpoints/latest
time anvil model deploy --name my-model --replicas 1
```

---

## Technical Implementation Details

### Project Structure

```
~/anvil/benchmarks/
├── cmd/
│   └── bench/
│       └── main.go               # Benchmark runner CLI
├── suite/
│   ├── scheduler_bench.go        # Scheduler latency benchmarks
│   ├── checkpoint_bench.go       # Checkpoint I/O benchmarks
│   ├── model_load_bench.go       # Model loading benchmarks
│   ├── e2e_bench.go              # End-to-end latency
│   └── common.go                 # Shared utilities, warmup, stats
├── results/
│   └── .gitkeep
├── ci/
│   ├── regression_gate.go        # CI performance gate logic
│   └── compare.go                # Statistical comparison
├── Makefile
└── README.md

~/anvil/capacity/
├── model.py                      # M/M/c queue model
├── what_if.py                    # What-if scenario analysis
├── simulator.py                  # Discrete event simulator
├── visualize.py                  # Generates capacity charts
└── tests/
    └── test_model.py
```

### Benchmark Framework

```go
// benchmarks/suite/common.go
package suite

import (
    "encoding/json"
    "math"
    "sort"
    "time"
)

type BenchmarkResult struct {
    Name       string        `json:"name"`
    Iterations int           `json:"iterations"`
    P50        time.Duration `json:"p50_ms"`
    P95        time.Duration `json:"p95_ms"`
    P99        time.Duration `json:"p99_ms"`
    Mean       time.Duration `json:"mean_ms"`
    StdDev     time.Duration `json:"stddev_ms"`
    Throughput float64       `json:"throughput_ops_sec"`
}

type Benchmark struct {
    WarmupIterations int
    Iterations       int
    latencies        []time.Duration
}

func (b *Benchmark) Run(name string, fn func() error) (*BenchmarkResult, error) {
    // Warmup phase
    for i := 0; i < b.WarmupIterations; i++ {
        if err := fn(); err != nil {
            return nil, fmt.Errorf("warmup failed: %w", err)
        }
    }

    // Measurement phase
    b.latencies = make([]time.Duration, 0, b.Iterations)
    for i := 0; i < b.Iterations; i++ {
        start := time.Now()
        if err := fn(); err != nil {
            return nil, fmt.Errorf("iteration %d failed: %w", i, err)
        }
        b.latencies = append(b.latencies, time.Since(start))
    }

    return b.computeStats(name), nil
}

func (b *Benchmark) computeStats(name string) *BenchmarkResult {
    sort.Slice(b.latencies, func(i, j int) bool { return b.latencies[i] < b.latencies[j] })
    n := len(b.latencies)
    return &BenchmarkResult{
        Name:       name,
        Iterations: n,
        P50:        b.latencies[n*50/100],
        P95:        b.latencies[n*95/100],
        P99:        b.latencies[n*99/100],
        Mean:       mean(b.latencies),
        StdDev:     stddev(b.latencies),
        Throughput: float64(n) / totalDuration(b.latencies).Seconds(),
    }
}
```

### Scheduler Optimization

```go
// benchmarks/suite/scheduler_bench.go
package suite

func BenchScheduler(ctx context.Context, client client.Client) (*BenchmarkResult, error) {
    b := &Benchmark{WarmupIterations: 10, Iterations: 100}

    // Pre-populate queue with 1000 pending jobs
    for i := 0; i < 1000; i++ {
        createPendingJob(ctx, client, fmt.Sprintf("bench-job-%d", i))
    }

    return b.Run("scheduler_decision_latency", func() error {
        job := createPendingJob(ctx, client, fmt.Sprintf("bench-trigger-%d", time.Now().UnixNano()))
        return waitForScheduled(ctx, client, job, 5*time.Second)
    })
}

// Optimization: batch scoring with parallel node evaluation
type OptimizedScheduler struct {
    workerCount   int
    nodeScoreCache *sync.Map
    batchSize     int
}

func (s *OptimizedScheduler) ScheduleBatch(ctx context.Context, jobs []*v1alpha1.TrainingJob) []ScheduleDecision {
    nodeCh := make(chan *corev1.Node, 100)
    resultCh := make(chan nodeScore, 100)

    // Parallel node scoring
    for i := 0; i < s.workerCount; i++ {
        go func() {
            for node := range nodeCh {
                score := s.scoreNode(ctx, node, jobs)
                resultCh <- score
            }
        }()
    }
    // ... collect results and assign
}
```

### Capacity Model (M/M/c Queue)

```python
# capacity/model.py
import math
from scipy.special import factorial
import numpy as np

class MMcCapacityModel:
    """M/M/c queue model for GPU cluster capacity planning."""

    def __init__(self, num_gpus: int, arrival_rate: float, service_rate: float):
        """
        Args:
            num_gpus: Number of GPU slots (servers in queue theory)
            arrival_rate: Jobs per hour (lambda)
            service_rate: Jobs completed per GPU per hour (mu)
        """
        self.c = num_gpus
        self.lam = arrival_rate
        self.mu = service_rate
        self.rho = arrival_rate / (num_gpus * service_rate)

    def erlang_c(self) -> float:
        """Probability that an arriving job must wait."""
        c, rho = self.c, self.rho
        sum_terms = sum((c * rho)**k / factorial(k, exact=True) for k in range(c))
        last_term = (c * rho)**c / (factorial(c, exact=True) * (1 - rho))
        return last_term / (sum_terms + last_term)

    def avg_wait_time(self) -> float:
        """Average wait time in queue (hours)."""
        if self.rho >= 1.0:
            return float('inf')
        return self.erlang_c() / (self.c * self.mu * (1 - self.rho))

    def avg_queue_length(self) -> float:
        """Average number of jobs waiting."""
        return self.lam * self.avg_wait_time()

    def utilization(self) -> float:
        """Server utilization (0 to 1)."""
        return self.rho

    def what_if_add_gpus(self, additional: int) -> dict:
        """Predict impact of adding GPUs."""
        current_wait = self.avg_wait_time()
        new_model = MMcCapacityModel(self.c + additional, self.lam, self.mu)
        new_wait = new_model.avg_wait_time()
        return {
            "current_gpus": self.c,
            "new_gpus": self.c + additional,
            "current_wait_minutes": current_wait * 60,
            "new_wait_minutes": new_wait * 60,
            "wait_reduction_percent": (1 - new_wait / current_wait) * 100,
            "new_utilization": new_model.utilization(),
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--gpu-type", type=str, default="rtx5080")
    parser.add_argument("--arrival-rate", type=float, required=True, help="Jobs per hour")
    parser.add_argument("--avg-job-duration", type=str, required=True, help="e.g., 2h")
    args = parser.parse_args()

    duration_hours = parse_duration(args.avg_job_duration)
    service_rate = 1.0 / duration_hours

    model = MMcCapacityModel(args.gpus, args.arrival_rate, service_rate)
    print(f"Utilization: {model.utilization():.1%}")
    print(f"Avg wait time: {model.avg_wait_time() * 60:.1f} minutes")
    print(f"Avg queue length: {model.avg_queue_length():.1f} jobs")
    print(f"P(wait): {model.erlang_c():.1%}")
```

### Performance Regression Gate

```go
// benchmarks/ci/regression_gate.go
package ci

import (
    "fmt"
    "math"
)

type RegressionCheck struct {
    MaxRegressionPercent float64 // e.g., 10.0 for 10%
    MinSampleSize        int
    ConfidenceLevel      float64 // e.g., 0.95
}

func (rc *RegressionCheck) Compare(baseline, current *BenchmarkResult) *RegressionReport {
    p99Diff := float64(current.P99-baseline.P99) / float64(baseline.P99) * 100

    report := &RegressionReport{
        Metric:     fmt.Sprintf("%s P99", baseline.Name),
        Baseline:   baseline.P99,
        Current:    current.P99,
        DiffPercent: p99Diff,
        Regressed:  p99Diff > rc.MaxRegressionPercent,
    }

    if report.Regressed {
        report.Message = fmt.Sprintf(
            "REGRESSION: %s P99 increased by %.1f%% (threshold: %.1f%%). Baseline: %v, Current: %v",
            baseline.Name, p99Diff, rc.MaxRegressionPercent, baseline.P99, current.P99,
        )
    }
    return report
}
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Benchmark results inconsistent | Increase warmup iterations; ensure no other workloads running; pin CPU governor to performance |
| Scheduler bench can't create 1000 jobs | Use a dedicated `bench` namespace with relaxed quotas; or use fake/dry-run mode |
| Capacity model predicts infinity | Utilization ρ ≥ 1.0 (system overloaded). Reduce arrival rate or increase GPUs |
| Parallel checkpoint slower | Check if storage backend supports concurrent writes; try fewer workers with larger chunks |
| CI gate too sensitive | Increase `MaxRegressionPercent` or require multiple consecutive failures before blocking |
| Connection pool exhaustion | Increase pool size; add metrics for pool utilization; check for connection leaks |
| What-if tool crashes on large GPU counts | Check factorial overflow in Erlang-C; use log-space computation for large c |

---

## Agent Handoff Template

```
Resume Anvil Phase C, Week 18: Infrastructure Performance Engineering.

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB, 32GB RAM, Ubuntu.
State: Phases A+B complete, Weeks 15-17 done. Full observability stack, CLI, GPU health controller all operational.

Current goal: Build benchmark suite, optimize infrastructure latency, implement capacity model with what-if analysis, and create CI performance regression gates.
Key files: ~/anvil/benchmarks/ (suite, CI gate), ~/anvil/capacity/ (queue model, what-if)
Test with: `make bench-all` for benchmarks, `python model.py --gpus 8 --arrival-rate 5 --avg-job-duration 2h` for capacity.

Specific task: [DESCRIBE WHAT TO DO NEXT]
Constraints: Benchmarks must be reproducible. Capacity model must validate against simulation within 15% error. CI gate must not produce false positives on normal variance.
```

---

## Out of Scope

- Application-level profiling (focus is infrastructure, not model code)
- GPU kernel optimization (CUDA-level, not infra-level)
- Network bandwidth optimization between nodes (assume adequate network)
- Storage tiering optimization (NVMe vs HDD placement)
- Auto-tuning (manual knobs documented, not auto-adjusted)
- Load testing the CLI itself (focus is on backend infrastructure)
- Comparison with other platforms (benchmarks are self-referential)
