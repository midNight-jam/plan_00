# Week 13: Observability and Load Testing
> Phase: 2 | Project: Forge | Estimated Duration: 7 days

## Context

You've built a high-performance inference engine (Weeks 8-12). Now you need to SEE what it's doing in production. Observability is the difference between "it's slow sometimes" and "p99 latency spiked because the KV-cache eviction rate tripled after tenant X started sending 4K-token prompts." This week builds the full observability stack and validates the platform under realistic load.

This integrates with the Phase 1 platform (API gateway, multi-model serving) to provide end-to-end visibility from request arrival through generation to response.

**Prerequisites**: Weeks 8-12 complete (inference engine). Phase 1 platform running (API gateway, model serving).

**Builds on**: Instruments the full stack built across Phase 1 and Phase 2. Integrates all components into a monitored system.

## Learning Goals

- [ ] Understand the three pillars of observability: metrics, traces, logs — and when each is appropriate
- [ ] Understand Prometheus metric types: counter, gauge, histogram, summary — and when to use each
- [ ] Understand OpenTelemetry: spans, traces, attributes, propagation — distributed tracing through async systems
- [ ] Understand percentile latencies (p50, p95, p99) — why averages lie
- [ ] Understand load testing methodology: open vs closed loop, think time, arrival patterns
- [ ] Understand saturation testing: finding the breaking point (throughput plateau, latency explosion)
- [ ] Understand soak testing: detecting slow memory leaks and resource exhaustion over hours

## Implementation Goals

- [ ] Instrument the platform with Prometheus metrics (request latency, GPU utilization, model metrics, business metrics)
- [ ] Implement OpenTelemetry distributed tracing (full request lifecycle with span attributes)
- [ ] Build 4 Grafana dashboards: Operations, GPU, Models, Platform
- [ ] Implement structured logging with correlation IDs (trace_id in every log line)
- [ ] Build load testing suite with realistic traffic patterns
- [ ] Run saturation test: find the breaking point (max QPS before latency explosion)
- [ ] Run soak test: 4+ hours sustained load, detect memory leaks
- [ ] Implement alerting rules with runbooks
- [ ] Build automated health check that detects common failure modes

## Acceptance Criteria

1. **4 Grafana dashboards working**: Operations (request rates, latencies, errors), GPU (VRAM, utilization, temp), Models (tokens/sec, cache hits, queue depth), Platform (per-tenant, per-model breakdown)
2. **Traces visible in Jaeger**: Full request trace from API gateway → scheduler → inference → response, with span attributes (model, tokens, latency)
3. **Metrics are accurate**: Prometheus histograms match actual measured latencies (verified by comparing with manual timing in load test)
4. **Load test identifies breaking point**: Clear chart showing where throughput plateaus and latency explodes (saturation point)
5. **Soak test runs 4+ hours**: No memory leaks detected (RSS stable within 5%), no goroutine/connection leaks
6. **Alerting fires on p99 breach**: Alert triggers within 60s when p99 latency exceeds SLO threshold
7. **Correlation IDs propagate**: Every log line for a request carries the same trace_id, searchable across services
8. **GPU metrics collected**: nvidia-smi metrics exported to Prometheus (VRAM, utilization, temperature, power)
9. **Per-tenant metrics**: Can identify which tenant is consuming most resources
10. **Load test report generated**: Automated report with charts, breaking point, recommendations

## Validation Commands

```bash
# Start observability stack (Prometheus + Grafana + Jaeger)
docker compose -f docker/observability.yaml up -d

# Verify metrics endpoint
curl http://localhost:8000/metrics | head -50

# Verify traces are being collected
curl http://localhost:16686/api/traces?service=forge-inference&limit=5

# Run smoke load test (quick validation)
python -m forge.loadtest.run --profile smoke --duration 60 --output results/loadtest_smoke.json

# Run saturation test (find breaking point)
python -m forge.loadtest.run --profile saturation --start-qps 1 --end-qps 100 --step-duration 30 --output results/saturation.json

# Run soak test (memory leak detection)
python -m forge.loadtest.run --profile soak --qps 20 --duration 14400 --output results/soak.json

# Run bursty traffic test
python -m forge.loadtest.run --profile bursty --base-qps 10 --burst-qps 50 --burst-duration 10 --output results/bursty.json

# Check alerting rules syntax
promtool check rules config/alerting_rules.yaml

# Generate load test report
python -m forge.loadtest.report --input results/saturation.json --output results/loadtest_report.html

# Verify Grafana dashboards load
curl -s http://localhost:3000/api/dashboards/uid/forge-operations | jq '.dashboard.title'
curl -s http://localhost:3000/api/dashboards/uid/forge-gpu | jq '.dashboard.title'
curl -s http://localhost:3000/api/dashboards/uid/forge-models | jq '.dashboard.title'
curl -s http://localhost:3000/api/dashboards/uid/forge-platform | jq '.dashboard.title'

# Test alert firing
python -m forge.loadtest.trigger_alert --type p99_breach --duration 120
```

## Technical Implementation Details

### Component 1: Prometheus Metrics Instrumentation (Day 1-2)

**File: `src/forge/observability/metrics.py`**

```python
from prometheus_client import Histogram, Counter, Gauge, Info
import time

# Request metrics
REQUEST_LATENCY = Histogram(
    "forge_request_duration_seconds",
    "End-to-end request latency",
    ["model", "endpoint", "status"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

TTFT_LATENCY = Histogram(
    "forge_ttft_seconds",
    "Time to first token",
    ["model"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

TPOT_LATENCY = Histogram(
    "forge_tpot_seconds",
    "Time per output token (inter-token latency)",
    ["model"],
    buckets=[0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]
)

TOKENS_GENERATED = Counter(
    "forge_tokens_generated_total",
    "Total tokens generated",
    ["model", "tenant"]
)

# GPU metrics
GPU_VRAM_USED = Gauge(
    "forge_gpu_vram_used_bytes",
    "GPU VRAM used",
    ["gpu_id", "category"]  # category: weights, kv_cache, activations, free
)

GPU_UTILIZATION = Gauge(
    "forge_gpu_utilization_percent",
    "GPU compute utilization",
    ["gpu_id"]
)

GPU_TEMPERATURE = Gauge(
    "forge_gpu_temperature_celsius",
    "GPU temperature",
    ["gpu_id"]
)

# Model metrics
KV_CACHE_HIT_RATE = Gauge(
    "forge_kv_cache_hit_rate",
    "Prefix cache hit rate",
    ["model"]
)

SCHEDULER_QUEUE_DEPTH = Gauge(
    "forge_scheduler_queue_depth",
    "Number of requests waiting in scheduler queue",
    ["model", "state"]  # state: waiting, running, swapped
)

BATCH_SIZE = Histogram(
    "forge_batch_size",
    "Current batch size per iteration",
    ["model"],
    buckets=[1, 2, 4, 8, 16, 32, 64, 128]
)

# Business metrics
TENANT_TOKENS = Counter(
    "forge_tenant_tokens_total",
    "Tokens consumed per tenant",
    ["tenant_id", "model", "direction"]  # direction: input, output
)

class MetricsCollector:
    """Collects and exports GPU metrics from nvidia-smi."""
    
    def __init__(self, collection_interval: float = 5.0):
        self.interval = collection_interval
    
    def collect_gpu_metrics(self):
        """Scrape nvidia-smi and update Prometheus gauges."""
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            GPU_VRAM_USED.labels(gpu_id=str(i), category="used").set(mem_info.used)
            GPU_VRAM_USED.labels(gpu_id=str(i), category="free").set(mem_info.free)
            GPU_UTILIZATION.labels(gpu_id=str(i)).set(util.gpu)
            GPU_TEMPERATURE.labels(gpu_id=str(i)).set(temp)
```

### Component 2: OpenTelemetry Distributed Tracing (Day 2-3)

**File: `src/forge/observability/tracing.py`**

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.trace import StatusCode

tracer = trace.get_tracer("forge-inference")

class TracedInferenceEngine:
    """Wraps inference engine with OpenTelemetry tracing."""
    
    def __init__(self, engine):
        self.engine = engine
    
    async def generate(self, request):
        with tracer.start_as_current_span("inference.generate") as span:
            span.set_attribute("model", request.model)
            span.set_attribute("input_tokens", len(request.prompt_tokens))
            span.set_attribute("max_output_tokens", request.max_tokens)
            span.set_attribute("tenant_id", request.tenant_id)
            
            # Phase 1: Queue wait
            with tracer.start_as_current_span("inference.queue_wait") as queue_span:
                queue_start = time.time()
                await self.engine.scheduler.wait_for_slot(request)
                queue_span.set_attribute("wait_ms", (time.time() - queue_start) * 1000)
            
            # Phase 2: Prefill
            with tracer.start_as_current_span("inference.prefill") as prefill_span:
                prefill_start = time.time()
                prefix_hit = self.engine.prefix_cache.match(request.prompt_tokens)
                prefill_span.set_attribute("prefix_cache_hit", prefix_hit > 0)
                prefill_span.set_attribute("tokens_cached", prefix_hit)
                await self.engine.prefill(request)
                prefill_span.set_attribute("duration_ms", (time.time() - prefill_start) * 1000)
            
            # Phase 3: Decode (streaming)
            with tracer.start_as_current_span("inference.decode") as decode_span:
                tokens_generated = 0
                async for token in self.engine.decode_stream(request):
                    tokens_generated += 1
                    yield token
                decode_span.set_attribute("tokens_generated", tokens_generated)
                decode_span.set_attribute("tpot_ms", decode_span.duration_ms / tokens_generated)
            
            span.set_attribute("total_output_tokens", tokens_generated)
            span.set_status(StatusCode.OK)
```

### Component 3: Grafana Dashboards (Day 3-4)

**File: `config/grafana/dashboards/`**

Four dashboards as JSON provisioning files:

**1. Operations Dashboard (`forge-operations.json`)**
- Request rate (QPS) over time, split by model and status code
- Latency percentiles (p50, p95, p99) over time
- Error rate percentage
- TTFT distribution histogram
- Active connections gauge

**2. GPU Dashboard (`forge-gpu.json`)**
- VRAM usage stacked area (weights, KV-cache, activations, free)
- GPU utilization % over time
- Temperature over time with threshold line
- Memory allocation rate (blocks/sec)
- Eviction rate over time

**3. Models Dashboard (`forge-models.json`)**
- Tokens/sec per model over time
- Prefix cache hit rate per model
- Batch size distribution
- Queue depth per model
- Speculative decoding acceptance rate

**4. Platform Dashboard (`forge-platform.json`)**
- Per-tenant token consumption (stacked bar)
- Per-tenant latency percentiles
- Model popularity (requests per model)
- Cost estimation (tokens × price)
- Capacity planning: headroom % remaining

### Component 4: Load Testing Suite (Day 4-6)

**File: `src/forge/loadtest/runner.py`**

```python
import asyncio
import aiohttp
import numpy as np
from dataclasses import dataclass

@dataclass
class LoadProfile:
    name: str
    duration_seconds: int
    qps_schedule: callable  # time -> target QPS
    prompt_length_dist: callable  # () -> int
    output_length_dist: callable  # () -> int

PROFILES = {
    "smoke": LoadProfile(
        name="smoke",
        duration_seconds=60,
        qps_schedule=lambda t: 5,
        prompt_length_dist=lambda: np.random.randint(50, 200),
        output_length_dist=lambda: np.random.randint(20, 100),
    ),
    "saturation": LoadProfile(
        name="saturation",
        duration_seconds=600,
        qps_schedule=lambda t: int(1 + t / 6),  # Ramp from 1 to 100 QPS over 10 min
        prompt_length_dist=lambda: np.random.choice([50, 200, 500, 1000], p=[0.3, 0.4, 0.2, 0.1]),
        output_length_dist=lambda: np.random.randint(20, 200),
    ),
    "bursty": LoadProfile(
        name="bursty",
        duration_seconds=300,
        qps_schedule=lambda t: 50 if (t % 60) < 10 else 10,  # Burst every minute
        prompt_length_dist=lambda: np.random.randint(100, 500),
        output_length_dist=lambda: np.random.randint(50, 150),
    ),
    "soak": LoadProfile(
        name="soak",
        duration_seconds=14400,  # 4 hours
        qps_schedule=lambda t: 20,  # Steady 20 QPS
        prompt_length_dist=lambda: np.random.randint(100, 500),
        output_length_dist=lambda: np.random.randint(50, 200),
    ),
}

class LoadTestRunner:
    """Open-loop load generator with accurate timing."""
    
    def __init__(self, base_url: str, profile: LoadProfile):
        self.base_url = base_url
        self.profile = profile
        self.results = []
    
    async def run(self):
        """Generate load according to profile's QPS schedule."""
        start = time.time()
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start < self.profile.duration_seconds:
                elapsed = time.time() - start
                target_qps = self.profile.qps_schedule(elapsed)
                interval = 1.0 / target_qps if target_qps > 0 else 1.0
                
                # Fire request (don't await — open loop)
                asyncio.create_task(self._send_request(session, elapsed))
                
                # Wait for next interval (with jitter for realism)
                jitter = np.random.exponential(interval * 0.1)
                await asyncio.sleep(interval + jitter)
        
        return self._compile_results()
    
    async def _send_request(self, session, send_time):
        prompt_len = self.profile.prompt_length_dist()
        max_tokens = self.profile.output_length_dist()
        
        request_start = time.time()
        first_token_time = None
        tokens_received = 0
        
        async with session.post(f"{self.base_url}/v1/completions", json={
            "prompt": "x " * prompt_len,  # Dummy prompt of desired length
            "max_tokens": max_tokens,
            "stream": True
        }) as resp:
            async for chunk in resp.content:
                if first_token_time is None:
                    first_token_time = time.time()
                tokens_received += 1
        
        self.results.append({
            "send_time": send_time,
            "ttft_ms": (first_token_time - request_start) * 1000 if first_token_time else None,
            "total_ms": (time.time() - request_start) * 1000,
            "tokens": tokens_received,
            "prompt_len": prompt_len,
            "status": resp.status
        })
```

### Component 5: Alerting Rules (Day 6-7)

**File: `config/alerting_rules.yaml`**

```yaml
groups:
  - name: forge_latency
    rules:
      - alert: HighP99Latency
        expr: histogram_quantile(0.99, rate(forge_request_duration_seconds_bucket[5m])) > 2.0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "P99 latency exceeds 2s"
          runbook: "docs/runbooks/high-latency.md"
      
      - alert: HighP99LatencyCritical
        expr: histogram_quantile(0.99, rate(forge_request_duration_seconds_bucket[5m])) > 5.0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "P99 latency exceeds 5s — likely saturation"
          runbook: "docs/runbooks/high-latency.md"
  
  - name: forge_gpu
    rules:
      - alert: GPUMemoryHigh
        expr: forge_gpu_vram_used_bytes{category="used"} / (forge_gpu_vram_used_bytes{category="used"} + forge_gpu_vram_used_bytes{category="free"}) > 0.95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU VRAM usage above 95%"
          runbook: "docs/runbooks/gpu-memory.md"
      
      - alert: GPUTemperatureHigh
        expr: forge_gpu_temperature_celsius > 85
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "GPU temperature above 85°C"
  
  - name: forge_errors
    rules:
      - alert: HighErrorRate
        expr: rate(forge_request_duration_seconds_count{status="error"}[5m]) / rate(forge_request_duration_seconds_count[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Error rate exceeds 5%"
          runbook: "docs/runbooks/high-errors.md"
```

### Component 6: Soak Test Analysis (Day 7)

**File: `src/forge/loadtest/soak_analysis.py`**

```python
class SoakTestAnalyzer:
    """Detect memory leaks and resource exhaustion from soak test data."""
    
    def analyze(self, metrics_timeseries: dict) -> dict:
        results = {
            "memory_leak_detected": False,
            "connection_leak_detected": False,
            "latency_degradation_detected": False,
        }
        
        # Check RSS memory: fit linear regression, alert if slope > threshold
        rss_values = metrics_timeseries["process_rss_bytes"]
        slope = self._linear_slope(rss_values)
        if slope > 1024 * 1024:  # Growing > 1MB/hour
            results["memory_leak_detected"] = True
            results["memory_growth_mb_per_hour"] = slope / (1024 * 1024)
        
        # Check latency degradation: compare first hour vs last hour
        latencies = metrics_timeseries["request_latency_p99"]
        first_hour = latencies[:len(latencies)//4]
        last_hour = latencies[-len(latencies)//4:]
        
        if np.mean(last_hour) > np.mean(first_hour) * 1.2:
            results["latency_degradation_detected"] = True
            results["degradation_percent"] = (np.mean(last_hour) / np.mean(first_hour) - 1) * 100
        
        return results
```

## If You Get Stuck

**Prometheus metrics not appearing**: Check that the `/metrics` endpoint is registered on your HTTP server. Common issue: metrics registered but never incremented (use `REQUEST_LATENCY.labels(model="mistral").observe(duration)` — don't forget `.labels()` before `.observe()`).

**Traces not showing in Jaeger**: Verify the exporter is configured correctly (host/port). Check that `BatchSpanProcessor` is flushing (it batches by default — in tests, use `SimpleSpanProcessor` for immediate export). Verify trace context is propagating through async boundaries.

**Load test shows weird latency spikes**: Could be GC pauses (Python), CUDA synchronization points, or your load generator running out of connections. Increase aiohttp connection pool. Use `TCPConnector(limit=500)`.

**Grafana dashboard shows no data**: Check the Prometheus datasource is configured. Check the metric name exactly matches (copy from `/metrics` endpoint). Check the time range in Grafana matches when data was collected.

**Soak test OOMs**: Start with lower QPS. Monitor RSS in real-time with `watch -n1 'ps aux | grep forge'`. If it grows steadily, you have a leak — check for request objects not being freed, KV-cache blocks not returned, or asyncio tasks accumulating.

## Agent Handoff Template

```
I'm on Week 13 of Forge — implementing observability and load testing.
Spec: /Users/jmalviya/Documents/zz/dev/plan_00/forge/specs/phase2/week13-observability.md
Context: Phase 1 + Weeks 8-12 complete — full platform with inference engine, now need to instrument and validate under load.
I need: Prometheus metrics, OpenTelemetry tracing, 4 Grafana dashboards, load testing suite (saturation + soak), alerting rules.
Current state: [describe what's instrumented so far]
Key challenge: [metric instrumentation / trace propagation / load test design / dashboard creation / alert tuning]
```

## Out of Scope

- Log aggregation system (ELK/Loki) — just structured stdout logs with trace_id
- Chaos engineering / fault injection (Phase 3)
- Auto-scaling based on metrics (Phase 3)
- Cost monitoring / billing integration
- Multi-region observability
- Custom Prometheus exporters (use client libraries)
- Production deployment of observability stack (local Docker Compose only)
