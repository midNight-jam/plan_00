# Week 14: Performance Engineering Methodology
> Phase: 2 | Project: Forge | Estimated Duration: 7 days

## Context

Phase 2's final week. You've built an inference engine with continuous batching, memory management, speculative decoding, quantization, and observability. Now you step back and apply rigorous performance engineering methodology. The goal: systematically find and fix real performance bottlenecks using profiling data, not guesswork.

This is about building the SKILL of performance analysis — a methodology you'll use for the rest of your career. The deliverable is both measurable platform improvements AND a documented approach others can follow.

**Prerequisites**: Weeks 8-13 complete — full instrumented platform with load testing capability.

**Builds on**: Uses observability from Week 13 to identify bottlenecks. Applies knowledge from all Phase 2 weeks to fix them.

## Learning Goals

- [ ] Understand the performance engineering loop: define metric → baseline → profile → hypothesize → fix → measure
- [ ] Understand profiling tools: PyTorch profiler (GPU ops), cProfile (Python CPU), nvidia-smi (GPU utilization), memory_profiler (Python memory)
- [ ] Understand Amdahl's Law — why fixing a 5% bottleneck gives at most 5% improvement
- [ ] Understand the difference between latency optimization and throughput optimization (often opposing goals)
- [ ] Understand memory-bound vs compute-bound operations and how to distinguish them
- [ ] Understand Python-specific overhead: GIL contention, object allocation, async scheduling overhead
- [ ] Understand how to write a performance investigation report that others can reproduce

## Implementation Goals

- [ ] Build systematic profiling toolkit (automated scripts for each profiler type)
- [ ] Document the performance investigation methodology as a runbook
- [ ] Establish baselines: measure current platform performance across all key metrics
- [ ] Profile the full inference pipeline: identify top-5 time consumers
- [ ] Find and fix Performance Issue #1 (with before/after measurements)
- [ ] Find and fix Performance Issue #2 (with before/after measurements)
- [ ] Find and fix Performance Issue #3 (with before/after measurements)
- [ ] End-to-end optimization pass: reduce Python overhead, optimize tokenization, memory pre-allocation
- [ ] Verify cumulative improvement: run same load test as baseline, compare
- [ ] Write Phase 2 blog post: "Deep Diving into LLM Inference"

## Acceptance Criteria

1. **3 performance issues found and fixed**: Each with clear before/after numbers (latency or throughput), root cause documented
2. **Methodology documented**: Reproducible runbook that another engineer could follow to investigate performance
3. **Profiling toolkit working**: Single command produces comprehensive profile (GPU trace + CPU profile + memory snapshot)
4. **Baseline established**: All key metrics measured and recorded (throughput, TTFT, TPOT, p99 latency, GPU utilization)
5. **Cumulative improvement measurable**: End-of-week performance at least 20% better than start-of-week on primary metric (throughput or p99 latency)
6. **No regressions**: Fixes don't degrade other metrics (e.g., throughput fix doesn't break latency)
7. **Blog post written**: 2000-3000 word blog post covering Phase 2 journey, key insights, architecture decisions
8. **Profile reports archived**: All profiling data saved and reproducible (scripts + data + analysis)
9. **Optimization pass documented**: List of micro-optimizations applied, with individual impact estimates
10. **Phase 2 retrospective**: Summary of what worked, what was harder than expected, what you'd do differently

## Validation Commands

```bash
# Run profiling toolkit (generates comprehensive profile)
python -m forge.perf.profile_all --duration 60 --qps 20 --output results/perf/baseline_profile/

# View GPU trace
python -m forge.perf.view_trace --input results/perf/baseline_profile/gpu_trace.json
# Opens in chrome://tracing

# CPU profiling
python -m forge.perf.cpu_profile --duration 30 --output results/perf/cpu_profile.prof
python -m forge.perf.analyze_cpu --input results/perf/cpu_profile.prof --output results/perf/cpu_flamegraph.svg

# Memory profiling
python -m forge.perf.memory_profile --duration 60 --output results/perf/memory_timeline.json

# Establish baseline
python -m forge.perf.baseline --qps 20 --duration 120 --output results/perf/baseline.json

# Run after each fix to measure improvement
python -m forge.perf.baseline --qps 20 --duration 120 --output results/perf/after_fix_1.json
python -m forge.perf.baseline --qps 20 --duration 120 --output results/perf/after_fix_2.json
python -m forge.perf.baseline --qps 20 --duration 120 --output results/perf/after_fix_3.json

# Compare before/after
python -m forge.perf.compare --baseline results/perf/baseline.json --after results/perf/after_fix_3.json --output results/perf/improvement_report.html

# Final validation: full load test
python -m forge.loadtest.run --profile saturation --output results/perf/final_saturation.json

# Generate Phase 2 summary
python -m forge.perf.phase2_summary --results-dir results/perf/ --output results/perf/phase2_summary.html

# Unit tests for any new optimizations
pytest tests/unit/test_optimizations.py -v
```

## Technical Implementation Details

### Component 1: Profiling Toolkit (Day 1-2)

**File: `src/forge/perf/profiler_toolkit.py`**

```python
import torch
import cProfile
import pstats
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
import tracemalloc
import subprocess
import json

class ProfilerToolkit:
    """Unified profiling toolkit — one command, comprehensive results."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def profile_gpu(self, inference_fn, num_warmup=5, num_profile=20):
        """PyTorch profiler: GPU kernel timing, memory events, CUDA streams."""
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=2, warmup=num_warmup, active=num_profile),
            on_trace_ready=tensorboard_trace_handler(str(self.output_dir / "gpu_trace")),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
            for _ in range(2 + num_warmup + num_profile):
                inference_fn()
                prof.step()
        
        # Export summary table
        summary = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
        (self.output_dir / "gpu_summary.txt").write_text(summary)
        
        # Export memory timeline
        prof.export_memory_timeline(str(self.output_dir / "memory_timeline.html"))
        
        return prof
    
    def profile_cpu(self, inference_fn, num_calls=100):
        """cProfile: Python function-level CPU timing."""
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(num_calls):
            inference_fn()
        profiler.disable()
        
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        stats.dump_stats(str(self.output_dir / "cpu_profile.prof"))
        
        # Generate text report
        with open(self.output_dir / "cpu_profile.txt", "w") as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats("cumulative")
            stats.print_stats(50)
        
        return stats
    
    def profile_memory(self, inference_fn, num_calls=50):
        """Track Python memory allocations over time."""
        tracemalloc.start()
        snapshots = []
        
        for i in range(num_calls):
            inference_fn()
            if i % 10 == 0:
                snapshot = tracemalloc.take_snapshot()
                snapshots.append({
                    "iteration": i,
                    "top_allocations": [
                        {"file": str(stat.traceback), "size_kb": stat.size / 1024}
                        for stat in snapshot.statistics("lineno")[:20]
                    ]
                })
        
        tracemalloc.stop()
        with open(self.output_dir / "memory_allocations.json", "w") as f:
            json.dump(snapshots, f, indent=2)
    
    def monitor_nvidia_smi(self, duration_seconds: int, interval_ms: int = 100):
        """Continuous nvidia-smi monitoring during profiling."""
        cmd = [
            "nvidia-smi", "--query-gpu=timestamp,utilization.gpu,utilization.memory,"
            "memory.used,memory.free,temperature.gpu,power.draw",
            "--format=csv,nounits,noheader",
            f"--loop-ms={interval_ms}"
        ]
        # Run for duration, capture output
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        # ... capture and parse into timeseries
        pass
    
    def run_all(self, inference_fn, duration=60):
        """Run all profilers and generate unified report."""
        self.profile_gpu(inference_fn)
        self.profile_cpu(inference_fn)
        self.profile_memory(inference_fn)
        self.monitor_nvidia_smi(duration)
        self.generate_unified_report()
```

### Component 2: Performance Investigation Methodology (Day 2-3)

**File: `docs/runbooks/performance-investigation.md`**

Document the following methodology as a reproducible runbook:

```
PERFORMANCE INVESTIGATION METHODOLOGY
======================================

Step 1: DEFINE THE METRIC
- What exactly are you optimizing? (throughput? p99 latency? TTFT? memory?)
- What is the current value? What is the target?
- How will you measure it reliably? (load test config, warmup, repetitions)

Step 2: ESTABLISH BASELINE
- Run standardized benchmark (fixed QPS, duration, prompt distribution)
- Record: throughput, p50/p95/p99 latency, TTFT, GPU util, memory
- Run 3x to ensure reproducibility (variance < 5%)

Step 3: PROFILE
- GPU profile: which CUDA kernels take the most time?
- CPU profile: which Python functions are hotspots?
- Memory profile: where are allocations happening?
- I/O profile: any blocking waits? Network? Disk?

Step 4: HYPOTHESIZE
- Based on profiling data, what is the #1 bottleneck?
- What % of total time does it consume? (Amdahl's Law check)
- What is the theoretical improvement if this is fixed?

Step 5: FIX
- Implement the optimization
- Keep the change minimal and isolated (one thing at a time)
- Ensure correctness is maintained (run functional tests)

Step 6: MEASURE
- Re-run EXACT same benchmark as baseline
- Compare: did the target metric improve?
- Check: did any other metric regress?
- If improvement < expected, understand why

Step 7: DOCUMENT
- Root cause, fix applied, before/after numbers
- Whether the fix should be kept or reverted
- What to investigate next
```

### Component 3: Common Performance Issues to Find (Day 3-5)

Likely issues in the platform (investigate systematically, don't assume):

**Issue Category 1: Python Overhead**
```python
# Symptoms: GPU utilization < 80% even under load, CPU hotspot in scheduling logic
# Profile pattern: large gaps between CUDA kernels in GPU trace

# Common fixes:
# - Move scheduling logic to run WHILE GPU is computing (overlap CPU and GPU)
# - Pre-allocate tensors instead of creating new ones each iteration
# - Use torch.cuda.Stream for async operations
# - Batch Python operations (process N requests in one Python call, not N calls)

class OptimizedScheduler:
    def __init__(self):
        self.compute_stream = torch.cuda.Stream()
        self.schedule_stream = torch.cuda.Stream()
    
    def step(self):
        # Schedule next batch WHILE current batch is computing
        with torch.cuda.stream(self.schedule_stream):
            next_batch = self.prepare_next_batch()
        
        with torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_stream(self.schedule_stream)
            result = self.model(next_batch)
        
        return result
```

**Issue Category 2: Tokenization Bottleneck**
```python
# Symptoms: TTFT higher than expected, CPU spike during prefill setup
# Profile pattern: tokenizer.__call__ appears in CPU top-10

# Common fixes:
# - Pre-tokenize on a separate thread/process
# - Use fast tokenizers (Rust-based via HuggingFace tokenizers)
# - Batch tokenization (tokenize multiple prompts at once)
# - Cache tokenized system prompts

class AsyncTokenizer:
    def __init__(self, tokenizer, num_workers=2):
        self.tokenizer = tokenizer
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        self.cache = LRUCache(maxsize=1000)
    
    async def tokenize(self, text: str) -> list[int]:
        cache_key = hash(text)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        loop = asyncio.get_event_loop()
        tokens = await loop.run_in_executor(self.pool, self.tokenizer.encode, text)
        self.cache[cache_key] = tokens
        return tokens
```

**Issue Category 3: Memory Allocation Overhead**
```python
# Symptoms: periodic latency spikes, GPU memory fluctuates
# Profile pattern: cudaMalloc/cudaFree in GPU trace, Python GC pauses

# Common fixes:
# - Pre-allocate output tensors and reuse (tensor pool)
# - Use CUDA memory pool (torch.cuda.memory.CUDAPluggableAllocator)
# - Disable Python GC during critical path (gc.disable() / gc.enable() around generation)
# - Pre-allocate KV-cache blocks at startup, not on-demand

class TensorPool:
    """Pre-allocated tensor pool to avoid cudaMalloc during inference."""
    
    def __init__(self, shapes: dict[str, tuple], dtype=torch.float16, device="cuda"):
        self.pools = {}
        for name, shape in shapes.items():
            self.pools[name] = {
                "tensor": torch.empty(shape, dtype=dtype, device=device),
                "in_use": False
            }
    
    def get(self, name: str) -> torch.Tensor:
        pool = self.pools[name]
        assert not pool["in_use"], f"Tensor {name} already in use"
        pool["in_use"] = True
        return pool["tensor"]
    
    def release(self, name: str):
        self.pools[name]["in_use"] = False
        self.pools[name]["tensor"].zero_()  # Clear for safety
```

### Component 4: Measurement and Comparison (Day 5-6)

**File: `src/forge/perf/compare.py`**

```python
@dataclass
class BenchmarkResult:
    name: str
    throughput_tps: float
    ttft_p50_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p99_ms: float
    gpu_utilization_avg: float
    memory_peak_gb: float
    timestamp: str

class PerformanceComparator:
    """Compare before/after benchmark results with statistical rigor."""
    
    def compare(self, baseline: BenchmarkResult, after: BenchmarkResult) -> dict:
        improvements = {}
        for metric in ["throughput_tps", "ttft_p99_ms", "tpot_p99_ms", "gpu_utilization_avg"]:
            base_val = getattr(baseline, metric)
            after_val = getattr(after, metric)
            
            if metric in ["throughput_tps", "gpu_utilization_avg"]:
                # Higher is better
                change_pct = ((after_val - base_val) / base_val) * 100
            else:
                # Lower is better
                change_pct = ((base_val - after_val) / base_val) * 100
            
            improvements[metric] = {
                "baseline": base_val,
                "after": after_val,
                "improvement_pct": change_pct,
                "direction": "better" if change_pct > 0 else "worse"
            }
        
        return improvements
    
    def generate_report(self, baseline, fixes: list[dict], final) -> str:
        """Generate HTML report showing cumulative improvement across fixes."""
        # Waterfall chart: baseline → fix1 → fix2 → fix3 → final
        # Table: each fix with root cause, change, impact
        pass
```

### Component 5: Blog Post Outline (Day 7)

**File: `blog/phase2-deep-dive.md`**

Structure for the blog post "Deep Diving into LLM Inference":

1. **Introduction** — Why understanding inference internals matters (not just calling APIs)
2. **The Transformer Forward Pass** — Key insight about autoregressive generation and KV-cache
3. **The Scheduling Problem** — Why continuous batching is essential (static batching waste)
4. **Memory is the Bottleneck** — Block allocation, eviction, the OS analogy
5. **Speculative Decoding** — Getting 2x speedup by guessing ahead
6. **Quantization** — Shrinking models without (much) quality loss, Pareto tradeoffs
7. **Observability** — You can't optimize what you can't measure
8. **Performance Engineering** — The methodology, specific issues found, before/after
9. **Key Takeaways** — Top 5 things learned that apply broadly to systems engineering
10. **What's Next** — Phase 3 preview (multi-GPU, custom kernels, distributed serving)

Target: 2000-3000 words, technical audience, includes diagrams and charts from the work.

## If You Get Stuck

**Can't find significant performance issues**: Run the profiler under HIGHER load (closer to saturation). Bottlenecks become obvious under pressure. If the system handles 20 QPS fine, profile at 40-50 QPS where it starts to struggle.

**GPU utilization is already 95%+**: You're compute-bound. Optimizations shift to: better batching (more tokens per forward pass), quantization (less compute per token), or reducing non-compute time (scheduling, tokenization, data transfer). Also check if the 95% is real compute or includes memory-bound stalls.

**Fix helps in isolation but not in integration**: Interactions between optimizations are real. A memory optimization might reduce cache hit rate. Profile the integrated system, not just components.

**Blog post feels superficial**: Include specific numbers. "We reduced p99 latency from 850ms to 420ms by overlapping CPU scheduling with GPU compute" is 10x better than "we made it faster." Use the profiling data you collected.

**Not sure what counts as a "real" issue**: If it shows up in the profile AND fixing it gives measurable improvement (>5% on some metric), it's real. Don't optimize things that don't show up in the profile.

## Agent Handoff Template

```
I'm on Week 14 of Forge — performance engineering methodology (Phase 2 final week).
Spec: /Users/jmalviya/Documents/zz/dev/plan_00/forge/specs/phase2/week14-performance-engineering.md
Context: Full platform built (Phase 1 + Weeks 8-13). Need to systematically profile, find bottlenecks, fix them, and document methodology.
I need: profiling toolkit, baseline measurements, 3 performance fixes with before/after, methodology runbook, Phase 2 blog post.
Current state: [describe baseline numbers and what's been profiled so far]
Key challenge: [profiling setup / identifying bottlenecks / implementing fixes / measuring improvement / writing]
```

## Out of Scope

- Custom CUDA kernels (Phase 3 — Week 16)
- Multi-GPU optimization (Phase 3)
- Compiler-level optimizations (Triton/CUTLASS)
- Hardware-specific tuning (specific GPU architecture exploitation)
- Production deployment optimization (this is dev/research environment)
- Algorithmic improvements to model architecture
- Rewriting components in C++/Rust (optimize Python first, rewrite only if justified)
