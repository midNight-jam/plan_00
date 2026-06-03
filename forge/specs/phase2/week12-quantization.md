# Week 12: Quantization Pipeline
> Phase: 2 | Project: Forge | Estimated Duration: 7 days

## Context

Weeks 8-11 built the full inference engine. But we've been running in FP16 — every parameter uses 2 bytes. A 7B model is 14GB. Quantization compresses models to 4-bit (or less), cutting memory 4x and often improving throughput. But not all quantization is equal — some methods preserve quality better than others, and the tradeoffs depend on the model and use case.

This week you build an automated quantization pipeline: take any HuggingFace model, apply multiple quantization methods, measure quality degradation, benchmark speed, and produce a Pareto frontier showing the optimal quality-speed-memory tradeoff.

**Prerequisites**: Weeks 8-11 complete — full understanding of inference, attention, KV-cache, and serving.

**Builds on**: Full inference understanding lets you reason about WHY quantization works and where it breaks down.

## Learning Goals

- [ ] Understand quantization fundamentals — mapping continuous values to discrete levels (scales, zero-points)
- [ ] Understand GPTQ — Hessian-based post-training quantization (uses calibration data to minimize output error)
- [ ] Understand AWQ — activation-aware quantization (protects salient weights based on activation magnitudes)
- [ ] Understand NF4 — normalized float 4-bit (QLoRA's format, information-theoretically optimal for normal distributions)
- [ ] Understand GGUF — llama.cpp's format (various quant levels: Q4_0, Q4_K_M, Q5_K_M, Q8_0)
- [ ] Understand perplexity as a quality metric — and its limitations
- [ ] Understand TensorRT-LLM compilation — graph optimization, kernel fusion, quantized kernels
- [ ] Understand torch.compile modes — tracing, graph breaks, and compilation overhead

## Implementation Goals

- [ ] Build quantization comparison pipeline: HF model → 4 quantized variants + benchmarks
- [ ] Implement GPTQ quantization with calibration (using auto-gptq or gptq library)
- [ ] Implement AWQ quantization (using autoawq)
- [ ] Implement NF4/BnB quantization (using bitsandbytes)
- [ ] Convert model to GGUF format with multiple quant levels
- [ ] Build quality gate: measure perplexity on WikiText-2, reject if degradation > threshold
- [ ] Implement TensorRT-LLM compilation and benchmark
- [ ] Benchmark torch.compile with all modes (default, reduce-overhead, max-autotune)
- [ ] Build automated "Model Optimization Pipeline" CLI tool
- [ ] Generate Pareto frontier chart: quality vs throughput vs memory

## Acceptance Criteria

1. **4 quantization methods benchmarked**: GPTQ, AWQ, NF4, and GGUF all produce working quantized models from same base model
2. **Quality gates working**: Pipeline automatically rejects quantization if perplexity increases by more than configurable threshold (default: 5%)
3. **Perplexity measured correctly**: WikiText-2 perplexity calculated with proper sliding window, matches published baselines (±2%)
4. **TensorRT gives measurable speedup**: TRT-LLM compiled model achieves at least 1.5x throughput vs PyTorch eager
5. **torch.compile modes compared**: All three modes benchmarked with compilation time and steady-state throughput
6. **Pareto chart published**: Interactive chart showing quality (perplexity) vs speed (tokens/sec) vs memory (GB) for all variants
7. **Pipeline is automated**: Single command takes HF model ID, outputs optimized model + comprehensive report
8. **Memory reduction verified**: 4-bit models use ~4x less VRAM than FP16 baseline
9. **Latency benchmarked**: TTFT and TPOT measured for each variant under identical conditions
10. **Report generated**: Markdown report with recommendations (which method for which use case)

## Validation Commands

```bash
# Run full quantization pipeline
python -m forge.research.quant_pipeline --model "mistralai/Mistral-7B-v0.1" --methods gptq,awq,nf4,gguf --output results/quant_pipeline/

# Individual method tests
python -m forge.research.quantize --method gptq --model "mistralai/Mistral-7B-v0.1" --bits 4 --output models/mistral-7b-gptq/
python -m forge.research.quantize --method awq --model "mistralai/Mistral-7B-v0.1" --bits 4 --output models/mistral-7b-awq/
python -m forge.research.quantize --method nf4 --model "mistralai/Mistral-7B-v0.1" --output models/mistral-7b-nf4/
python -m forge.research.quantize --method gguf --model "mistralai/Mistral-7B-v0.1" --quant-type Q4_K_M --output models/mistral-7b-gguf/

# Quality gate: perplexity measurement
python -m forge.research.measure_perplexity --model models/mistral-7b-gptq/ --dataset wikitext2 --output results/perplexity_gptq.json
python -m forge.research.quality_gate --baseline-ppl 5.5 --threshold 0.05 --measured results/perplexity_gptq.json

# TensorRT-LLM compilation
python -m forge.research.compile_tensorrt --model "mistralai/Mistral-7B-v0.1" --output models/mistral-7b-trt/
python -m forge.research.bench_tensorrt --model models/mistral-7b-trt/ --output results/trt_bench.json

# torch.compile benchmark
python -m forge.research.bench_torch_compile --model "mistralai/Mistral-7B-v0.1" --modes default,reduce-overhead,max-autotune --output results/torch_compile_bench.json

# Throughput benchmark (all variants)
python -m forge.research.bench_throughput --models-dir models/ --num-requests 100 --output results/throughput_comparison.json

# Generate Pareto chart and report
python -m forge.research.pareto_analysis --results-dir results/ --output results/pareto_chart.html
python -m forge.research.generate_report --results-dir results/ --output results/quantization_report.md

# Unit tests
pytest tests/unit/test_quantization.py -v
```

## Technical Implementation Details

### Component 1: Quantization Pipeline Framework (Day 1)

**File: `src/forge/research/quant_pipeline.py`**

```python
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

class QuantMethod(Enum):
    GPTQ = "gptq"
    AWQ = "awq"
    NF4 = "nf4"
    GGUF = "gguf"

@dataclass
class QuantConfig:
    method: QuantMethod
    bits: int = 4
    group_size: int = 128
    calibration_samples: int = 128
    calibration_dataset: str = "c4"
    gguf_type: str = "Q4_K_M"  # For GGUF only

@dataclass
class QuantResult:
    method: QuantMethod
    model_path: Path
    size_gb: float
    perplexity: float
    throughput_tps: float
    ttft_ms: float
    tpot_ms: float
    memory_gb: float
    passed_quality_gate: bool

class QuantizationPipeline:
    """Automated model quantization and benchmarking pipeline."""
    
    def __init__(self, base_model: str, output_dir: Path):
        self.base_model = base_model
        self.output_dir = output_dir
        self.results: list[QuantResult] = []
    
    def run(self, methods: list[QuantMethod], quality_threshold: float = 0.05):
        # 1. Measure FP16 baseline perplexity
        baseline_ppl = self.measure_baseline_perplexity()
        
        # 2. For each method, quantize + measure
        for method in methods:
            config = QuantConfig(method=method)
            model_path = self.quantize(config)
            ppl = self.measure_perplexity(model_path)
            
            # Quality gate
            ppl_increase = (ppl - baseline_ppl) / baseline_ppl
            passed = ppl_increase <= quality_threshold
            
            # Benchmark
            throughput, ttft, tpot, memory = self.benchmark(model_path)
            
            self.results.append(QuantResult(
                method=method,
                model_path=model_path,
                size_gb=self.measure_size(model_path),
                perplexity=ppl,
                throughput_tps=throughput,
                ttft_ms=ttft,
                tpot_ms=tpot,
                memory_gb=memory,
                passed_quality_gate=passed
            ))
        
        # 3. Generate Pareto frontier
        self.plot_pareto()
        
        # 4. Generate report
        self.generate_report()
```

### Component 2: GPTQ Implementation (Day 2)

**File: `src/forge/research/quant_gptq.py`**

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

class GPTQQuantizer:
    """
    GPTQ: Post-Training Quantization using Hessian information.
    Key idea: quantize weights layer-by-layer, using second-order (Hessian) 
    information to minimize output error. Calibration data provides the Hessian.
    """
    
    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
    
    def quantize(self, model_id: str, output_path: Path, calibration_data: list[str]):
        quantize_config = BaseQuantizeConfig(
            bits=self.bits,
            group_size=self.group_size,
            desc_act=True,  # Activation order (slightly better quality)
        )
        
        model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)
        
        # Calibration: feed representative data through model to estimate Hessian
        # GPTQ uses this to decide which weights to round up vs down
        examples = self._prepare_calibration(calibration_data)
        model.quantize(examples)
        
        model.save_quantized(str(output_path))
        return output_path
    
    def _prepare_calibration(self, texts: list[str], max_length: int = 2048):
        """Tokenize calibration texts for GPTQ."""
        # Use C4 or similar diverse dataset
        # 128 samples of 2048 tokens each is standard
        pass
```

### Component 3: AWQ Implementation (Day 2-3)

**File: `src/forge/research/quant_awq.py`**

```python
from awq import AutoAWQForCausalLM

class AWQQuantizer:
    """
    AWQ: Activation-Aware Weight Quantization.
    Key idea: not all weights are equally important. Weights that correspond to 
    large activations matter more. Protect these "salient" channels by scaling 
    them up before quantization (then scale activations down to compensate).
    """
    
    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
    
    def quantize(self, model_id: str, output_path: Path):
        model = AutoAWQForCausalLM.from_pretrained(model_id)
        
        quant_config = {
            "zero_point": True,
            "q_group_size": self.group_size,
            "w_bit": self.bits,
            "version": "GEMM"  # GEMM or GEMV kernel
        }
        
        # AWQ searches for optimal per-channel scaling factors
        # that minimize quantization error for the most important weights
        model.quantize(calib_data="c4", quant_config=quant_config)
        model.save_quantized(str(output_path))
        return output_path
```

### Component 4: NF4 / BitsAndBytes (Day 3)

**File: `src/forge/research/quant_nf4.py`**

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

class NF4Quantizer:
    """
    NF4: Normalized Float 4-bit quantization (from QLoRA paper).
    Key idea: neural network weights are approximately normally distributed.
    NF4 chooses 16 quantization levels that are optimal for N(0,1) distribution
    (information-theoretically optimal — equal probability mass per bin).
    Also uses double quantization: quantize the quantization constants themselves.
    """
    
    def quantize(self, model_id: str, output_path: Path):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",         # NF4 data type
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,    # Quantize the quantization constants
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        model.save_pretrained(str(output_path))
        return output_path
```

### Component 5: Quality Gate — Perplexity Measurement (Day 4)

**File: `src/forge/research/perplexity.py`**

```python
import torch
from datasets import load_dataset

class PerplexityMeasurer:
    """Measures perplexity on WikiText-2 with sliding window."""
    
    def __init__(self, model, tokenizer, stride: int = 512, max_length: int = 2048):
        self.model = model
        self.tokenizer = tokenizer
        self.stride = stride
        self.max_length = max_length
    
    def measure(self) -> float:
        """Calculate perplexity using sliding window approach."""
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(dataset["text"])
        encodings = self.tokenizer(text, return_tensors="pt")
        
        seq_len = encodings.input_ids.size(1)
        nlls = []
        
        for begin in range(0, seq_len, self.stride):
            end = min(begin + self.max_length, seq_len)
            input_ids = encodings.input_ids[:, begin:end].to(self.model.device)
            target_ids = input_ids.clone()
            
            # Only calculate loss on the stride portion (avoid double-counting)
            target_ids[:, :-self.stride] = -100
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                nlls.append(outputs.loss.item())
            
            if end == seq_len:
                break
        
        perplexity = torch.exp(torch.tensor(nlls).mean()).item()
        return perplexity

class QualityGate:
    """Reject quantized models that degrade too much."""
    
    def __init__(self, baseline_perplexity: float, threshold: float = 0.05):
        self.baseline = baseline_perplexity
        self.threshold = threshold
    
    def check(self, measured_perplexity: float) -> tuple[bool, float]:
        degradation = (measured_perplexity - self.baseline) / self.baseline
        passed = degradation <= self.threshold
        return passed, degradation
```

### Component 6: TensorRT-LLM Compilation (Day 5)

**File: `src/forge/research/tensorrt_compile.py`**

```python
class TensorRTCompiler:
    """Compile model with TensorRT-LLM for optimized inference."""
    
    def compile(self, model_path: str, output_path: str, config: dict):
        # TensorRT-LLM compilation steps:
        # 1. Convert HF model to TRT-LLM checkpoint format
        # 2. Build TRT engine with specified optimizations
        # 3. Engine includes: kernel fusion, memory planning, quantized ops
        
        build_config = {
            "max_batch_size": config.get("max_batch_size", 32),
            "max_input_len": config.get("max_input_len", 2048),
            "max_output_len": config.get("max_output_len", 512),
            "dtype": "float16",
            "enable_custom_all_reduce": False,
            "use_paged_context_fmha": True,  # PagedAttention in TRT-LLM
        }
        
        # In practice: use trtllm-build CLI or Python API
        # trtllm-build --checkpoint_dir <path> --output_dir <path> --gemm_plugin float16
        pass

class TorchCompileBenchmark:
    """Benchmark torch.compile with different modes."""
    
    MODES = ["default", "reduce-overhead", "max-autotune"]
    
    def benchmark_all_modes(self, model, input_ids, num_warmup=5, num_bench=50):
        results = {}
        
        for mode in self.MODES:
            compiled = torch.compile(model, mode=mode)
            
            # Warmup (includes compilation time)
            compile_start = time.time()
            for _ in range(num_warmup):
                compiled(input_ids)
            torch.cuda.synchronize()
            compile_time = time.time() - compile_start
            
            # Benchmark steady-state
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_bench):
                compiled(input_ids)
            torch.cuda.synchronize()
            bench_time = time.time() - start
            
            results[mode] = {
                "compile_time_s": compile_time,
                "avg_latency_ms": (bench_time / num_bench) * 1000,
                "throughput_tps": self._calc_throughput(input_ids, bench_time, num_bench)
            }
        
        return results
```

### Component 7: Pareto Frontier Analysis (Day 6-7)

**File: `src/forge/research/pareto_analysis.py`**

```python
import plotly.graph_objects as go

class ParetoAnalysis:
    """Generate Pareto frontier showing quality vs speed vs memory tradeoffs."""
    
    def compute_pareto_frontier(self, results: list[QuantResult]) -> list[QuantResult]:
        """Find Pareto-optimal points (no other point dominates on all axes)."""
        pareto = []
        for r in results:
            dominated = False
            for other in results:
                if (other.perplexity <= r.perplexity and
                    other.throughput_tps >= r.throughput_tps and
                    other.memory_gb <= r.memory_gb and
                    other != r):
                    dominated = True
                    break
            if not dominated:
                pareto.append(r)
        return pareto
    
    def plot(self, results: list[QuantResult], output_path: str):
        """Interactive 3D Pareto chart with plotly."""
        fig = go.Figure()
        
        # All points
        fig.add_trace(go.Scatter3d(
            x=[r.perplexity for r in results],
            y=[r.throughput_tps for r in results],
            z=[r.memory_gb for r in results],
            text=[r.method.value for r in results],
            mode='markers+text',
            marker=dict(size=8, color=[self._method_color(r.method) for r in results])
        ))
        
        fig.update_layout(
            scene=dict(xaxis_title='Perplexity (lower=better)',
                      yaxis_title='Throughput (higher=better)',
                      zaxis_title='Memory GB (lower=better)')
        )
        fig.write_html(output_path)
```

## If You Get Stuck

**GPTQ calibration fails or is very slow**: Reduce calibration samples (64 instead of 128) and sequence length (1024 instead of 2048). GPTQ can take 1-4 hours for a 7B model — this is normal.

**AWQ not significantly better than GPTQ**: The difference is often small (~0.1 perplexity). AWQ's advantage is speed of quantization (no iterative process) and slightly better quality on some models. If they're similar, that's a valid finding.

**NF4 perplexity is worse**: BitsAndBytes NF4 is designed for QLoRA fine-tuning, not pure inference quality. It may show slightly higher perplexity than GPTQ/AWQ. Document this tradeoff.

**TensorRT-LLM won't install**: It has complex dependencies (CUDA version, TensorRT version). If installation fails, document the attempt and use torch.compile as the compilation benchmark instead. Skip TRT-LLM rather than spending days on environment issues.

**torch.compile graph breaks**: Some model operations cause graph breaks (dynamic shapes, data-dependent control flow). Use `torch._dynamo.explain(model)(input)` to identify breaks. Document which operations cause them.

## Agent Handoff Template

```
I'm on Week 12 of Forge — building the quantization pipeline.
Spec: /Users/jmalviya/Documents/zz/dev/plan_00/forge/specs/phase2/week12-quantization.md
Context: Weeks 8-11 complete — full inference engine with continuous batching, KV-cache memory management, speculative decoding.
I need: automated quantization pipeline comparing GPTQ/AWQ/NF4/GGUF, quality gates with perplexity, TensorRT compilation, torch.compile benchmarks, Pareto analysis.
Current state: [describe what's implemented so far]
Key challenge: [GPTQ calibration / perplexity measurement / TensorRT setup / Pareto visualization]
```

## Out of Scope

- Training-aware quantization (QAT) — we only do post-training quantization
- Custom quantization kernels (use existing libraries)
- Quantization of KV-cache (separate from weight quantization)
- FP8 quantization (requires Hopper+ GPUs)
- Pruning / sparsity (different optimization axis)
- Model distillation (training a smaller model)
- Serving the quantized models in production (just benchmarking here)
