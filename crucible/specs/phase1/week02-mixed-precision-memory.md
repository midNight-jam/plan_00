# Week 2: Mixed Precision and Memory Optimization

## Context

**Where it fits:** Phase 1 (Foundations), Week 2 of 7
**Prerequisites:** Week 1 (custom training loop), understanding of floating point representation, basic GPU architecture awareness
**What it builds on:** The training loop from Week 1 — we now optimize it to handle larger models within 16GB VRAM
**What comes next:** Week 3 (Data Pipelines) and Week 4 (LoRA) both depend on the memory techniques learned here

Training a 1B parameter model requires ~4GB just for parameters in FP32. Add gradients (4GB), optimizer states (8GB for AdamW), and activations — you're at 20+ GB before a single forward pass. This week teaches you to fit the "impossible" into 16GB.

---

## Learning Goals

- [ ] Explain the bit layout of FP32, FP16, and BF16 and the tradeoffs (range vs precision)
- [ ] Understand why FP16 training can diverge without loss scaling (gradient underflow)
- [ ] Explain what GradScaler does: scale loss up → gradients don't underflow → unscale before optimizer step
- [ ] Articulate why BF16 doesn't need loss scaling (same exponent range as FP32)
- [ ] Explain gradient checkpointing: discard activations during forward, recompute during backward
- [ ] Calculate memory savings from gradient checkpointing (√N memory for N layers)
- [ ] Understand gradient accumulation as a way to simulate larger batch sizes
- [ ] Break down the full memory equation: params + grads + optimizer states + activations

---

## Implementation Goals

- [ ] Implement FP16 training with manual GradScaler (not autocast-only)
- [ ] Implement BF16 training and compare stability with FP16
- [ ] Profile memory usage at each stage: after model load, after forward, after backward, after optimizer step
- [ ] Implement gradient checkpointing manually using `torch.utils.checkpoint`
- [ ] Implement gradient accumulation with proper loss normalization
- [ ] Combine all techniques: BF16 + gradient checkpointing + gradient accumulation
- [ ] Train a 1B parameter model on RTX 5080 (16GB) using combined techniques
- [ ] Benchmark: create table of memory/throughput with each technique enabled/disabled

---

## Acceptance Criteria

1. FP16 training with GradScaler produces training loss within 5% of FP32 baseline after 500 steps
2. A deliberate test shows FP16 WITHOUT GradScaler producing NaN/Inf within 200 steps on a numerically challenging setup
3. BF16 training completes 500 steps without NaN on the same setup that breaks raw FP16
4. Memory profiler shows gradient checkpointing reduces peak activation memory by at least 50%
5. Gradient accumulation over 4 micro-batches produces identical gradients to a single batch of 4x size (within tolerance)
6. A 1B parameter model (TinyLlama-1.1B or similar) trains successfully on 16GB VRAM with all optimizations
7. Benchmark table shows at least 3x memory reduction from FP32-naive to fully-optimized configuration
8. Throughput (tokens/sec) is measured for each configuration and BF16 shows at least 1.5x speedup over FP32
9. Memory breakdown pie chart correctly accounts for >95% of allocated GPU memory
10. Training run with all optimizations maintains gradient norm stability (no sudden spikes >10x median)

---

## Validation Commands

```bash
# Unit test: GradScaler correctly skips steps on inf gradients
python -m pytest tests/test_mixed_precision.py -v

# Compare FP16+scaler vs FP32 loss trajectories
python scripts/compare_precision.py --steps 500 --plot

# Show FP16 without scaler produces NaN
python scripts/fp16_no_scaler.py --steps 200 | grep -c "nan"

# Memory profiling for each configuration
python scripts/memory_profile.py --config fp32
python scripts/memory_profile.py --config fp16
python scripts/memory_profile.py --config bf16
python scripts/memory_profile.py --config bf16+checkpoint
python scripts/memory_profile.py --config bf16+checkpoint+accumulation

# Train 1B model with all optimizations
python scripts/train_1b.py --dtype bf16 --gradient-checkpointing --grad-accum-steps 8

# Verify gradient accumulation equivalence
python scripts/verify_grad_accum.py --micro-batches 4 --tolerance 1e-5

# Generate benchmark table
python scripts/benchmark.py --output results/memory_benchmark.md

# Check no NaN in full training run
python scripts/train_1b.py --steps 200 --check-nan
```

---

## Technical Implementation Details

### Project Structure

```
week02-mixed-precision-memory/
├── src/
│   ├── __init__.py
│   ├── mixed_precision.py     # FP16/BF16 training contexts
│   ├── grad_checkpoint.py     # Manual gradient checkpointing
│   ├── grad_accumulation.py   # Accumulation wrapper
│   ├── memory_profiler.py     # GPU memory breakdown tool
│   └── training_loop.py       # Extended from Week 1
├── scripts/
│   ├── train_1b.py            # Train 1B model
│   ├── compare_precision.py   # FP32 vs FP16 vs BF16
│   ├── memory_profile.py      # Profile memory stages
│   ├── benchmark.py           # Generate comparison table
│   └── verify_grad_accum.py   # Verify accumulation correctness
├── tests/
│   ├── test_mixed_precision.py
│   ├── test_grad_checkpoint.py
│   └── test_grad_accumulation.py
└── results/
    └── .gitkeep
```

### Manual GradScaler Implementation

```python
# src/mixed_precision.py
import torch

class ManualGradScaler:
    """Simplified GradScaler to understand the mechanism."""
    
    def __init__(self, init_scale=2**16, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._steps_since_growth = 0
    
    def scale_loss(self, loss):
        return loss * self.scale
    
    def unscale_and_step(self, optimizer, model):
        has_inf = False
        for param in model.parameters():
            if param.grad is not None:
                param.grad.div_(self.scale)
                if torch.any(torch.isinf(param.grad)) or torch.any(torch.isnan(param.grad)):
                    has_inf = True
                    break
        
        if has_inf:
            # Skip step, reduce scale
            self.scale *= self.backoff_factor
            self._steps_since_growth = 0
            optimizer.zero_grad()
            return False
        else:
            optimizer.step()
            self._steps_since_growth += 1
            if self._steps_since_growth >= self.growth_interval:
                self.scale *= self.growth_factor
                self._steps_since_growth = 0
            return True
```

### Memory Profiler

```python
# src/memory_profiler.py
import torch
from dataclasses import dataclass

@dataclass
class MemorySnapshot:
    stage: str
    allocated_mb: float
    reserved_mb: float
    peak_mb: float

def profile_training_step(model, batch, optimizer, dtype=torch.float32):
    device = next(model.parameters()).device
    snapshots = []
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    # After model load
    snapshots.append(MemorySnapshot(
        stage="model_loaded",
        allocated_mb=torch.cuda.memory_allocated(device) / 1e6,
        reserved_mb=torch.cuda.memory_reserved(device) / 1e6,
        peak_mb=torch.cuda.max_memory_allocated(device) / 1e6
    ))
    
    # After forward pass
    with torch.autocast(device_type='cuda', dtype=dtype) if dtype != torch.float32 else nullcontext():
        logits = model(batch['input_ids'])
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            batch['input_ids'][:, 1:].reshape(-1)
        )
    
    snapshots.append(MemorySnapshot(
        stage="after_forward",
        allocated_mb=torch.cuda.memory_allocated(device) / 1e6,
        reserved_mb=torch.cuda.memory_reserved(device) / 1e6,
        peak_mb=torch.cuda.max_memory_allocated(device) / 1e6
    ))
    
    # After backward
    loss.backward()
    snapshots.append(MemorySnapshot(
        stage="after_backward",
        allocated_mb=torch.cuda.memory_allocated(device) / 1e6,
        reserved_mb=torch.cuda.memory_reserved(device) / 1e6,
        peak_mb=torch.cuda.max_memory_allocated(device) / 1e6
    ))
    
    # After optimizer step
    optimizer.step()
    optimizer.zero_grad()
    snapshots.append(MemorySnapshot(
        stage="after_optim_step",
        allocated_mb=torch.cuda.memory_allocated(device) / 1e6,
        reserved_mb=torch.cuda.memory_reserved(device) / 1e6,
        peak_mb=torch.cuda.max_memory_allocated(device) / 1e6
    ))
    
    return snapshots
```

### Gradient Accumulation with Proper Normalization

```python
# src/grad_accumulation.py
import torch

def train_with_accumulation(model, dataloader, optimizer, scheduler, 
                            accum_steps=4, max_steps=1000, dtype=torch.bfloat16):
    model.train()
    device = next(model.parameters()).device
    step = 0
    
    for micro_step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        
        with torch.autocast(device_type='cuda', dtype=dtype):
            logits = model(input_ids[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1)
            )
            # Normalize by accumulation steps so effective loss matches large batch
            loss = loss / accum_steps
        
        loss.backward()
        
        if (micro_step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            
            if step >= max_steps:
                break
    
    return step
```

### Gradient Checkpointing Usage

```python
# src/grad_checkpoint.py
import torch
from torch.utils.checkpoint import checkpoint

def apply_gradient_checkpointing(model):
    """Apply gradient checkpointing to transformer layers."""
    if hasattr(model, 'transformer'):
        # GPT-2 style
        for layer in model.transformer.h:
            layer.forward = make_checkpointed(layer.forward)
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # LLaMA style
        for layer in model.model.layers:
            layer.forward = make_checkpointed(layer.forward)

def make_checkpointed(forward_fn):
    def checkpointed_forward(*args, **kwargs):
        # checkpoint requires at least one tensor input with requires_grad
        return checkpoint(forward_fn, *args, use_reentrant=False, **kwargs)
    return checkpointed_forward
```

### The Memory Equation

```python
# Reference calculation (not runnable code — conceptual)
def memory_estimate(param_count, batch_size, seq_len, hidden_dim, n_layers, dtype_bytes):
    """
    Memory breakdown for transformer training:
    
    FP32 (4 bytes): param_count * 4
    FP16 (2 bytes): param_count * 2
    
    AdamW states: 2 * param_count * 4 (always FP32: first moment + second moment)
    Gradients: param_count * dtype_bytes
    
    Activations per layer ≈ batch_size * seq_len * hidden_dim * dtype_bytes * ~10
    (attention scores, residuals, layer norms, FFN intermediates)
    
    Total without checkpointing:
      params + grads + optimizer_states + activations * n_layers
    
    Total with checkpointing:
      params + grads + optimizer_states + activations * sqrt(n_layers)
    """
    params_mem = param_count * dtype_bytes
    grads_mem = param_count * dtype_bytes
    optim_mem = param_count * 4 * 2  # AdamW: 2 FP32 states
    act_per_layer = batch_size * seq_len * hidden_dim * dtype_bytes * 10
    
    total_no_ckpt = params_mem + grads_mem + optim_mem + act_per_layer * n_layers
    total_with_ckpt = params_mem + grads_mem + optim_mem + act_per_layer * int(n_layers**0.5)
    
    return {
        'params_gb': params_mem / 1e9,
        'grads_gb': grads_mem / 1e9,
        'optimizer_gb': optim_mem / 1e9,
        'activations_no_ckpt_gb': (act_per_layer * n_layers) / 1e9,
        'activations_with_ckpt_gb': (act_per_layer * int(n_layers**0.5)) / 1e9,
        'total_no_ckpt_gb': total_no_ckpt / 1e9,
        'total_with_ckpt_gb': total_with_ckpt / 1e9,
    }
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| CUDA OOM even with BF16 | Check you're using `torch.autocast` context correctly. Model params may still be FP32 — cast model with `model.to(torch.bfloat16)` |
| GradScaler always skipping steps | Initial scale too high. Start with `2**10` instead of `2**16`. Or your loss function produces huge values |
| NaN with FP16, fine with FP32 | Classic underflow. Verify GradScaler is active. Check for `log(0)` or division by very small numbers in your loss |
| Gradient checkpointing not saving memory | Verify `use_reentrant=False`. Check that layers actually have `requires_grad=True` tensors |
| Grad accumulation gives different loss | Must divide loss by `accum_steps` BEFORE `.backward()`. Otherwise gradients are accum_steps × too large |
| 1B model won't load at all | Load in BF16: `model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)` |
| Memory profiler shows less than expected | `torch.cuda.memory_allocated` doesn't count cached allocator memory. Use `memory_reserved` for total |
| BF16 slightly worse quality than FP32 | Expected. BF16 has less mantissa precision. Difference should be <5% loss after convergence |

---

## Agent Handoff Template

```
I'm working on Week 2 of the Crucible Phase 1 project: Mixed Precision and Memory Optimization.

Hardware: RTX 5080 16GB VRAM, 32GB RAM, Ubuntu
Project path: ~/crucible/week02-mixed-precision-memory/

Current status: [DESCRIBE WHERE YOU ARE]

What I've completed:
- [x/o] Manual GradScaler implementation
- [x/o] FP16 vs BF16 comparison
- [x/o] Gradient checkpointing integration
- [x/o] Gradient accumulation
- [x/o] Memory profiler
- [x/o] 1B model training
- [x/o] Benchmark table

Memory budget: 16GB VRAM total. Target: train 1B model with batch_size >= 4, seq_len = 2048.

Current issue: [DESCRIBE THE PROBLEM]
Error message (if any): [PASTE ERROR]
Current memory usage: [nvidia-smi output or profiler output]

Please help me [SPECIFIC ASK].
```

---

## Out of Scope

- Multi-GPU memory sharing (Week 6 — distributed training)
- Quantization for training (Week 4 — QLoRA covers NF4)
- CPU offloading (DeepSpeed ZeRO-Offload)
- Custom CUDA kernels for mixed precision
- FlashAttention implementation (use as library, don't implement)
- Model parallelism strategies (Week 6)
- INT8 training or FP8 (H100-specific features)
