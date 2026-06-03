# Week 16: Custom GPU Kernels with Triton

## Context

**Phase:** 3 — Production Infrastructure & Advanced Systems
**Prerequisites:** Transformer internals understanding (Week 8), working inference server, familiarity with GPU execution model (warps, blocks, grids).
**Duration:** 1 week
**Difficulty:** Advanced

You've been using PyTorch's built-in operations as black boxes. Now you'll write the actual GPU kernels that execute on hardware. Triton lets you write GPU kernels in Python-like syntax while achieving near-CUDA performance. Understanding kernel fusion, memory hierarchy, and compute utilization is what separates "I use LLMs" from "I understand why this inference is fast or slow." Every optimization in vLLM, TensorRT-LLM, and FlashAttention reduces to these principles.

---

## Learning Goals

- Understand the GPU memory hierarchy: registers → shared memory → L2 → HBM (and bandwidth at each level)
- Learn Triton's programming model: blocks, programs, masks, pointer arithmetic
- Understand kernel fusion: why combining operations reduces memory traffic
- Learn to measure kernel performance: TFLOPS achieved vs peak, memory bandwidth utilization
- Understand auto-tuning: how block sizes and warp counts affect performance
- Develop intuition for when a custom kernel is worth writing vs using library ops

---

## Implementation Goals

- Write a fused RMSNorm + residual addition kernel (eliminate extra HBM reads)
- Write a tiled attention kernel demonstrating the FlashAttention block-processing concept
- Write a fused SwiGLU MLP kernel (gate projection + up projection + activation in one pass)
- Build a benchmarking harness that reports TFLOPS and memory bandwidth utilization
- Use Triton's `@triton.autotune` to find optimal configurations per hardware
- Compare all kernels against PyTorch eager-mode equivalents with statistical significance

---

## Acceptance Criteria

1. Fused RMSNorm kernel produces output matching `torch.nn.functional.rms_norm` within 1e-3 relative tolerance across varied input shapes (batch 1-64, seq_len 128-4096, hidden 2048-8192).
2. Fused RMSNorm kernel achieves at least 30% wall-clock speedup over the equivalent two-kernel PyTorch implementation (separate norm + residual add) at sequence length 2048+.
3. Tiled attention kernel produces correct output matching `F.scaled_dot_product_attention` within 1e-2 tolerance for causal attention with head_dim=128.
4. Tiled attention kernel demonstrates memory-efficient tiling by processing Q/K/V in blocks (no full N×N attention matrix materialized), verified by peak memory usage being O(block_size²) not O(seq_len²).
5. Fused SwiGLU kernel matches the reference `silu(gate @ x) * (up @ x)` computation within 1e-3 tolerance.
6. Fused SwiGLU kernel achieves measurable speedup over the unfused 3-operation PyTorch version at hidden_dim 4096+.
7. Benchmark harness reports achieved TFLOPS and percentage of theoretical GPU peak for each kernel.
8. Benchmark harness reports effective memory bandwidth (GB/s) and percentage of hardware memory bandwidth limit.
9. Auto-tuning produces at least 3 configurations per kernel and selects the best based on measured performance, with the tuned version outperforming the untuned default.
10. A comparison table shows each Triton kernel vs PyTorch default across 5 input sizes, with mean and standard deviation of speedup ratios.

---

## Validation Commands

```bash
# Run correctness tests
pytest tests/test_rmsnorm_kernel.py -v
pytest tests/test_attention_kernel.py -v
pytest tests/test_swiglu_kernel.py -v

# Run benchmarks
python benchmarks/bench_rmsnorm.py --sizes 2048,4096,8192 --output results/rmsnorm.json
python benchmarks/bench_attention.py --seq-lens 512,1024,2048,4096 --output results/attention.json
python benchmarks/bench_swiglu.py --hidden-dims 2048,4096,8192 --output results/swiglu.json

# Generate comparison report
python benchmarks/generate_report.py --input results/ --output results/comparison.md

# Verify memory efficiency of attention kernel
python tests/test_attention_memory.py --seq-len 4096 --block-size 128

# Run auto-tuning
python kernels/rmsnorm.py --autotune --device cuda:0
python kernels/attention.py --autotune --device cuda:0
python kernels/swiglu.py --autotune --device cuda:0

# Profile a kernel with Nsight
ncu --set full python benchmarks/bench_rmsnorm.py --sizes 4096 --profile
```

---

## Technical Implementation Details

### Kernel 1: Fused RMSNorm + Residual Add

```python
# kernels/rmsnorm.py
import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def rmsnorm_residual_kernel(
    X_ptr,          # Input tensor
    Residual_ptr,   # Residual to add
    Weight_ptr,     # RMSNorm weight (gamma)
    Out_ptr,        # Output tensor
    stride_x,      # Row stride for X
    N: tl.constexpr,  # Hidden dimension
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused: out = rmsnorm(x + residual) * weight"""
    row_idx = tl.program_id(0)

    # Pointer to start of this row
    x_row_ptr = X_ptr + row_idx * stride_x
    res_row_ptr = Residual_ptr + row_idx * stride_x
    out_row_ptr = Out_ptr + row_idx * stride_x

    # Load and fuse residual add
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(res_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Fused residual addition
    hidden = x + residual

    # RMSNorm: x / sqrt(mean(x^2) + eps)
    variance = tl.sum(hidden * hidden, axis=0) / N
    rstd = 1.0 / tl.sqrt(variance + eps)
    normed = hidden * rstd

    # Apply weight
    weight = tl.load(Weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    output = normed * weight

    # Store result
    tl.store(out_row_ptr + offsets, output.to(tl.float16), mask=mask)


def fused_rmsnorm_residual(x: torch.Tensor, residual: torch.Tensor,
                           weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Launch the fused RMSNorm + residual kernel."""
    batch_seq, hidden = x.shape
    out = torch.empty_like(x)

    grid = (batch_seq,)
    rmsnorm_residual_kernel[grid](
        x, residual, weight, out,
        stride_x=x.stride(0),
        N=hidden,
        eps=eps,
    )
    return out
```

### Kernel 2: Tiled Attention

```python
# kernels/attention.py
import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["seq_len", "head_dim"],
)
@triton.jit
def tiled_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    seq_len,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Tiled attention: process Q in blocks of BLOCK_M, iterate K/V in blocks of BLOCK_N.
    Never materializes the full seq_len x seq_len attention matrix.
    """
    batch_head_idx = tl.program_id(1)
    batch_idx = batch_head_idx // tl.num_programs(1)
    head_idx = batch_head_idx % tl.num_programs(1)
    block_m_idx = tl.program_id(0)

    # Offsets for this block of queries
    m_offsets = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, head_dim)

    # Load Q block: [BLOCK_M, head_dim]
    q_ptrs = (Q_ptr + batch_idx * stride_qb + head_idx * stride_qh +
              m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qk)
    q_mask = m_offsets[:, None] < seq_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Accumulators (online softmax)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # row max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # row sum of exp
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)  # output accumulator

    # Iterate over K/V blocks
    for block_n_start in range(0, seq_len, BLOCK_N):
        n_offsets = block_n_start + tl.arange(0, BLOCK_N)

        # Causal mask: only attend to positions <= current position
        causal_mask = m_offsets[:, None] >= n_offsets[None, :]

        # Load K block: [BLOCK_N, head_dim]
        k_ptrs = (K_ptr + batch_idx * stride_kb + head_idx * stride_kh +
                  n_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kk)
        k_mask = n_offsets[:, None] < seq_len
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # QK^T: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(causal_mask, qk, float("-inf"))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        l_i = alpha * l_i + beta * tl.sum(tl.exp(qk - m_ij[:, None]), axis=1)

        # Update accumulator
        p = tl.exp(qk - m_new[:, None])
        acc = alpha[:, None] * acc + tl.dot(p.to(tl.float16), tl.load(
            V_ptr + batch_idx * stride_vb + head_idx * stride_vh +
            n_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vk,
            mask=n_offsets[:, None] < seq_len, other=0.0
        ))
        m_i = m_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output
    out_ptrs = (Out_ptr + batch_idx * stride_ob + head_idx * stride_oh +
                m_offsets[:, None] * stride_om + d_offsets[None, :] * stride_ok)
    tl.store(out_ptrs, acc.to(tl.float16), mask=m_offsets[:, None] < seq_len)
```

### Kernel 3: Fused SwiGLU

```python
# kernels/swiglu.py
import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def fused_swiglu_kernel(
    X_ptr, Gate_W_ptr, Up_W_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_gk, stride_gn,
    stride_uk, stride_un,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused SwiGLU: out = silu(x @ gate_w) * (x @ up_w)
    Computes both matmuls and the activation in a single kernel,
    avoiding writing intermediate gate/up results to HBM.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulate gate and up projections simultaneously
    gate_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    up_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load X block [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + m_offsets[:, None] * stride_xm + k_offsets[None, :] * stride_xk
        x_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load Gate weight block [BLOCK_K, BLOCK_N]
        g_ptrs = Gate_W_ptr + k_offsets[:, None] * stride_gk + n_offsets[None, :] * stride_gn
        g_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        g = tl.load(g_ptrs, mask=g_mask, other=0.0)

        # Load Up weight block [BLOCK_K, BLOCK_N]
        u_ptrs = Up_W_ptr + k_offsets[:, None] * stride_uk + n_offsets[None, :] * stride_un
        u = tl.load(u_ptrs, mask=g_mask, other=0.0)

        gate_acc += tl.dot(x, g)
        up_acc += tl.dot(x, u)

    # Fused SiLU activation: silu(x) = x * sigmoid(x)
    gate_silu = gate_acc * tl.sigmoid(gate_acc)
    output = gate_silu * up_acc

    # Store
    out_ptrs = Out_ptr + m_offsets[:, None] * stride_om + n_offsets[None, :] * stride_on
    out_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(out_ptrs, output.to(tl.float16), mask=out_mask)
```

### Benchmarking Harness

```python
# benchmarks/harness.py
import torch
import triton
import time
from dataclasses import dataclass
from typing import Callable

@dataclass
class BenchmarkResult:
    name: str
    input_shape: tuple
    time_ms: float
    tflops: float
    tflops_pct_peak: float
    bandwidth_gbs: float
    bandwidth_pct_peak: float

def get_gpu_specs():
    """Get theoretical peak performance for current GPU."""
    props = torch.cuda.get_device_properties(0)
    # Approximate peak TFLOPS (fp16 tensor cores)
    if "A100" in props.name:
        return {"peak_tflops_fp16": 312.0, "peak_bandwidth_gbs": 2039.0}
    elif "4090" in props.name:
        return {"peak_tflops_fp16": 165.0, "peak_bandwidth_gbs": 1008.0}
    elif "3090" in props.name:
        return {"peak_tflops_fp16": 71.0, "peak_bandwidth_gbs": 936.0}
    else:
        return {"peak_tflops_fp16": 100.0, "peak_bandwidth_gbs": 900.0}

def benchmark_kernel(
    fn: Callable,
    flops: int,
    bytes_accessed: int,
    warmup: int = 25,
    rep: int = 100,
) -> BenchmarkResult:
    """Benchmark a kernel with proper warmup and statistics."""
    specs = get_gpu_specs()

    # Use triton's built-in benchmarking for accuracy
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.05, 0.95])
    median_ms = ms[0]

    tflops = (flops / 1e12) / (median_ms / 1e3)
    bandwidth_gbs = (bytes_accessed / 1e9) / (median_ms / 1e3)

    return BenchmarkResult(
        name="",
        input_shape=(),
        time_ms=median_ms,
        tflops=tflops,
        tflops_pct_peak=(tflops / specs["peak_tflops_fp16"]) * 100,
        bandwidth_gbs=bandwidth_gbs,
        bandwidth_pct_peak=(bandwidth_gbs / specs["peak_bandwidth_gbs"]) * 100,
    )
```

### Project Structure

```
forge-kernels/
├── kernels/
│   ├── __init__.py
│   ├── rmsnorm.py           # Fused RMSNorm + residual
│   ├── attention.py         # Tiled attention
│   └── swiglu.py            # Fused SwiGLU MLP
├── benchmarks/
│   ├── harness.py           # Benchmarking utilities
│   ├── bench_rmsnorm.py     # RMSNorm benchmark script
│   ├── bench_attention.py   # Attention benchmark script
│   ├── bench_swiglu.py      # SwiGLU benchmark script
│   └── generate_report.py   # Comparison table generator
├── tests/
│   ├── test_rmsnorm_kernel.py
│   ├── test_attention_kernel.py
│   ├── test_attention_memory.py
│   └── test_swiglu_kernel.py
├── results/
│   └── .gitkeep
├── requirements.txt
└── README.md
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| `triton.CompilationError` on kernel | Check that all `tl.constexpr` params are actually compile-time constants; check block size divides evenly |
| Output doesn't match PyTorch reference | Accumulate in float32 even if inputs are float16; check mask boundaries carefully |
| Kernel is slower than PyTorch | PyTorch may use cuBLAS/cuDNN which are heavily optimized; ensure you're comparing against eager mode, not compiled |
| Auto-tune takes forever | Reduce the number of configs during development; use full configs only for final benchmarking |
| `CUDA out of memory` during attention | Your block sizes are too large, or you're accidentally materializing the full attention matrix |
| Triton cache stale | Delete `~/.triton/cache/` and rerun |
| `tl.dot` shape mismatch | First arg must be [M, K], second must be [K, N]; use `tl.trans()` for K^T |
| Numbers look wrong at large seq_len | Online softmax numerical stability — ensure you subtract the running max before exp |

**Key Resources:**
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135) — understand the algorithm before coding
- [GPU Memory Hierarchy blog by Lei Mao](https://leimao.github.io/blog/CUDA-Memory-Hierarchy/)
- Run `ncu` (Nsight Compute) to see actual memory throughput and occupancy

---

## Agent Handoff Template

```
## Session State
- Phase: 3 / Week 16
- Current task: [what you're working on]
- Branch: forge/week16-triton-kernels

## What's Done
- [ ] RMSNorm kernel — correct output
- [ ] RMSNorm kernel — 30%+ speedup achieved
- [ ] Attention kernel — tiling implemented
- [ ] Attention kernel — correct with causal mask
- [ ] Attention kernel — memory efficient (no full N×N)
- [ ] SwiGLU kernel — fused computation
- [ ] SwiGLU kernel — correct output
- [ ] Benchmark harness reports TFLOPS/bandwidth
- [ ] Auto-tuning working for all kernels
- [ ] Comparison table generated

## Current Blocker
[Describe the exact error/issue]

## Key Files
- kernels/rmsnorm.py — fused norm kernel
- kernels/attention.py — tiled attention kernel
- kernels/swiglu.py — fused MLP kernel
- benchmarks/harness.py — benchmark infrastructure

## GPU Hardware
[Your GPU model, VRAM, compute capability]

## Next Step
[Exact next action to take]
```

---

## Out of Scope

- Writing raw CUDA C++ kernels (Triton only this week)
- Implementing full FlashAttention v2 with all optimizations (simplified tiling concept is sufficient)
- Multi-GPU kernel distribution (single GPU only)
- Backward pass / autograd integration for training
- Integration with the inference server (standalone kernel development)
- Quantized kernel variants (INT8/INT4 Triton kernels)
- Warp-level primitives (`tl.atomic`, warp shuffle)
- Custom memory allocators or CUDA graphs
