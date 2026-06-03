# Week 6: Distributed Training Concepts

## Context

**Where it fits:** Phase 1 (Foundations), Week 6 of 7
**Prerequisites:** Week 1 (training loop), Week 2 (memory optimization), understanding of processes and inter-process communication
**What it builds on:** All memory optimization techniques from Week 2 hit physical limits on one GPU — distributed training breaks through these limits by using multiple processes/GPUs
**What comes next:** Week 7 (consolidation) ties everything together. Phase 2+ will use distributed concepts for larger alignment experiments

You have one RTX 5080. But understanding distributed training is essential because: (1) you'll work with multi-GPU clusters in production, (2) FSDP/DeepSpeed are now standard even for single-GPU training (ZeRO-Offload), and (3) the concepts (AllReduce, sharding, pipeline) appear everywhere in ML systems.

---

## Learning Goals

- [ ] Explain Data Parallelism: replicate model, split data, average gradients via AllReduce
- [ ] Articulate the AllReduce algorithm (Ring AllReduce): each GPU sends/receives N-1 times for N GPUs
- [ ] Explain ZeRO optimization stages: Stage 1 (shard optimizer states), Stage 2 (+gradients), Stage 3 (+parameters)
- [ ] Understand FSDP as PyTorch's native ZeRO-3: parameters sharded, gathered on-demand for compute
- [ ] Explain Tensor Parallelism: split individual layers across GPUs (column-parallel, row-parallel)
- [ ] Explain Pipeline Parallelism: split model layers across GPUs, micro-batching to fill pipeline bubbles
- [ ] Articulate communication overhead: when does AllReduce time dominate compute time?
- [ ] Understand the parallelism choice tradeoffs: memory reduction vs communication cost vs implementation complexity

---

## Implementation Goals

- [ ] Implement DDP training on localhost with `torch.distributed` (multiple processes, one GPU simulating multi-GPU)
- [ ] Implement manual AllReduce: each process computes gradients, sync via `dist.all_reduce`
- [ ] Implement Ring AllReduce from scratch to understand the communication pattern
- [ ] Set up FSDP training with a transformer model (understand wrapping policies)
- [ ] Profile communication vs computation time for different model/batch sizes
- [ ] Implement a simplified Pipeline Parallelism (split model into 2 stages, micro-batching)
- [ ] Implement Tensor Parallelism for a single Linear layer (column-parallel split)
- [ ] Create memory comparison: DDP vs FSDP(ZeRO-1) vs FSDP(ZeRO-3) for same model

---

## Acceptance Criteria

1. DDP training on 2 processes produces identical loss trajectory as single-process with 2x batch size (within FP tolerance)
2. Manual AllReduce produces identical averaged gradients to `torch.distributed.all_reduce` (exact match)
3. Ring AllReduce implementation correctly averages tensors across 4 simulated processes
4. FSDP training completes without errors on a 1B model using 2 processes on single GPU (CPU offload for simulation)
5. Communication profiling shows clear breakdown: time spent in AllReduce vs forward vs backward
6. Pipeline parallelism with 2 stages and 4 micro-batches achieves >75% GPU utilization (vs sequential)
7. Tensor parallel Linear layer produces identical outputs to regular Linear on same input
8. Memory comparison shows FSDP ZeRO-3 using ~1/N per-process memory vs DDP (within 20% overhead)
9. All distributed code handles initialization, synchronization, and cleanup properly (no hanging processes)
10. Written explanation of when to use each parallelism strategy based on model size, GPU count, and network bandwidth

---

## Validation Commands

```bash
# Test DDP training (2 processes on single GPU with different portions of memory)
torchrun --nproc_per_node=2 scripts/train_ddp.py --steps 100 --verify-equivalence

# Test manual AllReduce
python -m pytest tests/test_allreduce.py -v

# Ring AllReduce test
torchrun --nproc_per_node=4 scripts/test_ring_allreduce.py

# FSDP training
torchrun --nproc_per_node=2 scripts/train_fsdp.py --model 1b --strategy full_shard

# Communication profiling
torchrun --nproc_per_node=2 scripts/profile_communication.py --model-sizes 125m,350m,1b

# Pipeline parallelism test
python scripts/test_pipeline_parallel.py --stages 2 --micro-batches 4

# Tensor parallelism test
torchrun --nproc_per_node=2 scripts/test_tensor_parallel.py --verify-output

# Memory comparison across strategies
python scripts/memory_comparison.py --model gpt2-large

# Verify no hung processes after crash
torchrun --nproc_per_node=2 scripts/test_cleanup.py --simulate-crash

# Generate parallelism decision guide
python scripts/generate_decision_guide.py --output results/parallelism_guide.md
```

---

## Technical Implementation Details

### Project Structure

```
week06-distributed-training/
├── src/
│   ├── __init__.py
│   ├── ddp.py                 # Data Distributed Parallel
│   ├── allreduce.py           # Manual and Ring AllReduce
│   ├── fsdp_training.py       # FSDP setup and training
│   ├── tensor_parallel.py     # Column/Row parallel linear
│   ├── pipeline_parallel.py   # Pipeline with micro-batching
│   ├── profiling.py           # Communication vs compute timing
│   └── utils.py               # Process group setup, cleanup
├── scripts/
│   ├── train_ddp.py
│   ├── train_fsdp.py
│   ├── test_ring_allreduce.py
│   ├── test_tensor_parallel.py
│   ├── test_pipeline_parallel.py
│   ├── profile_communication.py
│   └── memory_comparison.py
├── tests/
│   ├── test_allreduce.py
│   ├── test_ddp_equivalence.py
│   ├── test_fsdp.py
│   └── test_tensor_parallel.py
└── results/
    └── .gitkeep
```

### DDP Training Setup

```python
# src/ddp.py
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed():
    """Initialize process group for distributed training."""
    dist.init_process_group(backend="nccl")  # Use "gloo" for CPU-only
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

def train_ddp(model, dataset, epochs=1, lr=1e-4):
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
    
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, sampler=sampler, pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Ensure different shuffling per epoch
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            logits = model(input_ids[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1)
            )
            
            loss.backward()
            # DDP automatically calls AllReduce on gradients here
            optimizer.step()
            optimizer.zero_grad()
            
            if local_rank == 0:
                print(f"Loss: {loss.item():.4f}")
    
    cleanup_distributed()
```

### Ring AllReduce from Scratch

```python
# src/allreduce.py
import torch
import torch.distributed as dist

def ring_allreduce(tensor: torch.Tensor) -> torch.Tensor:
    """
    Implement Ring AllReduce to average a tensor across all processes.
    
    Algorithm:
    1. Scatter-reduce: each process sends chunks to next, receives from prev, accumulates
    2. All-gather: each process sends its completed chunk around the ring
    
    Total communication: 2*(N-1)/N * tensor_size (nearly 2x tensor size for large N)
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Split tensor into world_size chunks
    chunks = list(tensor.chunk(world_size))
    
    # Pad last chunk if needed
    chunk_sizes = [c.numel() for c in chunks]
    
    # Phase 1: Scatter-Reduce
    # Each process sends its chunk[send_idx] to next, receives into chunk[recv_idx] from prev
    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size
        
        send_to = (rank + 1) % world_size
        recv_from = (rank - 1) % world_size
        
        # Send and receive simultaneously
        send_buf = chunks[send_idx].contiguous()
        recv_buf = torch.empty_like(chunks[recv_idx])
        
        send_op = dist.isend(send_buf, dst=send_to)
        recv_op = dist.irecv(recv_buf, src=recv_from)
        
        send_op.wait()
        recv_op.wait()
        
        # Accumulate received chunk
        chunks[recv_idx] += recv_buf
    
    # Phase 2: All-Gather
    # Each process now has one fully-reduced chunk; broadcast it to all
    for step in range(world_size - 1):
        send_idx = (rank - step + 1) % world_size
        recv_idx = (rank - step) % world_size
        
        send_to = (rank + 1) % world_size
        recv_from = (rank - 1) % world_size
        
        send_buf = chunks[send_idx].contiguous()
        recv_buf = torch.empty_like(chunks[recv_idx])
        
        send_op = dist.isend(send_buf, dst=send_to)
        recv_op = dist.irecv(recv_buf, src=recv_from)
        
        send_op.wait()
        recv_op.wait()
        
        chunks[recv_idx] = recv_buf
    
    # Average
    result = torch.cat(chunks) / world_size
    return result[:tensor.numel()]  # Remove padding
```

### FSDP Training Setup

```python
# src/fsdp_training.py
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

def setup_fsdp(model, sharding_strategy="FULL_SHARD", cpu_offload=False):
    """
    Wrap model with FSDP.
    
    Sharding strategies:
    - FULL_SHARD (ZeRO-3): shard params + grads + optimizer states
    - SHARD_GRAD_OP (ZeRO-2): shard grads + optimizer states
    - NO_SHARD (DDP): replicate everything
    """
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # Auto-wrap policy: wrap each transformer layer individually
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={LlamaDecoderLayer}
    )
    
    fsdp_model = FSDP(
        model,
        sharding_strategy=strategy_map[sharding_strategy],
        mixed_precision=mixed_precision_policy,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True) if cpu_offload else None,
        device_id=torch.cuda.current_device(),
    )
    
    return fsdp_model
```

### Tensor Parallelism (Column-Parallel Linear)

```python
# src/tensor_parallel.py
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """
    Split output features across GPUs.
    Each GPU computes a slice of the output: Y_i = X @ W_i
    
    Full weight W: (in_features, out_features)
    Per-GPU weight W_i: (in_features, out_features // world_size)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        assert out_features % self.world_size == 0
        self.local_out_features = out_features // self.world_size
        
        self.linear = nn.Linear(in_features, self.local_out_features, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each GPU computes its slice
        local_output = self.linear(x)
        return local_output  # Caller handles gather if needed

class RowParallelLinear(nn.Module):
    """
    Split input features across GPUs.
    Each GPU computes Y_i = X_i @ W_i, then AllReduce to sum.
    
    Full weight W: (in_features, out_features)
    Per-GPU weight W_i: (in_features // world_size, out_features)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        assert in_features % self.world_size == 0
        self.local_in_features = in_features // self.world_size
        
        self.linear = nn.Linear(self.local_in_features, out_features, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x should already be split across the input dimension
        local_output = self.linear(x)
        
        # AllReduce to sum partial results
        dist.all_reduce(local_output, op=dist.ReduceOp.SUM)
        return local_output
```

### Pipeline Parallelism (Simplified)

```python
# src/pipeline_parallel.py
import torch
import torch.nn as nn
from typing import List

class PipelineStage(nn.Module):
    """One stage of a pipeline-parallel model."""
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def pipeline_forward(stages: List[PipelineStage], input_batch: torch.Tensor,
                     num_micro_batches: int = 4):
    """
    GPipe-style pipeline parallelism with micro-batching.
    
    Split batch into micro-batches, pipeline them through stages.
    This fills the pipeline and reduces bubble overhead.
    """
    micro_batches = input_batch.chunk(num_micro_batches)
    
    # Clock cycle schedule for 2 stages, 4 micro-batches:
    # Time:    t0    t1    t2    t3    t4    t5    t6    t7
    # Stage0: MB0   MB1   MB2   MB3   ---   ---   ---   ---
    # Stage1: ---   MB0   MB1   MB2   MB3   ---   ---   ---
    # (Backward runs after all forwards complete)
    
    # Forward pass: collect intermediate activations
    stage_outputs = [[] for _ in stages]
    
    for mb_idx, mb in enumerate(micro_batches):
        x = mb
        for stage_idx, stage in enumerate(stages):
            x = stage(x)
            stage_outputs[stage_idx].append(x)
    
    # Final output is concatenation of last stage outputs
    final_output = torch.cat(stage_outputs[-1], dim=0)
    
    # Bubble fraction = (num_stages - 1) / (num_stages - 1 + num_micro_batches)
    bubble_fraction = (len(stages) - 1) / (len(stages) - 1 + num_micro_batches)
    utilization = 1 - bubble_fraction
    
    return final_output, utilization
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| `torchrun` hangs on startup | Check `MASTER_ADDR` and `MASTER_PORT`. Default: `localhost:29500`. Kill existing processes on that port |
| NCCL error on single GPU | Use `backend="gloo"` for CPU tensors or single-GPU simulation. NCCL needs actual separate GPU devices |
| DDP gives different loss than single GPU | Expected: DDP averages gradients, so effective batch size is `batch_size * world_size`. Adjust LR accordingly |
| Ring AllReduce deadlocks | Send/receive order must be consistent. Use `isend`/`irecv` (non-blocking) to avoid circular wait |
| FSDP OOM on gather | The gathered parameters temporarily need full memory. Use `limit_all_gathers=True` or CPU offload |
| Pipeline parallelism slow | Increase micro-batches to reduce bubble. With 2 stages, need ≥4 micro-batches for >75% utilization |
| Tensor parallel output doesn't match | AllReduce sum must happen AFTER row-parallel forward. Verify split dimensions match |
| Processes don't exit cleanly | Always call `dist.destroy_process_group()` in finally block. Use `torchrun` which handles signals |

---

## Agent Handoff Template

```
I'm working on Week 6 of the Crucible Phase 1 project: Distributed Training Concepts.

Hardware: RTX 5080 16GB VRAM (single GPU), 32GB RAM, Ubuntu
Project path: ~/crucible/week06-distributed-training/
Note: Using multi-process on single GPU to simulate distributed (process-based parallelism)

Current status: [DESCRIBE WHERE YOU ARE]

What I've completed:
- [x/o] DDP training setup
- [x/o] Manual AllReduce
- [x/o] Ring AllReduce from scratch
- [x/o] FSDP training
- [x/o] Communication profiling
- [x/o] Tensor Parallelism (column/row)
- [x/o] Pipeline Parallelism
- [x/o] Memory comparison table

Backend being used: [gloo/nccl]
Number of processes: [HOW MANY]

Current issue: [DESCRIBE THE PROBLEM]
Error message (if any): [PASTE ERROR]

Please help me [SPECIFIC ASK]. I'm simulating multi-GPU on a single RTX 5080 using multiple processes.
```

---

## Out of Scope

- Actual multi-node training (we simulate on single machine)
- InfiniBand / RDMA networking
- Megatron-LM or DeepSpeed library internals (understand concepts, not their codebase)
- 3D parallelism optimization (combination of TP+PP+DP at scale)
- Expert parallelism (Mixture of Experts)
- Sequence parallelism
- Async distributed training (we focus on synchronous)
- Cloud infrastructure setup (AWS/GCP multi-GPU instances)
