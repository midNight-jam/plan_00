# Week 18: Distributed Training Infrastructure

## Context

**Where it fits:** Phase 3 (Evaluation, Safety, and Portfolio), Week 18 of 20.

**Prerequisites:**
- Completed Phase 1-2: working single-GPU training loops (SFT, DPO, RLHF)
- Understanding of PyTorch autograd, optimizer state, gradient accumulation
- Familiarity with mixed-precision training (fp16/bf16) from earlier weeks
- Basic understanding of inter-process communication concepts

**What it builds on:** All training in Phase 1-2 used single-GPU setups with gradient accumulation and quantization to fit in 16GB VRAM. This week extends your training infrastructure to support distributed training via FSDP (Fully Sharded Data Parallelism). Even on a single machine, you'll simulate multi-process training to understand the distributed paradigm used at scale.

**Why it matters:** Production LLM training always uses distributed infrastructure. Understanding FSDP, sharding strategies, communication patterns, and fault tolerance is essential for working at any lab. Even if you only have one GPU, understanding the distributed training abstraction is critical because: (1) it's how all large-scale training works, (2) interview questions test this knowledge, (3) debugging distributed training issues requires understanding the full stack.

---

## Learning Goals

- [ ] Understand data parallelism vs. model parallelism vs. fully sharded data parallelism
- [ ] Learn ZeRO optimization stages: ZeRO-1 (shard optimizer), ZeRO-2 (shard grad+opt), ZeRO-3 (shard everything)
- [ ] Understand FSDP sharding: how parameters are sharded across ranks, all-gather for forward, reduce-scatter for backward
- [ ] Learn mixed precision in distributed context: param dtype, buffer dtype, reduce dtype, compute dtype
- [ ] Understand gradient communication: all-reduce, reduce-scatter, ring-allreduce topology
- [ ] Learn fault-tolerant training: checkpointing, elastic training, handling worker failures
- [ ] Understand communication-computation overlap and how to maximize GPU utilization
- [ ] Learn async checkpointing: non-blocking saves that don't stall training

---

## Implementation Goals

- [ ] Implement FSDP wrapper for existing SFT training loop
- [ ] Configure sharding strategies: FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
- [ ] Implement proper mixed-precision policy with FSDP (MixedPrecision dataclass)
- [ ] Build distributed checkpoint saving/loading (torch.distributed.checkpoint)
- [ ] Implement async checkpointing using background threads
- [ ] Add fault tolerance: graceful handling of worker crashes, automatic restart
- [ ] Implement communication-computation overlap (prefetch, limit_all_gathers)
- [ ] Run multi-process training on single machine (simulate 2-4 "GPUs" with CPU or split GPU memory)
- [ ] Measure and log communication overhead vs. computation time
- [ ] Implement gradient clipping in distributed context
- [ ] Build elastic training support (worker count can change mid-training)

---

## Acceptance Criteria

1. FSDP-wrapped training loop runs successfully with `torchrun --nproc_per_node=2` on single machine, producing identical loss curves (within numerical precision) to single-process training with equivalent batch size.
2. All three sharding strategies (FULL_SHARD, SHARD_GRAD_OP, NO_SHARD) are implemented and configurable, with measured memory usage for each (FULL_SHARD uses least per-rank memory).
3. Mixed precision policy correctly configures param_dtype=torch.float16, reduce_dtype=torch.float32, buffer_dtype=torch.float32, and training is numerically stable (no NaN/Inf in 1000 steps).
4. Distributed checkpointing saves/loads model correctly: a checkpoint saved from 2-process training loads correctly into 1-process or 4-process configuration (resharding).
5. Async checkpointing is implemented: checkpoint save does not block training for more than 100ms (measured via timing instrumentation).
6. Communication overhead is measured per step: time spent in all-gather, reduce-scatter, and idle waiting, reported as percentage of total step time.
7. Communication-computation overlap is implemented (forward prefetch, backward prefetch) and reduces measured communication overhead by ≥20% compared to non-overlapped version.
8. Gradient clipping works correctly in distributed setting: global norm is computed across all shards and clipping produces same result as single-process training.
9. Training handles simulated worker failure: when one process is killed, remaining processes detect the failure and either restart or save checkpoint and exit gracefully (no hang, no corruption).
10. Scaling efficiency report: measure throughput (tokens/sec) for 1, 2, and 4 processes and report scaling efficiency (ideal = linear, measure actual fraction of ideal).

---

## Validation Commands

```bash
# Run FSDP training with 2 processes
cd ~/crucible/distributed
torchrun --nproc_per_node=2 --master_port=29500 \
  train_fsdp.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset data/sft_train.jsonl \
  --sharding_strategy full_shard \
  --mixed_precision bf16 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --max_steps 100 \
  --output_dir checkpoints/fsdp_test

# Compare sharding strategies
for strategy in full_shard shard_grad_op no_shard; do
  torchrun --nproc_per_node=2 --master_port=29500 \
    train_fsdp.py \
    --sharding_strategy $strategy \
    --max_steps 50 \
    --log_memory \
    --output_dir checkpoints/${strategy}_test
done

# Test checkpoint resharding (save with 2, load with 1)
torchrun --nproc_per_node=2 --master_port=29500 \
  save_checkpoint.py --output_dir checkpoints/2proc
python load_checkpoint.py \
  --checkpoint_dir checkpoints/2proc \
  --target_world_size 1

# Measure communication overhead
torchrun --nproc_per_node=2 --master_port=29500 \
  train_fsdp.py \
  --profile_communication \
  --max_steps 20 \
  --output_dir profiling/comm_overhead

# Test fault tolerance
python test_fault_tolerance.py --nproc=2 --kill_rank=1 --kill_after_steps=50

# Run scaling efficiency benchmark
python scaling_benchmark.py \
  --max_procs 4 \
  --steps_per_config 50 \
  --output scaling_results.json

# Validate gradient clipping consistency
python -m pytest tests/test_grad_clipping.py -v

# Test async checkpointing
torchrun --nproc_per_node=2 --master_port=29500 \
  train_fsdp.py \
  --async_checkpoint \
  --checkpoint_every 10 \
  --max_steps 50 \
  --measure_checkpoint_time
```

---

## Technical Implementation Details

### Project Structure

```
~/crucible/distributed/
├── train_fsdp.py                   # Main FSDP training script
├── save_checkpoint.py              # Distributed checkpoint save
├── load_checkpoint.py              # Checkpoint load with resharding
├── scaling_benchmark.py            # Throughput measurement
├── test_fault_tolerance.py         # Fault tolerance testing
├── fsdp/
│   ├── __init__.py
│   ├── wrapper.py                  # FSDP wrapping logic
│   ├── policies.py                 # Sharding and mixed precision policies
│   ├── checkpoint.py               # Distributed checkpointing
│   ├── async_checkpoint.py         # Non-blocking checkpoint
│   ├── communication.py            # Comm profiling and overlap
│   └── elastic.py                  # Elastic training support
├── tests/
│   ├── test_grad_clipping.py
│   ├── test_checkpoint.py
│   └── test_sharding.py
├── profiling/
│   └── comm_overhead/
└── checkpoints/
```

### FSDP Training Loop

```python
# train_fsdp.py
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from functools import partial
import time

def setup_distributed():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def get_sharding_strategy(name: str) -> ShardingStrategy:
    strategies = {
        "full_shard": ShardingStrategy.FULL_SHARD,        # ZeRO-3
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,  # ZeRO-2
        "no_shard": ShardingStrategy.NO_SHARD,            # DDP equivalent
    }
    return strategies[name]

def get_mixed_precision_policy(dtype: str) -> MixedPrecision:
    if dtype == "bf16":
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
    elif dtype == "fp16":
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
    return None

def wrap_model_fsdp(
    model,
    sharding_strategy: ShardingStrategy,
    mixed_precision: MixedPrecision,
    backward_prefetch: bool = True,
    forward_prefetch: bool = True,
    limit_all_gathers: bool = True,
):
    """Wrap model with FSDP using transformer layer auto-wrap policy."""
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE if backward_prefetch else None,
        forward_prefetch=forward_prefetch,
        limit_all_gathers=limit_all_gathers,
        device_id=torch.cuda.current_device(),
    )
    return model


def train(args):
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"Training with {world_size} processes")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    sharding_strategy = get_sharding_strategy(args.sharding_strategy)
    mixed_precision = get_mixed_precision_policy(args.mixed_precision)

    model = wrap_model_fsdp(model, sharding_strategy, mixed_precision)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Training loop
    model.train()
    step_times = []
    comm_times = []

    for step in range(args.max_steps):
        step_start = time.perf_counter()

        batch = get_batch(tokenizer, args.batch_size, local_rank)
        input_ids = batch["input_ids"].to(f"cuda:{local_rank}")
        labels = batch["labels"].to(f"cuda:{local_rank}")

        # Forward pass (triggers all-gather for FULL_SHARD)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps
        loss.backward()  # triggers reduce-scatter for FULL_SHARD

        if (step + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping (must use FSDP's clip_grad_norm_)
            grad_norm = model.clip_grad_norm_(args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        step_time = time.perf_counter() - step_start
        step_times.append(step_time)

        if rank == 0 and step % 10 == 0:
            tokens_per_sec = (
                args.batch_size * world_size * 512 / step_time  # assuming seq_len=512
            )
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            print(
                f"Step {step}: loss={loss.item():.4f}, "
                f"tokens/s={tokens_per_sec:.0f}, "
                f"mem={mem_gb:.1f}GB, "
                f"grad_norm={grad_norm:.2f}"
            )

    dist.destroy_process_group()
```

### Distributed Checkpointing

```python
# fsdp/checkpoint.py
import torch
import torch.distributed as dist
from torch.distributed.checkpoint import (
    save as dist_save,
    load as dist_load,
    FileSystemReader,
    FileSystemWriter,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig,
)
from pathlib import Path
import json

def save_distributed_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    path: str,
    step: int,
    use_sharded: bool = True,
):
    """
    Save distributed checkpoint.
    
    Sharded format: each rank saves its own shard. Fast to save,
    but requires same world_size to load (unless using dist.checkpoint).
    
    Full format: rank 0 saves complete model. Slow (requires all-gather)
    but loads anywhere.
    """
    Path(path).mkdir(parents=True, exist_ok=True)

    if use_sharded:
        with FSDP.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
        ):
            state_dict = {"model": model.state_dict()}
            optim_state = FSDP.optim_state_dict(model, optimizer)
            state_dict["optimizer"] = optim_state
            state_dict["step"] = step

            dist_save(state_dict, FileSystemWriter(path))
    else:
        # Full state dict (rank 0 only)
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
            state_dict = model.state_dict()
            if dist.get_rank() == 0:
                torch.save({
                    "model": state_dict,
                    "step": step,
                }, Path(path) / "full_model.pt")

    if dist.get_rank() == 0:
        metadata = {
            "step": step,
            "world_size": dist.get_world_size(),
            "format": "sharded" if use_sharded else "full",
        }
        with open(Path(path) / "metadata.json", "w") as f:
            json.dump(metadata, f)

    dist.barrier()


def load_distributed_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    path: str,
):
    """Load checkpoint with automatic resharding if world_size changed."""
    with FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True),
    ):
        state_dict = {"model": model.state_dict()}
        dist_load(state_dict, FileSystemReader(path))
        model.load_state_dict(state_dict["model"])

        # Load optimizer state
        optim_state = {"optimizer": FSDP.optim_state_dict(model, optimizer)}
        try:
            dist_load(optim_state, FileSystemReader(path))
            FSDP.optim_state_dict_to_load(model, optimizer, optim_state["optimizer"])
            optimizer.load_state_dict(optim_state["optimizer"])
        except Exception:
            pass  # optimizer state may not exist or be reshardable

    dist.barrier()
```

### Async Checkpointing

```python
# fsdp/async_checkpoint.py
import torch
import threading
import time
from pathlib import Path
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
import torch.distributed as dist

class AsyncCheckpointer:
    """Non-blocking checkpointing that saves in background thread."""

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._save_thread = None
        self._lock = threading.Lock()
        self.last_save_time_ms = 0.0

    def save(self, model: FSDP, optimizer, step: int):
        """
        Initiate async checkpoint save.
        Copies state to CPU in main thread (fast with offload_to_cpu),
        then writes to disk in background thread.
        """
        start = time.perf_counter()

        # Wait for previous save to complete
        if self._save_thread is not None and self._save_thread.is_alive():
            self._save_thread.join()

        # Extract state dict to CPU (this is the blocking part, but fast with FSDP)
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            optim_state = optimizer.state_dict()  # already on CPU for FSDP

        blocking_time_ms = (time.perf_counter() - start) * 1000

        # Write to disk in background
        save_path = self.save_dir / f"step_{step}"
        self._save_thread = threading.Thread(
            target=self._write_checkpoint,
            args=(state_dict, optim_state, step, save_path),
        )
        self._save_thread.start()

        self.last_save_time_ms = blocking_time_ms
        return blocking_time_ms

    def _write_checkpoint(self, state_dict, optim_state, step, path):
        path.mkdir(parents=True, exist_ok=True)
        rank = dist.get_rank() if dist.is_initialized() else 0
        torch.save({
            "model": state_dict,
            "optimizer": optim_state,
            "step": step,
        }, path / f"rank_{rank}.pt")

    def wait(self):
        """Wait for any pending save to complete."""
        if self._save_thread is not None:
            self._save_thread.join()
```

### Communication Profiling

```python
# fsdp/communication.py
import torch
import torch.distributed as dist
import time
from contextlib import contextmanager
from collections import defaultdict

class CommProfiler:
    """Profile communication operations in distributed training."""

    def __init__(self):
        self.timings = defaultdict(list)
        self._active = False
        self._hooks = []

    def start(self):
        self._active = True
        self._patch_collective_ops()

    def stop(self):
        self._active = False
        self._unpatch()

    def _patch_collective_ops(self):
        """Monkey-patch distributed ops to measure timing."""
        ops_to_patch = ["all_reduce", "all_gather", "reduce_scatter", "broadcast"]
        for op_name in ops_to_patch:
            original = getattr(dist, op_name)
            patched = self._make_timed_op(op_name, original)
            setattr(dist, f"_original_{op_name}", original)
            setattr(dist, op_name, patched)
            self._hooks.append(op_name)

    def _make_timed_op(self, name, original_fn):
        profiler = self

        def timed_op(*args, **kwargs):
            if not profiler._active:
                return original_fn(*args, **kwargs)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = original_fn(*args, **kwargs)
            if isinstance(result, dist.Work):
                result.wait()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            profiler.timings[name].append(elapsed)
            return result

        return timed_op

    def _unpatch(self):
        for op_name in self._hooks:
            original = getattr(dist, f"_original_{op_name}")
            setattr(dist, op_name, original)
        self._hooks = []

    def report(self) -> dict:
        total_comm = sum(sum(v) for v in self.timings.values())
        report = {
            "total_communication_time_s": total_comm,
            "per_operation": {},
        }
        for op, times in self.timings.items():
            report["per_operation"][op] = {
                "total_s": sum(times),
                "count": len(times),
                "mean_ms": (sum(times) / len(times)) * 1000 if times else 0,
                "max_ms": max(times) * 1000 if times else 0,
            }
        return report


def measure_scaling_efficiency(
    throughputs: dict[int, float],  # {num_procs: tokens_per_sec}
) -> dict:
    """
    Compute scaling efficiency.
    
    Ideal (linear) scaling: throughput_N = N * throughput_1
    Efficiency = throughput_N / (N * throughput_1)
    """
    if 1 not in throughputs:
        base_throughput = min(throughputs.values())
        base_procs = min(throughputs.keys())
    else:
        base_throughput = throughputs[1]
        base_procs = 1

    results = {}
    for n_procs, throughput in sorted(throughputs.items()):
        ideal = (n_procs / base_procs) * base_throughput
        efficiency = throughput / ideal
        results[n_procs] = {
            "throughput_tokens_per_sec": throughput,
            "ideal_throughput": ideal,
            "efficiency": efficiency,
            "speedup": throughput / base_throughput,
            "ideal_speedup": n_procs / base_procs,
        }
    return results
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| NCCL errors on startup | Ensure `MASTER_ADDR` and `MASTER_PORT` are set. Use `torchrun` which handles this. Check that NCCL can find your GPU (`NCCL_DEBUG=INFO`). |
| OOM with FSDP FULL_SHARD | The model may be too large even when sharded. Try `cpu_offload=CPUOffload(offload_params=True)` or use a smaller model for testing. |
| Training hangs during backward | One rank may have hit an error while others wait at collective op. Add timeout: `dist.init_process_group(..., timeout=timedelta(minutes=5))`. |
| Loss is different between 1-process and 2-process | Check batch size: effective batch = per_rank_batch * world_size. Use gradient accumulation to match effective batch. Also check reduce dtype. |
| Checkpoint loading fails with different world_size | Use `torch.distributed.checkpoint` (not `torch.save/load`) for reshardable checkpoints. |
| Communication overhead is >50% | Batch size too small (compute/comm ratio too low). Increase batch size or sequence length. Enable overlap with `backward_prefetch`. |
| Gradient norm differs across ranks | With FULL_SHARD, use `model.clip_grad_norm_()` (FSDP method) not `torch.nn.utils.clip_grad_norm_()`. FSDP handles the all-reduce internally. |
| Only have 1 GPU but want to test | Use `--nproc_per_node=2` with small model—processes share GPU. Or use CPU-only mode: `device_map="cpu"` with gloo backend. |

---

## Agent Handoff Template

```
Continue building distributed training infrastructure for ~/crucible/distributed/.

Current state: [describe what's implemented]

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB VRAM, 32GB RAM, Ubuntu.
Note: Single GPU, so multi-process training shares GPU or uses CPU simulation.

What's working: [list components]
What's broken: [describe failures]

Training configuration:
- Model: [name/size]
- Sharding strategy: [full_shard/shard_grad_op/no_shard]
- Mixed precision: [bf16/fp16/none]
- Num processes: [2/4]

Current measurements:
- Single-process throughput: [X tokens/sec]
- Multi-process throughput: [Y tokens/sec]
- Communication overhead: [Z%]

Next steps from acceptance criteria:
- [ ] [next unchecked criterion]

Key constraints:
- Only 1 physical GPU (16GB VRAM), must simulate multi-process
- NCCL backend for GPU, gloo for CPU-only testing
- Checkpoints must be reshardable across different world_sizes
```

---

## Out of Scope

- Multi-node training across machines (single-machine, multi-process only)
- Pipeline parallelism (we focus on FSDP/data parallelism only)
- Tensor parallelism (Megatron-style column/row parallel)
- Custom CUDA kernels for communication (we use NCCL)
- Cloud infrastructure setup (AWS, GCP instance management)
- DeepSpeed integration (we use native PyTorch FSDP)
- Training at actual scale (we simulate and measure, not train full models)
- Inference serving infrastructure (training infrastructure only)
