# Week 10: KV-Cache and Memory Management
> Phase: 2 | Project: Forge | Estimated Duration: 7 days

## Context

Week 9 built the continuous batching scheduler that dynamically manages sequences. But we hand-waved memory management — "just allocate KV-cache." In production, KV-cache dominates GPU memory. A 7B model serving 100 concurrent sequences at 2048 tokens each needs ~40GB just for KV-cache. This week you build the memory subsystem that makes high-throughput serving possible.

The key insight: treat GPU memory like an OS treats RAM. Fixed-size blocks, logical-to-physical mapping (page tables), eviction policies, and defragmentation. This is exactly what vLLM's PagedAttention does.

**Prerequisites**: Week 9 complete — working continuous batching scheduler with basic KV-cache allocation.

**Builds on**: The scheduler from Week 9 now gets a real memory manager instead of naive pre-allocation.

## Learning Goals

- [ ] Understand why naive KV-cache allocation wastes memory (pre-allocate max_seq_len per sequence)
- [ ] Understand block-based allocation — like OS page tables (fixed blocks, logical-to-physical mapping)
- [ ] Understand internal vs external fragmentation in GPU memory
- [ ] Understand eviction policies: LRU, priority-based (preempt low-priority sequences)
- [ ] Understand GPU-CPU memory hierarchy and PCIe transfer costs
- [ ] Understand copy-on-write semantics for parallel sampling (beam search, best-of-N)
- [ ] Understand memory profiling — where does VRAM actually go?

## Implementation Goals

- [ ] Implement block-based KV-cache allocator (fixed 16-token blocks, logical-to-physical table)
- [ ] Implement block reference counting (for copy-on-write)
- [ ] Implement LRU eviction policy (when VRAM full, swap LRU sequence's KV to CPU)
- [ ] Implement priority-based eviction (prefer evicting longer-waiting or lower-priority sequences)
- [ ] Implement GPU↔CPU KV-cache swap (async PCIe transfers)
- [ ] Implement defragmentation (compact blocks to eliminate holes)
- [ ] Implement copy-on-write for fork operations (beam search shares prefix KV)
- [ ] Build GPU memory profiler tool (shows breakdown: weights, KV, activations, free, fragmented)
- [ ] Build model offloading (cold models to CPU RAM, hot models in GPU)
- [ ] Integrate with Week 9's scheduler (memory-aware admission control)

## Acceptance Criteria

1. **Block allocator tested**: Allocate/free 1000 sequences without memory leaks — verified by checking all blocks return to free list
2. **Logical-to-physical mapping**: Sequences use non-contiguous physical blocks, attention still produces correct output
3. **Eviction measured**: When VRAM is 95% full, LRU eviction frees blocks and allows new sequences to enter
4. **Swap latency < 500ms**: Moving a sequence's KV-cache (2048 tokens) from GPU to CPU takes < 500ms round-trip
5. **Profiler produces charts**: Memory breakdown visualization shows weights, KV-cache (per sequence), activations, free space, fragmented space
6. **Fragmentation < 5%**: Under sustained mixed workload (1000 allocate/free cycles), unusable fragmented memory stays below 5%
7. **Copy-on-write works**: Beam search with beam_width=4 shares prefix KV and uses only ~1.3x memory (not 4x)
8. **Model offloading**: Cold model moved to CPU in < 2s, hot model loaded from CPU in < 2s
9. **No corruption**: All generated text remains coherent after eviction + swap-back cycles
10. **Integration test**: Full pipeline — scheduler admits sequences, allocator manages blocks, eviction triggers under pressure, sequences complete correctly

## Validation Commands

```bash
# Unit tests for block allocator
pytest tests/unit/test_block_allocator.py -v

# Stress test: rapid allocate/free cycles
python -m forge.research.stress_allocator --num-sequences 1000 --cycles 50 --output results/allocator_stress.json

# Eviction test: fill VRAM then add more sequences
python -m forge.research.test_eviction --num-sequences 200 --max-vram-gb 8 --output results/eviction.json

# Swap latency benchmark
python -m forge.research.bench_swap --seq-len 2048 --num-swaps 100 --output results/swap_latency.json

# Memory profiler visualization
python -m forge.research.gpu_profiler --model mistral-7b --num-sequences 50 --output results/memory_profile.html

# Fragmentation test under load
python -m forge.research.bench_fragmentation --duration 300 --arrival-rate 10 --output results/fragmentation.json

# Copy-on-write beam search test
python -m forge.research.test_cow --beam-width 4 --prompt "The meaning of life is" --output results/cow_test.json

# Integration test: full scheduler + memory manager
python -m forge.research.integration_test --workload mixed --duration 120 --output results/integration.json

# Correctness: verify outputs match non-eviction baseline
python -m forge.research.verify_correctness --mode eviction --reference results/baseline.json
```

## Technical Implementation Details

### Component 1: Block-Based KV-Cache Allocator (Day 1-2)

**File: `src/forge/research/block_allocator.py`**

```python
from dataclasses import dataclass
import torch

BLOCK_SIZE = 16  # tokens per block

@dataclass
class PhysicalBlock:
    block_id: int
    ref_count: int = 0
    device: str = "cuda"  # "cuda" or "cpu"
    data: torch.Tensor = None  # shape: [2, num_kv_heads, block_size, head_dim] (K and V)

class BlockAllocator:
    """Page-table style allocator for KV-cache blocks."""
    
    def __init__(self, num_gpu_blocks: int, num_cpu_blocks: int, num_kv_heads: int, head_dim: int):
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        
        # Pre-allocate all physical blocks on GPU
        self.gpu_pool = torch.zeros(
            num_gpu_blocks, 2, num_kv_heads, BLOCK_SIZE, head_dim,
            dtype=torch.float16, device="cuda"
        )
        # CPU pool for swapped-out blocks
        self.cpu_pool = torch.zeros(
            num_cpu_blocks, 2, num_kv_heads, BLOCK_SIZE, head_dim,
            dtype=torch.float16, device="cpu", pin_memory=True
        )
        
        self.free_gpu_blocks = list(range(num_gpu_blocks))
        self.free_cpu_blocks = list(range(num_cpu_blocks))
    
    def allocate_gpu(self) -> int:
        if not self.free_gpu_blocks:
            raise MemoryError("No free GPU blocks")
        return self.free_gpu_blocks.pop()
    
    def free_gpu(self, block_id: int):
        self.gpu_pool[block_id].zero_()
        self.free_gpu_blocks.append(block_id)
    
    def num_free_gpu_blocks(self) -> int:
        return len(self.free_gpu_blocks)

class SequenceBlockTable:
    """Logical-to-physical mapping for one sequence (like a page table)."""
    
    def __init__(self, seq_id: str):
        self.seq_id = seq_id
        self.block_table: list[int] = []  # logical index -> physical block_id
    
    def append_block(self, physical_block_id: int):
        self.block_table.append(physical_block_id)
    
    def get_physical_block(self, logical_idx: int) -> int:
        return self.block_table[logical_idx]
    
    def num_blocks(self) -> int:
        return len(self.block_table)
```

### Component 2: Eviction Manager (Day 2-3)

**File: `src/forge/research/eviction.py`**

```python
from collections import OrderedDict

class LRUEvictionPolicy:
    """Tracks access order, evicts least-recently-used sequences."""
    
    def __init__(self):
        self.access_order = OrderedDict()  # seq_id -> last_access_time
    
    def record_access(self, seq_id: str):
        self.access_order.move_to_end(seq_id)
    
    def get_eviction_candidate(self) -> str:
        # Least recently used = first in OrderedDict
        return next(iter(self.access_order))
    
    def remove(self, seq_id: str):
        del self.access_order[seq_id]

class PriorityEvictionPolicy:
    """Evicts based on priority score: (priority_level, -wait_time, -num_generated)."""
    
    def __init__(self):
        self.sequences = {}  # seq_id -> EvictionMetadata
    
    def get_eviction_candidate(self) -> str:
        # Lowest priority, longest waiting, fewest tokens generated
        return min(self.sequences, key=lambda s: self.sequences[s].eviction_score())

class SwapManager:
    """Handles async GPU↔CPU transfers for evicted sequences."""
    
    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.swap_stream = torch.cuda.Stream()  # Non-blocking transfers
    
    def swap_out(self, seq_block_table: SequenceBlockTable) -> list[int]:
        """Move sequence's blocks from GPU to CPU. Returns CPU block IDs."""
        cpu_blocks = []
        with torch.cuda.stream(self.swap_stream):
            for gpu_block_id in seq_block_table.block_table:
                cpu_block_id = self.allocator.allocate_cpu()
                # Async copy GPU -> CPU (pinned memory enables async)
                self.allocator.cpu_pool[cpu_block_id].copy_(
                    self.allocator.gpu_pool[gpu_block_id], non_blocking=True
                )
                self.allocator.free_gpu(gpu_block_id)
                cpu_blocks.append(cpu_block_id)
        self.swap_stream.synchronize()
        return cpu_blocks
    
    def swap_in(self, cpu_blocks: list[int]) -> list[int]:
        """Move sequence's blocks from CPU back to GPU. Returns GPU block IDs."""
        gpu_blocks = []
        with torch.cuda.stream(self.swap_stream):
            for cpu_block_id in cpu_blocks:
                gpu_block_id = self.allocator.allocate_gpu()
                self.allocator.gpu_pool[gpu_block_id].copy_(
                    self.allocator.cpu_pool[cpu_block_id], non_blocking=True
                )
                self.allocator.free_cpu(cpu_block_id)
                gpu_blocks.append(gpu_block_id)
        self.swap_stream.synchronize()
        return gpu_blocks
```

### Component 3: GPU Memory Profiler (Day 3-4)

**File: `src/forge/research/gpu_profiler.py`**

```python
class GPUMemoryProfiler:
    """Tracks and visualizes VRAM usage by category."""
    
    def __init__(self):
        self.snapshots = []
    
    def snapshot(self, label: str = ""):
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        self.snapshots.append({
            "label": label,
            "timestamp": time.time(),
            "allocated_mb": allocated / 1024**2,
            "reserved_mb": reserved / 1024**2,
            "breakdown": self._get_breakdown()
        })
    
    def _get_breakdown(self) -> dict:
        return {
            "model_weights_mb": self._measure_weights(),
            "kv_cache_mb": self._measure_kv_cache(),
            "activations_mb": self._measure_activations(),
            "fragmented_mb": self._measure_fragmentation(),
            "free_mb": self._measure_free(),
        }
    
    def render_html(self, output_path: str):
        """Generate interactive HTML chart with plotly showing memory over time."""
        # Stacked area chart: weights (constant), KV (grows), activations (spikes), free
        pass
    
    def render_waterfall(self, output_path: str):
        """Waterfall chart showing where each MB of VRAM is allocated."""
        pass
```

### Component 4: Copy-on-Write for Parallel Sampling (Day 4-5)

**File: `src/forge/research/cow_blocks.py`**

```python
class CopyOnWriteBlockManager:
    """Enables shared KV-cache blocks between forked sequences (beam search)."""
    
    def fork_sequence(self, parent_seq: SequenceBlockTable, child_seq_id: str) -> SequenceBlockTable:
        """Create a new sequence sharing parent's blocks (increment ref counts)."""
        child = SequenceBlockTable(child_seq_id)
        for block_id in parent_seq.block_table:
            self.ref_counts[block_id] += 1
            child.block_table.append(block_id)
        return child
    
    def write_block(self, seq: SequenceBlockTable, logical_idx: int, data: torch.Tensor):
        """Copy-on-write: if block is shared, copy before writing."""
        physical_id = seq.get_physical_block(logical_idx)
        if self.ref_counts[physical_id] > 1:
            # Block is shared — copy it first
            new_block = self.allocator.allocate_gpu()
            self.allocator.gpu_pool[new_block].copy_(self.allocator.gpu_pool[physical_id])
            self.ref_counts[physical_id] -= 1
            self.ref_counts[new_block] = 1
            seq.block_table[logical_idx] = new_block
            physical_id = new_block
        # Now safe to write
        self.allocator.gpu_pool[physical_id] = data
```

### Component 5: Defragmentation (Day 5-6)

**File: `src/forge/research/defrag.py`**

```python
class Defragmenter:
    """Compacts allocated blocks to eliminate fragmentation."""
    
    def should_defrag(self, allocator: BlockAllocator) -> bool:
        """Trigger defrag when fragmentation exceeds threshold."""
        total_free = allocator.num_free_gpu_blocks()
        max_contiguous = self._max_contiguous_free(allocator)
        fragmentation = 1.0 - (max_contiguous / total_free) if total_free > 0 else 0
        return fragmentation > 0.05  # 5% threshold
    
    def defragment(self, allocator: BlockAllocator, sequences: list[SequenceBlockTable]):
        """Move blocks to compact them. Update all block tables."""
        # Strategy: move allocated blocks to low addresses, leaving free space at end
        # Must be done carefully — sequences are reading these blocks
        # Use a temporary buffer or do it between scheduler steps
        
        new_mapping = self._compute_compaction_plan(allocator, sequences)
        for old_id, new_id in new_mapping.items():
            if old_id != new_id:
                allocator.gpu_pool[new_id].copy_(allocator.gpu_pool[old_id])
                self._update_block_tables(sequences, old_id, new_id)
```

### Component 6: Model Offloading (Day 6-7)

**File: `src/forge/research/model_offload.py`**

```python
class ModelOffloadManager:
    """Manages hot/cold model placement across GPU and CPU."""
    
    def __init__(self, gpu_budget_gb: float):
        self.gpu_budget = gpu_budget_gb * 1024**3
        self.loaded_models = {}  # model_id -> {"device": "cuda"/"cpu", "last_used": timestamp}
    
    def ensure_on_gpu(self, model_id: str):
        """Load model to GPU if not already there. Evict others if needed."""
        if self.loaded_models[model_id]["device"] == "cuda":
            return
        
        # Check if space available
        model_size = self._get_model_size(model_id)
        while self._gpu_used() + model_size > self.gpu_budget:
            self._evict_coldest_model()
        
        self._move_to_gpu(model_id)
    
    def _evict_coldest_model(self):
        """Move least-recently-used model from GPU to CPU."""
        coldest = min(
            (m for m, info in self.loaded_models.items() if info["device"] == "cuda"),
            key=lambda m: self.loaded_models[m]["last_used"]
        )
        self._move_to_cpu(coldest)
```

## If You Get Stuck

**Block allocator seems over-engineered**: Start with a simple free-list (just track which block IDs are free). Don't worry about contiguity at first — the whole point is that blocks DON'T need to be contiguous.

**Swap is slow**: Make sure you're using pinned (page-locked) CPU memory (`pin_memory=True`). Also use CUDA streams for async transfer — don't block the compute stream.

**Fragmentation math is confusing**: Think of it like a disk. If you have 100 free blocks but the largest contiguous chunk is only 5, fragmentation is 95%. With block-based allocation you rarely need contiguity, so fragmentation matters less — but the profiler should still track it.

**Copy-on-write is tricky**: Start without it. Implement beam search with full copies first. Then optimize: when forking, share blocks and only copy when writing. The ref_count tracks sharing.

**Not sure how much memory to allocate**: Formula for KV-cache per token: `2 * num_layers * num_kv_heads * head_dim * dtype_bytes`. For Mistral-7B in FP16: `2 * 32 * 8 * 128 * 2 = 131KB/token`. For 2048 tokens: ~262MB per sequence.

## Agent Handoff Template

```
I'm on Week 10 of Forge — implementing KV-cache memory management.
Spec: /Users/jmalviya/Documents/zz/dev/plan_00/forge/specs/phase2/week10-kv-cache-memory.md
Context: Weeks 8-9 complete — I have a transformer with KV-cache and a continuous batching scheduler.
I need: block-based allocator (page-table style), eviction (LRU + priority), GPU↔CPU swap, defragmentation, copy-on-write, memory profiler, model offloading.
Current state: [describe what's implemented so far]
Key challenge: [allocator design / swap performance / CoW correctness / profiler visualization]
```

## Out of Scope

- Custom CUDA kernels for attention (Week 16)
- Multi-GPU memory management (Phase 3)
- Tensor parallelism (Phase 3)
- Prefix caching (Week 11)
- Speculative decoding (Week 11)
- Quantized KV-cache storage (Week 12 touches this tangentially)
- Production-grade memory pool (this is learning infrastructure)
