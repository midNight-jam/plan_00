# Week 9: Continuous Batching
> Phase: 2 | Project: Forge | Estimated Duration: 7 days

## Context

Week 8 gave you the mental model of how a transformer generates tokens. Now you build the scheduling system that makes serving EFFICIENT. The key insight: static batching wastes enormous GPU cycles. Continuous batching keeps the GPU saturated by dynamically adding/removing sequences each iteration.

**Prerequisites**: Week 8 complete — understand forward pass, KV-cache, prefill vs decode.

**Builds on**: Uses your understanding from Week 8 to build a scheduler that manages multiple concurrent requests.

## Learning Goals

- [ ] Understand static vs continuous batching — WHY static wastes GPU
- [ ] Understand prefill vs decode phases — different compute characteristics
- [ ] Understand iteration-level scheduling — batch composition changes every step
- [ ] Understand padding waste — in static batching, short sequences pad to max length
- [ ] Understand GPU utilization — how to keep the GPU busy even with variable-length requests

## Implementation Goals

- [ ] Implement static batching baseline (to measure what you're improving)
- [ ] Implement continuous batching scheduler (add/remove sequences per iteration)
- [ ] Implement waiting queue with admission control (memory-based)
- [ ] Implement padding-free batching (concatenated sequences with position offsets)
- [ ] Handle prefill vs decode in same batch (chunked prefill)
- [ ] Build benchmark harness comparing static vs continuous vs vLLM
- [ ] Generate throughput/latency charts

## Acceptance Criteria

1. **Static baseline works**: Can batch 4 requests and generate to completion (with padding)
2. **Continuous scheduler works**: New requests enter mid-batch, completed ones leave immediately
3. **No padding**: Sequences of different lengths processed without padding tokens
4. **Admission control**: Scheduler rejects new sequences if KV-cache memory would be exceeded
5. **Throughput improvement**: Continuous batching achieves 2-4x throughput over static on mixed workloads
6. **Latency improvement**: Time-to-first-token is lower with continuous (no waiting for batch to fill)
7. **Benchmark chart**: Published chart showing throughput vs concurrent requests for both methods
8. **vLLM comparison**: Your continuous batching within 50% of vLLM throughput (vLLM has more optimizations)
9. **Variable workload test**: Mix of short (10 token) and long (500 token) requests handled efficiently
10. **Correct outputs**: Generated text is coherent (scheduler doesn't corrupt sequence state)

## Validation Commands

```bash
# Run static batching benchmark
python -m forge.research.bench_batching --mode static --num-requests 50 --output results/static.json

# Run continuous batching benchmark
python -m forge.research.bench_batching --mode continuous --num-requests 50 --output results/continuous.json

# Run vLLM benchmark (reference)
python -m forge.research.bench_batching --mode vllm --num-requests 50 --output results/vllm.json

# Compare results
python -m forge.research.compare_benchmarks --inputs results/static.json results/continuous.json results/vllm.json --output results/comparison_chart.png

# Variable workload test
python -m forge.research.bench_batching --mode continuous --workload mixed --output results/mixed.json

# Correctness test
pytest tests/unit/test_scheduler.py -v
```

## Technical Implementation Details

### Component 1: Static Batching Baseline (Day 1)

**File: `src/forge/research/static_batch.py`**

```python
class StaticBatchEngine:
    """Naive approach: collect requests, pad to same length, process as one batch."""
    
    def __init__(self, model, batch_size=4):
        self.model = model
        self.batch_size = batch_size
        self.pending = []
    
    def add_request(self, prompt_tokens, max_new_tokens):
        self.pending.append(Request(prompt_tokens, max_new_tokens))
        if len(self.pending) >= self.batch_size:
            return self._process_batch()
    
    def _process_batch(self):
        # Pad all sequences to max length in batch
        max_len = max(len(r.tokens) for r in self.pending)
        padded = [pad(r.tokens, max_len) for r in self.pending]
        
        # Generate: ALL sequences must reach max_new_tokens
        max_gen = max(r.max_new_tokens for r in self.pending)
        for step in range(max_gen):
            logits = self.model(padded)  # Even finished sequences are processed!
            # Sample next tokens, append
        
        # All results returned at once (long tail waits for longest)
        return results
```

Problems to observe and document:
- Short sequences waste compute (processed but already done)
- All requests return together (worst-case latency for all)
- Padding tokens waste memory and compute

### Component 2: Continuous Batching Scheduler (Day 2-4)

**File: `src/forge/research/continuous_batch.py`**

```python
class SequenceState(Enum):
    WAITING = "waiting"     # In queue, not yet scheduled
    PREFILLING = "prefilling"  # Processing initial prompt
    DECODING = "decoding"      # Generating tokens one at a time
    FINISHED = "finished"      # Hit stop token or max_tokens

class Sequence:
    id: str
    tokens: list[int]
    kv_cache: KVCache
    state: SequenceState
    generated_tokens: int
    max_new_tokens: int
    
class ContinuousBatchScheduler:
    def __init__(self, model, max_batch_size, max_kv_blocks):
        self.model = model
        self.waiting_queue = deque()
        self.running_batch: list[Sequence] = []
        self.max_batch_size = max_batch_size
        self.kv_block_manager = BlockManager(max_kv_blocks)
    
    def add_request(self, prompt_tokens, max_new_tokens):
        seq = Sequence(tokens=prompt_tokens, max_new_tokens=max_new_tokens)
        self.waiting_queue.append(seq)
    
    def schedule_step(self) -> list[Sequence]:
        """Called every iteration. Decides what to process this step."""
        # 1. Remove finished sequences from running batch (FREE their KV memory)
        self.running_batch = [s for s in self.running_batch if s.state != FINISHED]
        
        # 2. Try to admit new sequences from waiting queue
        while self.waiting_queue and len(self.running_batch) < self.max_batch_size:
            candidate = self.waiting_queue[0]
            if self.kv_block_manager.can_allocate(candidate.estimated_blocks):
                seq = self.waiting_queue.popleft()
                self.kv_block_manager.allocate(seq)
                seq.state = PREFILLING
                self.running_batch.append(seq)
            else:
                break  # No memory for more sequences
        
        return self.running_batch
    
    def step(self):
        """Execute one forward pass for the current batch."""
        batch = self.schedule_step()
        if not batch:
            return
        
        # Separate prefill and decode sequences
        # Prefill: process all prompt tokens at once
        # Decode: process single new token
        # (In advanced version: interleave them with chunked prefill)
        
        for seq in batch:
            if seq.state == PREFILLING:
                logits = self.model.prefill(seq.tokens, seq.kv_cache)
                seq.state = DECODING
            elif seq.state == DECODING:
                logits = self.model.decode_one(seq.last_token, seq.kv_cache)
            
            next_token = sample(logits)
            seq.tokens.append(next_token)
            seq.generated_tokens += 1
            
            if next_token == EOS or seq.generated_tokens >= seq.max_new_tokens:
                seq.state = FINISHED
                self.kv_block_manager.free(seq)
```

### Component 3: Padding-Free Batching (Day 4-5)

**File: `src/forge/research/packed_batch.py`**

Instead of padding to max length, concatenate all sequences and use position offsets:

```python
def prepare_packed_batch(sequences: list[Sequence]):
    # Concatenate all tokens: [seq1_tokens..., seq2_tokens..., seq3_tokens...]
    all_tokens = torch.cat([s.current_tokens for s in sequences])
    
    # Position IDs: each sequence starts at 0
    # [0,1,2,3, 0,1,2, 0,1,2,3,4,5]
    position_ids = torch.cat([torch.arange(len(s.current_tokens)) for s in sequences])
    
    # Attention mask: block diagonal (each sequence only attends to itself)
    # Use flash attention's varlen API for this
    cu_seqlens = torch.tensor([0] + [len(s.current_tokens) for s in sequences]).cumsum(0)
    
    return all_tokens, position_ids, cu_seqlens
```

This eliminates ALL padding waste — the GPU processes only real tokens.

### Component 4: Benchmark Harness (Day 5-6)

**File: `src/forge/research/bench_batching.py`**

Benchmark parameters:
- Number of requests: 50-200
- Input length distribution: uniform(10, 500) tokens
- Output length: uniform(20, 200) tokens
- Arrival pattern: Poisson process (requests arrive randomly)

Metrics to measure:
- **Throughput**: total tokens generated / total time (tokens/sec)
- **Time-to-first-token (TTFT)**: time from request arrival to first generated token
- **Inter-token latency**: average time between consecutive tokens
- **GPU utilization**: average % during the benchmark
- **Total time**: wall clock to complete all requests

### Component 5: Results Visualization (Day 6-7)

Generate comparison charts (matplotlib):
- Bar chart: throughput comparison (static vs continuous vs vLLM)
- Line chart: latency vs concurrent requests
- Box plot: TTFT distribution for each method
- Time series: GPU utilization over time during benchmark

Write a brief analysis document explaining WHY continuous batching wins.

## If You Get Stuck

**Batch construction is complex**: Start without packed batching. Just handle sequences one at a time in a loop, but with proper add/remove. Then add real batching.

**KV-cache management is confusing**: Start with unlimited memory (just pre-allocate for all sequences). Add the block manager / admission control after the basic scheduler works.

**Throughput worse than expected**: Check if you're accidentally recomputing KV-cache (should only compute new token's KV). Check if GPU is actually being utilized (not waiting on CPU scheduling logic).

**Can't match vLLM**: Expected — vLLM has years of optimization. Getting within 2-3x is a good result for a from-scratch implementation. The goal is understanding, not beating vLLM.

## Agent Handoff Template

```
I'm on Week 9 of Forge — implementing continuous batching.
Spec: /Users/jmalviya/Documents/zz/dev/plan_00/forge/specs/phase2/week09-continuous-batching.md
Context: Week 8 complete — I have a from-scratch transformer implementation with KV-cache.
I need: static batching baseline, continuous batching scheduler with admission control, padding-free batching, benchmark harness comparing both + vLLM.
Key challenge: [scheduler logic / packed attention / benchmark setup]
```

## Out of Scope

- Preemption/swapping (Week 10-11)
- Speculative decoding (Week 11)
- Prefix caching (Week 11)
- Quantization (Week 12)
- Integration with the platform (this is research code for now)
