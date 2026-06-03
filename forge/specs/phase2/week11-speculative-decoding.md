# Week 11: Speculative Decoding and Prefix Caching
> Phase: 2 | Project: Forge | Estimated Duration: 7 days

## Context

Weeks 9-10 built the scheduling and memory systems. Now you tackle two techniques that dramatically improve user-perceived performance. Speculative decoding reduces inter-token latency by having a small "draft" model guess ahead, then verifying in parallel. Prefix caching eliminates redundant prefill computation for repeated system prompts (the same system prompt gets sent with every request to an API).

These are the techniques that separate good inference engines from great ones. vLLM, TensorRT-LLM, and SGLang all implement variants of these.

**Prerequisites**: Weeks 9-10 complete — continuous batching scheduler with block-based KV-cache and memory management.

**Builds on**: Uses the scheduler (Week 9) and memory manager (Week 10) to implement advanced inference optimizations.

## Learning Goals

- [ ] Understand speculative decoding theory — why verifying N draft tokens in one pass is faster than generating N tokens sequentially
- [ ] Understand acceptance/rejection sampling — the math that guarantees output distribution matches target model
- [ ] Understand the draft-target model relationship — draft model must be much faster but reasonably accurate
- [ ] Understand prefix caching — reusing KV-cache for common prefixes across requests
- [ ] Understand radix tree / trie structure for prefix matching
- [ ] Understand preemption strategies — swap vs recompute tradeoffs
- [ ] Understand SLO-aware scheduling — meeting latency targets for different request priorities

## Implementation Goals

- [ ] Implement speculative decoding with configurable draft model and speculation length
- [ ] Implement acceptance/rejection sampling (modified rejection sampling from Leviathan et al.)
- [ ] Implement adaptive speculation length (increase when acceptance is high, decrease when low)
- [ ] Implement prefix cache with radix tree for O(prefix_len) lookup
- [ ] Implement cache eviction for prefix cache (LRU on prefix entries)
- [ ] Implement preemption: swap-based (move KV to CPU) and recompute-based (discard KV, recompute later)
- [ ] Implement SLO-aware scheduling (priority queues based on deadline proximity)
- [ ] Benchmark speculative decoding speedup vs vanilla autoregressive
- [ ] Benchmark prefix cache TTFT improvement for repeated system prompts
- [ ] Integrate all optimizations with the existing scheduler

## Acceptance Criteria

1. **Speculative decoding works**: Draft model generates N tokens, target model verifies in one forward pass, accepted tokens are emitted
2. **Correct distribution**: Output token distribution matches target-model-only generation (verified statistically over 1000 samples)
3. **Speedup measured**: Speculative decoding achieves 1.5-2.5x speedup over vanilla decoding on Mistral-7B (with a 1B draft model)
4. **Adaptive speculation**: Speculation length adjusts dynamically — longer on easy text, shorter on hard/unpredictable text
5. **Prefix cache reduces TTFT by 50%+**: For repeated system prompts (e.g., 500-token system prompt), second+ requests skip prefill entirely
6. **Radix tree works**: Prefix matching finds longest cached prefix in O(prefix_len) time
7. **Preemption works without data loss**: Preempted sequences resume correctly (same output as non-preempted)
8. **Swap vs recompute decision**: System chooses swap when KV is large (saves compute), recompute when KV is small (saves memory)
9. **SLO scheduling**: High-priority requests meet latency target even under load (p99 within SLO for priority traffic)
10. **Integration test**: All optimizations work together — speculative decoding + prefix cache + preemption in one serving session

## Validation Commands

```bash
# Speculative decoding correctness test
pytest tests/unit/test_speculative.py -v

# Distribution match test (KL divergence should be ~0)
python -m forge.research.verify_speculative_distribution --num-samples 1000 --output results/spec_distribution.json

# Speculative decoding benchmark
python -m forge.research.bench_speculative --draft-model "TinyLlama/TinyLlama-1.1B" --target-model "mistralai/Mistral-7B-v0.1" --spec-length 5 --num-requests 50 --output results/speculative_bench.json

# Adaptive speculation analysis
python -m forge.research.bench_speculative --adaptive --output results/adaptive_spec.json

# Prefix cache benchmark
python -m forge.research.bench_prefix_cache --system-prompt-len 500 --num-requests 100 --output results/prefix_cache.json

# Preemption correctness test
python -m forge.research.test_preemption --mode swap --output results/preemption_swap.json
python -m forge.research.test_preemption --mode recompute --output results/preemption_recompute.json

# SLO scheduling test
python -m forge.research.bench_slo --high-priority-slo-ms 200 --load-factor 0.8 --output results/slo_test.json

# Integration: all optimizations together
python -m forge.research.integration_test --features speculative,prefix_cache,preemption --output results/week11_integration.json

# Generate comparison charts
python -m forge.research.plot_week11 --input-dir results/ --output results/week11_charts.html
```

## Technical Implementation Details

### Component 1: Speculative Decoding Engine (Day 1-3)

**File: `src/forge/research/speculative.py`**

```python
import torch
from typing import Tuple

class SpeculativeDecoder:
    """
    Speculative decoding: draft model proposes, target model verifies.
    Based on: "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023)
    """
    
    def __init__(self, draft_model, target_model, spec_length: int = 5):
        self.draft_model = draft_model      # Small, fast model (e.g., 1B params)
        self.target_model = target_model    # Large, accurate model (e.g., 7B params)
        self.spec_length = spec_length      # Number of tokens to speculate
    
    def generate_step(self, input_ids: torch.Tensor, kv_caches) -> Tuple[list[int], dict]:
        """One speculative decoding step: draft N tokens, verify all at once."""
        stats = {"drafted": 0, "accepted": 0}
        
        # Phase 1: Draft model generates N tokens autoregressively (cheap)
        draft_tokens = []
        draft_probs = []
        draft_kv = kv_caches["draft"]
        
        current = input_ids
        for _ in range(self.spec_length):
            logits = self.draft_model(current, kv_cache=draft_kv)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            token = torch.multinomial(probs, 1)
            draft_tokens.append(token)
            draft_probs.append(probs)
            current = token
        
        stats["drafted"] = len(draft_tokens)
        
        # Phase 2: Target model verifies ALL draft tokens in ONE forward pass
        # Feed [input_ids, draft_token_0, draft_token_1, ..., draft_token_N-1]
        verify_input = torch.cat([input_ids] + draft_tokens, dim=-1)
        target_logits = self.target_model(verify_input, kv_cache=kv_caches["target"])
        
        # Phase 3: Accept/reject each draft token
        accepted_tokens = []
        for i, (draft_token, draft_prob) in enumerate(zip(draft_tokens, draft_probs)):
            target_prob = torch.softmax(target_logits[:, -(self.spec_length - i), :], dim=-1)
            
            if self._accept(draft_token, draft_prob, target_prob):
                accepted_tokens.append(draft_token)
                stats["accepted"] += 1
            else:
                # Reject: sample from adjusted distribution
                adjusted = self._adjusted_distribution(draft_prob, target_prob)
                resampled = torch.multinomial(adjusted, 1)
                accepted_tokens.append(resampled)
                break  # Stop accepting after first rejection
        
        # If all accepted, sample one more from target (bonus token)
        if len(accepted_tokens) == self.spec_length:
            bonus_logits = target_logits[:, -1, :]
            bonus_token = torch.multinomial(torch.softmax(bonus_logits, dim=-1), 1)
            accepted_tokens.append(bonus_token)
        
        return accepted_tokens, stats
    
    def _accept(self, draft_token, draft_prob, target_prob) -> bool:
        """Modified rejection sampling: accept with prob min(1, target_p/draft_p)."""
        token_id = draft_token.item()
        p_draft = draft_prob[0, token_id].item()
        p_target = target_prob[0, token_id].item()
        
        acceptance_prob = min(1.0, p_target / p_draft)
        return torch.rand(1).item() < acceptance_prob
    
    def _adjusted_distribution(self, draft_prob, target_prob) -> torch.Tensor:
        """When rejecting, sample from norm(max(0, target_p - draft_p))."""
        diff = target_prob - draft_prob
        adjusted = torch.clamp(diff, min=0)
        return adjusted / adjusted.sum()

class AdaptiveSpeculativeDecoder(SpeculativeDecoder):
    """Dynamically adjusts speculation length based on acceptance rate."""
    
    def __init__(self, draft_model, target_model, min_spec=2, max_spec=8):
        super().__init__(draft_model, target_model, spec_length=4)
        self.min_spec = min_spec
        self.max_spec = max_spec
        self.acceptance_history = []
    
    def update_spec_length(self, stats: dict):
        rate = stats["accepted"] / stats["drafted"] if stats["drafted"] > 0 else 0
        self.acceptance_history.append(rate)
        
        # Rolling average of last 10 steps
        recent_rate = sum(self.acceptance_history[-10:]) / min(10, len(self.acceptance_history))
        
        if recent_rate > 0.8:
            self.spec_length = min(self.max_spec, self.spec_length + 1)
        elif recent_rate < 0.4:
            self.spec_length = max(self.min_spec, self.spec_length - 1)
```

### Component 2: Prefix Cache with Radix Tree (Day 3-5)

**File: `src/forge/research/prefix_cache.py`**

```python
class RadixTreeNode:
    """Node in the prefix radix tree. Each edge represents a token sequence."""
    
    def __init__(self):
        self.children: dict[int, RadixTreeNode] = {}  # token_id -> child node
        self.kv_block_ids: list[int] = []  # physical block IDs storing KV for this prefix
        self.ref_count: int = 0
        self.last_access: float = 0.0

class PrefixCache:
    """Cache KV-cache blocks for common prefixes (system prompts, few-shot examples)."""
    
    def __init__(self, block_allocator, max_cache_blocks: int):
        self.root = RadixTreeNode()
        self.block_allocator = block_allocator
        self.max_cache_blocks = max_cache_blocks
        self.used_blocks = 0
    
    def match_prefix(self, token_ids: list[int]) -> Tuple[int, list[int]]:
        """Find longest cached prefix. Returns (matched_length, kv_block_ids)."""
        node = self.root
        matched_blocks = []
        matched_tokens = 0
        
        for token_id in token_ids:
            if token_id in node.children:
                node = node.children[token_id]
                matched_tokens += 1
                if node.kv_block_ids:
                    matched_blocks.extend(node.kv_block_ids)
            else:
                break
        
        node.last_access = time.time()
        node.ref_count += 1
        return matched_tokens, matched_blocks
    
    def insert_prefix(self, token_ids: list[int], kv_block_ids: list[int]):
        """Cache KV blocks for a token prefix."""
        if self.used_blocks + len(kv_block_ids) > self.max_cache_blocks:
            self._evict_lru()
        
        node = self.root
        for i, token_id in enumerate(token_ids):
            if token_id not in node.children:
                node.children[token_id] = RadixTreeNode()
            node = node.children[token_id]
        
        node.kv_block_ids = kv_block_ids
        self.used_blocks += len(kv_block_ids)
    
    def _evict_lru(self):
        """Remove least-recently-used prefix entries until space is available."""
        # Walk tree, find node with oldest last_access, remove its blocks
        pass
```

### Component 3: Preemption Manager (Day 5-6)

**File: `src/forge/research/preemption.py`**

```python
class PreemptionPolicy:
    SWAP = "swap"       # Move KV to CPU (saves recompute, costs memory + transfer time)
    RECOMPUTE = "recompute"  # Discard KV entirely (saves memory, costs recompute time)

class PreemptionManager:
    """Decides when and how to preempt sequences to free resources."""
    
    def __init__(self, swap_manager, scheduler):
        self.swap_manager = swap_manager
        self.scheduler = scheduler
    
    def should_preempt(self) -> bool:
        """Preempt when a high-priority request is waiting and no memory available."""
        has_high_priority_waiting = any(
            r.priority == "high" for r in self.scheduler.waiting_queue
        )
        no_free_blocks = self.scheduler.block_allocator.num_free_gpu_blocks() < 10
        return has_high_priority_waiting and no_free_blocks
    
    def select_victim(self) -> str:
        """Choose which running sequence to preempt."""
        # Prefer: lowest priority, most tokens already generated (closer to done = more KV to save)
        running = self.scheduler.running_batch
        return min(running, key=lambda s: (s.priority_score, -s.generated_tokens))
    
    def choose_policy(self, victim) -> str:
        """Swap if KV is large (expensive to recompute), recompute if small."""
        kv_blocks = victim.block_table.num_blocks()
        recompute_cost = victim.prompt_len  # tokens to re-prefill
        swap_cost = kv_blocks * 0.5  # estimated ms per block transfer
        
        if recompute_cost > swap_cost * 2:
            return PreemptionPolicy.SWAP
        return PreemptionPolicy.RECOMPUTE
    
    def preempt(self, seq_id: str, policy: str):
        """Execute preemption."""
        if policy == PreemptionPolicy.SWAP:
            cpu_blocks = self.swap_manager.swap_out(self.scheduler.get_block_table(seq_id))
            self.scheduler.mark_swapped(seq_id, cpu_blocks)
        else:
            self.scheduler.mark_for_recompute(seq_id)
            self.scheduler.free_blocks(seq_id)
```

### Component 4: SLO-Aware Scheduler (Day 6-7)

**File: `src/forge/research/slo_scheduler.py`**

```python
import heapq
from dataclasses import dataclass

@dataclass
class SLOTarget:
    ttft_ms: float       # Time-to-first-token target
    tpot_ms: float       # Time-per-output-token target
    priority: int        # 0=highest, lower is higher priority

class SLOAwareScheduler:
    """Scheduler that respects latency SLOs for different priority levels."""
    
    def __init__(self, base_scheduler):
        self.base_scheduler = base_scheduler
        self.priority_queues = {0: [], 1: [], 2: []}  # priority -> heap by deadline
    
    def add_request(self, request, slo: SLOTarget):
        request.slo = slo
        request.deadline = time.time() + (slo.ttft_ms / 1000.0)
        heapq.heappush(self.priority_queues[slo.priority], (request.deadline, request))
    
    def schedule_step(self):
        """Priority-first, deadline-aware scheduling."""
        # 1. Always serve priority 0 (preempt if needed)
        # 2. Then priority 1 if capacity
        # 3. Then priority 2 (best-effort)
        
        batch = []
        for priority in sorted(self.priority_queues.keys()):
            while self.priority_queues[priority] and self._has_capacity():
                deadline, request = heapq.heappop(self.priority_queues[priority])
                if self._approaching_deadline(request):
                    # Urgent — preempt lower priority if needed
                    self._ensure_admission(request)
                batch.append(request)
        
        return batch
```

## If You Get Stuck

**Speculative decoding distribution doesn't match**: The acceptance criterion is subtle. Make sure you're using `min(1, p_target/p_draft)` not just comparing tokens. The adjusted distribution for rejection must be `norm(max(0, p_target - p_draft))`.

**Prefix cache lookup is slow**: The radix tree should match at the granularity of blocks (16 tokens), not individual tokens. Build the tree with block-aligned boundaries.

**Preemption corrupts outputs**: After swap-in, verify the KV-cache is byte-identical to before swap-out. Use a checksum in debug mode. Recompute-based preemption must re-run prefill with the SAME random seed if any sampling was involved.

**Can't get 50% TTFT improvement**: Make sure you're measuring correctly — TTFT includes queue time. The prefix cache only eliminates prefill compute. If prefill is already fast (short prompt), the improvement is smaller. Use a 500+ token system prompt for clear benefit.

**SLO scheduling too complex**: Start without preemption. Just use priority queues — high priority requests go to front of waiting queue. Add preemption only after basic priority works.

## Agent Handoff Template

```
I'm on Week 11 of Forge — implementing speculative decoding and prefix caching.
Spec: /Users/jmalviya/Documents/zz/dev/plan_00/forge/specs/phase2/week11-speculative-decoding.md
Context: Weeks 8-10 complete — I have transformer internals, continuous batching, and block-based KV-cache memory management.
I need: speculative decoding (draft+verify), prefix caching (radix tree), preemption (swap vs recompute), SLO-aware scheduling.
Current state: [describe what's implemented so far]
Key challenge: [acceptance sampling math / prefix tree design / preemption correctness / SLO integration]
```

## Out of Scope

- Medusa/EAGLE-style self-speculative decoding (single model with multiple heads)
- Tree-based speculative decoding (multiple draft paths)
- Disaggregated prefill/decode (separate GPU pools)
- Multi-GPU scheduling (Phase 3)
- Production alerting on SLO violations (Week 13)
- Quantized draft models (Week 12)
