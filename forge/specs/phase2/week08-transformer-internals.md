# Week 8: Transformer Internals
> Phase: 2 | Project: Forge | Estimated Duration: 7 days

## Context

Phase 1 built a working platform. Phase 2 goes deep on inference. Before optimizing, you must understand what you're optimizing. This week you implement a transformer forward pass from scratch — no HuggingFace `generate()`, just raw PyTorch. This gives you the mental model needed for Weeks 9-14.

**Prerequisites**: Phase 1 complete. Familiarity with PyTorch basics (tensors, autograd, nn.Module).

**Builds on**: Separate experimental code — this is a learning module, not integrated into the platform yet.

## Learning Goals

- [ ] Understand multi-head attention at the math + code level (Q, K, V matrices, scores, softmax, output)
- [ ] Understand KV-caching — WHY recomputing attention for past tokens is wasteful
- [ ] Understand Rotary Position Embeddings (RoPE) — how position information is encoded
- [ ] Understand RMSNorm — why it replaced LayerNorm in modern transformers
- [ ] Understand the autoregressive generation loop — predict one token at a time
- [ ] Understand memory vs compute bottleneck — when is attention memory-bound vs compute-bound

## Implementation Goals

- [ ] Implement Multi-Head Attention from scratch (raw matmuls, no nn.MultiheadAttention)
- [ ] Implement RoPE (Rotary Position Embedding) manually
- [ ] Implement RMSNorm
- [ ] Implement SwiGLU MLP (the activation used in Llama/Mistral)
- [ ] Assemble into a complete transformer block
- [ ] Implement naive autoregressive generation (without KV-cache)
- [ ] Implement generation WITH KV-cache — measure the speedup
- [ ] Load real model weights (Mistral-7B) into your implementation and verify correctness
- [ ] Profile the forward pass — identify bottlenecks

## Acceptance Criteria

1. **Attention correct**: Your attention output matches `torch.nn.functional.scaled_dot_product_attention` within 1e-5
2. **RoPE correct**: Your RoPE output matches HuggingFace Mistral's RoPE within 1e-5
3. **Full block correct**: Your transformer block produces same output as HuggingFace model's first layer
4. **Generation works**: Your implementation generates coherent text from a prompt
5. **KV-cache speedup**: Generation WITH cache is 3-10x faster than WITHOUT for sequences > 100 tokens
6. **Weights loaded**: Real Mistral-7B weights loaded and inference produces meaningful text
7. **Profile report**: PyTorch profiler output saved showing time breakdown per operation
8. **Memory report**: Document showing VRAM usage: weights + KV-cache + activations breakdown
9. **Notebook**: Jupyter notebook demonstrating all components with explanations
10. **Unit tests**: Each component (attention, RoPE, RMSNorm, SwiGLU) has a correctness test

## Validation Commands

```bash
# Run correctness tests
pytest tests/unit/test_transformer_components.py -v

# Run full generation test
python -m forge.research.generate --prompt "The capital of France is" --max-tokens 50 --mode naive
python -m forge.research.generate --prompt "The capital of France is" --max-tokens 50 --mode cached

# Benchmark naive vs cached
python -m forge.research.benchmark_cache --seq-lengths 64,128,256,512 --output results/cache_benchmark.json

# Profile
python -m forge.research.profile_forward --output results/profile_trace.json
# View with: chrome://tracing (load the JSON)

# Memory breakdown
python -m forge.research.memory_analysis --model mistral-7b --output results/memory_breakdown.json
```

## Technical Implementation Details

### Component 1: Multi-Head Attention (Day 1-2)

**File: `src/forge/research/attention.py`**

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        # Grouped Query Attention (GQA): fewer KV heads than Q heads
        # Mistral uses num_heads=32, num_kv_heads=8
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
    
    def forward(self, hidden_states, position_ids, kv_cache=None):
        # 1. Project to Q, K, V
        # 2. Reshape to (batch, heads, seq, head_dim)
        # 3. Apply RoPE to Q and K
        # 4. If kv_cache provided, append new K,V to cache
        # 5. Expand KV heads if using GQA (repeat for Q head groups)
        # 6. Compute attention: scores = Q @ K^T / sqrt(d_k)
        # 7. Apply causal mask
        # 8. Softmax
        # 9. Output = scores @ V
        # 10. Reshape and project with o_proj
        pass
```

Key insight: Grouped Query Attention (GQA) uses fewer KV heads than Q heads, saving memory. Each group of Q heads shares one KV head.

### Component 2: Rotary Position Embeddings (Day 2)

**File: `src/forge/research/rope.py`**

```python
def compute_rope_frequencies(dim: int, max_seq_len: int, base: float = 10000.0):
    # Compute the rotation frequencies for each dimension pair
    # freqs[i] = 1 / (base ^ (2i / dim))
    # Then compute the rotation matrix angles for each position
    pass

def apply_rope(x: torch.Tensor, freqs: torch.Tensor):
    # Split x into pairs of dimensions
    # Apply rotation: (x1*cos - x2*sin, x1*sin + x2*cos)
    # This encodes position without explicit position embeddings
    pass
```

Key insight: RoPE encodes RELATIVE position — the attention score between tokens at positions m and n depends only on (m-n), not on absolute positions. This allows generalization to longer sequences.

### Component 3: RMSNorm + SwiGLU (Day 2-3)

**File: `src/forge/research/layers.py`**

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        # RMS = sqrt(mean(x^2))
        # output = x / RMS * weight
        # Simpler than LayerNorm: no mean subtraction, no bias
        pass

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x):
        # SwiGLU = down_proj(silu(gate_proj(x)) * up_proj(x))
        # The "gating" provides better gradient flow than standard ReLU MLP
        pass
```

### Component 4: Complete Transformer Block + Model (Day 3-4)

**File: `src/forge/research/transformer.py`**

```python
class TransformerBlock(nn.Module):
    def __init__(self, config):
        self.attention = MultiHeadAttention(...)
        self.mlp = SwiGLU(...)
        self.input_layernorm = RMSNorm(...)
        self.post_attention_layernorm = RMSNorm(...)
    
    def forward(self, x, position_ids, kv_cache=None):
        # Pre-norm architecture (norm before attention, not after)
        residual = x
        x = self.input_layernorm(x)
        x = self.attention(x, position_ids, kv_cache)
        x = residual + x
        
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x

class MistralModel(nn.Module):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_ids, position_ids, kv_caches=None):
        x = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            x = layer(x, position_ids, cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
```

### Component 5: Generation with and without KV-Cache (Day 4-5)

**File: `src/forge/research/generate.py`**

```python
def generate_naive(model, input_ids, max_new_tokens):
    """Naive: recompute entire sequence every step."""
    for _ in range(max_new_tokens):
        logits = model(input_ids)  # Full forward on ALL tokens
        next_token = sample(logits[:, -1, :])
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids

def generate_cached(model, input_ids, max_new_tokens):
    """Cached: only compute new token, reuse past KV."""
    kv_caches = [KVCache() for _ in range(model.num_layers)]
    
    # Prefill: process entire prompt at once
    logits = model(input_ids, kv_caches=kv_caches)
    next_token = sample(logits[:, -1, :])
    
    # Decode: one token at a time, using cached K,V
    for _ in range(max_new_tokens - 1):
        logits = model(next_token, kv_caches=kv_caches)  # Only 1 token!
        next_token = sample(logits[:, -1, :])
    
    return all_generated_tokens
```

Key insight: Without cache, generating N tokens requires O(N^2) computation (reprocessing everything each step). With cache, it's O(N) — each step only processes 1 new token.

### Component 6: Loading Real Weights (Day 5-6)

**File: `src/forge/research/weight_loader.py`**

Map HuggingFace weight names to your implementation:
```python
WEIGHT_MAP = {
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.attention.q_proj.weight",
    "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.attention.k_proj.weight",
    # ... etc
}
```

Load with safetensors for speed. Verify by comparing output on same input.

### Component 7: Profiling + Memory Analysis (Day 6-7)

Use `torch.profiler`:
```python
with torch.profiler.profile(activities=[ProfilerActivity.CUDA]) as prof:
    model(input_ids)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

Memory breakdown document:
- Model weights: num_params × bytes_per_param (e.g., 7B × 2 bytes = 14GB for FP16)
- KV-cache: 2 × num_layers × num_kv_heads × head_dim × seq_len × batch_size × bytes
- Activations: hidden_size × seq_len × batch_size × bytes (temporary, freed after each layer)
- Total: sum of above

## If You Get Stuck

**Weights don't match**: Most common issue is transposed dimensions or GQA head grouping. Use `model.state_dict().keys()` to see exact weight names and shapes.

**CUDA OOM loading 7B**: Load in float16 (not float32). Use `torch.float16` dtype. If still OOM, try a smaller model (3B) first to verify correctness, then scale up.

**Attention outputs don't match**: Check causal mask (lower triangular), check sqrt(d_k) scaling, check GQA head expansion.

**Simplified fallback**: If loading real weights is too complex, implement with random weights and just verify shapes and attention patterns are correct. The learning is in the implementation, not the specific outputs.

## Agent Handoff Template

```
I'm on Week 8 of Forge — implementing transformer internals from scratch.
Spec: /Users/jmalviya/Documents/zz/dev/plan_00/forge/specs/phase2/week08-transformer-internals.md
This is a research/learning module separate from the main platform.
I need: from-scratch implementation of multi-head attention (with GQA), RoPE, RMSNorm, SwiGLU, and autoregressive generation with KV-caching. Must load real Mistral-7B weights.
Focus on: [specific component - attention/RoPE/generation/weight loading]
```

## Out of Scope

- Integrating this into the platform (it's a learning exercise)
- FlashAttention (understand the concept, don't implement it)
- Training (forward pass only)
- Batched inference (single sequence for now — batching comes Week 9)
- Custom CUDA kernels (Week 16)
