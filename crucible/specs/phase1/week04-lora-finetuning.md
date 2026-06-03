# Week 4: LoRA and QLoRA Fine-Tuning

## Context

**Where it fits:** Phase 1 (Foundations), Week 4 of 7
**Prerequisites:** Week 1 (training loops), Week 2 (mixed precision, memory optimization), Week 3 (data pipelines), linear algebra basics (matrix rank, decomposition)
**What it builds on:** You can now train from scratch and manage memory — this week introduces parameter-efficient fine-tuning to adapt large pre-trained models without full retraining
**What comes next:** Week 5 (Instruction Tuning) uses QLoRA as the default fine-tuning method

LoRA is the single most important technique for practical LLM fine-tuning. It lets you adapt a 7B model on a single GPU by training <1% of parameters. Understanding it deeply — mathematically, implementationally, and empirically — is non-negotiable for an alignment engineer.

---

## Learning Goals

- [ ] Explain LoRA mathematically: W_new = W_frozen + B×A where B∈ℝ^(d×r), A∈ℝ^(r×k), r << min(d,k)
- [ ] Articulate why low-rank updates work: weight changes during fine-tuning have low intrinsic rank
- [ ] Explain the role of alpha (scaling factor): ΔW = (alpha/r) × B×A
- [ ] Understand which layers benefit most from LoRA (attention projections vs FFN)
- [ ] Explain QLoRA: NF4 quantization of base model, LoRA adapters in BF16 on top
- [ ] Articulate the NF4 data type: normalized float4, optimal for normally-distributed weights
- [ ] Explain double quantization in QLoRA (quantize the quantization constants)
- [ ] Compare memory requirements: full fine-tune vs LoRA vs QLoRA for 7B model
- [ ] Understand adapter merging: how to fold LoRA weights back into base model for inference

---

## Implementation Goals

- [ ] Implement LoRA layer from scratch (Linear → LoRALinear replacement)
- [ ] Inject LoRA adapters into a pre-trained model manually (no PEFT library)
- [ ] Verify: only adapter parameters have `requires_grad=True`, base model is frozen
- [ ] Use PEFT library: `LoraConfig`, `get_peft_model`, training, save/load adapters
- [ ] QLoRA setup: load model in 4-bit with `BitsAndBytesConfig`, apply LoRA
- [ ] Hyperparameter sweep: rank (4, 8, 16, 32, 64), alpha, target modules
- [ ] Merge adapters back into base model and verify identical inference outputs
- [ ] Fine-tune Mistral-7B with QLoRA on a code generation or instruction-following task
- [ ] Compare: full fine-tune (1B model) vs LoRA vs QLoRA on same task and dataset

---

## Acceptance Criteria

1. Custom LoRA implementation produces identical forward pass outputs as PEFT library (within 1e-5 tolerance)
2. Only LoRA parameters (A, B matrices) have gradients; base model parameters are confirmed frozen (grad is None)
3. Memory usage with QLoRA on 7B model is <10GB (fits in RTX 5080 16GB with room for activations)
4. Hyperparameter sweep produces a clear chart showing quality vs rank (diminishing returns after r=16 for the test task)
5. Merged model produces bit-identical outputs to adapter model on 100 test inputs
6. Training loss converges within 1 epoch on a 10K example fine-tuning dataset
7. QLoRA fine-tuned Mistral-7B shows measurable improvement on the target task vs base model (>10% on evaluation metric)
8. Adapter checkpoint is <100MB for rank-16 LoRA on 7B model (vs 14GB for full model)
9. Comparison table shows LoRA achieves >90% of full fine-tune quality with <5% of trainable parameters
10. Training throughput with QLoRA is >100 tokens/sec on RTX 5080

---

## Validation Commands

```bash
# Test custom LoRA implementation
python -m pytest tests/test_lora.py -v

# Compare custom LoRA vs PEFT library output
python scripts/compare_lora_impls.py --tolerance 1e-5

# Verify base model is frozen
python scripts/verify_frozen.py --model mistralai/Mistral-7B-v0.1

# Memory usage comparison
python scripts/memory_comparison.py --model mistralai/Mistral-7B-v0.1

# Hyperparameter sweep
python scripts/hp_sweep.py --ranks 4,8,16,32,64 --dataset code_alpaca

# Train QLoRA on Mistral-7B
python scripts/train_qlora.py \
  --model mistralai/Mistral-7B-v0.1 \
  --dataset code_alpaca \
  --rank 16 --alpha 32 \
  --epochs 1 --output adapters/mistral-code

# Merge and verify
python scripts/merge_adapter.py --base mistralai/Mistral-7B-v0.1 --adapter adapters/mistral-code
python scripts/verify_merge.py --merged merged_model/ --adapter adapters/mistral-code --num-tests 100

# Evaluate improvement
python scripts/evaluate.py --model merged_model/ --benchmark humaneval --baseline mistralai/Mistral-7B-v0.1

# Check adapter size
du -sh adapters/mistral-code/
```

---

## Technical Implementation Details

### Project Structure

```
week04-lora-finetuning/
├── src/
│   ├── __init__.py
│   ├── lora_layer.py          # LoRA from scratch
│   ├── lora_injection.py      # Inject adapters into existing model
│   ├── qlora_setup.py         # BitsAndBytes 4-bit + LoRA
│   ├── peft_wrapper.py        # PEFT library usage
│   ├── merge.py               # Adapter merging
│   └── train.py               # Fine-tuning loop (reuses Week 1/2)
├── scripts/
│   ├── train_qlora.py
│   ├── hp_sweep.py
│   ├── compare_lora_impls.py
│   ├── merge_adapter.py
│   ├── verify_merge.py
│   └── evaluate.py
├── tests/
│   ├── test_lora.py
│   ├── test_injection.py
│   └── test_merge.py
└── configs/
    ├── qlora_mistral.yaml
    └── lora_sweep.yaml
```

### LoRA Layer from Scratch

```python
# src/lora_layer.py
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """Replace a frozen Linear layer with a LoRA-augmented version."""
    
    def __init__(self, original_linear: nn.Linear, rank: int = 16, alpha: float = 32.0,
                 dropout: float = 0.05):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # Freeze original weights
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)
        
        # LoRA matrices: A projects down to rank, B projects back up
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize A with Kaiming, B with zeros (so ΔW starts at 0)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen forward
        base_output = self.original(x)
        
        # LoRA path: x → dropout → A → B → scale
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        
        return base_output + lora_output
    
    def merge(self) -> nn.Linear:
        """Merge LoRA weights into original linear layer."""
        merged = nn.Linear(
            self.original.in_features,
            self.original.out_features,
            bias=self.original.bias is not None
        )
        # W_merged = W_original + scaling * B @ A
        merged.weight.data = self.original.weight.data + self.scaling * (self.lora_B @ self.lora_A)
        if self.original.bias is not None:
            merged.bias.data = self.original.bias.data
        return merged
```

### LoRA Injection into Pre-trained Model

```python
# src/lora_injection.py
import torch.nn as nn
from .lora_layer import LoRALinear

def inject_lora(model, target_modules: list[str], rank: int = 16, alpha: float = 32.0):
    """
    Replace target linear layers with LoRA-augmented versions.
    
    target_modules: list of substrings to match (e.g., ['q_proj', 'v_proj', 'k_proj', 'o_proj'])
    """
    replaced = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(target in name for target in target_modules):
                # Navigate to parent and replace
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, parts[-1], lora_layer)
                replaced[name] = (module.in_features, module.out_features)
    
    # Verify: count trainable vs frozen parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Injected LoRA into {len(replaced)} layers")
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model, replaced

def get_trainable_params(model):
    """Return only parameters that require gradients (LoRA params)."""
    return [p for p in model.parameters() if p.requires_grad]
```

### QLoRA Setup

```python
# src/qlora_setup.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def setup_qlora(model_name: str, rank: int = 16, alpha: int = 32,
                target_modules: list[str] = None) -> tuple:
    """Load model in 4-bit and apply LoRA adapters."""
    
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # 4-bit quantization config (NF4 + double quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # Quantize quantization constants
    )
    
    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare for k-bit training (handle layer norms, etc.)
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def memory_report(model):
    """Report memory usage breakdown."""
    total_mem = torch.cuda.memory_allocated() / 1e9
    params = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    trainable = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / 1e9
    print(f"GPU Memory Allocated: {total_mem:.2f} GB")
    print(f"Model Parameters: {params:.2f} GB")
    print(f"Trainable Parameters: {trainable:.4f} GB")
```

### Hyperparameter Sweep

```python
# scripts/hp_sweep.py
import itertools
from src.qlora_setup import setup_qlora
from src.train import train_lora

sweep_config = {
    'rank': [4, 8, 16, 32, 64],
    'alpha': [16, 32, 64],
    'target_modules': [
        ['q_proj', 'v_proj'],                    # Attention only (minimal)
        ['q_proj', 'k_proj', 'v_proj', 'o_proj'], # All attention
        ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],  # All linear
    ],
}

results = []
for rank, alpha, targets in itertools.product(
    sweep_config['rank'], sweep_config['alpha'], sweep_config['target_modules']
):
    model, tokenizer = setup_qlora("mistralai/Mistral-7B-v0.1", rank=rank, alpha=alpha, target_modules=targets)
    metrics = train_lora(model, tokenizer, dataset="code_alpaca", epochs=1, eval_steps=100)
    results.append({
        'rank': rank, 'alpha': alpha, 'targets': targets,
        'final_loss': metrics['final_loss'],
        'eval_score': metrics['eval_score'],
        'trainable_params': metrics['trainable_params'],
        'memory_gb': metrics['peak_memory_gb'],
    })
    del model
    torch.cuda.empty_cache()
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| CUDA OOM loading 7B model | Use `BitsAndBytesConfig(load_in_4bit=True)`. 7B in NF4 ≈ 3.5GB VRAM |
| `bitsandbytes` not found | `pip install bitsandbytes`. Requires CUDA toolkit. Check `nvidia-smi` matches |
| LoRA loss not decreasing | Check that base model is frozen (`requires_grad=False`). Verify only adapter params in optimizer |
| Custom LoRA doesn't match PEFT | Check initialization: PEFT uses `kaiming_uniform` for A, zeros for B. Check `scaling = alpha/rank` |
| Merge changes outputs | Numerical precision: use `torch.float32` for merge arithmetic, then cast back |
| QLoRA training slower than expected | BnB 4-bit dequantization adds overhead. Ensure compute dtype is `bfloat16`. Use `torch.compile` if supported |
| `prepare_model_for_kbit_training` errors | Update PEFT and transformers to latest versions. Some models need `use_gradient_checkpointing=True` |
| Adapter save is too large | Check you're saving only adapter weights (`model.save_pretrained()` from PEFT), not full model |

---

## Agent Handoff Template

```
I'm working on Week 4 of the Crucible Phase 1 project: LoRA and QLoRA Fine-Tuning.

Hardware: RTX 5080 16GB VRAM, 32GB RAM, Ubuntu
Project path: ~/crucible/week04-lora-finetuning/

Current status: [DESCRIBE WHERE YOU ARE]

What I've completed:
- [x/o] LoRA layer from scratch
- [x/o] LoRA injection into pre-trained model
- [x/o] PEFT library integration
- [x/o] QLoRA setup (4-bit + LoRA)
- [x/o] Hyperparameter sweep
- [x/o] Adapter merging
- [x/o] Mistral-7B QLoRA training
- [x/o] Comparison table

Model: [WHICH MODEL]
Task: [WHAT FINE-TUNING TASK]
Current rank/alpha: [HYPERPARAMETERS]

Current issue: [DESCRIBE THE PROBLEM]
Error message (if any): [PASTE ERROR]
Memory usage: [nvidia-smi output]

Please help me [SPECIFIC ASK].
```

---

## Out of Scope

- DPO/RLHF alignment (Phase 2)
- Instruction tuning data preparation (Week 5)
- Distributed LoRA training (Week 6)
- DoRA, AdaLoRA, or other LoRA variants (mention but don't implement)
- Serving multiple LoRA adapters simultaneously (inference optimization)
- Model architecture changes or training from scratch
- Prompt engineering or evaluation benchmarks design
