# Week 17: Training and Alignment

## Context

**Phase:** 3 — Production Infrastructure & Advanced Systems
**Prerequisites:** Deep understanding of transformer architecture (Week 8), working inference pipeline, familiarity with PyTorch tensor operations and autograd.
**Duration:** 1 week
**Difficulty:** Advanced

You've mastered inference. Now you'll understand the other side: how models get their capabilities. Training and alignment are where model behavior is shaped. Understanding custom training loops, LoRA, and RLHF/DPO gives you the full picture — you'll know why models behave the way they do, what fine-tuning can and can't fix, and how alignment techniques actually work under the hood. This knowledge is essential for production ML engineers who need to adapt models to specific use cases.

---

## Learning Goals

- Understand the complete training loop: forward pass → loss → backward pass → optimizer step
- Learn mixed precision training: why fp16 forward + fp32 gradients + loss scaling works
- Understand gradient accumulation as a memory-compute tradeoff
- Learn gradient checkpointing: recompute activations during backward to save memory
- Understand LoRA: low-rank adaptation as parameter-efficient fine-tuning
- Learn DPO: how to align models from preference data without a separate reward model
- Understand the RLHF pipeline and why DPO is a simplification of it

---

## Implementation Goals

- Build a custom training loop from scratch (no HuggingFace Trainer)
- Implement mixed precision, gradient accumulation, and gradient checkpointing manually
- Implement LoRA fine-tuning on a small model (≤3B parameters)
- Implement the DPO loss function and train on a preference dataset
- Train a Bradley-Terry reward model from preference data
- Evaluate and compare: base model vs LoRA fine-tuned vs DPO-aligned
- Write a technical document explaining RLHF vs DPO tradeoffs with code

---

## Acceptance Criteria

1. Custom training loop runs for 100+ steps without errors, with loss decreasing monotonically (smoothed) on the training set.
2. Mixed precision training uses `torch.cuda.amp.autocast` for forward pass and `GradScaler` for backward pass, with no NaN/Inf values in gradients (verified by gradient norm logging).
3. Gradient accumulation correctly accumulates gradients over N micro-batches before stepping, producing identical results to a single batch of size N×micro_batch_size (verified numerically).
4. Gradient checkpointing reduces peak GPU memory by at least 30% compared to non-checkpointed training at the same batch size (measured and logged).
5. LoRA fine-tuning trains only the low-rank adapter weights (verified: base model weights unchanged, adapter weights updated), producing measurably different outputs on evaluation prompts compared to the base model.
6. DPO loss (log-sigmoid of reward margin between chosen and rejected) decreases over training steps, reaching a lower value than initialization by at least 20%.
7. The trained DPO model produces outputs that are preferred over base model outputs on held-out test prompts (win rate >55% measured by reward model or automated eval).
8. Reward model achieves >60% accuracy on a held-out preference validation set (above random baseline of 50%).
9. A comparison document (Markdown) presents: base vs LoRA vs DPO outputs on 10 evaluation prompts, with automated metrics (perplexity, reward score, win rate).
10. Technical write-up explains RLHF vs DPO with code snippets, covering: PPO objective, DPO derivation from reward model, practical tradeoffs (stability, compute, data requirements).

---

## Validation Commands

```bash
# Run the custom training loop
python train/train_loop.py \
  --model meta-llama/Llama-3.2-1B \
  --dataset tatsu-lab/alpaca \
  --batch-size 4 \
  --gradient-accumulation 8 \
  --mixed-precision \
  --gradient-checkpointing \
  --max-steps 200 \
  --output checkpoints/custom-loop/

# Verify gradient accumulation equivalence
python tests/test_grad_accumulation.py

# Run LoRA fine-tuning
python train/lora_finetune.py \
  --model meta-llama/Llama-3.2-1B \
  --dataset tatsu-lab/alpaca \
  --lora-rank 16 \
  --lora-alpha 32 \
  --max-steps 500 \
  --output checkpoints/lora/

# Train reward model
python train/reward_model.py \
  --model meta-llama/Llama-3.2-1B \
  --dataset argilla/ultrafeedback-binarized-preferences \
  --max-steps 300 \
  --output checkpoints/reward-model/

# Run DPO training
python train/dpo_train.py \
  --model meta-llama/Llama-3.2-1B \
  --ref-model meta-llama/Llama-3.2-1B \
  --dataset argilla/ultrafeedback-binarized-preferences \
  --beta 0.1 \
  --max-steps 300 \
  --output checkpoints/dpo/

# Evaluate all models
python eval/compare_models.py \
  --base meta-llama/Llama-3.2-1B \
  --lora checkpoints/lora/ \
  --dpo checkpoints/dpo/ \
  --reward-model checkpoints/reward-model/ \
  --prompts eval/test_prompts.json \
  --output results/comparison.md

# Run tests
pytest tests/ -v
```

---

## Technical Implementation Details

### Custom Training Loop

```python
# train/train_loop.py
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

class CustomTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Optimizer with weight decay (exclude bias and layernorm)
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if "bias" in name or "norm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = AdamW([
            {"params": decay_params, "weight_decay": 0.01},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=config.learning_rate, betas=(0.9, 0.95))

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.max_steps, eta_min=config.learning_rate * 0.1
        )
        self.scaler = GradScaler(enabled=config.mixed_precision)
        self.global_step = 0

    def train_step(self, batch):
        """Single training step with gradient accumulation."""
        self.model.train()
        total_loss = 0.0

        for micro_step in range(self.config.gradient_accumulation_steps):
            micro_batch = self._get_micro_batch(batch, micro_step)
            input_ids = micro_batch["input_ids"].cuda()
            labels = micro_batch["labels"].cuda()

            with autocast(enabled=self.config.mixed_precision):
                if self.config.gradient_checkpointing:
                    outputs = self._forward_with_checkpointing(input_ids, labels)
                else:
                    outputs = self.model(input_ids=input_ids, labels=labels)

                loss = outputs.loss / self.config.gradient_accumulation_steps

            self.scaler.scale(loss).backward()
            total_loss += loss.item()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )

        # Check for NaN gradients
        if not math.isfinite(grad_norm):
            print(f"WARNING: grad_norm={grad_norm}, skipping step")
            self.optimizer.zero_grad()
            return {"loss": float("nan"), "grad_norm": float(grad_norm)}

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1

        return {
            "loss": total_loss,
            "grad_norm": grad_norm.item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

    def _forward_with_checkpointing(self, input_ids, labels):
        """Manual gradient checkpointing: wrap each transformer layer."""
        embeddings = self.model.model.embed_tokens(input_ids)
        hidden = embeddings

        for layer in self.model.model.layers:
            hidden = checkpoint(layer, hidden, use_reentrant=False)

        hidden = self.model.model.norm(hidden)
        logits = self.model.lm_head(hidden)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        class Output:
            pass
        out = Output()
        out.loss = loss
        return out
```

### LoRA Implementation

```python
# train/lora.py
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer wrapping a frozen linear layer."""

    def __init__(self, original_linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
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

        # LoRA matrices: W' = W + (alpha/r) * B @ A
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming, B with zeros (so initial output = original)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = self.original(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_output + lora_output


def apply_lora(model, rank=16, alpha=32.0, target_modules=("q_proj", "v_proj")):
    """Replace target linear layers with LoRA-wrapped versions."""
    replaced = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = dict(model.named_modules())[parent_name]
                setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))
                replaced += 1

    # Only train LoRA parameters
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if "lora_" in name:
            param.requires_grad_(True)
            trainable += param.numel()
        else:
            param.requires_grad_(False)

    print(f"LoRA applied: {replaced} layers, {trainable:,} trainable / {total:,} total "
          f"({100*trainable/total:.2f}%)")
    return model
```

### DPO Training

```python
# train/dpo_train.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

class DPOTrainer:
    """
    Direct Preference Optimization:
    Loss = -log(sigmoid(beta * (log_pi(y_w|x) - log_pi(y_l|x) - log_ref(y_w|x) + log_ref(y_l|x))))

    Intuitively: increase probability of preferred response relative to rejected,
    but regularized against drifting too far from the reference model.
    """

    def __init__(self, model, ref_model, beta=0.1, lr=1e-6):
        self.model = model
        self.ref_model = ref_model
        self.beta = beta

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr, betas=(0.9, 0.95)
        )

    def compute_log_probs(self, model, input_ids, labels):
        """Compute per-token log probabilities for a sequence."""
        with torch.no_grad() if not model.training else torch.enable_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits

        # Shift for autoregressive: predict token t from tokens 0..t-1
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask padding tokens
        mask = (shift_labels != -100).float()
        sequence_log_prob = (per_token_log_probs * mask).sum(dim=-1)

        return sequence_log_prob

    def dpo_loss(self, batch):
        """Compute DPO loss for a batch of preference pairs."""
        chosen_ids = batch["chosen_input_ids"].cuda()
        chosen_labels = batch["chosen_labels"].cuda()
        rejected_ids = batch["rejected_input_ids"].cuda()
        rejected_labels = batch["rejected_labels"].cuda()

        # Policy model log probs
        self.model.train()
        pi_chosen = self.compute_log_probs(self.model, chosen_ids, chosen_labels)
        pi_rejected = self.compute_log_probs(self.model, rejected_ids, rejected_labels)

        # Reference model log probs (frozen)
        with torch.no_grad():
            ref_chosen = self.compute_log_probs(self.ref_model, chosen_ids, chosen_labels)
            ref_rejected = self.compute_log_probs(self.ref_model, rejected_ids, rejected_labels)

        # DPO loss: -log(sigmoid(beta * (pi_chosen - pi_rejected - ref_chosen + ref_rejected)))
        log_ratio_chosen = pi_chosen - ref_chosen
        log_ratio_rejected = pi_rejected - ref_rejected
        logits = self.beta * (log_ratio_chosen - log_ratio_rejected)

        loss = -F.logsigmoid(logits).mean()

        # Metrics
        with torch.no_grad():
            reward_chosen = self.beta * log_ratio_chosen
            reward_rejected = self.beta * log_ratio_rejected
            accuracy = (logits > 0).float().mean()

        return loss, {
            "loss": loss.item(),
            "reward_chosen": reward_chosen.mean().item(),
            "reward_rejected": reward_rejected.mean().item(),
            "accuracy": accuracy.item(),
            "reward_margin": (reward_chosen - reward_rejected).mean().item(),
        }

    def train_step(self, batch):
        loss, metrics = self.dpo_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return metrics
```

### Reward Model

```python
# train/reward_model.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class RewardModel(nn.Module):
    """
    Bradley-Terry reward model: given two responses, predict which is preferred.
    Architecture: LLM backbone (frozen or partially trained) + scalar reward head.
    """

    def __init__(self, model_name: str, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        hidden_size = self.backbone.config.hidden_size

        # Remove the LM head, add reward head
        self.backbone.lm_head = nn.Identity()
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    def forward(self, input_ids, attention_mask=None):
        """Return scalar reward for each sequence."""
        outputs = self.backbone.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden = outputs.last_hidden_state

        # Use last non-padding token's representation
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1) - 1
            last_hidden = hidden[torch.arange(hidden.size(0)), last_idx]
        else:
            last_hidden = hidden[:, -1, :]

        reward = self.reward_head(last_hidden).squeeze(-1)
        return reward

    def preference_loss(self, chosen_ids, rejected_ids, chosen_mask=None, rejected_mask=None):
        """
        Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
        The preferred response should receive a higher reward.
        """
        r_chosen = self.forward(chosen_ids, chosen_mask)
        r_rejected = self.forward(rejected_ids, rejected_mask)

        loss = -torch.nn.functional.logsigmoid(r_chosen - r_rejected).mean()
        accuracy = (r_chosen > r_rejected).float().mean()

        return loss, {"loss": loss.item(), "accuracy": accuracy.item()}
```

### Project Structure

```
forge-training/
├── train/
│   ├── train_loop.py        # Custom training loop
│   ├── lora.py              # LoRA implementation
│   ├── lora_finetune.py     # LoRA fine-tuning script
│   ├── dpo_train.py         # DPO training
│   ├── reward_model.py      # Bradley-Terry reward model
│   └── data.py              # Dataset loading/preprocessing
├── eval/
│   ├── compare_models.py    # Model comparison script
│   ├── test_prompts.json    # Evaluation prompts
│   └── metrics.py           # Perplexity, win rate, etc.
├── tests/
│   ├── test_train_loop.py
│   ├── test_grad_accumulation.py
│   ├── test_lora.py
│   └── test_dpo_loss.py
├── docs/
│   └── rlhf_vs_dpo.md      # Technical write-up
├── checkpoints/
│   └── .gitkeep
├── results/
│   └── .gitkeep
├── requirements.txt
└── README.md
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| OOM during training | Reduce batch size, enable gradient checkpointing, use bfloat16 instead of float16 |
| Loss is NaN | Check GradScaler — if loss scale goes to 0, your learning rate is too high; also check for inf in logits |
| LoRA doesn't change outputs | Verify adapter weights are actually being updated (print norm before/after); check that `requires_grad=True` on LoRA params |
| DPO loss doesn't decrease | Beta too high (try 0.05); reference model must be frozen; check that chosen/rejected aren't swapped |
| Reward model stuck at 50% | Model may be too frozen — unfreeze last 2-4 layers; also check data quality |
| Gradient accumulation mismatch | Division by accumulation steps must happen on the loss, not the gradients; verify with `torch.allclose` |
| Checkpointing doesn't save memory | Must use `use_reentrant=False` with modern PyTorch; also ensure you're wrapping the right layers |
| Model generates garbage after fine-tuning | Learning rate too high; too many steps (overfitting); check that padding tokens are masked in loss |

**Key Resources:**
- [DPO paper](https://arxiv.org/abs/2305.18290) — read Section 3 for the derivation
- [LoRA paper](https://arxiv.org/abs/2106.09685) — understand why rank 16 is often sufficient
- [PyTorch AMP tutorial](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — reference training loop
- [TRL library source](https://github.com/huggingface/trl) — reference DPO implementation (read after building your own)

---

## Agent Handoff Template

```
## Session State
- Phase: 3 / Week 17
- Current task: [what you're working on]
- Branch: forge/week17-training-alignment

## What's Done
- [ ] Custom training loop (mixed precision + grad accum + checkpointing)
- [ ] Training loop verified: loss decreases, no NaN
- [ ] LoRA implementation applied to target model
- [ ] LoRA fine-tuning produces different outputs
- [ ] DPO loss function implemented
- [ ] DPO training decreases loss
- [ ] Reward model trained (>60% accuracy)
- [ ] Model comparison eval complete
- [ ] Technical write-up on RLHF vs DPO

## Current Blocker
[Describe the exact error/issue]

## Key Files
- train/train_loop.py — custom training loop
- train/lora.py — LoRA implementation
- train/dpo_train.py — DPO trainer
- train/reward_model.py — Bradley-Terry model
- docs/rlhf_vs_dpo.md — technical write-up

## Hardware
[GPU model, VRAM, max batch size that fits]

## Next Step
[Exact next action to take]
```

---

## Out of Scope

- Full RLHF with PPO (DPO is the focus; mention PPO in write-up only)
- Distributed training (single GPU only)
- FSDP or DeepSpeed integration
- Training models larger than 3B parameters
- Collecting your own preference data (use existing datasets)
- Constitutional AI or RLAIF approaches
- Quantization-aware training (QAT)
- Curriculum learning or data mixing strategies
- Deploying fine-tuned models back to the inference server (that's integration, not training)
