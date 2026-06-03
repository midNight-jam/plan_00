# Week 9: RLHF with PPO

## Context

**Where it fits:** Phase 2 (Alignment Deep Dive), Week 2 of 7. This is the canonical alignment method — the approach that made ChatGPT possible. Understanding PPO-based RLHF deeply is the single most important skill for Anthropic/OpenAI interviews.

**Prerequisites:**
- Week 8: Trained reward model (checkpoint ready for scoring)
- Phase 1: Custom training loop, mixed precision, gradient accumulation
- Understanding of policy gradient methods (REINFORCE at minimum)
- Working SFT model as the starting policy

**What it builds on:** Your reward model from Week 8 provides the scoring signal. Your SFT model becomes the initial policy. The training loop infrastructure from Phase 1 handles the optimization.

**What it enables:** Week 10 (DPO) is motivated by PPO's complexity — you need to experience that complexity firsthand. Weeks 11-14 all compare against your PPO baseline.

---

## Learning Goals

- [ ] Explain the 4-model RLHF setup: policy, reference, reward, value (and why each exists)
- [ ] Derive the PPO clipped objective and explain why clipping prevents catastrophic updates
- [ ] Understand KL penalty: prevents policy from exploiting reward model weaknesses
- [ ] Explain GAE (Generalized Advantage Estimation): bias-variance tradeoff in advantage computation
- [ ] Articulate why a value model (critic) reduces variance compared to REINFORCE
- [ ] Describe mode collapse: what happens without KL penalty (policy finds degenerate high-reward outputs)
- [ ] Explain the generation→scoring→training loop and why it's expensive (4 forward passes minimum)

---

## Implementation Goals

- [ ] Implement the 4-model setup with proper memory management (fit in 16GB VRAM)
- [ ] Implement response generation from policy with sampling (temperature, top-p)
- [ ] Implement reward scoring pipeline (batch scoring with reward model)
- [ ] Implement KL divergence computation between policy and reference log-probs
- [ ] Implement GAE advantage estimation from per-token rewards
- [ ] Implement PPO clipped surrogate objective
- [ ] Implement value function loss (MSE on predicted vs actual returns)
- [ ] Implement the full training loop: generate → score → compute advantages → PPO update
- [ ] Monitor: reward curve, KL divergence, entropy, value loss over training
- [ ] Manual quality inspection: sample 20 responses before/after training

---

## Acceptance Criteria

1. Four models load correctly: policy (trainable), reference (frozen), reward (frozen from Week 8), value (trainable, initialized from policy).
2. All four models fit in 16GB VRAM simultaneously using bf16 + gradient checkpointing (policy/value may use LoRA).
3. Generation produces coherent completions from the policy model with configurable sampling parameters.
4. KL divergence between policy and reference is computed per-token and the mean KL stays below 10 nats throughout training.
5. PPO clipped objective is implemented correctly: `min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)`.
6. Mean reward increases by at least 0.3 reward units over 500 PPO steps compared to the SFT baseline.
7. Training runs at >2 samples/second on RTX 5080 (including generation time).
8. KL coefficient sweep (0.01, 0.05, 0.1, 0.2) produces different reward/KL tradeoff curves.
9. Value model loss decreases over training, indicating it learns to predict returns.
10. Manual inspection of 20 sampled responses shows qualitative improvement over SFT baseline (document in results/).

---

## Validation Commands

```bash
# Verify all 4 models load and fit in VRAM
python check_memory.py --policy TinyLlama/TinyLlama-1.1B-Chat-v1.0 --reward ./checkpoints/reward_model_1B --dtype bf16

# Run PPO training (smoke test - 50 steps)
python train_ppo.py --policy_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --reward_model ./checkpoints/reward_model_1B --num_steps 50 --batch_size 4 --kl_coeff 0.1 --output_dir ./checkpoints/ppo_smoke

# Monitor training metrics
python plot_ppo_metrics.py --log_dir ./logs/ppo_run --output plots/ppo_curves.png

# KL coefficient sweep
for kl in 0.01 0.05 0.1 0.2; do
    python train_ppo.py --kl_coeff $kl --num_steps 200 --output_dir ./checkpoints/ppo_kl_${kl} --log_dir ./logs/ppo_kl_${kl}
done

# Compare KL sweep results
python compare_kl_sweeps.py --log_dirs ./logs/ppo_kl_0.01,./logs/ppo_kl_0.05,./logs/ppo_kl_0.1,./logs/ppo_kl_0.2

# Sample responses before/after training
python sample_responses.py --model ./checkpoints/ppo_final --prompts prompts/eval_set.txt --output results/ppo_samples.jsonl

# Full training run (expect ~6-8 hours)
python train_ppo.py --policy_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --reward_model ./checkpoints/reward_model_1B --num_steps 2000 --batch_size 8 --kl_coeff 0.1 --clip_range 0.2 --output_dir ./checkpoints/ppo_full
```

---

## Technical Implementation Details

### Project Structure

```
crucible/phase2/week09/
├── train_ppo.py              # Main PPO training loop
├── ppo_trainer.py            # PPO algorithm implementation
├── models.py                 # 4-model management (loading, memory)
├── generation.py             # Policy generation with sampling
├── advantages.py             # GAE computation
├── check_memory.py           # VRAM usage verification
├── sample_responses.py       # Before/after comparison
├── compare_kl_sweeps.py      # KL coefficient analysis
├── plot_ppo_metrics.py       # Training curves
├── configs/
│   └── ppo_default.yaml
├── prompts/
│   └── eval_set.txt
└── results/
```

### Four-Model Setup

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from reward_model import RewardModel

class RLHFModels:
    """Manages the 4 models required for PPO-based RLHF."""

    def __init__(self, policy_name: str, reward_path: str, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(policy_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Policy model (trainable) - uses LoRA to save memory
        base_policy = AutoModelForCausalLM.from_pretrained(
            policy_name, torch_dtype=dtype, attn_implementation="flash_attention_2"
        ).to(device)
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
        self.policy = get_peft_model(base_policy, lora_config)
        self.policy.gradient_checkpointing_enable()

        # Reference model (frozen copy of initial policy)
        self.reference = AutoModelForCausalLM.from_pretrained(
            policy_name, torch_dtype=dtype, attn_implementation="flash_attention_2"
        ).to(device)
        self.reference.eval()
        for param in self.reference.parameters():
            param.requires_grad = False

        # Reward model (frozen, from Week 8)
        self.reward = RewardModel.load(reward_path).to(device)
        self.reward.eval()
        for param in self.reward.parameters():
            param.requires_grad = False

        # Value model (critic) - separate head on policy backbone
        self.value_head = torch.nn.Linear(
            base_policy.config.hidden_size, 1, dtype=dtype
        ).to(device)

    def get_memory_usage(self):
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
```

### Generation from Policy

```python
@torch.no_grad()
def generate_responses(policy, tokenizer, prompts, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """Generate responses from the current policy for a batch of prompts."""
    policy.eval()
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(policy.device) for k, v in inputs.items()}

    outputs = policy.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )

    # Extract only the generated tokens (not the prompt)
    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[:, prompt_length:]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    policy.train()
    return generated_ids, responses
```

### KL Divergence Computation

```python
def compute_kl_divergence(policy_logprobs, reference_logprobs):
    """
    Compute per-token KL divergence: KL(policy || reference).
    KL = policy_logprob - reference_logprob (approximation for KL when using log probs directly).

    For the full KL: sum over vocab of policy_prob * (policy_logprob - reference_logprob)
    But in practice we use the simplified per-token estimate.
    """
    kl = policy_logprobs - reference_logprobs  # Per-token KL estimate
    return kl

def get_per_token_logprobs(model, input_ids, attention_mask):
    """Compute log-probability of each token under the model."""
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # Shift: predict next token
        target_ids = input_ids[:, 1:]  # Shift: actual next tokens

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

        # Mask padding tokens
        mask = attention_mask[:, 1:]
        token_log_probs = token_log_probs * mask

    return token_log_probs
```

### GAE Advantage Estimation

```python
def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """
    Generalized Advantage Estimation.

    For language model RLHF:
    - rewards: per-token rewards (usually 0 except last token which gets the RM score - KL penalty)
    - values: value model predictions at each token position
    - gamma: discount factor (1.0 for language since episodes are short)
    - lam: GAE lambda (bias-variance tradeoff, 0.95 is standard)

    Returns advantages and returns (for value function training).
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(batch_size, device=rewards.device)

    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = 0  # Terminal state
        else:
            next_value = values[:, t + 1]

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae

    returns = advantages + values
    return advantages, returns
```

### PPO Update

```python
def ppo_step(policy, value_head, old_logprobs, states, actions, advantages, returns,
             clip_range=0.2, value_clip_range=0.2, max_grad_norm=1.0):
    """
    Single PPO update step.

    Args:
        old_logprobs: log-probs from the policy that generated the data
        states: input_ids for computing new log-probs
        actions: generated token ids
        advantages: GAE advantages (normalized)
        returns: target returns for value function
    """
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Get current policy log-probs
    new_logprobs = get_per_token_logprobs(policy, states, actions)

    # Policy ratio
    ratio = torch.exp(new_logprobs - old_logprobs)

    # Clipped surrogate objective
    policy_loss_1 = ratio * advantages
    policy_loss_2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

    # Value function loss
    values = value_head(states).squeeze(-1)
    value_loss = 0.5 * ((values - returns) ** 2).mean()

    # Entropy bonus (encourages exploration)
    entropy = -(new_logprobs * torch.exp(new_logprobs)).mean()

    # Total loss
    total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

    return total_loss, {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "clip_fraction": ((ratio - 1.0).abs() > clip_range).float().mean().item(),
        "approx_kl": (old_logprobs - new_logprobs).mean().item(),
    }
```

### Full PPO Training Loop

```python
def train_ppo(models, prompts_dataset, config):
    """Full PPO training loop: generate → score → advantage → update."""
    optimizer = torch.optim.AdamW(
        list(models.policy.parameters()) + list(models.value_head.parameters()),
        lr=config.lr, weight_decay=0.01
    )

    for step in range(config.num_steps):
        # 1. Sample batch of prompts
        prompt_batch = sample_prompts(prompts_dataset, config.batch_size)

        # 2. Generate responses from current policy
        generated_ids, responses = generate_responses(
            models.policy, models.tokenizer, prompt_batch
        )

        # 3. Score with reward model
        rewards = models.reward.score_batch(prompt_batch, responses)

        # 4. Compute KL penalty
        policy_logprobs = get_per_token_logprobs(models.policy, generated_ids, mask)
        ref_logprobs = get_per_token_logprobs(models.reference, generated_ids, mask)
        kl = compute_kl_divergence(policy_logprobs, ref_logprobs)
        kl_penalty = config.kl_coeff * kl

        # 5. Compute per-token rewards (RM score at last token minus KL at each token)
        token_rewards = -kl_penalty.clone()
        token_rewards[:, -1] += rewards  # Add RM score at the end

        # 6. Compute advantages with GAE
        values = models.value_head(hidden_states).squeeze(-1)
        advantages, returns = compute_gae(token_rewards, values.detach())

        # 7. PPO update (multiple epochs over same batch)
        for epoch in range(config.ppo_epochs):
            loss, metrics = ppo_step(
                models.policy, models.value_head,
                policy_logprobs.detach(), generated_ids, generated_ids,
                advantages, returns, config.clip_range
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(models.policy.parameters(), config.max_grad_norm)
            optimizer.step()

        # Logging
        if step % 10 == 0:
            mean_reward = rewards.mean().item()
            mean_kl = kl.mean().item()
            print(f"Step {step} | Reward: {mean_reward:.3f} | KL: {mean_kl:.3f} | {metrics}")
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| OOM with 4 models | Use LoRA for policy+value, quantize reference to 4-bit with bitsandbytes, keep reward model in 8-bit |
| Reward doesn't increase | Check KL coeff isn't too high (start with 0.01). Verify reward model gives non-constant scores on generated text. |
| KL explodes (>50) | KL coeff too low or clip range too large. Increase kl_coeff, reduce clip_range to 0.1. |
| Mode collapse (repetitive outputs) | KL penalty not working. Verify reference model is truly frozen. Add entropy bonus. |
| Training very slow | Generation is the bottleneck. Use larger batch with generation, then split for PPO updates. Use vLLM for generation if possible. |
| Value loss not decreasing | Value head may need higher learning rate than policy. Try separate lr: 1e-4 for value, 1e-5 for policy. |
| Gradient NaN | Clip gradients more aggressively (0.5 instead of 1.0). Check for log(0) in logprob computation. |

---

## Agent Handoff Template

```
Continue the Crucible Phase 2, Week 9 (RLHF with PPO) project.

Hardware: RTX 5080 16GB VRAM, 32GB RAM, Ubuntu.
Project location: crucible/phase2/week09/

Current state: [DESCRIBE WHAT'S DONE - e.g., "4 models load, generation works, but PPO update produces NaN"]
Blocked on: [DESCRIBE THE ISSUE]

The goal is to implement full PPO-based RLHF from scratch (not using TRL) with:
- 4 models: policy (LoRA), reference (frozen), reward (from Week 8), value (trainable head)
- Generation → Scoring → KL penalty → GAE → PPO clipped update loop
- Target: reward increases by 0.3+ over 500 steps, KL stays < 10 nats

Key files:
- train_ppo.py: Main training loop
- ppo_trainer.py: PPO algorithm (clipped objective, GAE)
- models.py: 4-model loading with memory management

Please [FIX/CONTINUE/DEBUG] the [SPECIFIC COMPONENT].
```

---

## Out of Scope

- DPO or other RL-free alignment methods (Week 10)
- Online RLHF with iterative reward model updates
- Multi-GPU training or FSDP
- Reward model training (already done in Week 8)
- RLHF at scale (>7B models)
- Human-in-the-loop annotation
- Safety filtering or content moderation during generation
- Alternative RL algorithms (REINFORCE, A2C, GRPO) except for brief comparison
