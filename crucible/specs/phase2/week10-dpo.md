# Week 10: Direct Preference Optimization (DPO)

## Context

**Where it fits:** Phase 2 (Alignment Deep Dive), Week 3 of 7. DPO is the most important alternative to PPO-based RLHF — it eliminates the reward model and RL training entirely, replacing them with a single supervised loss.

**Prerequisites:**
- Week 8: Reward model (for comparison evaluation)
- Week 9: Working PPO-RLHF pipeline (DPO is motivated as a simpler alternative)
- Understanding of the RLHF objective: maximize reward while staying close to reference policy
- SFT model as starting point (reference model)

**What it builds on:** DPO is derived by solving the RLHF optimization problem in closed form. You need to have experienced PPO's complexity (4 models, generation loop, RL instability) to understand why DPO's simplicity is revolutionary.

**What it enables:** Week 11 (Advanced Alignment) builds DPO variants (KTO, ORPO, SimPO). Week 14's comparison report needs DPO as a key baseline.

---

## Learning Goals

- [ ] Derive DPO loss from the RLHF objective: show that optimal policy satisfies r(x,y) = β * log(π(y|x)/π_ref(y|x)) + C
- [ ] Explain why DPO is "implicit reward modeling" — the policy IS the reward model
- [ ] Understand the DPO loss: why it pushes chosen log-probs up and rejected log-probs down
- [ ] Articulate DPO failure modes: distribution shift, overfitting to preferences, reference model staleness
- [ ] Explain β (temperature): controls how much policy can deviate from reference (analogous to KL coeff in PPO)
- [ ] Understand online vs offline DPO: why on-policy data matters
- [ ] Compare DPO to PPO: compute cost, stability, data efficiency, final quality

---

## Implementation Goals

- [ ] Implement DPO loss function from scratch (no TRL)
- [ ] Implement log-probability computation for chosen and rejected sequences
- [ ] Train DPO on same preference dataset used for reward model (HH-RLHF)
- [ ] Implement reference model management (frozen copy vs periodic update)
- [ ] Implement IPO (Identity Preference Optimization) as regularized variant
- [ ] Compare DPO vs PPO: use same base model, same data, measure win-rate
- [ ] Measure with LLM-as-judge: win-rate vs base model, vs SFT model, vs PPO model
- [ ] Analyze β sensitivity: train with β = 0.05, 0.1, 0.2, 0.5
- [ ] Track implicit reward margin during training: log(π/π_ref) for chosen vs rejected

---

## Acceptance Criteria

1. DPO loss is implemented from scratch matching: `-β * log(σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x))))`.
2. Per-sequence log-probability computation handles padding correctly (only scores response tokens, not prompt tokens).
3. Training converges within 3 epochs on HH-RLHF (loss decreases monotonically after warmup).
4. DPO model achieves >55% win-rate vs SFT baseline when judged by GPT-4/Claude (on 100 test prompts).
5. β sweep (0.05, 0.1, 0.2, 0.5) shows clear tradeoff: lower β = more change from reference, higher β = more conservative.
6. IPO variant is implemented with its loss function and compared against standard DPO.
7. Training completes in under 2 hours on RTX 5080 16GB for 1B model (much faster than PPO).
8. Implicit reward margin (log-ratio gap between chosen and rejected) increases during training.
9. Head-to-head comparison table: DPO vs PPO on compute time, VRAM, win-rate, training stability.
10. DPO model doesn't degenerate: perplexity on general text stays within 1.5x of SFT model (no catastrophic forgetting).

---

## Validation Commands

```bash
# Run DPO training (smoke test)
python train_dpo.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset anthropic_hh --max_samples 500 --beta 0.1 --epochs 1 --batch_size 4 --output_dir ./checkpoints/dpo_smoke

# Full DPO training
python train_dpo.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset anthropic_hh --beta 0.1 --epochs 3 --batch_size 8 --lr 5e-6 --output_dir ./checkpoints/dpo_full

# β sweep
for beta in 0.05 0.1 0.2 0.5; do
    python train_dpo.py --beta $beta --epochs 3 --output_dir ./checkpoints/dpo_beta_${beta} --log_dir ./logs/dpo_beta_${beta}
done

# IPO training
python train_ipo.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset anthropic_hh --tau 0.1 --epochs 3 --output_dir ./checkpoints/ipo

# Evaluate with LLM-as-judge
python eval_judge.py --model_a ./checkpoints/dpo_full --model_b ./checkpoints/sft_baseline --judge gpt-4 --num_prompts 100 --output results/dpo_vs_sft.json

# Compare DPO vs PPO
python compare_methods.py --dpo_model ./checkpoints/dpo_full --ppo_model ./checkpoints/ppo_full --output results/dpo_vs_ppo_comparison.json

# Check for catastrophic forgetting
python eval_perplexity.py --model ./checkpoints/dpo_full --dataset wikitext --split test

# Plot implicit rewards during training
python plot_implicit_rewards.py --log_dir ./logs/dpo_full --output plots/implicit_reward_margin.png
```

---

## Technical Implementation Details

### Project Structure

```
crucible/phase2/week10/
├── train_dpo.py              # Main DPO training script
├── train_ipo.py              # IPO variant training
├── dpo_loss.py               # DPO loss implementation
├── logprobs.py               # Log-probability computation
├── eval_judge.py             # LLM-as-judge evaluation
├── eval_perplexity.py        # Catastrophic forgetting check
├── compare_methods.py        # DPO vs PPO comparison
├── plot_implicit_rewards.py  # Training dynamics visualization
├── configs/
│   ├── dpo_default.yaml
│   └── ipo_default.yaml
└── results/
```

### DPO Loss Derivation and Implementation

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps: torch.Tensor,
             policy_rejected_logps: torch.Tensor,
             reference_chosen_logps: torch.Tensor,
             reference_rejected_logps: torch.Tensor,
             beta: float = 0.1) -> tuple[torch.Tensor, dict]:
    """
    Direct Preference Optimization loss.

    Derivation:
    - RLHF objective: max E[r(x,y)] - β * KL(π || π_ref)
    - Optimal policy: π*(y|x) = π_ref(y|x) * exp(r(x,y)/β) / Z(x)
    - Rearranging: r(x,y) = β * log(π*(y|x)/π_ref(y|x)) + β*log(Z(x))
    - Substituting into Bradley-Terry: P(y_w > y_l) = σ(r(y_w) - r(y_l))
    - The partition function Z(x) cancels!
    - Final: P(y_w > y_l) = σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))

    Loss = -log P(y_w > y_l)
    """
    # Log-ratios (implicit rewards)
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    # DPO implicit reward margin
    logits = beta * (chosen_logratios - rejected_logratios)

    # Negative log-sigmoid loss
    loss = -F.logsigmoid(logits).mean()

    # Metrics for monitoring
    metrics = {
        "loss": loss.item(),
        "chosen_reward": chosen_logratios.mean().item(),
        "rejected_reward": rejected_logratios.mean().item(),
        "reward_margin": (chosen_logratios - rejected_logratios).mean().item(),
        "accuracy": (logits > 0).float().mean().item(),
    }

    return loss, metrics
```

### IPO (Identity Preference Optimization) Loss

```python
def ipo_loss(policy_chosen_logps: torch.Tensor,
             policy_rejected_logps: torch.Tensor,
             reference_chosen_logps: torch.Tensor,
             reference_rejected_logps: torch.Tensor,
             tau: float = 0.1) -> tuple[torch.Tensor, dict]:
    """
    IPO loss — avoids overfitting to preferences by using a squared loss
    instead of log-sigmoid. More robust when preferences are noisy.

    Loss = (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x)) - 1/(2τ))^2
    """
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    margin = chosen_logratios - rejected_logratios
    loss = ((margin - 1.0 / (2 * tau)) ** 2).mean()

    metrics = {
        "loss": loss.item(),
        "margin": margin.mean().item(),
        "accuracy": (margin > 0).float().mean().item(),
    }
    return loss, metrics
```

### Log-Probability Computation

```python
def compute_sequence_logprobs(model, input_ids, attention_mask, prompt_lengths):
    """
    Compute sum of log-probabilities for the RESPONSE portion only.

    Critical: we must not include prompt tokens in the log-prob computation,
    only the response tokens. prompt_lengths tells us where the response starts.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (batch, seq_len, vocab)

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    # Per-token log-probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask: only count response tokens (not prompt)
    response_mask = torch.zeros_like(shift_mask)
    for i, pl in enumerate(prompt_lengths):
        response_mask[i, pl-1:] = shift_mask[i, pl-1:]

    # Sum log-probs over response tokens
    sequence_logprobs = (token_log_probs * response_mask).sum(dim=1)

    return sequence_logprobs
```

### DPO Training Loop

```python
def train_dpo(model, ref_model, train_loader, optimizer, scheduler, config):
    """DPO training loop - elegantly simple compared to PPO."""
    model.train()
    ref_model.eval()

    for epoch in range(config.epochs):
        epoch_metrics = []
        for batch_idx, batch in enumerate(train_loader):
            # batch contains: chosen_ids, rejected_ids, attention_masks, prompt_lengths
            chosen_ids = batch["chosen_ids"].to(model.device)
            rejected_ids = batch["rejected_ids"].to(model.device)
            chosen_mask = batch["chosen_mask"].to(model.device)
            rejected_mask = batch["rejected_mask"].to(model.device)
            prompt_lengths = batch["prompt_lengths"]

            # Compute log-probs under current policy
            policy_chosen_logps = compute_sequence_logprobs(model, chosen_ids, chosen_mask, prompt_lengths)
            policy_rejected_logps = compute_sequence_logprobs(model, rejected_ids, rejected_mask, prompt_lengths)

            # Compute log-probs under reference model (no grad needed)
            with torch.no_grad():
                ref_chosen_logps = compute_sequence_logprobs(ref_model, chosen_ids, chosen_mask, prompt_lengths)
                ref_rejected_logps = compute_sequence_logprobs(ref_model, rejected_ids, rejected_mask, prompt_lengths)

            # DPO loss
            loss, metrics = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=config.beta
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_metrics.append(metrics)

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} Step {batch_idx} | Loss: {metrics['loss']:.4f} | "
                      f"Margin: {metrics['reward_margin']:.3f} | Acc: {metrics['accuracy']:.3f}")

        avg_metrics = {k: sum(m[k] for m in epoch_metrics)/len(epoch_metrics) for k in epoch_metrics[0]}
        print(f"Epoch {epoch} complete | {avg_metrics}")
```

### LLM-as-Judge Evaluation

```python
import openai

JUDGE_PROMPT = """You are evaluating two AI assistant responses. Which response is better?

Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Which is better? Answer ONLY "A" or "B" or "tie"."""

def evaluate_with_judge(model_a_responses, model_b_responses, prompts, judge_model="gpt-4"):
    """Evaluate using LLM-as-judge for win-rate computation."""
    wins_a, wins_b, ties = 0, 0, 0

    for prompt, resp_a, resp_b in zip(prompts, model_a_responses, model_b_responses):
        judgment = openai.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                prompt=prompt, response_a=resp_a, response_b=resp_b
            )}],
            max_tokens=5, temperature=0
        ).choices[0].message.content.strip()

        if "A" in judgment:
            wins_a += 1
        elif "B" in judgment:
            wins_b += 1
        else:
            ties += 1

    total = wins_a + wins_b + ties
    return {
        "win_rate_a": wins_a / total,
        "win_rate_b": wins_b / total,
        "tie_rate": ties / total,
    }
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| DPO loss goes to 0 too fast | β too low (model overfits). Increase β to 0.2-0.5 or reduce learning rate. |
| Accuracy stuck at 50% | Log-probs not computed correctly. Verify prompt masking — only response tokens should be scored. |
| Reference model taking too much VRAM | Quantize reference to 4-bit with bitsandbytes, or compute ref logprobs in chunks with offloading. |
| Model generates worse after DPO | β too low causing distribution shift. Or training too long (overfit). Use early stopping on val accuracy. |
| Perplexity spikes (catastrophic forgetting) | Add SFT loss as regularizer: `total_loss = dpo_loss + α * sft_loss` with α=0.1. |
| LLM-as-judge is expensive | Use a smaller judge (Llama-3-70B via API) or compute win-rate on smaller subset (50 prompts). |
| IPO gives worse results than DPO | τ hyperparameter needs tuning. Try τ in {0.01, 0.05, 0.1, 0.5}. |

---

## Agent Handoff Template

```
Continue the Crucible Phase 2, Week 10 (DPO) project.

Hardware: RTX 5080 16GB VRAM, 32GB RAM, Ubuntu.
Project location: crucible/phase2/week10/

Current state: [DESCRIBE WHAT'S DONE]
Blocked on: [DESCRIBE THE ISSUE]

The goal is to implement DPO from scratch, train on HH-RLHF preference data, and compare
against the PPO model from Week 9. Target: >55% win-rate vs SFT baseline.

Key files:
- dpo_loss.py: DPO and IPO loss functions
- logprobs.py: Per-sequence log-probability computation (response-only)
- train_dpo.py: Training loop with reference model

Please [FIX/CONTINUE/DEBUG] the [SPECIFIC COMPONENT].
```

---

## Out of Scope

- Online DPO with iterative generation (future extension)
- KTO, ORPO, SimPO (Week 11)
- Reward model training (Week 8, already done)
- PPO training (Week 9, already done)
- Multi-turn DPO (Week 13)
- DPO for code generation or specific domains
- Theoretical analysis of DPO convergence properties
- Constitutional AI integration (Week 12)
