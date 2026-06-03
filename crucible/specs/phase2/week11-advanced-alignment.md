# Week 11: Advanced Alignment Methods

## Context

**Where it fits:** Phase 2 (Alignment Deep Dive), Week 4 of 7. Having implemented the two foundational approaches (PPO in Week 9, DPO in Week 10), you now implement the cutting-edge alternatives that address their limitations.

**Prerequisites:**
- Week 9: PPO-RLHF (understand its cost and instability)
- Week 10: DPO (understand its simplicity but also its failure modes)
- Same preference dataset (HH-RLHF) for fair comparison
- Mathematical comfort with log-ratios, sigmoid losses, odds ratios

**What it builds on:** Each method in this week solves a specific problem with DPO or PPO:
- KTO: solves "I don't have paired preferences, only thumbs up/down"
- ORPO: solves "I want to combine SFT and alignment in one step"
- SimPO: solves "reference model is expensive and introduces staleness"

**What it enables:** Week 14's comparison report will rank all methods. This week provides the most methods to compare. Understanding these positions you as someone who knows the full landscape, not just the two famous ones.

---

## Learning Goals

- [ ] Explain KTO's insight: human utility is asymmetric (losses loom larger than gains — Kahneman-Tversky prospect theory)
- [ ] Derive ORPO's odds ratio: why odds(chosen)/odds(rejected) replaces explicit preference modeling
- [ ] Understand SimPO: why length normalization + reference-free works (implicit reference via margin)
- [ ] Articulate when each method is appropriate given data constraints
- [ ] Explain the data requirements spectrum: RLHF (paired + RM) > DPO (paired) > KTO (unpaired) > ORPO (SFT data with quality labels)
- [ ] Understand statistical comparison: why multiple seeds and significance tests matter
- [ ] Compare compute costs: which methods need a reference model? which can combine with SFT?

---

## Implementation Goals

- [ ] Implement KTO loss function: separate treatment for desirable/undesirable examples
- [ ] Implement ORPO loss function: SFT loss + odds ratio penalty
- [ ] Implement SimPO loss function: length-normalized, reference-free margin
- [ ] Train each method on same data (HH-RLHF or derived unpaired version for KTO)
- [ ] Run with 3 different random seeds per method for statistical significance
- [ ] Build comparison table: compute time, VRAM, accuracy, win-rate for all 5 methods (PPO, DPO, KTO, ORPO, SimPO)
- [ ] Compute confidence intervals and statistical significance (paired bootstrap)
- [ ] Publish results in structured JSON + markdown report

---

## Acceptance Criteria

1. KTO loss correctly handles asymmetric treatment: desirable examples use `1 - σ(β * (log_ratio - KL_ref))`, undesirable use `1 - σ(β * (KL_ref - log_ratio))`.
2. ORPO loss combines negative log-likelihood with log-odds ratio: `L = L_NLL + λ * L_OR` where L_OR = -log(σ(log(odds_w/odds_l)))`.
3. SimPO loss is reference-free: uses average log-prob as reward with length normalization and a target margin γ.
4. Each method trains to convergence in under 3 hours on RTX 5080 16GB for 1B model.
5. KTO works with unpaired data (randomly split chosen/rejected into separate "good" and "bad" pools).
6. All methods run with 3 seeds; standard deviation is reported for every metric.
7. Comparison table with all 5+ methods exists with: training time, VRAM peak, val accuracy, win-rate vs SFT, win-rate vs each other.
8. At least one method achieves >60% win-rate vs SFT baseline (higher bar than DPO alone).
9. Statistical significance computed using paired bootstrap test (p < 0.05 to claim one method beats another).
10. Markdown report (2-3 pages) summarizes findings with clear recommendations for practitioners.

---

## Validation Commands

```bash
# KTO training
python train_kto.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset anthropic_hh --beta 0.1 --epochs 3 --seed 42 --output_dir ./checkpoints/kto_seed42

# ORPO training
python train_orpo.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset anthropic_hh --lambda_or 1.0 --epochs 3 --seed 42 --output_dir ./checkpoints/orpo_seed42

# SimPO training
python train_simpo.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset anthropic_hh --gamma 1.0 --beta 2.0 --epochs 3 --seed 42 --output_dir ./checkpoints/simpo_seed42

# Multi-seed runs
for method in kto orpo simpo; do
    for seed in 42 123 456; do
        python train_${method}.py --seed $seed --output_dir ./checkpoints/${method}_seed${seed} --log_dir ./logs/${method}_seed${seed}
    done
done

# Build comparison table
python build_comparison.py --methods ppo,dpo,kto,orpo,simpo --results_dir ./results --output results/full_comparison.json

# Statistical significance
python significance_test.py --results_dir ./results --method bootstrap --num_samples 10000

# Generate report
python generate_report.py --comparison results/full_comparison.json --output results/week11_report.md
```

---

## Technical Implementation Details

### Project Structure

```
crucible/phase2/week11/
├── train_kto.py              # KTO training
├── train_orpo.py             # ORPO training
├── train_simpo.py            # SimPO training
├── losses/
│   ├── kto.py                # KTO loss function
│   ├── orpo.py               # ORPO loss function
│   └── simpo.py              # SimPO loss function
├── build_comparison.py       # Multi-method comparison table
├── significance_test.py      # Statistical significance
├── generate_report.py        # Markdown report generation
├── configs/
│   ├── kto.yaml
│   ├── orpo.yaml
│   └── simpo.yaml
└── results/
```

### KTO (Kahneman-Tversky Optimization) Loss

```python
import torch
import torch.nn.functional as F

def kto_loss(policy_logps: torch.Tensor,
             reference_logps: torch.Tensor,
             is_desirable: torch.BoolTensor,
             beta: float = 0.1) -> tuple[torch.Tensor, dict]:
    """
    KTO loss from "KTO: Model Alignment as Prospect Theoretic Optimization"

    Key insight: Humans are loss-averse (Kahneman & Tversky).
    Losing utility hurts more than gaining it feels good.

    Unlike DPO, KTO doesn't need paired preferences — just binary labels
    (good/bad) for individual responses.

    For desirable (y_w): loss = 1 - σ(β * (log π(y|x)/π_ref(y|x) - E[KL]))
    For undesirable (y_l): loss = 1 - σ(β * (E[KL] - log π(y|x)/π_ref(y|x)))

    The E[KL] term is the expected KL divergence, estimated from the batch.
    """
    log_ratios = policy_logps - reference_logps

    # Estimate KL from reference (using the batch)
    kl_estimate = (policy_logps.exp() * (policy_logps - reference_logps)).mean().detach()

    # Split into desirable and undesirable
    desirable_logratios = log_ratios[is_desirable]
    undesirable_logratios = log_ratios[~is_desirable]

    # KTO losses with asymmetric treatment
    desirable_loss = torch.tensor(0.0, device=policy_logps.device)
    undesirable_loss = torch.tensor(0.0, device=policy_logps.device)

    if desirable_logratios.numel() > 0:
        desirable_loss = (1 - F.sigmoid(beta * (desirable_logratios - kl_estimate))).mean()

    if undesirable_logratios.numel() > 0:
        undesirable_loss = (1 - F.sigmoid(beta * (kl_estimate - undesirable_logratios))).mean()

    # Combined loss (λ_D and λ_U weight the two components)
    n_desirable = is_desirable.sum().float()
    n_undesirable = (~is_desirable).sum().float()
    lambda_d = n_undesirable / (n_desirable + n_undesirable)
    lambda_u = n_desirable / (n_desirable + n_undesirable)

    loss = lambda_d * desirable_loss + lambda_u * undesirable_loss

    metrics = {
        "loss": loss.item(),
        "desirable_loss": desirable_loss.item(),
        "undesirable_loss": undesirable_loss.item(),
        "kl_estimate": kl_estimate.item(),
        "desirable_mean_logratio": desirable_logratios.mean().item() if desirable_logratios.numel() > 0 else 0,
        "undesirable_mean_logratio": undesirable_logratios.mean().item() if undesirable_logratios.numel() > 0 else 0,
    }
    return loss, metrics
```

### ORPO (Odds Ratio Preference Optimization) Loss

```python
def orpo_loss(model_outputs_chosen, model_outputs_rejected,
              chosen_labels, rejected_labels,
              lambda_or: float = 1.0) -> tuple[torch.Tensor, dict]:
    """
    ORPO: Monolithic Preference Optimization without Reference Model.

    Key insight: Combine SFT objective with preference optimization.
    No need for a separate reference model — the SFT loss acts as regularizer.

    L_ORPO = L_SFT + λ * L_OR
    L_SFT = standard cross-entropy on chosen responses
    L_OR = -log(σ(log(odds_chosen / odds_rejected)))

    Where odds(y|x) = P(y|x) / (1 - P(y|x)) computed from average token probability.
    """
    # SFT loss on chosen responses
    shift_logits = model_outputs_chosen.logits[:, :-1, :]
    shift_labels = chosen_labels[:, 1:]
    sft_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100
    )

    # Compute average token probabilities for odds ratio
    def compute_avg_prob(logits, labels):
        """Average probability of the correct token across the sequence."""
        probs = F.softmax(logits[:, :-1, :], dim=-1)
        token_probs = probs.gather(2, labels[:, 1:].unsqueeze(-1)).squeeze(-1)
        mask = (labels[:, 1:] != -100).float()
        avg_prob = (token_probs * mask).sum(dim=1) / mask.sum(dim=1)
        return avg_prob

    prob_chosen = compute_avg_prob(model_outputs_chosen.logits, chosen_labels)
    prob_rejected = compute_avg_prob(model_outputs_rejected.logits, rejected_labels)

    # Odds ratio
    odds_chosen = prob_chosen / (1 - prob_chosen + 1e-8)
    odds_rejected = prob_rejected / (1 - prob_rejected + 1e-8)

    log_odds_ratio = torch.log(odds_chosen / (odds_rejected + 1e-8) + 1e-8)
    or_loss = -F.logsigmoid(log_odds_ratio).mean()

    # Combined
    total_loss = sft_loss + lambda_or * or_loss

    metrics = {
        "loss": total_loss.item(),
        "sft_loss": sft_loss.item(),
        "or_loss": or_loss.item(),
        "prob_chosen": prob_chosen.mean().item(),
        "prob_rejected": prob_rejected.mean().item(),
        "odds_ratio": (odds_chosen / (odds_rejected + 1e-8)).mean().item(),
    }
    return total_loss, metrics
```

### SimPO (Simple Preference Optimization) Loss

```python
def simpo_loss(policy_chosen_logps: torch.Tensor,
               policy_rejected_logps: torch.Tensor,
               chosen_lengths: torch.Tensor,
               rejected_lengths: torch.Tensor,
               beta: float = 2.0,
               gamma: float = 1.0) -> tuple[torch.Tensor, dict]:
    """
    SimPO: Simple Preference Optimization with a Reference-Free Reward.

    Key insights:
    1. No reference model needed (saves memory and removes staleness issue)
    2. Uses average log-probability as implicit reward (length-normalized)
    3. Adds a target reward margin γ to push chosen further above rejected

    Reward: r(y|x) = (1/|y|) * Σ log π(y_t|x, y_<t)  (average log-prob)
    Loss: -log σ(β * (r(y_w) - r(y_l) - γ))

    The γ margin prevents the loss from being satisfied by trivially small differences.
    """
    # Length-normalized rewards (average log-prob per token)
    chosen_rewards = policy_chosen_logps / chosen_lengths
    rejected_rewards = policy_rejected_logps / rejected_lengths

    # SimPO loss with margin
    logits = beta * (chosen_rewards - rejected_rewards - gamma)
    loss = -F.logsigmoid(logits).mean()

    metrics = {
        "loss": loss.item(),
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        "accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
    }
    return loss, metrics
```

### Statistical Comparison

```python
import numpy as np

def paired_bootstrap_test(scores_a, scores_b, num_samples=10000, confidence=0.95):
    """
    Paired bootstrap test for statistical significance.
    Tests whether method A is significantly better than method B.
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    n = len(scores_a)
    assert len(scores_b) == n

    observed_diff = scores_a.mean() - scores_b.mean()

    bootstrap_diffs = []
    for _ in range(num_samples):
        indices = np.random.randint(0, n, size=n)
        diff = scores_a[indices].mean() - scores_b[indices].mean()
        bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)
    p_value = (bootstrap_diffs <= 0).mean()  # One-sided: P(A not better than B)

    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "significant": p_value < alpha,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }

def build_comparison_table(results: dict) -> str:
    """Build markdown comparison table from all method results."""
    methods = list(results.keys())
    header = "| Method | Train Time | VRAM | Val Acc | Win-Rate vs SFT | Win-Rate vs DPO |"
    separator = "|--------|-----------|------|---------|-----------------|-----------------|"
    rows = [header, separator]

    for method in methods:
        r = results[method]
        row = f"| {method} | {r['time_hrs']:.1f}h | {r['vram_gb']:.1f}GB | {r['val_acc']:.1%} | {r['winrate_sft']:.1%} | {r['winrate_dpo']:.1%} |"
        rows.append(row)

    return "\n".join(rows)
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| KTO: desirable/undesirable imbalanced | Balance with λ_D and λ_U weights (already in implementation). Or subsample majority class. |
| ORPO: SFT loss dominates | Increase λ_or to 2.0-5.0 to give odds ratio more weight. |
| SimPO: length normalization gives weird results | Ensure chosen/rejected lengths exclude prompt tokens. Minimum length threshold of 5 tokens. |
| OOM with multiple training runs | Run sequentially, clear CUDA cache between runs: `torch.cuda.empty_cache()`. |
| Results not reproducible across seeds | Set all seeds: `torch.manual_seed(seed)`, `np.random.seed(seed)`, `torch.cuda.manual_seed_all(seed)`. Set `torch.backends.cudnn.deterministic = True`. |
| No significant difference between methods | Need more eval samples. Increase from 100 to 500 test prompts. Or accept that methods are similar (valid finding). |
| ORPO training unstable | Lower learning rate for ORPO (3e-6 instead of 5e-6). The SFT component can cause instability if lr too high. |

---

## Agent Handoff Template

```
Continue the Crucible Phase 2, Week 11 (Advanced Alignment Methods) project.

Hardware: RTX 5080 16GB VRAM, 32GB RAM, Ubuntu.
Project location: crucible/phase2/week11/

Current state: [DESCRIBE WHAT'S DONE - e.g., "KTO trains, ORPO implemented but untested, SimPO not started"]
Blocked on: [DESCRIBE THE ISSUE]

The goal is to implement KTO, ORPO, and SimPO from scratch, train each with 3 seeds,
and build a comparison table with statistical significance against DPO/PPO baselines.

Key files:
- losses/kto.py: KTO loss (asymmetric, unpaired preferences)
- losses/orpo.py: ORPO loss (SFT + odds ratio, no reference model)
- losses/simpo.py: SimPO loss (reference-free, length-normalized)
- build_comparison.py: Generates comparison table from all runs

Please [FIX/CONTINUE/DEBUG] the [SPECIFIC COMPONENT].
```

---

## Out of Scope

- Reward model training (Week 8)
- PPO implementation (Week 9)
- DPO implementation (Week 10)
- Constitutional AI (Week 12)
- Novel alignment methods not in papers (research contribution)
- Methods requiring >16GB VRAM without optimization
- Theoretical proofs of convergence for each method
- Human evaluation (too expensive for this phase)
