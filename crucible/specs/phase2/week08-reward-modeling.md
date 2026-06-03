# Week 8: Reward Modeling

## Context

**Where it fits:** Phase 2 (Alignment Deep Dive), Week 1 of 7. This is the foundation for everything that follows — RLHF, DPO, and all alignment methods require a reward signal.

**Prerequisites:**
- Phase 1 complete: custom training loop, mixed precision, LoRA/QLoRA, SFT pipeline
- Working SFT model (your Week 7 output)
- Understanding of classification heads on top of language models
- Familiarity with binary cross-entropy loss

**What it builds on:** Your SFT pipeline provides the base model architecture. The reward model reuses the same transformer backbone but replaces the language modeling head with a scalar reward head. Your experience with LoRA will help when training reward models efficiently.

**What it enables:** Weeks 9-14 all depend on having a working reward model. PPO uses it for scoring, DPO implicitly learns one, and your final comparison report requires it as a baseline.

---

## Learning Goals

- [ ] Understand the Bradley-Terry preference model mathematically: P(a ≻ b) = σ(r(a) - r(b))
- [ ] Explain why reward modeling is a reduction from ranking to classification
- [ ] Describe reward hacking: length bias, sycophancy, reward model exploitation
- [ ] Understand calibration: why raw reward scores must be interpretable
- [ ] Articulate the difference between pointwise reward and pairwise preference
- [ ] Explain why reward model size matters (capacity vs overfitting tradeoff)
- [ ] Describe how reward model quality bounds alignment quality (GIGO principle)

---

## Implementation Goals

- [ ] Load and preprocess HH-RLHF preference dataset (chosen/rejected pairs)
- [ ] Implement reward model architecture: base LM + scalar head on last token
- [ ] Implement Bradley-Terry loss function from scratch (no library calls)
- [ ] Train reward model with proper train/val split
- [ ] Evaluate: accuracy on held-out preferences (target >65%)
- [ ] Analyze reward distributions: plot chosen vs rejected score histograms
- [ ] Detect reward hacking signals: correlation between reward and response length
- [ ] Compare 1B vs 3B base model for reward modeling (accuracy vs compute)
- [ ] Save reward model for use in Week 9 (PPO) and Week 10 (DPO comparison)

---

## Acceptance Criteria

1. Reward model loads a pretrained LM and adds a scalar output head that produces a single float per sequence.
2. Bradley-Terry loss is implemented from scratch and matches: `-log(sigmoid(r_chosen - r_rejected))` averaged over batch.
3. Training loop handles preference pairs correctly — same prompt, two completions, one label.
4. Held-out accuracy exceeds 65% on Anthropic HH-RLHF test split (random baseline = 50%).
5. Reward distribution plot shows clear separation between chosen and rejected responses (mean difference > 0.5 reward units).
6. Length bias analysis shows Pearson correlation between reward score and token count (document the value, flag if |r| > 0.4).
7. Training runs in under 4 hours on RTX 5080 16GB for the 1B reward model variant.
8. Model checkpoint is saved in a format loadable by Week 9's PPO pipeline (state_dict + config).
9. Comparison table exists showing 1B vs 3B reward model: accuracy, training time, VRAM usage.
10. Reward model can score arbitrary (prompt, response) pairs via a clean inference API: `reward_model.score(prompt, response) -> float`.

---

## Validation Commands

```bash
# Verify dataset downloaded and preprocessed
python -c "from datasets import load_dataset; ds = load_dataset('Anthropic/hh-rlhf', split='test[:100]'); print(f'Loaded {len(ds)} examples, keys: {ds[0].keys()}')"

# Run reward model training (smoke test with small subset)
python train_reward_model.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset anthropic_hh --max_samples 500 --epochs 1 --batch_size 4 --output_dir ./checkpoints/reward_model_smoke

# Evaluate on held-out set
python eval_reward_model.py --checkpoint ./checkpoints/reward_model --split test --metrics accuracy,calibration,length_correlation

# Score a sample response
python -c "
from reward_model import RewardModel
rm = RewardModel.load('./checkpoints/reward_model')
score = rm.score('What is Python?', 'Python is a programming language.')
print(f'Reward score: {score:.4f}')
"

# Compare base model sizes
python compare_reward_models.py --models 1B,3B --dataset anthropic_hh --output results/reward_model_comparison.json

# Plot reward distributions
python plot_rewards.py --checkpoint ./checkpoints/reward_model --split test --output plots/reward_distribution.png

# Full training run (expect ~3-4 hours on RTX 5080)
python train_reward_model.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset anthropic_hh --epochs 3 --batch_size 8 --lr 1e-5 --output_dir ./checkpoints/reward_model_1B
```

---

## Technical Implementation Details

### Project Structure

```
crucible/phase2/week08/
├── train_reward_model.py       # Main training script
├── eval_reward_model.py        # Evaluation and metrics
├── reward_model.py             # Model architecture
├── data.py                     # Dataset loading and preprocessing
├── compare_reward_models.py    # 1B vs 3B comparison
├── plot_rewards.py             # Visualization
├── configs/
│   ├── reward_1b.yaml
│   └── reward_3b.yaml
└── results/
    ├── reward_model_comparison.json
    └── plots/
```

### Reward Model Architecture

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class RewardModel(nn.Module):
    def __init__(self, model_name: str, dtype=torch.bfloat16):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, attn_implementation="flash_attention_2"
        )
        # Remove the LM head, keep the transformer
        self.backbone.lm_head = nn.Identity()
        hidden_size = self.backbone.config.hidden_size

        # Scalar reward head: projects last hidden state to single value
        self.reward_head = nn.Linear(hidden_size, 1, dtype=dtype)
        nn.init.zeros_(self.reward_head.bias)
        nn.init.normal_(self.reward_head.weight, std=1 / (hidden_size + 1))

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state

        # Get hidden state at the last non-padding token
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_idx, sequence_lengths]

        reward = self.reward_head(last_hidden).squeeze(-1)
        return reward

    def score(self, prompt: str, response: str) -> float:
        """Clean inference API for scoring a single (prompt, response) pair."""
        text = f"{prompt}\n{response}"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.backbone.device) for k, v in inputs.items()}
        with torch.no_grad():
            reward = self.forward(**inputs)
        return reward.item()
```

### Bradley-Terry Loss

```python
def bradley_terry_loss(rewards_chosen: torch.Tensor, rewards_rejected: torch.Tensor) -> torch.Tensor:
    """
    Bradley-Terry preference loss.
    P(chosen > rejected) = sigmoid(r(chosen) - r(rejected))
    Loss = -log(P(chosen > rejected)) = -log(sigmoid(r_chosen - r_rejected))

    This is equivalent to binary cross-entropy where the label is always 1
    (chosen is always preferred).
    """
    return -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
```

### Data Preprocessing

```python
from datasets import load_dataset

def load_preference_data(dataset_name="Anthropic/hh-rlhf", max_samples=None):
    """Load and format preference pairs from HH-RLHF."""
    ds = load_dataset(dataset_name)

    def format_example(example):
        # HH-RLHF has 'chosen' and 'rejected' as full conversation strings
        return {
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    train_ds = ds["train"].map(format_example)
    test_ds = ds["test"].map(format_example)

    if max_samples:
        train_ds = train_ds.select(range(min(max_samples, len(train_ds))))

    return train_ds, test_ds

def collate_preference_pairs(batch, tokenizer, max_length=1024):
    """Tokenize chosen and rejected into a single batch."""
    chosen_texts = [ex["chosen"] for ex in batch]
    rejected_texts = [ex["rejected"] for ex in batch]

    chosen_enc = tokenizer(
        chosen_texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt"
    )
    rejected_enc = tokenizer(
        rejected_texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt"
    )
    return chosen_enc, rejected_enc
```

### Training Loop

```python
def train_reward_model(model, train_loader, optimizer, scheduler, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (chosen_enc, rejected_enc) in enumerate(train_loader):
            chosen_enc = {k: v.to(device) for k, v in chosen_enc.items()}
            rejected_enc = {k: v.to(device) for k, v in rejected_enc.items()}

            rewards_chosen = model(**chosen_enc)
            rewards_rejected = model(**rejected_enc)

            loss = bradley_terry_loss(rewards_chosen, rewards_rejected)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Track accuracy: chosen should have higher reward
            correct += (rewards_chosen > rewards_rejected).sum().item()
            total += rewards_chosen.size(0)
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                acc = correct / total if total > 0 else 0
                print(f"Epoch {epoch} | Step {batch_idx} | Loss: {loss.item():.4f} | Acc: {acc:.3f}")

        epoch_acc = correct / total
        print(f"Epoch {epoch} complete | Avg Loss: {total_loss/len(train_loader):.4f} | Acc: {epoch_acc:.3f}")
```

### Reward Hacking Detection

```python
import numpy as np
from scipy.stats import pearsonr

def detect_length_bias(model, dataset, tokenizer, device, num_samples=1000):
    """Check if reward correlates with response length."""
    rewards = []
    lengths = []

    for i in range(min(num_samples, len(dataset))):
        text = dataset[i]["chosen"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            reward = model(**inputs).item()

        rewards.append(reward)
        lengths.append(len(tokenizer.encode(text)))

    correlation, p_value = pearsonr(lengths, rewards)
    print(f"Length-reward correlation: r={correlation:.3f}, p={p_value:.4f}")

    if abs(correlation) > 0.4:
        print("WARNING: Significant length bias detected!")

    return correlation, p_value
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| OOM during training | Reduce batch_size to 2, enable gradient checkpointing: `model.backbone.gradient_checkpointing_enable()` |
| Accuracy stuck at 50% | Check data loading — ensure chosen/rejected aren't swapped. Verify loss is decreasing. |
| Loss is NaN | Reduce learning rate to 1e-6, add gradient clipping (already in code). Check for empty sequences. |
| Reward scores all same value | Reward head may not be training. Check that `reward_head` parameters have `requires_grad=True`. |
| Training too slow | Use `torch.compile(model)`, ensure Flash Attention 2 is working, use bf16. |
| HH-RLHF download fails | Use `datasets` cache: `HF_DATASETS_CACHE=~/.cache/huggingface/datasets`. Or use UltraFeedback as alternative. |
| 3B model OOM | Use LoRA on backbone (only train reward_head + LoRA adapters), or use gradient accumulation. |

---

## Agent Handoff Template

```
Continue the Crucible Phase 2, Week 8 (Reward Modeling) project.

Hardware: RTX 5080 16GB VRAM, 32GB RAM, Ubuntu.
Project location: crucible/phase2/week08/

Current state: [DESCRIBE WHAT'S DONE]
Blocked on: [DESCRIBE THE ISSUE]

The goal is to train a reward model on Anthropic HH-RLHF preference data using the
Bradley-Terry framework. The model should achieve >65% accuracy on held-out preferences.

Key files:
- reward_model.py: RewardModel class (base LM + scalar head)
- train_reward_model.py: Training loop with BT loss
- eval_reward_model.py: Accuracy, calibration, length bias analysis

Please [FIX/CONTINUE/DEBUG] the [SPECIFIC COMPONENT].
```

---

## Out of Scope

- RLHF/PPO training (Week 9)
- DPO or other preference optimization methods (Week 10+)
- Multi-objective reward models (balancing helpfulness vs harmlessness)
- Reward model ensembles
- Process reward models (step-level rewards for reasoning)
- Online/iterative reward model training
- Human annotation interface or RLHF data collection
- Deployment or serving the reward model in production
