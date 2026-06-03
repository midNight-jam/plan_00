# Week 13: Multi-Turn Alignment

## Context

**Where it fits:** Phase 2 (Alignment Deep Dive), Week 6 of 7. Most alignment research operates on single-turn (prompt → response) data. Real deployments are multi-turn conversations where quality accumulates or degrades across turns.

**Prerequisites:**
- Weeks 9-10: PPO and DPO (used as optimizers for conversation-level rewards)
- Week 8: Reward model (extended to score full conversations)
- Understanding of conversation data formats (multi-turn chat templates)
- Working SFT model that can handle multi-turn conversations

**What it builds on:** Single-turn alignment from Weeks 9-11 gives you the optimization tools. This week extends them to the harder setting where reward assignment across turns, credit assignment, and growing context create new challenges.

**What it enables:** Week 14's comparison report includes multi-turn as a dimension. Multi-turn alignment is highly relevant to production systems (ChatGPT, Claude) and is an active research frontier.

---

## Learning Goals

- [ ] Explain why single-turn alignment is insufficient: models can be aligned per-response but produce poor conversations
- [ ] Understand the credit assignment problem: which turn in a 5-turn conversation was responsible for the final quality?
- [ ] Articulate conversation-level vs turn-level rewards and when each is appropriate
- [ ] Explain context handling challenges: KV cache growth, attention to earlier turns, recency bias
- [ ] Understand conversation trees: branching at decision points to compare alternative continuations
- [ ] Explain rejection sampling and Best-of-N: generate many, keep the best (inference-time alignment)
- [ ] Describe how multi-turn RLHF differs from single-turn (longer episodes, sparse rewards)

---

## Implementation Goals

- [ ] Implement conversation-level reward scoring (reward model scores full conversation)
- [ ] Implement turn-level reward decomposition: assign per-turn rewards from conversation-level score
- [ ] Implement conversation tree construction: branch at each turn, compare alternatives
- [ ] Extend DPO to multi-turn: preference pairs are full conversations, not single responses
- [ ] Implement rejection sampling at scale: generate K completions per turn, keep best
- [ ] Implement Best-of-N sampling at inference: generate N, select by reward model
- [ ] Build multi-turn preference dataset from conversation trees
- [ ] Compare single-turn vs multi-turn aligned models on MT-Bench

---

## Acceptance Criteria

1. Conversation-level reward model scores full multi-turn conversations (not just last response) and produces meaningful differentiation (std > 0.3 across conversations).
2. Turn-level reward decomposition assigns per-turn credit that sums approximately to the conversation-level reward.
3. Conversation tree generates at least 3 alternative continuations at each branch point with measurably different reward scores.
4. Multi-turn DPO trains on conversation-level preference pairs (chosen conversation vs rejected conversation).
5. Rejection sampling generates K=8 alternatives per turn and selects top-1 by reward, improving conversation quality by >10% reward increase.
6. Best-of-N inference with N=4 produces higher-reward responses than greedy decoding on 80%+ of test prompts.
7. Multi-turn aligned model outperforms single-turn aligned model on MT-Bench or similar conversation benchmark by at least 0.3 points.
8. Training handles growing context correctly: no OOM for conversations up to 8 turns (up to 4096 tokens).
9. Credit assignment is validated: turns identified as high-reward actually contain better responses (verified by reward model scoring individual turns).
10. Full multi-turn pipeline (tree generation → preference extraction → DPO training) completes in under 10 hours.

---

## Validation Commands

```bash
# Score full conversations with reward model
python score_conversations.py --model ./checkpoints/reward_model_1B --conversations data/multiturn_test.jsonl --output results/conversation_scores.jsonl

# Decompose conversation reward into per-turn rewards
python turn_rewards.py --model ./checkpoints/reward_model_1B --conversations data/multiturn_test.jsonl --method shapley --output results/turn_rewards.jsonl

# Generate conversation trees
python generate_tree.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompts data/conversation_starts.jsonl --branches_per_turn 4 --max_turns 5 --output data/conversation_trees.jsonl

# Extract preference pairs from trees
python extract_preferences.py --trees data/conversation_trees.jsonl --reward_model ./checkpoints/reward_model_1B --output data/multiturn_preferences.jsonl

# Multi-turn DPO training
python train_multiturn_dpo.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --preferences data/multiturn_preferences.jsonl --beta 0.1 --epochs 3 --output_dir ./checkpoints/multiturn_dpo

# Rejection sampling
python rejection_sampling.py --model ./checkpoints/multiturn_dpo --prompts data/conversation_starts.jsonl --k 8 --reward_model ./checkpoints/reward_model_1B --output results/rejection_sampled.jsonl

# Best-of-N evaluation
python best_of_n.py --model ./checkpoints/multiturn_dpo --prompts data/eval_prompts.jsonl --n 4 --reward_model ./checkpoints/reward_model_1B --output results/best_of_n_eval.json

# Compare single-turn vs multi-turn alignment
python compare_alignment.py --single_turn_model ./checkpoints/dpo_full --multi_turn_model ./checkpoints/multiturn_dpo --benchmark mt_bench --output results/single_vs_multi.json
```

---

## Technical Implementation Details

### Project Structure

```
crucible/phase2/week13/
├── score_conversations.py       # Conversation-level reward scoring
├── turn_rewards.py              # Per-turn reward decomposition
├── generate_tree.py             # Conversation tree generation
├── extract_preferences.py       # Preference pairs from trees
├── train_multiturn_dpo.py       # Multi-turn DPO training
├── rejection_sampling.py        # K-of-N generation + selection
├── best_of_n.py                 # Inference-time Best-of-N
├── compare_alignment.py         # Single-turn vs multi-turn eval
├── data/
│   ├── conversation_starts.jsonl
│   ├── multiturn_test.jsonl
│   └── conversation_trees.jsonl
├── configs/
│   └── multiturn_dpo.yaml
└── results/
```

### Conversation-Level Reward Scoring

```python
import torch
from transformers import AutoTokenizer

class ConversationRewardModel:
    """Extends reward model to score full multi-turn conversations."""

    def __init__(self, reward_model, tokenizer, max_length=4096):
        self.model = reward_model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def format_conversation(self, turns: list[dict]) -> str:
        """Format multi-turn conversation into a single string."""
        formatted = ""
        for turn in turns:
            role = turn["role"]
            content = turn["content"]
            if role == "user":
                formatted += f"Human: {content}\n\n"
            else:
                formatted += f"Assistant: {content}\n\n"
        return formatted.strip()

    def score_conversation(self, turns: list[dict]) -> float:
        """Score an entire conversation."""
        text = self.format_conversation(turns)
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.model.device)

        with torch.no_grad():
            reward = self.model(**inputs)
        return reward.item()

    def score_up_to_turn(self, turns: list[dict], turn_idx: int) -> float:
        """Score conversation up to (and including) a specific turn."""
        partial_turns = turns[:turn_idx + 1]
        return self.score_conversation(partial_turns)
```

### Turn-Level Credit Assignment

```python
import numpy as np
from itertools import combinations

def shapley_turn_rewards(conversation: list[dict], reward_model, num_samples=100) -> list[float]:
    """
    Decompose conversation reward into per-turn credits using Shapley values.

    Shapley value for turn i = average marginal contribution of turn i
    across all possible orderings of turns.

    For efficiency, we approximate with Monte Carlo sampling.
    """
    n_turns = len([t for t in conversation if t["role"] == "assistant"])
    assistant_indices = [i for i, t in enumerate(conversation) if t["role"] == "assistant"]

    turn_values = np.zeros(n_turns)

    for _ in range(num_samples):
        perm = np.random.permutation(n_turns)

        for position in range(n_turns):
            # Coalition without current turn
            coalition_without = set(perm[:position])
            # Coalition with current turn
            coalition_with = coalition_without | {perm[position]}

            # Build conversations for each coalition
            turns_without = build_partial_conversation(conversation, assistant_indices, coalition_without)
            turns_with = build_partial_conversation(conversation, assistant_indices, coalition_with)

            score_without = reward_model.score_conversation(turns_without) if turns_without else 0
            score_with = reward_model.score_conversation(turns_with)

            marginal = score_with - score_without
            turn_values[perm[position]] += marginal / num_samples

    return turn_values.tolist()


def difference_decomposition(conversation: list[dict], reward_model) -> list[float]:
    """
    Simpler credit assignment: reward at turn t = score(conv[:t+1]) - score(conv[:t])

    Less theoretically justified than Shapley but much cheaper to compute.
    """
    turn_rewards = []
    prev_score = 0.0

    for i in range(len(conversation)):
        if conversation[i]["role"] == "assistant":
            score = reward_model.score_up_to_turn(conversation, i)
            turn_rewards.append(score - prev_score)
            prev_score = score

    return turn_rewards
```

### Conversation Tree Generation

```python
class ConversationTree:
    """Generate branching conversation trees for preference data extraction."""

    def __init__(self, model, tokenizer, reward_model, branches_per_turn=4, max_turns=5):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.branches_per_turn = branches_per_turn
        self.max_turns = max_turns

    def generate_tree(self, initial_prompt: str) -> dict:
        """Generate a conversation tree by branching at each assistant turn."""
        root = {
            "turns": [{"role": "user", "content": initial_prompt}],
            "children": [],
            "reward": None,
        }

        self._expand_node(root, depth=0)
        return root

    def _expand_node(self, node, depth):
        """Recursively expand a tree node with multiple response branches."""
        if depth >= self.max_turns:
            node["reward"] = self.reward_model.score_conversation(node["turns"])
            return

        # Generate multiple assistant responses (branches)
        for _ in range(self.branches_per_turn):
            response = self._generate_response(node["turns"])
            child_turns = node["turns"] + [{"role": "assistant", "content": response}]

            # Add a follow-up user turn (simulated or from dataset)
            if depth < self.max_turns - 1:
                follow_up = self._generate_user_followup(child_turns)
                child_turns.append({"role": "user", "content": follow_up})

            child = {
                "turns": child_turns,
                "children": [],
                "reward": None,
            }

            node["children"].append(child)
            self._expand_node(child, depth + 1)

    def _generate_response(self, turns: list[dict]) -> str:
        """Generate one assistant response given conversation history."""
        prompt = self._format_for_generation(turns)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=256, temperature=0.8, top_p=0.9, do_sample=True
            )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response

    def extract_preferences(self, tree: dict) -> list[dict]:
        """Extract preference pairs from tree (best vs worst at each branch point)."""
        preferences = []
        self._extract_from_node(tree, preferences)
        return preferences

    def _extract_from_node(self, node, preferences):
        """Find best/worst children at each node and create preference pair."""
        if not node["children"]:
            return

        # Score all children
        scored_children = []
        for child in node["children"]:
            reward = self._get_subtree_reward(child)
            scored_children.append((child, reward))

        scored_children.sort(key=lambda x: x[1], reverse=True)

        # Best vs worst as preference pair
        if len(scored_children) >= 2:
            best = scored_children[0][0]
            worst = scored_children[-1][0]
            preferences.append({
                "chosen": best["turns"],
                "rejected": worst["turns"],
                "reward_diff": scored_children[0][1] - scored_children[-1][1],
            })

        # Recurse into children
        for child in node["children"]:
            self._extract_from_node(child, preferences)

    def _get_subtree_reward(self, node) -> float:
        """Get reward for a subtree (leaf reward or average of children)."""
        if node["reward"] is not None:
            return node["reward"]
        if not node["children"]:
            node["reward"] = self.reward_model.score_conversation(node["turns"])
            return node["reward"]
        return np.mean([self._get_subtree_reward(c) for c in node["children"]])
```

### Multi-Turn DPO

```python
def multiturn_dpo_loss(model, ref_model, chosen_conversations, rejected_conversations,
                       tokenizer, beta=0.1):
    """
    DPO loss applied to full conversations.
    The key difference from single-turn: we compute log-probs over ALL assistant
    turns in the conversation, not just one response.
    """
    def compute_conversation_logprob(mdl, conversation):
        """Sum log-probs of all assistant tokens in the conversation."""
        full_text = format_conversation(conversation)
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

        outputs = mdl(**inputs)
        logits = outputs.logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        # Mask: only count assistant tokens
        assistant_mask = get_assistant_token_mask(conversation, tokenizer, inputs["input_ids"].shape[1])
        assistant_mask = assistant_mask[:, 1:]  # Shift to align with predictions

        masked_log_probs = (token_log_probs * assistant_mask).sum(dim=1)
        return masked_log_probs

    # Policy log-probs
    policy_chosen_lp = compute_conversation_logprob(model, chosen_conversations)
    policy_rejected_lp = compute_conversation_logprob(model, rejected_conversations)

    # Reference log-probs
    with torch.no_grad():
        ref_chosen_lp = compute_conversation_logprob(ref_model, chosen_conversations)
        ref_rejected_lp = compute_conversation_logprob(ref_model, rejected_conversations)

    # Standard DPO loss
    chosen_logratios = policy_chosen_lp - ref_chosen_lp
    rejected_logratios = policy_rejected_lp - ref_rejected_lp
    logits = beta * (chosen_logratios - rejected_logratios)

    loss = -torch.nn.functional.logsigmoid(logits).mean()
    return loss
```

### Best-of-N Sampling

```python
class BestOfNSampler:
    """Inference-time alignment via rejection sampling with reward model."""

    def __init__(self, model, tokenizer, reward_model, n=4):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.n = n

    def generate(self, conversation: list[dict]) -> str:
        """Generate N responses and return the one with highest reward."""
        prompt = format_for_generation(conversation)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        candidates = []
        for _ in range(self.n):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=256, temperature=0.8, top_p=0.9, do_sample=True
                )
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Score this candidate
            full_conv = conversation + [{"role": "assistant", "content": response}]
            reward = self.reward_model.score_conversation(full_conv)
            candidates.append((response, reward))

        # Return highest-reward response
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0], candidates[0][1]

    def generate_with_stats(self, conversation: list[dict]) -> dict:
        """Generate with full statistics for analysis."""
        prompt = format_for_generation(conversation)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        candidates = []
        for i in range(self.n):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=256, temperature=0.8, top_p=0.9, do_sample=True
                )
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            full_conv = conversation + [{"role": "assistant", "content": response}]
            reward = self.reward_model.score_conversation(full_conv)
            candidates.append({"response": response, "reward": reward, "index": i})

        candidates.sort(key=lambda x: x["reward"], reverse=True)
        return {
            "best": candidates[0],
            "worst": candidates[-1],
            "all": candidates,
            "reward_spread": candidates[0]["reward"] - candidates[-1]["reward"],
            "mean_reward": np.mean([c["reward"] for c in candidates]),
        }
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| OOM on long conversations | Truncate to last 2048 tokens (keep most recent turns). Or use sliding window attention. |
| Shapley values too expensive | Use difference decomposition instead (O(n) vs O(n!)). Or reduce num_samples to 20. |
| Tree generation takes too long | Reduce branches_per_turn to 2, max_turns to 3. Parallelize with batch generation. |
| Multi-turn DPO doesn't converge | Conversations may have too many tokens. Reduce max_length or only compute loss on last 2 assistant turns. |
| Best-of-N barely improves | N=4 may not be enough. Try N=8 or N=16. Or reward model may not discriminate well on multi-turn. |
| Conversation trees all similar | Increase generation temperature to 1.0 for more diversity in branches. |
| Credit assignment gives negative rewards to good turns | The decomposition is approximate. Use absolute difference instead of marginal contribution for stability. |

---

## Agent Handoff Template

```
Continue the Crucible Phase 2, Week 13 (Multi-Turn Alignment) project.

Hardware: RTX 5080 16GB VRAM, 32GB RAM, Ubuntu.
Project location: crucible/phase2/week13/

Current state: [DESCRIBE WHAT'S DONE]
Blocked on: [DESCRIBE THE ISSUE]

The goal is to extend alignment to multi-turn conversations:
- Conversation-level reward scoring
- Turn-level credit assignment (Shapley / difference decomposition)
- Conversation tree generation with preference extraction
- Multi-turn DPO training
- Best-of-N sampling at inference

Key files:
- generate_tree.py: Conversation tree construction
- turn_rewards.py: Per-turn credit assignment
- train_multiturn_dpo.py: DPO on conversation-level preferences
- best_of_n.py: Inference-time alignment via overgeneration

Please [FIX/CONTINUE/DEBUG] the [SPECIFIC COMPONENT].
```

---

## Out of Scope

- Real-time serving or latency optimization for Best-of-N
- Speculative decoding or parallel generation optimizations
- User simulation (generating realistic multi-turn user behavior at scale)
- Multi-turn safety beyond what's covered in Week 12 (CAI)
- Tool use or function calling in conversations
- Retrieval-augmented generation within conversations
- Production conversation memory / state management
- Evaluation on real user conversations (privacy concerns)
