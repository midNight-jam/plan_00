# Week 17: LLM-as-Judge and Human Preference Evaluation

## Context

**Where it fits:** Phase 3 (Evaluation, Safety, and Portfolio), Week 17 of 20.

**Prerequisites:**
- Completed Week 15: Evaluation Framework (benchmarks, statistical testing)
- Completed Week 16: Safety Evaluation (refusal detection, toxicity scoring)
- Multiple trained model variants to compare (base, SFT, DPO, RLHF, KTO)
- Understanding of preference data and pairwise comparisons from Phase 2

**What it builds on:** Benchmark scores (Week 15) measure narrow capabilities. Safety metrics (Week 16) measure harm avoidance. But neither captures overall response quality as perceived by humans. This week builds LLM-as-judge evaluation: using a strong model to evaluate weaker models, computing ELO ratings, and building an arena-style leaderboard.

**Why it matters:** LLM-as-judge is now the dominant evaluation paradigm for open-ended generation quality. Chatbot Arena (LMSYS) showed that ELO ratings from pairwise comparisons correlate strongly with human preferences. Understanding judge biases (position bias, verbosity bias, self-preference) and mitigation techniques is essential. This completes your evaluation toolkit: benchmarks + safety + preference-based evaluation.

---

## Learning Goals

- [ ] Understand LLM-as-judge methodology: when it works, when it fails, known biases
- [ ] Learn position bias and its mitigation (swap positions, average scores)
- [ ] Understand ELO rating systems: update rules, convergence properties, confidence estimation
- [ ] Learn scoring rubric design: what makes a good evaluation rubric
- [ ] Understand inter-annotator agreement metrics (Cohen's kappa, Krippendorff's alpha)
- [ ] Learn the relationship between pairwise comparisons and ranking (Bradley-Terry model)
- [ ] Understand limitations: self-preference bias, verbosity bias, sycophancy in judges

---

## Implementation Goals

- [ ] Implement pointwise LLM-judge (score single responses on 1-10 scale)
- [ ] Implement pairwise LLM-judge (compare two responses, pick winner)
- [ ] Build position-bias mitigation: evaluate both orderings, detect inconsistencies
- [ ] Implement ELO rating system with proper initialization and convergence detection
- [ ] Build arena-style evaluation: random matchups, accumulate ratings over many rounds
- [ ] Implement scoring rubrics for multiple quality dimensions (helpfulness, correctness, clarity, safety)
- [ ] Measure inter-annotator agreement: same judge with different prompts/temperatures
- [ ] Build confidence intervals for ELO ratings via bootstrap
- [ ] Create visual leaderboard with rankings and confidence bands
- [ ] Validate LLM-judge against benchmark scores: correlation analysis
- [ ] Compare evaluation methods: benchmarks vs. LLM-judge vs. win-rate

---

## Acceptance Criteria

1. Pointwise judge assigns scores (1-10) to model responses with inter-judge consistency ≥0.7 (Spearman correlation across 3 different judge prompt variants).
2. Pairwise judge picks winners between model pairs with position-bias mitigation (both orderings evaluated), and position-bias rate (where swapping order changes verdict) is measured and reported (<15% target).
3. ELO rating system correctly implements the update formula, converges within 500 matchups for 5 models, and produces stable rankings (±30 ELO across different random matchup orders).
4. Arena evaluation runs ≥1000 pairwise comparisons across all model pairs and produces ELO ratings with bootstrap 95% confidence intervals.
5. Scoring rubric produces per-dimension scores (helpfulness, correctness, clarity, safety) and the judge follows the rubric consistently (dimension scores correlate with rubric descriptions, not just overall quality).
6. Inter-annotator agreement (same model, same prompt, 3 judge runs) achieves Cohen's kappa ≥0.6 for pairwise decisions.
7. Leaderboard visualization shows model rankings with confidence intervals, and correctly identifies when models are statistically indistinguishable.
8. Correlation between ELO ratings and benchmark composite scores is computed and reported (expected: moderate positive correlation, r ≥ 0.5).
9. Known failure modes are documented with examples: verbosity bias (longer responses scored higher), self-preference bias, difficulty distinguishing closely-matched models.
10. Full evaluation pipeline takes a set of models + test prompts and produces complete leaderboard with all metrics in <2 hours wall time on target hardware.

---

## Validation Commands

```bash
# Run pairwise evaluation
cd ~/crucible/llm_judge
python pairwise_eval.py \
  --models ../models/base-7b,../models/sft-7b,../models/dpo-7b,../models/rlhf-7b \
  --judge_model ../models/sft-7b \
  --prompts data/eval_prompts.jsonl \
  --n_comparisons 1000 \
  --output results/pairwise_results.json

# Compute ELO ratings
python compute_elo.py \
  --matchups results/pairwise_results.json \
  --n_bootstrap 1000 \
  --output results/elo_ratings.json

# Run pointwise evaluation with rubric
python pointwise_eval.py \
  --models ../models/dpo-7b \
  --judge_model ../models/sft-7b \
  --prompts data/eval_prompts.jsonl \
  --rubric rubrics/helpfulness.yaml \
  --output results/pointwise_results.json

# Measure position bias
python position_bias_analysis.py \
  --matchups results/pairwise_results.json \
  --output results/position_bias.json

# Measure inter-annotator agreement
python agreement_analysis.py \
  --n_repeats 3 \
  --models ../models/dpo-7b,../models/rlhf-7b \
  --judge_model ../models/sft-7b \
  --prompts data/eval_prompts.jsonl \
  --output results/agreement.json

# Generate leaderboard
python generate_leaderboard.py \
  --elo_ratings results/elo_ratings.json \
  --benchmark_results ../evaluation/results/ \
  --output leaderboard.html

# Run full pipeline
python run_arena.py \
  --config arena_config.yaml \
  --output results/full_arena/

# Validate ELO convergence
python -m pytest tests/test_elo.py -v
```

---

## Technical Implementation Details

### Project Structure

```
~/crucible/llm_judge/
├── run_arena.py                    # Full arena pipeline
├── pairwise_eval.py                # Pairwise comparisons
├── pointwise_eval.py               # Pointwise scoring
├── compute_elo.py                  # ELO computation
├── position_bias_analysis.py       # Bias measurement
├── agreement_analysis.py           # IAA measurement
├── generate_leaderboard.py         # Visualization
├── judge/
│   ├── __init__.py
│   ├── pairwise.py                 # Pairwise judge
│   ├── pointwise.py                # Pointwise judge
│   ├── prompts.py                  # Judge prompt templates
│   ├── rubrics.py                  # Scoring rubric loader
│   └── bias_mitigation.py          # Position/verbosity bias handling
├── elo/
│   ├── __init__.py
│   ├── rating.py                   # ELO system
│   ├── matchmaking.py              # Matchup scheduling
│   └── convergence.py              # Convergence detection
├── rubrics/
│   ├── helpfulness.yaml
│   ├── correctness.yaml
│   ├── clarity.yaml
│   └── safety.yaml
├── data/
│   └── eval_prompts.jsonl          # 200+ diverse evaluation prompts
├── tests/
│   ├── test_elo.py
│   ├── test_judge.py
│   └── test_bias.py
└── results/
```

### ELO Rating System

```python
# elo/rating.py
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class ELORatingSystem:
    """
    ELO rating system for model comparison.
    
    The ELO update rule:
      E_A = 1 / (1 + 10^((R_B - R_A) / 400))
      R_A_new = R_A + K * (S_A - E_A)
    
    Where:
      E_A = expected score for player A
      R_A, R_B = current ratings
      S_A = actual score (1 for win, 0.5 for tie, 0 for loss)
      K = update magnitude (higher = faster convergence, more volatile)
    """
    initial_rating: float = 1000.0
    k_factor: float = 32.0
    ratings: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    match_count: dict = field(default_factory=lambda: defaultdict(int))

    def get_rating(self, model: str) -> float:
        if model not in self.ratings:
            self.ratings[model] = self.initial_rating
        return self.ratings[model]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Probability that A beats B given their ratings."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, model_a: str, model_b: str, outcome: float):
        """
        Update ratings based on match outcome.
        outcome: 1.0 = A wins, 0.0 = B wins, 0.5 = tie
        """
        ra = self.get_rating(model_a)
        rb = self.get_rating(model_b)

        ea = self.expected_score(ra, rb)
        eb = 1.0 - ea

        self.ratings[model_a] = ra + self.k_factor * (outcome - ea)
        self.ratings[model_b] = rb + self.k_factor * ((1 - outcome) - eb)

        self.match_count[model_a] += 1
        self.match_count[model_b] += 1
        self.history.append({
            "model_a": model_a, "model_b": model_b,
            "outcome": outcome,
            "rating_a_before": ra, "rating_b_before": rb,
            "rating_a_after": self.ratings[model_a],
            "rating_b_after": self.ratings[model_b],
        })

    def bootstrap_confidence_intervals(
        self, n_bootstrap: int = 1000, confidence: float = 0.95, seed: int = 42
    ) -> dict[str, tuple[float, float]]:
        """Compute CI by resampling match history and recomputing ratings."""
        rng = np.random.RandomState(seed)
        all_boot_ratings = defaultdict(list)

        for _ in range(n_bootstrap):
            boot_system = ELORatingSystem(
                initial_rating=self.initial_rating, k_factor=self.k_factor
            )
            indices = rng.choice(len(self.history), size=len(self.history), replace=True)
            for idx in indices:
                match = self.history[idx]
                boot_system.update(match["model_a"], match["model_b"], match["outcome"])
            for model, rating in boot_system.ratings.items():
                all_boot_ratings[model].append(rating)

        alpha = (1 - confidence) / 2
        cis = {}
        for model, boot_ratings in all_boot_ratings.items():
            boot_ratings = np.array(boot_ratings)
            cis[model] = (
                float(np.percentile(boot_ratings, 100 * alpha)),
                float(np.percentile(boot_ratings, 100 * (1 - alpha))),
            )
        return cis

    def get_leaderboard(self) -> list[dict]:
        cis = self.bootstrap_confidence_intervals()
        leaderboard = []
        for model in sorted(self.ratings, key=self.ratings.get, reverse=True):
            ci = cis.get(model, (self.ratings[model], self.ratings[model]))
            leaderboard.append({
                "model": model,
                "rating": self.ratings[model],
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                "n_matches": self.match_count[model],
            })
        return leaderboard

    def is_converged(self, window: int = 50, threshold: float = 5.0) -> bool:
        """Check if ratings have converged (change < threshold over last window matches)."""
        if len(self.history) < window:
            return False
        recent = self.history[-window:]
        for model in self.ratings:
            relevant = [m for m in recent if m["model_a"] == model or m["model_b"] == model]
            if not relevant:
                continue
            first_rating = relevant[0].get(
                "rating_a_after" if relevant[0]["model_a"] == model else "rating_b_after"
            )
            last_rating = relevant[-1].get(
                "rating_a_after" if relevant[-1]["model_a"] == model else "rating_b_after"
            )
            if abs(last_rating - first_rating) > threshold:
                return False
        return True
```

### Pairwise Judge with Bias Mitigation

```python
# judge/pairwise.py
import torch
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class PairwiseJudgment:
    winner: str  # "A", "B", or "tie"
    confidence: float
    reasoning: str
    position_consistent: bool  # True if verdict same when positions swapped

PAIRWISE_PROMPT_TEMPLATE = """You are an impartial judge. Compare the two responses below to the given prompt and determine which is better.

Evaluate based on: helpfulness, accuracy, relevance, clarity, and completeness.

[Prompt]
{prompt}

[Response A]
{response_a}

[Response B]
{response_b}

Which response is better? Think step by step, then conclude with exactly one of:
[[A]] if Response A is better
[[B]] if Response B is better
[[tie]] if they are equally good

Your judgment:"""


class PairwiseJudge:
    def __init__(self, judge_model, judge_tokenizer, mitigate_position_bias: bool = True):
        self.model = judge_model
        self.tokenizer = judge_tokenizer
        self.mitigate_position_bias = mitigate_position_bias

    def judge(self, prompt: str, response_a: str, response_b: str) -> PairwiseJudgment:
        # First ordering: A then B
        verdict_ab = self._get_verdict(prompt, response_a, response_b)

        if not self.mitigate_position_bias:
            return PairwiseJudgment(
                winner=verdict_ab, confidence=1.0,
                reasoning="", position_consistent=True,
            )

        # Second ordering: B then A (swap positions)
        verdict_ba = self._get_verdict(prompt, response_b, response_a)
        # Translate back: if BA says "A" wins, that means B wins in original ordering
        verdict_ba_translated = {"A": "B", "B": "A", "tie": "tie"}[verdict_ba]

        if verdict_ab == verdict_ba_translated:
            return PairwiseJudgment(
                winner=verdict_ab, confidence=1.0,
                reasoning="Consistent across both orderings",
                position_consistent=True,
            )
        else:
            # Inconsistent - default to tie or use confidence
            return PairwiseJudgment(
                winner="tie", confidence=0.5,
                reasoning=f"Position-dependent: AB={verdict_ab}, BA={verdict_ba_translated}",
                position_consistent=False,
            )

    def _get_verdict(self, prompt: str, resp_a: str, resp_b: str) -> str:
        judge_prompt = PAIRWISE_PROMPT_TEMPLATE.format(
            prompt=prompt, response_a=resp_a, response_b=resp_b
        )
        inputs = self.tokenizer(judge_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=512, temperature=0.0, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return self._extract_verdict(response)

    def _extract_verdict(self, text: str) -> str:
        # Look for [[A]], [[B]], or [[tie]]
        match = re.search(r'\[\[(A|B|tie)\]\]', text, re.IGNORECASE)
        if match:
            verdict = match.group(1).upper()
            return verdict if verdict in ("A", "B") else "tie"
        # Fallback: look for "Response A/B is better"
        if re.search(r'response\s*a\s*is\s*better', text, re.IGNORECASE):
            return "A"
        if re.search(r'response\s*b\s*is\s*better', text, re.IGNORECASE):
            return "B"
        return "tie"
```

### Inter-Annotator Agreement

```python
# judge/agreement.py
import numpy as np
from itertools import combinations

def cohens_kappa(annotations_1: list, annotations_2: list) -> float:
    """
    Compute Cohen's kappa for inter-annotator agreement.
    
    kappa = (p_o - p_e) / (1 - p_e)
    
    Where:
      p_o = observed agreement proportion
      p_e = expected agreement by chance
    """
    assert len(annotations_1) == len(annotations_2)
    n = len(annotations_1)
    categories = list(set(annotations_1 + annotations_2))

    # Observed agreement
    p_o = sum(a == b for a, b in zip(annotations_1, annotations_2)) / n

    # Expected agreement by chance
    p_e = 0.0
    for cat in categories:
        p1 = sum(1 for a in annotations_1 if a == cat) / n
        p2 = sum(1 for a in annotations_2 if a == cat) / n
        p_e += p1 * p2

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def measure_judge_consistency(
    judge, prompts: list[str], responses_a: list[str], responses_b: list[str],
    n_repeats: int = 3, temperature: float = 0.1,
) -> dict:
    """Run judge multiple times and measure self-consistency."""
    all_verdicts = []
    for _ in range(n_repeats):
        verdicts = []
        for prompt, ra, rb in zip(prompts, responses_a, responses_b):
            judgment = judge.judge(prompt, ra, rb)
            verdicts.append(judgment.winner)
        all_verdicts.append(verdicts)

    # Compute pairwise kappa for all pairs of runs
    kappas = []
    for i, j in combinations(range(n_repeats), 2):
        kappa = cohens_kappa(all_verdicts[i], all_verdicts[j])
        kappas.append(kappa)

    # Per-example agreement
    per_example_agreement = []
    for idx in range(len(prompts)):
        votes = [all_verdicts[r][idx] for r in range(n_repeats)]
        majority = max(set(votes), key=votes.count)
        agreement = votes.count(majority) / n_repeats
        per_example_agreement.append(agreement)

    return {
        "mean_kappa": float(np.mean(kappas)),
        "min_kappa": float(np.min(kappas)),
        "max_kappa": float(np.max(kappas)),
        "mean_per_example_agreement": float(np.mean(per_example_agreement)),
        "n_fully_consistent": sum(1 for a in per_example_agreement if a == 1.0),
        "n_prompts": len(prompts),
        "n_repeats": n_repeats,
    }
```

### Bradley-Terry Model for Rankings

```python
# elo/bradley_terry.py
import numpy as np
from scipy.optimize import minimize

def fit_bradley_terry(win_matrix: np.ndarray) -> np.ndarray:
    """
    Fit Bradley-Terry model to pairwise comparison data.
    
    The Bradley-Terry model:
      P(i beats j) = p_i / (p_i + p_j)
    
    We parameterize as log-strengths: theta_i = log(p_i)
      P(i beats j) = sigmoid(theta_i - theta_j)
    
    Args:
        win_matrix: n x n matrix where win_matrix[i][j] = number of times i beat j
    
    Returns:
        Estimated log-strength parameters (n-dimensional)
    """
    n = win_matrix.shape[0]

    def neg_log_likelihood(theta):
        nll = 0.0
        for i in range(n):
            for j in range(n):
                if i == j or win_matrix[i, j] == 0:
                    continue
                prob_i_beats_j = 1.0 / (1.0 + np.exp(theta[j] - theta[i]))
                nll -= win_matrix[i, j] * np.log(prob_i_beats_j + 1e-10)
        return nll

    # Fix first parameter to 0 for identifiability
    def nll_constrained(theta_free):
        theta = np.concatenate([[0.0], theta_free])
        return neg_log_likelihood(theta)

    result = minimize(nll_constrained, np.zeros(n - 1), method="L-BFGS-B")
    theta = np.concatenate([[0.0], result.x])

    # Convert to ELO scale: ELO = 400 * theta / ln(10) + 1000
    elo_ratings = 400 * theta / np.log(10) + 1000

    return elo_ratings
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Judge always picks Response A (position bias) | Enable position-bias mitigation (swap and average). If persists, add explicit instruction: "Position does not indicate quality." |
| ELO ratings don't converge | Increase number of matchups. Use lower K-factor (16 instead of 32). Ensure diverse matchup pairs. |
| Inter-annotator agreement is low (<0.5 kappa) | Judge prompt may be ambiguous. Add more specific rubric. Use temperature=0 for deterministic judging. |
| Judge gives same score to everything | Rubric may lack discriminative criteria. Add concrete examples of high vs. low quality responses to the judge prompt. |
| Verbosity bias (longer = higher score) | Add explicit instruction: "Conciseness is valued. Do not favor longer responses." Also measure response length correlation. |
| Models are too similar to distinguish | Increase sample size (more prompts). Use harder/more diverse prompts that elicit different capabilities. |
| Bootstrap CI for ELO is very wide | Need more matchups. 200+ per model pair is recommended for tight CIs. |

---

## Agent Handoff Template

```
Continue building the LLM-as-judge evaluation system for ~/crucible/llm_judge/.

Current state: [describe what's implemented]

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB VRAM, 32GB RAM, Ubuntu.

Models in arena:
- Base: [path]
- SFT: [path]
- DPO: [path]
- RLHF: [path]

Judge model: [path]
Eval prompts: [path, count]

What's working: [list components]
What's broken: [describe failures]

Current ELO ratings (if computed): [model: rating]
Position bias rate: [X%]
Inter-annotator kappa: [value]

Next steps from acceptance criteria:
- [ ] [next unchecked criterion]

Key constraints:
- Judge must fit in 16GB VRAM alongside one response-generating model
- Position bias must be measured and mitigated
- Need ≥1000 comparisons for stable ratings
```

---

## Out of Scope

- Actual human evaluation studies (we simulate with LLM-judge)
- Building a web-based arena interface (command-line evaluation only)
- Multi-turn conversation evaluation (single-turn responses only)
- Training a dedicated judge model (we use existing models as judges)
- Evaluating closed-source models via API (local models only)
- Reward model training (that was Phase 2; here we only evaluate)
- Real-time interactive evaluation (batch processing only)
