# Week 14: Alignment Methods Comparison (Research Report)

## Context

**Where it fits:** Phase 2 (Alignment Deep Dive), Week 7 of 7. This is the capstone — a rigorous empirical comparison of everything you've built in Weeks 8-13, presented as a research-grade report.

**Prerequisites:**
- Week 8: Trained reward model (evaluation tool)
- Week 9: PPO-RLHF trained model
- Week 10: DPO trained model + IPO variant
- Week 11: KTO, ORPO, SimPO trained models
- Week 12: Constitutional AI / RLAIF model
- Week 13: Multi-turn aligned model + Best-of-N
- All models trained on same base model and (where possible) same data

**What it builds on:** Every previous week provides a model to compare. This week is pure evaluation, analysis, and writing — no new training algorithms.

**What it enables:** This report IS your portfolio piece. It demonstrates research methodology, experimental rigor, and deep understanding. It positions you as research-grade, not just engineering-grade — critical for Anthropic/OpenAI applications.

---

## Learning Goals

- [ ] Design rigorous experiments: fixed base model, same compute budget, controlled variables
- [ ] Understand evaluation methodology: win-rate, benchmark scores, human preference simulation
- [ ] Articulate when each method excels vs fails (and explain WHY from first principles)
- [ ] Apply ablation methodology: change one variable, measure impact
- [ ] Present results with proper statistical analysis (confidence intervals, significance tests)
- [ ] Write in research paper style: abstract, introduction, methods, results, analysis, conclusion
- [ ] Communicate trade-offs clearly for practitioners (the blog post audience)

---

## Implementation Goals

- [ ] Run all models through unified evaluation suite (same prompts, same judge, same metrics)
- [ ] Compute win-rates: every model vs every other model (pairwise matrix)
- [ ] Run ablation studies: KL coefficient, LoRA rank, dataset size, number of training steps
- [ ] Compute benchmark scores: TruthfulQA, MT-Bench categories, safety benchmarks
- [ ] Generate charts: radar plots, bar charts, training curves overlay, ablation curves
- [ ] Perform statistical significance testing on all pairwise comparisons
- [ ] Write 8-10 page research report with proper formatting
- [ ] Write practitioner blog post (2000-3000 words) summarizing key findings
- [ ] Release all code, configs, and evaluation results for reproducibility

---

## Acceptance Criteria

1. All 7+ models evaluated on identical test set of at least 200 prompts across diverse categories (coding, creative, reasoning, safety).
2. Pairwise win-rate matrix computed with LLM-as-judge for all model pairs (N×N matrix where N≥7).
3. At least 3 ablation studies completed: (a) KL coefficient sweep, (b) training data size, (c) base model size or LoRA rank.
4. Statistical significance computed for all pairwise comparisons using paired bootstrap (p-values reported).
5. At least 4 evaluation dimensions: helpfulness win-rate, safety refusal rate, coherence score, and one benchmark (TruthfulQA or MT-Bench).
6. Research report is 8-10 pages with: abstract, introduction, related work, experimental setup, results, analysis, limitations, conclusion.
7. Blog post is 2000-3000 words with clear practitioner recommendations: "use X when you have Y constraint."
8. All charts generated programmatically (reproducible) — at least 6 figures in the report.
9. Results are reproducible: config files, random seeds, and evaluation prompts are all saved and documented.
10. Report identifies at least 3 non-obvious findings (things that contradict common assumptions or aren't obvious from reading papers).

---

## Validation Commands

```bash
# Run unified evaluation across all models
python run_evaluation.py --models ppo,dpo,ipo,kto,orpo,simpo,cai,multiturn --prompts data/eval_200.jsonl --judge gpt-4 --output results/unified_eval.json

# Compute pairwise win-rate matrix
python compute_winrates.py --eval_results results/unified_eval.json --output results/winrate_matrix.json

# Run ablation: KL coefficient
python run_ablation.py --ablation kl_coeff --values 0.01,0.05,0.1,0.2,0.5 --method dpo --output results/ablation_kl.json

# Run ablation: dataset size
python run_ablation.py --ablation data_size --values 500,1000,2000,5000,10000 --method dpo --output results/ablation_datasize.json

# Run ablation: LoRA rank
python run_ablation.py --ablation lora_rank --values 4,8,16,32,64 --method dpo --output results/ablation_rank.json

# Benchmark evaluation
python run_benchmarks.py --models ppo,dpo,kto,orpo,simpo,cai --benchmarks truthfulqa,mt_bench --output results/benchmarks.json

# Safety evaluation
python eval_safety_all.py --models ppo,dpo,kto,orpo,simpo,cai --harmful_prompts data/safety_prompts.jsonl --output results/safety_eval.json

# Statistical significance
python significance_all.py --eval_results results/unified_eval.json --method paired_bootstrap --num_samples 10000 --output results/significance.json

# Generate all charts
python generate_figures.py --results_dir results/ --output_dir figures/

# Compile research report
python compile_report.py --results_dir results/ --figures_dir figures/ --template templates/report.md --output report/alignment_comparison.md

# Word count check for blog post
wc -w report/blog_post.md
```

---

## Technical Implementation Details

### Project Structure

```
crucible/phase2/week14/
├── run_evaluation.py          # Unified evaluation pipeline
├── compute_winrates.py        # Pairwise win-rate matrix
├── run_ablation.py            # Ablation study runner
├── run_benchmarks.py          # Standard benchmark evaluation
├── eval_safety_all.py         # Safety evaluation across models
├── significance_all.py        # Statistical significance testing
├── generate_figures.py        # All charts and plots
├── compile_report.py          # Report assembly
├── data/
│   ├── eval_200.jsonl         # Main evaluation prompts
│   ├── safety_prompts.jsonl   # Safety test prompts
│   └── benchmark_configs/
├── results/                   # Raw results (JSON)
├── figures/                   # Generated plots (PNG/SVG)
├── report/
│   ├── alignment_comparison.md  # Research report
│   └── blog_post.md            # Practitioner blog post
└── templates/
    └── report.md              # Report template
```

### Unified Evaluation Pipeline

```python
import json
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class EvalConfig:
    models: list[str]
    eval_prompts_path: str
    judge_model: str = "gpt-4"
    num_prompts: int = 200
    categories: list[str] = None

    def __post_init__(self):
        if self.categories is None:
            self.categories = ["coding", "creative", "reasoning", "safety", "factual", "instruction_following"]

class UnifiedEvaluator:
    """Run all models through the same evaluation pipeline."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.prompts = self._load_prompts()
        self.results = {}

    def _load_prompts(self) -> list[dict]:
        with open(self.config.eval_prompts_path) as f:
            prompts = [json.loads(line) for line in f]
        return prompts[:self.config.num_prompts]

    def generate_all_responses(self):
        """Generate responses from all models on all prompts."""
        for model_name in self.config.models:
            print(f"Generating responses from {model_name}...")
            model, tokenizer = load_model(model_name)

            responses = []
            for prompt_data in self.prompts:
                response = generate_response(model, tokenizer, prompt_data["prompt"])
                responses.append({
                    "prompt": prompt_data["prompt"],
                    "category": prompt_data["category"],
                    "response": response,
                    "model": model_name,
                })

            self.results[model_name] = responses
            del model
            torch.cuda.empty_cache()

    def compute_pairwise_winrates(self) -> dict:
        """Compute win-rate for every pair of models using LLM-as-judge."""
        models = list(self.results.keys())
        n = len(models)
        winrate_matrix = {}

        for i in range(n):
            for j in range(i + 1, n):
                model_a, model_b = models[i], models[j]
                wins_a, wins_b, ties = 0, 0, 0

                for idx in range(len(self.prompts)):
                    resp_a = self.results[model_a][idx]["response"]
                    resp_b = self.results[model_b][idx]["response"]
                    prompt = self.prompts[idx]["prompt"]

                    judgment = judge_pair(prompt, resp_a, resp_b, self.config.judge_model)

                    if judgment == "A":
                        wins_a += 1
                    elif judgment == "B":
                        wins_b += 1
                    else:
                        ties += 1

                total = wins_a + wins_b + ties
                winrate_matrix[f"{model_a}_vs_{model_b}"] = {
                    "win_a": wins_a / total,
                    "win_b": wins_b / total,
                    "tie": ties / total,
                }

        return winrate_matrix
```

### Ablation Study Framework

```python
class AblationStudy:
    """Systematic ablation: vary one factor, hold everything else constant."""

    def __init__(self, base_config: dict, ablation_var: str, values: list):
        self.base_config = base_config
        self.ablation_var = ablation_var
        self.values = values
        self.results = {}

    def run(self):
        """Run training + evaluation for each ablation value."""
        for value in self.values:
            print(f"\n=== Ablation: {self.ablation_var} = {value} ===")

            config = self.base_config.copy()
            config[self.ablation_var] = value
            config["output_dir"] = f"./checkpoints/ablation_{self.ablation_var}_{value}"
            config["seed"] = 42  # Fixed seed for comparability

            # Train
            train_model(config)

            # Evaluate
            eval_results = evaluate_model(config["output_dir"])
            self.results[value] = eval_results

            # Clean up VRAM
            torch.cuda.empty_cache()

        return self.results

    def analyze(self) -> dict:
        """Analyze ablation results: identify trends, optimal values."""
        analysis = {
            "variable": self.ablation_var,
            "values_tested": self.values,
            "metrics": {},
        }

        for metric in ["win_rate", "reward", "kl_divergence", "perplexity"]:
            values_for_metric = []
            for val in self.values:
                if metric in self.results[val]:
                    values_for_metric.append((val, self.results[val][metric]))

            if values_for_metric:
                best_val = max(values_for_metric, key=lambda x: x[1] if metric != "perplexity" else -x[1])
                analysis["metrics"][metric] = {
                    "values": values_for_metric,
                    "best": best_val[0],
                    "best_score": best_val[1],
                }

        return analysis


# Specific ablation configurations
ABLATION_CONFIGS = {
    "kl_coeff": {
        "method": "dpo",
        "values": [0.01, 0.05, 0.1, 0.2, 0.5],
        "base_config": {"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "epochs": 3, "dataset": "anthropic_hh"},
    },
    "data_size": {
        "method": "dpo",
        "values": [500, 1000, 2000, 5000, 10000],
        "base_config": {"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "beta": 0.1, "epochs": 3},
    },
    "lora_rank": {
        "method": "dpo",
        "values": [4, 8, 16, 32, 64],
        "base_config": {"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "beta": 0.1, "epochs": 3},
    },
}
```

### Figure Generation

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_winrate_matrix(winrate_data: dict, models: list[str], output_path: str):
    """Generate heatmap of pairwise win-rates."""
    n = len(models)
    matrix = np.full((n, n), 0.5)  # Diagonal = 0.5

    for i in range(n):
        for j in range(i + 1, n):
            key = f"{models[i]}_vs_{models[j]}"
            if key in winrate_data:
                matrix[i][j] = winrate_data[key]["win_a"]
                matrix[j][i] = winrate_data[key]["win_b"]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=models, yticklabels=models,
                cmap="RdYlGn", center=0.5, vmin=0.3, vmax=0.7, ax=ax)
    ax.set_title("Pairwise Win-Rate Matrix (row wins against column)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_radar_chart(method_scores: dict, metrics: list[str], output_path: str):
    """Radar/spider chart comparing all methods across dimensions."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    for method, scores in method_scores.items():
        values = [scores.get(m, 0) for m in metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=method)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title("Method Comparison Across Dimensions")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_ablation_curves(ablation_results: dict, output_path: str):
    """Plot how metrics change as ablation variable changes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ["win_rate", "reward", "kl_divergence"]
    titles = ["Win-Rate vs SFT", "Mean Reward", "KL Divergence"]

    for ax, metric, title in zip(axes, metrics, titles):
        if metric in ablation_results["metrics"]:
            values = ablation_results["metrics"][metric]["values"]
            x = [v[0] for v in values]
            y = [v[1] for v in values]
            ax.plot(x, y, 'bo-', linewidth=2, markersize=8)
            ax.set_xlabel(ablation_results["variable"])
            ax.set_ylabel(metric)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_training_curves_overlay(log_dirs: dict, output_path: str):
    """Overlay training reward curves for all methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, log_dir in log_dirs.items():
        steps, rewards = load_training_log(log_dir)
        ax.plot(steps, rewards, linewidth=2, label=method)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Training Curves: All Alignment Methods")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
```

### Report Structure

```python
REPORT_TEMPLATE = """
# RLHF vs DPO vs KTO: A Practitioner's Comparison on Consumer Hardware

## Abstract
We present a rigorous empirical comparison of 7 alignment methods (PPO-RLHF, DPO, IPO, KTO,
ORPO, SimPO, Constitutional AI) trained on identical base models and evaluation protocols.
All experiments run on a single RTX 5080 (16GB VRAM). We find that [KEY FINDING 1],
[KEY FINDING 2], and [KEY FINDING 3].

## 1. Introduction
- Motivation: practitioners need guidance on which method to use
- Contribution: apples-to-apples comparison on consumer hardware
- Scope: 1B-3B models, single-GPU training

## 2. Related Work
- Alignment surveys (Casper et al., Wang et al.)
- Individual method papers
- Prior comparisons and their limitations

## 3. Experimental Setup
### 3.1 Base Model and Data
### 3.2 Training Configuration (Table)
### 3.3 Evaluation Protocol
### 3.4 Compute Budget

## 4. Results
### 4.1 Overall Win-Rates (Figure: heatmap)
### 4.2 Per-Category Analysis (Figure: radar chart)
### 4.3 Training Efficiency (Figure: curves + compute table)
### 4.4 Safety Evaluation
### 4.5 Ablation Studies

## 5. Analysis
### 5.1 When Does Each Method Shine?
### 5.2 Failure Modes
### 5.3 Practitioner Recommendations
### 5.4 Surprising Findings

## 6. Limitations

## 7. Conclusion

## Appendix
- A: Full hyperparameter tables
- B: Evaluation prompts
- C: Additional figures
"""
```

### Practitioner Recommendations Table

```python
def generate_recommendations_table():
    """Generate the key practitioner guidance table."""
    return """
| Constraint | Recommended Method | Why |
|------------|-------------------|-----|
| No paired preferences (only good/bad labels) | KTO | Only method that works with unpaired data |
| Minimal compute budget | SimPO | No reference model needed, fast training |
| Maximum quality, cost no object | PPO-RLHF | Still best when done right (but hardest) |
| Good quality, simple implementation | DPO | Best quality/complexity tradeoff |
| Combined SFT + alignment in one step | ORPO | Saves a training stage |
| Safety-critical application | CAI + DPO | Self-critique catches safety issues |
| Multi-turn conversations | Multi-turn DPO + Best-of-N | Turn-level credit + inference-time boost |
| Limited VRAM (<16GB) | SimPO or KTO | No reference model overhead |
"""
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Not all models comparable (different data) | Retrain outliers on same data. Document any differences clearly in "Experimental Setup." |
| LLM-as-judge is expensive | Budget: 200 prompts × N^2/2 pairs × $0.03/call. Use Claude Haiku or GPT-4-mini for cost. |
| Results don't show clear winner | That's a valid finding! Report it. Often methods are within noise for small models/datasets. |
| Charts look bad | Use seaborn theme: `sns.set_theme(style="whitegrid")`. Consistent color palette across all figures. |
| Report too long / too short | Target 8-10 pages (4000-5000 words). Cut methodology details to appendix if too long. |
| Can't compute significance | Need at least 100 eval samples per comparison. If N<100, report confidence intervals instead of p-values. |
| Ablation results are noisy | Run 3 seeds per ablation point. Report mean ± std. |

---

## Agent Handoff Template

```
Continue the Crucible Phase 2, Week 14 (Alignment Comparison Report) project.

Hardware: RTX 5080 16GB VRAM, 32GB RAM, Ubuntu.
Project location: crucible/phase2/week14/

Current state: [DESCRIBE WHAT'S DONE - e.g., "All models evaluated, win-rate matrix computed, ablations running"]
Blocked on: [DESCRIBE THE ISSUE]

The goal is to produce a research-grade comparison of all alignment methods from Weeks 8-13:
- Unified evaluation on 200+ prompts
- Pairwise win-rate matrix (N×N)
- Ablation studies (KL coeff, data size, LoRA rank)
- Statistical significance testing
- 8-10 page research report + 2000-word blog post

Key files:
- run_evaluation.py: Unified evaluation pipeline
- compute_winrates.py: Pairwise comparison
- run_ablation.py: Ablation study framework
- generate_figures.py: All charts and plots
- report/alignment_comparison.md: The research report

Please [FIX/CONTINUE/DEBUG] the [SPECIFIC COMPONENT].
```

---

## Out of Scope

- Training new models (all models should be from Weeks 8-13)
- Novel alignment methods (this is evaluation only)
- Submission to a conference (format is inspired by papers but this is a portfolio piece)
- Models larger than 3B (hardware constraint)
- Human evaluation with paid annotators
- Production deployment recommendations
- Cost analysis beyond compute time (API costs for judge are operational)
- Comparison with closed-source models (GPT-4, Claude — we compare our own models only)
