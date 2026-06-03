# Week 15: Comprehensive Evaluation Framework

## Context

**Where it fits:** Phase 3 (Evaluation, Safety, and Portfolio), Week 15 of 20.

**Prerequisites:**
- Completed Phase 1 (Foundations): tokenization, pretraining fundamentals, SFT
- Completed Phase 2 (Alignment): DPO, RLHF with PPO, KTO, Constitutional AI
- Working fine-tuned and aligned model variants from Weeks 9-14
- Familiarity with HuggingFace transformers, PyTorch, and model inference

**What it builds on:** You have trained multiple model variants (base, SFT, DPO, RLHF, KTO). Now you need to rigorously measure how good they actually are. This week builds the evaluation infrastructure that quantifies model quality across standard benchmarks and custom tasks.

**Why it matters:** Without rigorous evaluation, you cannot make defensible claims about model quality. Industry labs invest heavily in evaluation infrastructure because benchmarks drive research direction, model selection, and deployment decisions. Understanding evaluation methodology—including its limitations—is a core competency for alignment engineers.

---

## Learning Goals

- [ ] Understand the design and limitations of standard LLM benchmarks (MMLU, HumanEval, GSM8K, HellaSwag, TruthfulQA)
- [ ] Implement few-shot evaluation with proper prompt formatting (0-shot, 5-shot, chain-of-thought)
- [ ] Apply statistical rigor: bootstrap confidence intervals, multiple random seeds, significance testing
- [ ] Measure perplexity correctly on held-out data (handling tokenization edge cases)
- [ ] Recognize benchmark contamination risks and mitigation strategies
- [ ] Understand metric validity: what each benchmark actually measures vs. what it claims to measure
- [ ] Design custom evaluation tasks for domain-specific capabilities

---

## Implementation Goals

- [ ] Build modular evaluation harness supporting multiple benchmark formats
- [ ] Implement MMLU evaluation (57 subjects, multiple-choice format)
- [ ] Implement HumanEval evaluation (code generation, pass@k metric)
- [ ] Implement GSM8K evaluation (math reasoning, answer extraction)
- [ ] Implement HellaSwag evaluation (commonsense NLI, length-normalized scoring)
- [ ] Implement TruthfulQA evaluation (truthfulness + informativeness scoring)
- [ ] Build perplexity measurement pipeline on held-out data
- [ ] Create custom domain-specific evaluation tasks (minimum 50 examples)
- [ ] Implement bootstrap confidence intervals for all metrics
- [ ] Build automated pipeline: single command runs all benchmarks, produces JSON + HTML report
- [ ] Generate comparison table: base vs SFT vs DPO vs RLHF across all benchmarks

---

## Acceptance Criteria

1. Evaluation harness runs MMLU (5-shot) on any HuggingFace model and reports per-subject and aggregate accuracy with 95% CI within ±2% of published results for a known model (e.g., Llama-2-7B).
2. HumanEval evaluation computes pass@1, pass@10 using unbiased estimator with temperature sampling and reports results within ±3% of published values.
3. GSM8K evaluation correctly extracts numerical answers from chain-of-thought responses and reports accuracy with bootstrap CI (1000 resamples).
4. HellaSwag evaluation uses length-normalized log-probability scoring and handles the 4-choice format correctly.
5. TruthfulQA evaluation scores both truthfulness and informativeness, producing a 2D score per model.
6. Perplexity measurement on WikiText-103 test set matches published values (±0.5 perplexity) for a reference model.
7. Custom eval task suite contains ≥50 examples with clear scoring rubric and inter-rater reliability >0.8 (measured via self-consistency).
8. Bootstrap confidence intervals are computed for every metric, and significance testing (paired bootstrap) correctly identifies when two models differ at p<0.05.
9. Full evaluation pipeline runs all benchmarks with a single command and produces a structured JSON report + formatted HTML comparison table.
10. Evaluation across 3+ random seeds shows variance and the pipeline handles seed-dependent randomness (example ordering, sampling) reproducibly.

---

## Validation Commands

```bash
# Run the full evaluation pipeline on a small model
cd ~/crucible/evaluation
python run_eval.py \
  --model_path ../models/sft-7b \
  --benchmarks mmlu,humaneval,gsm8k,hellaswag,truthfulqa \
  --shots 0,5 \
  --seeds 42,123,456 \
  --output_dir results/sft-7b \
  --confidence_level 0.95

# Verify MMLU evaluation correctness against reference
python -m pytest tests/test_mmlu.py -v

# Run perplexity measurement
python measure_perplexity.py \
  --model_path ../models/base-7b \
  --dataset wikitext-103-test \
  --stride 512

# Run custom eval tasks
python run_eval.py \
  --model_path ../models/dpo-7b \
  --benchmarks custom \
  --task_dir tasks/domain_specific \
  --output_dir results/dpo-7b-custom

# Generate comparison report
python generate_report.py \
  --results_dirs results/base-7b,results/sft-7b,results/dpo-7b,results/rlhf-7b \
  --output report.html

# Validate statistical tests
python -m pytest tests/test_statistics.py -v

# Check for benchmark contamination
python contamination_check.py \
  --training_data ../data/train.jsonl \
  --benchmark_data benchmarks/mmlu/test.jsonl \
  --method ngram_overlap \
  --n 13
```

---

## Technical Implementation Details

### Project Structure

```
~/crucible/evaluation/
├── run_eval.py                    # Main entry point
├── generate_report.py             # Report generation
├── measure_perplexity.py          # Perplexity measurement
├── contamination_check.py         # Data contamination detection
├── eval_harness/
│   ├── __init__.py
│   ├── base.py                    # Abstract benchmark class
│   ├── mmlu.py                    # MMLU implementation
│   ├── humaneval.py               # HumanEval implementation
│   ├── gsm8k.py                   # GSM8K implementation
│   ├── hellaswag.py               # HellaSwag implementation
│   ├── truthfulqa.py              # TruthfulQA implementation
│   ├── custom_tasks.py            # Custom task loader
│   └── utils/
│       ├── prompt_format.py       # Few-shot prompt construction
│       ├── scoring.py             # Answer extraction and scoring
│       └── statistics.py          # Bootstrap CI, significance tests
├── tasks/
│   └── domain_specific/
│       ├── task_config.yaml
│       └── examples.jsonl
├── tests/
│   ├── test_mmlu.py
│   ├── test_statistics.py
│   └── test_scoring.py
└── results/
```

### Abstract Benchmark Class

```python
# eval_harness/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class EvalResult:
    metric_name: str
    value: float
    ci_lower: float
    ci_upper: float
    n_examples: int
    per_example_scores: np.ndarray
    metadata: dict

class Benchmark(ABC):
    def __init__(self, model, tokenizer, n_shots: int = 0, seed: int = 42):
        self.model = model
        self.tokenizer = tokenizer
        self.n_shots = n_shots
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def load_data(self) -> list[dict]:
        """Load benchmark examples."""
        pass

    @abstractmethod
    def format_prompt(self, example: dict, few_shot_examples: list[dict]) -> str:
        """Format a single evaluation prompt with few-shot examples."""
        pass

    @abstractmethod
    def score(self, model_output: str, example: dict) -> float:
        """Score a single model output. Returns 0 or 1 for accuracy."""
        pass

    def evaluate(self) -> EvalResult:
        data = self.load_data()
        few_shot_pool = data[:self.n_shots * 2]
        eval_data = data[self.n_shots * 2:]

        few_shot_examples = self.rng.choice(
            few_shot_pool, size=min(self.n_shots, len(few_shot_pool)), replace=False
        ).tolist()

        scores = []
        for example in eval_data:
            prompt = self.format_prompt(example, few_shot_examples)
            output = self._generate(prompt)
            score = self.score(output, example)
            scores.append(score)

        scores = np.array(scores)
        mean = scores.mean()
        ci_lower, ci_upper = self._bootstrap_ci(scores)

        return EvalResult(
            metric_name=self.metric_name,
            value=mean,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_examples=len(scores),
            per_example_scores=scores,
            metadata={"n_shots": self.n_shots, "seed": self.seed},
        )

    def _bootstrap_ci(
        self, scores: np.ndarray, n_bootstrap: int = 1000, confidence: float = 0.95
    ) -> tuple[float, float]:
        boot_means = np.array([
            self.rng.choice(scores, size=len(scores), replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        alpha = (1 - confidence) / 2
        return float(np.percentile(boot_means, 100 * alpha)), \
               float(np.percentile(boot_means, 100 * (1 - alpha)))

    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
```

### MMLU Implementation

```python
# eval_harness/mmlu.py
import torch
from .base import Benchmark, EvalResult

class MMLUBenchmark(Benchmark):
    metric_name = "accuracy"
    CHOICES = ["A", "B", "C", "D"]

    def __init__(self, model, tokenizer, subjects: Optional[list] = None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.subjects = subjects  # None = all 57

    def load_data(self) -> list[dict]:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all")
        examples = []
        for item in ds["test"]:
            if self.subjects and item["subject"] not in self.subjects:
                continue
            examples.append({
                "question": item["question"],
                "choices": item["choices"],
                "answer": self.CHOICES[item["answer"]],
                "subject": item["subject"],
            })
        return examples

    def format_prompt(self, example: dict, few_shot_examples: list[dict]) -> str:
        prompt = f"The following are multiple choice questions about {example['subject']}.\n\n"
        for fs in few_shot_examples:
            prompt += self._format_question(fs) + f"\nAnswer: {fs['answer']}\n\n"
        prompt += self._format_question(example) + "\nAnswer:"
        return prompt

    def _format_question(self, ex: dict) -> str:
        q = f"Question: {ex['question']}\n"
        for letter, choice in zip(self.CHOICES, ex["choices"]):
            q += f"{letter}. {choice}\n"
        return q

    def score(self, model_output: str, example: dict) -> float:
        predicted = model_output.strip()[:1].upper()
        return float(predicted == example["answer"])

    def evaluate_logprob(self) -> EvalResult:
        """Alternative: score by comparing logprobs of A/B/C/D tokens."""
        data = self.load_data()
        few_shot_pool = data[:self.n_shots * 2]
        eval_data = data[self.n_shots * 2:]
        few_shot_examples = self.rng.choice(
            few_shot_pool, size=self.n_shots, replace=False
        ).tolist()

        choice_tokens = [
            self.tokenizer.encode(c, add_special_tokens=False)[0]
            for c in self.CHOICES
        ]
        scores = []
        for example in eval_data:
            prompt = self.format_prompt(example, few_shot_examples)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits[0, -1, :]
            choice_logits = logits[choice_tokens]
            predicted_idx = choice_logits.argmax().item()
            predicted = self.CHOICES[predicted_idx]
            scores.append(float(predicted == example["answer"]))

        scores = np.array(scores)
        ci_lower, ci_upper = self._bootstrap_ci(scores)
        return EvalResult(
            metric_name="accuracy",
            value=scores.mean(),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_examples=len(scores),
            per_example_scores=scores,
            metadata={"method": "logprob", "n_shots": self.n_shots},
        )
```

### Statistical Testing

```python
# eval_harness/utils/statistics.py
import numpy as np
from typing import Optional

def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap significance test.
    Tests H0: mean(scores_a) == mean(scores_b)
    Returns p-value and whether the difference is significant at various thresholds.
    """
    assert len(scores_a) == len(scores_b), "Score arrays must have same length"
    rng = np.random.RandomState(seed)
    observed_diff = scores_a.mean() - scores_b.mean()

    count = 0
    for _ in range(n_bootstrap):
        indices = rng.randint(0, len(scores_a), size=len(scores_a))
        boot_diff = scores_a[indices].mean() - scores_b[indices].mean()
        if observed_diff > 0 and boot_diff <= 0:
            count += 1
        elif observed_diff <= 0 and boot_diff > 0:
            count += 1

    p_value = (count + 1) / (n_bootstrap + 1)  # +1 for continuity correction

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "significant_at_0.05": p_value < 0.05,
        "significant_at_0.01": p_value < 0.01,
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_confidence_interval(
    scores: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
    method: str = "percentile",
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval.
    Methods: 'percentile' (basic), 'bca' (bias-corrected and accelerated).
    """
    rng = np.random.RandomState(seed)
    boot_means = np.array([
        rng.choice(scores, size=len(scores), replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    if method == "percentile":
        alpha = (1 - confidence) / 2
        return (
            float(np.percentile(boot_means, 100 * alpha)),
            float(np.percentile(boot_means, 100 * (1 - alpha))),
        )
    elif method == "bca":
        # Bias-corrected and accelerated bootstrap
        from scipy import stats
        observed = scores.mean()
        z0 = stats.norm.ppf(np.mean(boot_means < observed))

        # Acceleration (jackknife estimate)
        n = len(scores)
        jackknife_means = np.array([
            np.delete(scores, i).mean() for i in range(n)
        ])
        jack_mean = jackknife_means.mean()
        num = np.sum((jack_mean - jackknife_means) ** 3)
        den = 6 * (np.sum((jack_mean - jackknife_means) ** 2) ** 1.5)
        a = num / den if den != 0 else 0

        alpha = (1 - confidence) / 2
        z_alpha = stats.norm.ppf(alpha)
        z_1_alpha = stats.norm.ppf(1 - alpha)

        # Adjusted percentiles
        p_lower = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        p_upper = stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))

        return (
            float(np.percentile(boot_means, 100 * p_lower)),
            float(np.percentile(boot_means, 100 * p_upper)),
        )
    else:
        raise ValueError(f"Unknown method: {method}")
```

### Perplexity Measurement

```python
# measure_perplexity.py
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def measure_perplexity(
    model_path: str,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-v1",
    split: str = "test",
    stride: int = 512,
    max_length: Optional[int] = None,
) -> dict:
    """
    Measure perplexity using sliding window approach.
    Stride < max_length creates overlapping windows for more accurate measurement.
    Only the non-overlapping portion of each window contributes to the loss.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    seq_len = input_ids.size(1)
    if max_length is None:
        max_length = model.config.max_position_embeddings

    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end  # only score non-overlapping tokens

        input_chunk = input_ids[:, begin:end].to(model.device)
        target_chunk = input_chunk.clone()
        target_chunk[:, :-target_len] = -100  # mask overlapping prefix

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)
            neg_log_likelihood = outputs.loss * target_len

        nlls.append(neg_log_likelihood.item())
        prev_end = end

        if end == seq_len:
            break

    total_nll = sum(nlls)
    total_tokens = prev_end
    perplexity = np.exp(total_nll / total_tokens)

    return {
        "perplexity": float(perplexity),
        "total_tokens": total_tokens,
        "avg_nll": total_nll / total_tokens,
        "stride": stride,
        "max_length": max_length,
    }
```

### Automated Report Generation

```python
# generate_report.py
import json
from pathlib import Path
from jinja2 import Template

REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Evaluation Report</title>
<style>
  table { border-collapse: collapse; margin: 20px 0; }
  th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
  th { background: #4CAF50; color: white; }
  .best { font-weight: bold; color: #2e7d32; }
  .ci { font-size: 0.8em; color: #666; }
</style>
</head>
<body>
<h1>Model Evaluation Comparison</h1>
<table>
  <tr>
    <th>Benchmark</th>
    {% for model in models %}<th>{{ model }}</th>{% endfor %}
  </tr>
  {% for bench in benchmarks %}
  <tr>
    <td>{{ bench }}</td>
    {% for model in models %}
    <td class="{{ 'best' if results[model][bench].is_best else '' }}">
      {{ "%.1f"|format(results[model][bench].value * 100) }}%
      <span class="ci">({{ "%.1f"|format(results[model][bench].ci_lower * 100) }}-{{ "%.1f"|format(results[model][bench].ci_upper * 100) }})</span>
    </td>
    {% endfor %}
  </tr>
  {% endfor %}
</table>
</body>
</html>
"""

def generate_report(results_dirs: list[str], output_path: str):
    all_results = {}
    for rdir in results_dirs:
        model_name = Path(rdir).name
        with open(Path(rdir) / "results.json") as f:
            all_results[model_name] = json.load(f)

    # Mark best per benchmark
    benchmarks = list(next(iter(all_results.values())).keys())
    models = list(all_results.keys())

    for bench in benchmarks:
        best_val = max(all_results[m][bench]["value"] for m in models)
        for m in models:
            all_results[m][bench]["is_best"] = all_results[m][bench]["value"] == best_val

    template = Template(REPORT_TEMPLATE)
    html = template.render(results=all_results, models=models, benchmarks=benchmarks)
    Path(output_path).write_text(html)
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| MMLU accuracy far below published results | Check prompt format carefully—spacing, newlines, and choice letter format matter. Use logprob method instead of generation for consistency. |
| HumanEval execution errors | Use Docker sandboxing for code execution. Ensure timeout handling (some generated code runs forever). |
| GSM8K answer extraction fails | Model may not follow "The answer is X" format. Use regex with fallback: extract last number in response. |
| Perplexity doesn't match reference | Check stride/window settings. Verify tokenizer matches (some models use different tokenizers than documented). |
| Bootstrap CI is too wide | Increase n_bootstrap (try 10000). If still wide, you may need more eval examples or the model is genuinely high-variance. |
| Out of memory during evaluation | Use `torch.no_grad()`, reduce batch size, or use 4-bit quantization for eval (note this changes results slightly). |
| Contamination detected | Document which benchmarks may be contaminated. Use held-out test sets or recently-created benchmarks as alternatives. |

---

## Agent Handoff Template

```
Continue building the evaluation framework for ~/crucible/evaluation/.

Current state: [describe what's implemented so far]

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB VRAM, 32GB RAM, Ubuntu.

Models to evaluate:
- Base model: [path]
- SFT model: [path]
- DPO model: [path]
- RLHF model: [path]

What's working: [list benchmarks that pass tests]
What's broken: [describe failures]

Next steps from acceptance criteria:
- [ ] [next unchecked criterion]

Key constraints:
- All benchmarks must fit in 16GB VRAM (use quantization or CPU offload if needed)
- Results must include bootstrap confidence intervals
- Single-command pipeline must work end-to-end
```

---

## Out of Scope

- Building new benchmarks from scratch (we use existing validated benchmarks)
- Training new models (evaluation only—models come from Phase 1-2)
- Multi-GPU evaluation (single GPU focus; distributed eval is Week 18)
- Human evaluation studies (human preference evaluation is Week 17)
- Benchmark dataset creation/curation (we use published datasets)
- Real-time/streaming evaluation (batch evaluation only)
- Evaluation of closed-source/API models (local models only)
