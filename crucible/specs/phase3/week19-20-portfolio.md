# Weeks 19-20: Portfolio Polish

## Context

**Where it fits:** Phase 3 (Evaluation, Safety, and Portfolio), Weeks 19-20 of 20 (final two weeks).

**Prerequisites:**
- Completed Phases 1-2: trained base, SFT, DPO, RLHF, KTO, Constitutional AI model variants
- Completed Weeks 15-18: evaluation framework, safety evaluation, LLM-as-judge, distributed infrastructure
- All evaluation results available: benchmark scores, safety metrics, ELO ratings
- Working code for all training pipelines (SFT, DPO, RLHF, KTO)

**What it builds on:** You have spent 18 weeks building real training and alignment infrastructure. Now you package everything into a portfolio that demonstrates your competence to hiring managers, research teams, and the open-source community. This includes blog posts, a research report, training recipes, model cards, and a polished demo.

**Why it matters:** Technical ability without communication is invisible. The difference between a strong candidate and a hired candidate is often the portfolio. Blog posts demonstrate you can explain complex ideas clearly. A research report shows you can conduct rigorous experiments. Training recipes show practical engineering skill. A demo makes your work tangible to non-technical stakeholders.

---

## Learning Goals

- [ ] Learn technical writing for ML audiences: balancing rigor with accessibility
- [ ] Understand research paper structure: abstract, introduction, methods, experiments, results, discussion
- [ ] Learn how to create compelling visualizations: training curves, comparison charts, ablation tables
- [ ] Understand model card format: capabilities, limitations, intended use, ethical considerations
- [ ] Learn how to structure reproducible experiments: configs, seeds, hardware specs, runtime
- [ ] Understand open-source best practices: README, LICENSE, CONTRIBUTING, documentation
- [ ] Learn how to build effective interactive demos that highlight model differences

---

## Implementation Goals

- [ ] Write blog post 1: "Building RLHF from Scratch: The Full Pipeline Explained"
- [ ] Write blog post 2: "DPO vs RLHF vs KTO: A Practitioner's Guide"
- [ ] Write blog post 3: "Evaluating LLMs Rigorously: Beyond Benchmarks"
- [ ] Write blog post 4: "Constitutional AI on Consumer Hardware: Self-Improving Models"
- [ ] Write research report: "Alignment Methods Comparison on Consumer Hardware" (8-10 pages)
- [ ] Create training recipes for SFT, DPO, RLHF on different hardware configs
- [ ] Write model cards for each trained model variant
- [ ] Build interactive demo: side-by-side comparison of model variants
- [ ] Polish repository: clean code, documentation, reproducible results
- [ ] Create README with project overview, results summary, and quickstart guide

---

## Acceptance Criteria

1. Blog post 1 (RLHF) is 2000-3000 words, includes architecture diagrams, code snippets, training curves, and explains reward model training + PPO loop clearly enough for someone with basic ML knowledge.
2. Blog post 2 (DPO vs RLHF vs KTO) includes quantitative comparison table, training efficiency comparison (time, memory, stability), and clear practitioner recommendations with trade-off analysis.
3. Blog post 3 (Evaluation) explains benchmark limitations, contamination risks, and the case for multi-method evaluation (benchmarks + LLM-judge + safety), with concrete examples of misleading benchmark results.
4. Blog post 4 (Constitutional AI) explains self-improvement loop, CAI loss function, and practical results of training on consumer hardware with specific VRAM/time measurements.
5. Research report follows academic structure (abstract, intro, methods, experiments, results, discussion), contains ≥5 figures/tables, reports results with confidence intervals, and includes proper ablation studies.
6. Training recipes are step-by-step runnable: a reader with the same hardware can reproduce results by following the recipe (all commands, configs, expected outputs documented).
7. Model cards include: model description, training data summary, intended use, limitations, evaluation results, carbon footprint estimate, and example outputs for each variant.
8. Interactive demo runs locally, shows side-by-side outputs from ≥4 model variants for the same prompt, allows custom prompts, and highlights differences with visual indicators.
9. Repository has: comprehensive README, LICENSE (Apache 2.0), requirements.txt with pinned versions, clear directory structure, and all experiments reproducible with documented commands.
10. All artifacts are cross-referenced: blog posts link to code, research report references specific experiments, model cards link to training recipes, demo loads models documented in model cards.

---

## Validation Commands

```bash
# Build and serve blog posts locally
cd ~/crucible/portfolio
mkdocs serve --dev-addr 127.0.0.1:8000

# Word count check for blog posts
for post in blog/post1_rlhf.md blog/post2_comparison.md blog/post3_evaluation.md blog/post4_constitutional.md; do
  echo "$post: $(wc -w < $post) words"
done

# Verify all code snippets in blogs actually run
python verify_code_snippets.py --blog_dir blog/

# Build research report PDF
cd report
pdflatex alignment_comparison.tex
bibtex alignment_comparison
pdflatex alignment_comparison.tex
pdflatex alignment_comparison.tex
echo "Pages: $(pdfinfo alignment_comparison.pdf | grep Pages | awk '{print $2}')"

# Run training recipes (dry run to verify commands work)
for recipe in recipes/*.yaml; do
  python run_recipe.py --config $recipe --dry_run --verify_commands
done

# Validate model cards
python validate_model_cards.py --cards_dir model_cards/

# Launch demo
cd demo
pip install -r requirements.txt
python app.py --models_dir ../models --port 7860

# Repository quality checks
cd ~/crucible
python -m pylint src/ --disable=C --fail-under=7.0
python -m pytest tests/ -v --tb=short
pip install . --dry-run  # verify package installs

# Check all cross-references resolve
python check_references.py --portfolio_dir portfolio/

# Generate final results summary
python generate_summary.py --all_results evaluation/results/ --output portfolio/RESULTS.md
```

---

## Technical Implementation Details

### Project Structure

```
~/crucible/portfolio/
├── blog/
│   ├── post1_rlhf.md
│   ├── post2_comparison.md
│   ├── post3_evaluation.md
│   ├── post4_constitutional.md
│   └── figures/
│       ├── rlhf_architecture.png
│       ├── training_curves.png
│       ├── comparison_table.png
│       └── eval_radar.png
├── report/
│   ├── alignment_comparison.tex
│   ├── alignment_comparison.bib
│   ├── figures/
│   └── tables/
├── recipes/
│   ├── sft_recipe.yaml
│   ├── dpo_recipe.yaml
│   ├── rlhf_recipe.yaml
│   ├── kto_recipe.yaml
│   └── constitutional_recipe.yaml
├── model_cards/
│   ├── base_7b.md
│   ├── sft_7b.md
│   ├── dpo_7b.md
│   ├── rlhf_7b.md
│   └── kto_7b.md
├── demo/
│   ├── app.py
│   ├── requirements.txt
│   └── templates/
├── verify_code_snippets.py
├── validate_model_cards.py
├── check_references.py
└── mkdocs.yml
```

### Blog Post Structure (Post 2 Example)

```markdown
# DPO vs RLHF vs KTO: A Practitioner's Guide

## TL;DR

| Method | Training Time | Peak VRAM | Stability | Quality (ELO) |
|--------|-------------|-----------|-----------|----------------|
| SFT    | 2.5 hrs     | 14.2 GB   | ★★★★★    | 1000 ± 25      |
| DPO    | 3.1 hrs     | 15.1 GB   | ★★★★☆    | 1045 ± 30      |
| RLHF   | 8.4 hrs     | 15.8 GB   | ★★★☆☆    | 1062 ± 28      |
| KTO    | 2.8 hrs     | 14.5 GB   | ★★★★☆    | 1035 ± 32      |

## Introduction
[Hook: the alignment method you choose matters less than you think...]

## Background: What Problem Are We Solving?
[Brief preference learning formulation]

## Method 1: RLHF (The Classic)
[Architecture diagram, PPO explanation, reward model]

### The Math
The PPO objective with KL penalty:

$$\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_{x \sim D, y \sim \pi_\theta} \left[ 
\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) 
- \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

### Practical Notes
- Requires training separate reward model first
- PPO is notoriously unstable—learning rate warmup critical
- Reward hacking: model exploits reward model weaknesses

## Method 2: DPO (The Simplification)
[Direct derivation from RLHF objective]

### The Math
DPO loss eliminates the reward model entirely:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l) \sim D} \left[
\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} 
- \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right) \right]$$

### Practical Notes
- Single training stage (no separate reward model)
- More memory efficient (one model + reference)
- Sensitive to reference model quality

## Method 3: KTO (The Minimal Assumption)
[Explain Kahneman-Tversky optimization]

### The Math
KTO only needs binary signal (good/bad), not paired preferences:

$$\mathcal{L}_{\text{KTO}}(\theta) = \mathbb{E}_{x,y \sim D} \left[
w(y) \cdot \left(1 - v_{\text{KTO}}(x, y; \beta)\right) \right]$$

where $v_{\text{KTO}} = \sigma(\beta (r_\theta(x,y) - z_{\text{ref}}))$ for desirable $y$
and $v_{\text{KTO}} = \sigma(\beta (z_{\text{ref}} - r_\theta(x,y)))$ for undesirable $y$.

## Head-to-Head Comparison
[Benchmark results table, training curves, failure mode analysis]

## Recommendations
[Decision tree: when to use what method]
```

### Research Report Structure (LaTeX)

```latex
% report/alignment_comparison.tex
\documentclass[11pt,a4paper]{article}
\usepackage{amsmath,amssymb,graphicx,booktabs,hyperref}
\usepackage[margin=1in]{geometry}

\title{Alignment Methods Comparison on Consumer Hardware:\\
  A Practitioner's Empirical Study}
\author{[Your Name]}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
We present a systematic comparison of four alignment methods---SFT, DPO, RLHF 
(PPO), and KTO---trained and evaluated entirely on consumer hardware 
(RTX 5080, 16GB VRAM). Using a unified evaluation framework combining 
standard benchmarks, safety metrics, and LLM-as-judge preference evaluation, 
we find that [key finding 1], [key finding 2], and [key finding 3]. 
We release all code, model weights, and evaluation results to enable reproduction.
\end{abstract}

\section{Introduction}
% Motivation, research questions, contributions

\section{Related Work}
% RLHF (Ouyang et al.), DPO (Rafailov et al.), KTO (Ethayarajh et al.)
% Consumer hardware training (LoRA, QLoRA)

\section{Methods}
\subsection{Training Setup}
% Hardware, model (Llama-2-7B or similar), data
% Hyperparameters table

\subsection{Alignment Methods}
% Brief mathematical description of each

\subsection{Evaluation Framework}
% Benchmarks, safety metrics, LLM-as-judge
% Statistical methodology

\section{Experiments}
\subsection{Training Dynamics}
% Learning curves, convergence, stability

\subsection{Benchmark Results}
% MMLU, HumanEval, GSM8K, HellaSwag, TruthfulQA
% Table with confidence intervals

\subsection{Safety Evaluation}
% Toxicity, jailbreak resistance, refusal calibration

\subsection{Preference Evaluation}
% ELO ratings, win rates

\subsection{Ablation Studies}
% Effect of beta, learning rate, data size

\section{Results and Discussion}
% Key findings, surprising results, limitations

\section{Conclusion}
% Summary, recommendations, future work

\bibliographystyle{plain}
\bibliography{alignment_comparison}
\end{document}
```

### Training Recipe Format

```yaml
# recipes/dpo_recipe.yaml
name: "DPO Training Recipe"
description: "Fine-tune a 7B model with Direct Preference Optimization"
hardware_requirements:
  gpu: "NVIDIA RTX 5080 (16GB VRAM) or equivalent"
  ram: "32GB"
  disk: "50GB free"
  os: "Ubuntu 22.04+"

prerequisites:
  - "Base model downloaded: meta-llama/Llama-2-7b-hf"
  - "SFT model trained (see sft_recipe.yaml)"
  - "Preference dataset prepared: data/preferences.jsonl"

environment_setup:
  python_version: "3.10+"
  install_commands:
    - "pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121"
    - "pip install transformers==4.40.0 datasets==2.19.0"
    - "pip install peft==0.10.0 bitsandbytes==0.43.0"
    - "pip install wandb tqdm numpy scipy"

data_preparation:
  format: |
    Each line in preferences.jsonl should be:
    {"prompt": "...", "chosen": "...", "rejected": "..."}
  minimum_examples: 10000
  recommended_examples: 50000
  source_suggestions:
    - "UltraFeedback (filtered)"
    - "HH-RLHF preference pairs"
    - "Custom domain-specific preferences"

training_config:
  model_name: "meta-llama/Llama-2-7b-hf"
  sft_model_path: "./models/sft-7b"
  output_dir: "./models/dpo-7b"
  
  # DPO hyperparameters
  beta: 0.1
  learning_rate: 5.0e-7
  lr_scheduler: "cosine"
  warmup_ratio: 0.1
  num_epochs: 1
  per_device_batch_size: 2
  gradient_accumulation_steps: 8
  max_length: 1024
  max_prompt_length: 512
  
  # LoRA config (for 16GB VRAM)
  use_lora: true
  lora_r: 64
  lora_alpha: 128
  lora_dropout: 0.05
  lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  
  # Precision
  bf16: true
  gradient_checkpointing: true

training_command: |
  python train_dpo.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --sft_model_path ./models/sft-7b \
    --dataset_path data/preferences.jsonl \
    --output_dir ./models/dpo-7b \
    --beta 0.1 \
    --learning_rate 5e-7 \
    --num_epochs 1 \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lora_r 64 \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --save_steps 500 \
    --wandb_project crucible-dpo

expected_results:
  training_time: "~3 hours"
  peak_vram: "~15 GB"
  final_loss: "0.55-0.65"
  expected_improvement:
    mmlu: "+1-2% over SFT"
    human_preference: "+3-5% win rate over SFT"
    safety: "significantly improved refusal on harmful prompts"

troubleshooting:
  - symptom: "Loss stays flat"
    solution: "Increase learning rate to 1e-6, check data format"
  - symptom: "Loss diverges (NaN)"
    solution: "Reduce learning rate to 1e-7, check beta value"
  - symptom: "OOM error"
    solution: "Reduce batch size to 1, increase gradient accumulation"
```

### Model Card Template

```markdown
# Model Card: DPO-7B

## Model Description

- **Base model:** Llama-2-7B
- **Training method:** Direct Preference Optimization (DPO)
- **Training data:** UltraFeedback (50K preference pairs)
- **Hardware:** NVIDIA RTX 5080 (16GB VRAM), 32GB RAM
- **Training time:** 3.1 hours
- **Framework:** PyTorch 2.3 + Transformers 4.40

## Intended Use

This model is intended for research purposes only. It demonstrates DPO alignment
on consumer hardware and serves as a comparison point in the Crucible project.

**Primary use:** Research into alignment methods on consumer hardware.
**Out of scope:** Production deployment, medical/legal advice, content generation without oversight.

## Training Details

| Parameter | Value |
|-----------|-------|
| Beta (KL weight) | 0.1 |
| Learning rate | 5e-7 |
| Epochs | 1 |
| Effective batch size | 16 |
| LoRA rank | 64 |
| Precision | bfloat16 |
| Peak VRAM | 15.1 GB |

## Evaluation Results

| Benchmark | Score | 95% CI |
|-----------|-------|--------|
| MMLU (5-shot) | 46.2% | (45.1, 47.3) |
| HumanEval (pass@1) | 14.8% | (11.2, 18.4) |
| GSM8K (CoT) | 22.1% | (20.3, 23.9) |
| TruthfulQA | 41.5% | (39.2, 43.8) |
| ELO Rating | 1045 | (1015, 1075) |

### Safety Metrics

| Metric | Score |
|--------|-------|
| Harmful refusal rate | 87% |
| Benign compliance rate | 94% |
| Jailbreak resistance | 72% |
| Toxicity rate | 3.2% |

## Limitations

- Trained on limited data (50K examples) compared to production models
- Single-GPU training limits model size and data throughput
- Safety training is basic—not suitable for deployment without additional safeguards
- Performance on non-English languages is not evaluated

## Ethical Considerations

- Model may reproduce biases present in training data
- Refusal calibration is imperfect—may refuse some legitimate requests
- Not intended for generating content without human review
- Carbon footprint: estimated 1.2 kg CO2 for training run

## How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "./models/dpo-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

prompt = "Explain quantum computing in simple terms."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
```

### Interactive Demo (Gradio)

```python
# demo/app.py
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

class ModelComparisonDemo:
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.tokenizer = None
        self._load_models()

    def _load_models(self):
        base_path = "meta-llama/Llama-2-7b-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(base_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.float16, device_map="auto"
        )
        self.models["Base"] = base_model

        variants = ["sft-7b", "dpo-7b", "rlhf-7b", "kto-7b"]
        for variant in variants:
            adapter_path = self.models_dir / variant
            if adapter_path.exists():
                model = PeftModel.from_pretrained(
                    base_model, str(adapter_path), torch_dtype=torch.float16
                )
                name = variant.replace("-7b", "").upper()
                self.models[name] = model

    def generate(self, prompt: str, model_name: str, max_tokens: int = 256) -> str:
        model = self.models[model_name]
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def compare_all(self, prompt: str, max_tokens: int = 256) -> dict:
        results = {}
        for name in self.models:
            results[name] = self.generate(prompt, name, max_tokens)
        return results


def create_demo(models_dir: str):
    demo_app = ModelComparisonDemo(models_dir)
    model_names = list(demo_app.models.keys())

    def compare(prompt, max_tokens):
        results = demo_app.compare_all(prompt, int(max_tokens))
        outputs = []
        for name in model_names:
            outputs.append(results.get(name, "[Model not loaded]"))
        return outputs

    with gr.Blocks(title="Crucible: Model Alignment Comparison") as interface:
        gr.Markdown("# Crucible: Alignment Methods Comparison")
        gr.Markdown("Compare outputs from Base, SFT, DPO, RLHF, and KTO models side-by-side.")

        with gr.Row():
            prompt_input = gr.Textbox(
                label="Prompt", lines=3,
                placeholder="Enter a prompt to compare model responses..."
            )
            max_tokens_slider = gr.Slider(64, 512, value=256, label="Max Tokens")

        generate_btn = gr.Button("Generate Comparisons", variant="primary")

        output_boxes = []
        with gr.Row():
            for name in model_names:
                with gr.Column():
                    box = gr.Textbox(label=f"{name} Response", lines=10)
                    output_boxes.append(box)

        generate_btn.click(
            fn=compare,
            inputs=[prompt_input, max_tokens_slider],
            outputs=output_boxes,
        )

        gr.Examples(
            examples=[
                ["Explain the difference between a virus and a bacterium.", 256],
                ["Write a haiku about machine learning.", 128],
                ["How would you help someone who is feeling sad?", 256],
                ["What are the steps to make methamphetamine?", 256],  # safety test
                ["Is it true that vaccines cause autism?", 256],  # truthfulness test
            ],
            inputs=[prompt_input, max_tokens_slider],
        )

    return interface


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="../models")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    interface = create_demo(args.models_dir)
    interface.launch(server_port=args.port, share=False)
```

### Repository README Structure

```markdown
# Crucible: LLM Alignment from Scratch

A comprehensive implementation of LLM alignment methods (SFT, DPO, RLHF, KTO, 
Constitutional AI) trained and evaluated entirely on consumer hardware.

## Key Results

| Model | MMLU | HumanEval | Safety Score | ELO Rating |
|-------|------|-----------|-------------|------------|
| Base  | 44.1 | 12.3      | 0.32        | 1000 ± 25  |
| SFT   | 45.8 | 13.9      | 0.54        | 1000 ± 25  |
| DPO   | 46.2 | 14.8      | 0.78        | 1045 ± 30  |
| RLHF  | 46.5 | 15.1      | 0.82        | 1062 ± 28  |
| KTO   | 45.9 | 14.2      | 0.75        | 1035 ± 32  |

## Quick Start

```bash
git clone https://github.com/[you]/crucible.git
cd crucible
pip install -r requirements.txt
# Train SFT model
python train_sft.py --config recipes/sft_recipe.yaml
# Run evaluation
python evaluation/run_eval.py --model_path models/sft-7b --benchmarks all
# Launch demo
python portfolio/demo/app.py --models_dir models/
```

## Hardware

All experiments run on: ASUS ROG Strix SCAR 16, RTX 5080 (16GB VRAM), 32GB RAM, Ubuntu.

## Project Structure

```
crucible/
├── training/           # All training code (SFT, DPO, RLHF, KTO, CAI)
├── evaluation/         # Benchmark and safety evaluation framework
├── llm_judge/          # LLM-as-judge and ELO rating system
├── distributed/        # FSDP distributed training infrastructure
├── safety_eval/        # Red-teaming and safety evaluation
├── portfolio/          # Blog posts, report, recipes, demo
├── models/             # Trained model checkpoints
├── data/               # Training and evaluation data
└── tests/              # Unit and integration tests
```
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Blog posts feel too academic/dry | Add personal anecdotes, practical tips, and "things that surprised me" sections. Show failed experiments. |
| Research report lacks rigor | Add confidence intervals to ALL numbers. Include ablation studies. Acknowledge limitations explicitly. |
| Demo is too slow (loading multiple models) | Use PEFT adapters sharing one base model. Load models sequentially, keep only active model on GPU. |
| Training recipes don't reproduce | Pin ALL dependency versions. Include exact git commit hash. Document hardware/driver versions. |
| Figures look unprofessional | Use matplotlib with seaborn style (`plt.style.use('seaborn-v0_8-paper')`). Consistent color scheme. Export at 300 DPI. |
| Model cards feel generic | Include specific failure examples. Show real outputs (good and bad). Be honest about limitations. |
| Repository is messy | Create a fresh branch, restructure, then cherry-pick working code. Use `pre-commit` hooks for formatting. |

---

## Agent Handoff Template

```
Continue polishing the portfolio for ~/crucible/portfolio/.

Current state: [describe what's written/built]

Hardware: ASUS ROG Strix SCAR 16, RTX 5080 16GB VRAM, 32GB RAM, Ubuntu.

Blog posts status:
- Post 1 (RLHF): [draft/complete/word count]
- Post 2 (Comparison): [draft/complete/word count]
- Post 3 (Evaluation): [draft/complete/word count]
- Post 4 (Constitutional): [draft/complete/word count]

Research report: [status, page count]
Training recipes: [which ones complete]
Model cards: [which ones complete]
Demo: [working/broken/not started]
Repo polish: [README/tests/CI status]

Available results data:
- Benchmark results: [path]
- Safety results: [path]
- ELO ratings: [path]

Next steps from acceptance criteria:
- [ ] [next unchecked criterion]

Key constraints:
- Blog posts should be 2000-3000 words each
- Research report should be 8-10 pages
- All artifacts must cross-reference each other
- Demo must load on 16GB VRAM
```

---

## Out of Scope

- Publishing to arXiv or peer-reviewed venues (local report only)
- Building a production web application (local demo only)
- Marketing or social media strategy for the portfolio
- Video tutorials or recorded presentations
- Hosting models on HuggingFace Hub (local-only this phase)
- Building a personal website (blog posts are markdown files)
- Writing job applications or interview prep materials
- Collaboration with others (solo project)
