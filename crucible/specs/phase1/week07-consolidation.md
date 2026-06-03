# Week 7: Phase 1 Consolidation

## Context

**Where it fits:** Phase 1 (Foundations), Week 7 of 7 — the capstone week
**Prerequisites:** Weeks 1-6 completed (training loop, memory optimization, data pipelines, LoRA/QLoRA, SFT, distributed concepts)
**What it builds on:** Everything. This week integrates all components into a reproducible, production-quality training pipeline
**What comes next:** Phase 2 (Alignment: DPO, RLHF, reward modeling) builds directly on the tools and pipeline you consolidate here

This is not a "new concepts" week. This is an engineering week: make everything robust, reproducible, well-documented, and composable. You'll also produce deliverables that demonstrate your capabilities.

---

## Learning Goals

- [ ] Understand experiment tracking: why logging configs, metrics, and artifacts matters for ML research
- [ ] Articulate reproducibility requirements: seeds, deterministic ops, logged hyperparameters, code versioning
- [ ] Explain the value of a training recipe: a step-by-step guide that another engineer can follow exactly
- [ ] Understand evaluation methodology: choosing benchmarks, measuring significance, avoiding p-hacking
- [ ] Articulate what makes code production-quality: tests, types, documentation, error handling
- [ ] Explain the comparative evaluation framework: same data, same compute budget, different methods

---

## Implementation Goals

- [ ] Integrate Weights & Biases for all training runs (metrics, configs, artifacts, system metrics)
- [ ] Implement reproducibility infrastructure: seed everything, log git hash, save config YAML
- [ ] Write a training recipe document: end-to-end guide from raw data to deployed model
- [ ] Build end-to-end pipeline: `raw_data → tokenize → filter → train(SFT) → evaluate → save`
- [ ] Write a blog post: "Training LLMs from Scratch: What HuggingFace Trainer Hides from You"
- [ ] Add unit tests for all training utilities (>80% coverage on core modules)
- [ ] Run comparative evaluation: base model vs LoRA vs QLoRA vs full SFT on same benchmark
- [ ] Create documentation for all modules built in Phase 1

---

## Acceptance Criteria

1. W&B dashboard shows at least 5 training runs with full metrics (loss, LR, grad norm, eval scores, system GPU/memory)
2. Any training run can be exactly reproduced: same seed + same config + same code commit = same loss at step 1000 (within 1e-4)
3. Training recipe document is followed by someone else (or future you) end-to-end without additional help needed
4. End-to-end pipeline runs with a single command: `python run_pipeline.py --config configs/sft_mistral.yaml`
5. Blog post is >2000 words, includes code examples, diagrams, and concrete performance numbers
6. Test suite has >80% line coverage on `src/` modules and all tests pass
7. Comparison table shows metrics for base/LoRA/QLoRA/SFT on at least 2 benchmarks with error bars
8. All code passes `ruff check` (linting) and `ruff format --check` (formatting)
9. README.md exists with installation instructions, quick start, and architecture overview
10. Git history shows clean, logical commits with meaningful messages (squash/rebase as needed)

---

## Validation Commands

```bash
# Run full test suite with coverage
python -m pytest tests/ -v --cov=src --cov-report=html

# Verify coverage threshold
python -m pytest tests/ --cov=src --cov-fail-under=80

# Run linting
ruff check src/ scripts/
ruff format --check src/ scripts/

# Reproducibility test: two runs with same config produce same results
python scripts/reproducibility_test.py --config configs/repro_test.yaml --runs 2 --tolerance 1e-4

# End-to-end pipeline
python run_pipeline.py --config configs/sft_mistral.yaml --dry-run
python run_pipeline.py --config configs/sft_mistral.yaml

# W&B sync verification
python scripts/verify_wandb.py --project crucible-phase1 --min-runs 5

# Run comparison experiment
python scripts/run_comparison.py --methods base,lora,qlora,full_sft --dataset alpaca_eval

# Generate comparison table
python scripts/generate_comparison_table.py --output results/comparison.md

# Build documentation
python -m pdoc src/ --output-dir docs/api/

# Verify blog post word count
wc -w docs/blog_post.md

# Check all configs are valid YAML
python scripts/validate_configs.py configs/
```

---

## Technical Implementation Details

### Project Structure (Full Phase 1)

```
crucible-phase1/
├── src/
│   ├── __init__.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── loop.py             # Core training loop (Week 1)
│   │   ├── optimizer.py        # Custom AdamW (Week 1)
│   │   ├── scheduler.py        # LR schedulers (Week 1)
│   │   └── mixed_precision.py  # FP16/BF16 (Week 2)
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── checkpointing.py    # Gradient checkpointing (Week 2)
│   │   ├── accumulation.py     # Gradient accumulation (Week 2)
│   │   └── profiler.py         # Memory profiling (Week 2)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tokenizer.py        # BPE + wrappers (Week 3)
│   │   ├── packing.py          # Sequence packing (Week 3)
│   │   ├── mixing.py           # Data mixing (Week 3)
│   │   ├── streaming.py        # Streaming datasets (Week 3)
│   │   └── quality.py          # Data filtering (Week 3)
│   ├── peft/
│   │   ├── __init__.py
│   │   ├── lora.py             # LoRA from scratch (Week 4)
│   │   ├── qlora.py            # QLoRA setup (Week 4)
│   │   └── merge.py            # Adapter merging (Week 4)
│   ├── sft/
│   │   ├── __init__.py
│   │   ├── formats.py          # Dataset format parsers (Week 5)
│   │   ├── chat_template.py    # Chat templates (Week 5)
│   │   ├── loss_masking.py     # Masked loss (Week 5)
│   │   └── evaluation.py       # SFT evaluation (Week 5)
│   ├── distributed/
│   │   ├── __init__.py
│   │   ├── ddp.py              # DDP utilities (Week 6)
│   │   ├── fsdp.py             # FSDP setup (Week 6)
│   │   └── allreduce.py        # AllReduce implementations (Week 6)
│   └── utils/
│       ├── __init__.py
│       ├── logging.py          # W&B + CSV logging
│       ├── checkpoint.py       # Save/load checkpoints
│       ├── reproducibility.py  # Seed management
│       └── config.py           # YAML config loading
├── scripts/
│   ├── run_pipeline.py         # End-to-end pipeline
│   ├── run_comparison.py       # Method comparison
│   └── reproducibility_test.py
├── tests/
│   ├── test_training/
│   ├── test_data/
│   ├── test_peft/
│   ├── test_sft/
│   └── conftest.py
├── configs/
│   ├── sft_mistral.yaml
│   ├── qlora_sweep.yaml
│   └── repro_test.yaml
├── docs/
│   ├── blog_post.md
│   ├── training_recipe.md
│   └── api/
├── results/
│   └── comparison.md
├── pyproject.toml
├── README.md
└── Makefile
```

### W&B Integration

```python
# src/utils/logging.py
import wandb
import os
from pathlib import Path
from datetime import datetime

class ExperimentLogger:
    """Unified logging to W&B and local CSV."""
    
    def __init__(self, config: dict, project: str = "crucible-phase1",
                 run_name: str = None, tags: list[str] = None):
        self.config = config
        self.local_log_dir = Path(config.get('log_dir', 'runs')) / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.local_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize W&B
        self.run = wandb.init(
            project=project,
            name=run_name or config.get('run_name', None),
            config=config,
            tags=tags or [],
        )
        
        # Log code and git info
        wandb.run.log_code("src/")
        self._log_git_info()
    
    def _log_git_info(self):
        import subprocess
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
            git_diff = subprocess.check_output(['git', 'diff', '--stat']).decode().strip()
            wandb.config.update({'git_hash': git_hash, 'git_dirty': bool(git_diff)})
        except subprocess.CalledProcessError:
            pass
    
    def log_step(self, metrics: dict, step: int):
        wandb.log(metrics, step=step)
        # Also save to local CSV for offline access
        self._append_csv(metrics, step)
    
    def log_artifact(self, path: str, name: str, artifact_type: str = "model"):
        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(path)
        self.run.log_artifact(artifact)
    
    def _append_csv(self, metrics: dict, step: int):
        import csv
        csv_path = self.local_log_dir / 'metrics.csv'
        file_exists = csv_path.exists()
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step'] + sorted(metrics.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow({'step': step, **metrics})
    
    def finish(self):
        wandb.finish()
```

### Reproducibility Infrastructure

```python
# src/utils/reproducibility.py
import torch
import numpy as np
import random
import os

def set_seed(seed: int, deterministic: bool = True):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)

def get_experiment_snapshot(config: dict) -> dict:
    """Capture full experiment state for reproducibility."""
    import subprocess
    import platform
    
    snapshot = {
        'config': config,
        'seed': config.get('seed', None),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    
    try:
        snapshot['git_hash'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode().strip()
        snapshot['git_branch'] = subprocess.check_output(
            ['git', 'branch', '--show-current']
        ).decode().strip()
    except subprocess.CalledProcessError:
        pass
    
    return snapshot
```

### End-to-End Pipeline

```python
# scripts/run_pipeline.py
"""End-to-end training pipeline: raw data → trained model."""
import yaml
import argparse
from pathlib import Path
from src.utils.reproducibility import set_seed, get_experiment_snapshot
from src.utils.logging import ExperimentLogger
from src.data.streaming import StreamingTextDataset
from src.data.quality import InstructionFilter
from src.sft.formats import parse_dataset
from src.sft.chat_template import ChatTemplateProcessor
from src.peft.qlora import setup_qlora
from src.training.loop import train
from src.sft.evaluation import evaluate_model

def run_pipeline(config_path: str, dry_run: bool = False):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    
    if dry_run:
        print("Dry run — validating config and data paths...")
        validate_config(config)
        print("Config valid. Pipeline would run the following steps:")
        print("  1. Load and filter dataset")
        print("  2. Tokenize with chat template")
        print("  3. Setup QLoRA model")
        print("  4. Train with SFT loss masking")
        print("  5. Evaluate on benchmarks")
        print("  6. Save model and artifacts")
        return
    
    logger = ExperimentLogger(config, run_name=config.get('run_name'))
    
    # Step 1: Load and filter data
    print("[1/6] Loading and filtering dataset...")
    raw_data = parse_dataset(config['data']['path'], config['data']['format'])
    quality_filter = InstructionFilter(**config['data'].get('filter_params', {}))
    filtered_data = [d for d in raw_data if quality_filter.filter(d)]
    print(f"  Kept {len(filtered_data)}/{len(raw_data)} examples")
    
    # Step 2: Tokenize
    print("[2/6] Tokenizing with chat template...")
    processor = ChatTemplateProcessor(config['model']['name'], max_length=config['data']['max_length'])
    tokenized = [processor.process(conv) for conv in filtered_data]
    
    # Step 3: Setup model
    print("[3/6] Setting up QLoRA model...")
    model, tokenizer = setup_qlora(
        config['model']['name'],
        rank=config['lora']['rank'],
        alpha=config['lora']['alpha'],
    )
    
    # Step 4: Train
    print("[4/6] Training...")
    train_result = train(
        model=model,
        train_data=tokenized[:int(len(tokenized)*0.95)],
        val_data=tokenized[int(len(tokenized)*0.95):],
        config=config['training'],
        logger=logger,
    )
    
    # Step 5: Evaluate
    print("[5/6] Evaluating...")
    eval_results = evaluate_model(model, tokenizer, config['evaluation'])
    logger.log_step(eval_results, step=train_result['total_steps'])
    
    # Step 6: Save
    print("[6/6] Saving model and artifacts...")
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    
    snapshot = get_experiment_snapshot(config)
    with open(output_dir / "experiment_snapshot.yaml", 'w') as f:
        yaml.dump(snapshot, f)
    
    logger.log_artifact(str(output_dir / "adapter"), "sft-adapter")
    logger.finish()
    
    print(f"\nPipeline complete! Results saved to {output_dir}")
    print(f"W&B run: {logger.run.url}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.config, dry_run=args.dry_run)
```

### Comparison Experiment

```python
# scripts/run_comparison.py
"""Run comparative evaluation across training methods."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.peft.qlora import setup_qlora
from src.sft.evaluation import evaluate_model

METHODS = {
    'base': {'description': 'Pre-trained model, no fine-tuning'},
    'lora_r8': {'description': 'LoRA rank=8, all attention layers'},
    'lora_r16': {'description': 'LoRA rank=16, all attention layers'},
    'qlora_r16': {'description': 'QLoRA rank=16, 4-bit base'},
    'full_sft_1b': {'description': 'Full fine-tune on 1B model (different model)'},
}

BENCHMARKS = ['mmlu_subset', 'alpaca_eval', 'humaneval_subset']

def run_comparison(model_name: str, dataset_path: str, output_path: str):
    results = {}
    
    for method_name, method_config in METHODS.items():
        print(f"\nEvaluating: {method_name} ({method_config['description']})")
        
        if method_name == 'base':
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif 'qlora' in method_name:
            model, tokenizer = setup_qlora(model_name, rank=16)
            # Load pre-trained adapter
            model.load_adapter(f"models/{method_name}/adapter")
        # ... etc
        
        method_results = {}
        for benchmark in BENCHMARKS:
            score = evaluate_model(model, tokenizer, {'benchmark': benchmark})
            method_results[benchmark] = score
        
        results[method_name] = method_results
        del model
        torch.cuda.empty_cache()
    
    # Generate markdown comparison table
    generate_table(results, output_path)
    return results

def generate_table(results: dict, output_path: str):
    lines = ["# Phase 1 Method Comparison\n"]
    lines.append("| Method | " + " | ".join(BENCHMARKS) + " | Memory (GB) | Trainable Params |")
    lines.append("|--------|" + "|".join(["--------"] * (len(BENCHMARKS) + 2)) + "|")
    
    for method, scores in results.items():
        row = f"| {method} | "
        row += " | ".join(f"{scores.get(b, 'N/A'):.2f}" if isinstance(scores.get(b), float) else "N/A" 
                          for b in BENCHMARKS)
        row += f" | {scores.get('memory_gb', 'N/A')} | {scores.get('trainable_params', 'N/A')} |"
        lines.append(row)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
```

### Sample Config YAML

```yaml
# configs/sft_mistral.yaml
run_name: "mistral-7b-sft-alpaca"
seed: 42

model:
  name: "mistralai/Mistral-7B-v0.1"

data:
  path: "data/alpaca_cleaned.jsonl"
  format: "alpaca"
  max_length: 2048
  filter_params:
    min_instruction_len: 10
    min_response_len: 20
    max_response_len: 4096

lora:
  rank: 16
  alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  dropout: 0.05

training:
  epochs: 1
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  warmup_ratio: 0.03
  max_grad_norm: 1.0
  dtype: "bfloat16"
  gradient_checkpointing: true
  eval_steps: 100
  save_steps: 500

evaluation:
  benchmarks: ["alpaca_eval", "mmlu_subset"]
  num_samples: 100

output_dir: "models/mistral-sft-alpaca"
log_dir: "runs/"
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| W&B rate limiting | Use `wandb.log` batching or `commit=False` for high-frequency logging. Log every N steps instead of every step |
| Reproducibility fails across machines | CUDA non-determinism: disable `cudnn.benchmark`, set `CUBLAS_WORKSPACE_CONFIG`. Accept small differences across hardware |
| Test coverage below 80% | Focus on `src/` core modules. Mock external dependencies (model loading, GPU ops). Test data processing thoroughly |
| Pipeline fails mid-way | Add checkpointing between stages. Use `try/except` with state saving. Re-run from last checkpoint |
| Comparison table has high variance | Run each method 3 times with different seeds. Report mean ± std. Use same eval prompts |
| Blog post too technical | Write for audience who knows ML basics but not training internals. Include "why this matters" for each concept |
| Ruff linting has many errors | Run `ruff check --fix` for auto-fixable issues. Then address remaining manually |
| Config validation fails | Use `pydantic` or `dataclasses` for typed config. Validate before expensive GPU operations |

---

## Agent Handoff Template

```
I'm working on Week 7 of the Crucible Phase 1 project: Consolidation.

Hardware: RTX 5080 16GB VRAM, 32GB RAM, Ubuntu
Project path: ~/crucible/crucible-phase1/

Current status: [DESCRIBE WHERE YOU ARE]

What I've completed:
- [x/o] W&B integration
- [x/o] Reproducibility infrastructure
- [x/o] Training recipe document
- [x/o] End-to-end pipeline
- [x/o] Blog post draft
- [x/o] Unit tests (current coverage: XX%)
- [x/o] Comparison table
- [x/o] Code quality (linting, formatting)
- [x/o] Documentation and README

Previous weeks' code location: [PATHS TO WEEK 1-6 CODE]

Current issue: [DESCRIBE THE PROBLEM]
Error message (if any): [PASTE ERROR]

Please help me [SPECIFIC ASK]. Goal is production-quality code with full reproducibility and clear documentation.
```

---

## Out of Scope

- New ML techniques not covered in Weeks 1-6
- Phase 2 content (RLHF, DPO, reward modeling)
- Deployment to production (serving, API endpoints)
- Building a web UI or demo application
- Publishing the blog post (write it, don't publish)
- Submitting to benchmarks or leaderboards
- Optimizing for inference speed (this is training-focused)
- Building custom evaluation benchmarks (use existing ones)
