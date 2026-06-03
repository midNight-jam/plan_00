# Weeks 18-20: Portfolio Polish and Storytelling

## Context

**Phase:** 3 — Production Infrastructure & Advanced Systems (Final Weeks)
**Prerequisites:** All prior weeks complete. Working platform with inference server, continuous batching, quantization, monitoring, K8s operator, custom kernels, and training/alignment.
**Duration:** 3 weeks
**Difficulty:** Moderate (writing and polish, not new systems)

You've built something substantial. Now make it undeniable. The difference between "I built a side project" and "I built something impressive" is storytelling, benchmarks, and polish. These weeks transform raw code into a portfolio piece that demonstrates depth, rigor, and communication ability. Hiring managers and senior engineers won't read your code first — they'll read your README, skim your blog posts, and watch your demo video. Make those artifacts compelling.

---

## Learning Goals

- Learn to write technical content that demonstrates depth without being inaccessible
- Understand how to design reproducible benchmarks that withstand scrutiny
- Learn Architecture Decision Records as a tool for documenting trade-offs
- Develop the skill of explaining complex systems simply (Feynman technique)
- Understand open-source release best practices (licensing, CONTRIBUTING, semantic versioning)
- Learn video storytelling: hook → problem → solution → results → call to action

---

## Implementation Goals

- Write 4 technical blog posts (1500-2500 words each) covering key platform components
- Create a comprehensive benchmark report: Forge vs Ollama vs vLLM
- Write 5-7 Architecture Decision Records documenting key design choices
- Polish the README into a pitch document
- Record a 3-5 minute demo video showing the platform end-to-end
- Complete GitHub polish: CI, badges, versioning, contributor guide
- Execute open-source release checklist

---

## Acceptance Criteria

1. Four blog posts are published (as markdown in `docs/blog/`), each between 1500-2500 words, with code snippets, diagrams, and clear narrative structure (intro → problem → approach → results → conclusion).
2. Benchmark report includes reproducible scripts that anyone can run to verify results, testing Forge vs Ollama vs vLLM across at least 4 dimensions (throughput, latency p50/p95/p99, time-to-first-token, memory efficiency) on at least 2 model sizes.
3. Five or more Architecture Decision Records exist in `docs/adr/`, each following the format: Title, Status, Context, Decision, Consequences, and dated.
4. Demo video (3-5 minutes) is recorded and linked from README, showing: deploy from scratch → load model → serve requests → load test → monitor dashboard → demonstrate optimization (quantization or batching effect).
5. README contains: one-line description, architecture diagram (Mermaid or image), feature list with status, benchmark summary table, quick-start guide (< 5 commands to running), and links to all documentation.
6. CI pipeline runs on every push: lint, unit tests, integration tests (can be mocked), with status badges displayed in README.
7. Repository uses semantic versioning with a tagged `v1.0.0` release including release notes summarizing all capabilities.
8. CONTRIBUTING.md exists with: development setup, code style guide, PR process, and issue template references.
9. Git history is clean: meaningful commit messages, no "fix typo" chains, squashed feature branches, linear history on main.
10. All documentation cross-references correctly (no broken links), code examples in blog posts are tested and run, and the project can be cloned and started by following only the README instructions.

---

## Validation Commands

```bash
# Verify blog posts
for post in docs/blog/*.md; do
  wc -w "$post"  # Should be 1500-2500 words
done

# Run benchmark suite
cd benchmarks/
./run_all.sh  # Runs Forge vs Ollama vs vLLM comparisons
cat results/summary.md

# Verify ADRs
ls docs/adr/  # Should have 5+ files
for adr in docs/adr/*.md; do
  grep -q "## Status" "$adr" && grep -q "## Decision" "$adr" && echo "$adr: valid"
done

# Check README structure
grep -c "^#" README.md  # Should have clear sections
grep -q "Quick Start" README.md
grep -q "Architecture" README.md
grep -q "mermaid\|\.png\|\.svg" README.md  # Has diagram

# Verify CI
cat .github/workflows/ci.yml
gh run list --limit 5  # Recent runs should be green

# Check version tag
git tag --list 'v*'
git show v1.0.0 --stat

# Verify CONTRIBUTING.md
test -f CONTRIBUTING.md && echo "exists"
grep -q "Development Setup" CONTRIBUTING.md
grep -q "Pull Request" CONTRIBUTING.md

# Test quick-start from scratch
git clone <repo-url> /tmp/forge-test
cd /tmp/forge-test
# Follow README quick-start — should work in < 5 commands

# Link checker
find docs/ -name "*.md" -exec grep -l "\[.*\](.*)" {} \; | \
  xargs -I {} markdown-link-check {}
```

---

## Technical Implementation Details

### Blog Post 1: "Building a Multi-Model AI Platform from Scratch"

```markdown
# Outline (docs/blog/01-building-platform.md)

## Hook (100 words)
- Why I built my own inference platform instead of using existing tools
- What I learned that I couldn't learn from documentation

## The Architecture (400 words)
- High-level system diagram
- Component breakdown: model manager, inference engine, API layer, monitoring
- Design principles: modularity, observability, GPU-awareness

## Key Technical Decisions (600 words)
- Why Python + Rust (hot path in Rust, orchestration in Python)
- Memory management: pre-allocated KV cache pools
- API design: OpenAI-compatible with extensions

## The Hard Parts (400 words)
- GPU memory fragmentation and how we solved it
- Request scheduling: fairness vs throughput
- Model loading: lazy loading vs pre-loading tradeoffs

## Results (300 words)
- Performance numbers (link to benchmark report)
- What surprised me
- What I'd do differently

## Conclusion (200 words)
- Key takeaways for someone starting a similar project
```

### Blog Post 2: "Understanding Continuous Batching: Why Your LLM Server is Slow"

```markdown
# Outline (docs/blog/02-continuous-batching.md)

## Hook (100 words)
- Static batching wastes 40-60% of GPU compute — here's why
- Visual: GPU utilization timeline with static vs continuous batching

## The Problem (400 words)
- LLM generation is autoregressive: sequences finish at different times
- Static batching: pad all sequences to max length, waste compute on padding
- Diagram showing wasted compute with 8 sequences of varying lengths

## The Solution: Continuous Batching (600 words)
- Iteration-level scheduling: decide batch composition every iteration
- When a sequence finishes, immediately start a new one
- Implementation details: request queue, iteration scheduler, output buffer
- Code walkthrough of the scheduling loop

## PagedAttention Connection (400 words)
- Why continuous batching needs non-contiguous KV cache
- Block-level memory management enables flexible scheduling
- How vLLM combines these two ideas

## Benchmarking the Impact (300 words)
- Methodology: fixed model, fixed hardware, vary batch strategy
- Results table: throughput, latency percentiles
- When does continuous batching matter most?

## Implementation Gotchas (200 words)
- Prefill vs decode phases have different compute profiles
- Chunked prefill: don't let a long prompt starve decoding requests

## Conclusion (100 words)
```

### Blog Post 3: "A Practical Guide to LLM Quantization"

```markdown
# Outline (docs/blog/03-quantization-guide.md)

## Hook (100 words)
- Run a 70B model on a single 24GB GPU — quantization makes it possible
- But which method? I benchmarked them all.

## What Is Quantization? (300 words)
- Reducing precision: FP16 → INT8 → INT4
- Why it works: model weights are approximately normally distributed
- The fundamental tradeoff: memory savings vs quality loss

## Methods Compared (800 words)
- GPTQ: post-training, layer-by-layer optimal rounding
- AWQ: activation-aware, protects salient weights
- GGUF/llama.cpp: CPU-friendly, mixed precision per layer
- bitsandbytes NF4: training-friendly, normal float
- Each with: how it works (2 sentences), when to use it, code to apply it

## The Benchmark (500 words)
- Setup: same model, same prompts, same hardware
- Dimensions: perplexity, generation speed (tok/s), memory usage, quality (human eval proxy)
- Results table with winner highlighted per dimension
- Key insight: "best" depends on your constraint (memory vs speed vs quality)

## Practical Recommendations (200 words)
- Decision tree: which method for your use case
- When NOT to quantize

## Conclusion (100 words)
```

### Blog Post 4: "Writing My First GPU Kernels with Triton"

```markdown
# Outline (docs/blog/04-triton-kernels.md)

## Hook (100 words)
- I always treated GPU operations as black boxes. Then I wrote my own.
- What I learned about why inference is slow (and how to make it faster)

## GPU Programming Model (400 words)
- The memory hierarchy and why it matters for performance
- Compute vs memory bound: how to tell which one you are
- Diagram: registers → SRAM → L2 → HBM with bandwidth numbers

## Kernel 1: Fused RMSNorm (500 words)
- Why fusion matters: the unfused version reads from HBM 3x
- Triton implementation walkthrough
- The key insight: reduce memory traffic, not compute
- Benchmark result: 35% faster than PyTorch

## Kernel 2: Tiled Attention (500 words)
- The FlashAttention insight: never materialize N×N matrix
- Block processing with online softmax
- Why this is O(N) memory instead of O(N²)
- Benchmark result and comparison

## Kernel 3: Fused SwiGLU (300 words)
- Three operations become one: gate + up + activation
- Performance gain from eliminating intermediate stores

## Lessons Learned (200 words)
- Auto-tuning matters more than clever code
- Profile before optimizing
- Most ops are memory-bound, not compute-bound

## Conclusion (100 words)
```

### Benchmark Report Structure

```python
# benchmarks/run_comparison.py
"""
Benchmark Forge vs Ollama vs vLLM on identical hardware and prompts.
Produces a reproducible report with statistical significance.
"""
import json
import subprocess
import time
from dataclasses import dataclass
from typing import List

@dataclass
class BenchmarkConfig:
    models: List[str] = ("llama-3.2-1b", "llama-3.2-3b")
    prompt_lengths: List[int] = (128, 512, 1024, 2048)
    output_lengths: List[int] = (64, 256, 512)
    concurrency_levels: List[int] = (1, 4, 16, 64)
    num_requests: int = 200
    warmup_requests: int = 20

@dataclass
class BenchmarkResult:
    system: str
    model: str
    concurrency: int
    prompt_len: int
    output_len: int
    throughput_tps: float  # tokens per second (output)
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    ttft_ms: float  # time to first token
    memory_gb: float
    gpu_utilization_pct: float

def run_benchmark_suite(config: BenchmarkConfig) -> List[BenchmarkResult]:
    results = []
    systems = ["forge", "ollama", "vllm"]

    for system in systems:
        start_server(system)
        wait_for_healthy(system)

        for model in config.models:
            load_model(system, model)
            for concurrency in config.concurrency_levels:
                for prompt_len in config.prompt_lengths:
                    for output_len in config.output_lengths:
                        result = run_single_benchmark(
                            system=system,
                            model=model,
                            concurrency=concurrency,
                            prompt_len=prompt_len,
                            output_len=output_len,
                            num_requests=config.num_requests,
                            warmup=config.warmup_requests,
                        )
                        results.append(result)

        stop_server(system)

    return results

def generate_report(results: List[BenchmarkResult], output_path: str):
    """Generate markdown report with tables and analysis."""
    report = ["# Forge vs Ollama vs vLLM: Comprehensive Benchmark Report\n"]
    report.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Hardware:** {get_hardware_info()}\n")

    # Summary table
    report.append("## Summary\n")
    report.append("| Metric | Forge | Ollama | vLLM | Winner |")
    report.append("|--------|-------|--------|------|--------|")

    # ... generate comparison tables per dimension

    with open(output_path, "w") as f:
        f.write("\n".join(report))
```

### Architecture Decision Record Template

```markdown
# ADR-001: KV Cache Memory Management Strategy

## Status
Accepted (2024-XX-XX)

## Context
LLM inference requires storing key-value pairs for all previous tokens.
Naive allocation (contiguous per-sequence) leads to 60-80% memory waste
due to fragmentation and over-allocation for variable-length sequences.

We evaluated three approaches:
1. Contiguous allocation with maximum sequence length
2. PagedAttention (block-based allocation like OS virtual memory)
3. Ring buffer with eviction

## Decision
We chose PagedAttention-style block allocation because:
- Near-zero fragmentation (blocks are fixed size, ~16 tokens)
- Enables continuous batching (sequences don't need contiguous memory)
- Supports prefix caching (shared blocks for common prefixes)
- Proven at scale by vLLM

## Consequences
**Positive:**
- Memory utilization improved from ~40% to ~95%
- Can serve 3-4x more concurrent requests
- Enables prefix caching for repeated prompts

**Negative:**
- Added complexity: block table management, copy-on-write
- Non-contiguous memory requires gather operations in attention kernel
- Debugging memory issues is harder (indirection layer)
```

### README Structure

```markdown
# Forge: GPU-Native LLM Inference Platform

> High-performance multi-model inference with continuous batching, quantization,
> and Kubernetes-native deployment. Built from scratch to understand every layer.

[Architecture diagram here — Mermaid or SVG]

## Features

| Component | Status | Description |
|-----------|--------|-------------|
| Inference Engine | ✅ | Custom engine with PagedAttention |
| Continuous Batching | ✅ | Iteration-level scheduling |
| Quantization | ✅ | GPTQ, AWQ, GGUF support |
| K8s Operator | ✅ | GPU-aware scheduling, auto-scaling |
| Custom Kernels | ✅ | Triton: RMSNorm, Attention, SwiGLU |
| Monitoring | ✅ | Prometheus + Grafana dashboards |
| Training | ✅ | LoRA, DPO alignment |

## Benchmark Highlights

| Metric | Forge | Ollama | vLLM |
|--------|-------|--------|------|
| Throughput (tok/s, batch=16) | X | Y | Z |
| Latency p95 (ms, single) | X | Y | Z |
| Memory efficiency | X% | Y% | Z% |

[Full benchmark report →](docs/benchmarks/report.md)

## Quick Start

​```bash
# Clone and setup
git clone https://github.com/you/forge.git && cd forge
make setup  # installs dependencies, downloads small model

# Start serving
forge serve --model llama-3.2-1b --port 8080

# Test it
curl http://localhost:8080/v1/completions \
  -d '{"prompt": "Hello", "max_tokens": 50}'

# Run benchmarks
forge bench --model llama-3.2-1b --concurrency 16
​```

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Blog Posts](docs/blog/)
- [Architecture Decision Records](docs/adr/)
- [API Reference](docs/api.md)
- [Contributing](CONTRIBUTING.md)

## License

MIT
```

### Demo Video Script

```markdown
# Demo Video Script (3-5 minutes)

## 0:00-0:20 — Hook
"I built a complete LLM inference platform from scratch. Let me show you
what it does and why it's fast."

## 0:20-1:00 — Deploy
- Show terminal: `forge serve --model llama-3.2-3b --quantize awq`
- Model loads, shows VRAM usage
- Show Grafana dashboard coming alive

## 1:00-1:40 — Serve
- Send a single request via curl
- Show streaming response
- Point out time-to-first-token metric

## 1:40-2:30 — Load Test
- Run: `forge bench --concurrency 64 --duration 60s`
- Show Grafana: throughput climbing, batching kicking in
- Point out continuous batching effect (GPU util stays >90%)

## 2:30-3:20 — Optimize
- Show: "Now let's quantize to INT4"
- Re-run benchmark: show 2x throughput improvement
- Show memory savings on dashboard

## 3:20-4:00 — Scale (K8s)
- Apply InferenceService CRD
- Show operator creating pods
- Increase load → auto-scale triggers → new replica appears

## 4:00-4:30 — Closing
- Show architecture diagram
- "Every component was built from scratch"
- Link to repo and blog posts
```

### CI Configuration

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff mypy
      - run: ruff check .
      - run: ruff format --check .
      - run: mypy src/ --ignore-missing-imports

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v4

  integration:
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/integration/ -v --timeout=120

  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install markdown-link-check
      - run: find docs/ -name "*.md" | xargs markdown-link-check
```

### CONTRIBUTING.md Template

```markdown
# Contributing to Forge

## Development Setup

​```bash
# Clone
git clone https://github.com/you/forge.git
cd forge

# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v
​```

## Code Style

- Python: formatted with `ruff format`, linted with `ruff check`
- Type hints required for all public functions
- Docstrings: Google style for public APIs
- No comments that merely narrate code

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/description`
2. Make changes, ensuring tests pass: `pytest tests/ -v`
3. Run linting: `ruff check . && ruff format --check .`
4. Commit with meaningful messages (imperative mood)
5. Push and open a PR against `main`
6. Ensure CI passes
7. Request review

## Commit Messages

Format: `<type>: <description>`

Types: feat, fix, refactor, docs, test, bench, ci

Examples:
- `feat: add INT4 quantization support`
- `fix: resolve OOM in continuous batching at high concurrency`
- `bench: add vLLM comparison for 7B model`
```

### Release Checklist

```markdown
# Open-Source Release Checklist

## Code Quality
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] No linting errors (`ruff check .`)
- [ ] Type checking passes (`mypy src/`)
- [ ] No secrets in repository (`git log --all -p | grep -i "api_key\|secret\|password"`)
- [ ] No large binary files in git history

## Documentation
- [ ] README has architecture diagram
- [ ] README has quick-start (verified on clean machine)
- [ ] API documentation complete
- [ ] All blog posts proofread
- [ ] ADRs dated and status set
- [ ] CONTRIBUTING.md complete
- [ ] LICENSE file present (MIT)
- [ ] CHANGELOG.md for v1.0.0

## CI/CD
- [ ] CI runs on push and PR
- [ ] All jobs green
- [ ] Badges in README (CI status, coverage, version)
- [ ] Release workflow creates GitHub release on tag

## Git
- [ ] Clean history (no debug commits, no "WIP")
- [ ] Feature branches squash-merged
- [ ] Semantic version tag: v1.0.0
- [ ] Release notes written

## Final Verification
- [ ] Clone repo fresh → follow README → working in < 5 min
- [ ] Demo video linked and accessible
- [ ] All markdown links resolve
- [ ] Repository description and topics set on GitHub
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Blog post too long/unfocused | Each post should have ONE core insight — cut everything that doesn't serve it |
| Benchmarks show Forge slower than vLLM | That's fine — be honest, explain why (vLLM has years of optimization), show what you learned |
| Demo video is boring | Start with the impressive result (high throughput number), then show how you got there |
| README is cluttered | Use the "5-second test" — what does a reader understand in 5 seconds? Lead with that. |
| CI keeps failing | Fix the flaky tests first; use `--timeout` and `--reruns` for genuinely flaky integration tests |
| Git history is messy | Use `git rebase -i` to squash fixup commits; don't rewrite history that's already pushed to main |
| ADRs feel forced | Only write ADRs for decisions where you considered alternatives — skip trivial choices |
| Benchmark results vary between runs | Run at least 3 trials, report mean + std; warm up the GPU; disable CPU frequency scaling |

**Key Resources:**
- [Architecture Decision Records](https://adr.github.io/) — format and examples
- [Semantic Versioning](https://semver.org/)
- [Art of README](https://github.com/noffle/art-of-readme)
- [Technical blogging guide by Dan Luu](https://danluu.com/writing-non-advice/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

## Agent Handoff Template

```
## Session State
- Phase: 3 / Weeks 18-20
- Current task: [what you're working on]
- Branch: forge/portfolio-polish

## What's Done
- [ ] Blog post 1: Building the Platform
- [ ] Blog post 2: Continuous Batching
- [ ] Blog post 3: Quantization Guide
- [ ] Blog post 4: Triton Kernels
- [ ] Benchmark report with reproducible scripts
- [ ] ADRs (5-7 total)
- [ ] README polished with diagram and quick-start
- [ ] Demo video recorded
- [ ] CI pipeline configured and green
- [ ] CONTRIBUTING.md written
- [ ] Git history cleaned
- [ ] v1.0.0 tagged and released
- [ ] Final clone-and-run verification

## Current Blocker
[Describe the exact issue]

## Key Files
- docs/blog/ — blog posts
- docs/adr/ — architecture decision records
- benchmarks/ — comparison scripts
- .github/workflows/ci.yml — CI configuration
- README.md — main pitch document
- CONTRIBUTING.md — contributor guide

## Next Step
[Exact next action to take]
```

---

## Out of Scope

- Paid hosting for blog posts (Markdown in repo is sufficient)
- Professional video editing (screen recording with voiceover is fine)
- Marketing or social media promotion
- Building a documentation site (GitHub-rendered markdown is sufficient)
- Performance optimization beyond what's already built (these weeks are about presentation, not new features)
- Multi-language documentation
- Automated deployment pipeline for the project itself (just document how to deploy)
- Community management, issue triage, or ongoing maintenance planning
