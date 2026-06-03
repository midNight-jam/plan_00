# Crucible: Training and Alignment Engineering — The Full Plan (20 Weeks)

## Philosophy: "Understand the Why Before the How"

- **Phase 1 (Weeks 1-7)**: Master Training — Build the muscle of training models from raw PyTorch, understand every optimization trick, prepare data properly
- **Phase 2 (Weeks 8-14)**: Master Alignment — Implement EVERY major alignment method from scratch, understand the math, compare rigorously
- **Phase 3 (Weeks 15-20)**: Evaluate + Publish — Build comprehensive evaluation, safety testing, and produce research-quality outputs

The key differentiator: You don't just USE alignment libraries — you IMPLEMENT the algorithms from the loss functions up. This means you can whiteboard RLHF/DPO in an interview and explain WHY each component exists.

---

## Why This Track Matters

Anthropic, OpenAI, and xAI don't hire people who can `pip install trl` and call `DPOTrainer.train()`. They hire people who can:
- Derive the DPO loss from the RLHF objective on a whiteboard
- Explain why KL divergence prevents reward hacking
- Design a reward model training pipeline and know its failure modes
- Build evaluation systems that actually measure what matters
- Think about safety rigorously, not as an afterthought

This track makes you dangerous in those interviews.

---

## Phase 1: Training Foundations (Weeks 1-7)

### Week 1: PyTorch Training from Scratch

**The Confidence Builder.** Before alignment, you need rock-solid training fundamentals. Build a complete training loop WITHOUT HuggingFace Trainer — raw PyTorch.

Build:
- Custom training loop: data loading, forward pass, loss, backward, optimizer step
- Learning rate schedulers from scratch: linear warmup, cosine decay
- Implement AdamW optimizer from scratch (understand momentum, second moments, weight decay)
- Train GPT-2 124M on a text dataset, observe loss curves
- Proper train/val split, gradient norm monitoring, overfitting detection

**Deliverable**: A from-scratch training script that trains a language model and produces clean loss curves.

**Interview signal**: "I understand every line of the training loop — no magic, no abstractions."

---

### Week 2: Mixed Precision and Memory Optimization

**The Efficiency Layer.** On 16GB VRAM, you can't waste a single byte. Learn every memory trick.

Build:
- FP16 training with GradScaler (understand gradient underflow/overflow)
- BF16 training (why it's better for LLMs — no loss scaling needed)
- Gradient checkpointing: implement manually, measure memory savings vs compute cost
- Gradient accumulation: simulate batch_size=64 when you can only fit batch_size=4
- Memory profiling: document exactly where VRAM goes (params, grads, optimizer states, activations)
- The memory equation: params × 16 bytes for AdamW FP32 (param + grad + 2 optimizer states × 4 bytes each)

**Deliverable**: Train a 1B model on 16GB VRAM using all techniques combined. Published memory breakdown.

---

### Week 3: Data Pipelines for LLM Training

**The Overlooked Fundamental.** Bad data = bad model. Most engineers skip this.

Build:
- Implement BPE tokenization from scratch (simplified) to understand the algorithm
- Dataset preparation: tokenize + pack sequences efficiently (fill context windows, no waste)
- Data mixing: combine datasets with configurable ratios (like Llama 2's recipe)
- Data quality: deduplication (MinHash), perplexity-based filtering
- Streaming data loading (don't OOM on large datasets)
- Dynamic batching by sequence length (minimize padding waste)

**Deliverable**: A reusable data pipeline: raw text → tokenized → packed → training-ready batches.

---

### Week 4: LoRA and QLoRA Fine-Tuning

**The Practical Skill.** This is how 90% of production fine-tuning works.

Build:
- Implement LoRA from scratch: inject low-rank matrices into attention layers, freeze base
- Understand the math: W_new = W_frozen + B×A where B∈R^(d×r), A∈R^(r×d), r << d
- Use PEFT library: LoraConfig, get_peft_model, merge/unmerge
- QLoRA: 4-bit NF4 quantized base + LoRA adapters (the memory-efficient miracle)
- Hyperparameter study: rank (4, 8, 16, 32, 64), alpha, which modules to target
- Compare: full fine-tune vs LoRA vs QLoRA on same task (quality, speed, memory)
- Fine-tune Mistral-7B with QLoRA on instruction-following

**Deliverable**: Comparison table showing quality/memory/speed across methods. Working QLoRA pipeline.

---

### Week 5: Instruction Tuning (SFT)

**The Bridge to Alignment.** SFT is stage 1 of the alignment pipeline.

Build:
- Understand: base model predicts next token. SFT model follows instructions. The difference is the data.
- Prepare instruction dataset (Alpaca/ShareGPT format, proper chat templates)
- Loss masking: only compute loss on ASSISTANT tokens, not on the prompt
- Multi-turn conversation handling: proper special token management
- SFT training pipeline on 3-7B model with QLoRA
- Compare: base model vs SFT model on instruction-following benchmarks

**Deliverable**: An SFT pipeline that turns a base model into an instruction-following model. Side-by-side examples showing the difference.

---

### Week 6: Distributed Training Concepts

**The Scale Understanding.** You won't run distributed training on one GPU, but you MUST understand it for interviews.

Build/Understand:
- Data Parallelism: implement AllReduce with torch.distributed (multi-process on localhost)
- Implement Ring AllReduce algorithm to understand the communication pattern
- FSDP: understand what's sharded (ZeRO-1: optimizer, ZeRO-2: +gradients, ZeRO-3: +params)
- Tensor Parallelism: how to split a linear layer across GPUs (column-parallel, row-parallel)
- Pipeline Parallelism: micro-batching, bubble overhead calculation
- The communication vs computation tradeoff at scale

**Deliverable**: Working multi-process AllReduce demo + written document explaining each parallelism strategy with diagrams. This is your interview cheat sheet.

---

### Week 7: Phase 1 Consolidation

Build:
- Integrate all pieces: data → train (with LoRA) → evaluate
- Set up W&B experiment tracking for all future runs
- Write training recipe: step-by-step guide for fine-tuning new tasks
- Blog post: "Training LLMs from Scratch: What HuggingFace Trainer Hides from You"
- Code quality, tests, documentation

**Phase 1 Milestone**: You can now train, fine-tune, and evaluate LLMs with full understanding of every component.

---

## Phase 2: Alignment Deep Dive (Weeks 8-14)

**This is the crown jewel of the Crucible track.** After these 7 weeks, you'll have implemented every major alignment method and can discuss any of them at the code level.

### Week 8: Reward Modeling

**The Foundation of RLHF.** Before you can optimize for human preferences, you need a model that predicts them.

Build:
- Bradley-Terry preference model: P(a preferred over b) = sigmoid(reward(a) - reward(b))
- Take a language model, add a scalar value head, train on preference pairs
- Dataset: HH-RLHF or UltraFeedback (human preference data)
- Training: cross-entropy loss on preference pairs
- Evaluation: accuracy on held-out preferences (target: >65%)
- Understand reward hacking: length bias, sycophancy, format gaming

**Deliverable**: Trained reward model that scores responses and agrees with human preferences.

---

### Week 9: RLHF with PPO (The Big One)

**This is the week that wins Anthropic/OpenAI interviews.** Full RLHF from scratch.

Build:
- 4 models running simultaneously:
  - **Policy model** (being trained to generate better responses)
  - **Reference model** (frozen copy — the KL anchor)
  - **Reward model** (from Week 8 — scores responses)
  - **Value model** (critic — predicts expected reward for variance reduction)
- The training loop:
  1. Policy generates responses to prompts
  2. Reward model scores each response
  3. Compute KL penalty (policy vs reference — prevents mode collapse)
  4. Compute advantage estimates (GAE — Generalized Advantage Estimation)
  5. PPO update: clipped objective, value function loss, entropy bonus
- Monitor: reward over time, KL divergence, policy entropy, response quality

**Deliverable**: Working RLHF pipeline that measurably improves model responses over SFT baseline.

**Interview signal**: "I implemented PPO for language models from scratch. I can explain why KL penalty exists, how GAE reduces variance, and why clipping stabilizes training."

---

### Week 10: DPO (Direct Preference Optimization)

**The elegant alternative to RLHF.** Fewer models, simpler training, comparable results.

Build:
- Derive DPO loss from the RLHF objective (understand the mathematical connection)
- Key insight: DPO defines an implicit reward model — no explicit reward model needed
- Implement DPO loss: log-probability ratio between chosen/rejected responses
- Train on same preference data as RLHF (fair comparison)
- Compare DPO vs RLHF: stability, compute cost, quality, failure modes
- Implement offline DPO + understand distribution shift problem
- IPO (Identity Preference Optimization) as a regularized variant

**Deliverable**: Working DPO pipeline + comparison against RLHF (same data, same base model, published results).

---

### Week 11: Advanced Alignment Methods

**Stay current.** The field moves fast — know ALL the options.

Implement from scratch:
- **KTO** (Kahneman-Tversky Optimization): only needs good/bad labels, not paired preferences
- **ORPO** (Odds Ratio Preference Optimization): combines SFT + alignment in one step
- **SimPO** (Simple Preference Optimization): reference-free, length-normalized

Build comparison:
- All methods on same dataset, same base model, same compute budget
- Metrics: LLM-as-judge win-rate, benchmark scores, training stability, compute cost
- Table showing when each method is best (data requirements, stability, quality)

**Deliverable**: 5-method comparison table with statistical significance (multiple seeds).

---

### Week 12: Constitutional AI and Self-Critique

**Anthropic's approach.** Directly relevant if that's your target.

Build:
- Define a constitution: set of principles (helpfulness, harmlessness, honesty)
- Pipeline: generate → self-critique ("Does this violate any principle?") → revise
- Use revised responses as preference data for DPO training
- Automated red-teaming: generate adversarial prompts, find model weaknesses
- Iterative refinement: train → red-team → identify failures → retrain

**Deliverable**: Constitutional AI pipeline that iteratively improves model safety.

---

### Week 13: Multi-Turn Alignment

**The unsolved problem.** Most alignment is single-turn. Real conversations are multi-turn.

Build:
- Conversation-level rewards (score the full conversation, not just last response)
- Credit assignment: which turn contributed to good/bad conversation outcome?
- Conversation trees: branch at different points, compare trajectories
- Best-of-N at inference: generate N responses per turn, select best
- Multi-turn DPO: apply preference optimization at conversation level

**Deliverable**: Multi-turn alignment pipeline with conversation-level evaluation.

---

### Week 14: Alignment Methods Comparison (Research Report)

**The capstone of Phase 2.** Produce a research-quality comparison.

Build:
- Fixed experimental setup across ALL methods (Weeks 8-13)
- Metrics: win-rate (LLM-judge), benchmark scores, safety scores, human preference simulation
- Ablation studies: vary KL coefficient, rank, data size, base model
- Analysis: when does each method shine? when does it fail?
- Research report (8-10 pages): introduction, methods, results, analysis, conclusion
- Blog post: "RLHF vs DPO vs KTO: A Practitioner's Comparison"

**Deliverable**: Research-quality report + blog post. This is what makes your profile stand out from pure engineers.

---

## Phase 3: Evaluation, Safety + Portfolio (Weeks 15-20)

### Week 15: Comprehensive Evaluation Framework

Build benchmark harness (MMLU, HumanEval, GSM8K, HellaSwag, TruthfulQA) with statistical rigor (bootstrap CIs, significance testing). Compare all model variants systematically.

### Week 16: Safety Evaluation and Red-Teaming

Build safety test suite: toxicity, jailbreak resistance (100+ adversarial prompts), refusal calibration, bias evaluation. Produce safety scorecard for each model variant.

### Week 17: LLM-as-Judge and Human Preference Evaluation

Build LLM-as-judge system, ELO rating for models, arena-style pairwise comparison. Validate judge against human preferences.

### Week 18: Distributed Training Infrastructure

Implement FSDP wrapper, understand sharding strategies, fault-tolerant training with checkpoint recovery. Run multi-process training simulation.

### Weeks 19-20: Portfolio Polish

4 blog posts, research report, training recipes, model cards, demo showing base vs SFT vs DPO vs RLHF side-by-side.

---

## What You DON'T Need to Do

- **Skip**: Pre-training a model from scratch (you'd need 8xA100s — understand it conceptually)
- **Skip**: Building a full RLHF system that runs at scale (demonstrate understanding on small scale)
- **Skip**: Implementing every optimizer variant (AdamW is enough)
- **Skip**: Multi-GPU training (understand concepts, simulate with multi-process)
- **Skip**: Building evaluation infrastructure from scratch (use lm-evaluation-harness as base)

---

## The Resume Lines This Produces

> "Implemented complete RLHF pipeline from scratch (PPO with KL-penalized reward optimization), DPO, KTO, ORPO, and Constitutional AI on consumer hardware. Published rigorous comparison across 5 alignment methods with ablation studies, demonstrating 18% win-rate improvement over SFT baseline."

> "Built comprehensive LLM evaluation framework covering capability benchmarks (MMLU, HumanEval, GSM8K), safety evaluation (jailbreak resistance, refusal calibration), and LLM-as-judge preference evaluation with ELO rating system."

---

## Daily Routine

- **1.5 hours**: Study — read the key paper sections, watch alignment talks (Anthropic talks are gold)
- **4-5 hours**: Implement — write code, run training, debug loss curves
- **30 min**: Document — training logs, loss curve analysis, what worked/failed
- **30 min**: Write — blog paragraphs, comparison notes, research report sections

---

## Key Resources (Don't Read Everything — Focus on These)

1. InstructGPT paper (OpenAI) — the original RLHF paper, read Section 3 (method)
2. DPO paper — read Section 3 (derivation from RLHF objective)
3. Constitutional AI paper (Anthropic) — read Section 2 (method)
4. Anthropic's "Training a Helpful and Harmless AI" — the full alignment picture
5. Ziegler et al. "Fine-Tuning Language Models from Human Preferences" — early RLHF
6. KTO, ORPO, SimPO papers — read abstracts + method sections only

---

## Phase Gates

**End of Phase 1 (Week 7)**: Can train/fine-tune models from scratch, explain every optimizer/memory trick.

**End of Phase 2 (Week 14)**: Can implement RLHF, DPO, KTO from scratch. Can derive DPO loss on a whiteboard. Published comparison with statistical rigor.

**End of Phase 3 (Week 20)**: Full evaluation + safety pipeline. Research report published. Blog posts live. Can ace any alignment interview question.
