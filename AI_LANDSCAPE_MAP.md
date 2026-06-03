# The AI Landscape Map

> A complete topographical map of the AI field. Use this to locate any conversation, understand connections, and never feel lost again.

---

## How to Read This Map

Think of AI as a continent with 8 major regions. Each region has sub-areas (like states), landmarks (key papers/tools), and roads connecting to other regions. When someone is talking about something, find which region it belongs to, and you immediately understand the context.

---

## THE MAP: 8 Major Regions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          THE AI LANDSCAPE                                    │
│                                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐               │
│  │ 1. MODEL    │───▶│ 2. TRAINING  │───▶│ 3. ALIGNMENT    │               │
│  │ ARCHITECTURE│    │              │    │    & SAFETY      │               │
│  └──────┬──────┘    └──────┬───────┘    └────────┬────────┘               │
│         │                  │                     │                         │
│         ▼                  ▼                     ▼                         │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐               │
│  │ 4. INFERENCE│◀───│ 5. INFRA-    │───▶│ 6. DATA         │               │
│  │ & SERVING   │    │ STRUCTURE    │    │    ENGINEERING   │               │
│  └──────┬──────┘    └──────────────┘    └─────────────────┘               │
│         │                                                                  │
│         ▼                                                                  │
│  ┌─────────────┐    ┌──────────────┐                                      │
│  │ 7. APPLI-   │◀───│ 8. RESEARCH  │                                      │
│  │ CATIONS     │    │ FRONTIERS    │                                      │
│  └─────────────┘    └──────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Region 1: MODEL ARCHITECTURE

**What it is**: The mathematical and structural design of neural networks. The "blueprints" of AI models.

### Sub-Areas

| Sub-Area | What It Means | Key Terms You'll Hear |
|----------|--------------|----------------------|
| **Transformers** | The dominant architecture since 2017. ALL modern LLMs are transformers. | Self-attention, multi-head attention, encoder-decoder |
| **Attention Mechanisms** | How the model decides which parts of the input to focus on | Q/K/V (Query/Key/Value), scaled dot-product, causal mask |
| **Position Encoding** | How models know word ORDER (transformers have no inherent sequence sense) | RoPE, ALiBi, learned positional embeddings, relative position |
| **Normalization** | Keeping numbers in a healthy range during computation | LayerNorm, RMSNorm (modern), BatchNorm (older, vision) |
| **Activation Functions** | Non-linearities that give networks their power | ReLU (old), GELU, SiLU/Swish, SwiGLU (modern LLMs) |
| **Architecture Variants** | Different configurations of the transformer | Decoder-only (GPT), Encoder-only (BERT), Encoder-Decoder (T5) |
| **Mixture of Experts (MoE)** | Not all parameters active per token — route to "expert" sub-networks | Routing, top-k experts, load balancing, Mixtral |
| **State Space Models** | Alternative to transformers — linear-time sequence processing | Mamba, S4, RWKV, Hyena |
| **Multimodal** | Models that handle multiple data types (text + images + audio) | Vision encoder, cross-attention, CLIP, projection layers |

### Key Connections
- Architecture → **Training** (architecture determines what's trainable and how)
- Architecture → **Inference** (architecture determines memory/compute requirements)
- Architecture → **Research** (new architectures are active research)

### Landmarks (Papers/Models)
- "Attention Is All You Need" (2017) — the transformer paper
- GPT series (OpenAI) — decoder-only, autoregressive
- Llama series (Meta) — open decoder-only with RoPE, RMSNorm, SwiGLU
- Mixtral (Mistral) — mixture of experts
- Mamba (Gu & Dao) — state space model alternative

---

## Region 2: TRAINING

**What it is**: The process of teaching models from data. Everything from data preparation to optimization.

### Sub-Areas

| Sub-Area | What It Means | Key Terms You'll Hear |
|----------|--------------|----------------------|
| **Pre-training** | Training from scratch on massive data (trillions of tokens) | Next-token prediction, causal language modeling, compute-optimal |
| **Fine-tuning** | Adapting a pre-trained model to a specific task/domain | Full fine-tune, transfer learning, domain adaptation |
| **Parameter-Efficient (PEFT)** | Fine-tuning only a SMALL part of the model | LoRA, QLoRA, adapters, prefix tuning, prompt tuning |
| **Instruction Tuning (SFT)** | Teaching models to follow instructions | Supervised fine-tuning, chat format, loss masking |
| **Optimization** | How gradients update weights | AdamW, learning rate schedulers, warmup, weight decay |
| **Mixed Precision** | Training with lower precision numbers to save memory | FP16, BF16, loss scaling, GradScaler |
| **Distributed Training** | Splitting training across multiple GPUs | Data parallel, tensor parallel, pipeline parallel, FSDP, ZeRO |
| **Data Curation** | Preparing and cleaning training data | Deduplication, filtering, tokenization, data mixing |
| **Scaling Laws** | Mathematical relationships between compute/data/params and performance | Chinchilla optimal, compute-optimal training, power laws |
| **Continual Learning** | Updating models without forgetting previous knowledge | Catastrophic forgetting, replay buffers, elastic weight consolidation |

### Key Connections
- Training → **Alignment** (SFT is stage 1 of alignment)
- Training → **Infrastructure** (distributed training needs cluster management)
- Training → **Data** (garbage in = garbage out)

### Landmarks
- "Scaling Laws for Neural Language Models" (Kaplan et al.)
- "Training Compute-Optimal LLMs" (Chinchilla paper)
- LoRA paper (Hu et al.)
- "LLaMA: Open and Efficient Foundation Language Models" (Meta)

### Who Works Here
- Pre-training: OpenAI, Anthropic, Google, Meta, xAI (very few companies do this)
- Fine-tuning: Almost every AI company
- Distributed training infra: NVIDIA, Microsoft (DeepSpeed), Meta (FSDP)

---

## Region 3: ALIGNMENT AND SAFETY

**What it is**: Making models helpful, harmless, and honest. The process of aligning model behavior with human values.

### Sub-Areas

| Sub-Area | What It Means | Key Terms You'll Hear |
|----------|--------------|----------------------|
| **RLHF** | Reinforcement Learning from Human Feedback — train with human preferences | Reward model, PPO, KL penalty, value model, policy |
| **DPO** | Direct Preference Optimization — simpler alternative to RLHF | Implicit reward, log-probability ratio, offline optimization |
| **Reward Modeling** | Training a model to predict human preferences | Bradley-Terry, preference pairs, reward hacking |
| **Constitutional AI** | Self-improvement through principles (Anthropic's approach) | Self-critique, revision, principles, RLAIF |
| **Safety** | Preventing harmful outputs | Refusal, jailbreaks, toxicity, red-teaming |
| **Evaluation** | Measuring model capabilities and safety | Benchmarks (MMLU, HumanEval), LLM-as-judge, ELO |
| **Interpretability** | Understanding WHAT models learn internally | Mechanistic interpretability, probing, feature visualization |
| **Scalable Oversight** | How to supervise models smarter than humans | Debate, recursive reward modeling, weak-to-strong |

### Key Connections
- Alignment → **Training** (alignment IS training with preference data)
- Alignment → **Applications** (aligned models are what users interact with)
- Alignment → **Research** (active frontier — no solved problems here)

### Landmarks
- "Training language models to follow instructions" (InstructGPT/RLHF — OpenAI)
- "Direct Preference Optimization" (DPO — Rafailov et al.)
- "Constitutional AI" (Anthropic)
- "Scaling Laws for Reward Model Overoptimization"

### Who Works Here
- Anthropic (alignment-first company), OpenAI alignment team, DeepMind alignment team
- This is THE differentiator between "AI company" and "AI safety company"

---

## Region 4: INFERENCE AND SERVING

**What it is**: Running trained models efficiently to serve predictions. Making models FAST and CHEAP in production.

### Sub-Areas

| Sub-Area | What It Means | Key Terms You'll Hear |
|----------|--------------|----------------------|
| **KV-Cache** | Caching past computations to avoid recomputing | PagedAttention, block allocation, cache eviction |
| **Batching** | Processing multiple requests together | Static batching, continuous batching, iteration-level scheduling |
| **Quantization** | Reducing model precision to use less memory and compute | INT8, INT4, GPTQ, AWQ, NF4, GGUF |
| **Model Serving** | Systems that host models and handle requests | vLLM, TGI (HuggingFace), TensorRT-LLM, Ollama, Triton |
| **Speculative Decoding** | Use small model to draft, large model to verify | Draft model, acceptance sampling, parallel verification |
| **Kernel Optimization** | Custom GPU code for specific operations | FlashAttention, Triton kernels, fused operations |
| **Model Compilation** | Converting models to optimized formats | TensorRT, torch.compile, ONNX, graph optimization |
| **Edge/On-Device** | Running models on phones, laptops, embedded | GGML, llama.cpp, MLX (Apple), quantization for edge |

### Key Connections
- Inference → **Architecture** (model structure determines inference patterns)
- Inference → **Infrastructure** (serving needs clusters, scheduling, monitoring)
- Inference → **Applications** (apps consume inference APIs)

### Landmarks
- vLLM paper ("Efficient Memory Management for LLM Serving with PagedAttention")
- FlashAttention (Dao et al.)
- "Fast Transformer Decoding" (speculative decoding)

### Who Works Here
- vLLM team (UC Berkeley), NVIDIA (TensorRT), HuggingFace (TGI)
- Modal, Anyscale, Replicate, Together AI, Fireworks AI
- Every company serving LLMs

### YOUR PROJECTS HERE: Forge lives in this region

---

## Region 5: INFRASTRUCTURE

**What it is**: The hardware, systems, and platforms that make AI possible at scale.

### Sub-Areas

| Sub-Area | What It Means | Key Terms You'll Hear |
|----------|--------------|----------------------|
| **GPU Hardware** | The physical chips that run AI | H100, A100, RTX 5080, TPU, VRAM, HBM, NVLink |
| **Cluster Management** | Managing thousands of GPUs | Kubernetes, Slurm, scheduling, job orchestration |
| **Networking** | Connecting GPUs for distributed work | InfiniBand, RDMA, NCCL, AllReduce, network topology |
| **Storage** | Where data and models live | Object storage, distributed filesystems, checkpoints |
| **MLOps/MLPlatform** | Tools for ML lifecycle management | MLflow, W&B, feature stores, model registries |
| **Cloud AI** | Cloud providers' AI offerings | AWS SageMaker, GCP Vertex AI, Azure ML |
| **Observability** | Monitoring AI systems in production | Prometheus, GPU metrics, model monitoring, drift detection |
| **Cost Optimization** | Making AI affordable | Spot instances, GPU sharing, MIG, time-slicing |

### Key Connections
- Infrastructure → **Training** (training runs ON infrastructure)
- Infrastructure → **Inference** (serving runs ON infrastructure)
- Infrastructure → **Data** (data pipelines are infrastructure)

### Who Works Here
- NVIDIA (hardware + software), cloud providers (AWS, GCP, Azure)
- Anyscale (Ray), Modal, CoreWeave, Lambda Labs
- Every AI company's platform/infra team

### YOUR PROJECTS HERE: Anvil lives in this region

---

## Region 6: DATA ENGINEERING

**What it is**: Everything about the data that feeds AI systems.

### Sub-Areas

| Sub-Area | What It Means | Key Terms You'll Hear |
|----------|--------------|----------------------|
| **Pre-training Data** | Massive web-scale datasets for initial training | Common Crawl, The Pile, RedPajama, deduplication |
| **Synthetic Data** | AI-generated data for training AI | Self-instruct, distillation data, constitutional data |
| **Preference Data** | Human judgments of what's good/bad | RLHF data, preference pairs, annotation |
| **Evaluation Data** | Benchmarks and test sets | MMLU, HumanEval, GSM8K, contamination concerns |
| **Data Quality** | Ensuring data is clean and useful | Filtering, dedup, toxicity removal, perplexity filtering |
| **Feature Engineering** | Creating useful inputs for models | Feature stores, embeddings, representation learning |
| **Vector Databases** | Storing and searching embeddings | Pinecone, Qdrant, Weaviate, Milvus, pgvector |
| **Data Versioning** | Tracking what data was used when | DVC, Delta Lake, lakeFS |

### Key Connections
- Data → **Training** (data IS the training signal)
- Data → **Applications** (RAG uses vector databases)
- Data → **Alignment** (preference data drives alignment)

### YOUR PROJECTS HERE: Conduit lives in this region

---

## Region 7: APPLICATIONS

**What it is**: How AI models are used to solve real problems.

### Sub-Areas

| Sub-Area | What It Means | Key Terms You'll Hear |
|----------|--------------|----------------------|
| **RAG** | Retrieval-Augmented Generation — ground models in external knowledge | Chunking, embeddings, retrieval, context window, reranking |
| **Agents** | Models that can take actions, use tools, plan multi-step tasks | Tool use, function calling, planning, ReAct, chain-of-thought |
| **Code Generation** | Models that write/understand code | Copilot, code completion, test generation, debugging |
| **Chatbots/Assistants** | Conversational AI products | Claude, ChatGPT, multi-turn, system prompts |
| **Search** | AI-powered information retrieval | Semantic search, neural ranking, hybrid search |
| **Multimodal Apps** | Apps using text + images + audio + video | Vision-language models, image generation, TTS, STT |
| **Autonomous Systems** | AI that operates independently | Self-driving, robotics, game agents |

### Key Connections
- Applications → **Inference** (apps consume model predictions)
- Applications → **Data** (RAG connects apps to data)
- Applications → **Alignment** (apps need aligned models to be safe)

---

## Region 8: RESEARCH FRONTIERS

**What it is**: Active areas where the field is advancing. What people are trying to figure out right now.

### Sub-Areas

| Sub-Area | What It Means | Key Terms You'll Hear |
|----------|--------------|----------------------|
| **Reasoning** | Making models think step-by-step, solve complex problems | Chain-of-thought, tree-of-thought, o1-style reasoning, test-time compute |
| **Long Context** | Handling very long inputs (100K+ tokens) | Context extension, ring attention, landmark attention |
| **Efficiency** | Making models smaller/faster without quality loss | Distillation, pruning, architectural innovations |
| **Multimodal** | Unified models for all modalities | Omni models, any-to-any, world models |
| **Agents** | Models that can plan, use tools, and act autonomously | Tool use, memory, planning, self-correction |
| **Synthetic Data** | Using AI to generate training data for AI | Self-improvement, data augmentation, RLAIF |
| **Scaling** | What happens as models get bigger | Emergent abilities, phase transitions, scaling laws |
| **Interpretability** | Understanding model internals | Sparse autoencoders, circuit analysis, probing |
| **World Models** | Models that understand physics/causality, not just text patterns | Simulation, planning, Sora-style video models |

---

## THE GLOSSARY: Terms That Come Up Everywhere

### Model Sizes (When People Say...)

| Term | Meaning | Example |
|------|---------|---------|
| "Small model" | 1-7B parameters | Llama-3-8B, Mistral-7B |
| "Medium model" | 13-70B parameters | Llama-2-70B, Mixtral-8x22B |
| "Frontier model" | 100B+ (or MoE equivalent) | GPT-4, Claude 3.5, Gemini Ultra |
| "Foundation model" | A pre-trained model before task-specific tuning | Any base model |
| "Open-weight" | Weights downloadable, but training data/code may not be shared | Llama, Mistral |
| "Open-source" | Full code, data, training recipe shared | OLMo (AI2) |

### Common Abbreviations

| Abbreviation | Full Form | Region |
|---|---|---|
| LLM | Large Language Model | General |
| SFT | Supervised Fine-Tuning | Training/Alignment |
| RLHF | Reinforcement Learning from Human Feedback | Alignment |
| DPO | Direct Preference Optimization | Alignment |
| PPO | Proximal Policy Optimization | Alignment |
| KV-cache | Key-Value Cache | Inference |
| FSDP | Fully Sharded Data Parallel | Training/Infra |
| MoE | Mixture of Experts | Architecture |
| RAG | Retrieval-Augmented Generation | Applications |
| PEFT | Parameter-Efficient Fine-Tuning | Training |
| LoRA | Low-Rank Adaptation | Training |
| QLoRA | Quantized LoRA | Training |
| VRAM | Video RAM (GPU memory) | Infra/Inference |
| HBM | High Bandwidth Memory | Hardware |
| TTFT | Time To First Token | Inference |
| TPS | Tokens Per Second | Inference |
| FLOPS | Floating Point Operations Per Second | Hardware |
| NCCL | NVIDIA Collective Communication Library | Infra |
| MIG | Multi-Instance GPU | Infra |

### Concepts Everyone References

| Concept | One-Line Explanation |
|---------|---------------------|
| **Attention** | The mechanism that lets tokens "look at" other tokens to understand context |
| **Autoregressive** | Generate one token at a time, using all previous tokens as input |
| **Emergent abilities** | Capabilities that appear suddenly at certain model scales |
| **Hallucination** | Model generates confident but factually incorrect information |
| **Grounding** | Connecting model outputs to verified external knowledge (RAG does this) |
| **Prompt engineering** | Crafting inputs to get better outputs (no model changes) |
| **In-context learning** | Model learns from examples in the prompt (few-shot) without training |
| **Transfer learning** | Pre-train on general data, fine-tune on specific task |
| **Distillation** | Train a small model to mimic a large model's outputs |
| **Tokenization** | Converting text to numbers the model can process (BPE, SentencePiece) |
| **Embedding** | Dense vector representation of text/data in continuous space |
| **Perplexity** | How "surprised" the model is by text (lower = better language model) |
| **Loss** | The number being minimized during training (cross-entropy for LLMs) |
| **Gradient** | The direction to adjust weights to reduce loss |
| **Overfitting** | Model memorizes training data instead of learning patterns |
| **Inference** | Running a trained model to get predictions (not training) |
| **Latency** | Time from request to response |
| **Throughput** | Requests (or tokens) processed per second |
| **Context window** | Maximum input length the model can handle (4K, 32K, 128K, 1M tokens) |

---

## COMPANY MAP: Who Does What

### The Frontier Labs (Train Frontier Models)
| Company | Known For | Regions |
|---------|-----------|---------|
| **OpenAI** | GPT-4, ChatGPT, o1 reasoning | Architecture, Training, Alignment, Applications |
| **Anthropic** | Claude, Constitutional AI, safety-first | Alignment, Safety, Training |
| **Google DeepMind** | Gemini, AlphaFold, research breadth | Research, Architecture, Training |
| **Meta AI** | Llama (open-weight), PyTorch | Training, Architecture, Open Research |
| **xAI** | Grok, massive compute | Training, Infrastructure |
| **Mistral** | Efficient open models (Mixtral MoE) | Architecture, Training |

### Infrastructure and Serving
| Company | Known For | Regions |
|---------|-----------|---------|
| **NVIDIA** | GPU hardware, CUDA, TensorRT | Infrastructure, Hardware |
| **Anyscale** | Ray (distributed computing) | Infrastructure |
| **Modal** | Serverless GPU compute | Infrastructure, Inference |
| **Replicate** | Model hosting/serving | Inference |
| **Together AI** | Open model inference | Inference |
| **CoreWeave** | GPU cloud provider | Infrastructure |
| **Fireworks AI** | Fast model serving | Inference |
| **vLLM team** | Open-source inference engine | Inference |

### ML Platform and Tools
| Company | Known For | Regions |
|---------|-----------|---------|
| **Databricks** | MLflow, data + ML platform | Data, MLOps |
| **Weights & Biases** | Experiment tracking | Training, MLOps |
| **Hugging Face** | Model hub, transformers library, TGI | All regions (democratization) |
| **Cohere** | Enterprise LLMs, embeddings, RAG | Applications, Training |
| **LangChain/LlamaIndex** | Application frameworks for LLMs | Applications |

---

## YOUR POSITION ON THE MAP

```
Region 1 (Architecture):     Week 8 of Forge teaches you this
Region 2 (Training):         Crucible covers this entirely
Region 3 (Alignment):        Crucible Phase 2 is deep here
Region 4 (Inference):        Forge is DEEP in this region
Region 5 (Infrastructure):   Anvil owns this region
Region 6 (Data):             Conduit Phase 1 covers this
Region 7 (Applications):     Forge Weeks 2-3 (RAG) touches this
Region 8 (Research):         Touched across all projects
```

---

## CONVERSATION DECODER

When someone says... they're talking about this region:

| They Say | They Mean | Region |
|----------|-----------|--------|
| "We need to reduce TTFT" | Time-to-first-token in inference is too slow | Inference |
| "The model is hallucinating" | Outputs are factually wrong | Applications/Alignment |
| "We should try DPO instead" | Switching alignment method from RLHF | Alignment |
| "KV-cache is fragmenting" | Memory management issue in serving | Inference |
| "Let's quantize to INT4" | Reduce model size for faster/cheaper inference | Inference |
| "The reward model is being gamed" | Model found shortcuts to get high reward without being helpful | Alignment |
| "We need more FLOPS" | Need more compute hardware | Infrastructure |
| "Training is communication-bound" | GPUs waiting on network, not computing | Infra/Training |
| "Let's use LoRA instead of full fine-tune" | Save memory by training less parameters | Training |
| "The retrieval recall is low" | RAG is not finding relevant documents | Applications/Data |
| "We should add a MoE layer" | Use sparse architecture for efficiency | Architecture |
| "The model collapsed during RLHF" | Model started giving degenerate outputs (mode collapse) | Alignment |
| "We need to check for data contamination" | Test data might have leaked into training | Data/Evaluation |
| "Scaling law says we need more data" | Mathematical prediction of performance vs data size | Research |
| "The context window isn't long enough" | Model can't handle the input length needed | Architecture |
| "Continuous batching improved throughput 3x" | Better scheduling of inference requests | Inference |
| "We're seeing drift in production" | Input data patterns changed, model may be degrading | MLOps/Data |
| "The pod is OOMing" | Kubernetes pod running out of GPU memory | Infrastructure |
| "Let's try speculative decoding" | Use small model to speed up large model inference | Inference |
| "We need gang scheduling for training" | All training workers must be scheduled together | Infrastructure |

---

## HOW REGIONS CONNECT (The Roads Between Them)

```
Architecture ──defines──▶ Training (what you CAN train)
Training ──produces──▶ Alignment (SFT model → alignment input)
Alignment ──produces──▶ Applications (aligned model → user-facing product)
Applications ──requires──▶ Inference (serving the model to users)
Inference ──runs on──▶ Infrastructure (hardware and orchestration)
Infrastructure ──enables──▶ Training (clusters for training jobs)
Data ──feeds──▶ Training (training data)
Data ──feeds──▶ Alignment (preference data)
Data ──feeds──▶ Applications (RAG knowledge bases)
Research ──improves──▶ All regions (new methods, architectures, techniques)
```

---

## THE TIMELINE: How a Model Goes from Idea to Production

```
1. ARCHITECTURE DESIGN     → "Let's use a 7B decoder-only transformer with GQA and RoPE"
2. DATA CURATION           → "Clean 2T tokens from web, books, code, filter quality"
3. PRE-TRAINING            → "Train for 2 weeks on 256 GPUs, watch loss curve"
4. SFT (Instruction Tune)  → "Fine-tune on 100K instruction-response pairs"
5. ALIGNMENT (RLHF/DPO)    → "Train on human preferences to be helpful and safe"
6. EVALUATION              → "Run MMLU, HumanEval, safety evals, red-team"
7. OPTIMIZATION            → "Quantize to INT4, compile with TensorRT"
8. DEPLOYMENT              → "Deploy on K8s with auto-scaling, monitoring"
9. MONITORING              → "Track drift, quality, latency in production"
10. FEEDBACK               → "Collect user feedback, retrain/improve"
```

This is the FULL lifecycle. Different roles own different stages:
- **Crucible**: Steps 1-6
- **Forge**: Steps 7-8
- **Anvil**: Steps 8-9 (infrastructure layer)
- **Conduit**: Steps 2, 6-10 (end-to-end lifecycle)

---

## QUICK REFERENCE: "What Should I Read/Watch to Understand Region X?"

| Region | Best Single Resource |
|--------|---------------------|
| Architecture | Andrej Karpathy "Let's Build GPT" (YouTube, 2hr) |
| Training | Karpathy "Let's Build the GPT Tokenizer" + nanoGPT repo |
| Alignment | Anthropic's "Core Views on AI Safety" blog post |
| Inference | vLLM paper + their GitHub README |
| Infrastructure | Google SRE Book (free online) + K8s docs |
| Data | Chip Huyen "Designing ML Systems" (book) |
| Applications | LangChain docs + OpenAI cookbook |
| Research | "Situational Awareness" (Leopold Aschenbrenner) for big picture |
