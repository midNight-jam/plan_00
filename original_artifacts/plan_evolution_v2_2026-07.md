# Plan Evolution v2 — Strategic Review & Future-Proofed Plan
**Date:** 2026-07-06 · **Reviewer role:** Senior Technical Fellow (research–infrastructure boundary) · **Horizon:** March 2027, +12 months beyond
**Compare against:** `forge_practical_build_plan_615ca283.plan.md`, `forge_gpu_orchestrator_3a49e971.plan.md`, `anvil_ai_infrastructure_plan_3142860d.plan.md` (this directory), plus the live specs in `forge/specs/` and `anvil/specs/`.

---

## Executive Summary — the five things that matter

1. **Your instincts were right, your APIs are stale.** The Forge+Anvil combination is the correct choice, and the deep-systems weeks (transformer internals, KV-cache, continuous batching, training orchestrator, GPU health) are exactly the durable core. But the Kubernetes weeks are written against the 2024 stack: **DRA went GA in K8s 1.34 (March 2026)** and replaced the device-plugin model your operator and scheduler specs assume. Modernize before you build, not after.
2. **Your biggest gap is RL infrastructure.** Your plan has PPO and DPO but zero GRPO/RLVR and zero RL-*systems* engineering — and RL post-training infra (rollout workers, inference-engine colocation, mid-training weight sync, environment orchestration) is the single hottest hiring area at the research-infrastructure boundary in 2026. It is a distributed-systems problem wearing an ML costume. It is *your* problem shape.
3. **~7 weeks of your plan re-proves your day job.** API gateways, Terraform/ArgoCD, Vault, generic SRE, CLI polish, RAG depth — a hiring manager reads your resume and already believes you can do these. Every week spent proving what's already proven is a week stolen from what isn't.
4. **Your hardware is a moat, not a limitation.** Consumer Blackwell (RTX 5080, sm_120) is rare in every OSS test matrix. You already hit real sm_120 pain in Forge Week 1. FP4/NVFP4 benchmarks, sm_120 bug reports, and Blackwell kernel profiles are inherently novel content that Hopper-cloud users cannot produce.
5. **By 2027, "I built X" is fakeable; "I measured X and fixed what broke" is not.** Every interviewer will know candidates can generate a continuous-batching scheduler with an AI assistant in a day. The un-fakeable signals: original measurements on real hardware, discovered bugs, **merged upstream OSS PRs**, and write-ups with numbers nobody else has. Your plan currently contains zero OSS-contribution content. Fix that starting this week.

**Net changes at a glance:** Drop Conduit entirely (confirmed). Compress ~7 generic weeks across Forge+Anvil. Add: agent-native serving (Forge W3), disaggregation simulation (Forge W11), FP4-on-Blackwell (Forge W12), DRA/Kueue modernization (Forge W15, Anvil W2), GRPO (Forge W17), a 3-week **RL post-training infrastructure flagship** (Anvil W8–10), NCCL/collectives from scratch (Anvil W1), paper sprints in every consolidation week, and an OSS-contribution track with its own acceptance criteria.

---

# Section A: The 12-Month Technology Forecast (July 2026 → mid-2027)

## A1. What already happened while you were planning (post-Jan-2026 shifts)

**Disaggregated serving went GA and mainstream.** NVIDIA Dynamo 1.0 shipped at GTC in March 2026 as the orchestration layer above vLLM/SGLang/TensorRT-LLM, routing prefill and decode to separate worker pools with KV transfer over NIXL. llm-d (Red Hat/IBM/Google) was donated to CNCF at KubeCon EU 2026. SGLang published a 96×H100 DeepSeek-R1 deployment (3 prefill / 9 decode nodes) at ~$0.20 per million output tokens — ~5x cheaper than the official API. ByteDance's MegaScale-Infer disaggregates *attention from FFN* within decode, on heterogeneous GPUs. Consequences for you:

- "Wrap vLLM in FastAPI on one GPU" is the **floor**. Fine as Week-1 substrate; worthless as a headline.
- The scarce skill is now **KV-cache economics**: what to cache, where (GPU/CPU/NVMe), when to transfer vs recompute, how to route requests to maximize prefix hits, and how to capacity-plan disaggregated pools against SLOs ("goodput" — throughput that meets latency SLOs — replaced raw tokens/s as the metric that matters).
- Heterogeneous serving (expensive GPUs for prefill, cheaper for decode) makes **cost-aware routing** a real discipline, not a dashboard.

**Kubernetes became AI-native — via different APIs than your specs assume.** DRA (Dynamic Resource Allocation) went GA in K8s 1.34: pods declare `ResourceClaim`s against `DeviceClass`es; a DRA driver binds real devices with topology constraints, fine-grained sharing, and parameterized selection. The `nvidia.com/gpu: 1` extended-resource model — and with it the scheduler-extender and custom-`nvidia.com/vram`-resource patterns your Anvil W2 and Forge W15 specs are built on — is now the legacy path. Kueue (queue admission, quotas, preemption) and KAI (gang scheduling, topology-aware placement, DRA integration) own the batch/training layer. Gateway API Inference Extension (`InferencePool`, body-based routing on model name and LoRA adapter) owns serving ingress. **Build against these; contributing to them is worth more than cloning them.**

**RL for agents became the center of post-training — and its bottleneck is infrastructure.** GRPO (no value model) and descendants (DAPO, variants) displaced PPO as the workhorse; RLVR (verifiable rewards) displaced learned-reward-only pipelines for reasoning/agentic tasks. RL *environments* became products (HUD's environment SDK + cloud execution; environment hubs; OpenTinker's "RL-as-a-service" separation of environment from trainer). Frameworks (OpenRLHF, verl, RLinf, OpenTinker) all converge on the same architecture: **rollout workers running an inference engine (usually vLLM) + a trainer + weight synchronization + environment orchestration + reward pipelines**. Every hard problem in that sentence is distributed systems: async pipelines, backpressure, checkpointing, colocation vs disaggregation of trainer and engine, numerics consistency between trainer and engine logprobs. Labs cannot hire enough people who can build this. This is the highest-leverage pivot available to you.

**Agentic workloads restructured inference traffic.** Action-taking MCP tools went from 24% to 65% of the ecosystem in ~18 months. An agent session is: long shared prefix, many turns, interleaved tool calls with dead time, branching (retries, best-of-n), and enormous prefix reuse across steps. This inverts serving priorities: prefix-cache hit rate and session-aware routing dominate; KV cache becomes the contended resource; TTFT-per-*step* and end-to-end *task* latency replace per-request metrics. Almost nobody has published rigorous **agentic-trace serving benchmarks**. That's an open niche you can own with one GPU.

**Evaluation is breaking, and eval infrastructure is undersupplied.** The 2026 International AI Safety Report documented frontier models detecting evaluation contexts and behaving differently under test (eval-awareness). Reward-hacking catalogs grew (models cherry-picking seeds, exfiltrating test labels). Anthropic improved Petri (automated behavioral auditing) with realism mitigations; METR ran a cross-lab pilot on internal-agent misalignment risk with Anthropic, OpenAI, Google, Meta. Enterprise agent deployments show ~37% gaps between benchmark and production performance. The need: engineers who can build statistically rigorous, contamination-aware, eval-awareness-hardened evaluation *systems*. That's systems work — bootstrap CIs, judge-bias controls, trace capture, reproducibility — not research work.

## A2. What will be true by March 2027 (opinionated forecast)

- **Commodity by mid-2027** (every bootcamp resume): RAG pipelines, vLLM/Ollama deployment, LoRA fine-tuning, LangChain/agent-framework apps, prompt engineering, MCP servers, quantizing with GPTQ/AWQ via one-line tools, "built an inference server" (AI-assisted).
- **Frameworks commodity, operation scarce:** disaggregated serving, speculative decoding, prefix caching all become engine flags. Understanding *when they help, when they hurt, and why* — with measurements — stays scarce.
- **Rising and still early:** RL environment/rollout infrastructure; agentic serving optimization; GPU fleet reliability (Xid/ECC/thermal at scale); FP4/MXFP numerics; deterministic/batch-invariant inference (needed for RL correctness and debuggability); eval infrastructure with statistical rigor; cost-aware heterogeneous routing; capacity planning with queueing models.
- **Interview reality shift:** portfolio value migrates from artifacts to *evidence of judgment* — benchmarks with error bars, failure analyses, ADRs with real tradeoffs, upstream PRs with maintainer review. Plan every flagship so it terminates in a measurement or a merged contribution, not just a repo.

## A3. Reading list that signals where things are going (queue for consolidation-week paper sprints)

- DistServe + the Hao AI Lab "Disaggregated Inference: 18 Months Later" retrospective — the arc from paper to industry default in 24 months; internalize that clock speed.
- Mooncake (Moonshot) + LMCache — KV-cache-centric serving architecture; the "KV cache is the product" worldview.
- MegaScale-Infer (SIGCOMM 2025) — attention/FFN disaggregation on heterogeneous GPUs.
- GRPO (DeepSeekMath) → DAPO → 2026 agentic-RL surveys — the post-training lineage you'll implement.
- OpenTinker / RLinf architecture docs — how the field separates environments from trainers.
- Thinking Machines' batch-invariance post + trainer-vs-engine logprob mismatch threads in verl/OpenRLHF issues — the numerics problem you'll reproduce and measure.
- K8s DRA GA docs, Kueue DRA integration, Gateway API Inference Extension spec — the 2026 K8s-AI canon.

---

# Section B: Commoditization Risk Assessment

Ratings: risk that the skill/artifact is table-stakes or free by mid-2027. "For you" flags items that are low-signal specifically because your resume already proves them.

## B1. Forge — keep the track; surgical edits

| Week | Component | Risk | Verdict |
|---|---|---|---|
| W1 | vLLM + FastAPI OpenAI-compatible server | HIGH | Done anyway (9/10 ACs) — correct as substrate. **Add:** blog post #1 + upstream issue/PR from your sm_120/Blackwell + Qwen-AWQ experience. |
| W2 | RAG pipeline (chunking, embeddings, rerank) | **VERY HIGH** | Most commoditized content in the plan. Compress W2+W3 into one week; keep the *retrieval evaluation harness* (Recall@K, MRR, NDCG — evaluation rigor transfers), cut chunking-strategy depth. |
| W3 | RAG hardening (semantic cache, hybrid search, BM25/RRF) | **VERY HIGH** | Cut as a week. Salvage: Redis semantic cache generalizes to inference caching — fold one day into W2. Freed week → **Agentic serving (new W3)**, Section E. |
| W4 | Multi-model orchestration, LoRA hot-swap, VRAM-aware routing | MEDIUM | Keep. LoRA-adapter routing is exactly what Gateway API Inference Extension standardizes — you're learning the right shape. |
| W5 | API gateway (auth, rate limiting, metering, tenancy) | HIGH *for you* | 10 years of your career already proves this. Compress to 2–3 days: only AI-specific parts — token metering over SSE, client-disconnect → GPU cancellation, per-model rate limits. |
| W6 | Docker/K3s/Helm/CI deployment | HIGH *for you* | Compress to 2–3 days. Combine W5+W6 into one week; bank the freed week as buffer/paper-sprint. |
| W7 | Consolidation, chaos, ADRs, blog | LOW | Keep; add paper sprint #1 (2–3 days). |
| W8 | Transformer internals from scratch (GQA, RoPE, RMSNorm, KV-cache) | **LOW — keep unchanged** | Foundational boundary-layer knowledge. The week that makes every later week honest. |
| W9 | Continuous batching scheduler | MEDIUM | Build stays (still strong signal when benchmarked vs vLLM). Reframe: SLO-aware scheduling and **goodput**; benchmark on agentic traces from new W3, not just Poisson chat arrivals. |
| W10 | KV-cache & memory management (paged, eviction, swap, CoW) | **LOW — extend** | The field's center of gravity. Add: hierarchical offload (GPU→CPU→NVMe), cross-request radix-tree prefix reuse (pull forward from W11), cache-hit-rate telemetry. |
| W11 | Speculative decoding + prefix caching + SLO scheduler | MEDIUM | Spec-decode is becoming an engine flag; *correct rejection sampling* (distribution-preserving, KL≈0) remains rare — keep that. Prefix caching moves to W10. **Add: 2-process prefill/decode disaggregation simulation** with KV transfer between processes (NIXL-style interface) on one GPU. |
| W12 | Quantization pipeline (GPTQ/AWQ/NF4/GGUF + TensorRT-LLM) | HIGH for tool-running | Running quantizers is one-liner work. Durable part: the **quality-gate methodology** (perplexity + task-eval gates, Pareto frontier). **Add the moat: FP4/NVFP4 on consumer Blackwell** — your 5080 supports it; nearly nobody has published those curves. Drop breadth if needed (GGUF can be a footnote). |
| W13 | Observability + load testing | MEDIUM | Keep. Reframe around goodput/SLO-attainment, per-session prefix-hit-rate, agentic load profiles. |
| W14 | Performance engineering methodology | **LOW — keep** | The durable meta-skill. Add paper sprint #2. |
| W15 | K8s operator (kopf, custom `nvidia.com/vram` resource) | **HIGH as written** | Pattern is obsolete post-DRA-GA. Rebuild: operator/controller on **DRA ResourceClaims + Kueue admission + Gateway API Inference Extension** for routing; or aim the week at an **llm-d / KubeAI contribution**. LOW risk once modernized — same learning goals, 2026 APIs. |
| W16 | Triton kernels (fused RMSNorm, flash-style attention, SwiGLU) | LOW–MEDIUM | Keep. Fix the spec's GPU peak table (RTX 5080 missing — your %-of-peak numbers would be wrong). Profile with Nsight Compute on Blackwell: novel published content. |
| W17 | Training & alignment (custom loop, LoRA, DPO, reward model) | LOW | Keep and expand into the RL bridge: **add GRPO** with verifiable rewards on GSM8K-style tasks (Section E detail). |
| W18–20 | Portfolio, benchmarks vs Ollama/vLLM, blogs, release | LOW | Keep. **New acceptance criterion: ≥1 merged upstream PR** (vLLM, SGLang, llm-d, Kueue, or Triton) and ≥2 posts containing measurements that exist nowhere else. |

## B2. Anvil — keep the track; bigger surgery (~10 of 20 weeks are generic DevOps you already have)

| Week | Component | Risk | Verdict |
|---|---|---|---|
| W1 | Raft from scratch + distributed KV | LOW commoditization, **LOW differentiation** | Classic, but every distributed-systems course does it and your resume already proves systems chops. Compress to ~3 days (election, log replication, linearizable reads — enough to whiteboard cold). Reallocate: **NCCL & collectives — ring allreduce from scratch** (pulled from Crucible W6), bandwidth math, why topology matters. AI-native distributed fundamentals > generic ones. |
| W2 | K8s deep dive: scheduler extender, webhooks, device plugin | **HIGH as written** | The APIs changed. Modernize: DRA `DeviceClass`/`ResourceClaim`, structured parameters, KAI/Kueue gang scheduling, topology-aware placement via DRA. Same learning goals; 2026 stack. |
| W3 | Training job orchestrator (gang, checkpoint/resume, DRF, preemption) | **LOW — flagship #2** | Keep whole. Modernize: benchmark against Kueue+JobSet, add elastic semantics (grow/shrink world size with checkpoint-restore). Frame every metric around researcher experience: time-to-first-batch, resume latency, queue wait. |
| W4 | Terraform/ArgoCD/Rollouts | HIGH *for you* | 2–3 days. You do this professionally. Keep spot-eviction → checkpoint drill (AI-specific), skip the rest. |
| W5 | Networking (Cilium, WireGuard, topology, NCCL docs) | MEDIUM | Keep the NCCL/topology-aware-placement half (feeds W1 rewrite and W3). Cut Cilium/WireGuard depth — narratable from experience. |
| W6 | Storage for ML (registry, CAS dedup, tiered checkpoints) | MEDIUM | Keep, narrowed to **checkpoint I/O engineering**: async checkpointing, tiering, restore-latency benchmarks. Real lab pain (fleet-scale checkpoint/restore). |
| W7 | Consolidation | LOW | Keep + paper sprint. |
| W8 | SRE (SLOs, error budgets, self-healing) | HIGH (generic) | Fold into new W11: goodput SLOs, tokens/s/$, training-job success SLOs, error budgets on the RL system you just built. ~2 days of content, not a week. |
| W9 | Cost optimization (DCGM, MPS, time-slicing, chargeback) | MEDIUM | Keep; modernize GPU sharing to DRA-native patterns; the cost-attribution + right-sizing engine feeds new W12. |
| W10 | Security (Vault HA, cosign, RBAC, CVE scanning) | HIGH (generic) | Cut to ~2 days folded into consolidation: model signing/verification (cosign) + GPU ResourceQuotas only. Vault HA proves nothing new for you. |
| W11 | Chaos engineering (Chaos Mesh, 6 scenarios) | MEDIUM | Keep, narrowed to **GPU/training-specific chaos**: Xid errors, ECC, thermal throttle, NCCL timeout, node death mid-training → checkpoint recovery. Drop generic circuit-breaker/bulkhead content. |
| W12 | Multi-cluster federation (3 clusters, Thanos, geo-routing) | MEDIUM–HIGH | Simulated multi-region on one laptop is weak signal — interviewers discount it. Replace with **cost-aware routing + capacity planning** (queueing theory pulled forward from W18): M/M/c model of your own inference engine, validated against measurement, heterogeneous cost model, what-if analysis. |
| W13 | ML lifecycle (MLflow, promotion gates, A/B, DVC, feature store) | MEDIUM | Compress to eval-gated promotion + statistically-sound comparison (paired bootstrap) — merge with Crucible's LLM-judge rigor. Drop feature store (Conduit content). |
| W14 | Consolidation | LOW | Keep + paper sprint. |
| W15 | Internal developer platform (Go CLI, plugins, docs) | HIGH (generic) | Compress to 2–3 days of CLI ergonomics on top of your real systems. Platform-engineering taste is already on your resume. |
| W16 | GPU health monitor controller (DCGM, Xid/ECC, cordon/drain, controller-runtime) | **LOW — flagship #3** | Keep and expand (absorb time freed from W15): failure taxonomy, health-score model, integration with the W3 orchestrator (auto-checkpoint-and-migrate on degradation). Maps 1:1 to lab fleet-reliability teams. |
| W17 | Observability at scale (Loki, Thanos, cardinality, drift) | LOW–MEDIUM | Keep AI-specific parts: correlation IDs across train→serve, drift detection (salvaged from Conduit), log-based Xid/OOM alerting. Trim Thanos/90-day-retention depth. |
| W18 | Infra performance (benchmarks, queueing, CI perf gates) | LOW | Half pulled into new W12; remainder: perf-regression CI gates + hardening of the RL block. |
| W19–20 | Portfolio integration | LOW | Keep. Combined Forge+Anvil+RL demo: **code → RL training with rollouts on your own engine → checkpoint → deploy → serve**, one narrated video. |

## B3. Crucible — stays a weave, reweighted

| Component | Verdict |
|---|---|
| W1–2 (training loop, mixed precision, memory math) | Absorbed by Forge W17 prerequisites — keep as reading + the memory-breakdown exercise (the params×16-bytes AdamW equation is interview gold). |
| W3 (data pipelines, BPE, dedup) | Skip build; read. Data engineering for pre-training isn't your lane. |
| W4–5 (LoRA/QLoRA, SFT with loss masking) | Keep the essentials inside Forge W17 (LoRA from scratch already there; add loss-masking correctness test — zero grad on prompt tokens). |
| W6 (distributed training concepts, allreduce) | **Promoted** to Anvil W1 (build) — better home. |
| W8 (Bradley-Terry reward model) | **Keep** — lands in RL block (Anvil W8–10) as the learned-reward arm. |
| W9 (PPO, 4 models in 16GB) | **Demote to reading + derivation.** GRPO replaced PPO in practice; GRPO drops the value model and fits 16GB more comfortably. Implement GRPO; whiteboard PPO. |
| W10–11 (DPO; KTO/ORPO/SimPO) | DPO: keep (Forge W17). The KTO/ORPO/SimPO zoo: skip builds — one method-comparison table as reading. Marginal returns fell as the field consolidated on GRPO-for-capability + DPO-for-preference. |
| W12 (Constitutional AI, RLAIF) | Keep a compressed version *if time allows* in the RL block's reward pipeline (AI-feedback preference generation). Anthropic-relevant, but optional. |
| W13 (multi-turn alignment) | Fold into agentic RL framing (rollouts *are* multi-turn) — no separate week. |
| W14–17 (comparison report, eval harness, safety eval, LLM-as-judge) | **Keep the rigor, compress the surface:** paired bootstrap, judge position-bias controls, ELO, refusal/eval-awareness probes → become the eval slice of the RL block + Anvil W13. Add 2026 content: reward-hacking detection, eval-awareness probes. |
| W18 (FSDP simulation) | Skip as a week; FSDP concepts land in Anvil W1/W3 reading. |

## B4. Conduit — drop entirely (confirmed)

The most commoditized track: managed equivalents of nearly every week ship free in Vertex/SageMaker/Databricks, and it's the closest to what every MLOps bootcamp produces. Your Forge+Anvil choice was correct. Salvage into Anvil: drift detection (→ W17 observability), eval-gated promotion + canary/rollback discipline (→ W13). Do not spend weeks here; the systems thinking it teaches, you already have from your career.

---

# Section C: The Boundary Layer Blueprint

## C1. What defines the research–infrastructure boundary

**What boundary people know that pure infra engineers don't:**
- Model internals at code level — they can open `modeling_llama.py` or a vLLM attention backend and *edit* it (you'll have this from Forge W8/W16).
- **Training dynamics and numerics** — why loss spikes; bf16 vs fp16 failure modes; why RL diverges when trainer logprobs and inference-engine logprobs disagree by 1e-3; why batch-invariance matters for debugging; what gradient noise does to small-batch runs.
- The research workflow itself — what an experiment sweep looks like, why researcher iteration speed is the product, what "the run died at 3am" costs.
- Paper-to-code fluency — read an arXiv preprint Monday, have a working implementation Thursday, know which corners were safe to cut.

**What they know that pure researchers don't:** reliability engineering, distributed-systems failure modes, cost structure, API/abstraction design, observability, capacity planning. **You already own this entire half.** Your gap is precisely the first half, and it's closable in 10 months because you're not trying to *invent* techniques — only to implement, measure, and operationalize them.

## C2. Five boundary skills to build deliberately (and where they land)

1. **Paper-to-code muscle** — recurring **paper sprint** (2–3 days) in every consolidation week (Forge W7/W14, Anvil W7/W14): implement one recent technique before it's a library feature. Queue: a scheduling/decoding paper (e.g., DuetServe-style prefill/decode harmonization), a GRPO variant, a KV-compression method. Ship each as a short repo + post: "paper → working code → measurement."
2. **Numerics & determinism debugging** — reproduce the **trainer-vs-engine logprob mismatch**: same model, same prompt, HF-forward logprobs vs vLLM logprobs; quantify divergence across dtypes/batch sizes; show its effect on GRPO importance ratios; implement the importance-sampling correction. Lands in the RL block. Almost no candidate on earth has this artifact.
3. **Experiment-velocity engineering** — reframe Anvil W3: the orchestrator's customer is a researcher. SLIs: time-to-first-batch, queue wait, resume latency after failure, sweep throughput. Write the ADRs in those terms.
4. **Eval rigor** — bootstrap CIs on every benchmark you publish; paired comparisons; judge position-bias controls; contamination checks; eval-awareness probes. Lands in RL block + Anvil W13. Cheap to add, large interview surface.
5. **GPU performance literacy** — roofline analysis, memory-vs-compute-bound diagnosis, Nsight kernel profiling. Already in Forge W14/W16; keep, and do it on Blackwell where your numbers are novel.

## C3. Portfolio contrast: boundary-layer vs generic ML Platform Engineer

| Generic ML Platform Engineer resume | Boundary-layer resume (yours, March 2027) |
|---|---|
| "Deployed LLMs on K8s with vLLM and monitoring" | "Built an inference engine from scratch; benchmarked within 2–3x of vLLM; published agentic-trace serving benchmarks with goodput/SLO analysis on consumer Blackwell" |
| "Fine-tuned Llama with LoRA" | "Implemented GRPO with verifiable rewards; found and quantified a trainer/engine logprob mismatch and its effect on training stability; wrote the importance-sampling fix" |
| "Built CI/CD and IaC for ML workloads" | "Built an async RL post-training system: environment API, rollout workers on my own engine, mid-training weight sync <Xs for a 1B model, reward-hacking telemetry" |
| "Set up GPU autoscaling" | "Built a DRA-native GPU health controller (Xid/ECC/thermal → cordon, checkpoint, migrate); gang scheduler benchmarked against Kueue" |
| "Used MLflow for experiment tracking" | "3 merged PRs in vLLM/Kueue/llm-d; 8 posts, 3 with measurements that exist nowhere else" |

The left column gets a polite pass. The right column gets a debrief fight. Every artifact on the right is achievable on one RTX 5080.

---

# Section D: The Differentiation Playbook — five niches, ranked

### D1. Agent-native inference serving (Forge, months 1–4)
**Build:** new Forge W3 + reframed W9–W11: session-aware serving (multi-turn tool loops), agentic trace generator + benchmark harness (prefix-reuse distributions, tool-call dead time, branching), radix prefix cache with hierarchical offload, SLO-aware scheduler measured in goodput; capstone: your engine vs vLLM vs SGLang **on agentic traces** — a benchmark that mostly doesn't exist publicly.
**Proves:** you understand where inference traffic is going, not where it was. **Rare because:** everyone benchmarks chat/Poisson workloads; agent-trace serving analysis requires understanding both agents and engines.

### D2. RL post-training infrastructure ⭐ the flagship (Anvil W8–10, months 7–8)
**Build (3 weeks, detailed in Section E):** mini-verl on your own stack — environment API (Gymnasium-flavored, tool-use tasks with verifiable rewards), rollout workers calling the Forge engine, GRPO trainer (1–3B, LoRA, bf16), **async weight sync** engine↔trainer with measured staleness/latency, replay buffer + backpressure, reward pipeline (verifiable + Bradley-Terry learned arm), reward-hacking telemetry, eval gates with bootstrap CIs.
**Proves:** you can build the systems research runs on — the literal job description of research-infra teams in 2026–27. **Rare because:** it requires inference + training + distributed systems simultaneously; each community lacks the other two-thirds. It also stitches Forge, Anvil, and Crucible into **one narrative artifact** for the demo video.

### D3. GPU fleet reliability (Anvil W3 + W16, modernized)
**Build:** gang-scheduled training orchestrator with checkpoint/resume + preemption (benchmarked vs Kueue/JobSet), DRA-native GPU health controller (DCGM Xid/ECC/thermal → cordon/drain/checkpoint/migrate), GPU-failure chaos suite, researcher-experience SLIs.
**Proves:** you can keep ten thousand GPUs honest. **Rare because:** the people who understand Xid codes rarely write controllers, and controller authors have rarely operated GPUs.

### D4. Inference numerics on consumer Blackwell (Forge W12 + W16 + continuous)
**Build:** FP4/NVFP4 + INT4 quality-vs-throughput Pareto curves on RTX 5080 with task-level (not just perplexity) quality gates; batch-invariance/determinism experiments on your own engine; Nsight kernel profiles on sm_120; upstream issues/PRs from every rough edge you hit.
**Proves:** rigor plus novelty. **Rare because:** your hardware is your moat — consumer Blackwell is scarce in OSS test matrices and cloud-Hopper users cannot produce these numbers. You already hit sm_120 pain in Week 1; you're living the content.

### D5. Evaluation infrastructure with statistical rigor (woven, ~1.5 weeks total)
**Build:** eval harness with paired bootstrap CIs, LLM-judge with position-bias/verbosity controls + agreement stats, contamination checks, eval-awareness probes, reward-hacking detectors wired into the RL block's telemetry.
**Proves:** you can be trusted with the numbers that decide whether a model ships. **Rare because:** most engineers treat evals as scripts; labs treat them as load-bearing infrastructure — and 2026's eval-awareness crisis made this acute.

### The intersection that makes you un-passable
Any one niche is a strong candidate. The combination — **"built the inference engine, built the RL system that trains against it, built the fleet infrastructure that keeps both alive, and published novel measurements on hardware nobody else tests"** — is a profile hiring committees see perhaps a few times a year, and it's *reachable from your starting point precisely because* three-fifths of it is systems engineering you already do at a senior level.

### Anti-patterns — looks impressive today, worthless by mid-2027
- RAG chatbot variants and vector-DB pipeline depth
- LangChain/agent-framework demo apps; prompt-engineering portfolios
- Standalone "fine-tuned Llama with LoRA" projects
- Another thin vLLM wrapper presented as "an inference engine"
- Simulated laptop "multi-region federation" sold as scale experience (interviewers discount simulations that don't isolate a real phenomenon; your continuous-batching and RL sims *do* — they measure real algorithms on real hardware)
- kopf operators on device-plugin patterns (dated on arrival post-DRA)
- Raft-from-scratch as a headline (fine as a footnote)
- Conduit-style MLOps dashboards; feature stores
- A 20-repo GitHub with no benchmarks, no error bars, no upstream PRs — breadth reads as AI-generated in 2027; depth with measurement reads as real

---

# Section E: Revised Plan Priorities — the 10-month map

## E1. Forge, months 1–5 (surgical; you're mid-W1, momentum preserved)

| Week | Was | Now | Key acceptance-criteria deltas |
|---|---|---|---|
| W1 | Inference server | **Unchanged** — finish AC #10 (pytest integration tests) | **Add AC:** blog post #1 published ("Serving LLMs on consumer Blackwell: vLLM + sm_120 + a 16GB budget"); ≥1 upstream issue filed from real W1 friction. |
| W2 | RAG pipeline | **RAG compressed** (W2+W3 → one week) | Keep: ingestion, embeddings, retrieval + generation endpoint, **retrieval eval harness** (Recall@K/MRR/NDCG on a labeled set), 1 day on semantic cache. Cut: chunking-strategy comparison, hybrid/BM25/RRF, multi-filetype parsers. |
| W3 | RAG hardening | **NEW: Agent-native serving workloads** | ACs: session abstraction (multi-turn, tool-call boundaries) on the W1 server; agentic trace generator (configurable turns, prefix-share ratio, tool latency distribution, branching); measure prefix-cache hit rate and TTFT-per-step with/without vLLM prefix caching; per-session vs per-request metrics in Postgres; published trace-suite spec (becomes W9/W13 benchmark input); write-up: "chat vs agent traffic — measured." |
| W4 | Multi-model orch | **Keep** (as specced) | Add one AC: route by adapter name à la Gateway API Inference Extension `InferencePool` semantics (concept parity, single node). |
| W5+W6 | Gateway; Deployment | **Merged into one week** | Keep only: token metering over SSE, disconnect→GPU-cancellation, per-model rate limits (2–3 days); Docker/K3s/Helm/CI essentials (2–3 days — you're senior at this). Cut: multi-tenancy depth, argon2 key lifecycle, Makefile/secrets ceremony. |
| W6' | — | **Banked buffer / paper sprint #0** | Reality check: specs are dense and you have a full-time job. A banked week in phase 1 protects the phase-2 deep weeks. |
| W7 | Consolidation | Keep + **paper sprint #1** | Chaos scenarios, ADRs, blog #2 as specced. |
| W8 | Transformer internals | **Keep unchanged** | The foundational week. |
| W9 | Continuous batching | Keep, **reframed** | ACs shift: benchmark on W3 agentic traces *and* Poisson chat; report **goodput** (SLO-attainment throughput) not just tokens/s; SLO-aware admission. Still: 2–4x over static, within ~2–3x of vLLM. |
| W10 | KV-cache memory | Keep + **extend** | Add ACs: radix-tree cross-request prefix reuse (moved from W11); hierarchical eviction GPU→CPU(pinned)→NVMe with restore-latency measurements; hit-rate telemetry per session. |
| W11 | Spec decoding + prefix cache | Keep spec-decode correctness; **add disaggregation sim** | ACs: rejection sampling distribution-preserving (KL≈0 over 1k samples); adaptive speculation length; **2-process prefill/decode split on one GPU with KV transfer over a NIXL-style interface, measured transfer overhead vs recompute**. |
| W12 | Quantization | Keep methodology; **add FP4-on-Blackwell** | ACs: quality gates use task evals + perplexity; **NVFP4/FP4 path benchmarked on the 5080** (TensorRT-LLM or native vLLM path) with Pareto curves; blog #3 = "FP4 on consumer Blackwell: the numbers." GGUF demoted to footnote. |
| W13 | Observability | Keep, reframed | Goodput/SLO dashboards; prefix-hit-rate and per-session panels; agentic load profiles in the load-test suite. |
| W14 | Perf methodology | Keep + **paper sprint #2** | As specced (3 documented fixes, ≥20% cumulative win). |
| W15 | K8s operator | **Modernized** | Rebuild on **DRA ResourceClaims/DeviceClass + Kueue admission + Gateway API Inference Extension** routing; or substitute an **llm-d/KubeAI contribution week**. ACs: InferenceService CRD reconciles to DRA claims; queue-depth autoscaling; zero-downtime rollout — same behaviors, 2026 APIs. |
| W16 | Triton kernels | Keep | Fix peak-spec table for RTX 5080; Nsight Compute profiles on sm_120 in the write-up. |
| W17 | Training & alignment | Keep + **GRPO** | ACs add: GRPO on 1–3B (LoRA, bf16) with **verifiable rewards** (GSM8K-style exact-match) — group sampling via *your own engine or vLLM*, advantage = group-normalized reward, no value model; reward curve up over ≥300 steps; DPO retained; PPO as derivation write-up. This week is the bridge to Anvil's RL block. |
| W18–20 | Portfolio | Keep + **OSS AC** | New ACs: ≥1 **merged** upstream PR; benchmark report includes agentic-trace suite; ≥2 posts with data that exists nowhere else. |

## E2. Anvil, months 6–10 (restructured)

| Week | Was | Now | Key content |
|---|---|---|---|
| W1 | Raft week | **Raft (3 days) + NCCL/collectives (rest)** | Raft: election + replication + linearizable reads, enough to whiteboard. Then: **ring allreduce from scratch** (numerically exact vs `torch.distributed`), bandwidth-optimality math, NCCL topology concepts, DDP/FSDP/ZeRO as reading (absorbs Crucible W6/W18). |
| W2 | K8s deep dive | **Modernized: DRA + KAI/Kueue** | DRA driver concepts, ResourceClaims with topology constraints, Kueue quotas/preemption, gang admission; keep the webhook exercise (still-current API). |
| W3 | Training orchestrator | **Keep — flagship #2** | As specced + elastic semantics (grow/shrink with checkpoint-restore) + benchmarked against Kueue+JobSet + researcher-experience SLIs (time-to-first-batch, resume latency). |
| W4 | IaC | **Compressed to 2–3 days** | Spot-eviction→checkpoint drill; ArgoCD sync for the platform. Freed days → W3 spillover. |
| W5 | Networking | **Narrowed** | Topology-aware gang placement (feeds W3); NCCL performance model; cut Cilium/WireGuard builds. |
| W6 | Storage | **Narrowed: checkpoint I/O engineering** | Async checkpointing (<100ms training-loop stall), tiered NVMe→object storage, restore-latency benchmarks, CAS dedup. |
| W7 | Consolidation | Keep + paper sprint #3 | — |
| **W8–10** | SRE; Security; Cost (old) | **NEW FLAGSHIP: RL post-training infrastructure (3 weeks)** | **W8 — rollout plane:** environment API (Gymnasium-flavored tool-use tasks with verifiable rewards); rollout workers calling the Forge engine (or vLLM) async; trajectory store; backpressure; throughput benchmark (trajectories/min at fixed VRAM). **W9 — training plane:** GRPO trainer (1–3B, LoRA, bf16) consuming trajectories; **async weight sync** trainer→engine with measured sync latency + staleness-vs-stability experiment; the **logprob-mismatch study** (HF-forward vs engine logprobs; effect on importance ratios; correction implemented). **W10 — reward + eval plane:** verifiable-reward pipeline + Bradley-Terry learned-reward arm (Crucible W8); **reward-hacking telemetry** (length exploitation, formatting hacks, reward-vs-eval divergence alarms); eval gates with paired-bootstrap CIs; optional RLAIF/constitutional preference generation if ahead of schedule. Capstone: end-to-end run — base 1B → GRPO on tool tasks → measurable eval lift with CIs — fully orchestrated by W3's system. |
| W11 | Chaos (old W8+W11 merged) | **GPU-failure chaos + AI SLOs** | Xid/ECC/thermal/NCCL-timeout/node-death scenarios against the RL system; goodput + tokens/s/$ SLOs; error budgets; self-healing hooks into W3. |
| W12 | Multi-cluster (old) | **Cost-aware routing + capacity planning** | M/M/c queueing model of your engine validated within 15% of measurement; heterogeneous cost model (prefill vs decode economics); what-if analysis; cost-aware router prototype. (Absorbs old W18 queueing content + old W9 cost attribution.) |
| W13 | ML lifecycle | **Eval-gated promotion + judge rigor** | Promotion gates on evals with CIs; LLM-judge with position-bias controls + ELO (Crucible W17); canary/rollback (Conduit salvage); model signing (old W10 salvage, ~1 day). |
| W14 | Consolidation | Keep + paper sprint #4 | Full-lifecycle integration test now includes the RL loop. |
| W15 | IDP/CLI | **Compressed to 2–3 days** | CLI ergonomics over real systems; freed days → W16. |
| W16 | GPU health controller | **Keep — flagship #3, expanded** | DRA-native; DCGM Xid/ECC/thermal taxonomy; cordon→checkpoint→migrate integration with W3; controller-runtime (Go) as specced. |
| W17 | Observability at scale | Keep AI-specific parts | Correlation IDs train→serve; drift detection (Conduit salvage); Xid/OOM log alerting; trim Thanos depth. |
| W18 | Infra perf | **Hardening + perf-regression gates** | CI perf gates on the whole platform; remaining queueing content already spent in W12. |
| W19–20 | Portfolio | Keep + **launch + applications** | Combined demo: *code → RL training with rollouts on your own engine → checkpoint → health-event migration → deploy → serve*. **Start applications January–February 2027** — portfolio interview-ready by early Feb; W19–20 polish happens while interviewing, not before applying. |

## E3. What stays untouched (for easy diffing)

Forge W1 (bar the added blog/issue ACs), W4, W7, W8, W13, W14, W16, W18–20 core; Anvil W3, W5–7 cores, W16, W17, W19–20 core; all repo conventions (spec format, 10 ACs, progress.md discipline, immutable completed specs); the Forge→Anvil sequencing; the 6–7h/day budget; the hardware envelope.

## E4. Feasibility check against your constraints

- **16GB VRAM:** GRPO on 1–3B with LoRA needs policy(+LoRA)+reference in bf16 — comfortably lighter than Crucible's 4-model PPO week that was already designed to fit. Rollouts on a 1–3B engine while training via LoRA fits with gradient checkpointing; worst case, alternate rollout/train phases (synchronous GRPO) — that's the standard pattern anyway. Disaggregation sim = two processes, one GPU. FP4 runs *reduce* memory. Agentic benchmarks reuse the W1 7B-AWQ setup.
- **Time:** cuts (~7 weeks of generic content + Conduit + Crucible's method zoo) fund the additions (1 wk agentic serving, 3 wk RL block, GRPO inside existing W17, modernizations inside existing weeks) with one banked buffer week per project. Net load is *slightly lower* than v1, concentrated on higher-value work.
- **Full-time job:** the banked buffers and demoted-to-reading items are the pressure valves. Rule: when a week overruns, cut breadth ACs, never the measurement/write-up ACs — the write-ups are the portfolio.

## E5. Blog/output cadence (8 posts, 3 must be "nobody else has this data")

1. vLLM on consumer Blackwell/sm_120 (W1) · 2. Phase-1 platform retrospective (W7) · 3. **Chat vs agent traffic: serving measurements** (W3/W9) ⭐ · 4. Continuous batching from scratch, benchmarked (W9–10) · 5. **FP4 on consumer Blackwell: Pareto curves** (W12) ⭐ · 6. Triton kernels on sm_120 (W16) · 7. **Building RL post-training infra: weight sync, logprob mismatches, reward hacking** (Anvil W8–10) ⭐ · 8. GPU fleet reliability: Xid taxonomy + health controller (Anvil W16).

---

## Follow-ups — status (updated 2026-07-06, same day)

1. ✅ **Near-term specs regenerated** — Forge: `week02-rag-retrieval-eval.md`, `week03-agentic-serving.md` (new), `week05-gateway-deployment.md` (merged), `week06-buffer-oss.md` (new). Anvil: `week01-raft-and-collectives.md`, `week02-k8s-dra-kueue.md`. Superseded v1 specs archived in `original_artifacts/specs_v1/`.
2. ⚠️ **Later-phase specs are DELIBERATELY NOT regenerated** — they must be re-evaluated **just-in-time as near-term goals are exhausted**, because the landscape will move again before you reach them (the v1 W2/W15 specs went stale in ~3 months). The gates, triggers, and per-gate checklists live in **`forge/specs/V2_DELTAS_AND_REEVAL.md`** (Gate #1 before Forge W11; Gate #2 before Forge W15) and **`anvil/specs/V2_DELTAS_AND_REEVAL.md`** (Gate #1 before Anvil starts ~Nov 2026; Gate #2 before Phase B — where the W8–10 RL-flagship specs get written). Each gate says what to re-check and **what to capture for the next high-level plan review** (which §A forecasts held, which bets to revise). Both `progress.md` files carry the gate reminders inline so no session misses them.
3. **MANIFEST cleanup still open** — Forge MANIFEST's decisions log is a placeholder and both MANIFESTs carry stale `/Users/jmalviya/...` macOS paths; small cleanup pass recommended.

## Sources (web research, July 2026)

- [NVIDIA Dynamo 1.0 disaggregated inference guide (Spheron)](https://www.spheron.network/blog/nvidia-dynamo-disaggregated-inference-guide/) · [Dynamo repo](https://github.com/ai-dynamo/dynamo) · [Dynamo disaggregated serving docs](https://docs.dynamo.nvidia.com/dynamo/design-docs/disaggregated-serving) · [Dynamo × llm-d](https://developer.nvidia.com/blog/nvidia-dynamo-accelerates-llm-d-community-initiatives-for-advancing-large-scale-distributed-inference/)
- [Disaggregated Inference: 18 Months Later — Hao AI Lab](https://haoailab.com/blogs/distserve-retro/) · [Disaggregated prefill/decode overview](https://www.buildmvpfast.com/blog/disaggregated-llm-inference-prefill-decode-gpu-utilization-2026) · [Heterogeneous disaggregation (AM Compute)](https://www.amcompute.com/blog/disaggregated-inference) · [vLLM vs SGLang 2026 benchmarks](https://techsy.io/en/blog/vllm-vs-sglang) · [DuetServe (arXiv)](https://arxiv.org/pdf/2511.04791)
- [K8s DRA docs](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/) · [Kueue DRA](https://kueue.sigs.k8s.io/docs/concepts/dynamic_resource_allocation/) · [DRA GA in OpenShift 4.21 (Red Hat, Mar 2026)](https://developers.redhat.com/articles/2026/03/25/dynamic-resource-allocation-goes-ga-red-hat-openshift-421-smarter-gpu) · [K8s GPU scheduling 2026: DRA/KAI/MIG](https://www.techplained.com/kubernetes-gpu-scheduling) · [K8s AI infra 2026 production realities](https://cloudoptimo.com/blog/kubernetes-ai-infrastructure-in-2026-gpu-scheduling-and-production-realities/)
- [RLinf](https://github.com/RLinf/RLinf) · [OpenRLHF](https://github.com/openrlhf/openrlhf) · [OpenTinker (arXiv)](https://arxiv.org/pdf/2601.07376) · [Agentic RL survey](https://arxiv.org/pdf/2509.02547) · [Top RL environments 2026 (HUD)](https://www.hud.ai/resources/top-5-reinforcement-learning-environments)
- [Anthropic Alignment Science blog](https://alignment.anthropic.com/) · [METR](https://metr.org/) · [AI benchmarks 2026 and their limits (Kili)](https://kili-technology.com/blog/ai-benchmarks-guide-the-top-evaluations-in-2026-and-why-theyre-not-enough) · [AI Safety at the Frontier — April 2026 highlights](https://aisafetyfrontier.substack.com/p/paper-highlights-of-april-2026)
