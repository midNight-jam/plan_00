# Strategic Plan Evolution Prompt
# Use this in a fresh session with the most capable model available

---

## THE PROMPT (copy everything below this line)

---

Role: You are a Senior Technical Fellow at Anthropic — someone who has spent 15+ years at the intersection of ML research and systems engineering, who oversees technology evolution, understands where the field is heading in the next 12-24 months, and has direct visibility into what makes someone invaluable at the research-infrastructure boundary. You think in decades, not quarters.

Context: I am a senior backend/platform engineer (Java, Kubernetes, Helm, GCP, BigQuery) building a 10-month intensive portfolio to transition into top AI firms (Anthropic, OpenAI, xAI). I have an ASUS ROG Strix with RTX 5080 (16GB VRAM) and Ubuntu.

I have already created 4 detailed career track plans they are present in the current dir.. (20 weeks each):
**Forge** (AI Platform + Inference): Custom inference engine, continuous batching, KV-cache management, quantization, K8s operator, Triton kernels
**Anvil** (AI Infrastructure): Raft consensus, K8s schedulers, training job orchestration, SRE, chaos engineering, multi-cluster
**Crucible** (Training + Alignment): RLHF from scratch, DPO, Constitutional AI, reward modeling, evaluation frameworks, safety
**Conduit** (ML Systems): Feature stores, pipeline orchestration, A/B testing, drift detection, auto-retraining, governance

My chosen primary path: Forge (months 1-5) + Anvil (months 6-10), with elements of Crucible woven in.

---

## MY THREE CRITICAL ASKS:

### Ask 1: Future-Proof the Plan (12-Month Technology Horizon)

Technology is moving at breakneck speed. By the time I finish in 10 months (March 2027), the landscape will have shifted. I need you to:

1. **Identify what in my current plan risks becoming commoditized** — what will vLLM/Ollama/cloud providers simply give away for free by then? What will every bootcamp graduate have on their resume by mid-2027?

2. **Identify what will become MORE valuable** — what emerging areas are still early enough that deep expertise by March 2027 would be rare and highly sought after?

3. **Recommend specific additions or pivots to my plan** — concrete topics, tools, or projects I should add/replace to stay ahead of the commoditization curve. Be specific about WHICH weeks to modify and WHAT to replace them with.

Think about:
Where inference optimization is headed (compilation, hardware-aware scheduling, heterogeneous compute)
Where training/alignment is headed (synthetic data, process supervision, scalable oversight)
Where infrastructure is headed (AI-native orchestration, inference routers, cost-aware systems)
What new capabilities (reasoning, agents, tool use, long context, multimodal) will reshape infrastructure needs
What papers from the last 6 months signal future directions

### Ask 2: Position at the Research-Infrastructure Boundary

At Anthropic, there's a role called "Applied AI Product Engineer" — they sit at the boundary between researchers and production systems. They don't do pure research, but they deeply understand it. They don't do pure infrastructure, but they build the systems that make research possible.

I want to position myself at a similar critical  layer — not the surface layer (just serving models) and not pure theory (just writing papers). I want to be at the critical boundary where:
Research ideas get translated into production systems
Infrastructure decisions are informed by deep ML understanding  
Novel techniques (new attention patterns, new training methods, new eval approaches) get their FIRST implementation before they become library features

**Help me understand:**
1. What specific skills define this boundary layer? What do these people know that pure infra engineers don't, and what do they know that pure researchers don't?
2. What should I ADD to my plan to develop these boundary skills?
3. What projects would demonstrate I can operate at this layer?
4. What would the resume/portfolio look like for someone who lands this role vs someone who lands a generic "ML Platform Engineer" role?

### Ask 3: Differentiation Strategy (The Anti-Commoditization Moat)

Here's my fear: In 10 months, thousands of engineers will have built inference servers with vLLM, deployed models with K8s, and fine-tuned with LoRA. The basic playbook is already becoming common knowledge.

**I need the niche that can't be mass-produced.** Help me identify:

1. **What are the 3-5 technical differentiators** that will still be rare in March 2027? Things that require genuine deep understanding, not just following tutorials. Things that make interviewers say "we've never seen a candidate who built THIS."

2. **What combination of skills creates a unique profile?** Not just depth in one area, but a specific INTERSECTION that's extremely rare and extremely valuable.

3. **What should I REMOVE from my plan** because it will be table-stakes (everyone will have it) and replace with what?

4. **What are the "anti-patterns"** — things that LOOK impressive on a resume today but will be worthless by mid-2027?

---

## FORMAT YOUR RESPONSE AS:

### Section A: The 12-Month Technology Forecast
What's coming, what's dying, what's emerging. Be specific and opinionated.

### Section B: Commoditization Risk Assessment
For each major component of my plan, rate the commoditization risk (HIGH/MEDIUM/LOW) and explain. Identify what to cut.

### Section C: The Boundary Layer Blueprint
Exact skills, projects, and portfolio artifacts that demonstrate research-adjacent engineering capability.

### Section D: The Differentiation Playbook
The 3-5 specific niches I should target, with concrete additions to my plan (which weeks, what to build, what it proves).

### Section E: Revised Plan Priorities
Given all the above, what's the OPTIMAL ordering and focus for my 12 months? What do I add, what do I remove, what do I keep but reframe?

---

## CONSTRAINTS ON YOUR RESPONSE:

Be brutally honest. If parts of my plan are going to become worthless, tell me now.
Be specific. Don't say "stay current with research" — say "implement process reward models because X."
Think like someone who HIRES at Anthropic. What would make you unable to pass on a candidate?
Consider that I'm starting from backend/platform engineering, not ML research. My advantage is SYSTEMS THINKING applied to ML problems.
The plan must remain executable on a single RTX 5080 (16GB VRAM). No "just train a 70B model" suggestions.
Assume I'm working 6-7 hours/day on this alongside a full-time job.



Take your time, think step by step, thoroughly and deeply, cover every aspect that i have caputured in my 4 projs, [forge, anvil, conduit, crucible], I would rather wait for a while, even hour, for a deeply considered response rather than a quick surface level reply. This decision affects a key turning point in ones career , 

nit - if you are going to generate a new plan, put it under "original_artifacts" so that I can compare prev to now
