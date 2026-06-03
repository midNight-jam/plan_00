# Week 5: Instruction Tuning (SFT)

## Context

**Where it fits:** Phase 1 (Foundations), Week 5 of 7
**Prerequisites:** Week 3 (data pipelines), Week 4 (QLoRA fine-tuning), understanding of chat/conversation structure
**What it builds on:** You can now fine-tune efficiently with QLoRA and build data pipelines — this week applies both to the specific problem of making a model follow instructions
**What comes next:** Week 6 (distributed training) and Week 7 (consolidation). Phase 2 builds RLHF/DPO on top of your SFT model

A base language model predicts the next token. An instruction-tuned model follows commands. The gap between these is SFT — Supervised Fine-Tuning on instruction/response pairs. This is the first step in alignment: teaching the model what a "good response" looks like.

---

## Learning Goals

- [ ] Explain why a pre-trained model doesn't follow instructions (it continues text, doesn't answer questions)
- [ ] Articulate what SFT does differently: train on (instruction, response) pairs with loss only on response tokens
- [ ] Understand the three common dataset formats: Alpaca, ShareGPT, OpenAI Chat
- [ ] Explain loss masking: why we don't compute loss on system/user messages (only assistant responses)
- [ ] Understand chat templates: how special tokens delimit turns in a conversation
- [ ] Articulate the quality-over-quantity principle: 1K high-quality examples > 100K low-quality
- [ ] Explain the difference between single-turn and multi-turn instruction tuning
- [ ] Understand catastrophic forgetting: how aggressive SFT can degrade base model capabilities

---

## Implementation Goals

- [ ] Build dataset parser for Alpaca format, ShareGPT format, and OpenAI chat format
- [ ] Implement proper chat template application using tokenizer's `apply_chat_template`
- [ ] Implement loss masking: compute cross-entropy only on assistant tokens
- [ ] Build SFT training pipeline using QLoRA on a 3-7B model
- [ ] Implement multi-turn conversation handling (maintain context across turns)
- [ ] Create data quality filter: length, formatting, deduplication
- [ ] Compare base model vs SFT model on instruction-following benchmarks
- [ ] Implement proper evaluation: perplexity on held-out instructions + generation quality (manual and automated)

---

## Acceptance Criteria

1. Dataset parser correctly handles all three formats (Alpaca, ShareGPT, OpenAI) and produces uniform internal representation
2. Loss masking is verified: gradient is zero for prompt/system token positions (checked via hook on embedding layer)
3. Chat template produces correct special token sequences (verified against reference implementation for target model)
4. SFT-trained model responds to instructions coherently (manual evaluation: 8/10 responses are relevant and complete)
5. Base model vs SFT model comparison shows clear qualitative difference on 20 held-out instructions
6. Training loss converges below 1.5 within 1 epoch on a 10K instruction dataset
7. No catastrophic forgetting: SFT model retains >80% accuracy on a general knowledge benchmark (e.g., HellaSwag subset)
8. Multi-turn conversations maintain coherent context (model correctly references earlier turns)
9. Data quality filter removes at least 15% of a raw instruction dataset (demonstrating it catches real issues)
10. Full SFT pipeline (data prep → train → evaluate) completes in under 4 hours on RTX 5080

---

## Validation Commands

```bash
# Test dataset parsers
python -m pytest tests/test_dataset_formats.py -v

# Verify loss masking
python scripts/verify_loss_mask.py --model mistralai/Mistral-7B-v0.1 --sample 10

# Apply chat template and inspect tokens
python scripts/inspect_chat_template.py --model mistralai/Mistral-7B-v0.1 --format sharegpt

# Train SFT model
python scripts/train_sft.py \
  --model mistralai/Mistral-7B-v0.1 \
  --dataset data/instructions_10k.jsonl \
  --format alpaca \
  --rank 16 --alpha 32 \
  --epochs 1 \
  --output models/mistral-sft

# Compare base vs SFT
python scripts/compare_models.py \
  --base mistralai/Mistral-7B-v0.1 \
  --sft models/mistral-sft \
  --prompts data/eval_prompts.jsonl \
  --output results/comparison.md

# Evaluate forgetting
python scripts/eval_forgetting.py --model models/mistral-sft --benchmark hellaswag --subset 500

# Run data quality filter
python scripts/filter_dataset.py --input data/raw_instructions.jsonl --output data/filtered.jsonl --report

# Generate sample outputs for manual review
python scripts/generate_samples.py --model models/mistral-sft --num 20 --output results/samples.md

# Multi-turn test
python scripts/test_multiturn.py --model models/mistral-sft --conversations data/multiturn_test.jsonl
```

---

## Technical Implementation Details

### Project Structure

```
week05-instruction-tuning/
├── src/
│   ├── __init__.py
│   ├── formats/
│   │   ├── __init__.py
│   │   ├── alpaca.py          # Alpaca format parser
│   │   ├── sharegpt.py        # ShareGPT format parser
│   │   └── openai_chat.py     # OpenAI chat format parser
│   ├── chat_template.py       # Chat template application
│   ├── loss_masking.py        # Masked cross-entropy loss
│   ├── data_quality.py        # Filtering and cleaning
│   ├── sft_dataset.py         # SFT-specific dataset class
│   ├── train_sft.py           # SFT training loop
│   └── evaluation.py          # Automated evaluation
├── scripts/
│   ├── train_sft.py
│   ├── compare_models.py
│   ├── verify_loss_mask.py
│   ├── filter_dataset.py
│   └── generate_samples.py
├── tests/
│   ├── test_dataset_formats.py
│   ├── test_loss_masking.py
│   ├── test_chat_template.py
│   └── test_quality_filter.py
└── data/
    ├── raw_instructions.jsonl
    └── eval_prompts.jsonl
```

### Dataset Format Parsers

```python
# src/formats/alpaca.py
from dataclasses import dataclass

@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str

@dataclass
class Conversation:
    messages: list[Message]

def parse_alpaca(example: dict) -> Conversation:
    """
    Alpaca format:
    {"instruction": "...", "input": "...", "output": "..."}
    """
    messages = []
    
    # Build user message (instruction + optional input)
    user_content = example['instruction']
    if example.get('input', '').strip():
        user_content += f"\n\nInput: {example['input']}"
    
    messages.append(Message(role="user", content=user_content))
    messages.append(Message(role="assistant", content=example['output']))
    
    return Conversation(messages=messages)

# src/formats/sharegpt.py
def parse_sharegpt(example: dict) -> Conversation:
    """
    ShareGPT format:
    {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
    """
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    messages = []
    
    for turn in example['conversations']:
        role = role_map.get(turn['from'], turn['from'])
        messages.append(Message(role=role, content=turn['value']))
    
    return Conversation(messages=messages)

# src/formats/openai_chat.py
def parse_openai_chat(example: dict) -> Conversation:
    """
    OpenAI format:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]}
    """
    messages = [Message(role=m['role'], content=m['content']) for m in example['messages']]
    return Conversation(messages=messages)
```

### Loss Masking Implementation

```python
# src/loss_masking.py
import torch
import torch.nn.functional as F

def compute_sft_loss(logits: torch.Tensor, labels: torch.Tensor, 
                     loss_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss only on assistant tokens.
    
    Args:
        logits: (batch, seq_len, vocab_size) - model predictions
        labels: (batch, seq_len) - token IDs (shifted by 1 externally or here)
        loss_mask: (batch, seq_len) - 1.0 for assistant tokens, 0.0 for prompt tokens
    """
    # Shift: predict next token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous()
    
    # Flatten
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    flat_mask = shift_mask.view(-1)
    
    # Compute per-token loss
    per_token_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
    
    # Apply mask: zero out loss on non-assistant tokens
    masked_loss = per_token_loss * flat_mask
    
    # Average only over assistant tokens
    num_assistant_tokens = flat_mask.sum()
    if num_assistant_tokens == 0:
        return torch.tensor(0.0, device=logits.device)
    
    return masked_loss.sum() / num_assistant_tokens


def create_loss_mask(input_ids: torch.Tensor, tokenizer, conversations: list) -> torch.Tensor:
    """
    Create mask where 1 = assistant token (compute loss), 0 = prompt/system token (ignore).
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros(batch_size, seq_len, dtype=torch.float32)
    
    for i, conv in enumerate(conversations):
        # Find where assistant responses start/end in token sequence
        # This depends on the chat template's special tokens
        current_pos = 0
        full_text = ""
        
        for msg in conv.messages:
            msg_text = tokenizer.apply_chat_template([{'role': msg.role, 'content': msg.content}],
                                                      tokenize=False, add_generation_prompt=False)
            msg_tokens = tokenizer.encode(msg_text, add_special_tokens=False)
            
            if msg.role == "assistant":
                # Mark these positions in the mask
                end_pos = current_pos + len(msg_tokens)
                mask[i, current_pos:min(end_pos, seq_len)] = 1.0
            
            current_pos += len(msg_tokens)
    
    return mask
```

### Chat Template Application

```python
# src/chat_template.py
from transformers import AutoTokenizer

class ChatTemplateProcessor:
    """Apply model-specific chat templates to conversations."""
    
    def __init__(self, model_name: str, max_length: int = 2048):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def process(self, conversation: 'Conversation') -> dict:
        """Convert conversation to tokenized inputs with loss mask."""
        messages = [{'role': m.role, 'content': m.content} for m in conversation.messages]
        
        # Full conversation tokenized
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        full_tokens = self.tokenizer(full_text, max_length=self.max_length,
                                      truncation=True, return_tensors='pt')
        
        # Build loss mask by tokenizing incrementally
        loss_mask = self._build_mask(messages, full_tokens['input_ids'].shape[1])
        
        return {
            'input_ids': full_tokens['input_ids'].squeeze(0),
            'attention_mask': full_tokens['attention_mask'].squeeze(0),
            'loss_mask': loss_mask,
        }
    
    def _build_mask(self, messages: list[dict], total_len: int) -> torch.Tensor:
        """Build per-token loss mask by finding assistant response boundaries."""
        import torch
        mask = torch.zeros(total_len)
        
        # Tokenize messages incrementally to find boundaries
        cumulative = ""
        for msg in messages:
            prefix_len = len(self.tokenizer.encode(cumulative, add_special_tokens=False)) if cumulative else 0
            cumulative = self.tokenizer.apply_chat_template(
                messages[:messages.index(msg) + 1], tokenize=False
            )
            current_len = len(self.tokenizer.encode(cumulative, add_special_tokens=False))
            
            if msg['role'] == 'assistant':
                mask[prefix_len:min(current_len, total_len)] = 1.0
        
        return mask
```

### Data Quality Filter

```python
# src/data_quality.py
from dataclasses import dataclass

@dataclass
class FilterStats:
    total: int = 0
    passed: int = 0
    too_short: int = 0
    too_long: int = 0
    empty_response: int = 0
    low_quality: int = 0
    duplicate: int = 0

class InstructionFilter:
    def __init__(self, min_instruction_len=10, min_response_len=20,
                 max_response_len=4096, max_repetition_ratio=0.3):
        self.min_instruction_len = min_instruction_len
        self.min_response_len = min_response_len
        self.max_response_len = max_response_len
        self.max_repetition_ratio = max_repetition_ratio
        self.seen_hashes = set()
        self.stats = FilterStats()
    
    def filter(self, conversation: 'Conversation') -> bool:
        self.stats.total += 1
        
        user_msgs = [m for m in conversation.messages if m.role == 'user']
        asst_msgs = [m for m in conversation.messages if m.role == 'assistant']
        
        if not user_msgs or not asst_msgs:
            self.stats.empty_response += 1
            return False
        
        # Length checks
        if any(len(m.content) < self.min_instruction_len for m in user_msgs):
            self.stats.too_short += 1
            return False
        
        if any(len(m.content) < self.min_response_len for m in asst_msgs):
            self.stats.empty_response += 1
            return False
        
        if any(len(m.content) > self.max_response_len for m in asst_msgs):
            self.stats.too_long += 1
            return False
        
        # Repetition check (detect "I'm an AI" repeated patterns)
        for m in asst_msgs:
            words = m.content.split()
            if len(words) > 10:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < (1 - self.max_repetition_ratio):
                    self.stats.low_quality += 1
                    return False
        
        # Deduplication (hash-based exact match)
        content_hash = hash(tuple(m.content for m in conversation.messages))
        if content_hash in self.seen_hashes:
            self.stats.duplicate += 1
            return False
        self.seen_hashes.add(content_hash)
        
        self.stats.passed += 1
        return True
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| Loss mask is all zeros | Chat template may add tokens you're not accounting for. Print tokenized output with `tokenizer.decode` per-token to debug |
| Model outputs gibberish after SFT | Likely wrong chat template at inference. Use same template used during training. Check EOS token handling |
| Loss doesn't decrease | Verify data is correctly formatted. Print a few training examples decoded. Check learning rate (1e-4 to 2e-4 for SFT) |
| Catastrophic forgetting | Reduce learning rate, reduce epochs (1 epoch often sufficient), increase regularization. Try smaller rank |
| Multi-turn context lost | Ensure full conversation history is in the input, not just last turn. Check max_length truncation |
| Chat template not found for model | Some models need custom templates. Set `tokenizer.chat_template` manually. Check model card for format |
| OOM during training | Reduce max_length from 2048 to 1024. Use gradient accumulation. Ensure QLoRA is active |
| Evaluation is subjective | Use automated metrics: MMLU subset, instruction-following accuracy on templated tasks |

---

## Agent Handoff Template

```
I'm working on Week 5 of the Crucible Phase 1 project: Instruction Tuning (SFT).

Hardware: RTX 5080 16GB VRAM, 32GB RAM, Ubuntu
Project path: ~/crucible/week05-instruction-tuning/

Current status: [DESCRIBE WHERE YOU ARE]

What I've completed:
- [x/o] Dataset format parsers (Alpaca, ShareGPT, OpenAI)
- [x/o] Chat template application
- [x/o] Loss masking implementation
- [x/o] Data quality filtering
- [x/o] SFT training pipeline
- [x/o] Multi-turn handling
- [x/o] Base vs SFT comparison
- [x/o] Forgetting evaluation

Model: [WHICH MODEL]
Dataset: [WHICH DATASET, SIZE]
Training config: [RANK, ALPHA, LR, EPOCHS]

Current issue: [DESCRIBE THE PROBLEM]
Error message (if any): [PASTE ERROR]

Please help me [SPECIFIC ASK]. The model should follow instructions while retaining general capabilities.
```

---

## Out of Scope

- RLHF / DPO (Phase 2 — this is supervised only)
- Constitutional AI or self-critique methods
- Multi-modal instruction tuning (vision-language)
- Human evaluation at scale (use automated proxies)
- Prompt engineering for better base model performance (we're training, not prompting)
- Deployment / serving the SFT model
- Safety filtering or content moderation of training data
- Creating instruction datasets from scratch (use existing open datasets)
