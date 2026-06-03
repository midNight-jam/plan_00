# Week 1: PyTorch Training from Scratch

## Context

**Where it fits:** Phase 1 (Foundations), Week 1 of 7
**Prerequisites:** Python proficiency, basic PyTorch tensor operations, understanding of neural network forward/backward pass
**What it builds on:** General ML knowledge — this week strips away all high-level abstractions to build a training loop from raw components
**What comes next:** Week 2 applies memory optimization techniques to the training loop you build here

This is the foundation week. Everything in the Crucible depends on understanding what happens inside a training loop at the lowest practical level. By the end, you will never need to wonder "what does `trainer.train()` actually do?"

---

## Learning Goals

- [ ] Understand every component of a training loop: forward pass, loss computation, backward pass, optimizer step, scheduler step, zeroing gradients
- [ ] Explain why we zero gradients (accumulation by default in PyTorch)
- [ ] Implement AdamW from scratch and articulate the difference between weight decay and L2 regularization
- [ ] Explain teacher forcing in language model training
- [ ] Understand the relationship between batch size, learning rate, and convergence
- [ ] Articulate what a learning rate scheduler does and why warmup helps
- [ ] Read and interpret loss curves: overfitting, underfitting, learning rate too high/low
- [ ] Understand gradient norms as a training health diagnostic

---

## Implementation Goals

- [ ] Custom `DataLoader` with proper shuffling, batching, and sequence packing
- [ ] Cross-entropy loss function for causal language modeling (with shift by 1)
- [ ] AdamW optimizer implemented from scratch (matching PyTorch's behavior)
- [ ] Three LR schedulers from scratch: linear warmup, cosine decay, cosine with warm restarts
- [ ] Training loop with proper train/eval mode switching
- [ ] Gradient norm logging and optional gradient clipping
- [ ] Loss/LR/grad-norm logging to CSV and matplotlib plots
- [ ] Train GPT-2 124M on WikiText-103 or OpenWebText subset
- [ ] Train/validation split with periodic evaluation
- [ ] Checkpoint saving and resumption

---

## Acceptance Criteria

1. Training loop runs on RTX 5080 and produces monotonically decreasing training loss over 1000 steps on WikiText-103
2. Custom AdamW produces identical parameter updates to `torch.optim.AdamW` on a test case (within floating point tolerance 1e-6)
3. Cosine LR scheduler output matches `torch.optim.lr_scheduler.CosineAnnealingLR` within 1e-7 tolerance
4. Linear warmup scheduler increases LR linearly from 0 to target over exactly N steps
5. Validation loss is computed without gradient tracking and model in eval mode
6. Gradient norms are logged every step and a training run with lr=1e-1 shows exploding gradients (norm > 100)
7. Training can be stopped and resumed from checkpoint with identical loss trajectory (within FP tolerance)
8. Loss curves (train and val) are saved as PNG files showing clear overfitting on a tiny dataset (100 samples)
9. DataLoader properly handles last incomplete batch (drop or pad) and reshuffles each epoch
10. Full training run of 5000 steps completes in under 30 minutes on RTX 5080

---

## Validation Commands

```bash
# Run unit tests for custom AdamW
python -m pytest tests/test_adamw.py -v

# Run unit tests for LR schedulers
python -m pytest tests/test_schedulers.py -v

# Verify training loop produces decreasing loss
python scripts/train.py --steps 100 --dataset tiny | grep "step_loss" | python scripts/verify_decreasing.py

# Compare custom AdamW vs torch AdamW
python scripts/compare_optimizers.py --steps 50 --tolerance 1e-6

# Run full training and generate loss curves
python scripts/train.py --steps 5000 --log-dir runs/week01 --plot

# Verify checkpoint resume produces same loss
python scripts/train.py --steps 100 --checkpoint runs/week01/ckpt_50.pt --verify-resume

# Profile training step time
python scripts/benchmark_step.py --batch-size 8 --seq-len 1024

# Verify gradient clipping is working
python scripts/train.py --steps 100 --clip-grad 1.0 | grep "grad_norm" | python scripts/verify_clipped.py
```

---

## Technical Implementation Details

### Project Structure

```
week01-training-from-scratch/
├── src/
│   ├── __init__.py
│   ├── model.py              # Load GPT-2 124M (from HF for weights only)
│   ├── data.py               # Custom dataset and dataloader
│   ├── optimizer.py          # AdamW from scratch
│   ├── scheduler.py          # LR schedulers from scratch
│   ├── training_loop.py      # The core training loop
│   └── utils.py              # Logging, checkpointing
├── scripts/
│   ├── train.py              # Main entry point
│   ├── compare_optimizers.py # Verify custom == torch
│   └── benchmark_step.py     # Time per step
├── tests/
│   ├── test_adamw.py
│   ├── test_schedulers.py
│   └── test_dataloader.py
└── configs/
    └── default.yaml
```

### Custom AdamW Implementation

```python
# src/optimizer.py
import torch
from torch.optim import Optimizer

class CustomAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                m, v = state['exp_avg'], state['exp_avg_sq']

                # Decoupled weight decay (NOT L2 reg — applied to params directly)
                p.mul_(1 - lr * wd)

                # Moment updates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']
                m_hat = m / bc1
                v_hat = v / bc2

                # Parameter update
                p.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)
```

### LR Scheduler: Cosine with Warmup

```python
# src/scheduler.py
import math

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.current_step <= self.warmup_steps:
                # Linear warmup
                lr = base_lr * (self.current_step / self.warmup_steps)
            else:
                # Cosine decay
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            param_group['lr'] = lr

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
```

### Training Loop Core

```python
# src/training_loop.py
import torch
from pathlib import Path

def train(model, train_loader, val_loader, optimizer, scheduler, config):
    model.train()
    device = torch.device('cuda')
    model.to(device)
    
    log = {'step': [], 'train_loss': [], 'val_loss': [], 'lr': [], 'grad_norm': []}
    
    for step, batch in enumerate(train_loader):
        if step >= config['max_steps']:
            break
        
        input_ids = batch['input_ids'].to(device)
        # Shift targets by 1 for causal LM (teacher forcing)
        labels = input_ids[:, 1:].contiguous()
        logits = model(input_ids[:, :-1])
        
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        loss.backward()
        
        # Log gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('max_grad_norm', float('inf')))
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        log['step'].append(step)
        log['train_loss'].append(loss.item())
        log['lr'].append(scheduler.get_lr()[0])
        log['grad_norm'].append(grad_norm.item())
        
        if step % config['eval_interval'] == 0:
            val_loss = evaluate(model, val_loader, device)
            log['val_loss'].append(val_loss)
            model.train()
        
        if step % config['save_interval'] == 0:
            save_checkpoint(model, optimizer, scheduler, step, config['save_dir'])
    
    return log

def evaluate(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            input_ids = batch['input_ids'].to(device)
            labels = input_ids[:, 1:].contiguous()
            logits = model(input_ids[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
            total_loss += loss.item()
            count += 1
    return total_loss / count
```

### Data Loading with Sequence Packing

```python
# src/data.py
import torch
from torch.utils.data import Dataset, DataLoader

class PackedTextDataset(Dataset):
    """Packs tokenized text into fixed-length sequences for efficient training."""
    
    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        # Pack into sequences of seq_len + 1 (need +1 for label shift)
        n_sequences = len(token_ids) // (seq_len + 1)
        self.data = torch.tensor(token_ids[:n_sequences * (seq_len + 1)], dtype=torch.long)
        self.data = self.data.view(n_sequences, seq_len + 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {'input_ids': self.data[idx]}

def create_dataloaders(token_ids, seq_len, batch_size, val_split=0.05):
    n_val = int(len(token_ids) * val_split)
    train_dataset = PackedTextDataset(token_ids[:-n_val], seq_len)
    val_dataset = PackedTextDataset(token_ids[-n_val:], seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| CUDA OOM on GPT-2 124M | Reduce batch size to 4 or seq_len to 512. 124M should fit in ~2GB with batch=8, seq=1024 |
| Loss not decreasing | Check LR (try 3e-4 to 1e-3 for 124M). Verify labels are shifted correctly. Print first batch logits shape |
| Custom AdamW doesn't match torch | Check bias correction is applied. Weight decay is decoupled (applied to params, not gradients) |
| Cosine scheduler produces NaN | Check division by zero when warmup_steps == total_steps |
| Gradient norms exploding | Normal at high LR. Try gradient clipping at 1.0. If persistent at low LR, check data for anomalies |
| Checkpoint resume gives different loss | Ensure you save and load optimizer state_dict AND scheduler state, not just model weights |
| DataLoader is slow | Use `num_workers=4`, `pin_memory=True`. Pre-tokenize dataset to disk |
| Val loss increases while train loss decreases | Overfitting. Expected on small datasets. This is a success case for detection |

---

## Agent Handoff Template

```
I'm working on Week 1 of the Crucible Phase 1 project: PyTorch Training from Scratch.

Hardware: RTX 5080 16GB VRAM, 32GB RAM, Ubuntu
Project path: ~/crucible/week01-training-from-scratch/

Current status: [DESCRIBE WHERE YOU ARE]

What I've completed:
- [x/o] Custom AdamW optimizer
- [x/o] LR schedulers (warmup, cosine, cosine+restarts)
- [x/o] Training loop with logging
- [x/o] Data loading with sequence packing
- [x/o] Checkpoint save/resume
- [x/o] Validation loop
- [x/o] Loss curve plotting

Current issue: [DESCRIBE THE PROBLEM]
Error message (if any): [PASTE ERROR]

What I've tried: [LIST ATTEMPTS]

Please help me [SPECIFIC ASK] while maintaining the constraint that NO HuggingFace Trainer is used — everything must be raw PyTorch.
```

---

## Out of Scope

- Multi-GPU / distributed training (Week 6)
- Mixed precision / memory optimization (Week 2)
- LoRA or any parameter-efficient methods (Week 4)
- Data pipeline complexity beyond basic packing (Week 3)
- RLHF or alignment techniques (Phase 2)
- Model architecture implementation (using pre-trained weights)
- Deployment or inference optimization
- Wandb integration (Week 7)
