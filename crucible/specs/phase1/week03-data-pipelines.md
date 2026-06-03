# Week 3: Data Pipelines for LLM Training

## Context

**Where it fits:** Phase 1 (Foundations), Week 3 of 7
**Prerequisites:** Week 1 (training loop that consumes batches), Week 2 (memory awareness for large datasets), familiarity with text encoding
**What it builds on:** Weeks 1-2 gave you the training loop; this week builds everything upstream — the pipeline that feeds data into that loop
**What comes next:** Week 4 (LoRA fine-tuning) uses the data pipeline you build here, Week 5 (SFT) builds specialized instruction data formats on top of it

The model is only as good as the data it trains on. This week you build a production-grade data pipeline that handles tokenization, sequence packing, data mixing, quality filtering, and streaming — the exact infrastructure that makes the difference between a toy project and a real training run.

---

## Learning Goals

- [ ] Understand BPE tokenization at the algorithmic level (byte-pair encoding merge rules)
- [ ] Articulate the difference between SentencePiece (unigram vs BPE) and tiktoken
- [ ] Explain sequence packing: why naive padding wastes compute and how packing fills context windows
- [ ] Understand attention masks for packed sequences (prevent cross-contamination between documents)
- [ ] Explain data mixing ratios and why they matter (domain balance)
- [ ] Articulate curriculum learning: why ordering data by difficulty can improve training
- [ ] Understand deduplication: why duplicates hurt (memorization) and how MinHash detects near-duplicates
- [ ] Explain streaming datasets: memory-efficient iteration over datasets larger than RAM

---

## Implementation Goals

- [ ] Implement simplified BPE tokenizer from scratch (train on a small corpus)
- [ ] Build tokenization pipeline using tiktoken/SentencePiece for production use
- [ ] Implement sequence packing with proper attention masks (no cross-document attention)
- [ ] Build data mixer that combines multiple datasets with configurable ratios
- [ ] Implement curriculum learning: sort by length, then by perplexity
- [ ] Build MinHash deduplication pipeline for text data
- [ ] Implement streaming dataset that reads from disk without loading all data into memory
- [ ] Build custom collation function with dynamic batching by sequence length
- [ ] Create end-to-end pipeline: raw text files → training-ready DataLoader

---

## Acceptance Criteria

1. Custom BPE tokenizer trained on 1MB of text produces valid tokenization (roundtrip encode→decode is lossless)
2. Sequence packing achieves >90% token utilization (vs <60% with naive padding) on a dataset with variable-length documents
3. Packed attention mask correctly prevents attention between documents (verified by inspecting attention patterns)
4. Data mixer produces batches with correct ratios (within 5% of target) measured over 1000 batches
5. Streaming dataset processes a 10GB dataset using <500MB of RAM at any point
6. MinHash deduplication identifies >80% of near-duplicate pairs in a synthetic test set
7. Dynamic batching reduces padding waste by at least 40% compared to fixed batching
8. End-to-end pipeline processes raw text to first training batch in under 60 seconds for a 1GB dataset
9. Curriculum learning implementation produces batches ordered by length (monotonically non-decreasing within each epoch)
10. Pipeline is deterministic: same seed produces identical batches across two runs

---

## Validation Commands

```bash
# Test custom BPE tokenizer roundtrip
python -m pytest tests/test_bpe.py -v

# Verify sequence packing utilization
python scripts/measure_packing_efficiency.py --dataset wikitext --seq-len 2048

# Verify attention mask prevents cross-document attention
python scripts/verify_attention_mask.py --visualize

# Test data mixing ratios
python scripts/test_mixing.py --ratios "code:0.3,wiki:0.5,books:0.2" --batches 1000

# Benchmark streaming dataset memory usage
python scripts/benchmark_streaming.py --dataset-size 10GB --max-ram 500MB

# Run deduplication on test set
python scripts/dedup_test.py --method minhash --num-perm 128

# Measure dynamic batching efficiency
python scripts/dynamic_batching.py --compare-fixed --dataset wikitext

# End-to-end pipeline test
python scripts/e2e_pipeline.py --input data/raw/ --output data/processed/ --seq-len 2048

# Determinism test
python scripts/determinism_test.py --seed 42 --batches 100

# Profile pipeline throughput (tokens/sec)
python scripts/throughput_benchmark.py --workers 4
```

---

## Technical Implementation Details

### Project Structure

```
week03-data-pipelines/
├── src/
│   ├── __init__.py
│   ├── bpe_tokenizer.py       # BPE from scratch
│   ├── tokenization.py        # Production tokenizer wrapper
│   ├── packing.py             # Sequence packing with attention masks
│   ├── mixing.py              # Multi-dataset mixer
│   ├── curriculum.py          # Curriculum learning
│   ├── dedup.py               # MinHash deduplication
│   ├── streaming.py           # Memory-efficient streaming
│   ├── collation.py           # Custom collation and dynamic batching
│   └── pipeline.py            # End-to-end orchestration
├── scripts/
│   ├── e2e_pipeline.py
│   ├── measure_packing_efficiency.py
│   ├── benchmark_streaming.py
│   ├── dedup_test.py
│   └── throughput_benchmark.py
├── tests/
│   ├── test_bpe.py
│   ├── test_packing.py
│   ├── test_mixing.py
│   ├── test_streaming.py
│   └── test_dedup.py
└── data/
    └── raw/
        └── .gitkeep
```

### BPE Tokenizer from Scratch

```python
# src/bpe_tokenizer.py
from collections import Counter
import re

class SimpleBPE:
    """Minimal BPE tokenizer to understand the algorithm."""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = []  # List of (pair_to_merge) in order learned
        self.vocab = {}   # token_id -> bytes
    
    def train(self, text: str):
        # Start with byte-level vocabulary (256 base tokens)
        tokens = list(text.encode('utf-8'))
        self.vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256
        
        while next_id < self.vocab_size:
            # Count adjacent pairs
            pairs = Counter()
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += 1
            
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = pairs.most_common(1)[0][0]
            self.merges.append(best_pair)
            
            # Merge all occurrences
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(next_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            self.vocab[next_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            tokens = new_tokens
            next_id += 1
    
    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode('utf-8'))
        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    merge_id = 256 + self.merges.index(pair)
                    new_tokens.append(merge_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens
    
    def decode(self, token_ids: list[int]) -> str:
        byte_seq = b''.join(self.vocab[t] for t in token_ids)
        return byte_seq.decode('utf-8', errors='replace')
```

### Sequence Packing with Attention Masks

```python
# src/packing.py
import torch
import numpy as np

class SequencePacker:
    """Pack multiple documents into fixed-length sequences with attention masks."""
    
    def __init__(self, seq_len: int, pad_token_id: int = 0):
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
    
    def pack(self, documents: list[list[int]]) -> list[dict]:
        """
        Pack documents into sequences. Each sequence may contain multiple documents.
        Returns attention mask that prevents cross-document attention.
        """
        packed_sequences = []
        current_tokens = []
        current_doc_ids = []  # Track which doc each token belongs to
        doc_counter = 0
        
        for doc in documents:
            if len(current_tokens) + len(doc) > self.seq_len:
                # Emit current sequence
                if current_tokens:
                    packed_sequences.append(self._finalize(current_tokens, current_doc_ids))
                current_tokens = []
                current_doc_ids = []
                doc_counter = 0
            
            # If single doc exceeds seq_len, chunk it
            if len(doc) > self.seq_len:
                for i in range(0, len(doc), self.seq_len):
                    chunk = doc[i:i + self.seq_len]
                    packed_sequences.append(self._finalize(chunk, [0] * len(chunk)))
                continue
            
            current_tokens.extend(doc)
            current_doc_ids.extend([doc_counter] * len(doc))
            doc_counter += 1
        
        if current_tokens:
            packed_sequences.append(self._finalize(current_tokens, current_doc_ids))
        
        return packed_sequences
    
    def _finalize(self, tokens, doc_ids):
        # Pad to seq_len
        pad_len = self.seq_len - len(tokens)
        input_ids = tokens + [self.pad_token_id] * pad_len
        
        # Build causal attention mask that blocks cross-document attention
        # Shape: (seq_len, seq_len) — True means "can attend"
        n = len(tokens)
        attn_mask = torch.zeros(self.seq_len, self.seq_len, dtype=torch.bool)
        
        for i in range(n):
            for j in range(i + 1):  # Causal: can only attend to past
                if doc_ids[i] == doc_ids[j]:  # Same document
                    attn_mask[i, j] = True
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': attn_mask,
            'num_real_tokens': n,
        }
```

### Data Mixer

```python
# src/mixing.py
import random
from typing import Iterator

class DataMixer:
    """Mix multiple datasets according to specified ratios."""
    
    def __init__(self, datasets: dict[str, Iterator], ratios: dict[str, float], seed: int = 42):
        assert abs(sum(ratios.values()) - 1.0) < 1e-6, "Ratios must sum to 1.0"
        assert set(datasets.keys()) == set(ratios.keys()), "Dataset and ratio keys must match"
        
        self.datasets = datasets
        self.ratios = ratios
        self.rng = random.Random(seed)
    
    def __iter__(self):
        sources = list(self.ratios.keys())
        weights = [self.ratios[s] for s in sources]
        
        while True:
            # Weighted random selection of source
            source = self.rng.choices(sources, weights=weights, k=1)[0]
            try:
                yield next(self.datasets[source])
            except StopIteration:
                # Remove exhausted dataset, renormalize
                idx = sources.index(source)
                sources.pop(idx)
                weights.pop(idx)
                del self.datasets[source]
                if not sources:
                    break
                total = sum(weights)
                weights = [w / total for w in weights]
```

### MinHash Deduplication

```python
# src/dedup.py
import hashlib
import numpy as np
from dataclasses import dataclass

@dataclass
class MinHashSignature:
    doc_id: int
    signature: np.ndarray

class MinHashDedup:
    """Near-duplicate detection using MinHash + LSH."""
    
    def __init__(self, num_perm: int = 128, ngram_size: int = 5, threshold: float = 0.8):
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.threshold = threshold
        # Random hash parameters
        self.a = np.random.randint(1, 2**31, size=num_perm)
        self.b = np.random.randint(0, 2**31, size=num_perm)
        self.p = 2**31 - 1  # Mersenne prime
    
    def _ngrams(self, text: str) -> set[str]:
        words = text.split()
        return {' '.join(words[i:i+self.ngram_size]) for i in range(len(words) - self.ngram_size + 1)}
    
    def _hash_ngram(self, ngram: str) -> int:
        return int(hashlib.md5(ngram.encode()).hexdigest(), 16) % self.p
    
    def compute_signature(self, text: str, doc_id: int) -> MinHashSignature:
        ngrams = self._ngrams(text)
        if not ngrams:
            return MinHashSignature(doc_id, np.full(self.num_perm, np.inf))
        
        hashes = np.array([self._hash_ngram(ng) for ng in ngrams])
        
        # MinHash: for each permutation, take minimum hash value
        signature = np.full(self.num_perm, np.inf)
        for h in hashes:
            perm_hashes = (self.a * h + self.b) % self.p
            signature = np.minimum(signature, perm_hashes)
        
        return MinHashSignature(doc_id, signature)
    
    def jaccard_estimate(self, sig1: MinHashSignature, sig2: MinHashSignature) -> float:
        return np.mean(sig1.signature == sig2.signature)
    
    def find_duplicates(self, documents: list[str]) -> set[tuple[int, int]]:
        signatures = [self.compute_signature(doc, i) for i, doc in enumerate(documents)]
        duplicates = set()
        
        for i in range(len(signatures)):
            for j in range(i + 1, len(signatures)):
                sim = self.jaccard_estimate(signatures[i], signatures[j])
                if sim >= self.threshold:
                    duplicates.add((i, j))
        
        return duplicates
```

### Streaming Dataset

```python
# src/streaming.py
import json
import mmap
from pathlib import Path
from torch.utils.data import IterableDataset

class StreamingTextDataset(IterableDataset):
    """Memory-efficient streaming from large files using memory-mapped IO."""
    
    def __init__(self, file_paths: list[str], tokenizer, seq_len: int, shuffle_buffer: int = 10000):
        self.file_paths = [Path(p) for p in file_paths]
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle_buffer = shuffle_buffer
    
    def __iter__(self):
        buffer = []
        token_buffer = []
        
        for file_path in self.file_paths:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    tokens = self.tokenizer.encode(line)
                    token_buffer.extend(tokens)
                    
                    # Yield complete sequences from token buffer
                    while len(token_buffer) >= self.seq_len + 1:
                        seq = token_buffer[:self.seq_len + 1]
                        token_buffer = token_buffer[self.seq_len + 1:]
                        buffer.append(seq)
                        
                        # Shuffle buffer
                        if len(buffer) >= self.shuffle_buffer:
                            import random
                            random.shuffle(buffer)
                            while len(buffer) > self.shuffle_buffer // 2:
                                yield {'input_ids': buffer.pop()}
        
        # Flush remaining buffer
        import random
        random.shuffle(buffer)
        for item in buffer:
            yield {'input_ids': item}
```

---

## If You Get Stuck

| Problem | Solution |
|---------|----------|
| BPE training too slow | Limit training corpus to 1-5MB. Use Counter for pair frequencies. Real tokenizers use C++ |
| Packed attention mask is wrong shape | Should be `(seq_len, seq_len)` for each sample. For batch, expand to `(batch, 1, seq_len, seq_len)` for multi-head attention |
| Data mixer ratios drift over time | Measure every 1000 batches. Use reservoir sampling for exact ratios at small scale |
| Streaming dataset runs out of memory | Ensure you're not collecting all results in a list. Use `yield`, not `return`. Check `shuffle_buffer` size |
| MinHash too slow for large datasets | Use LSH banding to avoid O(n²) comparisons. Band signatures into buckets |
| Tokenizer encode/decode not roundtripping | Byte-fallback tokens: some bytes may not decode cleanly. Use `errors='replace'` for diagnostics only |
| Dynamic batching causes training instability | Sequence length variation changes effective compute per batch. Normalize loss per token, not per sequence |
| Pipeline not deterministic | Set seeds in: Python random, numpy, torch, DataLoader worker_init_fn |

---

## Agent Handoff Template

```
I'm working on Week 3 of the Crucible Phase 1 project: Data Pipelines for LLM Training.

Hardware: RTX 5080 16GB VRAM, 32GB RAM, Ubuntu
Project path: ~/crucible/week03-data-pipelines/

Current status: [DESCRIBE WHERE YOU ARE]

What I've completed:
- [x/o] BPE tokenizer from scratch
- [x/o] Sequence packing with attention masks
- [x/o] Data mixer
- [x/o] Curriculum learning
- [x/o] MinHash deduplication
- [x/o] Streaming dataset
- [x/o] Dynamic batching / collation
- [x/o] End-to-end pipeline integration

Dataset I'm working with: [NAME, SIZE, FORMAT]
Current issue: [DESCRIBE THE PROBLEM]
Error message (if any): [PASTE ERROR]

Please help me [SPECIFIC ASK] while maintaining memory efficiency (target: process 10GB+ data with <500MB RAM).
```

---

## Out of Scope

- Training the model (Weeks 1-2 cover this; this week is data-only)
- Instruction/chat formatting (Week 5)
- Web scraping or data collection from the internet
- Multimodal data (images, audio)
- Tokenizer training at scale (we implement to understand, then use existing tokenizers)
- Distributed data loading across machines
- Data privacy / PII detection
- Legal considerations of training data
