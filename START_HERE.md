# START HERE

Welcome to the **Takens-Based Transformer (TBT)** proof-of-concept implementation.

This repository demonstrates that **attention mechanisms in transformers can be replaced by explicit phase-space embeddings** derived from Takens' delay-coordinate reconstruction theorem.

---

## ⚠️ Important: This is Proof-of-Principle Code

This software was developed **purely to demonstrate that the core idea works**. It is:

- ✅ A working demonstration of the Takens-based approach
- ✅ Suitable for experimentation and research exploration
- ✅ Intentionally minimal and readable

It is **NOT**:

- ❌ Production-ready software
- ❌ Optimized for performance or scale
- ❌ A drop-in replacement for standard transformers

Please read [`SCOPE_AND_LIMITATIONS.md`](SCOPE_AND_LIMITATIONS.md) for full context.

---

## What's in This Repository?

This repository contains:

1. **Core Takens embedding implementation** - replacing attention with delay-coordinate reconstruction
2. **MARINA architecture** - a multi-channel extension with identity-aware encoding
3. **Training scripts** - for different modes (Q&A, Question-Bridge-Answer, continuous text)
4. **Inference tools** - for testing trained models
5. **Utility tools** - for vocabulary management and dataset preparation

---

## Quick Start

### 1. Requirements

```bash
# Core dependencies
pip install torch numpy

# Optional for dataset handling
pip install pandas
```

### 2. Understanding the Architecture

Before diving into code, read about the conceptual foundation:

- **Paper**: "Introducing the Takens-Based Transformer" (Haylett, 2025)  
  Available at: https://finitemechanics.com/papers/takens_transformer.pdf
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - How the code implements these ideas

### 3. Training Your First Model

The simplest way to start:

```python
# Edit configuration in train_flexible.py
# Then run it - no command-line arguments needed!
python train_flexible.py
```

See **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** for full details on:
- Different training modes (QA, QBA, CONTINUOUS)
- Chunking strategies
- Configuration options

### 4. Running Inference

Once you have a trained model:

```python
# Edit paths in unified_run.py
# Then run it
python unified_run.py
```

See **[INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)** for:
- Standard Q&A inference
- Question-Bridge-Answer (QBA) mode
- Vocabulary checking

---

## Documentation Structure

This repository includes several documentation files:

| File | Purpose |
|------|---------|
| **START_HERE.md** (this file) | Entry point and orientation |
| **[README.md](README.md)** | High-level overview and citations |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Technical architecture and file breakdown |
| **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** | How to train models |
| **[INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)** | How to run trained models |
| **[CODE_REFERENCE.md](CODE_REFERENCE.md)** | Detailed technical reference |
| **[SCOPE_AND_LIMITATIONS.md](SCOPE_AND_LIMITATIONS.md)** | Important context on intent and limitations |
| **[PROVENANCE.md](PROVENANCE.md)** | Development timeline and authorship |
| **[LICENSE.md](LICENSE.md)** | Mozilla Public License 2.0 |

---

## File Organization

```
takens-embedding-transformer/
│
├── START_HERE.md              ← You are here
├── README.md
├── ARCHITECTURE.md
├── TRAINING_GUIDE.md
├── INFERENCE_GUIDE.md
├── CODE_REFERENCE.md
├── SCOPE_AND_LIMITATIONS.md
├── PROVENANCE.md
├── LICENSE.md
│
├── Core Architecture Files
│   ├── takens_embedding.py    # Delay-coordinate embeddings
│   └── tbt_architecture.py    # Base TBT transformer
│
├── MARINA Multi-Channel Extension
│   ├── mvec_encoder.py        # Multi-channel encoding
│   ├── mvec_model.py          # Identity-aware model
│   ├── mvec_dataset.py        # Dataset handling
│   └── mvec_training.py       # Training utilities
│
├── Training Scripts
│   ├── train_flexible.py      # Main training script
│   ├── flexible_dataset.py    # Multiple chunking modes
│   ├── flexible_training.py   # Flexible training
│   └── unified_training.py    # Unified framework
│
├── Inference Scripts
│   ├── unified_run.py         # Standard inference
│   ├── unified_run_qba.py     # QBA two-phase generation
│   └── unified_run_vocab_check.py  # With vocab checking
│
├── Utilities
│   ├── training_utils.py      # Generic training utilities
│   ├── vocab_checker.py       # Check for OOV words
│   └── vocab_updater.py       # Update vocabulary files
│
└── models/                     # Your trained models (create this)
    └── utils/                  # Your utility scripts (create this)
```

---

## Typical Workflows

### Workflow 1: Train a Simple Q&A Model

1. Prepare a CSV file with `question` and `answer` columns
2. Edit `train_flexible.py` configuration section
3. Set `CHUNKING_MODE = 'pairs'` and `DATA_PATH` to your CSV
4. Run: `python train_flexible.py`
5. Test with `unified_run.py`

### Workflow 2: Train with Question-Bridge-Answer

1. Prepare CSV with `question`, `bridge`, and `answer` columns
2. Use `mvec_dataset.py` with `mode=EncodingMode.QBA`
3. Train using `mvec_training.py`
4. Test with `unified_run_qba.py` to see two-phase generation

### Workflow 3: Continuous Text Modeling

1. Prepare text file or CSV with `text` column
2. Set `CHUNKING_MODE = 'continuous'` in `train_flexible.py`
3. Adjust `chunk_size` for END marker frequency
4. Train and test

---

## Key Concepts

### Takens Embedding

Instead of attention between all token pairs, TBT uses **delay-coordinate embeddings**:

```
For position t, create embedding:
[x(t), x(t-τ₁), x(t-τ₂), ..., x(t-τₙ)]
```

This reconstructs the phase space of the underlying dynamics. See `takens_embedding.py` and **[ARCHITECTURE.md](ARCHITECTURE.md)**.

### Multi-Channel Identity (MARINA)

The MARINA extension adds **identity channels**:

- `USER` (0) - Input from user
- `INTERNAL` (1) - Hidden reasoning (e.g., bridge in QBA)
- `VISIBLE` (2) - Output to user

This allows the model to maintain separate "thought" and "speech" streams. See `mvec_encoder.py` and `mvec_model.py`.

### Chunking Strategies

Different chunking modes create different geometric structures:

- **pairs** - Isolated Q&A pairs (fragmented landscape)
- **conversation** - Chained Q&A pairs (flowing landscape)
- **sliding** - Overlapping windows (continuous landscape)
- **paragraph** - Natural text boundaries
- **document** - Full documents

See `flexible_dataset.py` and **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**.

---

## Common Questions

### Q: Can I use this in production?

**No.** This is proof-of-principle code. It demonstrates the concept works but would require significant engineering effort to make production-ready. See [SCOPE_AND_LIMITATIONS.md](SCOPE_AND_LIMITATIONS.md).

### Q: How does this compare to standard transformers?

This implementation:
- Uses ~1-2M parameters vs. billions in large models
- Trains on small datasets (thousands of examples)
- Focuses on conceptual clarity over optimization
- Demonstrates geometric structure in language

It's not designed to compete on benchmarks, but to explore a different structural foundation.

### Q: What can I do with this?

Great uses:
- Understand the geometric interpretation of language modeling
- Experiment with delay-coordinate approaches
- Explore identity-aware architectures
- Create small domain-specific models
- Research phase-space structure in NLP

### Q: Do I need a GPU?

No. These models are small enough to train on CPU. A GPU helps but isn't required.

### Q: Where do I get training data?

You'll need to prepare your own:
- CSV files with question/answer pairs
- Text files for continuous modeling
- Tagged conversations for QBA mode

Start with small datasets (hundreds to thousands of examples).

---

## Getting Help

1. **Read the documentation** - Start with [ARCHITECTURE.md](ARCHITECTURE.md)
2. **Check the code** - Most files have built-in test/example code at the bottom
3. **Review scope** - [SCOPE_AND_LIMITATIONS.md](SCOPE_AND_LIMITATIONS.md) explains what's out of scope
4. **Contact the author** - kevin.haylett@gmail.com

---

## Citation

If you use or build upon this work:

### Paper Citation

```
Kevin R. Haylett, Introducing the Takens-Based Transformer, December 2025.
Available at: https://finitemechanics.com/papers/takens_transformer.pdf
```

### Code Citation

```
Haylett, K. R. (2025).
Takens Embedding Transformer.
GitHub repository:
https://github.com/KevinHaylett/takens-embedding-transformer
```

---

## Further Reading

- **Finite Mechanics**: https://www.finitemechanics.com
- **Geofinitism Framework**: https://geofinitism.com
- **Author's Blog**: https://kevinhaylett.substack.com

---

## Next Steps

1. ✅ You've read this file
2. ➡️ Read **[ARCHITECTURE.md](ARCHITECTURE.md)** to understand how it works
3. ➡️ Read **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** to train your first model
4. ➡️ Experiment and explore!

---

**Remember**: This is a starting point, not a destination. The goal is to open a line of inquiry, not to provide final answers.

---

*"Simul Pariter" - Together Equally*

Kevin R. Haylett, PhD  
Independent Research  
kevin.haylett@gmail.com
