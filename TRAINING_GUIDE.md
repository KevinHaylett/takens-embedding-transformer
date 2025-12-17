# Training Guide

This guide explains how to train Takens-Based Transformer models.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Preparation](#data-preparation)
3. [Training Modes](#training-modes)
4. [Chunking Strategies](#chunking-strategies)
5. [Configuration Options](#configuration-options)
6. [Training Workflows](#training-workflows)
7. [Monitoring Training](#monitoring-training)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

The simplest way to train a model:

```python
# 1. Edit train_flexible.py configuration section
# 2. Set your DATA_PATH and other parameters
# 3. Run:
python train_flexible.py
```

No command-line arguments needed - everything is configured by editing the file.

---

## Data Preparation

### CSV Format

Training data should be in CSV format with specific column names depending on the mode:

#### For QA Mode
```csv
question,answer
What is the capital of France?,Paris
How hot is Mercury?,Very hot - about 430°C during the day
```

#### For QBA Mode (Question-Bridge-Answer)
```csv
question,bridge,answer
What is the capital of France?,geography France capital,Paris is the capital of France
How hot is Mercury?,planet Mercury temperature,Mercury is very hot - about 430°C during the day
```

The "bridge" column contains **hidden reasoning** that guides the model from question to answer.

#### For CONTINUOUS Mode
```csv
text
Once upon a time there was a wise old sage who lived in the mountains.
The sage taught many students the ways of wisdom and patience.
```

### Text Format

For continuous text modeling, you can also use plain `.txt` files:

```python
# In flexible_dataset.py, set:
DATA_FORMAT = 'text'
DATA_PATH = 'my_corpus.txt'
```

### Data Size Guidelines

This is proof-of-principle code optimized for **small datasets**:

- **Minimum**: 100-500 examples
- **Typical**: 1,000-10,000 examples
- **Maximum practical**: 50,000-100,000 examples

For larger datasets, you may need to modify batch processing and memory management.

### Creating Your Dataset

Example Python script to create QA dataset:

```python
import csv

qa_pairs = [
    ("What is Python?", "A high-level programming language"),
    ("What is a list?", "An ordered collection of items"),
    # ... more pairs
]

with open('my_qa_data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['question', 'answer'])
    writer.writerows(qa_pairs)
```

---

## Training Modes

### Mode 1: QA (Question-Answer)

**Best for**: Simple Q&A pairs, lookup tasks, factual responses

**Structure**:
```
USER: [question tokens]
VISIBLE: [answer tokens] END
```

**Use case**: Direct question → answer mapping without intermediate reasoning.

**Example**:
```python
CHUNKING_MODE = 'pairs'  # Each Q&A isolated
# or
CHUNKING_MODE = 'conversation'  # Chain multiple Q&A together
```

---

### Mode 2: QBA (Question-Bridge-Answer)

**Best for**: Complex reasoning, multi-step thinking, transparent AI

**Structure**:
```
USER: [question tokens]
INTERNAL: [bridge tokens] END     # Hidden from user
VISIBLE: [answer tokens] END       # Shown to user
```

**The bridge**: A geometric pathway through meaning-space. The model generates internal tokens that guide it from question to answer, but these are never shown to the user.

**Use case**: When you want the model to have explicit "thinking" space before answering.

**Training with QBA**:
```python
# In your CSV:
question,bridge,answer
What causes rain?,water cycle evaporation condensation,Rain forms when water vapor condenses in clouds

# The bridge should be:
# - Short (5-20 words)
# - Semantic keywords/concepts
# - Not a full sentence
# - Represents reasoning pathway
```

**Why QBA works**:
- Creates 3-basin structure in phase space
- Bridge acts as geometric scaffold
- Improves answer quality
- Enables reasoning inspection

---

### Mode 3: CONTINUOUS

**Best for**: Text generation, story continuation, language modeling

**Structure**:
```
VISIBLE: [all tokens] END END END ...
```

END markers inserted periodically based on `chunk_size`.

**Use case**: General text generation without Q&A structure.

**Configuration**:
```python
mode = EncodingMode.CONTINUOUS
chunk_size = 50  # Insert END marker every 50 tokens
```

---

## Chunking Strategies

Chunking determines how you split data into training samples. This is **critical** because it shapes the geometric landscape in phase space.

### Strategy 1: PAIRS (Fragmented Landscape)

```python
CHUNKING_MODE = 'pairs'
```

**What it does**: Each Q&A pair is an isolated training sample.

**Phase space structure**:
```
Basin 1: Q1 → A1
Basin 2: Q2 → A2 (disconnected from Basin 1)
Basin 3: Q3 → A3 (disconnected from others)
```

**Pros**:
- Simple
- No sample overlap
- Clean basin boundaries

**Cons**:
- Fragmented phase space
- No conversational flow
- Less manifold connectivity

**Best for**: Independent Q&A, lookup tasks, factual questions

---

### Strategy 2: CONVERSATION (Flowing Landscape)

```python
CHUNKING_MODE = 'conversation'
TURNS_PER_CONVERSATION = 3  # Chain 3 Q&A pairs
```

**What it does**: Chains multiple Q&A pairs into conversations.

**Phase space structure**:
```
Q1 → A1 → Q2 → A2 → Q3 → A3
(connected trajectory through phase space)
```

**Pros**:
- Conversational flow
- Richer manifold structure
- Context between turns
- Better geometric connectivity

**Cons**:
- Longer sequences
- More memory
- May mix unrelated topics if data isn't structured

**Best for**: Dialogue, conversational AI, context-aware responses

---

### Strategy 3: SLIDING (Continuous Landscape)

```python
CHUNKING_MODE = 'sliding'
MAX_SEQ_LEN = 256
STRIDE = 128  # 50% overlap
```

**What it does**: Creates overlapping windows that slide across the text.

**Phase space structure**:
```
Sample 1: [tokens 0-255]
Sample 2: [tokens 128-383]  (overlaps with sample 1)
Sample 3: [tokens 256-511]  (overlaps with sample 2)
```

**Pros**:
- Maximum manifold connectivity
- Smooth transitions
- Best for continuous text
- No abrupt boundaries

**Cons**:
- Data duplication (overlaps)
- Longer training
- More memory

**Best for**: Continuous text modeling, story generation, language modeling

**Configuration**:
- Small stride (64-128) = More overlap, slower training, better connectivity
- Large stride (200-256) = Less overlap, faster training, more fragmented

---

### Strategy 4: PARAGRAPH

```python
CHUNKING_MODE = 'paragraph'
```

**What it does**: Splits text on paragraph boundaries.

**Best for**: Structured documents, articles, essays

**Pros**:
- Natural boundaries
- Preserves logical units

**Cons**:
- Variable sequence lengths
- May exceed `MAX_SEQ_LEN`

---

### Strategy 5: DOCUMENT

```python
CHUNKING_MODE = 'document'
```

**What it does**: Each document is one training sample.

**Best for**: Short documents, full-context modeling

**Warning**: Documents must fit in `MAX_SEQ_LEN` or will be truncated.

---

## Configuration Options

### Essential Parameters

Edit these in `train_flexible.py`:

```python
# ============================================================
# DATA CONFIGURATION
# ============================================================

DATA_PATH = 'your_data.csv'      # Path to your CSV file
DATA_FORMAT = 'csv'               # 'csv', 'text', or 'tagged_text'
CHUNKING_MODE = 'pairs'           # See chunking strategies above

# ============================================================
# SEQUENCE CONFIGURATION
# ============================================================

MAX_SEQ_LEN = 256                 # Maximum tokens per sample
TURNS_PER_CONVERSATION = 3        # For 'conversation' mode
STRIDE = 128                      # For 'sliding' mode

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

EMBED_DIM = 128                   # Token embedding dimension
HIDDEN_DIM = 128                  # Hidden layer dimension
NUM_LAYERS = 4                    # Number of TBT layers
IDENTITY_EMBED_DIM = 32           # Identity embedding dimension
MAX_DELAY = 128                   # Maximum Takens delay
DROPOUT = 0.1                     # Dropout probability

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 8                    # Batch size
EPOCHS = 50                       # Training epochs
LEARNING_RATE = 3e-4              # Initial learning rate
VAL_SPLIT = 0.1                   # Validation split (0.1 = 10%)

# ============================================================
# LOSS WEIGHTS
# ============================================================

WORD_LOSS_WEIGHT = 1.0            # Weight for word prediction loss
END_LOSS_WEIGHT = 1.0             # Weight for end signal loss

# ============================================================
# SAVE PATHS
# ============================================================

SAVE_PATH = 'my_model.pt'         # Where to save trained model
SAVE_VOCAB = 'my_vocab.json'      # Where to save vocabulary

# ============================================================
# DEVICE
# ============================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

---

### Parameter Guidelines

#### Embedding Dimensions

```python
EMBED_DIM = 128     # Small models, quick training
EMBED_DIM = 256     # Medium models, better capacity
EMBED_DIM = 512     # Larger models, more expressiveness
```

**Rule of thumb**: Start with 128, increase if underfitting.

#### Hidden Dimension

```python
HIDDEN_DIM = EMBED_DIM  # Typical choice
```

Can be different, but must match if using `tie_weights=True`.

#### Number of Layers

```python
NUM_LAYERS = 2      # Minimal model
NUM_LAYERS = 4      # Standard (recommended)
NUM_LAYERS = 6-8    # Deeper models
```

**Trade-off**: More layers = more capacity but slower training and potential overfitting.

#### Batch Size

```python
BATCH_SIZE = 4      # Small memory, slower
BATCH_SIZE = 8      # Standard (recommended)
BATCH_SIZE = 16-32  # Larger batches, faster but more memory
```

**GPU memory guide**:
- 4GB GPU: batch_size = 4-8
- 8GB GPU: batch_size = 8-16
- 16GB GPU: batch_size = 16-32

#### Learning Rate

```python
LEARNING_RATE = 3e-4   # Standard (recommended)
LEARNING_RATE = 1e-4   # Conservative (if training unstable)
LEARNING_RATE = 5e-4   # Aggressive (if training slow)
```

We use **cosine annealing** scheduler by default:
- Starts at LEARNING_RATE
- Gradually decreases to LEARNING_RATE * 0.1
- Smooth decay over epochs

#### Loss Weights

```python
WORD_LOSS_WEIGHT = 1.0
END_LOSS_WEIGHT = 1.0
```

**When to adjust**:
- If model never stops generating: Increase END_LOSS_WEIGHT to 2.0 or 3.0
- If model stops too early: Decrease END_LOSS_WEIGHT to 0.5
- Usually default 1.0:1.0 works well

---

## Training Workflows

### Workflow 1: Simple Q&A Model

**Goal**: Train a basic question-answer model.

```python
# 1. Prepare data
# Create qa_data.csv with 'question' and 'answer' columns

# 2. Configure train_flexible.py
DATA_PATH = 'qa_data.csv'
CHUNKING_MODE = 'pairs'
MAX_SEQ_LEN = 128
EMBED_DIM = 128
NUM_LAYERS = 4
BATCH_SIZE = 8
EPOCHS = 50

# 3. Run
python train_flexible.py

# 4. Output files
# - my_model.pt (trained model)
# - my_vocab.json (vocabulary)
```

**Expected results**:
- Training should converge in 20-50 epochs
- Validation loss should plateau
- Model saved at best validation loss

---

### Workflow 2: QBA Reasoning Model

**Goal**: Train model with explicit reasoning bridges.

```python
# 1. Prepare QBA data
# CSV with 'question', 'bridge', 'answer' columns

# 2. Use mvec_dataset.py with QBA mode
from mvec_encoder import MVecEncoder, EncodingMode
from mvec_dataset import MVecDataset

encoder = MVecEncoder()
encoder.build_vocab(texts)

dataset = MVecDataset(
    csv_path='qba_data.csv',
    encoder=encoder,
    mode=EncodingMode.QBA,
    max_seq_len=256
)

# 3. Train with mvec_training.py
# (See example in mvec_training.py or train_flexible.py)

# 4. Test with unified_run_qba.py
# Shows two-phase generation: bridge → answer
```

---

### Workflow 3: Conversational Model

**Goal**: Train model that maintains context across turns.

```python
# 1. Prepare conversational data
# CSV with multiple Q&A pairs

# 2. Configure for conversation chunking
DATA_PATH = 'conversation_data.csv'
CHUNKING_MODE = 'conversation'
TURNS_PER_CONVERSATION = 3  # Chain 3 Q&A pairs
MAX_SEQ_LEN = 512           # Longer sequences needed

# 3. Train
python train_flexible.py
```

**Tips**:
- Start with 2-3 turns, increase gradually
- Longer sequences need more memory
- May need to reduce batch size

---

### Workflow 4: Continuous Text Model

**Goal**: General text generation.

```python
# 1. Prepare text corpus
# Either CSV with 'text' column or plain .txt file

# 2. Configure for continuous mode
DATA_PATH = 'corpus.txt'
DATA_FORMAT = 'text'
CHUNKING_MODE = 'continuous'
MAX_SEQ_LEN = 256

# 3. In code, use EncodingMode.CONTINUOUS
mode = EncodingMode.CONTINUOUS
chunk_size = 50  # END markers every 50 tokens

# 4. Train
python train_flexible.py
```

---

## Monitoring Training

### Console Output

Training shows progress every N batches:

```
Epoch 1/50
────────────────────────────────────────────────────────────
Training: 100%|████████| 125/125 [00:45<00:00, 2.75it/s, loss=4.5231]
Train Loss: 4.5231 | Time: 45.23s
Validating: 100%|████████| 14/14 [00:03<00:00, 4.12it/s]
Val Loss: 4.3892
✓ Saved best model to my_model.pt
Learning Rate: 0.000300

Epoch 2/50
────────────────────────────────────────────────────────────
...
```

### What to Watch

**Good training**:
- Train loss steadily decreases
- Val loss decreases and plateaus
- Losses converge (don't diverge)
- Learning rate gradually decreases

**Bad signs**:
- Loss stays constant or increases
- Val loss much higher than train loss (overfitting)
- Loss = NaN (training collapsed)
- No improvement after many epochs

---

### Training Metrics

At the end of training:

```
Training Complete!
══════════════════════════════════════════════════════════════════════
Best validation loss: 2.3451
Model saved to: my_model.pt
Vocabulary saved to: my_vocab.json

Final validation metrics:
  Total loss: 2.3451
  Word loss: 2.1203
  End loss: 0.2248
```

**Interpreting losses**:
- **Word loss**: Cross-entropy for next-word prediction
  - Good: 2.0-3.0 for small models
  - Better: 1.5-2.0
  - Excellent: <1.5
  
- **End loss**: Binary cross-entropy for stop signals
  - Good: 0.2-0.4
  - Better: 0.1-0.2
  - Excellent: <0.1

**Perplexity**: `exp(word_loss)`
- Word loss 2.0 → Perplexity ~7.4
- Word loss 1.5 → Perplexity ~4.5
- Lower is better

---

### Checkpoints

The best model is automatically saved when validation loss improves:

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'model_config': {...},  # Model architecture
    'train_metrics': {...}, # Training losses
    'val_metrics': {...}    # Validation losses
}
```

To load:

```python
checkpoint = torch.load('my_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## Troubleshooting

### Problem: Loss is NaN

**Causes**:
- Learning rate too high
- Numerical instability
- Bad initialization

**Solutions**:
```python
LEARNING_RATE = 1e-4  # Reduce from 3e-4
DROPOUT = 0.1         # Add dropout if not present
# Check for NaN in data
```

---

### Problem: No Improvement

**Symptoms**: Loss stays constant over many epochs.

**Causes**:
- Learning rate too low
- Model too small
- Data too hard
- Optimization stuck

**Solutions**:
```python
LEARNING_RATE = 5e-4  # Increase
NUM_LAYERS = 6        # Increase model capacity
EMBED_DIM = 256       # Increase
# Try different random seed
```

---

### Problem: Overfitting

**Symptoms**: Train loss << Val loss (e.g., train=1.0, val=3.0)

**Solutions**:
```python
DROPOUT = 0.2         # Increase dropout
VAL_SPLIT = 0.15      # More validation data
# Add more training data
# Reduce model size
# Early stopping will help
```

---

### Problem: Model Never Stops Generating

**Symptoms**: Generation goes to max_tokens, never predicts END.

**Solutions**:
```python
END_LOSS_WEIGHT = 2.0  # Increase from 1.0
# Check that training data has END markers
# Verify end_ids in dataset
```

---

### Problem: Model Stops Too Early

**Symptoms**: Generates 1-2 tokens then stops.

**Solutions**:
```python
END_LOSS_WEIGHT = 0.5  # Decrease from 1.0
# Check distribution of END markers in training data
# May need more training epochs
```

---

### Problem: Out of Memory

**Symptoms**: CUDA out of memory error.

**Solutions**:
```python
BATCH_SIZE = 4        # Reduce batch size
MAX_SEQ_LEN = 128     # Reduce sequence length
EMBED_DIM = 64        # Reduce model size
NUM_LAYERS = 2        # Reduce layers
# Or train on CPU (slower but no memory limit)
DEVICE = 'cpu'
```

---

### Problem: Slow Training

**Solutions**:
- Use GPU if available
- Increase batch size (if memory allows)
- Reduce validation frequency
- Use smaller validation set
- Reduce number of layers

---

### Problem: Vocabulary Issues

**Symptoms**: Many `<unk>` tokens, poor quality outputs.

**Solutions**:
```python
# Build vocabulary from more data
encoder.build_vocab(texts, min_freq=1)  # Lower frequency threshold

# Or manually add words
from vocab_updater import VocabUpdater
updater = VocabUpdater('vocab.json')
updater.add_words(['word1', 'word2'])
updater.save()
```

---

## Advanced Topics

### Custom Loss Functions

To modify loss calculation, edit `mvec_training.py`:

```python
class MVecTrainer:
    def _compute_losses(self, ...):
        # Modify loss computation here
        word_loss = ...
        end_loss = ...
        return word_loss, end_loss
```

### Custom Datasets

To create custom data loading, subclass `MVecDataset`:

```python
class MyCustomDataset(MVecDataset):
    def _load_data(self, csv_path):
        # Custom data loading logic
        pass
```

### Learning Rate Schedules

Current: Cosine annealing

To change, edit `train_flexible.py`:

```python
# Current
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.1
)

# Alternative: Step decay
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.5
)

# Alternative: Reduce on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
```

---

## Next Steps

After training:

1. **Test your model**: See [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)
2. **Understand results**: Review [ARCHITECTURE.md](ARCHITECTURE.md)
3. **Iterate**: Adjust configuration and retrain

---

**Remember**: This is proof-of-principle code. Expect to experiment, iterate, and learn. The goal is understanding the approach, not achieving state-of-the-art results.
