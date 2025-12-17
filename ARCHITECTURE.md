# Architecture Guide

This document explains the technical architecture of the Takens-Based Transformer and how the code files work together.

---

## Table of Contents

1. [Conceptual Overview](#conceptual-overview)
2. [Core Architecture](#core-architecture)
3. [MARINA Extension](#marina-extension)
4. [Data Flow](#data-flow)
5. [File-by-File Breakdown](#file-by-file-breakdown)
6. [Key Design Decisions](#key-design-decisions)

---

## Conceptual Overview

### From Attention to Phase Space

Standard transformers use **attention mechanisms** to model relationships between tokens:

```
Attention: Query × Key → Weights → Weighted sum of Values
```

The Takens-Based Transformer replaces this with **explicit phase-space reconstruction**:

```
Takens: [x(t), x(t-τ₁), x(t-τ₂), ..., x(t-τₙ)] → Feed-forward layers
```

### Why This Works

**Takens' Theorem** (1981) shows that a dynamical system's behavior can be reconstructed from time-delayed observations of a single variable. Applied to language:

- Language sequences are observations from an underlying "semantic dynamical system"
- Token embeddings at different delays reconstruct the semantic state space
- Feed-forward layers operate on this reconstructed space

**Result**: We get sequence modeling without explicit attention, using geometry instead.

---

## Core Architecture

The architecture has three main layers:

### Layer 1: Takens Embedding (`takens_embedding.py`)

Converts a sequence of token embeddings into delay-coordinate embeddings.

**Input**: `[batch, seq_len, embed_dim]`  
**Output**: `[batch, seq_len, (num_delays+1) × embed_dim]`

**Key components**:
- `TakensEmbedding` - Basic delay-coordinate construction
- `AdaptiveTakensEmbedding` - Learnable projection of delays
- Exponential delay spacing: [1, 2, 4, 8, 16, 32, 64, 128]

### Layer 2: Feed-Forward Processing (`tbt_architecture.py`)

Processes the reconstructed phase space through feed-forward layers.

**Key components**:
- `TBTLayer` - Layer norm + feed-forward + residual connection
- `TakensTransformer` - Stack of TBT layers
- **No attention mechanism** - pure feed-forward processing

### Layer 3: Output Projection

Projects hidden states back to vocabulary space.

```python
hidden_state → Linear → vocab_size logits
```

---

## MARINA Extension

MARINA (Multi-channel Architecture for Reasoning and INteraction with Awareness) extends the base TBT with **identity-aware channels**.

### The Three Identity Channels

```
0: USER     - Input from user (questions, prompts)
1: INTERNAL - Hidden reasoning (not shown to user)
2: VISIBLE  - Output to user (answers, responses)
```

### How Identity Works

1. **During Training**:
   - Each token has both a word ID and an identity ID
   - The model learns different patterns for each identity
   - INTERNAL tokens act as "geometric bridges" through meaning-space

2. **During Inference**:
   - USER tokens encode the question
   - Model generates INTERNAL tokens (if in QBA mode)
   - Model generates VISIBLE tokens as the answer

### Architecture Changes

```python
# Base TBT
input = token_embedding(word_ids)

# MARINA
input = concat([
    token_embedding(word_ids),
    identity_embedding(identity_ids)
])
```

The model also has **dual output heads**:

```python
word_logits = word_head(hidden)    # Predict next word
end_logits = end_head(hidden)      # Predict if sequence should end
```

This allows the model to learn natural stopping points for each identity.

---

## Data Flow

### Training Flow

```
1. CSV Data
   ↓
2. MVecEncoder.encode()
   → (word_ids, identity_ids, end_ids)
   ↓
3. MVecDataset.__getitem__()
   → (input_ids, target_ids, identity_ids, end_ids)
   ↓
4. DataLoader (with collate_fn)
   → Batched tensors
   ↓
5. MVecLanguageModel.forward()
   → (word_logits, end_logits, word_loss, end_loss)
   ↓
6. MVecTrainer.train_epoch()
   → Backprop, optimizer step
   ↓
7. Save checkpoint
```

### Inference Flow

```
1. Load trained model + vocabulary
   ↓
2. Encode user question
   → (word_ids, identity_ids) with USER identity
   ↓
3. MVecLanguageModel.generate()
   ↓
   For each step:
     - Forward pass
     - Sample next word from logits
     - Check end_logits for stopping
     - Append to sequence with VISIBLE identity
   ↓
4. Decode token_ids back to text
   ↓
5. Return answer
```

---

## File-by-File Breakdown

### Core Architecture Files

#### `takens_embedding.py`

**Purpose**: Implements delay-coordinate embedding based on Takens' theorem.

**Key classes**:
- `TakensEmbedding` - Basic implementation
  - Creates grid: `[batch, seq_len, num_delays+1, embed_dim]`
  - Fills with delayed versions of input
  - Handles padding for positions near sequence start
  
- `AdaptiveTakensEmbedding` - Learnable variant
  - Wraps `TakensEmbedding`
  - Adds learnable projection layer
  - Compresses full Takens dimension to desired output_dim

**Key functions**:
- `create_exponential_delays(max_delay, base=2)` - Generate [1, 2, 4, 8, ...]
- `create_logarithmic_delays(max_delay, num_delays)` - Generate log-spaced delays

**Usage**:
```python
takens = TakensEmbedding(embedding_dim=256, delays=[1, 2, 4, 8])
grid = takens(x)  # [B, L, 5, 256]
flat = takens.flatten_grid(grid)  # [B, L, 1280]
```

---

#### `tbt_architecture.py`

**Purpose**: Core transformer architecture without attention.

**Key classes**:

1. `TBTFeedForward`
   - Standard FFN: Linear → Activation → Dropout → Linear
   - Default: 4x expansion (dim → 4×dim → dim)

2. `TBTLayer`
   - Pre-norm architecture: LayerNorm → FFN → Residual
   - No attention sublayer

3. `TakensTransformer`
   - Stack of TBT layers
   - Input → Takens embedding → Layers → Final norm
   - Can use adaptive or standard Takens embedding

4. `TBTLanguageModel`
   - Complete language model
   - Token embedding (+ optional positional embedding)
   - TakensTransformer
   - Output projection to vocabulary
   - Optional weight tying
   - Includes `.generate()` method

**Configuration options**:
- `use_adaptive_takens`: Use learnable projection (recommended)
- `use_positional`: Add positional embeddings (optional, can disable)
- `tie_weights`: Tie input/output embeddings
- `delays`: Delay structure (default: exponential up to 128)

---

### MARINA Multi-Channel Files

#### `mvec_encoder.py`

**Purpose**: Encodes text with multi-channel identity information.

**Key class**: `MVecEncoder`

**Three encoding modes**:

1. **QA Mode** (Question → Answer)
   ```python
   encode(EncodingMode.QA, question="...", answer="...")
   → Sets identity: USER for question, VISIBLE for answer
   → Adds END marker after answer
   ```

2. **QBA Mode** (Question → Bridge → Answer)
   ```python
   encode(EncodingMode.QBA, question="...", bridge="...", answer="...")
   → USER for question
   → INTERNAL for bridge (hidden reasoning)
   → VISIBLE for answer
   → END markers after bridge and answer
   ```

3. **CONTINUOUS Mode** (Plain text)
   ```python
   encode(EncodingMode.CONTINUOUS, text="...")
   → All tokens marked VISIBLE
   → Optional chunking with END markers every N tokens
   ```

**Vocabulary management**:
```python
encoder = MVecEncoder()
encoder.build_vocab(texts, min_freq=1)  # Build from corpus
encoder.save_vocab("vocab.json")         # Save
encoder.load_vocab("vocab.json")         # Load
```

**Special tokens**:
- `<pad>` (ID: 1) - Padding
- `<unk>` (ID: 0) - Unknown words

---

#### `mvec_model.py`

**Purpose**: Language model with identity-aware architecture.

**Key class**: `MVecLanguageModel`

**Architecture**:
```python
input_ids, identity_ids → 
  token_embedding(input_ids) +
  identity_embedding(identity_ids) →
  concat → 
  TakensTransformer →
  word_head → word_logits
  end_head → end_logits
```

**Dual outputs**:
1. **Word predictions**: `[batch, seq_len, vocab_size]`
2. **End predictions**: `[batch, seq_len, 2]` (NO/YES)

**Generation method**:
```python
model.generate(
    input_ids,           # Input tokens
    identity_ids,        # Input identities
    max_new_tokens=100,  # Max length
    temperature=0.8,     # Sampling temperature
    top_k=50,            # Top-k filtering
    stop_on_end=True,    # Stop when end=YES
    generate_identity=2  # Identity for generated tokens
)
```

**Key parameters**:
- `use_identity_embed`: Enable identity channels (True for MARINA)
- `identity_embed_dim`: Dimension of identity embeddings (typically 16-32)
- `tie_weights`: Whether to tie input/output word embeddings

---

#### `mvec_dataset.py`

**Purpose**: Dataset class for multi-channel training.

**Key class**: `MVecDataset`

**Supports three modes**:
- QA: Requires `question`, `answer` columns
- QBA: Requires `question`, `bridge`, `answer` columns
- CONTINUOUS: Requires `text` or `content` column

**Returns**:
```python
(input_ids, target_ids, identity_ids, end_ids)
where:
  input_ids = word_ids[:-1]   # Current tokens
  target_ids = word_ids[1:]   # Next tokens (shifted)
  identity_ids = identities[:-1]
  end_ids = end_markers[1:]
```

**Collation function**:
```python
collate_mvec_batch(batch)
→ Pads all sequences to same length within batch
→ Returns batched tensors
```

---

#### `mvec_training.py`

**Purpose**: Training utilities for MARINA.

**Key class**: `MVecTrainer`

**Features**:
- Weighted loss: `total_loss = word_weight × word_loss + end_weight × end_loss`
- Gradient clipping
- Learning rate scheduling
- Validation monitoring
- Early stopping
- Checkpoint saving

**Usage**:
```python
trainer = MVecTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device='cuda',
    word_loss_weight=1.0,
    end_loss_weight=1.0
)

history = trainer.train(
    num_epochs=50,
    save_path='model.pt',
    scheduler=scheduler,
    early_stopping_patience=10
)
```

---

### Flexible Training Files

#### `flexible_dataset.py`

**Purpose**: Dataset with multiple chunking strategies.

**Chunking modes**:

1. **pairs** - Each Q&A pair is isolated
   - Fragments phase space
   - Good for independent Q&A

2. **conversation** - Chain N Q&A pairs together
   - Creates conversational flow
   - Better manifold structure

3. **sliding** - Overlapping windows with stride
   - Maximum connectivity
   - Good for continuous text

4. **paragraph** - Split on paragraph boundaries
   - Natural text structure

5. **document** - Full documents
   - Preserve complete context

**Key parameters**:
```python
FlexibleMVecDataset(
    data_path='data.csv',
    encoder=encoder,
    mode='conversation',       # Chunking mode
    max_seq_len=256,
    turns_per_conversation=3,  # For conversation mode
    stride=128,                # For sliding mode
    min_seq_len=10            # Filter short sequences
)
```

---

#### `train_flexible.py` / `flexible_training.py`

**Purpose**: Main training script with configuration.

**How to use**:
1. Edit CONFIGURATION section at top of file
2. Run: `python train_flexible.py`
3. No command-line arguments needed

**Key configuration**:
```python
DATA_PATH = 'your_data.csv'
CHUNKING_MODE = 'pairs'  # or 'conversation', 'sliding', etc.
MAX_SEQ_LEN = 256
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 3e-4
```

---

### Inference Files

#### `unified_run.py`

**Purpose**: Standard Q&A inference.

**How to use**:
1. Edit configuration section (MODEL_PATH, VOCAB_PATH)
2. Edit TEST_QUESTIONS list
3. Run: `python unified_run.py`
4. Optional: Interactive mode after tests

**Features**:
- Loads trained model and vocabulary
- Generates answers with temperature and top-k sampling
- Shows generation step-by-step (word + end probability)
- Interactive mode for live testing

---

#### `unified_run_qba.py`

**Purpose**: Two-phase Question-Bridge-Answer inference.

**How it works**:

**Phase 1 - Bridge (INTERNAL)**:
```
Question → Generate bridge tokens (identity=1)
Bridge acts as hidden geometric pathway
```

**Phase 2 - Answer (VISIBLE)**:
```
Question + Bridge → Generate answer tokens (identity=2)
Answer flows from bridge context
```

**Configuration**:
```python
MAX_BRIDGE_TOKENS = 20   # Bridge should be short
MAX_ANSWER_TOKENS = 100  # Answer can be longer
SHOW_BRIDGE = True       # Set False for production
```

The bridge represents the model's "reasoning" - the geometric trajectory through semantic space before producing the visible answer.

---

#### `unified_run_vocab_check.py`

**Purpose**: Inference with vocabulary safety checking.

**Features**:
- Checks questions for out-of-vocabulary (OOV) words before inference
- Suggests alternatives for OOV words
- Can skip or warn on OOV questions
- Interactive mode with safety checks

**Why this matters**:
- OOV words become `<unk>` tokens
- Model may produce degraded outputs
- Early detection prevents confusion

---

### Utility Files

#### `training_utils.py`

**Purpose**: Generic training utilities (used by base TBT).

**Key components**:

1. **TextDataset** - Character/word level text dataset
2. **TimeSeriesDataset** - For sequence prediction tasks
3. **Trainer** - Generic training loop
4. **Data generators**:
   - `generate_lorenz_attractor()` - Classic chaotic system
   - `normalize_timeseries()` - Normalize data

**Metrics**:
- `compute_perplexity(loss)`
- `compute_mse(predictions, targets)`
- `compute_mae(predictions, targets)`

---

#### `vocab_checker.py`

**Purpose**: Check text for out-of-vocabulary words.

**Key class**: `VocabChecker`

**Usage**:
```python
checker = VocabChecker('vocab.json')

# Check single text
in_vocab, oov, stats = checker.check_text("Your question here")

# Check safety
if checker.is_safe_for_inference(question):
    answer = generate_answer(question)
else:
    print("Warning: OOV words detected")

# Get suggestions
suggestions = checker.suggest_alternatives('unknownword')
```

**Methods**:
- `check_text(text)` - Returns in-vocab, OOV tokens, and stats
- `check_questions(questions)` - Batch check multiple questions
- `is_safe_for_inference(text)` - Boolean safety check
- `suggest_alternatives(word)` - Fuzzy matching suggestions

---

#### `vocab_updater.py`

**Purpose**: Add new words to existing vocabulary.

**Usage**:
```python
updater = VocabUpdater('vocab.json')
updater.add_words(['newword1', 'newword2'])
updater.save('vocab_updated.json')
```

**⚠️ Warning**: Only use for small additions! For major changes, rebuild vocabulary from scratch.

**Why**: Model checkpoint still expects old vocab_size. After updating vocab, you must either:
1. Retrain from scratch with new vocab
2. Restore original vocab file

---

## Key Design Decisions

### 1. Why Exponential Delays?

Exponential spacing [1, 2, 4, 8, 16, ...] provides:
- Fine resolution at short timescales
- Coarse resolution at long timescales
- Efficient coverage of wide temporal range
- Matches natural language structure (word → phrase → sentence → paragraph)

### 2. Why Adaptive Takens Embedding?

`AdaptiveTakensEmbedding` adds a learnable projection layer:
- Allows model to weight delays differently
- Can compress high-dimensional Takens space
- Improves training stability
- Recommended for all applications

### 3. Why Identity Channels?

The USER/INTERNAL/VISIBLE distinction:
- Mirrors human communication (thought vs. speech)
- Allows explicit hidden reasoning (QBA bridge)
- Enables different generation patterns per identity
- Improves geometric separation in phase space

### 4. Why Dual Output Heads?

Word + End predictions:
- Model learns natural stopping points
- Each identity can have different end patterns
- Prevents over-generation or under-generation
- More expressive than fixed-length sequences

### 5. Why Small Models?

1-2M parameters vs. billions because:
- Proof-of-principle focus
- Interpretability matters
- CPU training viable
- Geometric effects visible
- Domain-specific applications

### 6. Why No Positional Embeddings (Optional)?

Positional embeddings are **optional** in TBT:
- Takens delays already encode temporal structure
- Position information emerges from delay patterns
- Can be added if needed (use `use_positional=True`)
- Most experiments work without them

---

## Phase Space Interpretation

### What Does "Geometric" Mean Here?

The core insight: **Language has geometry**.

When we embed tokens and apply Takens delays, we're reconstructing a **manifold** - a geometric space where:

- Each point represents a semantic state
- Trajectories represent meaning flow
- Distance measures semantic similarity
- Basins represent different content domains

### Visualizing the Architecture

```
Text sequence:
"What is the capital of France?"

↓ Token embeddings

[what_vec, is_vec, the_vec, capital_vec, of_vec, France_vec, ?_vec]

↓ Takens embedding at position t=6 (France)

[
  France_vec,        # τ=0  (current)
  of_vec,            # τ=1  (1 step back)
  capital_vec,       # τ=2  (2 steps back)
  is_vec,            # τ=4  (4 steps back)
  what_vec,          # τ=8  (8 steps back)
  <pad>,             # τ=16 (before sequence)
  ...
]

↓ Flatten & project

Reconstructed state vector (256-dim)

↓ Feed-forward layers

Process geometric relationships

↓ Output heads

word_logits: "Paris" has high probability
end_logits: Not end of sequence yet
```

### Basin Structure

The training data creates **basins** in phase space:

```
Question Basin → Bridge Basin → Answer Basin
(USER identity) → (INTERNAL) → (VISIBLE identity)
```

Well-trained models show:
- Clear basin separation
- Smooth transitions (bridges)
- Stable attractors (consistent answers)

This is why chunking strategy matters - it shapes the geometric landscape.

---

## Next Steps

Now that you understand the architecture:

1. **For Training**: Read [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
2. **For Inference**: Read [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)
3. **For Details**: See [CODE_REFERENCE.md](CODE_REFERENCE.md)

---

*Remember: This architecture isn't trying to be "better" than attention - it's demonstrating that an entirely different structural foundation is viable.*
