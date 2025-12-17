# Inference Guide

This guide explains how to run trained Takens-Based Transformer models for inference.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Standard Q&A Inference](#standard-qa-inference)
3. [QBA Two-Phase Generation](#qba-two-phase-generation)
4. [Inference with Vocabulary Checking](#inference-with-vocabulary-checking)
5. [Generation Parameters](#generation-parameters)
6. [Understanding Outputs](#understanding-outputs)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

To run a trained model:

```python
# 1. Edit unified_run.py configuration
MODEL_PATH = 'path/to/your_model.pt'
VOCAB_PATH = 'path/to/your_vocab.json'
TEST_QUESTIONS = ["Your test questions here"]

# 2. Run
python unified_run.py
```

The script will:
1. Load the model and vocabulary
2. Run test questions
3. Enter interactive mode for live testing

---

## Standard Q&A Inference

### Using `unified_run.py`

This is the main inference script for standard question-answer models.

### Configuration

Edit the configuration section at the top:

```python
# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = 'models/my_model.pt'
VOCAB_PATH = 'models/my_vocab.json'
DEVICE = 'cpu'  # or 'cuda' if available

# Test questions
TEST_QUESTIONS = [
    "What is the capital of France?",
    "How hot is Mercury?",
    "What is Python?",
]

# Generation settings
MAX_TOKENS = 100        # Maximum tokens to generate
TEMPERATURE = 0.8       # Sampling temperature (0.1-2.0)
TOP_K = 50              # Top-k sampling (0 = disabled)
```

### Running Test Questions

```bash
python unified_run.py
```

Output:

```
══════════════════════════════════════════════════════════════════════
Marina Simple Test
══════════════════════════════════════════════════════════════════════

Loading model...
Vocabulary: 5,431 words
Model loaded: 1,127,895 parameters

Training metrics:
  Word loss: 2.1203
  End loss: 0.2248

══════════════════════════════════════════════════════════════════════
TESTING
══════════════════════════════════════════════════════════════════════

══════════════════════════════════════════════════════════════════════
Question: What is the capital of France?
══════════════════════════════════════════════════════════════════════

Step   Word            End Prob  
────────────────────────────────────────
0      Paris           15.2%
1      is              8.3%
2      the             6.1%
3      capital         12.4%
4      of              9.7%
5      France          45.6%

Stopped after 6 tokens (end_prob=45.6%)

Answer: Paris is the capital of France
══════════════════════════════════════════════════════════════════════
```

### Interactive Mode

After test questions, the script enters interactive mode:

```
Interactive mode - type your question or 'quit' to exit:

You: What is machine learning?
Marina: Machine learning is a type of artificial intelligence that allows computers to learn from data

You: quit
Goodbye!
```

---

## Generation Process

### How Generation Works

```python
def generate_answer(question, verbose=True):
    # 1. Encode question as USER tokens
    q_words, q_identity, q_end = encoder.encode_sequence(
        question, identity="USER", is_last_in_turn=True
    )
    
    # 2. Create input tensors
    input_ids = torch.tensor([q_words])
    identity_ids = torch.tensor([q_identity])
    
    # 3. Generate loop
    for step in range(MAX_TOKENS):
        # Forward pass
        word_logits, end_logits, _, _ = model(input_ids, identity_ids)
        
        # Sample next word
        next_word = sample_from_logits(word_logits[-1])
        
        # Check end probability
        end_prob = sigmoid(end_logits[-1, 1])
        
        # Append to sequence with VISIBLE identity
        input_ids = append(input_ids, next_word)
        identity_ids = append(identity_ids, VISIBLE)
        
        # Stop if end probability > 0.5
        if end_prob > 0.5:
            break
    
    return decode(answer_tokens)
```

### Generation Parameters

#### Temperature

Controls randomness in sampling:

```python
TEMPERATURE = 0.1   # Very deterministic, focused
TEMPERATURE = 0.5   # Somewhat focused
TEMPERATURE = 0.8   # Balanced (recommended)
TEMPERATURE = 1.0   # More creative
TEMPERATURE = 1.5   # Very creative
TEMPERATURE = 2.0   # Very random
```

**Effect**:
- Lower = more predictable, repetitive
- Higher = more diverse, potentially incoherent

**When to adjust**:
- Factual Q&A: Use 0.5-0.8
- Creative generation: Use 1.0-1.5
- Exploration: Use 1.5-2.0

#### Top-K Sampling

Limits sampling to top K most probable tokens:

```python
TOP_K = 0      # Disabled (sample from all tokens)
TOP_K = 10     # Very focused
TOP_K = 50     # Balanced (recommended)
TOP_K = 100    # More diverse
```

**Effect**:
- Lower K = more focused, less diverse
- Higher K = more diverse, potentially off-topic

**Recommended**: 40-50 for most cases

#### Max Tokens

```python
MAX_TOKENS = 50     # Short answers
MAX_TOKENS = 100    # Medium answers (recommended)
MAX_TOKENS = 200    # Long answers
```

The model can stop early if end_prob > 0.5, so this is just an upper limit.

---

## QBA Two-Phase Generation

### Using `unified_run_qba.py`

For models trained with Question-Bridge-Answer mode.

### What is QBA?

QBA generates in **two phases**:

**Phase 1 - Bridge (INTERNAL)**:
- Hidden reasoning tokens
- Not shown to user
- Acts as geometric pathway through meaning-space

**Phase 2 - Answer (VISIBLE)**:
- Generated using bridge as context
- This is what the user sees
- Flows naturally from hidden reasoning

### Configuration

```python
# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = 'models/qba_model.pt'
VOCAB_PATH = 'models/qba_vocab.json'
DEVICE = 'cpu'

TEST_QUESTIONS = [
    "What is the capital of France?",
    "What is the boiling point of water?",
]

# Generation settings
MAX_BRIDGE_TOKENS = 20   # Bridge should be short
MAX_ANSWER_TOKENS = 100  # Answer can be longer
TEMPERATURE = 0.8
TOP_K = 50

# Display options
SHOW_BRIDGE = True  # Set False to hide bridge (production mode)
```

### Output Example

```
══════════════════════════════════════════════════════════════════════
Question: What is the capital of France?
══════════════════════════════════════════════════════════════════════

🌉 BRIDGE (INTERNAL - Hidden Reasoning):
Step   Word                 End Prob  
─────────────────────────────────────────────
0      geography            8.2%
1      France               12.4%
2      capital              35.7%
Bridge complete (3 tokens)

💬 ANSWER (VISIBLE - User Output):
Step   Word                 End Prob  
─────────────────────────────────────────────
0      Paris                18.3%
1      is                   9.1%
2      the                  7.2%
3      capital              15.4%
4      of                   11.8%
5      France               52.3%
Answer complete (6 tokens)

══════════════════════════════════════════════════════════════════════
🌉 Bridge (hidden): geography France capital
💬 Answer: Paris is the capital of France
══════════════════════════════════════════════════════════════════════
```

### Production Mode

In production, you typically hide the bridge:

```python
SHOW_BRIDGE = False
```

Output:

```
You: What is the capital of France?
Marina: Paris is the capital of France
```

The bridge is still generated internally, but not displayed.

### Inspecting Bridges

For research/debugging, you can examine bridges:

```python
def generate_answer_qba(question, verbose=True):
    # ... generation code ...
    
    return bridge, answer  # Return both

bridge, answer = generate_answer_qba("What causes rain?")
print(f"Bridge: {bridge}")      # "water cycle evaporation condensation"
print(f"Answer: {answer}")       # "Rain forms when water vapor condenses..."
```

This lets you:
- Understand model's reasoning pathway
- Compare bridges for similar questions
- Debug generation issues
- Visualize semantic trajectories

---

## Inference with Vocabulary Checking

### Using `unified_run_vocab_check.py`

This script adds vocabulary safety checks before inference.

### Why Vocabulary Checking?

**Problem**: If a question contains out-of-vocabulary (OOV) words, they become `<unk>` tokens, leading to:
- Degraded outputs
- Confusing behavior
- Poor answers

**Solution**: Check questions before inference, warn user, suggest alternatives.

### How It Works

```python
from vocab_checker import VocabChecker

checker = VocabChecker('vocab.json')

# Check question
is_safe = checker.is_safe_for_inference(question)

if not is_safe:
    _, oov_words, stats = checker.check_text(question)
    print(f"Warning: Question contains OOV words: {oov_words}")
    print(f"Coverage: {stats['coverage_pct']:.1f}%")
    
    # Suggest alternatives
    for word in oov_words:
        suggestions = checker.suggest_alternatives(word)
        print(f"  '{word}' → {suggestions}")
    
    # Ask user
    proceed = input("Proceed anyway? (y/n): ")
```

### Output Example

```
══════════════════════════════════════════════════════════════════════
PRE-FLIGHT VOCABULARY CHECK
══════════════════════════════════════════════════════════════════════

✓ Question 1: Where is Lady Serendipity?
   Coverage: 85.7% | In vocab: 6 | OOV: 1

   OOV words: Serendipity

   Suggestions:
     'Serendipity' → serendipitous, serendipity's, serendipitously

✓ Question 2: How do I find Kaevin the Listener?
   Coverage: 100.0% | In vocab: 7 | OOV: 0

══════════════════════════════════════════════════════════════════════
TESTING WITH VOCAB SAFETY CHECKS
══════════════════════════════════════════════════════════════════════

⚠ WARNING: Question 1 contains OOV words!
   OOV words: Serendipity
   This will produce <unk> tokens in the output.

   Suggestions:
     'Serendipity' → serendipitous, serendipity's

   Skipping this question due to OOV words.
```

### Standalone Vocabulary Checker

You can also use `vocab_checker.py` directly:

```bash
python vocab_checker.py
```

Edit the configuration section to check your own questions:

```python
VOCAB_PATH = 'models/my_vocab.json'

TEST_QUESTIONS = [
    "Your question 1",
    "Your question 2",
]
```

Output:

```
══════════════════════════════════════════════════════════════════════
VOCABULARY CHECK
══════════════════════════════════════════════════════════════════════

Text: Where is Lady Serendipity?

Tokens: ['Where', 'is', 'Lady', 'Serendipity', '?']

Statistics:
  Total tokens: 5
  In vocabulary: 4
  Out of vocabulary: 1
  Coverage: 80.0%

⚠ WARNING: 1 unique OOV words found!

OOV Words:
  • 'Serendipity' (appears 1x)
    Suggestions: serendipitous, serendipity's, serendipitously

══════════════════════════════════════════════════════════════════════
```

---

## Understanding Outputs

### Reading Step-by-Step Generation

```
Step   Word            End Prob  
────────────────────────────────────────
0      Paris           15.2%
1      is              8.3%
2      the             6.1%
3      capital         12.4%
4      of              9.7%
5      France          45.6%
```

**Columns**:
- **Step**: Generation step number
- **Word**: Token generated at this step
- **End Prob**: Probability that this should be the last token

**Interpretation**:
- End prob starts low (~5-15%)
- Gradually increases as answer completes
- Model stops when end_prob > 50%
- If end_prob stays low, hits MAX_TOKENS limit

### Normal vs. Abnormal Patterns

**Normal pattern**:
```
Step 0: word1  (end_prob=10%)
Step 1: word2  (end_prob=12%)
Step 2: word3  (end_prob=15%)
...
Step N: wordN  (end_prob=55%) → STOPS
```

**Abnormal pattern 1** - Never stops:
```
Step 0-99: words...  (end_prob always <20%)
Hit MAX_TOKENS limit
```
**Fix**: Increase `END_LOSS_WEIGHT` in training

**Abnormal pattern 2** - Stops too early:
```
Step 0: word1  (end_prob=65%) → STOPS immediately
```
**Fix**: Decrease `END_LOSS_WEIGHT` in training

**Abnormal pattern 3** - Repetition:
```
Step 0-10: "the the the the the..."
```
**Fix**: Increase temperature or use different top_k

---

## Advanced Inference Techniques

### Batch Inference

To process multiple questions efficiently:

```python
questions = ["Q1", "Q2", "Q3"]

# Encode all questions
encoded = [encoder.encode_sequence(q, identity="USER") 
           for q in questions]

# Pad to same length
input_ids = pad_sequence([torch.tensor(e[0]) for e in encoded])
identity_ids = pad_sequence([torch.tensor(e[1]) for e in encoded])

# Single forward pass for all questions
with torch.no_grad():
    word_logits, end_logits, _, _ = model(input_ids, identity_ids)

# Generate answers in parallel
# (requires modifying generate() method for batch support)
```

### Controlling Generation Style

#### More Deterministic (Factual)

```python
TEMPERATURE = 0.3
TOP_K = 10
```

Use for: Factual Q&A, lookups, precise answers

#### More Creative (Open-ended)

```python
TEMPERATURE = 1.2
TOP_K = 100
```

Use for: Story generation, creative tasks, brainstorming

#### Balanced (General Use)

```python
TEMPERATURE = 0.8
TOP_K = 50
```

Use for: Most applications

### Manual Stopping

Override automatic stopping:

```python
def generate_answer(question, max_tokens=100, stop_on_end=False):
    # ... generation code ...
    
    # Generate exactly max_tokens regardless of end_prob
    for step in range(max_tokens):
        # ...
        if stop_on_end and end_prob > 0.5:
            break
```

Set `stop_on_end=False` to always generate full sequence.

---

## Troubleshooting

### Problem: Poor Quality Outputs

**Symptoms**: Incoherent, nonsensical, or irrelevant answers.

**Possible causes**:
1. Model undertrained
2. Question OOV words
3. Wrong temperature/top_k
4. Model-data mismatch

**Solutions**:
```python
# Check vocabulary coverage
checker = VocabChecker(VOCAB_PATH)
checker.check_text(question)

# Adjust generation parameters
TEMPERATURE = 0.5  # More focused
TOP_K = 20         # More constrained

# Check training metrics
checkpoint = torch.load(MODEL_PATH)
print(checkpoint['val_metrics'])
# If word_loss > 3.0, model needs more training
```

---

### Problem: Repetitive Outputs

**Symptoms**: "the the the the" or same phrase repeated.

**Solutions**:
```python
# Increase temperature
TEMPERATURE = 1.0  # More diverse

# Increase top_k
TOP_K = 100

# Or add repetition penalty (requires code modification)
```

---

### Problem: Never Stops Generating

**Symptoms**: Always generates MAX_TOKENS, never early stops.

**Solutions**:
```python
# Check end probabilities in verbose output
# If end_prob always <20%, model wasn't trained for stopping

# If end_prob reasonable but still doesn't stop:
# Lower the stopping threshold
if end_prob > 0.3:  # Instead of 0.5
    break

# Or retrain with higher END_LOSS_WEIGHT
```

---

### Problem: Stops Too Early

**Symptoms**: Generates 1-2 tokens then stops.

**Solutions**:
```python
# Raise stopping threshold
if end_prob > 0.7:  # Instead of 0.5
    break

# Or retrain with lower END_LOSS_WEIGHT
```

---

### Problem: `<unk>` Tokens in Output

**Symptoms**: Output contains "<unk>" or unexpected tokens.

**Causes**:
- Question has OOV words
- Model vocabulary too small
- Encoding/decoding mismatch

**Solutions**:
```python
# Check vocabulary
checker = VocabChecker(VOCAB_PATH)
is_safe = checker.is_safe_for_inference(question)

# Add missing words to vocabulary
from vocab_updater import VocabUpdater
updater = VocabUpdater(VOCAB_PATH)
updater.add_words(['missing', 'words'])
updater.save('vocab_updated.json')

# ⚠️ Then retrain model with updated vocabulary
```

---

### Problem: Slow Inference

**Solutions**:
- Use GPU: `DEVICE = 'cuda'`
- Reduce MAX_TOKENS
- Reduce model size (requires retraining)
- Use smaller vocabulary (requires retraining)
- Batch multiple questions together

---

### Problem: Out of Memory During Inference

**Solutions**:
```python
# Reduce sequence length
MAX_TOKENS = 50

# Move to CPU
DEVICE = 'cpu'

# Clear cache between questions
import torch
torch.cuda.empty_cache()
```

---

## Programmatic Usage

### Using in Your Own Code

```python
import torch
from mvec_encoder import MVecEncoder
from mvec_model import MVecLanguageModel
from tbt_architecture import create_exponential_delays

# Load model
encoder = MVecEncoder()
encoder.load_vocab('vocab.json')

checkpoint = torch.load('model.pt')
config = checkpoint['model_config']

model = MVecLanguageModel(
    vocab_size=config['vocab_size'],
    embed_dim=config['embed_dim'],
    hidden_dim=config['hidden_dim'],
    num_layers=config['num_layers'],
    max_seq_len=512,
    delays=create_exponential_delays(128),
    dropout=0.1,
    use_identity_embed=config['use_identity_embed'],
    identity_embed_dim=config['identity_embed_dim'],
    tie_weights=False
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate answer
def ask(question):
    q_words, q_identity, _ = encoder.encode_sequence(
        question, identity="USER", is_last_in_turn=True
    )
    
    input_ids = torch.tensor([q_words])
    identity_ids = torch.tensor([q_identity])
    
    generated_ids, generated_identities = model.generate(
        input_ids,
        identity_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50,
        stop_on_end=True,
        generate_identity=2
    )
    
    # Decode only the answer part (skip question)
    answer_ids = generated_ids[0, len(q_words):].tolist()
    answer = encoder.decode_sequence(answer_ids)
    
    return answer

# Use it
answer = ask("What is machine learning?")
print(answer)
```

---

## Next Steps

After successful inference:

1. **Experiment with parameters**: Try different temperatures, top_k values
2. **Test edge cases**: Complex questions, long questions, ambiguous questions
3. **Evaluate systematically**: Create test sets, measure quality
4. **Iterate training**: Based on inference results, adjust training

---

**Remember**: These are small proof-of-principle models. Don't expect ChatGPT-level performance. The goal is demonstrating the geometric approach, not achieving state-of-the-art results.
