# Code Reference

Comprehensive technical reference for all code files in the Takens-Based Transformer repository.

---

## Table of Contents

1. [Core Architecture](#core-architecture)
2. [MARINA Extension](#marina-extension)
3. [Training System](#training-system)
4. [Inference Scripts](#inference-scripts)
5. [Utilities](#utilities)

---

## Core Architecture

### `takens_embedding.py`

Implements delay-coordinate embeddings based on Takens' theorem.

#### Class: `TakensEmbedding`

Creates delay-coordinate embeddings from token sequences.

**Constructor**:
```python
TakensEmbedding(
    embedding_dim: int,
    delays: Optional[List[int]] = None,
    pad_value: float = 0.0
)
```

**Parameters**:
- `embedding_dim`: Dimension of input embeddings
- `delays`: List of delay positions (default: [1, 2, 4, 8, 16, 32, 64, 128])
- `pad_value`: Value for positions before sequence start

**Methods**:

**`.forward(x)`**:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: [batch, seq_len, embedding_dim]
    Returns:
        grid: [batch, seq_len, num_delays+1, embedding_dim]
    """
```

Creates Takens grid where each position contains:
- Current embedding: x(t)
- Delayed embeddings: x(t-τ₁), x(t-τ₂), ..., x(t-τₙ)

**`.flatten_grid(grid)`**:
```python
def flatten_grid(self, grid: torch.Tensor) -> torch.Tensor:
    """
    Args:
        grid: [batch, seq_len, num_delays+1, embed_dim]
    Returns:
        [batch, seq_len, (num_delays+1)*embed_dim]
    """
```

**`.get_output_dim()`**:
```python
def get_output_dim(self) -> int:
    """Returns (num_delays + 1) * embedding_dim"""
```

**Example**:
```python
takens = TakensEmbedding(
    embedding_dim=256,
    delays=[1, 2, 4, 8, 16, 32, 64, 128]
)

x = torch.randn(4, 100, 256)  # [batch, seq, dim]
grid = takens(x)               # [4, 100, 9, 256]
flat = takens.flatten_grid(grid)  # [4, 100, 2304]
```

---

#### Class: `AdaptiveTakensEmbedding`

Learnable variant with projection layer.

**Constructor**:
```python
AdaptiveTakensEmbedding(
    embedding_dim: int,
    delays: Optional[List[int]] = None,
    output_dim: Optional[int] = None,
    dropout: float = 0.1
)
```

**Parameters**:
- `embedding_dim`: Dimension of input embeddings
- `delays`: Delay positions (default: [1, 2, 4, 8, 16, 32, 64, 128])
- `output_dim`: Dimension after projection (default: full Takens dim)
- `dropout`: Dropout probability

**Architecture**:
```
input → TakensEmbedding → flatten → Linear → LayerNorm → Dropout → output
```

**Methods**:

**`.forward(x)`**:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: [batch, seq_len, embedding_dim]
    Returns:
        [batch, seq_len, output_dim]
    """
```

**Example**:
```python
adaptive = AdaptiveTakensEmbedding(
    embedding_dim=256,
    delays=[1, 2, 4, 8, 16, 32, 64],
    output_dim=512,
    dropout=0.1
)

x = torch.randn(4, 100, 256)
out = adaptive(x)  # [4, 100, 512]
```

---

#### Helper Functions

**`create_exponential_delays(max_delay, base=2)`**:
```python
def create_exponential_delays(max_delay: int, base: int = 2) -> List[int]:
    """
    Creates [1, 2, 4, 8, 16, ..., max_delay]
    
    Args:
        max_delay: Maximum delay value
        base: Exponential base (default 2)
    
    Returns:
        List of integer delays
    """
```

**`create_logarithmic_delays(max_delay, num_delays)`**:
```python
def create_logarithmic_delays(max_delay: int, num_delays: int) -> List[int]:
    """
    Creates logarithmically-spaced delays.
    
    Args:
        max_delay: Maximum delay value
        num_delays: Number of delays to create
    
    Returns:
        List of integer delays
    """
```

---

### `tbt_architecture.py`

Core transformer architecture without attention.

#### Class: `TBTFeedForward`

Standard feed-forward network.

**Constructor**:
```python
TBTFeedForward(
    dim: int,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.1,
    activation: str = 'gelu'
)
```

**Parameters**:
- `dim`: Input/output dimension
- `hidden_dim`: Hidden dimension (default: 4 * dim)
- `dropout`: Dropout probability
- `activation`: 'gelu' or 'relu'

**Architecture**:
```
input → Linear(dim, hidden_dim) → GELU → Dropout → 
Linear(hidden_dim, dim) → Dropout → output
```

---

#### Class: `TBTLayer`

Single TBT layer (norm + feedforward + residual).

**Constructor**:
```python
TBTLayer(
    dim: int,
    ff_hidden_dim: Optional[int] = None,
    dropout: float = 0.1
)
```

**Architecture**:
```
input → LayerNorm → FeedForward → add residual → output
```

Pre-norm architecture for stability.

---

#### Class: `TakensTransformer`

Stack of TBT layers with Takens embedding.

**Constructor**:
```python
TakensTransformer(
    input_dim: int,
    hidden_dim: int,
    num_layers: int = 6,
    delays: Optional[list] = None,
    ff_hidden_multiplier: int = 4,
    dropout: float = 0.1,
    use_adaptive_takens: bool = True
)
```

**Parameters**:
- `input_dim`: Input embedding dimension
- `hidden_dim`: Hidden dimension for layers
- `num_layers`: Number of TBT layers
- `delays`: Takens delay structure
- `ff_hidden_multiplier`: FFN expansion factor
- `dropout`: Dropout probability
- `use_adaptive_takens`: Use learnable projection (recommended)

**Architecture**:
```
input → AdaptiveTakensEmbedding → TBTLayer × N → LayerNorm → output
```

**Methods**:

**`.forward(x)`**:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: [batch, seq_len, input_dim]
    Returns:
        [batch, seq_len, hidden_dim]
    """
```

**Example**:
```python
tbt = TakensTransformer(
    input_dim=256,
    hidden_dim=256,
    num_layers=4,
    delays=create_exponential_delays(64),
    dropout=0.1,
    use_adaptive_takens=True
)

x = torch.randn(4, 100, 256)
hidden = tbt(x)  # [4, 100, 256]
```

---

#### Class: `TBTLanguageModel`

Complete language model for text generation.

**Constructor**:
```python
TBTLanguageModel(
    vocab_size: int,
    embed_dim: int = 512,
    hidden_dim: int = 512,
    num_layers: int = 6,
    max_seq_len: int = 8192,
    delays: Optional[list] = None,
    dropout: float = 0.1,
    tie_weights: bool = True,
    use_positional: bool = False
)
```

**Parameters**:
- `vocab_size`: Size of vocabulary
- `embed_dim`: Token embedding dimension
- `hidden_dim`: Hidden dimension
- `num_layers`: Number of TBT layers
- `max_seq_len`: Maximum sequence length
- `delays`: Takens delay structure
- `dropout`: Dropout probability
- `tie_weights`: Share input/output embeddings
- `use_positional`: Add positional embeddings (optional)

**Architecture**:
```
token_ids → TokenEmbedding (+PosEmbedding) → Dropout → 
TakensTransformer → Linear(vocab_size) → logits
```

**Methods**:

**`.forward(input_ids, labels=None)`**:
```python
def forward(
    self,
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Args:
        input_ids: [batch, seq_len]
        labels: [batch, seq_len] (optional, for loss)
    
    Returns:
        logits: [batch, seq_len, vocab_size]
        loss: Scalar (if labels provided)
    """
```

**`.generate(input_ids, max_new_tokens, temperature, top_k)`**:
```python
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None
) -> torch.Tensor:
    """
    Autoregressive generation.
    
    Args:
        input_ids: [1, seq_len] - Initial tokens
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering (None = disabled)
    
    Returns:
        [1, seq_len + generated_len]
    """
```

**Example**:
```python
model = TBTLanguageModel(
    vocab_size=5000,
    embed_dim=256,
    hidden_dim=256,
    num_layers=4,
    dropout=0.1,
    tie_weights=True,
    use_positional=False
)

# Training
input_ids = torch.randint(0, 5000, (8, 128))
labels = torch.randint(0, 5000, (8, 128))
logits, loss = model(input_ids, labels)

# Generation
prompt = torch.randint(0, 5000, (1, 10))
generated = model.generate(prompt, max_new_tokens=50, temperature=0.8)
```

---

## MARINA Extension

### `mvec_encoder.py`

Multi-channel encoding with identity awareness.

#### Enum: `EncodingMode`

```python
class EncodingMode(Enum):
    QA = "qa"              # Question → Answer
    QBA = "qba"            # Question → Bridge → Answer
    CONTINUOUS = "continuous"  # Plain text
```

---

#### Class: `MVecEncoder`

Encodes text with multi-channel identity information.

**Constructor**:
```python
MVecEncoder()
```

**Special Tokens**:
- `<unk>`: ID 0 - Unknown words
- `<pad>`: ID 1 - Padding

**Identity Values**:
- `USER = 0`: Input from user
- `INTERNAL = 1`: Hidden reasoning
- `VISIBLE = 2`: Output to user

**Methods**:

**`.build_vocab(texts, min_freq=1)`**:
```python
def build_vocab(
    self,
    texts: List[str],
    min_freq: int = 1
) -> None:
    """
    Build vocabulary from corpus.
    
    Args:
        texts: List of text strings
        min_freq: Minimum word frequency to include
    """
```

**`.save_vocab(path)` / `.load_vocab(path)`**:
```python
def save_vocab(self, path: str) -> None:
    """Save vocabulary to JSON"""

def load_vocab(self, path: str) -> None:
    """Load vocabulary from JSON"""
```

**`.encode(mode, **kwargs)`**:
```python
def encode(
    self,
    mode: EncodingMode,
    question: str = None,
    bridge: str = None,
    answer: str = None,
    text: str = None,
    chunk_size: int = None
) -> Tuple[List[int], List[int], List[int]]:
    """
    Encode text with identity channels.
    
    Returns:
        (word_ids, identity_ids, end_ids)
    """
```

**Encoding Modes**:

**QA Mode**:
```python
word_ids, identity_ids, end_ids = encoder.encode(
    EncodingMode.QA,
    question="What is Python?",
    answer="A programming language"
)

# Identity structure:
# [USER, USER, USER] + [VISIBLE, VISIBLE, VISIBLE, VISIBLE]
# [0,0,0,0] + [1] at end
```

**QBA Mode**:
```python
word_ids, identity_ids, end_ids = encoder.encode(
    EncodingMode.QBA,
    question="What is Python?",
    bridge="programming language definition",
    answer="Python is a programming language"
)

# Identity structure:
# [USER...] + [INTERNAL...] + [VISIBLE...]
# [0] after question, [1] after bridge, [1] at end
```

**CONTINUOUS Mode**:
```python
word_ids, identity_ids, end_ids = encoder.encode(
    EncodingMode.CONTINUOUS,
    text="Long text...",
    chunk_size=50  # END marker every 50 tokens
)

# Identity structure:
# [VISIBLE, VISIBLE, VISIBLE, ...]
# [1] every chunk_size tokens
```

**`.encode_sequence(text, identity, is_last_in_turn)`**:
```python
def encode_sequence(
    self,
    text: str,
    identity: str,
    is_last_in_turn: bool = False
) -> Tuple[List[int], List[int], List[int]]:
    """
    Encode single sequence with specified identity.
    
    Args:
        text: Text to encode
        identity: "USER", "INTERNAL", or "VISIBLE"
        is_last_in_turn: Add END marker at end
    
    Returns:
        (word_ids, identity_ids, end_ids)
    """
```

**`.decode_word(word_id)` / `.decode_sequence(word_ids)`**:
```python
def decode_word(self, word_id: int) -> str:
    """Convert single word ID to text"""

def decode_sequence(self, word_ids: List[int]) -> str:
    """Convert sequence of word IDs to text"""
```

**`.get_vocab_size()`**:
```python
def get_vocab_size(self) -> int:
    """Return vocabulary size"""
```

**Example**:
```python
encoder = MVecEncoder()

# Build vocabulary
texts = ["What is Python?", "A programming language"]
encoder.build_vocab(texts)
encoder.save_vocab("vocab.json")

# Encode QA pair
word_ids, identity_ids, end_ids = encoder.encode(
    EncodingMode.QA,
    question="What is Python?",
    answer="A programming language"
)

# Decode
text = encoder.decode_sequence(word_ids)
print(text)  # "What is Python? A programming language"
```

---

### `mvec_model.py`

Language model with identity-aware architecture.

#### Class: `MVecLanguageModel`

Extends TBT with identity channels and dual output heads.

**Constructor**:
```python
MVecLanguageModel(
    vocab_size: int,
    embed_dim: int = 256,
    hidden_dim: int = 256,
    num_layers: int = 4,
    max_seq_len: int = 512,
    delays: Optional[list] = None,
    dropout: float = 0.1,
    tie_weights: bool = True,
    use_identity_embed: bool = True,
    identity_embed_dim: int = 32
)
```

**Parameters**:
- `vocab_size`: Size of vocabulary
- `embed_dim`: Word embedding dimension
- `hidden_dim`: Hidden dimension
- `num_layers`: Number of TBT layers
- `max_seq_len`: Maximum sequence length
- `delays`: Takens delay structure
- `dropout`: Dropout probability
- `tie_weights`: Share input/output embeddings
- `use_identity_embed`: Enable identity channels
- `identity_embed_dim`: Identity embedding dimension

**Architecture**:
```
word_ids, identity_ids →
  token_embed(word_ids) [embed_dim] +
  identity_embed(identity_ids) [identity_embed_dim] →
  concat [embed_dim + identity_embed_dim] →
  TakensTransformer →
  hidden [hidden_dim] →
    ├─ word_head → word_logits [vocab_size]
    └─ end_head → end_logits [2]
```

**Methods**:

**`.forward(input_ids, identity_ids, word_labels, end_labels)`**:
```python
def forward(
    self,
    input_ids: torch.Tensor,
    identity_ids: Optional[torch.Tensor] = None,
    word_labels: Optional[torch.Tensor] = None,
    end_labels: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Forward pass with dual outputs.
    
    Args:
        input_ids: [batch, seq_len] - Word token IDs
        identity_ids: [batch, seq_len] - Identity channel
        word_labels: [batch, seq_len] - Target words (for loss)
        end_labels: [batch, seq_len] - Target end signals (for loss)
    
    Returns:
        word_logits: [batch, seq_len, vocab_size]
        end_logits: [batch, seq_len, 2]
        word_loss: Scalar (if word_labels provided)
        end_loss: Scalar (if end_labels provided)
    """
```

**`.generate(input_ids, identity_ids, ...)`**:
```python
def generate(
    self,
    input_ids: torch.Tensor,
    identity_ids: Optional[torch.Tensor] = None,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    stop_on_end: bool = True,
    generate_identity: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate with end signal detection.
    
    Args:
        input_ids: [1, seq_len] - Input tokens
        identity_ids: [1, seq_len] - Identity for input
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        stop_on_end: Stop when end=YES predicted
        generate_identity: Identity for generated tokens
    
    Returns:
        generated_ids: [1, total_len]
        generated_identities: [1, total_len]
    """
```

**Example**:
```python
model = MVecLanguageModel(
    vocab_size=5000,
    embed_dim=128,
    hidden_dim=128,
    num_layers=4,
    use_identity_embed=True,
    identity_embed_dim=32
)

# Training
input_ids = torch.randint(0, 5000, (8, 100))
identity_ids = torch.randint(0, 3, (8, 100))
word_labels = torch.randint(0, 5000, (8, 100))
end_labels = torch.randint(0, 2, (8, 100))

word_logits, end_logits, word_loss, end_loss = model(
    input_ids, identity_ids, word_labels, end_labels
)

# Generation
prompt_ids = torch.randint(0, 5000, (1, 10))
prompt_identity = torch.zeros(1, 10, dtype=torch.long)  # USER

generated_ids, generated_identity = model.generate(
    prompt_ids,
    prompt_identity,
    max_new_tokens=50,
    temperature=0.8,
    stop_on_end=True,
    generate_identity=2  # VISIBLE
)
```

---

### `mvec_dataset.py`

Dataset class for multi-channel training.

#### Class: `MVecDataset`

PyTorch Dataset for MARINA training.

**Constructor**:
```python
MVecDataset(
    csv_path: str,
    encoder: MVecEncoder,
    max_seq_len: int = 128,
    mode: EncodingMode = EncodingMode.QA,
    chunk_size: Optional[int] = None
)
```

**Parameters**:
- `csv_path`: Path to CSV data file
- `encoder`: MVecEncoder instance
- `max_seq_len`: Maximum sequence length
- `mode`: EncodingMode (QA, QBA, or CONTINUOUS)
- `chunk_size`: For CONTINUOUS, chunk size for END markers

**CSV Format by Mode**:

**QA Mode**: Requires `question`, `answer` columns  
**QBA Mode**: Requires `question`, `bridge`, `answer` columns  
**CONTINUOUS Mode**: Requires `text` or `content` column

**Methods**:

**`.__len__()`**:
```python
def __len__(self) -> int:
    """Return number of samples"""
```

**`.__getitem__(idx)`**:
```python
def __getitem__(
    self,
    idx: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get training sample.
    
    Returns:
        input_ids: [seq_len-1] - Current tokens
        target_ids: [seq_len-1] - Next tokens
        identity_ids: [seq_len-1] - Identity channel
        end_ids: [seq_len-1] - End markers
    """
```

**Example**:
```python
from mvec_encoder import MVecEncoder, EncodingMode
from mvec_dataset import MVecDataset

encoder = MVecEncoder()
encoder.build_vocab(texts)

dataset = MVecDataset(
    csv_path='qa_data.csv',
    encoder=encoder,
    max_seq_len=256,
    mode=EncodingMode.QA
)

# Get sample
input_ids, target_ids, identity_ids, end_ids = dataset[0]
```

---

#### Function: `collate_mvec_batch`

Collation function for DataLoader.

```python
def collate_mvec_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad sequences to same length within batch.
    
    Args:
        batch: List of (input_ids, target_ids, identity_ids, end_ids)
    
    Returns:
        Padded tensors with batch_first=True
    """
```

**Usage**:
```python
from torch.utils.data import DataLoader
from mvec_dataset import MVecDataset, collate_mvec_batch

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_mvec_batch
)
```

---

### `mvec_training.py`

Training utilities for MARINA models.

#### Class: `MVecTrainer`

Trainer with dual-loss optimization.

**Constructor**:
```python
MVecTrainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: optim.Optimizer,
    device: str = 'cuda',
    gradient_clip: float = 1.0,
    log_interval: int = 100,
    word_loss_weight: float = 1.0,
    end_loss_weight: float = 1.0
)
```

**Parameters**:
- `model`: MVecLanguageModel instance
- `train_loader`: Training DataLoader
- `val_loader`: Validation DataLoader
- `optimizer`: PyTorch optimizer
- `device`: 'cuda' or 'cpu'
- `gradient_clip`: Max gradient norm
- `log_interval`: Log every N batches
- `word_loss_weight`: Weight for word prediction loss
- `end_loss_weight`: Weight for end signal loss

**Loss Calculation**:
```python
total_loss = word_loss_weight * word_loss + end_loss_weight * end_loss
```

**Methods**:

**`.train(num_epochs, save_path, scheduler, early_stopping_patience)`**:
```python
def train(
    self,
    num_epochs: int,
    save_path: Optional[str] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    early_stopping_patience: Optional[int] = None
) -> Dict[str, List]:
    """
    Train for multiple epochs.
    
    Args:
        num_epochs: Number of training epochs
        save_path: Path to save best model
        scheduler: Learning rate scheduler
        early_stopping_patience: Stop after N epochs without improvement
    
    Returns:
        history: Dict with 'train' and 'val' metrics
    """
```

**`.train_epoch()` / `.validate()`**:
```python
def train_epoch(self) -> Dict[str, float]:
    """Train for one epoch, returns metrics"""

def validate(self) -> Dict[str, float]:
    """Validate on validation set, returns metrics"""
```

**Example**:
```python
import torch.optim as optim
from mvec_training import MVecTrainer

optimizer = optim.AdamW(model.parameters(), lr=3e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=3e-5
)

trainer = MVecTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device='cuda',
    gradient_clip=1.0,
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

## Training System

### `flexible_dataset.py`

Dataset with multiple chunking strategies.

#### Class: `FlexibleMVecDataset`

Dataset supporting different chunking modes.

**Constructor**:
```python
FlexibleMVecDataset(
    data_path: str,
    encoder: MVecEncoder,
    mode: str = 'pairs',
    data_format: str = 'csv',
    max_seq_len: int = 256,
    turns_per_conversation: int = 2,
    stride: int = 128,
    min_seq_len: int = 10
)
```

**Parameters**:
- `data_path`: Path to data file
- `encoder`: MVecEncoder instance
- `mode`: 'pairs', 'conversation', 'sliding', 'paragraph', 'document'
- `data_format`: 'csv', 'text', 'tagged_text'
- `max_seq_len`: Maximum sequence length
- `turns_per_conversation`: For 'conversation' mode
- `stride`: For 'sliding' mode
- `min_seq_len`: Filter sequences shorter than this

**Chunking Modes**:

**'pairs'**: Each Q&A pair is isolated
```
Sample 1: Q1 → A1
Sample 2: Q2 → A2
Sample 3: Q3 → A3
```

**'conversation'**: Chain N Q&A pairs
```
Sample 1: Q1 → A1 → Q2 → A2
Sample 2: Q3 → A3 → Q4 → A4
```

**'sliding'**: Overlapping windows
```
Sample 1: tokens[0:256]
Sample 2: tokens[128:384]  (overlaps with Sample 1)
Sample 3: tokens[256:512]
```

**'paragraph'**: Split on paragraph boundaries

**'document'**: Whole documents

**Example**:
```python
from flexible_dataset import FlexibleMVecDataset

dataset = FlexibleMVecDataset(
    data_path='data.csv',
    encoder=encoder,
    mode='conversation',
    max_seq_len=256,
    turns_per_conversation=3
)
```

---

#### Function: `collate_flexible_batch`

Collation function for flexible chunking.

```python
def collate_flexible_batch(batch):
    """Collate function for DataLoader"""
```

---

### `train_flexible.py` / `flexible_training.py`

Main training scripts with configuration.

**Usage**:
1. Edit CONFIGURATION section
2. Run: `python train_flexible.py`

**Configuration Variables**:
```python
# Data
DATA_PATH = 'your_data.csv'
DATA_FORMAT = 'csv'
CHUNKING_MODE = 'pairs'

# Sequence
MAX_SEQ_LEN = 256
TURNS_PER_CONVERSATION = 3
STRIDE = 128

# Model
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 4
IDENTITY_EMBED_DIM = 32
MAX_DELAY = 128
DROPOUT = 0.1

# Training
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 3e-4
VAL_SPLIT = 0.1

# Loss
WORD_LOSS_WEIGHT = 1.0
END_LOSS_WEIGHT = 1.0

# Save
SAVE_PATH = 'model.pt'
SAVE_VOCAB = 'vocab.json'

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

---

### `unified_training.py`

Unified training framework (alternative to train_flexible.py).

Similar functionality, different organization. Can be used for more complex training scenarios.

---

### `training_utils.py`

Generic training utilities for base TBT.

#### Class: `TextDataset`

Character/word-level text dataset.

```python
TextDataset(
    text: str,
    vocab: Dict[str, int],
    seq_len: int = 128,
    stride: Optional[int] = None
)
```

---

#### Class: `TimeSeriesDataset`

For sequence prediction tasks.

```python
TimeSeriesDataset(
    data: np.ndarray,
    seq_len: int = 100,
    prediction_length: int = 1,
    stride: int = 1
)
```

---

#### Class: `Trainer`

Generic trainer for TBT models.

```python
Trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: Optional[optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: str = 'cuda',
    gradient_clip: Optional[float] = 1.0,
    log_interval: int = 100
)
```

---

#### Helper Functions

**Data generation**:
```python
generate_lorenz_attractor(num_steps, dt, sigma, rho, beta)
normalize_timeseries(data)
denormalize_timeseries(data, stats)
```

**Metrics**:
```python
compute_perplexity(loss)
compute_mse(predictions, targets)
compute_mae(predictions, targets)
compute_rmse(predictions, targets)
```

**Vocabulary**:
```python
create_vocab_from_text(text)
decode_tokens(token_ids, idx_to_char)
```

---

## Inference Scripts

### `unified_run.py`

Standard Q&A inference.

**Configuration**:
```python
MODEL_PATH = 'path/to/model.pt'
VOCAB_PATH = 'path/to/vocab.json'
DEVICE = 'cpu'

TEST_QUESTIONS = [...]

MAX_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 50
```

**Key Function**:
```python
def generate_answer(question, verbose=True):
    """Generate Marina's answer to a question"""
```

**Usage**:
```bash
python unified_run.py
```

---

### `unified_run_qba.py`

Two-phase QBA inference.

**Configuration**:
```python
MODEL_PATH = 'path/to/qba_model.pt'
VOCAB_PATH = 'path/to/qba_vocab.json'

MAX_BRIDGE_TOKENS = 20
MAX_ANSWER_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 50
SHOW_BRIDGE = True  # Show/hide bridge
```

**Key Function**:
```python
def generate_answer_qba(question, verbose=True):
    """
    Two-phase generation:
    1. Generate bridge (INTERNAL)
    2. Generate answer (VISIBLE)
    
    Returns: (bridge, answer)
    """
```

**Usage**:
```bash
python unified_run_qba.py
```

---

### `unified_run_vocab_check.py`

Inference with vocabulary checking.

**Configuration**:
```python
MODEL_PATH = 'path/to/model.pt'
VOCAB_PATH = 'path/to/vocab.json'

TEST_QUESTIONS = [...]
```

**Additional Features**:
- Pre-flight vocabulary checks
- OOV word detection
- Suggestion of alternatives
- Safety confirmation before inference

**Usage**:
```bash
python unified_run_vocab_check.py
```

---

## Utilities

### `vocab_checker.py`

Check text for out-of-vocabulary words.

#### Class: `VocabChecker`

**Constructor**:
```python
VocabChecker(vocab_path: Optional[str] = None)
```

**Methods**:

**`.load_vocab(path)`**:
```python
def load_vocab(self, vocab_path: str) -> None:
    """Load vocabulary from JSON"""
```

**`.check_text(text, verbose)`**:
```python
def check_text(
    self,
    text: str,
    verbose: bool = True
) -> Tuple[List[str], List[str], Dict]:
    """
    Check text for OOV words.
    
    Returns:
        (in_vocab_tokens, oov_tokens, stats_dict)
    """
```

**`.check_questions(questions, verbose)`**:
```python
def check_questions(
    self,
    questions: List[str],
    verbose: bool = True
) -> Dict:
    """Check multiple questions"""
```

**`.is_safe_for_inference(text, allow_oov)`**:
```python
def is_safe_for_inference(
    self,
    text: str,
    allow_oov: bool = False
) -> bool:
    """Check if text is safe for inference"""
```

**`.suggest_alternatives(oov_word, max_suggestions)`**:
```python
def suggest_alternatives(
    self,
    oov_word: str,
    max_suggestions: int = 5
) -> List[Tuple[str, int]]:
    """Suggest similar words from vocabulary"""
```

**Example**:
```python
from vocab_checker import VocabChecker

checker = VocabChecker('vocab.json')

# Check text
in_vocab, oov, stats = checker.check_text("Where is Lady Serendipity?")
print(f"Coverage: {stats['coverage_pct']:.1f}%")
print(f"OOV words: {oov}")

# Safety check
if checker.is_safe_for_inference(question):
    answer = generate_answer(question)
else:
    print("Warning: OOV words detected")

# Get suggestions
for word in oov:
    suggestions = checker.suggest_alternatives(word)
    print(f"{word} → {suggestions}")
```

**Standalone usage**:
```bash
python vocab_checker.py
```

Edit configuration section to check your own questions.

---

### `vocab_updater.py`

Add words to existing vocabulary.

#### Class: `VocabUpdater`

**Constructor**:
```python
VocabUpdater(vocab_path: str)
```

**Methods**:

**`.add_words(words)`**:
```python
def add_words(self, words: List[str]) -> int:
    """
    Add new words to vocabulary.
    
    Returns:
        Number of words actually added
    """
```

**`.save(output_path)`**:
```python
def save(self, output_path: str = None) -> None:
    """Save updated vocabulary"""
```

**`.get_stats()`**:
```python
def get_stats(self) -> None:
    """Print vocabulary statistics"""
```

**⚠️ Warning**: After updating vocabulary, you must retrain the model! The model checkpoint still uses the old vocab_size.

**Example**:
```python
from vocab_updater import VocabUpdater

updater = VocabUpdater('vocab.json')
updater.add_words(['newword1', 'newword2'])
updater.save('vocab_updated.json')

# ⚠️ Now retrain model with updated vocabulary
```

**Standalone usage**:
```bash
python vocab_updater.py
```

Edit configuration section to add your own words.

---

## Testing and Examples

Most files include test code at the bottom:

```python
if __name__ == "__main__":
    # Test code here
    pass
```

To run tests for a specific file:

```bash
python takens_embedding.py
python mvec_encoder.py
python vocab_checker.py
# etc.
```

This will run built-in examples and validation checks.

---

## Summary of Key Files

**Must-read for training**:
- `train_flexible.py` - Main training script
- `mvec_encoder.py` - Understand encoding
- `mvec_dataset.py` - Understand data loading

**Must-read for inference**:
- `unified_run.py` - Standard inference
- `mvec_model.py` - Understand model architecture
- `vocab_checker.py` - Vocabulary safety

**Must-read for understanding**:
- `takens_embedding.py` - Core concept
- `tbt_architecture.py` - Base architecture
- `mvec_model.py` - MARINA extension

**Advanced**:
- `flexible_dataset.py` - Chunking strategies
- `unified_run_qba.py` - Two-phase generation
- `vocab_updater.py` - Vocabulary management

---

*This reference documents the code as it exists. Remember: this is proof-of-principle code optimized for clarity and experimentation, not production use.*
