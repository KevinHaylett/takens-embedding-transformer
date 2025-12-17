"""
Unified Training with CSV and Plain Text Support
 - unified_training.py
====================================
Handles both CSV files (QA/QBA format) and plain text files automatically.
Integrates with the three-mode encoder (QA, QBA, CONTINUOUS).

USAGE:
1. Edit CONFIGURATION section below
2. Set DATA_PATH to either:
   - CSV file with question,answer columns (QA mode)
   - CSV file with question,bridge,answer columns (QBA mode)
   - Plain text file (.txt) (CONTINUOUS mode)
3. Press F5 in Spyder to run

The system will automatically detect the file format and encoding mode!
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import csv
import sys
import os
from pathlib import Path

sys.path.append('.')

from mvec_encoder import MVecEncoder, EncodingMode
from flexible_dataset import FlexibleMVecDataset, collate_flexible_batch
from mvec_model import MVecLanguageModel
from tbt_architecture import create_exponential_delays
from mvec_training import MVecTrainer


# ============================================================
# CONFIGURATION - EDIT THESE
# ============================================================


# Training data file (can be CSV or plain text)
# Set File first line = text or question,answer
DATA_PATH = r'c:\Marina\models\solar_system\solar_system.csv' #'brown.txt'  # or 'my_book.txt', 'my_data.csv', etc.



# Chunking mode (how to group sequences)
# - For CSV/QA data: 'pairs', 'conversation' work best
# - For plain text: 'sliding', 'paragraph', 'document' work best
CHUNKING_MODE = 'pairs'  # 'pairs', 'conversation', 'paragraph', 'sliding', 'document'

# Chunking parameters
MAX_SEQ_LEN = 256
TURNS_PER_CONVERSATION = 1#3  # For conversation mode with CSV
STRIDE = 128                # For sliding mode with text
CHUNK_SIZE = 150            # For continuous text chunking

# Model architecture
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 4
IDENTITY_EMBED_DIM = 32
MAX_DELAY = 128
DROPOUT = 0.1

# Training parameters
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 3e-4
VAL_SPLIT = 0.1

# Loss weights
WORD_LOSS_WEIGHT = 1.0
END_LOSS_WEIGHT = 1.0

# Save paths (auto-generated from DATA_PATH if not specified)
SAVE_PATH = None  # Will become 'Terrain_01.pt' or 'my_book.pt'
SAVE_VOCAB = None  # Will become 'Terrain_vocab.json' or 'my_book_vocab.json'

# System
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# AUTO-DETECTION FUNCTIONS
# ============================================================

def detect_file_format(file_path: str) -> str:
    """
    Automatically detect if file is CSV or plain text.
    Returns: 'csv' or 'text'
    """
    ext = Path(file_path).suffix.lower()
    
    if ext == '.csv':
        return 'csv'
    elif ext in ['.txt', '.text', '.md', '.markdown']:
        return 'text'
    else:
        # Try to detect by content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                # Check if it looks like CSV
                if ',' in first_line and any(header in first_line.lower() 
                                            for header in ['question', 'answer', 'text', 'user', 'response']):
                    return 'csv'
        except:
            pass
    
    # Default to text
    return 'text'


def detect_csv_encoding_mode(csv_path: str) -> str:
    """
    Detect if CSV is QA, QBA, or CONTINUOUS format.
    Returns: 'qa', 'qba', or 'continuous'
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = [h.lower() for h in reader.fieldnames]
        
        if 'bridge' in headers:
            return 'qba'
        elif 'text' in headers:
            return 'continuous'
        else:
            return 'qa'


def get_encoding_mode_enum(mode_str: str) -> EncodingMode:
    """Convert string to EncodingMode enum"""
    mode_map = {
        'qa': EncodingMode.QA,
        'qba': EncodingMode.QBA,
        'continuous': EncodingMode.CONTINUOUS
    }
    return mode_map.get(mode_str.lower(), EncodingMode.CONTINUOUS)


def auto_detect_encoding_and_format(data_path: str) -> tuple:
    """
    Automatically detect both file format and encoding mode.
    Returns: (data_format, encoding_mode, description)
    """
    data_format = detect_file_format(data_path)
    
    if data_format == 'csv':
        encoding_mode = detect_csv_encoding_mode(data_path)
        
        descriptions = {
            'qa': 'CSV with Question-Answer pairs',
            'qba': 'CSV with Question-Bridge-Answer (reasoning)',
            'continuous': 'CSV with continuous text'
        }
        description = descriptions.get(encoding_mode, 'CSV file')
        
    else:  # text file
        encoding_mode = 'continuous'
        description = 'Plain text file'
    
    return data_format, encoding_mode, description


def collect_texts_for_vocab(data_path: str, data_format: str, encoding_mode: str) -> list:
    """Collect all texts for vocabulary building"""
    texts = []
    
    if data_format == 'csv':
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if encoding_mode == 'qba':
                    q = row.get('question') or row.get('user', '')
                    b = row.get('bridge', '')
                    a = row.get('answer') or row.get('response', '')
                    texts.extend([q, b, a])
                    
                elif encoding_mode == 'continuous':
                    texts.append(row.get('text', ''))
                    
                else:  # QA format
                    q = row.get('question') or row.get('user', '')
                    a = row.get('answer') or row.get('response', '')
                    texts.extend([q, a])
    else:
        # Plain text file - read entire content
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split into paragraphs or sentences for vocab building
            # This helps with better tokenization
            paragraphs = content.split('\n\n')
            texts.extend([p.strip() for p in paragraphs if p.strip()])
    
    return texts


def generate_save_paths(data_path: str) -> tuple:
    """Generate model and vocab save paths from data file name"""
    base_name = Path(data_path).stem
    model_path = f"{base_name}_model.pt"
    vocab_path = f"{base_name}_vocab.json"
    return model_path, vocab_path


def recommend_chunking_mode(encoding_mode: str, data_format: str) -> str:
    """Suggest best chunking mode based on data type"""
    if data_format == 'csv':
        if encoding_mode in ['qa', 'qba']:
            return 'conversation'  # Good for Q&A flow
        else:
            return 'sliding'  # Good for continuous CSV text
    else:
        return 'sliding'  # Good for plain text files


# ============================================================
# TRAINING
# ============================================================

print("=" * 70)
print("Marina Unified Training System")
print("CSV + Plain Text Support")
print("=" * 70)

# Check if file exists
if not os.path.exists(DATA_PATH):
    print(f"\n❌ ERROR: File not found: {DATA_PATH}")
    print("\nPlease set DATA_PATH to an existing file:")
    print("  - CSV file: 'my_data.csv' (with question,answer columns)")
    print("  - Text file: 'my_book.txt' (plain text)")
    sys.exit(1)

# Auto-detect format and encoding
print(f"\n📁 Input file: {DATA_PATH}")
data_format, encoding_mode, description = auto_detect_encoding_and_format(DATA_PATH)

print(f"   Format detected: {description}")
print(f"   Data format: {data_format.upper()}")
print(f"   Encoding mode: {encoding_mode.upper()}")

# Generate save paths if not specified
if SAVE_PATH is None or SAVE_VOCAB is None:
    auto_model_path, auto_vocab_path = generate_save_paths(DATA_PATH)
    if SAVE_PATH is None:
        SAVE_PATH = auto_model_path
    if SAVE_VOCAB is None:
        SAVE_VOCAB = auto_vocab_path
    print(f"   Model will save to: {SAVE_PATH}")
    print(f"   Vocab will save to: {SAVE_VOCAB}")

# Check if chunking mode is appropriate
recommended_mode = recommend_chunking_mode(encoding_mode, data_format)
if CHUNKING_MODE != recommended_mode:
    print(f"\n💡 Recommendation: For {description},")
    print(f"   CHUNKING_MODE='{recommended_mode}' might work better")
    print(f"   Currently using: CHUNKING_MODE='{CHUNKING_MODE}'")

print(f"\nChunking mode: {CHUNKING_MODE}")

# Describe the approach
mode_descriptions = {
    'qa': "Question-Answer pairs (USER → VISIBLE)",
    'qba': "Question-Bridge-Answer with reasoning (USER → INTERNAL → VISIBLE)",
    'continuous': "Continuous text stream (all VISIBLE)"
}

chunk_descriptions = {
    'pairs': "isolated Q&A pairs",
    'conversation': f"conversations with {TURNS_PER_CONVERSATION} turns",
    'sliding': f"sliding windows (stride={STRIDE})",
    'paragraph': "paragraph-based chunks",
    'document': "full documents"
}

print(f"\nStrategy:")
print(f"  Encoding: {mode_descriptions.get(encoding_mode, encoding_mode)}")
print(f"  Chunking: {chunk_descriptions.get(CHUNKING_MODE, CHUNKING_MODE)}")
print()

# Validate mode combinations
if encoding_mode == 'continuous' and CHUNKING_MODE == 'pairs':
    print("⚠️  WARNING: CONTINUOUS encoding with PAIRS chunking")
    print("    Consider using 'sliding' or 'paragraph' for better text flow")
    print()

if encoding_mode in ['qa', 'qba'] and CHUNKING_MODE in ['paragraph', 'document']:
    print("⚠️  WARNING: Q&A encoding with document-level chunking")
    print("    Consider using 'pairs' or 'conversation' chunking")
    print()

# Create/load encoder
print("1. Building vocabulary...")
encoder = MVecEncoder()

# Collect texts based on format
texts = collect_texts_for_vocab(DATA_PATH, data_format, encoding_mode)

# Remove empty texts
texts = [t for t in texts if t.strip()]

print(f"   Processing {len(texts)} text segments...")
encoder.build_vocab(texts, min_freq=1)
encoder.save_vocab(SAVE_VOCAB)

vocab_size = encoder.get_vocab_size()
print(f"   Vocabulary: {vocab_size} words")
print(f"   Saved to: {SAVE_VOCAB}")

# Convert encoding mode to enum
encoding_mode_enum = get_encoding_mode_enum(encoding_mode)

# Create dataset
print(f"\n2. Creating dataset...")

full_dataset = FlexibleMVecDataset(
    data_path=DATA_PATH,
    encoder=encoder,
    mode=CHUNKING_MODE,
    data_format=data_format,
    max_seq_len=MAX_SEQ_LEN,
    turns_per_conversation=TURNS_PER_CONVERSATION,
    stride=STRIDE,
    min_seq_len=10
)

print(f"   Total samples: {len(full_dataset)}")

# Check if dataset is too small
if len(full_dataset) < 10:
    print(f"\n⚠️  WARNING: Small dataset ({len(full_dataset)} samples)")
    if data_format == 'text':
        print(f"   For plain text, try:")
        print(f"   • Decreasing STRIDE to create more overlapping windows")
        print(f"   • Using CHUNKING_MODE='sliding' with smaller stride")
    else:
        print(f"   For CSV data, try:")
        print(f"   • Adding more rows to your CSV")
        print(f"   • Using CHUNKING_MODE='conversation' with TURNS_PER_CONVERSATION=1")
    print()

# Split train/val
val_size = int(VAL_SPLIT * len(full_dataset))
train_size = len(full_dataset) - val_size

# Ensure at least 1 sample in validation if dataset is small
if val_size == 0 and len(full_dataset) > 1:
    val_size = 1
    train_size = len(full_dataset) - 1
    print(f"   Small dataset: Using 1 sample for validation")
elif val_size == 0:
    # Dataset too small for validation split
    val_size = 0
    train_size = len(full_dataset)
    VAL_SPLIT = 0.0
    print(f"   Dataset too small for validation split - training only")

if val_size > 0:
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
else:
    train_dataset = full_dataset
    val_dataset = None

print(f"   Training samples: {len(train_dataset)}")
if val_dataset is not None:
    print(f"   Validation samples: {len(val_dataset)}")

# CRITICAL CHECK: Stop if no samples were created
if len(train_dataset) == 0:
    print("\n" + "=" * 70)
    print("❌ ERROR: No training samples created!")
    print("=" * 70)
    print("\n🔍 Diagnosis:")
    print(f"   File: {DATA_PATH}")
    print(f"   Format: {data_format}")
    print(f"   Encoding: {encoding_mode}")
    print(f"   Chunking: {CHUNKING_MODE}")
    print(f"   Max seq len: {MAX_SEQ_LEN}")
    print(f"   Stride: {STRIDE}")
    
    print("\n💡 IMMEDIATE ACTIONS:")
    print("   1. Run quick_debug.py to diagnose the issue")
    print("      • Edit: TEST_FILE = '{}'".format(DATA_PATH))
    print("      • Run it to see what's wrong")
    print()
    print("   2. Check your file has enough content:")
    if data_format == 'text':
        print("      • Plain text needs 100+ words minimum")
        print("      • Try: MAX_SEQ_LEN=64, STRIDE=32")
    else:
        print("      • CSV needs 10+ rows minimum")
        print("      • Check your CSV has correct columns")
    print()
    print("   3. Try smaller settings:")
    print("      MAX_SEQ_LEN = 64")
    print("      STRIDE = 32")
    print("      min_seq_len = 5")
    print()
    print("   4. Verify encoder.encode_continuous() works:")
    print("      from mvec_encoder import MVecEncoder")
    print("      encoder = MVecEncoder()")
    print("      result = encoder.encode_continuous('test text', max_len=256)")
    print("      print(result)  # Should NOT be None")
    
    print("\n" + "=" * 70)
    sys.exit(1)

# Show sample info
if len(full_dataset) > 0:
    sample = full_dataset[0]
    print(f"\n   Sample 0: {len(sample[0])} tokens")
    
    # Show channel distribution
    identity_ids = sample[2]
    channels = {0: 'USER', 1: 'INTERNAL', 2: 'VISIBLE'}
    channel_counts = {}
    for ch_id in identity_ids.tolist():
        ch_name = channels.get(ch_id, f'Unknown({ch_id})')
        channel_counts[ch_name] = channel_counts.get(ch_name, 0) + 1
    
    print(f"   Channels: {channel_counts}")

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_flexible_batch
)

if val_dataset is not None:
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_flexible_batch
    )
else:
    val_loader = None

# Create model
print("\n3. Creating model...")

model = MVecLanguageModel(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    max_seq_len=512,
    delays=create_exponential_delays(MAX_DELAY),
    dropout=DROPOUT,
    use_identity_embed=True,
    identity_embed_dim=IDENTITY_EMBED_DIM,
    tie_weights=False
)

num_params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {num_params:,}")
print(f"   Device: {DEVICE}")

# Optimizer and scheduler
optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=LEARNING_RATE * 0.1
)

# Create trainer
print("\n4. Training...")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Learning rate: {LEARNING_RATE}")

trainer = MVecTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader if val_loader is not None else train_loader,
    optimizer=optimizer,
    device=DEVICE,
    gradient_clip=1.0,
    log_interval=10,
    word_loss_weight=WORD_LOSS_WEIGHT,
    end_loss_weight=END_LOSS_WEIGHT
)

# Train
history = trainer.train(
    num_epochs=EPOCHS,
    save_path=SAVE_PATH,
    scheduler=scheduler,
    early_stopping_patience=10
)

# Results
print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)
print(f"Best validation loss: {trainer.best_val_loss:.4f}")
print(f"Model saved to: {SAVE_PATH}")
print(f"Vocabulary saved to: {SAVE_VOCAB}")

# Show final metrics
if history['val']:
    final_val = history['val'][-1]
    print(f"\nFinal validation metrics:")
    print(f"  Total loss: {final_val['total_loss']:.4f}")
    print(f"  Word loss: {final_val['word_loss']:.4f}")
    print(f"  End loss: {final_val['end_loss']:.4f}")

# Dataset statistics
print("\n" + "=" * 70)
print("Dataset Statistics:")
print("=" * 70)

if len(train_dataset) > 0:
    sample_lengths = []
    for i in range(min(len(full_dataset), 100)):
        try:
            sample = full_dataset[i]
            sample_lengths.append(len(sample[0]))
        except:
            pass
    
    if sample_lengths:
        avg_len = sum(sample_lengths) / len(sample_lengths)
        max_len = max(sample_lengths)
        min_len = min(sample_lengths)
        
        print(f"Average tokens per sample: {avg_len:.1f}")
        print(f"Min tokens: {min_len}")
        print(f"Max tokens: {max_len}")
        print(f"Total samples: {len(full_dataset)}")

# Format-specific recommendations
print("\n" + "=" * 70)
print("Format-Specific Tips:")
print("=" * 70)

if data_format == 'csv':
    print("\n📊 CSV File Tips:")
    print()
    if encoding_mode == 'qa':
        print("✓ Your CSV has question,answer columns")
        print("  Good for: chatbot training, Q&A pairs")
        print()
        print("  Recommended chunking:")
        print("    CHUNKING_MODE = 'pairs' (isolated pairs)")
        print("    CHUNKING_MODE = 'conversation' (flowing context)")
    
    elif encoding_mode == 'qba':
        print("✓ Your CSV has question,bridge,answer columns")
        print("  Good for: reasoning chains, manifold experiments")
        print()
        print("  Recommended chunking:")
        print("    CHUNKING_MODE = 'conversation' (best for QBA)")
        print("    TURNS_PER_CONVERSATION = 3-5")
    
    elif encoding_mode == 'continuous':
        print("✓ Your CSV has text column")
        print("  Good for: continuous text processing")
        print()
        print("  Recommended chunking:")
        print("    CHUNKING_MODE = 'sliding' or 'paragraph'")

else:  # text file
    print("\n📄 Plain Text File Tips:")
    print()
    print("✓ Your file contains continuous text")
    print("  Good for: books, articles, general text")
    print()
    print("  Recommended chunking:")
    print("    CHUNKING_MODE = 'sliding'")
    print("    Adjust STRIDE for overlap control:")
    print("      STRIDE = 64  (75% overlap, slower but thorough)")
    print("      STRIDE = 128 (50% overlap, balanced)")
    print("      STRIDE = 192 (25% overlap, faster)")
    print()
    print("  Alternative chunking:")
    print("    CHUNKING_MODE = 'paragraph' (natural boundaries)")
    print("    CHUNKING_MODE = 'document' (maximum context)")

# General recommendations
print("\n" + "=" * 70)
print("Experimentation Guide:")
print("=" * 70)

print("\n🔬 Try These Combinations:")
print()
print("For chatbot training:")
print("  • Use CSV with question,answer columns")
print("  • Set CHUNKING_MODE = 'conversation'")
print("  • Set TURNS_PER_CONVERSATION = 3-5")
print()
print("For book/article learning:")
print("  • Use plain .txt file")
print("  • Set CHUNKING_MODE = 'sliding'")
print("  • Set STRIDE = 128 (50% overlap)")
print("  • Set MAX_SEQ_LEN = 256-512")
print()
print("For reasoning experiments:")
print("  • Use CSV with question,bridge,answer columns")
print("  • Set CHUNKING_MODE = 'conversation'")
print("  • Experiment with TURNS_PER_CONVERSATION")
print()
print("For maximum context:")
print("  • Set CHUNKING_MODE = 'document'")
print("  • Increase MAX_SEQ_LEN = 512+")
print("  • Reduce BATCH_SIZE if memory issues")

print("\n" + "=" * 70)
print("\n✅ Training complete! Your model is ready to use.")
print(f"   Load with: torch.load('{SAVE_PATH}')")
print(f"   Vocab at: {SAVE_VOCAB}")
print("\n" + "=" * 70)
