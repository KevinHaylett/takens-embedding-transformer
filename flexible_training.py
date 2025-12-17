"""
Flexible Training with Flexible Chunking
 - flexible_training.py
====================================
Uses flexible_dataset.py for different chunking strategies.
Now integrated with the new three-mode encoder (QA, QBA, CONTINUOUS).

USAGE:
1. Edit CONFIGURATION section below
2. Press F5 in Spyder to run
3. Change ENCODING_MODE and CHUNKING_MODE to test different strategies
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import csv
import sys
import os

sys.path.append('.')

from mvec_encoder import MVecEncoder, EncodingMode
from flexible_dataset import FlexibleMVecDataset, collate_flexible_batch
from mvec_model import MVecLanguageModel
from tbt_architecture import create_exponential_delays
from mvec_training import MVecTrainer


# ============================================================
# CONFIGURATION - EDIT THESE
# ============================================================

# Data and encoding mode
DATA_PATH = 'Terrain_01.csv'
DATA_FORMAT = 'csv'  # 'csv', 'text', 'tagged_text'

# NEW: Encoding mode for vocabulary building
# Note: Actual encoding is determined by CSV structure
# - CSV with question,answer → QA encoding
# - CSV with question,bridge,answer → QBA encoding  
# - CSV with text → CONTINUOUS encoding
ENCODING_MODE = 'continuous' #'qa'  # 'qa', 'qba', 'continuous' (used for vocab building)

# Chunking mode (how to group sequences)
CHUNKING_MODE = 'sliding' #'pairs'  # 'pairs', 'conversation', 'paragraph', 'sliding', 'document'

# Chunking parameters
MAX_SEQ_LEN = 256
TURNS_PER_CONVERSATION = 3  # For conversation mode
STRIDE = 128                # For sliding mode
CHUNK_SIZE = 150            # For CONTINUOUS encoding mode

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

# Save paths
SAVE_PATH = 'Terrain_01.pt'
SAVE_VOCAB = 'Terrain_vocab.json'

# System
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_encoding_mode_enum(mode_str: str) -> EncodingMode:
    """Convert string to EncodingMode enum"""
    mode_map = {
        'qa': EncodingMode.QA,
        'qba': EncodingMode.QBA,
        'continuous': EncodingMode.CONTINUOUS
    }
    return mode_map.get(mode_str.lower(), EncodingMode.QA)


def detect_data_format_from_csv(csv_path: str) -> str:
    """Detect if CSV is QA or QBA format"""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = [h.lower() for h in reader.fieldnames]
        
        if 'bridge' in headers:
            return 'qba'
        elif 'text' in headers:
            return 'continuous'
        else:
            return 'qa'


def collect_texts_for_vocab(data_path: str, data_format: str, encoding_mode: str) -> list:
    """Collect all texts for vocabulary building"""
    texts = []
    
    if data_format == 'csv':
        detected_format = detect_data_format_from_csv(data_path)
        
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if detected_format == 'qba' or encoding_mode == 'qba':
                    q = row.get('question') or row.get('user', '')
                    b = row.get('bridge', '')
                    a = row.get('answer') or row.get('response', '')
                    texts.extend([q, b, a])
                    
                elif detected_format == 'continuous' or encoding_mode == 'continuous':
                    texts.append(row.get('text', ''))
                    
                else:  # QA format
                    q = row.get('question') or row.get('user', '')
                    a = row.get('answer') or row.get('response', '')
                    texts.extend([q, a])
    else:
        # Text file
        with open(data_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    
    return texts


# ============================================================
# TRAINING
# ============================================================

print("=" * 70)
print("Marina Training with Flexible Chunking + New Encoding Modes")
print("=" * 70)
print(f"\nEncoding mode: {ENCODING_MODE.upper()}")
print(f"Chunking mode: {CHUNKING_MODE}")

# Describe what this combination does
mode_descriptions = {
    'qa': "Simple Question-Answer (USER → VISIBLE)",
    'qba': "Question-Bridge-Answer (USER → INTERNAL → VISIBLE)",
    'continuous': "Continuous text stream (all VISIBLE)"
}

chunk_descriptions = {
    'pairs': "isolated Q&A pairs (fragmented landscape)",
    'conversation': f"conversations with {TURNS_PER_CONVERSATION} Q&A pairs (flowing landscape)",
    'sliding': f"sliding windows (stride={STRIDE}) (continuous landscape)",
    'paragraph': "paragraph-based chunks (natural boundaries)",
    'document': "full documents (maximum context)"
}

print(f"\nEncoding: {mode_descriptions.get(ENCODING_MODE, ENCODING_MODE)}")
print(f"Chunking: {chunk_descriptions.get(CHUNKING_MODE, CHUNKING_MODE)}")
print()

# Validate mode combinations
if ENCODING_MODE == 'continuous' and CHUNKING_MODE == 'pairs':
    print("⚠️  WARNING: CONTINUOUS encoding with PAIRS chunking")
    print("    Consider using 'paragraph' or 'sliding' chunking for better flow")
    print()

if ENCODING_MODE in ['qa', 'qba'] and CHUNKING_MODE in ['paragraph', 'document']:
    print("⚠️  WARNING: Q&A encoding with document-level chunking")
    print("    Consider using 'pairs' or 'conversation' chunking")
    print()

# Create/load encoder
print("1. Building vocabulary...")
encoder = MVecEncoder()

# Collect texts based on format
texts = collect_texts_for_vocab(DATA_PATH, DATA_FORMAT, ENCODING_MODE)

# Remove empty texts
texts = [t for t in texts if t.strip()]

encoder.build_vocab(texts, min_freq=1)
encoder.save_vocab(SAVE_VOCAB)

vocab_size = encoder.get_vocab_size()
print(f"Vocabulary: {vocab_size} words")
print(f"Saved to: {SAVE_VOCAB}")

# Convert encoding mode to enum
encoding_mode_enum = get_encoding_mode_enum(ENCODING_MODE)

# Create dataset
print(f"\n2. Creating dataset...")
print(f"   Encoding: {ENCODING_MODE}")
print(f"   Chunking: {CHUNKING_MODE}")
print(f"   Note: Encoding mode is determined by CSV structure")

# FlexibleMVecDataset will use the encoder's methods based on data structure
# QA: question,answer columns → encoder.encode_qa()
# QBA: question,bridge,answer columns → encoder.encode_qba()  
# CONTINUOUS: text column → encoder.encode_continuous()
full_dataset = FlexibleMVecDataset(
    data_path=DATA_PATH,
    encoder=encoder,
    mode=CHUNKING_MODE,
    data_format=DATA_FORMAT,
    max_seq_len=MAX_SEQ_LEN,
    turns_per_conversation=TURNS_PER_CONVERSATION,
    stride=STRIDE,
    min_seq_len=10
)

print(f"Total samples: {len(full_dataset)}")

# Check if dataset is too small
if len(full_dataset) < 10:
    print(f"\n⚠️  WARNING: Very small dataset ({len(full_dataset)} samples)")
    print(f"   For better training, consider:")
    print(f"   • Adding more data to your CSV")
    print(f"   • Using CHUNKING_MODE='conversation' with TURNS_PER_CONVERSATION=1")
    print(f"   • Using example_qa_data.csv which has 10 samples")
    print()

# Split train/val
val_size = int(VAL_SPLIT * len(full_dataset))
train_size = len(full_dataset) - val_size

# Ensure at least 1 sample in validation if dataset is small
if val_size == 0 and len(full_dataset) > 1:
    val_size = 1
    train_size = len(full_dataset) - 1
    print(f"⚠️  Small dataset: Using 1 sample for validation")
elif val_size == 0:
    # Dataset too small for validation split
    val_size = 0
    train_size = len(full_dataset)
    VAL_SPLIT = 0.0
    print(f"⚠️  Dataset too small for validation split - training only")

if val_size > 0:
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
else:
    train_dataset = full_dataset
    val_dataset = None

print(f"Training samples: {len(train_dataset)}")
if val_dataset is not None:
    print(f"Validation samples: {len(val_dataset)}")
else:
    print(f"Validation samples: 0 (training only mode)")

# Show sample info
if len(full_dataset) > 0:
    sample = full_dataset[0]
    print(f"Sample 0 length: {len(sample[0])} tokens")
    
    # Show channel distribution
    identity_ids = sample[2]
    channels = {0: 'USER', 1: 'INTERNAL', 2: 'VISIBLE'}
    channel_counts = {}
    for ch_id in identity_ids.tolist():
        ch_name = channels.get(ch_id, f'Unknown({ch_id})')
        channel_counts[ch_name] = channel_counts.get(ch_name, 0) + 1
    
    print(f"Channel distribution: {channel_counts}")

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
    val_loader = None  # No validation loader

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
print(f"Model parameters: {num_params:,}")
print(f"Device: {DEVICE}")

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
print(f"Word loss weight: {WORD_LOSS_WEIGHT}")
print(f"End loss weight: {END_LOSS_WEIGHT}")
if val_loader is None:
    print(f"⚠️  Training without validation (dataset too small)")

trainer = MVecTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader if val_loader is not None else train_loader,  # Use train as val if no validation
    optimizer=optimizer,
    device=DEVICE,
    gradient_clip=1.0,
    log_interval=10,
    word_loss_weight=WORD_LOSS_WEIGHT,
    end_loss_weight=END_LOSS_WEIGHT
)

# Add a flag to skip validation if needed
if val_loader is None:
    print("Note: Validation will use training data (no separate val set)")

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
    
    print("\n" + "=" * 70)
    print("Configuration Analysis:")
    print("=" * 70)
    
    # Encoding mode analysis
    print(f"\nENCODING MODE: {ENCODING_MODE.upper()}")
    if ENCODING_MODE == 'qa':
        print("✓ Simple question-answer encoding")
        print("✓ Two channels: USER, VISIBLE")
        print("✓ Fast training, direct associations")
        print()
        print("To add explicit reasoning, try:")
        print("  Set ENCODING_MODE = 'qba'")
        
    elif ENCODING_MODE == 'qba':
        print("✓ Question-bridge-answer with internal reasoning")
        print("✓ Three channels: USER, INTERNAL, VISIBLE")
        print("✓ Explicit semantic pathways")
        print("✓ Rich manifold structure")
        print()
        print("For faster training without bridges, try:")
        print("  Set ENCODING_MODE = 'qa'")
        
    elif ENCODING_MODE == 'continuous':
        print("✓ Continuous text stream encoding")
        print("✓ One channel: VISIBLE (all text)")
        print("✓ General language learning")
        if CHUNK_SIZE:
            print(f"✓ Chunking every {CHUNK_SIZE} tokens")
    
    # Chunking mode analysis
    print(f"\nCHUNKING MODE: {CHUNKING_MODE.upper()}")
    if CHUNKING_MODE == 'pairs':
        print("✓ Creates fragmented phase space")
        print("✓ Each Q&A is isolated - no flow between samples")
        print()
        print("To create flowing landscape, try:")
        print("  Set CHUNKING_MODE = 'conversation'")
        print(f"  Adjust TURNS_PER_CONVERSATION = 3-5")
        
    elif CHUNKING_MODE == 'conversation':
        print(f"✓ Chains {TURNS_PER_CONVERSATION} Q&A pairs")
        print("✓ Creates connected trajectories through phase space")
        print("✓ Context flow between turns")
        print("✓ Conversational continuity")
        print("✓ Richer manifold structure")
        print()
        print("To adjust conversation length:")
        print("  Increase TURNS_PER_CONVERSATION for longer contexts")
        
    elif CHUNKING_MODE == 'sliding':
        print(f"✓ Sliding windows with stride={STRIDE}")
        print("✓ Creates continuous overlapping sequences")
        print("✓ Maximum manifold connectivity")
        print("✓ Smooth phase space coverage")
        print()
        print("To adjust overlap:")
        print("  Decrease STRIDE for more overlap")
        print("  Increase STRIDE for less redundancy")
    
    elif CHUNKING_MODE == 'paragraph':
        print("✓ Natural paragraph boundaries")
        print("✓ Semantic coherence within chunks")
        print("✓ Good balance of context and boundaries")
    
    elif CHUNKING_MODE == 'document':
        print("✓ Full document context")
        print("✓ Maximum context window")
        print("✓ Best for understanding long-range dependencies")

# Dataset statistics
print("\n" + "=" * 70)
print("Dataset Statistics:")
print("=" * 70)

if len(train_dataset) > 0:
    # Get lengths safely
    sample_lengths = []
    for i in range(min(len(full_dataset), 100)):  # Sample first 100
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

# Recommendations
print("\n" + "=" * 70)
print("Recommendations:")
print("=" * 70)

print("\n💡 Experiment with different combinations:")
print()
print("For chatbot training:")
print("  ENCODING_MODE = 'qa'")
print("  CHUNKING_MODE = 'pairs' or 'conversation'")
print()
print("For reasoning/manifold experiments:")
print("  ENCODING_MODE = 'qba'")
print("  CHUNKING_MODE = 'conversation' (flowing contexts)")
print()
print("For book pre-training:")
print("  ENCODING_MODE = 'continuous'")
print("  CHUNKING_MODE = 'sliding' or 'paragraph'")
print("  Adjust CHUNK_SIZE = 150-200")
print()
print("For maximum context:")
print("  CHUNKING_MODE = 'document'")
print("  Increase MAX_SEQ_LEN = 512+")

print("\n" + "=" * 70)
