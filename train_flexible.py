"""
Flexible Training with Flexible Chunking
 - flexible_training.py
====================================
Uses flexible_dataset.py for different chunking strategies.

USAGE:
1. Edit CONFIGURATION section below
2. Press F5 in Spyder to run
3. Change CHUNKING_MODE to test different strategies
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import csv
import sys

sys.path.append('.')

from mvec_encoder import MVecEncoder
from flexible_dataset import FlexibleMVecDataset, collate_flexible_batch
from mvec_model import MVecLanguageModel
from tbt_architecture import create_exponential_delays
from mvec_training import MVecTrainer


# ============================================================
# CONFIGURATION - EDIT THESE
# ============================================================

# Data and chunking
DATA_PATH = 'example_qa_data.csv'
DATA_FORMAT = 'csv'  # 'csv', 'text', 'tagged_text'
CHUNKING_MODE = 'pairs'  # 'pairs', 'conversation', 'paragraph', 'sliding', 'document'

# Chunking parameters
MAX_SEQ_LEN = 256
TURNS_PER_CONVERSATION = 1  # For conversation mode
STRIDE = 128                # For sliding mode

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
SAVE_PATH = 'dialog.pt'
SAVE_VOCAB ='dialog_vocab.json'             # 'marina_flexible_vocab.json'

# System
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# TRAINING
# ============================================================

print("=" * 70)
print("Marina Training with Flexible Chunking")
print("=" * 70)
print(f"\nChunking mode: {CHUNKING_MODE}")
print(f"This creates: ", end='')

if CHUNKING_MODE == 'pairs':
    print("isolated Q&A pairs (fragmented landscape)")
elif CHUNKING_MODE == 'conversation':
    print(f"conversations with {TURNS_PER_CONVERSATION} Q&A pairs (flowing landscape)")
elif CHUNKING_MODE == 'sliding':
    print(f"sliding windows (stride={STRIDE}) (continuous landscape)")
else:
    print(f"{CHUNKING_MODE} chunking")

print()

# Create/load encoder
print("1. Building vocabulary...")
encoder = MVecEncoder()

texts = []
if DATA_FORMAT == 'csv':
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.extend([row['question'], row['answer']])
else:
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        texts.append(f.read())

encoder.build_vocab(texts, min_freq=1)
encoder.save_vocab(SAVE_VOCAB)

vocab_size = encoder.get_vocab_size()
print(f"Vocabulary: {vocab_size} words")
print(f"Saved to: {SAVE_VOCAB}")

# Create dataset
print(f"\n2. Creating dataset (mode={CHUNKING_MODE})...")
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

# Split train/val
val_size = int(VAL_SPLIT * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_flexible_batch
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_flexible_batch
)

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
print(f"End loss weight: {END_LOSS_WEIGHT}")

trainer = MVecTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
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
    
    print("\n" + "=" * 70)
    print("Chunking Strategy Impact:")
    print("=" * 70)
    
    if CHUNKING_MODE == 'pairs':
        print("PAIRS mode creates fragmented phase space.")
        print("Each Q&A is isolated - no flow between samples.")
        print()
        print("Try CONVERSATION mode to create flowing landscape:")
        print("  Set CHUNKING_MODE = 'conversation'")
        
    elif CHUNKING_MODE == 'conversation':
        print(f"CONVERSATION mode chains {TURNS_PER_CONVERSATION} Q&A pairs.")
        print("Creates connected trajectories through phase space.")
        print()
        print("Model learns:")
        print("  - Context flow between turns")
        print("  - Conversational continuity")
        print("  - Richer manifold structure")
        
    elif CHUNKING_MODE == 'sliding':
        print(f"SLIDING mode with stride={STRIDE}.")
        print("Creates continuous overlapping sequences.")
        print("Maximum manifold connectivity.")


print(f"Avg tokens per sample: {sum(len(s) for s in train_dataset)/len(train_dataset):.1f}")
print(f"Longest sample: {max(len(s) for s in full_dataset)} tokens")
print(f"Effective coverage: {len(full_dataset) * (MAX_SEQ_LEN - STRIDE if 'sliding' in CHUNKING_MODE else MAX_SEQ_LEN) / len(texts):.1f}x")
print("\n" + "=" * 70)
