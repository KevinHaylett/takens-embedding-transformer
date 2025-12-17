"""
MVec Training Script: mvec_training.py
Train Marina with multi-channel data.
Supports QA, QBA, and CONTINUOUS encoding modes.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import os
from tqdm import tqdm
import json
import csv

from mvec_encoder import MVecEncoder, EncodingMode
from mvec_dataset import MVecDataset, collate_mvec_batch
from mvec_model import MVecLanguageModel


class MVecTrainer:
    """Trainer for MVec language model"""
    
    def __init__(
        self,
        model: MVecLanguageModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: str = 'cpu',
        gradient_clip: float = 1.0,
        log_interval: int = 10,
        word_loss_weight: float = 1.0,
        end_loss_weight: float = 1.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.gradient_clip = gradient_clip
        self.log_interval = log_interval
        self.word_loss_weight = word_loss_weight
        self.end_loss_weight = end_loss_weight
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()
        total_word_loss = 0.0
        total_end_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids, target_ids, identity_ids, end_ids = batch
            
            # Move to device
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            identity_ids = identity_ids.to(self.device)
            end_ids = end_ids.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            word_logits, end_logits, word_loss, end_loss = self.model(
                input_ids,
                identity_ids,
                word_labels=target_ids,
                end_labels=end_ids
            )
            
            # Combined loss
            loss = self.word_loss_weight * word_loss + self.end_loss_weight * end_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Track losses
            total_word_loss += word_loss.item()
            total_end_loss += end_loss.item()
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                avg_word = total_word_loss / num_batches
                avg_end = total_end_loss / num_batches
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'word': f'{avg_word:.4f}',
                    'end': f'{avg_end:.4f}'
                })
        
        return {
            'total_loss': total_loss / num_batches,
            'word_loss': total_word_loss / num_batches,
            'end_loss': total_end_loss / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate on validation set"""
        self.model.eval()
        total_word_loss = 0.0
        total_end_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            input_ids, target_ids, identity_ids, end_ids = batch
            
            # Move to device
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            identity_ids = identity_ids.to(self.device)
            end_ids = end_ids.to(self.device)
            
            # Forward pass
            word_logits, end_logits, word_loss, end_loss = self.model(
                input_ids,
                identity_ids,
                word_labels=target_ids,
                end_labels=end_ids
            )
            
            loss = self.word_loss_weight * word_loss + self.end_loss_weight * end_loss
            
            total_word_loss += word_loss.item()
            total_end_loss += end_loss.item()
            total_loss += loss.item()
            num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'word_loss': total_word_loss / num_batches,
            'end_loss': total_end_loss / num_batches
        }
    
    def train(
        self,
        num_epochs: int,
        save_path: str,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        early_stopping_patience: int = 5
    ) -> dict:
        """Train for multiple epochs"""
        patience_counter = 0
        history = {'train': [], 'val': []}
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 70)
            
            # Train
            train_metrics = self.train_epoch()
            history['train'].append(train_metrics)
            
            print(f"Train - Loss: {train_metrics['total_loss']:.4f} | "
                  f"Word: {train_metrics['word_loss']:.4f} | "
                  f"End: {train_metrics['end_loss']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            history['val'].append(val_metrics)
            
            print(f"Val   - Loss: {val_metrics['total_loss']:.4f} | "
                  f"Word: {val_metrics['word_loss']:.4f} | "
                  f"End: {val_metrics['end_loss']:.4f}")
            
            # Save best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                patience_counter = 0
                
                # Save model config for inference
                model_config = {
                    'vocab_size': self.model.vocab_size,
                    'embed_dim': self.model.embed_dim,
                    'hidden_dim': self.model.hidden_dim,
                    'num_layers': len(self.model.tbt.layers),
                    'use_identity_embed': self.model.use_identity_embed,
                    'identity_embed_dim': self.model.identity_embed_dim
                }
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'history': history,
                    'model_config': model_config
                }, save_path)
                print(f"[SAVED] Best model saved to {save_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break
            
            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['total_loss'])
                else:
                    scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Learning Rate: {current_lr:.6f}")
        
        return history


def detect_encoding_mode(csv_path: str) -> EncodingMode:
    """
    Detect the encoding mode from CSV structure.
    
    CSV Formats:
    - QA: columns 'question', 'answer' (or 'user', 'response')
    - QBA: columns 'question', 'bridge', 'answer' (or 'user', 'bridge', 'response')
    - CONTINUOUS: column 'text'
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
    
    if not headers:
        raise ValueError("CSV file is empty or has no headers")
    
    headers_lower = [h.lower() for h in headers]
    
    # Check for CONTINUOUS mode
    if 'text' in headers_lower:
        return EncodingMode.CONTINUOUS
    
    # Check for QBA mode (has bridge column)
    if 'bridge' in headers_lower:
        return EncodingMode.QBA
    
    # Check for QA mode
    if ('question' in headers_lower and 'answer' in headers_lower) or \
       ('user' in headers_lower and 'response' in headers_lower):
        return EncodingMode.QA
    
    raise ValueError(
        f"Could not detect encoding mode from headers: {headers}\n"
        "Expected one of:\n"
        "  - QA: 'question', 'answer' or 'user', 'response'\n"
        "  - QBA: 'question', 'bridge', 'answer' or 'user', 'bridge', 'response'\n"
        "  - CONTINUOUS: 'text'"
    )


def collect_texts_from_csv(csv_path: str, mode: EncodingMode) -> list:
    """Collect all texts from CSV for vocabulary building"""
    texts = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            if mode == EncodingMode.QA:
                # Try both naming conventions
                q = row.get('question') or row.get('user', '')
                a = row.get('answer') or row.get('response', '')
                texts.extend([q, a])
                
            elif mode == EncodingMode.QBA:
                # Try both naming conventions
                q = row.get('question') or row.get('user', '')
                b = row.get('bridge', '')
                a = row.get('answer') or row.get('response', '')
                texts.extend([q, b, a])
                
            elif mode == EncodingMode.CONTINUOUS:
                texts.append(row.get('text', ''))
    
    return texts


def main():
    parser = argparse.ArgumentParser(description='Train Marina MVec Language Model')
    
    # Data
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to training CSV')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'qa', 'qba', 'continuous'],
                        help='Encoding mode (auto-detect from CSV if not specified)')
    parser.add_argument('--vocab', type=str, default=None, 
                        help='Path to saved vocabulary (if pre-built)')
    parser.add_argument('--chunk-size', type=int, default=None,
                        help='Chunk size for CONTINUOUS mode (optional)')
    
    # Model
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--max-seq-len', type=int, default=128)
    parser.add_argument('--max-delay', type=int, default=64)
    parser.add_argument('--identity-embed-dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--word-loss-weight', type=float, default=1.0)
    parser.add_argument('--end-loss-weight', type=float, default=1.0)
    
    # System
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-path', type=str, default='mvec_marina_model.pt')
    parser.add_argument('--save-vocab', type=str, default='mvec_vocab.json')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Marina MVec Training")
    print("=" * 70)
    
    # Detect or set encoding mode
    print("\n1. Detecting encoding mode...")
    if args.mode == 'auto':
        encoding_mode = detect_encoding_mode(args.data)
        print(f"Auto-detected mode: {encoding_mode.value.upper()}")
    else:
        mode_map = {
            'qa': EncodingMode.QA,
            'qba': EncodingMode.QBA,
            'continuous': EncodingMode.CONTINUOUS
        }
        encoding_mode = mode_map[args.mode]
        print(f"Using specified mode: {encoding_mode.value.upper()}")
    
    # Show mode description
    mode_descriptions = {
        EncodingMode.QA: "Simple Question-Answer (USER → VISIBLE)",
        EncodingMode.QBA: "Question-Bridge-Answer (USER → INTERNAL → VISIBLE)",
        EncodingMode.CONTINUOUS: "Continuous text stream (all VISIBLE)"
    }
    print(f"Mode description: {mode_descriptions[encoding_mode]}")
    
    if encoding_mode == EncodingMode.CONTINUOUS and args.chunk_size:
        print(f"Chunking enabled: {args.chunk_size} tokens per chunk")
    
    # Create/load encoder
    print("\n2. Setting up encoder and vocabulary...")
    encoder = MVecEncoder()
    
    if args.vocab and os.path.exists(args.vocab):
        print(f"Loading vocabulary from {args.vocab}")
        encoder.load_vocab(args.vocab)
    else:
        print(f"Building vocabulary from {args.data}")
        texts = collect_texts_from_csv(args.data, encoding_mode)
        encoder.build_vocab(texts, min_freq=1)
        encoder.save_vocab(args.save_vocab)
        print(f"Vocabulary saved to {args.save_vocab}")
    
    vocab_size = encoder.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    # Create dataset
    print("\n3. Loading dataset...")
    full_dataset = MVecDataset(
        args.data, 
        encoder, 
        max_seq_len=args.max_seq_len,
        mode=encoding_mode,
        chunk_size=args.chunk_size
    )
    
    # Split train/val
    val_size = int(args.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_mvec_batch,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_mvec_batch,
        num_workers=0
    )
    
    # Create model
    print("\n4. Creating model...")
    from tbt_architecture import create_exponential_delays
    
    model = MVecLanguageModel(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        delays=create_exponential_delays(args.max_delay),
        dropout=args.dropout,
        use_identity_embed=True,
        identity_embed_dim=args.identity_embed_dim,
        tie_weights=False  # Can't tie with identity channel
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Device: {args.device}")
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.1
    )
    
    # Create trainer
    print("\n5. Training...")
    trainer = MVecTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=args.device,
        gradient_clip=1.0,
        log_interval=10,
        word_loss_weight=args.word_loss_weight,
        end_loss_weight=args.end_loss_weight
    )
    
    # Train
    history = trainer.train(
        num_epochs=args.epochs,
        save_path=args.save_path,
        scheduler=scheduler,
        early_stopping_patience=10
    )
    
    # Results
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Model saved to: {args.save_path}")
    print(f"Vocabulary saved to: {args.save_vocab}")
    print(f"Encoding mode used: {encoding_mode.value.upper()}")
    
    # Save history with metadata
    history_path = args.save_path.replace('.pt', '_history.json')
    history_with_meta = {
        'history': history,
        'metadata': {
            'encoding_mode': encoding_mode.value,
            'chunk_size': args.chunk_size if encoding_mode == EncodingMode.CONTINUOUS else None,
            'vocab_size': vocab_size,
            'model_params': num_params,
            'final_train_loss': history['train'][-1]['total_loss'],
            'final_val_loss': history['val'][-1]['total_loss'],
            'best_val_loss': trainer.best_val_loss
        }
    }
    
    with open(history_path, 'w') as f:
        json.dump(history_with_meta, f, indent=2)
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
