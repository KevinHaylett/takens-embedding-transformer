"""
Training Utilities for Takens-Based Transformer: training_utils.py
Includes data loaders, metrics, and training loops.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from tqdm import tqdm
import time


# ============================================================================
# Language Modeling Utilities
# ============================================================================

class TextDataset(Dataset):
    """
    Simple dataset for character or token-level language modeling.
    """
    
    def __init__(
        self,
        text: str,
        vocab: Dict[str, int],
        seq_len: int = 128,
        stride: Optional[int] = None
    ):
        self.text = text
        self.vocab = vocab
        self.seq_len = seq_len
        self.stride = stride or seq_len  # Non-overlapping by default
        
        # Convert text to token ids
        self.tokens = [vocab.get(char, vocab.get('<unk>', 0)) for char in text]
        
        # Calculate number of sequences
        self.num_sequences = max(1, (len(self.tokens) - seq_len) // self.stride)
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_len + 1  # +1 for target
        
        if end_idx > len(self.tokens):
            # Pad if necessary
            sequence = self.tokens[start_idx:] + [0] * (end_idx - len(self.tokens))
        else:
            sequence = self.tokens[start_idx:end_idx]
        
        # Input and target (shifted by 1)
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, labels


def create_vocab_from_text(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create vocabulary from text.
    
    Returns:
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
    """
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char


#def decode_tokens(token_ids: torch.Tensor, idx_to_char: Dict[int, str]) -> str:
  #  """Decode token ids back to text."""
 #   return ''.join([idx_to_char.get(idx.item(), '?') for idx in token_ids])

def decode_tokens(token_ids, idx_to_char):
    """
    Convert token indices back to text.
    Handles both tensor and regular integer inputs.
    """
    if len(token_ids) == 0:
        return ""
    
    # Handle both tensor and regular integer inputs
    if hasattr(token_ids[0], 'item'):  # It's a tensor
        chars = [idx_to_char.get(idx.item(), '?') for idx in token_ids]
    else:  # It's regular integers
        chars = [idx_to_char.get(idx, '?') for idx in token_ids]
    
    return ''.join(chars)




# ============================================================================
# Time Series Utilities
# ============================================================================

class TimeSeriesDataset(Dataset):
    """
    Dataset for time series prediction.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 100,
        prediction_length: int = 1,
        stride: int = 1
    ):
        """
        Args:
            data: [num_timesteps, num_features]
            seq_len: Length of input sequence
            prediction_length: Number of steps to predict
            stride: Stride for creating sequences
        """
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.prediction_length = prediction_length
        self.stride = stride
        
        # Calculate number of sequences
        total_len = seq_len + prediction_length
        self.num_sequences = max(1, (len(data) - total_len) // stride + 1)
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_len
        target_end = end_idx + self.prediction_length
        
        # Input sequence
        x = self.data[start_idx:end_idx]
        
        # Target (next steps)
        y = self.data[end_idx:target_end]
        
        return x, y


def generate_lorenz_attractor(
    num_steps: int = 10000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8/3
) -> np.ndarray:
    """
    Generate Lorenz attractor time series.
    
    Returns:
        Array of shape [num_steps, 3] containing (x, y, z) coordinates
    """
    def lorenz_derivatives(state, sigma, rho, beta):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return np.array([dx, dy, dz])
    
    # Initialize
    state = np.array([1.0, 1.0, 1.0])
    trajectory = np.zeros((num_steps, 3))
    
    # Integrate using Euler method
    for i in range(num_steps):
        trajectory[i] = state
        derivatives = lorenz_derivatives(state, sigma, rho, beta)
        state = state + derivatives * dt
    
    return trajectory


def normalize_timeseries(data: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Normalize time series to zero mean and unit variance.
    
    Returns:
        Normalized data and statistics dict for denormalization
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-8
    normalized = (data - mean) / std
    stats = {'mean': mean, 'std': std}
    return normalized, stats


def denormalize_timeseries(data: np.ndarray, stats: Dict) -> np.ndarray:
    """Denormalize time series data."""
    return data * stats['std'] + stats['mean']


# ============================================================================
# Training Loop
# ============================================================================

class Trainer:
    """
    Generic trainer for TBT models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        gradient_clip: Optional[float] = 1.0,
        log_interval: int = 100
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.gradient_clip = gradient_clip
        self.log_interval = log_interval
        
        # Optimizer (default to AdamW)
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=3e-4,
                betas=(0.9, 0.999),
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            if len(batch) == 2:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
            else:
                x = batch.to(self.device)
                y = None
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if hasattr(self.model, 'forward'):
                if y is not None:
                    _, loss = self.model(x, y)
                else:
                    output = self.model(x)
                    loss = output if isinstance(output, torch.Tensor) else output[1]
            else:
                raise ValueError("Model must have forward method")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Update stats
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate on validation set."""
        if self.val_loader is None:
            return float('nan')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            # Move to device
            if len(batch) == 2:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
            else:
                x = batch.to(self.device)
                y = None
            
            # Forward pass
            if y is not None:
                _, loss = self.model(x, y)
            else:
                output = self.model(x)
                loss = output if isinstance(output, torch.Tensor) else output[1]
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(
        self,
        num_epochs: int,
        save_path: Optional[str] = None,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train for multiple epochs.
        
        Returns:
            Dictionary of training history
        """
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            start_time = time.time()
            train_loss = self.train_epoch()
            epoch_time = time.time() - start_time
            
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f} | Time: {epoch_time:.2f}s")
            
            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                print(f"Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    if save_path:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                        }, save_path)
                        print(f"✓ Saved best model to {save_path}")
                else:
                    patience_counter += 1
                
                # Early stopping
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping after {epoch + 1} epochs")
                    break
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if self.val_loader else train_loss)
                else:
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Learning Rate: {current_lr:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }


# ============================================================================
# Metrics
# ============================================================================

def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss."""
    return np.exp(loss)


def compute_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute mean squared error."""
    return torch.mean((predictions - targets) ** 2).item()


def compute_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute mean absolute error."""
    return torch.mean(torch.abs(predictions - targets)).item()


def compute_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute root mean squared error."""
    return np.sqrt(compute_mse(predictions, targets))


if __name__ == "__main__":
    print("Testing training utilities...\n")
    
    # Test text dataset
    print("=" * 60)
    print("Testing TextDataset")
    print("=" * 60)
    
    sample_text = "hello world! this is a test of the takens transformer."
    vocab, idx_vocab = create_vocab_from_text(sample_text)
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sample characters: {list(vocab.keys())[:10]}")
    
    dataset = TextDataset(sample_text, vocab, seq_len=10)
    print(f"Dataset size: {len(dataset)}")
    
    x, y = dataset[0]
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Input text: '{decode_tokens(x, idx_vocab)}'")
    print(f"Target text: '{decode_tokens(y, idx_vocab)}'")
    
    # Test time series generation
    print("\n" + "=" * 60)
    print("Testing Lorenz Attractor Generation")
    print("=" * 60)
    
    lorenz_data = generate_lorenz_attractor(num_steps=1000)
    print(f"Lorenz data shape: {lorenz_data.shape}")
    print(f"X range: [{lorenz_data[:, 0].min():.2f}, {lorenz_data[:, 0].max():.2f}]")
    print(f"Y range: [{lorenz_data[:, 1].min():.2f}, {lorenz_data[:, 1].max():.2f}]")
    print(f"Z range: [{lorenz_data[:, 2].min():.2f}, {lorenz_data[:, 2].max():.2f}]")
    
    # Test normalization
    normalized, stats = normalize_timeseries(lorenz_data)
    print(f"\nNormalized mean: {normalized.mean(axis=0)}")
    print(f"Normalized std: {normalized.std(axis=0)}")
    
    # Test time series dataset
    ts_dataset = TimeSeriesDataset(lorenz_data, seq_len=50, prediction_length=10)
    print(f"\nTime series dataset size: {len(ts_dataset)}")
    
    x, y = ts_dataset[0]
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
