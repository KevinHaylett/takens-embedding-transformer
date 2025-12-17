"""
Takens Embedding Module: takens_embedding.py
Implements delay-coordinate embeddings based on Takens' theorem
for use in the Takens-Based Transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class TakensEmbedding(nn.Module):
    """
    Constructs Takens delay-coordinate embeddings from a sequence.
    
    For each position t, creates a vector by concatenating the current
    embedding with embeddings from exponentially-spaced past positions:
    [x(t), x(t-τ₁), x(t-τ₂), ..., x(t-τₘ)]
    
    Args:
        embedding_dim: Dimension of input embeddings
        delays: List of delay positions (e.g., [1, 2, 4, 8, 16, 32, 64, 128])
        pad_value: Value to use for positions before sequence start
    """
    
    def __init__(
        self,
        embedding_dim: int,
        delays: Optional[List[int]] = None,
        pad_value: float = 0.0
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.delays = delays if delays is not None else [1, 2, 4, 8, 16, 32, 64, 128]
        self.pad_value = pad_value
        self.num_delays = len(self.delays)
        
        # Output dimension is (num_delays + 1) * embedding_dim
        # +1 because we include the current position x(t)
        self.output_dim = (self.num_delays + 1) * embedding_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Construct Takens embeddings for input sequence.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]
            
        Returns:
            Takens grid of shape [batch_size, seq_len, num_delays+1, embedding_dim]
            Can be reshaped to [batch_size, seq_len, (num_delays+1)*embedding_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embedding_dim, \
            f"Input embedding dim {embed_dim} doesn't match expected {self.embedding_dim}"
        
        # Create grid to hold all delay embeddings
        # Shape: [batch, seq_len, num_delays+1, embed_dim]
        grid = torch.zeros(
            batch_size, 
            seq_len, 
            self.num_delays + 1, 
            embed_dim,
            dtype=x.dtype,
            device=x.device
        )
        
        # First slot is always the current position x(t)
        grid[:, :, 0, :] = x
        
        # Fill in delayed positions
        for delay_idx, delay in enumerate(self.delays, start=1):
            if delay >= seq_len:
                # If delay is larger than sequence, fill with pad_value
                grid[:, :, delay_idx, :] = self.pad_value
            else:
                # Shift sequence by delay positions
                # For positions < delay, pad with pad_value
                if delay > 0:
                    grid[:, delay:, delay_idx, :] = x[:, :-delay, :]
                    grid[:, :delay, delay_idx, :] = self.pad_value
                else:
                    grid[:, :, delay_idx, :] = x
        
        return grid
    
    def flatten_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Flatten the Takens grid along the delay dimension.
        
        Args:
            grid: Shape [batch, seq_len, num_delays+1, embed_dim]
            
        Returns:
            Shape [batch, seq_len, (num_delays+1)*embed_dim]
        """
        batch_size, seq_len, num_delays_plus_one, embed_dim = grid.shape
        return grid.reshape(batch_size, seq_len, -1)
    
    def get_output_dim(self) -> int:
        """Return the output dimension after flattening."""
        return self.output_dim
    
    def extra_repr(self) -> str:
        return f'embedding_dim={self.embedding_dim}, delays={self.delays}, output_dim={self.output_dim}'


class AdaptiveTakensEmbedding(nn.Module):
    """
    Learnable variant of Takens embedding where delay weights are learned.
    
    Instead of simple concatenation, this module learns to weight different
    delays, allowing the model to adaptively focus on relevant time scales.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        delays: Optional[List[int]] = None,
        output_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.delays = delays if delays is not None else [1, 2, 4, 8, 16, 32, 64, 128]
        self.num_delays = len(self.delays)
        
        # Takens embedding dimension before projection
        takens_dim = (self.num_delays + 1) * embedding_dim
        
        # Output dimension (can be compressed from full Takens dimension)
        self.output_dim = output_dim if output_dim is not None else takens_dim
        
        # Base Takens embedding
        self.takens = TakensEmbedding(embedding_dim, delays)
        
        # Learnable projection to compress or transform Takens embedding
        self.projection = nn.Linear(takens_dim, self.output_dim)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embedding_dim]
            
        Returns:
            [batch_size, seq_len, output_dim]
        """
        # Get Takens grid
        grid = self.takens(x)  # [B, L, M+1, D]
        
        # Flatten
        flat = self.takens.flatten_grid(grid)  # [B, L, (M+1)*D]
        
        # Project and normalize
        out = self.projection(flat)  # [B, L, output_dim]
        out = self.layer_norm(out)
        out = self.dropout(out)
        
        return out
    
    def get_output_dim(self) -> int:
        return self.output_dim


def create_exponential_delays(max_delay: int, base: int = 2) -> List[int]:
    """
    Create exponentially-spaced delays: [1, 2, 4, 8, 16, ...]
    
    Args:
        max_delay: Maximum delay value
        base: Base for exponential spacing (default 2)
        
    Returns:
        List of integer delays
    """
    delays = []
    delay = 1
    while delay <= max_delay:
        delays.append(delay)
        delay *= base
    return delays


def create_logarithmic_delays(max_delay: int, num_delays: int) -> List[int]:
    """
    Create logarithmically-spaced delays.
    
    Args:
        max_delay: Maximum delay value
        num_delays: Number of delays to create
        
    Returns:
        List of integer delays
    """
    import numpy as np
    delays = np.logspace(0, np.log10(max_delay), num_delays, dtype=int)
    delays = sorted(list(set(delays)))  # Remove duplicates and sort
    return delays


if __name__ == "__main__":
    # Test the Takens embedding
    print("Testing TakensEmbedding...")
    
    batch_size = 4
    seq_len = 100
    embed_dim = 64
    
    # Create random input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Test standard Takens embedding
    takens = TakensEmbedding(embedding_dim=embed_dim)
    grid = takens(x)
    print(f"Input shape: {x.shape}")
    print(f"Grid shape: {grid.shape}")
    print(f"Expected: [{batch_size}, {seq_len}, {len(takens.delays)+1}, {embed_dim}]")
    
    flat = takens.flatten_grid(grid)
    print(f"Flattened shape: {flat.shape}")
    print(f"Output dimension: {takens.get_output_dim()}")
    
    # Test adaptive Takens embedding
    print("\nTesting AdaptiveTakensEmbedding...")
    adaptive_takens = AdaptiveTakensEmbedding(
        embedding_dim=embed_dim,
        output_dim=256
    )
    out = adaptive_takens(x)
    print(f"Output shape: {out.shape}")
    print(f"Expected: [{batch_size}, {seq_len}, 256]")
    
    # Test delay generation
    print("\nTesting delay generation...")
    exp_delays = create_exponential_delays(128, base=2)
    print(f"Exponential delays (max=128): {exp_delays}")
    
    log_delays = create_logarithmic_delays(100, num_delays=8)
    print(f"Logarithmic delays (max=100, n=8): {log_delays}")
    
    print("\nAll tests passed! ✓")
