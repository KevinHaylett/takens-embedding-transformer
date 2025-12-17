"""
MVec Language Model: mvec_model.py
Extends TBTLanguageModel with identity-aware multi-channel output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
sys.path.append('/mnt/user-data/uploads')

from tbt_architecture import TakensTransformer, create_exponential_delays


class MVecLanguageModel(nn.Module):
    """
    Marina's language model with multi-channel encoding.
    
    Extends TBT architecture to:
    1. Accept identity channel as additional input
    2. Output both word predictions AND end predictions
    """
    
    def __init__(
        self,
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
    ):
        """
        Args:
            vocab_size: Size of word vocabulary
            embed_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
            delays: Takens delay structure
            dropout: Dropout probability
            tie_weights: Whether to tie input/output embeddings
            use_identity_embed: Whether to use identity channel
            identity_embed_dim: Dimension of identity embeddings
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.use_identity_embed = use_identity_embed
        self.identity_embed_dim = identity_embed_dim
        
        # Word embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Identity embeddings (USER=0, MARINA_INTERNAL=1, MARINA_VISIBLE=2)
        if use_identity_embed:
            self.identity_embed = nn.Embedding(3, identity_embed_dim)
            # Input to TBT is word + identity
            tbt_input_dim = embed_dim + identity_embed_dim
        else:
            tbt_input_dim = embed_dim
        
        self.dropout = nn.Dropout(dropout)
        
        # Takens-based transformer
        if delays is None:
            delays = create_exponential_delays(64)
        
        self.tbt = TakensTransformer(
            input_dim=tbt_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            delays=delays,
            dropout=dropout,
            use_adaptive_takens=True
        )
        
        # Output projections
        self.word_output = nn.Linear(hidden_dim, vocab_size)
        self.end_output = nn.Linear(hidden_dim, 2)  # Binary: NO=0, YES=1
        
        # Optional weight tying
        if tie_weights and not use_identity_embed:
            assert embed_dim == hidden_dim, "embed_dim must equal hidden_dim for weight tying"
            self.word_output.weight = self.token_embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.token_embed.weight, std=0.02)
        if self.use_identity_embed:
            nn.init.normal_(self.identity_embed.weight, std=0.02)
        nn.init.normal_(self.word_output.weight, std=0.02)
        nn.init.normal_(self.end_output.weight, std=0.02)
        if self.word_output.bias is not None:
            nn.init.zeros_(self.word_output.bias)
        if self.end_output.bias is not None:
            nn.init.zeros_(self.end_output.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        identity_ids: Optional[torch.Tensor] = None,
        word_labels: Optional[torch.Tensor] = None,
        end_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len] - Word token IDs
            identity_ids: [batch, seq_len] - Identity channel (USER/INTERNAL/VISIBLE)
            word_labels: [batch, seq_len] - Target word IDs (for loss)
            end_labels: [batch, seq_len] - Target end signals (for loss)
            
        Returns:
            word_logits: [batch, seq_len, vocab_size]
            end_logits: [batch, seq_len, 2]
            word_loss: Scalar (if labels provided)
            end_loss: Scalar (if labels provided)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get word embeddings
        tok_emb = self.token_embed(input_ids)  # [B, L, embed_dim]
        
        # Add identity embeddings if enabled
        if self.use_identity_embed:
            if identity_ids is None:
                # Default to MARINA_VISIBLE if not provided
                identity_ids = torch.full((batch_size, seq_len), 2, dtype=torch.long, device=device)
            
            id_emb = self.identity_embed(identity_ids)  # [B, L, identity_embed_dim]
            x = torch.cat([tok_emb, id_emb], dim=-1)  # [B, L, embed_dim + identity_embed_dim]
        else:
            x = tok_emb
        
        x = self.dropout(x)
        
        # Pass through TBT
        hidden = self.tbt(x)  # [B, L, hidden_dim]
        
        # Dual output heads
        word_logits = self.word_output(hidden)  # [B, L, vocab_size]
        end_logits = self.end_output(hidden)    # [B, L, 2]
        
        # Compute losses if labels provided
        word_loss = None
        end_loss = None
        
        if word_labels is not None:
            word_loss = F.cross_entropy(
                word_logits.reshape(-1, self.vocab_size),
                word_labels.reshape(-1),
                ignore_index=-100
            )
        
        if end_labels is not None:
            end_loss = F.cross_entropy(
                end_logits.reshape(-1, 2),
                end_labels.reshape(-1),
                ignore_index=-100
            )
        
        return word_logits, end_logits, word_loss, end_loss
    
    def generate(
        self,
        input_ids: torch.Tensor,
        identity_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        stop_on_end: bool = True,
        generate_identity: int = 2  # Default: MARINA_VISIBLE
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate text with end signal detection.
        
        Args:
            input_ids: [1, seq_len] - Input token IDs
            identity_ids: [1, seq_len] - Identity channel for input
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None = no filtering)
            stop_on_end: Whether to stop when end=YES is predicted
            generate_identity: Identity to use for generated tokens
            
        Returns:
            generated_ids: [1, total_len] - Generated sequence
            generated_identities: [1, total_len] - Identity channel
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Initialize identity if not provided
        if identity_ids is None:
            identity_ids = torch.full_like(input_ids, 2)  # Default to VISIBLE
        
        generated_ids = input_ids.clone()
        generated_identities = identity_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                word_logits, end_logits, _, _ = self.forward(
                    generated_ids,
                    generated_identities
                )
                
                # Get predictions for last position
                next_word_logits = word_logits[:, -1, :] / temperature
                next_end_logits = end_logits[:, -1, :]
                
                # Sample next word
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(next_word_logits, min(top_k, next_word_logits.size(-1)))
                    next_word_logits[next_word_logits < v[:, [-1]]] = -float('inf')
                
                probs = F.softmax(next_word_logits, dim=-1)
                next_word = torch.multinomial(probs, num_samples=1)
                
                # Predict end signal
                end_probs = F.softmax(next_end_logits, dim=-1)
                next_end = torch.argmax(end_probs, dim=-1, keepdim=True)
                
                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_word], dim=1)
                
                next_identity = torch.full_like(next_word, generate_identity)
                generated_identities = torch.cat([generated_identities, next_identity], dim=1)
                
                # Check if we should stop
                if stop_on_end and next_end.item() == 1:  # YES
                    break
        
        return generated_ids, generated_identities


if __name__ == "__main__":
    print("Testing MVecLanguageModel...")
    print("=" * 70)
    
    # Create model
    print("\n1. Creating model...")
    model = MVecLanguageModel(
        vocab_size=1000,
        embed_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        use_identity_embed=True,
        identity_embed_dim=16
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {num_params:,} parameters")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 4
    seq_len = 20
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    identity_ids = torch.randint(0, 3, (batch_size, seq_len))
    word_labels = torch.randint(0, 1000, (batch_size, seq_len))
    end_labels = torch.randint(0, 2, (batch_size, seq_len))
    
    word_logits, end_logits, word_loss, end_loss = model(
        input_ids,
        identity_ids,
        word_labels,
        end_labels
    )
    
    print(f"  Word logits shape: {word_logits.shape}")
    print(f"  End logits shape: {end_logits.shape}")
    print(f"  Word loss: {word_loss.item():.4f}")
    print(f"  End loss: {end_loss.item():.4f}")
    
    # Test generation
    print("\n3. Testing generation...")
    prompt_ids = torch.randint(0, 1000, (1, 10))
    prompt_identity = torch.full((1, 10), 0)  # USER
    
    generated_ids, generated_identities = model.generate(
        prompt_ids,
        prompt_identity,
        max_new_tokens=30,
        temperature=1.0,
        stop_on_end=True,
        generate_identity=2  # MARINA_VISIBLE
    )
    
    print(f"  Input length: {prompt_ids.shape[1]}")
    print(f"  Generated length: {generated_ids.shape[1]}")
    print(f"  Total tokens: {generated_ids.shape[1]}")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
