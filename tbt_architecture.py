"""
Takens-Based Transformer Architecture: tbt_architecture.py
Attention-free sequence model using delay-coordinate embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from takens_embedding import TakensEmbedding, AdaptiveTakensEmbedding, create_exponential_delays


class TBTFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TBTLayer(nn.Module):
    def __init__(self, dim: int, ff_hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = TBTFeedForward(dim, ff_hidden_dim, dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class TakensTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 6,
        delays: Optional[list] = None,
        ff_hidden_multiplier: int = 4,
        dropout: float = 0.1,
        use_adaptive_takens: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if use_adaptive_takens:
            self.takens_embed = AdaptiveTakensEmbedding(
                embedding_dim=input_dim,
                delays=delays,
                output_dim=hidden_dim,
                dropout=dropout
            )
        else:
            self.takens_embed = TakensEmbedding(embedding_dim=input_dim, delays=delays)
            takens_dim = self.takens_embed.get_output_dim()
            self.takens_projection = nn.Linear(takens_dim, hidden_dim)
        self.use_adaptive_takens = use_adaptive_takens

        self.layers = nn.ModuleList([
            TBTLayer(dim=hidden_dim, ff_hidden_dim=hidden_dim * ff_hidden_multiplier, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_adaptive_takens:
            x = self.takens_embed(x)
        else:
            grid = self.takens_embed(x)
            x = self.takens_embed.flatten_grid(grid)
            x = self.takens_projection(x)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class TBTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 6,
        max_seq_len: int = 8192,
        delays: Optional[list] = None,
        dropout: float = 0.1,
        tie_weights: bool = True,
        use_positional: bool = False #True   # ← NEW: controlled from CLI
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.use_positional = use_positional

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        if use_positional:
            self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        self.dropout = nn.Dropout(dropout)

        if delays is None:
            delays = create_exponential_delays(128)

        self.tbt = TakensTransformer(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            delays=delays,
            dropout=dropout,
            use_adaptive_takens=True
        )

        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        if tie_weights:
            assert embed_dim == hidden_dim
            self.output_proj.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        if self.use_positional:
            nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        tok_emb = self.token_embed(input_ids)

        x = tok_emb
        
        ## remove to get rid of positional ecodings    
        if self.use_positional:
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_emb = self.pos_embed(positions)
            x = x + pos_emb
        x = self.dropout(x)
        ##
        hidden = self.tbt(x)
        logits = self.output_proj(hidden)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), labels.reshape(-1), ignore_index=-100)

        return logits, loss

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0, top_k: Optional[int] = None):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = self.forward(input_ids)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# TBTTimeSeriesModel unchanged — keep your original
class TBTTimeSeriesModel(nn.Module):
    # ... (keep exactly as you had it)
    pass


if __name__ == "__main__":
    # ... (keep your test code)
    pass