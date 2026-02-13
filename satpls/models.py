"""Neural network models: Set Transformer and Deep Sets for satellite scheduling.

References:
    Set Transformer: Lee et al., ICML 2019
    Deep Sets: Zaheer et al., NeurIPS 2017
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Set Transformer building blocks
# ---------------------------------------------------------------------------

class MultiheadAttentionBlock(nn.Module):
    """MAB(X, Y) = LayerNorm(H + FFN(H)) where H = LayerNorm(X + Attn(X,Y,Y))"""

    def __init__(self, dim: int, heads: int = 4, ff_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        h = self.norm1(x + self.attn(x, y, y, need_weights=False)[0])
        return self.norm2(h + self.ffn(h))


class SetAttentionBlock(nn.Module):
    """SAB(X) = MAB(X, X) -- self-attention among set elements."""

    def __init__(self, dim: int, heads: int = 4, ff_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mab = MultiheadAttentionBlock(dim, heads, ff_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.mab(x, x)


class PoolingByMultiheadAttention(nn.Module):
    """PMA: learnable seed vectors attend to set elements."""

    def __init__(self, dim: int, num_seeds: int = 1, heads: int = 4,
                 ff_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(num_seeds, dim))
        self.mab = MultiheadAttentionBlock(dim, heads, ff_dim, dropout)

    def forward(self, z: Tensor) -> Tensor:
        seeds = self.seeds.unsqueeze(0).expand(z.shape[0], -1, -1)
        return self.mab(seeds, z)


# ---------------------------------------------------------------------------
# Set Transformer Scheduler
# ---------------------------------------------------------------------------

class SetTransformerScheduler(nn.Module):
    """Set Transformer for satellite scheduling.

    Architecture: Input embed -> SAB stack -> PMA pool -> decision head
    """

    def __init__(
        self,
        local_dim: int,
        global_dim: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        num_pma_seeds: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.embed_dim = embed_dim

        self.input_embed = nn.Sequential(nn.Linear(local_dim, embed_dim), nn.ReLU())
        self.encoder = nn.ModuleList([
            SetAttentionBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.pma = PoolingByMultiheadAttention(embed_dim, num_pma_seeds, num_heads, ff_dim, dropout)
        self.head = nn.Sequential(
            nn.Linear(embed_dim + embed_dim + global_dim, ff_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, 1),
        )

    def forward(self, local: Tensor, global_: Tensor) -> Tensor:
        """
        Args:
            local: (batch, num_sats, local_dim)
            global_: (batch, global_dim)
        Returns:
            logits: (batch, num_sats)
        """
        B, N, _ = local.shape
        z = self.input_embed(local)
        for sab in self.encoder:
            z = sab(z)

        pooled = self.pma(z).mean(dim=1, keepdim=True).expand(B, N, self.embed_dim)
        g = global_.unsqueeze(1).expand(B, N, self.global_dim)
        return self.head(torch.cat([z, pooled, g], dim=-1)).squeeze(-1)

    def predict_topk(self, local: Tensor, global_: Tensor, k: int) -> Tensor:
        """Return binary mask selecting top-k satellites."""
        logits = self.forward(local, global_)
        _, idx = torch.topk(logits, k, dim=1)
        mask = torch.zeros_like(logits)
        mask.scatter_(1, idx, 1.0)
        return mask


# ---------------------------------------------------------------------------
# Deep Sets Scheduler
# ---------------------------------------------------------------------------

class DeepSetsScheduler(nn.Module):
    """Deep Sets baseline: encode -> pool -> decide."""

    def __init__(
        self,
        local_dim: int,
        global_dim: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(local_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim + embed_dim + global_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, local: Tensor, global_: Tensor) -> Tensor:
        B, N, L = local.shape
        emb = self.encoder(local.view(B * N, L)).view(B, N, self.embed_dim)
        pooled = emb.mean(dim=1, keepdim=True).expand(B, N, self.embed_dim)
        g = global_.unsqueeze(1).expand(B, N, self.global_dim)
        return self.head(torch.cat([emb, pooled, g], dim=-1)).squeeze(-1)

    def predict_topk(self, local: Tensor, global_: Tensor, k: int) -> Tensor:
        logits = self.forward(local, global_)
        _, idx = torch.topk(logits, k, dim=1)
        mask = torch.zeros_like(logits)
        mask.scatter_(1, idx, 1.0)
        return mask
