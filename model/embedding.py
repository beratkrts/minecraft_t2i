"""
Embedding module for the PixelArt Transformer.

Three learned embedding tables combined:
  - CategoryEmbedding : (345, d_model)   — one token per Quick Draw class
  - PatchEmbedding    : (256, d_model)   — one token per 2x2 patch vocab entry
  - RowEmbedding      : (8,   d_model)   — 2D positional, row in 8x8 grid
  - ColEmbedding      : (8,   d_model)   — 2D positional, col in 8x8 grid

Forward input:
  cat_idx : (B,)    integer category indices 0-344
  tokens  : (B, T)  image patch tokens 0-255, T in [0, 64]

Forward output:
  (B, T+1, d_model)  — category token first, then T patch tokens with 2D position
"""

import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, n_categories: int, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model

        self.category_embed = nn.Embedding(n_categories, d_model)
        self.patch_embed    = nn.Embedding(vocab_size, d_model)
        self.row_embed      = nn.Embedding(8, d_model)
        self.col_embed      = nn.Embedding(8, d_model)

    def forward(self, cat_idx: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        cat_idx : (B,)
        tokens  : (B, T)  — may be empty (T=0) during the first generation step
        returns : (B, T+1, d_model)
        """
        # Category token — no position embedding applied
        cat_emb = self.category_embed(cat_idx).unsqueeze(1)   # (B, 1, d_model)

        if tokens.shape[1] == 0:
            return cat_emb

        # Patch tokens + 2D position
        patch_emb = self.patch_embed(tokens)                   # (B, T, d_model)

        T = tokens.shape[1]
        pos = torch.arange(T, device=tokens.device)
        pos_emb = self.row_embed(pos // 8) + self.col_embed(pos % 8)  # (T, d_model)
        patch_emb = patch_emb + pos_emb.unsqueeze(0)          # (B, T, d_model)

        return torch.cat([cat_emb, patch_emb], dim=1)          # (B, T+1, d_model)
