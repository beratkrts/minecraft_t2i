"""
Full GPT-style autoregressive transformer for pixel-art generation.

Architecture:
  Embeddings  →  N × TransformerBlock  →  LayerNorm  →  Linear head (vocab_size)

TransformerBlock uses pre-norm (LayerNorm before attention and FFN),
residual connections, and ReLU activation in the FFN (integer-friendly for
Minecraft redstone inference after int8 quantization).

Forward returns logits of shape (B, T+1, vocab_size) where:
  logits[:, i, :] is the prediction for image token i (given category + tokens 0..i-1)
  logits[:, T, :] is unused during training (no target past the last token)
"""

import torch
import torch.nn as nn

from model.embedding import Embeddings
from model.attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class PixelArtTransformer(nn.Module):
    def __init__(
        self,
        n_categories: int = 345,
        vocab_size:    int = 16,
        d_model:       int = 64,
        n_heads:       int = 4,
        n_layers:      int = 4,
        d_ff:          int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        self.embeddings = Embeddings(n_categories, vocab_size, d_model)
        self.blocks     = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, cat_idx: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        cat_idx : (B,)      integer category indices
        tokens  : (B, T)    image patch tokens, T in [0, 64]

        returns : (B, T+1, vocab_size)
          logits[:, i, :] predicts image token i
          use logits[:, :T, :] vs tokens for cross-entropy loss
        """
        x = self.embeddings(cat_idx, tokens)   # (B, T+1, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)                    # (B, T+1, vocab_size)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
