"""
Causal multi-head self-attention.

Config:  d_model=64, n_heads=4  →  head_dim=16
Mask:    lower-triangular (each position only attends to itself and earlier positions)

Uses torch.nn.functional.scaled_dot_product_attention (PyTorch ≥ 2.0) with
is_causal=True — avoids materialising the full T×T mask tensor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        # Fused QKV projection — no bias (saves params, common in small transformers)
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, d_model)
        returns : (B, T, d_model)
        """
        B, T, C = x.shape

        # Project and split into Q, K, V
        qkv = self.qkv(x)                                         # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)                             # each (B, T, C)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask (flash-attention path when available)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, n_heads, T, head_dim)

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)      # (B, T, C)
        return self.out_proj(out)
