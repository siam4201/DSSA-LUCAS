"""
models/cross_attention.py
--------------------------
Step 2: Cross-Modal Attention (CMA)

  f_cross = MultiHead(Q=f_tab·W_Q,  K=f_vis·W_K,  V=f_vis·W_V)

CRITICAL — Directionality:
  • Tabular features  → Query  (Q) : "what do I want to find?"
  • Visual features   → Key    (K) : "what is available?"
  • Visual features   → Value  (V) : "what to retrieve?"

This forces the tabular representation to selectively gate the visual
features.  If the tabular Query signals high moisture, it will
mathematically down-weight the visual Keys associated with dark organic
matter, resolving visual ambiguity.

Handles both modes from the visual encoder:
  Pooled mode  → f_vis: (B, d)       → unsqueezed to (B, 1, d)
  Spatial mode → f_vis: (B, T, d)    → used directly as T key tokens

A standard Transformer decoder sub-block (attention + FFN + residuals)
is used for robustness.
"""

import torch
import torch.nn as nn
from typing import Optional


class CrossModalAttention(nn.Module):
    """
    Multi-head Cross-Modal Attention block.

    Tabular = Query, Visual = Key & Value.
    Includes:
      - Multi-head attention with projection matrices W_Q, W_K, W_V (built
        into nn.MultiheadAttention).
      - Residual connection on the query (f_tab).
      - LayerNorm.
      - Small FFN (2-layer MLP) mirroring a Transformer decoder block.

    Args:
        shared_dim  : Feature dimension d (Q, K, V all in ℝ^d).
        num_heads   : Number of attention heads.
        dropout     : Attention and FFN dropout.
        ffn_mult    : FFN hidden width = ffn_mult × shared_dim.
    """

    def __init__(
        self,
        shared_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_mult: int = 2,
    ):
        super().__init__()

        # nn.MultiheadAttention expects (seq_len, batch, d) by default.
        # batch_first=True makes it (batch, seq_len, d) — cleaner.
        self.attn = nn.MultiheadAttention(
            embed_dim=shared_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Post-attention residual & norm on the query
        self.norm1 = nn.LayerNorm(shared_dim)

        # FFN
        ffn_dim = shared_dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(shared_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, shared_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(shared_dim)

    def forward(
        self,
        f_tab: torch.Tensor,
        f_vis: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            f_tab: (B, d)      — tabular features (Query)
            f_vis: (B, d)      — pooled visual features (Key & Value)
                OR (B, T, d)   — spatial patch tokens (Key & Value)
            key_padding_mask   : Optional (B, T) bool mask for padding tokens.

        Returns:
            f_cross: (B, d)    — attended, fused representation
        """
        # Ensure query has sequence dimension: (B, 1, d)
        if f_tab.dim() == 2:
            q = f_tab.unsqueeze(1)          # (B, 1, d)
        else:
            q = f_tab

        # Ensure key/value has sequence dimension: (B, T, d)
        if f_vis.dim() == 2:
            kv = f_vis.unsqueeze(1)         # (B, 1, d)
        else:
            kv = f_vis                      # (B, T, d)

        # Cross-attention: Q=tabular, K=V=visual
        attn_out, _ = self.attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
        )                                   # (B, 1, d)

        # Squeeze back to (B, d) + residual from original query
        attn_out = attn_out.squeeze(1)      # (B, d)
        x = self.norm1(f_tab + attn_out)    # residual on f_tab

        # FFN block
        f_cross = self.norm2(x + self.ffn(x))

        return f_cross                      # (B, d)
