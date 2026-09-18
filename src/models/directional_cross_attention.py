"""
models/directional_cross_attention.py
------------------------------------
Implements Forward (Chem -> Vision), Reverse (Vision -> Chem),
and Bidirectional (Co-Attention) Cross-Modal Attention with
fine-grained Per-Feature Token granularity.

Granularities:
  - Forward: Q = Chemistry Tokens (B, M, d), K, V = Visual Patches (B, 49, d)
             Attention Map: (B, M, 49) — "Where in the image is this nutrient reflected?"
  - Reverse: Q = Visual Patches (B, 49, d), K, V = Chemistry Tokens (B, M, d)
             Attention Map: (B, 49, M) — "Which nutrient explains this visual patch?"
  - Bidirectional: Jointly computes both directions and fuses representations via projection + LayerNorm.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class DirectionalCrossModalAttention(nn.Module):
    """
    Directional & Bidirectional Cross-Modal Attention Block.

    Args:
        shared_dim: Embedding dimension d (default 256).
        num_heads: Number of attention heads (default 8).
        direction: "forward" | "reverse" | "bidirectional".
        dropout: Dropout rate.
        ffn_mult: FFN expansion multiplier.
    """

    def __init__(
        self,
        shared_dim: int = 256,
        num_heads: int = 8,
        direction: str = "forward",
        dropout: float = 0.1,
        ffn_mult: int = 2,
    ):
        super().__init__()
        assert direction in ("forward", "reverse", "bidirectional"), f"Invalid direction: {direction}"
        self.direction = direction
        self.shared_dim = shared_dim
        self.num_heads = num_heads

        if self.direction in ("forward", "bidirectional"):
            self.attn_fwd = nn.MultiheadAttention(
                embed_dim=shared_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm_fwd_1 = nn.LayerNorm(shared_dim)
            self.ffn_fwd = nn.Sequential(
                nn.Linear(shared_dim, shared_dim * ffn_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(shared_dim * ffn_mult, shared_dim),
                nn.Dropout(dropout),
            )
            self.norm_fwd_2 = nn.LayerNorm(shared_dim)

        if self.direction in ("reverse", "bidirectional"):
            self.attn_rev = nn.MultiheadAttention(
                embed_dim=shared_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm_rev_1 = nn.LayerNorm(shared_dim)
            self.ffn_rev = nn.Sequential(
                nn.Linear(shared_dim, shared_dim * ffn_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(shared_dim * ffn_mult, shared_dim),
                nn.Dropout(dropout),
            )
            self.norm_rev_2 = nn.LayerNorm(shared_dim)

        if self.direction == "bidirectional":
            self.fuse_proj = nn.Sequential(
                nn.Linear(shared_dim * 2, shared_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(shared_dim, shared_dim),
                nn.LayerNorm(shared_dim),
            )

    def forward(
        self,
        t_tokens: torch.Tensor,
        f_vis_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            t_tokens: (B, M, d) or (B, d) — Tabular chemistry token(s).
            f_vis_tokens: (B, 49, d) or (B, d) — Visual spatial patch token(s).

        Returns:
            f_out: (B, d) — Attended multimodal representation.
            attn_weights: Attention map tensor.
        """
        # Ensure 3D tensors: (B, seq_len, d)
        if t_tokens.dim() == 2:
            t_tokens = t_tokens.unsqueeze(1)      # (B, 1, d)
        if f_vis_tokens.dim() == 2:
            f_vis_tokens = f_vis_tokens.unsqueeze(1) # (B, 1, d)

        # ── 1. Forward Path (Chem queries Vision) ─────────────────────────────
        out_fwd = None
        w_fwd = None
        if self.direction in ("forward", "bidirectional"):
            # Q = t_tokens (B, M, d), K,V = f_vis_tokens (B, T, d)
            attn_fwd_out, w_fwd = self.attn_fwd(
                query=t_tokens,
                key=f_vis_tokens,
                value=f_vis_tokens,
            ) # attn_fwd_out: (B, M, d), w_fwd: (B, M, T)
            h_fwd = self.norm_fwd_1(t_tokens + attn_fwd_out)
            h_fwd = self.norm_fwd_2(h_fwd + self.ffn_fwd(h_fwd))
            out_fwd = h_fwd.mean(dim=1) # Pool over M tokens -> (B, d)

        # ── 2. Reverse Path (Vision queries Chem) ─────────────────────────────
        out_rev = None
        w_rev = None
        if self.direction in ("reverse", "bidirectional"):
            # Q = f_vis_tokens (B, T, d), K,V = t_tokens (B, M, d)
            attn_rev_out, w_rev = self.attn_rev(
                query=f_vis_tokens,
                key=t_tokens,
                value=t_tokens,
            ) # attn_rev_out: (B, T, d), w_rev: (B, T, M)
            h_rev = self.norm_rev_1(f_vis_tokens + attn_rev_out)
            h_rev = self.norm_rev_2(h_rev + self.ffn_rev(h_rev))
            out_rev = h_rev.mean(dim=1) # Pool over T spatial patches -> (B, d)

        # ── 3. Return representation based on mode ────────────────────────────
        if self.direction == "forward":
            return out_fwd, w_fwd
        elif self.direction == "reverse":
            return out_rev, w_rev
        else: # bidirectional
            combined = torch.cat([out_fwd, out_rev], dim=-1) # (B, 2d)
            f_bdir = self.fuse_proj(combined)                # (B, d)
            return f_bdir, (w_fwd, w_rev)
