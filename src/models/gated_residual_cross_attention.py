"""
models/gated_residual_cross_attention.py
----------------------------------------
Implements Gated Residual Cross-Modal Attention:
    G = sigmoid(W_g [V_base; C_context])
    V_out = V_base + G * A(C, V)

Guarantees:
  1. Zero-Degradation Lower Bound: When G -> 0, V_out = V_base (Vision-Only baseline is preserved).
  2. Additive Chemistry Correction: Chemistry only injects targeted delta adjustments where confident.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class GatedResidualCrossModalAttention(nn.Module):
    """
    Gated Residual Cross-Modal Attention Block.

    Formula:
        V' = V + G * Attention(Q, K, V)
        where G = sigmoid(Linear([V_base ; C_context]))

    Args:
        shared_dim: Feature dimension d (default 256).
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
        assert direction in ("forward", "reverse", "bidirectional")
        self.direction = direction
        self.shared_dim = shared_dim

        # 1. Attention Engines
        if self.direction in ("forward", "bidirectional"):
            self.attn_fwd = nn.MultiheadAttention(
                embed_dim=shared_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm_fwd = nn.LayerNorm(shared_dim)
            self.ffn_fwd = nn.Sequential(
                nn.Linear(shared_dim, shared_dim * ffn_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(shared_dim * ffn_mult, shared_dim),
            )

        if self.direction in ("reverse", "bidirectional"):
            self.attn_rev = nn.MultiheadAttention(
                embed_dim=shared_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm_rev = nn.LayerNorm(shared_dim)
            self.ffn_rev = nn.Sequential(
                nn.Linear(shared_dim, shared_dim * ffn_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(shared_dim * ffn_mult, shared_dim),
            )

        if self.direction == "bidirectional":
            self.fuse_proj = nn.Sequential(
                nn.Linear(shared_dim * 2, shared_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(shared_dim, shared_dim),
            )

        # 2. Gating Mechanism: G = sigmoid(W_g [V ; C])
        self.gate_mlp = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, shared_dim),
            nn.Sigmoid(),
        )
        
        # Initialize gate bias to slightly negative so training starts near identity (pure vision)
        nn.init.constant_(self.gate_mlp[-2].bias, -1.0)

        # Output normalization
        self.out_norm = nn.LayerNorm(shared_dim)

    def forward(
        self,
        t_tokens: torch.Tensor,
        f_vis_tokens: torch.Tensor,
        v_base_pooled: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            t_tokens: (B, M, d) or (B, d) — Tabular chemistry token(s).
            f_vis_tokens: (B, 49, d) or (B, d) — Visual spatial patch tokens.
            v_base_pooled: (B, d) — Original pooled visual representation (Base Evidence).

        Returns:
            v_out: (B, d) — Refined visual representation: V_base + G * A(C, V)
            gate: (B, d) — Learned per-dimension gate values.
        """
        if t_tokens.dim() == 2:
            t_tokens = t_tokens.unsqueeze(1)
        if f_vis_tokens.dim() == 2:
            f_vis_tokens = f_vis_tokens.unsqueeze(1)

        t_pooled = t_tokens.mean(dim=1) # (B, d)

        # ── Step 1: Compute Cross-Modal Signal A(C, V) ────────────────────────
        if self.direction == "forward":
            attn_out, _ = self.attn_fwd(query=t_tokens, key=f_vis_tokens, value=f_vis_tokens)
            attn_sig = self.norm_fwd(self.ffn_fwd(attn_out).mean(dim=1))
        elif self.direction == "reverse":
            attn_out, _ = self.attn_rev(query=f_vis_tokens, key=t_tokens, value=t_tokens)
            attn_sig = self.norm_rev(self.ffn_rev(attn_out).mean(dim=1))
        else: # bidirectional
            fwd_out, _ = self.attn_fwd(query=t_tokens, key=f_vis_tokens, value=f_vis_tokens)
            rev_out, _ = self.attn_rev(query=f_vis_tokens, key=t_tokens, value=t_tokens)
            fwd_sig = self.norm_fwd(self.ffn_fwd(fwd_out).mean(dim=1))
            rev_sig = self.norm_rev(self.ffn_rev(rev_out).mean(dim=1))
            attn_sig = self.fuse_proj(torch.cat([fwd_sig, rev_sig], dim=-1))

        # ── Step 2: Compute Adaptive Gate G in (0, 1)^d ────────────────────────
        gate_input = torch.cat([v_base_pooled, t_pooled], dim=-1) # (B, 2d)
        gate = self.gate_mlp(gate_input)                          # (B, d)

        # ── Step 3: Gated Residual Visual Refinement ──────────────────────────
        # V' = V_base + G * Attention(C, V)
        v_out = self.out_norm(v_base_pooled + gate * attn_sig)    # (B, d)

        return v_out, gate
