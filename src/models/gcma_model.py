"""
models/gcma_model.py
--------------------
End-to-end Gated Cross-Modal Attention (GCMA) model.

Full pipeline:

  Image   →  VisualEncoder  → f_vis  ──────────────────────────────┐
                                       ↓ (K, V)                     │
  Tabular →  TabularEncoder → f_tab  → CrossModalAttention → f_cross │
                               ↓ (Q)                                 │
                                                                     ↓
                                         GMU(f_vis, f_tab, f_cross) → f_fused
                                                                     ↓
                                               Linear(d, num_classes) → logits

The model exposes `get_gate_values()` after each forward pass so you
can inspect the GMU gate vector z for interpretability.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .visual_encoder import VisualEncoder
from .tabular_encoder import TabularEncoder
from .cross_attention import CrossModalAttention
from .gmu import GatedMultimodalUnit


class GCMAModel(nn.Module):
    """
    Gated Cross-Modal Attention model for multimodal soil classification.

    Args:
        num_continuous          : Number of continuous tabular features.
        num_classes             : Output classes.
        shared_dim              : Shared projection dimension d.
        num_heads               : Attention heads.
        cnn_backbone            : timm backbone name.
        use_spatial_attention   : If True, visual encoder emits spatial tokens.
        hidden_dim              : GRN hidden width.
        num_grn_layers          : Number of GRN blocks.
        dropout                 : Dropout rate.
        pretrained              : Load pretrained CNN weights.
    """

    def __init__(
        self,
        num_continuous: int,
        num_classes: int = 4,
        shared_dim: int = 256,
        num_heads: int = 8,
        cnn_backbone: str = "efficientnet_b0",
        use_spatial_attention: bool = False,
        hidden_dim: int = 128,
        num_grn_layers: int = 2,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()

        # ── Step 1a: Visual Encoder ───────────────────────────────────────────
        self.visual_encoder = VisualEncoder(
            shared_dim=shared_dim,
            backbone=cnn_backbone,
            use_spatial_attention=use_spatial_attention,
            dropout=dropout,
            pretrained=pretrained,
        )

        # ── Step 1b: Tabular Encoder ──────────────────────────────────────────
        self.tabular_encoder = TabularEncoder(
            num_continuous=num_continuous,
            shared_dim=shared_dim,
            hidden_dim=hidden_dim,
            num_grn_layers=num_grn_layers,
            dropout=dropout,
        )

        # ── Step 2: Cross-Modal Attention ─────────────────────────────────────
        self.cross_attention = CrossModalAttention(
            shared_dim=shared_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ── Step 3: Gated Multimodal Unit ─────────────────────────────────────
        self.gmu = GatedMultimodalUnit(
            shared_dim=shared_dim,
            dropout=dropout,
        )

        # ── Classification Head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(shared_dim, num_classes),
        )

        # Internal storage for gate values (set during forward, read externally)
        self._last_gate_values: Optional[torch.Tensor] = None

        # Architectural summary
        self._use_spatial = use_spatial_attention
        self.shared_dim = shared_dim

    def forward(
        self,
        images: torch.Tensor,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images  : (B, 3, H, W)
            tabular : (B, num_continuous)

        Returns:
            logits  : (B, num_classes)
        """
        # ── Step 1 ────────────────────────────────────────────────────────────
        # f_vis: (B, d)        — pooled mode
        #        (B, T, d)     — spatial mode
        f_vis = self.visual_encoder(images)

        # f_tab: (B, d)
        f_tab = self.tabular_encoder(tabular)

        # ── Step 2 ────────────────────────────────────────────────────────────
        # f_cross: (B, d)
        f_cross = self.cross_attention(f_tab, f_vis)

        # In spatial mode, collapse f_vis to (B, d) for the GMU
        # (we pass the pooled mean of the spatial tokens)
        if self._use_spatial and f_vis.dim() == 3:
            f_vis_pooled = f_vis.mean(dim=1)    # (B, d)
        else:
            f_vis_pooled = f_vis

        # ── Step 3 ────────────────────────────────────────────────────────────
        f_fused, z = self.gmu(f_vis_pooled, f_tab, f_cross)

        # Store gate values for external inspection
        self._last_gate_values = z.detach()

        # ── Classification ────────────────────────────────────────────────────
        logits = self.classifier(f_fused)       # (B, num_classes)

        return logits

    def get_gate_values(self) -> Optional[torch.Tensor]:
        """
        Returns the GMU gate vector z from the most recent forward pass.
        Shape: (B, d)  with values in (0, 1).

        High z → model relies more on this fused feature.
        Low z  → model suppresses this channel.
        """
        return self._last_gate_values

    def parameter_count(self) -> dict:
        """Returns parameter counts per sub-module."""
        def count(m):
            return sum(p.numel() for p in m.parameters())

        return {
            "visual_encoder":   count(self.visual_encoder),
            "tabular_encoder":  count(self.tabular_encoder),
            "cross_attention":  count(self.cross_attention),
            "gmu":              count(self.gmu),
            "classifier":       count(self.classifier),
            "total":            count(self),
        }
