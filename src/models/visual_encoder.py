"""
models/visual_encoder.py
------------------------
Step 1a: Visual Modality Encoder

  f_vis = W_vis · φ_vis(x_img)  ∈ ℝ^d

Two operating modes controlled by `use_spatial_attention`:

  Pooled mode (use_spatial_attention=False):
    → CNN backbone → global average pool → Linear → LayerNorm → (B, d)

  Spatial mode (use_spatial_attention=True):
    → CNN backbone → feature map → flatten spatial dims
      → Linear → LayerNorm → (B, num_tokens, d)
    Provides a sequence of spatial patch tokens for richer cross-attention.

NOTE: Switching between modes requires no changes to cross_attention.py or
      gmu.py — the cross-attention module handles both a (B, d) and a
      (B, T, d) key/value sequence transparently.
"""

import torch
import torch.nn as nn
import timm
from typing import Tuple, Union


class VisualEncoder(nn.Module):
    """
    Lightweight CNN visual encoder with optional spatial token output.

    Args:
        shared_dim          : Target projection dimension d.
        backbone            : timm model name (e.g., 'efficientnet_b0').
        use_spatial_attention: If True, returns spatial tokens instead of
                              a single pooled vector.
        dropout             : Dropout applied before the projection.
        pretrained          : Load ImageNet-pretrained weights.
    """

    def __init__(
        self,
        shared_dim: int = 256,
        backbone: str = "efficientnet_b0",
        use_spatial_attention: bool = False,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.use_spatial_attention = use_spatial_attention
        self.shared_dim = shared_dim

        # ── Backbone ──────────────────────────────────────────────────────────
        # num_classes=0 removes the classification head.
        # global_pool="" disables global pooling so we can access the feature map.
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="" if use_spatial_attention else "avg",
        )

        # Determine backbone output channel count via a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.backbone(dummy)
            if use_spatial_attention:
                # feat shape: (1, C, H, W)
                cnn_out_dim = feat.shape[1]     # channels
                self._spatial_h = feat.shape[2]
                self._spatial_w = feat.shape[3]
                self.num_tokens = feat.shape[2] * feat.shape[3]
            else:
                # feat shape: (1, C)  after global avg pool
                cnn_out_dim = feat.shape[1]
                self.num_tokens = 1

        # ── Projection: W_vis ─────────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(cnn_out_dim, shared_dim)
        self.norm = nn.LayerNorm(shared_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input images

        Returns:
            Pooled mode   → (B, d)
            Spatial mode  → (B, num_tokens, d)
        """
        feat = self.backbone(x)     # (B, C) or (B, C, H, W)

        if self.use_spatial_attention:
            B, C, H, W = feat.shape
            # Flatten spatial dims: (B, H*W, C)
            feat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
            feat = self.dropout(feat)
            out = self.norm(self.proj(feat))    # (B, num_tokens, d)
        else:
            feat = self.dropout(feat)
            out = self.norm(self.proj(feat))    # (B, d)

        return out
