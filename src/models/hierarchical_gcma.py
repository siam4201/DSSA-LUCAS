"""
models/hierarchical_gcma.py
---------------------------
HierarchicalGCMAModel — dual-head variant of GCMAModel.

Architecture:
    Shared backbone  (VisualEncoder + TabularEncoder + CrossModalAttention + GMU)
        ├── coarse_head  →  Linear(d, num_coarse)   [Cropland / Woodland / Grassland]
        └── fine_head    →  Linear(d, num_fine)     [Cereals / OtherCrop / … 7 classes]

Training loss:
    L = alpha * CE(coarse_logits, coarse_labels)
      + (1-alpha) * CE(fine_logits, fine_labels)

Constrained decoding (inference):
    1. Predict coarse group  →  g = argmax(coarse_logits)
    2. Mask fine_logits      →  set logits outside group g to -inf
    3. Predict fine class    →  argmax(masked fine_logits)

This guarantees every fine prediction is consistent with its coarse group.

COARSE_TO_FINE mapping (matches label encoding in config.py / prepare_lc2_labels.py):
    Cropland  (0)  →  {Cereals(0), Other Cropland(1)}
    Woodland  (1)  →  {Broadleaf(2), Coniferous(3)}
    Grassland (2)  →  {Shrubland(4), Managed Grassland(5), Other Grassland(6)}
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .visual_encoder import VisualEncoder
from .tabular_encoder import TabularEncoder
from .cross_attention import CrossModalAttention
from .gmu import GatedMultimodalUnit


# ── Hierarchy definition ──────────────────────────────────────────────────────

COARSE_TO_FINE: dict[int, list[int]] = {
    0: [0, 1],        # Cropland  → Cereals, Other Cropland
    1: [2, 3],        # Woodland  → Broadleaf Woodland, Coniferous Woodland
    2: [4, 5, 6],     # Grassland → Shrubland, Managed Grassland, Other Grassland
}

NUM_COARSE = 3   # Cropland / Woodland / Grassland
NUM_FINE   = 7   # 7 LC2 classes


class HierarchicalGCMAModel(nn.Module):
    """
    Hierarchical Gated Cross-Modal Attention model.

    Args:
        num_continuous        : Number of continuous tabular features.
        num_coarse            : Number of coarse (Level-1) classes.  Default 3.
        num_fine              : Number of fine (Level-2) classes.    Default 7.
        shared_dim            : Shared projection dimension d.
        num_heads             : Attention heads in cross-modal attention.
        cnn_backbone          : timm backbone name.
        use_spatial_attention : If True, visual encoder emits spatial tokens.
        hidden_dim            : GRN hidden width in tabular encoder.
        num_grn_layers        : Number of GRN blocks.
        dropout               : Dropout rate.
        pretrained            : Load pretrained CNN weights.
    """

    def __init__(
        self,
        num_continuous: int,
        num_coarse: int = NUM_COARSE,
        num_fine: int = NUM_FINE,
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

        self.num_coarse = num_coarse
        self.num_fine   = num_fine
        self.shared_dim = shared_dim
        self._use_spatial = use_spatial_attention

        # ── Shared backbone (same as GCMAModel) ──────────────────────────────
        self.visual_encoder = VisualEncoder(
            shared_dim=shared_dim,
            backbone=cnn_backbone,
            use_spatial_attention=use_spatial_attention,
            dropout=dropout,
            pretrained=pretrained,
        )
        self.tabular_encoder = TabularEncoder(
            num_continuous=num_continuous,
            shared_dim=shared_dim,
            hidden_dim=hidden_dim,
            num_grn_layers=num_grn_layers,
            dropout=dropout,
        )
        self.cross_attention = CrossModalAttention(
            shared_dim=shared_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.gmu = GatedMultimodalUnit(
            shared_dim=shared_dim,
            dropout=dropout,
        )

        # ── Dual classification heads ─────────────────────────────────────────
        self.coarse_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(shared_dim, num_coarse),
        )
        self.fine_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(shared_dim, num_fine),
        )

        # Internal storage for gate values
        self._last_gate_values: Optional[torch.Tensor] = None

        # Pre-build the fine-class mask per coarse group for fast constrained decoding
        # mask[g][c] = 0.0 if class c belongs to group g, else -inf
        self._group_masks: list[torch.Tensor] = []
        for g in range(num_coarse):
            mask = torch.full((num_fine,), float("-inf"))
            for c in COARSE_TO_FINE.get(g, []):
                mask[c] = 0.0
            self._group_masks.append(mask)

    # ─────────────────────────────────────────────────────────────────────────

    def _encode(
        self, images: torch.Tensor, tabular: torch.Tensor
    ) -> torch.Tensor:
        """Shared encoding: images + tabular → f_fused (B, d)."""
        f_vis   = self.visual_encoder(images)
        f_tab   = self.tabular_encoder(tabular)
        f_cross = self.cross_attention(f_tab, f_vis)

        if self._use_spatial and f_vis.dim() == 3:
            f_vis_pooled = f_vis.mean(dim=1)
        else:
            f_vis_pooled = f_vis

        f_fused, z = self.gmu(f_vis_pooled, f_tab, f_cross)
        self._last_gate_values = z.detach()
        return f_fused

    def forward(
        self,
        images: torch.Tensor,
        tabular: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images  : (B, 3, H, W)
            tabular : (B, num_continuous)

        Returns:
            coarse_logits : (B, num_coarse)
            fine_logits   : (B, num_fine)
        """
        f_fused = self._encode(images, tabular)
        return self.coarse_head(f_fused), self.fine_head(f_fused)

    @torch.no_grad()
    def constrained_predict(
        self,
        images: torch.Tensor,
        tabular: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hierarchically constrained inference.

        Returns:
            coarse_preds      : (B,) — predicted coarse class
            fine_preds_raw    : (B,) — unconstrained fine predictions
            fine_preds_hier   : (B,) — constrained fine predictions
        """
        self.eval()
        coarse_logits, fine_logits = self.forward(images, tabular)

        coarse_preds   = coarse_logits.argmax(dim=1)
        fine_preds_raw = fine_logits.argmax(dim=1)

        # Apply group mask per sample
        device = fine_logits.device
        masked = fine_logits.clone()
        for i, g in enumerate(coarse_preds.tolist()):
            mask = self._group_masks[g].to(device)
            masked[i] += mask

        fine_preds_hier = masked.argmax(dim=1)
        return coarse_preds, fine_preds_raw, fine_preds_hier

    def get_gate_values(self) -> Optional[torch.Tensor]:
        """Returns the GMU gate vector z from the most recent forward pass."""
        return self._last_gate_values

    def parameter_count(self) -> dict:
        """Returns parameter counts per sub-module."""
        def count(m):
            return sum(p.numel() for p in m.parameters())
        return {
            "visual_encoder":  count(self.visual_encoder),
            "tabular_encoder": count(self.tabular_encoder),
            "cross_attention": count(self.cross_attention),
            "gmu":             count(self.gmu),
            "coarse_head":     count(self.coarse_head),
            "fine_head":       count(self.fine_head),
            "total":           count(self),
        }
