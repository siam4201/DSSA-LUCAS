"""
models/baseline_models.py
-------------------------
Standard baseline architectures for the multimodal comparison table:
  1. TabularOnlyModel : GRN Tabular Encoder -> MLP Classifier (no images)
  2. VisionOnlyModel  : EfficientNet-B0 Visual Encoder -> MLP Classifier (no chemistry)
  3. ConcatFusionModel: Direct early concatenation [f_vis, f_tab] -> MLP (no Attention, no GMU)
"""

import torch
import torch.nn as nn
from typing import Optional

from .visual_encoder import VisualEncoder
from .tabular_encoder import TabularEncoder


class TabularOnlyModel(nn.Module):
    """GRN Tabular Encoder -> Classification Head (No visual input)."""

    def __init__(
        self,
        num_continuous: int = 8,
        num_classes: int = 7,
        shared_dim: int = 256,
        hidden_dim: int = 128,
        num_grn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TabularEncoder(
            num_continuous=num_continuous,
            shared_dim=shared_dim,
            hidden_dim=hidden_dim,
            num_grn_layers=num_grn_layers,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(shared_dim, num_classes),
        )

    def forward(self, images: Optional[torch.Tensor], tabular: torch.Tensor) -> torch.Tensor:
        f_tab = self.encoder(tabular)
        return self.classifier(f_tab)


class VisionOnlyModel(nn.Module):
    """EfficientNet-B0 Visual Encoder -> Classification Head (No tabular input)."""

    def __init__(
        self,
        num_classes: int = 7,
        shared_dim: int = 256,
        cnn_backbone: str = "efficientnet_b0",
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.encoder = VisualEncoder(
            shared_dim=shared_dim,
            backbone=cnn_backbone,
            use_spatial_attention=False,
            dropout=dropout,
            pretrained=pretrained,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(shared_dim, num_classes),
        )

    def forward(self, images: torch.Tensor, tabular: Optional[torch.Tensor] = None) -> torch.Tensor:
        f_vis = self.encoder(images)
        return self.classifier(f_vis)


class ConcatFusionModel(nn.Module):
    """
    Standard Early Concatenation Multimodal Fusion Baseline.
    Concatenates [f_vis, f_tab] -> MLP Classifier (No Cross-Attention, No GMU).
    """

    def __init__(
        self,
        num_continuous: int = 8,
        num_classes: int = 7,
        shared_dim: int = 256,
        cnn_backbone: str = "efficientnet_b0",
        hidden_dim: int = 128,
        num_grn_layers: int = 2,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.visual_encoder = VisualEncoder(
            shared_dim=shared_dim,
            backbone=cnn_backbone,
            use_spatial_attention=False,
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
        self.fusion_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(shared_dim * 2, shared_dim),
            nn.ELU(),
            nn.LayerNorm(shared_dim),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, num_classes),
        )

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        f_vis = self.visual_encoder(images)
        f_tab = self.tabular_encoder(tabular)
        f_concat = torch.cat([f_vis, f_tab], dim=-1)
        return self.fusion_mlp(f_concat)
