"""
models/spatial_grid_gated_model.py
----------------------------------
Chemical-Conditioned Dense Spatial Gating (No-Pooling Vision-Chemistry Interaction).

Maintains the dense C x H x W (e.g. d x 7 x 7) spatial visual grid throughout the
cross-modal interaction, allowing chemistry to condition spatial patch representations
BEFORE any spatial pooling is applied.

Formulation:
  1. Visual Backbone: V in R^(B x d x H x W)
  2. Tabular/Chemistry: c in R^(B x d) (or per-feature tokens C_tok in R^(B x M x d))
  3. Broadcast: C' in R^(B x d x H x W)
  4. Spatial Gate: G = sigmoid(Conv1x1([V, C'])) in [0, 1]^(B x d x H x W)
  5. Correction: Delta_V = A(V, C) in R^(B x d x H x W)
     - Conv-Gated: Delta_V = Conv1x1(GELU(Conv1x1([V, C'])))
     - Dense Cross-Attn: Delta_V = Reshape(MHA(Q=V_grid, K=C_tok, V=C_tok))
  6. Modulated Grid: V' = V + G * Delta_V
  7. Post-Interaction Global Pool: v'_pooled = AvgPool(V') in R^(B x d)
  8. Final Multimodal Fusion (GMU) & Classification Head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

import timm
from .tabular_encoder import TabularEncoder
from .pgmr_router import PhysicsGuidedRelevanceRouter
from .gmu import GatedMultimodalUnit


class ChemicalSpatialGatingBlock(nn.Module):
    """
    Dense 1x1 Convolutional Spatial Gating on unpooled (B, d, H, W) feature maps.
    
    Computes:
      C' = Broadcast(W_c(c))
      G = sigmoid(Conv1x1([V, C']))
      Delta_V = Conv1x1(GELU(Conv1x1([V, C'])))
      V' = V + G * Delta_V
    """
    def __init__(
        self,
        shared_dim: int = 256,
        dropout: float = 0.1,
        use_channel_gate: bool = True,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.use_channel_gate = use_channel_gate

        # Chemistry projection
        self.chem_proj = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ELU(),
        )

        # Gate generator: [V; C'] -> (B, d or 1, H, W)
        gate_out_dim = shared_dim if use_channel_gate else 1
        self.gate_conv = nn.Sequential(
            nn.Conv2d(shared_dim * 2, shared_dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(shared_dim),
            nn.ELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(shared_dim, gate_out_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Cross-modal correction generator
        self.correction_conv = nn.Sequential(
            nn.Conv2d(shared_dim * 2, shared_dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(shared_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(shared_dim, shared_dim, kernel_size=1, bias=True),
        )

        self.norm = nn.GroupNorm(8, shared_dim)

        # Initialize gate bias to slightly negative to start near identity
        nn.init.constant_(self.gate_conv[-2].bias, -1.0)

    def forward(self, v_grid: torch.Tensor, c_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            v_grid: (B, d, H, W) visual feature map
            c_vec : (B, d) chemistry conditioning vector

        Returns:
            v_mod : (B, d, H, W) modulated visual grid
            gate  : (B, gate_dim, H, W) spatial gate map
        """
        B, d, H, W = v_grid.shape

        # 1. Project & Broadcast chemistry
        c_proj = self.chem_proj(c_vec)                                      # (B, d)
        c_grid = c_proj.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)    # (B, d, H, W)

        # 2. Concatenate
        x_concat = torch.cat([v_grid, c_grid], dim=1)                       # (B, 2d, H, W)

        # 3. Compute spatial gate & correction
        gate = self.gate_conv(x_concat)                                     # (B, d or 1, H, W)
        delta_v = self.correction_conv(x_concat)                            # (B, d, H, W)

        # 4. Gated residual
        v_mod = self.norm(v_grid + gate * delta_v)                          # (B, d, H, W)

        return v_mod, gate


class DenseSpatialCrossAttentionBlock(nn.Module):
    """
    Dense Cross-Attention where unpooled spatial grid cells act as queries
    and per-feature chemistry tokens act as keys/values, followed by spatial residual gating.
    
    Q = V_grid in R^(B x (HW) x d)
    K, V_val = C_tokens in R^(B x M x d)
    Delta_V = MultiheadAttention(Q, K, V_val) -> reshape to (B, d, H, W)
    V' = V + G * Delta_V
    """
    def __init__(
        self,
        shared_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared_dim = shared_dim

        self.attn = nn.MultiheadAttention(
            embed_dim=shared_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Spatial gating from [V_grid; Delta_V]
        self.gate_conv = nn.Sequential(
            nn.Conv2d(shared_dim * 2, shared_dim, kernel_size=1),
            nn.BatchNorm2d(shared_dim),
            nn.ELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(shared_dim, shared_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        self.norm1 = nn.LayerNorm(shared_dim)
        self.norm2 = nn.GroupNorm(8, shared_dim)

        nn.init.constant_(self.gate_conv[-2].bias, -1.0)

    def forward(
        self,
        v_grid: torch.Tensor,
        c_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            v_grid   : (B, d, H, W)
            c_tokens : (B, M, d) per-feature chemistry tokens
        """
        B, d, H, W = v_grid.shape
        num_spatial = H * W

        # Reshape visual grid to query sequence: (B, HW, d)
        v_seq = v_grid.permute(0, 2, 3, 1).reshape(B, num_spatial, d)

        # Cross-Attention: Visual Patches query Chemistry Tokens
        attn_out, _ = self.attn(query=v_seq, key=c_tokens, value=c_tokens) # (B, HW, d)
        attn_out = self.norm1(attn_out)

        # Reshape back to spatial grid: (B, d, H, W)
        delta_v = attn_out.reshape(B, H, W, d).permute(0, 3, 1, 2)

        # Compute spatial gate
        x_concat = torch.cat([v_grid, delta_v], dim=1)                      # (B, 2d, H, W)
        gate = self.gate_conv(x_concat)                                     # (B, d, H, W)

        # Gated residual
        v_mod = self.norm2(v_grid + gate * delta_v)                         # (B, d, H, W)

        return v_mod, gate


class SpatialGridGatedModel(nn.Module):
    """
    End-to-end Multimodal Network with Chemical-Conditioned Dense Spatial Gating.
    
    Modes:
      - "conv_gated": 1x1 conv spatial gating with broadcasted chemistry vector.
      - "dense_cross_attn": Spatial grid queries attend per-feature chemistry tokens.
    """
    def __init__(
        self,
        feature_cols: List[str],
        visible_cols: List[str],
        subsurface_cols: List[str],
        num_classes: int = 7,
        shared_dim: int = 256,
        num_heads: int = 8,
        mode: str = "conv_gated",
        cnn_backbone: str = "efficientnet_b0",
        use_physics_guidance: bool = True,
        hidden_dim: int = 128,
        num_grn_layers: int = 2,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        assert mode in ("conv_gated", "dense_cross_attn")
        self.mode = mode
        self.shared_dim = shared_dim
        self.num_classes = num_classes
        self.feature_cols = list(feature_cols)
        self.use_physics_guidance = use_physics_guidance

        self.visible_indices = [feature_cols.index(col) for col in visible_cols if col in feature_cols]
        self.subsurface_indices = [feature_cols.index(col) for col in subsurface_cols if col in feature_cols]

        # ── 1. Visual Backbone (Unpooled Output) ──────────────────────────────
        self.backbone = timm.create_model(
            cnn_backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",  # Keep spatial grid (B, C, H, W)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.backbone(dummy)
            cnn_out_channels = feat.shape[1]
            self.grid_h = feat.shape[2]
            self.grid_w = feat.shape[3]

        self.vis_proj = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(cnn_out_channels, shared_dim, kernel_size=1),
            nn.GroupNorm(8, shared_dim),
            nn.GELU(),
        )

        # ── 2. Tabular Encoders / Router ──────────────────────────────────────
        if self.use_physics_guidance:
            self.pgmr_router = PhysicsGuidedRelevanceRouter(
                feature_names=feature_cols,
                shared_dim=shared_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            self.tabular_encoder = None
        else:
            self.pgmr_router = None
            self.tabular_encoder = TabularEncoder(
                num_continuous=len(feature_cols),
                shared_dim=shared_dim,
                hidden_dim=hidden_dim,
                num_grn_layers=num_grn_layers,
                dropout=dropout,
            )

        # ── 3. Dense Spatial Cross-Modal Gating Block ─────────────────────────
        if self.mode == "conv_gated":
            self.spatial_gate_block = ChemicalSpatialGatingBlock(
                shared_dim=shared_dim,
                dropout=dropout,
                use_channel_gate=True,
            )
        elif self.mode == "dense_cross_attn":
            self.spatial_gate_block = DenseSpatialCrossAttentionBlock(
                shared_dim=shared_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

        # ── 4. Multimodal Fusion Unit (GMU) ───────────────────────────────────
        self.gmu = GatedMultimodalUnit(shared_dim=shared_dim, dropout=dropout)

        # ── 5. Classifier Head ────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(shared_dim, num_classes),
        )

        self._init_weights()
        self._last_gate_maps: Optional[torch.Tensor] = None

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        images: torch.Tensor,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Unpooled Visual Feature Grid
        feat_map = self.backbone(images)                 # (B, C, H, W)
        v_grid = self.vis_proj(feat_map)                 # (B, d, H, W)
        v_base_pooled = v_grid.mean(dim=[-2, -1])        # (B, d)

        # 2. Tabular Feature Encoding
        if self.use_physics_guidance:
            feature_tokens = []
            for i, enc in enumerate(self.pgmr_router.feature_encoders):
                feat_val = tabular[:, i:i+1]
                e_i = enc(feat_val)
                feature_tokens.append(e_i)
            c_tokens = torch.stack(feature_tokens, dim=1) # (B, M, d)
            c_vec = c_tokens.mean(dim=1)                 # (B, d)
        else:
            c_vec = self.tabular_encoder(tabular)        # (B, d)
            c_tokens = c_vec.unsqueeze(1)                # (B, 1, d)

        # 3. Dense Spatial Cross-Modal Interaction (No pooling prior to interaction)
        if self.mode == "conv_gated":
            v_mod_grid, gate_map = self.spatial_gate_block(v_grid, c_vec)
        else:
            v_mod_grid, gate_map = self.spatial_gate_block(v_grid, c_tokens)

        self._last_gate_maps = gate_map.detach()

        # 4. Post-Interaction Global Spatial Average Pooling
        v_mod_pooled = v_mod_grid.mean(dim=[-2, -1])     # (B, d)

        # 5. GMU Fusion: Base Visual, Chemistry, and Spatially Modulated Visual
        f_fused, z = self.gmu(v_base_pooled, c_vec, v_mod_pooled)

        # 6. Classification Head
        logits = self.classifier(f_fused)
        return logits

    def get_gate_maps(self) -> Optional[torch.Tensor]:
        return self._last_gate_maps
