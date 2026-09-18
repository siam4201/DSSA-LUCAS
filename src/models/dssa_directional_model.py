"""
models/dssa_directional_model.py
--------------------------------
DSSA Multimodal Architecture supporting:
  1. Attention Directions:
     - "forward":       Chemistry Queries Visual Spatial Patches (M x 49)
     - "reverse":       Visual Spatial Patches Query Per-Feature Chemistry Tokens (49 x M)
     - "bidirectional": Joint Forward + Reverse Cross-Modal Co-Attention
  2. Gated Residual Cross-Modal Modulation (use_residual_gate=True):
     - V' = V_base + G * A(C, V)
     - G = sigmoid(W_g [V_base; C])
     - Ensures a mathematical Zero-Degradation Floor (lower bounded by Vision-Only).
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple

from .visual_encoder import VisualEncoder
from .spatial_adapter import SpatialDecompositionAdapter
from .pgmr_router import PhysicsGuidedRelevanceRouter
from .tabular_encoder import TabularEncoder
from .directional_cross_attention import DirectionalCrossModalAttention
from .gated_residual_cross_attention import GatedResidualCrossModalAttention
from .gmu import GatedMultimodalUnit


class DSSADirectionalModel(nn.Module):
    """
    Directional DSSA Multimodal Network with per-feature token cross-attention
    and Gated Residual visual modulation.
    """

    def __init__(
        self,
        feature_cols: List[str],
        visible_cols: List[str],
        subsurface_cols: List[str],
        num_classes: int = 7,
        shared_dim: int = 256,
        num_heads: int = 8,
        attn_direction: str = "forward",
        use_residual_gate: bool = True,
        cnn_backbone: str = "efficientnet_b0",
        use_spatial_attention: bool = True,
        use_physics_guidance: bool = True,
        use_soil_gating: bool = True,
        hidden_dim: int = 128,
        num_grn_layers: int = 2,
        dropout: float = 0.1,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        assert attn_direction in ("forward", "reverse", "bidirectional")
        self.attn_direction = attn_direction
        self.use_residual_gate = use_residual_gate
        self.shared_dim = shared_dim
        self._use_spatial = use_spatial_attention
        self.use_physics_guidance = use_physics_guidance
        self.use_soil_gating = use_soil_gating
        self.num_classes = num_classes
        self.feature_cols = list(feature_cols)

        self.visible_indices = [feature_cols.index(col) for col in visible_cols if col in feature_cols]
        self.subsurface_indices = [feature_cols.index(col) for col in subsurface_cols if col in feature_cols]

        # ── 1. Visual Backbone & Spatial Adapter ──────────────────────────────
        self.visual_encoder = VisualEncoder(
            shared_dim=shared_dim,
            backbone=cnn_backbone,
            use_spatial_attention=use_spatial_attention,
            dropout=dropout,
            pretrained=pretrained,
        )

        if freeze_backbone:
            for p in self.visual_encoder.backbone.parameters():
                p.requires_grad = False

        self.spatial_adapter = SpatialDecompositionAdapter(
            shared_dim=shared_dim,
            hidden_dim=hidden_dim,
            num_patches=49 if use_spatial_attention else 1,
            dropout=dropout,
            use_soil_gating=use_soil_gating,
        )

        # ── 2. Tabular Router / Encoders ──────────────────────────────────────
        if self.use_physics_guidance:
            self.pgmr_router = PhysicsGuidedRelevanceRouter(
                feature_names=feature_cols,
                shared_dim=shared_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            self.tabular_encoder_vis = None
            self.tabular_encoder_sub = None
        else:
            self.pgmr_router = None
            self.tabular_encoder_vis = TabularEncoder(
                num_continuous=len(self.visible_indices),
                shared_dim=shared_dim,
                hidden_dim=hidden_dim,
                num_grn_layers=num_grn_layers,
                dropout=dropout,
            )
            self.tabular_encoder_sub = TabularEncoder(
                num_continuous=len(self.subsurface_indices),
                shared_dim=shared_dim,
                hidden_dim=hidden_dim,
                num_grn_layers=num_grn_layers,
                dropout=dropout,
            )

        # ── 3. Cross-Attention Modules (Gated Residual vs Standard) ───────────
        if self.use_residual_gate:
            self.cross_attn_surface = GatedResidualCrossModalAttention(
                shared_dim=shared_dim,
                num_heads=num_heads,
                direction=attn_direction,
                dropout=dropout,
            )
            self.cross_attn_subsurface = GatedResidualCrossModalAttention(
                shared_dim=shared_dim,
                num_heads=num_heads,
                direction=attn_direction,
                dropout=dropout,
            )
        else:
            self.cross_attn_surface = DirectionalCrossModalAttention(
                shared_dim=shared_dim,
                num_heads=num_heads,
                direction=attn_direction,
                dropout=dropout,
            )
            self.cross_attn_subsurface = DirectionalCrossModalAttention(
                shared_dim=shared_dim,
                num_heads=num_heads,
                direction=attn_direction,
                dropout=dropout,
            )

        # ── 4. Dual GMU Units + Master Fusion GMU ─────────────────────────────
        self.gmu_surface = GatedMultimodalUnit(shared_dim=shared_dim, dropout=dropout)
        self.gmu_subsurface = GatedMultimodalUnit(shared_dim=shared_dim, dropout=dropout)

        self.master_fusion = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.Sigmoid(),
        )

        # ── 5. Classifier Head ────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(shared_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        images: torch.Tensor,
        tabular: torch.Tensor,
        eval_domain: Optional[str] = None,
    ) -> torch.Tensor:
        # 1. Visual Feature Extraction & Spatial Attention Decomposition
        f_vis = self.visual_encoder(images)
        v_soil_pooled, v_canopy_pooled, f_soil_tokens, f_canopy_tokens, attention_maps = (
            self.spatial_adapter(f_vis)
        )

        # 2. Tabular Feature Tokenization
        if self.use_physics_guidance:
            feature_tokens = []
            for i, enc in enumerate(self.pgmr_router.feature_encoders):
                feat_val = tabular[:, i:i+1]
                e_i = enc(feat_val)
                feature_tokens.append(e_i)
            e_all = torch.stack(feature_tokens, dim=1) # (B, M, d)

            t_surface_tokens = e_all[:, self.visible_indices, :]       # (B, 2, d)
            t_subsurface_tokens = e_all[:, self.subsurface_indices, :] # (B, 6, d)
            
            t_surface_pooled = t_surface_tokens.mean(dim=1)            # (B, d)
            t_subsurface_pooled = t_subsurface_tokens.mean(dim=1)      # (B, d)
        else:
            x_vis = tabular[:, self.visible_indices]
            x_sub = tabular[:, self.subsurface_indices]
            t_surface_pooled = self.tabular_encoder_vis(x_vis)
            t_subsurface_pooled = self.tabular_encoder_sub(x_sub)
            t_surface_tokens = t_surface_pooled.unsqueeze(1)
            t_subsurface_tokens = t_subsurface_pooled.unsqueeze(1)

        # 3. Cross-Attention Execution
        if self.use_residual_gate:
            # Gated Residual: V' = V_base + G * A(C, V)
            soil_vis_input = f_soil_tokens if self._use_spatial else v_soil_pooled
            canopy_vis_input = f_canopy_tokens if self._use_spatial else v_canopy_pooled

            h_surface, g_surf = self.cross_attn_surface(
                t_tokens=t_surface_tokens,
                f_vis_tokens=soil_vis_input,
                v_base_pooled=v_soil_pooled,
            )
            h_subsurface, g_sub = self.cross_attn_subsurface(
                t_tokens=t_subsurface_tokens,
                f_vis_tokens=canopy_vis_input,
                v_base_pooled=v_canopy_pooled,
            )
        else:
            if self._use_spatial:
                h_surface, _ = self.cross_attn_surface(t_surface_tokens, f_soil_tokens)
                h_subsurface, _ = self.cross_attn_subsurface(t_subsurface_tokens, f_canopy_tokens)
            else:
                h_surface, _ = self.cross_attn_surface(t_surface_tokens, v_soil_pooled)
                h_subsurface, _ = self.cross_attn_subsurface(t_subsurface_tokens, v_canopy_pooled)

        # 4. Dual GMU
        f_surface, z_surface = self.gmu_surface(v_soil_pooled, t_surface_pooled, h_surface)
        f_subsurface, z_subsurface = self.gmu_subsurface(v_canopy_pooled, t_subsurface_pooled, h_subsurface)

        # 5. Domain Ablations
        if eval_domain == "surface_only":
            f_subsurface = torch.zeros_like(f_subsurface)
        elif eval_domain == "subsurface_only":
            f_surface = torch.zeros_like(f_surface)
        elif eval_domain == "vision_only":
            f_surface = v_soil_pooled
            f_subsurface = v_canopy_pooled

        # 6. Master Fusion
        f_combined = torch.cat([f_surface, f_subsurface], dim=-1)
        gate_weights = self.fusion_gate(f_combined)
        f_candidate = self.master_fusion(f_combined)
        f_fused = gate_weights * f_candidate

        # Diagnostics
        self._last_gates = {
            "z_surface": z_surface.detach(),
            "z_subsurface": z_subsurface.detach(),
            "z_fusion": gate_weights.detach(),
        }
        self._last_attention_maps = {
            k: v.detach() for k, v in attention_maps.items()
        }

        # 7. Logits
        logits = self.classifier(f_fused)
        return logits

    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        return getattr(self, "_last_attention_maps", {})
