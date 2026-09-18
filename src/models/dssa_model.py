"""
models/dssa_model.py
--------------------
Domain-Specific Spatial Attention (DSSA) Model for Multimodal Land Cover Classification.

Optimal Architecture (Experiment B):
  1. Spatial Decomposition Adapter with Dynamic Canopy-Gated Soil Suppression:
     - Decomposes spatial visual tokens into M_soil (bare ground matrix), M_canopy (vegetation), and M_bg (background).
     - Gated Zero-Soil Thresholding: Learns a scene-level canopy density descriptor c supervised by
       biophysical optical priors (ExG/VARI) and applies a soft gate g_soil = 1 - sigma(beta * (c - tau))
       to collapse soil attention toward zero over dense closed-canopy scenes.
  2. Physics-Guided Modality Relevance Router (PGMR):
     - Maps continuous chemical features to soil/canopy pathways with differentiable physics constraints
       (OC, CaCO3 -> Soil; N, P, K, EC, pH -> Canopy).
  3. Dual Domain-Aligned Cross-Attention:
     - Surface Grounding: Visible Chemistry (Q) <-> Soil Patches (K, V)
     - Subsurface Growth: Subsurface Chemistry (Q) <-> Foliage Patches (K, V)
  4. Gated Multimodal Units (GMU) & Master Fusion:
     - Dynamically gates visual vs. tabular contributions per domain.
     - Master GMU fuses surface and subsurface streams into a unified representation f_fused.
  5. Classifier Head:
     - High-capacity MLP classifier operating on the unified continuous cross-modal representation.
"""

import math
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple

from .visual_encoder import VisualEncoder
from .spatial_adapter import SpatialDecompositionAdapter
from .pgmr_router import PhysicsGuidedRelevanceRouter
from .tabular_encoder import TabularEncoder
from .cross_attention import CrossModalAttention
from .gmu import GatedMultimodalUnit


class DSSAModel(nn.Module):
    """
    Domain-Specific Spatial Attention (DSSA) Multimodal Network.
    
    Args:
        feature_cols        : Complete ordered list of continuous feature names.
        visible_cols        : Feature names representing surface/visible chemistry (e.g. OC, CaCO3).
        subsurface_cols     : Feature names representing subsurface chemistry (e.g. pH, N, P, K, EC).
        num_classes         : Number of output classes (e.g. 7).
        shared_dim          : Shared projection dimension d (default 256).
        num_heads           : Multi-head attention heads (default 8).
        cnn_backbone        : timm CNN backbone name (default 'efficientnet_b0').
        use_spatial_attention: True to emit spatial tokens (7x7=49).
        use_physics_guidance: True to enable differentiable Physics-Guided Relevance Matrix (PGMR).
        use_soil_gating     : True to enable dynamic canopy-density soil suppression gate.
        hidden_dim          : GRN & Adapter hidden dimension (default 128).
        num_grn_layers      : Number of GRN blocks in tabular encoders (default 2).
        dropout             : Dropout rate.
        pretrained          : If True, loads ImageNet pretrained CNN backbone (from local cache).
        freeze_backbone     : If True, freezes CNN backbone for parameter-efficient training.
    """

    def __init__(
        self,
        feature_cols: List[str],
        visible_cols: List[str],
        subsurface_cols: List[str],
        num_classes: int = 7,
        shared_dim: int = 256,
        num_heads: int = 8,
        cnn_backbone: str = "efficientnet_b0",
        use_spatial_attention: bool = True,
        use_physics_guidance: bool = True,
        use_soil_gating: bool = True,
        hidden_dim: int = 128,
        num_grn_layers: int = 2,
        use_gmu_final_fusion: bool = False,
        dropout: float = 0.1,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.shared_dim = shared_dim
        self._use_spatial = use_spatial_attention
        self.use_physics_guidance = use_physics_guidance
        self.use_soil_gating = use_soil_gating
        self.use_gmu_final_fusion = use_gmu_final_fusion
        self.num_classes = num_classes
        self.feature_cols = list(feature_cols)

        # Column indices for visible and subsurface splits (used when PGMR is disabled)
        self.visible_indices = [feature_cols.index(col) for col in visible_cols if col in feature_cols]
        self.subsurface_indices = [feature_cols.index(col) for col in subsurface_cols if col in feature_cols]

        # ── 1. Visual Backbone & Spatial Decomposition Adapter ────────────────
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

        # ── 2. Tabular Encoders (PGMR Router vs Fallback GRN Split) ───────────
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

        # ── 3. Dual Domain-Aligned Cross-Attention ────────────────────────────
        # Surface Grounding: t_visible / t_soil (Q) <-> f_soil (K, V)
        self.cross_attn_surface = CrossModalAttention(
            shared_dim=shared_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        # Subsurface Growth: t_subsurface / t_canopy (Q) <-> f_canopy (K, V)
        self.cross_attn_subsurface = CrossModalAttention(
            shared_dim=shared_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ── 4. Dual GMU Units + Master Fusion GMU ─────────────────────────────
        self.gmu_surface = GatedMultimodalUnit(shared_dim=shared_dim, dropout=dropout)
        self.gmu_subsurface = GatedMultimodalUnit(shared_dim=shared_dim, dropout=dropout)

        # Master Fusion GMU combining f_surface and f_subsurface
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

        # Optional GMU-style Modality Arbitration Gate (V' vs C)
        if self.use_gmu_final_fusion:
            self.final_modality_gate = nn.Sequential(
                nn.Linear(shared_dim * 2, shared_dim),
                nn.LayerNorm(shared_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(shared_dim, shared_dim),
                nn.Sigmoid(),
            )
            self.tab_global_proj = nn.Sequential(
                nn.Linear(shared_dim, shared_dim),
                nn.LayerNorm(shared_dim),
            )
        else:
            self.final_modality_gate = None
            self.tab_global_proj = None

        # ── 5. Final Classification Head ──────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(shared_dim, num_classes),
        )

        # ── 6. Strictly Tabular-Only Macro-Biome Head (Level-1 Ecological Prior) ──
        # Predicts 4 macro biomes: Cropland(0), Woodland(1), Shrubland(2), Grassland(3)
        self.num_macro_classes = 4
        self.tabular_macro_encoder = TabularEncoder(
            num_continuous=len(feature_cols),
            shared_dim=shared_dim,
            hidden_dim=hidden_dim,
            num_grn_layers=num_grn_layers,
            dropout=dropout,
        )
        self.tabular_macro_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(shared_dim, self.num_macro_classes),
        )

        # Hierarchy Mapping Matrix H in R^{num_classes x 4}: H[c, m] = 1 if fine class c in macro biome m
        # 0: Cereals (0), 1: Other Cropland (1) -> Cropland (0)
        # 2: Broadleaf (2), 3: Coniferous (3)   -> Woodland (1)
        # 4: Shrubland (4)                      -> Shrubland (2)
        # 5: Managed Grassland (5)              -> Grassland (3)
        # 6: Other Grassland (if 7-class)       -> Grassland (3)
        H = torch.zeros(num_classes, self.num_macro_classes, dtype=torch.float32)
        H[0, 0] = 1.0; H[1, 0] = 1.0   # Cropland
        H[2, 1] = 1.0; H[3, 1] = 1.0   # Woodland
        H[4, 2] = 1.0                  # Shrubland
        H[5, 3] = 1.0                  # Grassland
        if num_classes > 6:
            H[6, 3] = 1.0              # Other Grassland
        self.register_buffer("hierarchy_matrix", H)
        fine_map = [0, 0, 1, 1, 2, 3] if num_classes == 6 else [0, 0, 1, 1, 2, 3, 3]
        self.register_buffer("fine_to_macro_map", torch.tensor(fine_map, dtype=torch.long))

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
        lambda_h: float = 0.0,
        return_macro: bool = False,
    ):
        """
        Args:
            images      : (B, 3, H, W) RGB satellite / field images.
            tabular     : (B, num_features) normalized continuous chemistry features.
            eval_domain : None | "surface_only" | "subsurface_only" | "vision_only".
            lambda_h    : Continuous soft macro prior scale factor in [0, 1].
            return_macro: If True, returns (logits_final, macro_logits, logits_fine).

        Returns:
            logits_final: (B, num_classes) soft macro-conditioned or standard unnormalized class logits.
        """
        # ── Step 1: Strictly Tabular-Only Macro Prior ─────────────────────────
        f_tab_macro = self.tabular_macro_encoder(tabular)
        macro_logits = self.tabular_macro_head(f_tab_macro)   # (B, 4)

        # ── Step 2: Visual Feature Extraction & Spatial Attention Decomposition ──
        # f_vis: (B, 49, d) when spatial mode is active, else (B, d)
        f_vis = self.visual_encoder(images)

        # Decomposition: (B, d), (B, d), (B, 49, d), (B, 49, d), maps dict
        v_soil_pooled, v_canopy_pooled, f_soil_tokens, f_canopy_tokens, attention_maps = (
            self.spatial_adapter(f_vis)
        )

        # ── Step 3: Tabular Processing (PGMR Router vs Legacy GRN) ────────────
        if self.use_physics_guidance:
            # PGMR router routes features to soil (visible) and canopy (subsurface) tokens
            t_surface, t_subsurface, _ = self.pgmr_router(tabular)
        else:
            x_vis = tabular[:, self.visible_indices]
            x_sub = tabular[:, self.subsurface_indices]
            t_surface = self.tabular_encoder_vis(x_vis)
            t_subsurface = self.tabular_encoder_sub(x_sub)

        # ── Step 4: Domain-Aligned Cross-Attention ────────────────────────────
        if self._use_spatial:
            # Cross-Attention over spatial tokens
            h_surface = self.cross_attn_surface(t_surface, f_soil_tokens)
            h_subsurface = self.cross_attn_subsurface(t_subsurface, f_canopy_tokens)
        else:
            h_surface = self.cross_attn_surface(t_surface, v_soil_pooled)
            h_subsurface = self.cross_attn_subsurface(t_subsurface, v_canopy_pooled)

        # ── Step 5: Dual Gated Multimodal Units (GMU) ─────────────────────────
        f_surface, z_surface = self.gmu_surface(v_soil_pooled, t_surface, h_surface)
        f_subsurface, z_subsurface = self.gmu_subsurface(v_canopy_pooled, t_subsurface, h_subsurface)

        # ── Domain-Isolated Evaluation Ablations ──────────────────────────────
        if eval_domain == "surface_only":
            f_subsurface = torch.zeros_like(f_subsurface)
        elif eval_domain == "subsurface_only":
            f_surface = torch.zeros_like(f_surface)
        elif eval_domain == "vision_only":
            f_surface = v_soil_pooled
            f_subsurface = v_canopy_pooled

        # ── Master Fusion GMU ─────────────────────────────────────────────────
        f_combined = torch.cat([f_surface, f_subsurface], dim=-1)   # (B, 2d)
        gate_weights = self.fusion_gate(f_combined)                 # (B, d)
        f_candidate = self.master_fusion(f_combined)                # (B, d)
        v_dssa = gate_weights * f_candidate                         # (B, d) = V'

        # Optional Modality Arbitration Gate: g = sigma(W[V'; C]), F = g * V' + (1 - g) * C
        if self.use_gmu_final_fusion:
            c_global = (t_surface + t_subsurface) * 0.5
            c_proj = self.tab_global_proj(c_global)
            g_modality = self.final_modality_gate(torch.cat([v_dssa, c_proj], dim=-1))
            f_fused = g_modality * v_dssa + (1.0 - g_modality) * c_proj
            z_mod = g_modality.detach()
        else:
            f_fused = v_dssa
            z_mod = None

        # Store diagnostics for explainability
        self._last_gates = {
            "z_surface": z_surface.detach(),
            "z_subsurface": z_subsurface.detach(),
            "z_fusion": gate_weights.detach(),
        }
        if z_mod is not None:
            self._last_gates["z_modality"] = z_mod

        self._last_attention_maps = {
            k: v.detach() for k, v in attention_maps.items()
        }

        # ── Step 6: Multimodal Fine Classification ────────────────────────────
        logits_fine = self.classifier(f_fused)                      # (B, 7)

        # ── Step 7: Continuous Soft Macro Prior Conditioning ──────────────────
        if lambda_h > 0.0:
            p_macro = torch.softmax(macro_logits, dim=-1)           # (B, 4)
            # Expand macro probabilities to fine class space: H in R^{7 x 4}
            macro_prior_fine = torch.matmul(p_macro, self.hierarchy_matrix.T) # (B, 7)
            logits_final = logits_fine + lambda_h * torch.log(macro_prior_fine + 1e-8)
        else:
            logits_final = logits_fine

        if return_macro:
            return logits_final, macro_logits, logits_fine
        return logits_final

    def compute_macro_loss(self, macro_logits: torch.Tensor, fine_labels: torch.Tensor) -> torch.Tensor:
        """
        Computes Cross-Entropy loss for the strictly tabular macro head.
        Macro labels are derived cleanly from fine ground truth labels via fine_to_macro_map.
        """
        macro_labels = self.fine_to_macro_map[fine_labels]
        return torch.nn.functional.cross_entropy(macro_logits, macro_labels)


    def compute_spatial_physics_loss(self, images: torch.Tensor, canopy_weight: float = 0.05) -> torch.Tensor:
        """
        Computes biophysical optical supervision loss for spatial attention maps,
        plus canopy density estimator supervision loss for the dynamic soil gate.
        """
        if self._use_spatial and self.spatial_adapter is not None:
            loss_spatial = self.spatial_adapter.compute_spatial_supervision_loss(images)
            if self.use_soil_gating:
                loss_canopy = self.spatial_adapter.compute_canopy_density_loss(images)
                return loss_spatial + canopy_weight * loss_canopy
            return loss_spatial
        return torch.tensor(0.0, device=images.device)

    def compute_physics_loss(self) -> torch.Tensor:
        """Computes physics regularization loss L_physics = || A - A_prior ||_F^2."""
        if self.use_physics_guidance and self.pgmr_router is not None:
            return self.pgmr_router.compute_physics_loss()
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def get_relevance_matrix(self) -> Optional[torch.Tensor]:
        """Returns the normalized PGMR routing matrix A in R^{M x 2} if enabled."""
        if self.use_physics_guidance and self.pgmr_router is not None:
            return self.pgmr_router.get_relevance_matrix()
        return None

    def export_relevance_summary(self) -> List[Dict[str, float]]:
        """Returns a list of dicts with learned weights vs domain priors."""
        if self.use_physics_guidance and self.pgmr_router is not None:
            return self.pgmr_router.export_relevance_summary()
        return []

    def get_gate_values(self) -> dict:
        """Returns the dictionary of all GMU gate vectors from the last forward pass."""
        return self._last_gates

    def get_attention_maps(self) -> dict:
        """Returns the spatial attention heatmaps (soil_map, canopy_map, bg_map) from last forward pass."""
        return self._last_attention_maps

    def parameter_count(self) -> dict:
        """Returns parameter counts per sub-module."""
        def count(m):
            if m is None:
                return 0
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        counts = {
            "visual_encoder":        count(self.visual_encoder),
            "spatial_adapter":       count(self.spatial_adapter),
            "cross_attn_surface":    count(self.cross_attn_surface),
            "cross_attn_subsurface": count(self.cross_attn_subsurface),
            "gmu_surface":           count(self.gmu_surface),
            "gmu_subsurface":        count(self.gmu_subsurface),
            "master_fusion":         count(self.master_fusion) + count(self.fusion_gate),
            "classifier":            count(self.classifier),
        }
        if self.use_physics_guidance and self.pgmr_router is not None:
            counts["pgmr_router"] = count(self.pgmr_router)
        else:
            counts["tabular_encoder_vis"] = count(self.tabular_encoder_vis)
            counts["tabular_encoder_sub"] = count(self.tabular_encoder_sub)

        counts["total_trainable"] = sum(counts.values())
        return counts
