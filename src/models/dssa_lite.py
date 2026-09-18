"""
dssa_lite.py
------------
DSSA-Lite-v2 (Enhanced Edge-Optimized Architecture):
- Latent dimension: d = 128
- Multi-Head Attention: 4 heads with d_k = 32
- Dual Compact Gated Multimodal Units (Dual GMU: Surface GMU + Subsurface GMU + Master Fusion)
- Compact Gated Residual Networks (GRNs) for surface and subsurface tabular features
- Compact Low-Rank (rank=16) Physics-Guided Mineral Routing (PGMR)
- Dynamic Zero-Soil Gating on Surface Channel
- Supports Multi-Task auxiliary returns: Spatial Orthogonality Mask and Canopy Density
- Total Parameter Count: 4,123,653 (~4.12M total, only 528k adapter overhead)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CompactGRN(nn.Module):
    """Lightweight Gated Residual Network for continuous tabular features."""
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim * 2)  # For GLU
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.drop(F.elu(self.fc1(x)))
        glu = self.fc2(h)
        val, gate = glu.chunk(2, dim=-1)
        return self.norm(residual + val * torch.sigmoid(gate))

class CompactSpatialAdapter(nn.Module):
    """Lightweight 3-way spatial attention routing (Soil, Canopy, Background)."""
    def __init__(self, in_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 3)  # Soil=0, Canopy=1, Background=2
        )
        self.soil_proj = nn.Linear(in_dim, in_dim)
        self.canopy_proj = nn.Linear(in_dim, in_dim)
        self.norm_s = nn.LayerNorm(in_dim)
        self.norm_c = nn.LayerNorm(in_dim)

    def forward(self, tokens: torch.Tensor):
        # tokens: (B, 49, d)
        logits = self.attn_net(tokens)  # (B, 49, 3)
        weights = F.softmax(logits, dim=1)  # Softmax across spatial tokens
        
        m_soil = weights[:, :, 0:1]      # (B, 49, 1)
        m_canopy = weights[:, :, 1:2]    # (B, 49, 1)
        m_bg = weights[:, :, 2:3]        # (B, 49, 1)
        
        v_soil = self.norm_s(F.elu(self.soil_proj((m_soil * tokens).sum(dim=1))))
        v_canopy = self.norm_c(F.elu(self.canopy_proj((m_canopy * tokens).sum(dim=1))))
        
        # Canopy density score for Zero-Soil Gating & auxiliary supervision
        canopy_density = m_canopy.sum(dim=1)  # (B, 1)
        
        return v_soil, v_canopy, m_soil, m_canopy, canopy_density

class CompactPGMR(nn.Module):
    """Compact Physics-Guided Mineral Router (Rank-16 low-rank bilinear projection)."""
    def __init__(self, d: int = 128, rank: int = 16):
        super().__init__()
        self.vis_down = nn.Linear(d, rank, bias=False)
        self.tab_down = nn.Linear(d, rank, bias=False)
        self.out_proj = nn.Linear(rank, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, vis_tokens: torch.Tensor, tab_emb: torch.Tensor) -> torch.Tensor:
        # vis_tokens: (B, 49, d), tab_emb: (B, d)
        v_low = self.vis_down(vis_tokens)            # (B, 49, rank)
        t_low = self.tab_down(tab_emb).unsqueeze(1)  # (B, 1, rank)
        return self.norm(vis_tokens + self.out_proj(v_low * t_low))

class DualCompactGMU(nn.Module):
    """
    Dual-Stream Compact Gated Multimodal Unit:
    Preserves structural physical separation between surface and subsurface streams.
    """
    def __init__(self, d: int = 128, dropout: float = 0.15):
        super().__init__()
        # Surface GMU
        self.gate_surf = nn.Sequential(nn.Linear(d * 3, d), nn.Sigmoid())
        self.cand_surf = nn.Sequential(nn.Linear(d * 3, d), nn.ELU(), nn.LayerNorm(d))
        
        # Subsurface GMU
        self.gate_sub = nn.Sequential(nn.Linear(d * 3, d), nn.Sigmoid())
        self.cand_sub = nn.Sequential(nn.Linear(d * 3, d), nn.ELU(), nn.LayerNorm(d))
        
        # Master Fusion Unit
        self.master_gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())
        self.master_cand = nn.Sequential(nn.Linear(d * 2, d), nn.ELU(), nn.LayerNorm(d))
        self.drop = nn.Dropout(dropout)

    def forward(self, v_soil, t_surf, f_surf, v_canopy, t_sub, f_sub, canopy_density):
        # Dynamic Zero-Soil Gating on Surface Channel
        zero_soil_weight = torch.clamp(1.0 - (canopy_density / 49.0), min=0.05, max=1.0)
        f_surf_gated = f_surf * zero_soil_weight
        
        # 1. Surface Gating
        in_s = torch.cat([v_soil, t_surf, f_surf_gated], dim=-1)  # (B, 384)
        h_surf = self.gate_surf(in_s) * self.cand_surf(in_s)     # (B, 128)
        
        # 2. Subsurface Gating
        in_sub = torch.cat([v_canopy, t_sub, f_sub], dim=-1)      # (B, 384)
        h_sub = self.gate_sub(in_sub) * self.cand_sub(in_sub)    # (B, 128)
        
        # 3. Master Fusion
        in_m = torch.cat([h_surf, h_sub], dim=-1)                 # (B, 256)
        fused = self.drop(self.master_gate(in_m) * self.master_cand(in_m))  # (B, 128)
        return fused

class DSSALiteModel(nn.Module):
    """
    DSSA-Lite-v2 (Enhanced):
    Lightweight edge-optimized multimodal network with Dual GMUs & Multi-Task output.
    Total Parameters: 4,123,653 (~4.12M total)
    """
    def __init__(
        self,
        num_classes: int = 6,
        d: int = 128,
        num_heads: int = 4,
        pretrained: bool = True,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.d = d
        self.num_classes = num_classes

        # 1. Vision Backbone (EfficientNet-B0 feature extractor)
        self.visual_backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            features_only=True
        )
        # Stage 5 outputs 320 channels at 7x7 resolution -> project to d=128
        self.vis_proj = nn.Sequential(
            nn.Conv2d(320, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ELU()
        )

        # 2. Compact Spatial Adapter
        self.spatial_adapter = CompactSpatialAdapter(in_dim=d, hidden_dim=64)

        # 3. Compact Tabular GRN Encoders
        self.grn_surface = CompactGRN(in_dim=2, hidden_dim=64, out_dim=d, dropout=dropout)      # OC, CaCO3
        self.grn_subsurface = CompactGRN(in_dim=6, hidden_dim=64, out_dim=d, dropout=dropout)   # pH, N, P, K, EC

        # 4. Compact PGMR Routers
        self.pgmr_surface = CompactPGMR(d=d, rank=16)
        self.pgmr_subsurface = CompactPGMR(d=d, rank=16)

        # 5. Lightweight Directional Multi-Head Cross-Attention
        self.cross_attn_surface = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.cross_attn_subsurface = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, batch_first=True, dropout=dropout)

        # 6. Dual-Stream Compact GMUs + Master Fusion
        self.dual_gmu = DualCompactGMU(d=d, dropout=dropout)

        # 7. Classification Head
        self.classifier = nn.Linear(d, num_classes)

    def forward(self, images: torch.Tensor, tabular: torch.Tensor, return_aux: bool = False):
        # images: (B, 3, 224, 224), tabular: (B, 8)
        B = images.size(0)

        # 1. Vision Features
        raw_feats = self.visual_backbone(images)[-1]  # (B, 320, 7, 7)
        vis_proj = self.vis_proj(raw_feats)          # (B, 128, 7, 7)
        vis_tokens = vis_proj.flatten(2).transpose(1, 2)  # (B, 49, 128)

        # 2. Spatial Routing
        v_soil, v_canopy, m_soil, m_canopy, canopy_density = self.spatial_adapter(vis_tokens)

        # 3. Tabular Chemistry Decomposition
        t_surface = self.grn_surface(tabular[:, :2])       # (B, 128)
        t_subsurface = self.grn_subsurface(tabular[:, 2:]) # (B, 128)

        # 4. Physics-Guided Mineral Routing
        tokens_soil = self.pgmr_surface(vis_tokens, t_surface)
        tokens_canopy = self.pgmr_subsurface(vis_tokens, t_subsurface)

        # 5. Cross-Attention
        f_cross_surf, _ = self.cross_attn_surface(
            query=t_surface.unsqueeze(1),
            key=tokens_soil,
            value=tokens_soil
        )
        f_cross_sub, _ = self.cross_attn_subsurface(
            query=t_subsurface.unsqueeze(1),
            key=tokens_canopy,
            value=tokens_canopy
        )

        # 6. Dual GMU Gated Fusion (with Zero-Soil dynamic attenuation)
        fused = self.dual_gmu(
            v_soil=v_soil,
            t_surf=t_surface,
            f_surf=f_cross_surf.squeeze(1),
            v_canopy=v_canopy,
            t_sub=t_subsurface,
            f_sub=f_cross_sub.squeeze(1),
            canopy_density=canopy_density
        )

        # 7. Classification Logits
        logits = self.classifier(fused)  # (B, num_classes)

        if return_aux:
            return logits, m_soil, m_canopy, canopy_density
        return logits
