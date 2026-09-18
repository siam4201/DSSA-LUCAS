"""
models/spatial_adapter.py
-------------------------
Spatial Attention Decomposition Adapter with Background Rejection
and Dynamic Canopy-Gated Soil Suppression.

Given spatial visual tokens F_vis in R^(B x T x d) (where T = H x W = 49 for 7x7 grid):
  1. Computes 3-way spatial attention distributions across all T patches:
       - M_soil   : Soft attention map over bare soil / ground matrix patches.
       - M_canopy : Soft attention map over vegetation / foliage / crop canopy patches.
       - M_bg     : Soft attention map over background / sky / horizon / artifacts (suppressed).
  2. Produces pooled representation vectors:
       - v_soil   = sum_{t} M_soil[t] * F_vis[t]
       - v_canopy = sum_{t} M_canopy[t] * F_vis[t]
  3. Optionally modulates spatial tokens for downstream multi-head cross-attention:
       - F_soil_tokens   = F_vis * M_soil_spatial
       - F_canopy_tokens = F_vis * M_canopy_spatial
  4. Dynamic Zero-Soil Thresholding (canopy density gate):
       - A scene-level canopy density estimator c in [0,1] is learned from the mean
         patch embedding, supervised by biophysical ExG/VARI optical indices.
       - A learned sigmoid gate g_soil = 1 - sigma(beta * (c - tau)) suppresses
         v_soil and F_soil_tokens toward zero when dense closed-canopy is detected.
       - A post-gate LayerNorm prevents scale drift in downstream GMU / cross-attention.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List


class SpatialDecompositionAdapter(nn.Module):
    """
    Lightweight Spatial Attention Decomposition Adapter.

    Args:
        shared_dim  : Token feature dimension d (e.g. 256).
        hidden_dim  : Intermediate projection dimension (e.g. 128).
        num_patches : Number of spatial patches T (default 49 for 7x7).
        dropout     : Dropout probability.
        temperature : Softmax temperature scaling (lower = sharper masks).
    """

    def __init__(
        self,
        shared_dim: int = 256,
        hidden_dim: int = 128,
        num_patches: int = 49,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_soil_gating: bool = True,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.num_patches = num_patches
        self.grid_size = int(math.sqrt(num_patches)) if math.isqrt(num_patches)**2 == num_patches else 7
        self.temperature = temperature
        self.use_soil_gating = use_soil_gating

        # ── 1. Spatial Scoring Network ─────────────────────────────────────────
        # Maps each token (d) to logits for 3 semantic components: [Soil, Canopy, Background]
        self.router = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # 0: Soil, 1: Canopy, 2: Background
        )

        # ── 2. Feature Refinement Heads ───────────────────────────────────────
        # Lightweight transformations to align feature spaces after spatial selection
        self.soil_proj = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.canopy_proj = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # ── 3. Dynamic Canopy-Density Gate ────────────────────────────────────
        # Scene-level canopy density estimator: mean patch embedding → scalar c in [0,1].
        # Input is detached from the main task graph so the gate is trained only by
        # compute_canopy_density_loss(), preventing co-adaptation with the spatial router.
        self.canopy_density_estimator = nn.Linear(shared_dim, 1)

        # Learnable gate parameters.
        # tau  : soft threshold — initialized at Broadleaf median canopy_fraction (0.40)
        #        so the sigmoid sits in its gradient-active region from epoch 1.
        # beta : sharpness — 8.0 gives a transition width of ~0.25 units, wide enough
        #        for gradients to flow during early training, steep enough at convergence.
        self.tau  = nn.Parameter(torch.tensor(0.40))
        self.beta = nn.Parameter(torch.tensor(8.0))

        # Post-gate LayerNorm: normalizes the gated soil representation before it reaches
        # downstream GMU / cross-attention, preventing scale drift caused by g_soil → 0
        # in high-canopy scenes while v_canopy remains at its natural scale.
        self.gate_norm = nn.LayerNorm(shared_dim)

    def forward(
        self,
        f_vis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            f_vis: Spatial visual tokens of shape (B, T, d) or pooled (B, d).

        Returns:
            v_soil_pooled    : (B, d) - Aggregated (and canopy-gated) soil matrix vector
            v_canopy_pooled  : (B, d) - Aggregated vegetation canopy vector
            f_soil_tokens    : (B, T, d) - Spatially modulated (and gated) soil tokens
            f_canopy_tokens  : (B, T, d) - Spatially modulated canopy tokens
            attention_maps   : Dict containing:
                                - "soil_map"        : (B, 1, H, W)
                                - "canopy_map"      : (B, 1, H, W)
                                - "bg_map"          : (B, 1, H, W)
                                - "raw_weights"     : (B, 3, T)
                                - "canopy_density"  : (B, 1)  [if use_soil_gating]
                                - "soil_gate"       : (B, 1)  [if use_soil_gating]
        """
        # Handle fallback when spatial mode is not active (input is flat B, d)
        if f_vis.dim() == 2:
            v_soil = self.soil_proj(f_vis)
            v_canopy = self.canopy_proj(f_vis)
            dummy_maps = {
                "soil_map": torch.ones(f_vis.size(0), 1, 1, 1, device=f_vis.device),
                "canopy_map": torch.ones(f_vis.size(0), 1, 1, 1, device=f_vis.device),
                "bg_map": torch.zeros(f_vis.size(0), 1, 1, 1, device=f_vis.device),
            }
            return v_soil, v_canopy, v_soil.unsqueeze(1), v_canopy.unsqueeze(1), dummy_maps

        B, T, d = f_vis.shape
        H = W = self.grid_size

        # ── Step 1: Compute Token-Level Semantic Component Logits ─────────────
        # logits: (B, T, 3)
        logits = self.router(f_vis) / self.temperature

        # Compute spatial attention distributions across tokens for each concept:
        # Permute to (B, 3, T) then softmax across the T spatial patches
        # This ensures sum_{t=1}^T M_k[t] = 1 for each semantic map k in {soil, canopy, bg}
        spatial_logits = logits.permute(0, 2, 1)  # (B, 3, T)
        spatial_attn = F.softmax(spatial_logits, dim=-1)  # (B, 3, T)

        m_soil_weights = spatial_attn[:, 0:1, :]    # (B, 1, T)
        m_canopy_weights = spatial_attn[:, 1:2, :]  # (B, 1, T)
        m_bg_weights = spatial_attn[:, 2:3, :]      # (B, 1, T)

        # ── Step 2: Attention-Weighted Pooling ─────────────────────────────────
        # v_soil: (B, 1, T) @ (B, T, d) -> (B, 1, d) -> (B, d)
        v_soil_raw = torch.bmm(m_soil_weights, f_vis).squeeze(1)
        v_canopy_raw = torch.bmm(m_canopy_weights, f_vis).squeeze(1)

        v_soil_proj = self.soil_proj(v_soil_raw)      # (B, d) — pre-gate
        v_canopy_pooled = self.canopy_proj(v_canopy_raw)  # (B, d) — unchanged

        # ── Step 3: Spatial Token Modulation (for Cross-Attention) ────────────
        # Modulate tokens by their normalized spatial probability (scaled by T for scale invariance)
        f_soil_tokens_raw = self.soil_proj(f_vis * m_soil_weights.permute(0, 2, 1) * T)  # (B, T, d)
        f_canopy_tokens = self.canopy_proj(f_vis * m_canopy_weights.permute(0, 2, 1) * T)

        # ── Step 4: Dynamic Canopy-Density Gate ───────────────────────────────
        if self.use_soil_gating:
            # Scene-level canopy density estimator.
            # scene_embed is detached so only compute_canopy_density_loss() drives the
            # estimator's weights — preventing co-adaptation with the spatial router.
            scene_embed = f_vis.mean(dim=1).detach()                              # (B, d)
            c = torch.sigmoid(self.canopy_density_estimator(scene_embed))         # (B, 1)

            # Soft threshold gate: g_soil → 1 for sparse canopy, → 0 for closed canopy.
            g_soil = 1.0 - torch.sigmoid(self.beta * (c - self.tau))              # (B, 1)

            # Apply gate then LayerNorm to prevent downstream scale drift.
            v_soil_pooled = self.gate_norm(g_soil * v_soil_proj)                  # (B, d)

            # Apply gate to soil tokens: reshape to (B*T, d) for LayerNorm, then restore.
            g_soil_tok = g_soil.unsqueeze(1).expand_as(f_soil_tokens_raw)         # (B, T, d)
            f_soil_tokens = self.gate_norm(
                (g_soil_tok * f_soil_tokens_raw).reshape(B * T, -1)
            ).reshape(B, T, -1)                                                    # (B, T, d)
        else:
            c = None
            g_soil = None
            v_soil_pooled = v_soil_proj
            f_soil_tokens = f_soil_tokens_raw

        # ── Step 5: Format 2D Attention Maps for Explainability ────────────────
        soil_map = m_soil_weights.view(B, 1, H, W)
        canopy_map = m_canopy_weights.view(B, 1, H, W)
        bg_map = m_bg_weights.view(B, 1, H, W)

        attention_maps = {
            "soil_map": soil_map,
            "canopy_map": canopy_map,
            "bg_map": bg_map,
            "raw_weights": spatial_attn,
        }
        if self.use_soil_gating:
            attention_maps["canopy_density"] = c
            attention_maps["soil_gate"] = g_soil

        self._last_attention_maps = attention_maps

        return v_soil_pooled, v_canopy_pooled, f_soil_tokens, f_canopy_tokens, attention_maps

    @staticmethod
    def compute_biophysical_priors(images: torch.Tensor, grid_size: int = 7) -> torch.Tensor:
        """
        Extracts biophysical optical target priors directly from RGB images:
          - Vegetation/Canopy: Excess Green Index (ExG = 2G - R - B) & VARI
          - Soil/Ground: Coloration Index (CI = (R-G)/(R+G)) & low greenness
          - Background/Sky: Upper image plane + blue dominance

        Args:
            images    : (B, 3, H, W) ImageNet normalized input images.
            grid_size : Spatial grid size (default 7).

        Returns:
            priors    : (B, 3, grid_size, grid_size) normalized spatial distribution targets.
                        Channel 0: Soil prior
                        Channel 1: Canopy prior
                        Channel 2: Background/Sky prior
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        rgb = torch.clamp(images * std + mean, 0.0, 1.0)
        R, G, B = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

        # 1. Vegetation / Canopy Index
        exg = 2.0 * G - R - B
        vari = (G - R) / (G + R - B + 1e-4)
        p_canopy = F.relu(exg) + 0.5 * F.relu(vari) + 1e-3

        # 2. Soil / Mineral Coloration Index (warm chromophores, non-vegetative)
        ci = (R - G) / (R + G + 1e-4)
        p_soil = F.relu(ci) * F.relu(1.0 - 2.0 * F.relu(exg)) + 1e-3

        # 3. Background / Sky Prior (upper 40% vertical plane with blue dominance)
        B_sz, _, H, W = images.shape
        y_coords = torch.linspace(0.0, 1.0, steps=H, device=images.device).view(1, 1, H, 1).expand(B_sz, 1, H, W)
        sky_zone = F.relu(1.0 - 2.5 * y_coords)
        sky_blue = F.relu(B - R) + F.relu(B - G)
        p_bg = sky_zone * (sky_blue + 0.1) + 1e-3

        raw_priors = torch.cat([p_soil, p_canopy, p_bg], dim=1)  # (B, 3, H, W)
        p_grid = F.adaptive_avg_pool2d(raw_priors, (grid_size, grid_size))  # (B, 3, 7, 7)

        # Normalize spatially so sum across patches per channel equals 1
        p_norm = p_grid / (p_grid.sum(dim=(-1, -2), keepdim=True) + 1e-6)
        return p_norm

    def compute_spatial_supervision_loss(self, images: torch.Tensor) -> torch.Tensor:
        """
        Computes physical spatial consistency loss between model attention maps
        and biophysical optical targets.
        """
        if not hasattr(self, "_last_attention_maps") or self._last_attention_maps is None:
            return torch.tensor(0.0, device=images.device)

        soil_m = self._last_attention_maps.get("soil_map")
        canopy_m = self._last_attention_maps.get("canopy_map")
        bg_m = self._last_attention_maps.get("bg_map")

        if soil_m is None or canopy_m is None or bg_m is None or soil_m.numel() <= 1:
            return torch.tensor(0.0, device=images.device)

        # Predicted maps: (B, 3, H, W)
        pred_maps = torch.cat([soil_m, canopy_m, bg_m], dim=1)
        target_priors = self.compute_biophysical_priors(images, grid_size=self.grid_size)

        # MSE Loss + Cosine dissimilarity
        loss_mse = F.mse_loss(pred_maps, target_priors)

        # Spatial Cosine Alignment
        flat_pred = pred_maps.flatten(start_dim=2)  # (B, 3, T)
        flat_targ = target_priors.flatten(start_dim=2)  # (B, 3, T)
        cos_sim = F.cosine_similarity(flat_pred, flat_targ, dim=-1).mean()
        loss_align = 1.0 - cos_sim

        return loss_mse + 0.5 * loss_align

    def compute_canopy_density_loss(self, images: torch.Tensor) -> torch.Tensor:
        """
        Supervises the scene-level canopy density estimator `c` using the biophysical
        ExG/VARI optical prior (the same signal used in compute_spatial_supervision_loss).

        Target: mean canopy prior mass per sample (scalar in [0, 1]).
        This is the only gradient source that updates `canopy_density_estimator` weights,
        since scene_embed is detached from the main task in forward().
        """
        if not hasattr(self, "_last_attention_maps") or self._last_attention_maps is None:
            return torch.tensor(0.0, device=images.device)

        c = self._last_attention_maps.get("canopy_density")
        if c is None:
            return torch.tensor(0.0, device=images.device)

        target_priors = self.compute_biophysical_priors(images, grid_size=self.grid_size)
        # Mean canopy prior mass across all patches per sample: (B, 1)
        canopy_target = target_priors[:, 1:2].mean(dim=(-1, -2))  # (B, 1)
        return F.mse_loss(c, canopy_target.detach())

    @staticmethod
    def calibrate_threshold(
        adapter: "SpatialDecompositionAdapter",
        dataloader,
        forest_class_indices: List[int],
        device: torch.device,
        percentile: float = 60.0,
    ) -> float:
        """
        Computes an empirically calibrated tau initialization from the biophysical canopy
        prior distribution over forest-class training samples.

        Run this once before training starts and use the returned value to set:
            adapter.tau.data.fill_(tau_calibrated)

        Args:
            adapter             : Initialized SpatialDecompositionAdapter instance.
            dataloader          : Training DataLoader yielding (images, tabular, labels).
            forest_class_indices: Label indices corresponding to forest classes
                                  (e.g. [2, 3] for Broadleaf and Coniferous).
            device              : Torch device.
            percentile          : Percentile of canopy prior distribution to use as tau
                                  (default 60 — places gate slightly above the median of
                                  forest canopy density, in a gradient-active region).

        Returns:
            tau : float — the calibrated threshold value.
        """
        forest_set = set(forest_class_indices)
        c_vals: List[float] = []
        adapter.eval()
        with torch.no_grad():
            for batch in dataloader:
                images, _, labels = batch
                mask = torch.tensor([l.item() in forest_set for l in labels], dtype=torch.bool)
                if not mask.any():
                    continue
                imgs_forest = images[mask].to(device)
                priors = SpatialDecompositionAdapter.compute_biophysical_priors(
                    imgs_forest, grid_size=adapter.grid_size
                )
                # Mean canopy prior mass per sample (what c is supervised to predict)
                canopy_means = priors[:, 1].mean(dim=(-1, -2)).cpu().numpy()
                c_vals.extend(canopy_means.tolist())
        if not c_vals:
            return float(adapter.tau.item())  # fallback: keep current value
        tau = float(np.percentile(c_vals, percentile))
        return tau

    @torch.no_grad()
    def compute_spatial_alignment_metrics(self, images: torch.Tensor) -> Dict[str, float]:
        """
        Evaluates quantitative spatial alignment (Cosine similarity) between
        learned attention maps and biophysical ground truth indices.
        """
        if not hasattr(self, "_last_attention_maps") or self._last_attention_maps is None:
            return {}

        soil_m = self._last_attention_maps.get("soil_map")
        canopy_m = self._last_attention_maps.get("canopy_map")
        bg_m = self._last_attention_maps.get("bg_map")

        if soil_m is None or canopy_m is None or bg_m is None or soil_m.numel() <= 1:
            return {}

        target_priors = self.compute_biophysical_priors(images, grid_size=self.grid_size)

        def cos(a, b):
            return float(F.cosine_similarity(a.flatten(start_dim=1), b.flatten(start_dim=1), dim=-1).mean().item())

        return {
            "soil_biophysical_alignment": cos(soil_m, target_priors[:, 0:1]),
            "canopy_biophysical_alignment": cos(canopy_m, target_priors[:, 1:2]),
            "bg_biophysical_alignment": cos(bg_m, target_priors[:, 2:3]),
        }
