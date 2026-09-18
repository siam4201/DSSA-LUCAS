"""
models/pgmr_router.py
---------------------
Physics-Guided Modality Relevance (PGMR) Router for DSSA.

Transforms hard-coded chemistry splitting into a differentiable, physics-constrained
multimodal routing framework.

Key Mechanisms:
  1. Per-Feature Tokenization:
       Each continuous soil property x_i (e.g. OC, CaCO3, pH, N, P, K, EC) is projected
       independently into a shared embedding space e_i in R^d using Gated Residual Networks.
  
  2. Learnable Relevance Matrix (A in R^{M x 2}):
       A learnable parameter W_rel in R^{M x 2} models the coupling between M chemical
       properties and 2 visual pathways (Soil Surface vs. Vegetation Canopy).
       Soft routing weights are obtained via row-wise softmax:
           A = Softmax(W_rel, dim=-1)
       Decomposed chemistry tokens:
           t_soil   = sum_i A_{i, soil}   * e_i
           t_canopy = sum_i A_{i, canopy} * e_i

  3. Physics Prior Regularization (L_physics):
       A domain prior matrix A_prior encodes agronomic and soil spectroscopy principles.
       The training objective penalizes deviation from physical reality:
           L_physics = || A - A_prior ||_F^2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from .tabular_encoder import GatedResidualNetwork


class PhysicsGuidedRelevanceRouter(nn.Module):
    """
    Physics-Guided Modality Relevance (PGMR) Router.

    Args:
        feature_names   : Ordered list of continuous feature names (length M).
        shared_dim      : Projected token dimension d (e.g. 256).
        hidden_dim      : Internal GRN width (e.g. 128).
        dropout         : Dropout probability.
        custom_priors   : Optional dict mapping feature_name -> (soil_weight, canopy_weight).
    """

    # Standard Domain Prior based on Soil Spectroscopy & Plant Physiology
    DEFAULT_DOMAIN_PRIORS: Dict[str, Tuple[float, float]] = {
        # Soil / Surface-evident chemistry (pigment/mineral reflectance directly visible in soil patches)
        "OC":       (1.0, 0.0),   # Organic Carbon -> Dark humic chromophores
        "CaCO3":    (1.0, 0.0),   # Calcium Carbonates -> Pale chalky mineral tint
        
        # Subsurface / Canopy chemistry (invisible nutrients & physiological constraints)
        "pH_H2O":   (0.1, 0.9),   # Acidity -> Nutrient availability & root uptake
        "pH_CaCl2": (0.1, 0.9),   # Buffer pH -> Root-zone constraint
        "N":        (0.0, 1.0),   # Total Nitrogen -> Foliar chlorophyll & canopy vigor
        "P":        (0.0, 1.0),   # Phosphorus -> Canopy biomass / root energy
        "K":        (0.0, 1.0),   # Potassium -> Osmotic regulation & canopy health
        "EC":       (0.2, 0.8),   # Salinity -> Osmotic stress (minor surface efflorescence)
    }

    def __init__(
        self,
        feature_names: List[str],
        shared_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        custom_priors: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        super().__init__()
        self.feature_names = list(feature_names)
        self.num_features = len(self.feature_names)
        self.shared_dim = shared_dim

        # ── 1. Per-Feature Tokenizer ──────────────────────────────────────────
        # Each feature is embedded via an independent lightweight GRN block:
        # scalar input (dim=1) -> hidden_dim -> shared_dim
        self.feature_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ELU(),
                GatedResidualNetwork(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=shared_dim,
                    dropout=dropout,
                ),
            )
            for _ in range(self.num_features)
        ])

        # ── 2. Construct Prior Matrix A_prior ─────────────────────────────────
        priors_dict = dict(self.DEFAULT_DOMAIN_PRIORS)
        if custom_priors:
            priors_dict.update(custom_priors)

        prior_list = []
        for feat in self.feature_names:
            if feat in priors_dict:
                soil_p, canopy_p = priors_dict[feat]
            else:
                # Fallback for unknown feature: uninformative prior
                soil_p, canopy_p = 0.5, 0.5
            total = soil_p + canopy_p + 1e-8
            prior_list.append([soil_p / total, canopy_p / total])

        prior_tensor = torch.tensor(prior_list, dtype=torch.float32)
        self.register_buffer("A_prior", prior_tensor)  # (M, 2)

        # ── 3. Learnable Relevance Parameters W_rel ───────────────────────────
        # Initialize logits matching the prior distribution with a small temperature
        clamped_prior = torch.clamp(prior_tensor, min=1e-3, max=1.0 - 1e-3)
        init_logits = torch.log(clamped_prior)
        self.W_rel = nn.Parameter(init_logits.clone())

        # Layer normalization for stabilized combined representations
        self.norm_soil = nn.LayerNorm(shared_dim)
        self.norm_canopy = nn.LayerNorm(shared_dim)

    def get_relevance_matrix(self) -> torch.Tensor:
        """
        Returns normalized soft routing matrix A in R^{M x 2} in [0, 1].
        Column 0: Soil pathway weight
        Column 1: Canopy pathway weight
        """
        return F.softmax(self.W_rel, dim=-1)

    def forward(
        self,
        tabular: torch.Tensor,
        zero_subsurface: bool = False,
        zero_visible: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            tabular         : (B, num_features) raw continuous tabular chemistry.
            zero_subsurface : If True, zero out subsurface/canopy pathway (for ablation).
            zero_visible    : If True, zero out visible/soil pathway (for ablation).

        Returns:
            t_soil   : (B, shared_dim) soil-routed chemistry representation.
            t_canopy : (B, shared_dim) canopy-routed chemistry representation.
            A        : (M, 2) soft relevance routing matrix.
        """
        B = tabular.size(0)

        # 1. Per-feature Tokenization: list of (B, shared_dim)
        feature_tokens = []
        for i, encoder in enumerate(self.feature_encoders):
            feat_i = tabular[:, i:i+1]  # (B, 1)
            token_i = encoder(feat_i)   # (B, shared_dim)
            feature_tokens.append(token_i)

        # E: (B, M, shared_dim)
        E = torch.stack(feature_tokens, dim=1)

        # 2. Differentiable Soft Routing Matrix A: (M, 2)
        A = self.get_relevance_matrix()

        # Routing weights
        a_soil = A[:, 0].view(1, self.num_features, 1)    # (1, M, 1)
        a_canopy = A[:, 1].view(1, self.num_features, 1)  # (1, M, 1)

        # 3. Aggregate decomposed representations
        t_soil = (E * a_soil).sum(dim=1)      # (B, shared_dim)
        t_canopy = (E * a_canopy).sum(dim=1)  # (B, shared_dim)

        if zero_visible:
            t_soil = torch.zeros_like(t_soil)
        if zero_subsurface:
            t_canopy = torch.zeros_like(t_canopy)

        t_soil = self.norm_soil(t_soil)
        t_canopy = self.norm_canopy(t_canopy)

        return t_soil, t_canopy, A

    def compute_physics_loss(self) -> torch.Tensor:
        """
        Computes Frobenius-norm MSE loss between learned A and domain prior A_prior:
            L_physics = || A - A_prior ||_F^2
        """
        A = self.get_relevance_matrix()
        return F.mse_loss(A, self.A_prior)

    def export_relevance_summary(self) -> List[Dict[str, float]]:
        """
        Returns an interpretable list of dictionaries with learned weights vs priors.
        """
        A = self.get_relevance_matrix().detach().cpu().numpy()
        A_p = self.A_prior.detach().cpu().numpy()

        summary = []
        for i, feat in enumerate(self.feature_names):
            summary.append({
                "feature": feat,
                "learned_soil": float(A[i, 0]),
                "learned_canopy": float(A[i, 1]),
                "prior_soil": float(A_p[i, 0]),
                "prior_canopy": float(A_p[i, 1]),
                "delta_soil": float(A[i, 0] - A_p[i, 0]),
                "delta_canopy": float(A[i, 1] - A_p[i, 1]),
            })
        return summary
