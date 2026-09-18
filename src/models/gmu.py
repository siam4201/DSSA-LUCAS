"""
models/gmu.py
-------------
Step 3: Gated Multimodal Unit (GMU)

Given the three feature vectors from Steps 1 & 2:
  • f_vis   ∈ ℝ^d  (visual)
  • f_tab   ∈ ℝ^d  (tabular)
  • f_cross ∈ ℝ^d  (cross-attended)

The GMU computes:

  Gate vector:     z     = σ( W_z [f_vis, f_tab, f_cross] + b_z )     ∈ ℝ^d
  Candidate:       h     = tanh( W_f [f_vis, f_tab, f_cross] + b_f )   ∈ ℝ^d
  Fused output:    f_fused = z ⊙ h                                      ∈ ℝ^d

Properties:
  - The sigmoid gate squashes z ∈ (0, 1), acting as a soft switch.
  - When tabular sensor data is zeroed-out (missing), the gate learns
    to dynamically close the tabular channel and rely on visual features.
  - The gate values z can be inspected at inference to understand which
    modalities the model is prioritising per sample.
"""

import torch
import torch.nn as nn


class GatedMultimodalUnit(nn.Module):
    """
    Three-way GMU fusing f_vis, f_tab, and f_cross.

    Args:
        shared_dim  : Dimensionality of each input feature (d).
        dropout     : Dropout applied before the gate and candidate layers.
    """

    def __init__(self, shared_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        input_dim = shared_dim * 3          # [f_vis, f_tab, f_cross]

        self.dropout = nn.Dropout(dropout)

        # W_z, b_z  →  gating vector
        self.gate_layer = nn.Linear(input_dim, shared_dim)

        # W_f, b_f  →  candidate fusion
        self.fusion_layer = nn.Linear(input_dim, shared_dim)

        # Final normalisation
        self.norm = nn.LayerNorm(shared_dim)

    def forward(
        self,
        f_vis: torch.Tensor,
        f_tab: torch.Tensor,
        f_cross: torch.Tensor,
    ) -> tuple:
        """
        Args:
            f_vis  : (B, d)
            f_tab  : (B, d)
            f_cross: (B, d)

        Returns:
            f_fused: (B, d) — gated multimodal representation
            z      : (B, d) — gate values (for interpretability / logging)
        """
        concat = torch.cat([f_vis, f_tab, f_cross], dim=-1)    # (B, 3d)
        concat = self.dropout(concat)

        # Gate: z = σ(W_z · concat + b_z)
        z = torch.sigmoid(self.gate_layer(concat))              # (B, d)

        # Candidate: h = tanh(W_f · concat + b_f)
        h = torch.tanh(self.fusion_layer(concat))               # (B, d)

        # Gated fusion: element-wise product
        f_fused = z * h                                         # (B, d)
        f_fused = self.norm(f_fused)

        return f_fused, z
