"""
models/tabular_encoder.py
-------------------------
Step 1b: Tabular Modality Encoder

  f_tab = W_tab · φ_tab(x_tab)  ∈ ℝ^d

Architecture:
  - Gated Residual Network (GRN) block as φ_tab.
    Inspired by the Temporal Fusion Transformer (Lim et al., 2021).
    Uses GLU (Gated Linear Unit) gating + skip connection + LayerNorm.
  - Final Linear projection (W_tab) maps to shared_dim d.

The GRN provides a more expressive nonlinear transformation than a plain MLP
while still being lightweight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    """
    GLU gating: splits input into two halves, applies sigmoid to one,
    then element-wise multiplies.  output_dim = input_dim // 2.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        assert input_dim % 2 == 0, "input_dim must be even for GLU"
        self.fc = nn.Linear(input_dim, input_dim)   # produce gate + value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        half = h.shape[-1] // 2
        value, gate = h[..., :half], h[..., half:]
        return value * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """
    Single GRN block:
        y = LayerNorm(x + GLU(ELU(Linear(ELU(Linear(x))))))

    Args:
        input_dim   : Dimensionality of the input.
        hidden_dim  : Internal hidden dimensionality.
        output_dim  : Output dimensionality.
        dropout     : Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)   # × 2 for GLU
        self.glu = GatedLinearUnit(hidden_dim * 2)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # Skip connection (projected if dims differ)
        self.skip = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        h = F.elu(self.fc1(x))
        h = self.glu(self.fc2(h))           # (B, hidden_dim)
        h = self.dropout(h)
        h = self.fc_out(h)                  # (B, output_dim)

        return self.norm(h + residual)


class TabularEncoder(nn.Module):
    """
    Tabular encoder: GRN(s) → Linear projection → shared dim d.

    Args:
        num_continuous  : Number of continuous input features.
        shared_dim      : Target projection dimension d.
        hidden_dim      : GRN internal width.
        num_grn_layers  : Number of stacked GRN blocks.
        dropout         : Dropout rate.
    """

    def __init__(
        self,
        num_continuous: int,
        shared_dim: int = 256,
        hidden_dim: int = 128,
        num_grn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input → first hidden
        grns = [GatedResidualNetwork(num_continuous, hidden_dim, hidden_dim, dropout)]
        # Additional hidden layers
        for _ in range(num_grn_layers - 1):
            grns.append(GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout))
        self.grn_stack = nn.Sequential(*grns)

        # W_tab: project GRN output to shared_dim
        self.proj = nn.Linear(hidden_dim, shared_dim)
        self.norm = nn.LayerNorm(shared_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_continuous) normalised tabular features

        Returns:
            f_tab: (B, d)
        """
        h = self.grn_stack(x)               # (B, hidden_dim)
        h = self.dropout(h)
        f_tab = self.norm(self.proj(h))     # (B, d)
        return f_tab
