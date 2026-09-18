"""
config.py
---------
Central configuration for the Gated Cross-Modal Attention (GCMA) project.
All hyperparameters and data paths are defined here.

Environment overrides (set before running):
  TRAIN_SEED          — random seed (default: 42)
  HF_HOME             — HuggingFace cache root (default: D:\\models\\huggingface)
"""

import os
from dataclasses import dataclass, field
from typing import List

# ── HuggingFace / Triton cache redirection ────────────────────────────────────
# Mirrors the pattern from the thesis LLM pipeline (ref/homo_llama_config.py)
# so that timm's EfficientNet weights download to D:\models instead of
# the default C:\Users\...\AppData location.
HF_CACHE_DIR = r"D:\models"
os.environ.setdefault("HF_HOME",             os.path.join(HF_CACHE_DIR, "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE",   os.path.join(HF_CACHE_DIR, "huggingface", "datasets"))
os.environ.setdefault("TRITON_CACHE_DIR",    os.path.join(HF_CACHE_DIR, "triton"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ── Seed (overridable from shell: set TRAIN_SEED=3407) ───────────────────────
_TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "42"))


@dataclass
class Config:
    # ── Paths ────────────────────────────────────────────────────────────────
    # Paths are relative to the project root (d:\Data_Mining\), not src/
    data_root: str = r"d:\Data_Mining\data"
    image_dir: str = r"d:\Data_Mining\data\images"          # Kaggle dataset root
    tabular_csv: str = r"d:\Data_Mining\data\soil_tabular.csv"
    checkpoint_dir: str = r"d:\Data_Mining\checkpoints"
    log_dir: str = r"d:\Data_Mining\logs"

    # Level-2 LUCAS land-cover classes used as classification targets (6-class benchmark).
    # Removed transitional 'Other Grassland' (E10/E30) to establish a clean, non-overlapping taxonomy.
    soil_classes: List[str] = field(default_factory=lambda: [
        "Cereals",
        "Other Cropland",
        "Broadleaf Woodland",
        "Coniferous Woodland",
        "Shrubland",
        "Managed Grassland",
    ])
    num_classes: int = 6

    # Which CSV column contains the integer class label.
    # 'label'      -> original 3-class (Cropland / Woodland / Grassland)
    # 'label_lc2'  -> 6-class Level-2 hierarchy (current setting)
    label_col: str = "label_lc2"

    # ── Hierarchical classification settings ──────────────────────────────────
    # Coarse (Level-1) label column — already in the CSV as 'label' (0/1/2)
    coarse_label_col: str = "label"
    coarse_classes: List[str] = field(default_factory=lambda: [
        "Cropland", "Woodland", "Grassland"
    ])
    num_coarse_classes: int = 3
    # Loss weight for the coarse head: L = alpha*CE_coarse + (1-alpha)*CE_fine
    hier_loss_alpha: float = 0.3

    # Real LUCAS topsoil features — 8 CHEMISTRY features only.
    # Texture features (Clay, Sand, Silt, Coarse) are excluded because they
    # were only measured for new 2015 points (~4k rows), not revisited points
    # (~17k rows). Using chemistry-only keeps all ~21k rows without imputation.
    # Raw CSV → internal name mapping:
    #   "pH(H2O)"  → "pH_H2O"     "pH(CaCl2)" → "pH_CaCl2"
    #   OC, CaCO3, N, P, K, EC — unchanged
    # Reference: Ballabio et al. (2019), DOI: 10.1111/ejss.12752
    continuous_features: List[str] = field(default_factory=lambda: [
        "pH_H2O",   # pH in water      (raw: "pH(H2O)")
        "pH_CaCl2", # pH in CaCl2      (raw: "pH(CaCl2)")
        "OC",       # Organic Carbon (g/kg)
        "CaCO3",    # Calcium Carbonate (g/kg)
        "N",        # Total Nitrogen (g/kg)
        "P",        # Phosphorus (mg/kg)
        "K",        # Potassium (mg/kg)
        "EC",       # Electrical Conductivity (mS/m)
    ])
    num_continuous_features: int = 8    # 8 chemistry features (all measured for every point)

    # ── DSSA (Dual Surface vs Subsurface Decomposition) Settings ──────────────
    # Visually evident chemistry: features that directly affect optical surface color/tint
    visible_chemistry_features: List[str] = field(default_factory=lambda: [
        "OC",       # Organic Carbon (soil darkening/humic content)
        "CaCO3",    # Calcium Carbonate (pale/chalky tint)
    ])
    # Subsurface chemistry: invisible root-zone nutrients, acidity, salinity
    subsurface_chemistry_features: List[str] = field(default_factory=lambda: [
        "pH_H2O", "pH_CaCl2", "N", "P", "K", "EC"
    ])
    # Parameter-efficient fine-tuning toggle: if True, freeze CNN backbone and train only adapter
    freeze_backbone: bool = False

    # Constrained Modality Arbitration toggle
    use_constrained_arbitration: bool = True
    aux_modality_weight: float = 0.2

    # No categorical features used (LC1 label is the target, not a feature)
    categorical_features: List[str] = field(default_factory=list)
    num_categorical_features: int = 0

    val_split: float = 0.15          # fraction of train set used for validation
    test_split: float = 0.15         # fraction of total used for test
    random_seed: int = _TRAIN_SEED   # overridable via TRAIN_SEED env var

    # ── Model Architecture ────────────────────────────────────────────────────
    shared_dim: int = 256            # d — shared projection dimension
    num_heads: int = 8               # multi-head attention heads
    cnn_backbone: str = "efficientnet_b0"
    dropout: float = 0.1

    # Spatial attention toggle
    # False → pooled (single d-dim vector per modality)
    # True  → spatial patch tokens from the CNN feature map
    use_spatial_attention: bool = False
    spatial_tokens: int = 49         # 7×7 spatial grid (EfficientNet-B0 at 224px)

    # ── Training ──────────────────────────────────────────────────────────────
    task: str = "classification"     # "classification" or "regression"
    lr: float = 5e-4
    weight_decay: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    img_size: int = 224
    num_workers: int = 8
    use_amp: bool = True             # automatic mixed precision

    # Modality dropout augmentation during training
    # Teaches the cross-modal network to be resilient to missing/noisy modalities
    use_modality_dropout: bool = True
    p_drop_tabular: float = 0.10      # 10% Vision-only
    p_drop_vision: float = 0.10       # 10% Chemistry-only
    tabular_dropout_prob: float = 0.10 # Backward-compatibility alias

    # Early stopping
    patience: int = 15
    label_smoothing: float = 0.10

    # ── Misc ──────────────────────────────────────────────────────────────────
    device: str = "cuda"             # "cuda" or "cpu" (auto-detected in utils.py)

    def display(self) -> None:
        """Pretty-print configuration."""
        print_config(self)


# Singleton config instance used project-wide
cfg = Config()


def print_config(cfg: Config) -> None:
    """Pretty-print the active configuration."""
    print("\n" + "=" * 60)
    print("  Gated Cross-Modal Attention — Configuration")
    print("=" * 60)
    for k, v in cfg.__dict__.items():
        print(f"  {k:<35} {v}")
    print("=" * 60 + "\n")
