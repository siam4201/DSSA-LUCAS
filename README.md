# Dual Surface-Subsurface Adapter (DSSA)

Official PyTorch implementation of **Dual Surface-Subsurface Adapter (DSSA)** for multimodal land-cover classification using ground-level RGB imagery and soil physicochemical properties on the European LUCAS survey benchmark.

---

## Overview

Ground-level visual observations often fail to discriminate visually identical vegetation types or degraded soil conditions. DSSA integrates ground-level photography (EfficientNet-B0) with eight in-situ topsoil physicochemical measurements (`pH_H2O`, `pH_CaCl2`, `OC`, `CaCO3`, `N`, `P`, `K`, `EC`) via three key components:

1. **Spatial Attention Decomposition Adapter**: Dynamically segments visual spatial tokens into soil, canopy, and background areas.
2. **Physics-Guided Modality Relevance Router (PGMR)**: Routes edaphic properties along surface and subsurface pathways guided by domain constraints.
3. **Adaptive Zero-Soil Gating**: Attenuates unobservable soil cues in closed-canopy scenes.

### Benchmark Results (Frozen Held-Out Test Set, N = 2,917)

| Model Architecture | Modality Stream | Parameters | Test Accuracy | Macro F1 | Cohen's Kappa |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Vision-Only (EffNet-B0) | RGB Only | 4.34M | 80.04% | 0.7264 | 0.7421 |
| Early Concat Fusion | RGB + Chemistry | 4.75M | 81.17% | 0.7434 | 0.7582 |
| Swin-Tiny (ICCV 2021) | RGB Only | 27.52M | 83.41% | 0.7860 | 0.7918 |
| Capacity-Matched Cross-Transformer | RGB + Chemistry | 10.09M | 81.86% | 0.7631 | 0.7710 |
| **DSSA-Lite (Edge)** | **RGB + Chemistry** | **3.99M** | **80.63%** | **0.7487** | **0.7553** |
| **DSSA-Standard (Proposed)** | **RGB + Chemistry** | **8.42M** | **86.20%** | **0.8287** | **0.8286** |

---

## Repository Structure

```
.
├── checkpoints/
│   ├── best_dssa_full_standard.pth   # Proposed DSSA-Standard (86.20% Acc, 32.5 MB)
│   └── dssa_lite_best_seed42.pth     # Proposed DSSA-Lite (80.63% Acc, 15.5 MB)
├── src/
│   ├── models/
│   │   ├── dssa_model.py             # Full DSSA model implementation
│   │   ├── dssa_lite.py              # Lightweight edge DSSA implementation
│   │   ├── spatial_adapter.py        # Spatial attention decomposition adapter
│   │   ├── pgmr_router.py            # Physics-guided modality relevance router
│   │   ├── transformer_baselines.py  # Swin & Capacity-matched cross-transformer
│   │   └── ...
│   ├── config.py                     # Project configuration & hyperparameter defaults
│   ├── dataset.py                    # Multimodal dataset loader & leakage-free splits
│   ├── train_dssa.py                 # Training script for DSSA-Standard
│   ├── train_dssa_lite.py            # Training script for DSSA-Lite
│   ├── evaluate_dssa.py              # Evaluation script
│   └── run_transformer_baselines.py  # Transformer baseline training suite
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/siam4201/DSSA-LUCAS.git
cd DSSA-LUCAS
pip install -r requirements.txt
```

---

## Quick Start: Evaluation

Evaluate the included pre-trained DSSA checkpoint on the test set:

```bash
python src/evaluate_dssa.py --checkpoint checkpoints/best_dssa_full_standard.pth
```

Verify Transformer baseline checkpoints:

```bash
python src/verify_baseline_checkpoints.py
```

---

## Training

### 1. Train Proposed DSSA-Standard

```bash
python src/train_dssa.py --spatial --physics_weight 0.05 --spatial_phys_weight 0.05 --canopy_weight 0.05
```

### 2. Train DSSA-Lite (Edge-Optimized)

```bash
python src/train_dssa_lite.py --seed 42
```

### 3. Train Transformer Baselines

```bash
python src/run_transformer_baselines.py --train_model capacity_cross --batch_size 64
python src/run_transformer_baselines.py --train_model swin --batch_size 64
```

---

## Pre-trained Weights & Hugging Face

Pre-trained weights for DSSA-Standard and DSSA-Lite are available directly in `checkpoints/` and on the Hugging Face Model Hub:
- Hugging Face Model Hub: `https://huggingface.co/<your-hf-username>/DSSA-LUCAS`

