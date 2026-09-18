"""
dataset.py
----------
MultimodalSoilDataset: loads (image, tabular_features, label) tuples.

Expects a CSV produced by generate_tabular_data.py with columns:
    image_path, soil_class, pH, moisture, ..., label

Images are loaded from the paths recorded in the CSV.
Tabular features are z-score normalised using training-set statistics.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List

from config import cfg


# ── Image transforms ──────────────────────────────────────────────────────────

def get_transforms(split: str, img_size: int = 224):
    """
    Return torchvision transforms for 'train', 'val', or 'test'.
    Training applies augmentation; val/test only resize and normalise.
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet stats (good default for soil images)
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultimodalSoilDataset(Dataset):
    """
    Returns: (image_tensor [3,H,W], tabular_tensor [num_features], label [int])

    Args:
        df              : DataFrame with image_path, feature columns, and label.
        feature_cols    : List of continuous feature column names.
        scaler          : Fitted StandardScaler (None → fit on this split, warn).
        split           : 'train', 'val', or 'test'.
        img_size        : Image resolution.
        tabular_dropout : Probability of zeroing out ALL tabular features
                          (simulates missing sensor data, train split only).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        scaler: Optional[StandardScaler] = None,
        split: str = "train",
        img_size: int = 224,
        tabular_dropout: float = 0.0,
        label_col: str = "label",
        hierarchical: bool = False,
        coarse_label_col: str = "label",
    ):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.split = split
        self.img_size = img_size
        self.label_col = label_col
        self.hierarchical = hierarchical
        self.coarse_label_col = coarse_label_col
        self.tabular_dropout = tabular_dropout if split == "train" else 0.0
        self.transform = get_transforms(split, img_size)

        # Fit or apply scaler
        raw = self.df[self.feature_cols].values.astype(np.float32)
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(raw)
        self.scaler = scaler
        self.features = scaler.transform(raw).astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]

        # ── Image ─────────────────────────────────────────────────────────────
        img_path = row["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Return a black image if loading fails (robustness)
            image = Image.new("RGB", (self.img_size, self.img_size), color=0)
        image = self.transform(image)

        # ── Tabular ───────────────────────────────────────────────────────────
        tabular = torch.tensor(self.features[idx], dtype=torch.float32)

        # Tabular dropout: randomly zero out all features to teach GMU resilience
        if self.tabular_dropout > 0.0 and torch.rand(1).item() < self.tabular_dropout:
            tabular = torch.zeros_like(tabular)

        # -- Label(s) -----------------------------------------------------------
        if self.hierarchical:
            coarse_label = int(row[self.coarse_label_col])
            fine_label   = int(row[self.label_col])
            return image, tabular, coarse_label, fine_label
        label = int(row[self.label_col])
        return image, tabular, label


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(
    csv_path: str,
    feature_cols: List[str],
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    tabular_dropout: float = 0.1,
    seed: int = 42,
    label_col: str = "label",
    hierarchical: bool = False,
    coarse_label_col: str = "label",
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Splits the CSV into train/val/test sets and returns three DataLoaders
    plus the fitted scaler (to apply identical normalisation at inference).

    When hierarchical=True, each batch yields (image, tabular, coarse_label, fine_label).
    """
    df = pd.read_csv(csv_path)

    # Stratified split so class distribution is preserved
    train_df, test_df = train_test_split(
        df, test_size=test_split,
        stratify=df[label_col], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_split / (1.0 - test_split),
        stratify=train_df[label_col], random_state=seed
    )

    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values.astype(np.float32))

    ds_kwargs = dict(hierarchical=hierarchical, coarse_label_col=coarse_label_col)

    train_ds = MultimodalSoilDataset(
        train_df, feature_cols, scaler, "train", img_size, tabular_dropout,
        label_col, **ds_kwargs
    )
    val_ds = MultimodalSoilDataset(
        val_df, feature_cols, scaler, "val", img_size,
        label_col=label_col, **ds_kwargs
    )
    test_ds = MultimodalSoilDataset(
        test_df, feature_cols, scaler, "test", img_size,
        label_col=label_col, **ds_kwargs
    )

    train_loader = DataLoader(
        train_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, shuffle=False, batch_size=batch_size, num_workers=min(num_workers, 2),
        pin_memory=True, persistent_workers=False
    )
    test_loader = DataLoader(
        test_ds, shuffle=False, batch_size=batch_size, num_workers=min(num_workers, 2),
        pin_memory=True, persistent_workers=False
    )

    print(f"  Dataset split - Train: {len(train_ds)} | "
          f"Val: {len(val_ds)} | Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader, scaler

