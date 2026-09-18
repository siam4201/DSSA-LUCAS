"""
utils.py
--------
Shared utilities: device selection, reproducibility seeding,
EarlyStopping, metric tracking, and directory setup.
"""

import os
import random
import numpy as np
import torch
from typing import Optional


# ── Device ───────────────────────────────────────────────────────────────────

def get_device(preferred: str = "cuda") -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  [Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("  [Device] CUDA not available — using CPU")
    return device


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility while allowing modern CUDA kernel selection."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


# ── Directory setup ───────────────────────────────────────────────────────────

def setup_dirs(*dirs: str) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when validation loss doesn't improve for `patience` epochs.
    Saves the best model checkpoint automatically.
    """

    def __init__(self, patience: int = 10, delta: float = 1e-4,
                 checkpoint_path: str = "best_model.pth"):
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.best_loss: Optional[float] = None
        self.counter: int = 0
        self.early_stop: bool = False

    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Call after each epoch.
        Returns True if training should stop.
        """
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"    [EarlyStopping] Validation loss improved -> {val_loss:.4f} | checkpoint saved")
        else:
            self.counter += 1
            print(f"    [EarlyStopping] No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                print("    [EarlyStopping] Triggered - stopping training.")
        return self.early_stop


# ── Metric Tracker ────────────────────────────────────────────────────────────

class MetricTracker:
    """Accumulates running averages for loss and accuracy."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._loss_sum = 0.0
        self._correct = 0
        self._total = 0
        self._count = 0

    def update(self, loss: float, preds: torch.Tensor, labels: torch.Tensor):
        self._loss_sum += loss
        self._count += 1
        self._correct += (preds.argmax(dim=1) == labels).sum().item()
        self._total += labels.size(0)

    @property
    def avg_loss(self) -> float:
        return self._loss_sum / max(self._count, 1)

    @property
    def accuracy(self) -> float:
        return self._correct / max(self._total, 1)

    def compute(self) -> dict:
        return {"loss": self.avg_loss, "acc": self.accuracy}

    def report(self, prefix: str = "") -> str:
        return (f"{prefix}Loss: {self.avg_loss:.4f} | "
                f"Acc: {self.accuracy * 100:.2f}%")
