"""
train_dssa.py
-------------
Training script for the proposed Domain-Specific Spatial Attention (DSSA) model
incorporating:
  1. Spatial Attention Adapter with Dynamic Canopy-Gated Zero-Soil Suppression
  2. Physics-Guided Modality Relevance Matrix (PGMR) Router
  3. Domain-Aligned Cross-Attention & Dual Gated Multimodal Units (GMU)
  4. Biophysical Spatial Loss (ExG/VARI) + Canopy Density Supervision + Physics Regularization
"""

import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import cfg
from dataset import build_dataloaders
from models import DSSAModel, SpatialDecompositionAdapter
from utils import get_device, MetricTracker, EarlyStopping, set_seed


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool = True,
    ortho_weight: float = 0.1,
    physics_weight: float = 0.05,
    spatial_phys_weight: float = 0.05,
    canopy_weight: float = 0.05,
    use_modality_dropout: bool = True,
    p_drop_tabular: float = 0.10,
    p_drop_vision: float = 0.10,
) -> tuple:
    model.train()
    tracker = MetricTracker()
    total_phys_loss = 0.0
    total_spatial_phys = 0.0
    total_ortho_loss = 0.0
    num_batches = 0

    for images, tabular, labels in loader:
        images  = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        # Modality Dropout Augmentation during training
        if use_modality_dropout:
            B = images.size(0)
            rand_vals = torch.rand(B, device=device)
            # 10% Vision-Only (Drop Tabular)
            mask_drop_tab = (rand_vals < p_drop_tabular).unsqueeze(1)
            tabular = torch.where(mask_drop_tab, torch.zeros_like(tabular), tabular)
            # 10% Chemistry-Only (Drop Vision)
            mask_drop_vis = ((rand_vals >= p_drop_tabular) & (rand_vals < (p_drop_tabular + p_drop_vision))).view(B, 1, 1, 1)
            images = torch.where(mask_drop_vis, torch.zeros_like(images), images)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            logits = model(images, tabular)
            loss_main = criterion(logits, labels)

            # Spatial Disentanglement Loss (Orthogonality penalty between soil & canopy masks)
            if hasattr(model, "get_attention_maps") and model.get_attention_maps():
                attn_maps = model.get_attention_maps()
                soil_m = attn_maps.get("soil_map")
                canopy_m = attn_maps.get("canopy_map")
                if soil_m is not None and canopy_m is not None and soil_m.numel() > 1:
                    loss_ortho = (soil_m * canopy_m).sum(dim=(-1, -2, -3)).mean()
                else:
                    loss_ortho = 0.0
            else:
                loss_ortho = 0.0

            # Chemistry PGMR Regularization Loss
            if hasattr(model, "compute_physics_loss") and physics_weight > 0.0:
                loss_phys = model.compute_physics_loss()
            else:
                loss_phys = torch.tensor(0.0, device=device)

            # Visual Spatial Biophysical Supervision Loss (ExG / VARI / CI + Canopy Density)
            if hasattr(model, "compute_spatial_physics_loss") and spatial_phys_weight > 0.0:
                loss_spatial_phys = model.compute_spatial_physics_loss(
                    images, canopy_weight=canopy_weight
                )
            else:
                loss_spatial_phys = torch.tensor(0.0, device=device)

            loss = (
                loss_main
                + ortho_weight * loss_ortho
                + physics_weight * loss_phys
                + spatial_phys_weight * loss_spatial_phys
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        tracker.update(loss.item(), logits.detach(), labels)
        total_phys_loss += loss_phys.item() if isinstance(loss_phys, torch.Tensor) else float(loss_phys)
        total_spatial_phys += loss_spatial_phys.item() if isinstance(loss_spatial_phys, torch.Tensor) else float(loss_spatial_phys)
        total_ortho_loss += loss_ortho.item() if isinstance(loss_ortho, torch.Tensor) else float(loss_ortho)
        num_batches += 1

    metrics = tracker.compute()
    avg_phys = total_phys_loss / max(num_batches, 1)
    avg_spatial_phys = total_spatial_phys / max(num_batches, 1)
    avg_ortho = total_ortho_loss / max(num_batches, 1)

    return metrics["loss"], metrics["acc"], avg_phys, avg_spatial_phys, avg_ortho


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> tuple:
    model.eval()
    tracker = MetricTracker()

    for images, tabular, labels in loader:
        images  = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        with autocast("cuda", enabled=use_amp):
            logits = model(images, tabular)
            loss = criterion(logits, labels)

        tracker.update(loss.item(), logits, labels)

    metrics = tracker.compute()
    return metrics["loss"], metrics["acc"]


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    cfg.random_seed = args.seed

    device = get_device(cfg.device)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    if args.spatial:
        cfg.use_spatial_attention = True
    if args.freeze_backbone:
        cfg.freeze_backbone = True

    print("\n" + "=" * 60)
    print("  DSSA (Dynamic Zero-Soil + PGMR) Training Configuration")
    print("=" * 60)
    cfg.display()

    # ── Data Loaders ──────────────────────────────────────────────────────────
    print("\n  Loading dataset...")
    train_loader, val_loader, test_loader, _ = build_dataloaders(
        csv_path=cfg.tabular_csv,
        feature_cols=cfg.continuous_features,
        val_split=cfg.val_split,
        test_split=cfg.test_split,
        batch_size=cfg.batch_size,
        img_size=cfg.img_size,
        num_workers=cfg.num_workers,
        seed=args.seed,
        label_col=cfg.label_col,
    )
    print(
        f"  Dataset split - Train: {len(train_loader.dataset)} | "
        f"Val: {len(val_loader.dataset)} | "
        f"Test: {len(test_loader.dataset)}"
    )

    # ── Compute Class Balanced Weights ────────────────────────────────────────
    class_counts = train_loader.dataset.df[cfg.label_col].value_counts().sort_index().values
    alpha = args.class_weight_alpha
    raw_weights = 1.0 / (np.power(class_counts, alpha) + 1e-6)
    weights = raw_weights / raw_weights.sum() * len(class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"\n  Class balanced loss weights (alpha={alpha}):")
    for name, cnt, w in zip(cfg.soil_classes, class_counts, weights):
        print(f"    {name:<22} (N={cnt:>5}): weight = {w:.3f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    use_physics = not args.no_physics
    use_soil_gating = not args.no_soil_gating
    model = DSSAModel(
        feature_cols=cfg.continuous_features,
        visible_cols=cfg.visible_chemistry_features,
        subsurface_cols=cfg.subsurface_chemistry_features,
        num_classes=cfg.num_classes,
        shared_dim=cfg.shared_dim,
        num_heads=cfg.num_heads,
        cnn_backbone=cfg.cnn_backbone,
        use_spatial_attention=cfg.use_spatial_attention,
        use_physics_guidance=use_physics,
        use_soil_gating=use_soil_gating,
        dropout=cfg.dropout,
        pretrained=True,
        freeze_backbone=cfg.freeze_backbone,
    ).to(device)

    counts = model.parameter_count()
    print(f"\n  DSSA Model parameter counts (use_physics_guidance={use_physics}, use_soil_gating={use_soil_gating}):")
    for k, v in counts.items():
        print(f"    {k:<26} {v:>10,}")

    # ── Differential Optimizer / Scheduler / Loss ─────────────────────────────
    backbone_params = list(model.visual_encoder.backbone.parameters())
    backbone_param_ids = set(id(p) for p in backbone_params)
    head_params = [p for p in model.parameters() if id(p) not in backbone_param_ids and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.lr * 0.1, "weight_decay": cfg.weight_decay},
        {"params": head_params,     "lr": cfg.lr,       "weight_decay": cfg.weight_decay},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    amp_scaler = GradScaler(enabled=cfg.use_amp)

    mode_tag = "spatial" if cfg.use_spatial_attention else "pooled"
    if cfg.freeze_backbone:
        mode_tag += "_peft"
    if use_physics:
        mode_tag += f"_pgmr_w{args.physics_weight}"
    if cfg.use_spatial_attention and args.spatial_phys_weight > 0:
        mode_tag += f"_spatw{args.spatial_phys_weight}"
    if use_soil_gating:
        mode_tag += f"_soilgate_cw{args.canopy_weight}"

    ckpt_path = os.path.join(cfg.checkpoint_dir, f"dssa_best_{mode_tag}.pth")
    early_stop = EarlyStopping(patience=args.patience, checkpoint_path=ckpt_path)

    # ── Calibrate threshold tau from training canopy prior distribution ───────
    if use_soil_gating and cfg.use_spatial_attention:
        forest_indices = [2, 3]  # Broadleaf Woodland, Coniferous Woodland
        print("  Calibrating soil gate threshold (tau) from training canopy prior...")
        tau_cal = SpatialDecompositionAdapter.calibrate_threshold(
            adapter=model.spatial_adapter,
            dataloader=train_loader,
            forest_class_indices=forest_indices,
            device=device,
            percentile=60.0,
        )
        model.spatial_adapter.tau.data.fill_(tau_cal)
        print(f"  Calibrated tau = {tau_cal:.4f} (60th percentile of forest canopy prior)")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n  Starting DSSA training for up to {cfg.epochs} epochs (Physics W: {args.physics_weight}, Spatial Phys W: {args.spatial_phys_weight}, Canopy W: {args.canopy_weight})...\n")
    history = {"train_loss": [], "train_acc": [], "train_phys": [], "train_spatial_phys": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss, train_acc, train_phys, train_spatial_phys, train_ortho = train_one_epoch(
            model, train_loader, optimizer, criterion_ce, amp_scaler, device, cfg.use_amp,
            ortho_weight=args.ortho_weight,
            physics_weight=args.physics_weight,
            spatial_phys_weight=args.spatial_phys_weight,
            canopy_weight=args.canopy_weight,
            use_modality_dropout=not args.no_modality_dropout,
            p_drop_tabular=args.p_drop_tabular,
            p_drop_vision=args.p_drop_vision,
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion_ce, device, cfg.use_amp
        )

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_phys"].append(train_phys)
        history["train_spatial_phys"].append(train_spatial_phys)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch {epoch:>3}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} (ChemPhys: {train_phys:.4f}, SpatPhys: {train_spatial_phys:.4f}) Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # Log gate parameters for monitoring
        if use_soil_gating and cfg.use_spatial_attention:
            tau_v  = model.spatial_adapter.tau.item()
            beta_v = model.spatial_adapter.beta.item()
            print(f"           |-- Soil Gate | tau={tau_v:.4f}  beta={beta_v:.4f}")

        if early_stop(val_loss, model):
            break

    print(f"\n  Training completed. Best checkpoint saved -> {ckpt_path}")

    # Load best checkpoint for final evaluation
    model.load_state_dict(torch.load(ckpt_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion_ce, device, cfg.use_amp)
    print(f"  [Final Test Evaluation] Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    # Learned PGMR relevance matrix summary
    if use_physics and hasattr(model, "pgmr_router") and model.pgmr_router is not None:
        summary = model.pgmr_router.export_relevance_summary()
        if summary:
            print("  -- Learned Physics-Guided Modality Relevance Matrix (PGMR) --")
            print(f"  {'Feature':<12} {'Soil (Learned)':<16} {'Canopy (Learned)':<18} {'Prior [S / C]':<18} {'Delta (Soil)':<14}")
            print("  " + "-" * 78)
            for row in summary:
                s_l = f"{row['learned_soil']:.3f}"
                c_l = f"{row['learned_canopy']:.3f}"
                p_sc = f"[{row['prior_soil']:.1f} / {row['prior_canopy']:.1f}]"
                d_s = f"{row['delta_soil']:+.3f}"
                print(f"  {row['feature']:<12} {s_l:<16} {c_l:<18} {p_sc:<18} {d_s:<14}")
            print("  " + "-" * 78)

            pgmr_path = os.path.join(cfg.log_dir, f"pgmr_matrix_{mode_tag}.json")
            with open(pgmr_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  Relevance matrix summary saved -> {pgmr_path}")

    # Save training history
    hist_path = os.path.join(cfg.log_dir, f"history_dssa_{mode_tag}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved -> {hist_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DSSA model (Dynamic Zero-Soil + PGMR + Modality Dropout)")
    parser.add_argument("--spatial", action="store_true",
                        help="Enable spatial attention mode")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze visual CNN backbone for parameter-efficient adapter training")
    parser.add_argument("--physics_weight", type=float, default=0.05,
                        help="Loss weight lambda_phys for Chemistry PGMR Prior Regularization (default: 0.05)")
    parser.add_argument("--spatial_phys_weight", type=float, default=0.05,
                        help="Loss weight for Biophysical Spatial Attention Supervision (default: 0.05)")
    parser.add_argument("--canopy_weight", type=float, default=0.05,
                        help="Loss weight for Scene Canopy Density Supervision (default: 0.05)")
    parser.add_argument("--ortho_weight", type=float, default=0.1,
                        help="Loss weight for Spatial Mask Orthogonality (default: 0.1)")
    parser.add_argument("--no_physics", action="store_true",
                        help="Disable PGMR and use hardcoded legacy split")
    parser.add_argument("--no_soil_gating", action="store_true",
                        help="Disable Dynamic Zero-Soil Gating")
    parser.add_argument("--class_weight_alpha", type=float, default=0.50,
                        help="Class weighting exponent alpha in w_c = 1/(n_c^alpha) (default: 0.50 inverse-sqrt)")
    parser.add_argument("--no_modality_dropout", action="store_true",
                        help="Disable Modality Dropout during training (default: enabled)")
    parser.add_argument("--p_drop_tabular", type=float, default=0.10,
                        help="Probability of dropping tabular chemistry during training (default: 0.10)")
    parser.add_argument("--p_drop_vision", type=float, default=0.10,
                        help="Probability of dropping image vision during training (default: 0.10)")
    parser.add_argument("--label_smoothing", type=float, default=cfg.label_smoothing,
                        help="Label smoothing factor for CrossEntropyLoss (default: 0.10)")
    parser.add_argument("--patience", type=int, default=cfg.patience,
                        help="Early stopping patience in epochs (default: 15)")
    parser.add_argument("--seed", type=int, default=cfg.random_seed,
                        help=f"Random seed for reproducibility and stratified splits (default: {cfg.random_seed})")
    parser.add_argument("--workers", type=int, default=cfg.num_workers,
                        help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=cfg.epochs,
                        help="Epochs")

    args = parser.parse_args()
    if args.workers != cfg.num_workers:
        cfg.num_workers = args.workers
    if args.batch_size != cfg.batch_size:
        cfg.batch_size = args.batch_size
    if args.epochs != cfg.epochs:
        cfg.epochs = args.epochs

    main(args)
