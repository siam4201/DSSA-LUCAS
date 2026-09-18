"""
evaluate_dssa.py
----------------
Full evaluation and ablation of a trained DSSA checkpoint.

Reports:
  1. Full multimodal test accuracy (Image + Visible Chemistry + Subsurface Chemistry)
  2. Classification report (precision, recall, F1 per class)
  3. Domain Ablations:
       • Vision-only (both chemistry channels zeroed)
       • Surface-only chemistry (subsurface zeroed: tests value of hidden pH/NPK/EC)
       • Subsurface-only chemistry (visible zeroed: tests value of surface OC/CaCO3)
  4. Mean GMU gate values per class for:
       • Surface gating (z_surface)
       • Subsurface gating (z_subsurface)
       • Top-level fusion gating (z_fusion)
  5. Confusion matrix PNG plot

Usage:
    python evaluate_dssa.py --checkpoint checkpoints/dssa_best_spatial.pth --spatial
"""

import os
import argparse
import json
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from config import cfg
from dataset import build_dataloaders
from models import DSSAModel
from utils import get_device, set_seed


CLASS_NAMES = cfg.soil_classes


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool = True,
    eval_domain: Optional[str] = None,
):
    """Run inference with optional domain ablation flags."""
    model.eval()
    all_preds, all_labels = [], []
    all_z_surface, all_z_subsurface, all_z_fusion = [], [], []
    spatial_alignments = {"soil": [], "canopy": [], "bg": []}

    for images, tabular, labels in loader:
        images  = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(
                images,
                tabular,
                eval_domain=eval_domain,
            )

        preds = logits.argmax(dim=1)
        gates = model.get_gate_values()

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        if "z_surface" in gates:
            all_z_surface.append(gates["z_surface"].cpu())
            all_z_subsurface.append(gates["z_subsurface"].cpu())
            all_z_fusion.append(gates["z_fusion"].cpu())

        # Quantitative biophysical optical alignment
        if hasattr(model, "compute_spatial_alignment_metrics"):
            align = model.compute_spatial_alignment_metrics(images)
            if align:
                spatial_alignments["soil"].append(align.get("soil_biophysical_alignment", 0.0))
                spatial_alignments["canopy"].append(align.get("canopy_biophysical_alignment", 0.0))
                spatial_alignments["bg"].append(align.get("bg_biophysical_alignment", 0.0))

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    gate_dict = {}
    if all_z_surface:
        gate_dict["z_surface"]    = torch.cat(all_z_surface).numpy()
        gate_dict["z_subsurface"] = torch.cat(all_z_subsurface).numpy()
        gate_dict["z_fusion"]     = torch.cat(all_z_fusion).numpy()

    avg_spatial_align = {}
    for k, v in spatial_alignments.items():
        if v:
            avg_spatial_align[k] = float(np.mean(v))

    return all_preds, all_labels, gate_dict, avg_spatial_align


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
    )
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)
    ax.set_title("Confusion Matrix — DSSA Model", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved -> {save_path}")


def plot_gate_breakdown(gate_means_dict, save_path):
    """Plots comparative gate activities for Surface, Subsurface, and Fusion."""
    classes = CLASS_NAMES
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width, [gate_means_dict["surface"][c] for c in classes], width, label="Surface Chemistry Gate (z_surf)", color="#4C72B0")
    rects2 = ax.bar(x,         [gate_means_dict["subsurface"][c] for c in classes], width, label="Subsurface Chemistry Gate (z_sub)", color="#55A868")
    rects3 = ax.bar(x + width, [gate_means_dict["fusion"][c] for c in classes], width, label="Master Fusion Gate (z_fuse)", color="#DD8452")

    ax.set_ylabel("Mean Gate Activity (0 to 1)", fontsize=11)
    ax.set_title("DSSA Gate Activation Breakdown per Class", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Gate breakdown plot saved -> {save_path}")


def main(args):
    set_seed(cfg.random_seed)
    device = get_device(cfg.device)

    if args.spatial:
        cfg.use_spatial_attention = True

    # ── Data ──────────────────────────────────────────────────────────────────
    _, _, test_loader, _ = build_dataloaders(
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

    # ── Model ─────────────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Check if checkpoint was trained with PGMR, soil gating, or legacy split
    has_pgmr = any("pgmr_router" in k for k in ckpt.keys())
    has_soilgate = any("spatial_adapter.tau" in k for k in ckpt.keys())
    use_physics = has_pgmr and not args.no_physics
    use_soil_gating = has_soilgate and not args.no_soil_gating

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
        pretrained=False,
    ).to(device)

    print(f"\n  Loading DSSA checkpoint: {args.checkpoint}")
    print(f"  Architecture configuration: use_physics_guidance={use_physics}, use_soil_gating={use_soil_gating}")
    model.load_state_dict(ckpt, strict=False)

    # ── 1. Full Multimodal Evaluation ─────────────────────────────────────────
    print("\n  [1/4] Running Full Multimodal evaluation (Image + Surface + Subsurface)...")
    preds_full, labels, gates, spatial_align = run_inference(model, test_loader, device, cfg.use_amp)
    acc_full = (preds_full == labels).mean()

    print(f"\n  ===================================================")
    print(f"  |  DSSA Full Multimodal Accuracy : {acc_full*100:.2f}%          |")
    print(f"  ===================================================\n")

    print("  Classification Report:")
    print(classification_report(
        labels, preds_full, target_names=CLASS_NAMES,
        labels=list(range(len(CLASS_NAMES))), digits=3
    ))

    cm = confusion_matrix(labels, preds_full)
    cm_path = os.path.join(cfg.log_dir, "confusion_matrix_dssa.png")
    plot_confusion_matrix(cm, cm_path)

    # ── 2. Domain-Specific Ablations ──────────────────────────────────────────
    print("\n  [2/4] Running Subsurface-Only Chemistry Ablation (Visible stream zeroed)...")
    preds_sub_only, _, _, _ = run_inference(model, test_loader, device, cfg.use_amp, eval_domain="subsurface_only")
    acc_sub_only = (preds_sub_only == labels).mean()
    print(f"  Subsurface-only Chemistry Accuracy: {acc_sub_only*100:.2f}%")

    print("\n  [3/4] Running Surface-Only Chemistry Ablation (Subsurface stream zeroed)...")
    preds_vis_only, _, _, _ = run_inference(model, test_loader, device, cfg.use_amp, eval_domain="surface_only")
    acc_vis_only = (preds_vis_only == labels).mean()
    print(f"  Surface-only Chemistry Accuracy   : {acc_vis_only*100:.2f}%")

    print("\n  [4/4] Running Vision-Only Ablation (Chemistry streams zeroed)...")
    preds_vo, _, _, _ = run_inference(model, test_loader, device, cfg.use_amp, eval_domain="vision_only")
    acc_vo = (preds_vo == labels).mean()
    print(f"  Vision-only Accuracy              : {acc_vo*100:.2f}%")
    print(f"  Total Tabular Contribution        : +{(acc_full - acc_vo)*100:.2f}%")

    # ── 3. Gate Breakdown Analysis ────────────────────────────────────────────
    gate_means_dict = {"surface": {}, "subsurface": {}, "fusion": {}}
    for i, name in enumerate(CLASS_NAMES):
        mask = labels == i
        if mask.sum() > 0 and gates:
            gate_means_dict["surface"][name]    = float(gates["z_surface"][mask].mean())
            gate_means_dict["subsurface"][name] = float(gates["z_subsurface"][mask].mean())
            gate_means_dict["fusion"][name]     = float(gates["z_fusion"][mask].mean())

    if gates:
        gate_plot_path = os.path.join(cfg.log_dir, "gate_breakdown_dssa.png")
        plot_gate_breakdown(gate_means_dict, gate_plot_path)

        print("\n  Mean DSSA Gate Values per Class:")
        print(f"  {'Class':<22} {'z_surface':>10} {'z_subsurface':>14} {'z_fusion':>10}")
        print(f"  {'-'*60}")
        for name in CLASS_NAMES:
            z_s = gate_means_dict["surface"].get(name, 0.0)
            z_sub = gate_means_dict["subsurface"].get(name, 0.0)
            z_f = gate_means_dict["fusion"].get(name, 0.0)
            print(f"  {name:<22} {z_s:>10.4f} {z_sub:>14.4f} {z_f:>10.4f}")

    # ── 4. Biophysical Spatial Attention Alignment ────────────────────────────
    if spatial_align:
        print("\n  -- Quantitative Biophysical Spatial Alignment (Cosine Similarity) --")
        print(f"  Soil Attention Map vs. Soil Color Index (CI):        {spatial_align.get('soil', 0.0):.4f}")
        print(f"  Canopy Attention Map vs. Vegetation Index (ExG):     {spatial_align.get('canopy', 0.0):.4f}")
        print(f"  Background Mask vs. Sky/Horizon Optical Zone:        {spatial_align.get('bg', 0.0):.4f}")
        print("  " + "-" * 78)

    # ── 5. Relevance Matrix Summary (PGMR) ───────────────────────────────────
    if use_physics and hasattr(model, "export_relevance_summary"):
        pgmr_summary = model.export_relevance_summary()
        if pgmr_summary:
            print("\n  -- Learned Physics-Guided Modality Relevance Matrix (PGMR) --")
            print(f"  {'Feature':<12} {'Soil (Learned)':<16} {'Canopy (Learned)':<18} {'Prior [S / C]':<18} {'Delta (Soil)':<14}")
            print("  " + "-" * 78)
            for row in pgmr_summary:
                s_l = f"{row['learned_soil']:.3f}"
                c_l = f"{row['learned_canopy']:.3f}"
                p_sc = f"[{row['prior_soil']:.1f} / {row['prior_canopy']:.1f}]"
                d_s = f"{row['delta_soil']:+.3f}"
                print(f"  {row['feature']:<12} {s_l:<16} {c_l:<18} {p_sc:<18} {d_s:<14}")
            print("  " + "-" * 78)
    else:
        pgmr_summary = []

    # ── 6. Save Summary ───────────────────────────────────────────────────────
    summary = {
        "full_multimodal_accuracy": float(acc_full),
        "subsurface_only_chemistry_accuracy": float(acc_sub_only),
        "surface_only_chemistry_accuracy": float(acc_vis_only),
        "vision_only_accuracy": float(acc_vo),
        "tabular_contribution_pct": float((acc_full - acc_vo) * 100),
        "subsurface_isolated_gain": float((acc_sub_only - acc_vo) * 100),
        "surface_isolated_gain": float((acc_vis_only - acc_vo) * 100),
        "gate_means": gate_means_dict,
        "pgmr_relevance": pgmr_summary,
    }
    summary_path = os.path.join(cfg.log_dir, "evaluation_summary_dssa.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Evaluation summary saved -> {summary_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DSSA model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(cfg.checkpoint_dir, "dssa_best_spatial.pth"),
        help="Path to DSSA checkpoint",
    )
    parser.add_argument("--spatial", action="store_true", help="Enable spatial attention mode")
    parser.add_argument("--no_physics", action="store_true", help="Force legacy split mode")
    parser.add_argument("--no_soil_gating", action="store_true", help="Force disabled soil gating mode")
    parser.add_argument("--seed", type=int, default=cfg.random_seed, help=f"Random seed used during training (default: {cfg.random_seed})")
    args = parser.parse_args()
    main(args)
