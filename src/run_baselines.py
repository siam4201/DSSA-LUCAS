"""
run_baselines.py
----------------
Trains and evaluates standard baselines, then compiles a comprehensive
benchmark comparison table (Table 1) against all proposed architectures.

Models Evaluated:
  1. Tabular-Only (GRN)
  2. Vision-Only (EfficientNet-B0)
  3. Early Concatenation Fusion ([f_vis, f_tab] -> MLP)
  4. Late Decision Fusion (0.5 * Softmax(vis) + 0.5 * Softmax(tab))
  5. Flat GCMA (Pooled)
  6. Flat GCMA (Spatial)
  7. Hierarchical GCMA (Spatial)
  8. DSSA (Dual Surface vs Subsurface Decomposition Adapter - Proposed)

Output:
  - Prints formatted Table 1
  - Saves logs/benchmark_comparison_table.json
  - Saves logs/benchmark_table.md
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, accuracy_score, f1_score

from config import cfg
from dataset import build_dataloaders
from models import (
    TabularOnlyModel,
    VisionOnlyModel,
    ConcatFusionModel,
    GCMAModel,
    HierarchicalGCMAModel,
    DSSAModel,
)
from utils import get_device, set_seed, setup_dirs, EarlyStopping, MetricTracker

CLASS_NAMES = cfg.soil_classes


def train_baseline_model(model_name: str, model: nn.Module, train_loader, val_loader, device, epochs: int = 35):
    """Generic trainer for baseline models."""
    print(f"\n───────────────────────────────────────────────────────")
    print(f"  Training Baseline: {model_name} ({epochs} epochs max)")
    print(f"───────────────────────────────────────────────────────")
    
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"baseline_{model_name.lower().replace(' ', '_')}.pth")
    early_stop = EarlyStopping(patience=10, checkpoint_path=ckpt_path)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    amp_scaler = GradScaler(enabled=cfg.use_amp)
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_tracker = MetricTracker()
        
        for images, tabular, labels in train_loader:
            images = images.to(device, non_blocking=True)
            tabular = tabular.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=cfg.use_amp):
                logits = model(images, tabular)
                loss = criterion(logits, labels)
                
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            
            train_tracker.update(loss.item(), logits.detach(), labels)
            
        # Validation
        model.eval()
        val_tracker = MetricTracker()
        with torch.no_grad():
            for images, tabular, labels in val_loader:
                images = images.to(device, non_blocking=True)
                tabular = tabular.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast("cuda", enabled=cfg.use_amp):
                    logits = model(images, tabular)
                    loss = criterion(logits, labels)
                val_tracker.update(loss.item(), logits, labels)
                
        scheduler.step()
        elapsed = time.time() - t0
        
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"  Epoch {epoch:>2}/{epochs} | "
                f"Train Loss: {train_tracker.avg_loss:.4f} Acc: {train_tracker.accuracy*100:.2f}% | "
                f"Val Loss: {val_tracker.avg_loss:.4f} Acc: {val_tracker.accuracy*100:.2f}% | "
                f"Time: {elapsed:.1f}s"
            )
            
        if early_stop(val_tracker.avg_loss, model):
            break
            
    # Load best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, ckpt_path


@torch.no_grad()
def get_model_predictions_and_probs(model, test_loader, device):
    """Returns predictions, probabilities, and true labels."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    
    for images, tabular, labels in test_loader:
        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast("cuda", enabled=cfg.use_amp):
            logits = model(images, tabular)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
        
    return torch.cat(all_preds).numpy(), torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def main():
    set_seed(cfg.random_seed)
    device = get_device(cfg.device)
    setup_dirs(cfg.checkpoint_dir, cfg.log_dir)
    
    print("\n=======================================================")
    print("  Multimodal Benchmark & Comparison Suite")
    print("=======================================================")
    
    # ── Load Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, _ = build_dataloaders(
        csv_path=cfg.tabular_csv,
        feature_cols=cfg.continuous_features,
        val_split=cfg.val_split,
        test_split=cfg.test_split,
        batch_size=cfg.batch_size,
        img_size=cfg.img_size,
        num_workers=cfg.num_workers,
        tabular_dropout=getattr(cfg, "p_drop_tabular", 0.10),
        seed=cfg.random_seed,
        label_col=cfg.label_col,
    )
    
    results = {}
    
    # ── 1. Train & Eval Tabular-Only ──────────────────────────────────────────
    tab_model = TabularOnlyModel(
        num_continuous=cfg.num_continuous_features,
        num_classes=cfg.num_classes,
        shared_dim=cfg.shared_dim,
    ).to(device)
    tab_model, _ = train_baseline_model("tabular_only", tab_model, train_loader, val_loader, device, epochs=35)
    preds_tab, probs_tab, labels_test = get_model_predictions_and_probs(tab_model, test_loader, device)
    
    tab_acc = accuracy_score(labels_test, preds_tab) * 100
    results["Tabular-Only (GRN)"] = {
        "acc": tab_acc,
        "macro_f1": f1_score(labels_test, preds_tab, average="macro"),
        "weighted_f1": f1_score(labels_test, preds_tab, average="weighted"),
        "modality": "Tabular Only",
        "gain": 0.0,
    }
    
    # ── 2. Train & Eval Vision-Only ───────────────────────────────────────────
    vis_model = VisionOnlyModel(
        num_classes=cfg.num_classes,
        shared_dim=cfg.shared_dim,
        cnn_backbone=cfg.cnn_backbone,
        pretrained=True,
    ).to(device)
    vis_model, _ = train_baseline_model("vision_only", vis_model, train_loader, val_loader, device, epochs=30)
    preds_vis, probs_vis, _ = get_model_predictions_and_probs(vis_model, test_loader, device)
    
    vis_acc = accuracy_score(labels_test, preds_vis) * 100
    results["Vision-Only (EfficientNet-B0)"] = {
        "acc": vis_acc,
        "macro_f1": f1_score(labels_test, preds_vis, average="macro"),
        "weighted_f1": f1_score(labels_test, preds_vis, average="weighted"),
        "modality": "Vision Only",
        "gain": 0.0,
    }
    
    # ── 3. Train & Eval Early Concatenation Fusion ────────────────────────────
    concat_model = ConcatFusionModel(
        num_continuous=cfg.num_continuous_features,
        num_classes=cfg.num_classes,
        shared_dim=cfg.shared_dim,
        cnn_backbone=cfg.cnn_backbone,
        pretrained=True,
    ).to(device)
    concat_model, _ = train_baseline_model("concat_fusion", concat_model, train_loader, val_loader, device, epochs=30)
    preds_concat, probs_concat, _ = get_model_predictions_and_probs(concat_model, test_loader, device)
    
    concat_acc = accuracy_score(labels_test, preds_concat) * 100
    results["Early Concat Fusion"] = {
        "acc": concat_acc,
        "macro_f1": f1_score(labels_test, preds_concat, average="macro"),
        "weighted_f1": f1_score(labels_test, preds_concat, average="weighted"),
        "modality": "Vision + Tabular",
        "gain": concat_acc - vis_acc,
    }
    
    # ── 4. Late Decision Fusion (Ensemble of Vision + Tabular) ────────────────
    probs_late = 0.5 * probs_vis + 0.5 * probs_tab
    preds_late = probs_late.argmax(axis=-1)
    late_acc = accuracy_score(labels_test, preds_late) * 100
    results["Late Decision Fusion"] = {
        "acc": late_acc,
        "macro_f1": f1_score(labels_test, preds_late, average="macro"),
        "weighted_f1": f1_score(labels_test, preds_late, average="weighted"),
        "modality": "Vision + Tabular",
        "gain": late_acc - vis_acc,
    }
    
    # ── 5. Train & Eval Flat GCMA (Pooled) ────────────────────────────────────
    gcma_pooled = GCMAModel(
        num_continuous=cfg.num_continuous_features,
        num_classes=cfg.num_classes,
        shared_dim=cfg.shared_dim,
        num_heads=cfg.num_heads,
        cnn_backbone=cfg.cnn_backbone,
        use_spatial_attention=False,
        pretrained=True,
    ).to(device)
    gcma_pooled, _ = train_baseline_model("gcma_pooled", gcma_pooled, train_loader, val_loader, device, epochs=30)
    preds_gcma_p, _, _ = get_model_predictions_and_probs(gcma_pooled, test_loader, device)
    gcma_p_acc = accuracy_score(labels_test, preds_gcma_p) * 100
    results["Flat GCMA (Global Pooled)"] = {
        "acc": gcma_p_acc,
        "macro_f1": f1_score(labels_test, preds_gcma_p, average="macro"),
        "weighted_f1": f1_score(labels_test, preds_gcma_p, average="weighted"),
        "modality": "Vision + Tabular",
        "gain": gcma_p_acc - vis_acc,
    }

    # ── 6. Train & Eval Flat GCMA (Spatial Grid) ──────────────────────────────
    gcma_spatial = GCMAModel(
        num_continuous=cfg.num_continuous_features,
        num_classes=cfg.num_classes,
        shared_dim=cfg.shared_dim,
        num_heads=cfg.num_heads,
        cnn_backbone=cfg.cnn_backbone,
        use_spatial_attention=True,
        pretrained=True,
    ).to(device)
    gcma_spatial, _ = train_baseline_model("gcma_spatial", gcma_spatial, train_loader, val_loader, device, epochs=30)
    preds_gcma_s, _, _ = get_model_predictions_and_probs(gcma_spatial, test_loader, device)
    gcma_s_acc = accuracy_score(labels_test, preds_gcma_s) * 100
    results["Flat GCMA (Spatial Grid)"] = {
        "acc": gcma_s_acc,
        "macro_f1": f1_score(labels_test, preds_gcma_s, average="macro"),
        "weighted_f1": f1_score(labels_test, preds_gcma_s, average="weighted"),
        "modality": "Vision + Tabular",
        "gain": gcma_s_acc - vis_acc,
    }

    # ── 7. Evaluate DSSA (Proposed Multimodal Architecture) ───────────────────
    dssa_path = os.path.join(cfg.checkpoint_dir, "dssa_best_spatial_pgmr_w0.05_spatw0.05_soilgate_cw0.05.pth")
    if os.path.exists(dssa_path):
        ckpt_dssa = torch.load(dssa_path, map_location=device)
        has_pgmr = any("pgmr_router" in k for k in ckpt_dssa.keys())
        has_soilgate = any("spatial_adapter.tau" in k for k in ckpt_dssa.keys())

        dssa_model = DSSAModel(
            feature_cols=cfg.continuous_features,
            visible_cols=cfg.visible_chemistry_features,
            subsurface_cols=cfg.subsurface_chemistry_features,
            num_classes=cfg.num_classes,
            shared_dim=cfg.shared_dim,
            num_heads=cfg.num_heads,
            cnn_backbone=cfg.cnn_backbone,
            use_spatial_attention=True,
            use_physics_guidance=has_pgmr,
            use_soil_gating=has_soilgate,
            dropout=cfg.dropout,
            pretrained=False,
        ).to(device)
        dssa_model.load_state_dict(ckpt_dssa, strict=False)
        preds_dssa, _, _ = get_model_predictions_and_probs(dssa_model, test_loader, device)
        dssa_acc = accuracy_score(labels_test, preds_dssa) * 100
        results["DSSA + PGMR + Zero-Soil (Proposed)"] = {
            "acc": dssa_acc,
            "macro_f1": f1_score(labels_test, preds_dssa, average="macro"),
            "weighted_f1": f1_score(labels_test, preds_dssa, average="weighted"),
            "modality": "Vision + Surface + Subsurface",
            "gain": dssa_acc - vis_acc,
        }

    # ── Print & Save Comparison Table ─────────────────────────────────────────
    print("\n" + "=" * 94)
    print("  TABLE 1: Comprehensive Architectural Benchmark (Frozen 6-Class LUCAS Benchmark)")
    print("=" * 94)
    print(f"  {'Model Architecture':<40} {'Modality':<24} {'Accuracy':>10} {'Macro F1':>10} {'Gain vs Vis':>12}")
    print("  " + "─" * 90)
    
    rows_for_df = []
    for name, m in results.items():
        gain_str = f"+{m['gain']:.2f}%" if m['gain'] > 0 else f"{m['gain']:.2f}%"
        if name in ["Tabular-Only (GRN)", "Vision-Only (EfficientNet-B0)"]:
            gain_str = "—"
        print(f"  {name:<40} {m['modality']:<24} {m['acc']:>9.2f}% {m['macro_f1']:>10.4f} {gain_str:>12}")
        rows_for_df.append({
            "Model Architecture": name,
            "Modality": m["modality"],
            "Test Accuracy (%)": f"{m['acc']:.2f}%",
            "Macro F1": f"{m['macro_f1']:.4f}",
            "Weighted F1": f"{m['weighted_f1']:.4f}",
            "Gain vs. Vision-Only": gain_str,
        })
    print("=" * 94 + "\n")
    
    # Save JSON & Markdown
    df_table = pd.DataFrame(rows_for_df)
    json_path = os.path.join(cfg.log_dir, "architectural_benchmark_6class.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    md_path = r"d:\Data_Mining\reports\architectural_benchmark_6class.md"
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    df_table.to_markdown(md_path, index=False)
    print(f"  Benchmark JSON saved -> {json_path}")
    print(f"  Benchmark Markdown saved -> {md_path}")


if __name__ == "__main__":
    main()
