"""
Multi-Seed Architectural Benchmark Suite (3 Seeds: 42, 123, 999)
Evaluates all architectures on the frozen 6-class LUCAS dataset.
"""
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score

from config import cfg
from dataset import build_dataloaders
from models import (
    TabularOnlyModel,
    VisionOnlyModel,
    ConcatFusionModel,
    GCMAModel,
    DSSAModel,
)
from utils import get_device, set_seed, setup_dirs, EarlyStopping, MetricTracker

SEEDS = [42, 123, 999]

def train_baseline_model(model_name: str, model: nn.Module, train_loader, val_loader, device, epochs: int = 30, lr: float = 1e-4):
    """Generic trainer for baseline models."""
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"temp_{model_name.lower().replace(' ', '_')}.pth")
    early_stop = EarlyStopping(patience=10, checkpoint_path=ckpt_path)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
    amp_scaler = GradScaler(enabled=cfg.use_amp)
    
    for epoch in range(1, epochs + 1):
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
        if early_stop(val_tracker.avg_loss, model):
            break
            
    # Load best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model

@torch.no_grad()
def evaluate_model(model, test_loader, device):
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
    device = get_device(cfg.device)
    setup_dirs(cfg.checkpoint_dir, cfg.log_dir)
    
    print("\n" + "=" * 90)
    print("   MULTI-SEED ARCHITECTURAL BENCHMARK SUITE (Frozen 6-Class LUCAS Dataset)")
    print(f"   Seeds: {SEEDS} | Total Dataset N = 19,441 | Unseen Test Split N = 2,917")
    print("=" * 90 + "\n")
    
    models_to_test = [
        "Tabular-Only (GRN)",
        "Vision-Only (EfficientNet-B0)",
        "Early Concat Fusion",
        "Late Decision Fusion",
        "Flat GCMA (Global Pooled)",
        "Flat GCMA (Spatial Grid)",
        "DSSA + PGMR + Zero-Soil (Proposed)",
    ]
    
    # Store runs: {model_name: {"acc": [], "macro_f1": [], "weighted_f1": []}}
    benchmark_data = {m: {"acc": [], "macro_f1": [], "weighted_f1": []} for m in models_to_test}
    
    for seed_idx, seed in enumerate(SEEDS, 1):
        print(f"\n========================================================================")
        print(f"  >>> RUNNING BENCHMARK ON SEED {seed} ({seed_idx}/{len(SEEDS)})")
        print(f"========================================================================")
        set_seed(seed)
        
        train_loader, val_loader, test_loader, _ = build_dataloaders(
            csv_path=cfg.tabular_csv,
            feature_cols=cfg.continuous_features,
            val_split=cfg.val_split,
            test_split=cfg.test_split,
            batch_size=cfg.batch_size,
            img_size=cfg.img_size,
            num_workers=cfg.num_workers,
            tabular_dropout=getattr(cfg, "p_drop_tabular", 0.10),
            seed=seed,
            label_col=cfg.label_col,
        )
        
        # 1. Tabular-Only
        t0 = time.time()
        tab_m = TabularOnlyModel(
            num_continuous=cfg.num_continuous_features,
            num_classes=cfg.num_classes,
            shared_dim=cfg.shared_dim,
        ).to(device)
        tab_m = train_baseline_model(f"tab_{seed}", tab_m, train_loader, val_loader, device, epochs=35)
        preds_tab, probs_tab, labels_test = evaluate_model(tab_m, test_loader, device)
        acc_tab = accuracy_score(labels_test, preds_tab) * 100
        benchmark_data["Tabular-Only (GRN)"]["acc"].append(acc_tab)
        benchmark_data["Tabular-Only (GRN)"]["macro_f1"].append(f1_score(labels_test, preds_tab, average="macro"))
        benchmark_data["Tabular-Only (GRN)"]["weighted_f1"].append(f1_score(labels_test, preds_tab, average="weighted"))
        print(f"  [1/7] Tabular-Only        : Acc = {acc_tab:6.2f}% | Macro F1 = {benchmark_data['Tabular-Only (GRN)']['macro_f1'][-1]:.4f} ({time.time()-t0:.1f}s)")
        
        # 2. Vision-Only
        t0 = time.time()
        vis_m = VisionOnlyModel(
            num_classes=cfg.num_classes,
            shared_dim=cfg.shared_dim,
            cnn_backbone=cfg.cnn_backbone,
            pretrained=True,
        ).to(device)
        vis_m = train_baseline_model(f"vis_{seed}", vis_m, train_loader, val_loader, device, epochs=25, lr=5e-5)
        preds_vis, probs_vis, _ = evaluate_model(vis_m, test_loader, device)
        acc_vis = accuracy_score(labels_test, preds_vis) * 100
        benchmark_data["Vision-Only (EfficientNet-B0)"]["acc"].append(acc_vis)
        benchmark_data["Vision-Only (EfficientNet-B0)"]["macro_f1"].append(f1_score(labels_test, preds_vis, average="macro"))
        benchmark_data["Vision-Only (EfficientNet-B0)"]["weighted_f1"].append(f1_score(labels_test, preds_vis, average="weighted"))
        print(f"  [2/7] Vision-Only         : Acc = {acc_vis:6.2f}% | Macro F1 = {benchmark_data['Vision-Only (EfficientNet-B0)']['macro_f1'][-1]:.4f} ({time.time()-t0:.1f}s)")
        
        # 3. Early Concat Fusion
        t0 = time.time()
        concat_m = ConcatFusionModel(
            num_continuous=cfg.num_continuous_features,
            num_classes=cfg.num_classes,
            shared_dim=cfg.shared_dim,
            cnn_backbone=cfg.cnn_backbone,
            pretrained=True,
        ).to(device)
        concat_m = train_baseline_model(f"concat_{seed}", concat_m, train_loader, val_loader, device, epochs=25, lr=5e-5)
        preds_cat, _, _ = evaluate_model(concat_m, test_loader, device)
        acc_cat = accuracy_score(labels_test, preds_cat) * 100
        benchmark_data["Early Concat Fusion"]["acc"].append(acc_cat)
        benchmark_data["Early Concat Fusion"]["macro_f1"].append(f1_score(labels_test, preds_cat, average="macro"))
        benchmark_data["Early Concat Fusion"]["weighted_f1"].append(f1_score(labels_test, preds_cat, average="weighted"))
        print(f"  [3/7] Early Concat Fusion : Acc = {acc_cat:6.2f}% | Macro F1 = {benchmark_data['Early Concat Fusion']['macro_f1'][-1]:.4f} ({time.time()-t0:.1f}s)")
        
        # 4. Late Decision Fusion
        probs_late = 0.5 * probs_vis + 0.5 * probs_tab
        preds_late = probs_late.argmax(axis=-1)
        acc_late = accuracy_score(labels_test, preds_late) * 100
        benchmark_data["Late Decision Fusion"]["acc"].append(acc_late)
        benchmark_data["Late Decision Fusion"]["macro_f1"].append(f1_score(labels_test, preds_late, average="macro"))
        benchmark_data["Late Decision Fusion"]["weighted_f1"].append(f1_score(labels_test, preds_late, average="weighted"))
        print(f"  [4/7] Late Decision Fusion: Acc = {acc_late:6.2f}% | Macro F1 = {benchmark_data['Late Decision Fusion']['macro_f1'][-1]:.4f}")
        
        # 5. Flat GCMA (Pooled)
        t0 = time.time()
        gcma_p = GCMAModel(
            num_continuous=cfg.num_continuous_features,
            num_classes=cfg.num_classes,
            shared_dim=cfg.shared_dim,
            num_heads=cfg.num_heads,
            cnn_backbone=cfg.cnn_backbone,
            use_spatial_attention=False,
            pretrained=True,
        ).to(device)
        gcma_p = train_baseline_model(f"gcma_p_{seed}", gcma_p, train_loader, val_loader, device, epochs=25, lr=5e-5)
        preds_gp, _, _ = evaluate_model(gcma_p, test_loader, device)
        acc_gp = accuracy_score(labels_test, preds_gp) * 100
        benchmark_data["Flat GCMA (Global Pooled)"]["acc"].append(acc_gp)
        benchmark_data["Flat GCMA (Global Pooled)"]["macro_f1"].append(f1_score(labels_test, preds_gp, average="macro"))
        benchmark_data["Flat GCMA (Global Pooled)"]["weighted_f1"].append(f1_score(labels_test, preds_gp, average="weighted"))
        print(f"  [5/7] Flat GCMA (Pooled)  : Acc = {acc_gp:6.2f}% | Macro F1 = {benchmark_data['Flat GCMA (Global Pooled)']['macro_f1'][-1]:.4f} ({time.time()-t0:.1f}s)")
        
        # 6. Flat GCMA (Spatial Grid)
        t0 = time.time()
        gcma_s = GCMAModel(
            num_continuous=cfg.num_continuous_features,
            num_classes=cfg.num_classes,
            shared_dim=cfg.shared_dim,
            num_heads=cfg.num_heads,
            cnn_backbone=cfg.cnn_backbone,
            use_spatial_attention=True,
            pretrained=True,
        ).to(device)
        gcma_s = train_baseline_model(f"gcma_s_{seed}", gcma_s, train_loader, val_loader, device, epochs=25, lr=5e-5)
        preds_gs, _, _ = evaluate_model(gcma_s, test_loader, device)
        acc_gs = accuracy_score(labels_test, preds_gs) * 100
        benchmark_data["Flat GCMA (Spatial Grid)"]["acc"].append(acc_gs)
        benchmark_data["Flat GCMA (Spatial Grid)"]["macro_f1"].append(f1_score(labels_test, preds_gs, average="macro"))
        benchmark_data["Flat GCMA (Spatial Grid)"]["weighted_f1"].append(f1_score(labels_test, preds_gs, average="weighted"))
        print(f"  [6/7] Flat GCMA (Spatial) : Acc = {acc_gs:6.2f}% | Macro F1 = {benchmark_data['Flat GCMA (Spatial Grid)']['macro_f1'][-1]:.4f} ({time.time()-t0:.1f}s)")
        
        # 7. DSSA (Proposed)
        dssa_path = os.path.join(cfg.checkpoint_dir, "dssa_best_spatial_pgmr_w0.05_spatw0.05_soilgate_cw0.05.pth")
        if os.path.exists(dssa_path):
            ckpt_dssa = torch.load(dssa_path, map_location=device)
            has_pgmr = any("pgmr_router" in k for k in ckpt_dssa.keys())
            has_soilgate = any("spatial_adapter.tau" in k for k in ckpt_dssa.keys())
            dssa_m = DSSAModel(
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
            dssa_m.load_state_dict(ckpt_dssa, strict=False)
            preds_dssa, _, _ = evaluate_model(dssa_m, test_loader, device)
            acc_dssa = accuracy_score(labels_test, preds_dssa) * 100
            benchmark_data["DSSA + PGMR + Zero-Soil (Proposed)"]["acc"].append(acc_dssa)
            benchmark_data["DSSA + PGMR + Zero-Soil (Proposed)"]["macro_f1"].append(f1_score(labels_test, preds_dssa, average="macro"))
            benchmark_data["DSSA + PGMR + Zero-Soil (Proposed)"]["weighted_f1"].append(f1_score(labels_test, preds_dssa, average="weighted"))
            print(f"  [7/7] DSSA Proposed       : Acc = {acc_dssa:6.2f}% | Macro F1 = {benchmark_data['DSSA + PGMR + Zero-Soil (Proposed)']['macro_f1'][-1]:.4f}")

    # ── Compile Multi-Seed Statistical Summary Table ──────────────────────────
    print("\n" + "=" * 100)
    print("  TABLE 1: Comprehensive Multi-Seed Architectural Benchmark (Mean +/- Std Across 3 Seeds)")
    print("=" * 100)
    print(f"  {'Model Architecture':<36} {'Modality':<24} {'Test Accuracy (%)':>20} {'Macro F1':>16}")
    print("  " + "─" * 96)
    
    table_rows = []
    modality_map = {
        "Tabular-Only (GRN)": "Chemistry Only",
        "Vision-Only (EfficientNet-B0)": "Vision Only",
        "Early Concat Fusion": "Vision + Chemistry",
        "Late Decision Fusion": "Vision + Chemistry",
        "Flat GCMA (Global Pooled)": "Vision + Chemistry",
        "Flat GCMA (Spatial Grid)": "Vision + Chemistry",
        "DSSA + PGMR + Zero-Soil (Proposed)": "Vision + Surface + Subsurface",
    }
    
    vis_mean_acc = np.mean(benchmark_data["Vision-Only (EfficientNet-B0)"]["acc"])
    
    for name, data in benchmark_data.items():
        m_acc, s_acc = np.mean(data["acc"]), np.std(data["acc"])
        m_f1, s_f1 = np.mean(data["macro_f1"]), np.std(data["macro_f1"])
        m_wf1, s_wf1 = np.mean(data["weighted_f1"]), np.std(data["weighted_f1"])
        
        acc_str = f"{m_acc:.2f}% ± {s_acc:.2f}%"
        f1_str = f"{m_f1:.4f} ± {s_f1:.4f}"
        wf1_str = f"{m_wf1:.4f} ± {s_wf1:.4f}"
        gain_str = f"+{m_acc - vis_mean_acc:.2f}%" if name not in ["Tabular-Only (GRN)", "Vision-Only (EfficientNet-B0)"] else "—"
        
        print(f"  {name:<36} {modality_map[name]:<24} {acc_str:>20} {f1_str:>16}")
        
        table_rows.append({
            "Model Architecture": name,
            "Modality Stream": modality_map[name],
            "Test Accuracy (Mean ± Std)": acc_str,
            "Macro F1 (Mean ± Std)": f1_str,
            "Weighted F1 (Mean ± Std)": wf1_str,
            "Gain vs. Vision-Only": gain_str,
        })
    print("=" * 100 + "\n")
    
    # Save JSON and Markdown
    out_df = pd.DataFrame(table_rows)
    md_path = r"d:\Data_Mining\reports\multiseed_architectural_benchmark_6class.md"
    json_path = r"d:\Data_Mining\logs\multiseed_architectural_benchmark_6class.json"
    
    out_df.to_markdown(md_path, index=False)
    with open(json_path, "w") as f:
        json.dump({
            "benchmark_seeds": SEEDS,
            "results_by_model": benchmark_data,
            "summary_table": table_rows
        }, f, indent=2)
        
    print(f"  [OK] Saved Multi-Seed Benchmark Markdown -> {md_path}")
    print(f"  [OK] Saved Multi-Seed Benchmark JSON     -> {json_path}\n")

if __name__ == "__main__":
    main()
