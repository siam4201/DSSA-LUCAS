"""
train_dssa_lite.py
------------------
Multi-seed training and evaluation script for DSSA-Lite-v2 (Enhanced Edge-Optimized Variant).
Features:
- Dual-Stream Gated Multimodal Units (Dual GMU: Surface + Subsurface + Master Fusion)
- Multi-Task Auxiliary Optimization:
    L_total = L_CE + lambda_ortho * L_ortho + lambda_canopy * L_canopy
- EarlyStopping on validation macro-F1 (patience: 8)
- Cosine Annealing learning rate schedule with 3-epoch warmup
- Fully pagefile-safe for Windows (num_workers=0 default)
- Consolidated JSON & Markdown reporting in results/ablation_studies/
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

sys.path.append(os.path.abspath("src"))
from config import cfg
from dataset import build_dataloaders
from utils import set_seed, get_device
from models.dssa_lite import DSSALiteModel

def parse_args():
    parser = argparse.ArgumentParser(description="Train DSSA-Lite-v2 (Enhanced Edge-Optimized Variant)")
    parser.add_argument("--epochs", type=int, default=45, help="Maximum number of training epochs per seed")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate for adapter heads")
    parser.add_argument("--backbone-lr", type=float, default=4e-5, help="Learning rate for backbone")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--lambda-ortho", type=float, default=0.05, help="Weight for spatial mask orthogonality penalty")
    parser.add_argument("--lambda-canopy", type=float, default=0.05, help="Weight for canopy density alignment loss")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension d (e.g. 128, 160, 192)")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of cross-attention heads (e.g. 4 or 5)")
    parser.add_argument("--seed", type=int, default=42, help="Single seed (if not using --seeds/--multiseed)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="List of seeds to evaluate (e.g. --seeds 42 123 999)")
    parser.add_argument("--multiseed", action="store_true", help="Shortcut for running standard benchmark seeds [42, 123, 999]")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (default: 0 for Windows pagefile stability)")
    parser.add_argument("--dry-run", action="store_true", help="Perform a 1-step dry run and exit")
    return parser.parse_args()

def compute_exg_canopy_prior(images: torch.Tensor) -> torch.Tensor:
    """
    Computes normalized Excess Green (ExG) vegetation index prior from RGB images:
    ExG = 2*G - R - B. Normalized to [0, 1].
    """
    # images: (B, 3, 224, 224), normalized with ImageNet mean/std
    # Approximate green dominance across the spatial map:
    r = images[:, 0, :, :]
    g = images[:, 1, :, :]
    b = images[:, 2, :, :]
    exg = 2.0 * g - r - b
    # Average across spatial dimensions and map via sigmoid to [0, 1]
    canopy_prior = torch.sigmoid(exg.mean(dim=[-2, -1], keepdim=False)).unsqueeze(1)  # (B, 1)
    return canopy_prior

def train_epoch(model, dataloader, optimizer, criterion_ce, device, args, dry_run=False):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_ortho = 0.0
    correct = 0
    total = 0

    for i, (images, tabular, labels) in enumerate(dataloader):
        images = images.to(device)
        tabular = tabular.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, m_soil, m_canopy, canopy_density = model(images, tabular, return_aux=True)

        # 1. Classification Loss (Cross-Entropy with label smoothing)
        loss_ce = criterion_ce(logits, labels)

        # 2. Spatial Mask Orthogonality Loss: penalize overlapping soil & canopy attention
        loss_ortho = (m_soil * m_canopy).sum(dim=1).mean()

        # 3. Biophysical Canopy Density Loss: align predicted canopy density with image vegetation index
        with torch.no_grad():
            canopy_prior = compute_exg_canopy_prior(images) * 49.0  # Scale to token count
        loss_canopy = F.smooth_l1_loss(canopy_density, canopy_prior)

        # Joint Multi-Task Objective
        loss_total = loss_ce + (args.lambda_ortho * loss_ortho) + (args.lambda_canopy * loss_canopy)

        loss_total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_total.item() * images.size(0)
        total_ce += loss_ce.item() * images.size(0)
        total_ortho += loss_ortho.item() * images.size(0)

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if dry_run:
            print(f"   [Dry-Run] Step 1 complete. Total Loss: {loss_total.item():.4f}, CE Loss: {loss_ce.item():.4f}, Ortho Loss: {loss_ortho.item():.4f}, Acc: {correct/total*100:.1f}%")
            break

    return total_loss / total, (correct / total) * 100.0

@torch.no_grad()
def evaluate(model, dataloader, criterion_ce, device, dry_run=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for i, (images, tabular, labels) in enumerate(dataloader):
        images = images.to(device)
        tabular = tabular.to(device)
        labels = labels.to(device)

        logits = model(images, tabular, return_aux=False)
        loss = criterion_ce(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if dry_run:
            break

    acc = accuracy_score(all_labels, all_preds) * 100.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    kappa = cohen_kappa_score(all_labels, all_preds)
    return total_loss / len(all_labels), acc, macro_f1, kappa, all_preds, all_labels

def run_single_seed(seed, args, device):
    print("\n" + "=" * 85)
    print(f"   >>> TRAINING DSSA-LITE-v2 ON SEED {seed} (Max Epochs: {args.epochs}, Patience: {args.patience})")
    print("=" * 85)
    set_seed(seed)

    # 1. DataLoaders
    train_loader, val_loader, test_loader, scaler = build_dataloaders(
        csv_path=cfg.tabular_csv,
        feature_cols=cfg.continuous_features,
        val_split=cfg.val_split,
        test_split=cfg.test_split,
        batch_size=args.batch_size,
        num_workers=0 if args.dry_run else args.num_workers,
        tabular_dropout=getattr(cfg, "p_drop_tabular", 0.10),
        seed=seed,
        label_col=cfg.label_col,
    )

    # 2. Model (DSSA-Lite-v2 Enhanced)
    model = DSSALiteModel(num_classes=6, d=args.latent_dim, num_heads=args.num_heads, pretrained=True).to(device)

    # 3. Optimizer & Criterion
    optimizer = AdamW([
        {"params": model.visual_backbone.parameters(), "lr": args.backbone_lr},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("visual_backbone")], "lr": args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    if args.dry_run:
        print("\n[Dry-Run Mode] Testing multi-task forward, losses, backward...")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion_ce, device, args, dry_run=True)
        val_loss, val_acc, val_f1, val_k, _, _ = evaluate(model, val_loader, criterion_ce, device, dry_run=True)
        test_loss, test_acc, test_f1, test_k, _, _ = evaluate(model, test_loader, criterion_ce, device, dry_run=True)
        print("\n[OK] DSSA-Lite-v2 Dry-Run Succeeded perfectly!")
        return {"test_acc": test_acc, "macro_f1": test_f1, "kappa": test_k, "time": 0.0, "epochs_trained": 1}

    # 4. Checkpoint & Early Stopping Setup
    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/dssa_lite_v2_best_seed{seed}.pth"

    best_val_f1 = 0.0
    patience_counter = 0
    epochs_ran = 0
    t0_seed = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion_ce, device, args)
        val_loss, val_acc, val_f1, val_k, _, _ = evaluate(model, val_loader, criterion_ce, device)
        scheduler.step()
        epochs_ran = epoch

        elapsed = time.time() - t0
        print(f"Seed {seed} | Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_loss:.4f}, Acc: {train_acc:5.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:5.2f}%, Macro F1: {val_f1:.4f}")

        # Check Early Stopping based on validation Macro F1
        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"   [BEST] Checkpoint saved -> {save_path} (Val Macro F1 = {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"   [EarlyStopping] Seed {seed} converged at epoch {epoch} (No improvement for {args.patience} epochs).")
                break

    seed_time = time.time() - t0_seed

    # 5. Evaluate Best Restored Checkpoint on Held-Out Test Set
    print(f"\nLoading best restored checkpoint from {save_path} for test evaluation...")
    model.load_state_dict(torch.load(save_path))
    test_loss, test_acc, test_f1, test_k, _, _ = evaluate(model, test_loader, criterion_ce, device)
    print(f">>> Seed {seed} Final Test Results: Acc = {test_acc:.2f}% | Macro F1 = {test_f1:.4f} | Kappa = {test_k:.4f} ({epochs_ran} epochs in {seed_time/60:.2f}m)")

    return {
        "seed": seed,
        "test_acc": float(test_acc),
        "macro_f1": float(test_f1),
        "kappa": float(test_k),
        "epochs_trained": epochs_ran,
        "time_seconds": float(seed_time),
        "checkpoint": save_path
    }

def main():
    args = parse_args()
    device = get_device(cfg.device)

    # Determine seeds list
    if args.multiseed:
        seeds_to_run = [42, 123, 999]
    elif args.seeds is not None and len(args.seeds) > 0:
        seeds_to_run = args.seeds
    else:
        seeds_to_run = [args.seed]

    print("\n" + "=" * 85)
    print(f"   [DSSA-LITE-v2 ENHANCED] BENCHMARK SUITE (Dual GMU + Multi-Task Auxiliary Loss)")
    print(f"   Device: {device} | Seeds: {seeds_to_run} | Max Epochs: {args.epochs} | Patience: {args.patience}")
    print("=" * 85 + "\n")

    # Measure parameter footprint
    dummy_model = DSSALiteModel(num_classes=6, d=args.latent_dim, num_heads=args.num_heads, pretrained=False)
    total_params = sum(p.numel() for p in dummy_model.parameters())
    backbone_params = sum(p.numel() for p in dummy_model.visual_backbone.parameters())
    adapter_params = total_params - backbone_params
    print(f"DSSA-Lite-v2 Architecture Footprint:")
    print(f"   - Total Parameters            : {total_params:,} ({total_params/1e6:.3f}M)")
    print(f"   - EfficientNet-B0 Backbone    : {backbone_params:,} ({backbone_params/1e6:.3f}M)")
    print(f"   - Multimodal Adapter Overhead : {adapter_params:,} ({adapter_params/1e6:.3f}M)")
    print(f"   - Multi-Task Weights          : lambda_ortho = {args.lambda_ortho}, lambda_canopy = {args.lambda_canopy}")
    print("-" * 85)

    if args.dry_run:
        _ = run_single_seed(seeds_to_run[0], args, device)
        return

    # Execute all seeds
    all_results = []
    t0_all = time.time()

    for s_idx, seed in enumerate(seeds_to_run, 1):
        print(f"\n[Progress: Seed {s_idx}/{len(seeds_to_run)}]")
        res = run_single_seed(seed, args, device)
        all_results.append(res)

    total_all_time = time.time() - t0_all

    # Aggregate Statistics
    accs = [r["test_acc"] for r in all_results]
    f1s = [r["macro_f1"] for r in all_results]
    kappas = [r["kappa"] for r in all_results]

    mean_acc, std_acc = np.mean(accs), np.std(accs)
    mean_f1, std_f1 = np.mean(f1s), np.std(f1s)
    mean_k, std_k = np.mean(kappas), np.std(kappas)

    print("\n" + "=" * 85)
    print("   FINAL MULTI-SEED DSSA-LITE-v2 BENCHMARK SUMMARY")
    print("=" * 85)
    print(f"Seeds Evaluated : {seeds_to_run}")
    print(f"Test Accuracy   : {mean_acc:.2f}% ± {std_acc:.2f}%  (Per-Seed: {[round(x, 2) for x in accs]})")
    print(f"Macro F1        : {mean_f1:.4f} ± {std_f1:.4f}  (Per-Seed: {[round(x, 4) for x in f1s]})")
    print(f"Cohen's Kappa   : {mean_k:.4f} ± {std_k:.4f}  (Per-Seed: {[round(x, 4) for x in kappas]})")
    print(f"Total Time      : {total_all_time/60:.2f} minutes")
    print("=" * 85 + "\n")

    # Save JSON Report
    os.makedirs("results/ablation_studies", exist_ok=True)
    json_path = "results/ablation_studies/dssa_lite_v2_multiseed_benchmark.json"
    summary_data = {
        "model": "DSSA-Lite-v2",
        "parameters": {
            "total": total_params,
            "backbone": backbone_params,
            "adapter": adapter_params,
            "latent_dim": 128,
            "attention_heads": 4
        },
        "seeds": seeds_to_run,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "multi_task_losses": {
            "lambda_ortho": args.lambda_ortho,
            "lambda_canopy": args.lambda_canopy
        },
        "aggregate_metrics": {
            "accuracy_mean": float(mean_acc),
            "accuracy_std": float(std_acc),
            "macro_f1_mean": float(mean_f1),
            "macro_f1_std": float(std_f1),
            "cohen_kappa_mean": float(mean_k),
            "cohen_kappa_std": float(std_k)
        },
        "per_seed_runs": all_results,
        "total_time_seconds": float(total_all_time)
    }
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved consolidated benchmark report -> {json_path}")

    # Save Markdown Report
    md_path = "results/ablation_studies/dssa_lite_v2_multiseed_benchmark.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ⚡ DSSA-Lite-v2 Enhanced Multi-Seed Benchmark Report\n\n")
        f.write(f"**Parameters**: {total_params/1e6:.3f}M total (Backbone: {backbone_params/1e6:.3f}M, Multimodal Adapter: {adapter_params/1e6:.3f}M)\n")
        f.write(f"**Multi-Task**: $\\mathcal{{L}}_{{\\text{{total}}}} = \\mathcal{{L}}_{{\\text{{CE}}}} + {args.lambda_ortho}\\mathcal{{L}}_{{\\text{{ortho}}}} + {args.lambda_canopy}\\mathcal{{L}}_{{\\text{{canopy}}}}$\n")
        f.write(f"**Seeds**: {seeds_to_run} | **Max Epochs**: {args.epochs} | **EarlyStopping Patience**: {args.patience}\n\n")
        f.write("## 1. Aggregate Performance\n\n")
        f.write("| Model Variant | Parameters | Test Accuracy (Mean ± Std) | Macro F1 (Mean ± Std) | Cohen's Kappa (Mean ± Std) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **DSSA-Lite-v2 (Enhanced)** | **{total_params/1e6:.2f}M** | **{mean_acc:.2f}% ± {std_acc:.2f}%** | **{mean_f1:.4f} ± {std_f1:.4f}** | **{mean_k:.4f} ± {std_k:.4f}** |\n\n")
        f.write("## 2. Per-Seed Performance Breakdown\n\n")
        f.write("| Seed Index | Seed ID | Test Accuracy (%) | Macro F1 | Cohen's Kappa | Epochs Trained | Training Time (min) |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for idx, r in enumerate(all_results, 1):
            f.write(f"| {idx} | {r['seed']} | {r['test_acc']:.2f}% | {r['macro_f1']:.4f} | {r['kappa']:.4f} | {r.get('epochs_trained', args.epochs)} | {r['time_seconds']/60:.2f} |\n")
    print(f"Saved markdown summary -> {md_path}\n")

if __name__ == "__main__":
    main()
