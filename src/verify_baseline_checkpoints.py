"""
verify_baseline_checkpoints.py
-------------------------------
Independently re-evaluates all saved transformer baseline checkpoints
on the frozen LUCAS test set. No training — pure inference from disk.
Prints a full comparison table against DSSA-Standard.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from config import cfg
from dataset import build_dataloaders
from models.transformer_baselines import (
    SwinVisionModel,
    CapacityMatchedCrossTransformer,
)

def evaluate_checkpoint(ckpt_path, model, model_type, test_loader, device):
    if not os.path.exists(ckpt_path):
        return None
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, tabular, labels in test_loader:
            images  = images.to(device)
            tabular = tabular.to(device)
            labels  = labels.to(device)

            if model_type in ("vit", "swin", "efficientvit", "mobilenetv4"):
                logits = model(images)
            elif model_type in ("tabm", "ft_transformer"):
                logits = model(tabular)
            else:
                logits = model(images, tabular)

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    acc      = accuracy_score(all_targets, all_preds) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro",    zero_division=0)
    wt_f1    = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    kappa    = cohen_kappa_score(all_targets, all_preds)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    return dict(accuracy=acc, macro_f1=macro_f1, weighted_f1=wt_f1, kappa=kappa,
                params_m=n_params, n_samples=len(all_targets))


def main():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading frozen LUCAS test split (seed=42, stratified)...")
    _, _, test_loader, _ = build_dataloaders(
        csv_path=cfg.tabular_csv,
        feature_cols=cfg.continuous_features,
        val_split=cfg.val_split,
        test_split=cfg.test_split,
        batch_size=64,
        img_size=cfg.img_size,
        num_workers=0,
        tabular_dropout=0.0,
        seed=SEED,
        label_col=cfg.label_col,
    )

    ckpt_dir = os.path.join(cfg.checkpoint_dir, "transformers")

    models_to_verify = [
        ("Vision Transformer (Swin-Tiny)",     "swin",          SwinVisionModel(pretrained=False),               "swin_best.pth"),
        ("Capacity-Matched Cross-Transformer", "capacity_cross", CapacityMatchedCrossTransformer(pretrained=False), "capacity_cross_best.pth"),
    ]

    print("\n" + "="*72)
    print("   INDEPENDENT CHECKPOINT VERIFICATION - FROZEN TEST SET")
    print("="*72)

    rows = []
    for model_name, model_type, model, ckpt_name in models_to_verify:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        print(f"\n[Evaluating] {model_name}")
        print(f"  Checkpoint : {ckpt_path}")
        if not os.path.exists(ckpt_path):
            print("  STATUS: CHECKPOINT NOT FOUND - SKIPPING")
            continue

        res = evaluate_checkpoint(ckpt_path, model, model_type, test_loader, device)
        rows.append((model_name, res))
        print(f"  Params     : {res['params_m']:.2f}M")
        print(f"  N samples  : {res['n_samples']}")
        print(f"  Accuracy   : {res['accuracy']:.2f}%")
        print(f"  Macro F1   : {res['macro_f1']:.4f}")
        print(f"  Weighted F1: {res['weighted_f1']:.4f}")
        print(f"  Kappa      : {res['kappa']:.4f}")

    dssa_ref = dict(accuracy=86.20, macro_f1=0.8287, kappa=0.8286, params_m=8.42)

    print("\n\n" + "="*72)
    print("   FINAL COMPARISON TABLE (Frozen Test Set, N=2,917)")
    print("="*72)
    header = f"{'Model':<42} {'Params':>7} {'Accuracy':>10} {'Macro F1':>10} {'Kappa':>8}"
    print(header)
    print("-"*72)
    for model_name, res in rows:
        if res:
            print(f"{model_name:<42} {res['params_m']:>6.2f}M {res['accuracy']:>9.2f}% {res['macro_f1']:>10.4f} {res['kappa']:>8.4f}")
    print(f"{'DSSA-Standard (Proposed) [multi-seed ref]':<42} {dssa_ref['params_m']:>6.2f}M {dssa_ref['accuracy']:>9.2f}% {dssa_ref['macro_f1']:>10.4f} {dssa_ref['kappa']:>8.4f}")
    print("="*72)

    import json
    results_json = "results/ablation_studies/transformer_benchmark_results.json"
    if os.path.exists(results_json):
        with open(results_json) as f:
            saved = {m["name"]: m for m in json.load(f)}

        print("\n[CROSS-CHECK] Fresh evaluation vs. stored JSON values:")
        print(f"{'Model':<42} {'JSON Acc':>9} {'Fresh Acc':>10} {'Match?':>8}")
        print("-"*72)
        for model_name, res in rows:
            if res and model_name in saved:
                stored_acc = saved[model_name]["accuracy"]
                fresh_acc  = round(res["accuracy"], 2)
                match = "OK" if abs(stored_acc - fresh_acc) < 0.05 else "MISMATCH"
                print(f"{model_name:<42} {stored_acc:>8.2f}% {fresh_acc:>9.2f}% {match:>8}")


if __name__ == "__main__":
    main()
