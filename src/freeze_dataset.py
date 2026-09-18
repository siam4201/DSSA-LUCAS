"""
Freeze the 6-class LUCAS dataset specification and create immutable manifest.
"""
import os
import hashlib
import json
import shutil
import pandas as pd

def main():
    src_csv = "data/soil_tabular.csv"
    frozen_csv = "data/soil_tabular_6class_frozen.csv"
    manifest_path = "data/dataset_manifest_6class.json"

    print("=" * 75)
    print("      FREEZING 6-CLASS DATASET SPECIFICATION & CREATING MANIFEST")
    print("=" * 75)

    # 1. Create dedicated frozen copy
    shutil.copyfile(src_csv, frozen_csv)
    print(f"\n[1/4] Created immutable frozen copy -> {frozen_csv}")

    # 2. Compute SHA-256 and MD5 checksums
    hasher_sha256 = hashlib.sha256()
    hasher_md5 = hashlib.md5()
    with open(frozen_csv, "rb") as f:
        buf = f.read()
        hasher_sha256.update(buf)
        hasher_md5.update(buf)

    sha256_hash = hasher_sha256.hexdigest()
    md5_hash = hasher_md5.hexdigest()

    df = pd.read_csv(frozen_csv)
    total_samples = len(df)
    class_counts = df["label_lc2"].value_counts().sort_index()

    class_names = [
        "Cereals",
        "Other Cropland",
        "Broadleaf Woodland",
        "Coniferous Woodland",
        "Shrubland",
        "Managed Grassland"
    ]

    breakdown = {}
    for idx, (name, count) in enumerate(zip(class_names, class_counts)):
        breakdown[name] = {
            "label_id": idx,
            "count": int(count),
            "percentage": float(count / total_samples * 100.0)
        }

    # 3. Create Dataset Manifest
    manifest = {
        "dataset_name": "LUCAS_6Class_Level2_Frozen_Benchmark",
        "frozen_timestamp": "2026-08-20T18:24:00Z",
        "source_file": frozen_csv,
        "sha256_checksum": sha256_hash,
        "md5_checksum": md5_hash,
        "total_samples": total_samples,
        "num_classes": len(class_names),
        "classes": class_names,
        "class_breakdown": breakdown,
        "features": {
            "continuous_chemistry_features": [
                "pH_H2O", "pH_CaCl2", "OC", "CaCO3", "N", "P", "K", "EC"
            ],
            "spatial_visual_resolution": "224x224x3 (RGB)",
            "visual_backbone": "EfficientNet-B0 (7x7x256 patch grid)"
        },
        "standard_splits": {
            "train_pct": 70.0,
            "train_samples": int(round(total_samples * 0.70)),
            "val_pct": 15.0,
            "val_samples": int(round(total_samples * 0.15)),
            "test_pct": 15.0,
            "test_samples": total_samples - int(round(total_samples * 0.70)) - int(round(total_samples * 0.15))
        }
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[2/4] Saved dataset manifest -> {manifest_path}")
    print(f"      SHA256: {sha256_hash}")
    print(f"      MD5:    {md5_hash}")

    print(f"\n[3/4] Dataset Composition:")
    for name, stats in breakdown.items():
        print(f"      Class {stats['label_id']}: {name:<22} N = {stats['count']:>5} ({stats['percentage']:5.2f}%)")
    print(f"      Total N = {total_samples:,}")

if __name__ == "__main__":
    main()
