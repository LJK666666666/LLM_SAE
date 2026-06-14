"""快速 inspect crosslingual 跑出的激活矩阵，定位合理的 active 阈值。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/crosslingual_2")
    args = ap.parse_args()
    d = Path(args.dir)
    blob = torch.load(d / "activations.pt", map_location="cpu")
    Z_en = blob["Z_en"].numpy()
    Z_zh = blob["Z_zh"].numpy()
    print(f"Z_en.shape={Z_en.shape}")
    print(f"Z_en stats: max={Z_en.max():.4f} mean(of nonzero)={Z_en[Z_en>0].mean():.4f}  fraction>0={(Z_en>0).mean():.4f}")
    print(f"Z_zh stats: max={Z_zh.max():.4f} mean(of nonzero)={Z_zh[Z_zh>0].mean():.4f}  fraction>0={(Z_zh>0).mean():.4f}")
    mean_en = Z_en.mean(axis=0)
    mean_zh = Z_zh.mean(axis=0)
    max_mean = np.maximum(mean_en, mean_zh)
    print(f"per-feature max_mean distribution:")
    for q in [0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0]:
        print(f"  q={q:>5.3f}  max_mean={np.quantile(max_mean, q):.6f}")
    print(f"# features with max_mean > 0.01: {(max_mean > 0.01).sum()}")
    print(f"# features with max_mean > 0.001: {(max_mean > 0.001).sum()}")
    print(f"# features with max_mean > 0.0001: {(max_mean > 0.0001).sum()}")

    var_en = Z_en.var(axis=0)
    var_zh = Z_zh.var(axis=0)
    en_std = Z_en.std(axis=0); zh_std = Z_zh.std(axis=0)
    en_centered = Z_en - Z_en.mean(axis=0)
    zh_centered = Z_zh - Z_zh.mean(axis=0)
    corr = (en_centered * zh_centered).mean(axis=0) / (en_std * zh_std + 1e-8)
    is_active = max_mean > 0.001
    print(f"on active features ({is_active.sum()}):")
    print(f"  corr dist: q10={np.quantile(corr[is_active], 0.1):.3f}  q50={np.quantile(corr[is_active], 0.5):.3f}  q90={np.quantile(corr[is_active], 0.9):.3f}")
    print(f"  # corr>0.5: {((corr > 0.5) & is_active).sum()}")
    print(f"  # corr>0.3: {((corr > 0.3) & is_active).sum()}")
    print(f"  # corr<-0.2: {((corr < -0.2) & is_active).sum()}")

    # 语言特异性 ratio
    eps = 1e-8
    en_zh_ratio = var_en / (var_zh + eps)
    zh_en_ratio = var_zh / (var_en + eps)
    print(f"  # en_zh_ratio > 8 (lang specific en) within active: {((en_zh_ratio > 8) & is_active).sum()}")
    print(f"  # zh_en_ratio > 8 (lang specific zh) within active: {((zh_en_ratio > 8) & is_active).sum()}")


if __name__ == "__main__":
    main()
