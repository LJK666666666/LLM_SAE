"""可视化训练曲线 + （若存在）评估特征密度直方图。

输入
----
- ``{exp_dir}/history.csv``：训练 epoch 维度记录
- 可选 ``{exp_dir}/eval/feature_density.json``：每个 split 的特征激活密度

输出（{exp_dir}/figures/）
----
- loss_curves.png：train/val recon loss
- l0_curve.png：train/val L0
- explained_var.png：train/val explained variance
- lr_curve.png：学习率
- dead_frac.png：训练侧 dead feature 占比
- feature_density_hist.png（若有评估文件）

约定（CLAUDE.md 规则 4）：英文文字，不设 title。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  saved {path}")


def plot_history(hist: pd.DataFrame, out_dir: Path) -> None:
    x = hist["epoch"].to_numpy()

    # loss
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, hist["train_recon"], label="train_recon", linewidth=1.5)
    ax.plot(x, hist["val_recon"], label="val_recon", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction loss (sum of squares per token)")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, out_dir / "loss_curves.png")

    # L0
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, hist["train_l0"], label="train_L0", linewidth=1.5)
    ax.plot(x, hist["val_l0"], label="val_L0", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L0 (active features per token)")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, out_dir / "l0_curve.png")

    # explained variance
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, hist["train_expl_var"], label="train_expl_var", linewidth=1.5)
    ax.plot(x, hist["val_expl_var"], label="val_expl_var", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Explained variance")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, out_dir / "explained_var.png")

    # lr
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, hist["lr"], color="C2", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    _save(fig, out_dir / "lr_curve.png")

    # dead frac
    if "train_dead_frac" in hist.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, hist["train_dead_frac"], color="C3", linewidth=1.5,
                label="train_dead_frac")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dead feature fraction")
        ax.grid(alpha=0.3)
        ax.legend()
        _save(fig, out_dir / "dead_frac.png")


def plot_feature_density(densities: dict[str, list[float]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (label, vals) in enumerate(densities.items()):
        v = np.asarray(vals, dtype=np.float64)
        # 用 log10(density) 直方图，density 为 0 的设成 -inf 单独算
        active = v[v > 0]
        zero_frac = float((v == 0).mean())
        if active.size == 0:
            continue
        log_density = np.log10(active)
        ax.hist(log_density, bins=60, alpha=0.5,
                label=f"{label} (active={active.size}, dead={zero_frac*100:.1f}%)")
    ax.set_xlabel("log10(activation density per token)")
    ax.set_ylabel("Number of features")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, out_dir / "feature_density_hist.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True, type=str)
    args = ap.parse_args()
    exp_dir = Path(args.exp_dir)

    hist_path = exp_dir / "history.csv"
    if not hist_path.exists():
        sys.exit(f"找不到 {hist_path}")
    hist = pd.read_csv(hist_path)
    fig_dir = exp_dir / "figures"
    print(f"[viz] reading {hist_path}  ({len(hist)} epochs)")
    plot_history(hist, fig_dir)

    density_path = exp_dir / "eval" / "feature_density.json"
    if density_path.exists():
        print(f"[viz] reading {density_path}")
        densities = json.loads(density_path.read_text(encoding="utf-8"))
        plot_feature_density(densities, fig_dir)
    else:
        print(f"[viz] 跳过特征密度图（未找到 {density_path}，先跑 evaluate.py 即可）。")

    print(f"[viz] 完成。输出目录：{fig_dir}")


if __name__ == "__main__":
    main()
