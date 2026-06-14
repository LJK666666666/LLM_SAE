"""绘制跨语言 SAE 分析的论文图表。

约定：英文 label，无 title。
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
import torch


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  saved {path}")


def plot_concept_consistency(per_concept: list, Z_en: np.ndarray, Z_zh: np.ndarray,
                              fig_dir: Path):
    cos_vals = np.array([c["cos_en_zh"] for c in per_concept])

    # Permutation baseline: shuffle zh concepts then recompute
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Z_zh))
    Z_zh_shuf = Z_zh[perm]
    en_norm = Z_en / (np.linalg.norm(Z_en, axis=1, keepdims=True) + 1e-8)
    zh_norm = Z_zh_shuf / (np.linalg.norm(Z_zh_shuf, axis=1, keepdims=True) + 1e-8)
    perm_cos = (en_norm * zh_norm).sum(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cos_vals, bins=24, alpha=0.7, color="C0", label=f"Aligned (mean={cos_vals.mean():.3f})")
    ax.hist(perm_cos, bins=24, alpha=0.5, color="C3",
            label=f"Random pair (mean={perm_cos.mean():.3f})")
    ax.set_xlabel("cos(z_en, z_zh) per concept")
    ax.set_ylabel("Number of concepts")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, fig_dir / "concept_consistency_hist.png")


def plot_category_breakdown(cats: dict, fig_dir: Path):
    items = sorted(cats.items(), key=lambda kv: kv[1]["cos_mean"])
    labels = [k for k, _ in items]
    means = [v["cos_mean"] for _, v in items]
    stds = [v["cos_std"] for _, v in items]
    js = [v["jaccard_mean"] for _, v in items]
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].barh(y, means, xerr=stds, color="C0", alpha=0.85)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("cos(z_en, z_zh) (mean ± std across concepts)")
    axes[0].grid(alpha=0.3, axis="x")

    axes[1].barh(y, js, color="C2", alpha=0.85)
    axes[1].set_yticks(y); axes[1].set_yticklabels(labels)
    axes[1].set_xlabel("Jaccard@20 (top features overlap)")
    axes[1].grid(alpha=0.3, axis="x")

    _save(fig, fig_dir / "category_breakdown.png")


def plot_corr_distribution(feat_uni: dict, fig_dir: Path):
    feats = feat_uni["features"]
    corrs = np.array([f["corr"] for f in feats])
    var_en = np.array([f["var_en"] for f in feats])
    var_zh = np.array([f["var_zh"] for f in feats])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(corrs, bins=40, color="C0", alpha=0.85)
    ax.axvline(0.5, color="C3", linestyle="--", label="universal threshold (0.5)")
    ax.axvline(-0.2, color="C2", linestyle="--", label="anti-aligned threshold (-0.2)")
    ax.set_xlabel("Per-feature corr(Z_en[:, j], Z_zh[:, j])")
    ax.set_ylabel("Number of (active) features")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, fig_dir / "feature_corr_hist.png")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(np.log10(var_en + 1e-12), np.log10(var_zh + 1e-12),
               c=corrs, cmap="coolwarm", s=10, alpha=0.7, vmin=-1, vmax=1)
    lo, hi = -10, max(np.log10(var_en + 1e-12).max(), np.log10(var_zh + 1e-12).max())
    ax.plot([lo, hi], [lo, hi], color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("log10(var(Z_en[:, j]))")
    ax.set_ylabel("log10(var(Z_zh[:, j]))")
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("corr")
    _save(fig, fig_dir / "feature_var_scatter.png")


def plot_classification_bar(classification: dict, fig_dir: Path):
    counts = {k: len(v) for k, v in classification.items()
              if k not in ("dead", "_thresholds")}
    order = ["universal", "english_specific", "chinese_specific",
             "anti_aligned", "other"]
    labels = [k.replace("_", " ") for k in order]
    vals = [counts.get(k, 0) for k in order]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["C0", "C1", "C3", "C4", "C7"]
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.01, str(v), ha="center")
    ax.set_ylabel("Number of active features")
    ax.set_xlabel("Feature type")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, fig_dir / "feature_classification.png")


def plot_lens_token_kind(lens_universal: list, lens_ls: dict, fig_dir: Path):
    def agg(rows):
        c = e = m = o = 0
        for r in rows:
            k = r["token_kind_counts"]
            c += k["chinese"]; e += k["english"]; m += k["mixed"]; o += k["other"]
        return [c, e, m, o]

    groups = {
        "Universal": agg(lens_universal),
        "English-specific": agg(lens_ls.get("english_specific", [])),
        "Chinese-specific": agg(lens_ls.get("chinese_specific", [])),
    }
    kinds = ["chinese", "english", "mixed", "other"]
    bottoms = np.zeros(len(groups))
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["C3", "C0", "C2", "C7"]
    for i, kind in enumerate(kinds):
        vals = np.array([groups[g][i] for g in groups])
        ax.bar(x, vals, bottom=bottoms, label=kind, color=colors[i], alpha=0.9)
        bottoms = bottoms + vals
    ax.set_xticks(x); ax.set_xticklabels(groups.keys())
    ax.set_ylabel("Logit-lens top-20 token count (aggregated)")
    ax.legend(title="Token kind")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, fig_dir / "lens_token_kind.png")


def plot_steering_kl(steering_rows: list, fig_dir: Path):
    # 按 (feature_id, alpha) 平均，区分 en / zh
    by = {}
    for row in steering_rows:
        for r in row["results"]:
            key = (row["feature_id"], row["prompt_lang"], r["alpha"])
            by.setdefault(key, []).append(r["kl_from_base"])
    feats = sorted(set(k[0] for k in by))
    alphas = sorted(set(k[2] for k in by))
    fig, ax = plt.subplots(figsize=(7, 4))
    markers = {"en": "o", "zh": "s"}
    colors = ["C0", "C1", "C2", "C3", "C4"]
    for fi, f in enumerate(feats):
        for lang in ("en", "zh"):
            ys = []
            for a in alphas:
                vals = by.get((f, lang, a), [])
                ys.append(np.mean(vals) if vals else np.nan)
            ax.plot(alphas, ys, marker=markers[lang], color=colors[fi % len(colors)],
                    linestyle=("-" if lang == "en" else "--"),
                    label=f"f{f} [{lang}]")
    ax.set_xlabel("Steering coefficient α")
    ax.set_ylabel("KL from baseline (next-token, top-200)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, fig_dir / "steering_kl_vs_alpha.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xling-dir", required=True)
    ap.add_argument("--fig-dir", default=None,
                    help="若空，则写入 xling-dir/figures/ 与 paper/figures/")
    args = ap.parse_args()

    xling_dir = Path(args.xling_dir)
    out_fig = Path(args.fig_dir) if args.fig_dir else (xling_dir / "figures")
    paper_fig = Path("paper/figures")
    out_fig.mkdir(parents=True, exist_ok=True)
    paper_fig.mkdir(parents=True, exist_ok=True)

    # 数据
    per_concept = json.loads((xling_dir / "per_concept.json").read_text(encoding="utf-8"))
    cats = json.loads((xling_dir / "category_breakdown.json").read_text(encoding="utf-8"))
    feat_uni = json.loads((xling_dir / "feature_universality.json").read_text(encoding="utf-8"))
    classification = json.loads((xling_dir / "classification.json").read_text(encoding="utf-8"))
    act = torch.load(xling_dir / "activations.pt", map_location="cpu")
    Z_en = act["Z_en"].numpy(); Z_zh = act["Z_zh"].numpy()

    # 画图（同时输出到两个目录）
    for d in (out_fig, paper_fig):
        plot_concept_consistency(per_concept, Z_en, Z_zh, d)
        plot_category_breakdown(cats, d)
        plot_corr_distribution(feat_uni, d)
        plot_classification_bar(classification, d)

    lens_uni_path = xling_dir / "feature_lens_universal.json"
    lens_ls_path = xling_dir / "feature_lens_lang_specific.json"
    if lens_uni_path.exists() and lens_ls_path.exists():
        lens_uni = json.loads(lens_uni_path.read_text(encoding="utf-8"))
        lens_ls = json.loads(lens_ls_path.read_text(encoding="utf-8"))
        for d in (out_fig, paper_fig):
            plot_lens_token_kind(lens_uni, lens_ls, d)

    steer_path = xling_dir / "steering_results.json"
    if steer_path.exists():
        steering = json.loads(steer_path.read_text(encoding="utf-8"))
        for d in (out_fig, paper_fig):
            plot_steering_kl(steering, d)

    print(f"[done] figures saved to {out_fig} and {paper_fig}")


if __name__ == "__main__":
    main()
