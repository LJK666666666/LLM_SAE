"""跨语言概念-特征对齐分析。

实验目标
--------
对每个概念 c，分别在英文/中文模板下采集 hook_layer 的 SAE 激活，得到
``Z_en, Z_zh ∈ R^{N_concept × d_sae}``。从中推导：

1. **概念级别一致度**：对每个概念 c，``cos(z_en[c], z_zh[c])`` —— 越大说明
   "同一概念在两种语言下激活的 SAE 特征模式越像"。
2. **特征级别普适性**：对每个特征 j，``corr(Z_en[:, j], Z_zh[:, j])`` —— 越大
   说明该特征在跨概念维度上对中英文同步反应。
3. **特征类型分类**：Universal / English-specific / Chinese-specific /
   Anti-aligned，依据是单语方差 + 跨语相关。
4. **类别分项**：把概念按 category（具体名词 / 抽象 / 数学 / ...）分组，
   看哪一类概念跨语言一致度最高。

输出
----
``results/crosslingual_<tag>_<N>/`` 下：
- ``per_concept.json``：每个概念的 cos / top-K features (en/zh) / Jaccard
- ``feature_universality.json``：每个特征的 (var_en, var_zh, corr) 与归类
- ``Z_en.pt`` / ``Z_zh.pt``：原始激活矩阵
- ``figures/`` 下若干图

约定（CLAUDE.md 规则 4）：图中文字英文，不设 title。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml
from src.utils.exp_dir import make_or_resume_exp_dir
from src.models.qwen_loader import load_hooked_qwen
from src.models.sae_topk import TopKSAE, TopKSAEConfig
from src.models.sae_jumprelu import JumpReLUSAE, JumpReLUSAEConfig


def build_sae_from_cfg(cfg: dict) -> torch.nn.Module:
    sae_cfg = cfg["sae"]
    variant = sae_cfg.get("variant", "topk").lower()
    if variant == "topk":
        c = TopKSAEConfig(
            d_in=sae_cfg["d_in"], d_sae=sae_cfg["d_sae"],
            k=sae_cfg.get("k", 32), k_aux=sae_cfg.get("k_aux", 256),
            aux_loss_coef=sae_cfg.get("aux_loss_coef", 1.0 / 32),
            dead_steps_threshold=sae_cfg.get("dead_steps_threshold", 1000),
            normalize_decoder=sae_cfg.get("normalize_decoder", True),
        )
        return TopKSAE(c)
    if variant == "jumprelu":
        c = JumpReLUSAEConfig(
            d_in=sae_cfg["d_in"], d_sae=sae_cfg["d_sae"],
            sparsity_coef=sae_cfg.get("sparsity_coef", 1e-3),
            bandwidth=sae_cfg.get("bandwidth", 1e-3),
            init_threshold=sae_cfg.get("init_threshold", 0.001),
            normalize_decoder=sae_cfg.get("normalize_decoder", True),
        )
        return JumpReLUSAE(c)
    raise ValueError(f"未知 SAE 变体: {variant}")


def load_sae(exp_dir: Path, ckpt_name: str, device: str) -> torch.nn.Module:
    cfg = load_yaml(exp_dir / "config.yaml")
    sae = build_sae_from_cfg(cfg)
    state = torch.load(exp_dir / ckpt_name, map_location="cpu")
    sd = state.get("sae", state.get("model", state))
    sae.load_state_dict(sd, strict=False)
    sae.eval().to(device)
    return sae, cfg


@torch.no_grad()
def encode_prompts(hooked, sae, prompts: list[str], ctx_len: int = 48,
                   batch_size: int = 16, device: str = "cuda") -> np.ndarray:
    """对每个 prompt 取 hook_layer 激活，过 SAE encode 得到 z（仅 top-k 非零）。
    每个 prompt 输出 z 在 valid tokens 上的平均，形状 [N_prompts, d_sae]。
    """
    sae.eval()
    out = np.zeros((len(prompts), sae.cfg.d_sae), dtype=np.float32)
    tok = hooked.tokenizer
    for i in tqdm(range(0, len(prompts), batch_size), desc="encode"):
        texts = prompts[i:i + batch_size]
        enc = tok(texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=ctx_len)
        input_ids = enc["input_ids"].to(hooked.device)
        attn = enc["attention_mask"].to(hooked.device)
        act = hooked.get_activations(input_ids, attn)  # [B, T, D] bf16
        B, T, D = act.shape
        x = act.reshape(-1, D).float()
        z = sae.encode(x).cpu()  # [B*T, d_sae]
        mask = attn.reshape(-1).cpu().bool()  # [B*T]
        z = z.view(B, T, -1)
        mask = mask.view(B, T)
        # 对 valid tokens 做平均
        z_sum = (z * mask.unsqueeze(-1).float()).sum(dim=1)         # [B, d_sae]
        n_valid = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()  # [B, 1]
        z_mean = (z_sum / n_valid).numpy()
        out[i:i + len(texts)] = z_mean
    return out


def per_concept_consistency(Z_en: np.ndarray, Z_zh: np.ndarray) -> np.ndarray:
    """每个概念的 cos(z_en, z_zh)。"""
    en = Z_en / (np.linalg.norm(Z_en, axis=1, keepdims=True) + 1e-8)
    zh = Z_zh / (np.linalg.norm(Z_zh, axis=1, keepdims=True) + 1e-8)
    return (en * zh).sum(axis=1)


def jaccard_topk(Z_en: np.ndarray, Z_zh: np.ndarray, k: int = 20) -> np.ndarray:
    out = np.zeros(len(Z_en))
    for i in range(len(Z_en)):
        a = set(np.argsort(-Z_en[i])[:k].tolist())
        b = set(np.argsort(-Z_zh[i])[:k].tolist())
        if not a and not b:
            out[i] = 0.0
        else:
            out[i] = len(a & b) / max(len(a | b), 1)
    return out


def feature_correlation(Z_en: np.ndarray, Z_zh: np.ndarray) -> dict:
    """对每个特征 j 计算跨概念的中英文相关性、各自方差等。"""
    # 标准化（按列）
    en_mean = Z_en.mean(axis=0)
    zh_mean = Z_zh.mean(axis=0)
    en_std = Z_en.std(axis=0)
    zh_std = Z_zh.std(axis=0)
    en_centered = Z_en - en_mean
    zh_centered = Z_zh - zh_mean
    denom = (en_std * zh_std + 1e-8)
    corr = (en_centered * zh_centered).mean(axis=0) / denom

    # 激活频率（多少 concept 上该 feature mean>0）
    freq_en = (Z_en > 0).mean(axis=0)
    freq_zh = (Z_zh > 0).mean(axis=0)
    mean_en = Z_en.mean(axis=0)
    mean_zh = Z_zh.mean(axis=0)

    return {
        "corr": corr.astype(np.float32),
        "var_en": (en_std ** 2).astype(np.float32),
        "var_zh": (zh_std ** 2).astype(np.float32),
        "freq_en": freq_en.astype(np.float32),
        "freq_zh": freq_zh.astype(np.float32),
        "mean_en": mean_en.astype(np.float32),
        "mean_zh": mean_zh.astype(np.float32),
    }


def classify_features(stats: dict,
                      active_quantile: float = 0.0,
                      corr_universal: float = 0.5,
                      corr_anti: float = -0.2,
                      lang_specific_ratio: float = 8.0,
                      min_max_mean: float = 0.001) -> dict:
    """根据 var_en/var_zh/corr 将特征分类，使用相对阈值。

    规则
    ----
    - 活跃集合 :math:`A`：``max(mean_en, mean_zh) > min_max_mean``（按平均激活幅度筛掉
      "几乎从未激活" 的死特征）。``active_quantile`` 可作为可选的进一步比例下限。
    - **Universal**: ``j ∈ A`` 且 ``var_en ≥ q`` 且 ``var_zh ≥ q`` 且 ``corr ≥ corr_universal``
      （q = active set 内 var 的 10% 分位）。
    - **English-specific**: ``j ∈ A`` 且 ``var_en / (var_zh+eps) > lang_specific_ratio``。
    - **Chinese-specific**: 同上反向。
    - **Anti-aligned**: ``j ∈ A`` 两边 var 都过 q，且 ``corr ≤ corr_anti``。
    - **Dead**: ``j ∉ A``。
    - **Other**: ``A`` 中其余（弱对齐）。
    """
    var_en, var_zh, corr = stats["var_en"], stats["var_zh"], stats["corr"]
    mean_en, mean_zh = stats["mean_en"], stats["mean_zh"]
    eps = 1e-8
    max_mean = np.maximum(mean_en, mean_zh)
    active = max_mean > min_max_mean
    n_active = int(active.sum())
    # 在活跃集合上取 var 的 10% 分位作为"两边都不小"的下限
    if n_active > 0:
        q_en = float(np.quantile(var_en[active], 0.1))
        q_zh = float(np.quantile(var_zh[active], 0.1))
    else:
        q_en = q_zh = 0.0
    is_active_both = active & (var_en >= q_en) & (var_zh >= q_zh)
    universal = is_active_both & (corr >= corr_universal)
    anti = is_active_both & (corr <= corr_anti)
    en_only = active & (var_en / (var_zh + eps) > lang_specific_ratio) & (var_en > q_en)
    zh_only = active & (var_zh / (var_en + eps) > lang_specific_ratio) & (var_zh > q_zh)
    dead = ~active
    other = active & ~(universal | anti | en_only | zh_only)
    return {
        "universal": np.where(universal)[0].tolist(),
        "english_specific": np.where(en_only)[0].tolist(),
        "chinese_specific": np.where(zh_only)[0].tolist(),
        "anti_aligned": np.where(anti)[0].tolist(),
        "dead": np.where(dead)[0].tolist(),
        "other": np.where(other)[0].tolist(),
        "_thresholds": {"q_en": q_en, "q_zh": q_zh,
                        "n_active": n_active, "min_max_mean": min_max_mean},
    }


def category_breakdown(concepts: list[dict], cos_per_concept: np.ndarray,
                       jaccard_per_concept: np.ndarray) -> dict:
    cats = {}
    for c, cos_v, j_v in zip(concepts, cos_per_concept, jaccard_per_concept):
        cat = c["category"]
        cats.setdefault(cat, {"cos": [], "jaccard": [], "n": 0,
                              "concepts": []})
        cats[cat]["cos"].append(float(cos_v))
        cats[cat]["jaccard"].append(float(j_v))
        cats[cat]["n"] += 1
        cats[cat]["concepts"].append(c["id"])
    return {
        cat: {
            "n": v["n"],
            "cos_mean": float(np.mean(v["cos"])),
            "cos_std": float(np.std(v["cos"])),
            "jaccard_mean": float(np.mean(v["jaccard"])),
            "concepts": v["concepts"],
        }
        for cat, v in cats.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-exp-dir", default="results/topk_l12_local_2")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--concepts", default="data/psych/concepts.json")
    ap.add_argument("--out-tag", default="crosslingual")
    ap.add_argument("--ctx-len", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--topk-jaccard", type=int, default=20)
    args = ap.parse_args()

    exp_dir = Path(args.sae_exp_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 概念语料
    raw = json.loads(Path(args.concepts).read_text(encoding="utf-8"))
    concepts = raw["concepts"]
    templates_en = raw["templates_en"]
    templates_zh = raw["templates_zh"]
    T = len(templates_en)
    assert len(templates_zh) == T

    prompts_en, prompts_zh, idx_map = [], [], []
    for ci, c in enumerate(concepts):
        for t in templates_en:
            prompts_en.append(t.format(en=c["en"]))
            idx_map.append(ci)
        for t in templates_zh:
            prompts_zh.append(t.format(zh=c["zh"]))
    print(f"[setup] concepts={len(concepts)} templates={T} prompts_en={len(prompts_en)} prompts_zh={len(prompts_zh)}")

    # 2) 加载 SAE + hooked Qwen
    sae, train_cfg = load_sae(exp_dir, args.ckpt, device)
    hook_layer = train_cfg["model"]["hook_layer"]
    model_dir = train_cfg["model"]["model_dir"]
    print(f"[setup] SAE: d_in={sae.cfg.d_in} d_sae={sae.cfg.d_sae} hook_layer={hook_layer}")

    hooked = load_hooked_qwen(model_dir, hook_layer=hook_layer, device=device)

    # 3) 编码
    Z_en_flat = encode_prompts(hooked, sae, prompts_en, args.ctx_len, args.batch_size, device)
    Z_zh_flat = encode_prompts(hooked, sae, prompts_zh, args.ctx_len, args.batch_size, device)

    # 平均 over templates -> [N_concept, d_sae]
    Z_en = Z_en_flat.reshape(len(concepts), T, -1).mean(axis=1)
    Z_zh = Z_zh_flat.reshape(len(concepts), T, -1).mean(axis=1)
    print(f"[setup] Z_en.shape={Z_en.shape} Z_zh.shape={Z_zh.shape}")

    # 4) 指标
    cos_pc = per_concept_consistency(Z_en, Z_zh)
    jacc_pc = jaccard_topk(Z_en, Z_zh, k=args.topk_jaccard)
    feat_stats = feature_correlation(Z_en, Z_zh)
    classification = classify_features(feat_stats)

    # 5) 输出目录
    out_dir = make_or_resume_exp_dir(args.out_tag)
    print(f"[out] {out_dir}")

    # 6) 保存原始激活
    torch.save({"Z_en": torch.tensor(Z_en), "Z_zh": torch.tensor(Z_zh),
                "concept_ids": [c["id"] for c in concepts]},
               out_dir / "activations.pt")

    # 7) per_concept.json
    per_concept = []
    for i, c in enumerate(concepts):
        top_en = np.argsort(-Z_en[i])[:args.topk_jaccard].tolist()
        top_zh = np.argsort(-Z_zh[i])[:args.topk_jaccard].tolist()
        per_concept.append({
            "id": c["id"], "category": c["category"],
            "en": c["en"], "zh": c["zh"],
            "cos_en_zh": float(cos_pc[i]),
            "jaccard_top": float(jacc_pc[i]),
            "top_features_en": top_en,
            "top_features_zh": top_zh,
            "shared_features": list(set(top_en) & set(top_zh)),
        })
    (out_dir / "per_concept.json").write_text(
        json.dumps(per_concept, ensure_ascii=False, indent=2), encoding="utf-8")

    # 8) feature_universality.json
    feat_records = []
    for j in range(sae.cfg.d_sae):
        if feat_stats["var_en"][j] < 1e-6 and feat_stats["var_zh"][j] < 1e-6:
            continue
        feat_records.append({
            "feature_id": int(j),
            "corr": float(feat_stats["corr"][j]),
            "var_en": float(feat_stats["var_en"][j]),
            "var_zh": float(feat_stats["var_zh"][j]),
            "freq_en": float(feat_stats["freq_en"][j]),
            "freq_zh": float(feat_stats["freq_zh"][j]),
            "mean_en": float(feat_stats["mean_en"][j]),
            "mean_zh": float(feat_stats["mean_zh"][j]),
        })
    feat_records.sort(key=lambda r: -r["corr"])
    (out_dir / "feature_universality.json").write_text(
        json.dumps({
            "n_features_total": int(sae.cfg.d_sae),
            "n_features_active": len(feat_records),
            "classification_counts": {k: len(v) for k, v in classification.items()},
            "features": feat_records,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    # 写出 classification 时把 _thresholds 一并保留
    (out_dir / "classification.json").write_text(
        json.dumps(classification, indent=2), encoding="utf-8")
    print(f"[classify] thresholds={classification.get('_thresholds')}")

    # 9) 按 category 分项
    cats = category_breakdown(concepts, cos_pc, jacc_pc)
    (out_dir / "category_breakdown.json").write_text(
        json.dumps(cats, ensure_ascii=False, indent=2), encoding="utf-8")

    # 10) summary.txt
    overall_cos = float(np.mean(cos_pc))
    overall_jacc = float(np.mean(jacc_pc))
    n_universal = len(classification["universal"])
    n_en_spec = len(classification["english_specific"])
    n_zh_spec = len(classification["chinese_specific"])
    n_anti = len(classification["anti_aligned"])
    n_active = len(feat_records)

    summary = (
        f"=== Cross-lingual Concept Universality (Qwen3.5 / SAE L{hook_layer}) ===\n"
        f"SAE: variant={train_cfg['sae']['variant']} d_sae={sae.cfg.d_sae} k={train_cfg['sae'].get('k','-')}\n"
        f"#concepts={len(concepts)}  #templates(per lang)={T}\n\n"
        f"[Overall]\n"
        f"  mean cos(z_en, z_zh) = {overall_cos:.4f}\n"
        f"  mean Jaccard@{args.topk_jaccard} = {overall_jacc:.4f}\n\n"
        f"[Feature typology]  (active features: {n_active}/{sae.cfg.d_sae})\n"
        f"  Universal       : {n_universal}\n"
        f"  English-specific: {n_en_spec}\n"
        f"  Chinese-specific: {n_zh_spec}\n"
        f"  Anti-aligned    : {n_anti}\n"
        f"  Other (weak)    : {len(classification['other'])}\n"
        f"  Dead            : {len(classification['dead'])}\n\n"
        f"[Per-category cos mean]\n"
    )
    for cat, v in sorted(cats.items(), key=lambda kv: -kv[1]["cos_mean"]):
        summary += f"  {cat:18s} n={v['n']:3d}  cos={v['cos_mean']:.3f}±{v['cos_std']:.3f}  jaccard={v['jaccard_mean']:.3f}\n"

    print(summary)
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")

    print(f"[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
