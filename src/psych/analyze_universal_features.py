"""基于 logit lens 的 SAE 特征语义解读 + 跨语言 token 关联分析。

对每个候选 feature j：
1. 取 SAE decoder 方向 ``d_j = W_dec[j] ∈ R^{d_in}``。
2. 用 logit lens（沿 hook_layer 之后的剩余 transformer 视为单位映射的近似）：
   ``logits = LN(d_j) @ unembed.T``，取 top-K tokens。
3. 统计 top tokens 中中英语种比例（中文字符占比）作为 "cross-lingual binding" 证据。

输入
----
- ``--sae-exp-dir`` SAE 训练目录（含 config + ckpt）。
- ``--xling-dir`` ``crosslingual_features.py`` 的输出目录（读 ``classification.json``）。

输出（写到 xling-dir 下）
----
- ``feature_lens_universal.json``：top universal features 的 lens 解读
- ``feature_lens_lang_specific.json``：top english-/chinese-specific features
- ``lens_summary.txt``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml
from src.models.qwen_loader import load_hooked_qwen
from src.models.sae_topk import TopKSAE, TopKSAEConfig
from src.models.sae_jumprelu import JumpReLUSAE, JumpReLUSAEConfig


def _build_sae(cfg):
    sae_cfg = cfg["sae"]
    variant = sae_cfg.get("variant", "topk").lower()
    if variant == "topk":
        return TopKSAE(TopKSAEConfig(
            d_in=sae_cfg["d_in"], d_sae=sae_cfg["d_sae"],
            k=sae_cfg.get("k", 32), k_aux=sae_cfg.get("k_aux", 256),
            aux_loss_coef=sae_cfg.get("aux_loss_coef", 1.0 / 32),
            dead_steps_threshold=sae_cfg.get("dead_steps_threshold", 1000),
            normalize_decoder=sae_cfg.get("normalize_decoder", True),
        ))
    if variant == "jumprelu":
        return JumpReLUSAE(JumpReLUSAEConfig(
            d_in=sae_cfg["d_in"], d_sae=sae_cfg["d_sae"],
            sparsity_coef=sae_cfg.get("sparsity_coef", 1e-3),
            bandwidth=sae_cfg.get("bandwidth", 1e-3),
            init_threshold=sae_cfg.get("init_threshold", 0.001),
            normalize_decoder=sae_cfg.get("normalize_decoder", True),
        ))
    raise ValueError(variant)


def _is_chinese(s: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in s)


def _has_latin(s: str) -> bool:
    return any(("a" <= ch.lower() <= "z") for ch in s)


def _classify_token(text: str) -> str:
    text_stripped = text.strip()
    if not text_stripped:
        return "punct_or_blank"
    if _is_chinese(text_stripped):
        if _has_latin(text_stripped):
            return "mixed"
        return "chinese"
    if _has_latin(text_stripped):
        return "english"
    if any(ch.isdigit() for ch in text_stripped):
        return "digit"
    return "other"


@torch.no_grad()
def logit_lens_topk(direction: torch.Tensor, unembed: torch.Tensor,
                    norm_layer: torch.nn.Module | None,
                    tokenizer, k: int = 20) -> list[dict]:
    """对一个方向做 logit lens, 返回 top-k tokens 列表。

    direction: [d_in]
    unembed:   [vocab, d_in]
    """
    # 应用最终 LayerNorm/RMSNorm (沿用 model.final_norm 的统计) 提高可读性。
    # 实际上 LayerNorm 是 affine, RMSNorm 是 RMS-scale；我们直接用模型自带 norm。
    v = direction.unsqueeze(0)  # [1, d]
    if norm_layer is not None:
        v = norm_layer(v.to(next(norm_layer.parameters()).dtype))
    v = v.to(unembed.dtype)
    logits = v @ unembed.t()  # [1, vocab]
    logits = logits.squeeze(0).float()
    vals, idx = logits.topk(k)
    items = []
    for v_i, t_i in zip(vals.tolist(), idx.tolist()):
        tok = tokenizer.decode([t_i])
        items.append({"token_id": int(t_i), "token": tok,
                      "logit": float(v_i), "kind": _classify_token(tok)})
    return items


def _find_unembed_and_norm(model):
    """定位 Qwen3.5 文本部分的 lm_head 与 final layernorm。"""
    head = None
    norm = None
    # 1) 顶层 lm_head
    if hasattr(model, "lm_head"):
        head = model.lm_head
    # 2) language_model.lm_head
    if head is None and hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
        head = model.language_model.lm_head
    # 3) language_model.model.norm
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        sub = model.language_model.model
        for cand_name in ("norm", "final_layernorm", "ln_f"):
            if hasattr(sub, cand_name):
                norm = getattr(sub, cand_name)
                break
    elif hasattr(model, "model"):
        sub = model.model
        for cand_name in ("norm", "final_layernorm", "ln_f"):
            if hasattr(sub, cand_name):
                norm = getattr(sub, cand_name)
                break
    if head is None:
        raise RuntimeError("无法定位 lm_head")
    return head, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-exp-dir", default="results/topk_l12_local_2")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--xling-dir", required=True, type=str,
                    help="crosslingual_features.py 输出目录")
    ap.add_argument("--topk-features", type=int, default=50)
    ap.add_argument("--topk-tokens", type=int, default=20)
    args = ap.parse_args()

    xling_dir = Path(args.xling_dir)
    classification = json.loads((xling_dir / "classification.json").read_text(encoding="utf-8"))
    feat_uni_path = xling_dir / "feature_universality.json"
    feat_uni = json.loads(feat_uni_path.read_text(encoding="utf-8"))
    feats_by_id = {r["feature_id"]: r for r in feat_uni["features"]}

    cfg = load_yaml(Path(args.sae_exp_dir) / "config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Qwen
    hooked = load_hooked_qwen(cfg["model"]["model_dir"],
                              hook_layer=cfg["model"]["hook_layer"], device=device)
    head, norm = _find_unembed_and_norm(hooked.model)
    unembed = head.weight.detach()  # [vocab, d_in]
    print(f"[setup] unembed.shape={tuple(unembed.shape)}  norm={type(norm).__name__ if norm else None}")

    # 2) SAE
    sae = _build_sae(cfg)
    state = torch.load(Path(args.sae_exp_dir) / args.ckpt, map_location="cpu")
    sd = state.get("sae", state.get("model", state))
    sae.load_state_dict(sd, strict=False)
    sae.eval().to(device)
    W_dec = sae.W_dec.detach()  # [d_sae, d_in]

    # 3) 选 top features
    def _take(group: str, top: int) -> list[int]:
        ids = classification.get(group, [])
        if group == "universal":
            ids_sorted = sorted(ids, key=lambda j: -feats_by_id.get(j, {"corr": 0})["corr"])
        elif group in ("english_specific", "chinese_specific"):
            side = "var_en" if group == "english_specific" else "var_zh"
            ids_sorted = sorted(ids, key=lambda j: -feats_by_id.get(j, {side: 0})[side])
        elif group == "anti_aligned":
            ids_sorted = sorted(ids, key=lambda j: feats_by_id.get(j, {"corr": 0})["corr"])
        else:
            ids_sorted = ids
        return ids_sorted[:top]

    targets = {
        "universal": _take("universal", args.topk_features),
        "english_specific": _take("english_specific", args.topk_features),
        "chinese_specific": _take("chinese_specific", args.topk_features),
        "anti_aligned": _take("anti_aligned", min(20, args.topk_features)),
    }

    # 4) Lens
    out = {}
    for group, feat_ids in targets.items():
        print(f"[lens] group={group}  n={len(feat_ids)}")
        rows = []
        for j in feat_ids:
            d_j = W_dec[j].to(device)
            top_tokens = logit_lens_topk(d_j, unembed.to(device), norm,
                                          hooked.tokenizer, k=args.topk_tokens)
            kinds = [t["kind"] for t in top_tokens]
            chinese = sum(1 for k in kinds if k == "chinese")
            english = sum(1 for k in kinds if k == "english")
            mixed = sum(1 for k in kinds if k == "mixed")
            other = len(kinds) - chinese - english - mixed
            feat_row = feats_by_id.get(j, {})
            rows.append({
                "feature_id": int(j),
                "corr": feat_row.get("corr"),
                "var_en": feat_row.get("var_en"),
                "var_zh": feat_row.get("var_zh"),
                "freq_en": feat_row.get("freq_en"),
                "freq_zh": feat_row.get("freq_zh"),
                "top_tokens": top_tokens,
                "token_kind_counts": {
                    "chinese": chinese, "english": english,
                    "mixed": mixed, "other": other,
                },
            })
        out[group] = rows

    # 5) 落盘
    (xling_dir / "feature_lens_universal.json").write_text(
        json.dumps(out["universal"], ensure_ascii=False, indent=2), encoding="utf-8")
    (xling_dir / "feature_lens_lang_specific.json").write_text(
        json.dumps({"english_specific": out["english_specific"],
                    "chinese_specific": out["chinese_specific"],
                    "anti_aligned": out["anti_aligned"]},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # 6) summary
    def _agg_kinds(rows):
        c = e = m = o = 0
        for r in rows:
            k = r["token_kind_counts"]
            c += k["chinese"]; e += k["english"]; m += k["mixed"]; o += k["other"]
        total = c + e + m + o
        if total == 0:
            return "no data"
        return (f"chinese={c/total*100:.1f}%  english={e/total*100:.1f}%  "
                f"mixed={m/total*100:.1f}%  other={o/total*100:.1f}%")

    summary_lines = [
        f"=== SAE Feature Logit-Lens Analysis ===",
        f"Top-K features per group: {args.topk_features}",
        f"Top-K tokens per feature: {args.topk_tokens}",
        f"",
        f"[Token kind distribution across top tokens]",
        f"  Universal       : {_agg_kinds(out['universal'])}",
        f"  English-specific: {_agg_kinds(out['english_specific'])}",
        f"  Chinese-specific: {_agg_kinds(out['chinese_specific'])}",
        f"  Anti-aligned    : {_agg_kinds(out['anti_aligned'])}",
        f"",
        f"[A few universal feature snapshots]",
    ]
    for r in out["universal"][:8]:
        toks = " | ".join(t["token"].replace("\n", "\\n") for t in r["top_tokens"][:10])
        summary_lines.append(f"  f{r['feature_id']:>5d}  corr={r['corr']:.3f}  tokens: {toks}")
    summary_lines.append("")
    summary_lines.append("[A few English-specific snapshots]")
    for r in out["english_specific"][:5]:
        toks = " | ".join(t["token"].replace("\n", "\\n") for t in r["top_tokens"][:10])
        summary_lines.append(f"  f{r['feature_id']:>5d}  var_en={r['var_en']:.3f}  tokens: {toks}")
    summary_lines.append("")
    summary_lines.append("[A few Chinese-specific snapshots]")
    for r in out["chinese_specific"][:5]:
        toks = " | ".join(t["token"].replace("\n", "\\n") for t in r["top_tokens"][:10])
        summary_lines.append(f"  f{r['feature_id']:>5d}  var_zh={r['var_zh']:.3f}  tokens: {toks}")

    summary = "\n".join(summary_lines)
    print(summary)
    (xling_dir / "lens_summary.txt").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
