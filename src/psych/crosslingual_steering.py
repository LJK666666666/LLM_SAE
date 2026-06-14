"""跨语言 SAE steering：用普适 feature 引导生成，验证其在中英文中产生
"概念上一致" 的行为偏移。

方法
----
对一个候选 feature j：
- 取 ``d_j = W_dec[j]`` 作为 steering 向量。
- 在 hook_layer 之后向残差流加上 ``α * d_j / ||d_j||``。
- 在一组中英文中性 prompt 下做 greedy 生成（max_new_tokens=40），与 α=0 基线比较。

指标
----
- **Next-token KL**：steering 后下一个 token 分布与基线的 KL。
- **Top-token 偏移**：steering 后 top-5 候选 token 的语种构成。
- 定性：输出 generation 文本。

输出：写入 ``--xling-dir`` 下的 ``steering_results.json`` 与 ``steering_samples.txt``。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

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
    return JumpReLUSAE(JumpReLUSAEConfig(
        d_in=sae_cfg["d_in"], d_sae=sae_cfg["d_sae"],
        sparsity_coef=sae_cfg.get("sparsity_coef", 1e-3),
        bandwidth=sae_cfg.get("bandwidth", 1e-3),
        init_threshold=sae_cfg.get("init_threshold", 0.001),
        normalize_decoder=sae_cfg.get("normalize_decoder", True),
    ))


class SteeringHook:
    """加性 steering：output[0] += alpha * direction (向 hook 层 hidden_states)。"""

    def __init__(self, direction: torch.Tensor, alpha: float):
        self.direction = direction  # [d_in]
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        if isinstance(output, (tuple, list)):
            hs = output[0]
            rest = output[1:]
        else:
            hs, rest = output, None
        delta = (self.alpha * self.direction).to(hs.dtype).to(hs.device)
        hs_new = hs + delta  # broadcast over [B, T]
        if rest is not None:
            return (hs_new,) + tuple(rest)
        return hs_new


def _is_chinese(s):
    return any("一" <= ch <= "鿿" for ch in s)


def _has_latin(s):
    return any("a" <= ch.lower() <= "z" for ch in s)


def _token_kind(s):
    s = s.strip()
    if not s: return "blank"
    if _is_chinese(s) and _has_latin(s): return "mixed"
    if _is_chinese(s): return "zh"
    if _has_latin(s): return "en"
    if any(ch.isdigit() for ch in s): return "digit"
    return "other"


@torch.no_grad()
def greedy_generate(model, tokenizer, layer, direction, alpha, prompt,
                    max_new_tokens=40, device="cuda"):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    hook = SteeringHook(direction, alpha) if alpha != 0 else None
    handle = layer.register_forward_hook(hook) if hook is not None else None
    try:
        for _ in range(max_new_tokens):
            logits = model(input_ids=input_ids, use_cache=False).logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=-1)
            if next_id.item() == tokenizer.eos_token_id:
                break
    finally:
        if handle is not None:
            handle.remove()
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


@torch.no_grad()
def next_token_distribution(model, layer, tokenizer, prompt, direction, alpha, device):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    handle = None
    if alpha != 0:
        handle = layer.register_forward_hook(SteeringHook(direction, alpha))
    try:
        logits = model(input_ids=input_ids, use_cache=False).logits[:, -1, :]
    finally:
        if handle is not None:
            handle.remove()
    return F.log_softmax(logits, dim=-1).squeeze(0).float()


def kl_top(logp_a, logp_b, k=200):
    """对 a 的 top-k tokens 计算 KL(P_a || P_b)。"""
    top = logp_a.topk(k).indices
    p_a = logp_a[top].exp()
    return float((p_a * (logp_a[top] - logp_b[top])).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-exp-dir", default="results/topk_l12_local_2")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--xling-dir", required=True)
    ap.add_argument("--top-feature-rank", type=int, default=5,
                    help="挑 universal classification 中 corr 最高的前 N 个 feature 做 steering")
    ap.add_argument("--alphas", type=str, default="0,4,12,40")
    ap.add_argument("--max-new-tokens", type=int, default=30)
    args = ap.parse_args()

    xling_dir = Path(args.xling_dir)
    classification = json.loads((xling_dir / "classification.json").read_text(encoding="utf-8"))
    feat_uni = json.loads((xling_dir / "feature_universality.json").read_text(encoding="utf-8"))
    feats_by_id = {r["feature_id"]: r for r in feat_uni["features"]}

    cfg = load_yaml(Path(args.sae_exp_dir) / "config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hooked = load_hooked_qwen(cfg["model"]["model_dir"],
                              hook_layer=cfg["model"]["hook_layer"], device=device)
    hooked.remove_hook()  # 不需要原 hook，自己加 steering hook
    layer = hooked.text_module.layers[hooked.hook_layer]

    sae = _build_sae(cfg)
    state = torch.load(Path(args.sae_exp_dir) / args.ckpt, map_location="cpu")
    sd = state.get("sae", state.get("model", state))
    sae.load_state_dict(sd, strict=False)
    sae.eval().to(device)

    universals = classification.get("universal", [])
    universals_sorted = sorted(universals, key=lambda j: -feats_by_id.get(j, {"corr": 0})["corr"])
    top_feats = universals_sorted[:args.top_feature_rank]
    print(f"[steering] 选定 universal features (按 corr 排): {top_feats}")

    alphas = [float(x) for x in args.alphas.split(",")]

    prompts = [
        {"id": "en_neutral_1", "lang": "en", "text": "I will now describe a topic to you. The topic is"},
        {"id": "en_neutral_2", "lang": "en", "text": "Today's lecture is about"},
        {"id": "en_neutral_3", "lang": "en", "text": "Let me think for a moment. What I really want to discuss is"},
        {"id": "zh_neutral_1", "lang": "zh", "text": "我现在向你描述一个主题，这个主题是"},
        {"id": "zh_neutral_2", "lang": "zh", "text": "今天讲座的内容是"},
        {"id": "zh_neutral_3", "lang": "zh", "text": "让我想一下，我真正想讨论的是"},
    ]

    results = []
    samples_text_lines = []

    for j in top_feats:
        d_j = sae.W_dec[j].detach().to(device)
        d_j_unit = d_j / d_j.norm().clamp_min(1e-8)
        feat_meta = feats_by_id.get(j, {})
        for p in prompts:
            base_logp = next_token_distribution(hooked.model, layer, hooked.tokenizer,
                                                p["text"], d_j_unit, 0.0, device)
            per_alpha = []
            for a in alphas:
                if a == 0:
                    gen = greedy_generate(hooked.model, hooked.tokenizer, layer,
                                          d_j_unit, 0.0, p["text"],
                                          max_new_tokens=args.max_new_tokens, device=device)
                    per_alpha.append({"alpha": a, "gen": gen, "kl_from_base": 0.0})
                    continue
                logp = next_token_distribution(hooked.model, layer, hooked.tokenizer,
                                                p["text"], d_j_unit, a, device)
                kl = kl_top(base_logp, logp, k=200)
                gen = greedy_generate(hooked.model, hooked.tokenizer, layer,
                                      d_j_unit, a, p["text"],
                                      max_new_tokens=args.max_new_tokens, device=device)
                per_alpha.append({"alpha": a, "gen": gen, "kl_from_base": kl})

            results.append({
                "feature_id": j,
                "feature_corr": feat_meta.get("corr"),
                "feature_var_en": feat_meta.get("var_en"),
                "feature_var_zh": feat_meta.get("var_zh"),
                "prompt_id": p["id"], "prompt_lang": p["lang"],
                "prompt": p["text"],
                "results": per_alpha,
            })
            samples_text_lines.append("")
            samples_text_lines.append(f"=== feature j={j}  corr={feat_meta.get('corr'):.3f}  | prompt[{p['lang']}]: {p['text']!r}")
            for r in per_alpha:
                samples_text_lines.append(f"  α={r['alpha']:>5}  kl={r['kl_from_base']:.3f}  -> {r['gen']!r}")

    (xling_dir / "steering_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (xling_dir / "steering_samples.txt").write_text(
        "\n".join(samples_text_lines), encoding="utf-8")
    print(f"[done] saved {xling_dir/'steering_results.json'} and steering_samples.txt")


if __name__ == "__main__":
    main()
