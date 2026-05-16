"""SAE 评估脚本：从训练好的 ckpt 出发，在验证集上全面测一遍 SAE 效果。

指标分四类
----------
1. **重构**：MSE、explained variance、normalized MSE、cosine similarity。
2. **稀疏性**：L0、每特征激活频率、死特征数量。
3. **下游影响（KL/CE 替代损失，业界主流指标）**：把第 K 层残差流换成 SAE 重构，
   测下一 token 预测的 CE 与 logits KL；同时给"零消融"和"原始"两组基线。
4. **中英分项**（若 config 含 en/zh 双语本地路径）。

用法
----
    # 完整评估（需要能加载 Qwen3.5：Colab/Linux + 兼容 triton）
    python src/evaluate.py --exp-dir results/topk_l12_local_2

    # 仅基于 ckpt 与 history.csv 的本地评估（无需加载 LLM）
    python src/evaluate.py --exp-dir results/topk_l12_local_2 --ckpt-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml
from src.models.qwen_loader import load_hooked_qwen
from src.models.sae_topk import TopKSAE, TopKSAEConfig
from src.models.sae_jumprelu import JumpReLUSAE, JumpReLUSAEConfig
from src.data.corpora import LocalTextFileStream, LocalBilingualTextStream
from src.data.activation_store import ActivationStore, ActivationStoreConfig


# ---------- helpers ----------

def build_sae_from_cfg(cfg: dict) -> torch.nn.Module:
    sae_cfg = cfg["sae"]
    variant = sae_cfg.get("variant", "topk").lower()
    if variant == "topk":
        c = TopKSAEConfig(
            d_in=sae_cfg["d_in"],
            d_sae=sae_cfg["d_sae"],
            k=sae_cfg.get("k", 32),
            k_aux=sae_cfg.get("k_aux", 256),
            aux_loss_coef=sae_cfg.get("aux_loss_coef", 1.0 / 32),
            dead_steps_threshold=sae_cfg.get("dead_steps_threshold", 1000),
            normalize_decoder=sae_cfg.get("normalize_decoder", True),
        )
        return TopKSAE(c)
    if variant == "jumprelu":
        c = JumpReLUSAEConfig(
            d_in=sae_cfg["d_in"],
            d_sae=sae_cfg["d_sae"],
            sparsity_coef=sae_cfg.get("sparsity_coef", 1e-3),
            bandwidth=sae_cfg.get("bandwidth", 1e-3),
            init_threshold=sae_cfg.get("init_threshold", 0.001),
            normalize_decoder=sae_cfg.get("normalize_decoder", True),
        )
        return JumpReLUSAE(c)
    raise ValueError(f"未知 SAE 变体: {variant}")


def resolve_local_corpus_paths(cfg: dict) -> dict:
    """训练时配置可能写的是 Colab 路径（如 ../data/...）；本地评估时按相对项目根做兜底搜索。"""
    cp = cfg.get("corpus", {})
    candidates_dirs = [
        Path(""),  # 原路径直接试
        Path("data/local_corpus/fineweb_edu_subset"),
        Path("data/fineweb_edu_subset"),
    ]

    def find(orig: Optional[str]) -> Optional[str]:
        if not orig:
            return orig
        p = Path(orig)
        if p.exists():
            return str(p)
        name = p.name
        for d in candidates_dirs:
            cand = d / name
            if cand.exists():
                return str(cand)
        return orig  # 找不到，照原样返回，后续会报错

    fixed = dict(cp)
    for k in [
        "local_en_train_path",
        "local_zh_train_path",
        "local_en_val_path",
        "local_zh_val_path",
        "local_train_path",
        "local_val_path",
    ]:
        if k in fixed:
            fixed[k] = find(fixed.get(k))
    new_cfg = dict(cfg)
    new_cfg["corpus"] = fixed
    return new_cfg


def build_val_stream(cfg: dict):
    cp = cfg["corpus"]
    text_col = cp.get("local_text_col", "text")
    if cp.get("local_en_val_path") and cp.get("local_zh_val_path"):
        en = cp["local_en_val_path"]
        zh = cp["local_zh_val_path"]
        return {
            "mix": LocalBilingualTextStream(en, zh, en_ratio=cp.get("en_ratio", 0.5),
                                            seed=42, text_col=text_col),
            "en": LocalTextFileStream(en, text_col=text_col),
            "zh": LocalTextFileStream(zh, text_col=text_col),
        }
    if cp.get("local_val_path"):
        return {"mix": LocalTextFileStream(cp["local_val_path"], text_col=text_col)}
    raise ValueError("评估需要本地 val 路径（local_(en/zh)_val_path 或 local_val_path）。")


def make_store(hooked, text_stream, cfg, sae_batch_size=4096) -> ActivationStore:
    asc = cfg["activation_store"]
    s_cfg = ActivationStoreConfig(
        sae_batch_size=sae_batch_size,
        buffer_size_tokens=min(asc.get("buffer_size_tokens", 524288), 65536),
        refill_threshold=asc.get("refill_threshold", 0.5),
        ctx_len=asc.get("ctx_len", 512),
        model_batch_size=asc.get("model_batch_size", 8),
        device=str(next(hooked.model.parameters()).device),
    )
    return ActivationStore(hooked, text_stream, s_cfg)


# ---------- 阶段 1：纯激活上的重构指标 ----------

@torch.no_grad()
def eval_reconstruction(sae, store: ActivationStore, n_batches: int, label: str = "mix") -> dict:
    sae.eval()
    sums = {"mse_per_token": 0.0, "recon_loss_sum": 0.0, "expl_var": 0.0,
            "cosine": 0.0, "l0": 0.0, "var_x_sum": 0.0}
    feat_active_count = torch.zeros(sae.cfg.d_sae, device=next(sae.parameters()).device)
    feat_l0_per_batch_sum = 0.0
    n = 0
    total_tokens = 0
    for _ in tqdm(range(n_batches), desc=f"recon[{label}]", leave=False):
        x = next(store)
        x = x.to(next(sae.parameters()).device)
        out = sae(x)
        x_hat = out.x_hat
        diff = (x - x_hat).float()
        sums["mse_per_token"] += diff.pow(2).mean().item()
        sums["recon_loss_sum"] += diff.pow(2).sum(dim=-1).mean().item()
        var_x = x.float().var(dim=0).sum().clamp_min(1e-8)
        var_r = diff.var(dim=0).sum()
        sums["expl_var"] += (1 - var_r / var_x).item()
        sums["cosine"] += F.cosine_similarity(x.float(), x_hat.float(), dim=-1).mean().item()
        sums["l0"] += float(out.l0)
        sums["var_x_sum"] += float(var_x)
        # 特征激活计数
        z = out.z
        feat_active_count += (z != 0).any(dim=0).float() if False else (z != 0).float().sum(dim=0)
        feat_l0_per_batch_sum += (z != 0).any(dim=0).float().sum().item()
        total_tokens += x.shape[0]
        n += 1
    avg = {k: v / max(n, 1) for k, v in sums.items()}
    feat_density = (feat_active_count / max(total_tokens, 1)).cpu()
    dead_frac = float((feat_active_count == 0).float().mean())
    return {
        "label": label,
        "n_batches": n,
        "n_tokens": total_tokens,
        "mse_per_token": avg["mse_per_token"],
        "recon_loss_sum_dim": avg["recon_loss_sum"],
        "explained_variance": avg["expl_var"],
        "cosine_sim": avg["cosine"],
        "L0": avg["l0"],
        "var_x_total": avg["var_x_sum"],
        "dead_feature_frac": dead_frac,
        "n_active_unique_features": int((feat_active_count > 0).sum()),
        "feature_density_p50": float(feat_density.quantile(0.5)),
        "feature_density_p99": float(feat_density.quantile(0.99)),
        "feature_density_max": float(feat_density.max()),
        # 完整密度数组单独写文件
        "_feature_density": feat_density,
    }


# ---------- 阶段 2：替代损失（CE / KL） ----------

class SubstitutionHook:
    """注册到 decoder block 上：把残差流替换为 SAE 重构 / 零 / 不变。"""

    def __init__(self, sae, mode: str):
        assert mode in ("recon", "zero", "mean", "passthrough")
        self.sae = sae
        self.mode = mode
        self.handle = None

    def __call__(self, module, inputs, output):
        if isinstance(output, (tuple, list)):
            hs = output[0]
            rest = output[1:]
        else:
            hs, rest = output, None
        if self.mode == "passthrough":
            return None
        B, T, D = hs.shape
        with torch.no_grad():
            if self.mode == "recon":
                x = hs.reshape(-1, D).float()
                z = self.sae.encode(x)
                x_hat = self.sae.decode(z)
                hs_new = x_hat.to(hs.dtype).view(B, T, D)
            elif self.mode == "zero":
                hs_new = torch.zeros_like(hs)
            elif self.mode == "mean":
                m = hs.mean(dim=(0, 1), keepdim=True)
                hs_new = m.expand_as(hs)
            else:
                return None
        if rest is not None:
            return (hs_new,) + tuple(rest)
        return hs_new


@torch.no_grad()
def collect_text_batches(text_stream: Iterable[str], tokenizer, ctx_len: int,
                        batch_size: int, n_batches: int):
    """从文本流采固定的若干 batch，用于 CE/KL 评估（多模式之间使用同一批输入保证可比）。"""
    batches = []
    it = iter(text_stream)
    for _ in range(n_batches):
        texts = []
        while len(texts) < batch_size:
            try:
                t = next(it)
            except StopIteration:
                break
            if t and t.strip():
                texts.append(t)
        if not texts:
            break
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=ctx_len)
        batches.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    return batches


@torch.no_grad()
def _forward_logits(hooked, input_ids, attention_mask):
    """走完整 model forward，返回 logits（[B, T, V]）。需要 hook 在外部已注册。"""
    out = hooked.model(input_ids=input_ids.to(hooked.device),
                       attention_mask=attention_mask.to(hooked.device),
                       use_cache=False)
    # 兼容不同输出：CausalLMOutput.logits 或 dict
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict) and "logits" in out:
        return out["logits"]
    return out[0]


@torch.no_grad()
def _ce_from_logits(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[float, int]:
    """next-token CE on the masked tokens; returns (sum_loss, n_tokens)."""
    logits = logits[:, :-1, :].float()
    targets = input_ids[:, 1:].to(logits.device)
    mask = attention_mask[:, 1:].to(logits.device).bool()
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    nll = nll[mask]
    return float(nll.sum()), int(mask.sum())


@torch.no_grad()
def _kl_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor,
                    attention_mask: torch.Tensor) -> tuple[float, int]:
    logits_p = logits_p[:, :-1, :].float()
    logits_q = logits_q[:, :-1, :].float()
    mask = attention_mask[:, 1:].to(logits_p.device).bool()
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(-1)
    kl = kl[mask]
    return float(kl.sum()), int(mask.sum())


def eval_substitution(hooked, sae, text_stream, n_batches: int, model_batch_size: int,
                      ctx_len: int, label: str = "mix") -> dict:
    """三种 mode 跑同一批输入：original / recon / zero。

    内存策略：流式按 batch 处理。每个 batch 跑 3 次 forward（passthrough/recon/zero），
    在 GPU 上直接算 CE 和 KL（避免缓存整个 logits 张量——vocab=248K 时一个 batch 就 ~2GB）。
    """
    hooked.remove_hook()
    layer = hooked.text_module.layers[hooked.hook_layer]

    batches = collect_text_batches(text_stream, hooked.tokenizer, ctx_len,
                                   model_batch_size, n_batches)
    print(f"  [substitution/{label}] 收集 {len(batches)} 个 batch，文本已 tokenize。")

    def _run_mode_on_batch(mode: str, b: dict):
        sub = SubstitutionHook(sae, mode=mode)
        h = layer.register_forward_hook(sub)
        try:
            logits = _forward_logits(hooked, b["input_ids"], b["attention_mask"])
        finally:
            h.remove()
        return logits

    stats = {
        "passthrough": {"ce_sum": 0.0, "ce_n": 0, "kl_sum": 0.0, "kl_n": 0},
        "recon":       {"ce_sum": 0.0, "ce_n": 0, "kl_sum": 0.0, "kl_n": 0},
        "zero":        {"ce_sum": 0.0, "ce_n": 0, "kl_sum": 0.0, "kl_n": 0},
    }
    for b in tqdm(batches, desc=f"sub[{label}]", leave=False):
        logits_orig = _run_mode_on_batch("passthrough", b)
        ce_s, ce_t = _ce_from_logits(logits_orig, b["input_ids"], b["attention_mask"])
        stats["passthrough"]["ce_sum"] += ce_s
        stats["passthrough"]["ce_n"] += ce_t

        for mode in ["recon", "zero"]:
            logits = _run_mode_on_batch(mode, b)
            ce_s, ce_t = _ce_from_logits(logits, b["input_ids"], b["attention_mask"])
            stats[mode]["ce_sum"] += ce_s
            stats[mode]["ce_n"] += ce_t
            kl_s, kl_t = _kl_from_logits(logits_orig, logits, b["attention_mask"])
            stats[mode]["kl_sum"] += kl_s
            stats[mode]["kl_n"] += kl_t
            del logits
        del logits_orig
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = {}
    for mode, s in stats.items():
        results[mode] = {
            "ce_mean": s["ce_sum"] / max(s["ce_n"], 1),
            "ce_token_count": s["ce_n"],
            "kl_mean": s["kl_sum"] / max(s["kl_n"], 1) if s["kl_n"] else None,
        }

    # 把训练 hook 恢复回去（捕获模式，供后续如有需要）
    def _restore(_module, _inputs, output):
        hs = output[0] if isinstance(output, (tuple, list)) else output
        hooked._last_activation = hs.detach()
    hooked._handle = layer.register_forward_hook(_restore)

    delta_ce_recon = results["recon"]["ce_mean"] - results["passthrough"]["ce_mean"]
    delta_ce_zero = results["zero"]["ce_mean"] - results["passthrough"]["ce_mean"]
    ce_recovered = 1.0 - delta_ce_recon / max(delta_ce_zero, 1e-9)

    return {
        "label": label,
        "n_batches": len(batches),
        "n_eval_tokens": results["passthrough"]["ce_token_count"],
        "ce_original": results["passthrough"]["ce_mean"],
        "ce_with_sae": results["recon"]["ce_mean"],
        "ce_zero_ablation": results["zero"]["ce_mean"],
        "delta_ce_sae_vs_orig": delta_ce_recon,
        "delta_ce_zero_vs_orig": delta_ce_zero,
        "ce_loss_recovered_frac": ce_recovered,  # 越接近 1 越好
        "kl_sae_vs_orig": results["recon"]["kl_mean"],
        "kl_zero_vs_orig": results["zero"]["kl_mean"],
    }


# ---------- ckpt-only 模式（不加载 LLM） ----------

def run_ckpt_only_mode(exp_dir: Path, cfg: dict, ckpt_name: str, eval_dir: Path) -> None:
    """只看 SAE 权重统计 + history.csv 最终指标。本地无 GPU/无法加载 LLM 时用。"""
    import pandas as pd

    print(f"[eval] ckpt-only 模式：跳过 LLM 加载。exp_dir={exp_dir}")

    sae = build_sae_from_cfg(cfg).to("cpu").to(torch.float32)
    ckpt_path = exp_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sae.load_state_dict(ckpt["sae_state"])
    sae.eval()

    # ---- 权重内部统计 ----
    with torch.no_grad():
        W_enc = sae.W_enc.detach().float()
        W_dec = sae.W_dec.detach().float()
        b_enc = sae.b_enc.detach().float()
        b_dec = sae.b_dec.detach().float()

        dec_norms = W_dec.norm(dim=1)         # [d_sae]
        enc_norms = W_enc.norm(dim=0)         # [d_sae]
        # encoder-decoder 余弦（每个特征 enc 列与 dec 行）
        W_enc_T = W_enc.t()                   # [d_sae, d_in]
        cos_enc_dec = F.cosine_similarity(W_enc_T, W_dec, dim=-1)  # [d_sae]

        weight_stats = {
            "W_dec_row_norm_mean": float(dec_norms.mean()),
            "W_dec_row_norm_std": float(dec_norms.std()),
            "W_dec_row_norm_min": float(dec_norms.min()),
            "W_dec_row_norm_max": float(dec_norms.max()),
            "W_enc_col_norm_mean": float(enc_norms.mean()),
            "W_enc_col_norm_std": float(enc_norms.std()),
            "encoder_decoder_cosine_mean": float(cos_enc_dec.mean()),
            "encoder_decoder_cosine_p10": float(cos_enc_dec.quantile(0.1)),
            "encoder_decoder_cosine_p90": float(cos_enc_dec.quantile(0.9)),
            "b_enc_mean": float(b_enc.mean()),
            "b_enc_std": float(b_enc.std()),
            "b_dec_mean": float(b_dec.mean()),
            "b_dec_std": float(b_dec.std()),
            "n_params": int(sum(p.numel() for p in sae.parameters())),
        }

        # JumpReLU 阈值
        if hasattr(sae, "log_threshold"):
            theta = sae.log_threshold.detach().exp().float()
            weight_stats.update({
                "jumprelu_theta_mean": float(theta.mean()),
                "jumprelu_theta_std": float(theta.std()),
                "jumprelu_theta_p10": float(theta.quantile(0.1)),
                "jumprelu_theta_p90": float(theta.quantile(0.9)),
            })

        # dead-feature 计数器（如果存在）
        if hasattr(sae, "steps_since_active"):
            ssa = sae.steps_since_active.detach()
            dead_threshold = sae.cfg.dead_steps_threshold if hasattr(sae.cfg, "dead_steps_threshold") else 1000
            n_dead = int((ssa >= dead_threshold).sum())
            weight_stats.update({
                "dead_feature_count_in_ckpt": n_dead,
                "dead_feature_frac_in_ckpt": n_dead / sae.cfg.d_sae,
                "steps_since_active_max": int(ssa.max()),
                "steps_since_active_mean": float(ssa.float().mean()),
            })

    # ---- 训练历史最终指标 ----
    hist_path = exp_dir / "history.csv"
    hist_summary = {}
    last_row = {}
    best_row = {}
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        if len(hist) > 0:
            last = hist.iloc[-1]
            best_idx = hist["val_recon"].idxmin()
            best = hist.loc[best_idx]
            last_row = {
                "final_epoch": int(last["epoch"]),
                "final_lr": float(last["lr"]),
                "final_train_recon": float(last["train_recon"]),
                "final_val_recon": float(last["val_recon"]),
                "final_train_l0": float(last["train_l0"]),
                "final_val_l0": float(last["val_l0"]),
                "final_train_expl_var": float(last["train_expl_var"]),
                "final_val_expl_var": float(last["val_expl_var"]),
                "final_train_dead_frac": float(last.get("train_dead_frac", float("nan"))),
            }
            best_row = {
                "best_epoch": int(best["epoch"]),
                "best_val_recon": float(best["val_recon"]),
                "best_val_l0": float(best["val_l0"]),
                "best_val_expl_var": float(best["val_expl_var"]),
            }
            # 改善幅度
            init = hist.iloc[0]
            hist_summary = {
                "init_val_recon": float(init["val_recon"]),
                "init_val_expl_var": float(init["val_expl_var"]),
                "val_recon_drop_total": float(init["val_recon"] - last["val_recon"]),
                "val_recon_drop_frac": float((init["val_recon"] - last["val_recon"]) / init["val_recon"]),
                "val_expl_var_gain": float(last["val_expl_var"] - init["val_expl_var"]),
                "n_epochs_logged": int(len(hist)),
                "total_train_seconds": float(hist["elapsed_s"].sum()),
            }

    # ---- 写文件 ----
    out = {
        "exp_dir": str(exp_dir).replace("\\", "/"),
        "mode": "ckpt_only",
        "ckpt": ckpt_name,
        "ckpt_epoch": ckpt.get("trainer_state", {}).get("epoch"),
        "ckpt_best_val_loss": ckpt.get("trainer_state", {}).get("best_val_loss"),
        "config_summary": {
            "variant": cfg["sae"]["variant"],
            "d_in": cfg["sae"]["d_in"],
            "d_sae": cfg["sae"]["d_sae"],
            "k": cfg["sae"].get("k"),
            "hook_layer": cfg["model"]["hook_layer"],
        },
        "history_summary": hist_summary,
        "final_metrics": last_row,
        "best_metrics": best_row,
        "weight_stats": weight_stats,
    }
    (eval_dir / "eval_metrics_ckpt_only.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # 可读 summary
    lines = []
    lines.append(f"实验目录: {exp_dir}")
    lines.append(f"评估模式: ckpt-only（未加载 LLM；完整 KL/CE 替代损失需在能跑 Qwen3.5 的环境运行）")
    lines.append(f"权重: {ckpt_name}（来自 epoch={out['ckpt_epoch']}）")
    cs = out["config_summary"]
    lines.append(f"SAE: variant={cs['variant']} d_in={cs['d_in']} d_sae={cs['d_sae']} "
                 f"k={cs['k']} hook_layer={cs['hook_layer']}")
    lines.append(f"参数量: {weight_stats['n_params']/1e6:.2f} M")
    if hist_summary:
        lines.append("")
        lines.append("[训练历史汇总]")
        lines.append(f"  共 {hist_summary['n_epochs_logged']} epoch，累计训练时长 "
                     f"{hist_summary['total_train_seconds']/3600:.2f} h")
        lines.append(f"  初始 val_recon = {hist_summary['init_val_recon']:.4f}  "
                     f"→ 终末 val_recon = {last_row['final_val_recon']:.4f}  "
                     f"(↓ {hist_summary['val_recon_drop_frac']*100:.1f}%)")
        lines.append(f"  初始 val_expl_var = {hist_summary['init_val_expl_var']:.4f}  "
                     f"→ 终末 val_expl_var = {last_row['final_val_expl_var']:.4f}  "
                     f"(+{hist_summary['val_expl_var_gain']:.4f})")
        lines.append(f"  最佳 val_recon = {best_row['best_val_recon']:.4f} (epoch {best_row['best_epoch']})")
        lines.append(f"  终末 L0 = {last_row['final_val_l0']:.1f}（目标 k={cs['k']}）")
        lines.append(f"  终末 train_dead_frac = {last_row['final_train_dead_frac']:.4%}")
        lines.append(f"  终末 lr = {last_row['final_lr']:.2e}")
    lines.append("")
    lines.append("[SAE 权重统计]")
    ws = weight_stats
    lines.append(f"  W_dec 行范数: mean={ws['W_dec_row_norm_mean']:.4f} std={ws['W_dec_row_norm_std']:.4f} "
                 f"min={ws['W_dec_row_norm_min']:.4f} max={ws['W_dec_row_norm_max']:.4f}")
    lines.append(f"    （配置要求 normalize_decoder=True：上述应≈1.0）")
    lines.append(f"  W_enc 列范数: mean={ws['W_enc_col_norm_mean']:.4f} std={ws['W_enc_col_norm_std']:.4f}")
    lines.append(f"  encoder/decoder 行向余弦相似度: mean={ws['encoder_decoder_cosine_mean']:.4f} "
                 f"p10={ws['encoder_decoder_cosine_p10']:.4f} p90={ws['encoder_decoder_cosine_p90']:.4f}")
    lines.append(f"    （SAE 收敛良好时 encoder/decoder 通常会形成正相关字典对）")
    lines.append(f"  b_dec: mean={ws['b_dec_mean']:.4f} std={ws['b_dec_std']:.4f}")
    lines.append(f"  b_enc: mean={ws['b_enc_mean']:.4f} std={ws['b_enc_std']:.4f}")
    if "dead_feature_count_in_ckpt" in ws:
        lines.append(f"  ckpt 内 dead feature 数: {ws['dead_feature_count_in_ckpt']}/{cs['d_sae']} "
                     f"({ws['dead_feature_frac_in_ckpt']*100:.2f}%)")
        lines.append(f"  steps_since_active: mean={ws['steps_since_active_mean']:.1f} "
                     f"max={ws['steps_since_active_max']}")
    lines.append("")
    lines.append("[如何跑完整 KL/CE 替代损失评估]")
    lines.append("  本机加载 Qwen3.5-0.8B 报 fla/triton 错误（fla 的 @triton.autotune 在 Windows + Triton 3.0 下不兼容）。")
    lines.append("  在 Colab/Linux GPU 环境复现训练时同样的依赖后运行：")
    lines.append("    python src/evaluate.py --exp-dir results/topk_l12_local_2 \\")
    lines.append("        --ckpt best.pt --n-batches 100 --n-sub-batches 20")
    txt = "\n".join(lines)
    (eval_dir / "eval_summary.txt").write_text(txt, encoding="utf-8")

    print("\n" + txt)
    print(f"\n[eval] 已写入：{eval_dir / 'eval_metrics_ckpt_only.json'}")
    print(f"[eval] 已写入：{eval_dir / 'eval_summary.txt'}")


# ---------- 主入口 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True, type=str)
    ap.add_argument("--ckpt", default="best.pt", type=str, help="best.pt 或 last.pt")
    ap.add_argument("--n-batches", type=int, default=100, help="重构指标用多少 SAE batch")
    ap.add_argument("--n-sub-batches", type=int, default=20, help="替代损失用多少 model batch")
    ap.add_argument("--sub-batch-size", type=int, default=4,
                    help="替代损失评估时的 model_batch_size（小一点省显存）")
    ap.add_argument("--sae-batch-size", type=int, default=2048,
                    help="重构指标评估的 sae_batch_size")
    ap.add_argument("--device", default="auto", type=str)
    ap.add_argument("--skip-substitution", action="store_true")
    ap.add_argument("--skip-per-language", action="store_true")
    ap.add_argument("--ckpt-only", action="store_true",
                    help="不加载 LLM，只对 SAE 权重做内部统计 + 拉取 history.csv 最终指标。"
                         "用于 LLM 无法在本地加载的环境（如 Windows + 不兼容的 triton）。")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    cfg = load_yaml(exp_dir / "config.yaml")
    cfg = resolve_local_corpus_paths(cfg)

    eval_dir = exp_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if args.ckpt_only:
        run_ckpt_only_mode(exp_dir, cfg, args.ckpt, eval_dir)
        return

    # 设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[eval] device={device}  exp_dir={exp_dir}  ckpt={args.ckpt}")

    # Qwen + hook
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    hooked = load_hooked_qwen(
        model_dir=cfg["model"]["model_dir"],
        hook_layer=cfg["model"]["hook_layer"],
        device=device,
        dtype=dtype_map.get(cfg["model"].get("dtype", "bfloat16"), torch.bfloat16),
    )
    print(f"[eval] Qwen 加载完毕，hidden_size={hooked.hidden_size} hook_layer={hooked.hook_layer}")

    # SAE
    sae = build_sae_from_cfg(cfg).to(device).to(torch.float32)
    ckpt_path = exp_dir / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sae.load_state_dict(ckpt["sae_state"])
    sae.eval()
    print(f"[eval] SAE 加载完毕：{ckpt_path.name}（来自 epoch={ckpt.get('trainer_state', {}).get('epoch', '?')}）")

    # 文本流
    streams = build_val_stream(cfg)

    # ====== 阶段 1：重构指标（mix / en / zh）======
    recon_results = {}
    labels = ["mix"]
    if not args.skip_per_language:
        labels += [k for k in ["en", "zh"] if k in streams]
    for label in labels:
        store = make_store(hooked, streams[label], cfg, sae_batch_size=args.sae_batch_size)
        recon_results[label] = eval_reconstruction(sae, store, args.n_batches, label=label)

    # 保存特征密度并从 dict 中剥出去（不放进 json）
    densities = {}
    for label, r in recon_results.items():
        densities[label] = r.pop("_feature_density").numpy().tolist()
    (eval_dir / "feature_density.json").write_text(
        json.dumps({k: v for k, v in densities.items()}, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== 重构指标 ==========")
    for label, r in recon_results.items():
        print(f"[{label}] explained_var={r['explained_variance']:.4f}  "
              f"cosine={r['cosine_sim']:.4f}  L0={r['L0']:.1f}  "
              f"dead_feat_frac={r['dead_feature_frac']:.4f}  "
              f"unique_active={r['n_active_unique_features']}/{cfg['sae']['d_sae']}")

    # ====== 阶段 2：替代损失（mix）======
    sub_results = {}
    if not args.skip_substitution:
        # 给替代损失重新拿个流（之前那个被消费了一部分）
        cp = cfg["corpus"]
        text_col = cp.get("local_text_col", "text")
        if cp.get("local_en_val_path") and cp.get("local_zh_val_path"):
            sub_stream_mix = LocalBilingualTextStream(
                cp["local_en_val_path"], cp["local_zh_val_path"],
                en_ratio=cp.get("en_ratio", 0.5), seed=12345, text_col=text_col,
            )
        else:
            sub_stream_mix = LocalTextFileStream(cp["local_val_path"], text_col=text_col)
        sub_results["mix"] = eval_substitution(
            hooked, sae, sub_stream_mix,
            n_batches=args.n_sub_batches,
            model_batch_size=args.sub_batch_size,
            ctx_len=cfg["activation_store"].get("ctx_len", 512),
            label="mix",
        )

        print("\n========== 替代损失（mix） ==========")
        r = sub_results["mix"]
        print(f"  ce_original         = {r['ce_original']:.4f}")
        print(f"  ce_with_sae         = {r['ce_with_sae']:.4f}")
        print(f"  ce_zero_ablation    = {r['ce_zero_ablation']:.4f}")
        print(f"  ΔCE(sae - orig)     = {r['delta_ce_sae_vs_orig']:+.4f}")
        print(f"  ΔCE(zero - orig)    = {r['delta_ce_zero_vs_orig']:+.4f}")
        print(f"  CE 恢复率           = {r['ce_loss_recovered_frac']*100:.2f}%   (越接近 100% 越好)")
        print(f"  KL(sae || orig)     = {r['kl_sae_vs_orig']:.4f}")
        print(f"  KL(zero || orig)    = {r['kl_zero_vs_orig']:.4f}")

    # 保存
    out = {
        "exp_dir": str(exp_dir).replace("\\", "/"),
        "ckpt": args.ckpt,
        "config_summary": {
            "variant": cfg["sae"]["variant"],
            "d_in": cfg["sae"]["d_in"],
            "d_sae": cfg["sae"]["d_sae"],
            "k": cfg["sae"].get("k"),
            "hook_layer": cfg["model"]["hook_layer"],
            "epoch_in_ckpt": ckpt.get("trainer_state", {}).get("epoch"),
            "best_val_loss_in_ckpt": ckpt.get("trainer_state", {}).get("best_val_loss"),
        },
        "reconstruction": recon_results,
        "substitution": sub_results,
        "eval_args": vars(args),
    }
    (eval_dir / "eval_metrics.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 人类可读摘要
    lines = []
    lines.append(f"实验目录: {exp_dir}")
    lines.append(f"权重: {args.ckpt}   (来自 epoch={out['config_summary']['epoch_in_ckpt']})")
    lines.append(f"SAE: variant={out['config_summary']['variant']} "
                 f"d_in={out['config_summary']['d_in']} d_sae={out['config_summary']['d_sae']} "
                 f"k={out['config_summary']['k']} hook_layer={out['config_summary']['hook_layer']}")
    lines.append("")
    lines.append("重构指标:")
    for label, r in recon_results.items():
        lines.append(f"  [{label}] explained_var={r['explained_variance']:.4f} "
                     f"cosine={r['cosine_sim']:.4f} L0={r['L0']:.1f} "
                     f"dead={r['dead_feature_frac']:.4f} "
                     f"unique_active={r['n_active_unique_features']}/{cfg['sae']['d_sae']}")
    if sub_results:
        lines.append("")
        lines.append("替代损失（CE / KL）：")
        for label, r in sub_results.items():
            lines.append(f"  [{label}] CE  orig={r['ce_original']:.4f}  "
                         f"sae={r['ce_with_sae']:.4f}  zero={r['ce_zero_ablation']:.4f}")
            lines.append(f"          ΔCE  +{r['delta_ce_sae_vs_orig']:.4f} (sae)  "
                         f"+{r['delta_ce_zero_vs_orig']:.4f} (zero)")
            lines.append(f"          CE 恢复率 = {r['ce_loss_recovered_frac']*100:.2f}%")
            lines.append(f"          KL  sae={r['kl_sae_vs_orig']:.4f}  zero={r['kl_zero_vs_orig']:.4f}")
    (eval_dir / "eval_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[eval] 已写入：{eval_dir / 'eval_metrics.json'}")
    print(f"[eval] 已写入：{eval_dir / 'eval_summary.txt'}")
    print(f"[eval] 已写入：{eval_dir / 'feature_density.json'}")


if __name__ == "__main__":
    main()
