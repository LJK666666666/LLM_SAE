"""SAE 训练入口。

约定执行路径为项目根目录，例：

    # 烟雾测试（单 batch 跑通整条链路）
    python src/train.py --config configs/train_topk_smoke.yaml --tag smoke --max-iters 1

    # TopK 正式训练
    python src/train.py --config configs/train_topk.yaml --tag topk_l12 --d-sae 16384 --k 32

    # TopK 本地固定子集训练
    python src/train.py --config data/local_corpus/fineweb_edu_subset/train_config_local.yaml --tag topk_l12_local

    # 从 last.pt 恢复
    python src/train.py --config configs/train_topk.yaml --tag topk_l12 --resume

    # JumpReLU
    python src/train.py --config configs/train_jumprelu.yaml --tag jumprelu_l12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# 让 `python src/train.py` 在项目根执行时能 import src.*
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml, merge_cli_overrides, dump_yaml
from src.utils.exp_dir import make_or_resume_exp_dir
from src.utils.overall_metrics import append_record
from src.models.qwen_loader import load_hooked_qwen
from src.models.sae_topk import TopKSAE, TopKSAEConfig
from src.models.sae_jumprelu import JumpReLUSAE, JumpReLUSAEConfig
from src.data.corpora import (
    CorpusConfig,
    build_text_stream,
    WikitextStream,
    LocalTextFileStream,
    LocalBilingualTextStream,
)
from src.data.activation_store import ActivationStore, ActivationStoreConfig
from src.training.trainer import SAETrainer, TrainerConfig


# ---------- 构建 ----------

def build_sae(cfg: dict) -> torch.nn.Module:
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
    elif variant == "jumprelu":
        c = JumpReLUSAEConfig(
            d_in=sae_cfg["d_in"],
            d_sae=sae_cfg["d_sae"],
            sparsity_coef=sae_cfg.get("sparsity_coef", 1e-3),
            bandwidth=sae_cfg.get("bandwidth", 1e-3),
            init_threshold=sae_cfg.get("init_threshold", 0.001),
            normalize_decoder=sae_cfg.get("normalize_decoder", True),
        )
        return JumpReLUSAE(c)
    else:
        raise ValueError(f"未知 SAE 变体: {variant}")


def build_text_streams(cfg: dict) -> tuple[object, object]:
    """返回 (train_stream, val_stream)。简化处理：用同一个流但不同 seed。"""
    corp_cfg = cfg["corpus"]
    if corp_cfg.get("use_local_txt"):
        # 完全离线烟雾测试：纯 .txt，跳过 datasets 库依赖
        path = corp_cfg["use_local_txt"]
        return LocalTextFileStream(path), LocalTextFileStream(path)
    if corp_cfg.get("local_train_path"):
        text_col = corp_cfg.get("local_text_col", "text")
        train_path = corp_cfg["local_train_path"]
        val_path = corp_cfg.get("local_val_path", train_path)
        return LocalTextFileStream(train_path, text_col=text_col), \
               LocalTextFileStream(val_path, text_col=text_col)
    if corp_cfg.get("local_en_train_path") and corp_cfg.get("local_zh_train_path"):
        text_col = corp_cfg.get("local_text_col", "text")
        en_ratio = corp_cfg.get("en_ratio", 0.5)
        seed = cfg.get("trainer", {}).get("seed", 0)
        return (
            LocalBilingualTextStream(
                corp_cfg["local_en_train_path"],
                corp_cfg["local_zh_train_path"],
                en_ratio=en_ratio,
                seed=seed,
                text_col=text_col,
            ),
            LocalBilingualTextStream(
                corp_cfg.get("local_en_val_path", corp_cfg["local_en_train_path"]),
                corp_cfg.get("local_zh_val_path", corp_cfg["local_zh_train_path"]),
                en_ratio=en_ratio,
                seed=seed + 12345,
                text_col=text_col,
            ),
        )
    if corp_cfg.get("use_wikitext_only", False):
        return WikitextStream(cache_dir=corp_cfg.get("cache_dir", "data/cache")), \
               WikitextStream(cache_dir=corp_cfg.get("cache_dir", "data/cache"))
    base_kwargs = {k: v for k, v in corp_cfg.items()
                   if k not in (
                       "use_wikitext_only",
                       "use_local_txt",
                       "local_train_path",
                       "local_val_path",
                       "local_en_train_path",
                       "local_zh_train_path",
                       "local_en_val_path",
                       "local_zh_val_path",
                       "local_text_col",
                   )}
    train_cfg = CorpusConfig(**{**base_kwargs, "seed": cfg.get("trainer", {}).get("seed", 0)})
    val_cfg = CorpusConfig(**{**base_kwargs, "seed": cfg.get("trainer", {}).get("seed", 0) + 12345})
    return build_text_stream(train_cfg), build_text_stream(val_cfg)


def build_stores(cfg: dict, hooked, train_text, val_text):
    asc = cfg["activation_store"]
    device = next(hooked.model.parameters()).device
    s_cfg = ActivationStoreConfig(
        sae_batch_size=asc["sae_batch_size"],
        buffer_size_tokens=asc["buffer_size_tokens"],
        refill_threshold=asc.get("refill_threshold", 0.5),
        ctx_len=asc.get("ctx_len", 512),
        model_batch_size=asc.get("model_batch_size", 8),
        device=str(device),
    )
    return ActivationStore(hooked, train_text, s_cfg), ActivationStore(hooked, val_text, s_cfg)


# ---------- main ----------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str, help="yaml 配置文件路径")
    p.add_argument("--tag", required=True, type=str, help="实验 tag，决定 {results_root}/{tag}_{n}/ 编号")
    p.add_argument("--results-root", type=str, default="results",
                   help="训练结果根目录，默认 results；Colab 可用 ../drive/MyDrive/results")
    p.add_argument("--resume", action="store_true", help="从同 tag 最大编号目录的 last.pt 恢复")
    p.add_argument("--max-iters", type=int, default=None,
                   help="若 >0：跳过完整训练，只跑这么多 batch 用于快速验证")
    # 高频可覆盖参数（CLAUDE.md 规则 17）
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--d-sae", dest="sae.d_sae", type=int, default=None)
    p.add_argument("--k", dest="sae.k", type=int, default=None)
    p.add_argument("--steps-per-epoch", dest="trainer.steps_per_epoch", type=int, default=None)
    p.add_argument("--max-epochs", dest="trainer.max_epochs", type=int, default=None)
    p.add_argument("--save-every-steps", dest="trainer.save_every_steps", type=int, default=None)
    p.add_argument("--device", dest="model.device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    raw_cfg = load_yaml(args.config)
    overrides = vars(args).copy()
    # 把 trainer.lr 用 --lr 顶上去
    if overrides.get("lr") is not None:
        overrides["trainer.lr"] = overrides.pop("lr")
    # 删除非配置覆盖项
    for k in ["config", "tag", "results_root", "resume", "max_iters"]:
        overrides.pop(k, None)
    cfg = merge_cli_overrides(raw_cfg, overrides)

    # 实验目录
    exp_dir = make_or_resume_exp_dir(args.tag, resume=args.resume, results_root=args.results_root)
    print(f"[main] 实验目录: {exp_dir}")
    dump_yaml(cfg, exp_dir / "config.yaml")
    (exp_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    # 设备
    requested_device = cfg["model"].get("device", "auto")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    print(f"[main] device={device}")

    # 模型 + hook
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    hooked = load_hooked_qwen(
        model_dir=cfg["model"]["model_dir"],
        hook_layer=cfg["model"]["hook_layer"],
        device=device,
        dtype=dtype_map.get(cfg["model"].get("dtype", "bfloat16"), torch.bfloat16),
    )
    print(f"[main] Qwen3.5 加载完成，hidden_size={hooked.hidden_size} hook_layer={hooked.hook_layer}")

    # 数据
    train_text, val_text = build_text_streams(cfg)
    train_store, val_store = build_stores(cfg, hooked, train_text, val_text)

    # SAE
    sae = build_sae(cfg).to(device).to(torch.float32)  # SAE 用 fp32 训练
    n_params = sum(p.numel() for p in sae.parameters())
    print(f"[main] SAE 参数量: {n_params/1e6:.2f}M  variant={cfg['sae']['variant']} d_sae={cfg['sae']['d_sae']}")

    # Trainer
    tr_cfg_d = cfg["trainer"]
    trainer = SAETrainer(
        sae=sae,
        train_store=train_store,
        val_store=val_store,
        cfg=TrainerConfig(
            lr=tr_cfg_d.get("lr", 3e-4),
            weight_decay=tr_cfg_d.get("weight_decay", 0.0),
            betas=tuple(tr_cfg_d.get("betas", (0.9, 0.999))),
            grad_clip=tr_cfg_d.get("grad_clip", 1.0),
            steps_per_epoch=tr_cfg_d.get("steps_per_epoch", 1000),
            val_steps=tr_cfg_d.get("val_steps", 50),
            max_epochs=tr_cfg_d.get("max_epochs", 200),
            lr_decay_factor=tr_cfg_d.get("lr_decay_factor", 0.7),
            lr_patience_epochs=tr_cfg_d.get("lr_patience_epochs", 3),
            early_stop_lr_patience_epochs=tr_cfg_d.get("early_stop_lr_patience_epochs", 15),
            seed=tr_cfg_d.get("seed", 0),
            grad_accum=tr_cfg_d.get("grad_accum", 1),
            save_every_steps=tr_cfg_d.get("save_every_steps", 0),
        ),
        exp_dir=exp_dir,
        device=device,
        full_config_snapshot=cfg,
    )

    if args.resume:
        trainer.load_resume()

    # 跑！
    t0 = time.time()
    summary = trainer.fit(max_iters_override=args.max_iters)
    elapsed = time.time() - t0

    # 汇总指标到 results/overall_*
    record = {
        "exp_dir": str(exp_dir).replace("\\", "/"),
        "results_root": str(Path(args.results_root)).replace("\\", "/"),
        "tag": args.tag,
        "variant": cfg["sae"]["variant"],
        "hook_layer": cfg["model"]["hook_layer"],
        "d_in": cfg["sae"]["d_in"],
        "d_sae": cfg["sae"]["d_sae"],
        "k": cfg["sae"].get("k"),
        "lr": cfg["trainer"]["lr"],
        "steps_per_epoch": cfg["trainer"]["steps_per_epoch"],
        "n_params_sae_M": round(n_params / 1e6, 3),
        "elapsed_s": round(elapsed, 1),
        **{f"summary.{k}": v for k, v in summary.items()},
    }
    if args.max_iters is None:
        # 完整训练才写最终指标
        try:
            with (exp_dir / "history.json").open(encoding="utf-8") as f:
                hist = json.load(f)
            if hist:
                last = hist[-1]
                for key in ["val_recon", "val_l0", "val_expl_var", "lr"]:
                    if key in last:
                        record[f"final.{key}"] = last[key]
        except Exception:
            pass
    append_record(record, results_root=args.results_root)
    (exp_dir / "metrics_final.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[main] 完成。耗时 {elapsed:.1f}s。汇总已写入 "
          f"{Path(args.results_root) / 'overall_config_metrics.{csv,json}'}")


if __name__ == "__main__":
    main()
