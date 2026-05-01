"""SAE 训练循环：epoch 制 + 每 epoch 末保存 best/last + 支持 --resume + tqdm 进度条。

epoch 定义
----------
SAE 训练通常以 token 数计量（如 1B token），本项目将"每 epoch"折算为 ``steps_per_epoch``
个 SAE batch。以此映射 CLAUDE.md 的"3 epoch lr 不降则衰减、15 epoch 不降则早停"。

每 epoch 流程
-------------
1. train: 跑 ``steps_per_epoch`` 个 batch，平均损失 / L0 / dead_frac / explained_var
2. val: 跑 ``val_steps`` 个 batch（不更新参数，但仍走 ActivationStore 流式 refill）
3. 学习率调度（基于 val_recon_loss）
4. 保存 last.pt；若 val_recon_loss 创新低则保存 best.pt
5. 追加 history.csv / .json
6. 早停判断
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm
import pandas as pd


@dataclass
class TrainerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0
    steps_per_epoch: int = 1000
    val_steps: int = 50
    max_epochs: int = 200
    lr_decay_factor: float = 0.7
    lr_patience_epochs: int = 3
    early_stop_lr_patience_epochs: int = 15
    seed: int = 0
    grad_accum: int = 1
    save_every_steps: int = 0


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    epochs_since_val_improve: int = 0
    epochs_since_lr_drop: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


class SAETrainer:
    """通用 SAE trainer，TopK / JumpReLU 都能用（接口统一：forward 返回 ``out.loss``）。"""

    def __init__(
        self,
        sae: nn.Module,
        train_store: Iterable[torch.Tensor],
        val_store: Iterable[torch.Tensor],
        cfg: TrainerConfig,
        exp_dir: Path,
        device: torch.device,
        full_config_snapshot: Optional[dict[str, Any]] = None,
    ):
        self.sae = sae.to(device)
        self.train_store = iter(train_store)
        self.val_store = iter(val_store)
        self.cfg = cfg
        self.exp_dir = Path(exp_dir)
        self.device = device
        self.full_cfg_snapshot = full_config_snapshot or {}
        self.optimizer = AdamW(
            sae.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )
        self.state = TrainState()

        self._last_path = self.exp_dir / "last.pt"
        self._best_path = self.exp_dir / "best.pt"
        self._history_csv = self.exp_dir / "history.csv"
        self._history_json = self.exp_dir / "history.json"

    # ---------- ckpt ----------
    def _save_ckpt(self, path: Path) -> None:
        torch.save(
            {
                "sae_state": self.sae.state_dict(),
                "sae_cfg": getattr(self.sae, "cfg", None).__dict__ if getattr(self.sae, "cfg", None) else {},
                "optim_state": self.optimizer.state_dict(),
                "trainer_state": asdict(self.state),
                "full_cfg": self.full_cfg_snapshot,
            },
            path,
        )

    def _save_history(self) -> None:
        df = pd.DataFrame(self.state.history)
        df.to_csv(self._history_csv, index=False)
        self._history_json.write_text(
            json.dumps(self.state.history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_resume(self, path: Optional[Path] = None) -> None:
        path = path or self._last_path
        if not path.exists():
            raise FileNotFoundError(f"恢复失败，找不到 {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.sae.load_state_dict(ckpt["sae_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        ts = ckpt["trainer_state"]
        self.state = TrainState(
            epoch=ts["epoch"],
            global_step=ts["global_step"],
            best_val_loss=ts["best_val_loss"],
            epochs_since_val_improve=ts["epochs_since_val_improve"],
            epochs_since_lr_drop=ts["epochs_since_lr_drop"],
            history=ts.get("history", []),
        )
        # 同步 lr：ckpt optimizer 已带 lr，无需额外操作
        print(f"[trainer] 已从 {path} 恢复：epoch={self.state.epoch} step={self.state.global_step} "
              f"best_val_loss={self.state.best_val_loss:.6f}")

    # ---------- core ----------
    def _step(self, x: torch.Tensor) -> dict[str, float]:
        x = x.to(self.device, non_blocking=True)
        out = self.sae(x)
        loss = out.loss / self.cfg.grad_accum
        loss.backward()
        return {
            "loss": float(out.loss.detach()),
            "recon": float(out.recon_loss.detach()),
            "l0": float(out.l0.detach()),
            "dead_frac": float(out.dead_frac.detach()),
            "expl_var": float(out.explained_variance.detach()),
        }

    def _train_epoch(self) -> dict[str, float]:
        self.sae.train()
        sums = {"loss": 0.0, "recon": 0.0, "l0": 0.0, "dead_frac": 0.0, "expl_var": 0.0}
        n = 0
        pbar = tqdm(range(self.cfg.steps_per_epoch),
                    desc=f"Epoch {self.state.epoch+1} train", leave=False)
        for i in pbar:
            x = next(self.train_store)
            metrics = self._step(x)
            for k, v in metrics.items():
                sums[k] += v
            n += 1
            if (i + 1) % self.cfg.grad_accum == 0:
                # 移除 W_dec 沿径向的梯度分量
                if hasattr(self.sae, "remove_parallel_grad_component_"):
                    self.sae.remove_parallel_grad_component_()
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.sae.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if hasattr(self.sae, "_normalize_decoder_") and getattr(self.sae.cfg, "normalize_decoder", False):
                    self.sae._normalize_decoder_()
                self.state.global_step += 1
                if self.cfg.save_every_steps > 0 and self.state.global_step % self.cfg.save_every_steps == 0:
                    self._save_ckpt(self._last_path)
            if i % 20 == 0:
                pbar.set_postfix(loss=f"{metrics['loss']:.4f}",
                                 l0=f"{metrics['l0']:.1f}",
                                 dead=f"{metrics['dead_frac']:.2%}")
        return {k: v / max(n, 1) for k, v in sums.items()}

    @torch.no_grad()
    def _val_epoch(self) -> dict[str, float]:
        self.sae.eval()
        sums = {"loss": 0.0, "recon": 0.0, "l0": 0.0, "expl_var": 0.0}
        n = 0
        for _ in tqdm(range(self.cfg.val_steps), desc="val", leave=False):
            x = next(self.val_store).to(self.device, non_blocking=True)
            out = self.sae(x)
            sums["loss"] += float(out.loss)
            sums["recon"] += float(out.recon_loss)
            sums["l0"] += float(out.l0)
            sums["expl_var"] += float(out.explained_variance)
            n += 1
        return {k: v / max(n, 1) for k, v in sums.items()}

    def _adjust_lr(self, val_recon: float) -> None:
        improved = val_recon < self.state.best_val_loss - 1e-6
        if improved:
            self.state.best_val_loss = val_recon
            self.state.epochs_since_val_improve = 0
            self.state.epochs_since_lr_drop = 0
        else:
            self.state.epochs_since_val_improve += 1
            self.state.epochs_since_lr_drop += 1
            if self.state.epochs_since_val_improve >= self.cfg.lr_patience_epochs:
                for pg in self.optimizer.param_groups:
                    pg["lr"] *= self.cfg.lr_decay_factor
                self.state.epochs_since_val_improve = 0
                print(f"[trainer] val 未改善 {self.cfg.lr_patience_epochs} epoch，"
                      f"lr → {self.optimizer.param_groups[0]['lr']:.3e}")

    def _early_stop(self) -> bool:
        return self.state.epochs_since_lr_drop >= self.cfg.early_stop_lr_patience_epochs

    # ---------- public ----------
    def fit(self, max_iters_override: Optional[int] = None) -> dict[str, Any]:
        """主训练入口。``max_iters_override``：若给出（如 1），则只跑这么多 batch 就退出，
        用于 CLAUDE.md 要求的快速验证。"""
        if max_iters_override is not None and max_iters_override > 0:
            print(f"[trainer] --max-iters {max_iters_override}：单 batch 快速验证模式。")
            self.sae.train()
            for i in range(max_iters_override):
                x = next(self.train_store).to(self.device)
                out = self.sae(x)
                out.loss.backward()
                if hasattr(self.sae, "remove_parallel_grad_component_"):
                    self.sae.remove_parallel_grad_component_()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if hasattr(self.sae, "_normalize_decoder_") and getattr(self.sae.cfg, "normalize_decoder", False):
                    self.sae._normalize_decoder_()
                print(f"  iter {i+1}: loss={float(out.loss):.4f} recon={float(out.recon_loss):.4f} "
                      f"l0={float(out.l0):.1f} dead={float(out.dead_frac):.2%} "
                      f"expl_var={float(out.explained_variance):.4f}")
            self._save_ckpt(self._last_path)
            print(f"[trainer] 已保存验证用 ckpt: {self._last_path}")
            return {"mode": "max_iters_smoke_test", "iters": max_iters_override}

        # state.epoch 含义："已完成的 epoch 数"。resume 时下一个要跑的 epoch index = state.epoch。
        start_epoch = self.state.epoch
        for epoch in range(start_epoch, self.cfg.max_epochs):
            t0 = time.time()
            train_m = self._train_epoch()
            self.state.epoch = epoch + 1  # 已完成本轮训练；验证失败时也能从下一轮继续
            if self.cfg.save_every_steps > 0:
                self._save_ckpt(self._last_path)
            val_m = self._val_epoch()
            dt = time.time() - t0

            cur_lr = self.optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch + 1,
                "global_step": self.state.global_step,
                "lr": cur_lr,
                "train_loss": train_m["loss"],
                "train_recon": train_m["recon"],
                "train_l0": train_m["l0"],
                "train_dead_frac": train_m["dead_frac"],
                "train_expl_var": train_m["expl_var"],
                "val_loss": val_m["loss"],
                "val_recon": val_m["recon"],
                "val_l0": val_m["l0"],
                "val_expl_var": val_m["expl_var"],
                "elapsed_s": dt,
            }
            self.state.history.append(row)
            self._save_history()

            print(f"[epoch {epoch+1:3d}] lr={cur_lr:.2e} "
                  f"train_loss={train_m['loss']:.4f} val_recon={val_m['recon']:.4f} "
                  f"l0={val_m['l0']:.1f} expl_var={val_m['expl_var']:.4f} dt={dt:.1f}s")

            self._adjust_lr(val_m["recon"])
            self._save_ckpt(self._last_path)
            if val_m["recon"] <= self.state.best_val_loss + 1e-9:
                self._save_ckpt(self._best_path)

            if self._early_stop():
                print(f"[trainer] 早停：{self.cfg.early_stop_lr_patience_epochs} epoch lr 未下降。")
                break

        return {
            "best_val_loss": self.state.best_val_loss,
            "final_epoch": self.state.epoch,
            "final_lr": self.optimizer.param_groups[0]["lr"],
            "history_len": len(self.state.history),
        }
