"""TopK Sparse Autoencoder.

参考：Gao et al. "Scaling and evaluating sparse autoencoders" (OpenAI, 2024)
     SAELens 中 ``TopKSAE`` / ``TrainingSAE`` 相关代码。

核心
----
1. encode: ``pre = W_enc @ (x - b_dec) + b_enc``; 保留 top-k 激活，其余置零。
2. decode: ``x_hat = W_dec @ z + b_dec``。
3. 主损失：``||x - x_hat||²``。
4. Auxiliary loss (aux_k)：用 "dead feature" 上的 pre-activation 去拟合残差 ``x - x_hat``，
   取其中 top-k_aux 项，缓解 dead neuron 问题（OpenAI 2024 §3.2）。
5. Dead feature 判定：最近 ``dead_steps_threshold`` 步内从未被 top-k 选中视为 dead。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TopKSAEConfig:
    d_in: int                          # = hidden_size，e.g. 1024
    d_sae: int                         # 字典规模，e.g. 16384（通常 d_in*8~32）
    k: int = 32                        # top-k 稀疏度（L0 目标）
    k_aux: int = 256                   # aux_k_loss 的 k（OpenAI: 512；字典小则调低）
    aux_loss_coef: float = 1.0 / 32.0  # α，OpenAI 建议 1/32
    dead_steps_threshold: int = 1000   # 连续多少 step 未激活则视为 dead
    normalize_decoder: bool = True     # W_dec 每列 L2 归一化（SAELens 默认）
    init_decoder_as_encoder_T: bool = True

    def __post_init__(self) -> None:
        if self.k > self.d_sae:
            raise ValueError(f"k={self.k} 不能大于 d_sae={self.d_sae}")
        if self.k_aux > self.d_sae:
            self.k_aux = min(self.k_aux, self.d_sae)


class TopKSAE(nn.Module):
    """Top-K Sparse Autoencoder.

    Shapes
    ------
    输入 ``x``:   ``[B, d_in]``（调用者负责把 ``[B, T, d_in]`` 展成 ``[B*T, d_in]``）
    隐藏 ``z``:   ``[B, d_sae]``
    输出 ``x̂``:   ``[B, d_in]``
    """

    def __init__(self, cfg: TopKSAEConfig):
        super().__init__()
        self.cfg = cfg

        # 编码器：先减去 decoder bias（"centered" 版本，SAELens 默认）。
        self.W_enc = nn.Parameter(torch.empty(cfg.d_in, cfg.d_sae))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
        self.W_dec = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_in))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in))

        # 初始化：Kaiming + 归一化到单位列范数。
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        if cfg.init_decoder_as_encoder_T:
            with torch.no_grad():
                self.W_dec.copy_(self.W_enc.t())
        else:
            nn.init.kaiming_uniform_(self.W_dec, a=5 ** 0.5)
        if cfg.normalize_decoder:
            self._normalize_decoder_()

        # dead-feature 追踪：每个特征"距离上次被选中的 step 数"
        self.register_buffer("steps_since_active", torch.zeros(cfg.d_sae, dtype=torch.long))

    # ---------- util ----------
    @torch.no_grad()
    def _normalize_decoder_(self) -> None:
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norms)

    @torch.no_grad()
    def remove_parallel_grad_component_(self) -> None:
        """SAELens 技巧：训练中去除 W_dec 梯度沿列方向（径向）的分量，
        防止归一化把梯度白白抵消。应当在 optimizer.step() 之前调用。
        """
        if self.W_dec.grad is None:
            return
        W = self.W_dec.data
        g = self.W_dec.grad
        # 每行（每个 feature 的 decoder 向量）上投影
        proj = (g * W).sum(dim=1, keepdim=True) * W
        g.sub_(proj)

    # ---------- core ----------
    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        """仅计算 pre-activation（不加 top-k）。"""
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    @staticmethod
    def _topk(pre: torch.Tensor, k: int) -> torch.Tensor:
        """在最后一维取 top-k，其余置零（ReLU 式保留值，不是 hard assign）。"""
        if k >= pre.shape[-1]:
            return F.relu(pre)
        values, indices = pre.topk(k, dim=-1)
        values = F.relu(values)
        out = torch.zeros_like(pre)
        out.scatter_(-1, indices, values)
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.encode_pre(x)
        return self._topk(pre, self.cfg.k)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> "TopKSAEOutput":
        pre = self.encode_pre(x)
        z = self._topk(pre, self.cfg.k)
        x_hat = self.decode(z)

        recon_loss = (x_hat - x).pow(2).sum(dim=-1).mean()

        # aux_k_loss on dead features
        if self.training and self.cfg.aux_loss_coef > 0:
            dead_mask = self.steps_since_active >= self.cfg.dead_steps_threshold  # [d_sae]
            num_dead = int(dead_mask.sum())
            if num_dead > 0:
                k_aux = min(self.cfg.k_aux, num_dead)
                dead_pre = pre.masked_fill(~dead_mask.unsqueeze(0), float("-inf"))
                z_aux = self._topk(dead_pre, k_aux)
                residual = (x - x_hat).detach()
                residual_hat = z_aux @ self.W_dec  # 不含 b_dec
                aux_loss = (residual_hat - residual).pow(2).sum(dim=-1).mean()
            else:
                aux_loss = recon_loss.new_zeros(())
        else:
            aux_loss = recon_loss.new_zeros(())

        total_loss = recon_loss + self.cfg.aux_loss_coef * aux_loss

        # 更新 dead-feature 计数（只在 training 时）
        if self.training:
            with torch.no_grad():
                active = (z.abs().sum(dim=0) > 0)  # [d_sae]
                self.steps_since_active = torch.where(
                    active,
                    torch.zeros_like(self.steps_since_active),
                    self.steps_since_active + 1,
                )

        # 监控
        with torch.no_grad():
            l0 = (z != 0).float().sum(dim=-1).mean()
            dead_frac = (self.steps_since_active >= self.cfg.dead_steps_threshold).float().mean()
            # explained variance: 1 - Var(residual)/Var(x)
            var_x = x.float().var(dim=0).sum().clamp_min(1e-8)
            var_r = (x - x_hat).float().var(dim=0).sum()
            explained_var = 1 - var_r / var_x

        return TopKSAEOutput(
            loss=total_loss,
            recon_loss=recon_loss.detach(),
            aux_loss=aux_loss.detach(),
            l0=l0.detach(),
            dead_frac=dead_frac.detach(),
            explained_variance=explained_var.detach(),
            x_hat=x_hat,
            z=z,
        )


@dataclass
class TopKSAEOutput:
    loss: torch.Tensor
    recon_loss: torch.Tensor
    aux_loss: torch.Tensor
    l0: torch.Tensor
    dead_frac: torch.Tensor
    explained_variance: torch.Tensor
    x_hat: torch.Tensor = field(repr=False)
    z: torch.Tensor = field(repr=False)
