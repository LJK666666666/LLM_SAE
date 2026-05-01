"""JumpReLU Sparse Autoencoder.

参考：Rajamanoharan et al. "Jumping Ahead: Improving Reconstruction Fidelity
     with JumpReLU Sparse Autoencoders" (DeepMind, 2024)。

核心
----
1. encode: ``pre = W_enc @ (x - b_dec) + b_enc``
2. JumpReLU 阈值化：``z = pre · 1[pre > θ_j]``，θ_j 为每个 feature 的可学阈值（exp 参数化保证正）。
3. 损失：``L = ||x - x̂||² + λ · L0_estimator(pre, θ)``，
   L0 用 STE（Straight-Through Estimator）：前向用阶跃函数，反向用 rectangular kernel。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class JumpReLUSAEConfig:
    d_in: int
    d_sae: int
    sparsity_coef: float = 1e-3       # λ，需根据数据规模调
    bandwidth: float = 1e-3            # ε，STE 矩形核宽度（DeepMind 论文用 1e-3）
    init_threshold: float = 0.001
    normalize_decoder: bool = True


class _RectangleKernel(torch.autograd.Function):
    """STE：前向 H(x) = 1[x>0]；反向 d/dx ≈ rectangle kernel of width=bandwidth/2."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(x)
        ctx.bandwidth = bandwidth
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        bw = ctx.bandwidth
        # 矩形核：在 |x| < bw/2 时 1/bw，否则 0
        grad_x = (x.abs() < bw / 2).to(grad_out.dtype) / bw * grad_out
        return grad_x, None


def heaviside_ste(x: torch.Tensor, bandwidth: float) -> torch.Tensor:
    return _RectangleKernel.apply(x, bandwidth)


class JumpReLUSAE(nn.Module):
    """JumpReLU SAE.

    参数化：``θ = exp(log_threshold)`` 保证阈值始终 >0。
    """

    def __init__(self, cfg: JumpReLUSAEConfig):
        super().__init__()
        self.cfg = cfg

        self.W_enc = nn.Parameter(torch.empty(cfg.d_in, cfg.d_sae))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
        self.W_dec = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_in))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in))
        self.log_threshold = nn.Parameter(
            torch.full((cfg.d_sae,), float(torch.tensor(cfg.init_threshold).log()))
        )

        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
        if cfg.normalize_decoder:
            self._normalize_decoder_()

    @torch.no_grad()
    def _normalize_decoder_(self) -> None:
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norms)

    @torch.no_grad()
    def remove_parallel_grad_component_(self) -> None:
        if self.W_dec.grad is None:
            return
        W = self.W_dec.data
        g = self.W_dec.grad
        proj = (g * W).sum(dim=1, keepdim=True) * W
        g.sub_(proj)

    def threshold(self) -> torch.Tensor:
        return self.log_threshold.exp()

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.encode_pre(x)
        theta = self.threshold()
        gate = (pre > theta).to(pre.dtype)
        return pre * gate  # 推理时硬阶跃

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> "JumpReLUSAEOutput":
        pre = self.encode_pre(x)
        theta = self.threshold()  # [d_sae]

        # JumpReLU with STE for the gate
        gate = heaviside_ste(pre - theta, self.cfg.bandwidth)
        z = pre * gate

        x_hat = self.decode(z)
        recon_loss = (x_hat - x).pow(2).sum(dim=-1).mean()

        # L0 estimator: 同样用 STE，使阈值参数有梯度
        l0_per_token = heaviside_ste(pre - theta, self.cfg.bandwidth).sum(dim=-1)
        sparsity_loss = l0_per_token.mean()

        total_loss = recon_loss + self.cfg.sparsity_coef * sparsity_loss

        with torch.no_grad():
            l0 = (z != 0).float().sum(dim=-1).mean()
            var_x = x.float().var(dim=0).sum().clamp_min(1e-8)
            var_r = (x - x_hat).float().var(dim=0).sum()
            explained_var = 1 - var_r / var_x
            dead_frac = (z.abs().sum(dim=0) == 0).float().mean()

        return JumpReLUSAEOutput(
            loss=total_loss,
            recon_loss=recon_loss.detach(),
            sparsity_loss=sparsity_loss.detach(),
            l0=l0.detach(),
            dead_frac=dead_frac.detach(),
            explained_variance=explained_var.detach(),
            x_hat=x_hat,
            z=z,
        )


@dataclass
class JumpReLUSAEOutput:
    loss: torch.Tensor
    recon_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    l0: torch.Tensor
    dead_frac: torch.Tensor
    explained_variance: torch.Tensor
    x_hat: torch.Tensor = field(repr=False)
    z: torch.Tensor = field(repr=False)
