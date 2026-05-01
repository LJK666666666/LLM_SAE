"""SAE 评估指标：L0、dead fraction、explained variance、KL divergence vs 原模型输出。

KL divergence 计算
------------------
1. 用 Qwen 跑一段文本，记录第 12 层残差流和最终 logits。
2. 用 SAE 重构第 12 层激活；将 ``model.model.layers[12].forward`` 的输出替换为重构值；
   再让模型继续后半段 forward，得到 logits'。
3. KL(logits || logits')；越小说明 SAE 越保留对下游有用的信息。

为节省工程量，本文件只暴露 ``compute_kl_substitution``，
其内部实现需要在 ``trainer`` 中临时换 hook，所以放在 trainer 里调用。
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.no_grad()
def kl_div_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """KL(P || Q)，logits 为 ``[*, vocab]``。返回每 token KL 的均值。"""
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(-1).mean()


@torch.no_grad()
def reconstruction_metrics(x: torch.Tensor, x_hat: torch.Tensor) -> dict[str, float]:
    x = x.float()
    x_hat = x_hat.float()
    mse = (x - x_hat).pow(2).mean().item()
    var_x = x.var(dim=0).sum().clamp_min(1e-8)
    var_r = (x - x_hat).var(dim=0).sum()
    explained_var = (1 - var_r / var_x).item()
    cosine = F.cosine_similarity(x, x_hat, dim=-1).mean().item()
    return {"mse": mse, "explained_variance": explained_var, "cosine_sim": cosine}
