"""激活缓冲区：从文本流批量送入 Qwen → 抓 hook 激活 → 打乱后吐 SAE batch。

参考：SAELens ``ActivationsStore``（saelens/training/activations_store.py）。

工作流
------
1. 维护一个滚动缓冲区，容量 ``buffer_size_tokens``（如 256K token 的 hidden_states）。
2. 缓冲不足时，从 text 流不断取文本→tokenize→截到 ``ctx_len`` →喂模型→收集激活→塞缓冲。
3. 缓冲满后整体打乱，按 ``sae_batch_size`` 切片吐出，吐到 ``refill_threshold`` 以下时再 refill。

为何要打乱
----------
SAE 训练假设激活近似 i.i.d. 采样。直接按 token 顺序训练会让相邻 batch 高度相关，
拖慢收敛、加剧 dead feature。SAELens / OpenAI 论文均使用大缓冲 + 随机打乱。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Iterable

import torch

from ..models.qwen_loader import HookedQwen


@dataclass
class ActivationStoreConfig:
    sae_batch_size: int = 4096          # SAE 训练 batch 大小（token 数）
    buffer_size_tokens: int = 524288    # 缓冲区容量（token 数），约 0.5M
    refill_threshold: float = 0.5       # 用到这个比例以下时 refill
    ctx_len: int = 512                  # 文本最长 token 数（截断）
    model_batch_size: int = 8           # 喂入 Qwen 的句子数 per forward
    dtype_storage: torch.dtype = torch.bfloat16  # 缓冲存储 dtype（省显存）
    dtype_train: torch.dtype = torch.float32     # 吐给 SAE 的 dtype
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ActivationStore:
    def __init__(
        self,
        hooked: HookedQwen,
        text_stream: Iterable[str],
        cfg: ActivationStoreConfig,
    ):
        self.hooked = hooked
        self.text_iter = iter(text_stream)
        self.cfg = cfg

        d = hooked.hidden_size
        self.buffer = torch.empty(
            (0, d), dtype=cfg.dtype_storage, device=cfg.device
        )
        self.cursor = 0

    # ---------- internal ----------
    def _gather_one_chunk(self) -> torch.Tensor:
        """收集 ``model_batch_size`` 句的激活，返回 ``[N_tokens, d_in]``。"""
        cfg = self.cfg
        tok = self.hooked.tokenizer
        texts = []
        # 文本可能很短/很长；尽量凑够 model_batch_size 条非空文本。
        while len(texts) < cfg.model_batch_size:
            try:
                t = next(self.text_iter)
            except StopIteration:
                break
            if t and t.strip():
                texts.append(t)
        if not texts:
            raise RuntimeError("文本流耗尽，无法 refill 激活缓冲。")

        enc = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.ctx_len,
        )
        input_ids = enc["input_ids"].to(cfg.device)
        attn_mask = enc["attention_mask"].to(cfg.device)

        acts = self.hooked.get_activations(input_ids, attn_mask)  # [B, T, d]
        # mask 掉 padding token（attn_mask=0）
        mask = attn_mask.bool().unsqueeze(-1)  # [B, T, 1]
        acts = acts.masked_select(mask).view(-1, self.hooked.hidden_size)
        return acts.to(cfg.dtype_storage)

    def _refill(self) -> None:
        cfg = self.cfg
        target = cfg.buffer_size_tokens
        # 保留未消费的尾部
        remaining = self.buffer[self.cursor:]
        chunks = [remaining]
        cur = remaining.shape[0]
        while cur < target:
            chunk = self._gather_one_chunk()
            chunks.append(chunk)
            cur += chunk.shape[0]
        new_buf = torch.cat(chunks, dim=0)
        # 整体打乱
        perm = torch.randperm(new_buf.shape[0], device=new_buf.device)
        self.buffer = new_buf[perm].contiguous()
        self.cursor = 0

    # ---------- public ----------
    def __iter__(self) -> "ActivationStore":
        return self

    def __next__(self) -> torch.Tensor:
        cfg = self.cfg
        # refill 触发条件
        if self.buffer.shape[0] - self.cursor < cfg.sae_batch_size or \
           self.buffer.shape[0] - self.cursor < cfg.refill_threshold * cfg.buffer_size_tokens:
            self._refill()
        end = self.cursor + cfg.sae_batch_size
        batch = self.buffer[self.cursor:end]
        self.cursor = end
        return batch.to(cfg.dtype_train)

    def take(self, n_batches: int) -> Iterator[torch.Tensor]:
        for _ in range(n_batches):
            yield next(self)
