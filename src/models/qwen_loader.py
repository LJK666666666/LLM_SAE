"""加载本地 Qwen3.5-0.8B 文本子模型并在指定层的 decoder block 输出注册 forward hook。

设计要点
---------
- 模型为 ``Qwen3_5ForConditionalGeneration``（多模态）。仅训练文本 SAE，
  因此通过 ``model.language_model`` 访问文本子模块（其下 ``.model.layers[i]`` 为 decoder block）。
- ``register_forward_hook`` 直接抓取 decoder block 输出（残差流），形状
  ``[batch, seq_len, hidden_size]``。
- 不做 generation，仅 forward 抓激活；模型设为 ``eval()`` 且 ``requires_grad_(False)``。
- 兼容 ``trust_remote_code=True``（Qwen3.5 较新，可能依赖 hub 上的实现）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, AutoConfig


DEFAULT_MODEL_DIR = Path("Qwen3.5-0.8B")  # 项目根目录相对路径
DEFAULT_HOOK_LAYER = 12                    # 24 层中点


@dataclass
class HookedQwen:
    """封装：HF 模型 + tokenizer + 抓取到的最近一次激活。"""

    model: nn.Module
    tokenizer: object
    text_module: nn.Module          # decoder stack（含 .layers）
    hook_layer: int
    hidden_size: int
    device: torch.device
    dtype: torch.dtype
    _handle: object = None
    _last_activation: Optional[torch.Tensor] = None

    # ---------- API ----------
    def remove_hook(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @torch.no_grad()
    def get_activations(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向一次，返回 hook 层激活，形状 [batch, seq_len, hidden_size]。

        参数
        ----
        input_ids: ``[batch, seq_len]``，long tensor
        attention_mask: 同形状 0/1（可选）
        """
        self._last_activation = None
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # 仅文本 forward。Qwen3_5ForConditionalGeneration 接受 input_ids 直接做语言建模。
        # 用 text_module 直接 forward 最干净（避免触发 vision 路径）。
        _ = self.text_module(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        if self._last_activation is None:
            raise RuntimeError("Hook 未捕获到激活，可能模型 forward 路径异常。")
        return self._last_activation


def _resolve_text_module(model: nn.Module) -> nn.Module:
    """在多模态 Qwen3_5 上定位文本 decoder 主体（其 .layers 是 decoder block 列表）。

    优先级：
        model.language_model.model  →  model.language_model  →  model.model  →  model
    """
    candidates = []
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            candidates.append(lm.model)
        if hasattr(lm, "layers"):
            candidates.append(lm)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        candidates.append(model.model)
    if hasattr(model, "layers"):
        candidates.append(model)
    for c in candidates:
        if hasattr(c, "layers") and len(c.layers) > 0:
            return c
    raise AttributeError(
        "无法定位 Qwen3.5 文本 decoder 主体（找不到 .layers）。请检查 transformers 版本是否 ≥4.57。"
    )


def load_hooked_qwen(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    hook_layer: int = DEFAULT_HOOK_LAYER,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
) -> HookedQwen:
    """加载本地 Qwen3.5-0.8B，在第 ``hook_layer`` 个 decoder block 输出注册 hook。"""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir.resolve()}")

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)

    # 优先 AutoModelForCausalLM；若架构不支持（多模态可能要 AutoModel），fallback。
    # transformers v5+ 用 dtype；旧版用 torch_dtype。
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
    except (ValueError, KeyError):
        try:
            model = AutoModel.from_pretrained(
                model_dir,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )
        except TypeError:
            model = AutoModel.from_pretrained(
                model_dir,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    text_module = _resolve_text_module(model)
    n_layers = len(text_module.layers)
    if not (0 <= hook_layer < n_layers):
        raise ValueError(f"hook_layer={hook_layer} 越界，模型共 {n_layers} 层。")

    # Hidden size：优先取 config.text_config.hidden_size；fallback 到 config.hidden_size
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        hidden_size = config.text_config.hidden_size
    else:
        hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        # 兜底：从模型 embedding 形状推断
        emb = next((m for m in model.modules() if isinstance(m, nn.Embedding)), None)
        hidden_size = emb.embedding_dim if emb is not None else 1024

    hooked = HookedQwen(
        model=model,
        tokenizer=tokenizer,
        text_module=text_module,
        hook_layer=hook_layer,
        hidden_size=hidden_size,
        device=torch.device(device),
        dtype=dtype,
    )

    # decoder block 输出通常是 tuple，第 0 个是 hidden_states。
    def _hook(_module, _inputs, output):
        hs = output[0] if isinstance(output, (tuple, list)) else output
        # detach 以免影响 SAE 前向；保持 dtype（bf16）以省显存。
        hooked._last_activation = hs.detach()

    hooked._handle = text_module.layers[hook_layer].register_forward_hook(_hook)
    return hooked


# 便于 `python src/models/qwen_loader.py` 单跑做最小验证
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--hook-layer", type=int, default=DEFAULT_HOOK_LAYER)
    parser.add_argument("--text", default="你好，世界。Hello, world.")
    args = parser.parse_args()

    hooked = load_hooked_qwen(args.model_dir, hook_layer=args.hook_layer)
    enc = hooked.tokenizer(args.text, return_tensors="pt")
    act = hooked.get_activations(enc["input_ids"], enc.get("attention_mask"))
    print(f"hook_layer={hooked.hook_layer}  hidden_size={hooked.hidden_size}")
    print(f"activation.shape={tuple(act.shape)}  dtype={act.dtype}  device={act.device}")
    print(f"activation 统计: mean={act.float().mean().item():.4f}  std={act.float().std().item():.4f}")
