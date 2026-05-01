# `src/models/` 模块记忆

## 文件
- `qwen_loader.py` — 加载 Qwen3.5 文本子模块 + 第 12 层残差流 hook
- `sae_topk.py` — TopK SAE（OpenAI 2024）
- `sae_jumprelu.py` — JumpReLU SAE（DeepMind 2024，STE 矩形核）

## 关键约定

### 接口形状
- **激活输入到 SAE**：`[B, d_in]`（`d_in=hidden_size=1024`）。
  调用者负责把 `[batch, seq_len, d_in]` 展平到 `[batch*seq_len, d_in]`。
- **激活 dtype**：从 hook 抓回是 `bfloat16`（省显存），训练 SAE 时 cast 成 `float32` 以避免精度问题。

### TopK SAE
- Loss = `recon_loss + α * aux_k_loss`
- `aux_k_loss` 用 dead 特征上的 pre-activation 拟合 `x - x̂` 的残差
- Dead 特征定义：`steps_since_active >= dead_steps_threshold`
- `remove_parallel_grad_component_()` 在 optimizer.step() 之前调用，去除 W_dec 沿列方向的梯度（与单位列范数约束兼容）
- `_normalize_decoder_()` 在 optimizer.step() 之后调用

### JumpReLU SAE
- 阈值 `θ = exp(log_threshold)`，每 feature 一个
- 阶跃函数 STE：前向 `1[pre>θ]`，反向 rectangular kernel of width=bandwidth
- L0 estimator 也用 STE 让阈值参数获得梯度

## 待补充
- 量化前的 W_enc/W_dec 保持 float32 训练
- 推理/feature dashboard 见 `src/evaluate.py`（待写）
