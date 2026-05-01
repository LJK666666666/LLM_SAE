# 设计决策记录

## 1. Hook 位置：第 12 层 decoder block 输出（残差流）

- **原因**：24 层 Qwen3.5 的中间层。Anthropic / DeepMind / OpenAI 主流 SAE 工作均选择中间层残差流，特征兼具语法与语义抽象，可解释性最佳。
- **实现**：`model.language_model.model.layers[12]` 注册 `register_forward_hook`，取 `output[0]`（hidden_states）。
- **形状**：`[batch, seq_len, 1024]` → 展平到 token 维度供 SAE 训练。

## 2. SAE 变体：先 TopK，后 JumpReLU

- **TopK SAE**（OpenAI "Scaling and evaluating sparse autoencoders" 2024）：
  - 编码后保留 top-k 激活，其余置零。
  - 损失：`||x - decode(topk(encode(x)))||² + α·aux_k_loss`。
  - aux_k_loss：用 dead features 重构残差，缓解 dead neuron 问题。
- **JumpReLU SAE**（DeepMind "JumpReLU SAEs" 2024）：
  - 用阈值化 ReLU + L0 估计（STE 直通梯度），介于 TopK 与 vanilla 之间。
  - 损失：`||x - decode(JumpReLU(Wx+b))||² + λ·L0_estimator`。
- **不选 vanilla L1**：shrinkage 问题严重，已不主流。
- **不选 Gated**：性能介于 vanilla 与 JumpReLU 之间，价值不大。

## 3. 参考库：SAELens 算法参考 + 自行精简实现

- **不直接使用** `sae_lens.LanguageModelSAERunnerConfig` 训练流程：因其底座是 `transformer_lens.HookedTransformer`，**不支持 Qwen3.5**（混合 linear/full attention 的新多模态架构）。
- **自行实现 SAE 类**（~150 行 / 个），算法严格对照 SAELens 源码。
- **自行实现 ActivationsStore 等价物**：滚动缓冲区 + 在线打乱。

## 4. 训练语料：中英混合

- **英文**：HuggingFace `HuggingFaceFW/fineweb-edu`（streaming）。
- **中文**：`opencsg/chinese-fineweb-edu` 或 `Skywork/SkyPile-150B`（streaming）。
- **混合比例**：默认 1:1（按抽样次数；由于分词差异，不严格等于 token 数 1:1）。
- **可在 yaml 中调整数据集名与比例**。
- **Parquet streaming 风险控制**：只读取 `text` 列，并设置 `streaming_batch_size=1024`。原因是中文 FineWeb streaming 曾在 PyArrow 读取过大 record batch 时申请约 2.45GB CPU 内存失败。

## 5. 多模态处理

- 仅训练文本 SAE。加载时 `Qwen3_5ForConditionalGeneration` → 取 `model.language_model`。
- vision 塔权重也加载（无法跳过单文件 safetensors 中部分参数），但 forward 不调用。
- 显存够则 `bfloat16`；不够则用 `device_map="auto"`。

## 6. 学习率与早停

按 CLAUDE.md 默认：3 epoch val_loss 不降则 lr*0.7，15 epoch lr 不降则早停。
对 SAE 训练的"epoch"定义：每 N=`steps_per_epoch` batch 视作一个 epoch（正式配置默认 1000；SAE 训练通常用 token 数而非 epoch，本项目折中映射）。

## 7. 长 epoch checkpoint 策略

- **epoch 末**：完成 train + val 后保存 `last.pt`；val_recon 创新低保存 `best.pt`。
- **epoch 中**：`save_every_steps>0` 时，每 N 个 optimizer step 覆盖保存 `last.pt`，正式配置默认 100。
- **验证前**：训练阶段结束后、进入验证前也保存一次，避免验证 streaming 崩溃导致整轮训练权重丢失。
- **取舍**：中途恢复不精确记录 epoch 内 batch offset，可能重复部分训练统计；优先满足长时间训练中断后保留最新权重。
