# `src/training/` 模块记忆

## 文件
- `trainer.py` — 通用 SAE trainer（TopK / JumpReLU 都能用）
- `metrics.py` — 重构指标 + KL 散度

## 关键约定

### epoch 折算
- SAE 一般以 token 数计训练量，本项目折算：
  - 1 "epoch" = `steps_per_epoch` 个 SAE batch（默认 1000）
  - 1 batch = `sae_batch_size` token（默认 4096）
  - → 1 epoch ≈ 4M token
- CLAUDE.md 的"3 epoch 不降则 lr*0.7、15 epoch 不降则早停"按此 epoch 计

### 检查点
- 每 epoch 末**必定**保存 last.pt（含 SAE 权重 + optimizer + trainer 状态 + 完整配置快照）
- val_recon 创新低则同步保存 best.pt
- 恢复：`--resume` 自动读 last.pt，沿用 epoch / global_step / 历史
- 长 epoch 额外保护：`trainer.save_every_steps` > 0 时，每 N 个 optimizer step 覆盖保存 `last.pt`
- 每个 epoch 的训练阶段结束后、进入验证前也会在 `save_every_steps` 启用时保存一次；这样验证阶段 streaming 崩溃时，至少保留本轮训练后的权重
- 注意：若在 epoch 中途从 step checkpoint 恢复，当前实现不会精确记住 epoch 内 batch offset，可能会从当前权重重复跑该 epoch 的剩余/全部训练统计；优先目标是避免权重丢失

### 单 batch 验证
- `--max-iters 1` 跳过完整 fit，只跑指定 batch 数后直接保存 ckpt 退出
- 用于快速验证整条链路（model 加载 → hook → store → SAE → 反向 → 保存）

### 梯度处理顺序
1. `loss.backward()`
2. `sae.remove_parallel_grad_component_()` — 去除 W_dec 沿径向梯度
3. `clip_grad_norm_`
4. `optimizer.step()`
5. `sae._normalize_decoder_()` — 重新单位化列范数
