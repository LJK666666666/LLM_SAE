# 给用户的关键信息（请优先阅读）

> 本文件汇总：(1) 主要运行命令；(2) 配置/参数所在位置；(3) Claude 代我做出的开放性决策（你应当审阅）；(4) 环境陷阱与已知风险。
> 索引位置：[`_memory/index.md`](index.md)。其他详细记忆：[`design_decisions.md`](design_decisions.md) / [`architecture.md`](architecture.md) / [`naming_conventions.md`](naming_conventions.md) / [`progress.md`](progress.md)。

---

## 0. 项目信息

- 作者：966279

## 1. 主要运行命令（执行路径=项目根）

### 1.1 离线 SAE 单元测试（不依赖 HF / 不加载 Qwen，秒级完成）
```bash
python tests/test_sae_smoke.py
```

### 1.2 端到端烟雾测试（加载真 Qwen3.5，但用本地 .txt，无需 HF 网络）
```bash
# 单 batch 跑通整条链路（约 30s）
python src/train.py --config configs/train_topk_smoke.yaml --tag smoke --max-iters 1

# 完整 2 epoch（验证 trainer + ckpt + history）
python src/train.py --config configs/train_topk_smoke.yaml --tag smoke

# 从 last.pt 恢复并继续到 epoch 4
python src/train.py --config configs/train_topk_smoke.yaml --tag smoke --resume --max-epochs 4
```

### 1.3 正式训练（推荐：先下载固定子集，再本地读取训练）
```bash
# 1) 下载固定中英文本地子集（只在准备数据时需要 HF 网络）
python src/data/download_subset.py --config configs/train_topk.yaml \
    --output-dir data/local_corpus/fineweb_edu_subset \
    --en-train-docs 100000 --zh-train-docs 100000 \
    --en-val-docs 5000 --zh-val-docs 5000

# 2) 使用脚本生成的本地配置训练 TopK SAE
python src/train.py --config data/local_corpus/fineweb_edu_subset/train_config_local.yaml \
    --tag topk_l12_local

# 3) 从本地子集训练的 last.pt 恢复
python src/train.py --config data/local_corpus/fineweb_edu_subset/train_config_local.yaml \
    --tag topk_l12_local --resume

# 4) 本地子集版：调字典 / k / lr / 训练长度（高频参数走命令行）
python src/train.py --config data/local_corpus/fineweb_edu_subset/train_config_local.yaml \
    --tag topk_l12_local_d32k \
    --d-sae 32768 --k 64 --lr 1e-4 --steps-per-epoch 2000 --max-epochs 50

# 5) 本地子集版：长 epoch 时提高保存频率，降低中断损失
python src/train.py --config data/local_corpus/fineweb_edu_subset/train_config_local.yaml \
    --tag topk_l12_local --save-every-steps 50
```

### 1.4 Colab / 云端推荐命令（数据放本地盘，结果放 Drive）
```bash
# 若项目位于 /content/LLM_SAE，先把固定子集下载到 /content/data 附近的本地盘
python src/data/download_subset.py --config configs/train_topk.yaml \
    --output-dir ../data/fineweb_edu_subset \
    --en-train-docs 100000 --zh-train-docs 100000 \
    --en-val-docs 5000 --zh-val-docs 5000

# 训练时读取 /content 本地盘数据，结果保存到 Google Drive
python src/train.py --config ../data/fineweb_edu_subset/train_config_local.yaml \
    --tag topk_l12_local \
    --results-root ../drive/MyDrive/results

# 云端恢复训练：必须使用同一个 config 和同一个 results_root
python src/train.py --config ../data/fineweb_edu_subset/train_config_local.yaml \
    --tag topk_l12_local \
    --results-root ../drive/MyDrive/results --resume
```

### 1.5 直接 HF streaming 训练（备选：适合短跑/调试）
```bash
# TopK SAE（默认字典 16384，k=32；训练期间仍依赖 HF streaming）
python src/train.py --config configs/train_topk.yaml --tag topk_l12

# 从 streaming 训练的 last.pt 恢复（仅当对应实验目录已有 last.pt 时可用）
python src/train.py --config configs/train_topk.yaml --tag topk_l12 --resume

# streaming 版调字典 / k / lr / 训练长度
python src/train.py --config configs/train_topk.yaml --tag topk_l12_d32k \
    --d-sae 32768 --k 64 --lr 1e-4 --steps-per-epoch 2000 --max-epochs 50

# JumpReLU SAE
python src/train.py --config configs/train_jumprelu.yaml --tag jumprelu_l12
```

### 1.6 单文件最小验证（只验证 Qwen 加载 + hook）
```bash
python src/models/qwen_loader.py --hook-layer 12 --text "你好。Hello."
```

---

## 2. 配置/参数：在哪改？

### 2.1 高频参数（命令行覆盖，CLAUDE.md 规则 17）
| 标志 | 默认 | 说明 |
|---|---|---|
| `--lr` | yaml | Adam 学习率 |
| `--d-sae` | yaml | 字典维度（建议 d_in×8 ~ d_in×32） |
| `--k` | yaml | TopK 稀疏度（典型 16~64） |
| `--steps-per-epoch` | yaml | 每"epoch"的 SAE batch 数 |
| `--max-epochs` | yaml | 总 epoch 上限 |
| `--save-every-steps` | yaml | 每 N 个 optimizer step 覆盖保存 `last.pt`；0 表示仅按 epoch 保存 |
| `--results-root` | `results` | 实验结果根目录；Colab 推荐 `../drive/MyDrive/results` |
| `--max-iters N` | None | 跳过完整训练，仅跑 N batch 验证 |
| `--device` | yaml | cuda / cpu / auto |
| `--resume` | False | 从同 tag 最大编号目录的 last.pt 恢复 |

### 2.2 低频参数（改 yaml，`configs/`）
- `configs/train_topk.yaml` — TopK 正式训练默认配置
- `configs/train_topk_local.yaml` — TopK 本地固定子集训练模板
- `configs/train_jumprelu.yaml` — JumpReLU 正式训练默认配置
- `configs/train_topk_smoke.yaml` — 烟雾测试（极小数据/缓冲）

**主要可调段**：
- `model.hook_layer` — 改钩取层（当前 12，可试 6 / 18）
- `model.dtype` — bfloat16 / float16 / float32
- `sae.k_aux` / `sae.aux_loss_coef` — TopK 的 dead-feature aux loss 强度
- `sae.dead_steps_threshold` — 多少 step 不激活算 dead
- `sae.sparsity_coef` / `sae.bandwidth` — JumpReLU 关键超参
- `corpus.en_ratio` — 中英比例（默认 0.5/0.5）
- `corpus.en_name` / `zh_name` — 切换数据集
- `corpus.streaming_batch_size` — HF Parquet streaming 的 record batch 行数（默认 1024，避免 PyArrow 一次申请几 GB 内存）
- `corpus.local_train_path` / `local_val_path` — 单文件本地 JSONL/TXT 训练入口
- `corpus.local_en_train_path` / `local_zh_train_path` / `local_en_val_path` / `local_zh_val_path` — 中英文分文件本地训练入口
- `activation_store.sae_batch_size` — SAE batch 大小（默认 4096 token）
- `activation_store.buffer_size_tokens` — 缓冲容量（默认 524288 ≈ 0.5M token）
- `activation_store.ctx_len` — 单条文本最长 token 数（默认 512）
- `trainer.lr_patience_epochs` / `early_stop_lr_patience_epochs` — 调度/早停耐心
- `trainer.save_every_steps` — 长 epoch 的额外 checkpoint 频率（正式配置默认 100）

### 2.3 实验目录命名
约定：`{results_root}/{tag}_{n}/`，{n} 自动递增；`results_root` 默认 `results`，可用 `--results-root` 指到云盘。详见 [`naming_conventions.md`](naming_conventions.md)。

### 2.4 数据来源
- 烟雾测试：`data/smoke_text.txt`（你可手动编辑/扩充）
- 正式：HuggingFace `HuggingFaceFW/fineweb-edu` + `opencsg/chinese-fineweb-edu`（streaming）
- streaming 模式不会提前下载完整语料；训练时边读边取样，HF Hub 元信息通常会进用户缓存，`data/cache/` 主要用于 datasets 缓存配置。
- 当前只读取 `text` 列，并设置 `streaming_batch_size=1024`，用于降低 Parquet streaming 的 CPU 内存峰值。
- 正式长跑推荐先运行 `src/data/download_subset.py` 生成固定 JSONL 子集；脚本会输出 `manifest.json`、`corpus_local.yaml` 和可直接训练的 `train_config_local.yaml`。

---

## 3. 我代你做出的开放性决策（建议审阅，可推翻重来）

> 这些决定我自行做了，但每个都不是唯一正确选项。若发现不合你意，请直接指出。

### 3.1 ckpt 恢复时 `state.epoch` 语义 = "已完成 epoch 数"
- **Why**：避免 resume 后 history.csv 出现重复 epoch 行（之前是该 bug，已修）。
- **影响**：恢复时打印 `恢复：epoch=N`，下一个跑的就是 epoch N+1。
- **可换方案**：用 `state.epoch_completed` 与 `state.epoch_running` 两个字段更明确。

### 3.2 SAE 一律以 fp32 训练，激活以 bfloat16 缓存
- **Why**：SAE 参数小、训练敏感，fp32 稳定；激活量大用 bf16 省显存。
- **影响**：每个 batch 取出时做 bf16→fp32 cast。
- **可换方案**：用 amp / fp16 训练 SAE 进一步省显存（但要监控 NaN）。

### 3.3 "epoch" 折算为 `steps_per_epoch=1000` 个 SAE batch
- **Why**：SAE 文献用 token 数衡量，CLAUDE.md 用 epoch；折中让 lr/早停规则可用。
- **当前默认**：1 epoch ≈ 1000 batch × 4096 token ≈ 4M token；正式训练通常 50~200 epoch ≈ 200M~800M token。
- **可换方案**：直接以"训练 token 数"为单位重写调度（更贴 SAE 文献），或调大 `steps_per_epoch`。

### 3.4 验证集 = 同源数据流换 seed（不切独立验证集）
- **Why**：SAE 训练以"激活分布上的重构"为目标，验证集只需统计上独立即可。
- **影响**：理论上 train/val 分布完全相同，仅采样独立。
- **可换方案**：从某个 held-out 文档子集生成验证激活；或用不同语料源做 cross-domain 验证。

### 3.5 Decoder 列做 L2 单位归一化 + 训练中去除径向梯度
- **Why**：SAELens / Anthropic 默认做法，避免特征 norm 漂移。
- **影响**：W_dec 行（每个特征的方向）始终是单位向量。
- **可换方案**：不归一化（让 norm 自由），用 weight decay 控制幅度。

### 3.6 中英比例 1:1（按抽样次数，不是按 token 数）
- **Why**：简单；实际 token 数会因分词差异略偏。
- **影响**：若想严格按 token 数 1:1，需要在 store 层重新加权。
- **可换方案**：调 `corpus.en_ratio`，或改成"按 token 数 quota"。

### 3.7 hook 层 12 选 decoder block **整体输出**（残差流），而非 attn / mlp 单独输出
- **Why**：SAE 主流做法，可解释性研究兼容性最佳。
- **可换方案**：如果想训"功能特化"SAE，可换钩 `attention.o_proj` 或 `mlp.down_proj` 输出。

### 3.8 烟雾测试用 20 行混合中英文本（`data/smoke_text.txt`）
- **Why**：完全离线、零外部依赖、覆盖典型概念词。
- **可换方案**：换成你领域相关的小语料更利于看 SAE 是否捕到关心的概念。

---

## 4. 环境陷阱（已为你处理 / 待你确认）

### 4.1 ✅ 已处理：transformers 必须 ≥ 5.6.0.dev
- release 4.57.1 **不识别** `model_type=qwen3_5`，会报 `KeyError: 'qwen3_5'`。
- 已通过 `pip install git+https://github.com/huggingface/transformers.git` 升级。
- **风险**：dev 版可能引入 API breaking change，正式发版前请勿混用。

### 4.2 ✅ 已处理：nibabel 与 NumPy 2.x 不兼容
- 现象：`datasets` import 触发旧版 `nibabel`，旧版调用 NumPy 2.x 已移除的 `np.sctypes`，导致正式 HF 数据流初始化失败。
- 已确认当时环境：`numpy 2.2.6`、`datasets 4.7.0`、`nibabel 3.2.0`。
- 处理建议已给出：升级 `nibabel` 到支持 NumPy 2.x 的版本，或保守降级 `numpy<2`。用户随后正式训练已越过该导入错误。
- 烟雾测试仍可用 `LocalTextFileStream` 走本地 .txt，不依赖 datasets。

### 4.3 ℹ Qwen3.5 fast path 警告
```
[transformers] The fast path is not available because one of the required library is not installed.
Falling back to torch implementation.
```
- 来源：缺 `flash-linear-attention` / `causal-conv1d`，Qwen3.5 的 linear attention 走 PyTorch 慢实现。
- **影响**：模型 forward 慢一些，不影响正确性。
- **解决（可选）**：按警告链接装 `flash-linear-attention` 和 `causal-conv1d`，能显著加速激活采集。

### 4.4 ⚠ HF streaming 训练的中断风险
- `streaming: true` 不会提前下载完整数据集，一个 epoch 结束后也不能假设后续完全离线可跑。
- 若 HF 网络断开、远程分片读取失败或被限流，训练仍可能中断；建议设置 `HF_TOKEN` 提高限流额度。
- 2026-04-29 遇到过 `pyarrow.lib.ArrowMemoryError: realloc of size 2449473536 failed`，发生在中文 FineWeb streaming 验证阶段。已通过只读 `text` 列和限制 `streaming_batch_size=1024` 降低风险。
- 原 `results/topk_l12_5/` 没有 `last.pt` / `history.csv`，说明中断发生在首次保存 checkpoint 前，无法恢复那段进度。
- 已新增 `save_every_steps`：正式配置默认每 100 个 optimizer step 写 `last.pt`，并在训练 epoch 结束、进入验证前额外保存一次，降低长 epoch 白跑风险。

---

## 5. 待用户决策的开放性问题（按优先级）

1. **正式训练的字典维度 d_sae**：默认 16384（16x），是否要试 32768（32x，特征更细）或 8192（8x，更密集）？
2. **训练规模**：默认 max_epochs=200 ≈ 800M token；SAE 文献常用 1B~10B token，你能接受多长训练？
3. **JumpReLU 的 sparsity_coef**：默认 1e-3 是 DeepMind 论文起点，但要根据 d_sae 规模调；通常需要扫 1e-4 ~ 1e-2。
4. **是否要写评估脚本**：`src/evaluate.py` 计算 KL substitution（用 SAE 重构替换原激活后看 logits KL），是 SAE 质量金标准。当前未写。
5. **是否要写可视化**：从 `results/overall_config_metrics.csv` 出 loss/L0 对比图，方便扫超参。当前未写。

---

## 6. 关键文件速查

- 训练入口：`src/train.py`
- SAE 模型：`src/models/sae_topk.py`、`src/models/sae_jumprelu.py`
- Qwen 加载/Hook：`src/models/qwen_loader.py`
- 训练循环：`src/training/trainer.py`
- 数据流：`src/data/corpora.py`、`src/data/activation_store.py`
- 工具：`src/utils/exp_dir.py`、`src/utils/overall_metrics.py`
- 完整使用文档：[`GUIDE/quickstart.md`](../GUIDE/quickstart.md)
- 项目级约束（CLAUDE.md 规则）：[`feedback_workflow.md`（持久化记忆）]
