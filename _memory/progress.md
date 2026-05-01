# 进度与里程碑

## 2026-04-16

- [x] 与用户确认 4 项核心设计：hook 位置（L12 残差流）、SAE 变体（TopK 优先 + JumpReLU）、参考库（SAELens）、语料（中英混合）
- [x] 建立项目骨架：`src/`, `configs/`, `results/`, `_memory/`, `GUIDE/`, `tests/`
- [x] 写记忆系统：`_memory/index.md`, `design_decisions.md`, `architecture.md`, `naming_conventions.md`, 本文件
- [x] 实现 `src/models/qwen_loader.py` 与 hook（L12 decoder block 输出）
- [x] 实现 `src/models/sae_topk.py`（含 aux_k_loss、dead-feature 计数、decoder 列归一化）
- [x] 实现 `src/models/sae_jumprelu.py`（STE 矩形核、可学每特征阈值）
- [x] 实现 `src/data/corpora.py`（双语 streaming + wikitext + 本地 txt 三种回退）
- [x] 实现 `src/data/activation_store.py`（滚动缓冲 + 整体打乱 + padding mask）
- [x] 实现 `src/training/trainer.py`（epoch / lr scheduler / early stop / best-last ckpt / resume）
- [x] 实现 `src/utils/exp_dir.py`、`overall_metrics.py`、`config.py`
- [x] 配置文件：`configs/train_topk.yaml`, `train_jumprelu.yaml`, `train_topk_smoke.yaml`
- [x] 入口：`src/train.py`
- [x] 离线 SAE 单元测试：`tests/test_sae_smoke.py`（5/5 通过）
- [x] 端到端 `--max-iters 1` 验证（results/smoke_1/）
- [x] 端到端完整 2-epoch + resume 验证（results/smoke_2/，loss 5.62→4.27 持续下降）
- [x] 升级 transformers 到 source（5.6.0.dev0），解决 qwen3_5 不识别问题
- [x] GUIDE/quickstart.md 编写完毕

待用户后续选择性推进：
- [ ] 中英混合 HF 数据正式训练（已越过 nibabel/numpy 2.x 导入问题；仍需关注 HF streaming 网络与内存稳定性）
- [ ] JumpReLU SAE 在真实 Qwen 上端到端跑（代码已就绪，未在大模型上跑过）
- [ ] 评估脚本 `src/evaluate.py`：KL substitution 指标
- [ ] 可视化脚本：loss/L0 曲线、特征激活分布

## 2026-04-29

- [x] 解释正式训练日志：HF `trust_remote_code` 不再支持、匿名请求限流、streaming 只解析数据分片列表，不会预下载完整语料
- [x] 确认数据缓存位置：`data/cache/` 基本为空；HF Hub 元信息缓存位于用户 Hugging Face cache；streaming 模式训练中仍会继续访问远程数据
- [x] 诊断 `ArrowMemoryError: realloc of size 2449473536 failed`：中文 FineWeb streaming 验证阶段 PyArrow 读取 Parquet batch 时 CPU 内存申请失败，并伴随远程数据主机断连重试
- [x] 确认 `results/topk_l12_5/` 无 `last.pt` / `history.csv`，中断发生在首次 checkpoint 前，因此无法从该次已训练进度恢复
- [x] 修改 `src/data/corpora.py`：移除 `trust_remote_code=True`；仅读取 `text` 列；新增 `streaming_batch_size` 限制 Parquet streaming batch 行数
- [x] 修改 `src/training/trainer.py` 与 `src/train.py`：新增 `trainer.save_every_steps` / `--save-every-steps`；训练中周期性覆盖保存 `last.pt`；进入验证前额外保存
- [x] 更新 `configs/train_topk.yaml` / `configs/train_jumprelu.yaml`：默认 `streaming_batch_size: 1024`、`save_every_steps: 100`
- [x] 语法检查通过：`python -m compileall src\data\corpora.py src\train.py src\training\trainer.py`
- [x] 支持云端结果路径：`src/train.py` 新增 `--results-root`；`src/utils/exp_dir.py` 与 `src/utils/overall_metrics.py` 改为按指定 root 保存实验目录和 overall 指标；文档记录 Colab 推荐 `../drive/MyDrive/results`
- [x] 支持本地固定子集：新增 `src/data/download_subset.py`，可从 HF streaming 下载固定中英文 train/val JSONL；`corpora.py` 支持本地 JSONL/TXT 按行循环读取和本地中英混合；新增 `configs/train_topk_local.yaml`
