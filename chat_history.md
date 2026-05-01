# 对话历史

## 2026-04-16 #1

**用户**：针对 Qwen3.5-0.8B 模型训练一个 SAE 模型，参考或直接使用相关开源代码库。

**关键决策**（通过 AskUserQuestion 与用户确认）：
- Hook 位置：第 12 层 decoder block 输出（残差流，24 层中点）
- SAE 变体：先实现 TopK SAE 跑通，再加 JumpReLU SAE
- 参考库：SAELens（但因 transformer_lens 不支持 Qwen3.5 多模态架构，决定**自行精简实现** SAE 类，仅以 SAELens 为算法参考）
- 训练语料：中英混合（FineWeb-Edu + Chinese-Fineweb-Edu 类）

**警示已告知用户**：
1. transformer_lens 大概率不支持 Qwen3.5 → 改用 transformers 原生 + register_forward_hook
2. 多模态模型仅取 language_model 子模块训练文本 SAE

**本次工作内容**：
- 建立完整项目骨架 `src/{models,data,training,utils} configs/ results/ _memory/ GUIDE/ tests/`
- 记忆系统：`_memory/{index,design_decisions,architecture,naming_conventions,progress}.md` 与各模块 `memory.md`
- 模型层：`qwen_loader.py`（多模态文本子模块定位 + L12 forward hook）；`sae_topk.py`（含 aux_k_loss、dead-feature 计数、decoder 列归一化）；`sae_jumprelu.py`（STE 矩形核、可学每特征阈值）
- 数据层：`corpora.py`（双语 HF streaming + wikitext + 本地 txt 三种回退）；`activation_store.py`（滚动缓冲 + 整体打乱 + padding mask）
- 训练层：`trainer.py`（epoch/lr scheduler/early stop/best-last ckpt/resume/`--max-iters` 单 batch 模式）；`metrics.py`（KL/explained variance）
- 工具：`exp_dir.py`（自动 `results/{tag}_{n}/`）、`overall_metrics.py`、`config.py`
- 入口：`src/train.py`；配置：`configs/train_{topk,jumprelu,topk_smoke}.yaml`
- 离线 SAE 单元测试：`tests/test_sae_smoke.py` 5/5 通过
- **环境处理**：用户确认后从 source 升级 `transformers` → `5.6.0.dev0`（release 4.57.1 不识别 `qwen3_5` model_type）
- **端到端验证**（results/smoke_2/）：2 epoch + resume 后 val_recon 5.44→4.08 持续下降，expl_var 0.10→0.24，证明链路正确
- 修复 resume 后 epoch 编号复用 bug（`state.epoch` 语义改为"已完成 epoch 数"）；在 smoke_3/ 重新验证通过，epoch 1→2→3→4 连贯

## 2026-04-16 #2

**用户**：在 `_memory/` 维护一个 `user.md`，记录主要命令、配置位置、Claude 自行做出的决策等。

**工作内容**：
- 写 `_memory/user.md`：包含 6 节
  1. 主要运行命令（离线测试 / 烟雾 / 正式训练 / 单文件验证）
  2. 配置参数位置（高频走命令行表格 / 低频走 yaml 段说明）
  3. Claude 代做的 8 项开放性决策（含 why、影响、可换方案）：epoch 语义、SAE fp32 / 激活 bf16、epoch 折算、验证集策略、decoder 归一化、中英 ratio、hook 层位置、smoke 文本
  4. 环境陷阱（transformers 已升级 / nibabel 待处理 / fast path 警告）
  5. 待用户决策的开放性问题 5 项
  6. 关键文件速查
- 更新 `_memory/index.md` 把 user.md 标为"首次进入项目优先看"
- **已知环境陷阱**：`datasets` import 触发 `nibabel` → `np.sctypes` 报错（NumPy 2.x），LocalTextFileStream 路径绕过；正式 HF 数据训练前需 `pip uninstall -y nibabel` 或升级
- 文档：`GUIDE/quickstart.md` 完整命令示例；`README.md` 顶层介绍

## 2026-04-29 #1

**用户**：正式运行 `python src/train.py --config configs/train_topk.yaml --tag topk_l12` 时连续询问依赖报错、HF streaming 提示、数据下载位置、streaming 是否会在 epoch 后完成下载，以及训练中断含义。

**关键诊断**：
- `datasets -> nibabel -> numpy` 导入报错原因：旧版 `nibabel 3.2.0` 调用 NumPy 2.x 已移除的 `np.sctypes`；建议升级 `nibabel` 或降级 `numpy<2`。随后正式训练已越过该错误。
- `trust_remote_code is not supported anymore` 来自新版 `datasets`，数据集已按标准 Parquet 解析，属于警告；项目代码后续移除该参数。
- HF streaming 不会预下载完整 FineWeb；一个 epoch 后仍会继续远程读取样本，训练中可能受网络、限流和远程分片读取影响。
- `pyarrow.lib.ArrowMemoryError: realloc of size 2449473536 failed` 发生在中文 FineWeb streaming 验证阶段，表示 PyArrow 读取 Parquet batch 时 CPU 内存申请约 2.45GB 失败；同时出现远程数据主机断连重试。
- 检查 `results/topk_l12_5/` 发现没有 `last.pt` / `best.pt` / `history.csv`，说明中断发生在首次 checkpoint 前，已训练进度没有落盘，不能 `--resume`。

**代码修改**：
- `src/data/corpora.py`：移除 `trust_remote_code=True`；只读取 `text` 列；新增 `CorpusConfig.streaming_batch_size` 并传给 HF Parquet loader，默认 1024，降低 streaming 内存峰值。
- `src/training/trainer.py`：新增 `TrainerConfig.save_every_steps`；训练中每 N 个 optimizer step 覆盖保存 `last.pt`；训练阶段结束、进入验证前额外保存一次；修正最终 summary 的 `final_epoch`。
- `src/train.py`：新增命令行参数 `--save-every-steps`，并传入 trainer。
- `configs/train_topk.yaml` / `configs/train_jumprelu.yaml`：新增 `streaming_batch_size: 1024` 和 `save_every_steps: 100`。

**验证**：
- `python -m compileall src\data\corpora.py src\train.py src\training\trainer.py` 通过。
- 联网取首条 HF 样本测试在 120 秒内超时，卡在 HF 访问/缓存权限提示，但未出现 `columns` / `batch_size` 参数不支持。

## 2026-04-29 #2

**用户**：给出代码编写要求，并要求把之前工作内容更新到 `_memory/` 和 `chat_history.md`。

**工作内容**：
- 更新 `_memory/user.md`：加入正式训练恢复命令、`--save-every-steps`、`corpus.streaming_batch_size`、HF streaming 不会完整预下载、`topk_l12_5` 无 checkpoint 不可恢复、PyArrow 内存错误与自动保存策略。
- 更新 `src/data/memory.md`：记录只读 `text` 列、`streaming_batch_size=1024`、移除 `trust_remote_code`、streaming 中途网络风险。
- 更新 `src/training/memory.md`：记录 `save_every_steps`、验证前 checkpoint，以及 epoch 中途恢复的权重优先/统计可能重复的取舍。
- 更新 `_memory/progress.md` 与 `_memory/design_decisions.md`：补充 2026-04-29 的诊断、修改和设计取舍。

## 2026-04-29 #3

**用户**：询问当前代码的内存请求是否稳定，以及之前内存问题更可能是代码问题还是外部环境问题。

**回答要点**：
- Qwen 取激活路径使用 `@torch.no_grad()`，模型前向不会保留反向图；SAE、模型权重、激活缓冲的显存需求主要由 `buffer_size_tokens`、`sae_batch_size`、`model_batch_size`、`ctx_len` 决定，较可预测。
- 之前的 `ArrowMemoryError` 是 CPU/PyArrow/HF streaming 读 Parquet 时的内存申请失败，不是 CUDA 显存或 SAE 参数量问题。
- 归因上更像“代码的数据读取方式没有限制列和 batch size”与“外部数据分片/网络/本机内存状态”共同触发；不是单纯环境问题，也不是训练循环的典型内存泄漏。
- 现已通过 `columns=[text_col]` 和 `streaming_batch_size=1024` 降低风险，但 streaming 远程数据仍不可能做到完全稳定；必要时继续减小 `streaming_batch_size` 和 activation buffer。

## 2026-04-29 #4

**用户**：希望将代码库放到云端运行，并把实验结果保存到 `/content/drive/MyDrive/results` 或 `../drive/MyDrive/results`。

**工作内容**：
- 指出 `/content/drive/MyDrive/results` 是 Colab 常用绝对路径，但与用户既有“路径尽量相对”的规则冲突；推荐项目位于 `/content/LLM_SAE` 时使用 `../drive/MyDrive/results`。
- `src/utils/exp_dir.py`：将硬编码 `results` 改为可传入 `results_root`，实验目录变为 `{results_root}/{tag}_{n}/`。
- `src/utils/overall_metrics.py`：`append_record` 支持同一个 `results_root`，确保 `overall_config_metrics.{csv,json}` 与实验目录在同一根目录。
- `src/train.py`：新增 `--results-root`，默认 `results`；训练和恢复都按该 root 扫描/写入；指标记录包含 `results_root`。
- 更新 `GUIDE/quickstart.md`、`README.md`、`_memory/user.md`、`_memory/naming_conventions.md`、`_memory/progress.md`。
- 验证：`python -m compileall src\train.py src\utils\exp_dir.py src\utils\overall_metrics.py` 通过。

## 2026-04-29 #5

**用户**：询问实时读取网络数据训练是否是主流做法，以及为什么不先下载一个子集到本地再读取。

**回答要点**：
- 对超大语料而言，streaming 是常见工程手段，尤其用于快速试验、数据集远大于本地磁盘、或从对象存储连续读取；但正式长时间实验通常更偏向固定版本的本地/近端分片，而不是训练时依赖公共网络。
- 当前项目使用 HF streaming 的原因是 FineWeb 体量很大，能避免一次性下载和管理全量数据，快速开始训练。
- 固定下载一个子集到本地更适合本项目当前阶段：更可复现、更稳定、避免 HF 限流/断连/PyArrow 远程读取问题，也方便 resume 后保持同一数据分布。
- 建议策略：调试继续 streaming；正式 SAE 长跑先抽取固定中英文子集到本地 JSONL/Parquet/TXT，再从本地流式读取；Colab 中建议训练时把数据放 `/content` 本地盘，结果保存到 Drive。

## 2026-04-29 #6

**用户**：要求添加代码，实现将数据集子集下载到本地，以及从本地加载数据训练。

**工作内容**：
- `src/data/download_subset.py`：新增数据准备脚本，从 `configs/train_topk.yaml` 读取 HF 数据集设置，通过 streaming 抽取固定数量的中英文 train/val 文档到本地 JSONL。支持 `--output-dir`、`--en-train-docs`、`--zh-train-docs`、`--en-val-docs`、`--zh-val-docs`、`--shuffle-buffer`、`--streaming-batch-size`、`--overwrite`，并显示 tqdm 进度条。
- 下载脚本输出 `train_en.jsonl`、`train_zh.jsonl`、`val_en.jsonl`、`val_zh.jsonl`、`manifest.json`、`corpus_local.yaml` 和可直接训练的 `train_config_local.yaml`。
- `src/data/corpora.py`：`LocalTextFileStream` 改为支持 `.txt` / `.jsonl` 按行循环读取，不一次性加载全文件；新增 `LocalBilingualTextStream`，支持本地中英文分文件按 `en_ratio` 混合采样。
- `src/train.py`：`build_text_streams` 支持 `local_train_path` / `local_val_path` 单文件入口，以及 `local_en_train_path` / `local_zh_train_path` / `local_en_val_path` / `local_zh_val_path` 分语言入口。
- `configs/train_topk_local.yaml`：新增本地固定子集训练模板。
- 更新 `GUIDE/quickstart.md`、`README.md`、`_memory/user.md`、`src/data/memory.md`、`_memory/progress.md`。

**验证**：
- `python -m compileall src\data\download_subset.py src\data\corpora.py src\train.py` 通过。
- `python src\data\download_subset.py --help` 正常显示参数。
- `LocalTextFileStream('data/smoke_text.txt')` 轻量读取通过。

## 2026-05-01 #1

**用户**：要求对 `_memory/user.md` 中的主要命令行进行同步更新，增加下载子集版。

**工作内容**：
- 重排 `_memory/user.md` 的“主要运行命令”部分：将正式训练改为优先推荐“先运行 `src/data/download_subset.py` 下载固定中英文 JSONL 子集，再用生成的 `train_config_local.yaml` 本地训练”。
- 增加本地子集版恢复训练、调参训练、提高 `save_every_steps` 的命令。
- 增加 Colab/云端推荐命令：数据下载到 `/content` 附近本地盘 `../data/fineweb_edu_subset`，实验结果通过 `--results-root ../drive/MyDrive/results` 保存到 Drive。
- 保留直接 HF streaming 训练作为短跑/调试备选，并明确训练期间仍依赖 HF streaming。
