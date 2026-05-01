# `src/data/` 模块记忆

## 文件
- `corpora.py` — 中英 streaming 文本流：FineWeb-Edu + Chinese-Fineweb-Edu 按 ratio 混合；HF 不可用时回退 wikitext
- `activation_store.py` — 滚动激活缓冲：tokenize → forward Qwen → mask padding → 打乱 → 切片吐 SAE batch
- `download_subset.py` — 从 HF streaming 抽取固定中英子集到本地 JSONL，写 manifest 与本地训练配置

## 关键约定

### 本地固定子集
- 推荐正式长跑前运行 `python src/data/download_subset.py ...`，生成 `train_en.jsonl`、`train_zh.jsonl`、`val_en.jsonl`、`val_zh.jsonl`。
- 每行 JSONL 格式为 `{"text": "...", "lang": "en|zh"}`。
- 脚本额外写 `manifest.json`、`corpus_local.yaml` 和完整 `train_config_local.yaml`，后者可直接交给 `src/train.py`。
- 训练加载器支持两种本地入口：单文件 `local_train_path` / `local_val_path`，以及中英文分文件 `local_en_*_path` / `local_zh_*_path`。
- 本地文件读取按行循环，不一次性加载全文件，适合较大的固定子集。

### HF streaming
- 正式配置 `streaming=true`：不会预先下载完整语料，一个 epoch 后仍会继续依赖 HF streaming 读取后续样本。
- 默认只读取 `text` 列，避免把不参与训练的元数据列读入 PyArrow batch。
- `corpus.streaming_batch_size=1024`：限制 Parquet streaming record batch 行数，降低 row group 过大导致的 CPU 内存峰值。
- 已移除数据集加载里的 `trust_remote_code=True`，适配新版 `datasets` 对 dataset loading script 的限制。

### 缓冲流转
- buffer dtype = bfloat16（省显存），吐给 SAE 时 cast float32（精度）
- buffer 大小默认 524288 tokens（≈ 0.5M），消费到 50% 以下触发 refill
- refill 时整体打乱，避免相邻 batch 相关
- padding token 通过 attention_mask 过滤掉，不进缓冲

### token 形状
- text 流 yield 字符串；tokenizer 在 store 内做（不需要数据集已 tokenize）
- ctx_len 默认 512；过长截断
- 一次模型 forward batch_size = `model_batch_size` 句

### 故障回退
- HF 网络/数据集名失败 → 回落 wikitext-2-raw-v1（必须能离线缓存）
- 文本流耗尽 → 自动重启同一个 dataset 流
- streaming 训练中途网络断开或远程分片读取失败时，未必能自动切换到 wikitext；正式长跑建议设置 `HF_TOKEN` 或准备本地语料。
