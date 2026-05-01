# 快速开始

## 1. 环境

```bash
pip install -r requirements.txt
# 关键：transformers 必须 ≥ 5.6.0.dev（当前 release 4.57.x 不识别 model_type=qwen3_5）
pip install "git+https://github.com/huggingface/transformers.git"
```

> 已知环境陷阱：若 `datasets` 触发 `nibabel` 导入 → `np.sctypes` 报错（NumPy 2.x 不兼容 nibabel <5.x），可
> `pip install --upgrade nibabel` 或 `pip uninstall -y nibabel`（datasets 不依赖它，仅 nifti 可选特性需要）。

## 2. 端到端烟雾测试（验证整条链路）

```bash
# 单 batch 验证（约 30s 完成，加载模型主要耗时）
python src/train.py --config configs/train_topk_smoke.yaml --tag smoke --max-iters 1

# 完整 2-epoch 烟雾（用本地 .txt，离线即可跑）
python src/train.py --config configs/train_topk_smoke.yaml --tag smoke

# 从 last.pt 恢复
python src/train.py --config configs/train_topk_smoke.yaml --tag smoke --resume --max-epochs 4
```

## 3. 正式训练 TopK SAE（中英混合数据）

需要 HF 网络可达 + datasets 可正常导入。

### 3.1 先下载固定本地子集（推荐长跑）

```bash
python src/data/download_subset.py --config configs/train_topk.yaml \
    --output-dir data/local_corpus/fineweb_edu_subset \
    --en-train-docs 100000 --zh-train-docs 100000 \
    --en-val-docs 5000 --zh-val-docs 5000
```

输出文件：

- `data/local_corpus/fineweb_edu_subset/train_en.jsonl`
- `data/local_corpus/fineweb_edu_subset/train_zh.jsonl`
- `data/local_corpus/fineweb_edu_subset/val_en.jsonl`
- `data/local_corpus/fineweb_edu_subset/val_zh.jsonl`
- `data/local_corpus/fineweb_edu_subset/manifest.json`
- `data/local_corpus/fineweb_edu_subset/corpus_local.yaml`
- `data/local_corpus/fineweb_edu_subset/train_config_local.yaml`

然后用本地子集训练：

```bash
python src/train.py --config data/local_corpus/fineweb_edu_subset/train_config_local.yaml \
    --tag topk_l12_local
```

也可以把 `corpus_local.yaml` 中的路径复制到 `configs/train_topk_local.yaml`。

### 3.2 直接 HF streaming 训练（适合短跑/调试）

```bash
python src/train.py --config configs/train_topk.yaml --tag topk_l12 \
    --d-sae 16384 --k 32

# 调字典大小 / lr / 训练长度
python src/train.py --config configs/train_topk.yaml --tag topk_l12_d32k \
    --d-sae 32768 --k 64 --lr 1e-4 --steps-per-epoch 2000 --max-epochs 50
```

### Colab / 云端保存到 Google Drive

先挂载 Drive，然后把结果根目录改到 Drive 下：

```python
from google.colab import drive
drive.mount("/content/drive")
```

若项目位于 `/content/LLM_SAE`，推荐使用相对路径，符合项目可迁移性约定：

```bash
python src/train.py --config configs/train_topk.yaml --tag topk_l12 \
    --results-root ../drive/MyDrive/results
```

长时间训练建议先把固定子集下载到 `/content/data/...`，训练时读 `/content` 本地盘，结果再写入 Drive：

```bash
python src/data/download_subset.py --config configs/train_topk.yaml \
    --output-dir ../data/fineweb_edu_subset \
    --en-train-docs 100000 --zh-train-docs 100000 \
    --en-val-docs 5000 --zh-val-docs 5000

python src/train.py --config configs/train_topk_local.yaml --tag topk_l12_local \
    --results-root ../drive/MyDrive/results
```

也可以直接使用脚本生成的配置：

```bash
python src/train.py --config ../data/fineweb_edu_subset/train_config_local.yaml \
    --tag topk_l12_local --results-root ../drive/MyDrive/results
```

也可以显式使用 Colab 绝对路径：

```bash
python src/train.py --config configs/train_topk.yaml --tag topk_l12 \
    --results-root /content/drive/MyDrive/results
```

恢复训练时必须使用同一个 `--results-root`：

```bash
python src/train.py --config configs/train_topk.yaml --tag topk_l12 \
    --results-root ../drive/MyDrive/results --resume
```

## 4. JumpReLU SAE

```bash
python src/train.py --config configs/train_jumprelu.yaml --tag jumprelu_l12
```

## 5. 离线 SAE 单元测试

```bash
python tests/test_sae_smoke.py
```

## 6. 输出位置

- `{results_root}/{tag}_{n}/last.pt` — 最近 epoch 权重（含 optimizer，供 `--resume`）
- `{results_root}/{tag}_{n}/best.pt` — 最佳 val_recon 权重
- `{results_root}/{tag}_{n}/history.{csv,json}` — 每 epoch 详细指标
- `{results_root}/{tag}_{n}/config.yaml` — 实验完整配置快照
- `{results_root}/{tag}_{n}/metrics_final.json` — 训练完成后的汇总
- `{results_root}/overall_config_metrics.{csv,json}` — 所有实验汇总

## 7. 常用命令行覆盖

| 标志 | 默认 | 说明 |
|---|---|---|
| `--lr` | yaml | Adam 学习率 |
| `--d-sae` | yaml | 字典维度 |
| `--k` | yaml | TopK 稀疏度 |
| `--steps-per-epoch` | yaml | 每"epoch" SAE batch 数 |
| `--max-epochs` | yaml | 总 epoch 上限 |
| `--results-root` | `results` | 实验结果根目录；Colab 推荐 `../drive/MyDrive/results` |
| `--max-iters N` | None | 跳过完整训练，只跑 N 个 batch 用于快速验证 |
| `--device` | yaml | cuda / cpu / auto |
| `--resume` | False | 从同 tag 最大编号目录的 last.pt 恢复 |
