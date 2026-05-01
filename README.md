# LLM_SAE: Qwen3.5-0.8B 稀疏自编码器训练

针对本地多模态大模型 `Qwen3.5-0.8B/` 的语言模型部分训练 Sparse Autoencoder（SAE）以提取可解释特征。

## 设计概要

| 项目 | 选择 |
|---|---|
| Hook 位置 | 第 12 层 decoder block 输出（残差流，24 层中点） |
| SAE 变体 | TopK SAE（OpenAI 2024）/ JumpReLU SAE（DeepMind 2024） |
| 参考库 | [SAELens](https://github.com/jbloomAus/SAELens)（算法参考，自行精简实现） |
| 训练语料 | 中英混合（FineWeb-Edu + Chinese-Fineweb-Edu，HF streaming） |

> 注：因 `transformer_lens` 不支持 Qwen3.5（混合 linear/full attention 的新多模态架构），未直接复用 SAELens 的 `HookedSAETransformer`，而是用 `transformers` 原生 + `register_forward_hook`。

## 目录结构

```
LLM_SAE/
├── src/                  # 源码
│   ├── models/           # qwen_loader, sae_topk, sae_jumprelu
│   ├── data/             # 双语语料 + 激活缓冲
│   ├── training/         # trainer, metrics
│   ├── utils/            # 实验目录、全局指标
│   ├── train.py          # 训练入口
│   └── evaluate.py       # 评估入口
├── configs/              # 训练 yaml
├── results/{tag}_{n}/    # 训练产物（默认根目录，可用 --results-root 改到云盘）
├── data/cache/           # HF 数据缓存
├── _memory/              # 工作记忆文档
├── GUIDE/                # 使用与设计说明
├── Qwen3.5-0.8B/         # 本地模型权重
└── tests/
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 单 batch 快速验证（验证整条链路）
python src/train.py --config configs/train_topk.yaml --max-iters 1 --tag debug

# 3. 正式训练 TopK SAE
python src/train.py --config configs/train_topk.yaml --tag topk_l12

# 3a. 正式长跑推荐：先下载固定本地子集，再训练
python src/data/download_subset.py --config configs/train_topk.yaml \
    --output-dir data/local_corpus/fineweb_edu_subset \
    --en-train-docs 100000 --zh-train-docs 100000 \
    --en-val-docs 5000 --zh-val-docs 5000
python src/train.py --config data/local_corpus/fineweb_edu_subset/train_config_local.yaml \
    --tag topk_l12_local

# 4. 从 last 恢复
python src/train.py --config configs/train_topk.yaml --tag topk_l12 --resume

# Colab/云端：保存到 Google Drive（项目位于 /content/LLM_SAE 时）
python src/train.py --config configs/train_topk.yaml --tag topk_l12 \
    --results-root ../drive/MyDrive/results
```

详细说明见 `GUIDE/`。
