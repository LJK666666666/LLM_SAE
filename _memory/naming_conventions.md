# 命名约定

## 实验目录：`{results_root}/{tag}_{n}/`

- `tag` 由命令行 `--tag` 指定，建议形如 `topk_l12_d16384`，`jumprelu_l12_d32768`。
- `results_root` 由命令行 `--results-root` 指定，默认 `results`。
- 云端/Colab 推荐使用相对路径 `../drive/MyDrive/results`；若用户明确指定，也支持 `/content/drive/MyDrive/results`。
- `{n}` 由 `src/utils/exp_dir.py` 自动扫描同 tag 已存在目录后 +1。
- 恢复训练（`--resume`）时写入已有最大编号目录，不创建新目录。
- 恢复训练时必须使用与原训练相同的 `--results-root`，否则会扫描另一个根目录并找不到 `last.pt`。

## 建议 tag 结构

```
{架构}_{hook层}_{字典维度}[_{扩展维度倍数}][_{数据}]
```

例：
- `topk_l12_d16384` — TopK，层 12，字典 16384 维（1024×16）
- `jumprelu_l12_d32768_ch` — JumpReLU，层 12，字典 32768 维，纯中文数据

## 文件命名（实验目录内部）

```
{results_root}/{tag}_{n}/
├── config.yaml           # 实际使用配置的快照
├── args.json             # 命令行参数快照
├── best.pt               # 最佳 val_loss 权重
├── last.pt               # 最近 epoch 权重（供 --resume）
├── history.csv           # 每 epoch 的 lr / train_loss / val_loss / L0 / dead_pct
├── history.json          # 同上 JSON 版
├── metrics_final.json    # 最终汇总指标
├── figures/              # L0、recon loss 曲线等
│   ├── loss_curve.png
│   └── l0_curve.png
└── eval/                 # 推理/评估产物
```
