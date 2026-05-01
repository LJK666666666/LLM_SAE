# 工作记忆索引

本文件汇总所有 memory 文件位置。其余子文件夹中可有局部 `memory.md` 模块化补充。

## 全局记忆（_memory/）

- [`user.md`](user.md) — **首次进入项目优先看这个**：主要命令、配置位置、Claude 代做的开放性决策、环境陷阱、待决问题
- [`design_decisions.md`](design_decisions.md) — SAE 训练的关键设计选择与理由（hook 位置、SAE 变体、库选择、语料）
- [`architecture.md`](architecture.md) — Qwen3.5-0.8B 模型架构关键参数与文本子模块定位
- [`progress.md`](progress.md) — 全过程进度与里程碑
- [`naming_conventions.md`](naming_conventions.md) — 实验目录命名 / 配置 tag 规则

## 局部记忆（各模块 `memory.md`）

- `src/models/memory.md` — SAE 实现要点与公式
- `src/data/memory.md` — 数据流与激活缓冲设计
- `src/training/memory.md` — 训练/评估超参数说明
