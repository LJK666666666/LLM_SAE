# Qwen3.5-0.8B 架构关键参数

> 信息来源：`Qwen3.5-0.8B/config.json`

## 模型类
- `Qwen3_5ForConditionalGeneration`（多模态：text + vision）
- model_type: `qwen3_5`

## 文本侧 (text_config)
| 参数 | 值 |
|---|---|
| hidden_size | **1024** ← SAE 输入维度 |
| num_hidden_layers | **24** |
| intermediate_size | 3584 |
| num_attention_heads | 8 |
| num_key_value_heads | 2 |
| head_dim | 256 |
| vocab_size | 248320 |
| dtype | bfloat16 |
| max_position_embeddings | 262144 |
| tie_word_embeddings | true |

## 注意力层模式 (layer_types, 24 层)
```
linear linear linear FULL
linear linear linear FULL
linear linear linear FULL
linear linear linear FULL
linear linear linear FULL
linear linear linear FULL
```
即 4 层一组：3 linear + 1 full。第 12 层（0-index）属于 `linear_attention`。
SAE 钩取的是 decoder block 整体输出（残差流），与注意力类型无关。

## 视觉侧 (vision_config)
- depth=12, hidden_size=768, out_hidden_size=1024
- 训练 SAE 时不调用，但权重需加载在显存中
