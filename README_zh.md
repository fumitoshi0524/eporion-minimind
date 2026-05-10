# Eiporion-MiniMind

基于 [MiniMind3](https://github.com/jingyaogong/minimind) 架构的轻量级 GPT 风格语言模型，将线性层替换为 **BitLinear**（int8 量化权重），并使用 **Eiporion** 可微量化训练框架进行训练。

## 概述

Eiporion-MiniMind 是一个全 int8 量化的 LLaMA 风格 decoder-only Transformer。所有注意力投影层（Q、K、V、O）和前馈网络投影层（gate、up、down）均使用 `BitLinear` 层，权重为 int8 精度，并配有逐行 float32 缩放因子。仅最终的 LM head 保留为标准 `nn.Linear`。

训练采用**可微量化训练（Differentiable Quantization Training, DQT）**——通过 8-bit 分块 Adam 梯度的随机取整（stochastic rounding），直接更新 int8 权重，由 `EiporionOptim` 优化器驱动。

| 属性 | 值 |
|----------|-------|
| 参数量 | ~64M |
| 隐藏层维度 | 768 |
| 层数 | 8 |
| 注意力头数 | 8（4 个 KV 头，GQA） |
| 头维度 | 96 |
| 中间层维度 | 2432（SwiGLU） |
| 词表大小 | 6400 |
| 最大序列长度 | 32768（支持 YaRN 扩展） |
| 权重量化精度 | int8（BitLinear） |
| 优化器 | EiporionOptim（8-bit Adam + 随机取整） |
| 训练数据 | pretrain_t2t_mini（MiniMind） |

## 架构

![模型结构](img/Model_Structure.png)

- **分组查询注意力（GQA）**，带 QK-Norm（对 query 和 key 头进行 RMSNorm）
- **RoPE** 位置编码，支持 YaRN 缩放以处理长上下文
- 前馈网络使用 **SwiGLU** 激活函数
- Embedding 与 LM head **权重共享**
- 所有线性投影层使用 **BitLinear** int8 权重；仅 lm_head 使用 float32

## 训练

```bash
python trainer/train_pretrain.py
```

关键超参数：
- 训练轮数：2
- 批次大小：64（梯度累积步数：4）
- 学习率：2e-4（余弦调度，预热 8000 步）
- 最大序列长度：512
- 权重衰减：0.1
- 梯度裁剪：1.0

### 训练损失曲线

![训练损失](img/train_loss.png)

## 评测结果

### C-Eval（0-shot）

| 科目 | 准确率 |
|------|-----|
| 美术学 | 45.45% |
| 离散数学 | 37.50% |
| 高中生物 | 36.84% |
| 导游资格 | 34.48% |
| 毛泽东思想 | 33.33% |
| 中学政治 | 33.33% |
| 植物保护 | 31.82% |
| **总体（52 个科目）** | **23.25%** |

### CMMLU（0-shot）

| 科目 | 准确率 |
|------|-----|
| 大学教育 | 31.78% |
| 大学工程水文学 | 30.19% |
| 小学数学 | 28.26% |
| 食品科学 | 27.97% |
| 艺术 | 27.50% |
| 高中化学 | 27.27% |
| **总体（67 个科目）** | **25.30%** |

### 通用评测（0-shot）

| 基准 | Acc | Acc Norm |
|-----------|-----|----------|
| ARC-Easy | 28.45% | 28.32% |
| HellaSwag | 27.32% | 28.58% |
| PIQA | 52.23% | 50.82% |
| WinoGrande | 52.64% | — |

## 评估

交互式对话评测：

```bash
python eval_llm.py
```

使用 lm-eval 批量评测：

```bash
lm_eval --model hf \
  --model_args pretrained=out/,trust_remote_code=True \
  --tasks ceval-valid,cmmlu,arc_easy,hellaswag,piqa,winogrande
```

## 导出 HuggingFace 格式

```bash
python scripts/export_hf.py
```

将 `.pth` 检查点导出为 `model.safetensors`，配置文件保存在 `out/` 目录。

## 项目结构

```
eporion-minimind/
├── model/
│   ├── model_eiporion.py    # 模型架构
│   ├── tokenizer.json       # 分词器（6400 词表，ChatML 格式）
│   └── tokenizer_config.json
├── dataset/
│   └── llm_dataset.py       # 数据集类
├── trainer/
│   ├── train_pretrain.py    # 预训练脚本
│   └── train_util.py        # 训练工具函数
├── scripts/
│   └── export_hf.py         # 导出 HuggingFace 格式
├── eval_llm.py              # 交互式评估
├── img/
│   ├── Model_Structure.png  # 模型结构图
│   └── train_loss.png       # 训练损失曲线
├── out/                     # 训练好的模型产物
└── pyproject.toml
```

## 依赖

- Python >= 3.12
- PyTorch（CUDA 12.4）
- transformers
- [eiporion](https://github.com/fumitoshi0524/eiporion)（BitLinear + EiporionOptim）
- bitsandbytes
- lm-eval
- datasets
- accelerate

## 致谢

- 模型架构参考 [MiniMind](https://github.com/jingyaogong/minimind)
- BitLinear 量化和 EiporionOptim 来自 [Eiporion](https://github.com/fumitoshi0524/eiporion) 项目
- 使用 MiniMind 的 pretrain_t2t_mini 数据集训练
