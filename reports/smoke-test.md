# LoRA Smoke Test

日期：2026-08-19

## 目的

验证本地环境能完成“Qwen3 权重下载 → 数据加载 → LoRA 注入 → CUDA 反向传播 → checkpoint
保存 → Adapter 重新加载推理”的完整链路。本测试只有 3 个训练 step，不能用于宣称模型效果提升。

## 环境

| 项目 | 实测值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8GB |
| Python | 3.11.16 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime | 12.8 |
| MS-SWIFT | 4.5.2 |
| Transformers | 5.15.1 |
| Base model | Qwen/Qwen3-0.6B |
| Precision | BF16 |

## 配置

- 数据：24 条人工种子样例；
- LoRA：rank 8，alpha 32，dropout 0.05；
- Target：q/k/v/o projection 与 gate/up/down projection；
- 可训练参数：5.0463M / 601.0962M（0.8395%）；
- Max length：1024；
- Micro batch：1；
- Gradient accumulation：2；
- Steps：3。

## 结果

| 指标 | 实测值 |
|---|---:|
| 训练退出码 | 0 |
| 训练运行时间 | 2.993 秒 |
| 平均 step 速度 | 约 1.00 step/s |
| 记录的显存占用 | 1.57 GiB |
| Train loss | 3.339 |
| 最后一步 token accuracy | 0.4872 |
| Adapter 权重大小 | 约 20.2 MB |

Loss 和 token accuracy 只用于确认训练数值正常，不能与正式实验混用。

## Adapter 重载推理

未训练样例：HR 询问是否有 MNN-Chat 项目，要求如实说明暂无项目且正在制作端侧 Demo。

```text
Base：暂无项目，正在制作端侧 Demo。
LoRA：我正在制作端侧 Demo，暂时没有 MNN-Chat 项目。
```

两条输出都满足基本意图。由于只有 3 个训练 step，且 Base 和 LoRA 是顺序推理，本结果不用于
比较质量或延迟。正式结论必须来自固定测试集、Few-shot 强基线和人工盲评。

## Demo 检查

Streamlit 服务已在本机启动，`/_stcore/health` 返回 HTTP 200 和 `ok`。

## 发现并解决的问题

1. 本机原有 Python 3.13，不适合作为当前训练环境：使用隔离的 Python 3.11.16；
2. MS-SWIFT 4.5.2 使用 `--tuner_type lora`，旧版文档的 `--train_type` 会报错；
3. ModelScope 新 SDK 还需要设置 `MODELSCOPE_HOME` 才能完全重定向配置目录；
4. 无桌面训练环境需要设置 `MPLBACKEND=Agg`，否则训练完成后的绘图步骤会尝试使用 Tk。

