# IM-Reply-Qwen Profile LoRA Model Card

## 状态

Adapter 已完成训练和评测，并发布至
[ModelScope：Say1kU/im-reply-qwen-lora](https://modelscope.cn/models/Say1kU/im-reply-qwen-lora)。
GitHub 仓库只提交代码、数据生成脚本、固定测试集和真实评测结果，不提交基础模型或本地
checkpoint。

## 基座与训练

- Base model：Qwen3-0.6B；
- Method：LoRA SFT；
- Framework：MS-SWIFT 4.5.2、Transformers 5.15.1、PEFT 0.19.1；
- Training data：96 条确定性合成招聘沟通样例；
- Epochs / steps：3 / 36；
- LoRA：rank 8、alpha 32、dropout 0.05、all-linear；
- Precision：BF16；
- Learning rate：1e-4，cosine scheduler；
- Effective batch：1 × gradient accumulation 8；
- Max length：1024；
- Hardware：NVIDIA GeForce RTX 4060 Laptop GPU（8GB）；
- Training time：103 秒；训练器记录峰值显存 1.57 GiB；
- Final training loss：0.7956。

训练 Loss 仅用于确认数值收敛，模型选择同时参考独立固定测试集的失败案例。最终 Demo 选择
checkpoint-36，而不是只按最低 Loss 选择。

## 预期用途

根据中文 IM 历史对话、回复目标、可用事实、语气和长度限制生成一条回复草稿。当前 Profile
版本重点展示招聘沟通中的真实经历引用和“不夸大经验”。输出应由用户确认后发送。

## 非预期用途

- 不用于自动代表用户发送消息；
- 不用于自动招聘筛选或评价候选人；
- 不用于法律、医疗、金融等高风险决策；
- 不应用来推断年龄、性别、民族、健康状况等受保护属性；
- 不保证事实正确，尤其在输入缺少信息或遭受对抗提示时。

## 评测

16 条未参与训练的固定招聘样例结果：

| 模型 | 规则全通过率 | 目标信息通过率 | 显式事实一致率 |
|---|---:|---:|---:|
| 原始 Qwen3-0.6B | 68.75% | 75.00% | 100.00% |
| 旧 3-step LoRA | 87.50% | 87.50% | 100.00% |
| Profile LoRA checkpoint-36 | **93.75%** | **93.75%** | 100.00% |

逐条输出、延迟、选择理由和失败案例见 `reports/profile-eval.md` 与
`reports/profile-eval.json`。

## 数据与局限

- 数据是围绕六条公开技能事实构造的合成对话，不是真实聊天记录；
- 96 条训练数据和 16 条测试数据规模很小，不能代表生产环境；
- 自动规则主要检查关键词、长度和显式冲突，不能替代语义级事实核验；
- 当前没有完成至少 50 条样例的人工盲评；
- 主要覆盖简短中文招聘沟通，长上下文、方言和对抗输入需要另行评测；
- Adapter 发布时仍需遵循具体 Qwen3 基座模型的许可证与 Model Card。
