# 七工具轨迹 LoRA SFT

2026-10-28，SFT 已完成：512 条、1 epoch、128 次更新，退出码 0、complete=true，无重复样本或未确认操作。最终 Sampler 真实图文采样成功，最终 State 经带优化器接口恢复后固定样本 NLL 与保存前完全一致。聚合证据见 [核验结果](evidence/sft-seven-v2.json)。正式 SFT 100 题评测和 RL 尚未启动；原 Base 评测继续，定时任务保持暂停。

输出为 `outputs/sft-seven-v2`，启动记录 `outputs/sft-seven-v2-launch.json`，日志/退出码 `logs/sft-seven-v2.log`、`logs/sft-seven-v2.exit`。v2 继承 v1 已确认的前八步和完整双权重，从第八步 State 恢复，补齐验证后继续第九步；迁移记录在同目录 migration.json。不要重新启动已完成的训练。

v1 在创建保存后的 Sampler 客户端时遗漏 SDK 必填的 base_model，尚未发送采样请求即抛 TypeError。已显式传入 base_model 与 model_path；已确认更新、两个 checkpoint 路径、账本和原源码快照完整保留，数据、训练顺序、超参数不变，没有重训前八步。该问题不涉及 Base 正在使用的源码。

| 最终记录 | 实测值 |
| --- | ---: |
| 训练输入 / 监督 tokens | 4626267 / 420240 |
| 128 次更新累计端到端耗时 | 664.334 秒（约 11 分钟，未包含准备、保存和恢复验证） |
| 单次更新耗时中位数 | 5.14 秒 |
| 固定训练样本初始 / 最终 NLL | 0.556990 / 0.432566 |
| 最终 State 恢复后同样本 NLL | 0.432566，与保存前一致 |
| 含诊断的输入/输出额度预留总量 | 4716275 tokens，未达到 6M 保护上限 |
| 已完成远程操作 | 128 次更新、8 次 forward、各 5 次 State/Sampler 保存、2 次采样验证 |

固定样本 NLL 降低约 22.3%，这是训练样本上的拟合记录，不能当作正式评测准确率提升。预留总量包括采样请求的最大输出额度，不等于实际账单；人民币费用尚未核对。最终两个真实路径保存在服务器 `outputs/sft-seven-v2/sft_result.json`，Mac 私有副本为 `outputs/sft-seven-v2-result.json`；不提交权重本体或内部路径到 Git。

## 固定配置

| 项目 | 配置 |
| --- | --- |
| 基模 / SDK | Qwen/Qwen3.5-4B / pytrio 0.2.9 |
| LoRA | rank 32，seed 42；SDK train_attn / train_mlp / train_unembed 显式设为 True |
| 数据 | data/selected-seven-v1/sft.jsonl，固定 512 条，1 epoch |
| Batch / 更新数 | 4 / 128；seed 42 打乱，每条仅训练一次 |
| 学习率 | AdamW 2e-5，其余采用 SDK 默认值 |
| 长度检查 | 32768；固定选集实际最长 14531，不截断或丢弃 |
| Loss | cross_entropy，assistant 内容和结束标记参与监督；手动右移，auto_shift=False |
| 归一化 | SDK 逻辑批次 active-token 全局分母，不再额外手动缩放梯度；单独记录加权平均 NLL |
| 提交保护预算 | 6M 输入/请求输出预留 tokens，包含诊断；不是人民币计费上限 |
| 双权重 checkpoint | 第 8、32、64、96、128 步 |

SFT 输入是发布轨迹中的图片、历史思考、工具动作和观察，不重新执行搜索或 OCR。原 200 题候选池、现用 100 题和 128 道 RL 题均不改动。512 条的图片字节哈希、数据哈希、模板/转换代码及 tokenizer/processor 文件哈希在创建训练客户端前核验。

## 前八步与恢复验证

1. 离线构造全部 Datum、检查图片/长度和已通过的 CPU 审计，读取平台训练能力；不在数据检查前创建 LoRA 训练客户端。
2. 在固定训练样本上取得初始 forward loss，再完成原定前八个 batch（32 条）。
3. 等更新完成后，固定权重，保存 `save_state` 和 `save_weights_for_sampler`；两者都返回路径后才写完整 checkpoint 清单。
4. 使用 Sampler 权重做一次真实图文采样，检查远端输入 token 与本地一致。诊断输出最多 128 tokens，只验证加载，不属于正式 100 题评测。
5. 使用 `create_training_client_from_state_with_optimizer` 新建恢复客户端，固定样本 NLL 与保存前按 rtol/atol=1e-3 比较，验证恢复权重；第九步以后继续原定顺序。采用恢复优化器的正式接口，具体优化器指标也逐步留存。
6. loss 无效或前八步完全无可观测变化、采样/恢复验证失败时停止并保留证据。正常则继续至第 128 步；不增加 epoch、不重复前八步、不根据评测集成绩选 checkpoint。

`pilot-verification.json`、`restore-verification-0008.json` 和 `sampler-health-0008.json` 只有真实检查通过后才生成。训练样本上的 loss 变化不能证明评测准确率提升。

## 保存与错误处理

每个 checkpoint 的两种远程路径保存到 `checkpoint-NNNN.json`，最新完整的一对保存为 `latest-checkpoint.json`。最终 `sft_result.json` 包含 Sampler/State 两个路径、训练步数与 complete 标记。Sampler 只用于推理；State 用于训练，SFT 中断恢复优化器，未来 SFT→RL 则按新阶段目标只加载权重、重新初始化优化器。

每次远程更新、forward、保存或采样之前写 pending 与请求账本，完成后保存原始结果；发送后不确定的操作不会自动重发。已确认更新超过最新完整 checkpoint 时，也不能静默恢复旧权重再训练同一批。`--resume` 仅接受配置一致、两类路径齐全、进度与 checkpoint 完全相同、无未确认操作的状态。恢复更早 checkpoint 涉及重算时，必须先明确核对原记录。

`metrics.jsonl` 保存逐步 NLL、输入/监督 token、学习率、耗时、训练及优化器指标；`requests.jsonl` 与 `operations/` 保留各项请求和回包。原始图片、训练数据、完整回包和模型权重只留服务器，Mac 仅保留聚合证据。人民币费用以平台实际账单为准。

当前 Base 评测始终绑定原始基模；SFT 使用独立 LoRA 状态，不修改其执行代码、提示词、模型路径和输出目录。共享服务排队可能影响耗时，不能把并行时的耗时差异直接解释为模型训练带来的加速。

## 运行命令

以下为已完成作业的历史命令，仅供查阅；不要重复运行。

```sh
.venv/bin/python -u train_sft.py \
  --data data/selected-seven-v1/sft.jsonl \
  --model-dir /data/jyh-yyh/.cache/modelscope/models/Qwen--Qwen3.5-4B/snapshots/master \
  --image-root data/source/sft --output-dir outputs/sft-seven-v2 \
  --rank 32 --epochs 1 --batch-size 4 --max-seq-len 32768 \
  --learning-rate 2e-5 --token-budget 6000000 --resume
```

旧的纯 CPU 预检完整保留在 `outputs/sft-seven-v1-initial-preflight`，未创建训练客户端。训练顺序、加权 loss、token 预算、未确认请求和恢复边界测试，以及项目完整单元测试两端均通过（50 项）。最终训练顺序与预先固定顺序逐条一致，全部 512 个 ID 恰好训练一次；8、32、64、96、128 步的五对权重均完整，执行源码与保存配置哈希一致，Base 运行源码哈希未变。
