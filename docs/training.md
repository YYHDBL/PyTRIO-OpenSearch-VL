# 训练与评测

从仓库根目录运行。先完成[数据准备](data.md)和[服务配置](services.md)，在自己的运行机器设置路径。下列路径是目录约定，原始数据和私有权重不随仓库发布。

## SFT

| 参数 | 本次设置 |
| --- | --- |
| 模型 / LoRA | Qwen3.5-4B / rank32 |
| 数据 / epoch | 2,048条完整轨迹 / 1轮 |
| batch / 学习率 | 4 / 2e-5 |
| 最大长度 | 32,768 tokens |
| 更新次数 | 512 |
| 监督范围 | Assistant生成内容及结束标记 |

先用 `audit_sft.py` 检查图片、模板、监督mask和长度，再运行训练。`model-dir` 指向本机 tokenizer / processor 目录：

```bash
uv run python audit_sft.py \
  --model-dir data/models/Qwen3.5-4B \
  --data data/selected-seven-2048-v1/sft.jsonl \
  --image-root data/source/sft \
  --output-dir outputs/sft-audit-2048 \
  --max-seq-len 32768

uv run python train_sft.py \
  --model-dir data/models/Qwen3.5-4B \
  --data data/selected-seven-2048-v1/sft.jsonl \
  --image-root data/source/sft \
  --audit-report outputs/sft-audit-2048/summary.json \
  --output-dir outputs/sft-2048-v1 \
  --expected-samples 2048 --epochs 1 --batch-size 4 \
  --rank 32 --learning-rate 2e-5 --max-seq-len 32768 \
  --checkpoint-every 128 --token-budget 22000000
```

训练保存 State 和 Sampler。State用于恢复训练，Sampler用于推理；不能互换。本次512次更新循环耗时47分37秒，含准备、保存和检查的全程54分19秒；固定训练样本NLL从0.5676降到0.4154，不是验证集loss。[完成证据](evidence/sft2048-v1.json)。

## Fatal-Aware RL

沿用作者的多轮搜索Fatal-Aware处理：连续3次工具错误触发fatal，从错误串起点屏蔽后缀，保留之前的有效交互；fatal轨迹的负优势归零。按作者运行代码使用RLOO组内优势和PPO裁剪目标。论文正文的标准差归一化与公开代码的RLOO存在差异，本实现遵循代码。

奖励为 `format × (0.8 × accuracy + 0.2 × query_quality)`。工具观察仅作为上下文，模型生成token参与学习；每批更新前重新计算Actor old logprob，之后保存新的State / Sampler供下一批使用。

| 参数 | 本次设置 |
| --- | --- |
| 初始化 | 2,048条SFT最终State |
| 任务 / 验证 | 100道 / 16道 |
| 每题采样 / 每批题数 | 4条 / 4道 |
| 上下文 / temperature | 32k / 0.7 |
| 最大交互轮数 | 50 |
| 学习率 / epoch | 1e-6 / 1 |
| loss聚合 | seq-mean-token-sum |
| ratio裁剪 / dual clip | 0.8至1.28 / 3 |
| KL loss | 不启用 |

`train_rl.py` 与 `rl_diagnostic.py` 读取 SFT 最终结果文件、作者源码和诊断数据；使用前检查两者中的目录约定。作者源码默认位于同级 `OpenSearch-VL/`，无需安装其完整训练栈。

```bash
# 默认只预检，不提交训练
uv run python train_rl.py \
  --output-dir outputs/rl-new \
  --selected-ids outputs/rl-selection-100-v1/train-ids.json \
  --validation-ids outputs/rl-selection-100-v1/validation-ids.json \
  --aggregation seq-mean-token-sum
```

确认题组、权重与预算后，在同一命令增加 `--execute` 开始训练。只有未完成且配置一致的运行才使用 `--resume`。本次最终93道任务、372条有效Rollout进入25次更新，7道服务异常按完整题组跳过；多轮数据共拆成1,598个Datum。

## 固定评测

```bash
uv run python agent_baseline.py \
  --data data/selected-eval100-v1/eval.jsonl \
  --output-dir outputs/eval-base \
  --max-context 65536 --max-tokens 32768 --max-turns 50 \
  --token-budget 25000000 --summary-token-budget 8000000 \
  --image-request-budget 800

uv run python judge_baseline.py \
  --results-dir outputs/eval-base --output-dir outputs/judge-base \
  --expected 100 --continue-on-error
```

SFT、RL评测在生成命令中增加 `--model-path` 指向自己的Sampler，分别使用新输出目录。benchmark使用temperature0、seed42；开发验证题每题4次，使用temperature0.7和32k，不能混合统计。

服务异常保留断点和未确认请求，不当作模型答错。完整结束需要检查冻结ID、回答哈希、评分条数及summary中的complete字段；退出码本身不足以证明评测完整。详情见[结果分析](training-stage-comparison.md)。
