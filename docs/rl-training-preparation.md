# RL训练准备

当前代码适配已接通并进行离线验证；诊断运行结束前不提交RL采样、远端训练、参数更新或checkpoint。已有诊断轨迹没有保存生成token及old logprob，不能直接作为RL更新输入。新的训练入口默认只离线预检，实际训练必须显式传入--execute，并检查诊断complete=true及已确认的选题ID文件。

## 已核对

- 作者 `compute_rloo_outcome_advantage` 使用四条同题奖励的leave-one-out基线，数学上为 `4/3 × (reward - 组均值)`，不除以组内标准差。论文正文公式写了标准差归一化；附录及作者启动脚本使用RLOO。这是不同的优势尺度。当前按作者运行实现使用RLOO，先纳入全部四条奖励，再对fatal轨迹负优势归零，已与作者原函数逐值比较通过；正式配置审核仍需说明该差异。
- PyTRIO 0.2.9已安装的SDK提供 `create_training_client_from_state(path)` 和 `create_training_client_from_state_with_optimizer(path)`。官方[继续训练说明](https://docs.pytrio.com/docs/guide/resume)说明前者只加载权重，后者恢复优化器；建议RL作为新阶段使用SFT State初始化权重、重建优化器，RL中断续训才恢复RL优化器。不得使用Sampler路径创建训练客户端。
- 使用PyTRIO[custom loss](https://docs.pytrio.com/docs/guide/custom_loss)显式实现作者token级PPO：low/high ratio为0.8/1.28，log ratio限制[-20,20]，负优势dual clip为3。作者运行配置loss为seq-mean-token-sum，而论文正文为seq-mean-token-mean；入口要求明确指定--aggregation，不自动选择。跨轮和micro-batch使用整个batch的可学习轨迹数为共同分母，token-mean模式另除以轨迹保留生成token总数。作者use_kl_loss=False、默认use_kl_in_reward=False，因此当前不加KL，不能因为配置中出现0.001系数就自动启用。候选运行设置为lr1e-6、每批4题×4条、1轮，正式运行前仍需确认题集和预算。
- `rl_prepare.build_datum` 保留图片chunk、将prompt和图片区间优势归零，生成区间使用当次采样token和old logprob。样本长度上限32768。`build_trajectory_datums` 校验相邻轮次的原始token与图片前缀，按fatal起点保留可学习轮次；不一致立即报错。分轮损失和梯度累积的离线计算已验证，服务端缩放仍需实测。

## 已接入模块

| 文件 | 职责 |
| --- | --- |
| rl_rollout.py | 原始token、old logprob、图文前缀、策略版本及工具断点；未知模型请求pending保护 |
| rl_prepare.py | RLOO、fatal单侧裁剪、图文target右移与观察mask、多轮前缀检查 |
| rl_loss.py | 作者PPO裁剪、dual clip、轨迹及micro-batch损失分母 |
| train_rl.py | 完整四轨迹组、Flash两类裁判、梯度累积、更新、State/Sampler成对保存和恢复 |

所有组内轨迹来自同一策略，更新前完成全部奖励与优势；每批更新后保存两类权重并切换新Sampler。服务故障停止当前训练batch，不以0奖励凑齐组。沿用已有remote_operation请求账本；有未确认操作或已更新超出checkpoint时拒绝自动重做。

CPU PyTorch使用项目uv.lock中的2.14.1，服务器为2.14.1+cpu。新采样上下文直接追加原始生成tokens和工具观察，不通过解码后再编码重建历史。工具观察导致下一轮超32k时记录context_limit，原观察保留且不再请求模型；未完整终端回答记no_final_answer。零优势轨迹保留损失分母，全批零优势时不执行optimizer。

Adam参数显式使用作者Megatron配置：lr1e-6、betas0.9/0.999、eps1e-8、weight_decay0.01、grad_clip1.0；不继承PyTRIO的不同默认值。来源为作者optim/megatron.yaml及Megatron optimizer_config.py，真实梯度裁剪行为仍需远端实测。

服务器离线预检示例：

```sh
.venv/bin/python train_rl.py --output-dir outputs/rl-preflight-v1/not-launched \
  --selected-ids outputs/rl-preflight-v1/ids-preview.json --aggregation seq-mean-token-sum
```

ids-preview.json仅为原128题ID的预览，未批准用于正式训练。预检输出network_requests=0、updates=0，不创建训练目录。实际--execute仍受诊断完整结束检查保护。

## 待实测及限制

本地全套71项测试通过，服务器相关14项测试通过，覆盖真实PNG图文往返、前缀变化拒绝、mask与target对齐、作者RLOO数值比较、PPO真实梯度和裁剪、分批损失等价、策略混用拒绝和无有效前缀处理。没有实际参数更新；服务端custom loss梯度缩放、真实32k多模态backward及双checkpoint加载等价性，按用户要求等筛题结束后验证。正式运行前还需确认选题、loss聚合取舍和费用额度，不能以离线通过代替远端实测。
