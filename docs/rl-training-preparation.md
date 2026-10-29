# RL训练准备

当前代码适配已接通并进行离线验证。用户随后明确授权在筛题运行期间先做独立RL smoke：仅2道训练题×4条、最多一次optimizer更新，使用独立复制模型，不启动正式训练。正式训练仍要求诊断complete=true及已确认的选题ID文件。已有诊断轨迹没有保存生成token及old logprob，不能直接作为RL更新输入；smoke重新采样，不复用旧模型回答。

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

## 独立smoke

`run_rl_smoke.sh`使用显式--smoke --execute，选择已完成诊断的rl-0002、rl-0003用于覆盖工具交互和组内奖励差异；这不是按效果选择正式训练题。--smoke必须恰好2个唯一训练ID，因此只运行一个batch、最多一次更新。主模型采样和训练操作各受1M token保护，摘要250k，当前修正版独立输出 `outputs/rl-smoke-2x4-v2/run`，日志及退出码为 `logs/rl-smoke-2x4-v2.*`。首轮v1完整保留，不改变原SFT权重，不恢复定时任务。

生成、评分、更新及双checkpoint完成后，`verify_rl_smoke.py --run-dir outputs/rl-smoke-2x4-v1/run`不追加更新：从最终Sampler采样32个token（temperature1），再让初始/最终State前向计算相同token概率，并验证最终State优化器可加载。SDK0.2.9的compute_logprobs在有图片时会在发送前拒绝，因此不能用该接口作图文重载核验。跨训练/采样backend采用预先固定的绝对误差0.05，不放宽阈值。未知请求保留pending，不自动重发。

首轮smoke已完成8条、27次backward和恰好1次optimizer，双checkpoint保存成功，生成训练脚本退出0。重载核验发现权重确有变化（最大logprob变化0.09046793），但State/Sampler最大差异0.11024642超过0.05，核验退出2；不能宣称完整smoke通过。两个backend各自可加载并不意味着原始概率数值相同。

随后核对作者agent_workflow_trainer.py发现，作者更新前调用Actor.compute_log_prob重算old_log_probs，而保留rollout_log_probs作对比。首轮适配漏接了此步骤，已在train_rl.py补齐：全部Actor旧概率在optimizer更新前计算，原始采样值单独保留于actor-old-logprobs.json。用独立初始State副本和真实图文completion进行一次前向及custom backward校准，Actor旧概率与custom前向最大差异0，无optimizer更新。原运行源码、旧轨迹和失败报告保留在smoke输出目录，不能改写原证据。

修正版v2完成8条生成与评分、16个训练Datum、16次Actor前向和16次custom backward，其中14个非零优势样本的Actor旧概率与custom前向最大差为0；另外2个零优势样本返回0 loss及零logprob占位，不参与概率一致性比较，损失分母保持原有7条可学习轨迹。随后恰好一次optimizer更新和两类checkpoint保存完成。以已保存训练端live前向核验State恢复，最大差为0（原定容差0.001）；32个新生成token的更新前后训练端最大差0.11215973，最终State/Sampler最大差0.03574139（原定容差0.05），本轮核验通过。此跨backend检查仅覆盖32tokens，不能证明所有输入逐位相同，也不能把v1到v2的概率差异缩小单独归因于Actor重算。

v2训练脚本最后的文件名glob将live-after-update前向也计为optimizer，原脚本退出1。核对操作账本、8份评分、无pending、完整最终checkpoint后恢复result.json，未重新训练、未追加optimizer，原退出码不改写。代码已改为按operation类型准确计数（包括resume路径），新增回归测试。依据见[首轮证据](evidence/rl-smoke-2x4-v1.json)、[修正版证据](evidence/rl-smoke-2x4-v2.json)；source-snapshot保留各轮实际执行源码，当前修复代码不冒充旧执行版本。

## 待实测及限制

离线测试覆盖真实PNG图文往返、前缀变化拒绝、mask与target对齐、作者RLOO数值比较、PPO真实梯度和裁剪、分批损失等价、策略混用拒绝、无有效前缀处理、Actor重算顺序及optimizer计数。独立两题真实采样、更新、双权重保存与重载已验证，但没有做完整32k图文backward压力验证，也没有证明整个服务端梯度数值与作者Megatron完全一致。正式训练仍未启动；开始前需完成筛题故障恢复、确认选题、loss聚合及预算，smoke通过不代表准确率提升。
