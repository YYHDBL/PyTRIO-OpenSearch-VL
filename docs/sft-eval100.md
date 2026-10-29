# 2048 条 SFT 最终模型：固定100题评测

2026-10-29，使用2048条、1epoch训练的最终第512步Sampler，固定100题中99题生成及GLM评分完成，另外eval-0006保留大图图搜故障断点。按用户要求对已有99题单独评分，expected仍为100，summary.complete=false。SFT为50/99，相同99题的Base为53/99；8题由错变对，11题由对变错，未进行人工复核，不能据此宣称稳定提升或退化。完整Base为53/100。

相同99题中，Base正常结束82题、8题轮数上限、8题无最终回答、1题上下文耗尽；SFT正常结束97题、2题轮数上限。工具调用693降至376，输出tokens430635降至98033。生成耗时合计7926.07秒降至4219.05秒，但缓存、服务和网络不同，不能把耗时差全部归因于训练。GLM评分首轮在第16题遭遇三次明确429，原响应和pending归档后续跑，已有15条评分复用；最终99条评分、无裁判pending。

| 项目 | 与Base的关系 |
| --- | --- |
| 模型 | 同一Qwen3.5-4B基模，载入最终SFT Sampler；不使用State作推理 |
| 数据 | 同一data/selected-eval100-v1/eval.jsonl，100个ID及图片哈希相同 |
| 提示词/工具/裁判 | 同版完整system/tools、七工具、DeepSeek/豆包/飞桨、27B摘要、GLM裁判 |
| 生成设置 | 同版65536总窗口、32768单轮上限、最多50轮、temperature0、seed42、EOS停止词 |
| 运行保护 | 同版主模型25M/摘要8M tokens、图搜800次尝试；预算耗尽停止并保留断点 |
| 工具缓存 | 相同查询和摘要可复用；不复用任何Base模型回答或评分 |

唯一源代码修复是checkpoint sampler创建显式同时传base_model和model_path，以满足固定SDK0.2.9的签名；Base分支行为不变。修复前测试重现缺少必填base_model，修复后56项测试通过。Base原执行文件按原launch哈希保存到服务器 `outputs/base-aligned-100-v1/source-snapshot/`，不改写旧输出配置或哈希。

预检配置与Base比较，变化字段恰为model_path、code_sha256、派生sha256，其余配置完全一致。随后在固定训练样本上作一次真实Sampler加载/图文token检查，输入4366、输出16tokens，本地与远端输入计数一致；这不是额外benchmark成绩，也不用于调参。证据见 [预检比较](evidence/sft2048-eval100-v1-preflight.json)、[加载检查](evidence/sft2048-eval100-v1-health.json)。

| 内容 | 服务器路径 |
| --- | --- |
| 原始启动记录 | outputs/sft2048-eval100-v1-launch.json |
| 恢复启动记录 | outputs/sft-eval-recovery-v2/launch.json |
| 生成输出 | outputs/sft2048-aligned-100-v2 |
| 评分输出 | outputs/judge-sft2048-aligned-100-v2 |
| 生成日志 | logs/sft2048-aligned-100-v2.log |
| 评分日志 | logs/judge-sft2048-aligned-100-v2.log |
| 有限脚本退出码 | logs/sft2048-aligned-100-v2.exit |

## 单题服务故障隔离

eval-0006依次裁剪两次、对原图作EDSR x4，再检索增强图。3200×2400的增强PNG为5689309字节，Base64为7585748字符，豆包返回HTTP413，原始正文为Request Entity Too Large。原程序将ToolError上抛至进程，v1因此在2题完整时退出；模型没有收到413反馈。

按用户要求，保留模型动作、上下文、原始图片和全部生成设置，不新增错误提示，不自动换原图、裁剪、缩小或切换搜索后端。现在ToolError只保留该题断点并记入generation-status.json的deferred，继续后续题目。配置错误、代码错误和预算耗尽仍终止整批。未确认提交保持pending，续跑不会自动重发。只在全部100题完成后退出0并进入评分；有deferred时退出2，不能把未完成题算错、删出分母或宣称完整成绩。

大图传输尚未解决：服务器内存中核验无损WebP像素、尺寸完全相同，Base64可降至5399516字符，但一次独立真实请求返回HTTP502。该诊断原始回包和pending保留在outputs/sft-eval-recovery-v2/doubao-lossless-probe，次数纳入工具尝试账本，实际费用未核对，不自动重试。正式适配器保留原编码和缓存身份，没有采用未经验证的编码方案。

v1原执行源码按启动哈希保存到其source-snapshot；v2只变更code_sha256和派生sha256。两份完整答案及第三题断点仅迁移config_sha256，其他字段完全一致；主模型、摘要、DeepSeek三份账本字节不变。原v1和413pending保持原状，依据见[迁移证据](evidence/sft2048-eval100-v2-recovery.json)。最终57项测试通过，其中使用真实持久化pending验证故障隔离不发网络请求、不添加模型反馈，并验证预算耗尽继续终止。

启动脚本为run_sft_eval.sh，从服务器私有训练结果读取实际Sampler路径，不在Git写入真实内部权重路径。先查现有PID、lock、partial、token账本和pending，禁止重复启动、覆盖或盲目重发状态不明请求。

完成后需核对100个ID恰好各生成/评分一次、complete=true、退出码0、请求/费用与缓存一致。报告总分、各来源正确数、同题配对变化、终止原因及工具错误；以此判断SFT是否改善准确率、反复检索和无法结束思考。GLM矛盾回包或基础设施错误单独处理，不能混为模型答错；人民币费用仍需实际账单核对。
