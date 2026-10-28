# 运行前评测配置核对

核对日期：2026-10-28。固定100题Base生成与评分已完成并核验，53/100正确，见[结果报告](base-100-results.md)；启动依据为服务器 `outputs/baseline-launch-aligned-100-v1.json`，不要重新启动。旧批次仅作诊断。2048条SFT训练已完成，正式SFT评测尚未运行；定时任务保持暂停，不启动RL。

## 固定配置

| 项目 | 本次配置 |
| --- | --- |
| 模型 | PyTRIO `Qwen/Qwen3.5-4B` 原始基模，不加载 LoRA/checkpoint |
| 数据 | 从原冻结 200 题按 seed 42 固定 100 题，每题生成一条轨迹；不依据模型表现选题 |
| 配比 | SimpleVQA / InfoSeek / VDR 各 20；FVQA / MMSearch / LiveVQA / BrowseComp-VL 各 10 |
| 主模型采样 | temperature 0、seed 42、最多 50 个 assistant 轮次 |
| 单轮输出上限 | 32768 tokens，包含思考与输出；实际为 `min(32768, 65536 - 当前图文输入 tokens)` |
| 总上下文窗口 | 65536 tokens，包含 system、图片、全部历史思考/输出和工具观察 |
| 停止条件 | EOS `<|im_end|>`；不再额外使用 `</tool_call>` 停止词；不强制作答、不在评测中连续三错后提前终止 |
| 提示词与工具 | 公开 SFT 的完整 system/tools；七种能力、八个工具名，不使用旧三工具提示词 |
| 图像处理 | crop、sharpen、perspective_correct 使用 PIL/OpenCV；super_resolution 为真实 EDSR x4，已移除额外的 1048576 输入像素限制 |
| 文字检索/读页 | DeepSeek Search/Fetch，最多五条搜索结果；API 查询最长 128 字符，超长明确报错，不静默截断 |
| 图搜 | 豆包 Global visual；DocCount 10、MaxSnippetLength 300；保留网页来源和图片链接 |
| OCR | 飞桨 PaddleOCR-VL-1.6 异步 Jobs，复用 job ID 和结果缓存 |
| 独立摘要 | PyTRIO Qwen3.8-27B；1024 输出 tokens、temperature 0.3、top_p 0.95、top_k 20、seed 42、关闭模板思考 |
| 摘要输入 | 沿用作者摘要提示词：网页正文前 2000 字符、图片检索 JSON 前 3000 字符 |
| 裁判 | 用户指定 GLM-5.3-Flash，Anthropic Messages；temperature 0、最多 1024 tokens、关闭思考 |
| 评分 | 作者完整 assistant 轨迹拼接、答案抽取规则和裁判提示词；报告各来源、micro / macro Pass@1 |
| 网络 | 以上服务全部由 CPU 服务器直连，不依赖 Mac 代理；启动脚本清除遗留代理变量 |
| 运行预算 | 主模型 25M tokens、摘要 8M tokens、豆包请求尝试 800；达到预算停止整批并保留断点，不把未完成题算错 |

主模型没有额外关闭思考。作者 `InferenceConfig.enable_thinking=False` 字段在已核对的 HF runner 中没有被使用，不能据该字段声称作者评测实际关闭了思考。

100 题文件为 `data/selected-eval100-v1/eval.jsonl`，SHA-256：`9f2f0e3d9117defccccff5a78eaa05099bdd78b3e489eb0540953900e1f5a6f1`；[抽样清单](evidence/eval100-v1-manifest.json) 保存算法与全部原始 ID。原 200 题 SHA-256 仍为 `59629e3226f24a1ea5c2f2ea06be92e25489844d66fcec627347d7380a9119b3`，完整保留且继续排除在训练外。来源内部未逐题型均匀分层；新 BrowseComp-VL 为 Level 1/2 = 3/7。这是自定义子集，不能表述为论文七个 benchmark 全量成绩。

## 长度能力与作者设置

参考版本固定为 `OpenSearch-VL@1ac26e2c35a3e813e022813c88888db457d88f7d`。作者公开代码默认单轮 `max_new_tokens=32768`、temperature 0、最多 50 轮。本次使用这些推理默认值；尚未确认论文每次实验的全部实际启动参数，不能声称与论文配置完全相同。

旧客户端的 16384 总窗口已移除。PyTRIO 模型目录标注 32768，而 sampling 能力报告 `max_seq_len=65536`、`max_tokens_per_request=65536`。已真实验证：65000 token 输入被接受；32000 图文输入加声明 32768 输出额度的请求被接受，远端输入计数一致。证据见 [容量检查](evidence/context-capacity-v1.json) 和 [配置预检](evidence/alignment-preflight-v2.json)。容量检查只输出了 2 tokens，证明请求容量，不证明实际长生成质量或每次都能成功生成满 32768 tokens。

65536 仍是本次采样服务窗口限制。输入超过 32768 后，单轮可生成额度逐渐小于 32768；达到窗口上限记录 `context_limit`。不截断历史、不丢弃图片、不额外要求立即给答案。论文训练中的 SFT 长度和 RL 响应长度不能直接当成最终评测参数；训练服务当前上限与采样服务也不同。

## 评分与比较边界

作者先用双换行拼接全部 assistant 文本，再依次抽取 boxed、answer 标签、response 标签、最后思考标签后的正文，最后才回退到文本末尾。因此没有显式最终回答标签的轨迹仍可能得到可评分文本，不能直接判错。抽取函数与作者的 34 个离线用例完全一致，裁判提示词逐字节一致，见 [对照检查](evidence/alignment-oracle-v1.json)。裁判最多收到抽取结果的前 8000 字符，与参考代码一致。

仍存在以下差异：4B 基模与后续 LoRA、小规模数据和自定义评测子集；DeepSeek / 豆包 / 飞桨服务；27B 摘要代替作者 32B；GLM 代替 GPT-4o；采样窗口受 PyTRIO 限制。当前 system 使用发布 SFT 的原文，与参考推理 prompt 有勾号/空白的 7 字符差异；工具观察使用 SFT 的 `tool_response` 模板，参考推理入口使用 observation 封装。这些适配必须在 Base/SFT/RL 的评测中保持一致，不能描述为所有实现逐字节相同。

## 保存与运行边界

旧 35 题输出、partial、账本和原执行代码快照保留；其中一条主模型请求结果未确认，不能当作零费用，也不自动重发。新配置完整生成冻结的 100 题，不复用诊断批次或十题 smoke 的模型回答和评分。相同请求的工具事实及摘要缓存可以复用，缓存命中不记为新的 API 消耗。

当前 45 项单元测试在 Mac 与服务器均通过，`bash -n run_baseline.sh` 通过。配置预检不创建 sampler、不生成评测回答。十题 smoke 已完成生成与评分，实际配置见 [记录](evidence/smoke-aligned-10-v1-config.json)；新的完整 100 题已启动，生成与评分尚未完成。小规模实际检查不等于长期稳定性保证。EDSR 不再有额外像素限制，大图 CPU 超分可能较慢或触及库/内存边界，应明确处理实际错误。

正式运行已由 `run_baseline.sh` 启动，输出为 `outputs/base-aligned-100-v1/`、`outputs/judge-base-aligned-100-v1/`；显式指定新 100 题 --data 和评分 --expected 100，生成/评分日志及退出码为 `logs/base-aligned-100-v1.log`、`logs/judge-base-aligned-100-v1.log`、`logs/base-aligned-100-v1.exit`。十题 smoke 使用独立目录；不恢复定时任务。

完成标准是冻结 100 个 ID 恰好各一次、100 条评分、summary.complete=true、退出码 0，并核对主模型/摘要账本和第三方用量。未确认提交必须保留，基础设施故障不能混作模型答错；客户端预算与 token 记录不替代平台人民币账单。
