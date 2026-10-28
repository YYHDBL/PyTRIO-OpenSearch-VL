# 七工具固定评测

2026-10-28：固定100题Base生成与GLM评分已完整结束，53/100正确、complete=true、退出码0。固定ID和账本已核验，缓存评分重跑无新增请求，结果见[正式报告](base-100-results.md)。启动依据为服务器 `outputs/baseline-launch-aligned-100-v1.json`，评分矛盾回包恢复依据在 `outputs/baseline-recovery-aligned-100-v1/`；不要重新启动已完成批次。2048条SFT训练已完成，SFT正式评测和RL未启动，定时任务保持暂停。

## 数据与生成

固定 100 题：SimpleVQA、InfoSeek、VDR 各 20，FVQA、MMSearch、LiveVQA、BrowseComp-VL 各 10；后者 Level 1/2 为 3/7。从原冻结 200 题按 seed 42 分层抽取，保留来源比例，不依据模型成绩或耗时挑题。原 200 题保留且继续排除在训练外；未逐题型均匀分层。详见 [数据说明](data.md)。不能称作论文全量评测。

主模型为 PyTRIO Qwen3.5-4B Base；temperature 0、seed 42、最多 50 个 assistant 轮次。总窗口 65536 tokens，单轮生成上限 32768，实际按窗口剩余空间扣减。只保留 EOS 停止词，不强制作答，不在评测时连续三错后提前终止。图文 token 必须与远端计数一致；模型输入不包含参考答案或 image_caption。

使用公开 SFT 的完整 system/tools，七种能力及 web_search 别名均保留。文字搜索与抓取为 DeepSeek，图搜为豆包 Global visual，OCR 为 PaddleOCR-VL-1.6，独立摘要为 Qwen3.8-27B；PIL/OpenCV/EDSR 处理图片。全部由服务器直连，不依赖 Mac 代理。提示词字符差异、观察模板及后端差异见 [协议说明](protocol-review.md)。

## 新批次与历史输出

`run_baseline.sh` 已按以下独立路径启动。启动前已检查进程、配置、两端执行文件、数据和图片哈希。后续先读现有 PID、锁、日志、退出码及断点，不能重复启动或在已有运行期间改动源代码。

| 内容 | 路径 |
| --- | --- |
| 固定输入 | `data/selected-eval100-v1/eval.jsonl` |
| 新回答与断点 | `outputs/base-aligned-100-v1/` |
| 生成日志 | `logs/base-aligned-100-v1.log` |
| 新评分与汇总 | `outputs/judge-base-aligned-100-v1/`，expected=100 |
| 评分日志 | `logs/judge-base-aligned-100-v1.log` |
| 整段脚本退出码 | `logs/base-aligned-100-v1.exit` |

旧三工具 145 题、旧百炼故障批次、三题 smoke 以及最近停止的 35 题均保留作诊断，不混入新比较。旧 35 题还保留一条未确认主模型请求，不能当作零费用或自动重发。旧执行代码快照与启动哈希一致。历史 smoke 的 2/3 使用旧窗口/旧答案抽取，不代表本次新评测。

新批次完整生成 100 题，不复用诊断批次或十题 smoke 的模型回答；完全相同请求的外部工具事实与摘要缓存可以跨阶段复用。直接调用评测入口时显式指定上述 --data，不能误用默认的原 200 题路径。当前缓存为 `data/tool_cache/author-seven-v5/`、`data/tool_cache/ocr-v1/`，缓存命中不记新 API 消耗。

## 评分、费用与完成标准

裁判为用户指定的 GLM-5.3-Flash（Anthropic Messages），temperature 0、max_tokens 1024、关闭思考。裁判提示词与作者 eval_with_gpt4o.py 逐字节一致；模型与论文 GPT-4o 不同。

按作者规则拼接完整 assistant 轨迹并抽取答案；没有显式最终标签不直接判错，只有无可抽取文本才按空文本处理。抽取规则的 34 个对照用例一致，见 [核验](evidence/alignment-oracle-v1.json)。裁判失败或基础设施异常保存断点并停止，不能静默计错；评分缓存必须与回答和裁判配置哈希一致。

预算主模型 25M tokens、摘要 8M tokens、豆包请求尝试 800。模型未完成请求按输入加请求输出额度预留，未知旧消耗不按零处理；达到预算停止整批，保留尚未完成题。预算不属于单题作答长度，也不等于人民币硬上限。分别核对主模型、摘要、DeepSeek、豆包、OCR 与 GLM，保留原始响应和未确认提交；历史支出不能遗漏。PyTRIO 代金券不覆盖第三方费用。

完成标准为冻结 100 个 ID 恰好各一次、评分 100 条、summary.complete=true、退出码 0；同时核对账本、模型终止原因和实际账单。报告各来源正确数/总数、micro 和 macro Pass@1。仅有进程退出码或生成完成不能证明评测交付完成。
