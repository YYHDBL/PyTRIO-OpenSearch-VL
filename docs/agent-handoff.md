# PyTRIO × OpenSearch-VL 接手说明

更新日期：2026-10-29。当前有限任务为[RL训练前难度诊断](rl-diagnostic.md)：输出 `outputs/rl-diagnostic-128x4-v1`，初始launcher PID2744777，以实时PID为准。固定SFT模型、128题×4条、32k、temperature0.7、七工具、Flash两类奖励裁判；没有参数更新或自动筛题。日志和退出码见诊断说明，不重复启动。已有一条 `rl-0004-r1` 因飞桨HTTP400/任务提交队列已满保留断点，不当作模型答错。定时任务保持PAUSED，不得自行恢复或创建。

SFT输出 `outputs/sft2048-aligned-100-v2`、评分 `outputs/judge-sft2048-aligned-100-v2`：99题生成和评分完成，SFT50/99，同题Base53/99；第100题eval-0006因大图图搜HTTP413保留，summary.complete=false。生成脚本退出2是缺题状态，不能当作仍在运行；评分由用户另行授权已完成。详情见[SFT评测说明](sft-eval100.md)。Base完整53/100、2048条SFT完整512步及两类权重均保留。

先读 [Base结果](base-100-results.md)、[2048条SFT结果](sft-2048.md)、[评测配置](evaluation-alignment.md)。实时文件、进程和最新用户指令优先于文档；历史网络恢复方法仅见 [恢复记录](ocr-network-recovery.md)，不要恢复已被国内直连替代的Mac代理。

## 项目与执行边界

目标是用PyTRIO完成OpenSearch-VL核心训练流程的小规模方法复现，记录真实实验与费用用于博客。当前主模型Qwen3.5-4B、LoRA rank32；当前SFT2048条×1epoch；后续RL原约定128题×4条rollout，先验收32条。比较Base/SFT/RL时使用同一冻结100题、提示词、工具、生成设置和裁判。

| 内容 | Mac | CPU服务器 |
| --- | --- | --- |
| 实验仓库 | `/Users/yyhdbl/Desktop/PyTRIO-OpenSearch-VL` | `/data/jyh-yyh/PyTRIO-OpenSearch-VL` |
| 作者仓库 | `/Users/yyhdbl/Desktop/opensearch-vl` | `/data/jyh-yyh/OpenSearch-VL` |

SSH为 `jyh-yyh@172.29.5.2`，已有密钥认证；服务器Python为项目 `.venv/bin/python`。默认工作目录可能是作者仓库，操作时显式指定实验目录。实验远端 `https://github.com/YYHDBL/PyTRIO-OpenSearch-VL`；作者固定版本 `1ac26e2c35a3e813e022813c88888db457d88f7d`；本轮整理前提交基线为6356220，当前Git状态以实时查询为准。

数据、图片、权重本体和原始API回包只放服务器；Mac只保存代码、依赖、私有配置及聚合证据。两端 `.env` 为私有、Git忽略、权限600；不要打印密钥、不要重新索要。用户最新要求优先，禁止Git回滚/reset/清理，保留当前修改。禁止读写 `/tmp`；中间结果使用已忽略的outputs/logs，源代码修改使用文件编辑工具。

## 已冻结数据

完整SFT/RL文件已下载并按官方大小、LFS SHA-256或Git blob核验，八个图片ZIP校验CRC并解压，不重做下载。SFT源revision为 `2c1c460af4fa15bd63210cbf426a96664b959944`，RL为 `8ef567289043eef004b13da83b0e7bb7f5ae2daa`，评测直接来源VDR发布子集revision为 `deeaf45779a3bbd407d8f0ccb9b4831fc78e81c9`。

| 服务器文件 | 数量 | SHA-256 |
| --- | ---: | --- |
| data/selected/eval.jsonl（保留候选池） | 200 | 59629e3226f24a1ea5c2f2ea06be92e25489844d66fcec627347d7380a9119b3 |
| data/selected-eval100-v1/eval.jsonl（当前评测） | 100 | 9f2f0e3d9117defccccff5a78eaa05099bdd78b3e489eb0540953900e1f5a6f1 |
| data/selected/rl_train.jsonl | 128 | fe38a4057291e98e633de4d8f917d04431f2baeb66c13c3a8344ccab86459af2 |
| data/selected-seven-v1/sft.jsonl（旧模型） | 512 | 92913ce5a508a285b65599dc83c8b79bcb2924130dea66d8b43471e260eaca31 |
| data/selected-seven-2048-v1/sft.jsonl（当前模型） | 2048 | 359b59edaa2961d193b7afa1ef592a5a57425ab58f7d7264b76991301a16569f |

100题从原200题以seed42按来源比例固定，原ID和行内容保留，不根据smoke成绩或耗时筛选。SimpleVQA/InfoSeek/VDR各20，其余四项各10；BrowseComp-VL L1/L2为3/7。全部原200题继续排除在训练外；原128道RL不变。来源内部未逐题型均匀分层，FVQA与论文所引2017版本关系仍未确认，不能声称原论文全量评测。

2048条筛选与审计已完成：检索/裁剪OCR/增强=1228/615/205，短/中/长工具链=1024/819/205，透视8、锐化60、超分164条。排除冻结题面和图片字节哈希、重复图片、未闭合最终回答、同动作重复超过两次及已知后端不兼容参数。全部通过独立图文mask、target右移、EOS及图片顺序审计，最长27600tokens，不使用旧32000字符初筛、不静默截断。图片重压缩和语义改写仍可能漏检，公开答案事实质量没有逐条重新判定。详见 [数据分布](sft-2048.md)。

## 当前模型与服务

SDK固定pytrio0.2.9。主模型Qwen/Qwen3.5-4B，独立摘要Qwen/Qwen3.8-27B，裁判用户指定GLM-5.3-Flash（Anthropic Messages）。文字搜索/抓取为DeepSeek Search/Fetch，图片检索为豆包Global visual，OCR为PaddleOCR-VL-1.6；裁剪/锐化/透视/EDSR超分使用CPU。全部服务器直连，启动脚本清除遗留代理，PADDLE_OCR_PROXY留空，不依赖Mac在线。用户原10080隧道不改动。

EDSR权重 `data/models/EDSR_x4.pb` 为38573255字节，Git blob SHA-1为 `09b872b26a6ba8d43abe433e65e65477ff8714ea`；真实x4执行通过，额外1048576输入像素限制已移除，大图仍有CPU耗时/资源边界。

生产工具缓存 `data/tool_cache/author-seven-v5`，OCR缓存 `data/tool_cache/ocr-v1`。名称中的vl16指OCR模型版本，不是16k窗口。只在明确连接未建立时安全重试；发送后状态不明保留pending，先核对原回包，不盲目重复付费POST。OCR复用job ID；豆包特定无结果响应保留原业务码，不能把其他错误放宽校验当成功。缓存命中不当作新请求，不编造来源网页。

## 已完成的Base与恢复

正式输出 `outputs/base-aligned-100-v1`、`outputs/judge-base-aligned-100-v1`；日志和退出码 `logs/base-aligned-100-v1.log`、`logs/judge-base-aligned-100-v1.log`、`logs/base-aligned-100-v1.exit`。100个固定ID各一次，评分完整，53题正确，micro53%、macro55%，退出码0。82题final_answer、9题turn_limit、8题no_final_answer、1题context_limit。743次工具调用中39次模型参数错误：24次裁剪越界、15次DeepSeek查询超过128字符；后者需记录为替换服务限制。

原GLM第51题eval-0101返回HTTP200/end_turn但两个correct字段矛盾，保留完整回包及usage后重评该题一次，再续评。恢复记录在 `outputs/baseline-recovery-aligned-100-v1`；原100个模型回答和前50条有效评分哈希不变，主模型/工具账本不变。最终裁判103次HTTP响应：101次200（含1次弃用矛盾回答）、2次429；原失败消耗已计入费用证据。评分缓存重跑新增请求0、五份账本不变。

Base主模型输入16435392/输出440864 tokens，摘要输入721814/输出117942；DeepSeek226次成功非空搜索，豆包52次完成调用。客户端统计不替代人民币账单；历史批次和旧未确认请求另记。生成约2小时16分钟，不含后续评分及恢复等待。

## 已完成SFT与权重用途

当前2048条训练输出 `outputs/sft-2048-v1`，启动记录 `outputs/sft-2048-v1-launch.json`，日志/退出码 `logs/sft-2048-v1.log`、`logs/sft-2048-v1.exit`。rank32、batch4、1epoch、lr2e-5、512步，从Base初始化新adapter；未在旧512模型上追加。五对checkpoint在8/128/256/384/512步，最终complete=true、退出码0，2048个ID各一次、无pending。更新累计47.6分钟，含CPU准备/保存启动至退出54.3分钟；训练输入20434350、监督1988460 tokens，含诊断预留20530658tokens。固定训练样本NLL从0.567562到0.415434，最终State恢复后完全一致，不能据此宣称任务准确率提升。

真实Sampler/State路径均在服务器 `outputs/sft-2048-v1/sft_result.json`；Mac私有副本 `outputs/sft2048-v1-result.json`，不要提交内部路径或权重本体。Sampler只用于推理；State用于训练，原SFT中断应恢复优化器，新RL阶段建议只加载权重、重建优化器。前8步和最终两类加载均已真实验证；同阶段恢复不能重复已确认更新。

旧512条模型仍在 `outputs/sft-seven-v2/sft_result.json`，Mac私有副本 `outputs/sft-seven-v2-result.json`。旧v1八步后Sampler创建遗漏base_model导致TypeError，v2继承八步从State恢复完成，无重复训练。原执行源码快照在 `outputs/sft-seven-v2/completed-source-snapshot/train_sft.py`，旧数据、账本和权重保留，不重训。

## 后续工作边界

agent_baseline.py的model_path分支已修复为同时传base_model/model_path；同一最终Sampler真实训练样本图文检查成功。当前本地60项测试通过。SFT v1到v2仅代码与派生哈希变更，已生成前缀和账本保留。Base原源码在 `outputs/base-aligned-100-v1/source-snapshot`，SFT v1在其source-snapshot，v2在 `outputs/sft2048-aligned-100-v2/source-snapshot-v2`。当前诊断仅增加可选随机采样、长度和连续三错终止设置，Base/SFT默认采样参数不变；生成和裁判均保护未确认提交。用户已明确要求将当前代码推送GitHub，运行未完成如实记录，不改运行代码来凑结果。

大图问题仍待解决。原PNG图搜413pending保留；一次隔离的同像素无损WebP请求返回502，其pending及原始回包在 `outputs/sft-eval-recovery-v2/doubao-lossless-probe`。正式代码未使用WebP，不重试这条诊断未知请求，不自动压缩/缩放、换后端或给模型提示。该题未恢复，其他99题已生成并按用户授权评分，summary.complete=false。恢复耗时会进入断点题elapsed_seconds，不能直接将该数值作为纯推理延迟比较。

RL参数更新尚未实现/启动。用户确认尽量沿用论文及作者代码，4B+LoRA、Flash两类裁判、32k训练轨迹为已确认替换；训练rollout也限制32k，不静默截断已有历史，正式评测仍64k。当前只做难度诊断。后续仍需核对并实现old logprob、observation/fatal mask、RLOO/组内优势及单侧裁剪，不能把诊断脚本称为已完成RL训练实现。

定时任务保持PAUSED。旧三工具 `base-v1` 145题、百炼故障及中间批次均只作诊断；旧 `base-seven-v5` 35题和一条未确认主模型请求保留，禁止清除后自动重发。旧smoke分数不能混入正式比较。任务费用未完成人民币账单核对，发布内容必须使用真实截图和实际结果，不伪造提升或支出。
