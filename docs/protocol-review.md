# 作者协议与当前实现核对

当前配置见 [运行前核对](evaluation-alignment.md)：旧 v5 的 35 题已停止且仅作诊断，新评测未启动。文字检索使用 DeepSeek Search/Fetch，图搜为豆包 Global visual；独立摘要为 Qwen3.8-27B。定时任务保持暂停。

参考代码固定在 `shawn0728/OpenSearch-VL@1ac26e2c35a3e813e022813c88888db457d88f7d`。作者推理提示词见 [prompts.py](https://github.com/shawn0728/OpenSearch-VL/blob/1ac26e2c35a3e813e022813c88888db457d88f7d/opensearch_vl/opensearch_infer/prompts.py)，工具定义见 [tools.py](https://github.com/shawn0728/OpenSearch-VL/blob/1ac26e2c35a3e813e022813c88888db457d88f7d/opensearch_vl/opensearch_infer/tools.py)。

## 核实结果

当前选中的 512 条 SFT 使用同一份作者 system 文本，共 10764 字符；原推理代码的提示词为 10757 字符，差异限于 OCR 成功提示中的勾号和空白字符。两者都描述七类工具能力。工具 JSON 实际有八个名称，其中 `web_search` 与 `text_search` 共用文字检索实现。

原 SFT 工具定义独立放在 `tools` 字段，不能只保留 `system`。转换器已补回这一字段，按原仓库 `QwenToolUtils` 的 JSON 格式加入 system；针对真实 512 条记录，生成结果与原函数逐字节相同。没有把 Qwen3.5 默认的另一种 XML 函数格式套入作者的 JSON 动作轨迹。

## 与已停止评测的差异

| 项目 | 作者公开实现 | 已停止的 base-v1 |
|---|---|---|
| System prompt | 完整 Visual Investigation Agent 规则、工作流和例子 | 自行编写的简短三工具说明 |
| 工具声明 | 七类能力，八个工具名 | 仅 crop、image_search、text_search |
| 文字检索 | Serper、Jina、独立 Qwen 摘要 | 原始摘要与前两页截取正文 |
| 图片检索 | 视觉匹配结果经过独立 Qwen 摘要 | 直接返回部分标题、链接与来源 |
| 工具反馈 | 公开推理入口使用 observation 封装；SFT 模板使用 tool_response | 自定义 ChatML 中的 tool_response |
| 默认推理轮数 | 代码默认最多 50 轮 | 最多 6 轮，之后追加强制作答 |
| 单轮生成长度 | InferenceConfig 默认 32768 | 1536 |

上表是固定版本代码的默认值，不等同于已经确认作者所有论文实验的实际启动参数。base-v1 保留的 145 题仅作诊断，不能表述为作者原版协议复现结果；旧目录已有停止标记，程序拒绝续跑。

## 七工具环境准备情况

| 能力 | 原实现依赖 | 当前准备情况 |
|---|---|---|
| 裁剪 | PIL | 真实调用已验证 |
| 图片搜索 | Lens/图像服务及结果摘要 | 豆包 Global visual，URL/纯 Base64、来源网页和独立 Qwen 摘要已验证，不再调用百炼或 SerpApi |
| 文字搜索 | Serper、Jina、独立 Qwen 摘要 | DeepSeek Search/Fetch，保留作者摘要提示词及独立 Qwen3.8-27B |
| 锐化 | OpenCV 高斯模糊与加权运算 | 已安装 OpenCV contrib，作者算法真实调用通过 |
| 透视校正 | OpenCV 轮廓检测与透视变换 | 已接入；联合检查图片未找到四边形，明确返回原图未变更 |
| 超分辨率 | OpenCV dnn_superres、EDSR 权重 | 权重已校验，真实执行 64×64 → 256×256；已移除额外输入像素限制，大图 CPU 执行仍受耗时与资源边界影响 |
| OCR/版面解析 | 版面解析接口及认证 | 已接入用户提供的 PaddleOCR-VL-1.6 异步接口，通过真实图片与缓存检查；作者服务版本未确认 |

OCR 通过远程接口运行，无需本机安装 Paddle；CPU 使用 OpenCV contrib 处理图像，语言模型采样和训练由 PyTRIO 远程执行。联合检查证据位于服务器 `outputs/seven-tool-check/summary.json`，包含真实多图 SFT 与评测模板的整段 token 一致性检查（样本 sft-0339，13344 tokens）。

当前实现缺少或损坏 EDSR 权重会报错，不把原图当作超分成功。工具观察使用作者 JSON 动作协议及 SFT 的 tool_response 模板，图片保留实际尺寸和坐标。文字检索最多五项，优先使用返回正文，缺失时 Fetch；明确不可抓取时标记摘要片段，网络失败暂停。豆包结果保留真实来源网页；缺失来源时明确标记，不编造溯源。

## 修正原则

七工具 SFT 已重选到 `data/selected-seven-v1/`，覆盖八个工具名并完成 512/512 CPU 审计，冻结的 eval/RL 不变。新的 Base 使用公开 SFT 完整 system/tools，temperature 0、seed 42、最多 50 个 assistant 轮次。采样总窗口 65536，单轮生成上限 32768，按剩余空间扣减；只使用 EOS 停止词。不使用旧六轮限制、强制最终作答或评测时连续三错提前终止。答案从完整轨迹按作者规则抽取，裁判提示词一致。配置、数据、代码与提示词哈希随运行保存。

与论文的差异明确保留：4B 基模与 LoRA、受 PyTRIO 服务限制的 64k 采样窗口、Qwen3.8-27B 替代作者 32B 摘要模型、飞桨 OCR、DeepSeek 搜索/抓取、豆包图搜、GLM 替代 GPT-4o，以及自定义 100 题子集。发布 SFT 与参考推理 prompt 的字符差异、tool_response 与 observation 封装差异也应保留说明。Base/SFT/RL 评测应共享同版条件。作者公开代码默认 MAX_TURNS=50，与当前设置一致；论文各实验实际启动参数未全部确认，不能声称所有设置完全相同。SFT 训练已完成并验证双权重，Base 正式评测仍在进行；SFT 正式评测和 RL 未启动。历史恢复见 [连接记录](ocr-network-recovery.md)。
