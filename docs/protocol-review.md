# 作者协议与当前实现核对

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
| 图片搜索 | Lens/图像服务及结果摘要 | SerpApi 上传与搜索已验证，作者 Qwen 摘要未接入 |
| 文字搜索 | Serper、Jina、独立 Qwen 摘要 | 搜索与网页读取已验证，作者 Qwen 摘要未接入 |
| 锐化 | OpenCV 高斯模糊与加权运算 | 原代码可参考，当前环境尚未安装 OpenCV |
| 透视校正 | OpenCV 轮廓检测与透视变换 | 原代码可参考，当前环境尚未安装 OpenCV |
| 超分辨率 | OpenCV dnn_superres、EDSR 权重 | 当前未配置权重，也未安装 OpenCV contrib |
| OCR/版面解析 | LAYOUT_PARSING_API_URL 及认证 | 当前没有接口配置，也没有安装本地 OCR |

服务器检查未发现当前虚拟环境中的 cv2、paddleocr、paddle 或 torch，也没有 tesseract。仅以配置名判断服务是否就绪，没有输出真实凭据。七工具本身不受 PyTRIO 限制；图像处理可在 CPU 运行，模型采样和训练由 PyTRIO 远程执行。

原代码在缺少超分权重时会返回原图；环境验收时必须明确识别这种情况，不能把未执行增强记成成功。OCR、搜索结果摘要若更换后端，需要记录差异并在 Base、SFT、RL 中统一使用。

## 修正原则

优先按作者七工具准备；作者提示词、工具定义、动作文本、图文顺序与观测格式作为可核验文件保存。当前 CPU 长度与 mask 验收不意味着七工具环境已经就绪。确认完整执行协议后，重新验收 SFT、在独立目录重跑 Base，再比较 SFT 和 RL。固定 200 题评测集保留；若重新筛选覆盖七工具的 SFT，使用新的清单和版本，保留原文件。

当前没有恢复付费评测或启动训练。
