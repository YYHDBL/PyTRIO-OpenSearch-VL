# 服务配置与用量

当前实验已配置全部所需服务，真实凭据只保存在 Mac 与服务器项目根目录的私有 `.env`，权限 600。不要打印变量值或将其写入示例、日志、文档和 Git。恢复任务时直接读取现有配置，无需重新申请 Key。

| 服务 | 用途 | 配置 |
| --- | --- | --- |
| [PyTRIO](https://pytrio.com/dashboard) | 4B 主模型采样、27B 独立摘要及后续训练 | PYTRIO_API_KEY |
| [DeepSeek Search/Fetch](https://api-docs.deepseek.com/zh-cn/guides/search/) | 文字搜索与网页正文，服务器直连 | DEEPSEEK_API_KEY |
| [豆包 Global 图搜图](https://docs.volcengine.com/docs/Networkedsearch/doubao-search-global-edition?lang=zh) | SearchType=visual，图片 URL 或纯 Base64，服务器直连 | DOUBAO_SEARCH_API_KEY |
| [PaddleOCR](https://aistudio.baidu.com/paddleocr/task) | 异步 OCR/版面解析，PaddleOCR-VL-1.6 | PADDLE_OCR_JOB_URL、PADDLE_OCR_MODEL、PADDLE_OCR_TOKEN、PADDLE_OCR_PROXY |
| [智谱](https://open.bigmodel.cn/) | GLM-5.3-Flash 答案裁判，Anthropic Messages 协议 | JUDGE_BASE_URL、JUDGE_API_KEY、JUDGE_MODEL、JUDGE_PROTOCOL |

当前图搜调用豆包 Global 的直接搜索接口，不再调用百炼辅助模型。SERPER_API_KEY、SERPAPI_API_KEY、JINA_API_KEY 和 DASHSCOPE_* 仅为历史配置。check_services.py 与正式入口均检查当前服务变量。作者的独立检索摘要仍由 PyTRIO Qwen3.8-27B 完成。

当前所有服务均从 CPU 服务器直连；PADDLE_OCR_PROXY 已清空，新入口清除通用代理环境，不依赖 Mac 10081/1082 转发。豆包检查见 [图搜验收](doubao-image.md)，其他服务的 [v4 检查](services-stability.md) 保留为历史证据。不能照历史隧道记录重新启用代理或评测。

PyTRIO 代金券仅覆盖约定的 PyTRIO 消耗，豆包、GLM 等第三方用量单独记录。豆包 visual 单价与实际费用待控制台核对；各平台额度以实际账户为准，不把历史免费额度当作长期保证。

主模型和摘要分别记录 tokens.jsonl、summary-tokens.jsonl；豆包真实调用记入 doubao-usage.jsonl，缓存命中排除，保留 RequestId、返回数量、原始业务码与空结果分类；DeepSeek 成功请求另有 deepseek-usage.jsonl；GLM usage 在逐题评分中。响应原文、状态码和请求 ID 私有留存，不保存认证头。客户端用量不等于账单；历史百炼检查和未确认提交也需独立核对。
