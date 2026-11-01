# 服务配置

在项目根目录复制 `.env.example` 为 `.env`，填写自己的凭据。不要提交 `.env`、认证头、原始服务回包或模型权重路径。

| 服务 | 用途 | 环境变量 |
| --- | --- | --- |
| [PyTRIO](https://pytrio.com/dashboard) | 主模型训练、采样及默认摘要 | `PYTRIO_API_KEY` |
| [DeepSeek](https://api-docs.deepseek.com/zh-cn/guides/search/) | 文字搜索和网页读取 | `DEEPSEEK_API_KEY` |
| [豆包搜索 Global](https://docs.volcengine.com/docs/networked-search/doubao-search-global-edition?lang=zh) | `visual` 以图搜图，需要开通对应权限 | `DOUBAO_SEARCH_API_KEY` |
| [PaddleOCR](https://aistudio.baidu.com/paddleocr/task) | 异步版面解析 | `PADDLE_OCR_TOKEN`，其余默认值见模板 |
| [智谱](https://open.bigmodel.cn/) | 答案和搜索质量裁判 | `JUDGE_BASE_URL`、`JUDGE_API_KEY`、`JUDGE_MODEL`、`JUDGE_PROTOCOL` |

本实验裁判使用 GLM-5.3-Flash，协议为 Anthropic Messages。裁判提示词来自作者源码，模型与论文的 GPT-4o 不同。

摘要默认使用 PyTRIO。可选的 `--summary-backend local` 指向自备的 OpenAI 兼容接口，需要 `LOCAL_SUMMARY_*` 变量；不要求使用私有地址或 Mac 转发。比较模型效果时，应固定摘要模型和部署设置，避免额外变量。

`check_services.py` 检查配置；`--pytrio` 额外只读查询支持模型。`check_seven_tools.py`、`check_service_stability.py` 等真实检查会调用服务，应先阅读各自 `--help` 并设置预算。

搜索、OCR、摘要和裁判费用分别计算。程序记录 tokens、请求次数、缓存命中和请求 ID，但这些统计不替代平台账单。服务超时且发送状态不明时保留 pending，避免自动重复提交付费请求；连接建立前的失败才进行有界重试。
