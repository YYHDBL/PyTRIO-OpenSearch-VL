# 豆包 Global 图搜图接入验收

2026-10-27，image_search 已切换为豆包 Global 的 visual 接口，CPU 服务器直连。文字搜索与网页抓取继续使用 DeepSeek，独立摘要继续使用 PyTRIO Qwen3.8-27B；作者主模型 system/tools 和七类工具、八个工具名不变。未启动 benchmark、SFT 或 RL，定时任务保持 PAUSED。

## 同图检查

使用与上一轮百炼检查相同的三张图，返回数量请求均为 10。聚合证据见 [doubao-visual-check-v1.json](evidence/doubao-visual-check-v1.json)，原始回包仅保存在服务器 `outputs/doubao-visual-check-v1/`。最初另有一次公开 URL 权限探测，记录在 `outputs/doubao-image-v1/`。

| 图片 | 豆包实测 | 上轮百炼记录 |
| --- | --- | --- |
| 原图 Base64 | 0.884 秒，10 条结果，10 个来源网页 | 85.365 秒，10 条匹配 |
| 公开图片 URL | 0.728 秒，10 条结果，10 个来源网页 | 49.707 秒，10 条匹配 |
| 300×300 裁剪图 Base64 | 明确无结果，0 条；原始业务码 500 保留 | 31.991 秒，0 条匹配 |

这不是并发压测或识别准确率评测，时间也非两平台同时采集，不能据此保证长期性能。裁剪图初次解析未记录完整耗时，因此不把后续缓存耗时冒充 API 延迟。

## 行为与恢复

请求使用 SearchType=visual、空 Query、DocCount=10、MaxSnippetLength=300。内存原图直接编码为纯 Base64，不加 Data URL 前缀，不改变尺寸与裁剪坐标；公网 URL 也已实测。保留来源网页 Url 与图片 ImageUrl 的区别，独立 Qwen 摘要使用真实标题和来源网页，不编造图片溯源。

裁剪图的实际回包同时满足 ErrorCode=500、TotalDocCount=0、Documents=null，以及明确的空搜索消息。适配器仅对此组合返回“无视觉匹配”，原始 500 和 empty_search 分类进入账本；其他 500、权限/额度错误和坏格式仍抛服务错误。该真实空结果回包经离线解析后恢复缓存，没有再次 HTTP 请求；依据为服务器 `empty-result-recovery.json`。

三张图各有独立真实 RequestId。重复调用命中缓存，不新增图搜请求；图搜到 Qwen 摘要的完整链路通过，摘要缓存重跑也未新增采样请求。原始回包、状态码和请求 ID 私有留存，权限 600；只重试确定未发送的连接失败，发送后的不确定状态保留 pending。没有回退百炼。

两端相关单元测试 37 个通过；代码编译、shell 语法、Git 空白检查通过。检查脚本为 `check_doubao_image.py`，新目录运行；显式 --resume 可复用已核验响应和缓存，未知状态请求仍会被阻止。

## 版本与费用

工具缓存为 `data/tool_cache/author-seven-v5/`。旧 `base-seven-v5` 已停止，35 题只作诊断；新评测目标为 `base-aligned-v1` / `judge-base-aligned-v1`，目前未启动，配置见 [运行前核对](evaluation-alignment.md)。旧百炼代码、账本、失败 pending 和 v1/v3/v4 证据保留用于诊断，不作为当前后端或新成绩。

当前凭据为私有 DOUBAO_SEARCH_API_KEY，运行不再需要 DASHSCOPE_API_KEY。豆包 live 调用记录在 doubao-usage.jsonl，缓存命中排除，保留 RequestId、返回数量、原始业务码与空结果分类；裁判汇总为 doubao_image_search。没有将“无结果”当作免费或把调用次数等同于账单。visual 单价及实际费用仍需控制台核对。

上述是接入、三图行为、缓存和摘要链检查，未包含长期负载或 200 题准确率结论。当前先汇报新配置，不启动评测；定时任务保持暂停。
