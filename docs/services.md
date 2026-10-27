# 服务申请与配置

核查日期：2026-10-26。先领取可用免费额度，按小试调用量再判断是否需要付费。

| 服务 | 用途 | 申请入口 | .env 变量 |
|---|---|---|---|
| PyTRIO | 模型训练、采样、checkpoint | [控制台](https://pytrio.com/dashboard)、[API Key](https://pytrio.com/keys) | `PYTRIO_API_KEY` |
| Serper | 文字搜索 | [注册](https://serper.dev/signup) | `SERPER_API_KEY` |
| SerpApi | Google Lens 全网以图搜图 | [注册](https://serpapi.com/users/sign_up)、[API Key 控制台](https://serpapi.com/manage-api-key) | `SERPAPI_API_KEY` |
| Jina Reader，可选 | 网页正文读取 | [Reader 与 Key 入口](https://jina.ai/reader/) | `JINA_API_KEY` |
| PaddleOCR | OCR 与版面解析 | [官方任务页](https://aistudio.baidu.com/paddleocr/task) | `PADDLE_OCR_JOB_URL`、`PADDLE_OCR_MODEL`、`PADDLE_OCR_TOKEN` |

Serper 官网目前提供新账号 2,500 次免费搜索，无需信用卡。[官方说明](https://serper.dev/)

SerpApi 免费档每月 250 次搜索、每小时最多 50 次；额度由同账号的各搜索 API 共用。这里只将它用于图片搜索。重复图片可按内容缓存，但新裁剪图会产生新的检索，因此免费额度不能保证覆盖整轮 RL 与三组评测。[官方价格](https://serpapi.com/pricing)

SerpApi 提供上传图片取得 `image_id` 后调用 Lens 的接口，可用于原图和本地裁剪图，上传大小上限为 500 KB。接入时应核验图片上传、有效期与账号限制；当前不需要另行申请对象存储。[官方上传说明](https://serpapi.com/google-lens-upload-an-image)

Jina Reader 不带 Key 可使用基础读取，官网当前列出 20 次/分钟；新 Key 提供 10M 免费 token。读取能力不等于绕过目标网站访问限制。[官方说明](https://jina.ai/reader/)

2026-10-27，飞桨配置为用户提供的异步 Jobs 接口和 `PaddleOCR-VL-1.6`，使用 Bearer Token。提交、轮询、结果 JSONL 解析与缓存已经真实图片验证；它与作者未公开的版面解析后端不能视为已确认同一版本。官方配额文档当前写明每用户每模型每日 20000 页，实际账户可用性仍需以平台为准。[配额说明](https://ai.baidu.com/ai-doc/AISTUDIO/Xmjclapam)、[接入说明](ocr.md)

答案判定、网页摘要与搜索质量评分还需要冻结的模型。若已有 OpenAI 兼容模型接口，可提供 `JUDGE_BASE_URL`、`JUDGE_API_KEY` 和 `JUDGE_MODEL` 复用；否则后续评估 PyTRIO 内的独立冻结 sampler，不必现在额外注册付费模型账号。

将真实密钥填在仓库根目录的私有 `.env`，不要填进 `.env.example`、README 或 GitHub issue。`.env` 已被忽略。只读检查用 `uv run python check_services.py --pytrio`，不会创建 sampler 或训练任务。

数据准备与训练脚本还需一个 CPU 执行环境及工作目录。PyTRIO 负责 GPU 模型计算，不提供本项目数据整理与搜索调度的本地运行位置。确定远程执行位置后再在那里下载数据。
