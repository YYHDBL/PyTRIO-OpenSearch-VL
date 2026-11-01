# PyTRIO × OpenSearch-VL

用 PyTRIO 远程训练 Qwen3.5-4B，让模型通过看图、搜索和图像处理完成多模态问答。仓库包含数据筛选、多模态 SFT、Multi-Turn Search Fatal-Aware RL、工具环境和评测代码。

本地或 CPU 服务器负责数据、工具执行、奖励计算和流程调度，主模型训练与采样由 PyTRIO 执行。实验借鉴 [OpenSearch-VL](https://github.com/shawn0728/OpenSearch-VL) 的公开数据和方法，使用更小的模型与数据规模，不代表论文全量复现成绩。

## 实验结果

SFT 使用 2,048 条完整轨迹，LoRA rank 32，训练一轮、512次更新。RL 从 SFT State 开始，计划100道任务，实际93道任务、372条在线 Rollout 参与25次更新，另留16道开发验证题。

| 同一99道评测题 | Base | SFT | SFT + RL |
| --- | ---: | ---: | ---: |
| 答对题数 | 53 | 50 | 49 |
| 正常结束并作答 | 82 | 97 | 97 |
| 工具调用次数 | 693 | 376 | 380 |
| 主模型输出 tokens | 430,635 | 98,033 | 98,766 |

SFT 减少了反复搜索和长输出；RL 后格式奖励改善，但本轮未观察到准确率提升。完整100题中，SFT有1题服务故障，因此三阶段采用共同完成的99题。评测采用 GLM-5.3-Flash 裁判；RL评测的摘要部署与前两阶段存在差异，不能将所有变化归因于训练。

- [完整结果与逐题分析](docs/training-stage-comparison.md)
- [RL benchmark 与开发验证集结果](docs/rl-results.md)
- [训练配置与执行流程](docs/training.md)

## 工具环境

保留作者七类能力、八个工具名，沿用原始提示词。`text_search` 和 `web_search` 是两个文字检索入口。

| 能力 | 实现 |
| --- | --- |
| 文字检索、网页读取 | DeepSeek Search / Fetch |
| 以图搜图 | 豆包搜索 Global visual |
| OCR / 版面解析 | PaddleOCR-VL-1.6 |
| 裁剪、锐化、透视校正 | Pillow / OpenCV，CPU 执行 |
| 超分辨率 | EDSR x4，CPU 执行 |
| 检索结果摘要 | 独立摘要模型，默认使用 PyTRIO |
| 答案与搜索质量评分 | GLM-5.3-Flash，Anthropic Messages 接口 |

搜索、OCR 和裁判需要各自的服务权限，费用独立于 PyTRIO。[配置说明](docs/services.md)列出所需变量和服务入口。

## 安装

推荐 Python 3.13，使用 [uv](https://docs.astral.sh/uv/) 管理环境：

```bash
git clone https://github.com/YYHDBL/PyTRIO-OpenSearch-VL.git
cd PyTRIO-OpenSearch-VL
uv sync --frozen --extra train
cp .env.example .env
chmod 600 .env
```

填写 `.env` 后检查配置：

```bash
uv run python -m opensearch_vl.check_services
```

该命令只检查配置；真实服务调用、训练和推理会产生用量。数据、模型处理器与 EDSR 权重需自行准备，仓库不附带私有权重或实验账号。[数据准备](docs/data.md)说明公开来源与目录约定。

## 使用

从数据准备到训练、评分的命令见[训练指南](docs/training.md)。所有命令从仓库根目录执行。

```bash
# 查看入口参数，不启动实验
uv run python -m opensearch_vl.train_sft --help
uv run python -m opensearch_vl.train_rl --help
uv run python -m opensearch_vl.agent_baseline --help
uv run python -m opensearch_vl.judge_baseline --help

# 离线测试，不调用付费服务
uv run python -m unittest discover -s tests -v
```

SFT 保存 State 和 Sampler 两类权重：前者恢复训练，后者用于采样。RL 默认只执行预检，必须显式传入 `--execute` 才会开始训练。断点、用量账本与不确定请求保护均保留；恢复实验前需检查现有输出，不应直接删除 pending 文件。

## 代码结构

| 位置 | 内容 |
| --- | --- |
| `opensearch_vl/train_sft.py`、`protocol.py`、`audit_sft.py` | 多模态数据转换、监督 mask、SFT 与审计 |
| `opensearch_vl/train_rl.py`、`rl_*.py` | 在线采样、奖励、RLOO 优势、Fatal-Aware mask、PPO 更新 |
| `opensearch_vl/agent_baseline.py`、`agent_tools.py` | Agent 运行与七类工具调度 |
| `opensearch_vl/deepseek_search.py`、`doubao_image.py`、`paddle_ocr.py` | 外部服务适配、缓存与用量记录 |
| `opensearch_vl/select_*.py` | 数据筛选和去重 |
| `opensearch_vl/judge_baseline.py`、`answer_extraction.py` | 答案提取与裁判评分 |
| `analysis/` | 结果统计、真实轨迹导出和绘图 |
| `tests/` | 离线回归测试 |
| `docs/`、`docs/evidence/` | 使用说明、结果报告和聚合证据 |
| `templates/` | 多模态对话模板 |

运行生成的 `data/`、`outputs/`、`logs/`、密钥与训练权重均不提交。历史交接、排障、一次性启动脚本和未发布博文也不属于公开仓库内容。

## 图表

```bash
uv sync --frozen --extra train --extra plots
uv run python analysis/plot_rl_metrics.py
uv run python analysis/plot_training_stages.py
```

上述两项使用已发布的聚合数据。SFT绘图还需要运行日志导出的 `outputs/sft-figures-v1/metrics.jsonl` 和 `summary.json`；不附带原始记录时不能从零生成 SFT 曲线。绘图脚本当前使用 macOS 中文字体路径，其他系统需指定或替换为可用中文字体。

## 来源与许可

- [OpenSearch-VL 论文](https://arxiv.org/abs/2605.05185)与[代码](https://github.com/shawn0728/OpenSearch-VL)，参考提交 `1ac26e2c35a3e813e022813c88888db457d88f7d`。
- [公开 SFT 轨迹](https://huggingface.co/datasets/OpenSearch-VL/Search-VL-SFT-36K)、[RL 任务](https://huggingface.co/datasets/OpenSearch-VL/Search-VL-RL-8K)、[评测数据](https://huggingface.co/datasets/Osilly/Vision-DeepResearch-Eval)。
- [PyTRIO 文档](https://docs.pytrio.com/docs)与 [PyTRIO Skill](https://github.com/SwanHubX/pytrio-skill)。

本仓库代码采用 [Apache-2.0](LICENSE)。数据、模型及第三方服务遵循各自许可与使用条款。
