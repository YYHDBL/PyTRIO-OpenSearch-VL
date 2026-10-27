# PyTRIO-OpenSearch-VL

用 PyTRIO 做 OpenSearch-VL 核心方法的小规模复现：图片问答、工具调用、SFT 冷启动、fatal-aware RL 和固定评测。

## 实验范围

| 项目 | 配置 |
|---|---|
| 基模 | Qwen3.5-4B，LoRA rank 32 |
| SFT | 作者公开轨迹中筛选 512 条，1 epoch |
| RL | 128 道训练题，每题 4 条轨迹，共 512 条 rollout |
| 工具 | 按作者七类能力准备，环境尚未完整验收 |
| 评测 | 200 题冻结子集，比较 Base、SFT、RL |
| 预算 | 暂按 PyTRIO ¥500 上限安排，先小试实测消耗 |

这是更换基模和训练栈、缩小规模后的方法复现。结果尚未产生，不能据此宣称复现论文成绩。

当前付费运行已停止：此前 base-v1 使用自行改写的三工具提示词，SFT 数据则保留作者提示词，145 题结果只作诊断。正在按 [协议核对](docs/protocol-review.md) 修正，七工具环境未验收完成前不恢复正式评测或训练。

## 数据与执行位置

当前开发电脑只保存代码、依赖与私有凭据，**不下载数据集**。原始图片、轨迹、切分、训练调度和评测在后续指定的 CPU 服务器或远程执行环境处理，模型训练与推理由 PyTRIO GPU 执行。

CPU 服务器保存完整 SFT 和 RL 发布数据，合计下载量约 15.78 GB，随后再筛选实验子集。固定版本、下载命令和日志说明见 [docs/data.md](docs/data.md)。原作者代码仓库与本实验仓库分开放置，作为实现参考。

GitHub 仓库只存代码、配置模板和说明，不存数据集或真实密钥。当前没有自动下载数据或自动启动训练的 GitHub Actions。

## 配置服务

```sh
uv sync --python 3.13
cp .env.example .env
chmod 600 .env
```

把密钥填入 `.env`，然后检查配置。检查命令不会输出密钥：

```sh
uv run python check_services.py
uv run python check_services.py --pytrio
```

`--pytrio` 只读查询支持模型，不发起模型推理、创建训练任务或更新参数。服务申请网址与免费额度见 [docs/services.md](docs/services.md)。

## 当前实现

- `select_experiment_data.py`：从服务器上的原始数据中生成固定的 512 条 SFT、128 条 RL 和 200 题评测数据。具体来源和筛选规则见 [docs/data.md](docs/data.md)。
- `protocol.py`、`templates/`：图文输入与 assistant-only SFT mask 转换，保留作者 system 文本，并加入独立 tools 字段。验收方法与结果见 [SFT CPU 验收](docs/sft-audit.md)。
- `train_sft.py`：PyTRIO LoRA SFT 入口草稿，未运行远程训练。
- `agent_baseline.py`、`agent_tools.py`：固定评测集上的多轮评测，支持裁剪、Serper 文字搜索、SerpApi 图片搜索、Jina 网页读取、搜索缓存、逐题断点续跑和 token/工具调用上限。
- `judge_baseline.py`：沿用 OpenSearch-VL 的裁判提示词，通过 Anthropic Messages 接口调用固定的 GLM-5.3-Flash，记录逐题判定与总体、分来源成绩。该裁判与论文的 GPT-4o 有差异。
- fatal-aware RL 尚待实现。

## 固定评测

旧批次 `outputs/base-v1/` 已停止，程序会拒绝续跑。以下是既有接口形式；修正后的完整协议须在新目录重新验收再运行，这些命令会产生模型与搜索服务消耗：

```sh
.venv/bin/python agent_baseline.py --output-dir outputs/base-v1
.venv/bin/python judge_baseline.py \
  --results-dir outputs/base-v1 --output-dir outputs/judge-base-v1
```

默认使用 Qwen3.5-4B、temperature 0、seed 42，每题最多 6 轮工具交互，耗尽后追加一次最终作答，每轮最多生成 1536 token。连续 3 次工具错误终止；空回答、未完成回答计为无最终答案。工具接口失败、网页读取失败、达到轮数上限均保留记录。

评测程序默认总输入和输出上限 500 万 token，图片搜索尝试上限 220 次、文字搜索尝试上限 1500 次。调用上限跨同一缓存目录累计；token 上限按本次输出目录累计。它们用于限制调用量，不等同于人民币账单上限。中断或未返回的模型请求保留 65536 token 预留；SDK 内部重试的真实费用以控制台为准。

代码、数据和参数的哈希写入 `config.json`。重新运行相同命令会跳过已完成题目并恢复中断轨迹；代码或配置变化时必须使用新输出目录。正式评测与流程检查使用不同目录，评测题面中不加入参考答案或发布包的 `image_caption`。

`data/tool_cache/` 保存成功搜索与网页结果，跨 Base、SFT、RL 复用相同请求结果，减少搜索漂移和重复消耗。文字搜索返回前五项，尝试读取前两项正文，每页最多保留 5000 字符；正文读取失败会明确返回 `page_error`，仍保留搜索摘要。这与论文的专门网页摘要模型不同。

从已校验并解压的原始数据生成切分；已有输出目录时脚本会报错，避免覆盖既有结果：

```sh
uv sync --frozen
uv run python select_experiment_data.py
```

筛选时的字符数限制不等同于多模态 token 数。当前数据的长度和 mask 验收见报告；训练前仍须统一完整工具协议并复核 RL 参考答案质量。

## 来源

- [OpenSearch-VL 原项目](https://github.com/shawn0728/OpenSearch-VL)，参考版本 `1ac26e2c35a3e813e022813c88888db457d88f7d`。
- [SFT 轨迹](https://huggingface.co/datasets/OpenSearch-VL/Search-VL-SFT-36K)与 [RL 任务](https://huggingface.co/datasets/OpenSearch-VL/Search-VL-RL-8K)。
- [PyTRIO 文档](https://docs.pytrio.com/docs)。
- [Spec-o3 的 PyTRIO 迁移案例](https://pytrio.com/blog/spec-o3-zh/)及其[参考实现](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/09-spec-o3)。
