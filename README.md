# PyTRIO-OpenSearch-VL

2026-10-29：[RL训练前难度诊断](docs/rl-diagnostic.md)正在运行，固定SFT模型对128道训练题各采样4条、32k窗口、Flash奖励裁判；只诊断，不更新权重或自动筛题。实时进度以服务器输出为准。[SFT评测](docs/sft-eval100.md)已有99题评分，GLM判定50题正确；相同99题的Base为53题正确，另1题服务故障保留，不能宣称100题完整评测。[LoRA冷启动](docs/sft-2048.md)已完成512次更新，两类权重均验证。旧512模型保留，定时任务保持暂停。

[作者RL流程的PyTRIO适配](docs/rl-training-preparation.md)已接入独立训练入口train_rl.py，默认零网络预检，包含原始采样token/logprob、RLOO、fatal mask、PPO custom loss和State/Sampler保存恢复。本地71项、服务器相关14项离线测试通过；真实远端RL更新尚未实测，按用户要求等诊断结束后进行。

用 PyTRIO 做 OpenSearch-VL 核心方法的小规模复现：图片问答、工具调用、SFT 冷启动、fatal-aware RL 和固定评测。

原[512条SFT](docs/sft.md)和扩大后的2048条SFT均已完成，分别保存于服务器 `outputs/sft-seven-v2/`、`outputs/sft-2048-v1/`。Base评测和评分在 `outputs/base-aligned-100-v1/`、`outputs/judge-base-aligned-100-v1/`，均完整结束，不重复启动。此前[十题smoke](docs/smoke-aligned-10-v1.md)完整、4/10正确；当前SFT评测输出为 `outputs/sft2048-aligned-100-v2/`。评测配置见[运行前核对](docs/evaluation-alignment.md)。旧批次及未确认请求保留；服务直连，定时任务保持暂停，RL未启动。

## 实验范围

| 项目 | 配置 |
|---|---|
| 基模 | Qwen3.5-4B，LoRA rank 32 |
| SFT | 当前2048条平衡轨迹，1 epoch；旧512条模型保留 |
| RL | 128 道训练题，每题 4 条轨迹，共 512 条 rollout |
| 工具 | 作者七类能力、八个工具名，真实联合调用已通过 |
| 评测 | 100 题冻结子集，比较 Base、SFT、RL；原 200 题保留 |
| 预算 | 暂按 PyTRIO ¥500 上限安排，先小试实测消耗 |

这是更换基模和训练栈、缩小规模后的方法复现。正式 Base/SFT/RL 对比尚未完成，不能据此宣称复现论文成绩。

七工具 SFT 已完成2048/2048 CPU审计及真实一轮训练；正式SFT评测的实时状态以服务器输出为准，fatal-aware RL尚未运行。此前三工具 `base-v1` 的145题只作诊断，程序拒绝续跑该目录。协议差异见[协议核对](docs/protocol-review.md)。

## 数据与执行位置

当前开发电脑只保存代码、依赖、聚合证据与私有凭据，**不下载数据集**。原始图片、轨迹、切分、训练调度和评测位于 CPU 服务器 `/data/jyh-yyh/PyTRIO-OpenSearch-VL`，模型训练与推理由 PyTRIO GPU 执行。

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

- `select_experiment_data.py`：生成原始固定切分；`select_seven_sft.py` 在不改动 200 题评测和 128 道 RL 题的前提下，重选七工具 SFT，输出 `data/selected-seven-v1/sft.jsonl`。具体来源和筛选规则见 [数据说明](docs/data.md)。
- `protocol.py`、`templates/`：图文输入与 assistant-only SFT mask 转换，保留作者 system 文本，并加入独立 tools 字段。验收方法与结果见 [SFT CPU 验收](docs/sft-audit.md)。
- `train_sft.py`：已完成真实训练的 PyTRIO LoRA SFT 入口，512 条、1 epoch；离线检查、逐步日志、双权重保存及恢复保护见 [SFT 说明](docs/sft.md)。最终权重已验证，未据训练 loss 宣称任务准确率提升。
- `agent_baseline.py`、`agent_tools.py`：作者完整 system/tools、七工具执行、搜索缓存、逐题断点及 token/调用上限。文字检索使用 DeepSeek Search/Fetch，图片检索使用豆包 Global visual，两者使用独立冻结的 Qwen3.8-27B 摘要。图像处理使用 PIL/OpenCV/EDSR，OCR 使用飞桨服务。
- `deepseek_search.py`、`service_io.py`：国内搜索/抓取、原始回包留存、安全重试与用量记录。`check_service_stability.py` 在服务器独立目录检查真实服务，不代替正式评测。
- `doubao_image.py`、`check_doubao_image.py`：豆包图搜图接入及真实检查，区分网页与图片来源，保留回包、缓存及用量。`bailian_image.py` 仅保留作历史诊断，评测不调用它。
- `paddle_ocr.py`、`check_ocr.py`：飞桨异步版面解析适配与真实图片检查，使用固定的 PaddleOCR-VL-1.6。验证结果与后端差异见 [OCR 接入](docs/ocr.md)。
- `judge_baseline.py`：沿用 OpenSearch-VL 的裁判提示词，通过 Anthropic Messages 接口调用固定的 GLM-5.3-Flash，记录逐题判定与总体、分来源成绩。该裁判与论文的 GPT-4o 有差异。
- `train_rl.py`、`rl_rollout.py`、`rl_prepare.py`、`rl_loss.py`：fatal-aware RL的PyTRIO适配，默认离线预检，真实训练链路待实测。

## 固定评测

以下有限批次已按用户授权启动，先检查已有进程，不重复执行。正式运行使用独立新目录，不复用诊断批次或十题 smoke 的模型回答：

```sh
nohup ./run_baseline.sh > logs/base-aligned-100-v1-launch.log 2>&1 < /dev/null &
```

脚本显式读取 `data/selected-eval100-v1/eval.jsonl`，清除遗留代理环境，先生成 `outputs/base-aligned-100-v1/`，再按 expected=100 评分到 `outputs/judge-base-aligned-100-v1/`。生成日志为 `logs/base-aligned-100-v1.log`，评分日志为 `logs/judge-base-aligned-100-v1.log`，整段退出码为 `logs/base-aligned-100-v1.exit`。退出码 0 之外，还需检查恰好 100 个固定 ID 和评分摘要 `complete=true`。

Qwen3.5-4B 使用 temperature 0、seed 42、最多 50 个 assistant 轮次。总窗口 65536 tokens，单轮最多生成 32768 tokens，实际按窗口剩余空间扣减；只使用 EOS 停止词。不追加强制最终回答，也不因连续三次模型参数错误提前终止评测。沿用作者完整轨迹答案抽取规则，没有显式最终回答不直接判错。容量证据、剩余差异及限制见 [评测配置](docs/evaluation-alignment.md)。

正式脚本的主模型 token 上限 25M、摘要 8M、图搜请求尝试 800。未返回请求按实际输入加请求输出上限预留，不再固定预留 16384 tokens；未知旧消耗不能按零处理。达到预算停止整批并保留断点。预算不进入配置哈希，这些上限不等于人民币账单。主模型和摘要分别记录 token，豆包真实调用记录 `doubao-usage.jsonl`，裁判汇总 `doubao_image_search`，保留空结果与原始业务码。实际账单需独立核对，第三方费用不由 PyTRIO 代金券覆盖。

代码、数据和参数的哈希写入 `config.json`。重新运行相同命令会跳过已完成题目并恢复中断轨迹；代码或配置变化时必须使用新输出目录。正式评测与流程检查使用不同目录，评测题面中不加入参考答案或发布包的 `image_caption`。

`data/tool_cache/author-seven-v5/` 保存当前后端搜索、网页及摘要，跨阶段复用相同请求。优先使用搜索返回正文，缺失时调用 Fetch；只有目标页明确不可用时才注明使用搜索摘要，网络故障暂停。豆包提供来源网页和图片地址，二者分开保留；缺失来源时明确说明。OCR 缓存另存 `data/tool_cache/ocr-v1/`；原始回包权限 600，不提交 Git。

现有切分已经生成，恢复运行时直接使用。首次在新的服务器准备数据时，可从已校验并解压的原始数据生成；已有输出目录时脚本拒绝覆盖：

```sh
uv sync --frozen
uv run python select_experiment_data.py
uv run python select_seven_sft.py
```

筛选时的字符数限制不等同于多模态 token 数。七工具 SFT 审计 512/512 通过、零超长，见 [验收报告](docs/sft-audit.md)；它不包含梯度与权重更新验证。网络故障、连接阶段重试及安全续跑见 [恢复记录](docs/ocr-network-recovery.md)。

## 来源

- [OpenSearch-VL 原项目](https://github.com/shawn0728/OpenSearch-VL)，参考版本 `1ac26e2c35a3e813e022813c88888db457d88f7d`。
- [SFT 轨迹](https://huggingface.co/datasets/OpenSearch-VL/Search-VL-SFT-36K)与 [RL 任务](https://huggingface.co/datasets/OpenSearch-VL/Search-VL-RL-8K)。
- [PyTRIO 文档](https://docs.pytrio.com/docs)。
- [Spec-o3 的 PyTRIO 迁移案例](https://pytrio.com/blog/spec-o3-zh/)及其[参考实现](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/09-spec-o3)。
