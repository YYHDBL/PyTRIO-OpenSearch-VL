# PyTRIO-OpenSearch-VL

用 PyTRIO 做 OpenSearch-VL 核心方法的小规模复现：图片问答、工具调用、SFT 冷启动、fatal-aware RL 和固定评测。

## 实验范围

| 项目 | 配置 |
|---|---|
| 基模 | Qwen3.5-4B，LoRA rank 32 |
| SFT | 作者公开轨迹中筛选 512 条，1 epoch |
| RL | 128 道训练题，每题 4 条轨迹，共 512 条 rollout |
| 工具 | 裁剪、以图搜图、文字搜索与网页读取 |
| 评测 | 100 题冻结子集，比较 Base、SFT、RL |
| 预算 | 暂按 PyTRIO ¥500 上限安排，先小试实测消耗 |

这是更换基模和训练栈、缩小规模后的方法复现。结果尚未产生，不能据此宣称复现论文成绩。

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

- `prepare_data.py`：对已准备的轨迹与图片抽样、验证和去重。尚未实际生成 512/128/100 切分，数据语义验收和 token 长度检查仍需完成。
- `protocol.py`、`templates/`：图文输入与 assistant-only SFT mask 草稿，尚未用真实数据验证。
- `train_sft.py`：PyTRIO LoRA SFT 入口草稿，未运行远程训练。
- 多轮 rollout、搜索适配、fatal-aware RL 与统一评测尚待实现。

在远程执行环境准备好数据并完成核验后再运行 SFT；下面命令会产生 PyTRIO 消耗：

```sh
uv sync --extra train
uv run python prepare_data.py \
  --sft-json /server/data/sft/source.json \
  --sft-image-root /server/data/sft \
  --rl-jsonl /server/data/rl/rl_data.jsonl \
  --rl-image-root /server/data/rl \
  --output-dir /server/experiment/selected

uv run python train_sft.py \
  --data /server/experiment/selected/sft.jsonl \
  --image-root /server/data/sft \
  --output-dir /server/experiment/outputs/sft
```

## 来源

- [OpenSearch-VL 原项目](https://github.com/shawn0728/OpenSearch-VL)，参考版本 `1ac26e2c35a3e813e022813c88888db457d88f7d`。
- [SFT 轨迹](https://huggingface.co/datasets/OpenSearch-VL/Search-VL-SFT-36K)与 [RL 任务](https://huggingface.co/datasets/OpenSearch-VL/Search-VL-RL-8K)。
- [PyTRIO 文档](https://docs.pytrio.com/docs)。
- [Spec-o3 的 PyTRIO 迁移案例](https://pytrio.com/blog/spec-o3-zh/)及其[参考实现](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/09-spec-o3)。
