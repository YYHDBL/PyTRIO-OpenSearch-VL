# 远端完整数据与参考源码

完整原始数据保存在 CPU 服务器，开发电脑不下载数据集。实验采用 512 条 SFT、128 道 RL 题和 200 题评测；完整原始数据用于筛选、去重和核验。

## 数据版本

按 Hugging Face 发布文件大小统计，以下总量包含图片 ZIP、轨迹 JSON、额外 plan-mode 轨迹以及发布说明等文件。该大小为下载量，解压需要额外空间。

| 数据 | 固定 revision | 文件数 | 下载字节数 |
|---|---|---:|---:|
| `OpenSearch-VL/Search-VL-SFT-36K` | `2c1c460af4fa15bd63210cbf426a96664b959944` | 24 | 13,073,649,606 |
| `OpenSearch-VL/Search-VL-RL-8K` | `8ef567289043eef004b13da83b0e7bb7f5ae2daa` | 5 | 2,707,030,005 |
| 合计 | | 29 | 15,780,679,611（约 15.78 GB） |

在本实验仓库的远端根目录执行：

```sh
uv run --frozen hf download OpenSearch-VL/Search-VL-SFT-36K \
  --repo-type dataset \
  --revision 2c1c460af4fa15bd63210cbf426a96664b959944 \
  --local-dir data/source/sft --max-workers 2

uv run --frozen hf download OpenSearch-VL/Search-VL-RL-8K \
  --repo-type dataset \
  --revision 8ef567289043eef004b13da83b0e7bb7f5ae2daa \
  --local-dir data/source/rl --max-workers 2
```

官方 CLI 会复用已完成文件；断线后重跑相同命令继续，避免添加 `--force-download`。当前使用 HTTP 代理时，应先确认远端代理端口可用，并设置 `HF_HUB_DISABLE_XET=1` 走 HTTP 下载。

SFT 与 RL 的发布文件已逐一与官方版本的大小、Git blob 或 LFS SHA-256 核验；八个图片 ZIP 均已检查 CRC 并解压。

## 固定切分

运行 `select_experiment_data.py` 后，服务器 `data/selected/` 下包含 `eval.jsonl`、`sft.jsonl`、`rl_train.jsonl`、`eval_images/` 和 `manifest.json`。数据文件不会提交到 Git。

评测从 [Vision-DeepResearch-Eval](https://huggingface.co/datasets/Osilly/Vision-DeepResearch-Eval) 固定版本 `deeaf45779a3bbd407d8f0ccb9b4831fc78e81c9` 的已发布子集抽样，种子 42：SimpleVQA、InfoSeek、VDR 各 40 题，FVQA、MMSearch、LiveVQA、BrowseComp-VL 各 20 题；BrowseComp-VL 中 Level 1 为 6 题，Level 2 为 14 题。来源文件、行号、原始 ID 和图片哈希均保存在记录中。

SFT 从七个作者数据来源分别抽取 74、73、73、73、73、73、73 条，仅保留实际调用 crop、image_search、text_search 的轨迹，保留全部图片，要求对话中的图片占位符与图片数一致、对话文本不超过 32000 字符。字符限制只是初筛；训练前必须检查实际图文 token 数与 mask。

RL 排除空参考答案、超过 500 字符的答案和明显未完成的思考语句；各来源至少 5 题，其余按候选数量分配。上述规则不能保证参考答案的事实正确性。

抽样按评测、RL、SFT 的顺序进行；跨集合检查归一化题面和图片字节 SHA-256，集合内避免重复图片。相同问句配不同图片可在同一训练集合出现；图片重编码或语义改写不一定被发现。

VDR 发布的 `fvqa_300.parquet` 是本次 FVQA 子集的直接来源；OpenSearch-VL 论文引用的 2017 FVQA 与该文件的具体来源关系尚未确认，不能声称两者完全相同。原项目的 [评测划分问题](https://github.com/shawn0728/OpenSearch-VL/issues/10) 也尚未给出作者答复。

## 参考源码

作者原始 [OpenSearch-VL](https://github.com/shawn0728/OpenSearch-VL) 仓库单独放置，与本实验仓库并列，用于查阅数据处理、工具协议、奖励与 fatal-aware 实现。它的训练依赖不安装到本实验的 PyTRIO 环境中。
