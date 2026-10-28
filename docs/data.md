# 远端完整数据与参考源码

完整原始数据保存在 CPU 服务器，开发电脑不下载数据集。实验采用 512 条 SFT、128 道 RL 题和 100 题评测；完整原始数据用于筛选、去重和核验。原 200 题候选池完整保留，不转入训练。

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

2026-10-28，按用户要求从上述冻结候选池再固定 100 题，保存至 `data/selected-eval100-v1/eval.jsonl`：SimpleVQA、InfoSeek、VDR 各 20，另外四项各 10；BrowseComp-VL Level 1/2 为 3/7。使用共享 Python random.Random(42)，按 benchmark 名排序，组内按原 sample_id 排序后 sample；BrowseComp-VL 按 source_file 分别抽 3/7，最终按原 sample_id 排序。不读取或依据 smoke 正确性、耗时或工具次数筛题，保留原 ID、题面、答案和图片路径。原 200 题及训练集文件不改，图片哈希再次核验；剩余候选不加入训练。

新 100 题 SHA-256 为 `9f2f0e3d9117defccccff5a78eaa05099bdd78b3e489eb0540953900e1f5a6f1`。选择算法、全部 ID、配比及父集哈希见服务器同目录 manifest.json 与 [聚合清单](evidence/eval100-v1-manifest.json)。Base/SFT/RL 必须使用同一份新 100 题；没有进一步逐题型均匀分层，不代表原论文全量 benchmark。

七工具 SFT 使用 `data/selected-seven-v1/sft.jsonl`，由 `select_seven_sft.py` 在冻结 eval/RL 之外重新筛选。七个来源分别取 74、73、73、73、73、73、73 条，保留完整图片及作者 system/tools；图片占位符与图片数一致，对话不超过 32000 字符。选集实际包含 crop 58、image_search 470、layout_parsing 41、perspective_correct 2、sharpen 4、super_resolution 9、text_search 543、web_search 2 次调用。八个工具名均覆盖，稀有工具样本较少。字符限制只是初筛，后续图文 token 和 mask 审计已 512/512 通过，详见 [验收报告](sft-audit.md)。旧 `data/selected/sft.jsonl` 是仅筛三类实际动作的历史版本，不用于新的七工具 SFT 训练。

RL 排除空参考答案、超过 500 字符的答案和明显未完成的思考语句；各来源至少 5 题，其余按候选数量分配。上述规则不能保证参考答案的事实正确性。

抽样按评测、RL、SFT 的顺序进行；跨集合检查归一化题面和图片字节 SHA-256，集合内避免重复图片。相同问句配不同图片可在同一训练集合出现；图片重编码或语义改写不一定被发现。

VDR 发布的 `fvqa_300.parquet` 是本次 FVQA 子集的直接来源；OpenSearch-VL 论文引用的 2017 FVQA 与该文件的具体来源关系尚未确认，不能声称两者完全相同。原项目的 [评测划分问题](https://github.com/shawn0728/OpenSearch-VL/issues/10) 也尚未给出作者答复。

## 参考源码

作者原始 [OpenSearch-VL](https://github.com/shawn0728/OpenSearch-VL) 仓库单独放置，与本实验仓库并列，用于查阅数据处理、工具协议、奖励与 fatal-aware 实现。它的训练依赖不安装到本实验的 PyTRIO 环境中。
