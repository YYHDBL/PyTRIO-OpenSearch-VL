# 远端完整数据与参考源码

完整原始数据保存在 CPU 服务器，开发电脑不下载数据集。训练仍采用约定的 512 条 SFT、128 个 RL 任务和 100 题评测；完整原始数据用于筛选、去重和核验。

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

当前远端后台下载日志为 `logs/download-sft.log` 和 `logs/download-rl.log`。相应的 `logs/download-sft.exit`、`logs/download-rl.exit` 出现且内容为 `0`，表示对应 CLI 正常完成；没有退出文件时仍需检查进程和日志，不能视为下载完成。

所有下载完成后还要核验文件大小、ZIP 完整性、解压路径与图片引用。目录存在或下载进程已启动，都不代表数据已可用于训练。

## 参考源码

作者原始 [OpenSearch-VL](https://github.com/shawn0728/OpenSearch-VL) 仓库单独放置，与本实验仓库并列，用于查阅数据处理、工具协议、奖励与 fatal-aware 实现。它的训练依赖不安装到本实验的 PyTRIO 环境中。
