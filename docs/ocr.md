# 飞桨 OCR 接入

本实验使用用户提供的 `PaddleOCR-VL-1.6`，任务入口为 `https://paddleocr.aistudio-app.com/api/v2/ocr/jobs`，采用 Bearer Token 认证。真实 Token 仅保存在被 Git 忽略的私有 `.env` 中，权限为 600。

原 OpenSearch-VL 包装器调用同步 layout-parsing 服务；本适配器将飞桨的异步 JSONL 结果规范为相同的 `rec_texts`、`formatted_text`、`blocks` 等字段。工具反馈使用原包装器的 `Layout Parsing SUCCESS` 和 `ALL RECOGNIZED TEXT` 文本。文字块标签仍按作者的 paragraph_title、text、vision_footnote 过滤，其他版面块保留在结果缓存中。不能据此宣称 PaddleOCR-VL-1.6 与作者未公开的后端模型相同。

## 调用与恢复

`layout_parsing` 读取当前轨迹登记的 `image` 引用，不接受模型任意指定服务器文件。上传文件后保存任务 ID，查询 pending/running/done/failed 状态，默认一次等待上限 180 秒；单次网络请求和 GET 重试也有独立上限。查询网络错误或 429/5xx 最多尝试三次，POST 不自动重试。

提交前写入持久化标记。若未确认返回任务 ID 就断线，后续自动调用会被拒绝，需要先在服务平台核对，避免盲目重复上传。有任务 ID 时会继续查询原任务；成功后按图片 SHA-256、模型和选项缓存结果。当前提交记录上限为 1000 次尝试，不代表账户账单。工具调用次数和实际上传尝试分别记录。

结果 JSONL 下载不发送 OCR Token，不下载 Markdown 内的图片或可视化图。只保存规范化结果、任务信息和调用记录；这些文件保存在服务器 `data/tool_cache/ocr-v1/`，不会提交到 Git。

## 实测

2026-10-27，第一张真实 SFT 图片返回 1 个 image 块，没有文字，确认链路可用但不能作为文字提取成功证据。另一张数据集中的真实界面截图 `livevqa/images/interface_v5.png` 返回 1 页，包含 4 个 paragraph_title、5 个 text、1 个 table、2 个 image，按作者规则提取 9 个文字块、499 个字符。再次调用命中缓存，没有新增提交。

测试期间发生过连接和读取超时；累计记录过 3 次上传尝试、取得 2 个任务 ID，其中一次未收到 ID 的提交结果无法从客户端确认。上述防重复提交保护在后续代码中阻止这一情况自动重传。该免费接口的长时稳定性尚未验证。

```sh
.venv/bin/python check_ocr.py \
  --image data/source/sft/livevqa/images/interface_v5.png \
  --output outputs/ocr-smoke-text-v1.json --require-text
```

测试没有调用 PyTRIO 或裁判模型，也没有恢复已停止的 baseline 或启动训练。七工具的其余环境和统一协议仍需验收。

## EDSR 权重

原代码默认的 [EDSR_x4.pb](https://github.com/Saafke/EDSR_Tensorflow/blob/master/models/EDSR_x4.pb) 文件经仓库元数据核对为 38573255 字节，约 38.6 MB（36.8 MiB），Git blob 为 `09b872b26a6ba8d43abe433e65e65477ff8714ea`。当前尚未下载或运行该权重。运行依赖 OpenCV contrib 的 dnn_superres；CPU 耗时与内存取决于图片大小，不能由权重文件体积推断。
