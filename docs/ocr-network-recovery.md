# OCR 连接恢复记录

本文是 2026-10-27 的历史连接恢复记录，保留原故障证据及用户追加的 Serper 说明。当前服务已全部改为服务器直连，OCR 显式代理留空，DeepSeek 替代 Serper/Jina、豆包替代百炼；不要按下文历史配置恢复 Mac 代理。当前暂停边界与新评测配置见 [接手说明](agent-handoff.md) 和 [运行前核对](evaluation-alignment.md)。

以下记录当时恢复七工具 smoke 的过程。

## 定位证据与修改

服务器直连 `paddleocr.aistudio-app.com` 的复测中，DNS 约 0.05 秒、TCP 约 0.08 秒，随后 TLS 握手超时。原反向代理 `127.0.0.1:10080` 在 CONNECT 阶段超时。Mac 直连及其 `127.0.0.1:1082` 代理均可正常访问同一 HTTPS 地址；未认证 GET 返回 HTTP 405。

新增独立反向隧道 `127.0.0.1:10081` → Mac `127.0.0.1:1082` 后，服务器探测同一地址返回 HTTP 405，TLS 约 0.77 秒。保留原 10080 隧道和系统 DNS；以上证明当前存在直连 TLS 与原代理链路问题，不能把全部故障归结为 DNS。

`paddle_ocr.py` 的任务请求和结果下载都读取显式 `PADDLE_OCR_PROXY`，留空时保持直连；结果下载仍不发送 OCR Token。服务器私有 `.env` 已设 `PADDLE_OCR_PROXY=http://127.0.0.1:10081`。smoke 进程的通用代理也指向可用的 10081；Serper 等明确直连的实现保持原行为。

本地和服务器相关单元测试均为 25 个通过，Python 编译与 `git diff --check` 通过。现有测试另有未关闭读取文件的 ResourceWarning，与本次代理修改无关。

## 安全恢复断点

原始故障目录为服务器 `outputs/smoke-seven-v1/`。`eval-0000` 只完成了第一轮模型输出，在执行 `layout_parsing(img_1)` 前保存断点，POST 抛 `ConnectError`，未进入 HTTP 请求发送阶段。

恢复前核对了故障日志、断点中记录的错误类型、图片 SHA-256、原配置及代码组合哈希，确认相对旧运行只有 OCR 显式代理支持这一项源代码变化。原 `.submitting.json` 移入 `outputs/ocr-network-recovery/` 存档，原运行和日志保留；读取/写入超时等提交状态不确定的情况仍不能据此自动重传。

新目录 `outputs/smoke-seven-v2/` 继承第一轮模型输出、token ledger 和服务计数，更新配置与代码哈希。完整迁移依据保存在 `outputs/ocr-network-recovery/recovery.json`。没有重新采样已保存的第一轮，也没有将旧三工具 baseline 混入此次运行。

## 真实调用结果

恢复后的 `eval-0000` 已重新提交 OCR，取得任务 ID，轮询到 `done`，下载并缓存结果，然后继续模型推理并生成最终答案。结果为 1 页、0 个文字块；这是该图片的成功解析响应，不能把它作为文字识别准确率证据。此前真实界面截图提取 499 字符的记录仍保留。

3 题 smoke 使用 `eval-0000`、`eval-0100`、`eval-0190`，运行日志在 `logs/smoke-seven-v2.log`，结束码在 `logs/smoke-seven-v2.exit`。完整三题及评分结果应以服务器实时输出为准，单题成功不等于全套评测完成。

迁移继承了原 `started_at`，所以 `eval-0000` 的 `elapsed_seconds` 含此前网络故障和暂停时间，不能直接当作连续推理耗时或 GPU 用时。

## Serper 间歇连接失败与 smoke-v3 迁移（同日追加）

`eval-0000`、`eval-0100` 完成后，`eval-0190` 在 Serper POST 上抛 `ConnectError`。实测直连 `google.serper.dev` 时通时断（三次探测一次超时、两次正常 403），与 OCR 直连问题同类但为间歇性，代理 10081 亦可通。这不是服务故障，而是单次连接失败即崩溃的运行策略过于脆弱。

代码修改（仅连接阶段，不改变任何工具语义与提示词）：

- `ToolEnvironment.request()` 对 `ConnectError`/`ConnectTimeout` 做最多 3 次尝试（退避 1s/2s）。连接未建立意味着请求必然未发出，重试不会重复计费或产生副作用；发送后的任何失败仍立即上抛。
- 百炼 `search()` 在连接未建立时删除 `.pending.json` 并提示可安全重试；发送后失败仍保留 pending，禁止自动重传。

两端单元测试 27 个通过。smoke 按 `smoke-seven-v2` 同样的迁移纪律迁至 `outputs/smoke-seven-v3/`：继承两题完成结果、`eval-0190` 断点与全部 token 账本，先让脚本生成新 `config.json` 再修补记录哈希，无任何重新采样；配置差异仅 `code_sha256` 与衍生 `sha256`。迁移依据存 `outputs/ocr-network-recovery/recovery-serper-retry.json`。工具缓存仍共享 `data/tool_cache/author-seven-v3`，已缓存的工具结果不重复计费。

## Jina 代理恢复与隧道维护（同日追加）

`smoke-seven-v3` 的原进程没有通用代理环境，Jina 直连多次连接重试。确认主模型和摘要账本均无进行中的请求后，只终止该进程并带标准 HTTP(S)_PROXY 重启原目录，配置和轨迹前缀保持不变。依据在服务器 `outputs/ocr-network-recovery/recovery-jina-proxy.json`。

复测发现旧 10081 隧道仍监听，但 HTTPS 失败；Mac 的 1082 访问同一 Jina 地址正常。新隧道先在 10082 验证 HTTP 200，再替换本实验原 10081 转发；替换后的 10081 复测 HTTP 200，约 3 秒。用户的 10080 隧道与系统 DNS 均保留。新的独立 SSH master 使用 Mac 项目下被忽略的 `outputs/experiment-tunnel.sock`，15 秒 keepalive、连续 3 次无响应后退出。恢复时仍须实际 HTTPS 探测，不能仅凭监听端口或 SSH 进程存在判断可用。

## 百炼回包解析失败，正式运行暂时停止（同日追加）

正式 `base-seven-v1` 完成 eval-0000 后，在 eval-0001 的首个 image_search(img_1) 调用中抛 `ValueError`，外部适配器报错并保留 pending，脚本退出码 1。提交时间为北京时间 2026-10-27 18:41:35，模型 qwen3.8-27b，图片 SHA-256 为 `aca9d78581f7b980d3023e45b802f3f47e66a28c50cf9f77ca8a06e4b71fb702`。断点、账本和原提交均保留，没有再次 POST，没有将服务故障当作答错。

当前适配器仅在完整解析成功后缓存结果；此次没有保存原始响应、响应 ID 或具体 ValueError 原因，这是诊断记录的缺口。现有证据不能区分 JSON 解码失败、顶层状态未完成、工具输出结构不符等原因，不能据此断言百炼服务故障，也不能放宽校验或替换工具来继续。百炼成功账本为空不代表该次没有计费。

已核查 [官方图搜图文档](https://help.aliyun.com/zh/model-studio/image-search) 和 [获取响应接口](https://help.aliyun.com/zh/model-studio/retrieve-a-response)。获取接口需要 Response ID 且原调用 store=true；本次使用 store=false，无法靠现有 API 状态直接取回原回包。应先取得平台该时段的调用日志或原始响应；若用户明确允许一次可能重复计费的诊断请求，再以相同图片、模型与参数调用一次，先把完整回包、状态码、响应头中的请求 ID 存服务器私有输出，然后离线定位。该授权前不删除 pending、不自动重发，不调整提示词、工具或冻结数据。

故障依据：服务器 `outputs/ocr-network-recovery/failure-bailian-seven-v1.json`。本次只检查和记录故障，没有更改运行源代码，也没有迁移为新协议。

## 使用边界

新隧道依赖 Mac 在线、1082 代理可用及 SSH 连接保持。再次故障时先做无认证 HTTPS 探测，并核对已有任务 ID、缓存与提交标记，避免重复上传。当前尚未验证长时间批量运行稳定性，三题 smoke 不计入新的 200 题正式 baseline 成绩。

若新隧道已经退出，在 Mac 重新启动下列命令；已有健康隧道时无需重复启动：

```sh
ssh -N -T \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -R 127.0.0.1:10081:127.0.0.1:1082 \
  jyh-yyh@172.29.5.2
```
