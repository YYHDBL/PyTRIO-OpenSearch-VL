# SFT CPU 验收

2026-10-27，对七工具重选后的 `data/selected-seven-v1/sft.jsonl` 共 512 条轨迹完成 CPU 验收，没有创建训练客户端、更新权重或发起新远程采样。转换保留作者 system 文本，并按原仓库的 Qwen JSON 格式加入独立 tools 字段。逐条明细位于服务器 `outputs/sft-audit-seven-v1/samples.jsonl`，聚合证据见 [summary.json](evidence/sft-audit-seven-v1.json)。原始数据和图片只保存在服务器。

| 检查项 | 结果 |
|---|---:|
| assistant mask、target 右移和图文输入顺序 | 512/512 通过 |
| 输入长度上限 | 16384 token |
| 最短 / 中位数 / 平均 / 最长 | 5286 / 8789 / 9035.68 / 14531 |
| P95 | 11829.95 |
| 超过上限的轨迹 | 0 |
| 图文输入 token 总数 | 4626267 |
| 图片数 / 视觉 token 数 | 586 / 375110 |
| 有效监督 token 总数 | 420240 |
| 被屏蔽 token 总数 | 4206027 |
| 被监督的 assistant 结束标记 | 1640 |

验收器独立渲染对话，用 tokenizer 的字符偏移确定 assistant 内容和结束标记区间，再对比实际 `Datum`。检查 system、user、tool observation、图片和控制区间的权重为零；assistant 内容和结束标记参与监督；target 与完整序列右移一位后的监督 token 一致；实际文本和图片 chunk 顺序一致。

使用缓存的 Qwen3.5-4B tokenizer 和官方图片 processor，在 CPU 上按图片尺寸计算视觉 token。另复用 145 道已经产生的 baseline 首轮 `input_tokens`，本地计算与远端返回全部一致，没有新增远程调用。模型配置文件和处理代码哈希记录在聚合证据中。

运行方法：

```sh
.venv/bin/python audit_sft.py \
  --model-dir /data/jyh-yyh/.cache/modelscope/models/Qwen--Qwen3.5-4B/snapshots/master \
  --data data/selected-seven-v1/sft.jsonl \
  --image-root data/source/sft \
  --output-dir outputs/sft-audit-seven-v1
```

输出目录已存在时拒绝覆盖。可用 `python -m unittest test_sft_audit` 验证验收器能拒绝错位 target、错误监督范围和全零监督。

本次检查使用七工具 SFT 内的作者 system prompt、完整工具定义和自定义 ChatML 适配模板。七工具环境另有真实联合调用检查；两种检查均不包含梯度、loss 数值及 LoRA 更新的远程训练验证。提示词或模板变化后必须重新验收。历史三工具动作选集的审计证据 `sft-audit-v2.json` 保留用于追踪，不与新选集统计混用。
