# 数据准备与划分

原始图片、轨迹、模型处理器和切分文件放在执行训练的 CPU 机器上，不随代码提交。

## 公开来源

| 数据 | 固定版本 | 放置位置 |
| --- | --- | --- |
| [Search-VL-SFT-36K](https://huggingface.co/datasets/OpenSearch-VL/Search-VL-SFT-36K) | `2c1c460af4fa15bd63210cbf426a96664b959944` | `data/source/sft/` |
| [Search-VL-RL-8K](https://huggingface.co/datasets/OpenSearch-VL/Search-VL-RL-8K) | `8ef567289043eef004b13da83b0e7bb7f5ae2daa` | `data/source/rl/` |
| [Vision-DeepResearch-Eval](https://huggingface.co/datasets/Osilly/Vision-DeepResearch-Eval) | `deeaf45779a3bbd407d8f0ccb9b4831fc78e81c9` | 查看 `select_experiment_data.py` 中的源目录约定 |

使用 Hugging Face CLI 下载指定版本，按发布包解压图片；先校验文件大小与哈希、ZIP完整性和图片可解码性。SFT 与 RL 发布文件本次合计约15.78GB，解压图片及缓存需要额外空间。

另需准备 Qwen3.5-4B 的 tokenizer / processor 文件、作者源码和 EDSR x4 权重；不需要在 CPU 机器加载主模型训练权重。

## 筛选流程

`select_experiment_data.py` 生成200题评测候选池、128道RL候选和初始SFT切分；它们是实验中间数据，不等于最终训练规模。脚本采用固定种子，输出目录已存在时拒绝覆盖。

最终100题从200题候选池按固定种子42分层抽取：SimpleVQA、InfoSeek、VDR各20题，FVQA、MMSearch、LiveVQA、BrowseComp-VL各10题；BrowseComp-VL中Level 1/2为3/7。完整ID和抽样方式见[冻结清单](evidence/eval100-v1-manifest.json)。最终文件放在 `data/selected-eval100-v1/eval.jsonl`，SHA-256为 `9f2f0e3d9117defccccff5a78eaa05099bdd78b3e489eb0540953900e1f5a6f1`。当前仓库没有将这一步的一次性抽样过程封装成独立CLI；复用已发布清单重建时需验证哈希。

`select_balanced_sft.py --model-dir <tokenizer目录>` 从公开轨迹筛选2,048条，检索 / 裁剪或OCR / 图像增强约60% / 30% / 10%。检查实际工具动作、完整结束、重复动作、图片、token长度和assistant-only mask；使用种子42，输出 `data/selected-seven-2048-v1/sft.jsonl`。筛选依赖初始七工具数据和审计结果，需先执行 `select_seven_sft.py` 与 `audit_sft.py`，审计输出放在 `outputs/sft-audit-seven-v1/`。[分布](evidence/sft2048-v1-manifest.json)和[CPU审计](evidence/sft-audit-2048-v1.json)保留聚合记录。

RL先对128道候选各采样4次，按完整题组、奖励和题面 / 图片去重后，整理100道训练题与16道开发验证题。[选题记录](evidence/rl-selection-100-v1.json)列出来源和分布；训练入口读取执行机器上的 `train-ids.json` 与 `validation-ids.json`，诊断轨迹不用于参数更新。该次划分依赖已有诊断结果，不是下载数据后直接得到的通用划分。

跨集合检查归一化题面与图片字节哈希。这不能完全识别重编码图片或语义改写。筛选未对所有参考答案重新人工核验，评测子集也不能代表论文全量成绩。
