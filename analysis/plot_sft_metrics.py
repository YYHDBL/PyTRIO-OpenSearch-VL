"""从已导出的真实SFT指标生成曲线、数据分布图和CSV，不调用训练服务。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/sft-figures-v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sft-figures-v1"))
    parser.add_argument("--font", type=Path,
                        default=Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"))
    args = parser.parse_args()
    metrics_path = args.input_dir / "metrics.jsonl"
    summary = json.loads((args.input_dir / "summary.json").read_text())
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert hashlib.sha256(metrics_path.read_bytes()).hexdigest() == summary["metrics_export_sha256"]
    assert [r["step"] for r in rows] == list(range(1, summary["updates"] + 1))
    assert sum(summary["source_counts"].values()) == summary["samples"]
    assert sum(r["input_tokens"] for r in rows) == summary["input_tokens"]
    assert sum(r["supervised_tokens"] for r in rows) == summary["supervised_tokens"]
    assert summary["fixed_sample_is_training_not_validation"]
    font_manager.fontManager.addfont(str(args.font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=args.font).get_name(),
        "font.size": 11, "axes.unicode_minus": False, "svg.fonttype": "path",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#CED4DA",
        "axes.labelcolor": "#334155", "text.color": "#172B3A", "xtick.color": "#526477",
        "ytick.color": "#526477", "savefig.facecolor": "white"})
    steps = np.array([r["step"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    weights = np.array([r["supervised_tokens"] for r in rows])
    seconds = np.array([r["seconds"] for r in rows])
    assert np.isfinite(loss).all() and (loss >= 0).all() and (weights > 0).all()
    assert np.isfinite(seconds).all() and (seconds > 0).all()
    window = 32
    smooth = np.convolve(loss * weights, np.ones(window), "valid") / np.convolve(weights, np.ones(window), "valid")
    checkpoints = summary["fixed_training_sample_nll"]
    x = [c["step"] for c in checkpoints]
    y = [c["nll"] for c in checkpoints]
    assert x[0] == 0 and x[-1] == summary["updates"] and len(set(x)) == len(x)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    fig.subplots_adjust(left=.08, right=.96, bottom=.11, top=.81, hspace=.58, wspace=.29)
    fig.suptitle("OpenSearch-VL · SFT 训练记录", x=.08, y=.965, ha="left", fontsize=23, fontweight="bold")
    fig.text(.08, .914, f'{summary["model"]}   /   LoRA rank {summary["rank"]}   /   2,048 条轨迹   /   512 次更新   /   1 epoch',
             fontsize=12, color="#526477")
    fig.text(.08, .865, f'训练循环 {summary["training_update_seconds"] / 60:.1f} 分钟     输入 {summary["input_tokens"] / 1e6:.2f}M tokens     监督 {summary["supervised_tokens"] / 1e6:.2f}M tokens',
             fontsize=12)
    ax = axes[0, 0]
    ax.plot(steps, loss, color="#A9C9E2", linewidth=.75, alpha=.7, label="每批实际 loss")
    ax.plot(steps[window - 1:], smooth, color="#16697A", linewidth=2.1, label="32 步监督 token 加权均值")
    ax.set(title="逐批训练 loss", xlabel="更新步数", ylabel="每监督 token 的 NLL")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.text(0, -.29, "每批题目不同，单步波动不能直接表示训练效果变化。", transform=ax.transAxes, fontsize=9, color="#64748B")
    ax = axes[0, 1]
    ax.plot(x, y, color="#BD6A30", marker="o", linewidth=2, markersize=5)
    ax.set(title="同一训练样本在各 checkpoint 的 NLL", xlabel="更新步数", ylabel="固定训练样本 NLL")
    ax.set_ylim(0, max(y) * 1.24)
    for index in [0, len(x) - 1]:
        ax.annotate(f"{y[index]:.4f}", (x[index], y[index]), xytext=(4, 10), textcoords="offset points", fontsize=10)
    ax.text(0, -.29, "固定训练样本复测；这条曲线不属于验证集 loss。", transform=ax.transAxes, fontsize=9, color="#64748B")
    ax = axes[1, 0]
    source = sorted(summary["source_counts"].items(), key=lambda item: item[1])
    bars = ax.barh([k for k, _ in source], [v for _, v in source], color="#468B9D", height=.65)
    ax.bar_label(bars, padding=5, fontsize=10)
    ax.set(title="2,048 条 SFT 轨迹的来源分布", xlabel="轨迹数量")
    ax.set_xlim(0, max(v for _, v in source) * 1.18)
    ax = axes[1, 1]
    ax.plot(steps, seconds, color="#6B7C93", linewidth=.8)
    ax.axhline(np.median(seconds), color="#BD6A30", linestyle="--", linewidth=1.5,
               label=f"中位数 {np.median(seconds):.2f} 秒")
    ax.set(title="每次训练更新的客户端耗时", xlabel="更新步数", ylabel="秒 / 次")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=10)
    ax.text(0, -.29, "包含请求等待；不含循环外的 CPU 准备与 checkpoint 保存。", transform=ax.transAxes, fontsize=9, color="#64748B")
    for ax in axes.flat:
        ax.grid(axis="y" if ax is not axes[1, 0] else "x", alpha=.18)
        ax.set_axisbelow(True)
        ax.title.set_fontsize(13)
    fig.text(.08, .025, "来源：sft-2048-v1 实际日志与数据审计汇总。Loss 下降不等同于独立评测准确率提升。", fontsize=10, color="#64748B")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "svg"]:
        fig.savefig(args.output_dir / ("sft-overview." + suffix), dpi=180)
    plt.close(fig)
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "fixed-training-sample.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "nll"])
        writer.writeheader()
        writer.writerows(checkpoints)
    print(json.dumps({"verified_steps": len(rows), "initial_fixed_sample_nll": y[0],
                      "final_fixed_sample_nll": y[-1], "median_update_seconds": float(np.median(seconds)),
                      "output": str(args.output_dir.resolve() / "sft-overview.png")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
