"""使用真实 SFT 日志生成博客用 loss 曲线，不调用训练服务。"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def main():
    directory = Path(__file__).resolve().parent.parent / "outputs/sft-figures-v1"
    metrics = directory / "metrics.jsonl"
    summary = json.loads((directory / "summary.json").read_text())
    assert hashlib.sha256(metrics.read_bytes()).hexdigest() == summary["metrics_export_sha256"]
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    steps = np.array([row["step"] for row in rows])
    loss = np.array([row["loss"] for row in rows])
    weights = np.array([row["supervised_tokens"] for row in rows])
    assert steps.tolist() == list(range(1, summary["updates"] + 1))
    assert np.isfinite(loss).all() and (weights > 0).all()
    font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=font).get_name(),
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "path",
    })
    window = 32
    smooth = np.convolve(loss * weights, np.ones(window), "valid") / np.convolve(weights, np.ones(window), "valid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    ax = axes[0]
    ax.plot(steps, loss, color="#B8CFE1", linewidth=.8, label="逐批 loss")
    ax.plot(steps[window - 1:], smooth, color="#246A91", linewidth=2,
            label="32步 token 加权均值")
    ax.set(title="训练 Loss", xlabel="更新步数", ylabel="NLL / 监督 token", ylim=(0, None))
    ax.legend(frameon=False, fontsize=10)
    checkpoints = summary["fixed_training_sample_nll"]
    x = [row["step"] for row in checkpoints]
    y = [row["nll"] for row in checkpoints]
    assert x[0] == 0 and x[-1] == summary["updates"]
    ax = axes[1]
    ax.plot(x, y, color="#246A91", marker="o", linewidth=2)
    ax.set(title="固定训练样本 NLL", xlabel="更新步数", ylabel="NLL", ylim=(.38, .61))
    for index in (0, -1):
        ax.annotate(f"{y[index]:.4f}", (x[index], y[index]), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=11)
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"sft-loss.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)
    print(directory / "sft-loss.png")


if __name__ == "__main__":
    main()
