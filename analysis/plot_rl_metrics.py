"""从真实训练批次汇总绘制 RL 曲线，不调用模型服务。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import numpy as np


def main():
    root = Path(__file__).resolve().parent.parent
    data = json.loads((root / "docs/evidence/rl-training-curves-v1.json").read_text())
    rows = np.array(data["batches"], dtype=float)
    steps, groups = rows[:, 0], rows[:, 1]
    assert data["complete"] and steps.tolist() == list(range(1, 26))
    assert int(groups.sum()) == data["completed_groups"] == 93
    assert np.isfinite(rows).all() and ((rows[:, 2:] >= 0) & (rows[:, 2:] <= 1)).all()
    font = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(),
                         "font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "path"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    titles = ["组合 Reward", "格式奖励", "训练采样答案正确率"]
    colors = ["#246A91", "#39876C", "#B67A35"]
    window = 5
    for i, (ax, title, color) in enumerate(zip(axes, titles, colors, strict=True)):
        values = rows[:, i + 2]
        # 题组数不同，平滑时按有效 Rollout 数加权；每组固定四条。
        smooth = np.convolve(values * groups, np.ones(window), "valid") / np.convolve(groups, np.ones(window), "valid")
        ax.plot(steps, values, color=color, alpha=.3, linewidth=1, marker="o", markersize=3, label="逐批均值")
        ax.plot(steps[window - 1:], smooth, color=color, linewidth=2.2, label="5批加权均值")
        ax.set(title=title, xlabel="训练批次", ylim=(0, 1.05), xticks=[1, 5, 10, 15, 20, 25])
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        if i == 2:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(frameon=False, fontsize=10, loc="lower right")
    output = root / "outputs/rl-figures-v1"
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"rl-training-curves.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)
    print(json.dumps({"batches": len(rows), "groups": int(groups.sum()),
                      "rollouts": int(groups.sum()) * 4, "output": str(output / "rl-training-curves.png")}))


if __name__ == "__main__":
    main()
