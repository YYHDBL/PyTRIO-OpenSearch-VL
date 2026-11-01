"""从核验后的聚合统计绘制共同99题的三阶段对比，不读取原始数据。"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent.parent


def main():
    data = json.loads((ROOT / "docs/evidence/training-stage-comparison-v1.json").read_text())
    names = ["Base", "SFT", "RL"]
    stages = data["stages"]
    assert all(stages[n]["questions"] == 99 for n in names)
    font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(),
        "font.size": 12, "axes.unicode_minus": False, "svg.fonttype": "path",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#CBD5E1",
        "text.color": "#1E293B", "axes.labelcolor": "#475569", "xtick.color": "#475569", "ytick.color": "#475569"})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    fig.subplots_adjust(left=.09, right=.96, bottom=.12, top=.80, wspace=.30, hspace=.53)
    fig.suptitle("Base → SFT → RL：哪些指标发生了变化？", x=.08, y=.965, ha="left", fontsize=23)
    fig.text(.08, .908, "同一批 99 道 benchmark 题 · 每题一次 · GLM 裁判 · 主模型 Qwen3.5-4B", fontsize=12, color="#475569")
    fig.text(.08, .858, "SFT：2,048 条轨迹 / 512 次更新     RL：93 道有效题 / 25 次更新", fontsize=12)
    colors = ["#94A3B8", "#2C7F9F", "#BE7A3B"]
    specs = [
        ("答对题数", [stages[n]["correct"] for n in names], "题 / 99题", 99),
        ("正常结束并给出最终回答", [stages[n]["termination_counts"]["final_answer"] for n in names], "题 / 99题", 108),
        ("工具调用总次数", [stages[n]["tool_calls"]["sum"] for n in names], "次", None),
        ("主模型输出 tokens", [stages[n]["output_tokens"]["sum"] / 1000 for n in names], "千 tokens", None)]
    for ax, (title, values, ylabel, upper) in zip(axes.flat, specs):
        bars = ax.bar(names, values, color=colors, width=.56)
        ax.set(title=title, ylabel=ylabel)
        labels = [f"{v:.0f}" for v in values] if ylabel != "千 tokens" else [f"{v:.1f}k" for v in values]
        ax.bar_label(bars, labels=labels, padding=6, fontsize=12)
        ax.set_ylim(0, upper if upper else max(values) * 1.22)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.text(.08, .047, "SFT 主要改善结束作答与资源消耗；本轮 RL 未显示准确率提升。", fontsize=12)
    fig.text(.08, .014, "限制：RL 改用了自部署摘要后端；另发现一题相同答案被裁判判出不同对错，故不能作严格因果结论。", fontsize=10, color="#64748B")
    output = ROOT / "outputs/training-stage-comparison-v1"
    output.mkdir(exist_ok=True)
    for suffix in ["png", "svg"]:
        fig.savefig(output / ("stage-comparison." + suffix), dpi=180, facecolor="white")
    plt.close(fig)
    print(str(output / "stage-comparison.png"))


if __name__ == "__main__":
    main()
