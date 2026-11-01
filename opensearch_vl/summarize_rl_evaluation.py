"""只读本地实际评分，汇总验证集及相同benchmark题目的模型对比。"""

import json
from pathlib import Path
import statistics

from .service_io import write_json


ROOT = Path(__file__).resolve().parent.parent


def mean_parts(scores):
    return {"scored": len(scores), "correct": sum(s["accuracy"] for s in scores),
            "accuracy": statistics.mean(s["accuracy"] for s in scores) if scores else None,
            "mean_reward": statistics.mean(s["reward"] for s in scores) if scores else None,
            "mean_format": statistics.mean(s["format_reward"] for s in scores) if scores else None,
            "mean_query": statistics.mean(s["query"] for s in scores) if scores else None}


def main():
    ids = json.loads((ROOT / "outputs/rl-selection-100-v1/validation-ids.json").read_text())
    expected = {identifier + f"-r{i}" for identifier in ids for i in range(4)}
    validation = ROOT / "outputs/rl-validation-16x4-v3"
    current = {f.name.removesuffix(".score.json"): json.loads(f.read_text()) for f in validation.glob("*.score.json")}
    assert set(current) <= expected
    previous = {identifier: json.loads((ROOT / "outputs/rl-diagnostic-128x4-v1" / (identifier + ".score.json")).read_text())
                for identifier in expected}
    assert all(s["id"] == identifier for identifier, s in current.items())
    status_path = validation / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        result = {"expected_questions": 16, "expected_rollouts": 64,
            "complete": status["complete"] and len(current) == 64,
            "rl": mean_parts(list(current.values())), "sft_all_64": mean_parts(list(previous.values())),
            "sft_same_completed_ids": mean_parts([previous[k] for k in current]),
            "gained": sum(previous[k]["accuracy"] == 0 and s["accuracy"] == 1 for k, s in current.items()),
            "lost": sum(previous[k]["accuracy"] == 1 and s["accuracy"] == 0 for k, s in current.items()),
            "missing_ids": sorted(expected - set(current)), "deferred": status["deferred"],
            "same_seeds_and_scoring_protocol": True, "validation_used_for_update": False,
            "limitations": ["16-question development set", "Flash judge not manually audited",
                           "temporary self-hosted summary backend differs from SFT reference; weights identity not verified",
                           "partial service failures are not scored as model mistakes"]}
        write_json(validation / "summary.json", result)
        print(json.dumps({"validation": result}, ensure_ascii=False), flush=True)
    judge = ROOT / "outputs/judge-rl-benchmark-100-v4"
    summary = judge / "summary.json"
    if summary.exists():
        folders = {"base": ROOT / "outputs/judge-base-aligned-100-v1",
                   "sft": ROOT / "outputs/judge-sft2048-aligned-100-v2", "rl": judge}
        scores = {name: {f.stem: json.loads(f.read_text()) for f in folder.glob("eval-*.json")}
                  for name, folder in folders.items()}
        common = sorted(set.intersection(*(set(rows) for rows in scores.values())))
        comparison = {"rl_summary_complete": json.loads(summary.read_text())["complete"],
            "same_question_count": len(common),
            "same_questions": {name: {"correct": sum(rows[k]["correct"] for k in common),
                "accuracy": sum(rows[k]["correct"] for k in common) / len(common) if common else None}
                for name, rows in scores.items()},
            "rl_vs_sft_gained": [k for k in common if not scores["sft"][k]["correct"] and scores["rl"][k]["correct"]],
            "rl_vs_sft_lost": [k for k in common if scores["sft"][k]["correct"] and not scores["rl"][k]["correct"]],
            "limitations": ["SFT has one unresolved service failure; compare identical completed IDs",
                           "RL uses a temporary self-hosted summary backend; comparison is not a strict single-variable experiment",
                           "custom 100-question benchmark subset, not the full paper benchmarks"]}
        write_json(judge / "paired-comparison.json", comparison)
        print(json.dumps({"benchmark_paired_comparison": comparison}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
