"""只读取真实实验记录，对相同题目的Base/SFT/RL进行统计；不调用服务。"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import unicodedata


ROOT = Path(__file__).resolve().parent.parent
RUNS = {"Base": "base-aligned-100-v1", "SFT": "sft2048-aligned-100-v2", "RL": "rl-benchmark-100-v4"}
JUDGES = {"Base": "judge-base-aligned-100-v1", "SFT": "judge-sft2048-aligned-100-v2", "RL": "judge-rl-benchmark-100-v4"}
FORMAT = re.compile(r"^<think>.*?</think>\s*(<tool_call>.*?</tool_call>|<response>.*?</response>|<answer>.*?</answer>)\s*$", re.S)


def stats(values):
    values = sorted(values)
    assert values
    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lower = int(position)
        return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)
    return {"sum": sum(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "p90": percentile(.9), "max": max(values)}


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def record_metrics(record):
    trace = record["trace"]
    actions = [t["action"] for t in trace if "action" in t]
    calls = Counter(json.dumps(a, sort_keys=True, ensure_ascii=False) for a in actions)
    errors = [t for t in trace if t.get("tool_error")]
    return {"termination": record["termination"], "normal_finish": record["termination"] == "final_answer",
            "rounds": len(trace), "tool_calls": len(actions), "tool_errors": len(errors),
            "input_tokens": sum(t["input_tokens"] for t in trace),
            "output_tokens": sum(t["output_tokens"] for t in trace),
            "max_context_tokens": max((t["input_tokens"] + t["output_tokens"] for t in trace), default=0),
            "repeat_same_action_extra_calls": sum(count - 1 for count in calls.values()),
            "strict_step_format_fraction": statistics.mean(bool(FORMAT.match(t["assistant"])) and not t.get("tool_error", False) for t in trace) if trace else 0,
            "actions": dict(Counter(a["name"] for a in actions)),
            "error_types": dict(Counter(t.get("observation", "missing_observation").strip()[:180] for t in errors)),
            "elapsed_seconds_including_recovery": record["elapsed_seconds"]}


def aggregate(ids, records, scores):
    per_question = {identifier: record_metrics(records[identifier]) for identifier in ids}
    result = {"questions": len(ids), "correct": sum(scores[k]["correct"] for k in ids),
              "accuracy": sum(scores[k]["correct"] for k in ids) / len(ids),
              "termination_counts": dict(Counter(per_question[k]["termination"] for k in ids)),
              "questions_with_tool_errors": sum(per_question[k]["tool_errors"] > 0 for k in ids),
              "questions_with_repeated_calls": sum(per_question[k]["repeat_same_action_extra_calls"] > 0 for k in ids)}
    for metric in ["rounds", "tool_calls", "tool_errors", "input_tokens", "output_tokens", "max_context_tokens", "repeat_same_action_extra_calls", "strict_step_format_fraction", "elapsed_seconds_including_recovery"]:
        result[metric] = stats([per_question[k][metric] for k in ids])
    for metric in ["actions", "error_types"]:
        counts = Counter()
        for k in ids:
            counts.update(per_question[k][metric])
        result[metric] = dict(counts)
    result["per_benchmark"] = {}
    for benchmark in sorted({records[k]["benchmark"] for k in ids}):
        chosen = [k for k in ids if records[k]["benchmark"] == benchmark]
        result["per_benchmark"][benchmark] = {"questions": len(chosen), "correct": sum(scores[k]["correct"] for k in chosen),
            "normal_finish": sum(per_question[k]["normal_finish"] for k in chosen),
            "tool_calls": sum(per_question[k]["tool_calls"] for k in chosen)}
    result["correct_by_termination"] = {termination: {"count": sum(per_question[k]["termination"] == termination for k in ids),
        "correct": sum(scores[k]["correct"] for k in ids if per_question[k]["termination"] == termination)}
        for termination in result["termination_counts"]}
    return result, per_question


def paired(before, after, ids, records, scores, metrics):
    changed = [k for k in ids if scores[before][k]["correct"] != scores[after][k]["correct"]]
    gained = [k for k in changed if scores[after][k]["correct"]]
    lost = [k for k in changed if not scores[after][k]["correct"]]
    same_text_changed_grade = [k for k in ids if normalized(scores[before][k]["evaluation_response"]) == normalized(scores[after][k]["evaluation_response"])
                               and scores[before][k]["correct"] != scores[after][k]["correct"]]
    recovered_finish = [k for k in ids if not metrics[before][k]["normal_finish"] and metrics[after][k]["normal_finish"]]
    return {"gained_ids": gained, "lost_ids": lost,
        "both_correct": sum(scores[before][k]["correct"] and scores[after][k]["correct"] for k in ids),
        "both_wrong": sum(not scores[before][k]["correct"] and not scores[after][k]["correct"] for k in ids),
        "identical_normalized_report_but_changed_judge": same_text_changed_grade,
        "identical_exact_report_but_changed_judge": [k for k in same_text_changed_grade
            if scores[before][k]["evaluation_response"] == scores[after][k]["evaluation_response"]],
        "recovered_normal_finish": {"count": len(recovered_finish),
            "correct_before": sum(scores[before][k]["correct"] for k in recovered_finish),
            "correct_after": sum(scores[after][k]["correct"] for k in recovered_finish)},
        "normal_finish_regressed": sum(metrics[before][k]["normal_finish"] and not metrics[after][k]["normal_finish"] for k in ids)}


def validation_accuracy():
    from rl_diagnostic import reward_parts

    path = ROOT / "outputs/rl-validation-16x4-v3"
    config = json.loads((path / "config.json").read_text())
    ids = json.loads((ROOT / "outputs/rl-selection-100-v1/validation-ids.json").read_text())
    current, previous, current_format, previous_format = [], [], [], []
    for question in ids:
        for replica in range(4):
            identifier = question + f"-r{replica}"
            record = json.loads((path / "rollouts" / (identifier + ".json")).read_text())
            parts = reward_parts(record)
            if parts["terminal_answer"]:
                result = json.loads((path / "judges" / (identifier + "-accuracy.json")).read_text())
                prompt = config["accuracy_prompt"].format(question=record["question"],
                    reference_answer=record["reference"], assistant_answer=parts["terminal_answer"])
                sha = hashlib.sha256(json.dumps({"prompt": prompt, "config": config["judge"]}, sort_keys=True).encode()).hexdigest()
                assert sha == result["sha256"]
                correct = int(result["value"])
            else:
                correct = 0
            score = path / (identifier + ".score.json")
            if score.exists():
                assert correct == json.loads(score.read_text())["accuracy"]
            current.append(correct)
            prior = json.loads((ROOT / "outputs/rl-diagnostic-128x4-v1" / (identifier + ".score.json")).read_text())
            previous.append(prior["accuracy"])
            current_format.append(parts["format_reward"])
            previous_format.append(prior["format_reward"])
    assert len(current) == len(previous) == 64
    return {"questions": 16, "rollouts": 64, "accuracy_complete": True,
            "rl_correct": sum(current), "sft_correct": sum(previous),
            "gained": sum(a and not b for a,b in zip(current, previous)),
            "lost": sum(not a and b for a,b in zip(current, previous)),
            "rl_mean_format_all64": statistics.mean(current_format),
            "sft_mean_format_all64": statistics.mean(previous_format),
            "rl_question_correct_distribution": dict(Counter(sum(current[i:i+4]) for i in range(0,64,4))),
            "sft_question_correct_distribution": dict(Counter(sum(previous[i:i+4]) for i in range(0,64,4))),
            "combined_reward_scored": 63, "missing_query_id": "rl-0115-r1", "new_api_requests": 0}


def main():
    output = ROOT / "outputs/training-stage-comparison-v1"
    output.mkdir(exist_ok=True)
    records = {name: {d["sample_id"]: d for f in (ROOT / "outputs" / folder).glob("eval-*.json")
                      if not f.name.endswith(".partial.json") for d in [json.loads(f.read_text())]} for name, folder in RUNS.items()}
    scores = {name: {d["sample_id"]: d for f in (ROOT / "outputs" / folder).glob("eval-*.json")
                     for d in [json.loads(f.read_text())]} for name, folder in JUDGES.items()}
    ids = sorted(set.intersection(*(set(scores[name]) & set(records[name]) for name in RUNS)))
    assert len(ids) == 99
    for name in RUNS:
        for k in ids:
            assert scores[name][k]["response_sha256"] == hashlib.sha256(json.dumps(records[name][k], sort_keys=True).encode()).hexdigest()
    assert all(len({records[name][k]["question"] for name in RUNS}) == 1 and len({records[name][k]["reference"] for name in RUNS}) == 1 for k in ids)
    aggregate_metrics, metrics = {}, {}
    for name in RUNS:
        aggregate_metrics[name], metrics[name] = aggregate(ids, records[name], scores[name])
        assert sum(d["questions"] for d in aggregate_metrics[name]["per_benchmark"].values()) == 99
        assert sum(d["correct"] for d in aggregate_metrics[name]["per_benchmark"].values()) == aggregate_metrics[name]["correct"]
    comparisons = {before + "_to_" + after: paired(before, after, ids, records, scores, metrics)
                   for before, after in [("Base", "SFT"), ("SFT", "RL"), ("Base", "RL")]}
    for before, after in [("Base", "SFT"), ("SFT", "RL"), ("Base", "RL")]:
        change = comparisons[before + "_to_" + after]
        assert change["both_correct"] + change["both_wrong"] + len(change["gained_ids"]) + len(change["lost_ids"]) == 99
        assert len(change["gained_ids"]) - len(change["lost_ids"]) == aggregate_metrics[after]["correct"] - aggregate_metrics[before]["correct"]
    cases = []
    changed = sorted(set(comparisons["Base_to_SFT"]["gained_ids"] + comparisons["Base_to_SFT"]["lost_ids"]
                         + comparisons["SFT_to_RL"]["gained_ids"] + comparisons["SFT_to_RL"]["lost_ids"]))
    for identifier in changed:
        cases.append({"id": identifier, "benchmark": records["Base"][identifier]["benchmark"],
            "stages": {name: {"correct": scores[name][identifier]["correct"], **metrics[name][identifier]} for name in RUNS}})
    result = {"cohort": "same 99 frozen benchmark questions", "ids": ids, "stages": aggregate_metrics,
        "paired": comparisons, "changed_cases_metrics": cases,
        "limitations": ["RL summary backend changed to user deployment; cannot isolate all effects to RL",
                        "GLM judgments not exhaustively manually audited", "single run and 99 paired questions",
                        "elapsed_seconds includes checkpoint recovery and cache/network differences",
                        "strict format is computed on all steps, not the training fatal-prefix reward"]}
    validation = json.loads((ROOT / "outputs/rl-validation-16x4-v3/summary.json").read_text())
    result["validation"] = {k:validation[k] for k in ["expected_questions", "expected_rollouts", "complete", "rl", "sft_same_completed_ids", "gained", "lost", "missing_ids"]}
    result["validation_accuracy_all_64"] = validation_accuracy()
    validation_ids = sorted(f.name.removesuffix(".score.json") for f in (ROOT / "outputs/rl-validation-16x4-v3").glob("*.score.json"))
    contributions = {"format": [], "accuracy": [], "query": [], "total": []}
    for identifier in validation_ids:
        before = json.loads((ROOT / "outputs/rl-diagnostic-128x4-v1" / (identifier + ".score.json")).read_text())
        after = json.loads((ROOT / "outputs/rl-validation-16x4-v3" / (identifier + ".score.json")).read_text())
        f0, f1 = before["format_reward"], after["format_reward"]
        h0, h1 = .8 * before["accuracy"] + .2 * before["query"], .8 * after["accuracy"] + .2 * after["query"]
        fmt = (f1 - f0) * (h0 + h1) / 2
        acc = .8 * (after["accuracy"] - before["accuracy"]) * (f0 + f1) / 2
        query = .2 * (after["query"] - before["query"]) * (f0 + f1) / 2
        total = after["reward"] - before["reward"]
        assert abs(fmt + acc + query - total) < 1e-10
        for name, value in [("format", fmt), ("accuracy", acc), ("query", query), ("total", total)]:
            contributions[name].append(value)
    result["validation_reward_change_decomposition"] = {
        "paired_rollouts": len(validation_ids), "method": "symmetric product decomposition; algebraic identity, not causal attribution",
        "mean_contribution": {name: statistics.mean(values) for name, values in contributions.items()}}
    # 训练题对照只反映在线采样策略混合，不能冒充最终checkpoint评测。
    train_rows = []
    sft_train_rows = []
    for batch in sorted((ROOT / "outputs/rl-100-v2").glob("batch-*/metrics.json")):
        for group in json.loads(batch.read_text())["groups"]:
            for replica in range(4):
                identifier = group["id"] + f"-r{replica}"
                train_rows.append(json.loads((batch.parent / (identifier + ".score.json")).read_text()))
                sft_train_rows.append(json.loads((ROOT / "outputs/rl-diagnostic-128x4-v1" / (identifier + ".score.json")).read_text()))
    def rewards(rows):
        return {"rollouts": len(rows), "correct": sum(r["accuracy"] for r in rows),
                **{"mean_" + field: statistics.mean(r[field] for r in rows) for field in ["reward", "format_reward", "query"]}}
    assert len(train_rows) == len(sft_train_rows) == 372
    result["training_pool_comparison"] = {"same_questions": 93, "sft_diagnostic": rewards(sft_train_rows),
        "rl_online_policy_mixture": rewards(train_rows), "same_seeds": False, "final_checkpoint_evaluation": False}
    (output / "aggregate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    summary = {"stages": {name: {k:v for k,v in d.items() if k not in ["per_benchmark", "elapsed_seconds_including_recovery"]} for name,d in aggregate_metrics.items()},
               "paired": comparisons, "training_pool_comparison": result["training_pool_comparison"]}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
