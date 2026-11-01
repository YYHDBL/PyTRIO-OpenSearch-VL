"""实际持久断点的裁判故障隔离，不发起网络请求。"""

import json
from pathlib import Path
import tempfile
import unittest


class JudgeIsolationTest(unittest.TestCase):
    def test_unconfirmed_score_defers_one_item_but_keeps_other_scores(self):
        from judge_baseline import judge_records

        root = Path(__file__).resolve().parent / "outputs"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            output = Path(directory)
            (output / "pending").mkdir()
            pending = output / "pending/eval-blocked.json"
            pending.write_text('{"state":"unconfirmed"}')
            original = pending.read_bytes()
            rows = [{"sample_id": name, "benchmark": "test", "termination": "no_final_answer",
                     "trace": [], "question": "", "reference": ""} for name in ["eval-blocked", "eval-empty"]]
            config = {"sha256": "same-protocol", "model": "test"}
            scores, deferred = judge_records(rows, output, config, continue_on_error=True)
            self.assertEqual([r["sample_id"] for r in scores], ["eval-empty"])
            self.assertEqual([r["sample_id"] for r in deferred], ["eval-blocked"])
            self.assertFalse(scores[0]["correct"])
            self.assertEqual(pending.read_bytes(), original)
            self.assertFalse((output / "requests.jsonl").exists())
            with self.assertRaises(RuntimeError):
                judge_records(rows, output, config, continue_on_error=False)
            (output / "eval-empty.json").write_text(json.dumps({"response_sha256": "wrong", "judge_config_sha256": "wrong"}))
            with self.assertRaises(ValueError):
                judge_records(rows[1:], output, config, continue_on_error=True)
