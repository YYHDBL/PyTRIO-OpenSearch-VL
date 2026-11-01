"""导出两条真实RL benchmark轨迹为单文件HTML，保留完整文本和原题图。"""

import base64
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "outputs/rl-benchmark-100-v4"
JUDGE = ROOT / "outputs/judge-rl-benchmark-100-v4"
DATA = ROOT / "data/selected-eval100-v1/eval.jsonl"
OUTPUT = ROOT / "outputs/trajectory-viewer-v1/rl-two-trajectories.html"
SAMPLE_IDS = ["eval-0184", "eval-0176"]


def main():
    rows = {row["sample_id"]: row for row in map(json.loads, DATA.read_text().splitlines())}
    trajectories = []
    for sample_id in SAMPLE_IDS:
        record = json.loads((RUN / (sample_id + ".json")).read_text())
        score = json.loads((JUDGE / (sample_id + ".json")).read_text())
        image_path = Path(rows[sample_id]["images"][0])
        image_bytes = image_path.read_bytes()
        assert len(image_bytes) < 5_000_000 and record["sample_id"] == score["sample_id"] == sample_id
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}[image_path.suffix.lower()]
        trajectories.append({"id": sample_id, "benchmark": record["benchmark"],
            "question": record["question"], "reference": record["reference"],
            "answer": record["answer"], "termination": record["termination"],
            "elapsed_seconds": record["elapsed_seconds"], "correct": score["correct"],
            "judge_model": score["judge_model"], "judge_reasoning": score.get("raw_judge", ""),
            "trace": record["trace"], "image": "data:" + mime + ";base64," + base64.b64encode(image_bytes).decode("ascii")})
    data = json.dumps(trajectories, ensure_ascii=False).replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    document = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RL评测轨迹 · 两个真实案例</title>
<style>
:root{color-scheme:light;--ink:#172a3a;--muted:#617180;--line:#dce4e8;--blue:#176d7a;--bg:#f5f7f7;--red:#9c4538;--green:#28724b}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.page{max-width:1180px;margin:auto;padding:36px 24px 60px}h1{font-size:30px;line-height:1.2;margin:0 0 10px}.intro{color:var(--muted);margin:0 0 26px}.case{background:white;border:1px solid var(--line);border-radius:14px;margin:20px 0 28px;overflow:hidden;box-shadow:0 3px 16px #1835440a}.head{padding:20px 24px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}.head h2{margin:0;font-size:20px}.badges{display:flex;gap:8px;flex-wrap:wrap}.badge{padding:4px 10px;border-radius:20px;background:#edf2f4;color:#425563;font-size:13px}.badge.good{background:#e5f3ea;color:var(--green)}.badge.bad{background:#f9e9e6;color:var(--red)}.question{padding:18px 24px 8px}.question b,.ref b{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.07em;margin-bottom:5px}.question p,.ref p{margin:0}.image{padding:8px 24px 18px}.image img{display:block;max-width:100%;max-height:300px;object-fit:contain;border:1px solid var(--line);border-radius:8px}.ref{margin:0 24px 18px;padding:12px 14px;background:#f7f9fa;border-radius:8px}.steps{border-top:1px solid var(--line)}details.step{border-bottom:1px solid var(--line)}details.step>summary{list-style:none;cursor:pointer;padding:13px 24px;display:flex;justify-content:space-between;gap:16px;font-weight:600}details.step>summary::-webkit-details-marker{display:none}details.step>summary:hover{background:#f8fafb}.kind{color:var(--blue);font-size:13px;white-space:nowrap}.step-body{padding:0 24px 20px;display:grid;gap:12px}.block{min-width:0}.block h3{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 5px}.block pre{margin:0;padding:12px 14px;background:#f6f8f9;border:1px solid #e8edef;border-radius:8px;white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;max-height:520px;overflow:auto}.judge{padding:18px 24px 24px}.judge p{margin:5px 0}.note{font-size:13px;color:var(--muted);margin-top:28px}
</style></head><body><main class="page"><h1>RL评测：两条真实轨迹</h1><p class="intro">最终RL Sampler · 冻结100题benchmark · 原始模型回合、工具参数、工具返回与GLM-5.3-Flash评分</p><div id="cases"></div><p class="note">来源为服务器保存的真实评测记录。图中展示初始题图；工具返回内新生成的裁剪图像以原始工具文字和图像ID呈现。裁判结论仅代表本次自动评分。</p></main>
<script id="trajectory-data" type="application/json">__DATA__</script>
<script>
const rows=JSON.parse(document.getElementById('trajectory-data').textContent), root=document.getElementById('cases');
const el=(tag,cls,txt)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(txt!==undefined)n.textContent=txt;return n};
for(const r of rows){const card=el('section','case');const head=el('div','head'),title=el('h2',null,`${r.id} · ${r.benchmark}`),badges=el('div','badges');for(const [t,c] of [[r.correct?'裁判：正确':'裁判：错误',r.correct?'good':'bad'],[r.termination, r.termination==='final_answer'?'good':'bad'],[`${r.trace.length} 轮模型调用`,'' ],[`${r.trace.filter(x=>x.action).length} 次工具调用`,'']])badges.append(el('span','badge '+c,t));head.append(title,badges);card.append(head);
const q=el('div','question'),qb=el('b',null,'评测问题');q.append(qb,el('p',null,r.question));card.append(q);
const im=el('div','image'),img=el('img');img.src=r.image;img.alt=r.id+' 原始题图';im.append(img);card.append(im);
const ref=el('div','ref');ref.append(el('b',null,'参考答案'),el('p',null,r.reference));card.append(ref);
const steps=el('div','steps');r.trace.forEach((s,i)=>{const det=el('details','step');det.open=r.trace.length<8||i<2;const summary=el('summary'),kind=s.action?s.action.name:(s.assistant.includes('<response>')||s.assistant.includes('<answer>')?'模型作答':'模型输出');summary.append(el('span',null,`第 ${i+1} 步`),el('span','kind',kind));det.append(summary);const body=el('div','step-body'),raw=el('div','block');raw.append(el('h3',null,'模型完整输出'),el('pre',null,s.assistant||''));body.append(raw);if(s.action){const tool=el('div','block'),args=JSON.stringify(s.action.arguments,null,2);tool.append(el('h3',null,'工具参数'),el('pre',null,args));body.append(tool)}if(s.observation!==undefined){const obs=el('div','block');obs.append(el('h3',null,'工具返回'),el('pre',null,s.observation));body.append(obs)}if(s.tool_error)body.append(el('span','badge bad','该步记录为工具参数错误'));det.append(body);steps.append(det)});card.append(steps);
const judge=el('div','judge');judge.append(el('h3',null,`裁判解释 · ${r.judge_model}`),el('p',null,r.judge_reasoning||'无文本理由'));if(r.answer)judge.append(el('h3',null,'模型最终答案'),el('pre',null,r.answer));card.append(judge);const time=el('p','note',`记录耗时 ${r.elapsed_seconds} 秒（完整记录值）`);card.append(time);root.append(card)}
</script></body></html>'''.replace("__DATA__", data)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(document, encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT.resolve()), "trajectories": [
        {"id": row["id"], "termination": row["termination"], "rounds": len(row["trace"]),
         "tool_calls": sum("action" in step for step in row["trace"]), "correct": row["correct"]}
        for row in trajectories], "html_bytes": OUTPUT.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
