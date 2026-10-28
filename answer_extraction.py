"""复用 OpenSearch-VL 的评测答案抽取规则，不额外要求最终回答标签。"""

import re


# 核心逻辑来自 OpenSearch-VL@1ac26e2 的 opensearch_vl/eval_with_gpt4o.py。
def extract_boxed_content(text: str) -> str | None:
    idx = text.find("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    content = text[start:i-1].strip()
    content = re.sub(r'\\text\{([^}]*)\}', r'\1', content)
    content = content.replace("\\#", "#").replace("\\%", "%")
    return content


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    boxed = extract_boxed_content(text)
    if boxed:
        return boxed
    answer_tag = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_tag:
        return answer_tag.group(1).strip()
    response_tag = re.search(r'<response>(.*?)</response>', text, re.DOTALL)
    if response_tag:
        return response_tag.group(1).strip()
    parts = text.split("</think>")
    if len(parts) > 1:
        last_part = parts[-1].strip()
        last_part = re.sub(r'<tool_call>.*?</tool_call>', '', last_part, flags=re.DOTALL).strip()
        if last_part:
            return last_part[:2000]
    return text[-2000:] if len(text) > 2000 else text


def evaluation_response(record: dict) -> str:
    # 正式轨迹与作者 pipeline 一样拼接全部 assistant 文本；无 trace 的服务健康检查保留其输入。
    text = "\n\n".join(turn["assistant"] for turn in record["trace"]) if "trace" in record else record.get("answer", "")
    return extract_final_answer(text)
