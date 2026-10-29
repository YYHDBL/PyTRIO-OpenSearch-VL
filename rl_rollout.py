"""PyTRIO多轮采样：持久化原始token、旧logprob、图文前缀和工具断点。"""

import base64
import json
from pathlib import Path
import re

import numpy as np
import pytrio as trio
from PIL import Image

from agent_baseline import outside_thought, parse_action, sample_prompt
from agent_tools import ToolError, model_image_bytes
from service_io import write_json


def dump_prompt(prompt):
    chunks = []
    for c in prompt.chunks:
        value = c.model_dump()
        if c.type == "image":
            value["data"] = base64.b64encode(c.data).decode()
        chunks.append(value)
    return {"chunks": chunks}


def load_prompt(value):
    chunks = []
    for c in value["chunks"]:
        c = dict(c)
        if c["type"] == "image":
            c["data"] = base64.b64decode(c["data"], validate=True)
        chunks.append(c)
    return trio.ModelInput.model_validate({"chunks": chunks})


def run_rollout(row, identifier, sampler, tokenizer, processor, tools, budget, root: Path, system, policy_id, seed):
    path, pending = root / f"{identifier}.json", root / f"{identifier}.pending.json"
    if pending.exists():
        raise ToolError("采样存在未确认提交，禁止重复请求")
    if path.exists():
        state = json.loads(path.read_text())
        if state["policy_id"] != policy_id or state["seed"] != seed:
            raise ValueError("已有轨迹属于不同策略或采样设置")
        if state["termination"]:
            return state
    else:
        if len(row["images"]) != 1:
            raise ValueError("当前训练任务必须为单图")
        with Image.open(row["images"][0]) as image:
            raw = model_image_bytes(image)
        prompt = sample_prompt(tokenizer, [["system", system, None],
            ["user", "<image>" + row["question"], base64.b64encode(raw).decode()]], processor)
        state = {"sample_id": identifier, "question": row["question"], "reference": row["answer"],
                 "policy_id": policy_id, "seed": seed, "trace": [], "phase": "sample", "errors": 0,
                 "prompt": dump_prompt(prompt), "images": {"img_1": base64.b64encode(raw).decode()},
                 "termination": ""}
        write_json(path, state)
    eos = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(eos) != 1:
        raise ValueError("Qwen EOS编码与协议不符")
    while not state["termination"]:
        if state["phase"] == "tool":
            images = {k: base64.b64decode(v, validate=True) for k, v in state["images"].items()}
            step = state["trace"][-1]
            try:
                observation, image = tools.execute(step["action"], images)
                state["errors"] = 0
            except (ValueError, KeyError, TypeError) as error:
                observation, image = f"Tool execution error:\n{type(error).__name__}: {error}", None
                step["tool_error"] = True
                state["errors"] += 1
            step["observation"] = observation
            state["images"] = {k: base64.b64encode(v).decode() for k, v in images.items()}
            old = load_prompt(state["prompt"])
            tail = sample_prompt(tokenizer, [["tool", observation + ("<image>" if image else ""),
                base64.b64encode(image).decode() if image else None]], processor)
            tokens = step["tokens"]
            separator = "\n" if tokens[-1] == eos[0] else "<|im_end|>\n"
            context = trio.ModelInput(chunks=[*old.chunks, trio.types.EncodedTextChunk(tokens=tokens),
                trio.types.EncodedTextChunk(tokens=tokenizer.encode(separator, add_special_tokens=False)), *tail.chunks])
            state["prompt"] = dump_prompt(context)
            state["phase"] = "sample"
            write_json(path, state)
        if state["errors"] >= 3:
            state["termination"] = "fatal_error"
            break
        if len(state["trace"]) >= 50:
            state["termination"] = "turn_limit"
            break
        prompt = load_prompt(state["prompt"])
        available = 32768 - len(prompt)
        if available < 1:
            state["termination"] = "context_limit"
            break
        write_json(pending, {"policy_id": policy_id, "turn": len(state["trace"]), "seed": seed})
        response = budget.sample(sampler, prompt, identifier,
            trio.SamplingParams(max_tokens=available, temperature=0.7, top_p=1.0, seed=seed,
                               stop=["<|im_end|>"]))
        sequence = response.sequences[0]
        tokens, logprobs = list(sequence.tokens), list(sequence.logprobs or [])
        if response.input_tokens != len(prompt) or not tokens or len(tokens) != len(logprobs) or not np.isfinite(logprobs).all():
            raise ValueError("采样图文token或old logprob无效；原请求标记保留")
        step = {"prompt": dump_prompt(prompt), "tokens": tokens, "logprobs": logprobs,
                "assistant": sequence.text, "stop_reason": sequence.stop_reason}
        state["trace"].append(step)
        if sequence.stop_reason == "length":
            state["termination"] = "length_limit"
        else:
            try:
                action = parse_action(sequence.text)
            except (ValueError, TypeError):
                step["tool_error"] = True
                # 无法解析的动作仍作为真实工具错误写回；保留原始采样前缀。
                action = {"name": "invalid_tool_call", "arguments": {}}
            if action is not None:
                step["action"] = action
                state["phase"] = "tool"
            else:
                terminal = re.search(r"<(response|answer)>.*?</\1>", outside_thought(sequence.text), re.S)
                state["termination"] = "final_answer" if terminal else "no_final_answer"
        write_json(path, state)
        pending.unlink()
    write_json(path, state)
    return state
