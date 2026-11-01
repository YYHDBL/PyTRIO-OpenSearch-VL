"""RL离线准备：RLOO组内优势和单轮图文训练样本。"""

import numpy as np
import pytrio as trio


def rloo_advantages(rewards, fatal):
    scores = np.asarray(rewards, dtype=np.float64)
    if scores.shape != (4,) or len(fatal) != 4 or not np.isfinite(scores).all():
        raise ValueError("RLOO需要同一题的4条有限奖励和4个fatal标记")
    if any(type(flag) is not bool for flag in fatal):
        raise ValueError("fatal标记必须为布尔值")
    advantages = (scores * 4 - scores.sum()) / 3
    return np.where(fatal, np.maximum(advantages, 0), advantages)


def build_datum(prompt, completion, old_logprobs, advantage):
    tokens = list(completion)
    logprobs = np.asarray(old_logprobs, dtype=np.float32)
    if not tokens or logprobs.shape != (len(tokens),) or not np.isfinite(logprobs).all() or not np.isfinite(advantage):
        raise ValueError("生成tokens、旧策略logprobs和优势必须完整且有限")
    if len(prompt) < 1 or len(prompt) + len(tokens) - 1 > 32768:
        raise ValueError("图文训练样本长度超出32k")
    chunks = list(prompt.chunks)
    if len(tokens) > 1:
        chunks.append(trio.types.EncodedTextChunk(tokens=tokens[:-1]))
    model_input = trio.ModelInput(chunks=chunks)
    padding = len(prompt) - 1
    return trio.Datum(model_input=model_input, loss_fn_inputs={
        "target_tokens": np.asarray([0] * padding + tokens, dtype=np.int64),
        "logprobs": np.concatenate((np.zeros(padding, dtype=np.float32), logprobs)),
        "advantages": np.asarray([0.0] * padding + [advantage] * len(tokens), dtype=np.float32),
    })


def build_trajectory_datums(steps, advantage, fatal_start):
    if not steps or (fatal_start is not None and not 0 <= fatal_start <= len(steps)):
        raise ValueError("轨迹为空或fatal起点越界")

    def prefix_units(prompt):
        units = []
        for chunk in prompt.chunks:
            if isinstance(chunk, trio.types.EncodedTextChunk):
                units.extend(chunk.tokens)
            elif isinstance(chunk, trio.ImageChunk) and chunk.expected_tokens:
                units.append((chunk.data, chunk.format, chunk.expected_tokens))
            else:
                raise ValueError("图片token数量或图文chunk无效")
        return units

    datums = []
    previous = None
    for index, step in enumerate(steps):
        units = prefix_units(step["prompt"])
        if previous is not None and units[:len(previous)] != previous:
            raise ValueError("多轮上下文未保留采样时的完整token前缀")
        if fatal_start is None or index < fatal_start:
            datums.append(build_datum(step["prompt"], step["tokens"], step["logprobs"], advantage))
        previous = units + list(step["tokens"])
    return datums
