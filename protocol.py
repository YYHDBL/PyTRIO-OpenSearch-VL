"""OpenSearch-VL ShareGPT 轨迹到 PyTRIO 多模态 Datum 的转换。"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytrio as trio
from PIL import Image

IMAGE_MARKER = "<image>"
IMAGE_PAD = "<|image_pad|>"
TEMPLATE_PATH = Path(__file__).parent / "templates/qwen3_5_opensearch.jinja"

# 沿用原仓库 LlamaFactory QwenToolUtils 的 JSON 工具格式，不改写作者 system 文本。
QWEN_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tool_text}"
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
    '<tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, '
    '"arguments": <args-json-object>}}\n</tool_call>'
)


def system_with_tools(system: str, tools: str | list[dict]) -> str:
    definitions = json.loads(tools) if isinstance(tools, str) and tools.strip() else tools
    if not definitions:
        return system
    if not isinstance(definitions, list):
        raise ValueError("tools 字段必须为 JSON 数组")
    wrapped = []
    names = set()
    for definition in definitions:
        if not isinstance(definition, dict):
            raise ValueError("工具定义必须是对象")
        tool = definition if definition.get("type") == "function" else {"type": "function", "function": definition}
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("工具定义缺少 function.name")
        if function["name"] in names:
            raise ValueError("工具定义名称重复")
        names.add(function["name"])
        wrapped.append(tool)
    text = "".join("\n" + json.dumps(tool, ensure_ascii=False) for tool in wrapped)
    return system + QWEN_TOOL_PROMPT.format(tool_text=text)


def _message_content(text: str, image_refs: Iterator[str], image_root: Path) -> str | list[dict]:
    parts = re.split(f"({re.escape(IMAGE_MARKER)})", text)
    content: list[dict] = []
    for part in parts:
        if not part:
            continue
        if part == IMAGE_MARKER:
            image_path = (image_root / next(image_refs)).resolve()
            if not image_path.is_relative_to(image_root.resolve()) or not image_path.is_file():
                raise FileNotFoundError(f"找不到轨迹图片或路径越界：{image_path}")
            content.append({"type": "image", "image": image_path})
        else:
            content.append({"type": "text", "text": part})
    return content[0]["text"] if len(content) == 1 and content[0]["type"] == "text" else content


def build_messages(row: dict[str, Any], image_root: Path) -> list[dict]:
    images = iter(row.get("images", []))
    messages = [{"role": "system", "content": system_with_tools(row.get("system", ""), row.get("tools", []))}]
    roles = {"human": "user", "gpt": "assistant", "observation": "tool"}
    for turn in row["conversations"]:
        source_role = turn["from"]
        if source_role not in roles:
            raise ValueError(f"不支持的轨迹角色：{source_role}")
        messages.append({
            "role": roles[source_role],
            "content": _message_content(turn["value"], images, image_root),
        })
    if next(images, None) is not None:
        raise ValueError("轨迹图片数多于对话中的 <image> 标记数")
    return messages


def _image_chunk(image_path: Path, image_processor: Any) -> trio.ImageChunk:
    with Image.open(image_path) as source:
        rgba = source.convert("RGBA")
        image = Image.alpha_composite(
            Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba
        ).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    patches = image_processor.get_number_of_image_patches(
        image.height,
        image.width,
        images_kwargs={},
    )
    expected_tokens = patches // int(image_processor.merge_size) ** 2
    if expected_tokens <= 0:
        raise ValueError(f"图片的 expected_tokens 非正数：{expected_tokens}")
    return trio.ImageChunk(data=buffer.getvalue(), format="png", expected_tokens=expected_tokens)


def _pack_image_chunks(ids: list[int], masks: list[int], messages: list[dict], tokenizer: Any, image_processor: Any):
    refs = [
        item["image"]
        for message in messages
        for item in (message["content"] if isinstance(message["content"], list) else [])
        if item["type"] == "image"
    ]
    image_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    chunks = []
    targets: list[int] = []
    weights: list[float] = []
    start = 0
    image_index = 0
    for index, token in enumerate(ids):
        if token != image_id:
            continue
        chunks.append(trio.types.EncodedTextChunk(tokens=ids[start:index]))
        targets.extend(ids[start:index])
        weights.extend(masks[start:index])
        if image_index >= len(refs):
            raise ValueError("模板中的图片标记数多于轨迹图片数")
        image_chunk = _image_chunk(refs[image_index], image_processor)
        chunks.append(image_chunk)
        targets.extend([0] * image_chunk.expected_tokens)
        weights.extend([0.0] * image_chunk.expected_tokens)
        image_index += 1
        start = index + 1
    if image_index != len(refs):
        raise ValueError(f"轨迹有 {len(refs)} 张图片，模板只渲染了 {image_index} 张")
    chunks.append(trio.types.EncodedTextChunk(tokens=ids[start:]))
    targets.extend(ids[start:])
    weights.extend(masks[start:])
    return trio.ModelInput(chunks=chunks), np.asarray(targets, dtype=np.int64), np.asarray(weights, dtype=np.float32)


def build_sft_datum(row: dict[str, Any], tokenizer: Any, image_processor: Any, image_root: Path) -> trio.Datum:
    messages = build_messages(row, image_root)
    rendered = tokenizer.apply_chat_template(
        messages,
        chat_template=TEMPLATE_PATH.read_text(encoding="utf-8"),
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    ids = list(rendered["input_ids"])
    masks = list(rendered["assistant_masks"])
    if len(ids) != len(masks):
        raise ValueError(f"token 与 assistant mask 长度不同：{len(ids)} != {len(masks)}")
    packed, targets, weights = _pack_image_chunks(ids, masks, messages, tokenizer, image_processor)
    if not packed.chunks or not isinstance(packed.chunks[-1], trio.types.EncodedTextChunk):
        raise ValueError("轨迹最后必须是可右移的文本 chunk")
    last = packed.chunks[-1]
    if not last.tokens:
        raise ValueError("轨迹末尾没有可用于自回归右移的文本 token")
    model_input = trio.ModelInput(chunks=[
        *packed.chunks[:-1],
        trio.types.EncodedTextChunk(tokens=last.tokens[:-1]),
    ])
    targets[weights == 0] = 0
    target_tokens = targets[1:]
    loss_weights = weights[1:]
    if len(model_input) != len(target_tokens) or len(target_tokens) != len(loss_weights):
        raise ValueError("SFT Datum 的 model_input、target_tokens、weights 必须等长")
    if not np.any(loss_weights):
        raise ValueError("轨迹没有 assistant 监督 token")
    return trio.Datum(
        model_input=model_input,
        loss_fn_inputs={"target_tokens": target_tokens, "weights": loss_weights},
    )
