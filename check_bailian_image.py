"""用官方公网图与服务器真实裁剪图验证百炼图搜图和计费字段。"""

from __future__ import annotations

import io
import json
from pathlib import Path
import time

from dotenv import load_dotenv
from PIL import Image

from agent_tools import ToolEnvironment
from bailian_image import BailianImageClient


def main() -> None:
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    cache = root / "data/tool_cache/bailian-image-check"
    budget = ToolEnvironment(cache, image_limit=2)
    client = BailianImageClient(cache / "responses")
    public_url = "https://img.alicdn.com/imgextra/i4/O1CN01YbrnSS1qtmsAkw0Ud_!!6000000005554-2-tps-788-450.png"
    with Image.open(root / "data/source/sft/livevqa/images/interface_v5.png") as source:
        crop = source.convert("RGB").crop((100, 100, 400, 400))
        stream = io.BytesIO()
        crop.save(stream, format="PNG")
        cropped_image = stream.getvalue()
    results = {}
    for name, image in (("public_url", public_url), ("server_crop_base64", cropped_image)):
        started = time.monotonic()
        result = client.search(image, before_submit=lambda: budget.charge("image_search"))
        results[name] = {"matches": len(result["matches"]), "usage": result["usage"],
                         "response_model": result["response_model"], "seconds": round(time.monotonic() - started, 2)}
        print(json.dumps({"check": name, **results[name]}, ensure_ascii=False), flush=True)
    summary = {"region": "cn-beijing", "provider": "bailian", "checks": results,
               "note": "image_search_call URLs are image URLs; website source links are not implied"}
    output = root / "outputs/bailian-image-check-v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
