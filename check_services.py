"""检查凭据配置；仅 --pytrio 会发起不含采样或训练的账号能力查询。"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
import pytrio as trio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytrio", action="store_true", help="只读查询当前账号的支持模型")
    args = parser.parse_args()
    load_dotenv(Path(__file__).parent / ".env", override=False)
    required = ("PYTRIO_API_KEY", "SERPER_API_KEY", "SERPAPI_API_KEY")
    for key in (*required, "JINA_API_KEY"):
        state = "已配置" if os.environ.get(key, "").strip() else "未配置"
        print(f"{key}: {state}")
    if args.pytrio:
        if not os.environ.get("PYTRIO_API_KEY", "").strip():
            parser.error("请先在 .env 配置 PYTRIO_API_KEY")
        try:
            models = trio.ServiceClient(api_key=os.environ["PYTRIO_API_KEY"]).get_supported_models()
        except Exception as error:
            # 部分 HTTP 异常包含请求 URL；不输出异常内容，避免日志泄露凭据。
            raise SystemExit(f"PyTRIO 查询失败：{type(error).__name__}") from None
        print(f"PyTRIO 支持模型：{models}")


if __name__ == "__main__":
    main()
