"""真实 HTTP 对象的回包留存不包含认证头。"""

import importlib.util
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import shutil
import threading
import unittest
import uuid

import httpx


class ServiceIOTest(unittest.TestCase):
    def test_real_http_elapsed_is_saved(self):
        from service_io import save_http_response

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(204)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        root = Path("outputs") / ("test-http-timing-" + uuid.uuid4().hex)
        root.mkdir(parents=True)
        thread.start()
        try:
            with httpx.Client(trust_env=False) as client:
                response = client.get(f"http://127.0.0.1:{server.server_port}/")
            save_http_response(root / "response.json", response)
            record = json.loads((root / "response.json").read_text())
            self.assertIn("elapsed_seconds", record)
            self.assertAlmostEqual(record["elapsed_seconds"], response.elapsed.total_seconds(), places=5)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
            shutil.rmtree(root)

    def test_response_bytes_are_preserved_without_credentials(self):
        self.assertIsNotNone(importlib.util.find_spec("service_io"))
        from service_io import save_http_response

        root = Path("outputs/test-service-io")
        self.assertFalse(root.exists())
        root.mkdir(parents=True)
        secret = uuid.uuid4().hex
        body = bytes(range(256))
        response = httpx.Response(200, content=body, headers={"x-request-id": "request-id", "set-cookie": secret},
                                  request=httpx.Request("POST", "https://example.org", headers={"Authorization": secret}))
        try:
            path = root / "response.json"
            save_http_response(path, response)
            record = json.loads(path.read_text())
            self.assertEqual(Path(record["body_path"]).read_bytes(), body)
            self.assertEqual(record["headers"], {"x-request-id": "request-id"})
            self.assertNotIn(secret, path.read_text())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
