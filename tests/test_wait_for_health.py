#!/usr/bin/env python3
from __future__ import annotations

import http.server
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WAIT = ROOT / "tools" / "wait_for_health.sh"


class Healthy(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *_args):
        pass


with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Healthy) as server:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/health"
    ready = subprocess.run(["bash", str(WAIT), url, "2"], capture_output=True)
    assert ready.returncode == 0, ready.stderr.decode()

timeout = subprocess.run(
    ["bash", str(WAIT), "http://127.0.0.1:1/health", "1"],
    capture_output=True,
)
assert timeout.returncode == 75, timeout.stderr.decode()

print("PASS: health wait succeeds on HTTP 200 and fails closed on timeout")
