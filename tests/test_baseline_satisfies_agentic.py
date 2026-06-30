#!/usr/bin/env python3
"""Test 1 — Baseline profile satisfies the `agentic` role.

temp request.toml (agentic requires structured_output+) + the REAL baseline
capability.toml + the REAL runtime capability record -> run the REAL resolver ->
assert status=resolved, model_id=qwen36-27b-fp8. Proves capability matching works
positively. No GPU, no network, no package install."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOLVER = ROOT / "tools" / "resolve_service_plan.py"

BASELINE_CAP = ROOT / "profiles" / "qwen36-27b-fp8" / "capability.toml"
RUNTIME_CAP = ROOT / "runtime" / "sglang" / "capability.toml"

REQUEST_TOML = """\
requested_roles = ["agentic"]
[roles.agentic]
required_model_capabilities = [
  "chat_completion",
  "tool_calling",
  "structured_output",
]
"""


def main() -> int:
    assert BASELINE_CAP.is_file(), f"missing {BASELINE_CAP}"
    assert RUNTIME_CAP.is_file(), f"missing {RUNTIME_CAP}"
    with tempfile.TemporaryDirectory() as td:
        tdpath = Path(td)
        (tdpath / "request.toml").write_text(REQUEST_TOML)
        models_dir = tdpath / "models"
        runtimes_dir = tdpath / "runtimes"
        models_dir.mkdir()
        runtimes_dir.mkdir()
        (models_dir / "baseline.toml").write_text(BASELINE_CAP.read_text())
        (runtimes_dir / "sglang.toml").write_text(RUNTIME_CAP.read_text())

        proc = subprocess.run(
            [
                sys.executable, str(RESOLVER),
                "--request", str(tdpath / "request.toml"),
                "--models-dir", str(models_dir),
                "--runtimes-dir", str(runtimes_dir),
                "--allow-unresolved",
            ],
            capture_output=True, text=True,
        )
        if proc.returncode not in (0, 2):
            print(f"resolver crashed (rc={proc.returncode}):\n{proc.stderr}", file=sys.stderr)
            return 1
        result = json.loads(proc.stdout)
        roles = result.get("roles", [])
        agentic = next((r for r in roles if r.get("role") == "agentic"), None)
        if not agentic or agentic.get("status") != "resolved":
            print(f"FAIL: agentic not resolved: {agentic}", file=sys.stderr)
            return 1
        if agentic.get("model_id") != "qwen36-27b-fp8":
            print(f"FAIL: wrong model_id: {agentic.get('model_id')}", file=sys.stderr)
            return 1
        print(f"PASS: baseline satisfies agentic -> model_id={agentic['model_id']}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
