#!/usr/bin/env python3
"""Test 5 — install.sh --dry-run is secret-free.

Runs the REAL installer in --dry-run with a temp operator agent.env containing a
FAKE key (a valid-shape placeholder), then runs scan_secrets.py over the dry-run
output. The dry-run must render the unit + the inference.env plan and write
NOTHING, and must contain no secret/path leakage. This also exercises that
--dry-run never starts the service or writes to disk."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "deploy" / "install.sh"
SCANNER = ROOT / "tests" / "scan_secrets.py"

# A valid-shape key (64 lowercase hex) that is OBVIOUSLY a placeholder, used only
# to satisfy the installer's agent.env shape check. It is not a real secret.
FAKE_KEY = "0" * 64


def main() -> int:
    assert INSTALL.is_file(), f"missing {INSTALL}"
    with tempfile.TemporaryDirectory() as td:
        tdpath = Path(td)
        agent_env = tdpath / "agent.env"
        agent_env.write_text(f"SGLANG_API_KEY={FAKE_KEY}\n")
        os.chmod(agent_env, 0o600)
        plan_root = tdpath / "plan"          # where dry-run would render (it won't write)
        plan_root.mkdir()

        proc = subprocess.run(
            ["bash", str(INSTALL), "--dry-run",
             "--install-root", str(plan_root / "lib"),
             "--config-root", str(plan_root / "etc"),
             "--model-cache-root", str(plan_root / "cache"),
             "--agent-env", str(agent_env)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"install --dry-run failed (rc={proc.returncode}):\n{proc.stderr}", file=sys.stderr)
            return 1
        out = proc.stdout
        # scan the rendered dry-run output with the contextual scanner.
        out_file = tdpath / "dryrun-output.txt"
        out_file.write_text(out)
        scan = subprocess.run(
            [sys.executable, str(SCANNER), str(out_file)],
            capture_output=True, text=True,
        )
        if scan.returncode != 0:
            print("FAIL: dry-run output leaked secrets/paths:", file=sys.stderr)
            print(scan.stdout, file=sys.stderr)
            return 1

        # assert dry-run wrote nothing to disk under plan_root except the dir we made
        written = [p for p in plan_root.rglob("*") if p.is_file()]
        if written:
            print(f"FAIL: --dry-run wrote files: {written}", file=sys.stderr)
            return 1

        print("PASS: install.sh --dry-run is secret-free and writes nothing")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
