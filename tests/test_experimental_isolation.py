#!/usr/bin/env python3
"""Test 6 — Experimental isolation: the adapter honors a caller-provided PORT and
does NOT source the production config that would clobber it.

Guards the load-bearing safety fix from review (claim 3): the experimental DFlash
launcher exports DGX_INFERENCE_EXPERIMENTAL=1 plus its own PORT/CONTAINER, and the
adapter must NOT source the production inference.env (which would overwrite those
and route the run onto the production slot/container).

We invoke the REAL adapter in emit-yaml mode, in experimental mode, with a caller
PORT that differs from a *production* inference.env (pointed at by CONFIG_ROOT).
The rendered runtime YAML carries the served port, so we assert it equals the
caller's port and not the production one. (The rendered YAML does not carry the
container name, so container-name isolation is exercised by the live launch path,
not here.)

No GPU, no docker: emit-yaml mode performs NO safety rails and launches nothing."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "runtime" / "sglang" / "adapters" / "sglang.sh"
RUNTIME_ROOT = ROOT / "runtime" / "sglang"
SPEC_REL = "profiles/qwen36-27b-fp8/sglang.toml"

CALLER_PORT = "30100"          # the experimental port
PROD_PORT = "30000"            # production port, recorded in the prod inference.env


def main() -> int:
    assert ADAPTER.is_file(), f"missing {ADAPTER}"
    td = tempfile.mkdtemp()
    # A "production" inference.env with the production port. If isolation is
    # broken, the adapter sources this and renders PROD_PORT.
    (Path(td) / "inference.env").write_text(
        f'MODEL_CACHE_ROOT="/tmp/nonexistent"\n'
        f'PORT="{PROD_PORT}"\n'
        f'CONTAINER_NAME="inference-agentic"\n'   # production container
        f'PROJECT_ROOT="{ROOT}"\n'
    )
    env = dict(os.environ)
    env["CONFIG_ROOT"] = td
    env["DGX_INFERENCE_EXPERIMENTAL"] = "1"        # the isolation guard
    env["PORT"] = CALLER_PORT                       # caller's port
    env["CONTAINER_NAME"] = "dflash-experimental"   # caller's container
    env["MODEL_CACHE_ROOT"] = "/tmp/nonexistent"

    proc = subprocess.run(
        ["bash", str(ADAPTER), "agentic", str(RUNTIME_ROOT), str(ROOT),
         "qwen36-27b-fp8", "model", SPEC_REL, "qwen3.6-27b-agentic", "emit-yaml"],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        print(f"adapter emit-yaml failed (rc={proc.returncode}):\n{proc.stderr}", file=sys.stderr)
        return 1
    out = proc.stdout
    # The rendered YAML must carry the CALLER's port, NOT the production port.
    port_lines = [l for l in out.splitlines() if l.startswith("port:")]
    if not port_lines:
        print("FAIL: no port line in rendered YAML", file=sys.stderr)
        return 1
    rendered_port = port_lines[0].split(":", 1)[1].strip()
    if rendered_port == PROD_PORT:
        print(f"FAIL: isolation broken — rendered PRODUCTION port {PROD_PORT} "
              f"instead of caller port {CALLER_PORT}", file=sys.stderr)
        return 1
    if rendered_port != CALLER_PORT:
        print(f"FAIL: rendered port {rendered_port} is neither caller ({CALLER_PORT}) "
              f"nor prod ({PROD_PORT})", file=sys.stderr)
        return 1
    print(f"PASS: experimental isolation — adapter honored caller port {CALLER_PORT} "
          f"(did not source prod config with port {PROD_PORT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
